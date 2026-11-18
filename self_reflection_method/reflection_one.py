import os
import re
import pandas as pd
import time
import json
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_classic.output_parsers import OutputFixingParser

current_folder = os.path.dirname(os.path.abspath(__file__))

# STAGE 1: OUTPUT FORMAT
class PatientAssessment(BaseModel):
    step_by_step_analysis: str = Field(description="Interpret the provided THRESHOLD FLAGS (already computed) in light of the retrieved guideline. State what each flag means clinically according to the guideline.")
    possible_infection_focus: str = Field(description="Anatomical infection source. Write 'Undetermined' only if no clues exist. If cough/low SpO2 -> Respiratory. If flank pain -> Urinary. If non-infectious -> N/A - Noninfectious.")
    possible_preliminary_diagnosis: str = Field(description="Most likely diagnosis based only on explicit data/text.")
    sepsis_risk: str = Field(description="Low/Moderate/High. Apply the guideline's OWN 'Definition of Sepsis' criteria. If clearly non-infectious (DKA, Heart Failure), this MUST be Low.")
    septik_shock_risk: str = Field(description="Low/Moderate/High. Requires fluid-resistant hypotension AND serum lactate >2 mmol/L. Can only be Moderate/High if sepsis_risk is at least Moderate.")
    recommended_empiric_antibiotic: str = Field(description="Antibiotic class. If sepsis_risk is Moderate/High, YOU MUST RECOMMEND AN ANTIBIOTIC. Write 'None' ONLY IF the case is strictly non-infectious.")
    hospitalization_requirement: str = Field(description="Yes/No + one-sentence justification.")
    icu_requirement: str = Field(description="Yes/No + one-sentence justification, based on septik_shock_risk and overall severity.")
    clinical_risk_assessment: str = Field(description="Brief summary of the patient's overall mortality/deterioration risk.")

parser = PydanticOutputParser(pydantic_object=PatientAssessment)

# STAGE 2: DOCTOR NOTE / COMPLAINT CLEANING
medical_dict = {
    "AC CA": "Lung Cancer", "KKY": "Congestive Heart Failure", "KOAH": "COPD",
    "SVO": "Cerebrovascular Event (Stroke)", "DM": "Diabetes Mellitus", "HT": "Hypertension",
    "OAB": "Out of Hospital Cardiac Arrest", "GDB": "Poor General Condition",
    "TDRO": "Treatment and follow-up recommended",
}

def temizle_ve_ayir(text):
    text = str(text).upper()
    for tr, en in medical_dict.items():
        text = text.replace(tr, en)
    text = re.sub(r'([A-ZÇĞİÖŞÜ]{2,}):', r'. \1:', text)
    return text.strip()

# STAGE 3: DETERMINISTIC THRESHOLD FLAGS
def sayi_parse_et(deger):
    if deger is None: return None
    s = str(deger).strip().replace(",", ".")
    s = s.lstrip("<>=").strip()
    if s.lower() in ("unknown", "bilinmiyor", "nan", ""): return None
    try: return float(s)
    except ValueError: return None

def esik_bayraklarini_hesapla(row):
    lactate = sayi_parse_et(row.get('LAKTAT'))
    creatinine = sayi_parse_et(row.get('KREATININ'))
    plt = sayi_parse_et(row.get('PLT'))
    spo2 = sayi_parse_et(row.get('SATURASYON'))
    rr = sayi_parse_et(row.get('SOLUNUM'))
    gcs = sayi_parse_et(row.get('GKS_TOPLAM'))

    bp_raw = str(row.get('TANSİYON', ''))
    sbp = None
    if '/' in bp_raw:
        try: sbp = float(bp_raw.split('/')[0].strip())
        except ValueError: sbp = None

    def bayrak(deger, kosul_metni, kosul_fn):
        if deger is None: return f"{kosul_metni}: UNKNOWN (value not available)"
        return f"{kosul_metni}: value={deger} -> {'THRESHOLD CROSSED' if kosul_fn(deger) else 'within normal range'}"

    satirlar = [
        bayrak(lactate, "Lactate (danger >2.0 mmol/L)", lambda v: v > 2.0),
        bayrak(creatinine, "Creatinine (danger >=1.2 mg/dL)", lambda v: v >= 1.2),
        bayrak(plt, "Platelets (danger <150 x10^3/uL)", lambda v: v < 150),
        bayrak(sbp, "Systolic BP (danger <=100 mmHg)", lambda v: v <= 100),
        bayrak(spo2, "SpO2 (danger <92%)", lambda v: v < 92),
        bayrak(rr, "Respiratory Rate (danger >22/min)", lambda v: v > 22),
        bayrak(gcs, "GCS (danger <15)", lambda v: v < 15),
    ]
    return "\n".join(satirlar)

# STAGE 4: GUIDELINE (PDF) + RETRIEVAL
print("Loading and splitting the Guideline (PDF)...")
loader = PyPDFLoader("nihms794087.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150)
splits = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 8, "fetch_k": 25})

GURULTU_PATTERNS = [
    "Funding/Support", "DSMB", "reports grants from", "reports serving",
    "reports receiving", "personal fees from", "Role of the Funder",
    "Author Contributions", "Corresponding Author", "School of Medicine",
    "University College", "Institute for", "outside the submitted work",
    "Society of Critical Care", "Society;", "Congress of", "Intensive Care Society",
    "Emergency Medicine;", "advisory board", "speaking from", "travel to meetings",
]

def chunk_gurultu_mu(text: str) -> bool:
    isabet = sum(1 for p in GURULTU_PATTERNS if p.lower() in text.lower())
    return isabet >= 1

def gurultuyu_filtrele(docs):
    return [d for d in docs if not chunk_gurultu_mu(d.page_content)]

TANIM_SORGULARI = [
    "Sepsis is defined as life-threatening organ dysfunction caused by a dysregulated host response to infection",
    "Septic shock definition circulatory and cellular metabolic abnormalities",
]

def sabit_tanimlari_getir(vectorstore, k=2):
    sonuclar = []
    gorulen = set()
    for sorgu in TANIM_SORGULARI:
        docs = vectorstore.similarity_search(sorgu, k=k)
        for d in docs:
            if not chunk_gurultu_mu(d.page_content) and d.page_content not in gorulen:
                sonuclar.append(d)
                gorulen.add(d.page_content)
    return sonuclar

def guideline_baglamini_hazirla(hasta_verisi: str):
    ham_docs = retriever.invoke(hasta_verisi)
    temiz_docs = gurultuyu_filtrele(ham_docs)
    sabit_docs = sabit_tanimlari_getir(vectorstore)
    mevcut_icerikler = {d.page_content for d in temiz_docs}
    for d in sabit_docs:
        if d.page_content not in mevcut_icerikler:
            temiz_docs.insert(0, d)
            mevcut_icerikler.add(d.page_content)
    return temiz_docs

# STAGE 5: DRAFT & REFLECTION LLMs
print("Connecting to the Qwen model...")
llm_draft = OllamaLLM(model="qwen2.5:14b-instruct", temperature=0.1)
llm_reflection = OllamaLLM(model="qwen2.5:14b-instruct", temperature=0.0, format="json", keep_alive=0)
fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=llm_reflection)

draft_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an ER physician. Use ONLY English. Base your evaluation strictly on the provided threshold flags and guidelines."),
    ("human", """
GUIDELINE:
{context}

PATIENT DATA:
{input}

THRESHOLDS:
{threshold_flags}

Write a comprehensive clinical assessment. Think about whether this is Sepsis, Heart Failure, or DKA. 
Determine the Sepsis Risk, Septic Shock Risk, and Antibiotics needed.
(Do not worry about JSON format yet, just output your medical reasoning).
""")
])

draft_chain = create_stuff_documents_chain(llm_draft, draft_prompt)

reflection_prompt = ChatPromptTemplate.from_template("""
You are an Expert Medical Auditor. Review the ER physician's Initial Draft and the Patient Data to generate the final, corrected JSON assessment.

PATIENT DATA:
{input}

THRESHOLDS:
{threshold_flags}

INITIAL DRAFT:
{draft_text}

CRITICAL AUDIT RULES (You MUST fix these errors if present in the draft):
1. THE ANTIBIOTIC CONTRADICTION: If the diagnosis involves "Sepsis" or "Infection", you MUST assign an empiric antibiotic class. You CANNOT say "None" for an infection.
2. THE NON-INFECTIOUS ERROR: If the case is strictly Heart Failure (e.g., low SpO2 + heart failure history + no fever) OR DKA (e.g., diabetes history + high lactate + acetone), you MUST set the diagnosis to that, set sepsis_risk to "Low", and recommended_empiric_antibiotic to "None".
3. THE "UNDETERMINED" LAZINESS: Do not use "Undetermined" for infection focus if there are clues. Cough/Dyspnea/Low SpO2 = Respiratory. Flank pain/Dysuria = Urinary.

Apply these corrections and output ONLY the final JSON matching this schema exactly:
{format_instructions}
""")

reflection_chain = reflection_prompt | llm_reflection

# STAGE 6: EXECUTION
print("Reading CSV...")
df = pd.read_csv("sepsis_dataset.csv", nrows=10)
df = df.fillna("Unknown")

all_results = []
file_name = os.path.join(current_folder, "self_reflection_first.json")

for index, row in df.iterrows():
    admission_no = str(row['BASVURU_NO'])
    print(f"\n[{index + 1}/10] Evaluating Admission No: {admission_no}...")

    doktor_notu = temizle_ve_ayir(row['SORU_METNI'])
    basvuru_sikayeti = temizle_ve_ayir(row['BSV_ŞİKAYETİ'])
    esikler = esik_bayraklarini_hesapla(row)

    patient_data = f"""
    Age:{row['YAS']} | Gender:{row['T_CINSIYET']}
    Chief Complaint:{basvuru_sikayeti}
    Doctor's Clinical Note:{doktor_notu}
    Vitals: Temp:{row['ATEŞ']} | Pulse:{row['NABIZ']} | BP:{row['TANSİYON']} | Resp Rate:{row['SOLUNUM']} | SpO2:{row['SATURASYON']} | GCS:{row['GKS_TOPLAM']}
    Labs: WBC:{row['WBC']} | PLT:{row['PLT']} | Creatinine:{row['KREATININ']} | Lactate:{row['LAKTAT']} | CRP:{row['CRP']}
    History: Diabetes({row['ÖZ_DİYABET']}), Heart Failure({row['ÖZ_KALP_YETMEZLİGİ']}), Cancer({row['ÖZ_CA']})
    """

    try:
        guideline_docs = guideline_baglamini_hazirla(patient_data)

        print("  -> Drafting initial assessment...")
        raw_draft = draft_chain.invoke({
            "input": patient_data,
            "context": guideline_docs,
            "threshold_flags": esikler,
        })

        print("  -> Auditing and generating final JSON...")
        raw_json_response = reflection_chain.invoke({
            "input": patient_data,
            "threshold_flags": esikler,
            "draft_text": raw_draft,
            "format_instructions": parser.get_format_instructions()
        })

        validated_data = fixing_parser.parse(raw_json_response)
        
        patient_output = {"admission_no": admission_no}
        patient_output.update(validated_data.dict())
        all_results.append(patient_output)

        print("  --- SUCCESS: Evaluated and corrected! ---")

    except Exception as e:
        print(f"!!! ERROR: Patient {admission_no} skipped. Error: {str(e)}")
        all_results.append({"admission_no": admission_no, "error": str(e)})

with open(file_name, "w", encoding="utf-8") as json_file:
    json.dump(all_results, json_file, ensure_ascii=False, indent=4)

print(f"\nProcess completed successfully! Results saved to '{file_name}'.")