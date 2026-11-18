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
from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.output_parsers import OutputFixingParser

current_folder = os.path.dirname(os.path.abspath(__file__))

# STAGE 1: OUTPUT FORMAT
class PatientAssessment(BaseModel):
    step_by_step_analysis: str = Field(description="Interpret the THRESHOLD FLAGS and retrieved guidelines. State what each flag means clinically.")
    possible_infection_focus: str = Field(description="Anatomical infection source. Use clues (cough=Respiratory, flank pain/dysuria=Urinary). Do not write 'Undetermined' if there are hints.")
    possible_preliminary_diagnosis: str = Field(description="The primary localized disease (e.g., Pneumonia, UTI, Diabetic Ketoacidosis, Heart Failure). DO NOT use the word 'Sepsis' here. Sepsis is a systemic complication, not a localized primary diagnosis.")
    sepsis_classification: str = Field(description="Strictly output 'Yes' or 'No'. Look AT THE RETRIEVED GUIDELINE. Does the patient meet the exact Sepsis-3 definition (Infection AND threshold-crossing organ dysfunction)? If they only have an infection without organ failure, or organ failure without infection, output 'No'.")
    septik_shock_risk: str = Field(description="Low/Moderate/High. Requires fluid-resistant hypotension AND serum lactate >2 mmol/L.")
    recommended_empiric_antibiotic: str = Field(description="CRITICAL LOCK: If 'sepsis_risk' is 'Moderate' or 'High', YOU ARE STRICTLY FORBIDDEN from writing 'None'. You MUST prescribe an antibiotic class. Write 'None' ONLY if sepsis_risk is 'Low' AND the case is 100% non-infectious.")
    hospitalization_requirement: str = Field(description="Yes/No + short justification.")
    icu_requirement: str = Field(description="Yes/No + justification.")
    clinical_risk_assessment: str = Field(description="Brief summary of mortality/deterioration risk.")

parser = PydanticOutputParser(pydantic_object=PatientAssessment)

# STAGE 2: DOCTOR NOTE / COMPLAINT CLEANING
medical_dict = {
    "AC CA": "Lung Cancer",
    "KKY": "Congestive Heart Failure",
    "KOAH": "COPD",
    "SVO": "Cerebrovascular Event (Stroke)",
    "DM": "Diabetes Mellitus",
    "HT": "Hypertension",
    "OAB": "Out of Hospital Cardiac Arrest / Emergency",
    "GDB": "Poor General Condition",
    "TDRO": "Treatment and follow-up recommended",
}


def temizle_ve_ayir(text):
    text = str(text).upper()
    for tr, en in medical_dict.items():
        text = text.replace(tr, en)
    text = re.sub(r'([A-ZÇĞİÖŞÜ]{2,}):', r'. \1:', text)
    return text.strip()


# STAGE 3: DETERMINISTIC THRESHOLD FLAGS (PYTHON, NOT LLM)
def sayi_parse_et(deger):
    if deger is None:
        return None
    s = str(deger).strip().replace(",", ".")
    s = s.lstrip("<>=").strip()
    if s.lower() in ("unknown", "bilinmiyor", "nan", ""):
        return None
    try:
        return float(s)
    except ValueError:
        return None


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
        try:
            sbp = float(bp_raw.split('/')[0].strip())
        except ValueError:
            sbp = None

    def bayrak(deger, kosul_metni, kosul_fn):
        if deger is None:
            return f"{kosul_metni}: UNKNOWN (value not available)"
        return f"{kosul_metni}: value={deger} -> {'THRESHOLD CROSSED' if kosul_fn(deger) else 'within normal range'}"

    satirlar = [
        bayrak(lactate, "Lactate (danger >2.0 mmol/L)", lambda v: v > 2.0),
        bayrak(lactate, "Lactate CRITICAL (danger >4.0 mmol/L - REQUIRES ICU)", lambda v: v > 4.0),
        bayrak(creatinine, "Creatinine (danger >=1.2 mg/dL)", lambda v: v >= 1.2),
        bayrak(plt, "Platelets (danger <150 x10^3/uL)", lambda v: v < 150),
        bayrak(sbp, "Systolic BP (danger <=100 mmHg)", lambda v: v <= 100),
        bayrak(spo2, "SpO2 (respiratory compromise <92%)", lambda v: v < 92),
        bayrak(spo2, "SpO2 CRITICAL (danger <90% - REQUIRES ICU)", lambda v: v < 90),
        bayrak(rr, "Respiratory Rate (danger >=22/min)", lambda v: v >= 22), 
        bayrak(gcs, "GCS (danger <15, altered mental status)", lambda v: v < 15),
    ]
    return "\n".join(satirlar)

# STAGE 4: GUIDELINE (PDF) + RETRIEVAL
print("Loading and splitting the Guideline (PDF)...")
loader = PyPDFLoader("nihms794087.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150)
splits = text_splitter.split_documents(docs)
print(f"Toplam {len(splits)} chunk oluşturuldu.")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8, "fetch_k": 25}
)

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


# STAGE 5: MODEL + PROMPT — TEK AŞAMA (3 aşamalı yapı kaldırıldı)
print("Connecting to the Qwen model (JSON Mode Active)...")
llm = OllamaLLM(model="qwen2.5:14b-instruct", temperature=0.0, format="json")
fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)

system_prompt = """You are an ER physician. Use ONLY English.
Never invent unmentioned history, symptoms, or vital signs.
Output ONLY valid JSON, no commentary. Base your sepsis evaluation STRICTLY on Sepsis-3 (qSOFA/SOFA) criteria."""

human_template = """
RETRIEVED GUIDELINE EXCERPTS (Sepsis-3 consensus definitions) — this is
your PRIMARY clinical authority. Do not invent criteria beyond what is
stated here or directly observed in the patient data:
--- GUIDELINE START ---
{context}
--- GUIDELINE END ---

PATIENT DATA:
{input}

PRE-COMPUTED THRESHOLD FLAGS (calculated in Python — trust these exact
values, do not recompute them yourself):
{threshold_flags}

TWO SEPARATE JUDGMENTS — apply the guideline's own criteria for each:
- sepsis_classification: uses the guideline's "Definition of Sepsis" (infection + organ dysfunction). Output strictly "Yes" or "No".
- septik_shock_risk: SEPARATELY requires fluid-resistant hypotension AND lactate >2 mmol/L. Output Low/Moderate/High.

CLINICAL REASONING & DIFFERENTIAL DIAGNOSIS (Think like a Senior Doctor):
1. NON-INFECTIOUS CAUSES vs. INFECTION: If History has Diabetes and patient has metabolic derangement (high lactate) without fever, consider DKA. If History has Heart Failure and patient has respiratory distress (low SpO2) without fever, consider Pulmonary Edema.
2. BIOMARKER OVERRIDE (Don't miss infections!): If Procalcitonin (PCT) or CRP is massively elevated, DO NOT dismiss the case as just a metabolic issue (like DKA) or pure Heart Failure. Massive inflammatory markers mean INFECTION is present.
3. CONCURRENT CONDITIONS (Dual Diagnosis): A patient can have a non-infectious emergency AND a localized infection at the same time (e.g., Heart Failure causing hypoxia, BUT patient also has cystitis/UTI). In such cases, keep Sepsis Risk 'Low' or 'Moderate' (if organ failure is driven by the heart/metabolism), but you MUST prescribe antibiotics for the localized infection.

CRITICAL RULES FOR CONSISTENCY (DO NOT CONTRADICT YOURSELF):
1. ANTIBIOTIC LOGIC: If your diagnosis mentions ANY infection (pneumonia, cystitis, UTI, sepsis), you CANNOT say "None" for antibiotics. You MUST provide an antibiotic class. Say "None" ONLY if the case is strictly 100% non-infectious (like pure DKA).
2. ICU LOGIC: Do not automatically deny ICU just because Septic Shock is absent. Patients with severe hypoxemia (SpO2 < 90%) or dangerous hyperlactatemia (Lactate > 4.0) need ICU monitoring.
3. VITAL SIGNS: GCS 15 = perfectly normal. Systolic BP > 100 = NOT hypotensive.
4. INFECTION FOCUS: Use clues! Cough/Dyspnea = Respiratory. Flank pain/Dysuria = Urinary. Write "Undetermined" only if absolutely no clues exist.

{format_instructions}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", human_template),
]).partial(format_instructions=parser.get_format_instructions())

question_answer_chain = create_stuff_documents_chain(llm, prompt)

# STAGE 6: CSV OKUMA VE JSON EXPORT
print("Reading the first 10 rows of the CSV file...")
df = pd.read_csv("sepsis_dataset.csv", nrows=10)
df = df.fillna("Unknown")

all_results = []
file_name = os.path.join(current_folder, "sepsis_raga_oncelikli.json")

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
    Temperature:{row['ATEŞ']} | Pulse:{row['NABIZ']} | Blood Pressure:{row['TANSİYON']} | Resp Rate:{row['SOLUNUM']} | SpO2:{row['SATURASYON']} | GCS:{row['GKS_TOPLAM']}
    WBC:{row['WBC']} | PLT:{row['PLT']} | NE:{row['NE']} | Creatinine:{row['KREATININ']} | Lactate:{row['LAKTAT']} | Procalcitonin:{row['PROKALSITONIN']} | CRP:{row['CRP']}
    Medical History: Diabetes({row['ÖZ_DİYABET']}), Heart Failure({row['ÖZ_KALP_YETMEZLİGİ']}), Renal Failure({row['ÖZ_BÖBREK_YETMEZLİGİ']}), Cancer({row['ÖZ_CA']}), COPD({row['ÖZ_KOAH_AMELİYAT']})
    """

    try:
        guideline_docs = guideline_baglamini_hazirla(patient_data)

        raw_answer = question_answer_chain.invoke({
            "input": patient_data,
            "context": guideline_docs,
            "threshold_flags": esikler,
        })

        validated_data = fixing_parser.parse(raw_answer)

        sepsis_cevabi = validated_data.sepsis_classification.lower()
        sok_str = validated_data.septik_shock_risk.lower()
        
        if "no" in sepsis_cevabi:
            validated_data.septik_shock_risk = "Low"

        sp02_val = sayi_parse_et(row.get('SATURASYON'))
        lactate_val = sayi_parse_et(row.get('LAKTAT'))
        
        icu_reasons = []
        if sp02_val is not None and sp02_val < 90:
            icu_reasons.append(f"SpO2 < 90% ({sp02_val}%)")
        if lactate_val is not None and lactate_val > 4.0:
            icu_reasons.append(f"Lactate > 4.0 ({lactate_val} mmol/L)")
        if sok_str in ["high", "moderate"]:
            icu_reasons.append("Septic Shock risk is Moderate/High")

        if icu_reasons:
            validated_data.icu_requirement = "Yes - Automatically indicated due to critical clinical threshold: " + " | ".join(icu_reasons)

        if "yes" in sepsis_cevabi:
            if "none" in validated_data.recommended_empiric_antibiotic.lower() or validated_data.recommended_empiric_antibiotic.strip() == "":
                validated_data.recommended_empiric_antibiotic = "Broad-spectrum antibiotics (Auto-assigned due to Sepsis Diagnosis)"
                

        patient_output = {"admission_no": admission_no}
        patient_output.update(validated_data.dict())
        all_results.append(patient_output)

        print("--- SUCCESS: Data validated and added to JSON ---")

    except Exception as e:
        print(f"!!! ERROR (Unrecoverable): Patient {admission_no} skipped. Error: {str(e)}")
        all_results.append({"admission_no": admission_no, "error": str(e)})

    time.sleep(2)

with open(file_name, "w", encoding="utf-8") as json_file:
    json.dump(all_results, json_file, ensure_ascii=False, indent=4)

print(f"\nProcess completed successfully! Results saved to '{file_name}'.")