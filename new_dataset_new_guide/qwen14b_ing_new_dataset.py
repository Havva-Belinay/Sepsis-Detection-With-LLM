import os
import pandas as pd
import time
import json
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.output_parsers import OutputFixingParser
import re

current_folder = os.path.dirname(os.path.abspath(__file__))

# STAGE 1: DEFINING THE OUTPUT FORMAT (PYDANTIC) IN ENGLISH
class PatientAssessment(BaseModel):
    step_by_step_analysis: str = Field(description="Extract exact values: Lactate, Creatinine, PLT, GCS, Systolic BP, SpO2, RR. State which cross danger thresholds (Lactate>2.0, Creatinine>=1.2, PLT<150, Systolic BP<=100, SpO2<92, RR>22).")
    possible_infection_focus: str = Field(description="Anatomical infection source ONLY if directly supported by text (e.g. 'dysuria'->urinary, 'cough'->respiratory). If no specific textual clue exists, write 'Undetermined' — do NOT guess.")
    possible_preliminary_diagnosis: str = Field(description="Most likely diagnosis based only on explicit data/text.")
    noninfectious_flag: str = Field(description="Yes/No. Yes ONLY if an explicit keyword/lab supports it: DKA needs 'acetone breath'/'ketone'/glucose data; Heart Failure needs 'pulmonary edema'/'orthopnea'/'CHF history'/troponin; Malignancy-related needs active cancer history + symptom link. If Yes, name the exact supporting phrase you found.")
    sepsis_risk: str = Field(description="Low/Moderate/High — STRICT rule-based, ignoring overall severity: based only on (infection focus present) + (organ dysfunction thresholds crossed). If noninfectious_flag=Yes, cap this at Low-Moderate regardless of labs.")
    overall_clinical_risk: str = Field(description="Low/Moderate/High — the patient's REAL deterioration risk regardless of cause (infectious or not). May be High even if sepsis_risk is Low, e.g. due to severe hypoxemia or DKA.")
    recommended_empiric_antibiotic: str = Field(description="Antibiotic class, or 'None — noninfectious presentation' if noninfectious_flag=Yes and no infection focus.")
    hospitalization_requirement: str = Field(description="Yes/No + one-sentence reason.")
    icu_requirement: str = Field(description="Yes/No + one-sentence reason, based on overall_clinical_risk, not sepsis_risk alone.")
    clinical_risk_assessment: str = Field(description="A brief summary of the patient's overall mortality and clinical deterioration risk")

parser = PydanticOutputParser(pydantic_object=PatientAssessment)

def temizle_ve_ayir(text):
    text = str(text).upper()
    for tr, en in medical_dict.items():
        text = text.replace(tr, en)
    text = re.sub(r'([A-ZÇĞİÖŞÜ]{2,}):', r'. \1:', text)
    return text

# STAGE 2: PREPARING THE GUIDELINE (PDF) AND RAG
print("Loading and splitting the Guideline (PDF)...")
loader = PyPDFLoader("nihms794087.pdf") 
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

print("Creating vector database (This may take a while on the first run)...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()

# STAGE 3: CONNECTING THE MODEL AND ENGLISH PROMPT ENGINEERING
print("Connecting to the Qwen model (JSON Mode Active - English)...")
llm = OllamaLLM(model="qwen2.5:14b-instruct", temperature=0.0, format="json")

fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)

system_prompt = """You are an ER physician. Use ONLY English.
Never invent unmentioned history or symptoms.
Output ONLY valid JSON, no commentary. Never write "SIRS"."""

human_template = """
GUIDELINE:
{context}

PATIENT DATA (Chief Complaint and Doctor's Note may contain the ONLY clue
to non-infectious causes — read them literally, do not skip):
{input}

TWO SEPARATE JUDGMENTS (do not merge them):
- sepsis_risk = strict infection-based classification (needs BOTH an
  infection focus AND organ dysfunction).
- overall_clinical_risk = the patient's true danger level, infectious or
  not. May differ from sepsis_risk.

NONINFECTIOUS OVERRIDE — evidence-gated:
Only set noninfectious_flag=Yes if you can quote the EXACT phrase/value:
- DKA: needs "acetone"/"ketone" smell OR glucose/ketone lab value.
- Heart Failure/Pulmonary edema: needs "edema"/"orthopnea"/"CHF"/
  "pulmonary edema" wording OR heart failure history + respiratory symptoms.
- Malignancy complication: needs active cancer history explicitly linked
  to the current complaint.
If no such exact phrase exists, noninfectious_flag=No.

RULES:
- GCS 15 = perfectly normal. BP>100 systolic = NOT hypotensive.
- Never group "elevated X and Y" if only one is elevated.
- possible_infection_focus: write "Undetermined" if no specific textual
  clue supports an anatomical source.

Fill the JSON. sepsis_risk and overall_clinical_risk are DIFFERENT
questions — answer both honestly even if they disagree.
{format_instructions}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", human_template),
]).partial(format_instructions=parser.get_format_instructions())

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# STAGE 4: READING DATA FROM CSV AND JSON EXPORT
print("Reading the first 10 rows of the CSV file...")
df = pd.read_csv("sepsis_dataset.csv", nrows=10)
df = df.replace(',', '.', regex=True) 
df = df.fillna("Unknown")

all_results = []
file_name = os.path.join(current_folder, "new_human_template.json")

for index, row in df.iterrows():
    admission_no = str(row['BASVURU_NO'])
    print(f"\n[{index + 1}/10] Evaluating Admission No: {admission_no}...")

    medical_dict = {
        "AC CA": "Lung Cancer",
        "KKY": "Congestive Heart Failure",
        "KOAH": "COPD",
        "SVO": "Cerebrovascular Event (Stroke)",
        "DM": "Diabetes Mellitus",
        "HT": "Hypertension",
        "OAB": "Out of Hospital Cardiac Arrest / Emergency",
        "GDB": "Poor General Condition",
        "TDRO": "Treatment and follow-up recommended"
    }

    doktor_notu = str(row['SORU_METNI']).upper()
    for tr_term, eng_term in medical_dict.items():
        doktor_notu = doktor_notu.replace(tr_term, eng_term)
    
    patient_data = f"""
    Age:{row['YAS']} | Gender:{row['T_CINSIYET']} 
    Chief Complaint:{row['BSV_ŞİKAYETİ']} 
    Doctor's Clinical Note:{doktor_notu}
    Temperature:{row['ATEŞ']} | Pulse:{row['NABIZ']} | Blood Pressure:{row['TANSİYON']} | Resp Rate:{row['SOLUNUM']} | SpO2:{row['SATURASYON']} | GCS:{row['GKS_TOPLAM']}
    WBC:{row['WBC']} | PLT:{row['PLT']} | NE:{row['NE']} | Creatinine:{row['KREATININ']} | Lactate:{row['LAKTAT']} | Procalcitonin:{row['PROKALSITONIN']} | CRP:{row['CRP']}
    Medical History: Diabetes({row['ÖZ_DİYABET']}), Heart Failure({row['ÖZ_KALP_YETMEZLİGİ']}), Renal Failure({row['ÖZ_BÖBREK_YETMEZLİGİ']}), Cancer({row['ÖZ_CA']}), COPD({row['ÖZ_KOAH_AMELİYAT']})
    """
    
    try:
        result = rag_chain.invoke({"input": patient_data})
        raw_answer = result["answer"]
        
        validated_data = fixing_parser.parse(raw_answer)
        
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

print(f"\nProcess completed successfully! Structured English data saved to the JSON file '{file_name}'.")