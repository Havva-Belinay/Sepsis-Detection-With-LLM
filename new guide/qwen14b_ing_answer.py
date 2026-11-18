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

current_folder = os.path.dirname(os.path.abspath(__file__))

# STAGE 1: DEFINING THE OUTPUT FORMAT (PYDANTIC) IN ENGLISH
class PatientAssessment(BaseModel):
    step_by_step_analysis: str = Field(description="First, extract exact values for Lactate, Creatinine, PLT, GCS, and BP. Then, explicitly state if any value crosses the danger threshold (Lactate>2.0, Creatinine>=1.2, PLT<150, Systolic BP<=100).")
    possible_infection_focus: str = Field(description="The probable source or system of the infection based on the guideline and data")
    possible_preliminary_diagnosis: str = Field(description="The most likely clinical preliminary diagnosis based on objective data")
    sepsis_or_septic_shock_risk: str = Field(description="Sepsis/Septic shock risk (Low, Moderate, or High) and a one-sentence reason")
    recommended_empiric_antibiotic: str = Field(description="The recommended empiric antibiotic class or general approach based on the guideline")
    hospitalization_requirement: str = Field(description="Hospitalization requirement (Yes or No) and a one-sentence justification")
    icu_requirement: str = Field(description="Intensive Care Unit requirement (Yes or No) and a one-sentence justification")
    clinical_risk_assessment: str = Field(description="A brief summary of the patient's overall mortality and clinical deterioration risk")

parser = PydanticOutputParser(pydantic_object=PatientAssessment)

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

system_prompt = """You are an expert clinical physician working in an emergency department.
You must use ONLY English medical terminology. Do not use Turkish, Chinese, or any other language.
Your responses must be academic, concise, and objective. Do not invent medical terms. 
CRITICAL CONTEXT: Many patients in this dataset DO NOT have sepsis. They may have simple infections, viral illnesses, or non-infectious conditions. Do NOT force a sepsis diagnosis. You will be evaluated on your ability to confidently RULE OUT sepsis when criteria are not strictly met.
Make logical deductions based ONLY on the provided data.
DO NOT write unnecessary explanations, comments, or introductory sentences. RETURN ONLY a valid JSON."""

human_template = """
Below is the Medical Guideline (Context) you MUST strictly follow when evaluating the patient:
--- GUIDELINE START ---
{context}
--- GUIDELINE END ---

Below is the patient's clinical data:
--- PATIENT DATA START ---
{input}
--- PATIENT DATA END ---

ATTENTION! YOU HAVE READ THE INFORMATION ABOVE. NOW IT IS TIME TO RESPOND. 
STRICTLY FOLLOW THESE RULES:
1. SPEAK ONLY IN ENGLISH. DO NOT USE ANY OTHER LANGUAGE.
2. DO NOT anchor your assessment on the "Unverified Triage Entry". Treat it as low-reliability information. You MUST base your clinical decisions, diagnosis, and risk assessments strictly on OBJECTIVE DATA (Vital Signs and Laboratory Results).
3. STRICTLY FORBIDDEN TERMS: You are EXPLICITLY BANNED from using the terms "SIRS" or "Systemic Inflammatory Response Syndrome" anywhere in your JSON output. 
4. DIAGNOSTIC BALANCE AND ORGAN DYSFUNCTION RULE: 
- If the patient has elevated inflammatory markers (WBC, CRP, Procalcitonin) BUT all organ functions are completely normal (GCS 15, normal BP, normal Creatinine <1.2, normal Lactate <2.0), Sepsis risk is LOW. You MUST write "Uncomplicated Infection" or "Localized Infection".
- HOWEVER, if the patient has ANY sign of acute organ dysfunction or tissue hypoperfusion (e.g., Lactate > 2.0, Creatinine >= 1.2, Platelets < 150, or Systolic BP <= 100), you MUST classify the Sepsis risk as MODERATE or HIGH, even if the GCS is 15. Do not ignore kidney (creatinine) or cellular (lactate) failures.
5. ZERO TOLERANCE FOR VITAL SIGN HALLUCINATION: You MUST read the exact numbers provided. A GCS of 15 means PERFECTLY NORMAL mental status; NEVER call it "altered mental status". If Blood Pressure is in the normal range, NEVER claim the patient has hypotension. Do not invent abnormal vitals to justify a high-risk score.
6. FILL OUT ONLY THE JSON FORMAT BELOW. 

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
file_name = os.path.join(current_folder, "eng_json_clinical_assessment_results_virgul.json")

for index, row in df.iterrows():
    admission_no = str(row['BASVURU_NO'])
    print(f"\n[{index + 1}/10] Evaluating Admission No: {admission_no}...")
    
    patient_data = f"""
    Age:{row['YAS']} | Gender:{row['T_CINSIYET']} | Chief Complaint:{row['BSV_ŞİKAYETİ']} | Unverified Triage Entry:{row['ÖN_TANI_ADI']}
    Temperature:{row['ATEŞ']} | Pulse:{row['NABIZ']} | Blood Pressure:{row['TANSİYON']} | Resp Rate:{row['SOLUNUM']} | SpO2:{row['SATURASYON']} | GCS:{row['GKS_TOPLAM']}
    WBC:{row['WBC']} | PLT:{row['PLT']} | NE:{row['NE']} | Creatinine:{row['KREATININ']} | Lactate:{row['LAKTAT']} | Procalcitonin:{row['PROKALSITONIN']} | CRP:{row['CRP']}
    Medical History - Diabetes:{row['ÖZ_DİYABET']} | Heart Failure:{row['ÖZ_KALP_YETMEZLİGİ']} | Renal Failure:{row['ÖZ_BÖBREK_YETMEZLİGİ']}
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