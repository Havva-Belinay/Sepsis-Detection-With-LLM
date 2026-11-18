import os
import re
import json
import time
import pandas as pd

from pydantic import BaseModel, Field

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_ollama import OllamaLLM

from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from langchain_classic.output_parsers import OutputFixingParser


current_folder = os.path.dirname(os.path.abspath(__file__))


class PatientAssessment(BaseModel):

    step_by_step_analysis: str = Field(
        description=(
            "Summarize the patient's objective findings. "
            "Mention important abnormalities, whether infection is supported, "
            "whether organ dysfunction is present, and whether the organ "
            "dysfunction can reasonably be attributed to infection. "
            "Do not treat a single abnormal laboratory value as diagnostic "
            "of sepsis."
        )
    )

    possible_infection_focus: str = Field(
        description=(
            "Most likely anatomical infection focus based on the patient's "
            "symptoms and clinical information. Examples: Lower respiratory "
            "tract, Upper urinary tract, Gastrointestinal. "
            "If infection is not supported, write "
            "'N/A - Infection not established'."
        )
    )

    possible_preliminary_diagnosis: str = Field(
        description=(
            "Most likely preliminary clinical diagnosis based ONLY on the "
            "provided patient data. Consider both infectious and "
            "non-infectious differential diagnoses."
        )
    )

    sepsis_risk: str = Field(
        description=(
            "Clinical risk category: Low, Moderate, or High. "
            "Use the Sepsis-3 definition from the retrieved guideline. "
            "Sepsis requires infection or suspected infection PLUS "
            "organ dysfunction attributable to the infection. "
            "Do not diagnose sepsis from lactate, CRP, procalcitonin, "
            "creatinine, platelet count, or hypoxemia alone."
        )
    )

    septik_shock_risk: str = Field(
        description=(
            "Clinical risk category: Low, Moderate, or High. "
            "Septic shock requires the guideline-defined combination "
            "of persistent hypotension requiring vasopressors despite "
            "adequate fluid resuscitation AND lactate >2 mmol/L. "
            "Lactate elevation alone is NOT septic shock."
        )
    )

    recommended_empiric_antibiotic: str = Field(
        description=(
            "If a bacterial infectious diagnosis is sufficiently supported, "
            "describe an empiric antibiotic approach only when supported "
            "by the available clinical information or guideline context. "
            "Do not invent an exact regimen. "
            "If infection is not sufficiently established, write "
            "'Not indicated from available data'."
        )
    )

    hospitalization_requirement: str = Field(
        description=(
            "Yes or No followed by a concise justification based on "
            "the patient's overall clinical severity."
        )
    )

    icu_requirement: str = Field(
        description=(
            "Yes or No followed by a concise justification. "
            "Consider overall severity, respiratory compromise, "
            "hemodynamic instability, altered mental status and "
            "septic shock criteria."
        )
    )

    clinical_risk_assessment: str = Field(
        description=(
            "Brief summary of the patient's overall risk of clinical "
            "deterioration. Distinguish sepsis-related risk from "
            "non-infectious causes of deterioration when appropriate."
        )
    )


parser = PydanticOutputParser(
    pydantic_object=PatientAssessment
)

# STAGE 2

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
    """
    Cleans and expands known Turkish medical abbreviations.

    This function does NOT perform diagnosis.
    """

    if text is None:
        return "UNKNOWN"

    text = str(text).strip()

    if text == "":
        return "UNKNOWN"

    text = text.upper()

    for tr, en in medical_dict.items():
        text = text.replace(tr, en)

    text = re.sub(
        r'([A-ZÇĞİÖŞÜ]{2,}):',
        r'. \1:',
        text
    )

    return text.strip()


# STAGE 3


def parse_numeric(value):
    """
    Converts a numeric clinical value into float.

    IMPORTANT:
    The original value is preserved separately because values
    such as '<0.17' contain censoring information.
    """

    if value is None:
        return None

    s = str(value).strip()

    if s.lower() in (
        "",
        "unknown",
        "bilinmiyor",
        "nan",
        "none",
        "not available"
    ):
        return None

    s = s.replace(",", ".")

    s = re.sub(r'^[<>=\s]+', '', s)

    try:
        return float(s)

    except ValueError:
        return None


def parse_bp(value):
    """
    Extracts systolic and diastolic BP.

    Returns:
        systolic_bp, diastolic_bp
    """

    if value is None:
        return None, None

    s = str(value).strip().replace(",", ".")

    if "/" not in s:
        return None, None

    try:

        parts = s.split("/")

        systolic = float(parts[0].strip())
        diastolic = float(parts[1].strip())

        return systolic, diastolic

    except (ValueError, IndexError):

        return None, None


def get_raw_value(row, column):
    """
    Returns the original CSV value.
    """

    value = row.get(column)

    if value is None:
        return "UNKNOWN"

    if pd.isna(value):
        return "UNKNOWN"

    return str(value).strip()


def klinik_veriyi_hazirla(row):
    """
    Converts the raw CSV row into a structured clinical dictionary.

    NO SEPSIS DECISION IS MADE HERE.
    """

    systolic_bp, diastolic_bp = parse_bp(
        row.get("TANSİYON")
    )

    return {

        "age": parse_numeric(
            row.get("YAS")
        ),

        "temperature": parse_numeric(
            row.get("ATEŞ")
        ),

        "pulse": parse_numeric(
            row.get("NABIZ")
        ),

        "systolic_bp": systolic_bp,

        "diastolic_bp": diastolic_bp,

        "respiratory_rate": parse_numeric(
            row.get("SOLUNUM")
        ),

        "spo2": parse_numeric(
            row.get("SATURASYON")
        ),

        "gcs": parse_numeric(
            row.get("GKS_TOPLAM")
        ),

        "wbc": parse_numeric(
            row.get("WBC")
        ),

        "neutrophil": parse_numeric(
            row.get("NE")
        ),

        "platelets": parse_numeric(
            row.get("PLT")
        ),

        "creatinine": parse_numeric(
            row.get("KREATININ")
        ),

        "lactate": parse_numeric(
            row.get("LAKTAT")
        ),

        "procalcitonin": parse_numeric(
            row.get("PROKALSITONIN")
        ),

        "crp": parse_numeric(
            row.get("CRP")
        ),

        "diabetes": get_raw_value(
            row,
            "ÖZ_DİYABET"
        ),

        "heart_failure": get_raw_value(
            row,
            "ÖZ_KALP_YETMEZLİGİ"
        ),

        "renal_failure": get_raw_value(
            row,
            "ÖZ_BÖBREK_YETMEZLİGİ"
        ),

        "cancer": get_raw_value(
            row,
            "ÖZ_CA"
        ),

        "copd": get_raw_value(
            row,
            "ÖZ_KOAH_AMELİYAT"
        ),
    }


def create_clinical_flags(clinical):
    """
    Creates objective clinical flags.

    IMPORTANT:
    These are NOT sepsis diagnostic criteria.

    They only tell the LLM which findings are present.
    """

    flags = {}

    rr = clinical.get("respiratory_rate")
    sbp = clinical.get("systolic_bp")
    gcs = clinical.get("gcs")

    flags["qsofa_rr_ge_22"] = (
        None if rr is None else rr >= 22
    )

    flags["qsofa_sbp_le_100"] = (
        None if sbp is None else sbp <= 100
    )

    flags["qsofa_altered_mentation"] = (
        None if gcs is None else gcs < 15
    )


    lactate = clinical.get("lactate")
    creatinine = clinical.get("creatinine")
    platelets = clinical.get("platelets")
    spo2 = clinical.get("spo2")

    flags["lactate_gt_2"] = (
        None if lactate is None else lactate > 2
    )

    flags["creatinine_ge_1_2"] = (
        None if creatinine is None else creatinine >= 1.2
    )

    flags["platelets_lt_150"] = (
        None if platelets is None else platelets < 150
    )

    flags["spo2_lt_92"] = (
        None if spo2 is None else spo2 < 92
    )

    return flags


def flag_text(value):
    """
    Converts True/False/None into explicit text.
    """

    if value is None:
        return "UNKNOWN"

    if value:
        return "PRESENT"

    return "ABSENT"


def klinik_veriyi_metin_haline_getir(
    clinical,
    flags,
    chief_complaint,
    doctor_note
):
    """
    Creates the final patient context given to the LLM.
    """

    def v(key):

        value = clinical.get(key)

        if value is None:
            return "UNKNOWN"

        return str(value)

    return f"""
==============================
PATIENT DATA
==============================

Demographics:
Age: {v("age")}

Clinical information:
Chief Complaint:
{chief_complaint}

Doctor's Clinical Note:
{doctor_note}

==============================
VITAL SIGNS
==============================

Temperature: {v("temperature")} °C
Pulse: {v("pulse")} /min
Blood Pressure: {v("systolic_bp")}/{v("diastolic_bp")} mmHg
Respiratory Rate: {v("respiratory_rate")} /min
SpO2: {v("spo2")} %
GCS: {v("gcs")}

==============================
LABORATORY RESULTS
==============================

WBC: {v("wbc")} x10^3/uL
Neutrophils: {v("neutrophil")} x10^3/uL
Platelets: {v("platelets")} x10^3/uL
Creatinine: {v("creatinine")} mg/dL
Lactate: {v("lactate")} mmol/L
Procalcitonin: {v("procalcitonin")} ng/mL
CRP: {v("crp")} mg/L

==============================
MEDICAL HISTORY
==============================

Diabetes: {v("diabetes")}
Heart Failure: {v("heart_failure")}
Renal Failure: {v("renal_failure")}
Cancer: {v("cancer")}
COPD: {v("copd")}

==============================
OBJECTIVE CLINICAL FLAGS
==============================

IMPORTANT:
These flags are NOT diagnoses.
They are only deterministic observations calculated
from the raw patient data.

qSOFA respiratory rate >=22:
{flag_text(flags["qsofa_rr_ge_22"])}

qSOFA systolic BP <=100:
{flag_text(flags["qsofa_sbp_le_100"])}

qSOFA altered mentation (GCS <15):
{flag_text(flags["qsofa_altered_mentation"])}

Lactate >2 mmol/L:
{flag_text(flags["lactate_gt_2"])}

Creatinine >=1.2 mg/dL:
{flag_text(flags["creatinine_ge_1_2"])}

Platelets <150 x10^3/uL:
{flag_text(flags["platelets_lt_150"])}

SpO2 <92%:
{flag_text(flags["spo2_lt_92"])}
"""


# STAGE 4

print(
    "Loading and splitting the Sepsis-3 guideline..."
)

loader = PyPDFLoader(
    "nihms794087.pdf"
)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=150
)

splits = text_splitter.split_documents(
    docs
)

print(
    f"Total guideline chunks created: {len(splits)}"
)



embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings
)


GURULTU_PATTERNS = [

    "Funding/Support",
    "DSMB",
    "reports grants from",
    "reports serving",
    "reports receiving",
    "personal fees from",
    "Role of the Funder",
    "Author Contributions",
    "Corresponding Author",
    "outside the submitted work",
    "Society of Critical Care",
    "Congress of",
    "advisory board",
    "speaking from",
    "travel to meetings",
]


def chunk_gurultu_mu(text):

    text_lower = text.lower()

    return any(
        pattern.lower() in text_lower
        for pattern in GURULTU_PATTERNS
    )



GUIDELINE_QUERIES = {

    "sepsis_definition": """
    Sepsis definition:
    life-threatening organ dysfunction caused by a
    dysregulated host response to infection.
    Sepsis-3 consensus definition.
    """,

    "organ_dysfunction": """
    Sepsis-associated organ dysfunction.
    SOFA score.
    Acute organ dysfunction involving respiratory,
    coagulation, liver, cardiovascular, central nervous
    system and renal systems.
    """,

    "infection": """
    Infection and suspected infection.
    Clinical evidence of infection.
    Relationship between infection and sepsis.
    Distinguishing infection from sepsis.
    """,

    "septic_shock": """
    Septic shock definition.
    Persistent hypotension requiring vasopressors
    to maintain MAP 65 mmHg or greater and serum
    lactate greater than 2 mmol/L despite adequate
    volume resuscitation.
    """,

    "qsofa": """
    qSOFA:
    respiratory rate 22 or greater,
    altered mentation,
    systolic blood pressure 100 mmHg or less.
    Clinical use and limitations of qSOFA.
    """,
}


def guideline_parcalarini_getir(
    vectorstore,
    queries,
    k_per_topic=3
):
    """
    Retrieves guideline knowledge for each required
    clinical concept.
    """

    retrieved = []

    seen = set()

    for topic, query in queries.items():

        topic_docs = vectorstore.similarity_search(
            query,
            k=k_per_topic
        )

        for doc in topic_docs:

            content = doc.page_content.strip()

            if not content:
                continue

            if chunk_gurultu_mu(content):
                continue

            if content in seen:
                continue

            seen.add(content)

            doc.metadata["clinical_topic"] = topic

            retrieved.append(doc)

    return retrieved


GUIDELINE_DOCS = guideline_parcalarini_getir(
    vectorstore,
    GUIDELINE_QUERIES,
    k_per_topic=3
)


print(
    f"Selected {len(GUIDELINE_DOCS)} clinical guideline chunks."
)


def guideline_context_hazirla(docs):
    """
    Converts guideline documents into a clear context block.
    """

    sections = []

    for i, doc in enumerate(
        docs,
        start=1
    ):

        topic = doc.metadata.get(
            "clinical_topic",
            "general"
        )

        page = doc.metadata.get(
            "page",
            "unknown"
        )

        sections.append(
            f"""
--- GUIDELINE CHUNK {i} ---
Clinical topic: {topic}
Source page: {page}

{doc.page_content}
"""
        )

    return "\n".join(sections)


GUIDELINE_CONTEXT = guideline_context_hazirla(
    GUIDELINE_DOCS
)


# STAGE 5

print(
    "Connecting to Qwen 2.5 14B..."
)

llm = OllamaLLM(
    model="qwen2.5:14b-instruct",
    temperature=0.0,
    format="json"
)


fixing_parser = OutputFixingParser.from_llm(
    parser=parser,
    llm=llm
)


system_prompt = """
You are an emergency medicine physician performing
a structured clinical assessment.

Your task is to determine whether the patient has
evidence compatible with Sepsis-3 sepsis or septic shock.

The retrieved excerpts from the provided Sepsis-3
guideline are your PRIMARY AUTHORITY for definitions.

You must distinguish five different situations:

1. Infection
2. Infection without sepsis
3. Sepsis
4. Septic shock
5. Non-infectious disease that may mimic sepsis

IMPORTANT:

Do NOT diagnose sepsis from a single abnormal laboratory
value.

Do NOT assume:

lactate >2 = sepsis

creatinine >=1.2 = sepsis

platelets <150 = sepsis

CRP elevation = sepsis

procalcitonin elevation = sepsis

hypoxemia = sepsis

These findings may support clinical concern, but they
do not independently establish Sepsis-3.

Sepsis requires:

INFECTION OR SUSPECTED INFECTION

PLUS

ORGAN DYSFUNCTION ATTRIBUTABLE TO THE INFECTION.

The relationship between infection and organ dysfunction
is essential.

Consider alternative explanations such as:

- heart failure
- pulmonary edema
- diabetic ketoacidosis
- chronic renal disease
- malignancy
- other metabolic disease
- other cardiopulmonary disease

Do not force an infectious diagnosis.

Septic shock is a separate diagnosis.

Do NOT diagnose septic shock merely because lactate
is elevated.

Use the guideline-defined septic shock criteria.

qSOFA is a risk-screening tool.

qSOFA is NOT the definition of sepsis.

Never invent:

- symptoms
- laboratory values
- vital signs
- diagnoses
- medical history
- treatment
- vasopressor use
- fluid resuscitation

If information is unavailable, say that it is unavailable.

Use the exact patient values provided.

GCS 15 means normal mentation.

Systolic BP >100 mmHg is not hypotension.

Return ONLY valid JSON.
"""


human_template = """

========================================
SEPSIS-3 GUIDELINE EXCERPTS
========================================

The following text was retrieved from the
provided Sepsis-3 guideline.

Use it as the primary source for the
definitions of sepsis, organ dysfunction,
qSOFA and septic shock.

{context}


========================================
PATIENT INFORMATION
========================================

{input}


========================================
CLINICAL REASONING PROCEDURE
========================================

Evaluate the patient in this order.

STEP 1 — INFECTION

First determine whether there is objective or
clinical evidence of infection.

Use:

- chief complaint
- symptoms
- clinical note
- anatomical clues
- inflammatory markers
- medical history

Do not assume infection merely because CRP,
WBC, procalcitonin or lactate is elevated.

If infection is uncertain, explicitly state this.


STEP 2 — ORGAN DYSFUNCTION

Determine whether there is objective evidence
of acute organ dysfunction.

Consider:

- respiratory system
- cardiovascular system
- neurological system
- renal system
- coagulation system
- other clinically relevant findings

Do not convert an isolated abnormal laboratory
value into sepsis.


STEP 3 — CAUSAL RELATIONSHIP

If organ dysfunction exists, determine whether
there is evidence that the dysfunction is
attributable to infection.

Consider alternative explanations.

For example:

Severe hypoxemia in a patient with heart failure
may have a cardiogenic explanation.

Elevated lactate in a patient with a metabolic
emergency may have a non-infectious explanation.

Renal dysfunction may be chronic or caused by
another condition.

Do not automatically attribute every abnormality
to infection.


STEP 4 — SEPSIS

Apply the retrieved Sepsis-3 definition.

Conceptually:

INFECTION
+
INFECTION-RELATED ORGAN DYSFUNCTION
=
SEPSIS

If infection is present without convincing
infection-related organ dysfunction, do not
classify the patient as septic.

If organ dysfunction exists but infection is
not established or another cause is more likely,
do not automatically classify the patient as septic.


STEP 5 — SEPTIC SHOCK

Evaluate septic shock separately.

Do not infer septic shock from lactate elevation.

Look specifically for:

- persistent hypotension
- vasopressor requirement
- adequate volume resuscitation
- lactate >2 mmol/L

If vasopressor use or adequate fluid resuscitation
cannot be established from the data, septic shock
cannot be confirmed.


STEP 6 — DIFFERENTIAL DIAGNOSIS

Consider non-infectious explanations when appropriate.

Examples:

- heart failure
- pulmonary edema
- diabetic ketoacidosis
- chronic renal disease
- malignancy
- metabolic disease
- cardiopulmonary disease

Do not force the patient into a sepsis diagnosis.


STEP 7 — FINAL RISK CLASSIFICATION

sepsis_risk MUST be exactly one of:

Low
Moderate
High

This is a clinical risk category and does not replace
the Sepsis-3 definition.

septik_shock_risk MUST be exactly one of:

Low
Moderate
High

Never classify septic shock as High based only on
elevated lactate.


========================================
ANTIBIOTIC RULE
========================================

Only recommend empiric antibiotics if an infectious
diagnosis is sufficiently supported.

Do NOT prescribe antibiotics solely because:

- lactate is elevated
- CRP is elevated
- procalcitonin is elevated
- WBC is elevated
- platelets are low
- creatinine is elevated

Do not invent an exact antibiotic regimen.

If the available information is insufficient to select
a reliable regimen, write:

"Guideline-specific regimen cannot be determined
from the available data."


========================================
CONSISTENCY RULES
========================================

The following statements in your output must not
contradict each other.

If you classify the condition as purely non-infectious,
do not simultaneously describe confirmed bacterial
infection.

If you say septic shock is present, the required
shock evidence must be described.

If GCS is 15, do not describe altered mentation.

If systolic BP is above 100 mmHg, do not describe
hypotension.

If infection is not established, do not state that
sepsis is definitely present.

========================================
OUTPUT
========================================

Return ONLY the following JSON structure:

{format_instructions}
"""


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", human_template),
    ]
).partial(
    format_instructions=parser.get_format_instructions()
)


question_answer_chain = create_stuff_documents_chain(
    llm,
    prompt
)

# STAGE 6

print(
    "Reading the first 10 rows of the CSV file..."
)

df = pd.read_csv(
    "sepsis_dataset.csv",
    nrows=10
)


df = df.fillna("Unknown")


all_results = []

debug_results = []


file_name = os.path.join(
    current_folder,
    "esnek_model.json"
)


debug_file_name = os.path.join(
    current_folder,
    "esnek_model_debug.json"
)


for index, row in df.iterrows():

    admission_no = str(
        row["BASVURU_NO"]
    )

    print(
        f"\n[{index + 1}/{len(df)}] "
        f"Evaluating Admission No: "
        f"{admission_no}..."
    )

    try:
        doktor_notu = temizle_ve_ayir(
            row.get(
                "SORU_METNI",
                "UNKNOWN"
            )
        )

        basvuru_sikayeti = temizle_ve_ayir(
            row.get(
                "BSV_ŞİKAYETİ",
                "UNKNOWN"
            )
        )


        clinical = klinik_veriyi_hazirla(
            row
        )

        clinical_flags = create_clinical_flags(
            clinical
        )

        patient_data = klinik_veriyi_metin_haline_getir(
            clinical=clinical,
            flags=clinical_flags,
            chief_complaint=basvuru_sikayeti,
            doctor_note=doktor_notu
        )

        raw_answer = question_answer_chain.invoke(
            {
                "input": patient_data,
                "context": GUIDELINE_DOCS,
            }
        )

        validated_data = fixing_parser.parse(
            raw_answer
        )

        validated_dict = (
            validated_data.model_dump()
            if hasattr(
                validated_data,
                "model_dump"
            )
            else validated_data.dict()
        )

        patient_output = {
            "admission_no": admission_no
        }

        patient_output.update(
            validated_dict
        )

        all_results.append(
            patient_output
        )

        retrieved_chunks = []

        for doc in GUIDELINE_DOCS:

            retrieved_chunks.append(
                {
                    "clinical_topic": doc.metadata.get(
                        "clinical_topic",
                        "unknown"
                    ),
                    "page": doc.metadata.get(
                        "page",
                        "unknown"
                    ),
                    "content": doc.page_content
                }
            )


        debug_results.append(
            {
                "admission_no": admission_no,

                "clinical_data": clinical,

                "clinical_flags": clinical_flags,

                "patient_context_sent_to_llm": patient_data,

                "retrieved_guideline_chunks": (
                    retrieved_chunks
                ),

                "raw_model_output": raw_answer,

                "validated_output": validated_dict,
            }
        )


        print(
            "--- SUCCESS ---"
        )

        print(
            f"Sepsis risk: "
            f"{validated_dict.get('sepsis_risk')}"
        )

        print(
            f"Septic shock risk: "
            f"{validated_dict.get('septik_shock_risk')}"
        )


    except Exception as e:

        error_message = str(e)

        print(
            f"!!! ERROR: Patient "
            f"{admission_no} failed."
        )

        print(
            f"Error details: {error_message}"
        )

        all_results.append(
            {
                "admission_no": admission_no,
                "error": error_message
            }
        )

        debug_results.append(
            {
                "admission_no": admission_no,

                "clinical_data": (
                    clinical
                    if "clinical" in locals()
                    else None
                ),

                "clinical_flags": (
                    clinical_flags
                    if "clinical_flags" in locals()
                    else None
                ),

                "patient_context_sent_to_llm": (
                    patient_data
                    if "patient_data" in locals()
                    else None
                ),

                "raw_model_output": (
                    raw_answer
                    if "raw_answer" in locals()
                    else None
                ),

                "validated_output": None,

                "error": error_message
            }
        )

    for variable_name in [
        "clinical",
        "clinical_flags",
        "patient_data",
        "raw_answer"
    ]:

        if variable_name in locals():
            del locals()[variable_name]

    time.sleep(2)

with open(
    file_name,
    "w",
    encoding="utf-8"
) as json_file:

    json.dump(
        all_results,
        json_file,
        ensure_ascii=False,
        indent=4
    )

with open(
    debug_file_name,
    "w",
    encoding="utf-8"
) as debug_json_file:

    json.dump(
        debug_results,
        debug_json_file,
        ensure_ascii=False,
        indent=4
    )

print()
print("=" * 60)
print("PROCESS COMPLETED")
print("=" * 60)

print(
    f"Clinical results:\n{file_name}"
)

print(
    f"\nDebug information:\n{debug_file_name}"
)

print(
    f"\nPatients processed: {len(all_results)}"
)