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

mevcut_klasor = os.path.dirname(os.path.abspath(__file__))

# AŞAMA 1: ÇIKTI FORMATININ (PYDANTIC) TANIMLANMASI
class HastaDegerlendirmesi(BaseModel):
    olasi_enfeksiyon_odagi: str = Field(description="Kılavuza ve verilere göre enfeksiyonun muhtemel kaynağı veya sistemi")
    olasi_on_tani: str = Field(description="Hastanın tablosuna göre en olası klinik ön tanı")
    sepsis_riski: str = Field(description="Sepsis/Septik şok riski (Düşük, Orta veya Yüksek) ve tek cümlelik nedeni")
    ampirik_antibiyotik: str = Field(description="Kılavuza göre önerilen ampirik antibiyotik sınıfı veya genel yaklaşımı")
    yatis_gerekliligi: str = Field(description="Yatış gerekliliği (Evet veya Hayır) ve tek cümlelik gerekçesi")
    yogun_bakim_ihtiyaci: str = Field(description="Yoğun bakım ihtiyacı (Evet veya Hayır) ve tek cümlelik gerekçesi")
    klinik_risk: str = Field(description="Hastanın genel mortalite ve klinik kötüleşme riskinin kısa özeti")

parser = PydanticOutputParser(pydantic_object=HastaDegerlendirmesi)

# AŞAMA 2: KILAVUZU (PDF) HAZIRLAMA VE RAG
print("Kılavuz (PDF) yükleniyor ve parçalanıyor...")
loader = PyPDFLoader("nihms794087.pdf") 
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

print("Vektör veritabanı oluşturuluyor (İlk çalışmada biraz zaman alabilir)...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()

# AŞAMA 3: MODELİ BAĞLAMA VE KOMUT MÜHENDİSLİĞİ
print("Qwen modeli bağlanıyor (JSON Modu Aktif)...")
llm = OllamaLLM(model="qwen2.5:14b-instruct", temperature=0.0, format="json")

onarici_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)

system_prompt = """Sen acil serviste görevli uzman bir Türk enfeksiyon hastalıkları hekimisin.
SADECE Türkçe tıbbi terminoloji kullanacaksın. Asla İngilizce, Çince veya başka bir dil kullanma.
Vereceğin yanıtlar kesinlikle akademik, kısa ve objektif olmalıdır. Uydurma tıbbi terimler kullanma.
Sadece verilen verilere dayanarak mantıksal çıkarımlar yap.
Gereksiz açıklama, yorum, veya "Anladım, işte sonuçlar" gibi giriş cümleleri YAZMA. SADECE geçerli bir JSON döndür."""

human_template = """
Aşağıda hastayı değerlendirirken KESİNLİKLE uyman gereken Tıbbi Kılavuz (Context) bulunmaktadır:
--- KILAVUZ BAŞLANGICI ---
{context}
--- KILAVUZ BİTİŞİ ---

Hastanın klinik verileri:
--- HASTA VERİSİ ---
{input}
--- BİTİŞ ---

ŞİMDİ DİKKAT! Aşağıdaki format talimatlarına KESİNLİKLE uyarak SADECE geçerli bir JSON objesi döndür:
{format_instructions}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", human_template),
]).partial(format_instructions=parser.get_format_instructions())

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# AŞAMA 4: CSV'DEN VERİ OKUMA VE JSON EXPORT
print("CSV dosyasının ilk 10 satırı okunuyor...")
df = pd.read_csv("sepsis_dataset.csv", nrows=10)
df = df.fillna("Bilinmiyor")

tum_sonuclar = []
dosya_adi = os.path.join(mevcut_klasor, "yz_klinik_degerlendirme_sonuclari.json")

for index, row in df.iterrows():
    hasta_no = str(row['BASVURU_NO'])
    print(f"\n[{index + 1}/10] Başvuru No: {hasta_no} değerlendiriliyor...")
    
    hasta_verisi = f"""
    Yaş:{row['YAS']} | Cinsiyet:{row['T_CINSIYET']} | Şikayet:{row['BSV_ŞİKAYETİ']} | ÖnTanı:{row['ÖN_TANI_ADI']}
    Ateş:{row['ATEŞ']} | Nabız:{row['NABIZ']} | Tansiyon:{row['TANSİYON']} | Solunum:{row['SOLUNUM']} | SpO2:{row['SATURASYON']} | GKS:{row['GKS_TOPLAM']}
    WBC:{row['WBC']} | PLT:{row['PLT']} | NE:{row['NE']} | Kreatinin:{row['KREATININ']} | Laktat:{row['LAKTAT']} | ProK:{row['PROKALSITONIN']} | CRP:{row['CRP']}
    Özgeçmiş - DM:{row['ÖZ_DİYABET']} | KKY:{row['ÖZ_KALP_YETMEZLİGİ']} | KBY:{row['ÖZ_BÖBREK_YETMEZLİGİ']}
    """
    
    try:
        sonuc = rag_chain.invoke({"input": hasta_verisi})
        ham_cevap = sonuc["answer"]
        
        dogrulanmis_veri = onarici_parser.parse(ham_cevap)
        
        hasta_cikti = {"basvuru_no": hasta_no}
        hasta_cikti.update(dogrulanmis_veri.dict())
        tum_sonuclar.append(hasta_cikti)
        
        print("--- BAŞARILI: Veri doğrulandı ve JSON'a eklendi ---")
        
    except Exception as e:
        print(f"!!! HATA (Onarılamadı): {hasta_no} numaralı hasta atlandı. Hata: {str(e)}")
        tum_sonuclar.append({"basvuru_no": hasta_no, "hata": str(e)})
        
    time.sleep(2) 

with open(dosya_adi, "w", encoding="utf-8") as json_dosya:
    json.dump(tum_sonuclar, json_dosya, ensure_ascii=False, indent=4)

print(f"\nİşlem başarıyla tamamlandı! Yapılandırılmış veriler '{dosya_adi}' adlı JSON dosyasına kaydedildi.")