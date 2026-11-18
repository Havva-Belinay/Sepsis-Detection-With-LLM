import os
import pandas as pd
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

mevcut_klasor = os.path.dirname(os.path.abspath(__file__))

# AŞAMA 1: KILAVUZU (PDF) HAZIRLAMA VE RAG
print("Kılavuz (PDF) yükleniyor ve parçalanıyor...")
loader = PyPDFLoader("Sepsis.pdf") 
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

print("Vektör veritabanı oluşturuluyor (İlk çalışmada biraz zaman alabilir)...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()

# AŞAMA 2: MODELİ BAĞLAMA VE KOMUT (TERS YÜZ EDİLMİŞ)
print("Llama modeli bağlanıyor (RAG Destekli - Kapsamlı Değerlendirme)...")
llm = OllamaLLM(model="qwen2.5:14b-instruct", temperature=0.0)

system_prompt = "Sen uzman bir Türk klinik doktorusun."

human_template = """
Aşağıda hastayı değerlendirirken KESİNLİKLE uyman gereken Tıbbi Kılavuz (Context) bulunmaktadır:
--- KILAVUZ BİLGİSİ BAŞLANGICI ---
{context}
--- KILAVUZ BİLGİSİ BİTİŞİ ---

Aşağıda ise hastanın klinik verileri bulunmaktadır:
--- HASTA VERİSİ BAŞLANGICI ---
{input}
--- HASTA VERİSİ BİTİŞİ ---

ŞİMDİ DİKKAT! YUKARIDAKİ BİLGİLERİ OKUDUN VE HAFIZANA ALDIN. 
ŞİMDİ YANIT VERME ZAMANI. AŞAĞIDAKİ KURALLARA KESİNLİKLE UY:
1. SADECE TÜRKÇE KONUŞ. İNGİLİZCE VEYA BAŞKA BİR DİL KULLANMA.
2. SADECE AŞAĞIDAKİ 7 MADDELİK ŞABLONU DOLDUR, ÖNCESİNDE VEYA SONRASINDA BAŞKA HİÇBİR CÜMLE (Örn: "Anladım", "İşte sonuçlar") YAZMA!

OLASI ENFEKSİYON ODAĞI: 
[Kılavuza ve verilere göre enfeksiyonun muhtemel kaynağını veya sistemini yaz]

OLASI ÖN TANI: 
[Hastanın tablosuna göre en olası klinik ön tanıyı yaz]

SEPSİS VEYA SEPTİK ŞOK RİSKİ: 
[Sepsis/Septik şok riskini Düşük / Orta / Yüksek olarak belirt ve tek cümleyle nedenini açıkla]

ÖNERİLEN AMPİRİK ANTİBİYOTİK TEDAVİ YAKLAŞIMI: 
[Kılavuza göre önerilen ampirik antibiyotik sınıfını veya genel yaklaşımını yaz]

YATIŞ GEREKLİLİĞİ: 
[Evet / Hayır şeklinde belirt ve tek cümleyle gerekçelendir]

YOĞUN BAKIM İHTİYACI ÖNGÖRÜSÜ: 
[Evet / Hayır şeklinde belirt ve tek cümleyle gerekçelendir]

KLİNİK RİSK DEĞERLENDİRMESİ: 
[Hastanın genel mortalite ve klinik kötüleşme riskini kısa bir özetle belirt]
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", human_template),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# AŞAMA 3: CSV'DEN VERİ OKUMA VE TEST
print("CSV dosyasının ilk 10 satırı okunuyor...")
df = pd.read_csv("sepsis_dataset.csv", nrows=10)
df = df.fillna("Bilinmiyor")

dosya_adi = os.path.join(mevcut_klasor, "tr_kapsamli_klinik_degerlendirme_qwen14.txt")

with open(dosya_adi, "w", encoding="utf-8") as dosya:
    dosya.write("=== YZ DESTEKLİ KLİNİK KARAR DEĞERLENDİRME SONUÇLARI ===\n\n")
    
    for index, row in df.iterrows():
        hasta_no = row['BASVURU_NO']
        print(f"\n[{index + 1}/10] Başvuru No: {hasta_no} RAG sistemi ile değerlendiriliyor...")
        
        hasta_verisi = f"""
        --- DEMOGRAFİK VE BAŞVURU BİLGİLERİ ---
        Yaş: {row['YAS']} | Cinsiyet: {row['T_CINSIYET']}
        Başvuru Şikayeti: {row['BSV_ŞİKAYETİ']}
        Ön Tanı: {row['ÖN_TANI_ADI']}
        
        --- HAYATİ BULGULAR (KLİNİK DURUM) ---
        Ateş: {row['ATEŞ']}
        Nabız: {row['NABIZ']}
        Tansiyon: {row['TANSİYON']}
        Solunum Hızı: {row['SOLUNUM']}
        Oksijen Saturasyonu: {row['SATURASYON']}
        Glaskow Koma Skoru: Toplam {row['GKS_TOPLAM']} (Göz: {row['GKS_E']}, Sözel: {row['GKS_V']}, Motor: {row['GKS_M']})
        
        --- LABORATUVAR SONUÇLARI ---
        Beyaz Kan Hücresi (WBC): {row['WBC']}
        Trombosit (PLT): {row['PLT']}
        Nötrofil (NE): {row['NE']}
        Kreatinin: {row['KREATININ']}
        Laktat: {row['LAKTAT']}
        Prokalsitonin: {row['PROKALSITONIN']}
        CRP: {row['CRP']}
        
        --- KRONİK HASTALIKLAR (ÖZGEÇMİŞ) ---
        Diyabet: {row['ÖZ_DİYABET']} | Tansiyon: {row['ÖZ_TANSİYON']} | Kalp Yetmezliği: {row['ÖZ_KALP_YETMEZLİGİ']}
        KOAH: {row['ÖZ_KOAH_AMELİYAT']} | Böbrek Yetmezliği: {row['ÖZ_BÖBREK_YETMEZLİGİ']} | Kanser: {row['ÖZ_CA']}
        """
        
        try:
            sonuc = rag_chain.invoke({"input": hasta_verisi})
            cevap = sonuc["answer"]
            
            print("--- MODELİN YANITI ---")
            print(cevap)
            print("="*60)
            
            dosya.write(f"HASTA BAŞVURU NO: {hasta_no}\n")
            dosya.write("-" * 40 + "\n")
            dosya.write(f"{cevap}\n")
            dosya.write("=" * 60 + "\n\n")
            
        except Exception as e:
            hata_mesaji = f"BU HASTADA BİR HATA OLUŞTU VE ATLANDI: {str(e)}"
            print(hata_mesaji)
            dosya.write(f"HASTA BAŞVURU NO: {hasta_no}\n")
            dosya.write(f"{hata_mesaji}\n")
            dosya.write("=" * 60 + "\n\n")
            
        time.sleep(2) 

print(f"\nİşlem başarıyla tamamlandı! Sonuçlar '{dosya_adi}' dosyasına kaydedildi.")