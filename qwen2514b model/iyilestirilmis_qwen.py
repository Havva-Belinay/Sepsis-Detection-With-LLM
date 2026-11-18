import os
import re
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

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 20}
)

# AŞAMA 2: MODELİ BAĞLAMA VE KOMUT
print("Qwen2.5-14B modeli bağlanıyor (RAG Destekli)...")
llm = OllamaLLM(model="qwen2.5:14b-instruct", temperature=0.0, num_predict=900)

system_prompt = """Sen bir acil servis / enfeksiyon hastalıkları alanında
uzmanlaşmış, klinik karar destek sistemi olarak görev yapan bir asistansın.

Görevin: Hastanın sepsis riskini değerlendirmek. Karar hiyerarşin şudur:
1. ÖNCELİKLE sana verilen Tıbbi Kılavuz'daki kriterleri uygula.
2. Kılavuzda hastanın durumuna doğrudan karşılık gelen bir kriter yoksa,
   genel kabul görmüş standart kriterlere (SIRS, qSOFA, Sepsis-3) başvur
   ve bunu açıkça belirt.
3. Analiz bölümünde tespit ettiğin bulgu ile SONUÇ bölümü çelişmemelidir.

MUTLAK DİL KURALI: Yanıtının HER KELİMESİ Türkçe harflerle (Latin alfabesi,
Türkçe özel karakterler dahil: ç, ğ, ı, ö, ş, ü) yazılmalıdır. Çince, Japonca,
Korece, Kiril alfabesi, Arapça veya bu diller dışındaki hiçbir alfabe/yazı
sistemi KESİNLİKLE kullanılamaz. İngilizce, İspanyolca gibi başka bir Latin
dilden tek bir kelime bile sızmamalıdır (istisna: WBC, CRP, PLT gibi zaten
Türkçe tıbbi literatürde de İngilizce kısaltmayla kullanılan laboratuvar
kısaltmaları serbesttir).

Kesin kurallar:
- Kesin tanı koyma, yalnızca risk seviyesi belirt.
- Nesnel, kısa ve klinik bir dil kullan; yorum veya spekülasyon yapma.
- "Orta" sonucunu sadece kriter kontrol listesi gerçekten karışık/sınırda
  çıktığında kullan; belirsizlik durumunda otomatik olarak "Orta" seçme.
- Aynı cümleyi veya paragrafı birden fazla kez tekrarlama."""

human_template = """
Aşağıda hastayı değerlendirirken KESİNLİKLE uyman gereken Tıbbi Kılavuz (Context) bulunmaktadır:
--- KILAVUZ BİLGİSİ BAŞLANGICI ---
{context}
--- KILAVUZ BİLGİSİ BİTİŞİ ---

Aşağıda ise hastanın klinik verileri bulunmaktadır:
--- HASTA VERİSİ BAŞLANGICI ---
{input}
--- HASTA VERİSİ BİTİŞİ ---

ŞİMDİ DİKKAT! AŞAĞIDAKİ KURALLARA KESİNLİKLE UY:
1. SADECE TÜRKÇE KONUŞ. BAŞKA HİÇBİR DİL VEYA ALFABE KULLANMA.
2. SADECE AŞAĞIDAKİ ŞABLONU DOLDUR, BAŞKA HİÇBİR CÜMLE YAZMA!
3. HER BAŞLIK İÇİN EN FAZLA 2-3 CÜMLE YAZ, TEKRARA DÜŞME.
4. KRİTER KONTROL LİSTESİ bölümünü, her satır için "VAR" veya "YOK"
   yazarak doldur - bu, SONUÇ bölümüne geçmeden önceki zorunlu adımdır.
5. SONUÇ, KRİTER KONTROL LİSTESİ ile tutarlı olmalıdır.

HAYATİ BULGULARIN ANALİZİ:
[Kılavuza göre hastanın hayati bulgularını burada Türkçe değerlendir]

LABORATUVAR SONUÇLARININ ANALİZİ:
[Kılavuza göre hastanın laboratuvar sonuçlarını burada Türkçe değerlendir]

KRİTER KONTROL LİSTESİ:
- Ateş >38.5°C veya <35°C: [VAR/YOK]
- Kalp atım hızı >90/dk: [VAR/YOK]
- Solunum sayısı >20/dk: [VAR/YOK]
- WBC >12.000 veya <4.000: [VAR/YOK]
- Laktat >2 mmol/L: [VAR/YOK]
- Hipotansiyon (belirgin düşüklük): [VAR/YOK]
- Trombosit <100.000/mL: [VAR/YOK]
- Bilinç bulanıklığı / GKS düşüklüğü: [VAR/YOK]

RİSK DEĞERLENDİRMESİ:
[Kılavuzdaki kurallara göre hastanın genel sepsis riskini burada Türkçe açıkla.
Kılavuzda karşılık yoksa SIRS/qSOFA/Sepsis-3'e başvurduğunu belirt.]

SONUÇ:
[Sadece şu üç kelimeden birini yaz: Yüksek / Orta / Düşük]
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", human_template),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# YENİ: DİL/KARAKTER DOĞRULAMA FONKSİYONU
NON_LATIN_PATTERN = re.compile(
    r'[\u4e00-\u9fff'   # CJK (Çince) temel karakterler
    r'\u3040-\u30ff'    # Hiragana + Katakana (Japonca)
    r'\uac00-\ud7af'    # Hangul (Korece)
    r'\u0400-\u04ff]'   # Kiril alfabesi (Rusça vb.)
)


def dil_temiz_mi(metin: str) -> bool:
    """Metinde Latin/Türkçe alfabesi dışında karakter olup olmadığını kontrol eder."""
    return not bool(NON_LATIN_PATTERN.search(metin))


def guvenli_invoke(rag_chain, girdi, max_deneme=3):
    """
    Modeli çağırır; yanıtta Latin-olmayan karakter (Çince/Japonca/Korece/
    Kiril) tespit edilirse, aynı girdiyle en fazla max_deneme kez tekrar
    dener. qwen14b denemelerinde gözlemlenen dil sızıntısı sorununu
    otomatik olarak azaltmayı hedefler.
    """
    son_cevap = None
    for deneme in range(1, max_deneme + 1):
        sonuc = rag_chain.invoke({"input": girdi})
        cevap = sonuc["answer"]
        son_cevap = cevap
        if dil_temiz_mi(cevap):
            return cevap, deneme
        print(f"  [UYARI] Latin-olmayan karakter tespit edildi, "
              f"{deneme}. deneme başarısız. Tekrar deneniyor...")
    return son_cevap, max_deneme


# AŞAMA 3: CSV'DEN VERİ OKUMA VE TEST
print("CSV dosyasının ilk 10 satırı okunuyor...")
df = pd.read_csv("sepsis_dataset.csv", nrows=10)
df = df.fillna("Bilinmiyor")

dosya_adi = os.path.join(mevcut_klasor, "tr_kisa_kilavuzlu_sonuclar_qwen14_v2.txt")

with open(dosya_adi, "w", encoding="utf-8") as dosya:
    dosya.write("=== SEPSİS RİSK DEĞERLENDİRME SONUÇLARI (RAG + CHECKLIST + DİL KONTROLÜ - QWEN14B) ===\n\n")

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
            cevap, kac_denemede = guvenli_invoke(rag_chain, hasta_verisi, max_deneme=3)

            temiz_mi = dil_temiz_mi(cevap)
            durum_etiketi = "TEMİZ" if temiz_mi else "DİL SIZINTISI VAR (3 denemede düzelmedi)"

            print(f"--- MODELİN YANITI ({kac_denemede}. denemede, {durum_etiketi}) ---")
            print(cevap)
            print("=" * 60)

            dosya.write(f"HASTA BAŞVURU NO: {hasta_no}\n")
            dosya.write(f"[Durum: {durum_etiketi} | Deneme sayısı: {kac_denemede}]\n")
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