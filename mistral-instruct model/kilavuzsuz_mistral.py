import os
import time
import pandas as pd
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

mevcut_klasor = os.path.dirname(os.path.abspath(__file__))

# 1. Modeli Bağlama
print("Llama modeli bağlanıyor...")
llm = OllamaLLM(model="mistral", temperature=0.0)

# 2. Prompt (Komut) Şablonunu Hazırlama
system_prompt = (
    "Sen uzman bir klinik doktorusun. Yalnızca akıcı ve profesyonel TÜRKÇE konuşursun. "
    "Görevin, sana verilen hasta verilerini inceleyip sepsis riskini değerlendirmektir. "
    "İngilizce, İspanyolca veya başka bir dil kullanmak KESİNLİKLE YASAKTIR. Uydurma kelimeler kullanma."
)

human_template = """
Aşağıdaki hasta verilerini incele:
{input}

DİKKAT: LÜTFEN SADECE AŞAĞIDAKİ ŞABLONU BİREBİR KULLANARAK TÜRKÇE YANIT VER. 
Şablonun dışına çıkma ve fazladan cümle ekleme:

HAYATİ BULGULARIN ANALİZİ: 
[Hayati bulguların tıbbi değerlendirmesini Türkçe yazın]

LABORATUVAR SONUÇLARININ ANALİZİ: 
[Laboratuvar değerlerinin tıbbi değerlendirmesini Türkçe yazın]

RİSK DEĞERLENDİRMESİ: 
[Genel sepsis riskini Türkçe olarak açıklayın]

SONUÇ: 
[Sadece şu üç kelimeden birini yazın: Yüksek / Orta / Düşük]
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", human_template),
])

chain = prompt | llm

# 3. CSV Dosyasını Okuma
print("CSV dosyasının ilk 10 satırı okunuyor...")
df = pd.read_csv("sepsis_dataset.csv", nrows=10)
df = df.fillna("Bilinmiyor")

# 4. Sonuçları TXT Dosyasına Yazma ve Modeli Döngüye Sokma
dosya_adi = os.path.join(mevcut_klasor, "kilavuzsuz_mistral_sonuçları.txt")

with open(dosya_adi, "w", encoding="utf-8") as dosya:
    dosya.write("=== SEPSİS RİSK DEĞERLENDİRME SONUÇLARI (KILAVUZSUZ BASELINE) ===\n\n")
    
    for index, row in df.iterrows():
        hasta_no = row['BASVURU_NO']
        print(f"\n[{index + 1}/10] Başvuru No: {hasta_no} numaralı hasta değerlendiriliyor...")
        
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
            sonuc = chain.invoke({"input": hasta_verisi})
            
            print("--- MODELİN YANITI ---")
            print(sonuc)
            print("="*60)
            
            dosya.write(f"HASTA BAŞVURU NO: {hasta_no}\n")
            dosya.write("-" * 40 + "\n")
            dosya.write(f"{sonuc}\n")
            dosya.write("=" * 60 + "\n\n")
            
        except Exception as e:
            hata_mesaji = f"BU HASTADA BİR HATA OLUŞTU VE ATLANDI: {str(e)}"
            print(hata_mesaji)
            dosya.write(f"HASTA BAŞVURU NO: {hasta_no}\n")
            dosya.write(f"{hata_mesaji}\n")
            dosya.write("=" * 60 + "\n\n")
            
        time.sleep(2)

print(f"\nİşlem başarıyla tamamlandı! Tüm sonuçlar klasörünüzdeki '{dosya_adi}' dosyasına kaydedildi.")