# Arduino Destekli Yüz Tanıma Sistemi

Python/Tkinter arayüzü, OpenCV YuNet + SFace derin öğrenme modelleri ve Arduino UNO ile çalışan yüz tanıma projesi.

Uygulama iki ana işlem yapar:

- **Kişi kaydı:** Kameradan otomatik 50 fotoğraf alır ve kişiyi eğitir.
- **Canlı tanıma:** Tanınan kişi için yeşil LED, bilinmeyen kişi için kırmızı LED ve buzzer uyarısı üretir.

## Özellikler

- **Derin öğrenme tabanlı tanıma:** OpenCV YuNet yüz tespiti + SFace embedding modeli.
- **Otomatik model indirme:** ONNX modelleri eksikse uygulama indirir.
- **Türkçe yol desteği:** ONNX modelleri Windows'ta ASCII güvenli klasöre alınır.
- **Arduino seri haberleşme:** 9600 baud üzerinden `G`, `R`, `O`, `B`, `T` komutları.
- **Bilinmeyen kişi uyarısı:** Yabancı yüz algılandığında kırmızı LED ve uzun buzzer uyarısı.
- **Bekleme modu:** Tanıma kapalıyken LED'ler kapalı kalır.
- **Veri artırma:** Eğitim sırasında ayna ve parlaklık varyasyonları kullanılır.

## Proje Yapısı

```text
.
├── app/
│   ├── app.py
│   └── requirements.txt
├── arduino/
│   └── face_recognition_arduino/
│       └── face_recognition_arduino.ino
└── README.md
```

## Donanım Bağlantıları

Bu sürüm kırmızı/yeşil LED mantığıyla sadeleştirilmiştir.

| Bileşen | Arduino pini |
|---|---|
| Buzzer (+) | D8 |
| Buzzer (-) | GND |
| LED ortak/en uzun bacak | +5V |
| Kırmızı LED bacağı | D9, 220Ω direnç ile |
| Yeşil LED bacağı | D7, 220Ω direnç ile |

> LED common anode ise `LOW = yanık`, `HIGH = sönük` mantığı kullanılır.

## Arduino Kurulumu

1. Arduino IDE ile `arduino/face_recognition_arduino/face_recognition_arduino.ino` dosyasını aç.
2. Kart olarak Arduino UNO seç.
3. Doğru COM portunu seç.
4. Kodu karta yükle.
5. Açılış testi sırasıyla kırmızı → yeşil → buzzer şeklinde çalışır.

## Python Kurulumu

Windows PowerShell:

```powershell
cd "c:\Users\PC6\Desktop\Yeni klasör\app"
py -3.13 -m pip install -r requirements.txt
py -3.13 app.py
```

Alternatif sanal ortam kurulumu:

```powershell
cd "c:\Users\PC6\Desktop\Yeni klasör\app"
py -3.13 -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## Kullanım

1. Arduino kodunu karta yükle.
2. Python uygulamasını başlat.
3. Üst menüden Arduino COM portunu seç ve **Bağlan** butonuna bas.
4. Sol panelden **Fotoğraf Çek ve Kaydet** ile kişiyi kaydet.
5. Model otomatik eğitilir.
6. Sağ panelden **Tanımayı Başlat** butonuna bas.

## Arduino Komutları

| Komut | Anlamı |
|---|---|
| `G` | Tanınan kişi: yeşil LED + kısa periyodik bip |
| `R` | Bilinmeyen kişi: kırmızı LED + uzun periyodik bip |
| `B` | Bekleme: LED ve buzzer kapalı |
| `O` | Tam kapatma |
| `T` | Açılış testini tekrar çalıştır |

## Tanıma Davranışı

- Tanınan kişi görülürse Arduino'ya `G` gönderilir.
- Bilinmeyen kişi görülürse Arduino'ya `R` gönderilir.
- Bilinmeyen kişi algılandıktan sonra kırmızı durum kısa süre korunur.
- Yeşile dönmek için aynı kayıtlı kişinin birkaç kare üst üste doğrulanması gerekir.

## Gereksinimler

Python paketleri `app/requirements.txt` içinde bulunur:

- `opencv-contrib-python`
- `numpy`
- `Pillow`
- `pyserial`

## Notlar

- İlk çalıştırmada YuNet ve SFace ONNX modelleri indirilebilir.
- Kayıtlı yüz fotoğrafları `app/faces/` altında tutulur.
- Embedding önbelleği `app/embeddings.pkl` dosyasında tutulur.
- Yüz fotoğrafları biyometrik veri olduğu için GitHub'a eklenmemesi önerilir.
