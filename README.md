# 🎾 Tenis Kortu Oyuncu Takip Sistemi

**Raspberry Pi üzerinde çalışan, gerçek zamanlı, yapay zeka kullanmadan tenis kortunda oyuncu takip ve konum belirleme sistemi.**

Sabit bir kameradan gelen görüntüde oyuncuyu bulur, takip eder ve kortun hangi bölgesinde olduğunu **metre cinsinden** raporlar. Sistem, tenis topu atma makinesinin üzerine monte edilmek üzere tasarlanmıştır — makine ileride oyuncunun konumuna göre nişan alacaktır.

> Pi **ekransız (headless)** çalışır. Kurulum ve canlı izleme **telefon tarayıcısından** yapılır — hiçbir uygulama kurmak gerekmez.

---

## 📺 Demo ve Test Videoları

Aşağıdaki videolar gerçek tenis kortunda, Raspberry Pi üzerinde kaydedilmiş ve arşivden aktarılmış takip test videolarıdır:

| Video | Algoritma / Koşu | Boyut |
|---|---|---|
| [`tracked_csrt_fast.mp4`](test_videolari/tracked_csrt_fast.mp4) | CSRT_FAST Segment 1 | ~35 MB |
| [`tracked_csrt_fast_2.mp4`](test_videolari/tracked_csrt_fast_2.mp4) | CSRT_FAST Segment 2 | ~35 MB |
| [`tracked_csrt_fast_3.mp4`](test_videolari/tracked_csrt_fast_3.mp4) | CSRT_FAST Segment 3 | ~35 MB |
| [`tracked_csrt_fast_4.mp4`](test_videolari/tracked_csrt_fast_4.mp4) | CSRT_FAST Segment 4 | ~35 MB |
| [`tracked_csrt_fast_7.mp4`](test_videolari/tracked_csrt_fast_7.mp4) | CSRT_FAST Kısa Demo | ~15 MB |
| [`tracked_mosse.mp4`](test_videolari/tracked_mosse.mp4) | MOSSE Yüksek Hız Testi 1 | ~35 MB |
| [`tracked_mosse_2.mp4`](test_videolari/tracked_mosse_2.mp4) | MOSSE Yüksek Hız Testi 2 | ~35 MB |

> Tüm videolar `test_videolari/` dizininde yer almaktadır ve Git LFS ile yönetilmektedir.

Her videoda şunları göreceksiniz:
- Korta projekte edilmiş **ızgara hücre numaraları** (perspektife uygun)
- Oyuncunun etrafındaki **takip kutusu**
- Sol üstte **HUD**: işleme FPS, oyun FPS, bölge numarası, metre koordinatı
- Sağ üstte **kuş bakışı mini kort paneli**: oyuncunun anlık konumu ve hareket izi

---

## 🧠 Nasıl Çalışır

### Büyük Resim

```
┌──────────────┐
│   KAMERA     │  USB / CSI / RTSP
└──────┬───────┘
       │  en yeni kare (eski kareler atılır, gecikme birikmez)
┌──────▼───────────────────────────────────────────────┐
│ 1. ARKA PLAN MODELİ                                  │
│    ~10 sn boyunca kare örneklenir                    │
│    Her piksel için medyan alınır                     │
│    → Oyuncusuz, temiz kort görüntüsü elde edilir     │
└──────┬───────────────────────────────────────────────┘
┌──────▼───────────────────────────────────────────────┐
│ 2. KORT TESPİTİ                                      │
│    Çizgi maskesi + Hough dönüşümü                    │
│    4 farklı aday görüntü denenir, en yüksek puanlı   │
│    seçilir; köşeler çizgi maskesine ince ayarlanır   │
│    Tutmazsa → telefondan elle ayarlama               │
└──────┬───────────────────────────────────────────────┘
┌──────▼───────────────────────────────────────────────┐
│ 3. HOMOGRAFİ                                         │
│    5 referans noktası → görüntü ↔ kort düzlemi       │
│    Artık her piksel metreye çevrilebilir             │
└──────┬───────────────────────────────────────────────┘
┌──────▼───────────────────────────────────────────────┐
│ 4. ANA DÖNGÜ (her kare)                              │
│    tespit → takip → ayak noktası → metre konum       │
│    → filtre → bölge → çizim → yayın                 │
└──────────────────────────────────────────────────────┘
```

---

### Adım 1 — Arka Plan Modeli (`background.py`)

Sistem açılışta ~10 saniye boyunca kare örnekler ve her piksel için **medyan** alır. Sonuç: oyuncunun hiç olmadığı, temiz kort fotoğrafı.

**Neden medyan, neden MOG2 değil?**  
MOG2 gibi uyarlanır modeller, duran bir oyuncuyu birkaç saniyede arka plana gömer — oyuncu görünmez olur. Medyan arka plan sabit kaldığı için bu sorunu yaşamaz.

**Arka plan nasıl tazelenir?**  
Işık değişimine uyum sağlamak için **her 30 karede bir** arka plan modeli güncellenir. Ancak tazeleme sırasında oyuncunun o anki kutusu **dışlanır** — aksi hâlde oyuncu yavaşça modele karışır.

---

### Adım 2 — Kort Tespiti ve Homografi (`court.py`)

**Otomatik tespit:**
1. Arka plan görüntüsü üzerinde çizgi maskesi çıkarılır (Canny + morfoloji)
2. Hough dönüşümü ile olası doğrular bulunur
3. Dört farklı aday görüntü (orijinal, açık, kapalı, vb.) denenir
4. Her aday için geometrik puanlama yapılır (doğruluk, dördüzlük, alan)
5. En yüksek puanlı aday seçilir
6. Köşeler, gerçek çizgi maskesine ICP benzeri bir yöntemle **ince ayarlanır**

**Elle ayarlama (otomatik başarısız olursa):**  
Telefon tarayıcısında 5 referans nokta görüntülenir. Kullanıcı noktaları **parmağıyla sürükler**, kort çizgileri **anlık güncellenir**. Parmağın altında büyüteç açılır. 90 saniye içinde onaylanmazsa otomatik tespit sonucuna devam edilir.

**Homografi:**  
Seçilen 5 referans noktası ile (`cv2.findHomography`) piksel ↔ metre dönüşümü kurulur. Referans noktaları file kesişimleri + yakın servis çizgisi kesişimleridir — uzak dip köşeler görüntüde çok sıkıştığı için kullanılmaz.

---

### Adım 3 — Oyuncu Tespiti (`detector.py`)

**Yöntem:** Klasik görüntü işleme. Yapay zeka yok.

**Boru hattı:**
```
kare − arka plan → mutlak fark → eşik → ön plan maskesi
→ morfoloji (açma + kapama) → bağlantılı bileşenler
→ alan / en-boy filtresi → ayak noktası doğrulaması → en büyük aday
```

**Önemli tasarım kararları:**

- **Neden renk modeli değil?**  
  Önceki sürüm kortun zemin rengine (LAB medyanı) uzaklığa bakıyordu. 20 karede **0 tespit** verdi: kortun dışındaki kırmızı alan da yeşilden uzak olduğundan maskenin %31'i ön plan sayılıyor, oyuncu bu dev blob'da kayboluyordu. Arka plan çıkarma her pikseli kendi geçmişiyle kıyaslar; tel örgü, bina, kırmızı zemin hepsi doğal olarak elenir. Aynı kare kümesinde 0/20 → 31/40 tespite yükseldi.

- **Gövde maskesi neden yukarı genişletilir?**  
  Uzak yarıda kort zemini görüntüde yalnızca ~40 piksel yüksekliğindedir; oyuncu ise dik durur ve gövdesi bu şeridin çok üstüne taşar. Zemin maskesiyle AND yapılınca yalnızca ayak bileği kalır — o da filtrelerden geçemez. Bu yüzden zemin maskesi, kamera geometrisinden türetilen bir değer kadar **yukarı doğru** genişletilir.

- **Ayak noktası doğrulaması:**  
  Her adayın alt-orta noktası homografiyle korta çevrilir. Oyuncunun beklenen yarısının dışında kalan aday atılır. Bu filtre olmadan tespitlerin yarısı rüzgârda sallanan ağaçtı.

- **Asimetrik padding:**  
  Bbox'ın alt kenarı (gerçek ayak noktası) kesilmez; yalnızca üst ve yanlarına padding eklenir. Alt kenara eklenen her 4 piksel uzak yarıda ~1,5 m konum hatasına dönüşüyordu.

---

### Adım 4 — Takip (`main.py` + `predictors.py`)

**Mevcut takipçiler:**

| Takipçi | Güçlü Yanı | Pi FPS |
|---|---|---|
| KCF | En yüksek AUC (0,721) | 38,1 |
| MOSSE | En hızlı | 187,6 |
| CSRT | En doğru kutu | 14,2 |
| CSRT_FAST | Hız/doğruluk dengesi | 33,7 |

**Mevcut hareket modelleri:**  
CV (Sabit Hız), CA (Sabit İvme), ABG (alfa-beta-gama), IMM (Etkileşimli Çoklu Model)

**Takip neden kaybolmuyor — üç emniyet katmanı:**

| Katman | Ne Yapar | Aralık |
|---|---|---|
| Yeniden senkron | Takip kutusunu tespitle hizalar | Her 18 kare |
| Sert yeniden tespit | Takipçiyi sıfırdan yeniden kurar | Her 45 kare |
| Arka plan tazeleme | Işık değişimine uyum sağlar | Her 30 kare |

**Sahne kesme tespiti (`shot.py`):**  
Video montajından veya kamera kesintisinden kaynaklanan sahneler tespit edilir. Kesme anında **dört şey birden sıfırlanır**: takipçi, Kalman filtresi, MOG2 modeli ve son konum. Aksi hâlde eski hız vektörü oyuncuyu yanlış yöne çeker ve "imkânsız sıçrama" denetimi sistemi kilitler.

---

### Adım 5 — Konum ve Bölge (`zones.py`)

**80 hücreli bölge şeması:**

```
       +-----------------------------------------------+
 LT1  |  25   26   27   28   29   30   31   32         |  RT1   ← üst bant (8×1)
 -----+-----------------------------------------------+-----
 LT2  |  17   18   19   20   21   22   23   24         |  RT2
 LT3  |   9   10   11   12   13   14   15   16         |  RT3   ← ana ızgara (8×3)
 LT4  |   1    2    3    4    5    6    7    8          |  RT4     alttan numaralanır
 -----+-----------------------------------------------+-----
 LB1  |  1x   2x   3x   4x   5x   6x   7x   8x        |  RB1
 LB2  |  9x  10x  11x  12x  13x  14x  15x  16x        |  RB2   ← alt bant (8×4)
 LB3  | 17x  18x  19x  20x  21x  22x  23x  24x        |  RB3     yukarıdan sayılır
 LB4  | 25x  26x  27x  28x  29x  30x  31x  32x        |  RB4
       +-----------------------------------------------+
```

Toplam: 24 (ana) + 8 (üst) + 32 (alt) + 16 (yan) = **80 hücre**

**Neden titremez — iki mekanizma:**

- **Medyan filtre (5 kare):** Ham ölçüm, kutu genişliği oynadıkça kare başına 0,5–1 m sıçrıyordu. 5 karenin medyanı ani sıçramaları eler. Kalman denendi fakat gürültü beyaz olmadığı için (ani sıçrama) medyan belirgin şekilde daha iyi çalıştı.
- **Bölge histerezisi (5 kare):** Yeni bölge kabul edilmeden önce 5 kare kararlı kalmalı. Bölge geçiş sayısı 21 → 9'a düştü (%57 azalma).

---

### Adım 6 — Canlı Yayın ve Arayüz (`mjpeg.py`)

- Telefon tarayıcısından `http://<pi-ip>:8080/` adresine gidilir, hiçbir uygulama gerekmez
- Görüntü **MJPEG** akışıyla yayınlanır; izleyici yoksa JPEG kodlaması **hiç çalışmaz** (maliyet sıfır)
- HUD: işleme FPS, video FPS, durum, bölge numarası, metre koordinatı
- Kuş bakışı mini kort paneli: aktif bölge vurgulu, hareket izi

---

## 📐 Ölçülen Başarım

İnsan etiketli yer gerçeğine karşı: **4 video, 9 segment, 540 kare** (sınırlayıcı kutu + elle tıklanmış ayak noktası + bölge etiketi), 144 değerlendirme koşusu.

### Takip Bileşeni

| Takipçi | AUC | Prec@20 | FPS (Pi) |
|---|---|---|---|
| **KCF** | **0,721** | 0,958 | 38,1 |
| MOSSE | 0,708 | **0,968** | **187,6** |
| CSRT | 0,706 | 0,941 | 14,2 |
| CSRT_FAST | 0,699 | 0,931 | 33,7 |

Hareket modelleri arasında istatistiksel olarak anlamlı fark yok (p = 0,07–0,78).

### Uçtan Uca Bölge Doğruluğu

| Ölçüm | Değer |
|---|---|
| Bölge doğruluğu (tüm sınırlar dahil) | %64,6 |
| Bölge doğruluğu (sınır belirsizliği hariç) | %71,1 |
| Rastgele tahmin karşılaştırması | %2,5 (80 hücrede) |

### Hata Analizi — Her Basamak Ayrı Ayrı

| Basamak | Doğruluk |
|---|---|
| Gerçek sistem | %57,2 |
| Kusursuz takip varsayımı | %60,4 |
| İnsan işaretli ayak noktası | %94,4 |

**Kritik bulgu:** Kaybın **3,1 puanı takipten**, **34,1 puanı "ayak noktası = kutunun alt-orta noktası" varsayımından** geliyor. Takipçiyi iyileştirmenin getirisi ihmal edilebilir; asıl kazanç **gerçek ayak noktası kestiriminden** gelecek.

### Perspektif Kısalması Etkisi

| Yön | Hata | Neden |
|---|---|---|
| Yanal (sol–sağ) | **22 cm** | Piksel başına ~0,016 m/px |
| Derinlik (uzak–yakın) | **92 cm** | Piksel başına ~0,279 m/px (17 kat kaba) |

Hata ∝ mesafe² / (odak × kamera yüksekliği) bağıntısıyla modellenmiş ve dört videonun ölçümlerine **%0,1–1,0** bağıl hatayla oturmuştur.

### Kalibrasyon ve Etiket Güvenilirliği

| Ölçüm | Değer |
|---|---|
| Yeniden projeksiyon hatası (medyan, fit dışı 12 nokta) | 3,5–4,0 piksel |
| Bölge etiket uyumu (Cohen's κ) | 0,884 ("neredeyse mükemmel") |
| Ayak noktası etiket tutarsızlığı | 32 cm |
| Gürültüden arındırılmış sistem hatası | 0,86 m |

---

## 🗂️ Dosya Yapısı

| Dosya | Satır | Görevi |
|---|---:|---|
| [`main.py`](main.py) | 1 306 | Ana döngü, kurulum sırası, çizim (modüler sürüm) |
| [`tenis_takip.py`](tenis_takip.py) | ~3 000 | **Tek dosya (All-in-One)** tüm boru hattı — bağımsız çalıştırma için |
| [`benchmark.py`](benchmark.py) | 181 | PC vs Raspberry Pi performans & aşama kıyaslama aracı |
| [`config.py`](config.py) | ~800 | Tüm ayarlar (gerekçeleriyle) |
| [`court.py`](court.py) | 964 | Kort tespiti, homografi, köşe ince ayarı |
| [`mjpeg.py`](mjpeg.py) | 604 | MJPEG yayını + telefon kurulum sayfası |
| [`zones.py`](zones.py) | 401 | 80 hücreli bölge şeması, mini kort paneli |
| [`detector.py`](detector.py) | 329 | Arka plan çıkarma ile oyuncu tespiti |
| [`livesource.py`](livesource.py) | 279 | Kamera soyutlaması (USB / CSI / RTSP / dosya) |
| [`predictors.py`](predictors.py) | 242 | CV / CA / ABG / IMM hareket modelleri |
| [`performance.py`](performance.py) | 200 | Aşama bazında FPS profilleme |
| [`analysis.py`](analysis.py) | 179 | Bölge geçiş kaydı ve özeti |
| [`background.py`](background.py) | 140 | Medyan arka plan modeli + tazeleme |
| [`framereader.py`](framereader.py) | 80 | İş parçacıklı dosya okuma |
| [`shot.py`](shot.py) | 64 | Sahne kesme tespiti |
| [`tools/compare.py`](tools/compare.py) | 345 | Yer gerçeği (GT) ile takip başarım/hata metrik analizi |
| [`tools/eval_runner.py`](tools/eval_runner.py) | 150 | Otomatik çoklu video/takipçi değerlendirme matrisi |
| [`tools/fps_bench.py`](tools/fps_bench.py) | 200 | Detaylı FPS profilleme aracı |
| [`tools/_matrix_run.py`](tools/_matrix_run.py) | 37 | 4 takipçi × 4 tahmin modeli toplu matris koşturucu |
| [`tools/_matrix_iso.py`](tools/_matrix_iso.py) | 50 | İzole ortam matris test betiği |
| [`court_corners.json`](court_corners.json) | - | Varsayılan kalibrasyon köşe koordinatları |
| [`test_videolari/`](test_videolari/) | - | Arşivden aktarılan 9 test videosu + `matrix/` & `matrix_iso/` sonuçları |
| [`veri/`](veri/) | - | 144 adet GT değerlendirme verisi (`eval_gt/`), referans etiketleri, indeksler ve overlayler |

---

## ⚙️ Kurulum ve Çalıştırma

### Gereksinimler

- Raspberry Pi 4 (veya daha yenisi önerilir) ya da x86_64 PC
- Python 3.11+
- OpenCV 4.x (contrib ile — CSRT ve KCF için)
- USB kamera, CSI şerit kablolu Pi kamerası veya RTSP ağ kamerası

### Kurulum

```bash
# Depoyu klonla
git clone https://github.com/purkusur-pixel/tenis-sahasi.git
cd tenis-sahasi

# Sanal ortam oluştur ve bağımlılıkları yükle
python3 -m venv venv
venv/bin/pip install opencv-contrib-python numpy

# Modüler ana sürümü çalıştır
venv/bin/python main.py

# VEYA tek dosya (stand-alone) sürümü çalıştır
venv/bin/python tenis_takip.py

# Performans kıyaslaması (benchmark) çalıştırmak için:
venv/bin/python benchmark.py 200 CSRT_FAST
```

### Açılışta Otomatik Başlatma (systemd)

```bash
sudo cp staj-takip.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable staj-takip
sudo systemctl start staj-takip
```

Günlükleri izlemek için:

```bash
journalctl -u staj-takip -f
```

---

## 📱 Telefondan Kullanım

1. **Pi'ye elektrik ver**, ~40 saniye bekle
2. **Pi'nin açtığı WiFi ağına bağlan** (veya aynı ağdaysan Pi'nin IP'sini bul)
3. Tarayıcıda `http://10.42.0.1:8080/` adresini aç

### Kurulum Ekranı

İlk açılışta **kort kurulum ekranı** gelir:
- Kort görüntüsü üzerinde **5 referans nokta** görünür
- Projekte edilmiş kort çizgileri anlık güncellenir

**Çizgiler kortla örtüşüyorsa → "Onayla"**  
**Örtüşmüyorsa →** parmağınla noktayı sürükle. Parmağın altındaki bölge büyüteçte gösterilir. Onaylamazsan ve otomatik tespit tutmuşsa **90 saniye sonra otomatik devam edilir.**

Onaydan sonra aynı adres **canlı yayına** döner.

---

## 🔧 Önemli Ayarlar

Tüm ayarlar [`config.py`](config.py) içinde, gerekçeleriyle birlikte yorumlanmıştır.

| Ayar | Varsayılan | Açıklama |
|---|---|---|
| `LIVE_MODE` | `True` | `False` → dosyadan çalışır |
| `LIVE_SOURCE` | `0` | USB indeksi / `"picamera2"` / RTSP adresi |
| `TRACKER_TYPE` | `CSRT_FAST` | KCF, MOSSE, CSRT, CSRT_FAST |
| `PREDICT_TYPE` | `CV` | CV, CA, ABG, IMM |
| `PROC_SCALE` | `0.6` | İşleme ölçeği — FPS için düşür |
| `SAVE_OUTPUT` | `True` | Video kaydı (~2,5–3 GB/saat) |
| `STREAM_PORT` | `8080` | Telefon arayüzü portu |
| `SETUP_ON_PHONE` | `True` | Telefondan kort ayarlama |
| `PLAYER_HALF` | `"far"` | Oyuncunun beklenen yarısı: `"far"`, `"near"`, `"both"` |

---

## ⚠️ Bilinen Sınırlılıklar

- **Otomatik kort tespiti her açıda çalışmıyor.** Test edilen 4 videodan birinde Hough çizgiler yakın servis çizgisini bulamadı; üretilen dörtgenler dışbükey çıkmadı. Telefondan elle ayarlama bu durumu kurtarıyor.
- **Ayak noktası bir vekil.** Kutunun alt-orta noktası kullanılıyor. Ölçümlere göre hatanın **baskın kaynağı** bu (%57 → %94 tavan). Ön plan maskesinden veya iskelet kestiriminden gerçek ayak noktası çıkarmak en yüksek getirili iyileştirme olur.
- **Derinlik doğruluğu kamera geometrisine bağlı.** Kamerayı yükseltmek veya odak uzaklığını artırmak hatayı doğrusal azaltır; korta yaklaştırmak (mesafe² terimi) en güçlü etkiyi verir.
- **Sabit kamera varsayımı.** Kamera bir çalışma boyunca kıpırdamamalı. Makineye monte edilirken dönen kafaya değil, sabit gövdeye takılmalıdır.
- **Tek oyuncu, tek kort.** Veri kümesi 4 video ile sınırlı.
- **Top makinesi tespiti yarım.** `DETECT_MACHINE` kodu var, ancak doğrulanmadı ve yanlış pozitif verebilir.

---

## 🚀 Sonraki Adımlar (Önerilen)

1. **Gerçek ayak noktası kestirimi** — ön plan maskesi veya MediaPipe iskelet ile (%57 → %94 tavan)
2. **Top tespiti** — kırmızı / sarı yuvarlak blob veya hafif bir YOLO modeli
3. **Makine nişan kontrolü** — konum verisini makine kontrol kartına aktar
4. **Çoklu oyuncu** — iki takipçi, iki ayrı bölge güncellemesi

---

## 📊 Çıktılar

Program çalışırken [`output/`](output/) klasörüne şunları yazar:

| Dosya | İçerik |
|---|---|
| `tracked_*.mp4` | Overlay'li kayıt videosu |
| `first_frame_court_detection.jpg` | Kalibrasyon anının ekran görüntüsü |
| `zone_events.csv` | Tüm bölge geçişleri (zaman damgası, bölge, koordinat) |
| `zone_summary.json` | Her bölgenin toplam süre ve ziyaret sayısı |
| `eval_*.csv` | Takipçi değerlendirme sonuçları |
| `_zone_map.jpg` | Statik bölge haritası |
| `_zones_preview.jpg` | 80 hücreli perspektif bölge ızgarası görsel önizlemesi |
| `_A1_court.jpg` | Örnek kort kalibrasyonu ve köşe çizgileri görseli |

---

*Raspberry Pi staj projesi — 2025*
