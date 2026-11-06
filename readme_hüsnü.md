# Tenis Kortu Oyuncu Takip Sistemi


## VAR


**Kort**
- Otomatik kort tespiti (4 aday görüntü, en iyisi seçilir)
- Köşe ince ayarı (çizgi maskesine hizalama)
- Telefondan elle ayarlama: nokta sürükleme + anlık çizgi güncelleme + büyüteç
- Her açılışta sıfırdan kalibrasyon

**Takip**
- 4 takipçi: KCF, MOSSE, CSRT, CSRT_FAST(kullanılan algoritma)
- 4 hareket modeli: CV(varsayılan), CA, ABG, IMM
- Periyodik yeniden senkron (18 kare) ve sert yeniden tespit (45 kare)
- Arka plan tazeleme (oyuncu kutusu dışlanarak)
- Medyan filtre + bölge histerezisi
- Sahne kesme tespiti
- Oyuncu-yarısı kısıtı

**Bölge**
- Kuş bakışı mini kort paneli + iz
- Bölge geçiş kaydı: CSV + JSON (süre, ziyaret sayısı, konum)

**Arayüz**
- MJPEG canlı yayın (telefon tarayıcısı, uygulama gerekmez)
- HUD: işleme FPS, video FPS, durum, bölge, metre konum
- Overlay + video kaydı

**Çalıştırma**
- systemd servisi, açılışta otomatik başlar
- Kamera yoksa 15 sn'de bir yeniden dener
- Aşama bazında FPS profilleme

---

## YOK

- Top atma / makine kontrolü
- Top tespiti ve takibi
- Çoklu oyuncu
- Hareketli kamera desteği
- Gerçek ayak noktası kestirimi (kutunun alt-orta noktası vekil olarak kullanılıyor)
- Vuruş / skor / oyun analizi
- Kayıtların otomatik numaralanması (bölge kayıtları her çalıştırmada üzerine yazılır)
- Bölge kaydının anlık diske yazılması (yalnızca program kapanırken yazılır)
- Çalışan top makinesi tespiti (kod var, doğrulanmadı)
- Kort tespitinin her açıda çalışması (4 videonun 1'inde başarısız)

---

## Kullanım

```bash
cd ~/staj5.2 && venv/bin/python main.py
```

Telefondan: Pi'nin WiFi ağına bağlan → `http://10.42.0.1:8080/`

Açılışta otomatik başlatma:

```bash
sudo cp staj-takip.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable staj-takip
```

---

## Ölçülen değerler

540 insan etiketli kare, 4 video, 144 değerlendirme koşusu.

| | Değer |
|---|---|
| En iyi takipçi | KCF, AUC 0,721 |
| En hızlı | MOSSE, 187,6 FPS |
| Bölge doğruluğu | %64,6 (sınır hariç %71,1) |
| Yanal hata | 0,22 m |
| Derinlik hatası | 0,92 m |
| Kalibrasyon hatası | 3,5–4,0 piksel |
| Etiket uyumu | κ = 0,884 |
| Çalışma hızı (Pi) | ~24 FPS |

Bence saha denemesinde kcf deneyin
---

## Ayarlar

`config.py` içinde. Sık kullanılanlar:

| Ayar | Varsayılan |
|---|---|
| `LIVE_MODE` | `True` |
| `LIVE_SOURCE` | `0` (USB indeksi) |
| `TRACKER_TYPE` | `CSRT_FAST` |
| `PREDICT_TYPE` | `CV` |
| `PROC_SCALE` | `0.6` |
| `SAVE_OUTPUT` | `True` (~2,5–3 GB/saat) |
| `STREAM_PORT` | `8080` |
