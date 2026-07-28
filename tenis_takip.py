"""
tenis_takip.py
==============
Tek oyunculu tenis kortu takip sistemi - TEK DOSYA surumu.

Calistirmak icin:  python tenis_takip.py
Gereken:           pip install opencv-contrib-python numpy
Video yolu:        BOLUM 1'deki VIDEO_PATH

AKIS:
    1) Video acilir, ilk kare PROC_SCALE ile olceklenir.
    2) Kortun 5 referans noktasi (file + yakin servis cizgisi kesisimleri):
       kayitli dosya -> OTOMATIK tespit -> elle secim. Ilk kare + tespit
       output/'a kaydedilir (bkz. save_first_frame_debug).
    3) Homography kurulur; izgara, overlay katmani ve panel BIR KEZ hazirlanir.
    4) Medyan arka plan modeli kurulur (videoya yayilmis karelerin piksel medyani).
    5) Oyuncu arka plan cikarma ile tespit edilir; CSRT tracker baslatilir.
    6) Her karede once SAHNE KESME kontrol edilir, sonra tracker guncellenir.
    7) MOG2 SADECE bbox icindeki hareket enerjisini olcmek icin okunur.
    8) Ayak noktasi homography ile kort koordinatina cevrilir.
    9) Kalman yumusatir; bolge (get_zone) saf aritmetikle belirlenir.
   10) Izgara overlay'i + aktif hucre vurgusu + kus bakisi panel cizilir.

KONTROLLER:
    q      : cikis
    BOSLUK : duraklat / devam
    r      : elle yeniden tespit

BIRLESTIRME NOTU:
    Bu dosya 7 bolumden olusur (eski moduller). Tek isim alaninda birlestikleri
    icin AYNI ADA sahip iki fonksiyon cifti yeniden adlandirilmistir:
        zones.build      -> zones_build          (background.build ile cakisiyordu)
        background.build -> background_build
        detector.init    -> detector_init        (shot.init ile cakisiyordu)
        shot.init        -> shot_init
    Baska hicbir davranis degismemistir.
"""

import csv
import itertools
import json
import os
import time
from collections import deque

import cv2
import numpy as np

# ==============================================================================
# BOLUM 1 - AYARLAR (eski BOLUM 1 (AYARLAR))
# ==============================================================================
# BOLUM 1 (AYARLAR)
# ---------
# Tenis oyuncusu takip sisteminin tum ayarlari.
#
# Bu dosya sadece sabit tanimlar icerir; hicbir calisma zamani mantigi barindirmaz.
# Diger tum moduller (BOLUM 2 (KORT), BOLUM 4 (TESPIT), BOLUM 5 (BOLGELER), BOLUM 6 (KESME), BOLUM 7 (ANA DONGU)) ayarlari
# buradan okur. Tuning yaparken yalnizca bu dosyaya dokunmak yeterlidir.
#
# Koordinat sistemi (dunya / kort duzlemi, metre):
#     x -> 0.0 ... COURT_WIDTH   (soldan saga)
#     y -> 0.0 ... COURT_LENGTH  (uzak dip cizgi -> yakin dip cizgi)

 
# ==============================================================================
# 1) VIDEO / GENEL CALISMA AYARLARI
# ==============================================================================
 
VIDEO_PATH = "input/C01.mp4"   # Webcam icin 0 (int).
 
PROC_SCALE = 0.6   # Tum boru hattinin calistigi kare olcegi. FPS icin dusur.
FRAME_SKIP = 1     # 1 -> her kare islenir.

# Kare KUCULTME interpolasyonu. INTER_AREA en kaliteli ama YAVAS; olculdu:
#     1920x1080 -> 1152x648 : AREA 6.83 ms  vs  LINEAR 2.54 ms
#     1152x648  -> MOG2 girdi: AREA 4.26 ms  vs  LINEAR 0.62 ms
#     -> toplam ~7.9 ms/kare kazanc (48.2 ms butcede %16).
# Kalite farki bu boru hatti icin onemsiz: ortalama piksel farki 0.78/255 ve
# BG_DIFF_THRESH(=24) esigini asan piksel orani yalnizca %0.096.
#
# TEK YERDE TANIMLI OLMASI SART: arka plan modeli (_scaled) ile ana dongudeki
# kareler AYNI interpolasyonu kullanmak zorunda. Farkli olurlarsa model ile
# kareler sistematik olarak farkli uretilir ve her karede SAHTE foreground cikar.
RESIZE_INTERP = cv2.INTER_LINEAR
 
SHOW_WINDOW = False   # Raspberry Pi (headless): anlik pencere YOK. imshow hic cagrilmaz,
                      # dongu tam hizda calisir. Cikti SAVE_OUTPUT ile dosyaya yazilir.
SHOW_SEEKBAR = True   # Video penceresine YouTube tarzi kaydirma cubugu ekler. Yalnizca
                      # SEEK edilebilir DOSYA kaynaginda VE SHOW_WINDOW acikken kurulur.
 
SAVE_OUTPUT = True              # Islenmis (overlay + panel + HUD) videoyu diske yaz.
OUTPUT_PATH = "output/tracked.mp4"   # Klasor yoksa otomatik olusturulur.

# Cikti videosunun FPS'i. None -> KAYNAK videonun FPS'inden otomatik hesaplanir
# (src_fps / FRAME_SKIP), boylece cikti GERCEK ZAMANLI hizda oynar.
# NEDEN OTOMATIK: sabit 30.0 idi, ama kaynaklar farkli (C01=30, match=25,
# C0004=50). 50 FPS'lik kaynagi 30 FPS olarak yazmak videoyu ~1.7x AGIR CEKIM
# yapiyordu. Elle sabitlemek istersen buraya sayi yaz.
OUTPUT_FPS = None

# Ilk karenin uzerine kort cizgileri + numarali/renkli referans noktalari
# cizilip output/'a kaydedilir mi? SHOW_WINDOW=False (Pi, headless) oldugunda
# ekranda hicbir gorsel geri bildirim olmadigindan, kort tespitinin dogru
# calisip calismadigini kontrol etmenin tek yolu budur. Tek seferlik (kurulum
# aninda) calistigi icin performansa etkisi yoktur.
SAVE_FIRST_FRAME_DEBUG = True
FIRST_FRAME_DEBUG_PATH = "output/first_frame_court_detection.jpg"

CORNERS_PATH = "court_corners.json"
 
KEY_QUIT = ord("q")
KEY_PAUSE = ord(" ")
KEY_REDETECT = ord("r")
 
# ==============================================================================
# 2) KORT MODELI (gercek dunya olculeri, metre)
# ==============================================================================
 
COURT_WIDTH = 10.97
COURT_LENGTH = 23.77
 
SINGLES_LEFT_X = 1.37
SINGLES_RIGHT_X = 9.60
CENTER_X = COURT_WIDTH / 2.0
NET_Y = COURT_LENGTH / 2.0
SERVICE_LINE_DIST = 6.40
FAR_SERVICE_Y = NET_Y - SERVICE_LINE_DIST
NEAR_SERVICE_Y = NET_Y + SERVICE_LINE_DIST
 
# Homography referans noktalari: FILE + YAKIN SERVIS CIZGISI kesisimleri (5 nokta).
#
# NEDEN ARTIK UZAK DIP KOSELER DEGIL:
#     Uzak dip kose kameraya EN UZAK noktadir; perspektif nedeniyle goruntude
#     birkac piksellik bir alana sikisir (olcum: eski court_corners.json
#     kaydinda iki uzak kose arasi yatayda ~380 px iken dikeydeki fark sadece
#     ~8 px idi). Boyle sikisik bir bolgede 1-2 pikselik bir tespit/tiklama
#     hatasi, kort duzlemine geri izdusurulunce METRE bazinda ciddi bir sapmaya
#     donusur (homography'nin bu bolgedeki hassasiyeti/turevi cok buyuktur).
#     Yakin dip cizgi de guvenilir DEGILDIR (cogu cekimde kadraj disinda kalir
#     / oyuncu-golge tarafindan kapanir). Bu yuzden capa artik SADECE kameraya
#     orta-yakin mesafede, HER ZAMAN net ve tam gorunen file ile yakin servis
#     cizgisi bandindan (NET_Y .. NEAR_SERVICE_Y) seciliyor.
#
# YENI CAPA NOKTALARI (sira SABITTIR; CORNER_LABELS ile birebir eslesir):
#     1) FILE      - SOL  ciftler yan cizgisi kesisimi : (0, NET_Y)
#     2) FILE      - SAG  ciftler yan cizgisi kesisimi : (COURT_WIDTH, NET_Y)
#     3) FILE      - orta servis cizgisi kesisimi      : (CENTER_X, NET_Y)
#     4) YAKIN SERVIS - SOL tekler yan cizgisi kesisimi : (SINGLES_LEFT_X, NEAR_SERVICE_Y)
#     5) YAKIN SERVIS - SAG tekler yan cizgisi kesisimi : (SINGLES_RIGHT_X, NEAR_SERVICE_Y)
#
#     Noktalar (1,2,3) dunyada es dogrusaldir (ucu de file uzerinde, y=NET_Y)
#     ama BAGIMSIZ olarak tespit edilir (orta servis cizgisi ile file'in
#     kesisimi); bu, en-kucuk-kareler (least-squares) uydurmasina fazladan bir
#     tutarlilik kisiti ekler ve TEK bir noktadaki hatanin butun homography'yi
#     suruklemesini engeller (bkz. compute_homography). Uzak dip koseler ARTIK
#     ana capa olarak KULLANILMAZ; sadece auto_detect_corners icindeki ROBUST
#     ic hesaplamada (yakin dip + yakin servis + cift yan cizgileri ile kurulan
#     yardimci homography) rol oynarlar, disariya hic verilmezler.
#
# TAM KORT NASIL OLUSUYOR:
#     Homography bir DUZLEMIN tamamini esler, yalnizca secilen noktalari degil.
#     Bu 5 capa yer duzlemini tanimlar; hem uzak yari (y -> 0 .. NET_Y) hem
#     yakin dip (y -> NEAR_SERVICE_Y .. COURT_LENGTH) matematiksel olarak
#     ekstrapole edilir. Izgara bu yuzden eskisi gibi TAM kortu (0 ..
#     COURT_LENGTH) kaplamaya devam eder -> BOLUM 5 (BOLGELER)'de hicbir
#     degisiklik gerekmez, bolge semasi birebir korunur.
COURT_CORNERS_WORLD = np.array(
    [
        [0.0, NET_Y],                       # 1) File - SOL ciftler yan cizgisi
        [COURT_WIDTH, NET_Y],               # 2) File - SAG ciftler yan cizgisi
        [CENTER_X, NET_Y],                  # 3) File - orta servis cizgisi
        [SINGLES_LEFT_X, NEAR_SERVICE_Y],   # 4) Yakin servis - SOL tekler yan cizgisi
        [SINGLES_RIGHT_X, NEAR_SERVICE_Y],  # 5) Yakin servis - SAG tekler yan cizgisi
    ],
    dtype=np.float32,
)

CORNER_LABELS = (
    "1/5: FILE - SOL ciftler yan cizgisi kesisimi",
    "2/5: FILE - SAG ciftler yan cizgisi kesisimi",
    "3/5: FILE - orta servis cizgisi kesisimi",
    "4/5: YAKIN SERVIS - SOL tekler yan cizgisi kesisimi",
    "5/5: YAKIN SERVIS - SAG tekler yan cizgisi kesisimi",
)

# Referans noktasi basina sabit, birbirinden ayirt edilebilir renk (BGR). Sira
# CORNER_LABELS ile birebir eslesir; ilk kare debug goruntusunde ve onay
# penceresinde "hangi nokta hangisi" karisikligini onlemek icin kullanilir.
REF_POINT_COLORS = (
    (0, 0, 255),      # 1: file sol         - kirmizi
    (0, 255, 0),      # 2: file sag         - yesil
    (255, 0, 255),    # 3: file orta        - magenta
    (255, 255, 0),    # 4: yakin servis sol - cyan
    (0, 165, 255),    # 5: yakin servis sag - turuncu
)

# Konum saglik denetimi toleransi (metre). Bolge bantlarindan buyuk olmali.
COURT_MARGIN = 9.0  # Alt bant derinligini (BOTTOM_BAND_DEPTH) kapsamalidir.

# ==============================================================================
# 2b) OTOMATIK KORT TESPITI (auto_detect_corners)
# ==============================================================================
# Kayitli kose yoksa once otomatik denenir; basarisiz olursa elle secime dusulur.

AUTO_DETECT_CORNERS = True
AUTO_CONFIRM = True       # Bulunan kortu onaya sun (ENTER: kabul, m: elle sec).

# --- Cizgi maskesi ---
LINE_NEIGHBOR_DIST = 4    # Cizginin iki yanina bakilan mesafe (px).
LINE_DARK_DELTA = 18      # Cizgi, yanlarindan en az bu kadar parlak olmali.
LINE_MIN_BRIGHT = 120     # Mutlak parlaklik alt siniri.
LINE_MIN_LENGTH = 40      # Bilesenin (dondurulmus kutu) uzun kenari en az (px).
LINE_MAX_THICK = 10       # Bilesenin kisa kenari en fazla (px) -> ince yapi.
LINE_MIN_WEIGHT = 90      # Birlesik dogrunun toplam segment uzunlugu en az (px).
LINE_HORIZ_MAX_ANGLE = 8.0  # Bu acinin altindakiler "yatay" (dip/servis cizgisi).

AUTO_LINE_TOL = 4.0       # Model cizgisi beyaz piksele bu kadar yakinsa "isabet".
AUTO_MIN_SCORE = 0.45     # Bu puanin altinda otomatik tespit reddedilir.
AUTO_RATIO_TOL = 0.03     # tekler/cift genislik orani (0.750) tolerans.
 
# ==============================================================================
# 3) BOLGE IZGARASI (overlay_rect.py semasi - birebir)
# ==============================================================================
#   Ana izgara  : GRID_COLS x GRID_ROWS  -> 1..24   (numaralar ALTTAN sayilir)
#   Ust bant    : GRID_COLS x TOP_BAND_ROWS    -> 25..32
#   Alt bant    : GRID_COLS x BOTTOM_BAND_ROWS -> 1x..32x (YUKARIDAN sayilir)
#   Yan seritler: her satir bandi icin bir dilim
#                 -> LT1..LT4 / LB1..LB4 / RT1..RT4 / RB1..RB4
#   Disarisi    : OUT
#   Toplam: 24 + 8 + 32 + 16 = 80 hucre
 
GRID_COLS = 8
GRID_ROWS = 3
 
TOP_BAND_ROWS = 1     # Uzak dip cizgi arkasi satir sayisi.
BOTTOM_BAND_ROWS = 4  # Yakin dip cizgi arkasi satir sayisi (1x..32x -> 4 satir).
 
# Bant derinlikleri (metre, TOPLAM; satir sayisina bolunur).
# overlay_rect.py'de bunlar ORANSAL idi (1/ROWS), yani alt bant kortun 4/3'u
# kadar (~31.7 m) uzuyordu. Ekran dortgeninde sorun degil, tenis kortunda o alan
# tribun arkasidir. Yapi ve etiketler ayni; sadece derinlik metrik hale getirildi.
# Birebir orijinal oran icin:
#   TOP_BAND_DEPTH    = COURT_LENGTH * TOP_BAND_ROWS    / GRID_ROWS   (= 7.92)
#   BOTTOM_BAND_DEPTH = COURT_LENGTH * BOTTOM_BAND_ROWS / GRID_ROWS   (= 31.69)
TOP_BAND_DEPTH = 2.50
BOTTOM_BAND_DEPTH = 8.00
 
# Yan serit genisligi (metre). Orijinalde bir sutun genisligi kadardir:
#   COURT_WIDTH / GRID_COLS = 1.371
SIDE_BAND_WIDTH = COURT_WIDTH / GRID_COLS
 
SHOW_TOP_BAND = True
SHOW_BOTTOM_BAND = True
SHOW_SIDE_BANDS = True
 
ZONE_OUT = "OUT"
 
# ==============================================================================
# 4) ILK TESPIT (medyan arka plan cikarma; AI yok)
# ==============================================================================
# Onceki surum kort renginin LAB medyanina uzakliga bakiyordu; bu kamerada HIC
# calismadi (20 karede 0 tespit). Kirmizi run-off alani kort yesilinden uzak
# oldugu icin maskenin ~%31'ini dolduruyor, oyuncuyu yutuyordu. Arka plan cikarma
# her pikseli KENDI gecmisiyle karsilastirir -> ayni karelerde 31/40 ve kutularin
# tamami dogru boyutta. Ayrinti: BOLUM 3 (ARKA PLAN)
 
DET_SCALE = 0.5   # Yalnizca blob alan esiklerinin olceklenmesinde kullanilir.
 
# Oyuncunun arandigi kort yarisi. "far" -> kameradan UZAK yari (y: 0 .. NET_Y).
# build_search_mask() bunu okuyup tespit ROI'sini kurar.
PLAYER_HALF = "far"     # "near" | "far" | "both"
SEARCH_MARGIN = 3.0     # Arama alanina kort disindan eklenen pay (metre).
 
# --- Medyan arka plan modeli ---
BG_SAMPLES = 25         # Medyan icin videoya yayilmis ornek kare sayisi.
BG_STRIDE = 12          # Webcam / kare sayisi bilinmiyorsa kac karede bir ornek.
BG_BLUR = (5, 5)        # Fark almadan once gurultu bastirma.
BG_DIFF_THRESH = 24     # Arka plandan bu kadar sapan piksel foreground sayilir.
 
# --- Govde maskesi (zemin poligonunun goruntude YUKARI genisletilmesi) ---
# Oyuncu zeminde durur ama goruntude DIK uzanir; zemin poligonu govdeyi kapsamaz.
# Genisletme miktari geometriden turetilir (_rise_px), sabit degildir.
PLAYER_HEIGHT_M = 2.0
BG_RISE_FALLBACK = 80   # Olcek hesaplanamazsa (projeksiyon basarisiz) kullanilir.
 
# --- Ayak noktasi dogrulamasi ---
# Govde maskesi yukari genisledigi icin arkadaki agac/tel hareketi de iceri girer.
# Adayin ALT-ORTA noktasi istenen yarinin bu kadar disindaysa atilir (metre).
FOOT_VALIDATE_MARGIN = 1.5
 
MORPH_OPEN_KERNEL = (3, 3)
MORPH_CLOSE_KERNEL = (7, 21)   # Uzun dikey cekirdek: parcalanmis govdeyi birlestirir.
MORPH_ITERATIONS = 1
 
# PLAYER_HALF = "far" iken oyuncu perspektif geregi piksel olarak KUCUK gorunur.
# Tespit hic tutmazsa ilk dusurulecek knob MIN_BLOB_AREA_RATIO'dur (or. 0.00012).
MIN_BLOB_AREA_RATIO = 0.00025
MAX_BLOB_AREA_RATIO = 0.06
 
MIN_ASPECT_RATIO = 0.9
MAX_ASPECT_RATIO = 5.0
 
BBOX_PADDING = 4
 
# ==============================================================================
# 5) TRACKER (tipi TRACKER_TYPE ile secilir)
# ==============================================================================
 
# --- TAKIPCI SECIMI (FPS DARBOGAZI) -----------------------------------------
# CSRT en DOGRU ama EN YAVAS takipcidir; dusuk FPS'in sebebi buydu (takip
# kesilince FPS 8 -> 25'e ciktigi icin darbogaz oldugu kesinlesti).
# Olculen update maliyeti (1152x648 kare, uzak yari oyuncu bbox'i):
#     "CSRT"       ~88.8 ms/kare  (~11 FPS tavani)   <- eski varsayilan
#     "CSRT_FAST"  ~20.5 ms/kare  (~49 FPS)          <- CSRT, olcek/segmentasyon kisilmis
#     "KCF"        ~16.5 ms/kare  (~61 FPS)          <- varsayilan (dengeli)
#     "MOSSE"      ~ 0.6 ms/kare  (~1541 FPS)        <- Pi icin en hizli; OLCEK YOK
#
# NEDEN HAFIF TAKIPCI YETERLI: hibrit resync (TRACK_RESYNC_INTERVAL) kutuyu her
# ~18 karede tespitle yeniden senkronluyor. Yani takipcinin kendi PAHALI olcek
# aramasina gerek yok; takipci sadece iki resync arasinda konumu tutar, olcegi ve
# kaymayi resync duzeltir. Takipci hedefi kaybederse mevcut redetect agi toparlar.
TRACKER_TYPE = "KCF"    # "CSRT" | "CSRT_FAST" | "KCF" | "MOSSE"

# True -> program baslarken hangi takipcinin kullanilacagi SORULUR; ENTER'a basmak
# yukaridaki TRACKER_TYPE'i secer. Pi'de otomatik baslatma (servis/autostart) icin
# False yap: soru sorulmaz, dogrudan TRACKER_TYPE kullanilir.
# NOT: stdin yoksa (servis olarak calisirken) soru zaten atlanir, program kilitlenmez.
ASK_TRACKER = True

# CSRT_FAST parametreleri (YALNIZCA TRACKER_TYPE = "CSRT_FAST" iken kullanilir).
CSRT_FAST_SCALES = 11           # varsayilan 33 -> pahali olcek aramasi kisilir
CSRT_FAST_ADMM = 2              # varsayilan 4  -> filtre optimizasyon adimi
CSRT_FAST_TEMPLATE = 120.0      # varsayilan 200
CSRT_FAST_SEGMENTATION = False  # varsayilan True -> kapatmak en buyuk hizlanmayi verir

TRACKER_FAIL_TOLERANCE = 1
REDETECT_INTERVAL = 2
 
BBOX_AREA_MIN_RATIO = 0.35
BBOX_AREA_MAX_RATIO = 2.60
 
BBOX_ASPECT_MIN = 0.6
BBOX_ASPECT_MAX = 6.0
 
VALIDATE_POSITION_ON_COURT = True
 
TRACK_VERIFY_INTERVAL = 15
TRACK_VERIFY_MIN_FG_RATIO = 0.12

# --- HIBRIT YENIDEN SENKRONIZASYON (Cozum 1) --------------------------------
# NEDEN: CSRT bir korelasyon-filtre takipcisidir; otelemeyi iyi ama OLCEGI cok
# zayif/tutucu tahmin eder. Oyuncu kameradan UZAKLASINCA gercek piksel boyu
# kuculur, fakat CSRT kutuyu neredeyse ayni boyutta tutar. Kutunun alt kenari
# gercek ayaktan asagida kaldigi icin homografi konumu KAMERAYA YAKIN gosterir
# (oyuncu "olmasi gerekenden daha onde" gorunur).
# COZUM: CSRT basariyla takip ederken BILE her N karede bir, zaten var olan
# hafif tespiti (detect() -> arka plan cikarma, YOLO/agir model DEGIL) calistirip
# CSRT'yi dogru boyutlu kutuyla yeniden baslatiriz. Boylece zamanla olusan
# konum ve BOYUT kaymasi periyodik olarak silinir.
TRACK_RESYNC_INTERVAL = 18     # Yeniden senkron periyodu (kare). 0 -> kapali. Onerilen 15-20.
TRACK_RESYNC_MAX_JUMP = 1.5    # Yeni tespit mevcut konuma bu mesafeden (m) uzaksa spurious
                               # blob sayilir ve resync REDDEDILIR (takip kacmasin diye).

# --- BOYUT AKLA-YATKINLIK DENETIMI (Cozum 2: otomatik kurtarma) -------------
# Homografiden oyuncunun o derinlikteki BEKLENEN piksel boyu hesaplanir
# (expected_height). Tracker kutusunun yuksekligi bu beklentinin cok disindaysa
# -ozellikle cok BUYUKse- CSRT olcegi kacirmis demektir: takibi sonlandirip
# yeniden tespit tetikleriz. "Kutu kuculmuyor" hatasini dogrudan yakalar.
TRACK_SIZE_CHECK = True
TRACK_SIZE_MIN_RATIO = 0.50    # kutu_h < beklenen_h * bu  -> gecersiz (cok kucuk)
TRACK_SIZE_MAX_RATIO = 1.80    # kutu_h > beklenen_h * bu  -> gecersiz (kuculmeyen buyuk kutu)

# ==============================================================================
# 6) BACKGROUND SUBTRACTION (MOG2) - SADECE HAREKET ANALIZI ICIN
# ==============================================================================
# MOG2 oyuncu BULMAK icin kullanilmaz. Maske sabit piksel gridi uzerinde uretilir,
# fakat YALNIZCA oyuncunun bbox'i icinden okunur.
 
# MOG2 boru hattinin CEKIRDEK islevi DEGILDIR: tek isi HUD'daki
# "HAREKETLI / DURAGAN" gostergesini beslemek. Olculdu: apply ~3.6 ms/kare ->
# Raspberry Pi'de luks. False yapinca hareket gostergesi de cizilmez.
# SAHNE KESME tespiti bundan BAGIMSIZ calismaya devam eder (CUT_DETECTION).
USE_MOG2 = False

MOG2_SCALE = 0.4
MOG2_HISTORY = 300
MOG2_VAR_THRESHOLD = 32.0
MOG2_DETECT_SHADOWS = False
MOG2_LEARNING_RATE = 0.005
 
MOG2_WARMUP = 15   # Model sifirlandiktan sonra enerji RAPORLANMAZ.
 
MOTION_ENERGY_THRESH = 0.10
MOTION_SMOOTHING = 0.35
 
# ==============================================================================
# 7) SAHNE KESME (CUT) TESPITI
# ==============================================================================
 
CUT_DETECTION = True
 
CUT_HIST_BINS = 32
CUT_CORR_THRESH = 0.70   # Yukselt -> daha cok tetiklenir. Dusur -> daha az.
CUT_MIN_GAP = 5
 
CUT_RESET_MOG2 = True
# CUT_REFRESH_COLOR kaldirildi: tespit artik medyan arka plan kullaniyor,
# kamera sabit oldugu icin model sahne kesmesinde de gecerli kalir.
 
# ==============================================================================
# 8) KONUM HESABI VE KALMAN
# ==============================================================================
 
USE_FOOT_POINT = True   # Homography yer duzlemini esler -> bbox ALT-ORTA noktasi.
FOOT_POINT_RATIO = 1.0
 
USE_KALMAN = True
 
KF_PROCESS_NOISE = 1e-2
KF_MEASUREMENT_NOISE = 5e-2
KF_ERROR_COV = 1.0
 
MAX_JUMP_PER_FRAME = 1.5  # metre/kare. Kesmede son konum sifirlandigi icin atlanir.
 
# ==============================================================================
# 9) ANALIZ (ilerleyen asamalar)
# ==============================================================================
 
FALLBACK_FPS = 30.0
 
MIN_MOVE_DIST = 0.03
SPEED_WINDOW = 5
 
DIRECTION_CHANGE_ANGLE = 60.0
DIRECTION_MIN_SPEED = 0.8
 
HEATMAP_CELL_SIZE = 0.5
HEATMAP_BLUR_KERNEL = (9, 9)
 
# ==============================================================================
# 10) IZGARA OVERLAY GORSELLIGI (video uzerine)
# ==============================================================================
# Izgara + numaralar + kort cizgileri BIR KEZ statik katmana render edilir.
# Her karede sadece maskeli kopya + aktif hucre vurgusu yapilir.
 
GRID_COLOR = (0, 255, 255)     # Ana izgara hucre kenarlari (sari).
TOP_BAND_COLOR = (0, 90, 255)     # Uzak dip cizgi arkasi bant (turuncu).
BOTTOM_BAND_COLOR = (0, 90, 255)  # Yakin dip cizgi arkasi bant (turuncu).
SIDE_COLOR = (0, 255, 0)       # Yan seritler (yesil).
GRID_THICKNESS = 1
 
SHOW_GRID_NUMBERS = True
GRID_NUMBER_COLOR = (255, 255, 255)
GRID_NUMBER_SCALE = 0.42
 
HIGHLIGHT_COLOR = (211, 0, 155)    # Aktif hucre dolgusu (mor).
HIGHLIGHT_BORDER = (255, 255, 255)
HIGHLIGHT_ALPHA = 0.45             # 0.0 - 1.0
 
# ==============================================================================
# 11) BOLGE PANELI (kus bakisi)
# ==============================================================================
 
SHOW_PANEL = True
 
PANEL_PPM = 9           # Piksel / metre. Panel buyukse dusur.
PANEL_MARGIN = 8
PANEL_HEADER = 26       # Baslik seridi yuksekligi (px).
PANEL_POS = (10, 10)    # Karedeki sol-ust kose.
PANEL_ALPHA = 0.88
 
PANEL_BG = (30, 30, 30)
PANEL_HEADER_BG = (60, 60, 60)
PANEL_BORDER = (255, 255, 255)
PANEL_TEXT_COLOR = (255, 255, 255)
 
PANEL_GRID_COLOR = (90, 160, 160)
PANEL_BAND_COLOR = (60, 90, 140)
PANEL_SIDE_COLOR = (70, 130, 70)
PANEL_COURT_COLOR = (255, 255, 255)
 
PANEL_SHOW_NUMBERS = True
PANEL_NUMBER_COLOR = (170, 170, 170)
PANEL_NUMBER_SCALE = 0.28
 
PANEL_HIGHLIGHT = (211, 0, 155)
PANEL_HIGHLIGHT_BORDER = (255, 255, 255)
PANEL_HIGHLIGHT_ALPHA = 0.75
 
PANEL_PLAYER_COLOR = (0, 0, 255)
PANEL_TRAIL_COLOR = (0, 200, 255)
 
TRAIL_LENGTH = 40

# ==============================================================================
# 11b) BOLGE GECIS ANALIZI
# ==============================================================================
# Oyuncu bir bolgeden digerine gectikce OLAY kaydedilir (kare basina degil).
# Cikti: output videosuna panel + CSV (her gecis) + JSON (bolge basina ozet).

ZONE_ANALYSIS = True
ZONE_EVENTS_CSV = "output/zone_events.csv"     # Her gecis bir satir.
ZONE_SUMMARY_JSON = "output/zone_summary.json"  # Bolge basina toplam sure + ziyaret.

SHOW_ZONE_PANEL = True        # Output videosunun sag-altina son gecis textbox'i.
ZONE_PANEL_MAX_LINES = 6      # Panelde gosterilecek son gecis sayisi.

# ==============================================================================
# 12) GENEL GORSELLESTIRME
# ==============================================================================
 
COLOR_BBOX = (0, 255, 0)
COLOR_BBOX_LOST = (0, 0, 255)
COLOR_FOOT = (0, 200, 255)
COLOR_COURT = (255, 200, 0)
COLOR_TEXT = (255, 255, 255)
COLOR_TEXT_BG = (0, 0, 0)
COLOR_MOTION_ON = (0, 255, 255)
COLOR_MOTION_OFF = (120, 120, 120)
COLOR_CUT = (0, 140, 255)
 
BBOX_THICKNESS = 2
LINE_THICKNESS = 1
 
FONT_SCALE = 0.5
FONT_THICKNESS = 1
 
SHOW_FPS = True
FPS_SMOOTHING = 0.9
 
CUT_BANNER_FRAMES = 12
 
DEBUG_SHOW_MASK = False
DEBUG_SHOW_DETECTION = False


# ==============================================================================
# BOLUM 2 - KORT GEOMETRISI, OTOMATIK TESPIT, HOMOGRAPHY (eski BOLUM 2 (KORT))
# ==============================================================================
# BOLUM 2 (KORT)
# --------
# Kort geometrisi: kose secimi, homography, nokta donusumleri, arama maskesi ve
# kort cizgilerinin cizimi.
#
# Bolge (zone) modeli ve panel artik BOLUM 5 (BOLGELER)'dedir; bu dosya yalnizca geometri ile
# ilgilenir.
#
# KOORDINAT SISTEMI:
#     Goruntu (piksel) <--H--> Kort (metre)
#     H     : goruntu -> kort
#     H_inv : kort    -> goruntu
#
# NOT: Kose secimi, boru hattinin calistigi kare uzerinde (PROC_SCALE uygulanmis)
#      yapilmalidir. Aksi halde piksel koordinatlari tutmaz.

CORNERS_PATH = CORNERS_PATH

# Kose semasi kimligi. Referans noktalarinin ANLAMI/SAYISI degistiginde bu
# yazi da degistirilmelidir.
# Neden gerekli: load_corners eskiden yalnizca kare BOYUTUNU dogruluyordu. Kose
# semasi "4 dip kose"den "uzak dip + file"ye gecince, eski bir court_corners.json
# sessizce yuklenip tamamen yanlis bir kort uretiyordu (yakin yari kameranin
# arkasina dusuyor, court_to_image None donuyordu). Sema uyusmazsa kayit reddedilir
# ve kullaniciya koseler yeniden sectirilir.
#
# "far_baseline+net" -> "net+near_service": uzak dip koseler goruntude asiri
# sikistigi icin capa artik 4 degil 5 nokta (file + yakin servis cizgisi
# kesisimleri). Surum bilerek degistirildi ki ESKI semayla kaydedilmis
# court_corners.json dosyalari SESSIZCE yanlis yorumlanmasin.
CORNER_SCHEME = "net+near_service"

_W = COURT_WIDTH
_L = COURT_LENGTH
_SL = SINGLES_LEFT_X
_SR = SINGLES_RIGHT_X
_CX = CENTER_X
_NET = NET_Y
_FS = FAR_SERVICE_Y
_NS = NEAR_SERVICE_Y

# Gercek tenis kortu cizgileri (dunya koordinatinda, metre).
COURT_LINES = [
    # Dis (cift) kort dortgeni
    ((0.0, 0.0), (_W, 0.0)),
    ((_W, 0.0), (_W, _L)),
    ((_W, _L), (0.0, _L)),
    ((0.0, _L), (0.0, 0.0)),
    # Tekler yan cizgileri
    ((_SL, 0.0), (_SL, _L)),
    ((_SR, 0.0), (_SR, _L)),
    # File
    ((0.0, _NET), (_W, _NET)),
    # Servis cizgileri
    ((_SL, _FS), (_SR, _FS)),
    ((_SL, _NS), (_SR, _NS)),
    # Orta servis cizgisi
    ((_CX, _FS), (_CX, _NS)),
    # Dip cizgi orta isaretleri
    ((_CX, 0.0), (_CX, 0.30)),
    ((_CX, _L - 0.30), (_CX, _L)),
]


# ==============================================================================
# 1) OTOMATIK KOSE TESPITI
# ==============================================================================
#
# NEDEN SERBEST MODEL ARAMASI DEGIL:
#     "Tum cizgileri bul, her model atamasini dene, en iyi puanlayani sec"
#     yaklasimi bu kamera acisinda CALISMIYOR. Kortun uzak yarisi file tarafindan
#     ortuldugu icin (file mesh'i tam o bandin onunde durur), derinlik atamasini
#     kaydiran YANLIS cozumler gorunen tum kanitla uyusur ve dogru cozumden daha
#     iyi puan alir. Ayirt edici kanit goruntude yoktur.
#
# BUNUN YERINE - projenin zaten varsaydigi yapiyi kullan (sabit kamera, yakin dip
# cizginin arkasinda). Bu kurulumda su cizgiler her zaman nettir:
#     - yakin dip cizgi     : en alttaki, en genis yatay cizgi
#     - yakin servis cizgisi: bir ustteki yatay cizgi
#     - cift yan cizgileri  : en distaki egik cizgi cifti
# Atama SABITLENDIGI icin belirsizlik kalkar; 4 kesisim -> homography -> kortun
# tamami (kare disina dusen yakin dip koseler dahil) ekstrapole edilir.
#
# TEKLER/CIFT AYRIMI:
#     Egik cizgiler arasinda hem tekler hem cift yan cizgileri vardir; yanlisini
#     secmek x olcegini bozar. Ayrim gercek kort oranindan yapilir:
#         tekler_genisligi / cift_genisligi = 8.23 / 10.97 = 0.750
#     Bu oran perspektiften bagimsiz DEGILDIR, ama dip cizgi goruntude neredeyse
#     yatay oldugundan (kamera kort eksenine bakar) yeterince kararlidir.


def _line_mask(frame):
    """
    Kort cizgisi maskesi.

    "Parlak piksel" olcutu yetmez: bina duz parlaktir, tel orgu ince agdir; ikisi
    de beyaz sayilinca cizgi tespiti coker. Kort cizgisi INCE bir yapidir: parlak,
    ve DIK yonde her iki yani belirgin KOYU. Bina bu testi gecemez.

    Ardindan bilesenler dondurulmus kutu (minAreaRect) ile elenir: uzun+ince olan
    kalir, agac yapragi gibi tikiz dokular gider. Eksen hizali kutu KULLANILMAZ -
    capraz bir cizginin eksen hizali kutusu her iki boyutta da buyuktur, bu yuzden
    yan cizgiler yanlislikla "kalin" sayilip elenirdi.
    """
    d = LINE_NEIGHBOR_DIST
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
    left, right = np.roll(g, d, axis=1), np.roll(g, -d, axis=1)
    up, down = np.roll(g, d, axis=0), np.roll(g, -d, axis=0)

    t = LINE_DARK_DELTA
    vert = (g - left > t) & (g - right > t)
    horiz = (g - up > t) & (g - down > t)
    mask = ((vert | horiz) & (g > LINE_MIN_BRIGHT)).astype(np.uint8) * 255
    mask[:, :d] = 0
    mask[:, -d:] = 0
    mask[:d, :] = 0
    mask[-d:, :] = 0

    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(mask)
    for c in contours:
        if len(c) < 5:
            continue
        (_, _), (a, b), _ = cv2.minAreaRect(c)
        if max(a, b) < LINE_MIN_LENGTH or min(a, b) > LINE_MAX_THICK:
            continue
        cv2.drawContours(out, [c], -1, 255, -1)
    return out


def _merge_lines(segments):
    """Hough segmentlerini sonsuz dogrulara birlestirir (aci + ofset yakinligiyla)."""
    groups = []
    for x1, y1, x2, y2 in segments:
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        ang = ang + 180 if ang < -90 else (ang - 180 if ang > 90 else ang)
        rho = x1 * np.cos(np.radians(ang + 90.0)) + y1 * np.sin(np.radians(ang + 90.0))
        length = float(np.hypot(x2 - x1, y2 - y1))
        for g in groups:
            if abs(g["ang"] - ang) < 2.5 and abs(g["rho"] - rho) < 12.0:
                tot = g["w"] + length
                g["ang"] = (g["ang"] * g["w"] + ang * length) / tot
                g["rho"] = (g["rho"] * g["w"] + rho * length) / tot
                g["w"] = tot
                g["pts"] += [(x1, y1), (x2, y2)]
                break
        else:
            groups.append({"ang": ang, "rho": rho, "w": length, "pts": [(x1, y1), (x2, y2)]})

    lines = []
    for g in groups:
        if g["w"] < LINE_MIN_WEIGHT:
            continue
        pts = np.array(g["pts"], dtype=np.float64)
        mean = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - mean)          # toplam en kucuk kareler
        n = np.array([-vt[0][1], vt[0][0]])
        lines.append({"n": n, "c": float(n @ mean), "ang": g["ang"], "w": g["w"]})
    return lines


def _intersect(l1, l2):
    A = np.array([l1["n"], l2["n"]])
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    if abs(det) < 1e-8:
        return None
    return np.linalg.solve(A, np.array([l1["c"], l2["c"]]))


def _x_at(line, y):
    n, c = line["n"], line["c"]
    return (c - n[1] * y) / n[0] if abs(n[0]) > 1e-9 else 1e9


def _y_at(line, x):
    n, c = line["n"], line["c"]
    return (c - n[0] * x) / n[1] if abs(n[1]) > 1e-9 else 1e9


def auto_detect_corners(frame, verbose=True):
    """
    Kortun 5 referans noktasini (file + yakin servis cizgisi kesisimleri)
    otomatik bulur.

    Ic hesaplama ROBUST cizgilerden (yakin dip + yakin servis + cift yan
    cizgileri) kurulan yardimci bir homography kullanir; sonra bu yardimci
    homography ile COURT_CORNERS_WORLD icindeki 5 dunya noktasi goruntuye
    izdusurulerek disariya verilir. Boylece disari her zaman GUNCEL sema
    donmus olur; ic hesaplama semasi degismez.

    Donus: COURT_CORNERS_WORLD sirasinda (N, 2) float32 dizi, veya
           guvenilir bir cozum bulunamazsa None (-> elle secime dusulur).
    """
    h, w = frame.shape[:2]
    mask = _line_mask(frame)

    segments = cv2.HoughLinesP(mask, 1, np.pi / 360, threshold=55,
                               minLineLength=60, maxLineGap=18)
    if segments is None:
        return None

    # OpenCV surumune gore sekil (N,1,4) veya (N,4) olabilir (or. OpenCV 5.x
    # (N,4) donuyor, eski surumler (N,1,4)); reshape(-1,4) ikisini de N x
    # (x1,y1,x2,y2) haline getirir.
    lines = _merge_lines(segments.reshape(-1, 4))
    horiz = sorted([l for l in lines if abs(l["ang"]) < LINE_HORIZ_MAX_ANGLE],
                   key=lambda l: -_y_at(l, w * 0.5))          # en alttaki once
    steep = [l for l in lines if abs(l["ang"]) >= LINE_HORIZ_MAX_ANGLE]
    if len(horiz) < 2 or len(steep) < 2:
        return None

    # --- Dogrulama puanlayicisi ---------------------------------------------
    # SADECE yakin yari (dunya y >= NET_Y) puanlanir: uzak yari file tarafindan
    # ortuludur, oradaki model noktalari her adayda "isabetsiz" sayilir ve puani
    # anlamsizlastirir. Atama sabit oldugu icin yakin yari ayirt etmeye yeter.
    painted = [ln for ln in COURT_LINES
               if not (abs(ln[0][1] - _NET) < 1e-6 and abs(ln[1][1] - _NET) < 1e-6)]
    dist = cv2.distanceTransform(cv2.bitwise_not(mask), cv2.DIST_L2, 3)
    samples = np.array([(ax + (bx - ax) * t, ay + (by - ay) * t)
                        for (ax, ay), (bx, by) in painted
                        for t in np.linspace(0, 1, 60)
                        if ay + (by - ay) * t >= _NET - 1e-6])

    def verify(H_inv):
        den = H_inv[2, 0] * samples[:, 0] + H_inv[2, 1] * samples[:, 1] + H_inv[2, 2]
        if np.any(den <= 1e-9):
            return -1.0
        px = (H_inv[0, 0] * samples[:, 0] + H_inv[0, 1] * samples[:, 1] + H_inv[0, 2]) / den
        py = (H_inv[1, 0] * samples[:, 0] + H_inv[1, 1] * samples[:, 1] + H_inv[1, 2]) / den
        ok = (px >= 0) & (px < w - 1) & (py >= 0) & (py < h - 1)
        if ok.sum() < 60:
            return -1.0
        d = dist[py[ok].astype(np.int32), px[ok].astype(np.int32)]
        return float((d < AUTO_LINE_TOL).mean())

    world_quad = np.array([[0.0, _L], [_W, _L], [_W, _NS], [0.0, _NS]], dtype=np.float32)
    best = None

    for base, serv in itertools.combinations(horiz[:4], 2):
        y_base = _y_at(base, w * 0.5)
        if y_base <= _y_at(serv, w * 0.5):
            continue                                   # dip cizgi servisin ALTINDA
        if not (0.55 * h <= y_base <= 1.05 * h):
            continue                                   # dip cizgi karenin alt kisminda

        ordered = sorted(steep, key=lambda l: _x_at(l, y_base))
        for pair in _doubles_candidates(ordered, y_base, w):
            ls, rs = pair
            pts = [_intersect(ls, base), _intersect(rs, base),
                   _intersect(rs, serv), _intersect(ls, serv)]
            if any(p is None for p in pts):
                continue
            img = np.array(pts, dtype=np.float32)
            if not cv2.isContourConvex(img.astype(np.int32)):
                continue
            try:
                H_inv = cv2.getPerspectiveTransform(world_quad, img).astype(np.float64)
            except cv2.error:
                continue
            if H_inv[2, 0] * _CX + H_inv[2, 1] * _NET + H_inv[2, 2] < 0:
                H_inv = -H_inv
            score = verify(H_inv)
            if best is None or score > best[0]:
                best = (score, H_inv)

    if best is None or best[0] < AUTO_MIN_SCORE:
        if verbose:
            got = "{:.3f}".format(best[0]) if best else "yok"
            print("Otomatik kort tespiti basarisiz (puan {} < {}).".format(
                got, AUTO_MIN_SCORE))
        return None

    score, H_inv = best
    corners = []
    for wx, wy in COURT_CORNERS_WORLD:
        p = court_to_image(H_inv, float(wx), float(wy))
        if p is None:
            return None
        corners.append(p)
    if verbose:
        print("Otomatik kort tespiti: puan {:.3f}".format(score))
    return np.array(corners, dtype=np.float32)


def _doubles_candidates(ordered, y_base, frame_w):
    """
    Egik cizgiler arasindan CIFT yan cizgisi ciftlerini secer.

    Once gercek kort oranindan yararlanilir: dort cizgilik (a<b<c<d) altkumelerde
        (x_c - x_b) / (x_d - x_a)  ~  8.23 / 10.97 = 0.750
    saglaniyorsa (a, d) cift, (b, c) tekler yan cizgisidir. Bu, tekler cizgilerini
    yanlislikla cift sanmayi engeller (x olcegini bozardi).

    Oran testi tutmazsa en distaki birkac cift yedek olarak denenir.
    """
    xs = [_x_at(l, y_base) for l in ordered]
    out = []
    ratio = (_SR - _SL) / _W
    for a, b, c, d in itertools.combinations(range(len(ordered)), 4):
        outer = xs[d] - xs[a]
        if outer < 0.4 * frame_w:
            continue
        if abs((xs[c] - xs[b]) / outer - ratio) < AUTO_RATIO_TOL:
            out.append((ordered[a], ordered[d]))
    for i in range(min(2, len(ordered))):
        for j in range(max(0, len(ordered) - 2), len(ordered)):
            if i < j and xs[j] - xs[i] >= 0.4 * frame_w:
                out.append((ordered[i], ordered[j]))
    seen, uniq = set(), []
    for p in out:
        k = (id(p[0]), id(p[1]))
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    return uniq


def confirm_corners(frame, corners, window_name="Otomatik Kort Tespiti"):
    """
    Otomatik bulunan kortu kullaniciya gosterip onay ister.

    Yanlis bir otomatik tespit sessizce her seyi bozacagi icin bu adim vardir.

    Kontroller:  ENTER / c : kabul      m : elle sec      q / ESC : iptal
    Donus: True (kabul) / False (elle secime gec)
    """
    H_inv = compute_homography(corners)[1]
    canvas = frame.copy()
    draw_lines(canvas, project_lines(H_inv, COURT_LINES), COLOR_COURT,
               LINE_THICKNESS + 1)
    draw_ref_points(canvas, corners, show_labels=False)
    _put_text(canvas, "Kort otomatik bulundu - dogru mu?", (10, 24), 0.7, 2)
    _put_text(canvas, "ENTER: kabul | m: elle sec | q: iptal", (10, 50), 0.55, 1)

    cv2.namedWindow(window_name)
    try:
        while True:
            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 10, ord("c")):
                return True
            if key == ord("m"):
                return False
            if key in (ord("q"), 27):
                return False
    finally:
        cv2.destroyWindow(window_name)


# ==============================================================================
# 2) KOSE SECIMI (elle - otomatik tespit basarisiz olursa)
# ==============================================================================

def select_corners(frame, window_name="Kort Koseleri"):
    """
    Kullanicidan kortun 5 referans noktasini fare ile sectirir.

    Secim sirasi (COURT_CORNERS_WORLD / CORNER_LABELS ile AYNI sira):
        file-sol -> file-sag -> file-orta -> yakin servis-sol -> yakin servis-sag
    (bkz. COURT_CORNERS_WORLD basligi icin ayrintili gerekce).

    Yani capa artik kortun uzak yarisi DEGIL, file ile yakin servis cizgisi
    arasindaki bant; tam kort bu noktalardan ekstrapole edilir.

    Kontroller:
        Sol tik  : nokta ekle
        z        : son noktayi geri al
        r        : sifirla
        ENTER/c  : onayla (5 nokta secildiyse)
        q/ESC    : iptal

    Donus: (5, 2) float32 dizi veya None.
    """
    n_points = len(CORNER_LABELS)
    points = []
    cursor = [0, 0]

    def on_mouse(event, x, y, flags, param):
        cursor[0], cursor[1] = x, y
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < n_points:
            points.append((float(x), float(y)))

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    h, w = frame.shape[:2]
    lens_half = 22
    lens_zoom = 6
    lens_size = lens_half * 2 * lens_zoom

    result = None
    while True:
        canvas = frame.copy()

        if points:
            draw_ref_points(canvas, points, show_labels=False)
        # Dortgen anahatti file-sol/file-sag/yakin servis-sag/yakin servis-sol
        # sirasiyla cizilir (file-orta digerleriyle es dogrusal oldugundan
        # anahatta dahil edilmez, sadece nokta olarak gosterilir).
        quad_order = [i for i in (0, 1, 4, 3) if i < len(points)]
        if len(quad_order) >= 2:
            pts = np.array([points[i] for i in quad_order], dtype=np.int32)
            cv2.polylines(canvas, [pts], len(points) == n_points, COLOR_COURT, 2)

        cx, cy = cursor
        cv2.line(canvas, (cx, 0), (cx, h), (60, 60, 60), 1)
        cv2.line(canvas, (0, cy), (w, cy), (60, 60, 60), 1)

        # Zoom lens: hassas kose secimi icin imlecin cevresini buyutur.
        x0 = max(0, min(w - 1, cx - lens_half))
        y0 = max(0, min(h - 1, cy - lens_half))
        x1 = max(1, min(w, cx + lens_half))
        y1 = max(1, min(h, cy + lens_half))
        patch = frame[y0:y1, x0:x1]
        if patch.size > 0:
            lens = cv2.resize(patch, (lens_size, lens_size),
                              interpolation=cv2.INTER_NEAREST)
            lx = (cx - x0) * lens_zoom
            ly = (cy - y0) * lens_zoom
            cv2.line(lens, (lx, 0), (lx, lens_size), (0, 0, 255), 1)
            cv2.line(lens, (0, ly), (lens_size, ly), (0, 0, 255), 1)
            cv2.rectangle(lens, (0, 0), (lens_size - 1, lens_size - 1),
                          COLOR_COURT, 2)
            dst_x = 10 if cx > w // 2 else w - lens_size - 10
            dst_y = 10 if cy > h // 2 else h - lens_size - 10
            if 0 <= dst_x and dst_x + lens_size <= w and 0 <= dst_y and dst_y + lens_size <= h:
                canvas[dst_y:dst_y + lens_size, dst_x:dst_x + lens_size] = lens

        info = CORNER_LABELS[len(points)] if len(points) < n_points \
            else "{0}/{0} tamam -> ENTER: onayla".format(n_points)
        _put_text(canvas, info, (10, 24), 0.7, 2)
        _put_text(canvas, "z: geri al | r: sifirla | q: iptal", (10, 50), 0.55, 1)

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("z") and points:
            points.pop()
        elif key == ord("r"):
            points.clear()
        elif key in (13, 10, ord("c")) and len(points) == n_points:
            result = np.array(points, dtype=np.float32)
            break
        elif key in (ord("q"), 27):
            result = None
            break

    cv2.destroyWindow(window_name)
    return result


def _video_id():
    """
    Kayitli koselerin HANGI videoya ait oldugunu belirten kimlik.

    NEDEN GEREKLI: iki farkli video ayni cozunurlukte olabilir (or. C01.mp4 ve
    match.mp4 ikisi de 1920x1080 -> PROC_SCALE sonrasi ikisi de 1152x648). Eski
    sema yalnizca genislik/yukseklik denetledigi icin video degistirildiginde
    ONCEKI kortun koseleri SESSIZCE yukleniyor ve tum homography kayiyordu.
    Kimlik uyusmazsa load_corners None doner -> koseler yeniden sectirilir.
    """
    return str(VIDEO_PATH)


def save_corners(image_pts, frame_shape, path=CORNERS_PATH):
    """Secilen koseleri JSON olarak kaydeder (hangi videoya ait oldugu dahil)."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    data = {
        "scheme": CORNER_SCHEME,
        "video": _video_id(),   # Koseler bu videoya aittir; baskasinda gecersiz.
        "corners": np.asarray(image_pts, dtype=np.float32).reshape(-1, 2).tolist(),
        "width": int(frame_shape[1]),
        "height": int(frame_shape[0]),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_corners(frame_shape, path=CORNERS_PATH):
    """
    Kayitli koseleri yukler. Su durumlarda None doner (-> koseler yeniden sectirilir):
      - dosya yok,
      - kose semasi farkli (or. eski "4 dip kose" kaydi; anahtar hic yoksa KeyError),
      - kayit BASKA bir videoya ait (bkz. _video_id; kimliksiz eski kayitlar dahil),
      - kare boyutu kayittan farkli (PROC_SCALE degistiyse olcek tutmaz).
    """
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data["scheme"] != CORNER_SCHEME:
            return None
        # Kayit baska videoya aitse kullanma: ayni cozunurlukteki farkli bir video
        # sessizce YANLIS koseleri yuklerdi. Kimliksiz (eski) kayitlar da reddedilir.
        if data.get("video") != _video_id():
            return None
        if int(data["width"]) != int(frame_shape[1]):
            return None
        if int(data["height"]) != int(frame_shape[0]):
            return None
        return np.array(data["corners"], dtype=np.float32).reshape(-1, 2)
    except (KeyError, ValueError, TypeError, json.JSONDecodeError):
        return None


# ==============================================================================
# 2) HOMOGRAPHY
# ==============================================================================

def compute_homography(image_pts):
    """
    Goruntudeki referans noktalarindan homography hesaplar.

    Donus: (H, H_inv)
        H     : goruntu (piksel) -> kort (metre)
        H_inv : kort (metre)     -> goruntu (piksel)

    Girdi noktalari COURT_CORNERS_WORLD ile AYNI sirada olmalidir (bkz.
    CORNER_LABELS): file uzerinde 3 nokta (sol/sag/orta) + yakin servis
    cizgisi uzerinde 2 nokta (sol/sag tekler) - toplam 5 nokta.

    NEDEN 5 NOKTA + cv2.findHomography (getPerspectiveTransform DEGIL):
        Eskiden capa TAM OLARAK 4 nokta idi ve cv2.getPerspectiveTransform ile
        MINIMAL (fazla bilgi icermeyen) bir cozum kuruluyordu: capadaki kucucuk
        bir hata dogrudan ve TAM olarak homography'ye giriyordu. O 4 noktadan
        ikisi uzak dip koseydi ve goruntude birkac piksele sikistigi icin oradaki
        hata orantisiz buyuyordu. Simdi capa 5 noktaya cikarildi (fazladan-
        belirlenmis/over-determined sistem) ve cv2.findHomography en kucuk
        kareler (least-squares, method=0) ile TUM noktalarin ortak hatasini
        minimize eden TEK bir homography bulur. Boylece herhangi BIR noktadaki
        (or. file ortasi) kucuk bir tespit hatasi digerlerinin agirligiyla
        YUMUSATILIR; capa artik tek noktalik bir hataya karsi kirilgan degildir.

    Capa noktalari artik kortun uzak yarisinda degil, FILE ile YAKIN SERVIS
    CIZGISI arasindaki (kameraya orta-yakin) bantta; homography yine de yer
    DUZLEMININ TAMAMINI esler, uzak yari ve tum bantlar bu matristen ekstrapole
    edilir (bkz. COURT_CORNERS_WORLD basligi).

    Matrisler, projektif payda (w) kort alani icinde POZITIF olacak sekilde
    normalize edilir. Bu sayede kort disina tasan noktalarin kamera arkasina
    dusup dusmedigi paydanin isaretine bakilarak anlasilabilir.
    """
    world = COURT_CORNERS_WORLD
    img = np.asarray(image_pts, dtype=np.float32).reshape(-1, 2)
    if img.shape[0] != world.shape[0]:
        raise ValueError(
            "{} referans noktasi bekleniyordu, {} nokta geldi (CORNER_LABELS "
            "sirasina bakin).".format(world.shape[0], img.shape[0])
        )

    # Geometrik gecerlilik denetimi: file-sol, file-sag, yakin servis-sag, yakin
    # servis-sol dortgeni (indeks 0,1,4,3). File-orta (indeks 2) bu denetime
    # dahil EDILMEZ: dunyada digerleriyle es dogrusaldir (ucu de file uzerinde),
    # bir dortgenin kosesi olamaz.
    quad = img[[0, 1, 4, 3]].astype(np.int32)
    if abs(cv2.contourArea(quad)) < 1.0 or not cv2.isContourConvex(quad):
        raise ValueError(
            "Secilen referans noktalari gecerli bir dortgen olusturmuyor. Noktalari "
            "CORNER_LABELS sirasiyla secin (file sol -> file sag -> file orta "
            "-> yakin servis sol -> yakin servis sag)."
        )

    H, _ = cv2.findHomography(img, world, method=0)
    if H is None:
        raise ValueError("Homography hesaplanamadi (referans noktalari dejenere olabilir).")

    # H_inv, H'nin TAM matris tersidir - BAGIMSIZ ikinci bir en-kucuk-kareler
    # cozumu DEGIL. Boylece court_to_image(H_inv, *image_to_court(H, x, y)) HER
    # ZAMAN (x, y)'ye geri doner (yuvarlama disinda).
    H_inv = np.linalg.inv(H)

    # Isaret denetimi capa noktalarinin (dunya) AGIRLIK MERKEZINDE yapilir: bu
    # nokta H_inv'nin (kort -> goruntu) paydasini kullanir, cunku H_inv'ye girdi
    # DUNYA (metre) koordinatidir -- H'nin kendisine (goruntu/piksel bekler)
    # dunya koordinati vermek boyut hatasi olurdu. Sabit bir nokta yerine merkez
    # kullanilir; capa dizilimi degisirse de (COURT_CORNERS_WORLD guncellenirse)
    # merkez her zaman guvenli bir IC noktadir. H ve H_inv HER ZAMAN birbirinin
    # tam tersi kalmasi icin isaret ikisine BIRLIKTE uygulanir (inv(-M) = -inv(M)).
    ref_cx, ref_cy = world.mean(axis=0)
    if H_inv[2, 0] * ref_cx + H_inv[2, 1] * ref_cy + H_inv[2, 2] < 0:
        H = -H
        H_inv = -H_inv

    return H.astype(np.float64), H_inv.astype(np.float64)


def image_to_court(H, x, y):
    """
    Tek goruntu noktasini kort koordinatina cevirir (metre).
    Tek nokta icin cv2.perspectiveTransform'dan cok daha hizli.
    """
    w = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    if abs(w) < 1e-9:
        return None
    return ((H[0, 0] * x + H[0, 1] * y + H[0, 2]) / w,
            (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / w)


def court_to_image(H_inv, wx, wy):
    """
    Tek kort noktasini goruntu koordinatina cevirir (piksel).
    Nokta kameranin arkasina dusuyorsa (payda <= 0) None doner.
    """
    w = H_inv[2, 0] * wx + H_inv[2, 1] * wy + H_inv[2, 2]
    if w <= 1e-9:
        return None
    return ((H_inv[0, 0] * wx + H_inv[0, 1] * wy + H_inv[0, 2]) / w,
            (H_inv[1, 0] * wx + H_inv[1, 1] * wy + H_inv[1, 2]) / w)


def is_on_court(wx, wy, margin=None):
    """Konum kort + tolerans alani icinde mi? (tracker saglik denetimi)"""
    if margin is None:
        margin = COURT_MARGIN
    return (-margin <= wx <= _W + margin) and (-margin <= wy <= _L + margin)


# ==============================================================================
# 3) ARAMA MASKESI (tespit icin)
# ==============================================================================

def build_search_mask(frame_shape, H_inv, half=None, margin=None):
    """
    Tespitin yapilacagi bolgenin maskesini ve sinirlayici dikdortgenini uretir.
    Bir kez hesaplanir (kamera sabit).

    half   : "near" | "far" | "both"
    margin : kort disina eklenen pay (metre).

    Kort disina tasan koseler kameranin arkasina duserse pay otomatik kucultulur.
    Donus: (mask uint8 [0/255], (x, y, w, h))
    """
    if half is None:
        half = PLAYER_HALF
    if margin is None:
        margin = SEARCH_MARGIN

    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    m = float(margin)
    poly = None
    for _ in range(6):
        x0, x1 = -m, _W + m
        if half == "near":
            y0, y1 = _NET, _L + m
        elif half == "far":
            y0, y1 = -m, _NET
        else:
            y0, y1 = -m, _L + m

        corners = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
        projected = [court_to_image(H_inv, cx, cy) for cx, cy in corners]

        if all(p is not None for p in projected):
            poly = projected
            break
        m *= 0.5  # Kamera arkasina dusen kose var -> payi kucult.

    if poly is None:
        x0, x1 = 0.0, _W
        y0, y1 = (_NET, _L) if half == "near" else ((0.0, _NET) if half == "far" else (0.0, _L))
        poly = [court_to_image(H_inv, cx, cy)
                for cx, cy in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
        poly = [p for p in poly if p is not None]
        if len(poly) < 3:
            raise ValueError("Kort poligonu goruntuye projekte edilemedi. Koseleri kontrol edin.")

    pts = np.clip(np.array(poly, dtype=np.float32), -1e5, 1e5).astype(np.int32)
    cv2.fillConvexPoly(mask, pts, 255)

    rect = cv2.boundingRect(mask)
    if rect[2] == 0 or rect[3] == 0:
        raise ValueError("Arama maskesi bos. Kose secimi hatali olabilir.")

    return mask, rect


# ==============================================================================
# 4) CIZIM
# ==============================================================================

def project_lines(H_inv, lines):
    """
    Dunya cizgilerini goruntuye projekte eder. Kamera sabit oldugu icin BIR KEZ
    cagrilir; sonuc her karede yeniden kullanilir.
    """
    projected = []
    for (ax, ay), (bx, by) in lines:
        p1 = court_to_image(H_inv, ax, ay)
        p2 = court_to_image(H_inv, bx, by)
        if p1 is None or p2 is None:
            continue
        p1 = (int(round(max(-1e4, min(1e4, p1[0])))), int(round(max(-1e4, min(1e4, p1[1])))))
        p2 = (int(round(max(-1e4, min(1e4, p2[0])))), int(round(max(-1e4, min(1e4, p2[1])))))
        projected.append((p1, p2))
    return projected


def draw_lines(frame, projected, color, thickness=None):
    """Onceden projekte edilmis cizgileri cizer."""
    if thickness is None:
        thickness = LINE_THICKNESS
    for p1, p2 in projected:
        cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)


def _put_text(frame, text, org, scale=None, thickness=None):
    """Arka plan golgeli metin (her zeminde okunur)."""
    if scale is None:
        scale = FONT_SCALE
    if thickness is None:
        thickness = FONT_THICKNESS
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                COLOR_TEXT_BG, thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                COLOR_TEXT, thickness, cv2.LINE_AA)


def draw_ref_points(frame, points, show_labels=True):
    """
    Referans noktalarini numarali ve HER NOKTA icin AYRI RENKTE isaretler
    (renkler REF_POINT_COLORS'tan, sira CORNER_LABELS ile birebir eslesir).
    confirm_corners() ve save_first_frame_debug() bu tek fonksiyonu paylasir,
    iki ayri cizim kodu bakimda birbirinden sapmasin diye.

    show_labels=False -> sadece numara (kucuk onay/secim pencereleri icin, yer
                          dar oldugundan tam etiket sigmaz).
    show_labels=True  -> numara + tam referans noktasi aciklamasi (diske
                          kaydedilen dogrulama goruntusu icin).
    """
    h, w = frame.shape[:2]
    for i, (px, py) in enumerate(points):
        p = (int(round(px)), int(round(py)))
        color = REF_POINT_COLORS[i % len(REF_POINT_COLORS)]
        cv2.circle(frame, p, 7, color, -1, cv2.LINE_AA)
        cv2.circle(frame, p, 7, (0, 0, 0), 2, cv2.LINE_AA)   # kontrast halka
        label = CORNER_LABELS[i] if show_labels and i < len(CORNER_LABELS) \
            else str(i + 1)
        # Etiket metni, kare disina tasip kirpilmesin diye nokta sag kenara
        # yakinsa SOLA dogru yazilir (uzun tam etiketler icin onemli).
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        tx = p[0] + 12 if p[0] + 12 + tw < w else p[0] - 12 - tw
        ty = max(16, p[1] - 10)
        _put_text(frame, label, (tx, ty), 0.5, 2)


def save_first_frame_debug(frame, corners, H_inv, path):
    """
    Kort tespitini GORSEL olarak dogrulamak icin ilk kareyi -- uzerine
    projekte edilmis TUM kort cizgileri ve numarali/renkli referans
    noktalariyla birlikte -- diske (output/ altina) kaydeder.

    NEDEN GEREKLI: SHOW_WINDOW=False oldugunda (Raspberry Pi, headless
    calisma) ekranda hicbir gorsel geri bildirim yoktur; kort tespitinin/
    homography'nin dogru calisip calismadigini kontrol etmenin baska yolu
    kalmaz. Bu fonksiyon kurulum sirasinda TEK SEFERLIK cagrilir; video
    isleme dongusune hicbir performans maliyeti eklemez.
    """
    canvas = frame.copy()
    draw_lines(canvas, project_lines(H_inv, COURT_LINES), COLOR_COURT,
               LINE_THICKNESS + 1)
    draw_ref_points(canvas, corners, show_labels=True)
    _put_text(canvas, "Kort tespiti - referans noktalari (ilk kare)", (10, 24), 0.7, 2)

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    cv2.imwrite(path, canvas)
    print("Ilk kare kort tespiti kaydedildi: {}".format(path))


# ==============================================================================
# BOLUM 3 - MEDYAN ARKA PLAN MODELI (eski BOLUM 3 (ARKA PLAN))
# ==============================================================================
# BOLUM 3 (ARKA PLAN)
# -------------
# Sabit kamera icin medyan arka plan modeli.
#
# NEDEN MEDYAN:
#     Kamera sabit ve kortta TEK oyuncu var. Videonun tamamina yayilmis N kare
#     ustunde piksel bazinda medyan alinca oyuncu tamamen silinir: her piksel,
#     zamanin cok buyuk cogunlugunda arka plandir; oyuncu o pikselin uzerinden
#     yalnizca kisa bir sure gecer, yani medyani etkileyemeyecek kadar azinliktadir.
#     Sonuc: oyuncusuz, keskin, gurultusuz bir kort goruntusu.
#
# NEDEN MOG2 DEGIL:
#     MOG2 uyarlanir bir modeldir: duran oyuncu birkac saniye icinde arka plana
#     karisir ve kaybolur. Ayrica isinma (warmup) ister ve sahne kesmesinde coker.
#     Medyan model ise videonun tamamindan BIR KEZ kurulur, duragan oyuncuyu de
#     gorur ve zamanla kaymaz.
#
# NEDEN RENK MODELI DEGIL:
#     BOLUM 4 (TESPIT) eskiden kort renginin LAB medyanina uzakliga bakiyordu. Kortun
#     disindaki kirmizi run-off alani da kort renginden uzak oldugu icin maskenin
#     %31'ini dolduruyor, oyuncu bu dev blob'un icinde kayboluyordu (tespit 0/20).
#     Arka plan cikarma her pikseli KENDI gecmisiyle karsilastirdigi icin bu sorun
#     yapisal olarak ortadan kalkar.

def background_build(video_path, scale, n_samples=None, verbose=True):
    """
    Videodan yayilmis kareler ornekleyip medyan arka plani uretir.

    video_path : VIDEO_PATH (dosya yolu veya webcam indeksi)
    scale      : PROC_SCALE - boru hattiyla AYNI olcek kullanilmalidir.

    Donus: arka plan karesi (BGR, islenmis olcekte) veya kurulamazsa None.
    """
    if n_samples is None:
        n_samples = BG_SAMPLES

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    if total > n_samples:
        # Dosya: videonun TAMAMINA yayilmis kareler (oyuncu her seferinde baska yerde).
        for idx in np.linspace(0, total - 1, n_samples).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, raw = cap.read()
            if ok:
                frames.append(_scaled(raw, scale))
    else:
        # Webcam / kare sayisi bilinmiyor: ardisik kareleri atlayarak topla.
        grabbed = 0
        while len(frames) < n_samples and grabbed < n_samples * BG_STRIDE:
            ok, raw = cap.read()
            if not ok:
                break
            if grabbed % BG_STRIDE == 0:
                frames.append(_scaled(raw, scale))
            grabbed += 1

    cap.release()

    if len(frames) < 3:
        return None

    bg = np.median(np.stack(frames), axis=0).astype(np.uint8)
    if verbose:
        print("Arka plan modeli: {} kare medyani.".format(len(frames)))
    return bg


def _scaled(frame, scale):
    if scale == 1.0:
        return frame
    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=RESIZE_INTERP)


# ==============================================================================
# BOLUM 4 - OYUNCU TESPITI (eski BOLUM 4 (TESPIT))
# ==============================================================================
# BOLUM 4 (TESPIT)
# -----------
# Medyan arka plan cikarma ile oyuncu tespiti. YAPAY ZEKA YOK.
#
# Tespit yalnizca iki durumda calisir:
#   1) Videonun basinda, tracker'i baslatmak icin.
#   2) CSRT oyuncuyu kaybettiginde, yeniden kilitlenmek icin.
#
# NEDEN RENK MODELI BIRAKILDI:
#     Onceki surum kort zemininin LAB medyanina uzakliga bakiyordu. Bu kamerada hic
#     calismadi (20 karede 0 tespit): kortun disindaki KIRMIZI run-off alani da kort
#     yesilinden uzak oldugu icin maskenin ~%31'ini foreground yapiyor, oyuncu bu dev
#     blob'un icinde kayboluyordu. Arka plan cikarma her pikseli KENDI gecmisiyle
#     karsilastirir; run-off, tel orgu, bina - hepsi dogal olarak elenir.
#     Ayni kare kumesinde tespit 0/20 -> 31/40'a cikti ve kutularin TAMAMI dogru
#     boyutta oldu (bkz. BOLUM 3 (ARKA PLAN)).
#
# BORU HATTI:
#     kare - arka plan  ->  mutlak fark  ->  esik  ->  GOVDE maskesi  ->  morfoloji
#     ->  connectedComponentsWithStats  ->  alan/en-boy filtresi
#     ->  AYAK NOKTASI kort dogrulamasi  ->  en buyuk aday
#
# GOVDE MASKESI (kritik):
#     build_search_mask() ZEMIN duzlemi poligonunu verir. Uzak yari bu kamerada
#     yalnizca ~40 piksel yuksekliginde bir seritdir; oyuncu ise DIK durur ve govdesi
#     o seridin cok ustune tasar. Zemin maskesiyle AND'lenince geriye yalnizca ayak
#     bilegi kalir, o da en-boy/alan filtrelerine takilir. Bu yuzden zemin poligonu
#     goruntu duzleminde YUKARI dogru genisletilir (bkz. _rise_px).
#
# AYAK NOKTASI DOGRULAMASI:
#     Maskeyi yukari genisletmek arkadaki agac/tel hareketini de iceri alir. Her
#     adayin ALT-ORTA noktasi homography ile korta cevrilir; istenen yarinin disinda
#     kalan aday atilir. Ruzgarda sallanan agacin "ayagi" kortun metrelerce arkasina
#     duser -> elenir. Bu filtre olmadan tespitlerin yarisi agacti.
#
# DURUM (state):
#     Sinif yerine hafif bir dict. init() ile olusturulur, detect()/verify()'a verilir.

_OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_OPEN_KERNEL)
_CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_CLOSE_KERNEL)


def _rise_px(H_inv):
    """
    Zemin maskesinin goruntude YUKARI genisletilecegi piksel miktari.

    Sihirli sayi degil, geometriden turetilir: bir derinlikteki YATAY olcek (px/m)
    ayni derinlikteki DIK bir nesnenin piksel boyunu da verir (ikisi de bakis
    yonune diktir). File hizasi, aranan uzak yarinin kameraya en yakin - yani en
    buyuk olcekli - kenaridir; en kotu durumu verdigi icin oradan olculur.
    """
    a = court_to_image(H_inv, 0.0, NET_Y)
    b = court_to_image(H_inv, COURT_WIDTH, NET_Y)
    if a is None or b is None:
        return BG_RISE_FALLBACK
    px_per_m = abs(b[0] - a[0]) / COURT_WIDTH
    return int(round(PLAYER_HEIGHT_M * px_per_m))


def detector_init(frame, search_mask, search_rect, background, H, H_inv):
    """
    Tespit durumunu hazirlar. Kamera sabit varsayildigi icin BIR KEZ cagrilir.

    frame       : islenmis kare (PROC_SCALE uygulanmis, BGR)
    search_mask : aranan yarinin ZEMIN maskesi (margin=0 ile kurulmali)
    search_rect : kullanilmaz; rect govde maskesinden yeniden hesaplanir
    background  : background_build() ile uretilen medyan arka plan
    H, H_inv    : homography (ayak dogrulamasi ve olcek icin)
    """
    rise = _rise_px(H_inv)
    body = cv2.dilate(search_mask, np.ones((rise + 1, 1), np.uint8), anchor=(0, 0))
    rect = cv2.boundingRect(body)
    if rect[2] == 0 or rect[3] == 0:
        raise ValueError("Govde maskesi bos. Kose secimini kontrol edin.")

    fh, fw = frame.shape[:2]
    det_area = (fw * DET_SCALE) * (fh * DET_SCALE)

    return {
        "rect": rect,
        "body": body,
        "bg": cv2.GaussianBlur(background, BG_BLUR, 0),
        "H": H,
        "H_inv": H_inv,   # Cozum 2: expected_height() derinlige gore beklenen boyu buradan hesaplar.
        "rise": rise,
        "frame_w": fw,
        "frame_h": fh,
        "min_area": MIN_BLOB_AREA_RATIO * det_area,
        "max_area": MAX_BLOB_AREA_RATIO * det_area,
        "debug_mask": None,
    }


def _foreground(state, frame):
    """Arka plandan sapan pikseller, GOVDE maskesiyle sinirli."""
    diff = cv2.absdiff(cv2.GaussianBlur(frame, BG_BLUR, 0), state["bg"])
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, fg = cv2.threshold(gray, BG_DIFF_THRESH, 255, cv2.THRESH_BINARY)
    cv2.bitwise_and(fg, state["body"], dst=fg)

    it = MORPH_ITERATIONS
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, _OPEN_KERNEL, iterations=it)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, _CLOSE_KERNEL, iterations=it)
    return fg


def _in_search_half(wx, wy):
    """Ayak noktasi PLAYER_HALF yarisinda (+ pay) mi?"""
    m = FOOT_VALIDATE_MARGIN
    if not (-m <= wx <= COURT_WIDTH + m):
        return False
    if PLAYER_HALF == "far":
        return -m <= wy <= NET_Y + m
    if PLAYER_HALF == "near":
        return NET_Y - m <= wy <= COURT_LENGTH + m
    return -m <= wy <= COURT_LENGTH + m


def detect(state, frame):
    """
    Oyuncuyu tespit eder.

    Donus: (x, y, w, h) islenmis kare koordinatlarinda bbox, veya None.
    """
    fg = _foreground(state, frame)
    if DEBUG_SHOW_DETECTION:
        state["debug_mask"] = fg

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(fg, 8, cv2.CV_32S)

    best = None
    best_area = 0
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < state["min_area"] or area > state["max_area"]:
            continue

        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        if bw <= 0:
            continue
        aspect = bh / float(bw)
        if aspect < MIN_ASPECT_RATIO or aspect > MAX_ASPECT_RATIO:
            continue

        bx = stats[i, cv2.CC_STAT_LEFT]
        by = stats[i, cv2.CC_STAT_TOP]

        # Ayak noktasi (alt-orta) istenen yarida mi?
        world = image_to_court(state["H"], bx + bw * 0.5, by + bh)
        if world is None or not _in_search_half(world[0], world[1]):
            continue

        if area > best_area:
            best_area = area
            best = (bx, by, bw, bh)

    if best is None:
        return None

    pad = BBOX_PADDING
    x, y, bw, bh = best
    # ASIMETRIK PADDING: padding UST ve YAN kenarlara uygulanir (CSRT'nin baglam
    # yakalamasi icin faydali), ancak ALT kenara EKLENMEZ. Alt kenar = blob'un
    # GERCEK yer-temas noktasi (ayak). Alta pad eklemek ayak noktasini asagi
    # kaydiriyordu; uzak yaride perspektif bunu buyuk ve TUTARLI bir "kameraya
    # yakin" sistematik hataya ceviriyordu (olculdu: 4 px alt padding ~ 1.5 m).
    # Yukseklik: ust padding korunur (+pad), alt padding kaldirilir -> bh + pad.
    return _clamp_bbox(x - pad, y - pad, bw + 2 * pad, bh + pad,
                       state["frame_w"], state["frame_h"])


def verify(state, frame, bbox):
    """
    Tracker saglik denetimi: bbox icinde arka plandan sapan piksellerin orani.

    Tracker bos zemine kilitlendiyse bu oran cok dusuk cikar -> BOLUM 7 (ANA DONGU) bunu
    TRACK_VERIFY_MIN_FG_RATIO ile karsilastirip yeniden tespit tetikler.
    """
    x, y, w, h = (int(v) for v in bbox)
    x = max(0, min(state["frame_w"] - 1, x))
    y = max(0, min(state["frame_h"] - 1, y))
    w = max(1, min(state["frame_w"] - x, w))
    h = max(1, min(state["frame_h"] - y, h))

    patch = frame[y:y + h, x:x + w]
    if patch.size == 0:
        return 0.0

    diff = cv2.absdiff(cv2.GaussianBlur(patch, BG_BLUR, 0),
                       state["bg"][y:y + h, x:x + w])
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    return float(np.count_nonzero(gray > BG_DIFF_THRESH)) / gray.size


def expected_height(state, foot_x, foot_y):
    """
    Oyuncunun o DERINLIKTE beklenen piksel boyu (Cozum 2 boyut denetimi icin).

    Geometri _rise_px ile ayni: bir derinlikteki YATAY olcek (px/m) ayni
    derinlikteki DIK bir nesnenin piksel boyunu da verir (ikisi de bakis yonune
    diktir). Ayak noktasi once korta cevrilir, sonra o derinligin px/m olcegi
    PLAYER_HEIGHT_M ile carpilir.

    Tracker kutusunun yuksekligi bunun cok disindaysa -ozellikle cok BUYUKse-
    CSRT olcegi kacirmis demektir (kutu oyuncu uzaklasirken kuculmemistir).

    Donus: beklenen yukseklik (px), veya hesaplanamazsa None.
    """
    world = image_to_court(state["H"], foot_x, foot_y)
    if world is None:
        return None
    wy = world[1]
    a = court_to_image(state["H_inv"], 0.0, wy)
    b = court_to_image(state["H_inv"], COURT_WIDTH, wy)
    if a is None or b is None:
        return None
    px_per_m = abs(b[0] - a[0]) / COURT_WIDTH
    return PLAYER_HEIGHT_M * px_per_m


def _clamp_bbox(x, y, w, h, frame_w, frame_h):
    """Bbox'i kare sinirlarina kirpar ve tam sayiya cevirir. Gecersizse None doner."""
    x0 = int(round(max(0.0, x)))
    y0 = int(round(max(0.0, y)))
    x1 = int(round(min(float(frame_w), x + w)))
    y1 = int(round(min(float(frame_h), y + h)))

    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return (x0, y0, x1 - x0, y1 - y0)


# ==============================================================================
# BOLUM 5 - BOLGE MODELI, IZGARA OVERLAY, PANEL (eski BOLUM 5 (BOLGELER))
# ==============================================================================
# BOLUM 5 (BOLGELER)
# --------
# Kort bolge (zone) modeli, perspektif izgara overlay'i ve kus bakisi panel.
#
# IZGARA SEMASI (overlay_rect.py ile birebir ayni yapi ve etiketler):
#
#         +-----------------------------------------------+
#    LT1  |  25   26   27   28   29   30   31   32         |  RT1   <- ust bant (8x1)
#    -----+-----------------------------------------------+-----
#    LT2  |  17   18   19   20   21   22   23   24         |  RT2
#    LT3  |   9   10   11   12   13   14   15   16         |  RT3   <- ana izgara (8x3)
#    LT4  |   1    2    3    4    5    6    7    8         |  RT4      ALTTAN numaralanir
#    -----+-----------------------------------------------+-----
#    LB1  |  1x   2x   3x   4x   5x   6x   7x   8x         |  RB1
#    LB2  |  9x  10x  11x  12x  13x  14x  15x  16x         |  RB2   <- alt bant (8x4)
#    LB3  | 17x  18x  19x  20x  21x  22x  23x  24x         |  RB3      YUKARIDAN sayilir
#    LB4  | 25x  26x  27x  28x  29x  30x  31x  32x         |  RB4
#         +-----------------------------------------------+
#
#     Toplam: 24 (ana) + 8 (ust) + 32 (alt) + 16 (yan) = 80 hucre.
#     Yan serit dilimleri, satir sinirlariyla birebir hizalidir:
#       LT1 -> ust bant, LT2..LT4 -> ana izgara satirlari, LB1..LB4 -> alt bant satirlari.
#
# TASARIM:
#     Bolge sorgusu icin AYRI bir homography gerekmez. Normalize kort koordinati
#     zaten elimizdedir: s = wx / COURT_WIDTH, t = wy / COURT_LENGTH. Bu yuzden
#     get_zone() saf aritmetiktir -> O(1), her karede bedavaya calisir.
#
# ORIJINALDEN TEK FARK:
#     overlay_rect.py'de bant derinlikleri ORANSAL verilir (1/ROWS), yani alt bant
#     kortun 4/3'u kadar (~31.7 m) asagi uzar. Ekran dortgeninde sorun degil, ama
#     tenis kortunda o alan tribun arkasidir. Yapi ve etiketler aynen korunmus,
#     derinlik metre cinsinden ayarlanabilir yapilmistir (BOTTOM_BAND_DEPTH).
#     Birebir orijinal oran icin: BOTTOM_BAND_DEPTH = COURT_LENGTH * BOTTOM_BAND_ROWS / GRID_ROWS
#
# PERFORMANS:
#     Kamera sabit oldugundan izgara, numaralar ve kort cizgileri BIR KEZ statik bir
#     katmana render edilir. Her karede yapilan is sadece:
#       1) cv2.copyTo ile katmanin maskeli kopyalanmasi (tek numpy islemi),
#       2) aktif hucrenin vurgulanmasi (tek poligon).

 
 
_W = COURT_WIDTH
_L = COURT_LENGTH
 
_COLS = GRID_COLS
_ROWS = GRID_ROWS
_TOP_ROWS = TOP_BAND_ROWS
_BOT_ROWS = BOTTOM_BAND_ROWS
 
_TOP_DEPTH = TOP_BAND_DEPTH        # Toplam derinlik (metre).
_BOT_DEPTH = BOTTOM_BAND_DEPTH     # Toplam derinlik (metre).
_SIDE = SIDE_BAND_WIDTH            # Yan serit genisligi (metre).
 
_N_MAIN = _COLS * _ROWS                   # Ana izgara hucre sayisi (varsayilan 24).
 
_CELL_W = _W / float(_COLS)
 
# --- Satir sinirlari (dunya y, metre). Ust banttan alt banda kesintisiz. ---
# Bant sayilari: [0 .. TOP_ROWS-1] ust, [TOP_ROWS .. TOP_ROWS+ROWS-1] ana,
#                [TOP_ROWS+ROWS .. sonu] alt.
_Y_EDGES = []
for _i in range(_TOP_ROWS):
    _Y_EDGES.append(-_TOP_DEPTH + _i * (_TOP_DEPTH / _TOP_ROWS))
for _i in range(_ROWS):
    _Y_EDGES.append(_i * (_L / _ROWS))
for _i in range(_BOT_ROWS + 1):
    _Y_EDGES.append(_L + _i * (_BOT_DEPTH / _BOT_ROWS))
 
_Y_EDGES = np.array(_Y_EDGES, dtype=np.float64)
 
_N_BANDS = len(_Y_EDGES) - 1        # Toplam satir bandi (1 + 3 + 4 = 8)
_Y_MIN = float(_Y_EDGES[0])
_Y_MAX = float(_Y_EDGES[-1])
 
_SIDE_T_COUNT = _TOP_ROWS + _ROWS   # LT dilim sayisi (varsayilan 4)
 
 
# ==============================================================================
# 1) BOLGE SORGUSU (saf aritmetik)
# ==============================================================================
 
def _col_index(wx):
    col = int(wx / _CELL_W)
    return 0 if col < 0 else (_COLS - 1 if col > _COLS - 1 else col)
 
 
def _band_index(wy):
    """Satir bandi indeksi (0 = en ust bant, _N_BANDS-1 = en alt bant)."""
    b = int(np.searchsorted(_Y_EDGES, wy, side="right")) - 1
    if b < 0:
        return 0
    if b > _N_BANDS - 1:
        return _N_BANDS - 1
    return b
 
 
def get_zone(wx, wy):
    """Kort koordinatindan (metre) bolge etiketini dondurur."""
    if wy < _Y_MIN or wy > _Y_MAX or wx < -_SIDE or wx > _W + _SIDE:
        return ZONE_OUT
 
    band = _band_index(wy)
 
    # --- Yan seritler: LT1..LTn / LB1..LBn (yukaridan asagi) ---
    if wx < 0.0 or wx > _W:
        if not SHOW_SIDE_BANDS:
            return ZONE_OUT
        side = "L" if wx < 0.0 else "R"
        if band < _SIDE_T_COUNT:
            return "{}T{}".format(side, band + 1)
        return "{}B{}".format(side, band - _SIDE_T_COUNT + 1)
 
    col = _col_index(wx)
 
    # --- Ust bant: 25..32 ---
    if band < _TOP_ROWS:
        if not SHOW_TOP_BAND:
            return ZONE_OUT
        return str(_N_MAIN + band * _COLS + col + 1)
 
    # --- Ana izgara: 1..24 (ALTTAN yukari) ---
    if band < _TOP_ROWS + _ROWS:
        row = band - _TOP_ROWS                  # 0 = uzak (ust) satir
        return str((_ROWS - 1 - row) * _COLS + col + 1)
 
    # --- Alt bant: 1x..32x (YUKARIDAN asagi) ---
    if not SHOW_BOTTOM_BAND:
        return ZONE_OUT
    brow = band - _TOP_ROWS - _ROWS             # 0 = dip cizgiye en yakin satir
    return "{}x".format(brow * _COLS + col + 1)
 
 
# ==============================================================================
# 2) HUCRE TANIMLARI (dunya koordinatinda)
# ==============================================================================
 
def _build_cells():
    """
    Tum hucreleri (etiket, dunya dortgeni, tur) uretir.
    Tur: "main" | "top" | "bottom" | "side"  -> farkli renklerle cizilir.
    """
    cells = []
 
    for band in range(_N_BANDS):
        y0 = float(_Y_EDGES[band])
        y1 = float(_Y_EDGES[band + 1])
 
        # --- Kort ici hucreler ---
        if band < _TOP_ROWS:
            if SHOW_TOP_BAND:
                for c in range(_COLS):
                    label = str(_N_MAIN + band * _COLS + c + 1)
                    cells.append((label, (c * _CELL_W, y0, (c + 1) * _CELL_W, y1), "top"))
        elif band < _TOP_ROWS + _ROWS:
            row = band - _TOP_ROWS
            for c in range(_COLS):
                label = str((_ROWS - 1 - row) * _COLS + c + 1)
                cells.append((label, (c * _CELL_W, y0, (c + 1) * _CELL_W, y1), "main"))
        else:
            if SHOW_BOTTOM_BAND:
                brow = band - _TOP_ROWS - _ROWS
                for c in range(_COLS):
                    label = "{}x".format(brow * _COLS + c + 1)
                    cells.append((label, (c * _CELL_W, y0, (c + 1) * _CELL_W, y1), "bottom"))
 
        # --- Yan serit dilimleri (her satir bandi icin bir dilim) ---
        if SHOW_SIDE_BANDS:
            if band < _SIDE_T_COUNT:
                l_label = "LT{}".format(band + 1)
                r_label = "RT{}".format(band + 1)
            else:
                l_label = "LB{}".format(band - _SIDE_T_COUNT + 1)
                r_label = "RB{}".format(band - _SIDE_T_COUNT + 1)
            cells.append((l_label, (-_SIDE, y0, 0.0, y1), "side"))
            cells.append((r_label, (_W, y0, _W + _SIDE, y1), "side"))
 
    return cells
 
 
def _project_quad(H_inv, x0, y0, x1, y1):
    """Dunya dortgenini goruntuye projekte eder. Kamera arkasina dusen kose varsa None."""
    pts = []
    for wx, wy in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
        p = court_to_image(H_inv, wx, wy)
        if p is None:
            return None
        px = max(-10000.0, min(10000.0, p[0]))
        py = max(-10000.0, min(10000.0, p[1]))
        pts.append((int(round(px)), int(round(py))))
    return np.array(pts, dtype=np.int32)
 
 
# ==============================================================================
# 3) KURULUM (bir kez calisir)
# ==============================================================================
 
def zones_build(H_inv, frame_shape):
    """
    Bolge sistemini kurar: hucre poligonlari, statik overlay katmani ve panel tabani.
    Kamera sabit varsayildigi icin BIR KEZ cagrilir.
    """
    h, w = frame_shape[:2]
    layer = np.zeros((h, w, 3), dtype=np.uint8)
 
    cells = {}
    for label, quad, kind in _build_cells():
        poly = _project_quad(H_inv, *quad)
        center = court_to_image(H_inv,
                                      (quad[0] + quad[2]) * 0.5,
                                      (quad[1] + quad[3]) * 0.5)
        cells[label] = {"quad": quad, "poly": poly, "center": center, "kind": kind}
 
    color_of = {
        "main": GRID_COLOR,
        "top": TOP_BAND_COLOR,
        "bottom": BOTTOM_BAND_COLOR,
        "side": SIDE_COLOR,
    }
 
    for cell in cells.values():
        if cell["poly"] is None:
            continue
        cv2.polylines(layer, [cell["poly"]], True, color_of[cell["kind"]],
                      GRID_THICKNESS, cv2.LINE_AA)
 
    # Kort cizgileri izgaranin uzerine (daha belirgin).
    draw_lines(layer, project_lines(H_inv, COURT_LINES),
                     COLOR_COURT, LINE_THICKNESS + 1)
 
    if SHOW_GRID_NUMBERS:
        for label, cell in cells.items():
            c = cell["center"]
            if c is None:
                continue
            cx, cy = int(round(c[0])), int(round(c[1]))
            if cx < 0 or cy < 0 or cx >= w or cy >= h:
                continue
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                          GRID_NUMBER_SCALE, 2)
            org = (cx - tw // 2, cy + th // 2)
            cv2.putText(layer, label, org, cv2.FONT_HERSHEY_SIMPLEX,
                        GRID_NUMBER_SCALE, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(layer, label, org, cv2.FONT_HERSHEY_SIMPLEX,
                        GRID_NUMBER_SCALE, GRID_NUMBER_COLOR, 1, cv2.LINE_AA)
 
    # Maske: katmanda cizim olan pikseller. Her karede sadece bunlar kopyalanir.
    gray = cv2.cvtColor(layer, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
 
    state = {"cells": cells, "layer": layer, "mask": mask}
 
    if SHOW_PANEL:
        _build_panel(state)
 
    return state
 
 
def _world_to_panel(wx, wy):
    """Kort koordinatini panel piksel koordinatina cevirir."""
    px = PANEL_MARGIN + (wx + _SIDE) * PANEL_PPM
    py = PANEL_HEADER + PANEL_MARGIN + (wy - _Y_MIN) * PANEL_PPM
    return (int(round(px)), int(round(py)))
 
 
def panel_size():
    """Panelin (genislik, yukseklik) degeri. HUD'u panelin yanina hizalamak icin."""
    pw = int(round((_W + 2 * _SIDE) * PANEL_PPM)) + 2 * PANEL_MARGIN
    ph = int(round((_Y_MAX - _Y_MIN) * PANEL_PPM)) + 2 * PANEL_MARGIN \
        + PANEL_HEADER
    return pw, ph
 
 
def _build_panel(state):
    """Kus bakisi panelin degismeyen tabanini uretir (bir kez)."""
    pw, ph = panel_size()
 
    base = np.zeros((ph, pw, 3), dtype=np.uint8)
    base[:] = PANEL_BG
    cv2.rectangle(base, (0, 0), (pw - 1, PANEL_HEADER - 1),
                  PANEL_HEADER_BG, -1)
 
    rects = {}
    color_of = {
        "main": PANEL_GRID_COLOR,
        "top": PANEL_BAND_COLOR,
        "bottom": PANEL_BAND_COLOR,
        "side": PANEL_SIDE_COLOR,
    }
 
    for label, cell in state["cells"].items():
        x0, y0, x1, y1 = cell["quad"]
        p0 = _world_to_panel(x0, y0)
        p1 = _world_to_panel(x1, y1)
        rects[label] = (p0[0], p0[1], p1[0], p1[1])
 
        cv2.rectangle(base, p0, p1, color_of[cell["kind"]], 1)
 
        if PANEL_SHOW_NUMBERS:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                          PANEL_NUMBER_SCALE, 1)
            cx = (p0[0] + p1[0]) // 2 - tw // 2
            cy = (p0[1] + p1[1]) // 2 + th // 2
            cv2.putText(base, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                        PANEL_NUMBER_SCALE, PANEL_NUMBER_COLOR, 1, cv2.LINE_AA)
 
    # Kort cizgileri (panel uzerinde beyaz ve belirgin).
    for (ax, ay), (bx, by) in COURT_LINES:
        cv2.line(base, _world_to_panel(ax, ay), _world_to_panel(bx, by),
                 PANEL_COURT_COLOR, 1, cv2.LINE_AA)
    cv2.line(base, _world_to_panel(0.0, NET_Y),
             _world_to_panel(_W, NET_Y), PANEL_COURT_COLOR, 2, cv2.LINE_AA)
 
    state["panel"] = base
    state["panel_rects"] = rects
 
 
# ==============================================================================
# 4) CIZIM (her karede)
# ==============================================================================
 
def draw_overlay(frame, state, label=None):
    """
    Statik izgara katmanini kareye kopyalar ve aktif hucreyi vurgular.
    Katman onceden render edildigi icin maliyet tek bir maskeli kopyadir.
    """
    cv2.copyTo(state["layer"], state["mask"], frame)
 
    if label is None:
        return frame
 
    cell = state["cells"].get(label)
    if cell is None or cell["poly"] is None:
        return frame
 
    poly = cell["poly"]
    x, y, w, h = cv2.boundingRect(poly)
 
    fh, fw = frame.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(fw, x + w), min(fh, y + h)
    if x1 - x0 < 1 or y1 - y0 < 1:
        return frame
 
    roi = frame[y0:y1, x0:x1]
    tinted = roi.copy()
    shifted = poly - np.array([[x0, y0]], dtype=np.int32)
    cv2.fillConvexPoly(tinted, shifted, HIGHLIGHT_COLOR, cv2.LINE_AA)
    cv2.addWeighted(tinted, HIGHLIGHT_ALPHA, roi,
                    1.0 - HIGHLIGHT_ALPHA, 0.0, roi)
 
    cv2.polylines(frame, [poly], True, HIGHLIGHT_BORDER, 2, cv2.LINE_AA)
    return frame
 
 
def draw_panel(frame, state, label=None, world_pos=None, trail=None):
    """Kus bakisi paneli kareye yerlestirir: aktif hucre vurgulu, oyuncu ve izi ile."""
    if "panel" not in state:
        return frame
 
    view = state["panel"].copy()
 
    if label is not None and label in state["panel_rects"]:
        x0, y0, x1, y1 = state["panel_rects"][label]
        tinted = view.copy()
        cv2.rectangle(tinted, (x0, y0), (x1, y1), PANEL_HIGHLIGHT, -1)
        cv2.addWeighted(tinted, PANEL_HIGHLIGHT_ALPHA, view,
                        1.0 - PANEL_HIGHLIGHT_ALPHA, 0.0, view)
        cv2.rectangle(view, (x0, y0), (x1, y1), PANEL_HIGHLIGHT_BORDER, 2)
 
    if trail and len(trail) > 1:
        pts = np.array([_world_to_panel(px, py) for px, py in trail], dtype=np.int32)
        cv2.polylines(view, [pts], False, PANEL_TRAIL_COLOR, 1, cv2.LINE_AA)
 
    if world_pos is not None:
        p = _world_to_panel(world_pos[0], world_pos[1])
        cv2.circle(view, p, 4, PANEL_PLAYER_COLOR, -1, cv2.LINE_AA)
        cv2.circle(view, p, 6, PANEL_COURT_COLOR, 1, cv2.LINE_AA)
 
    text = "BOLGE: {}".format(label if label else "-")
    cv2.putText(view, text, (8, PANEL_HEADER - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, PANEL_TEXT_COLOR, 1, cv2.LINE_AA)
 
    fh, fw = frame.shape[:2]
    ph, pw = view.shape[:2]
    x, y = PANEL_POS
    if x < 0 or y < 0 or x + pw > fw or y + ph > fh:
        return frame  # Sigmiyorsa cizme (PANEL_PPM'i kucult).
 
    roi = frame[y:y + ph, x:x + pw]
    cv2.addWeighted(view, PANEL_ALPHA, roi, 1.0 - PANEL_ALPHA, 0.0, roi)
    cv2.rectangle(frame, (x, y), (x + pw - 1, y + ph - 1), PANEL_BORDER, 1)
    return frame


# ==============================================================================
# BOLUM 6 - SAHNE KESME TESPITI (eski BOLUM 6 (KESME))
# ==============================================================================
# BOLUM 6 (KESME)
# -------
# Sahne kesme (cut) tespiti.
#
# AMAC:
#     Kurgulanmis / kesintili videolarda kamera ayni kalir ama zaman atlar; oyuncu
#     bir kareden digerine "isinlanir". Bu an tespit edilmezse:
#       - CSRT eski konumdaki yamayi arar ve kaybolur,
#       - Kalman eski hiz vektoruyle oyuncuyu yanlis yone ceker,
#       - MOG2 arka plan modeli coker ve yuzlerce kare boyunca cop maske uretir,
#       - MAX_JUMP_PER_FRAME denetimi, dogru olan yeni olcumu "imkansiz sicrama"
#         sayip reddeder ve sistem kendini kilitler.
#
# YONTEM:
#     Gri histogram korelasyonu. Ardisik iki karenin histogramlari
#     cv2.compareHist(..., HISTCMP_CORREL) ile karsilastirilir. Kesmede korelasyon
#     sert duser. Karsilastirma, MOG2 icin zaten kucultulmus olan kare uzerinde
#     yapilir -> ek resize yok, maliyet ihmal edilebilir (~0.1 ms).
#
#     Histogram, kamera sabit oldugu icin fazlasiyla yeterlidir: oyuncunun hareketi
#     histogrami neredeyse hic degistirmez, sahne kesmesi ise sert degistirir.

def shot_init():
    """Kesme tespiti durumunu olusturur."""
    return {"hist": None, "last_cut": -10 ** 9}


def detect_cut(state, frame_small, frame_idx):
    """
    Bu karede sahne kesmesi var mi?

    frame_small : MOG2 icin kucultulmus BGR kare (yeniden olceklenmez).
    frame_idx   : gecerli kare numarasi.

    Donus: True (kesme) / False
    """
    gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

    hist = cv2.calcHist([gray], [0], None, [CUT_HIST_BINS], [0, 256])
    cv2.normalize(hist, hist, 0.0, 1.0, cv2.NORM_MINMAX)

    prev = state["hist"]
    state["hist"] = hist

    if prev is None:
        return False

    # Cok sik tetiklenmeyi engelle (gecis efektleri arka arkaya dusuk skor verebilir).
    if frame_idx - state["last_cut"] < CUT_MIN_GAP:
        return False

    corr = cv2.compareHist(prev, hist, cv2.HISTCMP_CORREL)
    if corr < CUT_CORR_THRESH:
        state["last_cut"] = frame_idx
        return True

    return False


# ==============================================================================
# BOLUM 7 - ANA DONGU (eski BOLUM 7 (ANA DONGU))
# ==============================================================================
# BOLUM 7 (ANA DONGU)
# -------
# Tek oyunculu tenis takip sisteminin ana dongusu.
#
# AKIS:
#     1) Video acilir, ilk kare PROC_SCALE ile olceklenir.
#     2) Kortun 4 kosesi yuklenir veya kullaniciya sectirilir.
#     3) Homography kurulur; izgara, overlay katmani ve panel BIR KEZ hazirlanir.
#     4) Oyuncu klasik goruntu isleme ile tespit edilir (BOLUM 4 (TESPIT)).
#     5) CSRT tracker baslatilir.
#     6) Her karede once SAHNE KESME kontrol edilir, sonra tracker guncellenir.
#     7) MOG2 SADECE bbox icindeki hareket enerjisini olcmek icin okunur.
#     8) Ayak noktasi homography ile kort koordinatina cevrilir.
#     9) Kalman yumusatir; bolge (get_zone) saf aritmetikle belirlenir.
#    10) Izgara overlay'i + aktif hucre vurgusu + kus bakisi panel cizilir.
#
# SAHNE KESME (cut) YONETIMI:
#     Kamera ayni kalir (homography gecerli), ama zaman atlar; oyuncu "isinlanir".
#     Kesme aninda DORT sey birden sifirlanir:
#       - CSRT tracker  : eski yamayi ariyordu, gecersiz.
#       - Kalman        : eski hiz vektoru oyuncuyu yanlis yone ceker.
#       - MOG2 modeli   : arka plan modeli coker; ogrenmesi beklenmez, YENIDEN kurulur.
#       - Son konum     : sifirlanmazsa MAX_JUMP denetimi, isinlanma sonrasi DOGRU
#                         olcumu "imkansiz sicrama" sayip reddeder ve sistem kilitlenir.
#
# KONTROLLER:
#     q      : cikis
#     BOSLUK : duraklat / devam
#     r      : elle yeniden tespit

put_text = _put_text


def _make_tracker(name):
    """Adi verilen takipciyi OpenCV surum farklarini tolere ederek olusturur (yoksa None)."""
    for factory in (
        lambda: getattr(cv2, "Tracker" + name + "_create")(),
        lambda: getattr(cv2, "Tracker" + name).create(),
        lambda: getattr(cv2.legacy, "Tracker" + name + "_create")(),
    ):
        try:
            return factory()
        except Exception:
            continue
    return None


def _make_csrt_fast():
    """
    CSRT, HIZ icin kisilmis parametrelerle (yoksa None).

    Hibrit resync kutuyu duzenli olarak tespitle tazeledigi icin CSRT'nin pahali
    33-olcekli aramasi ve segmentasyonu buyuk olcude GEREKSIZ. Bunlari kismak
    dogrulugu neredeyse hic bozmadan ~4x hizlandirir (olculdu: 88.8 -> 20.5 ms/kare).
    """
    try:
        p = cv2.TrackerCSRT_Params()
        p.number_of_scales = CSRT_FAST_SCALES
        p.admm_iterations = CSRT_FAST_ADMM
        p.template_size = CSRT_FAST_TEMPLATE
        p.use_segmentation = CSRT_FAST_SEGMENTATION
        return cv2.TrackerCSRT_create(p)
    except Exception:
        return None


def create_tracker():
    """
    TRACKER_TYPE'a gore takipci olusturur (OpenCV surumunden bagimsiz).

    NEDEN SECILEBILIR: CSRT en dogru ama EN YAVAS takipci; FPS darbogazi oydu
    (~88.8 ms/kare -> ~11 FPS tavani). Hibrit resync (TRACK_RESYNC_INTERVAL)
    kutuyu her ~18 karede tespitle senkronladigi icin takipcinin kendi OLCEK
    aramasi gereksizdir: takipci yalnizca iki resync arasinda konumu tutar.
    Olculen maliyetler ve secenekler icin bkz. BOLUM 1, "TAKIPCI SECIMI".

    Istenen takipci bu OpenCV surumunde yoksa CSRT'ye geri dusulur (cokmez).
    """
    kind = str(TRACKER_TYPE).upper()
    tracker = _make_csrt_fast() if kind == "CSRT_FAST" else _make_tracker(kind)

    if tracker is None and kind != "CSRT":
        print("UYARI: '{}' takipcisi bulunamadi -> CSRT'ye donuluyor.".format(kind))
        tracker = _make_tracker("CSRT")

    if tracker is None:
        raise RuntimeError(
            "Takipci olusturulamadi ({}). 'pip install opencv-contrib-python' gerekli.".format(kind))
    return tracker


# ==============================================================================
# BOLUM 6b - BOLGE GECIS ANALIZI (eski analysis.py)
# ==============================================================================
# Oyuncu bir bolgeden digerine gectiginde BIR olay kaydeder (kare basina DEGIL).
# analysis.init -> analysis_init olarak yeniden adlandirildi (init cakismasi).

def analysis_init(fps):
    """Gecis analizi durumunu olusturur. fps = efektif FPS (src_fps / FRAME_SKIP)."""
    return {
        "fps": float(fps) if fps and fps > 0 else 30.0,
        "events": [],
        "cur_zone": None,
        "enter_frame": 0,
        "dwell": {},
        "visits": {},
        "recent": [],
    }


def analysis_update(state, zone, frame_idx, pos):
    """Her karede cagrilir. Bolge DEGISTIYSE bir gecis olayi kaydeder."""
    if zone is None:
        return

    if state["cur_zone"] is None:
        state["cur_zone"] = zone
        state["enter_frame"] = frame_idx
        state["visits"][zone] = state["visits"].get(zone, 0) + 1
        return

    if zone == state["cur_zone"]:
        return

    # --- GECIS ---
    prev = state["cur_zone"]
    dwell_frames = max(0, frame_idx - state["enter_frame"])
    dwell_s = dwell_frames / state["fps"]

    state["dwell"][prev] = state["dwell"].get(prev, 0) + dwell_frames
    state["events"].append({
        "olay": len(state["events"]) + 1,
        "kare": frame_idx,
        "saniye": round(frame_idx / state["fps"], 2),
        "onceki_bolge": prev,
        "yeni_bolge": zone,
        "onceki_bolgede_sure_sn": round(dwell_s, 2),
        "x_m": round(pos[0], 2) if pos is not None else "",
        "y_m": round(pos[1], 2) if pos is not None else "",
    })
    state["recent"].append("{} -> {}  ({:.1f} sn)".format(prev, zone, dwell_s))

    state["cur_zone"] = zone
    state["enter_frame"] = frame_idx
    state["visits"][zone] = state["visits"].get(zone, 0) + 1


def analysis_draw_panel(state, frame, max_lines=6):
    """Output videosunun SAG-ALT kosesine son bolge gecislerini yazan textbox."""
    recent = state["recent"][-max_lines:]
    cur = "SIMDIKI: {}".format(state["cur_zone"] if state["cur_zone"] is not None else "-")
    lines = ["BOLGE GECISLERI"] + (recent if recent else ["(henuz gecis yok)"]) + [cur]

    fh, fw = frame.shape[:2]
    pad, line_h, box_w = 8, 20, 300
    box_h = pad * 2 + line_h * len(lines)
    x1, y1 = max(0, fw - box_w - 10), max(0, fh - box_h - 10)
    x2, y2 = fw - 10, fh - 10

    roi = frame[y1:y2, x1:x2]
    if roi.size:
        dark = roi.copy()
        dark[:] = (30, 30, 30)
        cv2.addWeighted(dark, 0.6, roi, 0.4, 0.0, roi)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

    y = y1 + pad + 12
    for i, txt in enumerate(lines):
        if i == 0:
            color = (0, 255, 255)
        elif txt.startswith("SIMDIKI"):
            color = (0, 200, 255)
        else:
            color = (255, 255, 255)
        cv2.putText(frame, txt, (x1 + pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, txt, (x1 + pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    color, 1, cv2.LINE_AA)
        y += line_h
    return frame


def analysis_save(state, final_frame, csv_path, json_path):
    """Gecis olaylarini CSV'ye, bolge basina ozeti JSON'a yazar."""
    if state["cur_zone"] is not None:
        rem = max(0, final_frame - state["enter_frame"])
        state["dwell"][state["cur_zone"]] = state["dwell"].get(state["cur_zone"], 0) + rem

    for path in (csv_path, json_path):
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)

    cols = ["olay", "kare", "saniye", "onceki_bolge", "yeni_bolge",
            "onceki_bolgede_sure_sn", "x_m", "y_m"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(state["events"])

    fps = state["fps"]
    summary = {
        z: {"toplam_sure_sn": round(frames / fps, 2),
            "ziyaret_sayisi": state["visits"].get(z, 0)}
        for z, frames in sorted(state["dwell"].items())
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"toplam_gecis": len(state["events"]),
                   "bolge_ozeti": summary,
                   "olaylar": state["events"]}, f, ensure_ascii=False, indent=2)

    return len(state["events"])


def ask_tracker():
    """
    Baslangicta hangi takipcinin kullanilacagini sorar ve TRACKER_TYPE'i ayarlar.

    ENTER       -> varsayilan kullanilir.
    Numara/isim -> o takipci secilir. Gecersiz girdi -> varsayilan.

    ASK_TRACKER = False ise hic sorulmaz (Pi'de otomatik baslatma icin).
    stdin yoksa (servis/autostart) EOFError yakalanir; program KILITLENMEZ,
    varsayilanla devam eder.
    """
    global TRACKER_TYPE

    if not ASK_TRACKER:
        return

    options = (
        ("KCF", "~17 ms/kare   - dengeli"),
        ("MOSSE", "~0.6 ms/kare  - en hizli (Raspberry Pi icin)"),
        ("CSRT", "~94 ms/kare   - en dogru ama cok yavas"),
        ("CSRT_FAST", "~21 ms/kare   - CSRT, olcek/segmentasyon kisilmis"),
    )
    names = [n for n, _ in options]
    default = str(TRACKER_TYPE).upper()

    print("\nHangi takipci kullanilsin?")
    for i, (name, note) in enumerate(options, 1):
        print("  {}) {:<10} {}{}".format(
            i, name, note, "   <- varsayilan" if name == default else ""))

    try:
        raw = input("Secim (numara veya isim, ENTER = {}): ".format(default)).strip()
    except (EOFError, KeyboardInterrupt):
        raw = ""   # stdin yok (autostart) -> varsayilanla devam.

    if raw:
        if raw.isdigit() and 1 <= int(raw) <= len(names):
            TRACKER_TYPE = names[int(raw) - 1]
        elif raw.upper() in names:
            TRACKER_TYPE = raw.upper()
        else:
            print("Gecersiz secim -> varsayilan kullanilacak.")

    print("Takipci: {}\n".format(TRACKER_TYPE))


def create_mog():
    """
    MOG2 arka plan cikaricisi. Sahne kesmesinde YENIDEN cagrilir: eski model
    coktugu icin ogrenmesini beklemek yuzlerce kare cop maske uretmek demektir.
    """
    return cv2.createBackgroundSubtractorMOG2(
        history=MOG2_HISTORY,
        varThreshold=MOG2_VAR_THRESHOLD,
        detectShadows=MOG2_DETECT_SHADOWS,
    )


def create_kalman():
    """Kort duzleminde (metre) sabit hizli Kalman. Durum: [x, y, vx, vy]."""
    dt = float(FRAME_SKIP)
    kf = cv2.KalmanFilter(4, 2, 0, cv2.CV_32F)

    kf.transitionMatrix = np.array([[1, 0, dt, 0],
                                    [0, 1, 0, dt],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], dtype=np.float32)
    kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], dtype=np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * KF_PROCESS_NOISE
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * KF_MEASUREMENT_NOISE
    kf.errorCovPost = np.eye(4, dtype=np.float32) * KF_ERROR_COV
    return kf


def is_bbox_valid(bbox, ref_area, frame_w, frame_h):
    """Tracker bbox'i geometrik olarak makul mu? (ucuz on denetim)"""
    x, y, w, h = bbox
    if w < 2 or h < 2:
        return False
    if x < 0 or y < 0 or x + w > frame_w or y + h > frame_h:
        return False

    aspect = h / float(w)
    if aspect < BBOX_ASPECT_MIN or aspect > BBOX_ASPECT_MAX:
        return False

    if ref_area > 0:
        ratio = (w * h) / float(ref_area)
        if ratio < BBOX_AREA_MIN_RATIO or ratio > BBOX_AREA_MAX_RATIO:
            return False

    return True


def foot_point(bbox):
    """Homography yer duzlemini esler -> gercek konum icin bbox'in ALT-ORTA noktasi."""
    x, y, w, h = bbox
    if USE_FOOT_POINT:
        return (x + w * 0.5, y + h * FOOT_POINT_RATIO)
    return (x + w * 0.5, y + h * 0.5)


def motion_energy(fgmask, bbox, mog_scale):
    """MOG2 maskesinin YALNIZCA bbox icindeki foreground orani (0.0 - 1.0)."""
    mh, mw = fgmask.shape[:2]
    x, y, w, h = bbox

    x0 = max(0, min(mw - 1, int(x * mog_scale)))
    y0 = max(0, min(mh - 1, int(y * mog_scale)))
    x1 = max(x0 + 1, min(mw, int((x + w) * mog_scale)))
    y1 = max(y0 + 1, min(mh, int((y + h) * mog_scale)))

    patch = fgmask[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0

    # detectShadows=True ise golgeler 127 ile etiketlenir; sadece 255'leri say.
    return float(cv2.countNonZero(cv2.compare(patch, 200, cv2.CMP_GE))) / patch.size


def main():
    # Takipci secimi EN BASTA sorulur: agir kurulumdan (arka plan modeli, kose
    # secimi) once cevap alinsin, kullanici bosuna beklemesin.
    ask_tracker()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise IOError("Video acilamadi: {}".format(VIDEO_PATH))

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 1.0 or src_fps > 240.0:
        src_fps = FALLBACK_FPS

    ok, frame = cap.read()
    if not ok:
        raise IOError("Videodan ilk kare okunamadi.")

    scale = float(PROC_SCALE)
    if scale != 1.0:
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=RESIZE_INTERP)
    frame_h, frame_w = frame.shape[:2]

    # --- Kort kurulumu (BIR KEZ; kamera sabit) ---
    # Sira: kayitli koseler -> otomatik tespit -> elle secim.
    #
    # NEDEN confirm_corners/select_corners SHOW_WINDOW'A BAGLI: ikisi de
    # cv2.namedWindow/imshow/waitKey kullanir; SHOW_WINDOW=False (headless Pi,
    # DISPLAY yok) oldugunda bu cagrilar Qt/X11 hatasiyla patlar ("could not
    # connect to display", "xcb platform plugin"). Ekran yoksa zaten ne onay
    # penceresi ne fare ile secim mumkundur -> otomatik tespit dogrudan (onaysiz)
    # kabul edilir; guvenlik AUTO_MIN_SCORE esigi + kaydedilen
    # output/first_frame_court_detection.jpg ile (offline) saglanir.
    corners = load_corners(frame.shape)
    if corners is not None:
        print("Kayitli koseler yuklendi: {}".format(CORNERS_PATH))
    else:
        if AUTO_DETECT_CORNERS:
            corners = auto_detect_corners(frame)
            if corners is not None and AUTO_CONFIRM and SHOW_WINDOW:
                if not confirm_corners(frame, corners):
                    corners = None      # Kullanici reddetti -> elle sec.
        if corners is None and SHOW_WINDOW:
            corners = select_corners(frame)
        if corners is None:
            cap.release()
            if not SHOW_WINDOW:
                print(
                    "Otomatik kort tespiti basarisiz VE SHOW_WINDOW=False (headless): "
                    "elle kose secimi icin ekran/fare gerekir, bu ortamda ikisi de yok.\n"
                    "Cozum secenekleri:\n"
                    "  1) Bir kez, ekrani olan bir makinede/oturumda SHOW_WINDOW=True "
                    "yapip elle secim yapin -> koseler court_corners.json'a kaydedilir, "
                    "sonraki calistirmalar headless (SHOW_WINDOW=False) devam edebilir.\n"
                    "  2) Cizgi maskesi / AUTO_MIN_SCORE esiklerini video goruntusune "
                    "gore ayarlayip otomatik tespitin puanini yukseltin (bkz. "
                    "'Otomatik kort tespiti basarisiz (puan ...)' mesaji)."
                )
            else:
                print("Kose secimi iptal edildi.")
            return
        save_corners(corners, frame.shape)
        print("Koseler kaydedildi: {}".format(CORNERS_PATH))

    H, H_inv = compute_homography(corners)

    # Kort tespitini GORSEL olarak dogrulamak icin ilk kareyi + cizilen kort
    # cizgilerini + numarali/renkli referans noktalarini output/'a kaydet.
    # SHOW_WINDOW=False (headless Pi) oldugunda ekranda hicbir gorsel geri
    # bildirim olmadigi icin bu, tespitin dogrulugunu kontrol etmenin tek yolu.
    if SAVE_FIRST_FRAME_DEBUG:
        save_first_frame_debug(frame, corners, H_inv, FIRST_FRAME_DEBUG_PATH)

    # Tespit ROI'si: aranan yarinin ZEMIN poligonu. init bunu oyuncunun
    # govdesini kapsayacak sekilde goruntude yukari genisletir.
    search_mask, search_rect = build_search_mask(frame.shape, H_inv)

    bg = background_build(VIDEO_PATH, scale)
    if bg is None:
        cap.release()
        raise RuntimeError("Arka plan modeli kurulamadi: video okunamadi.")

    det_state = detector_init(frame, search_mask, search_rect, bg, H, H_inv)

    # Izgara, statik overlay katmani ve panel tabani: BIR KEZ render edilir.
    zone_state = zones_build(H_inv, frame.shape)

    # MOG2 yalnizca HUD'daki hareket gostergesini besler (oyuncu bulmak icin DEGIL).
    # USE_MOG2 = False -> hic kurulmaz, ~3.6 ms/kare kazanilir ve hareket etiketi
    # cizilmez. Sahne kesme tespiti bundan bagimsiz calismaya devam eder.
    mog = create_mog() if USE_MOG2 else None
    mog_scale = float(MOG2_SCALE)
    mog_warmup = MOG2_WARMUP if USE_MOG2 else 0

    shot_state = shot_init()

    # Bolge gecis analizi: efektif FPS ile (src_fps / FRAME_SKIP) -> olaylar
    # gercek video saniyesiyle damgalanir.
    eff_fps = src_fps / float(FRAME_SKIP)
    zone_track = analysis_init(eff_fps) if ZONE_ANALYSIS else None

    kf = create_kalman() if USE_KALMAN else None
    kf_ready = False

    writer = None
    if SAVE_OUTPUT:
        # Hedef klasoru olustur: yoksa VideoWriter SESSIZCE basarisiz olur
        # (nesne doner ama hicbir sey yazilmaz, hata da vermez).
        out_dir = os.path.dirname(OUTPUT_PATH)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # Cikti FPS'i: None -> kaynaktan turet. FRAME_SKIP'e bolunur cunku her
        # FRAME_SKIP karede yalnizca BIR kare yazilir; bolunmezse cikti hizlanir.
        out_fps = OUTPUT_FPS if OUTPUT_FPS else (src_fps / float(FRAME_SKIP))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, out_fps, (frame_w, frame_h))

        # Sessiz basarisizligi yakala: codec/yol sorunlarinda isOpened() False doner.
        if not writer.isOpened():
            cap.release()
            raise IOError("Cikti videosu acilamadi: {} (codec/yol sorunu?)".format(OUTPUT_PATH))
        print("Cikti kaydi: {}  ({}x{} @ {:.2f} FPS)".format(
            OUTPUT_PATH, frame_w, frame_h, out_fps))

    # HUD'u panelin sagina hizala.
    hud_x = 10
    if SHOW_PANEL:
        hud_x = PANEL_POS[0] + panel_size()[0] + 12

    # --------------------------------------------------------------------------
    # KAYDIRMA CUBUGU (YouTube tarzi seek) - OpenCV HighGUI trackbar; ek bagimlilik YOK.
    # Yalnizca SEEK edilebilir DOSYA kaynaginda kurulur (webcam veya kare sayisi
    # bilinmiyorsa otomatik atlanir). Cubugu suruklemek videonun herhangi bir anina atlar.
    # --------------------------------------------------------------------------
    window_name = "Tenis Oyuncu Takibi"
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    seekable = (SHOW_WINDOW and SHOW_SEEKBAR
                and not isinstance(VIDEO_PATH, int) and total_frames > 1)
    seek_to = [None]        # callback -> ana dongu: bekleyen hedef kare (kaynak indeksi).
    seek_syncing = [False]  # True iken trackbar guncellemesi PROGRAMATIKTIR -> callback yok say.

    def on_seek(pos):
        # Kullanici cubugu suruklerse hedef kareyi kaydeder. Oynatma ilerlerken cubugu
        # BIZ guncelledigimizde (setTrackbarPos) seek_syncing True olur; boylece programatik
        # guncelleme yanlislikla yeni bir seek tetiklemez (geri besleme dongusu engellenir).
        if not seek_syncing[0]:
            seek_to[0] = pos

    if seekable:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar("Kare", window_name, 0, total_frames - 1, on_seek)

    tracker = None
    bbox = None
    ref_area = 0
    fail_count = 0
    last_detect_frame = -10 ** 9
    position = None
    trail = deque(maxlen=max(1, TRAIL_LENGTH))
    smooth_motion = 0.0
    fps = 0.0
    frame_idx = 0
    cut_count = 0
    cut_banner = 0
    paused = False
    status = "LOST"

    max_jump = MAX_JUMP_PER_FRAME * FRAME_SKIP

    while True:
        # ----------------------------------------------------------------------
        # BEKLEYEN SEEK ISTEGI (kaydirma cubugu) - oynatirken de duraklatilmisken de.
        # Bir ana atlamak SAHNE KESMESI gibidir: tracker/Kalman/MOG2/son konum
        # gecersizdir. Kesme yolundaki ayni sifirlama uygulanir ki sistem yeni
        # konumda kullanici mudahalesi olmadan yeniden kilitlensin.
        # ----------------------------------------------------------------------
        force_process = False
        if seek_to[0] is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(seek_to[0]))
            seek_to[0] = None
            tracker = None
            bbox = None
            ref_area = 0
            fail_count = 0
            kf_ready = False
            position = None
            trail.clear()
            smooth_motion = 0.0
            last_detect_frame = -10 ** 9
            status = "SEEK"
            shot_state = shot_init()      # Kesme histogramini sifirla -> sahte "kesme" sayilmasin.
            if mog is not None and CUT_RESET_MOG2:
                mog = create_mog()        # MOG2 modeli atlamada coker; yeniden kur.
                mog_warmup = MOG2_WARMUP
            force_process = True          # Duraklatilmis olsak bile bu kareyi bir kez isle/goster.

        # Duraklatilmisken yalnizca bir seek istegi geldiyse (force_process) tek kare islenir.
        if paused and not force_process:
            key = cv2.waitKey(30) & 0xFF
            if key == KEY_PAUSE:
                paused = False
            elif key == KEY_QUIT:
                break
            continue

        t0 = time.perf_counter()

        if force_process:
            # Seek sonrasi hedef kareyi buradan oku (ilk-kare istisnasindan bagimsiz).
            ok, raw = cap.read()
            if not ok:
                paused = True     # Video sonuna/otesine atlandi: son duruma don, duraklat.
                continue
            for _ in range(FRAME_SKIP - 1):
                cap.grab()
            frame = cv2.resize(raw, None, fx=scale, fy=scale,
                               interpolation=RESIZE_INTERP) if scale != 1.0 else raw
        elif frame_idx > 0:  # Ilk kare zaten okundu.
            ok, raw = cap.read()
            if not ok:
                break
            for _ in range(FRAME_SKIP - 1):
                cap.grab()  # Atlanan kareler decode edilmez.
            frame = cv2.resize(raw, None, fx=scale, fy=scale,
                               interpolation=RESIZE_INTERP) if scale != 1.0 else raw

        frame_idx += 1

        # Kucultulmus kare HEM sahne kesme HEM MOG2 icin kullanilir; ikisi de
        # kapaliysa hic uretilmez (bos yere resize etme).
        mog_input = None
        if CUT_DETECTION or mog is not None:
            mog_input = cv2.resize(frame, None, fx=mog_scale, fy=mog_scale,
                                   interpolation=RESIZE_INTERP)

        # ----------------------------------------------------------------------
        # 1) SAHNE KESME KONTROLU (MOG2 apply'dan ONCE)
        # ----------------------------------------------------------------------
        cut = detect_cut(shot_state, mog_input, frame_idx) if CUT_DETECTION else False

        if cut:
            cut_count += 1
            cut_banner = CUT_BANNER_FRAMES

            tracker = None
            bbox = None
            ref_area = 0
            fail_count = 0
            kf_ready = False
            position = None      # KRITIK: MAX_JUMP denetimini bu kare icin devre disi birakir.
            trail.clear()
            smooth_motion = 0.0
            last_detect_frame = -10 ** 9   # Ayni karede yeniden tespit tetikle.
            status = "CUT"

            if mog is not None and CUT_RESET_MOG2:
                mog = create_mog()
                mog_warmup = MOG2_WARMUP

            # Tespit modeli (medyan arka plan) kesmede TAZELENMEZ: kamera sabit
            # oldugu icin arka plan kesme boyunca gecerli kalir. Eski renk modeli
            # isik/renk kaymasina duyarli oldugu icin tazelenmesi gerekiyordu.

        # MOG2 her karede guncellenir (kayipken bile), yoksa model donar.
        # Kapaliysa (USE_MOG2 = False) maske uretilmez; motion_ready hep False kalir,
        # boylece asagidaki hareket blogu ve etiketi kendiliginden devre disi kalir.
        fgmask = mog.apply(mog_input, learningRate=MOG2_LEARNING_RATE) \
            if mog is not None else None

        motion_ready = mog is not None and mog_warmup <= 0
        if mog is not None and mog_warmup > 0:
            mog_warmup -= 1

        # ----------------------------------------------------------------------
        # 2) CSRT TAKIBI
        # ----------------------------------------------------------------------
        tracked = False
        if tracker is not None:
            ok_t, raw_box = tracker.update(frame)
            if ok_t:
                candidate = tuple(int(round(v)) for v in raw_box)
                if is_bbox_valid(candidate, ref_area, frame_w, frame_h):
                    bbox = candidate
                    tracked = True

            # Periyodik saglik denetimi (Cozum 2: otomatik kurtarma).
            if tracked and TRACK_VERIFY_INTERVAL > 0 \
                    and frame_idx % TRACK_VERIFY_INTERVAL == 0:
                # (a) Tracker bos zemine mi kilitlendi? (kutu ici foreground orani)
                if verify(det_state, frame, bbox) < TRACK_VERIFY_MIN_FG_RATIO:
                    tracked = False
                # (b) Kutu boyutu o derinlikte akla yatkin mi? CSRT olcegi zayif
                #     adapte eder; oyuncu uzaklasinca kutu KUCULMEZ, konum kameraya
                #     yakin cikar. Beklenen boy homografiden gelir (expected_height).
                elif TRACK_SIZE_CHECK:
                    exp_h = expected_height(det_state, *foot_point(bbox))
                    if exp_h is not None and exp_h > 1.0:
                        ratio = bbox[3] / exp_h
                        if ratio < TRACK_SIZE_MIN_RATIO or ratio > TRACK_SIZE_MAX_RATIO:
                            tracked = False
                            tracker = None   # Takibi sonlandir -> ayni kare yeniden tespiti tetikler.

        # ----------------------------------------------------------------------
        # 2b) PERIYODIK YENIDEN SENKRONIZASYON (Cozum 1 - hibrit takip)
        # CSRT olcegi zayif adapte eder; oyuncu uzaklasinca kutu kuculmez ve konum
        # kameraya yakin cikar. Bu yuzden tracker BASARILI olsa bile her
        # TRACK_RESYNC_INTERVAL karede bir, mevcut HAFIF tespiti (detect() -> arka
        # plan cikarma, YOLO/agir model DEGIL) calistirip CSRT'yi dogru boyutlu
        # kutuyla yeniden baslatiriz. Boylece birikmis kayma silinir.
        # Kalman/iz sifirlanmaz -> takip surekliligi korunur.
        # ----------------------------------------------------------------------
        if tracked and tracker is not None and TRACK_RESYNC_INTERVAL > 0 \
                and frame_idx % TRACK_RESYNC_INTERVAL == 0:
            fresh = detect(det_state, frame)
            if fresh is not None:
                fw = image_to_court(H, *foot_point(fresh))
                # Yeni tespiti YALNIZCA kort-ici VE mevcut konuma makul mesafedeyse
                # kabul et; aksi halde spurious bir blob takibi kacirabilir.
                if fw is not None and is_on_court(fw[0], fw[1]) and (
                        position is None
                        or ((fw[0] - position[0]) ** 2 + (fw[1] - position[1]) ** 2)
                        <= TRACK_RESYNC_MAX_JUMP ** 2):
                    tracker = create_tracker()
                    tracker.init(frame, fresh)      # CSRT'yi tazelenmis kutuyla yeniden baslat.
                    bbox = fresh
                    ref_area = fresh[2] * fresh[3]  # Boyut referansini da tazele (geometri denetimi icin).

        # ----------------------------------------------------------------------
        # 3) KONUMU KORT DUZLEMINE TASI VE DOGRULA
        # ----------------------------------------------------------------------
        measurement = None
        if tracked:
            fx, fy = foot_point(bbox)
            world = image_to_court(H, fx, fy)

            if world is None or (VALIDATE_POSITION_ON_COURT
                                 and not is_on_court(world[0], world[1])):
                tracked = False
            elif position is not None:
                dx = world[0] - position[0]
                dy = world[1] - position[1]
                if (dx * dx + dy * dy) > (max_jump * max_jump):
                    tracked = False   # Imkansiz sicrama -> tracker baska nesneye atladi.
                else:
                    measurement = world
            else:
                measurement = world

        # ----------------------------------------------------------------------
        # 4) KAYIP YONETIMI VE YENIDEN TESPIT
        # ----------------------------------------------------------------------
        if tracked:
            fail_count = 0
            status = "TRACKING"
        elif not cut:
            fail_count += 1
            if fail_count > TRACKER_FAIL_TOLERANCE:
                tracker = None
                status = "LOST"
            else:
                status = "PREDICT"

        if tracker is None and (frame_idx - last_detect_frame) >= REDETECT_INTERVAL:
            last_detect_frame = frame_idx
            found = detect(det_state, frame)

            if found is not None:
                world = image_to_court(H, *foot_point(found))
                if world is not None and is_on_court(world[0], world[1]):
                    tracker = create_tracker()
                    tracker.init(frame, found)
                    bbox = found
                    ref_area = found[2] * found[3]
                    fail_count = 0
                    tracked = True
                    measurement = world
                    status = "DETECTED"
                    kf_ready = False
                    position = None
                    trail.clear()

        # ----------------------------------------------------------------------
        # 5) KALMAN
        # ----------------------------------------------------------------------
        if kf is not None:
            if not kf_ready:
                if measurement is not None:
                    kf.statePost = np.array([[measurement[0]], [measurement[1]],
                                             [0.0], [0.0]], dtype=np.float32)
                    kf.errorCovPost = np.eye(4, dtype=np.float32) * KF_ERROR_COV
                    kf_ready = True
                    position = measurement
            else:
                pred = kf.predict()
                if measurement is not None:
                    meas = np.array([[measurement[0]], [measurement[1]]], dtype=np.float32)
                    est = kf.correct(meas)
                else:
                    est = pred
                position = (float(est[0, 0]), float(est[1, 0]))
        elif measurement is not None:
            position = measurement

        display_pos = position if status in ("TRACKING", "DETECTED", "PREDICT") else None
        if display_pos is not None:
            trail.append(display_pos)

        zone = get_zone(display_pos[0], display_pos[1]) if display_pos else None

        # Bolge gecis analizi: bolge degistiyse olay kaydeder (kare basina degil).
        if zone_track is not None:
            analysis_update(zone_track, zone, frame_idx, display_pos)

        # ----------------------------------------------------------------------
        # 6) MOG2: KUTU ICI HAREKET ENERJISI (oyuncu bulmak icin DEGIL)
        # ----------------------------------------------------------------------
        if tracked and bbox is not None and motion_ready:
            energy = motion_energy(fgmask, bbox, mog_scale)
            a = MOTION_SMOOTHING
            smooth_motion = a * energy + (1.0 - a) * smooth_motion
        elif motion_ready:
            smooth_motion *= (1.0 - MOTION_SMOOTHING)

        moving = motion_ready and smooth_motion > MOTION_ENERGY_THRESH

        # ----------------------------------------------------------------------
        # 7) CIZIM
        # ----------------------------------------------------------------------
        # Statik izgara katmani (maskeli kopya) + aktif hucre vurgusu.
        draw_overlay(frame, zone_state, zone)

        if bbox is not None and display_pos is not None:
            x, y, w, h = bbox
            color = COLOR_BBOX if tracked else COLOR_BBOX_LOST
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, BBOX_THICKNESS)
            fx, fy = foot_point(bbox)
            cv2.circle(frame, (int(fx), int(fy)), 4, COLOR_FOOT, -1, cv2.LINE_AA)

        if SHOW_PANEL:
            draw_panel(frame, zone_state, zone, display_pos, list(trail))

        # Bolge gecis paneli (sag-alt): son gecisleri gosteren textbox.
        if zone_track is not None and SHOW_ZONE_PANEL:
            analysis_draw_panel(zone_track, frame, ZONE_PANEL_MAX_LINES)

        hud_y = 24
        if SHOW_FPS:
            put_text(frame, "FPS: {:.1f}".format(fps), (hud_x, hud_y), 0.6, 1)
            hud_y += 24

        put_text(frame, "DURUM: {}".format(status), (hud_x, hud_y), 0.6, 1)
        hud_y += 24
        put_text(frame, "BOLGE: {}".format(zone if zone else "-"), (hud_x, hud_y), 0.7, 2)
        hud_y += 26

        if display_pos is not None:
            put_text(frame, "KONUM: {:.2f} m, {:.2f} m".format(display_pos[0], display_pos[1]),
                     (hud_x, hud_y), 0.6, 1)
            hud_y += 24

        # Hareket gostergesi yalnizca MOG2 acikken cizilir; kapaliyken veri yok
        # (yoksa "ISINIYOR" sonsuza kadar ekranda kalirdi).
        if mog is not None:
            if not motion_ready:
                label, motion_color = "HAREKET: ISINIYOR", COLOR_MOTION_OFF
            elif moving:
                label = "HAREKETLI ({:.2f})".format(smooth_motion)
                motion_color = COLOR_MOTION_ON
            else:
                label = "DURAGAN ({:.2f})".format(smooth_motion)
                motion_color = COLOR_MOTION_OFF

            cv2.putText(frame, label, (hud_x, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        COLOR_TEXT_BG, 3, cv2.LINE_AA)
            cv2.putText(frame, label, (hud_x, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        motion_color, 1, cv2.LINE_AA)
            hud_y += 24

        put_text(frame, "KESME: {}".format(cut_count), (hud_x, hud_y), 0.6, 1)

        if cut_banner > 0:
            cut_banner -= 1
            cv2.rectangle(frame, (0, 0), (frame_w - 1, frame_h - 1), COLOR_CUT, 3)
            put_text(frame, "SAHNE KESME - YENIDEN TESPIT",
                     (frame_w // 2 - 170, frame_h - 20), 0.7, 2)

        # ----------------------------------------------------------------------
        # 8) FPS OLCUMU (islem suresi; goruntuleme haric)
        # ----------------------------------------------------------------------
        dt = time.perf_counter() - t0
        if dt > 0:
            inst = 1.0 / dt
            fps = inst if fps == 0.0 else \
                FPS_SMOOTHING * fps + (1.0 - FPS_SMOOTHING) * inst

        if writer is not None:
            writer.write(frame)

        if DEBUG_SHOW_MASK and fgmask is not None:
            cv2.imshow("MOG2 (sadece hareket analizi)", fgmask)
        if DEBUG_SHOW_DETECTION and det_state["debug_mask"] is not None:
            cv2.imshow("Tespit maskesi", det_state["debug_mask"])

        if SHOW_WINDOW:
            cv2.imshow(window_name, frame)
            if seekable:
                # Cubugu oynatma konumuna esitle. seek_syncing bayragi, bu programatik
                # guncellemenin callback'i (on_seek) tetiklemesini engeller.
                seek_syncing[0] = True
                cv2.setTrackbarPos("Kare", window_name,
                                   min(total_frames - 1, int(cap.get(cv2.CAP_PROP_POS_FRAMES))))
                seek_syncing[0] = False
            key = cv2.waitKey(1) & 0xFF
            if key == KEY_QUIT:
                break
            elif key == KEY_PAUSE:
                paused = True
            elif key == KEY_REDETECT:
                tracker = None
                kf_ready = False
                position = None
                fail_count = TRACKER_FAIL_TOLERANCE + 1
                last_detect_frame = -10 ** 9
                status = "LOST"
                trail.clear()

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    # Bolge gecis olaylarini + ozeti diske yaz (q ile cikista da calisir).
    if zone_track is not None:
        n = analysis_save(zone_track, frame_idx, ZONE_EVENTS_CSV, ZONE_SUMMARY_JSON)
        print("Bolge gecis analizi: {} olay -> {} , {}".format(
            n, ZONE_EVENTS_CSV, ZONE_SUMMARY_JSON))

    print("Islenen kare: {} | Sahne kesme: {} | Ortalama FPS: {:.1f} | Kaynak FPS: {:.1f}".format(
        frame_idx, cut_count, fps, src_fps))


if __name__ == "__main__":
    main()
