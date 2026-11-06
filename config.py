"""
config.py
---------
Tenis oyuncusu takip sisteminin tum ayarlari.
 
Bu dosya sadece sabit tanimlar icerir; hicbir calisma zamani mantigi barindirmaz.
Diger tum moduller (court.py, detector.py, zones.py, shot.py, main.py) ayarlari
buradan okur. Tuning yaparken yalnizca bu dosyaya dokunmak yeterlidir.
 
Koordinat sistemi (dunya / kort duzlemi, metre):
    x -> 0.0 ... COURT_WIDTH   (soldan saga)
    y -> 0.0 ... COURT_LENGTH  (uzak dip cizgi -> yakin dip cizgi)
"""
 
import cv2
import numpy as np

# ==============================================================================
# 1) VIDEO / GENEL CALISMA AYARLARI
# ==============================================================================

VIDEO_PATH = "input/A1.mp4"   # Webcam icin 0 (int).

PROC_SCALE = 0.6   # Tum boru hattinin calistigi kare olcegi. FPS icin dusur.
FRAME_SKIP = 1     # 1 -> her kare islenir (CSRT en dogru; kare atlama takibi bozuyordu).
                   # Hiz gerekiyorsa 2-3 yapilabilir ama CSRT'de accuracy dusuyor.

# Kare KUCULTME interpolasyonu. INTER_AREA en kaliteli ama YAVAS; olculdu:
#     1920x1080 -> 1152x648 : AREA 6.83 ms  vs  LINEAR 2.54 ms
#     1152x648  -> MOG2 girdi: AREA 4.26 ms  vs  LINEAR 0.62 ms
#     -> toplam ~7.9 ms/kare kazanc (48.2 ms bütcede %16).
# Kalite farki bu boru hatti icin onemsiz: ortalama piksel farki 0.78/255 ve
# BG_DIFF_THRESH(=24) esigini asan piksel orani yalnizca %0.096.
#
# TEK YERDE TANIMLI OLMASI SART: background.py (arka plan modeli) ile main.py
# (kareler) AYNI interpolasyonu kullanmak zorunda. Farkli olurlarsa model ile
# kareler sistematik olarak farkli uretilir ve her karede SAHTE foreground cikar.
RESIZE_INTERP = cv2.INTER_LINEAR

# Is parcacikli (threaded) kare okuma: decode + resize arka plan thread'inde yapilir,
# ana isleme ile UST USTE biner -> toplam kare suresi ~ max(decode, isleme). Olculdu:
# decode data-only modda kare suresinin ~%88'i; bunu gizlemek buyuk kazanc.
# KISIT: seek YOK (thread cap'i sahiplenir) -> seekbar ile birlikte kullanilamaz;
# THREADED_READ = True iken seekbar otomatik kapanir. Headless Pi icindir.
THREADED_READ = False   # NOT: bu Pi'de True iken ffmpeg "async_lock" assertion cokmesi
                        # TUTARLI tekrarliyor (libavcodec/pthread_frame.c). False stabil ve
                        # FPS'i neredeyse hic dusurmuyor (olculdu: 46.2 vs 48.1). Cokme
                        # cozulurse tekrar True yapilabilir.
THREADED_QUEUE = 4       # Kac kare onden decode edilecek (bellek/gecikme dengesi).

# ==============================================================================
# 1b) CANLI KAMERA KIPI  (top atma makinesi uzerine monte; gercek zamanli)
# ==============================================================================
# TASARIM KURALI: bu bolumdeki her sey BAYRAKLA acilir. LIVE_MODE=False iken
# boru hatti eskisi gibi dosyadan calisir ve BULGULAR.md'deki olcumler aynen
# yeniden uretilebilir. Canli kip mevcut mimariyi DEGISTIRMEZ, uzerine biner.
#
# DOSYA ile CANLI arasindaki YAPISAL farklar (kod bunlara gore ayrisir):
#   1. Geri sarma yok  -> cap.set(POS_FRAMES) canli kaynakta calismaz.
#   2. Kaynak iki kez acilamaz -> arka plan modeli ACIK cap'ten kurulur.
#   3. Akis bitmez     -> dosya sonu yerine sonsuz dongu + temiz Ctrl-C.
#   4. Kareler birikir -> islenmeyen kare TAMPONDA bekler, gecikme buyur.
#   5. Kort her kurulumda FARKLI acidan gorunur -> kayitli koseler gecersiz.

LIVE_MODE = True         # True -> canli kamera kipi (asagidaki ayarlar devreye girer).
                         # Dosyadan calistirmak istersen False yap (VIDEO_PATH kullanilir).

# Kaynak: "picamera2" (CSI serit kablolu Pi kamera), int (USB webcam indeksi)
# veya str (RTSP/dosya yolu). CSI kamera OpenCV'ye DOGRUDAN gorunmez -- Pi 4 +
# guncel Raspberry Pi OS'ta legacy V4L2 kamera arayuzu kapalidir; bu yuzden
# picamera2 sarmalayicisi gerekir (bkz. livesource.py).
#   "picamera2" -> CSI serit kablolu Pi kamera (Pi calisirken TAKILAMAZ,
#                  yeniden baslatmak gerekir)
#   0, 1, 2 ...  -> USB kamera indeksi (/dev/video0 -> 0). USB kamera calisirken
#                  takilip cikarilabilir; dogru indeksi bulmak icin: ls /dev/video*
#   "rtsp://..." -> ag kamerasi
LIVE_SOURCE = 0
LIVE_WIDTH = 1280        # Kameradan ISTENEN cozunurluk. Yuksek cozunurluk FPS
LIVE_HEIGHT = 720        # yemez (isleme PROC_SCALE ile kucultulur) ama gecikme
LIVE_FPS = 30            # ve bellek maliyeti artar.

# picamera2'de "RGB888" isimlendirmesi yaniltici: dizi pratikte BGR duzeninde
# gelir (OpenCV'nin bekledigi sira). Kamera takilinca renkler TERS gorunurse
# (gok mavisi turuncu cikiyorsa) bunu True yapin -- tek dokunusla duzelir.
LIVE_BGR_SWAP = False
LIVE_WARMUP_SEC = 0.5    # Kamera baslarken pozlama/beyaz dengesi otursun diye beklenen sure.
                         # Onemli: arka plan modeli TAM BU kareler uzerine kurulur;
                         # karanlik ilk kareler modele girerse tespit bozulur.

# En yeni kareyi isle, birikenleri AT. Dosyada her kare islenmeliydi (offline
# dogruluk); canlida ise ONEMLI OLAN GECIKMEDIR: 5 kare geride kalmis dogru bir
# konum, makineye nisan aldirmak icin ise yaramaz. Ayri bir thread surekli okur
# ve yalnizca en son kareyi tutar (bkz. livesource.LatestFrame).
LIVE_DROP_STALE = True
LIVE_FIRST_FRAME_TIMEOUT = 5.0   # Acilista ilk kare icin beklenecek en fazla sure (sn).
LIVE_READ_TIMEOUT = 2.0          # Yeni kare gelmezse okuma kac saniye sonra basarisiz sayilir.
LIVE_MISS_TOLERANCE = 5          # Ust uste kac basarisiz okumadan sonra gercekten koptu sayilir.
                                 # Dosyada okuma hatasi = video sonu; canlida ise
                                 # gecici olabilir, hemen cikmak yanlis olur.

# --- Arka plan tazeleme (YALNIZCA canli kipte; bkz. background.refresh) ------
# Dosyada medyan videonun tamamindan alindigi icin model eskimezdi. Canlida
# model acilistaki birkac saniyeden kurulur; isik/golge degistikce eskir ve
# sahte on plan uretir. Oyuncunun KUTUSU DISLANARAK yavas guncellenir --
# oyuncu modele sizarsa gorunmez olur (MOG2'yi bu yuzden kullanmiyoruz).
BG_REFRESH_INTERVAL = 30   # Kac karede bir tazelensin. 0 -> kapali.
BG_REFRESH_ALPHA = 0.02    # Karisim orani. Kucuk = yavas/kararli. 0.02 ~ 50 tazelemede tam donusum.
BG_REFRESH_PAD = 24        # Oyuncu kutusunun disina eklenen emniyet payi (islenmis olcekte piksel).

# --- Kort tespiti (canli kurulumda kritik) ----------------------------------
# Makine her kurulumda FARKLI acidan bakacagi icin kayitli koseler kullanilamaz;
# sistem kortu her acilista sifirdan bulmak zorunda. Bu, olculmus bir zayiflik:
# otomatik tespit 4 videonun 1'inde tamamen basarisiz olmustu (bkz. BULGULAR.md
# §6) ve ekransiz calismada elle duzeltme mumkun degil.
#
# Iki onlem: (a) tespiti OYUNCUSUZ medyan arka plan uzerinde yap -- oyuncu
# cizgilerin ustunde durdugunda/golge yaptiginda tespiti bozuyordu; (b) tek kare
# yerine birkac goruntu dene ve EN YUKSEK PUANLIYI sec.
COURT_ON_BG = True          # Kort tespitini ham kare yerine medyan arka planda yap.
COURT_DETECT_ATTEMPTS = 4   # Kac aday goruntu denenecek (arka plan + ham kareler).
COURT_ATTEMPT_STRIDE = 15   # Ham kare adaylari arasinda kac kare atlanacak.
COURT_IGNORE_SAVED = True   # Canli kipte court_corners.json'i YOK SAY (acilar degisiyor).

# --- Overlay yayini (telefondan izlemek icin; bkz. mjpeg.py) ----------------
# Pi ekransiz calisacak. Masaustunu acik tutmak hem yavaslatir hem korta laptop
# tasimayi gerektirir. Bunun yerine overlay ag uzerinden MJPEG olarak yayinlanir:
# telefonun tarayicisindan http://<pi-ip>:8080/ acilir. Istemci yokken maliyeti
# sifirdir (kare JPEG'e yalnizca istek varken kodlanir).
STREAM_ENABLED = True
STREAM_PORT = 8080
STREAM_QUALITY = 70      # JPEG kalitesi (1-100). Dusur -> daha az bant genisligi/CPU.
STREAM_MAX_FPS = 12      # Yayin hizi tavani. Isleme hizini SINIRLAMAZ, sadece kodlamayi.

# --- Telefondan kort kurulumu (son care) ------------------------------------
# Otomatik tespit basarisiz olursa program EKRANSIZ ortamda kurtarilamiyordu:
# elle secim fare/ekran ister, Pi'de ikisi de yok -> program mesaj basip
# cikiyordu. Olculen basarisizlik orani 4 videonun 1'i (BULGULAR.md §6), yani
# her dort kurulumdan biri hic baslamazdi. Bu bayrak acikken ayni secim
# tarayicidan yaptirilir: kullanici kortun kenarinda telefonuyla bes noktaya
# dokunur. Ayni port kullanilir (yayin henuz baslamamistir).
SETUP_ON_PHONE = True
SETUP_TIMEOUT_SEC = 600  # Tespit BASARISIZ iken beklenecek sure. Bu durumda
                         # onay gelmezse program baslayamaz, o yuzden uzun.

# Tespit BASARILI olsa bile sayfayi ac (dogrulama icin). Kullanici cizgilerin
# kortla ortustugunu GORUP onaylar; begenmezse noktalari surukleyip duzeltir.
# Onay gelmezse asagidaki sure sonunda otomatik koselerle DEVAM EDILIR --
# yani makine basinda kimse olmasa da sistem tek basina calisabilir.
SETUP_CONFIRM_ON_PHONE = True
SETUP_CONFIRM_TIMEOUT_SEC = 90

# --- PERFORMANS OLCUMU (performance.py) -------------------------------------
# Headless calismada terminale saniyede BIR kez FPS + asama sureleri basar,
# programin sonunda toplu ozet cikarir. Pi <-> Windows karsilastirmasi icindir.
# Algoritmayi ETKILEMEZ; sadece olcum. Kapatmak icin PERF_ENABLED = False.
PERF_ENABLED = True
PERF_INTERVAL = 1.0      # Canli rapor periyodu (saniye).
PERF_AVG_WINDOW = 30     # Hareketli ortalama FPS penceresi (kare sayisi).
 
SHOW_WINDOW = False   # Raspberry Pi (headless): anlik pencere YOK. imshow hic cagrilmaz,
                      # dongu tam hizda calisir. Cikti SAVE_OUTPUT ile dosyaya yazilir.
SHOW_SEEKBAR = True   # Video penceresine YouTube tarzi kaydirma cubugu ekler. Yalnizca
                      # SEEK edilebilir DOSYA kaynaginda VE SHOW_WINDOW acikken kurulur.

# Gorsellestirme (izgara overlay + mini kort paneli + HUD + bbox) yalnizca INSAN
# icindir; get_zone ve bolge CSV/JSON'u tamamen CIZIMDEN BAGIMSIZ calisir. Ekransiz
# Pi'de sadece veri (CSV) uretiliyorsa cizim SAF ISRAFTIR (olculdu ~9.9 ms/kare, %37).
# False -> hicbir sey cizilmez (zone_state bile kurulmaz), veri akisi aynen surer.
# NOT: annotated output videosu istiyorsan True olmali (yoksa video ham cikar).
DRAW_VISUALS = True

SAVE_OUTPUT = True              # Islenmis (overlay + panel + HUD) videoyu diske yaz.
OUTPUT_PATH = "output/tracked.mp4"   # SABLON. Klasor yoksa otomatik olusturulur.

# Cikti dosyasi adina SECILEN TAKIPCI eklensin ve var olan dosyanin UZERINE
# yazmak yerine YENI bir dosya olustursun mu?
#   True  -> "output/tracked.mp4" + MOSSE = "output/tracked_mosse.mp4"; o isim
#            zaten varsa "tracked_mosse_2.mp4", "_3" ... (her calistirma AYRI cikti).
#   False -> OUTPUT_PATH aynen kullanilir (eski davranis: her sefer uzerine yazar).
# Ad, takipci secimi (ask_tracker) KESINLESTIKTEN sonra belirlenir; ENTER'la
# varsayilani secsen bile dogru takipci adi kullanilir.
OUTPUT_NAME_BY_TRACKER = True

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

# --- DEGERLENDIRME (EVAL) LOGU --------------------------------------------------
# Tracker'in kare-kare KUTU + AYAK konumunu CSV'ye yazar (NORMALIZE 0..1). SADECE
# test/dogrulama icin: PC'de YOLO ile uretilen referans CSV ile compare.py'de
# kiyaslanip tracker DOGRULUK skoru (IoU, ayak hatasi) cikarilir. Nihai urunde
# KAPALI olmali. Adil kiyas icin FRAME_SKIP=1 kullan (kare numaralari eslessin).
EVAL_LOG = False
EVAL_LOG_PATH = "output/eval_track.csv"

# Video bitince TRACKER PERFORMANCE SUMMARY raporunu OTOMATIK bas (compare.py).
# Sart: EVAL_LOG acik + asagidaki YOLO referans CSV'si mevcut olmali. Referans
# yoksa rapor sessizce atlanir (calisma etkilenmez).
EVAL_AUTO_REPORT = True
EVAL_REFERENCE_CSV = "referans_A1.csv"   # PC'de yolo_reference.py ile uretilir

# Benchmark SEGMENTI: cok sayida kombinasyon (tracker x filtre) denenirken tum
# videoyu islemek saatler surer. Bu iki ayar isleme penceresini kisitlar; kare
# numaralari kaynak videoya gore yazildigi icin YOLO referansiyla hizali kalir.
#   EVAL_START_FRAME : baslangic kaynak karesi (0 = bastan)
#   EVAL_MAX_FRAMES  : islenecek kare sayisi   (0 = sonuna kadar)
EVAL_START_FRAME = 0
EVAL_MAX_FRAMES = 0
 
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
#     suruklemesini engeller (bkz. court.compute_homography). Uzak dip koseler
#     ARTIK ana capa olarak KULLANILMAZ; sadece court.auto_detect_corners
#     icindeki ROBUST ic hesaplamada (yakin dip + yakin servis + cift yan
#     cizgileri ile kurulan yardimci homography) rol oynarlar, disariya hic
#     verilmezler.
#
# TAM KORT NASIL OLUSUYOR:
#     Homography bir DUZLEMIN tamamini esler, yalnizca secilen noktalari degil.
#     Bu 5 capa yer duzlemini tanimlar; hem uzak yari (y -> 0 .. NET_Y) hem
#     yakin dip (y -> NEAR_SERVICE_Y .. COURT_LENGTH) matematiksel olarak
#     ekstrapole edilir. Izgara bu yuzden eskisi gibi TAM kortu (0 ..
#     COURT_LENGTH) kaplamaya devam eder -> zones.py'de hicbir degisiklik
#     gerekmez, bolge semasi birebir korunur.
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
# 2b) OTOMATIK KORT TESPITI (court.auto_detect_corners)
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

# --- KOSE INCE AYARI (line-mask hizalama; court.refine_corners) --------------
# 5 capa noktasi goruntude ince bir yatay banda sikisir -> homografinin DERINLIK
# yonundeki ekstrapolasyonu zayiftir ve kort modeli gercek korttan kayar
# (ozellikle file-orta ve uzak/yakin uclar). Bu adim, 5 nokta yerine projekte
# edilen TUM kort cizgilerinin gercek cizgi maskesine ortak hizasini kullanir:
# noktalar, projekte cizgiler beyaz cizgilere oturana kadar kucuk adimlarla
# kaydirilir (koordinat inisi / ICP benzeri). ASLA bozmaz: skor iyilesmezse
# orijinal noktalar korunur. Kurulumda TEK SEFER calisir -> FPS'e etkisi yok.
REFINE_CORNERS = True
REFINE_SAMPLES_PER_LINE = 40   # Her kort cizgisi boyunca ornek nokta sayisi.
REFINE_INIT_STEP = 8.0         # Baslangic arama adimi (px); yarilanarak kuculur.
REFINE_MIN_STEP = 0.5          # Bu adimin altinda arama durur (px hassasiyet).
REFINE_MAX_SHIFT = 40.0        # Bir nokta orijinalinden en fazla bu kadar kayabilir
                               # (px) -> yanlis/dis cizgiye kacmayi engeller.
# Robust maliyet: bir ornegin cizgi maskesine uzakligi bu esikte KIRPILIR. Ortulu
# (uzak yari) / yanlis-tespit cizgilerdeki uzak ornekler sabit katki verir, fit'i
# saptirmaz -> optimizasyon yakin/hizali cizgileri sikilastirmaya odaklanir.
REFINE_TRUNCATE = 16.0
# Ornekler yalnizca YAKIN yaridan mi alinsin, yoksa UZAK yariyi da kapsasin mi?
# UZAK yari bu kamerada buyuk olcude file/mesh tarafindan ORTULU oldugu icin
# (olculdu: dahil edilince skor 0.68 -> 0.39'a dustu, cogu ornek hicbir cizgiye
# oturmadi) yalnizca YAKIN yari kullanilir. Yan cizgilerin uzak kisimlari kismen
# gorunse de derinligi kisitlamaya yetmiyor.
REFINE_INCLUDE_FAR_HALF = False

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

# 8x3 ana izgaranin kapladigi dunya-y uzunlugu (metre).
# YARI-KORT SEMASI (referans overlay_rect ile birebir): ana izgara kortun BIR
# YARISINI kaplar -> file (net) izgaranin ALT kenarindadir; kortun DIGER yarisi
# (kameraya yakin yari) alt bantlarla (1x..32x, 8x4) doldurulur. Bu yuzden izgara
# TUM kort (COURT_LENGTH) yerine YARIM kort = NET_Y kadar uzanir (uzak dip -> file).
# Oyuncu bu yarida takip edildigi icin (PLAYER_HALF="far") izgara UZAK yaridir.
GRID_LENGTH = NET_Y

TOP_BAND_ROWS = 1     # Uzak dip cizgi arkasi satir sayisi.
BOTTOM_BAND_ROWS = 4  # File'dan yakin yariya dogru satir sayisi (1x..32x -> 4 satir).

# Bant derinlikleri (metre). Referans (overlay_rect) yapisinda ORANSALDIR: her bant
# satiri, ana izgara satiri kadar derindir (GRID_LENGTH / GRID_ROWS). Boylece ust
# bant 1 satir, alt bant 4 satir olur ve yari-kort semasi birebir korunur.
#   satir yuksekligi = GRID_LENGTH / GRID_ROWS  (= NET_Y / 3 ~ 3.96 m)
# Alt bant (4 satir ~15.85 m) file'dan baslar, yakin yariyi (11.885 m) kaplar ve
# yakin dip cizginin biraz otesine tasar -> referanstaki 1x..32x ile ayni.
TOP_BAND_DEPTH = GRID_LENGTH * TOP_BAND_ROWS / GRID_ROWS        # ~3.96 m
BOTTOM_BAND_DEPTH = GRID_LENGTH * BOTTOM_BAND_ROWS / GRID_ROWS  # ~15.85 m
 
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
# tamami dogru boyutta. Ayrinti: background.py
 
DET_SCALE = 0.5   # Yalnizca blob alan esiklerinin olceklenmesinde kullanilir.
 
# Oyuncunun arandigi kort yarisi. "far" -> kameradan UZAK yari (y: 0 .. NET_Y).
# court.build_search_mask() bunu okuyup tespit ROI'sini kurar.
PLAYER_HALF = "far"     # "near" | "far" | "both"
SEARCH_MARGIN = 3.0     # Arama alanina kort disindan eklenen pay (metre).

# OYUNCU-YARISI KISITI (fiziksel/domain prior): oyuncu yalnizca PLAYER_HALF
# yarisinda olabiliyorsa (or. A1'de hep uzak yari), konum KARSI yariya gecerse
# file cizgisine kirpilir. Sebep: sikisik uzak-yari perspektifi ayagi ara sira
# file otesine tasiyor; bu IMKANSIZ oldugundan gurultu sayilir. "both" veya iki
# oyuncu takip edilecekse False yap. Rapor icin: "oyuncu-yarisi kisiti uygulandi".
CONSTRAIN_PLAYER_HALF = True
 
# --- Medyan arka plan modeli ---
BG_SAMPLES = 25         # Medyan icin videoya yayilmis ornek kare sayisi.
BG_STRIDE = 12          # Webcam / kare sayisi bilinmiyorsa kac karede bir ornek.
BG_BLUR = (5, 5)        # Fark almadan once gurultu bastirma.
BG_DIFF_THRESH = 24     # Arka plandan bu kadar sapan piksel foreground sayilir.
 
# --- Govde maskesi (zemin poligonunun goruntude YUKARI genisletilmesi) ---
# Oyuncu zeminde durur ama goruntude DIK uzanir; zemin poligonu govdeyi kapsamaz.
# Genisletme miktari geometriden turetilir (detector._rise_px), sabit degildir.
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
# 4b) SABIT NESNE: TOP-ATMA MAKINESI (aluminyum/beyaz) TESPITI
# ==============================================================================
# Makine SABIT oldugu icin MEDYAN ARKA PLANDA kalir; kurulumda BIR KEZ aranir
# (detector.detect_machine). Ayirt: dusuk doygunluk + yuksek parlaklik (beyaz/
# aluminyum) + kort-ici + KOMPAKT blob (cizgi/bina degil). Bulunursa BOLGESI,
# bulunamazsa "BULUNAMADI" ekranin SAG UST'unde gosterilir.
# NOT: esikler GERCEK makine goruntusuyle ayarlanacak (henuz veri yok). A1'de
# makine olmadigi icin "bulunamadi" beklenir -> bu, bulunamadi yolunu test eder.
DETECT_MACHINE = True
MACHINE_MIN_VAL = 150        # HSV V alt siniri (parlak: beyaz/aluminyum)
MACHINE_MAX_SAT = 60         # HSV S ust siniri (dusuk doygunluk = beyazimsi/gri)
MACHINE_MORPH_KERNEL = (5, 5)
MACHINE_MIN_AREA = 300       # px^2 (PROC_SCALE olceginde) -> kucuk gurultu elenir
MACHINE_MAX_AREA = 20000     # cok buyuk (zemin/genis alan) elenir
MACHINE_ASPECT_MIN = 0.5     # h/w; ince beyaz CIZGILER (asiri en/boy) elenir
MACHINE_ASPECT_MAX = 4.0
COLOR_MACHINE = (255, 255, 0)  # makine isaret/etiket rengi (BGR, cyan-ish)

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
TRACKER_TYPE = "CSRT_FAST"    # "CSRT" | "CSRT_FAST" | "KCF" | "MOSSE"
# CSRT_FAST secildi: takip kalitesi cok iyi (kullanici tercihi) ve ~20 ms islemesi,
# threaded okumada ~14 ms decode'u tamamen gizler -> hem dogru takip hem yuksek FPS.

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
# hafif tespiti (detector.detect -> arka plan cikarma, YOLO/agir model DEGIL)
# calistirip CSRT'yi dogru boyutlu kutuyla yeniden baslatiriz. Boylece zamanla
# olusan konum ve BOYUT kaymasi periyodik olarak silinir.
TRACK_RESYNC_INTERVAL = 18     # Yeniden senkron periyodu (kare). 0 -> kapali. Onerilen 15-20.
                               # (Kosu jitter'i icin 10->18 geri alindi: sik resync,
                               # kararsiz tespit kutusunu tracker'a daha sik enjekte edip
                               # gurultuyu artiriyordu.)
TRACK_RESYNC_MAX_JUMP = 1.5    # Yeni tespit mevcut konuma bu mesafeden (m) uzaksa spurious
                               # blob sayilir ve resync REDDEDILIR (takip kacmasin diye).

# --- BOYUT AKLA-YATKINLIK DENETIMI (Cozum 2: otomatik kurtarma) -------------
# Homografiden oyuncunun o derinlikteki BEKLENEN piksel boyu hesaplanir
# (detector.expected_height). Tracker kutusunun yuksekligi bu beklentinin cok
# disindaysa -ozellikle cok BUYUKse- CSRT olcegi kacirmis demektir: takibi
# sonlandirip yeniden tespit tetikleriz. "Kutu kuculmuyor" hatasini dogrudan yakalar.
TRACK_SIZE_CHECK = True
TRACK_SIZE_MIN_RATIO = 0.50    # kutu_h < beklenen_h * bu  -> gecersiz (cok kucuk)
TRACK_SIZE_MAX_RATIO = 1.80    # kutu_h > beklenen_h * bu  -> gecersiz (kuculmeyen buyuk kutu)

# --- KUTU YUKSEKLIGI KIRPMA (ayak noktasi duzeltmesi; detector.cap_box_height) --
# Uzak yarida tespit blob'u sisip kutu ayaklarin ALTINA tastiginda ayak noktasi
# (kutu alti) yaniliyordu -> konum metrelerce kayiyor, kareler arasi zipliyor ve
# "cok uzun kutu" boyut denetimi takibi KESIYORDU. Kutu, o derinlikte beklenen
# oyuncu boyunu asarsa USTTEN kirpilir; ayak gercek yer seviyesine oturur ve
# takip KESINTISIZ surer. Kutu normal/kisaysa dokunulmaz.
CAP_BOX_HEIGHT = True
BOX_HEIGHT_CAP_FACTOR = 1.0    # Kirpma esigi = beklenen_boy (PLAYER_HEIGHT_M=2.0m) * bu.
                               # Ayak hala biraz ALTTA kaliyorsa ~0.9'a dusur
                               # (kutu altini ayaklara daha cok yaklastirir).

# --- OTOMATIK YENIDEN TESPIT (ekransiz 'r' tusu) ----------------------------
# SHOW_WINDOW=False (Pi, headless) oldugunda 'r' tusuna basilamaz; bu yuzden
# 'r'nin yaptigi SERT yeniden-tespit belirli araliklarla OTOMATIK tetiklenir.
# SERT = tracker + Kalman + son konum sifirlanir; boylece bir sonraki tespit,
# TRACK_RESYNC'in mesafe kisitini (TRACK_RESYNC_MAX_JUMP) ve MAX_JUMP denetimini
# BYPASS eder.
# NEDEN GEREKLI: resync yalnizca mevcut konuma YAKIN tespitleri kabul eder
# (continuity). Tracker YANLIS ama KARARLI bir yere kilitlenirse (or. oyuncunun
# diz hizasi; ozellikle MOSSE gibi OLCEKSIZ takipcilerde) dogru tespit "cok uzak
# sicrama" sanilip reddedilir ve hata KALICI olur. Sert yeniden-tespit konumu
# sifirladigi icin bu kisiti asar ve dogru tespiti kabul eder.
AUTO_REDETECT = True
AUTO_REDETECT_INTERVAL = 45    # Kac karede bir sert yeniden-tespit (~0.9 sn @50fps).
                               # NOT: cok dusuk (20) YAPMA -> her sert sifirlama MAX_JUMP
                               # korumasini bir kare devre disi birakir; uzak-yari derinlik
                               # cokmesi (y~0) o an suzulup gecer. 45 daha guvenli.

# --- OLCUM YUMUSATMA: MEDYAN FILTRE + BOLGE HISTEREZISI ----------------------
# Kosuda arka-plan-cikarma blob'u kararsiz (genislik 42<->98px zipliyor) -> ayak
# noktasi ve dolayisiyla dunya konumu ani sicramalarla gurultuluydu (jitter). Iki
# katmanli bastirma:
#   1) Konum MEDYAN filtresi: son N olcumun medyani -> tek-kare spike'lari eler
#      (ortalamadan farkli olarak aykiri degerden ETKILENMEZ; konumu DONDURMAZ,
#      gercek hareketi ~1-2 kare gecikmeyle izler). Olculdu: y-jitter 0.164->0.107.
#   2) BOLGE HISTEREZISI: oyuncu sinirdayken bolge etiketinin (or. 5<->6<->5x)
#      cakmasini onler; yeni bolge, ancak K kare US TUSTE gozlenirse gecerli olur.
# Ikisi de takibi KESMEZ (tracker/Kalman'a dokunmaz; sadece olcum/etiket yumusatir).
POSITION_MEDIAN_WINDOW = 5     # Konum medyani penceresi (kare). 1 -> kapali.
ZONE_HYSTERESIS_FRAMES = 5     # Yeni bolgeye gecmek icin gereken ardisik kare. 1 -> kapali.
                               # Dusur -> daha sik toparlar ama iz/Kalman daha sik sifirlanir.
                               # 0 veya AUTO_REDETECT=False -> kapali.

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

# --- KONUM TAHMIN MODELI (predictors.py; baslarken menuden secilir) ---------
# CV  = Sabit Hiz (baseline, en ucuz)     CA  = Sabit Ivme (hizlanma/yavaslama)
# ABG = alfa-beta-gama (hafif)             IMM = Coklu Model (en saglam: dur/sprint/donus)
# Tahmin: (a) kayipta bosluk doldurur, (b) arama penceresini besler. Uzak-yari
# derinlik sinirini COZMEZ (o geometri). Akademik kiyas icin menuden secilir.
PREDICT_TYPE = "CV"     # "CV" | "CA" | "ABG" | "IMM"
ASK_PREDICT = True      # True -> baslarken "hangi tahmin modeli?" sorulur (ask_predict).
                        # False -> dogrudan PREDICT_TYPE (Pi otomatik baslatma).

KF_PROCESS_NOISE = 1e-2        # Kucuk = daha cok YUMUSATMA (olcume daha az guvenir).
                               # Kosudaki blob gurultusunu (jitter) bastirmak icin
                               # dusuk tutulur. (Onceki 3e-2 deneme gurultuyu geciriyordu.)
KF_MEASUREMENT_NOISE = 5e-2    # X (YATAY/sutun) olcum gurultusu.

# --- ANIZOTROPIK DERINLIK SABITLEME (kozmetik) ------------------------------
# Uzak yari goruntude 5x sikisik (0.36 m/px) oldugundan DERINLIK (y) olcumu cok
# gurultuludur; oyuncu orada derinlikte az hareket eder. Bu yuzden Kalman'da y
# olcumune COK az guvenilir (yuksek gurultu) -> y "yapiskan" olur, zone SATIRI'nin
# ucusmasi (or. 22<->31) azalir. X (yatay) hassas kalir. NOT: gercek bir derinlik
# hareketini geciktirir; bu bir kamera-geometrisi band-aid'idir, tam dogru degildir.
# Kapatmak icin KF_MEASUREMENT_NOISE'a esitle (izotropik).
KF_MEASUREMENT_NOISE_Y = 1.0   # Y (DERINLIK) olcum gurultusu (X'in ~20 kati -> yapiskan).
                               # NOT: 10'a kadar denendi; oyuncu uzak DIP CIZGIDE (y=0, hem en
                               # sikisik bolge hem zone SINIRI) dururken zone ucusmasi yine de
                               # tam durmadi -> bu bir kamera-geometrisi siniri, yazilimla
                               # tamamen cozulmez. 1.0 genel uzak-yari jitter'ini yumusatir.
KF_ERROR_COV = 1.0

# alfa-beta-gama (ABG) sabit kazanclari (yalnizca PREDICT_TYPE="ABG").
ABG_ALPHA = 0.5    # konum duzeltme kazanci
ABG_BETA = 0.2     # hiz duzeltme kazanci
ABG_GAMMA = 0.05   # ivme duzeltme kazanci

# IMM parametreleri (yalnizca PREDICT_TYPE="IMM").
IMM_MANEUVER_FACTOR = 50.0   # manevra modunun process noise carpani (dengeli moda gore)
IMM_STAY = 0.95              # markov: bir sonraki karede AYNI modda kalma olasiligi
 
MAX_JUMP_PER_FRAME = 1.0  # metre/kare. Kesmede son konum sifirlandigi icin atlanir.
                          # 1.5->1.0: uzak yari 0.36 m/px (5x dusuk cozunurluk) oldugundan
                          # ayak birkac piksel zipladiginda derinlik metrelerce cokuyor;
                          # bu esik o "imkansiz" derinlik sicramalarini reddeder (Kalman
                          # tahminle duz tutar). 50fps'te gercek hareket zaten <=0.2 m/kare.
 
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
# 11b) BOLGE GECIS ANALIZI (analysis.py)
# ==============================================================================
# Oyuncu bir bolgeden digerine gectikce OLAY kaydedilir (kare basina degil).
# Cikti: output videosuna panel + CSV (her gecis) + JSON (bolge basina ozet).

ZONE_ANALYSIS = True
ZONE_EVENTS_CSV = "output/zone_events.csv"     # Her gecis bir satir.
ZONE_SUMMARY_JSON = "output/zone_summary.json"  # Bolge basina toplam sure + ziyaret.

SHOW_ZONE_PANEL = True        # Output videosunun sag-altina son gecis textbox'i.
ZONE_PANEL_MAX_LINES = 6      # Panelde gosterilecek son gecis sayisi.

# Canli terminal izleme: her ZONE_PRINT_INTERVAL video-saniyesinde bir, oyuncunun
# O ANKI bolgesini terminale basar. Ekransiz Pi'de bolge takibini SSH'tan gormek
# icin. Cizim/video gerekmez; get_zone ciktisini dogrudan terminale dokerek
# "bolge takip kodu bosuna gitmesin" saglar.
ZONE_PRINT = True
ZONE_PRINT_INTERVAL = 1.0     # video-saniyesi (1.0 -> her saniye bir satir).

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