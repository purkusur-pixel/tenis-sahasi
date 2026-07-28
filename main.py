"""
main.py
-------
Tek oyunculu tenis takip sisteminin ana dongusu.

AKIS:
    1) Video acilir, ilk kare PROC_SCALE ile olceklenir.
    2) Kortun 5 referans noktasi (file + yakin servis cizgisi kesisimleri)
       yuklenir veya kullaniciya sectirilir; ilk kare + tespit output/'a
       kaydedilir (bkz. court.save_first_frame_debug).
    3) Homography kurulur; izgara, overlay katmani ve panel BIR KEZ hazirlanir.
    4) Oyuncu klasik goruntu isleme ile tespit edilir (detector.py).
    5) CSRT tracker baslatilir.
    6) Her karede once SAHNE KESME kontrol edilir, sonra tracker guncellenir.
    7) MOG2 SADECE bbox icindeki hareket enerjisini olcmek icin okunur.
    8) Ayak noktasi homography ile kort koordinatina cevrilir.
    9) Kalman yumusatir; bolge (zones.get_zone) saf aritmetikle belirlenir.
   10) Izgara overlay'i + aktif hucre vurgusu + kus bakisi panel cizilir.

SAHNE KESME (cut) YONETIMI:
    Kamera ayni kalir (homography gecerli), ama zaman atlar; oyuncu "isinlanir".
    Kesme aninda DORT sey birden sifirlanir:
      - CSRT tracker  : eski yamayi ariyordu, gecersiz.*
      - Kalman        : eski hiz vektoru oyuncuyu yanlis yone ceker.
      - MOG2 modeli   : arka plan modeli coker; ogrenmesi beklenmez, YENIDEN kurulur.
      - Son konum     : sifirlanmazsa MAX_JUMP denetimi, isinlanma sonrasi DOGRU
                        olcumu "imkansiz sicrama" sayip reddeder ve sistem kilitlenir.

KONTROLLER:
    q      : cikis
    BOSLUK : duraklat / devam
    r      : elle yeniden tespit
"""

import csv
import os
import time
from collections import deque

import cv2
import numpy as np

import background
import config
import court
import detector
import analysis
import framereader
import performance
import predictors
import shot
import zones

put_text = court._put_text


def resolve_output_path(template, tracker_type):
    """
    Cikti yolunu SECILEN TAKIPCIYE gore adlandirir ve var olan dosyanin uzerine
    YAZMAZ -> her calistirma AYRI bir cikti videosu birakir.

    "output/tracked.mp4" + "MOSSE"  ->  "output/tracked_mosse.mp4"
    O isim zaten varsa "output/tracked_mosse_2.mp4", "_3", ... ilk BOS ismi dondurur.

    Takipci adi kucuk harfe cevrilir (dosya adi tutarli olsun diye).
    """
    root, ext = os.path.splitext(template)
    base = "{}_{}".format(root, str(tracker_type).lower())
    candidate = base + ext
    n = 2
    while os.path.exists(candidate):
        candidate = "{}_{}{}".format(base, n, ext)
        n += 1
    return candidate


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
        p.number_of_scales = config.CSRT_FAST_SCALES
        p.admm_iterations = config.CSRT_FAST_ADMM
        p.template_size = config.CSRT_FAST_TEMPLATE
        p.use_segmentation = config.CSRT_FAST_SEGMENTATION
        return cv2.TrackerCSRT_create(p)
    except Exception:
        return None


def create_tracker():
    """
    config.TRACKER_TYPE'a gore takipci olusturur (OpenCV surumunden bagimsiz).

    NEDEN SECILEBILIR: CSRT en dogru ama EN YAVAS takipci; FPS darbogazi oydu
    (~88.8 ms/kare -> ~11 FPS tavani). Hibrit resync (TRACK_RESYNC_INTERVAL)
    kutuyu her ~18 karede tespitle senkronladigi icin takipcinin kendi OLCEK
    aramasi gereksizdir: takipci yalnizca iki resync arasinda konumu tutar.
    Olculen maliyetler ve secenekler icin bkz. config.py bolum 5.

    Istenen takipci bu OpenCV surumunde yoksa CSRT'ye geri dusulur (cokmez).
    """
    kind = str(config.TRACKER_TYPE).upper()
    tracker = _make_csrt_fast() if kind == "CSRT_FAST" else _make_tracker(kind)

    if tracker is None and kind != "CSRT":
        print("UYARI: '{}' takipcisi bulunamadi -> CSRT'ye donuluyor.".format(kind))
        tracker = _make_tracker("CSRT")

    if tracker is None:
        raise RuntimeError(
            "Takipci olusturulamadi ({}). 'pip install opencv-contrib-python' gerekli.".format(kind))
    return tracker


def ask_tracker():
    """
    Baslangicta hangi takipcinin kullanilacagini sorar ve config.TRACKER_TYPE'i ayarlar.

    ENTER      -> config'teki varsayilan kullanilir.
    Numara/isim -> o takipci secilir. Gecersiz girdi -> varsayilan.

    config.ASK_TRACKER = False ise hic sorulmaz (Pi'de otomatik baslatma icin).
    stdin yoksa (servis/autostart) EOFError yakalanir; program KILITLENMEZ,
    varsayilanla devam eder.
    """
    if not config.ASK_TRACKER:
        return

    options = (
        ("KCF", "~17 ms/kare   - dengeli"),
        ("MOSSE", "~0.6 ms/kare  - en hizli (Raspberry Pi icin)"),
        ("CSRT", "~94 ms/kare   - en dogru ama cok yavas"),
        ("CSRT_FAST", "~21 ms/kare   - CSRT, olcek/segmentasyon kisilmis"),
    )
    names = [n for n, _ in options]
    default = str(config.TRACKER_TYPE).upper()

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
            # config modulu calisma aninda guncellenir; create_tracker() bunu okur.
            config.TRACKER_TYPE = names[int(raw) - 1]
        elif raw.upper() in names:
            config.TRACKER_TYPE = raw.upper()
        else:
            print("Gecersiz secim -> varsayilan kullanilacak.")

    print("Takipci: {}\n".format(config.TRACKER_TYPE))


def create_mog():
    """
    MOG2 arka plan cikaricisi. Sahne kesmesinde YENIDEN cagrilir: eski model
    coktugu icin ogrenmesini beklemek yuzlerce kare cop maske uretmek demektir.
    """
    return cv2.createBackgroundSubtractorMOG2(
        history=config.MOG2_HISTORY,
        varThreshold=config.MOG2_VAR_THRESHOLD,
        detectShadows=config.MOG2_DETECT_SHADOWS,
    )


def ask_predict():
    """
    Baslangicta hangi KONUM TAHMIN modelinin kullanilacagini sorar ve
    config.PREDICT_TYPE'i ayarlar (ask_tracker ile ayni desen; modeller predictors.py).

    config.ASK_PREDICT = False ise sorulmaz. stdin yoksa (autostart) varsayilanla devam.
    """
    if not config.ASK_PREDICT:
        return

    options = (
        ("CV", "Sabit Hiz      - baseline, en ucuz"),
        ("CA", "Sabit Ivme     - hizlanma/yavaslama"),
        ("ABG", "alfa-beta-gama - hafif, sabit kazanc"),
        ("IMM", "Coklu Model    - en saglam (dur/sprint/donus)"),
    )
    names = [n for n, _ in options]
    default = str(config.PREDICT_TYPE).upper()

    print("\nHangi tahmin modeli kullanilsin?")
    for i, (name, note) in enumerate(options, 1):
        print("  {}) {:<5} {}{}".format(
            i, name, note, "   <- varsayilan" if name == default else ""))

    try:
        raw = input("Secim (numara veya isim, ENTER = {}): ".format(default)).strip()
    except (EOFError, KeyboardInterrupt):
        raw = ""   # stdin yok (autostart) -> varsayilanla devam.

    if raw:
        if raw.isdigit() and 1 <= int(raw) <= len(names):
            config.PREDICT_TYPE = names[int(raw) - 1]
        elif raw.upper() in names:
            config.PREDICT_TYPE = raw.upper()
        else:
            print("Gecersiz secim -> varsayilan kullanilacak.")

    print("Tahmin modeli: {}\n".format(config.PREDICT_TYPE))


def is_bbox_valid(bbox, ref_area, frame_w, frame_h):
    """Tracker bbox'i geometrik olarak makul mu? (ucuz on denetim)"""
    x, y, w, h = bbox
    if w < 2 or h < 2:
        return False
    if x < 0 or y < 0 or x + w > frame_w or y + h > frame_h:
        return False

    aspect = h / float(w)
    if aspect < config.BBOX_ASPECT_MIN or aspect > config.BBOX_ASPECT_MAX:
        return False

    if ref_area > 0:
        ratio = (w * h) / float(ref_area)
        if ratio < config.BBOX_AREA_MIN_RATIO or ratio > config.BBOX_AREA_MAX_RATIO:
            return False

    return True


def foot_point(bbox):
    """Homography yer duzlemini esler -> gercek konum icin bbox'in ALT-ORTA noktasi."""
    x, y, w, h = bbox
    if config.USE_FOOT_POINT:
        return (x + w * 0.5, y + h * config.FOOT_POINT_RATIO)
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
    # Takipci ve TAHMIN modeli secimi EN BASTA sorulur: agir kurulumdan (arka plan
    # modeli, kose secimi) once cevap alinsin, kullanici bosuna beklemesin.
    ask_tracker()
    ask_predict()

    cap = cv2.VideoCapture(config.VIDEO_PATH)
    if not cap.isOpened():
        raise IOError("Video acilamadi: {}".format(config.VIDEO_PATH))

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 1.0 or src_fps > 240.0:
        src_fps = config.FALLBACK_FPS

    ok, frame = cap.read()
    if not ok:
        raise IOError("Videodan ilk kare okunamadi.")

    scale = float(config.PROC_SCALE)
    if scale != 1.0:
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=config.RESIZE_INTERP)
    frame_h, frame_w = frame.shape[:2]

    # Performans olcumu (headless FPS/asama profilleme; algoritmayi etkilemez).
    # Asama adlari islenme sirasindadir. NOT: bu projede YOLO / top tespiti-takibi
    # / ayri hiz hesabi YOKTUR; "oyuncu_tespit" arka plan cikarmadir, "konum_kalman"
    # Kalman konum kestirimidir. Var olan asamalar olculur, olmayanlar uydurulmaz.
    perf = performance.Performance(
        ["video_okuma", "oyuncu_tespit", "takip", "konum_kalman", "gorsellestirme"],
        report_interval=config.PERF_INTERVAL,
        avg_window=config.PERF_AVG_WINDOW,
        enabled=config.PERF_ENABLED,
    )

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
    perf.start_setup("court_detection")
    corners = court.load_corners(frame.shape)
    if corners is not None:
        print("Kayitli koseler yuklendi: {}".format(court.CORNERS_PATH))
    else:
        if config.AUTO_DETECT_CORNERS:
            corners = court.auto_detect_corners(frame)
            if corners is not None and config.AUTO_CONFIRM and config.SHOW_WINDOW:
                if not court.confirm_corners(frame, corners):
                    corners = None      # Kullanici reddetti -> elle sec.
        if corners is None and config.SHOW_WINDOW:
            corners = court.select_corners(frame)
        if corners is None:
            cap.release()
            if not config.SHOW_WINDOW:
                print(
                    "Otomatik kort tespiti basarisiz VE SHOW_WINDOW=False (headless): "
                    "elle kose secimi icin ekran/fare gerekir, bu ortamda ikisi de yok.\n"
                    "Cozum secenekleri:\n"
                    "  1) Bir kez, ekrani olan bir makinede/oturumda SHOW_WINDOW=True "
                    "yapip elle secim yapin -> koseler court_corners.json'a kaydedilir, "
                    "sonraki calistirmalar headless (SHOW_WINDOW=False) devam edebilir.\n"
                    "  2) config.py'deki cizgi maskesi / AUTO_MIN_SCORE esiklerini "
                    "video goruntusune gore ayarlayip otomatik tespitin puanini "
                    "yukseltin (bkz. 'Otomatik kort tespiti basarisiz (puan ...)' mesaji)."
                )
            else:
                print("Kose secimi iptal edildi.")
            return
        court.save_corners(corners, frame.shape)
        print("Koseler kaydedildi: {}".format(court.CORNERS_PATH))
    perf.stop_setup("court_detection")

    # Kort kalibrasyonu: koseler -> homography + tespit ROI'si.
    perf.start_setup("court_calibration")

    # Kose ince ayari: 5 referans noktasini, projekte kort cizgilerini gercek
    # cizgi maskesine hizalayacak sekilde kucuk adimlarla kaydirir (line-mask ICP;
    # bkz. court.refine_corners). 5 noktadan cok daha fazla kanit kullanir ->
    # homografiyi sikilastirir. Iyilesme yoksa noktalar aynen korunur (asla bozmaz).
    # Kurulumda TEK SEFER; FPS'e etkisi yok. Kayitli json'a YAZILMAZ (idempotent).
    if config.REFINE_CORNERS and corners is not None:
        corners, _score_b, _score_a = court.refine_corners(frame, corners)

    H, H_inv = court.compute_homography(corners)

    # Kort tespitini GORSEL olarak dogrulamak icin ilk kareyi + cizilen kort
    # cizgilerini + numarali/renkli referans noktalarini output/'a kaydet.
    # SHOW_WINDOW=False (headless Pi) oldugunda ekranda hicbir gorsel geri
    # bildirim olmadigi icin bu, tespitin dogrulugunu kontrol etmenin tek yolu.
    if config.SAVE_FIRST_FRAME_DEBUG:
        court.save_first_frame_debug(frame, corners, H_inv, config.FIRST_FRAME_DEBUG_PATH)

    # Tespit ROI'si: aranan yarinin ZEMIN poligonu. detector.init bunu oyuncunun
    # govdesini kapsayacak sekilde goruntude yukari genisletir.
    search_mask, search_rect = court.build_search_mask(frame.shape, H_inv)
    perf.stop_setup("court_calibration")

    bg = background.build(config.VIDEO_PATH, scale)
    if bg is None:
        cap.release()
        raise RuntimeError("Arka plan modeli kurulamadi: video okunamadi.")

    det_state = detector.init(frame, search_mask, search_rect, bg, H, H_inv)

    # --- SABIT MAKINE TESPITI (bir kez; makine sabit -> medyan arka planda kalir) ---
    # Makine hareket etmedigi icin her karede aranmaz: kurulumda BIR KEZ bulunur,
    # konumu/bolgesi sabit tutulur. (Tasinirsa yeniden baslatinca guncellenir.)
    machine_world = None
    machine_bbox = None
    machine_zone = None
    if config.DETECT_MACHINE:
        m_mask, _ = court.build_search_mask(frame.shape, H_inv, half="both",
                                            margin=config.SEARCH_MARGIN)
        rise = detector._rise_px(H_inv)
        m_body = cv2.dilate(m_mask, np.ones((rise + 1, 1), np.uint8), anchor=(0, 0))
        found = detector.detect_machine(bg, m_body, H)
        if found is not None:
            machine_world, machine_bbox = found
            machine_zone = zones.get_zone(machine_world[0], machine_world[1])
            print("Makine bulundu: bolge {} @ ({:.2f}, {:.2f}) m".format(
                machine_zone, machine_world[0], machine_world[1]))
        else:
            print("Makine BULUNAMADI (kort ici aluminyum/beyaz kompakt nesne yok).")

    # Cizim yalnizca insan icin (output videosu / ekran). config.DRAW_VISUALS = False
    # ise hicbir sey cizilmez; get_zone ve bolge CSV/JSON aynen calisir (headless
    # data-only Pi modu). zone_state SADECE cizim icin gerekli -> cizim yoksa kurulmaz.
    draw = config.DRAW_VISUALS
    zone_state = zones.build(H_inv, frame.shape) if draw else None

    # MOG2 yalnizca HUD'daki hareket gostergesini besler (oyuncu bulmak icin DEGIL).
    # config.USE_MOG2 = False -> hic kurulmaz, ~3.6 ms/kare kazanilir ve hareket
    # etiketi cizilmez. Sahne kesme tespiti bundan bagimsiz calismaya devam eder.
    mog = create_mog() if config.USE_MOG2 else None
    mog_scale = float(config.MOG2_SCALE)
    mog_warmup = config.MOG2_WARMUP if config.USE_MOG2 else 0

    shot_state = shot.init()

    # Bolge gecis analizi: efektif FPS ile (src_fps / FRAME_SKIP) -> olaylar
    # gercek video saniyesiyle damgalanir.
    eff_fps = src_fps / float(config.FRAME_SKIP)
    zone_track = analysis.init(eff_fps, config.ZONE_PRINT, config.ZONE_PRINT_INTERVAL) \
        if config.ZONE_ANALYSIS else None

    predictor = predictors.create(config.PREDICT_TYPE) if config.USE_KALMAN else None
    kf_ready = False

    writer = None
    if config.SAVE_OUTPUT:
        # Hedef klasoru olustur: yoksa VideoWriter SESSIZCE basarisiz olur
        # (nesne doner ama hicbir sey yazilmaz, hata da vermez).
        out_dir = os.path.dirname(config.OUTPUT_PATH)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # Cikti adi: secilen takipciye gore + uzerine yazmadan (bkz.
        # resolve_output_path). Klasor yukarida olusturuldugu icin var olan
        # dosyalar dogru sayilir. Kapaliysa OUTPUT_PATH aynen kullanilir.
        output_path = (resolve_output_path(config.OUTPUT_PATH, config.TRACKER_TYPE)
                       if config.OUTPUT_NAME_BY_TRACKER else config.OUTPUT_PATH)

        # Cikti FPS'i: None -> kaynaktan turet. FRAME_SKIP'e bolunur cunku her
        # FRAME_SKIP karede yalnizca BIR kare yazilir; bolunmezse cikti hizlanir.
        out_fps = config.OUTPUT_FPS if config.OUTPUT_FPS else (src_fps / float(config.FRAME_SKIP))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, out_fps, (frame_w, frame_h))

        # Sessiz basarisizligi yakala: codec/yol sorunlarinda isOpened() False doner.
        if not writer.isOpened():
            cap.release()
            raise IOError("Cikti videosu acilamadi: {} (codec/yol sorunu?)".format(output_path))
        print("Cikti kaydi: {}  ({}x{} @ {:.2f} FPS)".format(
            output_path, frame_w, frame_h, out_fps))

    # HUD'u panelin sagina hizala.
    hud_x = 10
    if config.SHOW_PANEL:
        hud_x = config.PANEL_POS[0] + zones.panel_size()[0] + 12

    # --------------------------------------------------------------------------
    # KAYDIRMA CUBUGU (YouTube tarzi seek) - OpenCV HighGUI trackbar; ek bagimlilik YOK.
    # Yalnizca SEEK edilebilir DOSYA kaynaginda kurulur (webcam veya kare sayisi
    # bilinmiyorsa otomatik atlanir). Cubugu suruklemek videonun herhangi bir anina atlar.
    # --------------------------------------------------------------------------
    window_name = "Tenis Oyuncu Takibi"
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Threaded okuma cap'i sahiplenir; seek (cap.set) ile cakisir -> birlikte kapali.
    seekable = (config.SHOW_WINDOW and config.SHOW_SEEKBAR and not config.THREADED_READ
                and not isinstance(config.VIDEO_PATH, int) and total_frames > 1)
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
    last_auto_redetect = 0        # Otomatik 'r' (AUTO_REDETECT) icin son tetik karesi.
    position = None
    trail = deque(maxlen=max(1, config.TRAIL_LENGTH))
    meas_hist = deque(maxlen=max(1, config.POSITION_MEDIAN_WINDOW))  # konum medyan filtresi
    stable_zone = None            # bolge histerezisi: raporlanan (stabil) bolge
    pending_zone = None           # gecise aday bolge
    pending_count = 0             # adayin ardisik gozlem sayisi
    smooth_motion = 0.0
    fps = 0.0
    frame_idx = 0
    cut_count = 0
    cut_banner = 0
    paused = False
    status = "LOST"

    max_jump = config.MAX_JUMP_PER_FRAME * config.FRAME_SKIP

    # DEGERLENDIRME LOGU: acik ise tracker'in kare-kare kutu+ayak konumunu CSV'ye
    # yaz (normalize 0..1). YOLO referansiyla compare.py'de kiyaslanir (bkz. config).
    eval_file = None
    eval_writer = None
    if config.EVAL_LOG:
        ed = os.path.dirname(config.EVAL_LOG_PATH)
        if ed:
            os.makedirs(ed, exist_ok=True)
        eval_file = open(config.EVAL_LOG_PATH, "w", newline="")
        eval_writer = csv.writer(eval_file)
        eval_writer.writerow(["frame", "x1", "y1", "x2", "y2", "fx", "fy"])
        print("Eval logu acik: {}".format(config.EVAL_LOG_PATH))

    # Threaded okuyucu: decode + resize'i arka planda yapar (bkz. framereader.py).
    # cap simdiden itibaren SADECE bu thread tarafindan okunur. Ilk kare (frame 0)
    # kurulumda kullanildi; okuyucu frame 1'den devam eder.
    reader = None
    if config.THREADED_READ:
        reader = framereader.ThreadedReader(
            cap, scale, config.RESIZE_INTERP, config.FRAME_SKIP, config.THREADED_QUEUE).start()

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
            shot_state = shot.init()      # Kesme histogramini sifirla -> sahte "kesme" sayilmasin.
            if mog is not None and config.CUT_RESET_MOG2:
                mog = create_mog()        # MOG2 modeli atlamada coker; yeniden kur.
                mog_warmup = config.MOG2_WARMUP
            force_process = True          # Duraklatilmis olsak bile bu kareyi bir kez isle/goster.

        # Duraklatilmisken yalnizca bir seek istegi geldiyse (force_process) tek kare islenir.
        if paused and not force_process:
            key = cv2.waitKey(30) & 0xFF
            if key == config.KEY_PAUSE:
                paused = False
            elif key == config.KEY_QUIT:
                break
            continue

        t0 = time.perf_counter()
        perf.frame_begin()          # Kare kronometresi (asama sayaclari sifirlanir).

        perf.start("video_okuma")
        if force_process:
            # Seek sonrasi hedef kareyi buradan oku (ilk-kare istisnasindan bagimsiz).
            ok, raw = cap.read()
            if not ok:
                paused = True     # Video sonuna/otesine atlandi: son duruma don, duraklat.
                continue
            for _ in range(config.FRAME_SKIP - 1):
                cap.grab()
            frame = cv2.resize(raw, None, fx=scale, fy=scale,
                               interpolation=config.RESIZE_INTERP) if scale != 1.0 else raw
        elif frame_idx > 0:  # Ilk kare zaten okundu.
            ok, raw = cap.read()
            if not ok:
                break
            for _ in range(config.FRAME_SKIP - 1):
                cap.grab()  # Atlanan kareler decode edilmez.
            frame = cv2.resize(raw, None, fx=scale, fy=scale,
                               interpolation=config.RESIZE_INTERP) if scale != 1.0 else raw
        perf.stop("video_okuma")

        frame_idx += 1

        # Kucultulmus kare HEM sahne kesme HEM MOG2 icin kullanilir; ikisi de
        # kapaliysa hic uretilmez (bos yere resize etme).
        mog_input = None
        if config.CUT_DETECTION or mog is not None:
            mog_input = cv2.resize(frame, None, fx=mog_scale, fy=mog_scale,
                                   interpolation=config.RESIZE_INTERP)

        # ----------------------------------------------------------------------
        # 1) SAHNE KESME KONTROLU (MOG2 apply'dan ONCE)
        # ----------------------------------------------------------------------
        cut = shot.detect_cut(shot_state, mog_input, frame_idx) if config.CUT_DETECTION else False

        if cut:
            cut_count += 1
            cut_banner = config.CUT_BANNER_FRAMES

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

            if mog is not None and config.CUT_RESET_MOG2:
                mog = create_mog()
                mog_warmup = config.MOG2_WARMUP

            # Tespit modeli (medyan arka plan) kesmede TAZELENMEZ: kamera sabit
            # oldugu icin arka plan kesme boyunca gecerli kalir. Eski renk modeli
            # isik/renk kaymasina duyarli oldugu icin tazelenmesi gerekiyordu.

        # MOG2 her karede guncellenir (kayipken bile), yoksa model donar.
        # Kapaliysa (config.USE_MOG2 = False) maske uretilmez; motion_ready hep False
        # kalir, boylece asagidaki hareket blogu ve etiketi kendiliginden devre disi.
        fgmask = mog.apply(mog_input, learningRate=config.MOG2_LEARNING_RATE) \
            if mog is not None else None

        motion_ready = mog is not None and mog_warmup <= 0
        if mog is not None and mog_warmup > 0:
            mog_warmup -= 1

        # ----------------------------------------------------------------------
        # 1b) OTOMATIK YENIDEN TESPIT ("ekransiz r tusu")
        # SHOW_WINDOW=False oldugundan 'r' tusuna basilamaz. Tracker BASARIYLA
        # takip ediyor olsa bile her AUTO_REDETECT_INTERVAL karede bir, 'r'nin
        # yaptigi SERT sifirlamayi otomatik uygular (asagidaki KEY_REDETECT ile
        # birebir ayni alanlar). Konum sifirlandigi icin bu kareki tespit,
        # resync'in mesafe kisitina takilmadan dogru kutuyu kabul eder -> yanlis
        # bir yere (or. diz hizasi) kalici kilitlenmeyi kirar.
        # tracker None iken tetiklenmez: kayip durumu zaten mevcut redetect agi
        # tarafindan toparlanir; burada yalnizca "takip ediyor ama suphe var"
        # durumu icin periyodik guvenlik yeniden-tespiti yapilir.
        # ----------------------------------------------------------------------
        if (config.AUTO_REDETECT and config.AUTO_REDETECT_INTERVAL > 0
                and tracker is not None and not cut
                and frame_idx - last_auto_redetect >= config.AUTO_REDETECT_INTERVAL):
            last_auto_redetect = frame_idx
            tracker = None
            kf_ready = False
            position = None
            fail_count = config.TRACKER_FAIL_TOLERANCE + 1
            last_detect_frame = -10 ** 9
            status = "LOST"
            trail.clear()

        # ----------------------------------------------------------------------
        # 2) CSRT TAKIBI
        # ----------------------------------------------------------------------
        tracked = False
        if tracker is not None:
            perf.start("takip")
            ok_t, raw_box = tracker.update(frame)
            perf.stop("takip")
            if ok_t:
                candidate = tuple(int(round(v)) for v in raw_box)
                # Sisen kutuyu o derinlikte beklenen boya kirp: ayak noktasi
                # ayaklara oturur VE "cok uzun kutu" boyut denetimi takibi
                # gereksiz kesmez (dogrulamadan ONCE kirpiliyor).
                if config.CAP_BOX_HEIGHT:
                    candidate = detector.cap_box_height(det_state, candidate)
                if is_bbox_valid(candidate, ref_area, frame_w, frame_h):
                    bbox = candidate
                    tracked = True

            # Periyodik saglik denetimi (Cozum 2: otomatik kurtarma).
            if tracked and config.TRACK_VERIFY_INTERVAL > 0 \
                    and frame_idx % config.TRACK_VERIFY_INTERVAL == 0:
                # (a) Tracker bos zemine mi kilitlendi? (kutu ici foreground orani)
                if detector.verify(det_state, frame, bbox) < config.TRACK_VERIFY_MIN_FG_RATIO:
                    tracked = False
                # (b) Kutu boyutu o derinlikte akla yatkin mi? CSRT olcegi zayif
                #     adapte eder; oyuncu uzaklasinca kutu KUCULMEZ, konum kameraya
                #     yakin cikar. Beklenen boy homografiden gelir (detector.expected_height).
                elif config.TRACK_SIZE_CHECK:
                    exp_h = detector.expected_height(det_state, *foot_point(bbox))
                    if exp_h is not None and exp_h > 1.0:
                        ratio = bbox[3] / exp_h
                        if ratio < config.TRACK_SIZE_MIN_RATIO \
                                or ratio > config.TRACK_SIZE_MAX_RATIO:
                            tracked = False
                            tracker = None   # Takibi sonlandir -> ayni kare yeniden tespiti tetikler.

        # ----------------------------------------------------------------------
        # 2b) PERIYODIK YENIDEN SENKRONIZASYON (Cozum 1 - hibrit takip)
        # CSRT olcegi zayif adapte eder; oyuncu uzaklasinca kutu kuculmez ve konum
        # kameraya yakin cikar. Bu yuzden tracker BASARILI olsa bile her
        # TRACK_RESYNC_INTERVAL karede bir, mevcut HAFIF tespiti (detector.detect ->
        # arka plan cikarma, YOLO/agir model DEGIL) calistirip CSRT'yi dogru
        # boyutlu kutuyla yeniden baslatiriz. Boylece birikmis kayma silinir.
        # Kalman/iz sifirlanmaz -> takip surekliligi korunur.
        # ----------------------------------------------------------------------
        if tracked and tracker is not None and config.TRACK_RESYNC_INTERVAL > 0 \
                and frame_idx % config.TRACK_RESYNC_INTERVAL == 0:
            perf.start("oyuncu_tespit")
            fresh = detector.detect(det_state, frame)
            perf.stop("oyuncu_tespit")
            if fresh is not None:
                fw = court.image_to_court(H, *foot_point(fresh))
                # Yeni tespiti YALNIZCA kort-ici VE mevcut konuma makul mesafedeyse
                # kabul et; aksi halde spurious bir blob takibi kacirabilir.
                if fw is not None and court.is_on_court(fw[0], fw[1]) and (
                        position is None
                        or ((fw[0] - position[0]) ** 2 + (fw[1] - position[1]) ** 2)
                        <= config.TRACK_RESYNC_MAX_JUMP ** 2):
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
            world = court.image_to_court(H, fx, fy)

            # OYUNCU-YARISI KISITI (domain prior): oyuncu fiziksel olarak yalnizca
            # PLAYER_HALF yarisinda olabilir. Sikisik uzak-yari perspektifi ayagi
            # ara sira file otesine (karsi yariya) tasiyor -> IMKANSIZ; konum file
            # cizgisine (NET_Y) kirpilir. Uzak yaride y<0 (dip cizgi arkasi) gecerli
            # kalir; yalnizca karsi yariya tasma engellenir.
            if world is not None and config.CONSTRAIN_PLAYER_HALF:
                # File cizgisinin hemen ICINE kirp (sinir yuvarlamasi karsi banda
                # dusmesin): uzak yari icin file'in ~2cm altina, yakin yari icin ustune.
                _m = 0.02
                if config.PLAYER_HALF == "far" and world[1] > config.NET_Y - _m:
                    world = (world[0], config.NET_Y - _m)
                elif config.PLAYER_HALF == "near" and world[1] < config.NET_Y + _m:
                    world = (world[0], config.NET_Y + _m)

            if world is None or (config.VALIDATE_POSITION_ON_COURT
                                 and not court.is_on_court(world[0], world[1])):
                tracked = False
            else:
                # KONUM MEDYAN FILTRESI: kosuda blob gurultusunun ani sicramalarini
                # bastirir. Medyan aykiri degerden ETKILENMEZ ve konumu DONDURMAZ
                # (gercek hareketi ~1-2 kare gecikmeyle izler). Ham olcum gecmise
                # eklenir, olcum olarak pencerenin medyani kullanilir.
                meas_hist.append(world)
                if len(meas_hist) >= 2:
                    world = (float(np.median([p[0] for p in meas_hist])),
                             float(np.median([p[1] for p in meas_hist])))
                if position is not None:
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
            if fail_count > config.TRACKER_FAIL_TOLERANCE:
                tracker = None
                status = "LOST"
            else:
                status = "PREDICT"

        if tracker is None and (frame_idx - last_detect_frame) >= config.REDETECT_INTERVAL:
            last_detect_frame = frame_idx
            perf.start("oyuncu_tespit")
            found = detector.detect(det_state, frame)
            perf.stop("oyuncu_tespit")

            if found is not None:
                world = court.image_to_court(H, *foot_point(found))
                if world is not None and court.is_on_court(world[0], world[1]):
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
                    meas_hist.clear()   # yeni track: eski (teleport oncesi) olcumleri medyana karistirma

        # ----------------------------------------------------------------------
        # 5) KALMAN
        # ----------------------------------------------------------------------
        # ("konum_kalman" asamasi: bu projede AYRI bir hiz hesabi yok; en yakin
        #  karsilik Kalman konum kestirimidir.)
        perf.start("konum_kalman")
        if predictor is not None:
            if not kf_ready:
                if measurement is not None:
                    predictor.reset(measurement)   # ilk olcumle baslat
                    kf_ready = True
                    position = measurement
            else:
                predictor.predict()                # bir sonraki kareyi kestir (a-priori)
                if measurement is not None:
                    position = predictor.update(measurement)   # olcumle duzelt
                else:
                    position = predictor.predicted()           # olcum yok -> tahmini kullan
        elif measurement is not None:
            position = measurement
        perf.stop("konum_kalman")

        display_pos = position if status in ("TRACKING", "DETECTED", "PREDICT") else None
        if display_pos is not None:
            trail.append(display_pos)

        raw_zone = zones.get_zone(display_pos[0], display_pos[1]) if display_pos else None

        # BOLGE HISTEREZISI: oyuncu sinirdayken konum jitter'i bolge etiketini
        # (or. 5<->6<->5x) cakmasina yol aciyordu. Yeni bolge, ancak ardisik
        # ZONE_HYSTERESIS_FRAMES kare gozlenirse GECERLI olur; aksi halde son
        # stabil bolge korunur. Takibi/konumu degistirmez, sadece etiketi stabilize eder.
        if raw_zone is None:
            stable_zone = None
            pending_zone = None
            pending_count = 0
        elif raw_zone == stable_zone:
            pending_zone = None
            pending_count = 0
        else:
            if raw_zone == pending_zone:
                pending_count += 1
            else:
                pending_zone = raw_zone
                pending_count = 1
            if stable_zone is None or pending_count >= config.ZONE_HYSTERESIS_FRAMES:
                stable_zone = raw_zone
                pending_zone = None
                pending_count = 0
        zone = stable_zone

        # Bolge gecis analizi: bolge degistiyse olay kaydeder (kare basina degil).
        if zone_track is not None:
            analysis.update(zone_track, zone, frame_idx, display_pos)

        # ----------------------------------------------------------------------
        # 6) MOG2: KUTU ICI HAREKET ENERJISI (oyuncu bulmak icin DEGIL)
        # ----------------------------------------------------------------------
        if tracked and bbox is not None and motion_ready:
            energy = motion_energy(fgmask, bbox, mog_scale)
            a = config.MOTION_SMOOTHING
            smooth_motion = a * energy + (1.0 - a) * smooth_motion
        elif motion_ready:
            smooth_motion *= (1.0 - config.MOTION_SMOOTHING)

        moving = motion_ready and smooth_motion > config.MOTION_ENERGY_THRESH

        # ----------------------------------------------------------------------
        # 7) CIZIM  (asama: gorsellestirme -> output videosuna cizilen her sey)
        # ----------------------------------------------------------------------
        perf.start("gorsellestirme")
        if draw:
            # Statik izgara katmani (maskeli kopya) + aktif hucre vurgusu.
            zones.draw_overlay(frame, zone_state, zone)

            if bbox is not None and display_pos is not None:
                x, y, w, h = bbox
                color = config.COLOR_BBOX if tracked else config.COLOR_BBOX_LOST
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, config.BBOX_THICKNESS)
                fx, fy = foot_point(bbox)
                cv2.circle(frame, (int(fx), int(fy)), 4, config.COLOR_FOOT, -1, cv2.LINE_AA)

            # SABIT MAKINE: bulunduysa kutu+ayak isareti; SAG UST'te bolge/BULUNAMADI.
            if config.DETECT_MACHINE:
                if machine_bbox is not None:
                    mx, my, mw, mh = machine_bbox
                    cv2.rectangle(frame, (mx, my), (mx + mw, my + mh), config.COLOR_MACHINE, 2)
                    cv2.circle(frame, (int(mx + mw * 0.5), int(my + mh)), 4,
                               config.COLOR_MACHINE, -1, cv2.LINE_AA)
                    put_text(frame, "MAKINE", (mx, my - 8), 0.5, 2)
                mtxt = "MAKINE: BULUNAMADI" if machine_zone is None \
                    else "MAKINE: {}".format(machine_zone)
                mcolor = config.COLOR_BBOX_LOST if machine_zone is None else config.COLOR_MACHINE
                (tw, _), _ = cv2.getTextSize(mtxt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                mxp = frame_w - tw - 14
                cv2.putText(frame, mtxt, (mxp, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            config.COLOR_TEXT_BG, 3, cv2.LINE_AA)
                cv2.putText(frame, mtxt, (mxp, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            mcolor, 1, cv2.LINE_AA)

            if config.SHOW_PANEL:
                zones.draw_panel(frame, zone_state, zone, display_pos, list(trail))

            # Bolge gecis paneli (sag-alt): son gecisleri gosteren textbox.
            if zone_track is not None and config.SHOW_ZONE_PANEL:
                analysis.draw_panel(zone_track, frame, config.ZONE_PANEL_MAX_LINES)

            hud_y = 24
            if config.SHOW_FPS:
                # ISLEME FPS = kodun kareyi ne kadar hizli isledigi (1/islem_suresi);
                #              MOSSE gibi hizli takipcide 30'un USTUNE cikabilir.
                # VIDEO FPS   = kaynak videonun kare hizi (sabit, cikti da bu hizda yazilir).
                # Ikisi FARKLI seylerdir; alt alta gosterilir ki karismasin.
                put_text(frame, "ISLEME FPS: {:.1f}".format(fps), (hud_x, hud_y), 0.6, 1)
                hud_y += 24
                put_text(frame, "VIDEO FPS: {:.1f}".format(src_fps), (hud_x, hud_y), 0.6, 1)
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
                    label, motion_color = "HAREKET: ISINIYOR", config.COLOR_MOTION_OFF
                elif moving:
                    label = "HAREKETLI ({:.2f})".format(smooth_motion)
                    motion_color = config.COLOR_MOTION_ON
                else:
                    label = "DURAGAN ({:.2f})".format(smooth_motion)
                    motion_color = config.COLOR_MOTION_OFF

                cv2.putText(frame, label, (hud_x, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            config.COLOR_TEXT_BG, 3, cv2.LINE_AA)
                cv2.putText(frame, label, (hud_x, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            motion_color, 1, cv2.LINE_AA)
                hud_y += 24

            put_text(frame, "KESME: {}".format(cut_count), (hud_x, hud_y), 0.6, 1)

            if cut_banner > 0:
                cut_banner -= 1
                cv2.rectangle(frame, (0, 0), (frame_w - 1, frame_h - 1), config.COLOR_CUT, 3)
                put_text(frame, "SAHNE KESME - YENIDEN TESPIT",
                         (frame_w // 2 - 170, frame_h - 20), 0.7, 2)
        perf.stop("gorsellestirme")

        # ----------------------------------------------------------------------
        # 8) FPS OLCUMU (islem suresi; goruntuleme haric)
        # ----------------------------------------------------------------------
        dt = time.perf_counter() - t0
        if dt > 0:
            inst = 1.0 / dt
            fps = inst if fps == 0.0 else \
                config.FPS_SMOOTHING * fps + (1.0 - config.FPS_SMOOTHING) * inst

        # EVAL: tracker'in bu karedeki kutu+ayak konumu (normalize; YOLO ile kiyas icin).
        if eval_writer is not None:
            ef = frame_idx - 1   # 0-tabanli kaynak kare (FRAME_SKIP=1 iken YOLO ile eslesir)
            if bbox is not None and display_pos is not None:
                bx, by, bw, bh = bbox
                efx, efy = foot_point(bbox)
                eval_writer.writerow([ef,
                                      round(bx / frame_w, 5), round(by / frame_h, 5),
                                      round((bx + bw) / frame_w, 5), round((by + bh) / frame_h, 5),
                                      round(efx / frame_w, 5), round(efy / frame_h, 5)])
            else:
                eval_writer.writerow([ef, "", "", "", "", "", ""])

        if writer is not None:
            writer.write(frame)

        # Kareyi kapat: FPS guncelle, saniyede bir canli PERF raporu bas.
        perf.frame_end()

        if config.DEBUG_SHOW_MASK and fgmask is not None:
            cv2.imshow("MOG2 (sadece hareket analizi)", fgmask)
        if config.DEBUG_SHOW_DETECTION and det_state["debug_mask"] is not None:
            cv2.imshow("Tespit maskesi", det_state["debug_mask"])

        if config.SHOW_WINDOW:
            cv2.imshow(window_name, frame)
            if seekable:
                # Cubugu oynatma konumuna esitle. seek_syncing bayragi, bu programatik
                # guncellemenin callback'i (on_seek) tetiklemesini engeller.
                seek_syncing[0] = True
                cv2.setTrackbarPos("Kare", window_name,
                                   min(total_frames - 1, int(cap.get(cv2.CAP_PROP_POS_FRAMES))))
                seek_syncing[0] = False
            key = cv2.waitKey(1) & 0xFF
            if key == config.KEY_QUIT:
                break
            elif key == config.KEY_PAUSE:
                paused = True
            elif key == config.KEY_REDETECT:
                tracker = None
                kf_ready = False
                position = None
                fail_count = config.TRACKER_FAIL_TOLERANCE + 1
                last_detect_frame = -10 ** 9
                status = "LOST"
                trail.clear()

    cap.release()
    if writer is not None:
        writer.release()
    if eval_file is not None:
        eval_file.close()
    cv2.destroyAllWindows()

    # Bolge gecis olaylarini + ozeti diske yaz (q ile cikista da calisir).
    if zone_track is not None:
        n = analysis.save(zone_track, frame_idx, config.ZONE_EVENTS_CSV, config.ZONE_SUMMARY_JSON)
        print("Bolge gecis analizi: {} olay -> {} , {}".format(
            n, config.ZONE_EVENTS_CSV, config.ZONE_SUMMARY_JSON))

    print("Islenen kare: {} | Sahne kesme: {} | Ortalama FPS: {:.1f} | Kaynak FPS: {:.1f}".format(
        frame_idx, cut_count, fps, src_fps))

    # Performans ozeti (asama sureleri + FPS) -> Pi <-> Windows karsilastirmasi.
    perf.summary()


if __name__ == "__main__":
    main()
