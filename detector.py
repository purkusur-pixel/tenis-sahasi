"""
detector.py
-----------
Medyan arka plan cikarma ile oyuncu tespiti. YAPAY ZEKA YOK.

Tespit yalnizca iki durumda calisir:
  1) Videonun basinda, tracker'i baslatmak icin.
  2) CSRT oyuncuyu kaybettiginde, yeniden kilitlenmek icin.

NEDEN RENK MODELI BIRAKILDI:
    Onceki surum kort zemininin LAB medyanina uzakliga bakiyordu. Bu kamerada hic
    calismadi (20 karede 0 tespit): kortun disindaki KIRMIZI run-off alani da kort
    yesilinden uzak oldugu icin maskenin ~%31'ini foreground yapiyor, oyuncu bu dev
    blob'un icinde kayboluyordu. Arka plan cikarma her pikseli KENDI gecmisiyle
    karsilastirir; run-off, tel orgu, bina - hepsi dogal olarak elenir.
    Ayni kare kumesinde tespit 0/20 -> 31/40'a cikti ve kutularin TAMAMI dogru
    boyutta oldu (bkz. background.py).

BORU HATTI:
    kare - arka plan  ->  mutlak fark  ->  esik  ->  GOVDE maskesi  ->  morfoloji
    ->  connectedComponentsWithStats  ->  alan/en-boy filtresi
    ->  AYAK NOKTASI kort dogrulamasi  ->  en buyuk aday

GOVDE MASKESI (kritik):
    court.build_search_mask() ZEMIN duzlemi poligonunu verir. Uzak yari bu kamerada
    yalnizca ~40 piksel yuksekliginde bir seritdir; oyuncu ise DIK durur ve govdesi
    o seridin cok ustune tasar. Zemin maskesiyle AND'lenince geriye yalnizca ayak
    bilegi kalir, o da en-boy/alan filtrelerine takilir. Bu yuzden zemin poligonu
    goruntu duzleminde YUKARI dogru genisletilir (bkz. _rise_px).

AYAK NOKTASI DOGRULAMASI:
    Maskeyi yukari genisletmek arkadaki agac/tel hareketini de iceri alir. Her
    adayin ALT-ORTA noktasi homography ile korta cevrilir; istenen yarinin disinda
    kalan aday atilir. Ruzgarda sallanan agacin "ayagi" kortun metrelerce arkasina
    duser -> elenir. Bu filtre olmadan tespitlerin yarisi agacti.

DURUM (state):
    Sinif yerine hafif bir dict. init() ile olusturulur, detect()/verify()'a verilir.
"""

import cv2
import numpy as np

import config
import court

_OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, config.MORPH_OPEN_KERNEL)
_CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, config.MORPH_CLOSE_KERNEL)


def _rise_px(H_inv):
    """
    Zemin maskesinin goruntude YUKARI genisletilecegi piksel miktari.

    Sihirli sayi degil, geometriden turetilir: bir derinlikteki YATAY olcek (px/m)
    ayni derinlikteki DIK bir nesnenin piksel boyunu da verir (ikisi de bakis
    yonune diktir). File hizasi, aranan uzak yarinin kameraya en yakin - yani en
    buyuk olcekli - kenaridir; en kotu durumu verdigi icin oradan olculur.
    """
    a = court.court_to_image(H_inv, 0.0, config.NET_Y)
    b = court.court_to_image(H_inv, config.COURT_WIDTH, config.NET_Y)
    if a is None or b is None:
        return config.BG_RISE_FALLBACK
    px_per_m = abs(b[0] - a[0]) / config.COURT_WIDTH
    return int(round(config.PLAYER_HEIGHT_M * px_per_m))


def init(frame, search_mask, search_rect, background, H, H_inv):
    """
    Tespit durumunu hazirlar. Kamera sabit varsayildigi icin BIR KEZ cagrilir.

    frame       : islenmis kare (PROC_SCALE uygulanmis, BGR)
    search_mask : aranan yarinin ZEMIN maskesi (margin=0 ile kurulmali)
    search_rect : kullanilmaz; rect govde maskesinden yeniden hesaplanir
    background  : background.build() ile uretilen medyan arka plan
    H, H_inv    : homography (ayak dogrulamasi ve olcek icin)
    """
    rise = _rise_px(H_inv)
    body = cv2.dilate(search_mask, np.ones((rise + 1, 1), np.uint8), anchor=(0, 0))
    rect = cv2.boundingRect(body)
    if rect[2] == 0 or rect[3] == 0:
        raise ValueError("Govde maskesi bos. Kose secimini kontrol edin.")

    fh, fw = frame.shape[:2]
    det_area = (fw * config.DET_SCALE) * (fh * config.DET_SCALE)

    return {
        "rect": rect,
        "body": body,
        "bg": cv2.GaussianBlur(background, config.BG_BLUR, 0),
        "H": H,
        "H_inv": H_inv,   # Cozum 2: expected_height() derinlige gore beklenen boyu buradan hesaplar.
        "rise": rise,
        "frame_w": fw,
        "frame_h": fh,
        "min_area": config.MIN_BLOB_AREA_RATIO * det_area,
        "max_area": config.MAX_BLOB_AREA_RATIO * det_area,
        "debug_mask": None,
    }


def _foreground(state, frame):
    """Arka plandan sapan pikseller, GOVDE maskesiyle sinirli."""
    diff = cv2.absdiff(cv2.GaussianBlur(frame, config.BG_BLUR, 0), state["bg"])
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, fg = cv2.threshold(gray, config.BG_DIFF_THRESH, 255, cv2.THRESH_BINARY)
    cv2.bitwise_and(fg, state["body"], dst=fg)

    it = config.MORPH_ITERATIONS
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, _OPEN_KERNEL, iterations=it)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, _CLOSE_KERNEL, iterations=it)
    return fg


def _in_search_half(wx, wy):
    """Ayak noktasi config.PLAYER_HALF yarisinda (+ pay) mi?"""
    m = config.FOOT_VALIDATE_MARGIN
    if not (-m <= wx <= config.COURT_WIDTH + m):
        return False
    if config.PLAYER_HALF == "far":
        return -m <= wy <= config.NET_Y + m
    if config.PLAYER_HALF == "near":
        return config.NET_Y - m <= wy <= config.COURT_LENGTH + m
    return -m <= wy <= config.COURT_LENGTH + m


def detect(state, frame):
    """
    Oyuncuyu tespit eder.

    Donus: (x, y, w, h) islenmis kare koordinatlarinda bbox, veya None.
    """
    fg = _foreground(state, frame)
    if config.DEBUG_SHOW_DETECTION:
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
        if aspect < config.MIN_ASPECT_RATIO or aspect > config.MAX_ASPECT_RATIO:
            continue

        bx = stats[i, cv2.CC_STAT_LEFT]
        by = stats[i, cv2.CC_STAT_TOP]

        # Ayak noktasi (alt-orta) istenen yarida mi?
        world = court.image_to_court(state["H"], bx + bw * 0.5, by + bh)
        if world is None or not _in_search_half(world[0], world[1]):
            continue

        if area > best_area:
            best_area = area
            best = (bx, by, bw, bh)

    if best is None:
        return None

    pad = config.BBOX_PADDING
    x, y, bw, bh = best
    # ASIMETRIK PADDING: padding UST ve YAN kenarlara uygulanir (CSRT'nin baglam
    # yakalamasi icin faydali), ancak ALT kenara EKLENMEZ. Alt kenar = blob'un
    # GERCEK yer-temas noktasi (ayak). Alta pad eklemek ayak noktasini asagi
    # kaydiriyordu; uzak yaride perspektif bunu buyuk ve TUTARLI bir "kameraya
    # yakin" sistematik hataya ceviriyordu (olculdu: 4 px alt padding ~ 1.5 m).
    # Yukseklik: ust padding korunur (+pad), alt padding kaldirilir -> bh + pad.
    box = _clamp_bbox(x - pad, y - pad, bw + 2 * pad, bh + pad,
                      state["frame_w"], state["frame_h"])
    if box is not None and config.CAP_BOX_HEIGHT:
        box = cap_box_height(state, box)   # sisen blob'u ayak seviyesine kirp
    return box


def cap_box_height(state, bbox):
    """
    Kutuyu, o DERINLIKTE beklenen oyuncu boyuna gore USTTEN kirpar.

    NEDEN: Uzak yarida oyuncu goruntude cok kucuktur; arka plan cikarma blob'u
    oyuncu + altindaki/ustundeki gurultuyu (golge, tel/fon farki) tek UZUN kutu
    yapabilir. Kutunun ALTI ayaklarin altina tastiginda:
      1) ayak noktasi (kutu alti) yaniliyor -> sikisik perspektifte metrelerce
         konum hatasi + kareler arasi ziplama,
      2) "cok uzun kutu" boyut denetimini (TRACK_SIZE_CHECK) tetikleyip takibi
         gereksiz KESIYORDU.
    Kutunun UST kenari (kafa) alt kenardan cok daha stabil oldugu icin, yuksekligi
    beklenen boya cekmek ayak noktasini gercek yer-temas seviyesine yaklastirir.
    Kutu zaten kisa/normalse DOKUNULMAZ -> takip kesilmez.

    Donus: yuksekligi kirpilmis (x, y, w, h) veya kirpma gerekmezse aynen bbox.
    """
    x, y, w, h = bbox
    exp_h = expected_height(state, x + w * 0.5, y + h)
    if exp_h is None or exp_h <= 1.0:
        return bbox
    cap = exp_h * config.BOX_HEIGHT_CAP_FACTOR
    if h <= cap:
        return bbox
    # Yeni ayak derinligini bir kez tazele (expected_height derinlige baglidir).
    exp_h2 = expected_height(state, x + w * 0.5, y + cap)
    if exp_h2 is not None and exp_h2 > 1.0:
        cap = exp_h2 * config.BOX_HEIGHT_CAP_FACTOR
    return (x, y, w, max(2, int(round(cap))))


def verify(state, frame, bbox):
    """
    Tracker saglik denetimi: bbox icinde arka plandan sapan piksellerin orani.

    Tracker bos zemine kilitlendiyse bu oran cok dusuk cikar -> main.py bunu
    config.TRACK_VERIFY_MIN_FG_RATIO ile karsilastirip yeniden tespit tetikler.
    """
    x, y, w, h = (int(v) for v in bbox)
    x = max(0, min(state["frame_w"] - 1, x))
    y = max(0, min(state["frame_h"] - 1, y))
    w = max(1, min(state["frame_w"] - x, w))
    h = max(1, min(state["frame_h"] - y, h))

    patch = frame[y:y + h, x:x + w]
    if patch.size == 0:
        return 0.0

    diff = cv2.absdiff(cv2.GaussianBlur(patch, config.BG_BLUR, 0),
                       state["bg"][y:y + h, x:x + w])
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    return float(np.count_nonzero(gray > config.BG_DIFF_THRESH)) / gray.size


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
    world = court.image_to_court(state["H"], foot_x, foot_y)
    if world is None:
        return None
    wy = world[1]
    a = court.court_to_image(state["H_inv"], 0.0, wy)
    b = court.court_to_image(state["H_inv"], config.COURT_WIDTH, wy)
    if a is None or b is None:
        return None
    px_per_m = abs(b[0] - a[0]) / config.COURT_WIDTH
    return config.PLAYER_HEIGHT_M * px_per_m


def _clamp_bbox(x, y, w, h, frame_w, frame_h):
    """Bbox'i kare sinirlarina kirpar ve tam sayiya cevirir. Gecersizse None doner."""
    x0 = int(round(max(0.0, x)))
    y0 = int(round(max(0.0, y)))
    x1 = int(round(min(float(frame_w), x + w)))
    y1 = int(round(min(float(frame_h), y + h)))

    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return (x0, y0, x1 - x0, y1 - y0)


def detect_machine(bg, search_mask, H):
    """
    SABIT top-atma makinesini (aluminyum/beyaz) MEDYAN ARKA PLAN goruntusu
    uzerinde bulur.

    NEDEN ARKA PLAN UZERINDE: Makine SABIT oldugu icin medyan arka planda KALIR
    (oyuncu hareketli oldugundan medyanla silinir). Yani makineyi oyuncudan
    ayirmanin bedava yolu, zaten kurulu medyan modelde aramaktir. Bu yuzden
    arka-plan-cikarma ile ARANMAZ (makine foreground degil, background'dur).

    AYIRT ETME: dusuk doygunluk (S) + yuksek parlaklik (V) = beyaz/aluminyum;
     + KORT-ICI (search_mask) + KOMPAKT blob (en/boy makul -> beyaz CIZGILER ve
    kort disi BINA elenir). En buyuk gecerli aday secilir.

    bg          : background.build ciktisi (medyan arka plan, BGR, islenmis olcek)
    search_mask : kort-ici (yukari genisletilmis) arama bolgesi, uint8 [0/255]
    H           : goruntu -> kort homografisi (ayak dogrulamasi icin)

    Donus: (world (x, y) metre, bbox (x, y, w, h)) veya None (bulunamazsa).
    """
    hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv,
                        (0, 0, int(config.MACHINE_MIN_VAL)),
                        (180, int(config.MACHINE_MAX_SAT), 255))
    cv2.bitwise_and(white, search_mask, dst=white)     # yalnizca kort ici

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.MACHINE_MORPH_KERNEL)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, k)   # ince gurultu/cizgi kirintisi
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, k)  # blob'u toparla

    n, _, stats, _ = cv2.connectedComponentsWithStats(white, 8, cv2.CV_32S)
    best = None
    best_area = 0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if w <= 0 or h <= 0:
            continue
        if area < config.MACHINE_MIN_AREA or area > config.MACHINE_MAX_AREA:
            continue
        aspect = h / float(w)
        if aspect < config.MACHINE_ASPECT_MIN or aspect > config.MACHINE_ASPECT_MAX:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        # Ayak (alt-orta) noktasi kort duzlemine dusuyor mu?
        world = court.image_to_court(H, x + w * 0.5, y + h)
        if world is None or not court.is_on_court(world[0], world[1]):
            continue
        if area > best_area:
            best_area = area
            best = (world, (x, y, w, h))
    return best
