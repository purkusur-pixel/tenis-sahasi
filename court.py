"""
court.py
--------
Kort geometrisi: kose secimi, homography, nokta donusumleri, arama maskesi ve
kort cizgilerinin cizimi.

Bolge (zone) modeli ve panel artik zones.py'dedir; bu dosya yalnizca geometri ile
ilgilenir.

KOORDINAT SISTEMI:
    Goruntu (piksel) <--H--> Kort (metre)
    H     : goruntu -> kort
    H_inv : kort    -> goruntu

NOT: Kose secimi, boru hattinin calistigi kare uzerinde (PROC_SCALE uygulanmis)
     yapilmalidir. Aksi halde piksel koordinatlari tutmaz.
"""

import itertools
import json
import os

import cv2
import numpy as np

import config

CORNERS_PATH = config.CORNERS_PATH

# Kose semasi kimligi. Referans noktalarinin ANLAMI/SAYISI degistiginde bu yazi
# da degistirilmelidir.
# Neden gerekli: load_corners eskiden yalnizca kare BOYUTUNU dogruluyordu. Kose
# semasi "4 dip kose"den "uzak dip + file"ye gecince, eski bir court_corners.json
# sessizce yuklenip tamamen yanlis bir kort uretiyordu (yakin yari kameranin
# arkasina dusuyor, court_to_image None donuyordu). Sema uyusmazsa kayit reddedilir
# ve kullaniciya koseler yeniden sectirilir.
#
# "far_baseline+net" -> "net+near_service": uzak dip koseler goruntude asiri
# sikistigi icin (bkz. config.py COURT_CORNERS_WORLD basligindaki gerekce)
# capa artik 4 degil 5 nokta (file + yakin servis cizgisi kesisimleri). Surum
# bilerek degistirildi ki ESKI semayla kaydedilmis court_corners.json dosyalari
# SESSIZCE yanlis yorumlanmasin; bu dosyalar otomatik reddedilip koseler (ya da
# otomatik tespit) yeniden calistirilir.
CORNER_SCHEME = "net+near_service"

_W = config.COURT_WIDTH
_L = config.COURT_LENGTH
_SL = config.SINGLES_LEFT_X
_SR = config.SINGLES_RIGHT_X
_CX = config.CENTER_X
_NET = config.NET_Y
_FS = config.FAR_SERVICE_Y
_NS = config.NEAR_SERVICE_Y

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
    d = config.LINE_NEIGHBOR_DIST
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
    left, right = np.roll(g, d, axis=1), np.roll(g, -d, axis=1)
    up, down = np.roll(g, d, axis=0), np.roll(g, -d, axis=0)

    t = config.LINE_DARK_DELTA
    vert = (g - left > t) & (g - right > t)
    horiz = (g - up > t) & (g - down > t)
    mask = ((vert | horiz) & (g > config.LINE_MIN_BRIGHT)).astype(np.uint8) * 255
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
        if max(a, b) < config.LINE_MIN_LENGTH or min(a, b) > config.LINE_MAX_THICK:
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
        if g["w"] < config.LINE_MIN_WEIGHT:
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
    homography ile config.COURT_CORNERS_WORLD icindeki 5 dunya noktasi
    (uzerinde HICBIRI dogrudan bu yardimci noktalarda olmayabilir; file ve
    tekler yan cizgileri matematiksel olarak ekstrapole edilir) goruntuye
    izdusurulerek disariya verilir. Boylece disari her zaman config.py'de
    tanimli GUNCEL sema donmus olur; ic hesaplama semasi degismez.

    Donus: config.COURT_CORNERS_WORLD sirasinda (N, 2) float32 dizi, veya
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
    horiz = sorted([l for l in lines if abs(l["ang"]) < config.LINE_HORIZ_MAX_ANGLE],
                   key=lambda l: -_y_at(l, w * 0.5))          # en alttaki once
    steep = [l for l in lines if abs(l["ang"]) >= config.LINE_HORIZ_MAX_ANGLE]
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
        return float((d < config.AUTO_LINE_TOL).mean())

    # Dunya karsiliklari. DIS (doubles) yan cizgileri her zaman; IC (singles) yan
    # cizgileri tespit edilirse EKLENIR. Servis cizgisi (SONSUZ dogru olarak) y=NS
    # duzlemidir; yan cizgilerle kesisimi dunyada (x, NS) noktasidir.
    W_DOUBLES = [(0.0, _L), (_W, _L), (_W, _NS), (0.0, _NS)]      # dL∩base, dR∩base, dR∩serv, dL∩serv
    W_SINGLES = [(_SL, _L), (_SR, _L), (_SL, _NS), (_SR, _NS)]    # sL∩base, sR∩base, sL∩serv, sR∩serv
    best = None

    for base, serv in itertools.combinations(horiz[:4], 2):
        y_base = _y_at(base, w * 0.5)
        if y_base <= _y_at(serv, w * 0.5):
            continue                                   # dip cizgi servisin ALTINDA
        if not (0.55 * h <= y_base <= 1.05 * h):
            continue                                   # dip cizgi karenin alt kisminda

        ordered = sorted(steep, key=lambda l: _x_at(l, y_base))
        for dL, sL, sR, dR in _doubles_candidates(ordered, y_base, w):
            d_pts = [_intersect(dL, base), _intersect(dR, base),
                     _intersect(dR, serv), _intersect(dL, serv)]
            if any(p is None for p in d_pts):
                continue
            quad = np.array(d_pts, dtype=np.float32)
            if not cv2.isContourConvex(quad.astype(np.int32)):
                continue

            # Aday karsilik kumeleri: (A) yalnizca doubles (4 nokta), (B) singles
            # de tespit edildiyse doubles+singles (8 nokta, over-determined LSQ).
            # IKISI DE denenip verify daha iyisini secer -> singles tespiti bir
            # cozumu bozarsa doubles-only kazanir (guvenli genisletme).
            corr_sets = [(W_DOUBLES, list(d_pts))]
            if sL is not None and sR is not None:
                s_pts = [_intersect(sL, base), _intersect(sR, base),
                         _intersect(sL, serv), _intersect(sR, serv)]
                if all(p is not None for p in s_pts):
                    corr_sets.append((W_DOUBLES + W_SINGLES, list(d_pts) + list(s_pts)))

            for world_pts, img_pts in corr_sets:
                world_arr = np.array(world_pts, dtype=np.float64)
                img_arr = np.array([(float(p[0]), float(p[1])) for p in img_pts], dtype=np.float64)
                try:
                    # findHomography (method=0): 4 nokta -> tam cozum; >4 nokta ->
                    # TUM karsiliklarin ortak hatasini minimize eden LSQ homography.
                    H_inv, _ = cv2.findHomography(world_arr, img_arr, method=0)
                except cv2.error:
                    continue
                if H_inv is None:
                    continue
                H_inv = H_inv.astype(np.float64)
                if H_inv[2, 0] * _CX + H_inv[2, 1] * _NET + H_inv[2, 2] < 0:
                    H_inv = -H_inv
                score = verify(H_inv)
                if best is None or score > best[0]:
                    best = (score, H_inv)

    if best is None or best[0] < config.AUTO_MIN_SCORE:
        if verbose:
            got = "{:.3f}".format(best[0]) if best else "yok"
            print("Otomatik kort tespiti basarisiz (puan {} < {}).".format(
                got, config.AUTO_MIN_SCORE))
        return None

    score, H_inv = best
    corners = []
    for wx, wy in config.COURT_CORNERS_WORLD:
        p = court_to_image(H_inv, float(wx), float(wy))
        if p is None:
            return None
        corners.append(p)
    if verbose:
        print("Otomatik kort tespiti: puan {:.3f}".format(score))
    return np.array(corners, dtype=np.float32)


def _doubles_candidates(ordered, y_base, frame_w):
    """
    Egik cizgilerden (dL, sL, sR, dR) dortlulerini secer: DIS (cift/doubles) ve
    IC (tekler/singles) yan cizgileri.

    Once gercek kort oranindan yararlanilir: dort cizgilik (a<b<c<d) altkumelerde
        (x_c - x_b) / (x_d - x_a)  ~  8.23 / 10.97 = 0.750
    saglaniyorsa (a, d) cift, (b, c) tekler yan cizgisidir -> (a, b, c, d) dondurulur.
    Boylece tekler cizgileri hem yanlislikla cift sanilmaz HEM DE homografiye
    fazladan karsilik olarak eklenebilir (bkz. auto_detect_corners, over-determined).

    Oran testi tutmazsa en distaki birkac cift YEDEK olarak (sL=sR=None) denenir.
    """
    xs = [_x_at(l, y_base) for l in ordered]
    out = []
    ratio = (_SR - _SL) / _W
    for a, b, c, d in itertools.combinations(range(len(ordered)), 4):
        outer = xs[d] - xs[a]
        if outer < 0.4 * frame_w:
            continue
        if abs((xs[c] - xs[b]) / outer - ratio) < config.AUTO_RATIO_TOL:
            out.append((ordered[a], ordered[b], ordered[c], ordered[d]))
    for i in range(min(2, len(ordered))):
        for j in range(max(0, len(ordered) - 2), len(ordered)):
            if i < j and xs[j] - xs[i] >= 0.4 * frame_w:
                out.append((ordered[i], None, None, ordered[j]))
    seen, uniq = set(), []
    for tup in out:
        k = (id(tup[0]), id(tup[3]))     # ayni (dL, dR) ciftini iki kez ekleme
        if k not in seen:
            seen.add(k)
            uniq.append(tup)
    return uniq


def refine_corners(frame, corners, verbose=True):
    """
    Referans noktalarini, projekte edilen TUM kort cizgilerini goruntudeki gercek
    beyaz cizgi maskesine hizalayacak sekilde ince ayar yapar (line-mask ICP benzeri
    koordinat inisi).

    NEDEN: 5 capa noktasi goruntude ince bir yatay banda sikisir; homografinin
    DERINLIK (yukari/asagi) yonundeki ekstrapolasyonu bu yuzden zayiftir ve kort
    modeli gercek korttan kayar (ozellikle file-orta ve uzak/yakin uclar). Bu
    fonksiyon 5 noktayi degil, projekte edilen cizgilerin BINLERCE piksellik ortak
    hizasini kullanir: yakin dip cizgi, servis cizgileri, tekler/cift yan cizgileri
    ve orta servis cizgisi boyunca yogun ornek alinir; noktalar bu ornekler gercek
    cizgilere oturana kadar kucuk adimlarla kaydirilir. Boylece 5 noktadan cok daha
    fazla kanit kullanilir, fit derinlikte stabillesir.

    GUVENLIK: skor (AUTO_LINE_TOL icindeki ornek orani) iyilesmezse ORIJINAL
    noktalar aynen dondurulur -> bu adim asla mevcut tespiti bozamaz.

    Donus: (refined_corners float32 (N,2), score_before, score_after).
    """
    h, w = frame.shape[:2]
    mask = _line_mask(frame)
    dist = cv2.distanceTransform(cv2.bitwise_not(mask), cv2.DIST_L2, 3)

    # File (net) cizgisi HARIC tum kort cizgileri boyunca yogun ornek (dunya).
    # Net dislanir: fiziksel file mesh'i maskede karmasik/guvenilmez bir yapidir
    # (auto_detect_corners.verify de ayni sebeple net'i disliyor).
    painted = [ln for ln in COURT_LINES
               if not (abs(ln[0][1] - _NET) < 1e-6 and abs(ln[1][1] - _NET) < 1e-6)]
    npl = int(config.REFINE_SAMPLES_PER_LINE)
    near_only = not config.REFINE_INCLUDE_FAR_HALF
    pts = []
    for (ax, ay), (bx, by) in painted:
        for t in np.linspace(0.0, 1.0, npl):
            wy = ay + (by - ay) * t
            if near_only and wy < _NET - 1e-6:
                continue               # yalnizca YAKIN yari isteniyorsa uzak yariyi atla
            pts.append((ax + (bx - ax) * t, wy))
    if len(pts) < 40:
        return np.asarray(corners, dtype=np.float32), 0.0, 0.0
    samples = np.array(pts, dtype=np.float64)
    sx, sy = samples[:, 0], samples[:, 1]

    trunc = float(config.REFINE_TRUNCATE)

    def cost_and_score(H_inv):
        """Projekte ornek noktalarinin cizgi maskesine ROBUST (kirpilmis) ortalama
        uzakligi (cost) ve AUTO_LINE_TOL icinde kalan orani (score). Cost dusuk =
        iyi hiza. Kirpma (trunc) ortulu/yanlis cizgi ornekelerinin fit'i
        saptirmasini engeller."""
        den = H_inv[2, 0] * sx + H_inv[2, 1] * sy + H_inv[2, 2]
        good = den > 1e-9
        if int(good.sum()) < 40:
            return 1e9, 0.0
        px = (H_inv[0, 0] * sx + H_inv[0, 1] * sy + H_inv[0, 2]) / den
        py = (H_inv[1, 0] * sx + H_inv[1, 1] * sy + H_inv[1, 2]) / den
        inside = good & (px >= 0) & (px < w - 1) & (py >= 0) & (py < h - 1)
        if int(inside.sum()) < 40:
            return 1e9, 0.0
        d = dist[py[inside].astype(np.int32), px[inside].astype(np.int32)]
        cost = float(np.minimum(d, trunc).mean())
        return cost, float((d < config.AUTO_LINE_TOL).mean())

    def cost_of(cand):
        try:
            _, H_inv = compute_homography(cand)
        except (ValueError, cv2.error, np.linalg.LinAlgError):
            return 1e9
        return cost_and_score(H_inv)[0]

    orig = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    n = orig.shape[0]
    try:
        _, H0 = compute_homography(orig)
    except (ValueError, cv2.error, np.linalg.LinAlgError):
        return np.asarray(corners, dtype=np.float32), 0.0, 0.0
    cost_before, score_before = cost_and_score(H0)

    # --- Hareket kumesi: tek-nokta + GRUP hareketleri --------------------------
    # Tek tek nokta oynatmak derinlikte takasa girer (bir noktayi duzeltmek
    # digerini bozar). GRUP hareketleri bir cizgiyi (file / servis) veya tum
    # kortu birlikte kaydirir; boylece "file noktalari topluca cok asagida" gibi
    # SISTEMATIK derinlik/kayma hatalari tek adimda duzeltilebilir. Her hareket
    # (etkilenen nokta indeksleri, dx, dy) birim yonudur; +-step ile olceklenir.
    moves = []
    for i in range(n):
        moves.append(([i], 1.0, 0.0))
        moves.append(([i], 0.0, 1.0))
    file_idx = [i for i in range(n) if i < 3]          # 0,1,2 -> file (net) hatti
    serv_idx = [i for i in range(n) if i >= 3]         # 3,4  -> yakin servis hatti
    if file_idx:
        moves.append((file_idx, 0.0, 1.0))            # file hattini birlikte derinlikte kaydir
    if serv_idx:
        moves.append((serv_idx, 0.0, 1.0))            # servis hattini birlikte derinlikte kaydir
    allp = list(range(n))
    moves.append((allp, 1.0, 0.0))                     # tum kort: global x oteleme
    moves.append((allp, 0.0, 1.0))                     # tum kort: global y oteleme

    # --- Koordinat/grup inisi: her hareketi +-step dene, iyilesirse tut. -------
    best = orig.copy()
    best_cost = cost_before
    step = float(config.REFINE_INIT_STEP)
    max_shift = float(config.REFINE_MAX_SHIFT)

    while step >= config.REFINE_MIN_STEP:
        improved = False
        for idxs, dx, dy in moves:
            for sign in (1.0, -1.0):
                cand = best.copy()
                for j in idxs:
                    cand[j, 0] += sign * dx * step
                    cand[j, 1] += sign * dy * step
                # Hicbir nokta orijinalinden max_shift'ten fazla kaymasin.
                if np.any(np.abs(cand - orig) > max_shift):
                    continue
                c = cost_of(cand)
                if c < best_cost - 1e-6:
                    best_cost, best, improved = c, cand, True
        if not improved:
            step *= 0.5

    try:
        _, Hb = compute_homography(best)
    except (ValueError, cv2.error, np.linalg.LinAlgError):
        return orig.astype(np.float32), score_before, score_before
    _, score_after = cost_and_score(Hb)

    # Iyilesme yoksa (skor artmadi) orijinali koru: bu adim asla bozmaz.
    if score_after <= score_before + 1e-6:
        if verbose:
            print("Kort ince ayari: iyilesme yok (skor {:.3f}); orijinal korunuyor.".format(
                score_before))
        return orig.astype(np.float32), score_before, score_before

    if verbose:
        print("Kort ince ayari: skor {:.3f} -> {:.3f} | ort. cizgi hatasi {:.2f} -> {:.2f} px.".format(
            score_before, score_after, cost_before, best_cost))
    return best.astype(np.float32), score_before, score_after


def confirm_corners(frame, corners, window_name="Otomatik Kort Tespiti"):
    """
    Otomatik bulunan kortu kullaniciya gosterip onay ister.

    Yanlis bir otomatik tespit sessizce her seyi bozacagi icin bu adim vardir.

    Kontroller:  ENTER / c : kabul      m : elle sec      q / ESC : iptal
    Donus: True (kabul) / False (elle secime gec)
    """
    H_inv = compute_homography(corners)[1]
    canvas = frame.copy()
    draw_lines(canvas, project_lines(H_inv, COURT_LINES), config.COLOR_COURT,
               config.LINE_THICKNESS + 1)
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

    Secim sirasi (config.COURT_CORNERS_WORLD / config.CORNER_LABELS ile AYNI sira):
        file-sol -> file-sag -> file-orta -> yakin servis-sol -> yakin servis-sag
    (bkz. config.py COURT_CORNERS_WORLD basligi icin ayrintili gerekce).

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
    n_points = len(config.CORNER_LABELS)
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
            cv2.polylines(canvas, [pts], len(points) == n_points, config.COLOR_COURT, 2)

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
                          config.COLOR_COURT, 2)
            dst_x = 10 if cx > w // 2 else w - lens_size - 10
            dst_y = 10 if cy > h // 2 else h - lens_size - 10
            if 0 <= dst_x and dst_x + lens_size <= w and 0 <= dst_y and dst_y + lens_size <= h:
                canvas[dst_y:dst_y + lens_size, dst_x:dst_x + lens_size] = lens

        info = config.CORNER_LABELS[len(points)] if len(points) < n_points \
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
    return str(config.VIDEO_PATH)


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

    Girdi noktalari config.COURT_CORNERS_WORLD ile AYNI sirada olmalidir (bkz.
    config.CORNER_LABELS): file uzerinde 3 nokta (sol/sag/orta) + yakin servis
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
    edilir (bkz. config.py COURT_CORNERS_WORLD basligi).

    Matrisler, projektif payda (w) kort alani icinde POZITIF olacak sekilde
    normalize edilir. Bu sayede kort disina tasan noktalarin kamera arkasina
    dusup dusmedigi paydanin isaretine bakilarak anlasilabilir.
    """
    world = config.COURT_CORNERS_WORLD
    img = np.asarray(image_pts, dtype=np.float32).reshape(-1, 2)
    if img.shape[0] != world.shape[0]:
        raise ValueError(
            "{} referans noktasi bekleniyordu, {} nokta geldi (config.CORNER_LABELS "
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
            "config.CORNER_LABELS sirasiyla secin (file sol -> file sag -> file orta "
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
        margin = config.COURT_MARGIN
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
        half = config.PLAYER_HALF
    if margin is None:
        margin = config.SEARCH_MARGIN

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
        thickness = config.LINE_THICKNESS
    for p1, p2 in projected:
        cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)


def _put_text(frame, text, org, scale=None, thickness=None):
    """Arka plan golgeli metin (her zeminde okunur)."""
    if scale is None:
        scale = config.FONT_SCALE
    if thickness is None:
        thickness = config.FONT_THICKNESS
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                config.COLOR_TEXT_BG, thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                config.COLOR_TEXT, thickness, cv2.LINE_AA)


def draw_ref_points(frame, points, show_labels=True):
    """
    Referans noktalarini numarali ve HER NOKTA icin AYRI RENKTE isaretler
    (renkler config.REF_POINT_COLORS'tan, sira config.CORNER_LABELS ile
    birebir eslesir). confirm_corners() ve save_first_frame_debug() bu tek
    fonksiyonu paylasir, iki ayri cizim kodu bakimda birbirinden sapmasin diye.

    show_labels=False -> sadece numara (kucuk onay/secim pencereleri icin, yer
                          dar oldugundan tam etiket sigmaz).
    show_labels=True  -> numara + tam referans noktasi aciklamasi (diske
                          kaydedilen dogrulama goruntusu icin).
    """
    h, w = frame.shape[:2]
    for i, (px, py) in enumerate(points):
        p = (int(round(px)), int(round(py)))
        color = config.REF_POINT_COLORS[i % len(config.REF_POINT_COLORS)]
        cv2.circle(frame, p, 7, color, -1, cv2.LINE_AA)
        cv2.circle(frame, p, 7, (0, 0, 0), 2, cv2.LINE_AA)   # kontrast halka
        label = config.CORNER_LABELS[i] if show_labels and i < len(config.CORNER_LABELS) \
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

    NEDEN GEREKLI: config.SHOW_WINDOW=False oldugunda (Raspberry Pi, headless
    calisma) ekranda hicbir gorsel geri bildirim yoktur; kort tespitinin/
    homography'nin dogru calisip calismadigini kontrol etmenin baska yolu
    kalmaz. Bu fonksiyon kurulum sirasinda TEK SEFERLIK cagrilir; video
    isleme dongusune hicbir performans maliyeti eklemez.
    """
    canvas = frame.copy()
    draw_lines(canvas, project_lines(H_inv, COURT_LINES), config.COLOR_COURT,
               config.LINE_THICKNESS + 1)
    draw_ref_points(canvas, corners, show_labels=True)
    _put_text(canvas, "Kort tespiti - referans noktalari (ilk kare)", (10, 24), 0.7, 2)

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    cv2.imwrite(path, canvas)
    print("Ilk kare kort tespiti kaydedildi: {}".format(path))
