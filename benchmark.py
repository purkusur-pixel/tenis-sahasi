"""
benchmark.py
------------
PC <-> Raspberry Pi performans karsilastirma araci.

Ana boru hattini (main.py ile AYNI asamalar) headless olarak N kare isler ve her
asamanin kare basina suresini + toplam FPS'i + makine bilgisini basar. Projeye
HICBIR sey yazmaz; sadece mevcut modulleri cagirir.

Ayni komutu iki makinede de calistir, ciktilari yan yana koy:
    python benchmark.py                 # varsayilan 120 kare
    python benchmark.py 300             # 300 kare
    python benchmark.py 200 MOSSE       # kare sayisi + takipci tipini zorla

Kort koseleri: kayitli court_corners.json varsa kullanilir; yoksa otomatik tespit
denenir; o da olmazsa kare merkezli kaba bir dortgenle olculur (asamalarin SURESI
gecerli kalir, yalnizca konum anlamsiz olur - benchmark icin sorun degil).
"""

import platform
import sys
import time
from collections import defaultdict

import cv2
import numpy as np

import background
import config
import court
import detector
import main as M
import shot
import zones


def _corners(frame, H_hint=None):
    """Kayitli -> otomatik -> kaba merkez dortgen sirasiyla kose bulur."""
    c = court.load_corners(frame.shape)
    if c is not None:
        return c, "kayitli (court_corners.json)"
    c = court.auto_detect_corners(frame, verbose=False)
    if c is not None:
        return c, "otomatik tespit"
    h, w = frame.shape[:2]
    # compute_homography artik 5 referans noktasi bekliyor (bkz. config.py
    # COURT_CORNERS_WORLD). Konum zaten anlamsiz (yalnizca asama SURELERI
    # olculuyor); ayni kaba dortgen korunup 5. nokta (file orta) 1. ve 2.
    # noktanin ortalamasi olarak eklendi.
    c = np.array([[w * 0.37, h * 0.53], [w * 0.70, h * 0.52],
                  [w * 0.535, h * 0.525], [w * 0.27, h * 0.59],
                  [w * 0.81, h * 0.58]], dtype=np.float32)
    return c, "KABA merkez dortgen (konum anlamsiz, sure gecerli)"


def run(n_frames, force_tracker=None):
    if force_tracker:
        config.TRACKER_TYPE = force_tracker.upper()

    cap = cv2.VideoCapture(config.VIDEO_PATH)
    if not cap.isOpened():
        raise IOError("Video acilamadi: {}".format(config.VIDEO_PATH))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or config.FALLBACK_FPS

    ok, raw = cap.read()
    if not ok:
        raise IOError("Ilk kare okunamadi.")
    scale = float(config.PROC_SCALE)
    frame = cv2.resize(raw, None, fx=scale, fy=scale, interpolation=config.RESIZE_INTERP)
    frame_h, frame_w = frame.shape[:2]

    corners, corner_src = _corners(frame)
    H, H_inv = court.compute_homography(corners)
    search_mask, _ = court.build_search_mask(frame.shape, H_inv)
    bg = background.build(config.VIDEO_PATH, scale)
    det_state = detector.init(frame, search_mask, search_mask, bg, H, H_inv)
    zone_state = zones.build(H_inv, frame.shape)
    shot_state = shot.init()
    mog = M.create_mog() if config.USE_MOG2 else None
    mog_scale = float(config.MOG2_SCALE)

    tracker = M.create_tracker()
    bbox = (int(frame_w * 0.35), int(frame_h * 0.45), 45, 80)
    tracker.init(frame, bbox)

    t = defaultdict(float)
    wall0 = time.perf_counter()
    done = 0

    for _ in range(n_frames):
        a = time.perf_counter()
        ok, raw = cap.read()
        if not ok:
            break
        t["decode (cap.read)"] += time.perf_counter() - a

        a = time.perf_counter()
        frame = cv2.resize(raw, None, fx=scale, fy=scale, interpolation=config.RESIZE_INTERP)
        t["resize -> PROC_SCALE"] += time.perf_counter() - a

        a = time.perf_counter()
        mog_input = None
        if config.CUT_DETECTION or mog is not None:
            mog_input = cv2.resize(frame, None, fx=mog_scale, fy=mog_scale,
                                   interpolation=config.RESIZE_INTERP)
        t["resize -> MOG2/kesme"] += time.perf_counter() - a

        a = time.perf_counter()
        if config.CUT_DETECTION:
            shot.detect_cut(shot_state, mog_input, done)
        t["shot.detect_cut"] += time.perf_counter() - a

        a = time.perf_counter()
        if mog is not None:
            mog.apply(mog_input, learningRate=config.MOG2_LEARNING_RATE)
        t["MOG2 apply"] += time.perf_counter() - a

        a = time.perf_counter()
        tracker.update(frame)
        t["tracker.update (%s)" % config.TRACKER_TYPE] += time.perf_counter() - a

        a = time.perf_counter()
        zones.draw_overlay(frame, zone_state, "12")
        zones.draw_panel(frame, zone_state, "12", (5.0, 6.0), [(5.0, 6.0)])
        t["cizim (overlay+panel)"] += time.perf_counter() - a

        done += 1

    wall = time.perf_counter() - wall0

    # Resync karesindeki tespit (her ~REDETECT/RESYNC karede bir)
    a = time.perf_counter()
    for _ in range(3):
        detector.detect(det_state, frame)
    det_ms = (time.perf_counter() - a) / 3 * 1000

    cap.release()
    return {"t": t, "done": done, "wall": wall, "det_ms": det_ms,
            "src_fps": src_fps, "res": (frame_w, frame_h), "corner_src": corner_src}


def report(r):
    print("=" * 60)
    print("PERFORMANS KIYASI")
    print("=" * 60)
    print("Makine     : {}  ({})".format(platform.node(), platform.machine()))
    print("Islemci    : {}".format(platform.processor() or "bilinmiyor"))
    print("Python/CV2 : {} / {}".format(platform.python_version(), cv2.__version__))
    print("Video      : {}  @ {:.0f} FPS  -> islem {}x{} (PROC_SCALE={})".format(
        config.VIDEO_PATH, r["src_fps"], r["res"][0], r["res"][1], config.PROC_SCALE))
    print("Takipci    : {}   |   MOG2: {}   |   Koseler: {}".format(
        config.TRACKER_TYPE, "acik" if config.USE_MOG2 else "KAPALI", r["corner_src"]))
    print("Islenen kare: {}".format(r["done"]))
    print("-" * 60)
    print("{:<32}{:>10}{:>14}".format("ASAMA", "ms/kare", "toplam %"))
    print("-" * 60)

    per = {k: v / r["done"] * 1000 for k, v in r["t"].items()}
    total = sum(per.values())
    for k in sorted(per, key=lambda x: -per[x]):
        print("{:<32}{:>10.2f}{:>13.1f}%".format(k, per[k], 100 * per[k] / total))
    print("-" * 60)
    print("{:<32}{:>10.2f}".format("TOPLAM (asama)", total))
    print("{:<32}{:>10.2f}   (her ~{} karede)".format(
        "  + resync detector.detect", r["det_ms"], config.TRACK_RESYNC_INTERVAL))

    wall_fps = r["done"] / r["wall"] if r["wall"] > 0 else 0.0
    print("-" * 60)
    print("GERCEK ISLEM HIZI : {:.1f} FPS  ({:.1f} ms/kare, duvar saati)".format(
        wall_fps, r["wall"] / r["done"] * 1000))
    rt = wall_fps / r["src_fps"] if r["src_fps"] else 0.0
    print("GERCEK ZAMAN ORANI: {:.2f}x  ({})".format(
        rt, "gercek zamandan HIZLI" if rt >= 1.0 else "gercek zamandan YAVAS"))
    print("=" * 60)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    n = int(args[0]) if args and args[0].isdigit() else 120
    trk = next((a for a in args if not a.isdigit()), None)
    report(run(n, trk))
