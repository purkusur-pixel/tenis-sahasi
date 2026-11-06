"""
fps_bench.py -- TEMIZ FPS olcumu (Pi).

Onceki FPS sayilari kirliydi: 16 kombinasyon PES PESE kosmustu ve segment icin
seek yapiliyordu; hem termal birikme hem seek maliyeti mutlak degerleri
bozuyordu (yalnizca goreli kiyas gecerliydi). Bu modul her kombinasyonu:

  - AYRI surecte,               (onceki kosunun bellek/onbellek etkisi kalmasin)
  - SEEK OLMADAN (kare 0'dan),  (arama maliyeti olcume karismasin)
  - AYNI kare sayisinda,        (esit is yuku)
  - 5 TEKRAR ile,               (zamanlama degisken -> ort +- ss anlamli)
  - CPU sicakligi loglanarak    (uzun kosuda throttling var mi?)

kosturur.

NOT: Dogruluk metrikleri icin tekrar YAPILMAZ -- boru hatti deterministik
oldugu olculdu (ayni kombinasyon iki kez kosturuldu, veri satirlari birebir
ayni). Orada tekrar sahte hassasiyet uretirdi.

Kullanim (Pi'de):
    venv/bin/python fps_bench.py --frames 300 --repeats 5 --out fps_bench.json
"""

import argparse
import json
import os
import statistics as st
import subprocess
import sys
import time

TRACKERS = ["MOSSE", "KCF", "CSRT_FAST", "CSRT"]
PREDICTS = ["CV", "CA", "ABG", "IMM"]


def cpu_temp():
    """CPU sicakligi (C). Throttling kontrolu icin."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as fh:
            return int(fh.read().strip()) / 1000.0
    except Exception:
        return None


def throttled():
    """Pi throttling bayragi (0x0 = temiz)."""
    try:
        out = subprocess.check_output(["vcgencmd", "get_throttled"], timeout=5)
        return out.decode().strip().split("=")[-1]
    except Exception:
        return None


CHILD = r'''
import json, sys, time
import config
config.ASK_TRACKER = False; config.ASK_PREDICT = False
config.TRACKER_TYPE = "{T}"; config.PREDICT_TYPE = "{P}"
config.EVAL_LOG = False            # yazma yok -> saf isleme hizi
config.EVAL_AUTO_REPORT = False
config.SAVE_OUTPUT = False         # video yazma FPS'e karismasin
config.DRAW_VISUALS = False   # cekirdek dongu olculur
config.ZONE_PRINT = False; config.ZONE_ANALYSIS = False
config.SAVE_FIRST_FRAME_DEBUG = False
config.EVAL_START_FRAME = 0        # SEEK YOK
config.EVAL_MAX_FRAMES = 0
config.VIDEO_PATH = "{VIDEO}"
import cv2, numpy as np, court, detector, background, zones, predictors, shot

# --- kurulum (olcume DAHIL DEGIL) ---
cap = cv2.VideoCapture(config.VIDEO_PATH)
ok, frame = cap.read()
s = float(config.PROC_SCALE)
frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=config.RESIZE_INTERP)
corners = court.load_corners(frame.shape)
if corners is None:
    corners = court.auto_detect_corners(frame, verbose=False)
if config.REFINE_CORNERS:
    corners, _, _ = court.refine_corners(frame, corners, verbose=False)
H, Hi = court.compute_homography(corners)
mask, rect = court.build_search_mask(frame.shape, Hi)
bg = background.build(config.VIDEO_PATH, s, verbose=False)
det = detector.init(frame, mask, rect, bg, H, Hi)
pred = predictors.create(config.PREDICT_TYPE)

def mk():
    kind = str(config.TRACKER_TYPE).upper()
    return main._make_csrt_fast() if kind == "CSRT_FAST" else main._make_tracker(kind)
import main

# --- olcum: SADECE kare isleme dongusu ---
tracker = None; bbox = None; ready = False; pos = None
n = {N}; times = []
for i in range(n):
    ok, raw = cap.read()
    if not ok: break
    t0 = time.perf_counter()
    fr = cv2.resize(raw, None, fx=s, fy=s, interpolation=config.RESIZE_INTERP)
    tracked = False
    if tracker is not None:
        ok_t, box = tracker.update(fr)
        if ok_t:
            bbox = tuple(int(round(v)) for v in box); tracked = True
    if tracker is None or not tracked:
        found = detector.detect(det, fr)
        if found is not None:
            tracker = mk()
            if tracker is not None:
                tracker.init(fr, found); bbox = found; tracked = True
    if tracked and bbox is not None:
        fx, fy = bbox[0] + bbox[2] * 0.5, bbox[1] + bbox[3]
        w = court.image_to_court(H, fx, fy)
        if w is not None:
            if not ready:
                pred.reset(w); ready = True; pos = w
            else:
                pred.predict(); pos = pred.update(w)
            zones.get_zone(pos[0], pos[1])
    times.append(time.perf_counter() - t0)
cap.release()
print(json.dumps({{"n": len(times), "total": sum(times),
                   "fps": len(times) / sum(times) if times else 0}}))
'''


def run_once(video, T, P, n):
    src = CHILD.format(T=T, P=P, N=n, VIDEO=video)
    r = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True,
                       cwd=os.getcwd(), timeout=900)
    for line in reversed(r.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    return None


def main_bench():
    ap = argparse.ArgumentParser(description="Temiz FPS olcumu")
    ap.add_argument("--video", default="input/A1.mp4")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default="fps_bench.json")
    args = ap.parse_args()

    print("video: {} | kare: {} | tekrar: {}  (cekirdek dongu; cizim/video yazma HARIC)".format(
        args.video, args.frames, args.repeats))
    print("baslangic sicakligi: {} C | throttled: {}".format(cpu_temp(), throttled()))
    print()

    results = {}
    for T in TRACKERS:
        for P in PREDICTS:
            fps_list, temps = [], []
            for rep in range(args.repeats):
                res = run_once(args.video, T, P, args.frames)
                if res:
                    fps_list.append(res["fps"])
                temps.append(cpu_temp())
            if not fps_list:
                print("  {:<11}{:<6} HATA".format(T, P))
                continue
            m = st.mean(fps_list)
            sd = st.pstdev(fps_list) if len(fps_list) > 1 else 0.0
            results["{}+{}".format(T, P)] = {
                "fps_mean": m, "fps_sd": sd, "fps_all": fps_list,
                "temp_end": temps[-1], "n_frames": args.frames,
            }
            print("  {:<11}{:<6} {:6.1f} +- {:4.1f} FPS   (sicaklik {} C)".format(
                T, P, m, sd, temps[-1]))

    print("\nbitis sicakligi: {} C | throttled: {}".format(cpu_temp(), throttled()))
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"video": args.video, "frames": args.frames,
                   "repeats": args.repeats, "scope": "core_loop",
                   "results": results}, fh, indent=1)
    print("-> {}".format(args.out))


if __name__ == "__main__":
    main_bench()
