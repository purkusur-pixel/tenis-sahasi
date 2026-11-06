"""
eval_runner.py -- GT segmentlerinde tracker x tahmin modeli koşuları (Pi).

Her (segment, tracker, tahmin) icin boru hattini SEGMENTIN TAMAMINDA (300 kare)
kosturur ve kare-kare ciktiyi CSV'ye yazar. Degerlendirme sonra yalnizca
ETIKETLI karelerde (her 5.) yapilir -- tracker sureklilige ihtiyac duydugu icin
tum segment islenmeli, ama GT yalnizca ornek karelerde var.

TEKRAR YOK: boru hatti deterministik oldugu OLCULDU (ayni kombinasyon iki kez
kosturuldu, veri satirlari birebir ayni). Dogruluk metrikleri icin tekrar
sahte hassasiyet uretirdi (5 ozdes deger, ss=0). Zamanlama tekrarlari AYRI
olarak fps_bench.py'de yapilir.

Cikti: eval_gt/<segment>__<TRACKER>__<PREDICT>.csv
"""

import json
import os
import time

import numpy as np

import config

SEGMENTS_JSON = "gt_index.json"
OUT_DIR = "eval_gt"

TRACKERS = ["MOSSE", "KCF", "CSRT_FAST", "CSRT"]
PREDICTS = ["CV", "CA", "ABG", "IMM"]

# Sabit kosul: menu yok, video yazma yok, cizim yok (hiz + disk).
config.ASK_TRACKER = False
config.ASK_PREDICT = False
config.EVAL_LOG = True
config.EVAL_AUTO_REPORT = False
config.SAVE_OUTPUT = False
config.DRAW_VISUALS = False          # gorsellestirme degerlendirmeyi etkilemez
config.ZONE_PRINT = False
config.ZONE_ANALYSIS = False
config.SAVE_FIRST_FRAME_DEBUG = False

import main  # noqa: E402  (config ayarlandiktan sonra)
import court  # noqa: E402


def install_human_corners(video):
    """
    INSAN tikladigi kort noktalarini sistemin bekledigi court_corners.json'a yazar.

    NEDEN: Sistemin OTOMATIK kort tespiti A2'de tamamen basarisiz oluyor ("puan yok")
    ve headless ortamda elle secime dusemedigi icin bosuna cikiyordu. Ayrica her video
    farkli kalibrasyon kaynagi kullanirsa (A1 kayitli, A3/C01 otomatik, A2 yok)
    takip hatasi ile kalibrasyon hatasi birbirine karisir.

    Bu yuzden DEGERLENDIRMEDE tum videolar AYNI (insan) kalibrasyonu kullanir; boylece
    olculen sey saf takip + bolge performansidir. Otomatik kalibrasyonun kendi
    dogrulugu/basarisizligi AYRICA raporlanir (held-out yeniden-projeksiyon hatasi).

    Insan noktalari TAM cozunurluktedir; sistem PROC_SCALE olceginde calistigi icin
    olceklenerek yazilir.
    """
    src = "court_{}.json".format(video)
    if not os.path.isfile(src):
        return False
    with open(src, encoding="utf-8") as fh:
        gt = json.load(fh)
    w, p = gt["world"], gt["points"]
    # config.COURT_CORNERS_WORLD ile AYNI sirada olmali -> CORNER_LABELS sirasi
    order = [n for n in config.CORNER_LABELS]
    pts = []
    for lbl in order:
        # etiketler "1/5: FILE - SOL ..." formatinda; dunya konumundan eslestir
        idx = config.CORNER_LABELS.index(lbl)
        wx, wy = config.COURT_CORNERS_WORLD[idx]
        match = None
        for name, ww in w.items():
            if ww["role"] == "fit" and abs(ww["x"] - wx) < 0.01 and abs(ww["y"] - wy) < 0.01:
                match = name
                break
        if match is None or match not in p:
            return False
        s = float(config.PROC_SCALE)
        pts.append([p[match][0] * s, p[match][1] * s])

    import cv2
    cap = cv2.VideoCapture(config.VIDEO_PATH)
    ok, fr = cap.read()
    cap.release()
    if not ok:
        return False
    s = float(config.PROC_SCALE)
    shape = (int(round(fr.shape[0] * s)), int(round(fr.shape[1] * s)), 3)
    court.save_corners(np.array(pts, dtype=np.float32), shape)
    return True

with open(SEGMENTS_JSON, encoding="utf-8") as fh:
    index = json.load(fh)
segments = index["segments"]

os.makedirs(OUT_DIR, exist_ok=True)
total = len(segments) * len(TRACKERS) * len(PREDICTS)
done = 0
t_all = time.time()

for seg in segments:
    video = seg["video"]
    # Windows yolu yerine Pi'deki karsiligi
    config.VIDEO_PATH = "input/{}.mp4".format(video)
    if not os.path.isfile(config.VIDEO_PATH):
        print("ATLANDI (video yok): {}".format(config.VIDEO_PATH), flush=True)
        continue
    if not install_human_corners(video):
        print("ATLANDI (insan kort GT'si yok/eslesmedi): {}".format(video), flush=True)
        continue
    config.EVAL_START_FRAME = seg["start"]
    config.EVAL_MAX_FRAMES = seg["end"] - seg["start"]

    for T in TRACKERS:
        for P in PREDICTS:
            done += 1
            out = os.path.join(OUT_DIR, "{}__{}__{}.csv".format(seg["folder"], T, P))
            if os.path.isfile(out):
                print("[{}/{}] atlandi (var): {}".format(done, total, out), flush=True)
                continue
            config.TRACKER_TYPE = T
            config.PREDICT_TYPE = P
            config.EVAL_LOG_PATH = out
            t0 = time.time()
            try:
                main.main()
                print("[{}/{}] {} {} {} -> {:.0f} sn".format(
                    done, total, seg["folder"], T, P, time.time() - t0), flush=True)
            except Exception as exc:
                print("[{}/{}] HATA {} {} {}: {}".format(
                    done, total, seg["folder"], T, P, exc), flush=True)

print("\nBITTI: {} kosu, toplam {:.1f} dk".format(total, (time.time() - t_all) / 60.0),
      flush=True)
