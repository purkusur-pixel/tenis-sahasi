"""
background.py
-------------
Sabit kamera icin medyan arka plan modeli.

NEDEN MEDYAN:
    Kamera sabit ve kortta TEK oyuncu var. Videonun tamamina yayilmis N kare
    ustunde piksel bazinda medyan alinca oyuncu tamamen silinir: her piksel,
    zamanin cok buyuk cogunlugunda arka plandir; oyuncu o pikselin uzerinden
    yalnizca kisa bir sure gecer, yani medyani etkileyemeyecek kadar azinliktadir.
    Sonuc: oyuncusuz, keskin, gurultusuz bir kort goruntusu.

NEDEN MOG2 DEGIL:
    MOG2 uyarlanir bir modeldir: duran oyuncu birkac saniye icinde arka plana
    karisir ve kaybolur. Ayrica isinma (warmup) ister ve sahne kesmesinde coker.
    Medyan model ise videonun tamamindan BIR KEZ kurulur, duragan oyuncuyu de
    gorur ve zamanla kaymaz.

NEDEN RENK MODELI DEGIL:
    detector.py eskiden kort renginin LAB medyanina uzakliga bakiyordu. Kortun
    disindaki kirmizi run-off alani da kort renginden uzak oldugu icin maskenin
    %31'ini dolduruyor, oyuncu bu dev blob'un icinde kayboluyordu (tespit 0/20).
    Arka plan cikarma her pikseli KENDI gecmisiyle karsilastirdigi icin bu sorun
    yapisal olarak ortadan kalkar.
"""

import numpy as np
import cv2

import config


def build(video_path, scale, n_samples=None, verbose=True):
    """
    Videodan yayilmis kareler ornekleyip medyan arka plani uretir.

    video_path : config.VIDEO_PATH (dosya yolu veya webcam indeksi)
    scale      : PROC_SCALE - boru hattiyla AYNI olcek kullanilmalidir.

    Donus: arka plan karesi (BGR, islenmis olcekte) veya kurulamazsa None.
    """
    if n_samples is None:
        n_samples = config.BG_SAMPLES

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
        while len(frames) < n_samples and grabbed < n_samples * config.BG_STRIDE:
            ok, raw = cap.read()
            if not ok:
                break
            if grabbed % config.BG_STRIDE == 0:
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
    # main.py ile AYNI interpolasyon (config.RESIZE_INTERP) kullanilmali; farkli
    # olursa arka plan modeli ile kareler uyusmaz ve sahte foreground olusur.
    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=config.RESIZE_INTERP)
