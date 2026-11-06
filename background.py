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


def build(video_path, scale, n_samples=None, verbose=True, cap=None):
    """
    Videodan yayilmis kareler ornekleyip medyan arka plani uretir.

    video_path : config.VIDEO_PATH (dosya yolu veya webcam indeksi)
    scale      : PROC_SCALE - boru hattiyla AYNI olcek kullanilmalidir.
    cap        : ZATEN ACIK bir cv2.VideoCapture (istege bagli).

    NEDEN cap PARAMETRESI VAR: canli kamerada aygit IKINCI KEZ ACILAMAZ. Boru
    hatti kamerayi main.py'de zaten acmis durumdadir; burada tekrar acmaya
    calismak ya hata verir ya da akisi calar -- canli kipte calismayi tumden
    engelleyen sorun buydu. cap verilirse ondan SIRAYLA okunur ve cap
    KAPATILMAZ (sahibi cagirandir); verilmezse eski dosya davranisi aynen kalir.

    Donus: arka plan karesi (BGR, islenmis olcekte) veya kurulamazsa None.
    """
    if n_samples is None:
        n_samples = config.BG_SAMPLES

    owned = cap is None            # cap'i biz actiysak kapatmak da bizim isimiz
    if owned:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

    # Geri sarma (POS_FRAMES) YALNIZCA kendi actigimiz dosyada guvenli: odunc
    # alinan cap ya canlidir (seek desteklemez) ya da ana dongunun okuma
    # konumunu tasir -- ikisinde de geri sarmak yanlis olur.
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if owned else 0
    frames = []

    if owned and total > n_samples:
        # Dosya: videonun TAMAMINA yayilmis kareler (oyuncu her seferinde baska yerde).
        for idx in np.linspace(0, total - 1, n_samples).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, raw = cap.read()
            if ok:
                frames.append(_scaled(raw, scale))
    else:
        # Canli / odunc cap / kare sayisi bilinmiyor: ardisik kareleri atlayarak topla.
        grabbed = 0
        while len(frames) < n_samples and grabbed < n_samples * config.BG_STRIDE:
            ok, raw = cap.read()
            if not ok:
                break
            if grabbed % config.BG_STRIDE == 0:
                frames.append(_scaled(raw, scale))
            grabbed += 1

    if owned:
        cap.release()

    if len(frames) < 3:
        return None

    bg = np.median(np.stack(frames), axis=0).astype(np.uint8)
    if verbose:
        print("Arka plan modeli: {} kare medyani.".format(len(frames)))
    return bg


def refresh(bg, frame, alpha, exclude_box=None):
    """
    Arka plan modelini, oyuncunun BULUNMADIGI yerlerde yavasca gunceller.

    NEDEN GEREKLI (yalnizca canli kipte): dosyada medyan, videonun TAMAMINDAN
    alindigi icin model bastan dogruydu ve eskimezdi. Canli akista ise model
    yalnizca acilistaki birkac saniyeden kurulur; sonra isik degisir (bulut,
    aksamustu), golgeler kayar. Model eskidikce her yerde sahte on plan uretir
    ve tespit bozulur.

    NEDEN OYUNCUNUN KUTUSU DISLANIYOR: oyuncunun uzerinden gecen pikselleri de
    guncellersek oyuncu yavas yavas arka plana KARISIR ve gorunmez olur -- yani
    tam da onlemeye calistigimiz sey (MOG2'yi bu yuzden kullanmiyoruz, bkz.
    dosya basi). exclude_box verildiginde o dikdortgen oldugu gibi korunur.

    alpha       : 0..1 karisim orani. Kucuk tut (or. 0.02) -- yavas ve kararli.
    exclude_box : (x, y, w, h) islenmis olcekte; None ise tum kare guncellenir.

    Donus: guncellenmis arka plan (yeni dizi). Girdi bozulmaz.
    """
    if bg is None or frame is None or alpha <= 0.0:
        return bg
    if frame.shape != bg.shape:
        return bg          # olcek uyusmazligi -> sessizce dokunma (sahte fg uretmesin)

    out = cv2.addWeighted(frame, float(alpha), bg, 1.0 - float(alpha), 0.0)

    if exclude_box is not None:
        x, y, bw, bh = [int(v) for v in exclude_box]
        h, w = bg.shape[:2]
        # Kutuyu bir miktar genislet: takip kutusu oyuncuyu tam sarmalayabilir,
        # kenardaki golge/kol pikselleri disarida kalirsa yine modele sizar.
        pad = int(config.BG_REFRESH_PAD)
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
        if x1 > x0 and y1 > y0:
            out[y0:y1, x0:x1] = bg[y0:y1, x0:x1]

    return out


def _scaled(frame, scale):
    if scale == 1.0:
        return frame
    # main.py ile AYNI interpolasyon (config.RESIZE_INTERP) kullanilmali; farkli
    # olursa arka plan modeli ile kareler uyusmaz ve sahte foreground olusur.
    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=config.RESIZE_INTERP)
