"""
shot.py
-------
Sahne kesme (cut) tespiti.

AMAC:
    Kurgulanmis / kesintili videolarda kamera ayni kalir ama zaman atlar; oyuncu
    bir kareden digerine "isinlanir". Bu an tespit edilmezse:
      - CSRT eski konumdaki yamayi arar ve kaybolur,
      - Kalman eski hiz vektoruyle oyuncuyu yanlis yone ceker,
      - MOG2 arka plan modeli coker ve yuzlerce kare boyunca cop maske uretir,
      - MAX_JUMP_PER_FRAME denetimi, dogru olan yeni olcumu "imkansiz sicrama"
        sayip reddeder ve sistem kendini kilitler.

YONTEM:
    Gri histogram korelasyonu. Ardisik iki karenin histogramlari
    cv2.compareHist(..., HISTCMP_CORREL) ile karsilastirilir. Kesmede korelasyon
    sert duser. Karsilastirma, MOG2 icin zaten kucultulmus olan kare uzerinde
    yapilir -> ek resize yok, maliyet ihmal edilebilir (~0.1 ms).

    Histogram, kamera sabit oldugu icin fazlasiyla yeterlidir: oyuncunun hareketi
    histogrami neredeyse hic degistirmez, sahne kesmesi ise sert degistirir.
"""

import cv2

import config


def init():
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

    hist = cv2.calcHist([gray], [0], None, [config.CUT_HIST_BINS], [0, 256])
    cv2.normalize(hist, hist, 0.0, 1.0, cv2.NORM_MINMAX)

    prev = state["hist"]
    state["hist"] = hist

    if prev is None:
        return False

    # Cok sik tetiklenmeyi engelle (gecis efektleri arka arkaya dusuk skor verebilir).
    if frame_idx - state["last_cut"] < config.CUT_MIN_GAP:
        return False

    corr = cv2.compareHist(prev, hist, cv2.HISTCMP_CORREL)
    if corr < config.CUT_CORR_THRESH:
        state["last_cut"] = frame_idx
        return True

    return False
