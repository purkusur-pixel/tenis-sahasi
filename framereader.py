"""
framereader.py
--------------
Is parcacikli (threaded) kare okuma.

NEDEN: profil, cizim kapaliyken kare suresinin ~%88'inin DECODE oldugunu gosterdi.
Decode I/O agirlikli, isleme (takip/tespit) CPU agirlikli - AYRI isler. Ardisik
yapinca toplam = decode + isleme. Decode'u arka plan thread'ine alirsak isleme ile
UST USTE biner; toplam ~ max(decode, isleme) olur. CSRT_FAST gibi ~20 ms'lik bir
takipci ile decode (~14 ms) TAMAMEN gizlenir -> takip kalitesi korunur, FPS artar.

KISIT: yalnizca ILERI-AKIS icindir (seek YOK). cap'i thread sahiplenir; seekbar ile
kullanilamaz. Headless Pi senaryosu icin. config.THREADED_READ ile acilir/kapanir.

Takibe HIC dokunmaz: yalnizca kareyi decode+resize edip hazir sunar.
"""

import queue
import threading

import cv2


class ThreadedReader:
    def __init__(self, cap, scale, interp, frame_skip=1, queue_size=4):
        """
        cap        : acilmis cv2.VideoCapture (bu andan sonra SADECE bu thread okur).
        scale      : PROC_SCALE (kare kucultme).
        interp     : cv2 interpolasyon (config.RESIZE_INTERP).
        frame_skip : FRAME_SKIP; her okunan kareden sonra (skip-1) kare atlanir (grab).
        queue_size : kac kare onden decode edilecegi (bellek/gecikme dengesi).
        """
        self.cap = cap
        self.scale = float(scale)
        self.interp = interp
        self.frame_skip = max(1, int(frame_skip))
        self.q = queue.Queue(maxsize=max(1, queue_size))
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._t.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            ok, raw = self.cap.read()
            if not ok:
                self._put(None)          # sentinel: video bitti
                break
            for _ in range(self.frame_skip - 1):
                self.cap.grab()          # atlanan kareler decode/retrieve EDILMEZ
            frame = cv2.resize(raw, None, fx=self.scale, fy=self.scale,
                               interpolation=self.interp) if self.scale != 1.0 else raw
            self._put(frame)

    def _put(self, item):
        # Kuyruk doluysa (ana dongu yavas) yer acilana kadar bekle; stop'ta cik.
        while not self._stop.is_set():
            try:
                self.q.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def read(self):
        """(ok, frame). Hazir kareyi dondurur (yoksa okuyucuyu bekler). Bitti -> (False, None)."""
        frame = self.q.get()
        if frame is None:
            return False, None
        return True, frame

    def stop(self):
        self._stop.set()
        # Thread put()'ta takili kalmasin diye kuyrugu bosalt.
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass
