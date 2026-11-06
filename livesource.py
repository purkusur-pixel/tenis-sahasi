"""
livesource.py
-------------
Canli goruntu kaynagi: CSI serit kablolu Pi kamera (picamera2), USB kamera /
RTSP (OpenCV) ya da test icin dosya.

NEDEN AYRI MODUL:
    main.py'nin okuma dongusu kaynagin ne oldugunu BILMEK ZORUNDA DEGIL. Bu
    modul cv2.VideoCapture ile AYNI arayuzu sunar (read / get / isOpened /
    release); boylece dongu tek satir degismeden dosya, USB kamera veya CSI
    kamera ile calisir.

NEDEN picamera2 GEREKIYOR:
    Guncel Raspberry Pi OS'ta eski V4L2 kamera arayuzu kapalidir; CSI serit
    kablolu kamera cv2.VideoCapture'a DOGRUDAN gorunmez. /dev/video* altinda
    gorunen aygitlar donanim kodekleridir, kamera degil.

NEDEN "EN YENI KARE" MANTIGI:
    Dosyada her kare islenmeliydi (offline dogruluk). Canlida onemli olan
    GECIKMEDIR: bes kare geride kalmis dogru bir konum, makineye nisan
    aldirmak icin ise yaramaz. Gercek kamera kaynaklarinda ayri bir thread
    surekli okur ve YALNIZCA EN SON kareyi tutar; ana dongu ne zaman isterse
    en tazesini alir, arada birikenler dusurulur.

DIKKAT -- SAHADA DOGRULANMADI:
    Bu dosya yazildiginda Pi'ye kamera HENUZ TAKILI DEGILDI. picamera2 yolu
    kod olarak hazirdir ama gercek donanimla denenmemistir. Diger her sey
    (dosyayi "akis gibi", geri sarmasiz okutarak) test edilmistir.
"""

import glob
import os
import sys
import threading
import time

import cv2

import config


def _enable_system_packages():
    """
    picamera2 ve libcamera'yi venv icinden gorunur kilar.

    NEDEN GEREKLI: bu ikisi SISTEM paketidir (/usr/lib/python3/dist-packages).
    libcamera baglantilari isletim sistemiyle birlikte derlenir; pip ile venv'e
    kurulamazlar. Proje venv'i --system-site-packages OLMADAN kuruldugu icin
    venv/bin/python bunlari goremiyordu -> canli kipte "No module named
    'picamera2'" ile duserdi.

    NEDEN sys.path'in SONUNA: basina eklenirse sistemdeki eski surumler
    (numpy vb.) venv'dekileri golgeler ve baska seyler bozulur. Sona eklenince
    venv paketleri oncelikli kalir; yalnizca venv'de OLMAYAN picamera2 ve
    libcamera sistemden alinir.
    """
    for pattern in ("/usr/lib/python3/dist-packages",
                    "/usr/lib/python3.*/dist-packages",
                    "/usr/lib/python3/site-packages"):
        for path in sorted(glob.glob(pattern)):
            if os.path.isdir(path) and path not in sys.path:
                sys.path.append(path)


# ---------------------------------------------------------------------------
# CSI kamera (picamera2)
# ---------------------------------------------------------------------------
class _Picamera2Source:
    def __init__(self, width, height, fps):
        # Import BURADA: picamera2 yalnizca Pi'de kurulu. PC'de test ederken
        # modulun tamami import edilemez olmasin.
        _enable_system_packages()
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise ImportError(
                "picamera2 bulunamadi ({}). Pi'de sistem paketi olarak kurulu "
                "olmali: sudo apt install -y python3-picamera2".format(exc))

        self._picam = Picamera2()
        cfg = self._picam.create_video_configuration(
            main={"size": (int(width), int(height)), "format": "RGB888"},
            controls={"FrameRate": float(fps)},
        )
        self._picam.configure(cfg)
        self._picam.start()
        # Sensor pozlama/beyaz dengesi otursun; ilk kareler karanlik/renksiz gelir
        # ve arka plan modeli tam bu kareler uzerine kurulacagi icin onemli.
        time.sleep(float(config.LIVE_WARMUP_SEC))
        self._opened = True
        self._fps = float(fps)

    def read(self):
        arr = self._picam.capture_array()
        if arr is None:
            return False, None
        # picamera2'de "RGB888" isimlendirmesi yaniltici: dizi pratikte BGR
        # duzeninde gelir, yani OpenCV'nin bekledigi sira. Donanimla
        # dogrulanana kadar ters cikma ihtimaline karsi bayrak birakildi.
        if config.LIVE_BGR_SWAP:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return True, arr

    def isOpened(self):
        return self._opened

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return self._fps
        return 0.0

    def release(self):
        if self._opened:
            try:
                self._picam.stop()
                self._picam.close()
            except Exception:
                pass
            self._opened = False


# ---------------------------------------------------------------------------
# OpenCV kaynagi (USB kamera / RTSP / dosya)
# ---------------------------------------------------------------------------
class _CvSource:
    def __init__(self, src, width=None, height=None, fps=None, live=True):
        self._cap = cv2.VideoCapture(src)
        if live and self._cap.isOpened():
            # MJPG'yi COZUNURLUKTEN ONCE iste. NEDEN: USB (UVC) kameralar
            # varsayilan olarak sikistirilmamis YUYV verir; 1280x720'de USB 2.0
            # bant genisligi buna ancak ~5 FPS yeter ve kamera sessizce oraya
            # duser. MJPG sikistirilmis oldugu icin ayni kabloda 30 FPS cikar.
            # Kamera MJPG desteklemiyorsa cagri sessizce yok sayilir, zarari yok.
            try:
                self._cap.set(cv2.CAP_PROP_FOURCC,
                              cv2.VideoWriter_fourcc(*"MJPG"))
            except Exception:
                pass
            # Surucu tamponunu 1'e cek: bekleyen kare birikmesin. Her surucu
            # desteklemez; desteklemeyende sessizce yok sayilir, bu yuzden
            # asagidaki "en yeni kare" thread'i asil emniyettir.
            try:
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            if width:
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
            if height:
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
            if fps:
                self._cap.set(cv2.CAP_PROP_FPS, float(fps))

    def read(self):
        return self._cap.read()

    def isOpened(self):
        return self._cap.isOpened()

    def get(self, prop):
        return self._cap.get(prop)

    def release(self):
        self._cap.release()


# ---------------------------------------------------------------------------
# "En yeni kare" sarmalayicisi
# ---------------------------------------------------------------------------
class LatestFrame:
    """
    Kaynagi ayri bir thread'de surekli okur, YALNIZCA en son kareyi tutar.

    Bu, framereader.ThreadedReader'dan farklidir: orada amac decode'u isleme
    ile ust uste bindirip verimi artirmakti ve KUYRUK vardi (hicbir kare
    kaybolmaz). Burada amac tam tersi -- eski kareleri BILEREK dusurup
    gecikmeyi sabit tutmak.

    NOT: yalnizca gercek canli kaynaklarda kullanilir. Dosyada kullanilirsa
    kareler atlanir (istenmez) ve ayrica bu Pi'de dosya decode'unda thread'li
    okuma ffmpeg "async_lock" cokmesi veriyordu (bkz. config.THREADED_READ).
    """

    def __init__(self, source):
        self._src = source
        self._lock = threading.Lock()
        self._frame = None
        self._seq = 0            # kare sayaci: ayni kareyi iki kez islememek icin
        self._last_seq = -1
        self._run = False
        self._thread = None
        self._err = None

    def start(self):
        self._run = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        # Ilk kare gelene kadar bekle ki cagiran hemen None ile karsilasmasin.
        t0 = time.time()
        while time.time() - t0 < float(config.LIVE_FIRST_FRAME_TIMEOUT):
            with self._lock:
                if self._frame is not None:
                    return self
            if self._err is not None:
                break
            time.sleep(0.01)
        return self

    def _loop(self):
        while self._run:
            try:
                ok, frame = self._src.read()
            except Exception as exc:          # surucu hatasi dongu yi oldurmesin
                self._err = exc
                time.sleep(0.05)
                continue
            if not ok or frame is None:
                # Canli kaynakta gecici okuma hatasi olagandir (usb sarsintisi,
                # ag paketi kaybi). Akisi bitmis SAYMA -- kisa bekleyip devam et.
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame
                self._seq += 1

    def read(self):
        """Bekleyen en yeni kareyi verir. Ayni kare iki kez verilmez."""
        deadline = time.time() + float(config.LIVE_READ_TIMEOUT)
        while True:
            with self._lock:
                if self._frame is not None and self._seq != self._last_seq:
                    self._last_seq = self._seq
                    return True, self._frame
            if time.time() > deadline:
                return False, None
            time.sleep(0.002)

    def isOpened(self):
        return self._src.isOpened()

    def get(self, prop):
        return self._src.get(prop)

    def release(self):
        self._run = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._src.release()


# ---------------------------------------------------------------------------
def open_source():
    """
    config'e gore uygun kaynagi acar.

    Donus: cv2.VideoCapture benzeri nesne (read/get/isOpened/release).
           Acilamazsa None.
    """
    src = config.LIVE_SOURCE

    if isinstance(src, str) and src.lower() == "picamera2":
        base = _Picamera2Source(config.LIVE_WIDTH, config.LIVE_HEIGHT, config.LIVE_FPS)
        live = True
    else:
        # int -> USB kamera indeksi, str -> RTSP adresi veya dosya yolu.
        # Dosya yolu TEST icindir: kamera yokken canli kod yolunu denemeye yarar.
        is_file = isinstance(src, str) and not src.lower().startswith(
            ("rtsp://", "http://", "https://", "udp://", "tcp://"))
        base = _CvSource(src, config.LIVE_WIDTH, config.LIVE_HEIGHT,
                         config.LIVE_FPS, live=not is_file)
        live = not is_file

    if not base.isOpened():
        return None

    # "En yeni kare" thread'i yalnizca GERCEK canli kaynakta. Dosyada kare
    # atlamak istenmez (test tekrarlanabilir olmali).
    if live and config.LIVE_DROP_STALE:
        return LatestFrame(base).start()
    return base
