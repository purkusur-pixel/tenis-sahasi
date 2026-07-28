"""
performance.py
--------------
Headless performans olcumu ve asama profillemesi.

TASARIM: tum olcum mantigi burada; main.py yalnizca bu sinifi kullanir. Yalnizca
standart kutuphane (time, platform, collections) -> Raspberry Pi ve Windows'ta
BIREBIR ayni calisir, iki cihazin FPS/sure degerleri dogrudan karsilastirilabilir.

OLCUM: time.perf_counter() (monotonik, yuksek cozunurluk).

RAPOR: her karede DEGIL, saniyede BIR kez basilir (terminal dolmasin). Ayrica
program sonunda toplu ozet basilir.

KULLANIM:
    perf = Performance(["video_okuma", "takip", ...])
    # tek seferlik kurulum olcumu:
    with perf.measure_setup("court_detection"):
        corners = court.auto_detect_corners(frame)
    # her kare:
    perf.frame_begin()
    perf.start("video_okuma"); ...; perf.stop("video_okuma")
    with perf.measure("takip"):
        tracker.update(frame)
    perf.frame_end()          # rapor gerekiyorsa burada basilir
    ...
    perf.summary()            # program sonunda
"""

import platform
import time
from collections import deque


class _Timer:
    """with perf.measure('takip'): ...  -> sureyi asamaya ekler."""
    __slots__ = ("_cb", "_name", "_t")

    def __init__(self, cb, name):
        self._cb = cb
        self._name = name

    def __enter__(self):
        self._t = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self._cb(self._name, time.perf_counter() - self._t)


class Performance:
    def __init__(self, stages, report_interval=1.0, avg_window=30,
                 enabled=True, label=None):
        """
        stages         : rapor sirasinda gosterilecek asama adlari (liste).
        report_interval: kac saniyede bir canli rapor basilacagi.
        avg_window     : hareketli ortalama FPS penceresi (kare sayisi).
        enabled        : False -> tum cagrilar no-op (sifir ek yuk).
        label          : rapor basligina yazilacak ad (or. "Windows" / "Pi").
        """
        self.enabled = bool(enabled)
        self.stages = list(stages)
        self.report_interval = float(report_interval)
        self.avg_window = int(avg_window)
        self.label = label

        self.t_start = time.perf_counter()
        self.t_last_report = self.t_start
        # loop_start ILK karede kurulur: FPS/gecen sure yalnizca DONGU'yu kapsar,
        # kurulum (arka plan modeli, kose yukleme - saniyeler surebilir) HARIC kalir.
        self.loop_start = None
        self.frames = 0
        self.frame_times = deque(maxlen=self.avg_window)   # son N kare TOPLAM suresi (s)
        self.cum = {s: 0.0 for s in self.stages}           # asama -> kumulatif sure (s)
        self.cur = {s: 0.0 for s in self.stages}           # bu karedeki asama suresi (s)
        self.setup = {}                                    # tek seferlik kurulum olcumleri
        self._open = {}                                    # start()/stop() acik olcumler
        self._t_frame = None
        self._last_dt = 0.0

    # ---------------------------------------------------------------- kurulum
    def measure_setup(self, name):
        """Tek seferlik (kurulum) olcumu icin context manager."""
        return _Timer(self._add_setup, name)

    def start_setup(self, name):
        """Tek seferlik kurulum bloklarini yeniden girintilemeden olcmek icin."""
        if self.enabled:
            self._open["setup:" + name] = time.perf_counter()

    def stop_setup(self, name):
        key = "setup:" + name
        if self.enabled and key in self._open:
            self._add_setup(name, time.perf_counter() - self._open.pop(key))

    def _add_setup(self, name, dt):
        if self.enabled:
            self.setup[name] = self.setup.get(name, 0.0) + dt

    # ------------------------------------------------------------------ kare
    def frame_begin(self):
        """Yeni karenin basi. Asama sayaclarini sifirlar, kare kronometresini baslatir."""
        if not self.enabled:
            return
        for s in self.cur:
            self.cur[s] = 0.0
        self._open.clear()
        self._t_frame = time.perf_counter()
        if self.loop_start is None:
            self.loop_start = self._t_frame

    def measure(self, name):
        """with perf.measure('takip'): ...  -> tek cagrilik asamalar icin."""
        return _Timer(self._add, name)

    def start(self, name):
        """Buyuk/dagitik bloklari yeniden girintilemeden olcmek icin (stop ile eslesir)."""
        if self.enabled:
            self._open[name] = time.perf_counter()

    def stop(self, name):
        if self.enabled and name in self._open:
            self._add(name, time.perf_counter() - self._open.pop(name))

    def _add(self, name, dt):
        if not self.enabled:
            return
        if name not in self.cum:          # bilinmeyen asama -> dinamik ekle
            self.cum[name] = 0.0
            self.cur[name] = 0.0
            self.stages.append(name)
        self.cum[name] += dt
        self.cur[name] += dt

    def frame_end(self):
        """Kareyi kapatir; FPS gunceller; gerekiyorsa (saniyede bir) canli rapor basar."""
        if not self.enabled:
            return
        now = time.perf_counter()
        self._last_dt = now - self._t_frame if self._t_frame else 0.0
        self.frame_times.append(self._last_dt)
        self.frames += 1
        if now - self.t_last_report >= self.report_interval:
            self._live_report(now)
            self.t_last_report = now

    # --------------------------------------------------------------- raporlar
    def _fps(self):
        inst = 1.0 / self._last_dt if self._last_dt > 0 else 0.0
        avg = (len(self.frame_times) / sum(self.frame_times)) if self.frame_times else 0.0
        return inst, avg

    def _live_report(self, now):
        inst, avg = self._fps()
        elapsed = now - (self.loop_start or now)   # dongu suresi (kurulum haric)
        parts = " | ".join("{} {:.1f}".format(s, self.cur[s] * 1000.0) for s in self.stages)
        print("[PERF] t={:5.1f}s kare={:5d} | FPS anlik={:5.1f} ort{}={:5.1f} | "
              "kare={:6.2f} ms  ({})".format(
                  elapsed, self.frames, inst, self.avg_window, avg,
                  self._last_dt * 1000.0, parts), flush=True)

    def summary(self):
        """Program sonunda toplu ozet. Iki cihazda karsilastirma icin ana cikti."""
        if not self.enabled:
            return
        # Gecen sure = DONGU suresi (ilk kareden simdiye); kurulum HARIC.
        elapsed = time.perf_counter() - (self.loop_start or time.perf_counter())
        overall_fps = self.frames / elapsed if elapsed > 0 else 0.0

        line = "=" * 64
        print("\n" + line, flush=True)
        print("PERFORMANS OZETI" + ("  [{}]".format(self.label) if self.label else ""), flush=True)
        print(line, flush=True)
        print("Makine        : {}  ({})".format(platform.node(), platform.machine()), flush=True)
        print("Islemci       : {}".format(platform.processor() or "bilinmiyor"), flush=True)
        print("Python        : {}".format(platform.python_version()), flush=True)
        print("Islenen kare  : {}".format(self.frames), flush=True)
        print("Gecen sure    : {:.2f} sn".format(elapsed), flush=True)
        print("Ortalama FPS  : {:.2f}".format(overall_fps), flush=True)

        if self.setup:
            print("-" * 64, flush=True)
            print("KURULUM (tek seferlik):", flush=True)
            for name, dt in self.setup.items():
                print("  {:<22}{:>9.1f} ms".format(name, dt * 1000.0), flush=True)

        if self.frames > 0:
            print("-" * 64, flush=True)
            print("{:<22}{:>12}{:>12}{:>9}".format(
                "KARE ASAMASI", "toplam ms", "ms/kare", "%"), flush=True)
            stage_total = sum(self.cum.values())
            for s in self.stages:
                tot = self.cum[s] * 1000.0
                per = self.cum[s] / self.frames * 1000.0
                pct = (self.cum[s] / stage_total * 100.0) if stage_total > 0 else 0.0
                print("  {:<20}{:>12.1f}{:>12.2f}{:>8.1f}%".format(s, tot, per, pct), flush=True)
            print("  {:<20}{:>12.1f}{:>12.2f}".format(
                "TOPLAM (asamalar)", stage_total * 1000.0,
                stage_total / self.frames * 1000.0), flush=True)
        print(line, flush=True)
