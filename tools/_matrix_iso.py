"""
A/B testi: ANIZOTROPIK derinlik damping'i KAPALI (izotropik olcum gurultusu).
Hipotez: KF_MEASUREMENT_NOISE_Y (=1.0) derinligi "yapiskan" yaptigi icin filtrelenmis
konum ham tracker kutusundan DAHA KOTU cikiyordu (FILT_px > AYAK_px). Izotropik
yapinca filtrelerin gercekten iyilestirip iyilestirmedigini olcuyoruz.
Ayni segment (700..1300) -> onceki matrisle birebir kiyaslanabilir.
"""
import time
import config

config.ASK_TRACKER = False
config.ASK_PREDICT = False
config.EVAL_LOG = True
config.EVAL_AUTO_REPORT = False
config.SAVE_OUTPUT = False
config.EVAL_START_FRAME = 700
config.EVAL_MAX_FRAMES = 600
config.ZONE_PRINT = False

# >>> TEK FARK: derinlik damping KAPALI (y olcum gurultusu = x ile ayni) <<<
config.KF_MEASUREMENT_NOISE_Y = config.KF_MEASUREMENT_NOISE

import main  # noqa: E402

TRACKERS = ["MOSSE", "KCF"]          # en hizli + en dogru (sure icin 2 tracker)
PREDICTS = ["CV", "CA", "ABG", "IMM"]

t_all = time.time()
n = 0
for T in TRACKERS:
    for P in PREDICTS:
        n += 1
        config.TRACKER_TYPE = T
        config.PREDICT_TYPE = P
        config.EVAL_LOG_PATH = "output/matrix_iso/eval_{}_{}.csv".format(T, P)
        print("\n===== [{}/8] {} + {} (izotropik) =====".format(n, T, P), flush=True)
        t0 = time.time()
        try:
            main.main()
        except Exception as exc:
            print("HATA ({} + {}): {}".format(T, P, exc), flush=True)
        print("--- {} + {} bitti: {:.0f} sn".format(T, P, time.time() - t0), flush=True)

print("\nIZOTROPIK MATRIS TAMAM: {} kombinasyon, {:.1f} dk".format(
    n, (time.time() - t_all) / 60.0), flush=True)
