"""Benchmark matrisi: 4 tracker x 4 tahmin modeli, ayni segmentte (adil kiyas)."""
import time
import config

# Menu sorma, video yazma (hiz); eval acik, otomatik rapor kapali (matris sonda).
config.ASK_TRACKER = False
config.ASK_PREDICT = False
config.EVAL_LOG = True
config.EVAL_AUTO_REPORT = False
config.SAVE_OUTPUT = False
config.EVAL_START_FRAME = 700
config.EVAL_MAX_FRAMES = 600
config.ZONE_PRINT = False

import main  # noqa: E402  (config ayarlandiktan sonra import edilmeli)

TRACKERS = ["MOSSE", "KCF", "CSRT_FAST", "CSRT"]
PREDICTS = ["CV", "CA", "ABG", "IMM"]

t_all = time.time()
n = 0
for T in TRACKERS:
    for P in PREDICTS:
        n += 1
        config.TRACKER_TYPE = T
        config.PREDICT_TYPE = P
        config.EVAL_LOG_PATH = "output/matrix/eval_{}_{}.csv".format(T, P)
        print("\n===== [{}/16] {} + {} =====".format(n, T, P), flush=True)
        t0 = time.time()
        try:
            main.main()
        except Exception as exc:
            print("HATA ({} + {}): {}".format(T, P, exc), flush=True)
        print("--- {} + {} bitti: {:.0f} sn".format(T, P, time.time() - t0), flush=True)

print("\nMATRIS TAMAM: {} kombinasyon, toplam {:.1f} dk".format(n, (time.time() - t_all) / 60.0),
      flush=True)
