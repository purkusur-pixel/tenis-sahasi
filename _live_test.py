"""
_live_test.py -- GECICI test betigi (kamera yokken canli kod yolunu dener).

  venv/bin/python _live_test.py file   -> LIVE_MODE=False, eski davranis (regresyon)
  venv/bin/python _live_test.py live   -> LIVE_MODE=True, A1.mp4 "akis gibi" okunur
                                          (geri sarma yok, arka plan acik cap'ten,
                                           kort tespiti temiz arka planda, cok denemeli)
"""
import sys

import config

config.ASK_TRACKER = False
config.ASK_PREDICT = False
config.SHOW_WINDOW = False
config.SAVE_OUTPUT = False        # test hizli olsun; video yazma olculmuyor
config.EVAL_LOG = False
config.DETECT_MACHINE = False

mode = sys.argv[1] if len(sys.argv) > 1 else "live"
video = sys.argv[2] if len(sys.argv) > 2 else "input/A1.mp4"
if len(sys.argv) > 3:                       # test icin bekleme suresi (her ikisi)
    config.SETUP_TIMEOUT_SEC = int(sys.argv[3])
    config.SETUP_CONFIRM_TIMEOUT_SEC = int(sys.argv[3])

if mode == "live":
    config.LIVE_MODE = True
    config.LIVE_SOURCE = video             # dosya yolu -> livesource onu AKIS gibi okur
    config.STREAM_ENABLED = True
    print("=== TEST: CANLI KOD YOLU (kaynak {}, akis gibi) ===".format(video))
else:
    config.LIVE_MODE = False
    config.VIDEO_PATH = video
    config.STREAM_ENABLED = False
    print("=== TEST: DOSYA KIPI (regresyon; {}) ===".format(video))

import main

main.main()
