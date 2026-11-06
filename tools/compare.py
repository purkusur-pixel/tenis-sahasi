"""
compare.py  --  TRACKER PERFORMANCE SUMMARY
--------------------------------------------------------------------------------
YOLO referansi (yolo_reference.py) ile Pi tracker'inin eval CSV'sini (main.py
EVAL_LOG) kare-kare kiyaslar ve profesyonel bir OZET RAPOR basar.

Kullanim:
    python compare.py <referans.csv> <tracker1.csv> [tracker2.csv ...]
    ornek: python compare.py referans_A1.csv eval_csrtfast.csv eval_mosse.csv

VERI KAYNAGI: her iki CSV de NORMALIZE (0..1) kutu + ayak noktasi icerir; olcek/
cozunurluk farki bu sayede yok olur. Tracker CSV'sinin basindaki '#' meta satirlari
(video, tracker, predict, frames, avg_fps, elapsed_sec) "Genel Bilgiler" bolumunu
besler -- AYRI bir meta dosyasi olusturulmaz.

METRIKLER (hepsi GORUNTU uzayinda -> uzak-yari derinlik sorunundan bagimsiz):
    IoU            : kutu ortusmesi (0..1)
    Foot error     : referans ayak <-> tracker ayak mesafesi (piksel @kaynak coz.)
    Coverage       : YOLO oyuncuyu bulmusken tracker'in da konum verdigi kare orani
    Lost / Re-acq  : takibin koptugu kare sayisi / yeniden yakalama sayisi
"""
import csv
import statistics as st
import sys

# Renk (opsiyonel): colorama varsa kullanilir, yoksa duz metne duser.
try:
    from colorama import Fore, Style, init as _color_init
    _color_init()
    C_TITLE, C_KEY, C_VAL, C_OK, C_WARN, C_BAD, C_DIM, C_RESET = (
        Fore.CYAN + Style.BRIGHT, Fore.WHITE, Fore.YELLOW + Style.BRIGHT,
        Fore.GREEN + Style.BRIGHT, Fore.YELLOW, Fore.RED + Style.BRIGHT,
        Style.DIM, Style.RESET_ALL)
except ImportError:                                    # colorama yoksa renksiz calis
    C_TITLE = C_KEY = C_VAL = C_OK = C_WARN = C_BAD = C_DIM = C_RESET = ""

SRC_W, SRC_H = 1920, 1080      # kaynak cozunurluk -> foot error'u piksele cevirmek icin
W = 62                         # rapor genisligi


# =============================================================================
# 1) VERI YUKLEME
# =============================================================================
def load(path):
    """
    CSV -> (frames dict, meta dict).
    frames: {kare_no: ((x1,y1,x2,y2), (fx,fy))} ; konum yoksa None.
    meta  : '#' ile baslayan satirlardan (key=value) toplanir.
    """
    frames, meta = {}, {}
    rows = []
    with open(path, newline="") as f:
        for line in f:
            if line.startswith("#"):
                kv = line[1:].strip()
                for part in kv.split():
                    if "=" in part:
                        k, v = part.split("=", 1)
                        meta[k] = v
            else:
                rows.append(line)

    reader = csv.DictReader(rows)
    for row in reader:
        try:
            fr = int(row["frame"])
        except (KeyError, ValueError):
            continue

        def g(k):
            v = row.get(k, "")
            return float(v) if v not in ("", None) else None

        if g("x1") is None:
            frames[fr] = None
        else:
            # 3. eleman: FILTRELENMIS konum (px,py) -- yalnizca tracker CSV'sinde
            # bulunur; tahmin modellerinin (CV/CA/ABG/IMM) kalitesini olcer.
            filt = (g("px"), g("py")) if g("px") is not None else None
            frames[fr] = ((g("x1"), g("y1"), g("x2"), g("y2")), (g("fx"), g("fy")), filt)
    return frames, meta


# =============================================================================
# 2) METRIKLER
# =============================================================================
def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def evaluate(ref, trk):
    """Referans ve tracker karelerini hizalayip tum metrikleri hesaplar."""
    # SEGMENT DUZELTMESI: tracker yalnizca bir kare araligini islediyse (benchmark
    # segmenti), referansi O ARALIGA kirp. Aksi halde kapsama, islenmeyen kareler
    # yuzunden yapay olarak dusuk cikar (or. 600/1388 = %43).
    if trk:
        lo, hi = min(trk), max(trk)
        ref = {f: v for f, v in ref.items() if lo <= f <= hi}

    total = len(ref)
    ref_have = sum(1 for v in ref.values() if v is not None)

    ious, foot, ffoot = [], [], []
    both = 0
    lost = 0            # YOLO oyuncuyu gormus ama tracker konum verememis
    reacq = 0           # kayip -> tekrar yakalama gecisi
    prev_lost = False

    for fr in sorted(ref):
        rv = ref[fr]
        if rv is None:
            continue
        tv = trk.get(fr)
        if tv is None:
            lost += 1
            prev_lost = True
            continue
        if prev_lost:
            reacq += 1
            prev_lost = False
        both += 1
        ious.append(iou(rv[0], tv[0]))
        dx = (rv[1][0] - tv[1][0]) * SRC_W
        dy = (rv[1][1] - tv[1][1]) * SRC_H
        foot.append((dx * dx + dy * dy) ** 0.5)
        # FILTRELENMIS konum hatasi -> tahmin modelinin (CV/CA/ABG/IMM) kalitesi
        if len(tv) > 2 and tv[2] is not None:
            fdx = (rv[1][0] - tv[2][0]) * SRC_W
            fdy = (rv[1][1] - tv[2][1]) * SRC_H
            ffoot.append((fdx * fdx + fdy * fdy) ** 0.5)

    m = {
        "total": total, "ref_have": ref_have, "both": both,
        "det_rate": 100.0 * ref_have / max(total, 1),
        "coverage": 100.0 * both / max(ref_have, 1),
        "lost": lost, "reacq": reacq,
    }
    if ious:
        m.update({
            "iou_mean": st.mean(ious), "iou_med": st.median(ious),
            "s50": 100.0 * sum(1 for i in ious if i > 0.50) / len(ious),
            "s75": 100.0 * sum(1 for i in ious if i > 0.75) / len(ious),
            "foot_mean": st.mean(foot), "foot_max": max(foot), "foot_med": st.median(foot),
            "filt_mean": st.mean(ffoot) if ffoot else 0.0,
        })
    else:
        m.update({"iou_mean": 0, "iou_med": 0, "s50": 0, "s75": 0,
                  "foot_mean": 0, "foot_max": 0, "foot_med": 0, "filt_mean": 0})

    # GENEL BASARI (seffaf formul; agirliklar raporda gosterilir):
    #   %50 ortalama IoU  +  %30 IoU>0.5 basari orani  +  %20 kapsama
    m["overall"] = 50.0 * m["iou_mean"] + 0.30 * m["s50"] + 0.20 * m["coverage"]
    m["status"] = ("Excellent" if m["overall"] >= 90 else
                   "Very Good" if m["overall"] >= 80 else
                   "Good" if m["overall"] >= 70 else
                   "Needs Improvement")
    return m


# =============================================================================
# 3) RAPOR CIZIMI
# =============================================================================
def _line(ch="-"):
    print(C_DIM + ch * W + C_RESET)


def _head(title):
    print(C_TITLE + "+" + "-" * (W - 2) + "+" + C_RESET)
    print(C_TITLE + "|" + title.center(W - 2) + "|" + C_RESET)
    print(C_TITLE + "+" + "-" * (W - 2) + "+" + C_RESET)


def _section(title):
    print()
    print(C_TITLE + title + C_RESET)
    _line()


def _row(key, value, color=None):
    col = color if color is not None else C_VAL
    print("  {}{:<34}{} {}{}{}".format(C_KEY, key, C_RESET, col, value, C_RESET))


def report(name, meta, m):
    _head("TRACKER PERFORMANCE SUMMARY")

    _section("1) GENEL BILGILER")
    _row("Video", meta.get("video", "-"))
    _row("Islenen kare", "{}".format(meta.get("frames", m["total"])))
    _row("Tracker", meta.get("tracker", name))
    _row("Tahmin modeli", meta.get("predict", "-"))
    _row("Ortalama FPS (isleme)", meta.get("avg_fps", "-"))
    _row("Kaynak FPS (video)", meta.get("src_fps", "-"))
    el = meta.get("elapsed_sec")
    _row("Islem suresi", "{} sn".format(el) if el else "-")

    _section("2) YOLO REFERANS KARSILASTIRMASI")
    _row("YOLO detection rate", "%{:.1f}   ({}/{} kare)".format(
        m["det_rate"], m["ref_have"], m["total"]))
    _row("Tracker coverage", "%{:.1f}   ({}/{} kare)".format(
        m["coverage"], m["both"], m["ref_have"]),
        C_OK if m["coverage"] >= 90 else C_WARN if m["coverage"] >= 70 else C_BAD)
    _row("Ortalama IoU", "{:.3f}".format(m["iou_mean"]),
         C_OK if m["iou_mean"] >= 0.7 else C_WARN if m["iou_mean"] >= 0.5 else C_BAD)
    _row("Medyan IoU", "{:.3f}".format(m["iou_med"]))
    _row("IoU > 0.50 kare orani", "%{:.1f}".format(m["s50"]),
         C_OK if m["s50"] >= 80 else C_WARN if m["s50"] >= 50 else C_BAD)
    _row("IoU > 0.75 kare orani", "%{:.1f}".format(m["s75"]))
    _row("Ortalama foot error", "{:.1f} px  (medyan {:.1f})".format(
        m["foot_mean"], m["foot_med"]),
        C_OK if m["foot_mean"] <= 20 else C_WARN if m["foot_mean"] <= 50 else C_BAD)
    _row("Maksimum foot error", "{:.1f} px".format(m["foot_max"]))
    _row("Kaybedilen kare", "{}".format(m["lost"]),
         C_OK if m["lost"] == 0 else C_WARN)
    _row("Yeniden yakalama", "{}".format(m["reacq"]))

    _section("3) GENEL BASARI")
    col = (C_OK if m["overall"] >= 80 else C_WARN if m["overall"] >= 70 else C_BAD)
    print()
    print(col + "   Overall Tracker Accuracy : {:.1f} %".format(m["overall"]) + C_RESET)
    print(col + "   Status                   : {}".format(m["status"]) + C_RESET)
    print()
    print(C_DIM + "   (skor = 0.50*ort.IoU + 0.30*IoU>0.5 orani + 0.20*kapsama)" + C_RESET)
    _line("=")


def matrix_table(rows):
    """
    TEK SAYFA benchmark matrisi: her (tracker x tahmin modeli) kombinasyonu icin
    dogruluk + isleme FPS. rows: [(meta, metrics), ...]
    """
    MW = 78
    print(C_TITLE + "=" * MW + C_RESET)
    print(C_TITLE + "TRACKER x FILTRE BENCHMARK MATRISI  (referans: YOLOv8s)".center(MW) + C_RESET)
    print(C_TITLE + "=" * MW + C_RESET)

    v = rows[0][0].get("video", "-") if rows else "-"
    fr = rows[0][0].get("frames", "-") if rows else "-"
    print(C_DIM + "  video: {}   |   kare: {}   |   metrikler goruntu uzayinda".format(v, fr) + C_RESET)
    print(C_DIM + "-" * MW + C_RESET)
    print("  {}{:<11}{:<7}{:>7}{:>8}{:>8}{:>9}{:>9}{:>8}{}".format(
        C_KEY, "TRACKER", "FILTRE", "IoU", "IoU>.5", "KAPSM",
        "AYAK_px", "FILT_px", "FPS", C_RESET))
    print(C_DIM + "-" * MW + C_RESET)

    ordered = sorted(rows, key=lambda r: -r[1]["overall"])
    best = ordered[0][1]["overall"] if ordered else 0
    fastest = max((float(r[0].get("avg_fps", 0) or 0) for r in rows), default=0)

    for meta, m in ordered:
        trk = meta.get("tracker", "?")
        pred = meta.get("predict", "?")
        fps = float(meta.get("avg_fps", 0) or 0)
        col = C_OK if m["overall"] == best else C_VAL
        fps_mark = "*" if fps == fastest and fps > 0 else " "
        print("  {}{:<11}{:<7}{:>7.3f}{:>7.1f}%{:>7.1f}%{:>9.1f}{:>9.1f}{:>7.1f}{}{}".format(
            col, trk, pred, m["iou_mean"], m["s50"], m["coverage"],
            m["foot_mean"], m["filt_mean"], fps, fps_mark, C_RESET))

    print(C_DIM + "-" * MW + C_RESET)
    print(C_DIM + "  AYAK_px = ham tracker kutusu hatasi | FILT_px = filtrelenmis konum hatasi" + C_RESET)
    print(C_DIM + "  (*) en yuksek isleme FPS | yesil satir = en yuksek genel skor" + C_RESET)

    # En iyiler
    if ordered:
        bt, bm = ordered[0]
        print()
        print(C_OK + "  EN DOGRU : {} + {}  ->  IoU {:.3f} | skor {:.1f}% | {:.1f} FPS".format(
            bt.get("tracker", "?"), bt.get("predict", "?"),
            bm["iou_mean"], bm["overall"], float(bt.get("avg_fps", 0) or 0)) + C_RESET)
        fr_ = max(rows, key=lambda r: float(r[0].get("avg_fps", 0) or 0))
        print(C_OK + "  EN HIZLI : {} + {}  ->  {:.1f} FPS | skor {:.1f}%".format(
            fr_[0].get("tracker", "?"), fr_[0].get("predict", "?"),
            float(fr_[0].get("avg_fps", 0) or 0), fr_[1]["overall"]) + C_RESET)
        # filtre bazinda ortalama (hangi filtre genelde daha iyi?)
        by_pred = {}
        for meta, m in rows:
            by_pred.setdefault(meta.get("predict", "?"), []).append(m["filt_mean"])
        if len(by_pred) > 1:
            print()
            print(C_KEY + "  FILTRE ORTALAMALARI (dusuk = iyi):" + C_RESET)
            for p, vals in sorted(by_pred.items(), key=lambda kv: st.mean(kv[1])):
                print("    {:<6} filtrelenmis ayak hatasi: {:.1f} px".format(p, st.mean(vals)))
    print(C_TITLE + "=" * MW + C_RESET)


def summary_table(rows):
    """Birden fazla tracker verildiyse karsilastirma tablosu."""
    print()
    print(C_TITLE + "KARSILASTIRMA TABLOSU" + C_RESET)
    _line()
    print("  {}{:<14}{:>7}{:>8}{:>9}{:>10}{:>9}{}".format(
        C_KEY, "TRACKER", "IoU", "IoU>.5", "KAPSAMA", "FOOT_px", "SKOR", C_RESET))
    _line()
    best = max(r[1]["overall"] for r in rows) if rows else 0
    for name, m in rows:
        col = C_OK if m["overall"] == best else C_VAL
        print("  {}{:<14}{:>7.3f}{:>7.1f}%{:>8.1f}%{:>10.1f}{:>8.1f}%{}".format(
            col, name, m["iou_mean"], m["s50"], m["coverage"],
            m["foot_mean"], m["overall"], C_RESET))
    _line("=")


# =============================================================================
# 4) MAIN
# =============================================================================
def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    matrix_mode = "--matrix" in sys.argv          # tek sayfa benchmark tablosu

    if len(args) < 2:
        print("Kullanim:")
        print("  python compare.py <referans.csv> <tracker.csv> [...]        # detayli rapor")
        print("  python compare.py --matrix <referans.csv> <eval_*.csv ...>  # tek sayfa matris")
        sys.exit(1)

    ref, _ = load(args[0])
    if matrix_mode:
        rows = []
        for path in args[1:]:
            trk, meta = load(path)
            rows.append((meta, evaluate(ref, trk)))
        matrix_table(rows)
        return

    results = []
    for path in args[1:]:
        trk, meta = load(path)
        m = evaluate(ref, trk)
        name = meta.get("tracker") or path.replace("\\", "/").rsplit("/", 1)[-1].replace(".csv", "")
        report(name, meta, m)
        results.append((name, m))

    if len(results) > 1:
        summary_table(results)


if __name__ == "__main__":
    main()
