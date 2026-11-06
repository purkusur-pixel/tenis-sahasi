"""
make_zone_overlay.py -- INSAN tikladigi kort noktalarindan bolge izgarasi overlay'i uretir.

NEDEN INSAN NOKTALARINDAN: Overlay, etiketleyiciye "ayak hangi hucrede" sorusunda
gorsel kilavuzluk eder. Bunu SISTEMIN otomatik kort tespitinden turetmek, bolge
GT'sini degerlendirilecek sistemin kalibrasyonuna bagimli kilardi (dongusellik).
annotate_court.py ile ELLE tiklanan noktalar sistemden bagimsizdir; overlay
yalnizca "fit" rolundeki capa noktalarindan kurulur.

Cikti: TAM COZUNURLUKTE (1920x1080) overlay PNG -- etiketleme kareleriyle birebir
hizali (gt_extract.py kareleri tam cozunurlukte yazar).

Kullanim (Pi'de, proje klasorunde):
    venv/bin/python make_zone_overlay.py court_gt_A1.json -o overlay_A1.png \
        [--width 1920 --height 1080]
"""

import argparse
import json
import sys

import cv2
import numpy as np

import config
import court
import zones


def main():
    p = argparse.ArgumentParser(description="Insan kort GT'sinden bolge overlay'i")
    p.add_argument("court_gt_json")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    args = p.parse_args()

    with open(args.court_gt_json, encoding="utf-8") as fh:
        gt = json.load(fh)

    world = gt["world"]
    clicked = gt["points"]

    # --- YALNIZCA "fit" rolundeki noktalarla homografi kur --------------------
    # config.COURT_CORNERS_WORLD ile AYNI sirada olmalari sart; annotate_court.py
    # bu sirayi koruyarak sorar.
    fit_img, fit_world = [], []
    for name, w in world.items():
        if w.get("role") != "fit":
            continue
        if name not in clicked:
            continue
        fit_img.append(clicked[name])
        fit_world.append([w["x"], w["y"]])

    if len(fit_img) < 4:
        sys.exit("Yetersiz 'fit' noktasi ({}): en az 4 gerekli. "
                 "annotate_court.py ile yesil (FIT) noktalari tiklayin.".format(len(fit_img)))

    img_pts = np.array(fit_img, dtype=np.float64)
    wld_pts = np.array(fit_world, dtype=np.float64)

    H, _ = cv2.findHomography(img_pts, wld_pts, method=0)   # goruntu -> dunya
    if H is None:
        sys.exit("Homografi kurulamadi (noktalar dejenere olabilir).")
    H_inv = np.linalg.inv(H)                                # dunya -> goruntu

    # Isaret normalizasyonu (court.compute_homography ile ayni mantik)
    ref = wld_pts.mean(axis=0)
    if H_inv[2, 0] * ref[0] + H_inv[2, 1] * ref[1] + H_inv[2, 2] < 0:
        H, H_inv = -H, -H_inv

    # --- bolge katmanini TAM COZUNURLUKTE render et ---------------------------
    shape = (args.height, args.width, 3)
    state = zones.build(H_inv.astype(np.float64), shape)
    layer = state["layer"]              # siyah zemin uzerine izgara + numaralar

    cv2.imwrite(args.output, layer)

    nz = int(np.count_nonzero(cv2.cvtColor(layer, cv2.COLOR_BGR2GRAY)))
    print("fit noktasi   : {}".format(len(fit_img)))
    print("overlay       : {}  ({}x{}, {} dolu piksel)".format(
        args.output, args.width, args.height, nz))

    # Dogrulama: tiklanan TEST noktalarinin yeniden-projeksiyon hatasi
    errs = []
    for name, w in world.items():
        if w.get("role") != "test" or name not in clicked:
            continue
        pr = court.court_to_image(H_inv, w["x"], w["y"])
        if pr is None:
            continue
        cx, cy = clicked[name]
        errs.append(float(np.hypot(pr[0] - cx, pr[1] - cy)))
    if errs:
        print("held-out test : {} nokta | yeniden-projeksiyon hatasi "
              "ort {:.1f} px, medyan {:.1f} px, max {:.1f} px".format(
                  len(errs), float(np.mean(errs)), float(np.median(errs)), max(errs)))
    else:
        print("held-out test : (turuncu TEST noktalari tiklanmamis -- "
              "kalibrasyon dogrulugu olculemez)")


if __name__ == "__main__":
    main()
