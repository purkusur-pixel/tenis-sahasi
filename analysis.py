"""
analysis.py
-----------
Bolge (zone) GECIS analizi.

Oyuncu bir bolgeden digerine gectiginde BIR olay kaydeder. Kare basina DEGIL,
yalnizca bolge DEGISTIGINDE cikti uretir (kullanici istegi: "frame basi degil,
oyuncunun degistirdigi bolge basina text").

IKI CIKTI:
  1) Output videosuna PANEL: son gecisleri gosteren textbox (SHOW_ZONE_PANEL).
  2) DOSYA: output/zone_events.csv (her gecis) + output/zone_summary.json
     (bolge basina toplam sure + ziyaret sayisi).

DURUM (state): sinif yerine hafif bir dict. init() ile olusturulur.

ZAMAN HESABI: frame_idx islenen kare sayisidir; gercek video zamani icin
EFEKTIF FPS (src_fps / FRAME_SKIP) kullanilir -> saniye = frame_idx / fps.
"""

import csv
import json
import os

import cv2


def init(fps, print_zone=False, print_interval=1.0):
    """
    Gecis analizi durumunu olusturur. fps = efektif FPS (src_fps / FRAME_SKIP).

    print_zone     : True -> her print_interval video-saniyesinde bir, o anki
                     bolgeyi terminale basar (headless canli izleme).
    print_interval : bolge yazdirma araligi (video-saniyesi).
    """
    return {
        "fps": float(fps) if fps and fps > 0 else 30.0,
        "events": [],        # gecis olaylari (dict listesi)
        "cur_zone": None,    # oyuncunun su an bulundugu bolge
        "enter_frame": 0,    # cur_zone'a girilen kare
        "dwell": {},         # bolge -> o bolgede gecirilen TOPLAM kare (ozet)
        "visits": {},        # bolge -> bolgeye giris sayisi (ozet)
        "recent": [],        # panelde gosterilecek son gecis metinleri
        "print_zone": bool(print_zone),
        "print_interval": float(print_interval),
        "last_print_vt": -1e9,   # en son bolge yazdirilan video-zamani (sn)
    }


def update(state, zone, frame_idx, pos):
    """
    Her karede cagrilir. Bolge DEGISTIYSE bir gecis olayi kaydeder.

    zone : o karedeki bolge etiketi (None -> takip yok/kayip; olay uretilmez,
           mevcut bolge surdurulur ki kisa takip bosluklari sahte olay yaratmasin).
    pos  : (x, y) metre, yeni bolgeye giris konumu (veya None).
    """
    # Canli izleme: her print_interval video-saniyesinde bir o anki bolgeyi bas.
    # (Gecis olmasi gerekmez; bolge degismese de saniyede bir yazilir.)
    if state["print_zone"]:
        vt = frame_idx / state["fps"]
        if vt - state["last_print_vt"] >= state["print_interval"]:
            state["last_print_vt"] = vt
            z = zone if zone is not None else "- (takip yok)"
            print("[BOLGE] {:6.1f}s -> bolge {}".format(vt, z), flush=True)

    if zone is None:
        return

    # Ilk gecerli bolge: olay yok, sadece baslat.
    if state["cur_zone"] is None:
        state["cur_zone"] = zone
        state["enter_frame"] = frame_idx
        state["visits"][zone] = state["visits"].get(zone, 0) + 1
        return

    if zone == state["cur_zone"]:
        return  # ayni bolge -> olay yok

    # --- GECIS ---
    prev = state["cur_zone"]
    dwell_frames = max(0, frame_idx - state["enter_frame"])
    dwell_s = dwell_frames / state["fps"]

    state["dwell"][prev] = state["dwell"].get(prev, 0) + dwell_frames
    state["events"].append({
        "olay": len(state["events"]) + 1,
        "kare": frame_idx,
        "saniye": round(frame_idx / state["fps"], 2),
        "onceki_bolge": prev,
        "yeni_bolge": zone,
        "onceki_bolgede_sure_sn": round(dwell_s, 2),
        "x_m": round(pos[0], 2) if pos is not None else "",
        "y_m": round(pos[1], 2) if pos is not None else "",
    })
    state["recent"].append("{} -> {}  ({:.1f} sn)".format(prev, zone, dwell_s))

    state["cur_zone"] = zone
    state["enter_frame"] = frame_idx
    state["visits"][zone] = state["visits"].get(zone, 0) + 1


def draw_panel(state, frame, max_lines=6):
    """
    Output videosunun SAG-ALT kosesine son bolge gecislerini yazan textbox.
    Sol-ustteki kus bakisi panelle cakismaz. Yalnizca gecis oldukca degisir.
    """
    recent = state["recent"][-max_lines:]
    cur = "SIMDIKI: {}".format(state["cur_zone"] if state["cur_zone"] is not None else "-")
    lines = ["BOLGE GECISLERI"] + (recent if recent else ["(henuz gecis yok)"]) + [cur]

    fh, fw = frame.shape[:2]
    pad, line_h, box_w = 8, 20, 300
    box_h = pad * 2 + line_h * len(lines)
    x1, y1 = max(0, fw - box_w - 10), max(0, fh - box_h - 10)
    x2, y2 = fw - 10, fh - 10

    # Yari saydam koyu arka plan.
    roi = frame[y1:y2, x1:x2]
    if roi.size:
        dark = roi.copy()
        dark[:] = (30, 30, 30)
        cv2.addWeighted(dark, 0.6, roi, 0.4, 0.0, roi)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

    y = y1 + pad + 12
    for i, txt in enumerate(lines):
        if i == 0:
            color = (0, 255, 255)                 # baslik (sari)
        elif txt.startswith("SIMDIKI"):
            color = (0, 200, 255)                 # simdiki bolge (turuncu)
        else:
            color = (255, 255, 255)               # gecis satirlari (beyaz)
        cv2.putText(frame, txt, (x1 + pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, txt, (x1 + pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    color, 1, cv2.LINE_AA)
        y += line_h
    return frame


def save(state, final_frame, csv_path, json_path):
    """
    Gecis olaylarini CSV'ye, bolge basina ozeti JSON'a yazar.

    final_frame : dongu bittigindeki son kare -> son (aktif) bolgenin suresini
                  kapatmak icin (o bolgeden 'cikis' olayi olmadigi icin).
    """
    # Son aktif bolgenin suresini ozete ekle (kapanis olayi yok).
    if state["cur_zone"] is not None:
        rem = max(0, final_frame - state["enter_frame"])
        state["dwell"][state["cur_zone"]] = state["dwell"].get(state["cur_zone"], 0) + rem

    for path in (csv_path, json_path):
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)

    # --- CSV: her gecis bir satir ---
    cols = ["olay", "kare", "saniye", "onceki_bolge", "yeni_bolge",
            "onceki_bolgede_sure_sn", "x_m", "y_m"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(state["events"])

    # --- JSON: bolge basina ozet (toplam sure + ziyaret) ---
    fps = state["fps"]
    summary = {
        z: {"toplam_sure_sn": round(frames / fps, 2),
            "ziyaret_sayisi": state["visits"].get(z, 0)}
        for z, frames in sorted(state["dwell"].items())
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"toplam_gecis": len(state["events"]),
                   "bolge_ozeti": summary,
                   "olaylar": state["events"]}, f, ensure_ascii=False, indent=2)

    return len(state["events"])
