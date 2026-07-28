"""
zones.py
--------
Kort bolge (zone) modeli, perspektif izgara overlay'i ve kus bakisi panel.
 
IZGARA SEMASI (overlay_rect.py ile birebir ayni yapi ve etiketler):
 
        +-----------------------------------------------+
   LT1  |  25   26   27   28   29   30   31   32         |  RT1   <- ust bant (8x1)
   -----+-----------------------------------------------+-----
   LT2  |  17   18   19   20   21   22   23   24         |  RT2
   LT3  |   9   10   11   12   13   14   15   16         |  RT3   <- ana izgara (8x3)
   LT4  |   1    2    3    4    5    6    7    8         |  RT4      ALTTAN numaralanir
   -----+-----------------------------------------------+-----
   LB1  |  1x   2x   3x   4x   5x   6x   7x   8x         |  RB1
   LB2  |  9x  10x  11x  12x  13x  14x  15x  16x         |  RB2   <- alt bant (8x4)
   LB3  | 17x  18x  19x  20x  21x  22x  23x  24x         |  RB3      YUKARIDAN sayilir
   LB4  | 25x  26x  27x  28x  29x  30x  31x  32x         |  RB4
        +-----------------------------------------------+
 
    Toplam: 24 (ana) + 8 (ust) + 32 (alt) + 16 (yan) = 80 hucre.
    Yan serit dilimleri, satir sinirlariyla birebir hizalidir:
      LT1 -> ust bant, LT2..LT4 -> ana izgara satirlari, LB1..LB4 -> alt bant satirlari.
 
TASARIM:
    Bolge sorgusu icin AYRI bir homography gerekmez. Normalize kort koordinati
    zaten elimizdedir: s = wx / COURT_WIDTH, t = wy / COURT_LENGTH. Bu yuzden
    get_zone() saf aritmetiktir -> O(1), her karede bedavaya calisir.
 
ORIJINALDEN TEK FARK:
    overlay_rect.py'de bant derinlikleri ORANSAL verilir (1/ROWS), yani alt bant
    kortun 4/3'u kadar (~31.7 m) asagi uzar. Ekran dortgeninde sorun degil, ama
    tenis kortunda o alan tribun arkasidir. Yapi ve etiketler aynen korunmus,
    derinlik metre cinsinden ayarlanabilir yapilmistir (config.BOTTOM_BAND_DEPTH).
    Birebir orijinal oran icin: BOTTOM_BAND_DEPTH = COURT_LENGTH * BOTTOM_BAND_ROWS / GRID_ROWS
 
PERFORMANS:
    Kamera sabit oldugundan izgara, numaralar ve kort cizgileri BIR KEZ statik bir
    katmana render edilir. Her karede yapilan is sadece:
      1) cv2.copyTo ile katmanin maskeli kopyalanmasi (tek numpy islemi),
      2) aktif hucrenin vurgulanmasi (tek poligon).
"""
 
import cv2
import numpy as np
 
import config
import court
 
_W = config.COURT_WIDTH
_L = config.COURT_LENGTH
_GL = config.GRID_LENGTH     # Ana izgaranin dunya-y uzunlugu (YARIM kort = NET_Y).

_COLS = config.GRID_COLS
_ROWS = config.GRID_ROWS
_TOP_ROWS = config.TOP_BAND_ROWS
_BOT_ROWS = config.BOTTOM_BAND_ROWS
 
_TOP_DEPTH = config.TOP_BAND_DEPTH        # Toplam derinlik (metre).
_BOT_DEPTH = config.BOTTOM_BAND_DEPTH     # Toplam derinlik (metre).
_SIDE = config.SIDE_BAND_WIDTH            # Yan serit genisligi (metre).
 
_N_MAIN = _COLS * _ROWS                   # Ana izgara hucre sayisi (varsayilan 24).
 
_CELL_W = _W / float(_COLS)
 
# --- Satir sinirlari (dunya y, metre). Ust banttan alt banda kesintisiz. ---
# Bant sayilari: [0 .. TOP_ROWS-1] ust, [TOP_ROWS .. TOP_ROWS+ROWS-1] ana,
#                [TOP_ROWS+ROWS .. sonu] alt.
_Y_EDGES = []
for _i in range(_TOP_ROWS):
    _Y_EDGES.append(-_TOP_DEPTH + _i * (_TOP_DEPTH / _TOP_ROWS))
for _i in range(_ROWS):
    _Y_EDGES.append(_i * (_GL / _ROWS))              # ana izgara YARIM kort: 0 -> NET_Y
for _i in range(_BOT_ROWS + 1):
    _Y_EDGES.append(_GL + _i * (_BOT_DEPTH / _BOT_ROWS))   # alt bantlar file'dan (NET_Y) baslar
 
_Y_EDGES = np.array(_Y_EDGES, dtype=np.float64)
 
_N_BANDS = len(_Y_EDGES) - 1        # Toplam satir bandi (1 + 3 + 4 = 8)
_Y_MIN = float(_Y_EDGES[0])
_Y_MAX = float(_Y_EDGES[-1])
 
_SIDE_T_COUNT = _TOP_ROWS + _ROWS   # LT dilim sayisi (varsayilan 4)
 
 
# ==============================================================================
# 1) BOLGE SORGUSU (saf aritmetik)
# ==============================================================================
 
def _col_index(wx):
    col = int(wx / _CELL_W)
    return 0 if col < 0 else (_COLS - 1 if col > _COLS - 1 else col)
 
 
def _band_index(wy):
    """Satir bandi indeksi (0 = en ust bant, _N_BANDS-1 = en alt bant)."""
    b = int(np.searchsorted(_Y_EDGES, wy, side="right")) - 1
    if b < 0:
        return 0
    if b > _N_BANDS - 1:
        return _N_BANDS - 1
    return b
 
 
def get_zone(wx, wy):
    """Kort koordinatindan (metre) bolge etiketini dondurur."""
    if wy < _Y_MIN or wy > _Y_MAX or wx < -_SIDE or wx > _W + _SIDE:
        return config.ZONE_OUT
 
    band = _band_index(wy)
 
    # --- Yan seritler: LT1..LTn / LB1..LBn (yukaridan asagi) ---
    if wx < 0.0 or wx > _W:
        if not config.SHOW_SIDE_BANDS:
            return config.ZONE_OUT
        side = "L" if wx < 0.0 else "R"
        if band < _SIDE_T_COUNT:
            return "{}T{}".format(side, band + 1)
        return "{}B{}".format(side, band - _SIDE_T_COUNT + 1)
 
    col = _col_index(wx)
 
    # --- Ust bant: 25..32 ---
    if band < _TOP_ROWS:
        if not config.SHOW_TOP_BAND:
            return config.ZONE_OUT
        return str(_N_MAIN + band * _COLS + col + 1)
 
    # --- Ana izgara: 1..24 (ALTTAN yukari) ---
    if band < _TOP_ROWS + _ROWS:
        row = band - _TOP_ROWS                  # 0 = uzak (ust) satir
        return str((_ROWS - 1 - row) * _COLS + col + 1)
 
    # --- Alt bant: 1x..32x (YUKARIDAN asagi) ---
    if not config.SHOW_BOTTOM_BAND:
        return config.ZONE_OUT
    brow = band - _TOP_ROWS - _ROWS             # 0 = dip cizgiye en yakin satir
    return "{}x".format(brow * _COLS + col + 1)
 
 
# ==============================================================================
# 2) HUCRE TANIMLARI (dunya koordinatinda)
# ==============================================================================
 
def _build_cells():
    """
    Tum hucreleri (etiket, dunya dortgeni, tur) uretir.
    Tur: "main" | "top" | "bottom" | "side"  -> farkli renklerle cizilir.
    """
    cells = []
 
    for band in range(_N_BANDS):
        y0 = float(_Y_EDGES[band])
        y1 = float(_Y_EDGES[band + 1])
 
        # --- Kort ici hucreler ---
        if band < _TOP_ROWS:
            if config.SHOW_TOP_BAND:
                for c in range(_COLS):
                    label = str(_N_MAIN + band * _COLS + c + 1)
                    cells.append((label, (c * _CELL_W, y0, (c + 1) * _CELL_W, y1), "top"))
        elif band < _TOP_ROWS + _ROWS:
            row = band - _TOP_ROWS
            for c in range(_COLS):
                label = str((_ROWS - 1 - row) * _COLS + c + 1)
                cells.append((label, (c * _CELL_W, y0, (c + 1) * _CELL_W, y1), "main"))
        else:
            if config.SHOW_BOTTOM_BAND:
                brow = band - _TOP_ROWS - _ROWS
                for c in range(_COLS):
                    label = "{}x".format(brow * _COLS + c + 1)
                    cells.append((label, (c * _CELL_W, y0, (c + 1) * _CELL_W, y1), "bottom"))
 
        # --- Yan serit dilimleri (her satir bandi icin bir dilim) ---
        if config.SHOW_SIDE_BANDS:
            if band < _SIDE_T_COUNT:
                l_label = "LT{}".format(band + 1)
                r_label = "RT{}".format(band + 1)
            else:
                l_label = "LB{}".format(band - _SIDE_T_COUNT + 1)
                r_label = "RB{}".format(band - _SIDE_T_COUNT + 1)
            cells.append((l_label, (-_SIDE, y0, 0.0, y1), "side"))
            cells.append((r_label, (_W, y0, _W + _SIDE, y1), "side"))
 
    return cells
 
 
def _project_quad(H_inv, x0, y0, x1, y1):
    """Dunya dortgenini goruntuye projekte eder. Kamera arkasina dusen kose varsa None."""
    pts = []
    for wx, wy in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
        p = court.court_to_image(H_inv, wx, wy)
        if p is None:
            return None
        px = max(-10000.0, min(10000.0, p[0]))
        py = max(-10000.0, min(10000.0, p[1]))
        pts.append((int(round(px)), int(round(py))))
    return np.array(pts, dtype=np.int32)
 
 
# ==============================================================================
# 3) KURULUM (bir kez calisir)
# ==============================================================================
 
def build(H_inv, frame_shape):
    """
    Bolge sistemini kurar: hucre poligonlari, statik overlay katmani ve panel tabani.
    Kamera sabit varsayildigi icin BIR KEZ cagrilir.
    """
    h, w = frame_shape[:2]
    layer = np.zeros((h, w, 3), dtype=np.uint8)
 
    cells = {}
    for label, quad, kind in _build_cells():
        poly = _project_quad(H_inv, *quad)
        center = court.court_to_image(H_inv,
                                      (quad[0] + quad[2]) * 0.5,
                                      (quad[1] + quad[3]) * 0.5)
        cells[label] = {"quad": quad, "poly": poly, "center": center, "kind": kind}
 
    color_of = {
        "main": config.GRID_COLOR,
        "top": config.TOP_BAND_COLOR,
        "bottom": config.BOTTOM_BAND_COLOR,
        "side": config.SIDE_COLOR,
    }
 
    for cell in cells.values():
        if cell["poly"] is None:
            continue
        cv2.polylines(layer, [cell["poly"]], True, color_of[cell["kind"]],
                      config.GRID_THICKNESS, cv2.LINE_AA)
 
    # Kort cizgileri izgaranin uzerine (daha belirgin).
    court.draw_lines(layer, court.project_lines(H_inv, court.COURT_LINES),
                     config.COLOR_COURT, config.LINE_THICKNESS + 1)
 
    if config.SHOW_GRID_NUMBERS:
        for label, cell in cells.items():
            c = cell["center"]
            if c is None:
                continue
            cx, cy = int(round(c[0])), int(round(c[1]))
            if cx < 0 or cy < 0 or cx >= w or cy >= h:
                continue
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                          config.GRID_NUMBER_SCALE, 2)
            org = (cx - tw // 2, cy + th // 2)
            cv2.putText(layer, label, org, cv2.FONT_HERSHEY_SIMPLEX,
                        config.GRID_NUMBER_SCALE, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(layer, label, org, cv2.FONT_HERSHEY_SIMPLEX,
                        config.GRID_NUMBER_SCALE, config.GRID_NUMBER_COLOR, 1, cv2.LINE_AA)
 
    # Maske: katmanda cizim olan pikseller. Her karede sadece bunlar kopyalanir.
    gray = cv2.cvtColor(layer, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
 
    state = {"cells": cells, "layer": layer, "mask": mask}
 
    if config.SHOW_PANEL:
        _build_panel(state)
 
    return state
 
 
def _world_to_panel(wx, wy):
    """Kort koordinatini panel piksel koordinatina cevirir."""
    px = config.PANEL_MARGIN + (wx + _SIDE) * config.PANEL_PPM
    py = config.PANEL_HEADER + config.PANEL_MARGIN + (wy - _Y_MIN) * config.PANEL_PPM
    return (int(round(px)), int(round(py)))
 
 
def panel_size():
    """Panelin (genislik, yukseklik) degeri. HUD'u panelin yanina hizalamak icin."""
    pw = int(round((_W + 2 * _SIDE) * config.PANEL_PPM)) + 2 * config.PANEL_MARGIN
    ph = int(round((_Y_MAX - _Y_MIN) * config.PANEL_PPM)) + 2 * config.PANEL_MARGIN \
        + config.PANEL_HEADER
    return pw, ph
 
 
def _build_panel(state):
    """Kus bakisi panelin degismeyen tabanini uretir (bir kez)."""
    pw, ph = panel_size()
 
    base = np.zeros((ph, pw, 3), dtype=np.uint8)
    base[:] = config.PANEL_BG
    cv2.rectangle(base, (0, 0), (pw - 1, config.PANEL_HEADER - 1),
                  config.PANEL_HEADER_BG, -1)
 
    rects = {}
    color_of = {
        "main": config.PANEL_GRID_COLOR,
        "top": config.PANEL_BAND_COLOR,
        "bottom": config.PANEL_BAND_COLOR,
        "side": config.PANEL_SIDE_COLOR,
    }
 
    for label, cell in state["cells"].items():
        x0, y0, x1, y1 = cell["quad"]
        p0 = _world_to_panel(x0, y0)
        p1 = _world_to_panel(x1, y1)
        rects[label] = (p0[0], p0[1], p1[0], p1[1])
 
        cv2.rectangle(base, p0, p1, color_of[cell["kind"]], 1)
 
        if config.PANEL_SHOW_NUMBERS:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                          config.PANEL_NUMBER_SCALE, 1)
            cx = (p0[0] + p1[0]) // 2 - tw // 2
            cy = (p0[1] + p1[1]) // 2 + th // 2
            cv2.putText(base, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                        config.PANEL_NUMBER_SCALE, config.PANEL_NUMBER_COLOR, 1, cv2.LINE_AA)
 
    # Kort cizgileri (panel uzerinde beyaz ve belirgin).
    for (ax, ay), (bx, by) in court.COURT_LINES:
        cv2.line(base, _world_to_panel(ax, ay), _world_to_panel(bx, by),
                 config.PANEL_COURT_COLOR, 1, cv2.LINE_AA)
    cv2.line(base, _world_to_panel(0.0, config.NET_Y),
             _world_to_panel(_W, config.NET_Y), config.PANEL_COURT_COLOR, 2, cv2.LINE_AA)
 
    state["panel"] = base
    state["panel_rects"] = rects
 
 
# ==============================================================================
# 4) CIZIM (her karede)
# ==============================================================================
 
def draw_overlay(frame, state, label=None):
    """
    Statik izgara katmanini kareye kopyalar ve aktif hucreyi vurgular.
    Katman onceden render edildigi icin maliyet tek bir maskeli kopyadir.
    """
    cv2.copyTo(state["layer"], state["mask"], frame)
 
    if label is None:
        return frame
 
    cell = state["cells"].get(label)
    if cell is None or cell["poly"] is None:
        return frame
 
    poly = cell["poly"]
    x, y, w, h = cv2.boundingRect(poly)
 
    fh, fw = frame.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(fw, x + w), min(fh, y + h)
    if x1 - x0 < 1 or y1 - y0 < 1:
        return frame
 
    roi = frame[y0:y1, x0:x1]
    tinted = roi.copy()
    shifted = poly - np.array([[x0, y0]], dtype=np.int32)
    cv2.fillConvexPoly(tinted, shifted, config.HIGHLIGHT_COLOR, cv2.LINE_AA)
    cv2.addWeighted(tinted, config.HIGHLIGHT_ALPHA, roi,
                    1.0 - config.HIGHLIGHT_ALPHA, 0.0, roi)
 
    cv2.polylines(frame, [poly], True, config.HIGHLIGHT_BORDER, 2, cv2.LINE_AA)
    return frame
 
 
def draw_panel(frame, state, label=None, world_pos=None, trail=None):
    """Kus bakisi paneli kareye yerlestirir: aktif hucre vurgulu, oyuncu ve izi ile."""
    if "panel" not in state:
        return frame
 
    view = state["panel"].copy()
 
    if label is not None and label in state["panel_rects"]:
        x0, y0, x1, y1 = state["panel_rects"][label]
        tinted = view.copy()
        cv2.rectangle(tinted, (x0, y0), (x1, y1), config.PANEL_HIGHLIGHT, -1)
        cv2.addWeighted(tinted, config.PANEL_HIGHLIGHT_ALPHA, view,
                        1.0 - config.PANEL_HIGHLIGHT_ALPHA, 0.0, view)
        cv2.rectangle(view, (x0, y0), (x1, y1), config.PANEL_HIGHLIGHT_BORDER, 2)
 
    if trail and len(trail) > 1:
        pts = np.array([_world_to_panel(px, py) for px, py in trail], dtype=np.int32)
        cv2.polylines(view, [pts], False, config.PANEL_TRAIL_COLOR, 1, cv2.LINE_AA)
 
    if world_pos is not None:
        p = _world_to_panel(world_pos[0], world_pos[1])
        cv2.circle(view, p, 4, config.PANEL_PLAYER_COLOR, -1, cv2.LINE_AA)
        cv2.circle(view, p, 6, config.PANEL_COURT_COLOR, 1, cv2.LINE_AA)
 
    text = "BOLGE: {}".format(label if label else "-")
    cv2.putText(view, text, (8, config.PANEL_HEADER - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, config.PANEL_TEXT_COLOR, 1, cv2.LINE_AA)
 
    fh, fw = frame.shape[:2]
    ph, pw = view.shape[:2]
    x, y = config.PANEL_POS
    if x < 0 or y < 0 or x + pw > fw or y + ph > fh:
        return frame  # Sigmiyorsa cizme (PANEL_PPM'i kucult).
 
    roi = frame[y:y + ph, x:x + pw]
    cv2.addWeighted(view, config.PANEL_ALPHA, roi, 1.0 - config.PANEL_ALPHA, 0.0, roi)
    cv2.rectangle(frame, (x, y), (x + pw - 1, y + ph - 1), config.PANEL_BORDER, 1)
    return frame