"""
mjpeg.py
--------
Iki is yapar, ikisi de tarayici uzerinden (Pi ekransiz calisir):

  1) Streamer      : overlay'i MJPEG olarak yayinlar -> telefondan canli izleme.
  2) setup_court() : kort koselerini telefondan AYARLATIR -> surukle + canli cizgi.

NEDEN TARAYICI:
    Pi kortta ekransiz calisacak. Masaustunu acik tutmak hem CPU yer (labwc,
    panel, VNC) hem de korta laptop tasimayi gerektirir. Kullanicinin elinde
    zaten telefon var; ek uygulama/eklenti olmadan tarayicidan hallediliyor.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

_BOUNDARY = "frameboundary"

_PAGE = b"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tenis Oyuncu Takibi</title>
<style>
 html,body{margin:0;background:#111;color:#eee;font-family:system-ui,sans-serif}
 h1{font-size:15px;font-weight:600;padding:10px 12px;margin:0}
 img{width:100%;height:auto;display:block}
 p{padding:8px 12px;margin:0;font-size:12px;color:#999}
</style></head>
<body>
<h1>Tenis Oyuncu Takibi &mdash; canli</h1>
<img src="/stream" alt="canli goruntu">
<p>Goruntu donmussa sayfayi yenileyin.</p>
</body></html>
"""


class Streamer:
    def __init__(self, port=8080, quality=70, max_fps=12):
        self.port = int(port)
        self.quality = int(quality)
        self.max_fps = float(max_fps)
        self._frame = None
        self._seq = 0
        self._lock = threading.Lock()
        self._server = None
        self._thread = None

    # -- ana dongunun cagirdigi tek sey ------------------------------------
    def update(self, frame):
        """Son kareyi kaydeder. UCUZ olmasi sart: kopyalama/kodlama YOK."""
        if frame is None:
            return
        with self._lock:
            self._frame = frame
            self._seq += 1

    def _snapshot(self, last_seq):
        """Yeni kare varsa (kare, seq) doner; yoksa (None, last_seq)."""
        with self._lock:
            if self._frame is None or self._seq == last_seq:
                return None, last_seq
            return self._frame, self._seq

    def start(self):
        streamer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *args):
                pass          # konsolu her istekle kirletme

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(_PAGE)))
                    self.end_headers()
                    self.wfile.write(_PAGE)
                elif self.path == "/stream":
                    self._stream()
                else:
                    self.send_error(404)

            def _stream(self):
                self.send_response(200)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Pragma", "no-cache")
                self.send_header(
                    "Content-Type",
                    "multipart/x-mixed-replace; boundary=" + _BOUNDARY)
                self.end_headers()

                enc = [int(cv2.IMWRITE_JPEG_QUALITY), streamer.quality]
                interval = 1.0 / streamer.max_fps if streamer.max_fps > 0 else 0.0
                last_seq = -1
                try:
                    while True:
                        t0 = time.time()
                        frame, last_seq = streamer._snapshot(last_seq)
                        if frame is None:
                            time.sleep(0.01)
                            continue
                        ok, buf = cv2.imencode(".jpg", frame, enc)
                        if not ok:
                            continue
                        data = buf.tobytes()
                        self.wfile.write(b"--" + _BOUNDARY.encode() + b"\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(
                            b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n")
                        self.wfile.write(data)
                        self.wfile.write(b"\r\n")
                        dt = time.time() - t0
                        if interval > dt:
                            time.sleep(interval - dt)
                except (BrokenPipeError, ConnectionResetError):
                    pass      # istemci sekmeyi kapatti -- olagan
                except Exception:
                    pass      # yayin hatasi ANA DONGUYU asla etkilemesin

        class Server(ThreadingHTTPServer):
            allow_reuse_address = True   # yeniden baslatinca "address in use" olmasin
            daemon_threads = True

        try:
            self._server = Server(("0.0.0.0", self.port), Handler)
        except OSError as exc:
            print("Yayin baslatilamadi (port {}): {}".format(self.port, exc))
            self._server = None
            return self

        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        if self._server is not None:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception:
                pass
            self._server = None


# ===========================================================================
# KORT KURULUM SAYFASI (surukle + canli cizgi)
# ===========================================================================
# TASARIM:
#   - Acilista mevcut 5 nokta VE onlardan uretilen TUM kort cizgileri cizili gelir.
#     Kullanici once "kort dogru oturmus mu" diye BAKAR; tikladigi bir sey yok.
#   - Begenmezse noktayi PARMAKLA SURUKLER. Tiklamaya gore avantaji: parmak
#     hedefi kapatmaz, noktayi goze goze tasirsin.
#   - Her hareket sonrasi homografi YENIDEN hesaplanir ve cizgiler ANLIK guncellenir.
#     Boylece kullanici sonucu tahmin etmez, GORUR.
#   - Buyutec: surukleme sirasinda parmagin altindaki bolge buyutulmus gosterilir.
#
# NEDEN HOMOGRAFI TARAYICIDA HESAPLANIYOR:
#   Cizgilerin surukleme sirasinda akici guncellenmesi lazim. Her parmak
#   hareketinde sunucuya sorup cevap beklemek (onlarca istek/saniye) hem agi
#   hem Pi'yi bosuna mesgul ederdi. 5 noktadan DLT en kucuk kareler + Gauss
#   eliminasyonu tarayicida milisaniyenin altinda cozuluyor.
_SETUP_PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>Kort Kurulumu</title>
<style>
 html,body{margin:0;background:#111;color:#eee;font-family:system-ui,sans-serif;
           -webkit-user-select:none;user-select:none;touch-action:manipulation}
 #hdr{padding:9px 12px;background:#1c1c1c;position:sticky;top:0;z-index:5}
 #ttl{font-size:15px;font-weight:600}
 #sub{font-size:12px;color:#9a9a9a;margin-top:3px;line-height:1.35}
 #wrap{position:relative;line-height:0;touch-action:none}
 #im{width:100%;height:auto;display:block}
 #cv{position:absolute;left:0;top:0;width:100%;height:100%}
 #loupe{position:absolute;right:8px;top:8px;width:128px;height:128px;
        border:2px solid #fff;border-radius:64px;display:none;z-index:4;
        box-shadow:0 2px 10px rgba(0,0,0,.6);background:#000}
 #bar{display:flex;gap:8px;padding:10px 12px;background:#1c1c1c;
      position:sticky;bottom:0}
 button{flex:1;padding:14px 8px;font-size:15px;font-weight:600;border:0;
        border-radius:8px;background:#333;color:#eee}
 #ok{background:#1f7a3a}
 #msg{padding:10px 12px;font-size:14px;color:#7fd18f;min-height:20px}
 #msg.err{color:#ff8f8f;font-weight:600}
</style></head>
<body>
<div id="hdr">
  <div id="ttl">Kort dogru mu?</div>
  <div id="sub">Cizgiler kortla ortusuyorsa <b>Onayla</b>. Ortusmuyorsa bir
   noktayi <b>parmakla surukle</b> &mdash; cizgiler aninda guncellenir.</div>
</div>
<div id="wrap">
  <img id="im" src="/frame.jpg" alt="kort">
  <canvas id="cv"></canvas>
  <canvas id="loupe" width="128" height="128"></canvas>
</div>
<div id="bar">
  <button id="reset">Basa don</button>
  <button id="ok">Onayla</button>
</div>
<div id="msg"></div>
<script>
var D = __DATA__;
var pts = D.pts.map(function(p){ return [p[0], p[1]]; });
var pts0 = D.pts.map(function(p){ return [p[0], p[1]]; });
var im = document.getElementById("im");
var cv = document.getElementById("cv");
var ctx = cv.getContext("2d");
var loupe = document.getElementById("loupe");
var lctx = loupe.getContext("2d");
var active = -1;

/* ---- 8 bilinmeyenli lineer sistem (Gauss, kismi pivotlama) -------------- */
function solve8(A, b){
  var n = 8, i, j, k;
  for (i = 0; i < n; i++){
    var piv = i;
    for (k = i + 1; k < n; k++) if (Math.abs(A[k][i]) > Math.abs(A[piv][i])) piv = k;
    if (Math.abs(A[piv][i]) < 1e-12) return null;
    var t = A[i]; A[i] = A[piv]; A[piv] = t;
    var tb = b[i]; b[i] = b[piv]; b[piv] = tb;
    for (k = i + 1; k < n; k++){
      var f = A[k][i] / A[i][i];
      if (!f) continue;
      for (j = i; j < n; j++) A[k][j] -= f * A[i][j];
      b[k] -= f * b[i];
    }
  }
  var x = new Array(n);
  for (i = n - 1; i >= 0; i--){
    var s = b[i];
    for (j = i + 1; j < n; j++) s -= A[i][j] * x[j];
    x[i] = s / A[i][i];
  }
  return x;
}

/* ---- dunya -> goruntu homografisi (DLT, en kucuk kareler) --------------- */
/* h33=1 alinir; her karsilik 2 denklem verir. 5 nokta -> 10 denklem, 8
   bilinmeyen -> normal denklemlerle (A^T A)h = A^T b cozulur.              */
function homography(world, img){
  var AtA = [], Atb = new Array(8).fill(0), i, j, k;
  for (i = 0; i < 8; i++) AtA.push(new Array(8).fill(0));
  for (k = 0; k < world.length; k++){
    var X = world[k][0], Y = world[k][1], x = img[k][0], y = img[k][1];
    var rows = [[X, Y, 1, 0, 0, 0, -X * x, -Y * x],
                [0, 0, 0, X, Y, 1, -X * y, -Y * y]];
    var rhs = [x, y];
    for (var r = 0; r < 2; r++){
      for (i = 0; i < 8; i++){
        Atb[i] += rows[r][i] * rhs[r];
        for (j = 0; j < 8; j++) AtA[i][j] += rows[r][i] * rows[r][j];
      }
    }
  }
  return solve8(AtA, Atb);
}

function project(h, X, Y){
  var d = h[6] * X + h[7] * Y + 1;
  if (Math.abs(d) < 1e-9) return null;
  return [(h[0] * X + h[1] * Y + h[2]) / d, (h[3] * X + h[4] * Y + h[5]) / d];
}

/* ---- cizim -------------------------------------------------------------- */
function draw(){
  var W = im.naturalWidth, H = im.naturalHeight;
  if (!W) return;
  cv.width = W; cv.height = H;
  ctx.clearRect(0, 0, W, H);

  var h = homography(D.world, pts);
  var okFit = false;
  if (h){
    ctx.strokeStyle = "rgba(0,225,255,.95)";
    ctx.lineWidth = Math.max(2, W / 500);
    ctx.beginPath();
    var drew = 0;
    for (var i = 0; i < D.lines.length; i++){
      var a = project(h, D.lines[i][0][0], D.lines[i][0][1]);
      var b = project(h, D.lines[i][1][0], D.lines[i][1][1]);
      if (!a || !b) continue;
      ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); drew++;
    }
    ctx.stroke();
    okFit = drew > 0;
  }
  document.getElementById("ttl").textContent =
    okFit ? "Kort dogru mu?" : "Noktalar gecersiz - duzelt";

  var R = Math.max(9, W / 90);
  for (var k = 0; k < pts.length; k++){
    ctx.beginPath();
    ctx.arc(pts[k][0], pts[k][1], R, 0, 6.2832);
    ctx.fillStyle = (k === active) ? "rgba(255,215,0,.85)" : "rgba(255,60,60,.6)";
    ctx.fill();
    ctx.lineWidth = Math.max(2, W / 600);
    ctx.strokeStyle = "#fff"; ctx.stroke();
    ctx.beginPath(); ctx.arc(pts[k][0], pts[k][1], 1.6, 0, 6.2832);
    ctx.fillStyle = "#fff"; ctx.fill();
    ctx.fillStyle = "#fff";
    ctx.font = "bold " + Math.round(R * 1.25) + "px system-ui";
    ctx.fillText(String(k + 1), pts[k][0] + R + 2, pts[k][1] - R - 2);
  }
}

/* ---- buyutec: parmak noktayi kapatmasin --------------------------------- */
function showLoupe(p){
  var Z = 3.5, S = 128;
  lctx.fillStyle = "#000"; lctx.fillRect(0, 0, S, S);
  lctx.save();
  lctx.beginPath(); lctx.arc(S / 2, S / 2, S / 2 - 1, 0, 6.2832); lctx.clip();
  lctx.drawImage(im, p[0] - S / (2 * Z), p[1] - S / (2 * Z), S / Z, S / Z, 0, 0, S, S);
  var h = homography(D.world, pts);
  if (h){
    lctx.strokeStyle = "rgba(0,225,255,.95)"; lctx.lineWidth = 1.5;
    lctx.beginPath();
    for (var i = 0; i < D.lines.length; i++){
      var a = project(h, D.lines[i][0][0], D.lines[i][0][1]);
      var b = project(h, D.lines[i][1][0], D.lines[i][1][1]);
      if (!a || !b) continue;
      lctx.moveTo((a[0] - p[0]) * Z + S / 2, (a[1] - p[1]) * Z + S / 2);
      lctx.lineTo((b[0] - p[0]) * Z + S / 2, (b[1] - p[1]) * Z + S / 2);
    }
    lctx.stroke();
  }
  lctx.restore();
  lctx.strokeStyle = "#ff3c3c"; lctx.lineWidth = 1;
  lctx.beginPath();
  lctx.moveTo(S / 2 - 12, S / 2); lctx.lineTo(S / 2 + 12, S / 2);
  lctx.moveTo(S / 2, S / 2 - 12); lctx.lineTo(S / 2, S / 2 + 12);
  lctx.stroke();
  loupe.style.display = "block";
  // Parmak sag ustteyse buyutec sola kacsin
  var r = im.getBoundingClientRect();
  var onRight = (p[0] / im.naturalWidth) > 0.55;
  var onTop = (p[1] / im.naturalHeight) < 0.35;
  loupe.style.right = onRight ? "auto" : "8px";
  loupe.style.left  = onRight ? "8px" : "auto";
  loupe.style.top   = onTop ? "auto" : "8px";
  loupe.style.bottom = onTop ? "8px" : "auto";
}

/* ---- olay -> goruntu koordinati ---------------------------------------- */
function toImg(e){
  var r = im.getBoundingClientRect();
  var t = (e.touches && e.touches[0]) ? e.touches[0] : e;
  return [(t.clientX - r.left) * im.naturalWidth / r.width,
          (t.clientY - r.top) * im.naturalHeight / r.height];
}
function nearest(p){
  var best = -1, bd = 1e18;
  for (var i = 0; i < pts.length; i++){
    var dx = pts[i][0] - p[0], dy = pts[i][1] - p[1], d = dx * dx + dy * dy;
    if (d < bd){ bd = d; best = i; }
  }
  var lim = im.naturalWidth / 8;          // yakalama yaricapi (genis: parmak kaba)
  return (bd <= lim * lim) ? best : -1;
}

function onDown(e){
  var p = toImg(e);
  active = nearest(p);
  if (active >= 0){ e.preventDefault(); pts[active] = p; draw(); showLoupe(p); }
}
function onMove(e){
  if (active < 0) return;
  e.preventDefault();
  var p = toImg(e);
  pts[active] = p; draw(); showLoupe(p);
}
function onUp(){ active = -1; loupe.style.display = "none"; draw(); }

var wrap = document.getElementById("wrap");
wrap.addEventListener("touchstart", onDown, {passive: false});
wrap.addEventListener("touchmove", onMove, {passive: false});
wrap.addEventListener("touchend", onUp);
wrap.addEventListener("mousedown", onDown);
window.addEventListener("mousemove", onMove);
window.addEventListener("mouseup", onUp);

document.getElementById("reset").onclick = function(){
  pts = pts0.map(function(p){ return [p[0], p[1]]; });
  draw();
};
document.getElementById("ok").onclick = function(){
  var btn = document.getElementById("ok");
  var msg = document.getElementById("msg");
  btn.disabled = true;
  btn.textContent = "Gonderiliyor...";
  msg.className = "";
  msg.textContent = "";

  var x = new XMLHttpRequest();
  x.open("POST", "/corners", true);
  x.setRequestHeader("Content-Type", "application/json");
  x.timeout = 8000;

  // BASARI: sayfanin TAMAMI degisir. Onceki surumde yalnizca en altta kucuk
  // yesil bir yazi cikiyordu; kullanici gonderdigini FARK ETMEDI ve tekrar
  // denedi. Geri bildirim goz kacirilamayacak kadar belirgin olmali.
  x.onload = function(){
    if (x.status >= 200 && x.status < 300){
      document.body.innerHTML =
        '<div style="padding:28px 20px;font-family:system-ui,sans-serif;' +
        'background:#123d22;color:#eaffef;min-height:100vh">' +
        '<div style="font-size:44px;line-height:1">&#10003;</div>' +
        '<h2 style="margin:14px 0 8px;font-size:21px">Kort onaylandi</h2>' +
        '<p style="margin:0;font-size:15px;line-height:1.5;color:#b8e6c6">' +
        'Takip basladi. Canli goruntuyu izlemek icin bu sayfayi <b>yenile</b>' +
        ' &mdash; ayni adres artik yayini gosteriyor.</p></div>';
    } else {
      fail("Sunucu reddetti (kod " + x.status + "). Sayfayi yenileyip tekrar dene.");
    }
  };
  x.onerror = function(){ fail("Baglanti kurulamadi. Pi ile ayni agda misin?"); };
  x.ontimeout = function(){ fail("Yanit gelmedi (8 sn). Sayfayi yenile."); };

  function fail(t){
    btn.disabled = false;
    btn.textContent = "Onayla";
    msg.className = "err";
    msg.textContent = t;
  }

  x.send(JSON.stringify({points: pts}));
};
im.onload = draw;
window.addEventListener("resize", draw);
if (im.complete) draw();
</script>
</body></html>
"""


def default_corners(width, height):
    """
    Otomatik tespit hic sonuc veremediginde gosterilecek BASLANGIC noktalari.

    Amac dogruluk degil, kullaniciya surukleyecek bir sey vermek: kabaca dogru
    yerde duran bir yamuk, tipik bir kort goruntusune gore. Sira
    config.CORNER_LABELS ile ayni.
    """
    w, h = float(width), float(height)
    return [
        (0.25 * w, 0.45 * h),   # file - sol ciftler
        (0.75 * w, 0.45 * h),   # file - sag ciftler
        (0.50 * w, 0.45 * h),   # file - orta
        (0.15 * w, 0.68 * h),   # yakin servis - sol tekler
        (0.85 * w, 0.68 * h),   # yakin servis - sag tekler
    ]


def setup_court(frame, world_pts, court_lines, initial_pts=None,
                port=8080, quality=85, timeout_sec=600, auto_ok=False):
    """
    Kort koselerini TELEFONDAN onaylatir/ayarlatir.

    Kullanici once mevcut noktalari VE onlardan uretilen kort cizgilerini gorur;
    ortusme iyiyse Onayla'ya basar, degilse noktalari surukler (cizgiler anlik
    guncellenir).

    frame        : gosterilecek goruntu (oyuncusuz medyan arka plan tercih edilir)
    world_pts    : referans noktalarinin dunya koordinatlari (config.COURT_CORNERS_WORLD)
    court_lines  : kort cizgileri, dunya koordinatlarinda (court.COURT_LINES)
    initial_pts  : baslangic goruntu noktalari; None ise makul bir yamuk kullanilir
    auto_ok      : otomatik tespit tuttu mu (yalnizca ekrana basilan mesaj icin)

    Donus: [(x, y) x5] veya kullanici sureye kadar onaylamazsa None.
    """
    ok, buf = cv2.imencode(".jpg", frame,
                           [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return None
    jpg = buf.tobytes()

    h_img, w_img = frame.shape[:2]
    if initial_pts is None:
        initial_pts = default_corners(w_img, h_img)

    data = {
        "world": [[float(p[0]), float(p[1])] for p in world_pts],
        "lines": [[[float(a[0]), float(a[1])], [float(b[0]), float(b[1])]]
                  for a, b in court_lines],
        "pts": [[float(p[0]), float(p[1])] for p in initial_pts],
    }
    page = _SETUP_PAGE.replace("__DATA__", json.dumps(data)).encode("utf-8")

    result = {"points": None}
    done = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, "text/html; charset=utf-8", page)
            elif self.path == "/frame.jpg":
                self._send(200, "image/jpeg", jpg)
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path != "/corners":
                self.send_error(404)
                return
            try:
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n).decode("utf-8"))
                pts = [(float(p[0]), float(p[1])) for p in body["points"]]
            except Exception:
                self.send_error(400)
                return
            if len(pts) != len(data["world"]):
                self.send_error(400)
                return
            result["points"] = pts
            self._send(200, "application/json", b'{"ok":true}')
            done.set()

        def _send(self, code, ctype, body):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    class Server(ThreadingHTTPServer):
        allow_reuse_address = True
        daemon_threads = True

    try:
        srv = Server(("0.0.0.0", int(port)), Handler)
    except OSError as exc:
        print("Kurulum sayfasi acilamadi (port {}): {}".format(port, exc))
        return None

    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print("=" * 64)
    print("KORT KURULUMU -> TELEFONDAN ONAYLA" if auto_ok
          else "OTOMATIK KORT TESPITI BASARISIZ -> TELEFONDAN AYARLA")
    for ip in local_ips():
        print("   Tarayicida ac:  http://{}:{}/".format(ip, port))
    if auto_ok:
        print("   Cizgiler kortla ortusuyorsa Onayla. Onaylamazsan {} sn sonra".format(
            timeout_sec))
        print("   otomatik bulunan koselerle devam edilir.")
    else:
        print("   Noktalari surukleyip kortu oturt, sonra Onayla ({} sn).".format(
            timeout_sec))
    print("=" * 64)

    got = done.wait(timeout=float(timeout_sec))
    time.sleep(0.3)          # tarayiciya yanit gitsin
    try:
        srv.shutdown()
        srv.server_close()
    except Exception:
        pass

    if not got:
        print("Telefondan onay gelmedi.")
        return None
    print("Koseler telefondan alindi.")
    return result["points"]


def local_ips():
    """Yayin adresini yazdirmak icin makinenin IPv4 adresleri."""
    import socket
    out = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127.") and ip not in out:
                out.append(ip)
    except Exception:
        pass
    if not out:
        # getaddrinfo bos donebilir; disari giden bir soket acmak en guveniliri
        # (paket GONDERILMEZ, yalnizca yerel adres ogrenilir).
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            out.append(s.getsockname()[0])
            s.close()
        except Exception:
            pass
    return out
