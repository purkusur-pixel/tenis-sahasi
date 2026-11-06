#!/bin/bash
# Projeyi sadeleştirir: olu kodu SILER, degerlendirme dosyalarini proje DISINA tasir.
# Tasima geri alinabilir (mv), silme yalnizca import edilmedigi dogrulanan iki dosyada.
set -u
cd ~/staj5.2 || exit 1

ARSIV=~/arsiv_staj
echo "ONCE:"
du -sh --exclude=venv --exclude=input . 2>/dev/null

# --- 1) OLU KOD: sil (hicbir yerde import edilmedigi dogrulandi) -------------
rm -f tenis_takip.py ball.py
rm -f __pycache__/tenis_takip*.pyc __pycache__/ball*.pyc
echo "silindi: tenis_takip.py (120K, kullanilmayan tek-dosya eski kopya), ball.py (0 bayt)"

# --- 2) DEGERLENDIRME: proje disina tasi -----------------------------------
mkdir -p "$ARSIV/betikler" "$ARSIV/veri" "$ARSIV/test_videolari"

for f in benchmark.py eval_runner.py fps_bench.py compare.py \
         _matrix_iso.py _matrix_run.py make_zone_overlay.py; do
  [ -e "$f" ] && mv -f "$f" "$ARSIV/betikler/"
done

for f in eval_gt referans_A1.csv fps_bench.json gt_index.json overlay_A1.png \
         overlay_A2.png overlay_A3.png overlay_C01.png; do
  [ -e "$f" ] && mv -f "$f" "$ARSIV/veri/"
done

# Test ciktilari: 300 MB'lik asil yuk. first_frame_court_detection.jpg KALIR --
# ekransiz calismada kort tespitini gozle dogrulamanin tek yolu odur.
mv -f output/tracked_*.mp4 "$ARSIV/test_videolari/" 2>/dev/null
mv -f output/matrix output/matrix_iso "$ARSIV/test_videolari/" 2>/dev/null
mv -f output/_zone_overlay.jpg "$ARSIV/veri/" 2>/dev/null

# Eski derlenmis artiklar
rm -rf __pycache__

echo
echo "SONRA (proje):"
du -sh --exclude=venv --exclude=input . 2>/dev/null
echo "ARSIV (proje disi, Pi'de duruyor):"
du -sh "$ARSIV"
echo
echo "=== PROJEDE KALAN .py ==="
ls -1 *.py
echo
echo "=== output/ ==="
ls -1 output/
