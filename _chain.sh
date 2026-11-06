#!/bin/bash
cd /home/pi/staj5.2
venv/bin/python -u eval_runner.py
echo "=== EVAL BITTI; 60 sn soguma ==="
sleep 60
venv/bin/python -u fps_bench.py --frames 300 --repeats 5 --out fps_bench.json
echo "=== HEPSI BITTI ==="
