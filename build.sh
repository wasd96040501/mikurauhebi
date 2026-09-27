#!/usr/bin/env bash
# 一键生成 v3（中文版 + 英文版）：配乐(score.py, MIDI+FluidSynth) -> 混音(mixdown.py, 音效+台词 blip) -> 画面(director.py) -> 合成成片
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
[ -x "$PY" ] || { python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt; }
command -v fluidsynth >/dev/null || brew install fluid-synth
mkdir -p out
$PY src/score.py out/music.wav
(cd src && ../$PY mixdown.py)
# 中文版 / 英文版画面（日文原句两版都保留），共用同一条音轨
for L in zh en; do
  (cd src && PV_LANG=$L ../$PY director.py ../out/v3_noaudio_$L.mp4)
  ffmpeg -y -loglevel error -i out/v3_noaudio_$L.mp4 -i out/soundtrack.wav \
    -c:v copy -c:a aac -b:a 256k -shortest -movflags +faststart out/ouroboros_pv_$L.mp4
  echo "done -> out/ouroboros_pv_$L.mp4"
done
