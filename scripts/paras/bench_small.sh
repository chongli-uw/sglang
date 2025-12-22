#!/usr/bin/bash
set -x

for nr in 1 4 16 32 64 128; do
    python -m sglang.bench_serving --num-prompts 1 --dataset-name random --random-input-len 4096 --num-responses $nr --disable-stream
done