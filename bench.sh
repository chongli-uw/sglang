#!/usr/bin/bash
set -x

for np in 2048; do
    for nr in 8 16; do
        # if not successful retry:
        run_cmd="python -m sglang.bench_serving --num-prompts $np --dataset-name dapo --dapo-max-resp-len 512 --disable-stream --num-responses $nr"
        until $run_cmd; do
            sleep 60
        done
    done
done