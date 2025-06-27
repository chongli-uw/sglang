#!/usr/bin/bash
while ! python -m sglang.bench_serving --num-prompts 2048 --dataset-name dapo --disable-stream --dapo-max-resp-len=8192 --output-details --disable-ignore-eos; do
    echo "Command failed. Retrying..."
    sleep 60 
done
