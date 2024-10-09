#!/bin/bash
# PYTHONPATH=. uv run tools/get_orientation.py --config ./configs/hrnet.yaml \
#     --pretrained pretrained/model_hboe.pt \
#     --dataset /media/jurgen/Documents/datasets/LTCC_ReID/ \
#     --dataset-name ltcc \
#     --target-set train \
#     --device cuda \
#     --batch-size 32

PYTHONPATH=. uv run tools/get_orientation.py --config ./configs/hrnet.yaml \
    --pretrained pretrained/model_hboe.pt \
    --dataset /media/jurgen/Documents/datasets/PRCC/ \
    --dataset-name prcc \
    --target-set query_diff \
    --device cuda \
    --batch-size 64