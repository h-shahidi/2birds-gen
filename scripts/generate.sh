#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python generate.py --model_prefix logs/ \
                   --in_path data/processed_test.json \
                   --out_path results