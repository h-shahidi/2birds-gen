#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python generate.py --model_prefix logs/ --out_path results