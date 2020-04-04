#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python train.py --config_path config.json --data_split 1
