#!/bin/sh -eu

export CUDA_VISIBLE_DEVICES=0

python3 main.py --outdir test

