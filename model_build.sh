#!/bin/sh
set -eu

export PYTHONUNBUFFERED=1
python3 src/data_generate.py && python3 src/model_train.py
