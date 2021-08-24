#!/bin/sh
set -eu

export PYTHONUNBUFFERED=1
exec python3 src/model_apply.py
