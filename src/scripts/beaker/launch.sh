#!/usr/bin/env bash

gantry run \
    --priority=high \
    --yes \
    --timeout=-1 \
    --gpus=8 \
    --beaker-image=petew/olmax \
    --env="PYTHONUNBUFFERED=1" \
    --allow-dirty \
    --gpu-type=h100 \
    --install 'pip install -e .' \
    -- "$@"
