#!/usr/bin/env bash

gantry run \
    --priority=high \
    --yes \
    --timeout=-1 \
    --gpus=8 \
    --beaker-image=petew/olmax \
    --env="PYTHONUNBUFFERED=1" \
    --env="NCCL_DEBUG=info" \
    --env-secret="BEAKER_TOKEN=PETEW_BEAKER_TOKEN" \
    --allow-dirty \
    --gpu-type=b200 \
    --install './src/scripts/beaker/setup_env.sh' \
    --host-networking \
    -- "$@"
