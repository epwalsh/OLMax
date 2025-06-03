#!/usr/bin/env bash

gantry run \
    --priority=high \
    --yes \
    --timeout=-1 \
    --gpus=8 \
    --beaker-image=petew/olmax \
    --env="PYTHONUNBUFFERED=1" \
    --env-secret="BEAKER_TOKEN=PETEW_BEAKER_TOKEN" \
    --allow-dirty \
    --gpu-type=h100 \
    --install './src/scripts/beaker/setup_env.sh' \
    --replicas=2 \
    --leader-selection \
    --host-networking \
    --propagate-failure \
    --propagate-preemption \
    --synchronized-start-timeout=5m \
    -- "$@"
