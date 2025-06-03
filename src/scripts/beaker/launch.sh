#!/usr/bin/env bash

gantry run \
    --priority=high \
    --yes \
    --timeout=-1 \
    --gpus=8 \
    --beaker-image=petew/olmax \
    --env-var="PYTHONUNBUFFERED=1" \
    --cluster=ai2/augusta-google-1 \
    --cluster=ai2/jupiter-cirrascale-2 \
    --cluster=ai2/ceres-cirrascale \
    --install 'pip install -e .' \
    -- "$@"

    # --cluster=ai2/titan-cirrascale \
