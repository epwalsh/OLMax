#!/usr/bin/env bash

gantry run \
    --priority=high \
    --yes \
    --timeout=-1 \
    --gpus=2 \
    --cluster=ai2/augusta-google-1 \
    --cluster=ai2/jupiter-cirrascale-2 \
    --install 'pip install "jax[cuda12]" -r requirements.txt' \
    -- "$@"
