#!/usr/bin/env bash

gantry run \
    --priority=high \
    --yes \
    --timeout=-1 \
    --gpus=8 \
    --beaker-image=petew/olmax \
    --cluster=ai2/augusta-google-1 \
    --cluster=ai2/jupiter-cirrascale-2 \
    --install 'pip install -e .' \
    -- "$@"
