#!/usr/bin/env bash

set -x

if [[ -d "/var/lib/tcpxo/lib64" ]]; then
    export NCCL_LIB_DIR="/var/lib/tcpxo/lib64"
    export LD_LIBRARY_PATH="/var/lib/tcpxo/lib64:$LD_LIBRARY_PATH"
fi

pip install -e ".[all]"
