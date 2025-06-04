#!/usr/bin/env bash

if [[ -d "/var/lib/tcpxo/lib64" ]]; then
    echo "Configuring NCCL for GPUDirect-TCPXO..."
    export NCCL_LIB_DIR="/var/lib/tcpxo/lib64"
    export LD_LIBRARY_PATH="/var/lib/tcpxo/lib64:$LD_LIBRARY_PATH"
    # shellcheck disable=SC1091
    source /var/lib/tcpxo/lib64/nccl-env-profile.sh
fi

pip install -e ".[all]"
