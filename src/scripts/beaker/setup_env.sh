#!/usr/bin/env bash

if [[ -d "/var/lib/tcpxo/lib64" ]]; then
    export LD_LIBRARY_PATH="/var/lib/tcpxo/lib64:$LD_LIBRARY_PATH"
fi

pip install -e ".[all]"
