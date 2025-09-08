#!/usr/bin/env bash

set -e

echo "Patching fused ring attention..."
wget https://gist.githubusercontent.com/epwalsh/8fbde5374638b62f49743a219831dc7c/raw/4c3dabe6d6ea51534fdc20ad6debaf1347fd7961/patched_attention.py
mv patched_attention.py /usr/local/lib/python3.12/dist-packages/transformer_engine/jax/cpp_extensions/attention.py
