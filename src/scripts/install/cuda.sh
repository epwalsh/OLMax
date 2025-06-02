#!/usr/bin/env bash

set -e

pip install "jax[cuda12]"
pip install "transformer_engine[jax]"
pip install -e ".[all]"
