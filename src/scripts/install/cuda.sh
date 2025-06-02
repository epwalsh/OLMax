#!/usr/bin/env bash

pip install "jax[cuda12]"
pip install "transformer_engine[jax]"
pip install -e ".[all]"
