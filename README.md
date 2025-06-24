# JAX-dev

## Benchmarks

Benchmarks on various transformer models can be executed via the script `src/integration_tests/train_transformer.py`.
Run `python src/integration_tests/train_transformer.py --help` to see the script's usage.

The benchmarks listed below were launched on Beaker via OLMax's `launch.beaker` module.
Run `python -m olmax.launch.beaker --help` to see the module's usage.

Each benchmark below was run with these trainer settings:
- Sequence length: `4096`
- Compute data type: `BF16`
- Optimizer data type: `FP32`

### Llama-like 7B

- [12,035 TPS/GPU](https://beaker.org/ex/01JXDC25DM5GJRMWNDXSJ2QA7J) on 2 Jupiter H100 nodes with full FSDP, micro-batch size of 2 instances/GPU
  ```fish
  python -m olmax.launch.beaker --cluster=ai2/jupiter-cirrascale-2 --nodes=2 -- \
    python src/integration_tests/train_transformer.py --recipe=llama_like_7B
  ```
- [11,525 TPS/GPU](https://beaker.org/ex/01JX9F1J0XM2P7B2HEA9BNZQ1H) on 2 Augusta H100 nodes with full FSDP, micro-batch size of 2 instances/GPU
  ```fish
  python -m olmax.launch.beaker --cluster=ai2/augusta-google-1 --nodes=2 -- \
    python src/integration_tests/train_transformer.py --recipe=llama_like_7B
  ```
- [12,217 TPS/GPU](https://beaker.org/ex/01JYFJ7KQ0CE6WJ1HXWQ40Y88B) on 1 Ceres H100 node with full FSDP, micro-batch size of 2 instances/GPU
  ```fish
  python -m olmax.launch.beaker --gpu-type=h100 -- \
    python src/integration_tests/train_transformer.py --recipe=llama_like_7B
  ```

### OLMo2 7B

- [11,122 TPS/GPU](https://beaker.org/ex/01JX98QAKVVMC8YJX3400T96TA) on 1 Jupiter H100 node with full FSDP, micro-batch size of 2 instances/GPU
  ```fish
  python -m olmax.launch.beaker --gpu-type=h100 -- \
    python src/integration_tests/train_transformer.py --recipe=olmo2_7B
  ```
- [11,146 TPS/GPU](https://beaker.org/ex/01JX98C8XPW4BSVVG8XEE3MHN2) on 2 Jupiter H100 nodes with full FSDP, micro-batch size of 2 instances/GPU
  ```fish
  python -m olmax.launch.beaker --gpu-type=h100 --nodes=2 -- \
    python src/integration_tests/train_transformer.py --recipe=olmo2_7B
  ```

### Gemma2 27B

- [5,358 TPS/GPU](https://beaker.org/ex/01JX3D0TECNJ28WVZE4Q5DKHKM) on 2 Titan B200 nodes with full FSDP, micro-batch size of 2 instances/GPU
  ```fish
  python -m olmax.launch.beaker --nodes=2 --gpu-type=b200 -- \
    python src/integration_tests/train_transformer.py --recipe=gemma2_like_27B
  ```

### Gemma3 27B

- [4,917 TPS/GPU](https://beaker.org/ex/01JX3CB9A4PD69Q715RPMDYWNY) on 2 Titan B200 nodes with full FSDP, micro-batch size of 2 instances/GPU
  ```fish
  python -m olmax.launch.beaker --nodes=2 --gpu-type=b200 -- \
    python src/integration_tests/train_transformer.py --recipe=gemma3_like_27B
  ```

### OLMo2 32B

- [2,155 TPS/GPU](https://beaker.org/ex/01JYHCTA0YM9X8B5F7457WESR0) on 2 Jupiter H100 nodes with full FSDP, micro-batch size of 4 instances/GPU, full block activation checkpointing.
  ```fish
  python -m olmax.launch.beaker --nodes=2 --gpu-type=h100 -- \
    python src/integration_tests/train_transformer.py --recipe=olmo2_32B
  ```
