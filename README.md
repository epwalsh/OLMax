# JAX-dev

## Benchmarks

### Llama-like 7B

**Common configuration:**
- Sequence length: `4096`
- Compute data type: `BF16`
- Optimizer data type: `FP32`

**Results:**
- [12,035 TPS/GPU](https://beaker.org/ex/01JXDC25DM5GJRMWNDXSJ2QA7J) on 2 Jupiter H100 nodes with full FSDP, micro-batch size of 2 instances/GPU
  ```fish
  python -m olmax.launch.beaker --cluster=ai2/jupiter-cirrascale-2 --nodes=2 -- \
    python src/integration_tests/train_transformer.py llama_like_7B \
    --batch-size-per-device=8192
  ```
- [11,525 TPS/GPU](https://beaker.org/ex/01JX9F1J0XM2P7B2HEA9BNZQ1H) on 2 Augusta H100 nodes with full FSDP, micro-batch size of 2 instances/GPU
  ```fish
  python -m olmax.launch.beaker --cluster=ai2/augusta-google-1 --nodes=2 -- \
    python src/integration_tests/train_transformer.py llama_like_7B \
    --batch-size-per-device=8192
  ```
- [12,173 TPS/GPU](https://beaker.org/ex/01JX98QTJRSNCCZ4CSYMJ7CXR6) on 1 Ceres H100 node with full FSDP, micro-batch size of 2 instances/GPU
  ```fish
  python -m olmax.launch.beaker --gpu-type=h100 -- \
    python src/integration_tests/train_transformer.py llama_like_7B \
    --batch-size-per-device=8192
  ```

### OLMo2 7B

**Common configuration:**
- Sequence length: `4096`
- Compute data type: `BF16`
- Optimizer data type: `FP32`

**Results:**
- [11,122 TPS/GPU](https://beaker.org/ex/01JX98QAKVVMC8YJX3400T96TA) on 1 Jupiter H100 node with full FSDP, micro-batch size of 2 instances/GPU
  ```fish
  python -m olmax.launch.beaker --gpu-type=h100 -- \
    python src/integration_tests/train_transformer.py olmo_7B \
    --batch-size-per-device=8192
  ```
- [11,146 TPS/GPU](https://beaker.org/ex/01JX98C8XPW4BSVVG8XEE3MHN2) on 2 Jupiter H100 nodes with full FSDP, micro-batch size of 2 instances/GPU
  ```fish
  python -m olmax.launch.beaker --gpu-type=h100 --nodes=2 -- \
    python src/integration_tests/train_transformer.py olmo_7B \
    --batch-size-per-device=8192
  ```

### Gemma2 27B

**Common configuration:**
- Sequence length: `4096`
- Compute data type: `BF16`
- Optimizer data type: `FP32`

**Results:**
- [5,358 TPS/GPU](https://beaker.org/ex/01JX3D0TECNJ28WVZE4Q5DKHKM) on 2 Titan B200 nodes with full FSDP, micro-batch size of 2 instances/GPU
  ```fish
  python -m olmax.launch.beaker --nodes=2 --gpu-type=b200 -- \
    python src/integration_tests/train_transformer.py gemma2_like_27B \
    --batch-size-per-device=8192
  ```

### Gemma3 27B

**Common configuration:**
- Sequence length: `4096`
- Compute data type: `BF16`
- Optimizer data type: `FP32`

**Results:**
- [4,917 TPS/GPU](https://beaker.org/ex/01JX3CB9A4PD69Q715RPMDYWNY) on 2 Titan B200 nodes with full FSDP, micro-batch size of 2 instances/GPU
  ```fish
  python -m olmax.launch.beaker --nodes=2 --gpu-type=b200 -- \
    python src/integration_tests/train_transformer.py gemma3_like_27B \
    --batch-size-per-device=8192
  ```

### OLMo2 32B

**Common configuration:**
- Sequence length: `4096`
- Compute data type: `BF16`
- Optimizer data type: `FP32`

**Results:**
- [2,144 TPS/GPU](https://beaker.org/ex/01JY24TWEJTCK18AJAGTE831ZM) on 2 Jupiter H100 nodes with full FSDP, micro-batch size of 4 instances/GPU, full block activation checkpointing.
  ```fish
  python -m olmax.launch.beaker --nodes=2 --gpu-type=h100 -- \
    python src/integration_tests/train_transformer.py olmo_32B \
    --layer_ac_policy='{type: nothing_saveable, prevent_cse: false}' \
    --scan_layers \
    --batch-size-per-device=16384
  ```
