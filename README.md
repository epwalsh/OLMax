# JAX-dev

## Benchmarks

### Llama-like 7B

**Common configuration:**
- Sequence length: `4096`
- Compute data type: `BF16`
- Optimizer data type: `FP32`

**Results:**
- [11,740 TPS/GPU](https://beaker.org/ex/01JWY01XXCSS291DW5KTG9GP76) on 2 Jupiter H100 nodes with full FSDP, micro-batch size of 2 instances/GPU
- [10,924 TPS/GPU](https://beaker.org/ex/01JWYHDW51H9R9H1Z0X3AZZW9D) on 2 Augusta H100 nodes with full FSDP, micro-batch size of 2 instances/GPU
- [12,157 TPS/GPU](https://beaker.org/ex/01JWY6Q9PRD5QZ8176K2FS8QP0) on 1 Augusta H100 node with full FSDP, micro-batch size of 2 instances/GPU

**Example launch command:**
```fish
python -m olmax.launch.beaker --allow-dirty --gpu-type=h100 -- python src/integration_tests/train_transformer.py --recipe=llama_like_7B --attn=cudnn
```

### Gemma2 27B

**Common configuration:**
- Sequence length: `4096`
- Compute data type: `BF16`
- Optimizer data type: `FP32`

**Results:**
- [4,617 TPS/GPU](https://beaker.org/ex/01JX3DF7HVHTMXNZD2P2SRQZFE) on 2 Titan B200 nodes with full FSDP, micro-batch size of 1 instance/GPU
- [5,358 TPS/GPU](https://beaker.org/ex/01JX3D0TECNJ28WVZE4Q5DKHKM) on 2 Titan B200 nodes with full FSDP, micro-batch size of 2 instances/GPU

**Example launch command:**
```fish
python -m olmax.launch.beaker --nodes=2 --gpu-type=b200 -- python src/integration_tests/train_transformer.py --recipe=gemma2_like_27B --attn=cudnn
```

### Gemma3 27B

**Common configuration:**
- Sequence length: `4096`
- Compute data type: `BF16`
- Optimizer data type: `FP32`

**Results:**
- [4,027 TPS/GPU](https://beaker.org/ex/01JX33WC54YRG2X4943VGQSGZQ) on 2 Titan B200 nodes with full FSDP, micro-batch size of 1 instance/GPU
- [4,917 TPS/GPU](https://beaker.org/ex/01JX3CB9A4PD69Q715RPMDYWNY) on 2 Titan B200 nodes with full FSDP, micro-batch size of 2 instances/GPU

**Example launch command:**
```fish
python -m olmax.launch.beaker --nodes=2 --gpu-type=b200 -- python src/integration_tests/train_transformer.py --recipe=gemma3_like_27B --attn=cudnn
```
