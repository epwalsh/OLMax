# JAX-dev

## Benchmarks

### Llama-like 7B

**Common configuration:**
- Sequence length: `4096`
- Compute data type: `BF16`
- Optimizer data type: `FP32`

**Results:**
- [11,740 TPS/GPU](https://beaker.org/ex/01JWY01XXCSS291DW5KTG9GP76) on 2 Jupiter H100 nodes with full FSDP
- [10,924 TPS/GPU](https://beaker.org/ex/01JWYHDW51H9R9H1Z0X3AZZW9D) on 2 Augusta H100 nodes with full FSDP
- [12,157 TPS/GPU](https://beaker.org/ex/01JWY6Q9PRD5QZ8176K2FS8QP0) on 1 Augusta H100 node with full FSDP

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
- [4,767 TPS/GPU](https://beaker.org/ex/01JWY650NCVWEBPW97V14K1SJN) on 2 Titan B200 nodes with full FSDP

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
- [ TPS/GPU]() on 2 Titan B200 nodes with full FSDP

**Example launch command:**
```fish
python -m olmax.launch.beaker --nodes=2 --gpu-type=b200 -- python src/integration_tests/train_transformer.py --recipe=gemma3_like_27B --attn=cudnn
```
