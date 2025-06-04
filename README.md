# JAX-dev

## Benchmarks

### Llama-like 7B

Setup:
- Sequence length: `4096`
- Parallelism: full FSDP
- Compute data type: `BF16`
- Optimizer data type: `FP32`

Launch command:

```fish
python -m olmax.launch.beaker --allow-dirty --nodes=2 --gpu-type=h100 -- python src/integration_tests/train_transformer.py --recipe=7B --attn=cudnn
```

Results:
- [11,537 TPS/GPU](https://beaker.org/ex/01JWXR4M5KFH578WFGYNZQDX17) on 2 Jupiter H100 nodes
- [10,653 TPS/GPU](https://beaker.org/ex/01JWXWWW70E9A9AD6EX8MW85S8) on 2 Augusta H100 nodes

### Gemma2 27B

Setup:
- Sequence length: `4096`
- Parallelism: full FSDP
- Compute data type: `BF16`
- Optimizer data type: `FP32`

Launch command:

```fish
python -m olmax.launch.beaker --nodes=2 --gpu-type=b200 -- python src/integration_tests/train_transformer.py --recipe=gemma2_27B --attn=cudnn --vocab-size=256000
```

Results:
- [4,717 TPS/GPU](https://beaker.org/ex/01JWW3NKVGXZKSB0DA6H1NJ4R5) on 2 Titan B200 nodes
