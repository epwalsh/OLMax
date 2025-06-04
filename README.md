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
- [11,740 TPS/GPU](https://beaker.org/ex/01JWY01XXCSS291DW5KTG9GP76) on 2 Jupiter H100 nodes
- [10,909 TPS/GPU](https://beaker.org/ex/01JWY0GKCNJM8C32WZ7H7MGM93) on 2 Augusta H100 nodes

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
- [4,731 TPS/GPU](https://beaker.org/ex/01JWY2W7TGN20NE6MB1RGQEP7S) on 2 Titan B200 nodes
