# OLMax

## Benchmarks

Benchmarks on various transformer models can be executed via the script `src/integration_tests/train_transformer.py`.
Run `python src/integration_tests/train_transformer.py --help` to see the script's usage.

The benchmarks listed below were launched on Beaker via OLMax's `launch.beaker` module.
Run `python -m olmax.launch.beaker --help` to see the module's usage.

### Profiling

Benchmarks launched on Beaker will automatically run a few steps with a profiler enabled, and the resulting trace files will be saved to the workload's result dataset (see [beaker.org/ds/01JZ18BCKD63WFGQYAMN0S9TEZ](https://beaker.org/ds/01JZ18BCKD63WFGQYAMN0S9TEZ), for example).
You can view these profiles with tensorboard or [ui.perfetto.dev](http://ui.perfetto.dev/) by following these steps:

1. First download all trace files:
   ```fish
   beaker dataset fetch 01JZ18BCKD63WFGQYAMN0S9TEZ --output=results/ --prefix=profiler
   ```
   Note that its important to maintain the paths of these files starting with `plugins` (e.g. `plugins/profile/...`) otherwise tensorboard won't load them.
2. Then either launch tensorboard:
   ```fish
   tensorboard --logdir=results/profiler
   ```
   And visit [http://localhost:6006/](http://localhost:6006/) through Chrome (this won't work with Safari).
   Or visit [ui.perfetto.dev](http://ui.perfetto.dev/) and load the `perfetto_trace.json.gz` file you just downloaded from Beaker.

### Known performance issues

- When using HSDP with gradient accumulation enabled (e.g. `--mesh-type=HSDP --num_microbatches=2`), gradients are all-reduced after each micro-batch when ideally we should only issue an all-reduce on the last micro-batch.

### Runs

All run below shared these common trainer settings:
- Sequence length: `4096`
- Compute data type: `BF16`
- Optimizer data type: `FP32`

#### Llama-like 7B

- [12,035 TPS/GPU](https://beaker.org/ex/01JXDC25DM5GJRMWNDXSJ2QA7J) on 2 Jupiter H100 nodes with full FSDP, micro-batch size of 2 instances/GPU
  ```fish
  python -m olmax.launch.beaker --cluster=ai3/jupiter-cirrascale-2 --nodes=2 -- \
    python src/integration_tests/train_transformer.py --recipe=llama_like_7B
  ```
- [11,854 TPS/GPU](https://beaker.org/ex/01JZ1S2X9QF8T3GK3KXME61K0Q) on 2 Augusta H100 nodes with full FSDP, micro-batch size of 2 instances/GPU
  ```fish
  python -m olmax.launch.beaker --cluster=ai2/augusta-google-1 --nodes=2 -- \
    python src/integration_tests/train_transformer.py --recipe=llama_like_7B
  ```
- [12,597 TPS/GPU](https://beaker.org/ex/01JZ1TRYXKNYQVHANM168D234X) on 1 H100 node with full FSDP, micro-batch size of 2 instances/GPU
  ```fish
  python -m olmax.launch.beaker --gpu-type=h100 -- \
    python src/integration_tests/train_transformer.py --recipe=llama_like_7B
  ```

#### OLMo2 7B

- [11,274 TPS/GPU](https://beaker.org/ex/01JZ1VQ7B1VASJBGAXQ4FYD7R2) on 2 Augusta H100 nodes with full FSDP, micro-batch size of 2 instances/GPU
  ```fish
  python -m olmax.launch.beaker --cluster=ai2/augusta-google-1 --nodes=2 -- \
    python src/integration_tests/train_transformer.py --recipe=olmo2_7B \
    --env.jax.compiler_enable_remat_pass=false
  ```
- [11,366 TPS/GPU](https://beaker.org/ex/01JZ3JB9KRXGQAS175GGSS5H8E) on 2 Augusta H100 nodes with node-wise HSDP, micro-batch size of 2 instances/GPU
  ```fish
  python -m olmax.launch.beaker --cluster=ai2/augusta-google-1 --nodes=2 -- \
    python src/integration_tests/train_transformer.py --recipe=olmo2_7B \
    --env.jax.compiler_enable_remat_pass=false \
    --mesh-type=HSDP
  ```
- [12,006 TPS/GPU](https://beaker.org/ex/01JZ1VB0RMCQWBSYCQHM4WQDMD) on 1 H100 node with full FSDP, micro-batch size of 2 instances/GPU
  ```fish
  python -m olmax.launch.beaker --gpu-type=h100 -- \
    python src/integration_tests/train_transformer.py --recipe=olmo2_7B \
    --env.jax.compiler_enable_remat_pass=false
  ```

#### Gemma2 27B

- [5,358 TPS/GPU](https://beaker.org/ex/01JX3D0TECNJ28WVZE4Q5DKHKM) on 2 Titan B200 nodes with full FSDP, micro-batch size of 2 instances/GPU
  ```fish
  python -m olmax.launch.beaker --nodes=2 --gpu-type=b200 -- \
    python src/integration_tests/train_transformer.py --recipe=gemma2_like_27B
  ```

#### Gemma3 27B

- [4,917 TPS/GPU](https://beaker.org/ex/01JX3CB9A4PD69Q715RPMDYWNY) on 2 Titan B200 nodes with full FSDP, micro-batch size of 2 instances/GPU
  ```fish
  python -m olmax.launch.beaker --nodes=2 --gpu-type=b200 -- \
    python src/integration_tests/train_transformer.py --recipe=gemma3_like_27B
  ```

#### OLMo2 32B

- [2,155 TPS/GPU](https://beaker.org/ex/01JYHCTA0YM9X8B5F7457WESR0) on 2 Jupiter H100 nodes with full FSDP, micro-batch size of 4 instances/GPU, full block activation checkpointing.
  ```fish
  python -m olmax.launch.beaker --nodes=2 --gpu-type=h100 -- \
    python src/integration_tests/train_transformer.py --recipe=olmo2_32B
  ```
