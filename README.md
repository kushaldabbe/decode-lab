# decode-lab

Single-request LLM decode measurement tool. Profiles prefill and decode phases of a causal language model on GPU and reports TTFT, decode throughput, and peak memory usage.

## Overview

`decode_lab` loads a causal language model, runs a single prompt through a fixed-length generation, and measures:

| Metric | Definition |
|---|---|
| TTFT | Time to first token (prefill latency) |
| Decode throughput | Tokens/sec during the decode phase |
| Total latency | End-to-end wall clock |
| Peak GPU memory | `torch.cuda.max_memory_allocated()` after generation |

Timing uses `time.perf_counter()` paired with `torch.cuda.synchronize()`. A warmup forward pass precedes measurement to eliminate first-call JIT/autotune overhead, and models are freed between runs (`del` + `torch.cuda.empty_cache()`). Outputs are written as CSV plus plots to `results/`.

## Usage

```powershell
python -m decode_lab
```

By default this runs `SmolLM2-135M` and `gpt2` in fp16 on CUDA across three prompt lengths, then writes:

- `results/run_<timestamp>.csv` — raw measurements
- `results/ttft_vs_prompt.png` — TTFT vs prompt length, per model
- `results/throughput_vs_model.png` — mean decode throughput per model
- `results/memory_breakdown.png` — weights vs inference overhead per model

### Models

The model list and input prompts are defined as constants in `src/decode_lab/__main__.py` (`MODEL_LIST`, `INPUT_TEXTS`). Any causal LM from the Hugging Face hub can be substituted; small models are appropriate for consumer GPUs (e.g. 4 GB VRAM).

## Results

Sample run on a GTX 1650 Ti (fp16, single request, no batching) — see `results/`.

| Model | Prompt len | TTFT (s) | Decode throughput (tok/s) | Peak mem (GB) |
|---|---|---|---|---|
| SmolLM2-135M | 5 | 0.061 | 12.0 | 0.262 |
| SmolLM2-135M | 36 | 0.276 | 19.2 | 0.266 |
| SmolLM2-135M | 117 | 0.236 | 18.8 | 0.278 |
| gpt2 | 5 | 0.082 | 71.1 | 0.264 |
| gpt2 | 37 | 0.132 | 103.9 | 0.270 |
| gpt2 | 121 | 0.232 | 105.8 | 0.284 |

Interpretation:

- **Decode throughput is flat across prompt lengths** but scales with model size. This is the signature of a memory-bandwidth-bound decode phase: each decoded token re-reads the full weight matrix, so throughput depends on parameter count and memory bandwidth rather than sequence length.
- **TTFT grows with prompt length**, since prefill is a compute-bound forward pass over the full prompt.

## Installation

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Project layout

```
decode-lab/
├── results/          Measurement outputs (CSV, plots)
├── src/decode_lab/   Package; __main__.py is the entry point
├── tests/            Pytest smoke tests
├── pyproject.toml    Project metadata + tool config (ruff, pytest, mypy)
├── requirements.txt  Runtime dependencies
└── requirements-dev.txt
```

## Development

```powershell
pytest
ruff check .
```
