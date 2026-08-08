# decode-lab — Single-Request Decode Lab

> First proof-of-work project in the LLM inference learning journey.

## Goal

Build a small local experiment that runs a single prompt through a causal language model and teaches you the foundations of LLM inference:

- tokenization
- prefill vs decode
- autoregressive generation
- TTFT (time to first token)
- tokens/sec
- GPU memory usage

## What you will build

A Python module (`decode_lab`) that:

1. Loads a small causal LM locally.
2. Tokenizes one prompt.
3. Runs generation for a fixed number of output tokens.
4. Prints clean measurements.

## Minimum outputs

For each run, capture:

| Metric | Description |
|---|---|
| Input token count | Tokens in the prompt |
| Output token count | Tokens generated |
| TTFT | Time to first token (prefill latency) |
| Decode throughput | tokens/sec during decode phase |
| Total latency | End-to-end wall clock |
| Peak GPU memory | `torch.cuda.max_memory_allocated()` |

## Example results (GTX 1650 Ti, fp16)

Committed sample run (see `results/`). All metrics from a single request, no batching.

| Model | Prompt len | TTFT (s) | Decode throughput (tok/s) | Peak mem (GB) |
|---|---|---|---|---|
| SmolLM2-135M | 5 | 0.061 | 12.0 | 0.262 |
| SmolLM2-135M | 36 | 0.276 | 19.2 | 0.266 |
| SmolLM2-135M | 117 | 0.236 | 18.8 | 0.278 |
| gpt2 | 5 | 0.082 | 71.1 | 0.264 |
| gpt2 | 37 | 0.132 | 103.9 | 0.270 |
| gpt2 | 121 | 0.232 | 105.8 | 0.284 |

Reading: decode throughput is **flat across prompt lengths** but scales with model size — the classic signature of a **memory-bandwidth-bound** decode phase (each new token re-reads the full weights; larger models are slower per token). TTFT grows with prompt length (more prefill compute). Plots in `results/ttft_vs_prompt.png`, `results/throughput_vs_model.png`, `results/memory_breakdown.png`.

## Setup

```powershell
# From the decode-lab directory
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Run

```powershell
python -m decode_lab
```

## Suggested models for GTX 1650 Ti (4GB VRAM)

| Model | Size | Why |
|---|---|---|
| `sshleifer/tiny-gpt2` | ~3 MB | Smoke test only — verifies pipeline |
| `HuggingFaceTB/SmolLM2-135M` | ~270 MB | First real model — fast, modern |
| `gpt2` | ~500 MB | The classic baseline |
| `Qwen/Qwen2.5-0.5B` | ~1 GB | Modern, capable, small |
| `microsoft/Phi-1.5` | ~3 GB | Pushes your GPU — good for memory profiling |

Start with `SmolLM2-135M`. Graduate up.

## Project layout

```
decode-lab/
├── .venv/               Virtual env (NOT committed)
├── results/             Example measurement outputs (CSV, plots) — committed
├── src/
│   └── decode_lab/      The importable Python package
│       ├── __init__.py
│       └── __main__.py  Entry point for `python -m decode_lab`
├── tests/               Pytest tests
│   └── test_basic.py
├── .gitignore
├── .gitattributes       Line-ending normalization
├── .python-version      Pins Python version (pyenv / uv compatible)
├── pyproject.toml       Tool config (ruff, pytest, mypy) + project metadata
├── README.md            This file
├── requirements.txt     Runtime dependencies (pinned)
└── requirements-dev.txt Dev dependencies (pytest, ruff)
```

> Personal lab notes (observations, questions) are kept in a private journal, not in this public repo.

## Questions you should be able to answer after this project

1. What happens during prefill?
2. What changes during decode?
3. Why is decode usually memory-bandwidth sensitive?
4. Why does batch size affect throughput and latency differently?
5. What exactly does TTFT include?

## Definition of done

You are done when you can:

- [ ] Run one prompt end-to-end
- [ ] Produce all 6 measurements in a clean table
- [ ] Explain each number in your own words (write it in `../notes/decode-lab/observations.md` in the private journal)
- [ ] Test passes: `pytest`
- [ ] Code passes lint: `ruff check .`

## Working rules

- **Write the code yourself.** AI is for explanation, debugging hints, and review — not writing the implementation.
- Prefer simple, readable code over clever abstractions.
- Commit early and often, with clear messages.
- Type hints on every function signature.
- `logging` module for measurement output (not `print` for production code).
