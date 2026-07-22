"""Entry point for `python -m decode_lab`.

Loads SmolLM2-135M and gpt2 (both fp16) on CUDA and measures inference
performance across three prompt lengths (short / medium / long).

For each (model, prompt) pair, measures:
- TTFT (s): single `model(**inputs)` forward pass, timed with cuda.synchronize.
- Total generation time (s): `model.generate(max_new_tokens=50, min_new_tokens=50)`.
- Decode throughput (tok/s): `50 / (total - ttft)` — excludes prefill.
- Peak GPU memory (GB): `torch.cuda.max_memory_allocated()` after generation.

Pipeline (separation of concerns):
1. MEASURE  → list[dict] of all measurements
2. RECORD   → write to results/run_<timestamp>.csv
3. PRESENT  → plots in results/

A warmup forward pass runs before measurement to eliminate first-call JIT
overhead. Models are freed between iterations via `del` + `empty_cache()`.
"""

from __future__ import annotations

import csv
import logging
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from tabulate import tabulate
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
MODEL_LIST = ["HuggingFaceTB/SmolLM2-135M", "gpt2"]

INPUT_TEXTS = [
    "The cat sat quietly.",
    (
        "Artificial intelligence is transforming software development by helping "
        "engineers write code, debug complex systems, understand unfamiliar "
        "repositories, and automate repetitive tasks, allowing teams to deliver "
        "features faster while maintaining high quality."
    ),
    (
        "Large language models process text by first converting it into tokens, "
        "then performing a prefill stage that builds the attention cache from the "
        "prompt before entering the decode stage where new tokens are generated one "
        "at a time. Measuring metrics such as time to first token, decode latency, "
        "and output throughput helps engineers understand inference performance. "
        "Optimizing batching, cache reuse, memory bandwidth, quantization, and "
        "scheduling can significantly improve responsiveness while reducing "
        "infrastructure costs. Careful benchmarking across different prompt lengths "
        "and output sizes provides a realistic view of production behavior and "
        "reveals bottlenecks that synthetic microbenchmarks might otherwise miss."
    ),
]

# Plot colors per model — kept consistent across plots for visual continuity.
MODEL_COLORS = {
    "HuggingFaceTB/SmolLM2-135M": "#1f77b4",
    "gpt2": "#ff7f0e",
}


def print_environment() -> None:
    """Log Python + torch + GPU info. Useful as a smoke test that setup works."""
    import sys

    logger.info("Python: %s", sys.version.split()[0])
    logger.info("PyTorch: %s", torch.__version__)
    logger.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
        logger.info(
            "Total VRAM: %.2f GB",
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )


# ---------------------------------------------------------------------------
# PHASE 1: MEASURE
# ---------------------------------------------------------------------------


def measure_one_prompt(
    model: AutoModelForCausalLM,
    inputs,
    tokenizer: AutoTokenizer,
) -> dict:
    """Measure TTFT, generation time, and peak memory for a single prompt.

    Args:
        model: causal LM, already on device, in inference_mode.
        inputs: tokenized BatchEncoding (input_ids + attention_mask) on device.
        tokenizer: tokenizer (for pad_token_id).

    Returns:
        Dict with: prompt_length, ttft_s, total_time_s, decode_throughput,
        peak_mem_gb.
    """
    prompt_length = inputs.input_ids.shape[-1]

    # TTFT: single forward pass on the prompt.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = model(**inputs)
    torch.cuda.synchronize()
    ttft_s = time.perf_counter() - t0

    # Total generation + peak memory.
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    _ = model.generate(
        **inputs,
        max_new_tokens=50,
        min_new_tokens=50,
        pad_token_id=tokenizer.eos_token_id,
    )
    torch.cuda.synchronize()
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1024**3
    total_time_s = time.perf_counter() - t1

    return {
        "prompt_length": prompt_length,
        "ttft_s": ttft_s,
        "total_time_s": total_time_s,
        "decode_throughput": 50 / (total_time_s - ttft_s),
        "peak_mem_gb": peak_mem_gb,
    }


def measure_model(checkpoint: str, prompts: list[str], device: str = "cuda") -> list[dict]:
    """Load one model, measure across all prompts, free VRAM, return measurements."""
    logger.info("Loading model: %s", checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.float16).to(device)
    weight_mem_gb = torch.cuda.memory_allocated() / 1024**3
    logger.info("Model weights: %.4f GB", weight_mem_gb)

    # Warmup — kills first-call JIT/cuDNN autotune noise.
    warmup_inputs = tokenizer("Hello there!", return_tensors="pt").to(device)
    _ = model(**warmup_inputs)

    measurements: list[dict] = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        result = measure_one_prompt(model, inputs, tokenizer)
        result["model"] = checkpoint
        result["weight_mem_gb"] = weight_mem_gb
        result["overhead_gb"] = result["peak_mem_gb"] - weight_mem_gb
        result["timestamp"] = datetime.now().isoformat(timespec="seconds")
        measurements.append(result)

        log_measurement_table(result)

    # Free VRAM before next model.
    del model
    del tokenizer
    torch.cuda.empty_cache()

    return measurements


def log_measurement_table(m: dict) -> None:
    """Log a single measurement as a GitHub-style table (in-run feedback)."""
    table = [
        ["Prompt length (tokens)", m["prompt_length"]],
        ["TTFT (s)", f"{m['ttft_s']:.6f}"],
        ["Total gen time (s)", f"{m['total_time_s']:.6f}"],
        ["Decode throughput (tok/s)", f"{m['decode_throughput']:.2f}"],
        ["Model weight mem (GB)", f"{m['weight_mem_gb']:.6f}"],
        ["Peak inference mem (GB)", f"{m['peak_mem_gb']:.6f}"],
        ["Inference overhead (GB)", f"{m['overhead_gb']:.6f}"],
    ]
    logger.info(
        "\n[%s] prompt_len=%d\n%s",
        m["model"],
        m["prompt_length"],
        tabulate(table, headers=["Metric", "Value"], tablefmt="github"),
    )


# ---------------------------------------------------------------------------
# PHASE 2: RECORD
# ---------------------------------------------------------------------------


def write_csv(measurements: list[dict], path: Path) -> None:
    """Write measurements to a CSV file. Creates parent dir if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(measurements[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(measurements)
    logger.info("Wrote CSV: %s (%d rows)", path, len(measurements))


# ---------------------------------------------------------------------------
# PHASE 3: PRESENT
# ---------------------------------------------------------------------------


def plot_ttft_vs_prompt_length(measurements: list[dict], path: Path) -> None:
    """Line plot: TTFT (ms) vs prompt length, one line per model."""
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    for model_name in sorted({m["model"] for m in measurements}):
        rows = sorted(
            [m for m in measurements if m["model"] == model_name],
            key=lambda r: r["prompt_length"],
        )
        xs = [r["prompt_length"] for r in rows]
        ys = [r["ttft_s"] * 1000 for r in rows]  # seconds → ms
        plt.plot(
            xs,
            ys,
            marker="o",
            linewidth=2,
            markersize=8,
            color=MODEL_COLORS.get(model_name),
            label=model_name,
        )

    plt.xlabel("Prompt length (tokens)")
    plt.ylabel("TTFT (ms)")
    plt.title("Time to First Token vs Prompt Length")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved plot: %s", path)


def plot_throughput_vs_model(measurements: list[dict], path: Path) -> None:
    """Bar chart: mean decode throughput per model (averaged across prompt lengths)."""
    path.parent.mkdir(parents=True, exist_ok=True)

    models = sorted({m["model"] for m in measurements})
    avg_throughputs = []
    for model_name in models:
        rows = [m for m in measurements if m["model"] == model_name]
        avg = sum(r["decode_throughput"] for r in rows) / len(rows)
        avg_throughputs.append(avg)

    colors = [MODEL_COLORS.get(m, "#888888") for m in models]
    plt.figure(figsize=(8, 5))
    plt.bar(models, avg_throughputs, color=colors)
    plt.ylabel("Decode throughput (tok/s)")
    plt.title("Mean decode throughput by model")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved plot: %s", path)


def plot_memory_breakdown(measurements: list[dict], path: Path) -> None:
    """Stacked bar: weights vs inference overhead, per model (longest prompt)."""
    path.parent.mkdir(parents=True, exist_ok=True)

    models = sorted({m["model"] for m in measurements})
    weight_mem = []
    overhead_mem = []
    for model_name in models:
        rows = [m for m in measurements if m["model"] == model_name]
        longest = max(rows, key=lambda r: r["prompt_length"])
        weight_mem.append(longest["weight_mem_gb"])
        # Clamp negative overhead to 0 — measurement artifact, not real signal.
        overhead_mem.append(max(0.0, longest["overhead_gb"]))

    plt.figure(figsize=(8, 5))
    plt.bar(models, weight_mem, label="Weights", color="#1f77b4")
    plt.bar(
        models,
        overhead_mem,
        bottom=weight_mem,
        label="Inference overhead (KV cache + activations)",
        color="#ff7f0e",
    )
    plt.ylabel("GPU memory (GB)")
    plt.title("Memory breakdown at longest prompt per model")
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved plot: %s", path)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


@torch.inference_mode()
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    print_environment()

    # === PHASE 1: MEASURE ===
    logger.info("=== Phase 1: MEASURE ===")
    all_measurements: list[dict] = []
    for checkpoint in MODEL_LIST:
        all_measurements.extend(measure_model(checkpoint, INPUT_TEXTS))

    # === PHASE 2: RECORD ===
    logger.info("=== Phase 2: RECORD ===")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"run_{timestamp}.csv"
    write_csv(all_measurements, csv_path)

    # === PHASE 3: PRESENT ===
    logger.info("=== Phase 3: PRESENT ===")
    plot_ttft_vs_prompt_length(all_measurements, RESULTS_DIR / "ttft_vs_prompt.png")
    plot_throughput_vs_model(all_measurements, RESULTS_DIR / "throughput_vs_model.png")
    plot_memory_breakdown(all_measurements, RESULTS_DIR / "memory_breakdown.png")

    logger.info(
        "Done. %d measurements across %d models. Outputs in %s.",
        len(all_measurements),
        len(MODEL_LIST),
        RESULTS_DIR,
    )


if __name__ == "__main__":
    main()
