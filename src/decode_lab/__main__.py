"""Entry point for `python -m decode_lab`.

Loads SmolLM2-135M (fp16) on CUDA and measures inference performance across
three prompt lengths (short / medium / long).

For each prompt, measures:
- TTFT (s): single `model(**inputs)` forward pass, timed with cuda.synchronize.
- Total generation time (s): `model.generate(max_new_tokens=50)`.
- Decode throughput (tok/s): `50 / (total - ttft)` — excludes prefill.

A warmup forward pass runs before measurement to eliminate first-call JIT
overhead. Results are logged as a GitHub-style table per prompt.
"""

from __future__ import annotations

import logging
import torch
import time

from transformers import AutoModelForCausalLM, AutoTokenizer
from tabulate import tabulate

logger = logging.getLogger(__name__)


def print_environment() -> None:
    """Print Python + torch + GPU info. Useful as a smoke test that setup works."""
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

@torch.inference_mode()
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print_environment()

    checkpoint = "HuggingFaceTB/SmolLM2-135M"
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, dtype=torch.float16)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.float16).to(device)

    # Warmup - GPU go brrrr
    inputs = tokenizer("Hello there!", return_tensors="pt").to(device)
    _ = model(**inputs)

    input_texts = [
        # Short (~5 tokens)
        "The cat sat quietly.",

        # Medium (~32 tokens)
        "Artificial intelligence is transforming software development by helping engineers write code, debug complex systems, understand unfamiliar repositories, and automate repetitive tasks, allowing teams to deliver features faster while maintaining high quality.",

        # Long (~128 tokens)
        "Large language models process text by first converting it into tokens, then performing a prefill stage that builds the attention cache from the prompt before entering the decode stage where new tokens are generated one at a time. Measuring metrics such as time to first token, decode latency, and output throughput helps engineers understand inference performance. Optimizing batching, cache reuse, memory bandwidth, quantization, and scheduling can significantly improve responsiveness while reducing infrastructure costs. Careful benchmarking across different prompt lengths and output sizes provides a realistic view of production behavior and reveals bottlenecks that synthetic microbenchmarks might otherwise miss."
    ]


    for prompt in input_texts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        # GPU work
        output = model(**inputs)
        next_token = output.logits[:,-1,:].argmax(dim=-1)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        ttft = t1 - t0

        # Total generation
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        out = model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()
        total_time = time.perf_counter() - t2

        # Decode througput
        decode_tp = 50/(total_time-ttft)

        table = [
            ["Prompt length (tokens)", inputs.input_ids.shape[-1]],
            ["TTFT (s)", ttft],
            ["Total gen time (s)", total_time],
            ["Decode throughput", decode_tp]
        ]
        logger.info("\n%s", tabulate(table, headers=["Metric", "Value"], tablefmt="github"))

if __name__ == "__main__":
    main()
