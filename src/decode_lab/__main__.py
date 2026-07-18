"""Entry point for `python -m decode_lab`.

This is intentionally a stub. You will implement the actual decode + measurement
logic yourself, following the project README and your notes.

Suggested implementation order:
1. Print environment info (this file, partially done below).
2. Load a small model + tokenizer (e.g., HuggingFaceTB/SmolLM2-135M).
3. Tokenize a prompt and time it.
4. Run model.forward once on the prompt (this is the PREFILL) and time it.
5. Loop: feed last token back in, generate next token (this is the DECODE).
6. Measure TTFT, decode throughput, total latency, peak GPU memory.

Rules:
- Write the code yourself. Use AI only for explanation, debugging hints, review.
- Add type hints to every function signature.
- Use `logging` (or `structlog`) for measurement output, not bare print statements.
"""

from __future__ import annotations

import logging

import torch

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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print_environment()

    # TODO: your implementation goes here.
    # See README.md "Definition of done" for what to build.
    logger.warning("decode_lab: not yet implemented — see src/decode_lab/__main__.py")


if __name__ == "__main__":
    main()
