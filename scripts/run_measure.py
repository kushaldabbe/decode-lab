#!/usr/bin/env python
"""Standalone measurement script — call this from the command line once you've
implemented the decode_lab package.

Example usage (after implementation):
    python scripts/run_measure.py --model HuggingFaceTB/SmolLM2-135M \
        --prompt "The future of AI is" --max-new-tokens 50

For now this is a stub showing the argparse pattern.
"""

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single decode measurement.")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M", help="HF model id")
    parser.add_argument("--prompt", default="The future of AI is", help="Input prompt")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--device", default="cuda", help="cuda | cpu | mps")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    logger.info("Args: %s", args)

    # TODO: implement
    # from decode_lab import ...
    # load model, run prefill + decode, measure, print table, save CSV to results/
    raise NotImplementedError("You will implement this as part of decode-lab.")


if __name__ == "__main__":
    main()
