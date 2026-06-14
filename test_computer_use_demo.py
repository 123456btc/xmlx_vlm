#!/usr/bin/env python3
"""Safe demo to test whether computer_use can load its model and reason about the screen.

This demo loads the configured GUI agent model, takes a screenshot, asks the model
to describe the screen, and prints the raw response. It does NOT execute any mouse
or keyboard actions, so it is safe to run on your real machine.
"""
import os
import sys
import time
from pathlib import Path

# Ensure local source tree is used
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import mlx.core as mx
from PIL import ImageGrab

from xmlx_vlm import load, generate
from xmlx_vlm.computer_use.gui_agent import (
    GUI_MODEL,
    min_pixels,
    max_pixels,
    system_prompt,
)


def main():
    print(f"Testing computer_use with model: {GUI_MODEL}")
    print("Loading model...")
    try:
        model, processor = load(
            GUI_MODEL,
            tokenizer_config={"min_pixels": min_pixels, "max_pixels": max_pixels},
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"FAILED to load model: {e}")
        raise

    print("Taking screenshot...")
    time.sleep(1)
    screenshot = ImageGrab.grab()
    print(f"Screenshot size: {screenshot.size}")

    query = "Describe what you see on the screen in one sentence."
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": system_prompt},
                {"type": "text", "text": f"Task: {query}"},
                {"type": "text", "text": "Past actions: []"},
                {"type": "image", "min_pixels": min_pixels, "max_pixels": max_pixels},
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    print("Running inference...")
    try:
        response = generate(
            model,
            processor,
            prompt,
            screenshot,
            temperature=0.1,
            max_tokens=1000,
            verbose=False,
        )
        mx.metal.clear_cache()
        print("\n=== Raw model response ===")
        print(response)
        print("==========================\n")

        # Try to parse the action dict (computer_use expects this format)
        try:
            parsed = eval(response)
            print("Parsed action:", parsed)
        except Exception as parse_err:
            print(f"Note: model did not return a valid Python dict: {parse_err}")
            print("This may mean the model is not trained for GUI action output.")

        print("Demo completed successfully.")
    except Exception as e:
        print(f"FAILED during inference: {e}")
        raise


if __name__ == "__main__":
    main()
