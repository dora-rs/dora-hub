"""TODO: Add docstring."""

import os
import re

import pyarrow as pa
import torch
from dora import Node
from kokoro import KPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
REPO_ID = "hexgrad/Kokoro-82M"

LANGUAGE = os.getenv("LANGUAGE", "en")
VOICE = os.getenv("VOICE", "af_heart")


# HACK: Mitigate rushing caused by lack of training data beyond ~100 tokens
# Simple piecewise linear fn that decreases speed as len_ps increases
def speed_callable(len_ps):
    speed = 0.8
    if len_ps <= 83:
        speed = 1
    elif len_ps < 183:
        speed = 1 - (len_ps - 83) / 500
    return speed * 1.1


def main():
    """TODO: Add docstring."""
    # Set up pipelines for English and Chinese
    pipeline = KPipeline(
        lang_code="z",
        repo_id=REPO_ID,
    )  # <= make sure lang_code matches voice
    node = Node()

    for event in node:
        if event["type"] == "INPUT":
            if event["id"] == "text":
                text = event["value"][0].as_py()

                if "<tool_call>" in text:
                    # Remove everything between <tool_call> and </tool_call>
                    text = re.sub(
                        r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL
                    ).strip()
                    if text == "":
                        continue
                # Split text with point or comma even chinese version
                texts = re.sub(r"([。,.，?!:])", r"\1\n", text)
                for text in texts.split("\n"):
                    # Skip if text start with <tool_call>

                    generator = pipeline(
                        text,
                        voice=VOICE,
                        speed=1.2,
                        split_pattern=r"\n+",
                    )
                    for _, (_, _, audio) in enumerate(generator):
                        audio = audio.numpy()
                        node.send_output(
                            "audio", pa.array(audio), {"sample_rate": 24000}
                        )


if __name__ == "__main__":
    main()
