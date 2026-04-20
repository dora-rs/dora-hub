"""TODO: Add docstring."""

import os
import re

import pyarrow as pa
import torch
from dora import Node
from kokoro import KPipeline
from kokoro_onnx import Kokoro
from misaki import zh

device = "cuda" if torch.cuda.is_available() else "cpu"
REPO_ID = os.getenv("REPO_ID", "hexgrad/Kokoro-82M")

LANGUAGE = os.getenv("LANGUAGE", "a")
VOICE = os.getenv("VOICE", "af_heart")


def main():
    """TODO: Add docstring."""
    # Set up pipelines for English and Chinese
    pipeline = KPipeline(
        lang_code=LANGUAGE,
        repo_id=REPO_ID,
    )  # <= make sure lang_code matches voice

    kokoro = Kokoro(
        "/Users/xaviertao/.cache/modelscope/hub/models/iic/test/model_fp16.onnx",
        "/Users/xaviertao/.cache/modelscope/hub/models/iic/test/voices-v1.0.bin",
    )
    g2p = zh.ZHG2P(version="1.0")
    g2p("测试一下")  # warm up g2p
    # warm up voice
    _generator = pipeline(
        "hello",
        voice=VOICE,
        speed=1.2,
        # split_pattern=r"\n+",
    )

    node = Node()

    for event in node:
        if event["type"] == "INPUT" and event["id"] == "text":
            text = event["value"][0].as_py()

            if "<tool_call>" in text:
                # Remove everything between <tool_call> and </tool_call>
                text = re.sub(
                    r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL
                ).strip()
                if text == "":
                    continue
            # Split text with point or comma even chinese version
            texts = re.sub(r"([。.，?!:])", r"\1\n", text)

            for text in texts.split("\n"):
                # if text is empty, skip
                if text.strip() in ["", ".", '"']:
                    continue
                print(f"Processing text: {text}")
                # Skip if text start with <tool_call>
                if re.findall(r"[\u4e00-\u9fff]+", text):
                    phonemes, _ = g2p(text)
                    sample, sr = kokoro.create(
                        phonemes,
                        voice=VOICE,
                        speed=1.1,
                        is_phonemes=True,
                        lang="cmn",
                    )
                else:
                    sample, sr = kokoro.create(
                        text,
                        voice=VOICE,
                        speed=1.1,
                        # split_pattern=r"\n+",
                    )
                node.send_output("audio", pa.array(sample.ravel()), {"sample_rate": sr})


if __name__ == "__main__":
    main()
