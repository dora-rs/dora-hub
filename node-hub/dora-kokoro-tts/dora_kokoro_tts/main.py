"""TODO: Add docstring."""

import os
import re

import pyarrow as pa
import torch
from dora import Node
from kokoro import KPipeline

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
    # warm up voice
    generator = pipeline(
        "hello",
        voice=VOICE,
        speed=1.2,
        split_pattern=r"\n+",
    )
    for _, (_, _, audio) in enumerate(generator):
        pass
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
                texts = re.sub(r"([。.，?!:])", r"\1\n", text)

                for text in texts.split("\n"):
                    if text.strip() == "":
                        continue
                    # Skip if text start with <tool_call>
                    if (
                        re.findall(r"[\u4e00-\u9fff]+", text)
                        and pipeline.lang_code != "z"
                    ):
                        pipeline = KPipeline(repo_id=REPO_ID, lang_code="z")
                    elif (
                        not re.findall(r"[\u4e00-\u9fff]+", text)
                        and pipeline.lang_code == "z"
                    ):
                        pipeline = KPipeline(
                            repo_id=REPO_ID, lang_code=LANGUAGE
                        )  # reset to default

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
