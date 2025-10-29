"""TODO: Add docstring."""

import re
import time

import numpy as np
import pyarrow as pa
from dora import Node

chunk_size = [0, 10, 5]  # [0, 10, 5] 600ms, [0, 8, 4] 480ms
encoder_chunk_look_back = 4  # number of chunks to lookback for encoder self-attention
decoder_chunk_look_back = (
    1  # number of encoder chunks to lookback for decoder cross-attention
)


def remove_text_noise(text: str, text_noise="") -> str:
    """Remove noise from text.

    Args:
        text (str): Original text
        text_noise (str): text to remove from the original text

    Returns:
        str: Cleaned text

    """
    # Handle the case where text_noise is empty
    if not text_noise.strip():
        return (
            text  # Return the original text if text_noise is empty or just whitespace
        )

    # Helper function to normalize text (remove punctuation, make lowercase, and handle hyphens)
    def normalize(s):
        # Replace hyphens with spaces to treat "Notre-Dame" and "notre dame" as equivalent
        s = re.sub(r"-", " ", s)
        # Remove other punctuation and convert to lowercase
        return re.sub(r"[^\w\s]", "", s).lower()

    # Normalize both text and text_noise
    normalized_text = normalize(text)
    normalized_noise = normalize(text_noise)

    # Split into words
    text_words = normalized_text.split()
    noise_words = normalized_noise.split()

    # Only remove parts of text_noise that are found in text
    cleaned_words = text_words[:]
    for noise_word in noise_words:
        if noise_word in cleaned_words:
            cleaned_words.remove(noise_word)

    # Reconstruct the cleaned text
    return " ".join(cleaned_words)


# Load model
def load_model():
    from funasr import AutoModel

    return AutoModel(
        model="paraformer-zh",
        punc_model="ct-punc",
        # spk_model="cam++",
        disable_update=True,
    )


BAD_SENTENCES = [
    "",
    " so",
    " So.",
    " so so",
    " What?",
    " We'll see you next time.",
    " I'll see you next time.",
    " We're going to come back.",
    " let's move on.",
    " Here we go.",
    " my",
    " All right. Thank you.",
    " That's what we're doing.",
    " That's what I wanted to do.",
    " I'll be back.",
    " Hold this. Hold this.",
    " Hold this one. Hold this one.",
    " And we'll see you next time.",
    " strength.",
    " Length.",
    "You",
    "You ",
    " You",
    "字幕",
    "字幕志愿",
    "中文字幕",
    "或或或或",
    "或",
    "我",
    "你",
    " you",
    "!",
    "THANK YOU",
    " Thank you.",
    " www.microsoft.com",
    " The",
    " BANG",
    " Silence.",
    " Sous-titrage Société Radio-Canada",
    " Sous",
    " Sous-",
    " i'm going to go to the next one.",
]


def cut_repetition(text, min_repeat_length=4, max_repeat_length=50):
    """TODO: Add docstring."""
    if len(text) == 0:
        return text
    # Check if the text is primarily Chinese (you may need to adjust this threshold)
    if sum(1 for char in text if "\u4e00" <= char <= "\u9fff") / len(text) > 0.5:
        # Chinese text processing
        for repeat_length in range(
            min_repeat_length,
            min(max_repeat_length, len(text) // 2),
        ):
            for i in range(len(text) - repeat_length * 2 + 1):
                chunk1 = text[i : i + repeat_length]
                chunk2 = text[i + repeat_length : i + repeat_length * 2]

                if chunk1 == chunk2:
                    return text[: i + repeat_length]
    else:
        # Non-Chinese (space-separated) text processing
        words = text.split()
        for repeat_length in range(
            min_repeat_length,
            min(max_repeat_length, len(words) // 2),
        ):
            for i in range(len(words) - repeat_length * 2 + 1):
                chunk1 = " ".join(words[i : i + repeat_length])
                chunk2 = " ".join(words[i + repeat_length : i + repeat_length * 2])

                if chunk1 == chunk2:
                    return " ".join(words[: i + repeat_length])

    return text


def main():
    """TODO: Add docstring."""
    text_noise = ""
    # For macos use mlx:
    model = load_model()

    node = Node()
    noise_timestamp = time.time()
    cache_audio = None
    for event in node:
        if event["type"] == "INPUT":
            if "text_noise" in event["id"]:
                text_noise = event["value"][0].as_py()
                text_noise = (
                    text_noise.replace("(", "")
                    .replace(")", "")
                    .replace("[", "")
                    .replace("]", "")
                )
                noise_timestamp = time.time()
            else:
                audio_input = event["value"].to_numpy()
                if cache_audio is not None:
                    audio = np.concatenate([cache_audio, audio_input])
                else:
                    audio = audio_input

                result = model.generate(
                    audio,
                )[0]

                text = result["text"]
                print(f"Raw text: {text}")
                text = text.replace(" ", "")
                if text.strip() == "" or text.strip() == ".":
                    continue

                node.send_output(
                    "text",
                    pa.array([text]),
                    {"language": "zh", "primitive": "text"},
                )
                node.send_output(
                    "speech_started",
                    pa.array([text]),
                )
                cache_audio = None
                audio = None
