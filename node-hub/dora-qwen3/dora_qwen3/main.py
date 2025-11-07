"""TODO: Add docstring."""

import json
import os
import sys

import pyarrow as pa
from dora import Node
from mlx_lm import load, generate, stream_generate

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You're a very succinct AI assistant with short answers.",
)

MODEL_NAME_OR_PATH = os.getenv("MODEL_NAME_OR_PATH", "mlx-community/Qwen3-8B-4bit")
model, tokenizer = load(MODEL_NAME_OR_PATH)
# warm up model
messages = [{"role": "user", "content": "hello world!"}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

response = generate(model, tokenizer, prompt=prompt, verbose=False)


ACTIVATION_WORDS = os.getenv("ACTIVATION_WORDS", "").split()


def main():
    """TODO: Add docstring."""
    history = []
    tools = None

    node = Node()

    for event in node:
        if event["type"] == "INPUT":
            # Warning: Make sure to add my_output_id and my_input_id within the dataflow.
            texts = event["value"].to_pylist()

            if "system_prompt" in event["id"]:
                history += [{"role": "system", "content": texts[0]}]
                continue
            if "tools" in event["id"]:
                tools = json.loads(texts[0])
                continue
            for text in texts:
                if text.startswith("<|system|>\n"):
                    history.append(
                        {
                            "role": "system",
                            "content": text.replace("<|system|>\n", ""),
                        }
                    )
                elif text.startswith("<|assistant|>\n"):
                    history.append(
                        {
                            "role": "assistant",
                            "content": text.replace("<|assistant|>\n", ""),
                        }
                    )
                elif text.startswith("<|tool|>\n"):
                    history.append(
                        {
                            "role": "tool",
                            "content": text.replace("<|tool|>\n", ""),
                        }
                    )
                elif text.startswith("<|user|>\n<|im_start|>\n"):
                    history.append(
                        {
                            "role": "user",
                            "content": text.replace("<|user|>\n<|im_start|>\n", "")
                            + " /no_think",
                        }
                    )
                    # If the last message was from the user, append the image URL to it

                else:
                    history.append(
                        {
                            "role": "user",
                            "content": text + " /no_think",
                        }
                    )

            words = text.lower().split()

            tmp_tools = event["metadata"].get("tools")
            tool_choice = event["metadata"].get("tool_choice", "auto")
            tmp_tools = json.loads(tmp_tools) if tmp_tools is not None else tools

            if len(ACTIVATION_WORDS) == 0 or any(
                word in ACTIVATION_WORDS for word in words
            ):
                # On linux, Windows
                if sys.platform == "darwin":
                    print(history)
                    prompt = tokenizer.apply_chat_template(
                        history  # Prompt
                        # tools=tools,
                        # tool_choice=tool_choice,
                    )
                    stream_params = {
                        "model": model,
                        "tokenizer": tokenizer,
                        "prompt": prompt,
                    }

                    accumulated_response = ""
                    chunk_buffer = ""
                    segment_index = 0
                    metadata = event["metadata"]
                    for token_response in stream_generate(**stream_params):
                        # Extract token text (stream_generate returns GenerationResponse objects)
                        token_text = (
                            token_response.text
                            if hasattr(token_response, "text")
                            else str(token_response)
                        )
                        chunk_buffer += token_text

                        # Check if we should send a chunk (on punctuation or size)
                        should_send = False

                        chunk_buffer = chunk_buffer.replace(
                            "<|im_start|>Assistant .", ""
                        )
                        if any(
                            p in chunk_buffer
                            for p in ["。", "！", "？", ".", "!", "?", "，", ",", "\n"]
                        ):
                            # Send on punctuation for more natural breaks
                            should_send = True

                        if should_send and chunk_buffer.strip():
                            # Clean chunk before sending
                            clean_chunk = chunk_buffer.strip()

                            # Remove dashes from output
                            clean_chunk = clean_chunk.replace("-", "")

                            # Skip if it's part of thinking tags
                            if "<think>" in clean_chunk or "</think>" in clean_chunk:
                                chunk_buffer = ""
                                continue

                            accumulated_response += clean_chunk
                            # Send chunk to TTS with metadata (including question_id)
                            node.send_output(
                                output_id="text",
                                data=pa.array([clean_chunk]),
                                metadata=metadata or {},
                            )

                            segment_index += 1
                            chunk_buffer = ""

                    # Send any remaining buffer
                    if chunk_buffer.strip():
                        print("hello world2")
                        clean_final_chunk = chunk_buffer.strip().replace("-", "")
                        accumulated_response += clean_chunk
                        node.send_output(
                            output_id="text",
                            data=pa.array([clean_final_chunk]),
                            metadata=metadata or {},
                        )
                    history += [{"role": "assistant", "content": accumulated_response}]


if __name__ == "__main__":
    main()
