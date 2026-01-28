"""TODO: Add docstring."""

import json
import os
import sys

import pyarrow as pa
from dora import Node
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You're a very succinct AI assistant with short answers.",
)

MODEL_NAME_OR_PATH = os.getenv("MODEL_NAME_OR_PATH", "Qwen/Qwen2.5-0.5B-Instruct-GGUF")
MODEL_FILE_PATTERN = os.getenv("MODEL_FILE_PATTERN", "*fp16.gguf")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", "0"))
N_THREADS = int(os.getenv("N_THREADS", "4"))
CONTEXT_SIZE = int(os.getenv("CONTEXT_SIZE", "4096"))
TOOLS_JSON = os.getenv("TOOLS_JSON")
tools = json.loads(TOOLS_JSON) if TOOLS_JSON is not None else None


def get_model_gguf():
    """TODO: Add docstring."""
    from llama_cpp import Llama

    return Llama.from_pretrained(
        repo_id=MODEL_NAME_OR_PATH,
        filename=MODEL_FILE_PATTERN,
        n_gpu_layers=N_GPU_LAYERS,
        n_ctx=CONTEXT_SIZE,
        n_threads=N_THREADS,
        verbose=False,
    )


def get_model_darwin():
    """TODO: Add docstring."""
    from mlx_lm import load

    model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-8bit")
    return model, tokenizer


def get_model_huggingface():
    """TODO: Add docstring."""
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


ACTIVATION_WORDS = os.getenv("ACTIVATION_WORDS", "").split()


def generate_hf(model, tokenizer, prompt: str, history) -> str:
    """TODO: Add docstring."""
    history += [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        history,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    history += [{"role": "assistant", "content": response}]
    return response, history


def main():
    """TODO: Add docstring."""
    history = []
    tools = None
    # If OS is not Darwin, use Huggingface model
    if sys.platform == "darwin":
        model = get_model_gguf()
    elif sys.platform == "linux":
        model, tokenizer = get_model_huggingface()
    else:
        model, tokenizer = get_model_darwin()

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
                            "content": [
                                {
                                    "type": "text",
                                    "text": text.replace("<|system|>\n", ""),
                                },
                            ],
                        }
                    )
                elif text.startswith("<|assistant|>\n"):
                    history.append(
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "text",
                                    "text": text.replace("<|assistant|>\n", ""),
                                },
                            ],
                        }
                    )
                elif text.startswith("<|tool|>\n"):
                    history.append(
                        {
                            "role": "tool",
                            "content": [
                                {
                                    "type": "text",
                                    "text": text.replace("<|tool|>\n", ""),
                                },
                            ],
                        }
                    )
                elif text.startswith("<|user|>\n<|im_start|>\n"):
                    history.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": text.replace(
                                        "<|user|>\n<|im_start|>\n", ""
                                    ),
                                },
                            ],
                        }
                    )
                elif text.startswith("<|user|>\n<|vision_start|>\n"):
                    # Handle the case where the text starts with <|user|>\n<|vision_start|>
                    image_url = text.replace("<|user|>\n<|vision_start|>\n", "")

                    # If the last message was from the user, append the image URL to it
                    if history[-1]["role"] == "user":
                        history[-1]["content"].append(
                            {
                                "type": "image",
                                "image": image_url,
                            }
                        )
                    else:
                        history.append(
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "image": image_url,
                                    },
                                ],
                            }
                        )
                else:
                    history.append(
                        {
                            "role": "user",
                            "content": text,
                        }
                    )

            words = text.lower().split()

            tmp_tools = event["metadata"].get("tools")
            tool_choice = event["metadata"].get("tool_choice", "auto")
            tmp_tools = json.loads(tmp_tools) if tmp_tools is not None else tools

            if len(ACTIVATION_WORDS) == 0 or any(
                word in ACTIVATION_WORDS for word in words
            ):
                response = model.create_chat_completion(
                    messages=history,  # Prompt
                    max_tokens=150,
                    tools=tools,
                    tool_choice=tool_choice,
                    stream=True,
                )

                accumulated_response = ""
                chunk_buffer = ""
                tool_buffer = ""
                tool_call = False
                segment_index = 0
                for chunk in response:
                    delta = chunk["choices"][0]["delta"]
                    if "content" not in delta:
                        continue

                    chunk_buffer += delta["content"]

                    # Check if we should send a chunk (on punctuation or size)
                    should_send = False

                    chunk_buffer = chunk_buffer.replace("<|im_start|>Assistant .", "")
                    if any(
                        p in chunk_buffer
                        for p in ["。", "！", "？", ".", "!", "?", "，", ",", "\n"]
                    ):
                        # Send on punctuation for more natural breaks
                        should_send = True

                    if should_send and chunk_buffer:
                        # Clean chunk before sending
                        clean_chunk = chunk_buffer

                        # Remove dashes from output
                        clean_chunk = clean_chunk.replace("-", "")

                        # Skip if it's part of thinking tags
                        if "<think>" in clean_chunk or "</think>" in clean_chunk:
                            chunk_buffer = ""
                            continue
                        elif "<tool_call>" in chunk_buffer or tool_call:
                            tool_buffer += chunk_buffer
                            tool_call = True
                            continue

                        if "</tool_call>" in chunk_buffer:
                            tool_call = False
                            node.send_output(
                                output_id="text",
                                data=pa.array(
                                    [tool_buffer.partition("</tool_call>")[0]]
                                ),
                            )
                            chunk_buffer = chunk_buffer.partition("</tool_call>")[1]
                            continue
                        accumulated_response += clean_chunk
                        # Send chunk to TTS with metadata (including question_id)
                        node.send_output(
                            output_id="text",
                            data=pa.array([clean_chunk]),
                        )

                        segment_index += 1
                        chunk_buffer = ""

                # Send any remaining buffer
                if chunk_buffer.strip():
                    clean_final_chunk = chunk_buffer.strip().replace("-", "")
                    accumulated_response += clean_final_chunk
                    node.send_output(
                        output_id="text",
                        data=pa.array([clean_final_chunk]),
                    )
                history += [{"role": "assistant", "content": accumulated_response}]


if __name__ == "__main__":
    main()
