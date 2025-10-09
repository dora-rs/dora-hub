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
                            "content": [
                                {"type": "text", "text": text},
                            ],
                        }
                    )

            words = text.lower().split()

            if "system_prompt" in event["id"]:
                history += [{"role": "system", "content": text}]
                print(f"System prompt set to: {text}")
                continue
            if "tools" in event["id"]:
                tools = json.loads(text)
                print(f"Tools set to: {tools}")
                continue

            tmp_tools = event["metadata"].get("tools")
            tmp_tools = json.loads(tmp_tools) if tmp_tools is not None else tools

            if len(ACTIVATION_WORDS) == 0 or any(
                word in ACTIVATION_WORDS for word in words
            ):
                print(f"Received input: {text}")
                # On linux, Windows
                if sys.platform == "darwin":
                    response = model.create_chat_completion(
                        messages=[{"role": "user", "content": text}],  # Prompt
                        max_tokens=200,
                        tools=tools,
                    )["choices"][0]["message"]["content"]
                elif sys.platform == "linux":
                    response, history = generate_hf(model, tokenizer, text, history)
                else:
                    from mlx_lm import generate

                    response = generate(
                        model,
                        tokenizer,
                        prompt=text,
                        verbose=False,
                        max_tokens=50,
                    )

                node.send_output(
                    output_id="text",
                    data=pa.array([response]),
                    metadata={},
                )


if __name__ == "__main__":
    main()
