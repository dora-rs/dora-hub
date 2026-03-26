# dora-smolagents

A minimal Dora node integrating smolagents with a local LLM (Ollama).

## Features
- Accepts natural language commands (agent/command)
- Uses smolagents for reasoning
- Executes tool-based actions
- Outputs structured JSON (agent/action)
- LLM-agnostic via LiteLLM

## Example
Input:
move forward

Output:
{"action": "move_forward"}

## Setup
pip install smolagents litellm
ollama serve
ollama pull mistral

## Run
dora run dataflow.yml

## Architecture
agent/command -> smolagents node -> agent/action

## Notes
- Uses local LLM (Ollama), no API key required
- Easily extensible with new tools and models

