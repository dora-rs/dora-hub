import os
import json
from smolagents import CodeAgent, Tool
from smolagents.models import LiteLLMModel

# --- Tools ---
class MoveForwardTool(Tool):
    name = "move_forward"
    description = "Move the robot forward"
    inputs = {}
    output_type = "object"

    def forward(self):
        return {"action": "move_forward"}

class StopTool(Tool):
    name = "stop_motion"
    description = "Stop the robot"
    inputs = {}
    output_type = "object"

    def forward(self):
        return {"action": "stop"}

# --- Agent setup ---
model = LiteLLMModel(
    model=os.getenv("SMOLAGENTS_MODEL", "ollama/mistral"),
    api_base=os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
)

agent = CodeAgent(
    tools=[MoveForwardTool(), StopTool()],
    model=model,
    max_steps=1
)

# --- Dora node ---
def handle_event(event):
    if event["type"] == "INPUT" and event["id"] == "command":
        command = event["value"]
        print(f"[TRACE] agent.input: {command}")

        import time
        start = time.perf_counter()
        result = agent.run(command)
        trace_output = json.dumps(result) if isinstance(result, dict) else result
        print(f"[TRACE] agent.output: {trace_output}")
        end = time.perf_counter()
        print(f"[TRACE] agent.latency: {end - start:.3f}s")

        # Extract clean action (fallback safe)
        if isinstance(result, dict):
            action = result
        else:
            action = {"action": "unknown"}

        return {
            "type": "OUTPUT",
            "id": "action",
            "value": json.dumps(action)
        }

from dora import Node

def main():
    node = Node("smol_agent")

    for event in node:
        result = handle_event(event)

        if result:
            node.send_output(result["id"], result["value"])


if __name__ == "__main__":
    main()
