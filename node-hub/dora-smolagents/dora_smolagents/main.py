import os
import json
import re
import pyarrow as pa
from dora import Node
from smolagents import CodeAgent, Tool
from smolagents.models import LiteLLMModel

class MoveForwardTool(Tool):
    name = "move_forward"
    description = "Move forward"
    inputs = {}
    output_type = "object"
    def forward(self):
        return {"action": "move_forward"}

class StopTool(Tool):
    name = "stop_motion"
    description = "Stop"
    inputs = {}
    output_type = "object"
    def forward(self):
        return {"action": "stop_motion"}

model = LiteLLMModel(
    model_id=os.getenv("SMOLAGENTS_MODEL", "ollama/mistral"),
    api_base=os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
)

agent = CodeAgent(
    tools=[MoveForwardTool(), StopTool()],
    model=model,
    max_steps=1
)

def handle_event(event):
    if event["type"] == "INPUT" and event["id"] == "command":
        command = event["value"][0].as_py()
        command = f'Return ONLY JSON: {{"action": "move_forward"}} or {{"action": "stop_motion"}}. Command: {command}'

        print("[TRACE] input:", command)

        result = agent.run(command)

        # Extract JSON
        if isinstance(result, str):
            match = re.search(r"{.*}", result)
            if match:
                try:
                    action = json.loads(match.group())
                except:
                    action = {"action": "unknown"}
            else:
                action = {"action": "unknown"}
        elif isinstance(result, dict):
            action = result
        else:
            action = {"action": "unknown"}

        # MAP → SPEED (correct place)
        if action.get("action") == "move_forward":
            action = {"speed": 1.0}
        elif action.get("action") == "stop_motion":
            action = {"speed": 0.0}
        else:
            action = {"speed": 0.0}

        print("[TRACE] output:", action)

        return {
            "id": "action",
            "value": pa.array([json.dumps(action)])
        }

def main():
    node = Node("smol_agent")
    for event in node:
        result = handle_event(event)
        if result:
            node.send_output(result["id"], result["value"])

if __name__ == "__main__":
    main()
