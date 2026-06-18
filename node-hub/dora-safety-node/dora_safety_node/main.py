import pyarrow as pa
from dora import Node
import json

MAX_SPEED = 1.0

def handle_event(event):
    if event["type"] == "INPUT" and event["id"] == "cmd_vel":
        data = event["value"]

        raw = data[0].as_py()
        print("RAW INPUT:", raw)

        try:
            parsed = json.loads(raw)
            if isinstance(parsed, str):
                parsed = json.loads(parsed)
            cmd = parsed
        except Exception as e:
            print("PARSE ERROR:", e)
            cmd = {"speed": 0}

        print("PARSED CMD:", cmd)

        speed = float(cmd.get("speed", 0))
        safe_speed = min(max(speed, -MAX_SPEED), MAX_SPEED)

        print("SAFE OUTPUT:", safe_speed)

        return {
            "id": "safe_cmd_vel",
            "value": pa.array([json.dumps({"speed": safe_speed, "reason": "safe"})])
        }

def main():
    node = Node("safety_node")

    for event in node:
        result = handle_event(event)
        if result:
            node.send_output(result["id"], result["value"])

if __name__ == "__main__":
    main()
