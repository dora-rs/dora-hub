"""TODO: Add docstring."""

from dora import Node
import time
import pyarrow as pa

node = Node()


for event in node:
    if event["type"] == "INPUT":
        actions = event["value"].to_numpy().copy().reshape((64, 14))

        for action in actions:
            gripper_left = action[6]
            gripper_right = action[13]
            action[13] = 0.3 if gripper_right < 0.45 else 0.6
            action[6] = 0.3 if gripper_left < 0.45 else 0.6
            node.send_output("jointstate_left", pa.array(action[:7], type=pa.float32()))
            node.send_output(
                "jointstate_right",
                pa.array(action[7:], type=pa.float32()),
            )
            time.sleep(0.02)
        print(actions)
