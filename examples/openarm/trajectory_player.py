"""Dora node: trajectory playback for motion planner output.

Receives a flattened joint trajectory array and plays it back one waypoint
per tick (10Hz).  Loops once then holds the final position.

Inputs:
    joint_trajectory: flattened float32 array with metadata (num_waypoints, num_joints)
    tick: timer trigger for playback rate

Outputs:
    joint_command: single waypoint (num_joints,) float32 with encoding="jointstate"
"""

import numpy as np
import pyarrow as pa
from dora import Node


def main():
    node = Node()

    trajectory = None  # (T, J) numpy array
    step = 0
    playing = False

    for event in node:
        if event["type"] == "INPUT":
            event_id = event["id"]
            metadata = event["metadata"]

            if event_id == "joint_trajectory":
                num_waypoints = int(metadata.get("num_waypoints", 0))
                num_joints = int(metadata.get("num_joints", 7))

                if num_waypoints == 0:
                    continue

                traj_flat = event["value"].to_numpy().astype(np.float32)
                trajectory = traj_flat.reshape(num_waypoints, num_joints)
                step = 0
                playing = True
                print(
                    f"[trajectory-player] Received trajectory: "
                    f"{num_waypoints} waypoints, {num_joints} joints"
                )

            elif event_id == "tick" and playing and trajectory is not None:
                if step < len(trajectory):
                    waypoint = trajectory[step]
                    out_metadata = metadata.copy()
                    out_metadata["encoding"] = "jointstate"
                    node.send_output(
                        "joint_command",
                        pa.array(waypoint, type=pa.float32()),
                        metadata=out_metadata,
                    )
                    step += 1
                else:
                    # Hold final position
                    waypoint = trajectory[-1]
                    out_metadata = metadata.copy()
                    out_metadata["encoding"] = "jointstate"
                    node.send_output(
                        "joint_command",
                        pa.array(waypoint, type=pa.float32()),
                        metadata=out_metadata,
                    )
                    playing = False
                    print("[trajectory-player] Trajectory complete, holding final position")

        elif event["type"] == "STOP":
            break


if __name__ == "__main__":
    main()
