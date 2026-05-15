"""Dora node: trajectory playback for motion planner output.

Receives a flattened joint trajectory array and plays it back one waypoint
per tick (10Hz).  Loops once then holds the final position.

Supports gripper actions: the trajectory metadata may contain a
``gripper_actions`` JSON string with a list of ``{"waypoint": N, "rad": float}``.
When the playback step reaches or passes a waypoint, the player sends a
``gripper_command`` output with the specified angle.

Inputs:
    joint_trajectory: flattened float32 array with metadata (num_waypoints, num_joints)
    tick: timer trigger for playback rate

Outputs:
    joint_command: single waypoint (num_joints,) float32 with encoding="jointstate"
    gripper_command: float32 array [rad] — sent at specific trajectory waypoints
"""

import json

import numpy as np
import pyarrow as pa
from dora import Node


def main():
    node = Node()

    trajectory = None  # (T, J) numpy array
    step = 0
    playing = False
    gripper_actions = []  # list of {"waypoint": int, "rad": float}
    gripper_fired = set()
    play_arm = "left"

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
                play_arm = metadata.get("arm", "left")

                # Parse gripper actions from metadata
                ga_str = metadata.get("gripper_actions", "")
                if ga_str:
                    try:
                        gripper_actions = json.loads(ga_str) if isinstance(ga_str, str) else ga_str
                    except (json.JSONDecodeError, TypeError):
                        gripper_actions = []
                else:
                    gripper_actions = []
                gripper_fired = set()

                print(
                    f"[trajectory-player] Received trajectory: "
                    f"{num_waypoints} waypoints, {num_joints} joints"
                )
                if gripper_actions:
                    print(f"[trajectory-player] Gripper actions: {gripper_actions}")

            elif event_id == "tick" and playing and trajectory is not None:
                if step < len(trajectory):
                    waypoint = trajectory[step]
                    out_metadata = metadata.copy()
                    out_metadata["encoding"] = "jointstate"
                    out_metadata["arm"] = play_arm
                    node.send_output(
                        "joint_command",
                        pa.array(waypoint, type=pa.float32()),
                        metadata=out_metadata,
                    )

                    # Fire gripper commands at the right waypoints
                    for action in gripper_actions:
                        wp = action["waypoint"]
                        if step >= wp and wp not in gripper_fired:
                            gripper_fired.add(wp)
                            rad = float(action["rad"])
                            print(
                                f"[trajectory-player] Gripper command: rad={rad:.3f} "
                                f"at step {step} (trigger wp={wp})"
                            )
                            node.send_output(
                                "gripper_command",
                                pa.array([rad], type=pa.float32()),
                                metadata={"arm": play_arm},
                            )

                    step += 1
                else:
                    # Hold final position
                    waypoint = trajectory[-1]
                    out_metadata = metadata.copy()
                    out_metadata["encoding"] = "jointstate"
                    out_metadata["arm"] = play_arm
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
