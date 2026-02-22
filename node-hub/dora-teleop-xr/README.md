# dora-teleop-xr

A Dora adapter node that provides teleoperation functionality using [teleop-xr](https://github.com/qrafty-ai/teleop_xr).

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Dora Framework 0.3.6 or higher
- A compatible VR headset or WebXR-enabled mobile device.
- `dora-rs-cli` (dev dependency for deployment).

### Installation

Install the package in editable mode:
```bash
uv pip install -e .
```

### Example
See [so101_ik_demo](../../examples/teleop-xr/so101_ik_demo.yml).

### Outputs

The node publishes 9 outputs based on the XR session state:

| Output | Type | Description |
|--------|------|-------------|
| `head_pose` | `float32[7]` | Head pose `[x, y, z, qx, qy, qz, qw]`. Metadata: `primitive: "pose"`. |
| `left_hand_pose` | `float32[7]` | Left controller/hand grip pose. Metadata: `primitive: "pose"`. |
| `right_hand_pose` | `float32[7]` | Right controller/hand grip pose. Metadata: `primitive: "pose"`. |
| `joint_state` | `JSON string` | Computed or received robot joint states. Metadata: `primitive: "jointstate"`. |
| `left_buttons` | `float64[]` | Flattened list of `[pressed, value]` pairs for all buttons. |
| `right_buttons` | `float64[]` | Flattened list of `[pressed, value]` pairs for all buttons. |
| `left_axes` | `float64[]` | List of axis values (e.g., thumbstick X/Y). |
| `right_axes` | `float64[]` | List of axis values (e.g., thumbstick X/Y). |
| `cmd_vel` | `float64[6]` | Velocity commands `[vx, vy, vz, wx, wy, wz]`. |
| `raw_xr_state` | `string` | Complete raw WebXR state message as a JSON string. |

#### Pose Format
All pose outputs use the `xyzquat` encoding: `[position_x, position_y, position_z, orientation_x, orientation_y, orientation_z, orientation_w]`. They include `primitive: "pose"` metadata for compatibility with `dora-rerun`.

#### Velocity Control (`cmd_vel`)
- **Linear X (Forward/Backward)**: Controlled by Right Thumbstick Y-axis.
- **Angular Z (Turn Left/Right)**: Controlled by Right Thumbstick X-axis.
- **Deadman Switch**: The Right Grip button must be held to enable velocity output.
- **Deadzone**: 20% to prevent drift.

### Inputs

1. **`tick`**: Triggers the node to process the latest received XR message and publish outputs.
2. **`joint_state`**: Receives robot joint states (as a JSON string) to be sent back to the WebXR client.
   - **Requirement**: Use `primitive: "jointstate"` metadata for this input to ensure proper handling.

## Configuration

### Environment Variables

The following variables can be set in your environment or YAML:

- `TELEOP_HOST`: WebSocket server host (default: `0.0.0.0`).
- `TELEOP_PORT`: WebSocket server port (default: `4443`).
- `INPUT_MODE`: Input mode for XR (`controller`, `hand`, or `auto`).
- `ROBOT_CLASS`: Python class path for internal IK (e.g., `teleop_xr.ik.robots.so101:SO101Robot`).
