# SO101 IK Visualization Demo with WebXR

This example demonstrates how to use a VR/AR headset to teleoperate a virtual SO101 robotic arm using Inverse Kinematics (IK). The motion of your VR controllers drives the robot arm in real-time, and the result is visualized both in the VR headset and in Rerun.

## Prerequisites

- **WebXR Compatible Headset**: Meta Quest 2/3/Pro, or any headset with a WebXR-compatible browser.
- **Network**: Headset and computer must be on the same local network.
- **Python 3.10+**

## Installation

1. Install `dora-rs`:
   ```bash
   pip install dora-rs
   ```

2. Build the example:
   ```bash
   dora build so101_ik_demo.yml --uv
   ```

## Running the Demo

1. Start the dora dataflow:
   ```bash
   dora run so101_ik_demo.yml --uv
   ```

2. Access the WebXR interface:
   - Look for the URL printed in the `teleop_xr` node logs (usually `https://<your-ip>:4443`).
   - Open this URL in your VR headset's browser.
   - **Note**: You may need to bypass the "Insecure Connection" warning as the node uses a self-signed certificate for HTTPS (required by WebXR).

3. Enter VR:
   - Click the "Enter VR" or "Enter AR" button in the browser.
   - You should see coordinate frames for your head and hands.
   - The SO101 robot arm will appear at the origin.

## Usage

- **Movement**: Use the **right controller thumbstick** to move the robot base or end-effector (depending on configuration).
- **IK Engagement**: 
  - Hold the **Grip button** on your right controller to "engage" the IK.
  - While holding the Grip, moving your hand will cause the robot arm to follow your controller's position and orientation.
- **Visualization**:
  - The robot arm is rendered in the VR headset.
  - A Rerun window will also open on your computer showing the 6DOF poses and the robot's joint state.
