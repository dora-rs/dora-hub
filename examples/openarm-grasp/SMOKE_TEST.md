# Smoke Test — Real Arms, Pick-and-Place

Six steps to confirm everything is wired up before running the full
chat-driven demo. Run on baguette unless noted.

## 1. Check the RGBD camera (web)

Open the OpenArm web UI: <https://huggingface.co/spaces/haixuantao/openarm>

You should see the **RealSense color + depth feed** rendering live in
the page. If both panels are blank or stuck on a single frame, the
`realsense-server` relay isn't publishing — restart it on baguette:

```bash
tmux kill-session -t realsense 2>/dev/null
tmux new-session -d -s realsense \
  '~/wser/target/release/realsense-server \
     --relay https://cdn.1ms.ai \
     --path anon/7e58263812ba/realsense-243222073892 \
     --width 1280 --height 720 --fps 15 --serial 243222073892'
```

Refresh the page; the camera feed should come up within a few seconds.

## 2. Check OpenArm is up

Verify the CAN-FD interfaces are present and UP, and a `can-server`
is publishing them:

```bash
ip link show can0    # state should be UP, mtu 72 (CAN-FD)
ip link show can1
pgrep -af can-server # one process, can0:fd can1:fd
```

If the interfaces are DOWN:

```bash
bash ~/setup-can.sh   # brings can0, can1 up at 1M nominal / 5M data
```

If `can-server` isn't running:

```bash
tmux new-session -d -s can_server \
  '~/wser/target/release/can-server can0:fd can1:fd \
     --key-dir ~/.config/xoq/keys \
     --moq-relay https://cdn.1ms.ai 2>&1 | tee /tmp/can_server.log'
```

The server's iroh IDs should match the `CAN_INTERFACE` keys in
`pick-and-place-chat.yml` — confirm with:

```bash
grep 'CAN bridge server running' /tmp/can_server.log
```

## 3. Activate `openarm_query` (keep it running)

`openarm_query` sends `ENABLE_MIT` to every motor and then a continuous
stream of zero-torque position queries. The arms hold still under no
commanded torque, but every joint angle is published over the bus →
MoQ → web. **Leave this running for the rest of the smoke test** so
the web client receives a continuous state stream.

```bash
~/wser/target/release/examples/openarm_query
```

Expected output:

```
left  J1=-0.02 J2=-0.04 J3=-0.09 J4=0.02 J5=-0.07 J6=-0.03 J7=0.06 Grip=0.00
right J1= 0.01 J2= 0.00 J3=-0.05 J4=-0.01 J5= 0.15 J6=-0.05 J7=0.00 Grip=0.00
```

If positions are stuck at zero or all `nan`, motors aren't powered or
the cable isn't seated — check the harness before going further.

(Ctrl+C exits and sends `DISABLE_MIT` on the way out.)

## 4. Verify the digital twin matches reality

With `openarm_query` running, the virtual arms in the web UI should
**mirror the physical arms in real time**. Manually wiggle one joint
on the physical arm — the corresponding joint on the on-screen arm
should move with it. That's the digital-twin loop closing.

If the virtual arm doesn't move, or moves with the wrong joints,
or is mounted at the wrong base pose:

- Open the **Settings panel** in the web UI (gear icon, top right).
- Confirm the left/right arm MoQ paths match what `can-server` is
  publishing (default `anon/xoq-can-can0/state` and
  `anon/xoq-can-can1/state`). Hit **Apply** to re-subscribe.
- If both arms are visible but their handedness is swapped, toggle
  the left/right assignment in Settings.
- If the base pose / orientation looks off, adjust the arm transform
  sliders in Settings until the on-screen arm geometry matches the
  physical setup.

Once the twin tracks 1:1, you're good.

## 5. Connect dora

Make sure the venv is active and dora-cli + node binaries are on PATH,
then launch the dataflow into the persistent `dora` tmux session:

```bash
source ~/dora-hub/.venv/bin/activate
export PATH=$HOME/.cargo/bin:$PATH
export LD_LIBRARY_PATH=$HOME/dora-hub/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH
cd ~/dora-hub/examples/openarm-grasp

tmux send-keys -t dora 'dora run pick-and-place-chat.yml' Enter
```

Wait ~60–90s for the VLM to load. Once it's up, both arm nodes log
`resp=8/8` per tick and `motion_planner` reports streaming intrinsics.

> Stop `openarm_query` first if it's still running — dora's
> arm-playback nodes also send MIT enables, and two clients writing
> to the same motors will fight each other.

## 6. Send a pick-and-place command

The `chat` node listens to the MoQ chat channel `anon/openarm-chat`
on `cdn.1ms.ai`. From the web UI, post:

```
@robot pick the yellow cube and place it in the white plate
```

Watch the chat acknowledge it, then `trajectory_status: ready` will
appear. Confirm by sending:

```
ok
```

The motion planner kicks off the trajectory and the arms execute
over real CAN. The digital twin reflects the motion live.

If anything misbehaves, see [QUICKSTART.md](QUICKSTART.md#troubleshooting)
for the failure-mode table.
