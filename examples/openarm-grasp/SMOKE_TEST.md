# Smoke Test — Real Arms, Pick-and-Place

Four steps to confirm everything is wired up before running the full
chat-driven demo. Run on baguette unless noted.

## 1. Check OpenArm is up

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

## 2. Zero-torque MIT query (no motion, just liveness)

`openarm_query` sends `ENABLE_MIT` then a stream of zero-torque
position queries. Motors enabled, no commanded torque → arms hold
still and report back joint angles. **If you see realistic angles
that change as you wiggle a joint by hand, the bus is healthy.**

```bash
~/wser/target/release/examples/openarm_query
```

Expected output:

```
left  J1=-0.02 J2=-0.04 J3=-0.09 J4=0.02 J5=-0.07 J6=-0.03 J7=0.06 Grip=0.00
right J1= 0.01 J2= 0.00 J3=-0.05 J4=-0.01 J5= 0.15 J6=-0.05 J7=0.00 Grip=0.00
```

If positions are stuck at zero or all `nan`, motors aren't powered
or the cable isn't seated — check the harness before going further.

`Ctrl+C` to exit (it sends `DISABLE_MIT` on shutdown).

## 3. Connect dora

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

## 4. Send a pick-and-place command

The `chat` node listens to the MoQ chat channel
`anon/openarm-chat` on `cdn.1ms.ai`. Open the OpenArm web UI at
<https://huggingface.co/spaces/haixuantao/openarm> and post:

```
@robot pick the yellow cube and place it in the white plate
```

Watch the chat acknowledge it, then `trajectory_status: ready` will
appear. Confirm by sending:

```
ok
```

The motion planner kicks off Phase 1 → Phase 2 of the trajectory and
the arms execute over real CAN. The web client shows live joint
state and the camera feed for the duration.

If anything misbehaves, see [QUICKSTART.md](QUICKSTART.md#troubleshooting)
for the failure-mode table.
