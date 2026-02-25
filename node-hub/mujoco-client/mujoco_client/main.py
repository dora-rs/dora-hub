"""Mujoco Client: This node is used to represent simulated robot, it can be used to read virtual positions, or can be controlled."""

import argparse
import json
import os
import time

import cv2
import mujoco
import mujoco.viewer
import numpy as np
import pyarrow as pa
from dora import Node

#Image constants (must match the downstream pipeline expectations)
_IMG_W: int = 960
_IMG_H: int = 600
_JPEG_QUALITY: int = 90

_IMG_META: dict[str, object] = {
    "encoding": "jpeg",
    "width":    _IMG_W,
    "height":   _IMG_H,
}


def _cam_topic(cam_name: str) -> str:
    """Derive a Dora-safe output topic name from a camera's XML name.

    Hyphens are replaced with underscores so the name is valid as a YAML
    key without quoting.  No other transformation is applied, keeping the
    mapping transparent: "camera-wrist-right" → "camera_wrist_right".
    """
    return cam_name.replace("-", "_")


class Client:
    """Generic MuJoCo simulation node."""

    def __init__(self, config: dict[str, str]) -> None:
        
        self.m = mujoco.MjModel.from_xml_path(filename=config["scene"])
        self.data = mujoco.MjData(self.m)

        print(
            f"Model loaded: nq={self.m.nq}  nv={self.m.nv}  "
            f"nu={self.m.nu}  ncam={self.m.ncam}",
            flush=True,
        )

        self._gl_ctx = mujoco.GLContext(_IMG_W, _IMG_H)
        self._gl_ctx.make_current()

        self._scene    = mujoco.MjvScene(self.m, maxgeom=10_000)
        self._cam      = mujoco.MjvCamera()
        self._opt      = mujoco.MjvOption()
        self._pert     = mujoco.MjvPerturb()
        self._viewport = mujoco.MjrRect(0, 0, _IMG_W, _IMG_H)
        self._mjr_ctx  = mujoco.MjrContext(self.m, mujoco.mjtFontScale.mjFONTSCALE_150)

        # mjCAMERA_FIXED → a named <camera> element from the XML.
        # fixedcamid is set per-render; type is set once here.
        self._cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

        # Pre-allocated pixel buffer — reused every render, no per-frame alloc.
        self._rgb_buf = np.zeros((_IMG_H, _IMG_W, 3), dtype=np.uint8)

        # Read every camera declared in the XML at startup.
        # Camera index i == the id returned by mj_name2id, so no runtime lookup.
        self._cameras: list[tuple[int, str]] = []   # (cam_id, dora_topic)
        for i in range(self.m.ncam):
            name  = self.m.camera(i).name
            topic = _cam_topic(name)
            self._cameras.append((i, topic))
            print(f"  camera[{i}] '{name}' → topic '{topic}'", flush=True)

        if not self._cameras:
            print("WARNING: no cameras found in model — no image output", flush=True)

        # Warn once at startup if the model has no actuators.
        if self.m.nu == 0:
            print("WARNING: model has no actuators (m.nu=0) — action input ignored", flush=True)

        self.node = Node(config["name"])

    def run(self) -> None:
        """Drive the physics + rendering event loop."""
        with mujoco.viewer.launch_passive(self.m, self.data) as viewer:
            viewer.sync()

            for event in self.node:
                if event["type"] != "INPUT":
                    continue

                eid = event["id"]

                if eid == "tick":
                    # Echo the tick first so the downstream dataflow does not
                    # stall while we spend time in mj_step.
                    self.node.send_output("tick", pa.array([]), event["metadata"])

                    if not viewer.is_running():
                        break

                    mujoco.mj_step(self.m, self.data)
                    viewer.sync()
                    self._publish_state()

                elif eid == "render":
                    # Decoupled from the physics tick: encoding four 960×600
                    # JPEGs at 250 Hz would block the control loop entirely.
                    self._render_and_publish()

                elif eid == "action":
                    self._handle_action(event["value"])

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _handle_action(self, value: pa.Array | pa.ChunkedArray) -> None:
        """Write the incoming flat action array into data.ctrl.

        The caller is responsible for ordering the values to match the
        actuator order declared in the XML (data.ctrl index order == the
        order of <actuator> elements in the scene file).
        """
        if self.m.nu == 0:
            return

        if not isinstance(value, (pa.Array, pa.ChunkedArray)):
            print(
                f"WARNING: action expected pa.Array, got {type(value).__name__} — ignored",
                flush=True,
            )
            return

        vals = np.asarray(value, dtype=np.float64)

        if len(vals) != self.m.nu:
            print(
                f"WARNING: action length {len(vals)} != m.nu {self.m.nu} "
                f"— writing first {min(len(vals), self.m.nu)} values",
                flush=True,
            )

        n = min(len(vals), self.m.nu)
        self.data.ctrl[:n] = vals[:n]

    def _publish_state(self) -> None:
        """Publish full generalised coordinates and ready status."""
        self.node.send_output(
            "joint_state",
            pa.array(self.data.qpos.astype(np.float32)),
        )
        self.node.send_output("status", pa.array(["ready"]))

    def _render_and_publish(self) -> None:
        """Render every model camera off-screen and publish as JPEG.

        Per-camera pipeline:
          mjv_updateScene → mjr_render → mjr_readPixels
          → cv2.flip (OpenGL bottom-up → top-down)
          → cv2.cvtColor RGB→BGR
          → cv2.imencode JPEG q90
          → node.send_output
        """
        # Re-assert our EGL context as current.  The passive viewer runs in a
        # background thread with its own GLFW context; this call is a ~1 µs
        # guard against any threading edge-cases.
        self._gl_ctx.make_current()

        for cam_id, topic in self._cameras:
            self._cam.fixedcamid = cam_id

            mujoco.mjv_updateScene(
                self.m, self.data,
                self._opt, self._pert,
                self._cam, mujoco.mjtCatBit.mjCAT_ALL,
                self._scene,
            )
            mujoco.mjr_render(self._viewport, self._scene, self._mjr_ctx)

            mujoco.mjr_readPixels(self._rgb_buf, None, self._viewport, self._mjr_ctx)

            bgr = cv2.cvtColor(cv2.flip(self._rgb_buf, 0), cv2.COLOR_RGB2BGR)

            ret, jpeg = cv2.imencode(
                ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY]
            )
            if ret:
                self.node.send_output(topic, pa.array(jpeg.ravel()), _IMG_META)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """Parse CLI arguments and start the simulation node."""
    parser = argparse.ArgumentParser(description="Generic MuJoCo simulation node")
    parser.add_argument("--name",  type=str, default="mujoco")
    parser.add_argument("--scene", type=str, help="Path to the MuJoCo XML scene file")
    args = parser.parse_args()

    scene = os.getenv("SCENE", args.scene)
    if not scene:
        raise ValueError("Provide --scene <path> or set the SCENE environment variable")

    config = {"name": args.name, "scene": scene}
    print("MuJoCo Client Configuration:", config, flush=True)

    Client(config).run()


if __name__ == "__main__":
    main()
