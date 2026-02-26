"""Mujoco Client: This node is used to represent simulated robot, it can be used to read virtual positions, or can be controlled."""

import argparse
import os

import cv2
import mujoco
import mujoco.viewer
import numpy as np
import pyarrow as pa
from dora import Node


class Client:
    """Generic MuJoCo simulation node."""

    def __init__(self, config: dict[str, object]) -> None:
        """Initialise model, optional off-screen renderer, and Dora node."""
        self.m = mujoco.MjModel.from_xml_path(filename=config["scene"])
        self.data = mujoco.MjData(self.m)

        self._render_cameras: bool = bool(config.get("render_cameras", False))
        self._cameras: list[tuple[int, str]] = []

        if self._render_cameras:
            img_w: int = int(config["img_width"])
            img_h: int = int(config["img_height"])
            jpeg_quality: int = int(config["jpeg_quality"])

            self._img_meta: dict[str, object] = {
                "encoding": "jpeg",
                "width": img_w,
                "height": img_h,
            }
            self._jpeg_quality = jpeg_quality

            self._gl_ctx = mujoco.GLContext(img_w, img_h)
            self._gl_ctx.make_current()

            self._scene    = mujoco.MjvScene(self.m, maxgeom=10_000)
            self._cam      = mujoco.MjvCamera()
            self._opt      = mujoco.MjvOption()
            self._pert     = mujoco.MjvPerturb()
            self._viewport = mujoco.MjrRect(0, 0, img_w, img_h)
            self._mjr_ctx  = mujoco.MjrContext(self.m, mujoco.mjtFontScale.mjFONTSCALE_150)

            # mjCAMERA_FIXED → a named <camera> element from the XML.
            # fixedcamid is set per-render; type is set once here.
            self._cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

            # Pre-allocated pixel buffer — reused every render, no per-frame alloc.
            self._rgb_buf = np.zeros((img_h, img_w, 3), dtype=np.uint8)

            # Read every camera declared in the XML at startup.
            # Camera index i == the id returned by mj_name2id, so no runtime lookup.
            for i in range(self.m.ncam):
                name = self.m.camera(i).name
                self._cameras.append((i, name))

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

                event_id = event["id"]

                if event_id == "tick":
                    # Echo the tick first so the downstream dataflow does not
                    # stall while we spend time in mj_step.
                    self.node.send_output("tick", pa.array([]), event["metadata"])

                    if not viewer.is_running():
                        break

                    mujoco.mj_step(self.m, self.data)
                    viewer.sync()
                    self._publish_state()

                elif event_id == "render" and self._render_cameras:
                    # Decoupled from the physics tick: encoding four 960×600
                    # JPEGs at 250 Hz would block the control loop entirely.
                    self._render_and_publish()

                elif event_id == "action":
                    self._handle_action(event["value"])

    def _handle_action(self, value: pa.Array) -> None:
        """Write the incoming flat action array into data.ctrl.

        The caller is responsible for ordering the values to match the
        actuator order declared in the XML (data.ctrl index order == the
        order of <actuator> elements in the scene file).
        """
        if self.m.nu == 0:
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
          → cv2.imencode JPEG
          → node.send_output
        """
        if not self._cameras:
            return

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
                ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
            )
            if ret:
                self.node.send_output(topic, pa.array(jpeg.ravel()), self._img_meta)


def main() -> None:
    """Handle dynamic nodes, ask for the name of the node in the dataflow."""
    parser = argparse.ArgumentParser(
        description="MuJoCo Client: This node is used to represent a MuJoCo simulation. "
        "It can be used instead of a follower arm to test the dataflow.",
    )

    parser.add_argument(
        "--name",
        type=str,
        required=False,
        help="The name of the node in the dataflow.",
        default="mujoco_client",
    )
    parser.add_argument(
        "--scene",
        type=str,
        required=False,
        help="The scene file of the MuJoCo simulation.",
    )
    parser.add_argument(
        "--cameras",
        action="store_true",
        help="Enable off-screen camera rendering and publish JPEG frames per 'render' tick.",
    )
    parser.add_argument(
        "--img-width",
        type=int,
        default=None,
        help="Width of rendered camera images in pixels (default: IMG_WIDTH env var or 960).",
    )
    parser.add_argument(
        "--img-height",
        type=int,
        default=None,
        help="Height of rendered camera images in pixels (default: IMG_HEIGHT env var or 600).",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=None,
        help="JPEG encoding quality 0-100 (default: JPEG_QUALITY env var or 90).",
    )

    args = parser.parse_args()

    if not os.getenv("SCENE") and args.scene is None:
        raise ValueError(
            "Please set the SCENE environment variable or pass the --scene argument.",
        )

    scene = os.getenv("SCENE", args.scene)

    config = {
        "name": args.name,
        "scene": scene,
        "render_cameras": args.cameras,
        "img_width":    args.img_width    or int(os.getenv("IMG_WIDTH",    960)),
        "img_height":   args.img_height   or int(os.getenv("IMG_HEIGHT",   600)),
        "jpeg_quality": args.jpeg_quality or int(os.getenv("JPEG_QUALITY",  90)),
    }

    print("MuJoCo Client Configuration:", config, flush=True)

    Client(config).run()


if __name__ == "__main__":
    main()
