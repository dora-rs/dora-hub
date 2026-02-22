"""Main module for dora-teleop-xr node."""

import asyncio
import importlib
import os
import threading
from typing import Protocol, TypedDict

import jax
import numpy as np
import pyarrow as pa
import uvicorn
from dora import Node
from numpy.typing import NDArray
from teleop_xr import Teleop
from teleop_xr.config import TeleopSettings
from teleop_xr.ik.controller import IKController
from teleop_xr.ik.solver import PyrokiSolver
from teleop_xr.messages import XRState


def load_robot_class(class_path: str):
    if ":" not in class_path:
        raise ValueError(
            f"Invalid ROBOT_CLASS format: '{class_path}'. Expected 'module:ClassName'"
        )
    module_path, class_name = class_path.split(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def get_robot_joint_order(robot_class) -> tuple[str, ...] | None:
    if hasattr(robot_class, "DEFAULT_RERUN_JOINT_ORDER"):
        return robot_class.DEFAULT_RERUN_JOINT_ORDER
    return None


class JointNamesProvider(Protocol):
    @property
    def actuated_joint_names(self) -> list[str]: ...


class IKStateContainer(TypedDict):
    q: NDArray[np.float32]
    active: bool
    xr_state: XRState | None


class IKWorker(threading.Thread):
    def __init__(
        self,
        controller: IKController,
        robot: object,
        teleop: Teleop,
        state_container: IKStateContainer,
    ):
        super().__init__(daemon=True)
        self.controller = controller
        self.robot = robot
        self.teleop = teleop
        self.state_container = state_container
        self.latest_xr_state: XRState | None = None
        self.new_state_event = threading.Event()
        self.running = True
        self.teleop_loop = None

    def update_state(self, state: XRState):
        self.latest_xr_state = state
        self.new_state_event.set()

    def set_teleop_loop(self, loop: asyncio.AbstractEventLoop):
        if self.teleop_loop is None:
            self.teleop_loop = loop
            if "q" in self.state_container:
                joint_dict = {
                    name: float(val)
                    for name, val in zip(
                        self.robot.actuated_joint_names, self.state_container["q"]
                    )
                }
                asyncio.run_coroutine_threadsafe(
                    self.teleop.publish_joint_state(joint_dict),
                    self.teleop_loop,
                )

    def run(self):
        while self.running:
            if not self.new_state_event.wait(timeout=0.1):
                continue

            self.new_state_event.clear()

            state = self.latest_xr_state
            if state is None:
                continue

            try:
                q_current = self.state_container["q"]
                was_active = self.controller.active

                new_config = np.asarray(
                    self.controller.step(state, q_current), dtype=np.float32
                )

                self.state_container["active"] = self.controller.active
                is_active = self.controller.active

                if not was_active and is_active:
                    print("IK Control Start - Taking Snapshots")

                if not np.array_equal(new_config, q_current):
                    self.state_container["q"] = new_config
                    joint_dict = {
                        name: float(val)
                        for name, val in zip(
                            self.robot.actuated_joint_names, new_config
                        )
                    }

                    if self.teleop_loop and self.teleop_loop.is_running():
                        asyncio.run_coroutine_threadsafe(
                            self.teleop.publish_joint_state(joint_dict),
                            self.teleop_loop,
                        )

            except Exception as e:
                print(f"Error in IK Worker: {e}")


def pose_to_array(pose):
    if not pose or not pose.position or not pose.orientation:
        return None
    pos = pose.position
    ori = pose.orientation
    return np.array(
        [
            pos.get("x", 0.0),
            pos.get("y", 0.0),
            pos.get("z", 0.0),
            ori.get("x", 0.0),
            ori.get("y", 0.0),
            ori.get("z", 0.0),
            ori.get("w", 1.0),
        ],
        dtype=np.float32,
    )


def reorder_joint_state_for_rerun(
    robot: JointNamesProvider,
    q_current: NDArray[np.float32],
    joint_order: tuple[str, ...] | None,
) -> NDArray[np.float32]:
    if joint_order is None:
        return q_current

    names = list(robot.actuated_joint_names)
    if not names:
        return q_current

    name_to_idx = {name: idx for idx, name in enumerate(names)}
    if not set(names).issubset(set(joint_order)):
        return q_current

    ordered = [
        q_current[name_to_idx[name]]
        for name in joint_order
        if name in name_to_idx
    ]
    if len(ordered) != len(q_current):
        return q_current
    return np.asarray(ordered, dtype=np.float32)


def main():
    jax.config.update("jax_platform_name", "cpu")

    robot_class_path = os.environ.get(
        "ROBOT_CLASS", "teleop_xr.ik.robots.so101:SO101Robot"
    )
    robot_class = load_robot_class(robot_class_path)

    node = Node()

    robot = robot_class()
    solver = PyrokiSolver(robot)
    controller = IKController(robot, solver)

    state_container: IKStateContainer = {
        "q": np.asarray(robot.get_default_config(), dtype=np.float32),
        "active": False,
        "xr_state": None,
    }

    robot_vis = robot.get_vis_config()
    settings = TeleopSettings(robot_vis=robot_vis)

    teleop = Teleop(settings)

    ik_worker = IKWorker(controller, robot, teleop, state_container)
    ik_worker.start()

    joint_order = get_robot_joint_order(robot_class)

    def on_xr_update(_pose, message):
        try:
            try:
                loop = asyncio.get_running_loop()
                ik_worker.set_teleop_loop(loop)
            except RuntimeError:
                pass

            xr_data = message.get("data", message)
            state = XRState.model_validate(xr_data)

            state_container["xr_state"] = state
            ik_worker.update_state(state)
        except Exception as e:
            print(f"Error in on_xr_update: {e}")

    teleop.subscribe(on_xr_update)

    import teleop_xr

    teleop_file = getattr(teleop_xr, "__file__", None)
    if teleop_file is None:
        raise ImportError("teleop_xr module has no __file__ attribute")
    teleop_dir = os.path.dirname(teleop_file)
    ssl_keyfile = os.path.join(teleop_dir, "key.pem")
    ssl_certfile = os.path.join(teleop_dir, "cert.pem")

    config = uvicorn.Config(
        teleop.app,
        host=settings.host,
        port=settings.port,
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
        log_level="warning",
    )
    server = uvicorn.Server(config)

    def run_server():
        server.run()

    teleop_thread = threading.Thread(target=run_server, daemon=True)
    teleop_thread.start()

    try:
        for event in node:
            event_type = event["type"]

            if event_type == "STOP":
                break

            if event_type == "INPUT":
                input_id = event["id"]

                if input_id == "tick":
                    xr_state: XRState | None = state_container["xr_state"]
                    if xr_state is None:
                        continue

                    head = None
                    left = None
                    right = None
                    for dev in xr_state.devices:
                        if dev.role == "head":
                            head = dev
                        elif dev.role == "controller" or dev.role == "hand":
                            if dev.handedness == "left":
                                left = dev
                            elif dev.handedness == "right":
                                right = dev

                    head_pose = pose_to_array(head.pose) if head else None
                    left_pose = pose_to_array(left.gripPose) if left else None
                    right_pose = pose_to_array(right.gripPose) if right else None

                    pose_metadata = {"primitive": "pose", "encoding": "xyzquat"}

                    if head_pose is not None:
                        node.send_output(
                            "head_pose",
                            pa.array(head_pose, type=pa.float32()),
                            pose_metadata,
                        )
                    if left_pose is not None:
                        node.send_output(
                            "left_hand_pose",
                            pa.array(left_pose, type=pa.float32()),
                            pose_metadata,
                        )
                    if right_pose is not None:
                        node.send_output(
                            "right_hand_pose",
                            pa.array(right_pose, type=pa.float32()),
                            pose_metadata,
                        )

                    q_current = state_container["q"]
                    q_rerun = reorder_joint_state_for_rerun(robot, q_current, joint_order)
                    node.send_output(
                        "joint_state",
                        pa.array(q_rerun, type=pa.float32()),
                        {
                            "encoding": "jointstate",
                            "primitive": "jointstate",
                        },
                    )

                    node.send_output(
                        "raw_xr_state",
                        pa.array([xr_state.model_dump_json()]),
                    )

    finally:
        ik_worker.running = False
        ik_worker.join(timeout=1.0)
        server.should_exit = True
        teleop_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
