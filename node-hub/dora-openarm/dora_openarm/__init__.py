"""Dora node for OpenArm bimanual robot — Damiao motor CAN bus driver."""

import os

readme_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "README.md")

try:
    with open(readme_path, encoding="utf-8") as f:
        __doc__ = f.read()
except FileNotFoundError:
    __doc__ = "README file not found."
