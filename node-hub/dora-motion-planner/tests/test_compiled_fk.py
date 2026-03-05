"""Tests for compiled forward kinematics (compiled_fk.py).

Verifies that CompiledFK produces identical results to pytorch_kinematics,
gradients flow correctly, and the adapter is a drop-in replacement for
pk.Chain in the TrajectoryOptimizer.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import torch
import pytorch_kinematics as pk

from dora_motion_planner.compiled_fk import CompiledFK, CompiledFKAdapter

URDF_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "examples"
    / "openarm"
    / "openarm_v10.urdf"
)

EE_LINK = "openarm_left_hand_tcp"


@pytest.fixture
def pk_chain():
    with open(URDF_PATH) as f:
        urdf_str = f.read()
    root = ET.fromstring(urdf_str)
    for link in root.findall(".//link"):
        for tag in ("visual", "collision", "inertial"):
            for elem in link.findall(tag):
                link.remove(elem)
    clean = ET.tostring(root, encoding="unicode")
    return pk.build_serial_chain_from_urdf(clean, EE_LINK).to(
        dtype=torch.float32, device="cpu"
    )


@pytest.fixture
def cfk():
    return CompiledFK("cpu")


def test_single_ee(pk_chain, cfk):
    """Single-sample EE transform matches pk."""
    q = torch.tensor([[0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7]])
    pk_mat = pk_chain.forward_kinematics(q).get_matrix()
    cfk_mat = cfk.forward(q)
    assert (pk_mat - cfk_mat).abs().max() < 1e-5


def test_batched_ee(pk_chain, cfk):
    """Batched EE transforms match pk."""
    q = torch.randn(64, 7) * 0.5
    pk_mat = pk_chain.forward_kinematics(q).get_matrix()
    cfk_mat = cfk.forward(q)
    assert (pk_mat - cfk_mat).abs().max() < 1e-4


def test_all_links(pk_chain, cfk):
    """All-link transforms match pk."""
    q = torch.randn(16, 7) * 0.5
    pk_all = pk_chain.forward_kinematics(q, end_only=False)
    cfk_all = cfk.forward_all(q)
    for name in cfk_all:
        if name in pk_all:
            diff = (pk_all[name].get_matrix() - cfk_all[name]).abs().max()
            assert diff < 1e-4, f"{name}: diff={diff}"


def test_gradients(cfk):
    """Gradients flow through compiled FK."""
    q = torch.randn(4, 7, requires_grad=True)
    ee = cfk.forward(q)
    loss = ee[:, :3, 3].sum()
    loss.backward()
    assert q.grad is not None
    assert q.grad.abs().sum() > 0


def test_adapter_drop_in(pk_chain, cfk):
    """CompiledFKAdapter provides the same interface as pk.Chain."""
    adapter = CompiledFKAdapter(cfk)
    q = torch.randn(8, 7) * 0.5

    # end_only=True
    pk_ee = pk_chain.forward_kinematics(q).get_matrix()
    ad_ee = adapter.forward_kinematics(q, end_only=True).get_matrix()
    assert (pk_ee - ad_ee).abs().max() < 1e-4

    # end_only=False
    pk_all = pk_chain.forward_kinematics(q, end_only=False)
    ad_all = adapter.forward_kinematics(q, end_only=False)
    for name in ad_all:
        if name in pk_all:
            diff = (pk_all[name].get_matrix() - ad_all[name].get_matrix()).abs().max()
            assert diff < 1e-4, f"{name}: diff={diff}"

    # get_link_names
    assert adapter.get_link_names() == pk_chain.get_link_names()


def test_adapter_to_noop(cfk):
    """adapter.to() returns self without error."""
    adapter = CompiledFKAdapter(cfk)
    assert adapter.to(dtype=torch.float32, device="cpu") is adapter


@pytest.mark.skipif(
    not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    reason="MPS not available",
)
def test_mps_forward():
    """CompiledFK works on MPS."""
    cfk = CompiledFK("mps")
    q = torch.randn(8, 7, device="mps")
    ee = cfk.forward(q)
    assert ee.shape == (8, 4, 4)
    assert ee.device.type == "mps"
