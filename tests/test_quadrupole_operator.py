"""Regression tests for the static atomic-quadrupole operator in EMLEBase.

Loads ``_emle_base`` directly by path so the test does not pull the full
``emle.models`` import chain (which currently needs a newer torchani than is
installed). Validates that ``get_static_energy`` with a quadrupole reproduces
the analytic Cartesian multipole potential, and that the full charge-quadrupole
kernel is invariant to the trace of the quadrupole (raw MBIS == detraced),
which is the property that the reduced kernel got wrong.
"""
import os
import importlib.util

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.normpath(os.path.join(_HERE, "..", "emle", "models", "_emle_base.py"))
_spec = importlib.util.spec_from_file_location("_emle_base_under_test", _BASE)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
EMLEBase = _mod.EMLEBase


def _random_system(seed=0):
    rng = np.random.default_rng(seed)
    A, M = 3, 6
    xyz_qm = rng.normal(0, 1.0, (A, 3))
    # Put MM points a few Bohr away so the multipole expansion is well behaved.
    xyz_mm = rng.normal(0, 1.0, (M, 3)) + rng.choice([-6, 6], (M, 1))
    s = np.abs(rng.normal(1.0, 0.1, A))
    q_core = rng.normal(0, 0.5, A)
    q_val = rng.normal(0, 0.5, A)
    mu = rng.normal(0, 0.3, (A, 3))
    theta6 = rng.normal(0, 0.3, (A, 6))  # xx,xy,xz,yy,yz,zz
    charges_mm = rng.normal(0, 0.5, M)
    return xyz_qm, xyz_mm, s, q_core, q_val, mu, theta6, charges_mm


def _theta_3x3(theta6):
    Q = np.zeros((theta6.shape[0], 3, 3))
    Q[:, 0, 0] = theta6[:, 0]
    Q[:, 0, 1] = Q[:, 1, 0] = theta6[:, 1]
    Q[:, 0, 2] = Q[:, 2, 0] = theta6[:, 2]
    Q[:, 1, 1] = theta6[:, 3]
    Q[:, 1, 2] = Q[:, 2, 1] = theta6[:, 4]
    Q[:, 2, 2] = theta6[:, 5]
    return Q


def _analytic_energy(xyz_qm, xyz_mm, s, q_core, q_val, mu, Q, charges_mm):
    rr = xyz_mm[None, :, :] - xyz_qm[:, None, :]        # (A,M,3)
    r = np.linalg.norm(rr, axis=-1)
    inv_r = 1.0 / r
    T0_slater = (1.0 - (1.0 + r / (2.0 * s[:, None])) * np.exp(-r / s[:, None])) / r
    v = np.sum(q_core[:, None] * inv_r + q_val[:, None] * T0_slater, axis=0)
    v += np.einsum("amx,ax,am->m", rr, mu, inv_r ** 3)
    rQr = np.einsum("amx,axy,amy->am", rr, Q, rr)
    trQ = np.trace(Q, axis1=1, axis2=2)
    v += np.sum((3.0 * rQr - r * r * trQ[:, None]) / (2.0 * r ** 5), axis=0)
    return float(np.sum(charges_mm * v))


def _call(xyz_qm, xyz_mm, s, q_core, q_val, mu, Q, charges_mm):
    A = xyz_qm.shape[0]
    M = xyz_mm.shape[0]
    mask = torch.ones(1, A, M, dtype=torch.bool)
    mesh = EMLEBase._get_mesh_data(
        torch.tensor(xyz_qm)[None], torch.tensor(xyz_mm)[None],
        torch.tensor(s)[None], mask,
    )
    return EMLEBase.get_static_energy(
        torch.tensor(q_core)[None], torch.tensor(q_val)[None],
        torch.tensor(charges_mm)[None], mesh,
        torch.tensor(mu)[None], torch.tensor(Q)[None],
    ).item()


def test_quadrupole_matches_analytic_potential():
    for seed in range(4):
        xyz_qm, xyz_mm, s, q_core, q_val, mu, theta6, charges_mm = _random_system(seed)
        Q = _theta_3x3(theta6)
        got = _call(xyz_qm, xyz_mm, s, q_core, q_val, mu, Q, charges_mm)
        ref = _analytic_energy(xyz_qm, xyz_mm, s, q_core, q_val, mu, Q, charges_mm)
        assert abs(got - ref) < 1e-9, f"seed {seed}: {got} vs {ref}"


def test_full_kernel_invariant_to_trace():
    # The full (3 rr rr - r^2 I)/(2 r^5) kernel gives the same energy for a
    # primitive (traced) quadrupole and its detraced counterpart. The old
    # reduced kernel (rr rr / r^5) did not, which blew up the MBIS reference.
    for seed in range(4):
        xyz_qm, xyz_mm, s, q_core, q_val, mu, theta6, charges_mm = _random_system(seed)
        Q = _theta_3x3(theta6)
        trQ = np.trace(Q, axis1=1, axis2=2)
        Q_detr = Q - (trQ[:, None, None] / 3.0) * np.eye(3)[None]
        e_raw = _call(xyz_qm, xyz_mm, s, q_core, q_val, mu, Q, charges_mm)
        e_detr = _call(xyz_qm, xyz_mm, s, q_core, q_val, mu, Q_detr, charges_mm)
        assert abs(e_raw - e_detr) < 1e-9, f"seed {seed}: {e_raw} vs {e_detr}"
