"""ABB IRB 1200-5/0.9 kinematics for the K2 cell (numpy only), with the ultrasonic knife on tool0.

Joint chain from USD/ABB-IRB1200.urdf; base pose as in the K2 scenes (side-mounted, +90 deg about Y).
The knife frame comes from tools/measure_knife output (cutting/knife.json): plate centred on tool0,
TCP = blade tip, TCP z = along the blade into the product, TCP x = cutting direction (blade width).
"""
import json
import os

import numpy as np

LIMITS = np.array([[-2.967, 2.967], [-1.745, 2.269], [-3.491, 1.222], [-4.712, 4.712], [-2.269, 2.269], [-6.283, 6.283]])
BASE_POS = np.array([0.225, 0.0, 1.9608])


def rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


BASE_ROT = ry(np.pi / 2)
CHAIN = [  # (offset in parent, joint axis rotation)
    (np.array([0, 0, 0.3991]), rz), (np.zeros(3), ry), (np.array([0, 0, 0.448]), ry),
    (np.array([0, 0, 0.042]), rx), (np.array([0.451, 0, 0]), ry), (np.array([0.082, 0, 0]), rx),
]
TOOL0_ROT = ry(np.pi / 2)


def fk_tool0(q):
    R, p = BASE_ROT.copy(), BASE_POS.copy()
    for (off, rot), qi in zip(CHAIN, q):
        p = p + R @ off
        R = R @ rot(qi)
    return R @ TOOL0_ROT, p


class Knife:
    """Rigid transform tool0 -> TCP, built from the measured knife geometry."""

    def __init__(self, path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "knife.json")):
        k = json.load(open(path))
        mount, up = np.array(k["mount"]), np.array(k["mount_normal_up"])
        tip, axis, width = np.array(k["tip"]), np.array(k["blade_axis"]), np.array(k["blade_width_dir"])
        z = -up / np.linalg.norm(up)                       # tool0 z points out of the flange, into the tool
        x = (tip - mount) - ((tip - mount) @ z) * z      # clocking: blade leans toward tool0 +x
        x /= np.linalg.norm(x)
        R_world_tool = np.stack([x, np.cross(z, x), z], axis=1)
        self.R_world_tool, self.mount = R_world_tool, mount  # pose of the knife as it sits in the CAD scene
        self.p_tcp = R_world_tool.T @ (tip - mount)         # TCP position in tool0
        a, w = R_world_tool.T @ axis, R_world_tool.T @ width
        w -= (w @ a) * a
        w /= np.linalg.norm(w)
        self.R_tcp = np.stack([w, np.cross(a, w), a], axis=1)  # TCP frame in tool0 (x cut dir, z blade)

    def tcp(self, q):
        R0, p0 = fk_tool0(q)
        return R0 @ self.R_tcp, p0 + R0 @ self.p_tcp


def target_rotation(cut_dir):
    """Blade straight down, cutting along cut_dir (horizontal)."""
    x = np.asarray(cut_dir, float)
    x = x / np.linalg.norm(x)
    z = np.array([0, 0, -1.0])
    return np.stack([x, np.cross(z, x), z], axis=1)


def rot_err(Ra, Rb):
    Re = Ra @ Rb.T
    return 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])


def ik(knife, R_goal, p_goal, q0, iters=200, tol_p=5e-4, tol_r=np.radians(0.5)):
    """Damped least squares on the TCP pose; returns (q, ok)."""
    q = np.array(q0, float)
    for _ in range(iters):
        R, p = knife.tcp(q)
        e = np.concatenate([p_goal - p, rot_err(R_goal, R)])
        if np.linalg.norm(e[:3]) < tol_p and np.linalg.norm(e[3:]) < tol_r:
            return q, bool(np.all((q >= LIMITS[:, 0]) & (q <= LIMITS[:, 1])))
        J = np.zeros((6, 6))
        for j in range(6):
            dq = np.zeros(6)
            dq[j] = 1e-5
            R2, p2 = knife.tcp(q + dq)
            J[:3, j] = (p2 - p) / 1e-5
            J[3:, j] = rot_err(R2, R) / 1e-5
        q = q + J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
        q = np.clip(q, LIMITS[:, 0], LIMITS[:, 1])
    return q, False


def joint_rates(knife, q, twist, damping=1e-4):
    """Joint speeds giving a TCP twist [vx, vy, vz, wx, wy, wz] (world frame) at q, damped least squares."""
    R, p = knife.tcp(q)
    J = np.zeros((6, 6))
    for j in range(6):
        dq = np.zeros(6)
        dq[j] = 1e-5
        R2, p2 = knife.tcp(np.asarray(q, float) + dq)
        J[:3, j] = (p2 - p) / 1e-5
        J[3:, j] = rot_err(R2, R) / 1e-5
    return J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), np.asarray(twist, float))
