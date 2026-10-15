"""Cut lines on dough pieces from their meshes (numpy only).

Each piece gets one straight score over the middle `span` of its footprint, sampled every `step`
metres: along the long axis for long pieces (aspect > `long_aspect`, "left to right"), along the belt
for round ones ("top to bottom"). The score runs `depth` under the top contour. `cut_path` adds the
way in and out: straight down from `lead_up` above the first cut point, straight up at the end.
"""
import numpy as np


def cut_path(contour, direction, offset=-0.003, lead_up=0.03):
    """Full knife path for one piece.

    contour: (N, 3) points on the top surface along the cut line; direction: cut direction (x, y).
    Returns (poly, i0, i1): approach -> cut -> retract, and the indices of the first and last cut
    points in poly.
    """
    S = np.asarray(contour, float)
    up = np.array([0.0, 0.0, 1.0])
    cut = S + offset * up
    poly = np.vstack([cut[0] + lead_up * up, cut, cut[-1] + lead_up * up])
    return poly, 1, len(cut)


def top_height(points, counts, indices, xy):
    """Exact top surface: for each (x, y), the highest mesh triangle straight above or below it (NaN if none)."""
    P = np.asarray(points, float)
    tris, k = [], 0
    for n in np.asarray(counts, int):  # fan-triangulate the polygons
        f = indices[k:k + n]
        tris += [(f[0], f[j], f[j + 1]) for j in range(1, n - 1)]
        k += n
    T = P[np.array(tris)]  # (T, 3 vertices, xyz)
    a, b, c = T[:, 0, :2], T[:, 1, :2], T[:, 2, :2]
    det = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
    good = np.abs(det) > 1e-12  # skip triangles seen edge-on from above
    T, a, b, c, det = T[good], a[good], b[good], c[good], det[good]
    out = np.full(len(xy), np.nan)
    for i, (px, py) in enumerate(np.asarray(xy, float)):
        w0 = ((b[:, 1] - c[:, 1]) * (px - c[:, 0]) + (c[:, 0] - b[:, 0]) * (py - c[:, 1])) / det
        w1 = ((c[:, 1] - a[:, 1]) * (px - c[:, 0]) + (a[:, 0] - c[:, 0]) * (py - c[:, 1])) / det
        w2 = 1.0 - w0 - w1
        hit = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        if hit.any():
            out[i] = (w0[hit] * T[hit, 0, 2] + w1[hit] * T[hit, 1, 2] + w2[hit] * T[hit, 2, 2]).max()
    return out


def piece_cut(points, depth=0.003, span=0.8, step=0.005, long_aspect=1.4, clearance=0.03, belt_dir=(1.0, 0.0),
              faces=None):
    """Cut for one piece mesh.

    `clearance` is how high above the first/last cut point the approach and retract points sit.
    `faces` = (counts, indices) of the mesh gives the exact top surface; without it the height is the
    highest vertex within 4 mm, which dips to the underside where the mesh is sparse.
    """
    P = np.asarray(points, float)
    top = P[P[:, 2] > P[:, 2].min() + 0.002]           # skip the base resting on the tray
    xy = top[:, :2]
    c = xy.mean(axis=0)
    ev, V = np.linalg.eigh(np.cov((xy - c).T))
    major = V[:, 1]
    ext_major = np.ptp((xy - c) @ major)
    ext_minor = np.ptp((xy - c) @ V[:, 0])
    if ext_major / max(ext_minor, 1e-6) > long_aspect:
        d, kind = major, "long"
    else:
        d, kind = np.asarray(belt_dir, float) / np.linalg.norm(belt_dir), "round"
    if d @ np.asarray(belt_dir) < 0:  # always cut in the belt direction
        d = -d
    proj = (xy - c) @ d
    lo, hi = proj.min(), proj.max()
    mid, half = (lo + hi) / 2, (hi - lo) * span / 2
    s = np.arange(mid - half, mid + half + 1e-9, step)
    line = c + np.outer(s, d)
    if faces is not None:
        z = top_height(P, faces[0], faces[1], line)
    else:
        from scipy.spatial import cKDTree  # only needed here; the scheduler uses cut_path alone
        near = cKDTree(xy).query_ball_point(line, r=0.004)
        z = np.array([top[idx, 2].max() if idx else np.nan for idx in near])
    keep = np.isfinite(z)
    contour = np.column_stack([line[keep], z[keep]])
    direction = [float(d[0]), float(d[1]), 0.0]
    poly, i0, i1 = cut_path(contour, direction, -depth, clearance)
    cut = poly[i0:i1 + 1]
    return {"kind": kind, "direction": direction, "surface_z": z[keep].tolist(), "contour": contour.tolist(),
            "points": cut.tolist(), "approach": poly[0].tolist(), "retract": poly[-1].tolist(),
            "path": poly.tolist(), "length_m": float(np.linalg.norm(cut[-1] - cut[0]))}


def order_pieces(cuts, start_xy):
    """Nearest-next order from the robot's start, so the tool travels little between cuts."""
    left, order, here = list(range(len(cuts))), [], np.asarray(start_xy, float)
    while left:
        k = min(left, key=lambda i: np.linalg.norm(np.asarray(cuts[i]["approach"][:2]) - here))
        order.append(k)
        left.remove(k)
        here = np.asarray(cuts[k]["retract"][:2])
    return order
