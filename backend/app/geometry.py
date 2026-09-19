"""Planar homography / measurement geometry.

The calibration rectangle is mapped from *image pixel coordinates* (u, v)
into the real-world plane (x, y) of a known rectangle of width W and height H.
The four corners must be supplied in order:

    0 = top-left  -> (0, 0)
    1 = top-right -> (W, 0)
    2 = bottom-right -> (W, H)
    3 = bottom-left  -> (0, H)

The 3x3 homography H is estimated with the normalized 8-point DLT (we have
exactly 4 point pairs => 8 equations) and validated for invertibility before
use.  The same numerical tolerances are mirrored by the frontend JavaScript
(frontend/app.js) so live results and the authoritative backend results agree.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Tunables (keep in sync with frontend/app.js)
# ---------------------------------------------------------------------------

DUP_TOL = 1e-7          # relative duplicate-point tolerance
COLINEAR_TOL = 1e-9     # |cross| / edge-length ratio for collinearity
MIN_PIVOT = 1e-9        # least-squares pivot threshold (singular DLT)
RESIDUAL_TOL = 1e-7     # max normalized DLT residual
H_DET_TOL = 1e-8        # |det H| relative to |H|^3
DENOM_TOL = 1e-9        # |w| / |H row 3| for mapping through infinity

_EPS = 1e-12


def _point(p) -> np.ndarray:
    return np.asarray(p, dtype=float).reshape(2)


# ---------------------------------------------------------------------------
# Elementary checks
# ---------------------------------------------------------------------------

def _duplicate(points: list[list[float]]) -> tuple[int, int, float] | None:
    """Return (i, j, min_len) for the first coincident point pair, else None."""
    pts = [_point(p) for p in points]
    diag = max(
        (float(np.hypot(*(pts[i] - pts[j]))) for i in range(len(pts)) for j in range(i + 1, len(pts))),
        default=0.0,
    )
    ref = max(diag, 1.0)
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if float(np.hypot(*(pts[i] - pts[j]))) < DUP_TOL * ref:
                return i, j, diag
    return None


def convex_ordered_quad(points: list[list[float]]) -> tuple[bool, str]:
    """True if the 4 points form a strictly convex quadrilateral in order.

    A concave or self-crossing ordering makes the projective map degenerate
    (one triangle folds over), so it is rejected with a readable reason.
    """
    dup = _duplicate(points)
    if dup is not None:
        return False, (
            f"角点 {dup[0] + 1} 与角点 {dup[1] + 1} 重合：四点必须彼此分离"
        )
    pts = [_point(p) for p in points]
    crosses = []
    for i in range(4):
        a = pts[i]
        b = pts[(i + 1) % 4]
        c = pts[(i + 2) % 4]
        cr = float((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
        crosses.append(cr)
    scale = max(
        float(np.hypot(*(pts[(i + 1) % 4] - pts[i]))) for i in range(4)
    ) ** 2
    if any(abs(cr) < COLINEAR_TOL * max(scale, 1.0) for cr in crosses):
        return False, "有三个或更多角点近似共线，四边形已退化"
    if not all(cr > 0 for cr in crosses) and not all(cr < 0 for cr in crosses):
        return False, "四个角点没有构成凸四边形（顺序错乱或自交），请按 左上→右上→右下→左下 依次点击"
    return True, ""


# ---------------------------------------------------------------------------
# Homography (DLT)
# ---------------------------------------------------------------------------

@dataclass
class HomographyResult:
    ok: bool
    H: list[list[float]] | None = None
    errors: list[str] | None = None


def solve_homography(
    image_points: list[list[float]],
    width: float,
    height: float,
) -> HomographyResult:
    """Solve image -> world homography for the calibration rectangle."""
    errors: list[str] = []
    if len(image_points) != 4:
        return HomographyResult(False, None, ["需要恰好 4 个角点"])
    if not (np.isfinite(width) and np.isfinite(height)) or width <= 0 or height <= 0:
        errors.append("已知矩形的宽和高必须为正数")
    convex, reason = convex_ordered_quad(image_points)
    if not convex:
        errors.append(reason)
    if errors:
        return HomographyResult(False, None, errors)

    world_points = [
        [0.0, 0.0],
        [float(width), 0.0],
        [float(width), float(height)],
        [0.0, float(height)],
    ]

    # Normalized DLT: isotropic scaling of both point sets for stability.
    src = np.asarray(image_points, dtype=float)
    dst = np.asarray(world_points, dtype=float)
    Ts, sn = _normalizer(src)
    Td, dn = _normalizer(dst)
    ns = (Ts @ np.vstack([src.T, np.ones((1, 4))]))[:2].T
    nd = (Td @ np.vstack([dst.T, np.ones((1, 4))]))[:2].T

    A = np.zeros((8, 9))
    for i in range(4):
        x, y = ns[i]
        X, Y = nd[i]
        A[2 * i] = [-x, -y, -1, 0, 0, 0, X * x, X * y, X]
        A[2 * i + 1] = [0, 0, 0, -x, -y, -1, Y * x, Y * y, Y]

    try:
        _, svals, Vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return HomographyResult(False, None, ["透视矩阵求解失败：方程组数值不稳定（SVD 未收敛）"])
    s_min = svals[-1]
    if s_min < MIN_PIVOT * max(svals[0], sn * dn, 1.0):
        return HomographyResult(False, None, [
            "透视变换不可逆：标定四点共线或构成退化四边形，无法从平面照片恢复唯一映射"
        ])

    h = Vt[-1].reshape(3, 3)
    Hn = np.linalg.solve(Td, h @ Ts)
    H = Hn / Hn[2, 2]

    # DLT residual: the 4 correspondences must be satisfied.
    res = 0.0
    for (u, v), (X, Y) in zip(image_points, world_points):
        w = H[2, 0] * u + H[2, 1] * v + H[2, 2]
        rx = (H[0, 0] * u + H[0, 1] * v + H[0, 2]) / w - X
        ry = (H[1, 0] * u + H[1, 1] * v + H[1, 2]) / w - Y
        res = max(res, float(np.hypot(rx, ry)))
    if res > RESIDUAL_TOL * max(width, height):
        return HomographyResult(False, None, [
            f"透视校准残差过大（{res:.3g} 单位），角点与已知矩形对应关系不一致，请检查点击顺序"
        ])

    # Invertibility guard.
    det = float(np.linalg.det(H))
    scale = float(np.linalg.norm(H)) ** 3
    if abs(det) < H_DET_TOL * max(scale, 1e-30):
        return HomographyResult(False, None, [
            "透视矩阵行列式接近 0，变换不可逆：四点几乎共线或矩形在照片中退化"
        ])
    row3_scale = float(np.hypot(H[2, 0], H[2, 1]))
    if row3_scale < _EPS:
        row3_scale = abs(float(H[2, 2]))
    for i, (u, v) in enumerate(image_points):
        w = abs(H[2, 0] * u + H[2, 1] * v + H[2, 2])
        if w < DENOM_TOL * max(row3_scale, 1.0):
            return HomographyResult(False, None, [
                f"角点 {i + 1} 映射时分母接近 0（落在无穷远线上），此视角下透视不可求逆"
            ])

    return HomographyResult(True, H.tolist())


def _normalizer(pts: np.ndarray) -> tuple[np.ndarray, float]:
    centroid = pts.mean(axis=0)
    d = np.hypot(*(pts - centroid).T).mean()
    if d < _EPS:
        return np.eye(3), 1.0
    s = np.sqrt(2.0) / d
    T = np.array([[s, 0, -s * centroid[0]], [0, s, -s * centroid[1]], [0, 0, 1]])
    return T, float(s)


# ---------------------------------------------------------------------------
# Point / polygon transformation
# ---------------------------------------------------------------------------

def apply_homography(H: list[list[float]] | np.ndarray, point: list[float]) -> tuple[float, float, float]:
    """Return (x, y, w) where w is the homogeneous denominator."""
    M = np.asarray(H, dtype=float)
    u, v = float(point[0]), float(point[1])
    x = M[0, 0] * u + M[0, 1] * v + M[0, 2]
    y = M[1, 0] * u + M[1, 1] * v + M[1, 2]
    w = M[2, 0] * u + M[2, 1] * v + M[2, 2]
    return x, y, w


def _segments_cross(p: np.ndarray, q: np.ndarray, a: np.ndarray, b: np.ndarray) -> bool:
    """Proper intersection of open segments p-q and a-b (excluding shared endpoints)."""
    def orient(r, s, t):
        return (s[0] - r[0]) * (t[1] - r[1]) - (s[1] - r[1]) * (t[0] - r[0])

    o1 = orient(p, q, a)
    o2 = orient(p, q, b)
    o3 = orient(a, b, p)
    o4 = orient(a, b, q)
    tol = 1e-10 * max(
        float(np.hypot(*(q - p))),
        float(np.hypot(*(b - a))),
        1.0,
    )
    if all(abs(o) < tol for o in (o1, o2, o3, o4)):
        # Collinear overlap -> self-intersecting.
        def on_seg(x, r, s):
            return (
                min(r[0], s[0]) - tol <= x[0] <= max(r[0], s[0]) + tol
                and min(r[1], s[1]) - tol <= x[1] <= max(r[1], s[1]) + tol
            )

        return any(on_seg(z, p, q) and on_seg(z, a, b) for z in (p, q, a, b)) and not (
            np.allclose(p, a, atol=tol) or np.allclose(p, b, atol=tol)
            or np.allclose(q, a, atol=tol) or np.allclose(q, b, atol=tol)
        )
    return ((o1 > tol and o2 < -tol) or (o1 < -tol and o2 > tol)) and (
        (o3 > tol and o4 < -tol) or (o3 < -tol and o4 > tol)
    )


def polygon_self_intersects(points: list[list[float]]) -> tuple[bool, str]:
    """Check a closed polygon for duplicate vertices or edge crossings."""
    n = len(points)
    if n < 3:
        return False, ""
    dup = _duplicate(points)
    if dup is not None:
        return True, f"顶点 {dup[0] + 1} 与顶点 {dup[1] + 1} 重合"
    pts = [_point(p) for p in points]
    for i in range(n):
        p, q = pts[i], pts[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i or (j + 1) % n == i or (i + 1) % n == j:
                continue  # edges sharing a vertex are allowed
            a, b = pts[j], pts[(j + 1) % n]
            if _segments_cross(p, q, a, b):
                return True, f"边 {i + 1} 与边 {j + 1} 相交"
    return False, ""


@dataclass
class MeasurementResult:
    ok: bool
    area: float | None = None
    perimeter: float | None = None
    world_points: list[list[float]] | None = None
    errors: list[str] | None = None


def measure_polygon(H: list[list[float]], image_points: list[list[float]]) -> MeasurementResult:
    """Transform a closed polygon and compute its area/perimeter in real units."""
    if len(image_points) < 3:
        return MeasurementResult(False, None, None, None, ["多边形至少需要 3 个顶点"])

    row3 = np.asarray(H, dtype=float)[2]
    row3_scale = float(np.hypot(row3[0], row3[1]))
    if row3_scale < _EPS:
        row3_scale = abs(float(row3[2]))

    world: list[list[float]] = []
    for i, p in enumerate(image_points):
        x, y, w = apply_homography(H, p)
        if abs(w) < DENOM_TOL * max(row3_scale, 1.0):
            return MeasurementResult(False, None, None, None, [
                f"多边形顶点 {i + 1} 的透视分母接近 0（位于无穷远线上），无法计算校正后坐标"
            ])
        world.append([x / w, y / w])

    crosses, reason = polygon_self_intersects(world)
    if crosses:
        return MeasurementResult(False, None, None, None, [
            f"多边形自交（{reason}），自交多边形没有唯一的面积，请调整顶点"
        ])

    n = len(world)
    area2 = 0.0
    perimeter = 0.0
    for i in range(n):
        x1, y1 = world[i]
        x2, y2 = world[(i + 1) % n]
        area2 += x1 * y2 - x2 * y1
        perimeter += float(np.hypot(x2 - x1, y2 - y1))
    return MeasurementResult(True, abs(area2) / 2.0, perimeter, world)
