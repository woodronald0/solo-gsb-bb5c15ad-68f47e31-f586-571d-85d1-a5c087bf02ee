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
CONTACT_TOL = 1e-8      # |cross| / edge-length: rings intersecting or tangent
ZERO_AREA_TOL = 1e-10   # |ring area| relative to ring diameter squared

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


# ---------------------------------------------------------------------------
# Regions with holes
#
# A region is one closed outer ring plus zero or more closed hole rings.
# Ring relations are always checked in *corrected world coordinates* (the
# projective map can change apparent containment in the photo):
#   - every hole must lie strictly inside the outer ring;
#   - holes must not intersect or touch the outer ring;
#   - holes must not intersect or touch one another (tangency included).
# Area = outer area - sum(hole areas); perimeter counts every boundary.
# ---------------------------------------------------------------------------

def _ring_diameter(pts: list[np.ndarray]) -> float:
    diag = 0.0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            diag = max(diag, float(np.hypot(*(pts[i] - pts[j]))))
    return diag


def _closed_segments_touch(p: np.ndarray, q: np.ndarray,
                           a: np.ndarray, b: np.ndarray) -> bool:
    """True when closed segments p-q and a-b have any common point.

    Includes proper crossings, endpoint-on-segment tangency and collinear
    overlap (the open-segment test used for self-intersection intentionally
    ignores shared endpoints — rings in a region may not share any).
    """
    def orient(r, s, t):
        return (s[0] - r[0]) * (t[1] - r[1]) - (s[1] - r[1]) * (t[0] - r[0])

    o1 = orient(p, q, a)
    o2 = orient(p, q, b)
    o3 = orient(a, b, p)
    o4 = orient(a, b, q)
    tol = CONTACT_TOL * max(float(np.hypot(*(q - p))), float(np.hypot(*(b - a))), 1.0)
    if all(abs(o) < tol for o in (o1, o2, o3, o4)):
        def on_seg(x, r, s):
            return (
                min(r[0], s[0]) - tol <= x[0] <= max(r[0], s[0]) + tol
                and min(r[1], s[1]) - tol <= x[1] <= max(r[1], s[1]) + tol
            )
        return any(on_seg(z, p, q) and on_seg(z, a, b)
                   for z in (p, q, a, b))
    return (
        ((o1 > tol and o2 < -tol) or (o1 < -tol and o2 > tol)
         or (-tol <= o1 <= tol) or (-tol <= o2 <= tol))
        and
        ((o3 > tol and o4 < -tol) or (o3 < -tol and o4 > tol)
         or (-tol <= o3 <= tol) or (-tol <= o4 <= tol))
    )


def _rings_touch(ring_a: list[np.ndarray], ring_b: list[np.ndarray]) -> tuple[int, int] | None:
    """Return (edge index in A, edge index in B) of the first touching edges."""
    for i, p in enumerate(ring_a):
        q = ring_a[(i + 1) % len(ring_a)]
        for j, a in enumerate(ring_b):
            b = ring_b[(j + 1) % len(ring_b)]
            if _closed_segments_touch(p, q, a, b):
                return i, j
    return None


def _point_on_ring_boundary(pt: np.ndarray, ring: list[np.ndarray], tol: float) -> bool:
    for i, a in enumerate(ring):
        b = ring[(i + 1) % len(ring)]
        cross = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
        if abs(cross) > tol * max(float(np.hypot(*(b - a))), 1.0):
            continue
        if (min(a[0], b[0]) - tol <= pt[0] <= max(a[0], b[0]) + tol
                and min(a[1], b[1]) - tol <= pt[1] <= max(a[1], b[1]) + tol):
            return True
    return False


def _point_in_ring_strict(pt: np.ndarray, ring: list[np.ndarray],
                          tol: float) -> str:
    """'inside' / 'outside' / 'boundary' for a point against one closed ring."""
    if _point_on_ring_boundary(pt, ring, tol):
        return "boundary"
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        a, b = ring[i], ring[j]
        if (a[1] > pt[1]) != (b[1] > pt[1]):
            xcross = a[0] + (b[0] - a[0]) * (pt[1] - a[1]) / (b[1] - a[1])
            if pt[0] < xcross - tol:
                inside = not inside
        j = i
    return "inside" if inside else "outside"


def _ring_signed_area2(pts: list[np.ndarray]) -> float:
    area2 = 0.0
    for i, a in enumerate(pts):
        b = pts[(i + 1) % len(pts)]
        area2 += a[0] * b[1] - b[0] * a[1]
    return area2


def _ring_perimeter(pts: list[np.ndarray]) -> float:
    return sum(
        float(np.hypot(*(pts[(i + 1) % len(pts)] - pts[i])))
        for i in range(len(pts))
    )


@dataclass
class RegionResult:
    ok: bool
    area: float | None = None
    perimeter: float | None = None
    outer_area: float | None = None
    hole_areas: list[float] | None = None
    hole_perimeters: list[float] | None = None
    world_points: list[list[float]] | None = None
    world_holes: list[list[list[float]]] | None = None
    errors: list[str] | None = None


def measure_region(
    H: list[list[float]],
    outer_image: list[list[float]],
    holes_image: list[list[list[float]]] | None = None,
) -> RegionResult:
    """Measure a region with holes; all ring checks use corrected world coords."""
    holes_image = holes_image or []
    if len(outer_image) < 3:
        return RegionResult(False, errors=["外环至少需要 3 个顶点"])

    row3 = np.asarray(H, dtype=float)[2]
    row3_scale = float(np.hypot(row3[0], row3[1]))
    if row3_scale < _EPS:
        row3_scale = abs(float(row3[2]))

    def to_world(points: list[list[float]], label: str) -> list[np.ndarray] | None:
        out: list[np.ndarray] = []
        for i, p in enumerate(points):
            x, y, w = apply_homography(H, p)
            if abs(w) < DENOM_TOL * max(row3_scale, 1.0):
                errs.append(f"{label}顶点 {i + 1} 的透视分母接近 0（位于无穷远线上），无法计算校正后坐标")
                return None
            out.append(np.array([x / w, y / w], dtype=float))
        return out

    errs: list[str] = []

    # --- per-ring structural validity (duplicates, self-crossing, zero area)
    def ring_errors(points: list[list[float]], world: list[np.ndarray],
                    label: str) -> list[str]:
        out: list[str] = []
        if len(points) < 3:
            out.append(f"{label} 至少需要 3 个顶点")
            return out
        crosses, reason = polygon_self_intersects([[float(p[0]), float(p[1])] for p in world])
        if crosses:
            out.append(f"{label} {reason}：自交或顶点重合的多边形没有唯一的面积，请调整顶点")
        if abs(_ring_signed_area2(world)) / 2.0 < ZERO_AREA_TOL * max(_ring_diameter(world) ** 2, 1.0):
            out.append(f"{label} 面积为 0（顶点近似共线或退化），请调整顶点")
        return out

    world_outer = to_world(outer_image, "外环")
    if world_outer is None:
        return RegionResult(False, errors=errs)
    errs += ring_errors(outer_image, world_outer, "外环")

    world_holes: list[list[np.ndarray]] = []
    for hi, hpts in enumerate(holes_image):
        wh = to_world(hpts, f"孔洞 {hi + 1}")
        # to_world appends a single error and returns None on failure
        if wh is None:
            continue
        world_holes.append(wh)
        errs += ring_errors(hpts, wh, f"孔洞 {hi + 1}")
    if errs:
        return RegionResult(False, errors=errs)

    tol = CONTACT_TOL * max(_ring_diameter(world_outer), 1.0)

    # --- outer ring vs each hole: no touch, strict containment
    for hi, wh in enumerate(world_holes):
        tag = f"孔洞 {hi + 1}"
        touch = _rings_touch(world_outer, wh)
        if touch is not None:
            errs.append(
                f"{tag} 与外环相交或相切（孔洞边 {touch[1] + 1} 碰到外环边 {touch[0] + 1}），"
                f"孔洞必须完全位于外环内部且互不接触"
            )
            continue
        bad_vertex = None
        for vi, v in enumerate(wh):
            if _point_in_ring_strict(v, world_outer, tol) != "inside":
                bad_vertex = vi
                break
        if bad_vertex is not None:
            errs.append(f"{tag} 顶点 {bad_vertex + 1} 位于外环之外，孔洞必须完全位于外环内部")

    # --- hole vs hole: disjoint (may not touch; nesting is not allowed either)
    pair_errors = 0
    for i in range(len(world_holes)):
        for j in range(i + 1, len(world_holes)):
            touch = _rings_touch(world_holes[i], world_holes[j])
            if touch is not None:
                errs.append(
                    f"孔洞 {i + 1} 与孔洞 {j + 1} 相交或相切"
                    f"（孔洞 {i + 1} 边 {touch[0] + 1} 与孔洞 {j + 1} 边 {touch[1] + 1} 接触），孔洞之间必须完全分离"
                )
                pair_errors += 1
            else:
                bad = None
                for vi, v in enumerate(world_holes[j]):
                    if _point_in_ring_strict(v, world_holes[i], tol) != "outside":
                        bad = (j, vi)
                        break
                if bad is None:
                    for vi, v in enumerate(world_holes[i]):
                        if _point_in_ring_strict(v, world_holes[j], tol) != "outside":
                            bad = (i, vi)
                            break
                if bad is not None:
                    errs.append(
                        f"孔洞 {bad[0] + 1} 与孔洞 {i + 1 if bad[0] == j else j + 1} "
                        f"未完全分离（孔洞 {bad[0] + 1} 顶点 {bad[1] + 1} 落在另一孔洞内或边界上），孔洞不允许相互包含"
                    )
                    pair_errors += 1
            if pair_errors >= 6:
                errs.append("……（还有更多孔洞冲突，未全部列出）")
                break
        if pair_errors >= 6:
            break
    if errs:
        return RegionResult(False, errors=errs)

    outer_area = abs(_ring_signed_area2(world_outer)) / 2.0
    hole_areas = [abs(_ring_signed_area2(wh)) / 2.0 for wh in world_holes]
    perimeter = _ring_perimeter(world_outer) + sum(
        _ring_perimeter(wh) for wh in world_holes
    )
    return RegionResult(
        True,
        area=outer_area - sum(hole_areas),
        perimeter=perimeter,
        outer_area=outer_area,
        hole_areas=hole_areas,
        hole_perimeters=[_ring_perimeter(wh) for wh in world_holes],
        world_points=[[float(p[0]), float(p[1])] for p in world_outer],
        world_holes=[[[float(p[0]), float(p[1])] for p in wh] for wh in world_holes],
        errors=[],
    )
