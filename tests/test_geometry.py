"""Numerical sanity tests for geometry.py using a known synthetic homography.

Run: python tests/test_geometry.py
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.geometry import (  # noqa: E402
    convex_ordered_quad,
    measure_polygon,
    measure_region,
    polygon_self_intersects,
    solve_homography,
)

# World -> image: a known, well-conditioned projective transform.
# A plane seen by a pinhole camera: image ~ K (r1*X + r2*Y + t), divided by
# depth.  Chosen so the whole rectangle stays well in front of the camera.
K = np.array([[800.0, 0.0, 512.0], [0.0, 800.0, 384.0], [0.0, 0.0, 1.0]])
r1 = np.array([0.999, 0.0, -0.0499])
r2 = np.array([0.02, 1.0, 0.03])
t = np.array([0.0, 0.0, 12.0])
# P restricted to world points (x, y, 0, 1): columns r1, r2, t.
P = K @ np.column_stack([r1, r2, t])  # 3x3 mapping (x,y,1) -> image


def project(x, y):
    v = P @ np.array([x, y, 1.0])
    assert v[2] > 1.0, "test point must be in front of camera"
    return [float(v[0] / v[2]), float(v[1] / v[2])]


W, H = 100.0, 60.0
corners_world = [[0, 0], [W, 0], [W, H], [0, H]]
corners_img = [project(x, y) for x, y in corners_world]

failures = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        failures.append(name)


print("== homography recovery ==")
res = solve_homography(corners_img, W, H)
check("solve ok", res.ok, str(res.errors))
if res.ok:
    Hm = np.array(res.H)
    max_err = 0.0
    for (x, y), (u, v) in zip(corners_world, corners_img):
        w = Hm[2, 0] * u + Hm[2, 1] * v + Hm[2, 2]
        xx = (Hm[0, 0] * u + Hm[0, 1] * v + Hm[0, 2]) / w
        yy = (Hm[1, 0] * u + Hm[1, 1] * v + Hm[1, 2]) / w
        max_err = max(max_err, math.hypot(xx - x, yy - y))
    check("corner remap residual ~0", max_err < 1e-4, f"err={max_err:.2e}")

    # Interior polygon: rectangle [20,70] x [10,50] -> area 50*40=2000, perim 180
    poly_img = [project(20, 10), project(70, 10), project(70, 50), project(20, 50)]
    m = measure_polygon(res.H, poly_img)
    check("interior polygon valid", m.ok, str(m.errors))
    check("area == 2000", m.ok and abs(m.area - 2000.0) < 1e-4, f"area={m.area:.6f}")
    check("perimeter == 180", m.ok and abs(m.perimeter - 180.0) < 1e-4, f"p={m.perimeter:.6f}")

    # Triangle: (0,0),(100,0),(0,60) -> area 3000, perim 100+60+sqrt(13600)
    tri = [project(0, 0), project(100, 0), project(0, 60)]
    mt = measure_polygon(res.H, tri)
    expect_p = 100 + 60 + math.hypot(100, 60)
    check("triangle area == 3000", mt.ok and abs(mt.area - 3000.0) < 1e-4, f"area={mt.area:.6f}")
    check("triangle perimeter", mt.ok and abs(mt.perimeter - expect_p) < 1e-4, f"p={mt.perimeter:.6f}")

print("== invalid calibrations ==")
# duplicate points
r = solve_homography([[0, 0], [100, 0], [100, 0], [0, 60]], W, H)
check("duplicate rejected", not r.ok and "重合" in r.errors[0])
# collinear
r = solve_homography([[0, 0], [50, 0], [100, 0], [150, 0]], W, H)
check("collinear rejected", not r.ok)
# self-crossing order (bowtie)
r = solve_homography([corners_img[0], corners_img[2], corners_img[1], corners_img[3]], W, H)
check("bowtie order rejected", not r.ok)
# bad dimensions
r = solve_homography(corners_img, 0, H)
check("zero width rejected", not r.ok)

print("== polygon self-intersection ==")
bowtie = [[0, 0], [10, 10], [10, 0], [0, 10]]
si = polygon_self_intersects(bowtie)
check("bowtie detected", si[0], str(si[1]))
okquad = [[0, 0], [10, 0], [10, 10], [0, 10]]
check("square clean", not polygon_self_intersects(okquad)[0])
dupped = [[0, 0], [10, 0], [10, 10], [10, 10]]
check("duplicate vertex detected", polygon_self_intersects(dupped)[0])

if res.ok:
    # self-intersecting polygon in image plane maps to self-intersection
    # build bowtie from projected world points
    bt_img = [project(0, 0), project(100, 60), project(100, 0), project(0, 60)]
    m = measure_polygon(res.H, bt_img)
    check("self-intersecting polygon blocked", not m.ok and "自交" in m.errors[0])

print("== regions with holes (world-coordinate ring relations) ==")
Hm = res.H if res.ok else None
if Hm:
    # outer [10,90] x [10,50] -> area 80*40=3200, perim 240
    outer = [project(10, 10), project(90, 10), project(90, 50), project(10, 50)]
    # hole [20,30] x [20,30] -> area 100, perim 40
    hole = [project(20, 20), project(30, 20), project(30, 30), project(20, 30)]
    m = measure_region(Hm, outer, [hole])
    check("region valid", m.ok, str(m.errors))
    check("net area 3100", m.ok and abs(m.area - 3100) < 1e-4, f"area={m.area}")
    check("total perim 280", m.ok and abs(m.perimeter - 280) < 1e-4, f"p={m.perimeter}")
    check("outer area 3200", m.ok and abs(m.outer_area - 3200) < 1e-4)
    check("hole area 100", m.ok and abs(m.hole_areas[0] - 100) < 1e-4)
    check("world holes returned", m.ok and len(m.world_holes) == 1)

    # two separate holes: each area 100, perim 40
    hole2 = [project(40, 20), project(50, 20), project(50, 30), project(40, 30)]
    m = measure_region(Hm, outer, [hole, hole2])
    check("two holes valid", m.ok and abs(m.area - 3000) < 1e-4
          and abs(m.perimeter - 320) < 1e-4, f"area={m.area} p={m.perimeter}")

    # tangent to outer edge (shares edge y=10)
    t = [project(20, 5), project(30, 5), project(30, 10), project(20, 10)]
    m = measure_region(Hm, outer, [t])
    check("tangent outer rejected", not m.ok and "相切" in m.errors[0], m.errors)

    # vertex sitting on outer boundary
    b = [project(20, 20), project(90, 30), project(20, 40)]
    m = measure_region(Hm, outer, [b])
    check("boundary vertex rejected", not m.ok
          and ("相切" in m.errors[0] or "相交" in m.errors[0]), m.errors)

    # hole outside the outer ring
    o = [project(92, 20), project(98, 20), project(98, 30), project(92, 30)]
    m = measure_region(Hm, outer, [o])
    check("outside hole rejected", not m.ok and "之外" in m.errors[0], m.errors)

    # holes touching each other (shared point)
    a1 = [project(20, 20), project(30, 20), project(30, 30), project(20, 30)]
    a2 = [project(30, 25), project(40, 25), project(40, 35), project(30, 35)]
    m = measure_region(Hm, outer, [a1, a2])
    check("touching holes rejected", not m.ok and "孔洞 1 与孔洞 2" in m.errors[0], m.errors)

    # nested holes
    n1 = [project(15, 15), project(85, 15), project(85, 45), project(15, 45)]
    n2 = [project(20, 20), project(30, 20), project(30, 30), project(20, 30)]
    m = measure_region(Hm, outer, [n1, n2])
    check("nested holes rejected", not m.ok and "包含" in " ".join(m.errors), m.errors)

    # degenerate (collinear) hole
    d = [project(20, 20), project(30, 20), project(40, 20)]
    m = measure_region(Hm, outer, [d])
    check("degenerate hole rejected", not m.ok and "面积为 0" in m.errors[0], m.errors)

    # self-crossing hole
    sc = [project(20, 20), project(40, 40), project(40, 20), project(20, 40)]
    m = measure_region(Hm, outer, [sc])
    check("self-cross hole rejected", not m.ok and any("自交" in e for e in m.errors), m.errors)

# identity-H direct cases (error message wording)
ident = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
m = measure_region(ident, [[0, 0], [10, 0], [10, 10], [0, 10]],
                   [[[2, 2], [4, 2], [4, 4], [2, 4]]])
check("identity net area", m.ok and abs(m.area - 96) < 1e-9 and abs(m.perimeter - 48) < 1e-9)
m = measure_region(ident, [[0, 0], [10, 0]], [])
check("too few outer verts", not m.ok and "外环" in m.errors[0])
m = measure_region(ident, [[0, 0], [10, 0], [10, 10]], [[[1, 1], [2, 1]]])
check("too few hole verts", not m.ok and any("孔洞 1" in e for e in m.errors), m.errors)

print()
if failures:
    print(f"{len(failures)} FAILURES: {failures}")
    sys.exit(1)
print("ALL GEOMETRY TESTS PASSED")
