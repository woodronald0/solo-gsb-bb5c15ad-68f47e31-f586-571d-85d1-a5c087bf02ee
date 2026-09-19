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

print()
if failures:
    print(f"{len(failures)} FAILURES: {failures}")
    sys.exit(1)
print("ALL GEOMETRY TESTS PASSED")
