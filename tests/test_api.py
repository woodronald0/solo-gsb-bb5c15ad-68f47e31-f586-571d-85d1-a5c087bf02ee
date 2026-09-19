"""End-to-end API test using FastAPI TestClient and a generated photo.

The photo renders a known rectangle in perspective (using the same camera
matrix as test_geometry.py); corners are queried from the projection and
posted to the API. Run: python tests/test_api.py
"""
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

client = TestClient(app)

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        fails.append(name)


# ---------------------------------------------------------------- camera ---
# A closer camera with noticeable perspective; the whole target fits a
# 1024x768 photo.
K = np.array([[800.0, 0, 512], [0, 800, 384], [0, 0, 1.0]])
r1 = np.array([0.99, 0.0, -0.1494])
r2 = np.array([0.06, 1.0, 0.09])
P = K @ np.column_stack([r1, r2, [0.0, 0.0, 1.2]])


def project(x, y):
    v = P @ np.array([x, y, 1.0])
    return v[0] / v[2], v[1] / v[2]


# Make a simple image (the pixels are irrelevant; geometry comes from the API
# points, but endpoints require a decodable image).
buf = io.BytesIO()
Image.new("RGB", (1024, 768), (40, 50, 60)).save(buf, "JPEG")
buf.seek(0)

print("== session + upload ==")
sid = client.post("/api/sessions").json()["session_id"]
check("session created", bool(sid))

r = client.post(f"/api/sessions/{sid}/image", files={"file": ("t.jpg", buf, "image/jpeg")})
check("image upload", r.status_code == 200, str(r.status_code))
check("image dims", r.json()["image"]["width"] == 1024)

r = client.post(f"/api/sessions/{sid}/image", files={"file": ("x.bin", b"not-an-image", "application/octet-stream")})
check("garbage upload rejected", r.status_code == 400)

r = client.get(f"/api/sessions/{'deadbeef'}")
check("missing session 404", r.status_code == 404)

print("== calibration ==")
W, H = 0.1, 0.06
corners = [project(0, 0), project(W, 0), project(W, H), project(0, H)]
payload = {"corners": [{"x": float(u), "y": float(v)} for u, v in corners],
           "rect_width": W, "rect_height": H, "unit": "mm"}
r = client.put(f"/api/sessions/{sid}/calibration", json=payload)
check("calibration 200", r.status_code == 200, str(r.status_code))
cal = r.json()["calibration"]
check("calibration valid", cal["valid"], cal.get("errors"))
check("H is 3x3", cal["valid"] and len(cal["H"]) == 3 and len(cal["H"][0]) == 3)

# degenerate calibration
bad = {"corners": [{"x": 0, "y": 0}, {"x": 50, "y": 0}, {"x": 100, "y": 0}, {"x": 150, "y": 0}],
       "rect_width": W, "rect_height": H, "unit": "mm"}
r = client.put(f"/api/sessions/{sid}/calibration", json=bad)
check("collinear rejected", r.status_code == 200 and not r.json()["calibration"]["valid"])

# restore good calibration for the rest
client.put(f"/api/sessions/{sid}/calibration", json=payload)

print("== polygons ==")
# interior rectangle [0.02,0.07] x [0.01,0.05] -> area .05*.04=.002, perim .18
poly_img = [project(0.02, 0.01), project(0.07, 0.01), project(0.07, 0.05), project(0.02, 0.05)]
r = client.post(f"/api/sessions/{sid}/polygons",
                json={"points": [{"x": float(u), "y": float(v)} for u, v in poly_img]})
check("polygon add", r.status_code == 200)
p1 = r.json()["polygon"]
check("poly area .002", p1["valid"] and abs(p1["area"] - 0.002) < 1e-9, f"area={p1['area']}")
check("poly perim .18", p1["valid"] and abs(p1["perimeter"] - 0.18) < 1e-9)

# self-intersecting polygon
bt = [project(0, 0), project(W, H), project(W, 0), project(0, H)]
r = client.post(f"/api/sessions/{sid}/polygons",
                json={"points": [{"x": float(u), "y": float(v)} for u, v in bt]})
p2 = r.json()["polygon"]
check("self-intersect blocked", not p2["valid"] and "自交" in p2["errors"][0], p2["errors"])

# update + delete: triangle (0,0),(0.05,0),(0,0.03) -> area .00075
r = client.put(f"/api/sessions/{sid}/polygons/{p2['id']}",
               json={"points": [{"x": float(u), "y": float(v)} for u, v in
                                [project(0, 0), project(0.05, 0), project(0, 0.03)]],
                     "name": "三角"})
p2u = r.json()["polygon"]
check("poly update valid", p2u["valid"] and abs(p2u["area"] - 0.00075) < 1e-9, f"area={p2u['area']}")
check("poly rename", p2u["name"] == "三角")

r = client.delete(f"/api/sessions/{sid}/polygons/{p2['id']}")
check("poly delete", r.status_code == 200)

# out-of-range vertex rejected
r = client.post(f"/api/sessions/{sid}/polygons",
                json={"points": [{"x": -5, "y": 0}, {"x": 1, "y": 0}, {"x": 1, "y": 1}]})
check("out-of-range rejected", r.status_code == 400)

print("== regions with holes ==")
# outer world rect [0.01,0.09] x [0.01,0.05]: area .08*.04=.0032, perim .24
outer_img = [project(0.01, 0.01), project(0.09, 0.01), project(0.09, 0.05), project(0.01, 0.05)]
# hole [0.02,0.03] x [0.02,0.03]: area .01*.01=.0001, perim .04
hole_img = [project(0.02, 0.02), project(0.03, 0.02), project(0.03, 0.03), project(0.02, 0.03)]
r = client.post(f"/api/sessions/{sid}/polygons",
                json={"points": [{"x": float(u), "y": float(v)} for u, v in outer_img],
                      "holes": [{"points": [{"x": float(u), "y": float(v)} for u, v in hole_img]}]})
check("region add", r.status_code == 200, str(r.status_code))
h1 = r.json()["polygon"]
check("region valid", h1["valid"], h1["errors"])
check("net area .0031", h1["valid"] and abs(h1["area"] - 0.0031) < 1e-9, f"area={h1['area']}")
check("total perim .28", h1["valid"] and abs(h1["perimeter"] - 0.28) < 1e-9, f"p={h1['perimeter']}")
check("outer area .0032", h1["valid"] and abs(h1["outer_area"] - 0.0032) < 1e-9)
check("hole payload", len(h1["holes"]) == 1 and h1["holes"][0]["valid"]
      and abs(h1["holes"][0]["area"] - 0.0001) < 1e-9, str(h1.get("holes")))
check("hole has id", bool(h1["holes"][0]["id"]))

# invalid candidate on PUT must be rejected with 409 and NOT overwrite stored data
hole_touch = [project(0.01, 0.01), project(0.02, 0.01), project(0.02, 0.02), project(0.01, 0.02)]
r = client.put(f"/api/sessions/{sid}/polygons/{h1['id']}",
               json={"points": [{"x": float(u), "y": float(v)} for u, v in outer_img],
                     "holes": [{"points": [{"x": float(u), "y": float(v)} for u, v in hole_touch]}]})
check("invalid update -> 409", r.status_code == 409, str(r.status_code))
if r.status_code == 409:
    body = r.json()
    check("409 keeps stored", body.get("conflict") and body["polygon"]["valid"]
          and abs(body["polygon"]["area"] - 0.0031) < 1e-9)
    check("409 reports candidate", "相切" in body["candidate"]["errors"][0]
          or "相交" in body["candidate"]["errors"][0], body["candidate"]["errors"])
# server really kept the last valid result
r = client.get(f"/api/sessions/{sid}")
stored = [p for p in r.json()["polygons"] if p["id"] == h1["id"]][0]
check("stored result intact", stored["valid"] and abs(stored["area"] - 0.0031) < 1e-9)

# hole fully outside the outer ring -> invalid (persisted on POST, marked invalid)
hole_out = [project(0.05, 0.0), project(0.06, 0.0), project(0.06, 0.005), project(0.05, 0.005)]
r = client.post(f"/api/sessions/{sid}/polygons",
                json={"points": [{"x": float(u), "y": float(v)} for u, v in outer_img],
                      "holes": [{"points": [{"x": float(u), "y": float(v)} for u, v in hole_out]}]})
bad_region = r.json()["polygon"]
check("outside hole invalid", not bad_region["valid"]
      and "之外" in bad_region["errors"][0], bad_region["errors"])
# invalid geometry IS persisted on creation; GET reflects it (no prior valid
# result to protect), and session reload keeps it
r = client.get(f"/api/sessions/{sid}")
stored_bad = [p for p in r.json()["polygons"] if p["id"] == bad_region["id"]]
check("invalid region persisted", bool(stored_bad) and not stored_bad[0]["valid"])

# two holes tangent to each other -> invalid
ha = [project(0.02, 0.02), project(0.03, 0.02), project(0.03, 0.03), project(0.02, 0.03)]
hb = [project(0.03, 0.025), project(0.04, 0.025), project(0.04, 0.035), project(0.03, 0.035)]
r = client.post(f"/api/sessions/{sid}/polygons",
                json={"points": [{"x": float(u), "y": float(v)} for u, v in outer_img],
                      "holes": [
                          {"points": [{"x": float(u), "y": float(v)} for u, v in ha]},
                          {"points": [{"x": float(u), "y": float(v)} for u, v in hb]}]})
tg = r.json()["polygon"]
check("tangent holes invalid", not tg["valid"] and "孔洞" in tg["errors"][0], tg["errors"])

# repair: two fully separated holes -> area .0032 - .0001 - .0001 = .003
hc = [project(0.07, 0.03), project(0.08, 0.03), project(0.08, 0.04), project(0.07, 0.04)]
r = client.put(f"/api/sessions/{sid}/polygons/{tg['id']}",
               json={"points": [{"x": float(u), "y": float(v)} for u, v in outer_img],
                     "holes": [
                         {"points": [{"x": float(u), "y": float(v)} for u, v in ha]},
                         {"points": [{"x": float(u), "y": float(v)} for u, v in hc]}]})
fix = r.json()["polygon"]
check("repair valid", fix["valid"], fix.get("errors"))
check("two-hole net area", fix["valid"] and abs(fix["area"] - 0.003) < 1e-9, f"area={fix['area']}")
check("two-hole perim", fix["valid"] and abs(fix["perimeter"] - 0.32) < 1e-9, f"p={fix['perimeter']}")

# remove holes (holes: []) -> plain region again
r = client.put(f"/api/sessions/{sid}/polygons/{h1['id']}",
               json={"points": [{"x": float(u), "y": float(v)} for u, v in outer_img],
                     "holes": []})
plain = r.json()["polygon"]
check("holes removed", plain["valid"] and abs(plain["area"] - 0.0032) < 1e-9
      and plain["holes"] == [])

print("== export ==")
r = client.get(f"/api/sessions/{sid}/export")
check("export 200", r.status_code == 200)
data = r.json()
check("schema tag", data["schema"] in ("photo-ruler/1.0", "photo-ruler/1.1"))
check("calibration in export", data["calibration"] and data["calibration"]["valid"])
# polygons present: original p1 (area .002), h1 (now holes removed),
# bad_region (invalid, outside hole), fix (two holes)
check("polygon count", len(data["polygons"]) == 4, str(len(data["polygons"])))
check("all valid flag", data["all_measurements_valid"] is False)
check("export area", abs(data["polygons"][0]["area"] - 0.002) < 1e-9)
check("homography present", len(data["calibration"]["homography_image_to_world"]) == 3)
exp_fix = [p for p in data["polygons"] if p["id"] == fix["id"]][0]
check("export holes", len(exp_fix["holes"]) == 2
      and all(h["world_points"] and abs(h["area"] - 0.0001) < 1e-9 for h in exp_fix["holes"]))
check("export hole img pts", len(exp_fix["holes"][0]["image_points"]) == 4)
check("export net area", abs(exp_fix["area"] - 0.003) < 1e-9)
check("export outer area", abs(exp_fix["outer_area"] - 0.0032) < 1e-9)

print("== persistence ==")
r = client.get(f"/api/sessions/{sid}")
check("session reload has data", r.json()["image"] is not None and len(r.json()["polygons"]) == 4)
reloaded_fix = [p for p in r.json()["polygons"] if p["id"] == fix["id"]][0]
check("holes survive reload", reloaded_fix["valid"] and len(reloaded_fix["holes"]) == 2)

r = client.get("/api/health")
check("health", r.json()["status"] == "ok")

print()
if fails:
    print(f"{len(fails)} FAILURES: {fails}")
    sys.exit(1)
print("ALL API TESTS PASSED")
