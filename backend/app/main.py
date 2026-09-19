"""FastAPI backend for the planar photo ruler calibration app.

Sessions are kept in memory and mirrored to ``data/sessions`` as JSON so a
container restart (with the volume mounted, see docker-compose.yml) keeps
measurements.  Images live in ``data/images``.
"""

from __future__ import annotations

import io
import json
import math
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

from .geometry import HomographyResult, measure_polygon, solve_homography

BASE_DIR = Path(__file__).resolve().parent
# backend/app/main.py -> parents: backend/, workspace/
DATA_DIR = BASE_DIR.parents[1] / "data"
IMG_DIR = DATA_DIR / "images"
SESS_DIR = DATA_DIR / "sessions"
STATIC_DIR = BASE_DIR.parents[1] / "frontend"
IMG_DIR.mkdir(parents=True, exist_ok=True)
SESS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="平面照片标尺校准", version="1.0.0")

_LOCK = threading.RLock()
_SESSIONS: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _session_path(session_id: str) -> Path:
    return SESS_DIR / f"{session_id}.json"


def _save_session(s: dict[str, Any]) -> None:
    tmp = _session_path(s["id"]).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(_session_path(s["id"]))


def _load_sessions() -> None:
    for f in SESS_DIR.glob("*.json"):
        try:
            s = json.loads(f.read_text(encoding="utf-8"))
            _SESSIONS[s["id"]] = s
        except (json.JSONDecodeError, KeyError):
            continue


_load_sessions()


def _get(session_id: str) -> dict[str, Any]:
    s = _SESSIONS.get(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail="会话不存在（可能已随容器重启丢失）")
    return s


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class Point(BaseModel):
    x: float = Field(..., description="图像像素坐标 u")
    y: float = Field(..., description="图像像素坐标 v")


class CalibrationIn(BaseModel):
    corners: list[Point] = Field(..., min_length=4, max_length=4)
    rect_width: float = Field(..., gt=0)
    rect_height: float = Field(..., gt=0)
    unit: str = Field("mm", min_length=1, max_length=16)


class PolygonIn(BaseModel):
    points: list[Point] = Field(..., min_length=1, max_length=2000)
    name: str | None = Field(None, max_length=80)


class PolygonPatch(BaseModel):
    points: list[Point] = Field(..., min_length=1, max_length=2000)
    name: str | None = Field(None, max_length=80)


def _cal_payload(s: dict[str, Any]) -> dict[str, Any] | None:
    cal = s.get("calibration")
    if not cal:
        return None
    return {
        "corners": cal["corners"],
        "rect_width": cal["rect_width"],
        "rect_height": cal["rect_height"],
        "unit": cal["unit"],
        "H": cal.get("H"),
        "valid": cal.get("valid", False),
        "errors": cal.get("errors", []),
    }


def _measure_payload(poly: dict[str, Any], cal: dict[str, Any] | None) -> dict[str, Any]:
    pts = [[p["x"], p["y"]] for p in poly["points"]]
    out = {
        "id": poly["id"],
        "name": poly["name"],
        "points": poly["points"],
        "valid": False,
        "area": None,
        "perimeter": None,
        "world_points": None,
        "errors": ["尚未完成透视校准"],
    }
    if cal and cal.get("valid") and cal.get("H"):
        r = measure_polygon(cal["H"], pts)
        out.update(
            valid=r.ok,
            area=r.area,
            perimeter=r.perimeter,
            world_points=r.world_points,
            errors=r.errors or [],
        )
    return out


# ---------------------------------------------------------------------------
# Session & image endpoints
# ---------------------------------------------------------------------------

@app.post("/api/sessions")
def create_session() -> dict[str, Any]:
    sid = uuid.uuid4().hex
    s = {
        "id": sid,
        "created_at": time.time(),
        "image": None,
        "calibration": None,
        "polygons": [],
    }
    with _LOCK:
        _SESSIONS[sid] = s
        _save_session(s)
    return {"session_id": sid}


@app.get("/api/sessions/{sid}")
def get_session(sid: str) -> dict[str, Any]:
    with _LOCK:
        s = _get(sid)
        cal = s.get("calibration")
        return {
            "session_id": sid,
            "image": s["image"],
            "calibration": _cal_payload(s),
            "polygons": [_measure_payload(p, cal) for p in s["polygons"]],
        }

@app.post("/api/sessions/{sid}/image")
async def upload_image(sid: str, file: UploadFile) -> dict[str, Any]:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="上传文件为空")
    if len(raw) > 30 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="图像过大（上限 30 MB）")
    try:
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img)  # honor camera orientation
        img.load()
    except Exception:
        raise HTTPException(status_code=400, detail="无法解析图像，请上传 JPG/PNG/WebP 等格式")
    if not (0 < img.width <= 20000 and 0 < img.height <= 20000):
        raise HTTPException(status_code=400, detail="图像尺寸不受支持（上限 20000 像素）")
    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")

    name = f"{sid}_{uuid.uuid4().hex[:8]}.jpg"
    img.save(IMG_DIR / name, "JPEG", quality=92)
    with _LOCK:
        s = _get(sid)
        # Drop stale geometry whenever the photo changes.
        s["image"] = {"filename": name, "width": img.width, "height": img.height}
        s["calibration"] = None
        s["polygons"] = []
        _save_session(s)
    return {"image": s["image"]}


@app.get("/api/images/{name}")
def get_image(name: str) -> FileResponse:
    path = IMG_DIR / name
    if not path.is_file() or "/" in name or ".." in name:
        raise HTTPException(status_code=404, detail="图像不存在")
    return FileResponse(path, media_type="image/jpeg")


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

@app.put("/api/sessions/{sid}/calibration")
def set_calibration(sid: str, body: CalibrationIn) -> dict[str, Any]:
    s = _get(sid)
    if not s.get("image"):
        raise HTTPException(status_code=400, detail="请先上传照片")
    img_w, img_h = s["image"]["width"], s["image"]["height"]
    corners = [[p.x, p.y] for p in body.corners]
    for i, (u, v) in enumerate(corners):
        if not (math.isfinite(u) and math.isfinite(v)) or not (-1e6 <= u <= 1e6) or not (-1e6 <= v <= 1e6):
            raise HTTPException(status_code=400, detail=f"角点 {i + 1} 坐标非法")
        if not (-1 <= u <= img_w + 1 and -1 <= v <= img_h + 1):
            raise HTTPException(status_code=400, detail=f"角点 {i + 1} 超出图像范围")

    res: HomographyResult = solve_homography(corners, body.rect_width, body.rect_height)
    cal = {
        "corners": [{"x": p[0], "y": p[1]} for p in corners],
        "rect_width": body.rect_width,
        "rect_height": body.rect_height,
        "unit": body.unit,
        "H": res.H,
        "valid": res.ok,
        "errors": res.errors or [],
    }
    s["calibration"] = cal
    with _LOCK:
        _save_session(s)
    return {"calibration": _cal_payload(s),
            "polygons": [_measure_payload(p, cal) for p in s["polygons"]]}


# ---------------------------------------------------------------------------
# Polygons
# ---------------------------------------------------------------------------

def _validate_points_in_image(s: dict[str, Any], points: list[Point]) -> None:
    img_w, img_h = s["image"]["width"], s["image"]["height"]
    for i, p in enumerate(points):
        if not (math.isfinite(p.x) and math.isfinite(p.y)):
            raise HTTPException(status_code=400, detail=f"顶点 {i + 1} 坐标非法")
        if not (-1 <= p.x <= img_w + 1 and -1 <= p.y <= img_h + 1):
            raise HTTPException(status_code=400, detail=f"顶点 {i + 1} 超出图像范围")


@app.post("/api/sessions/{sid}/polygons")
def add_polygon(sid: str, body: PolygonIn) -> dict[str, Any]:
    s = _get(sid)
    if not s.get("image"):
        raise HTTPException(status_code=400, detail="请先上传照片")
    if len(s["polygons"]) >= 200:
        raise HTTPException(status_code=400, detail="每个会话最多保留 200 个区域")
    _validate_points_in_image(s, body.points)
    name = (body.name or f"区域 {len(s['polygons']) + 1}").strip()[:80]
    poly = {
        "id": uuid.uuid4().hex,
        "name": name or f"区域 {len(s['polygons']) + 1}",
        "points": [{"x": p.x, "y": p.y} for p in body.points],
    }
    s["polygons"].append(poly)
    with _LOCK:
        _save_session(s)
    return {"polygon": _measure_payload(poly, s.get("calibration"))}


@app.put("/api/sessions/{sid}/polygons/{pid}")
def update_polygon(sid: str, pid: str, body: PolygonPatch) -> dict[str, Any]:
    s = _get(sid)
    _validate_points_in_image(s, body.points)
    for poly in s["polygons"]:
        if poly["id"] == pid:
            poly["points"] = [{"x": p.x, "y": p.y} for p in body.points]
            if body.name:
                poly["name"] = body.name
            with _LOCK:
                _save_session(s)
            return {"polygon": _measure_payload(poly, s.get("calibration"))}
    raise HTTPException(status_code=404, detail="多边形不存在")


@app.delete("/api/sessions/{sid}/polygons/{pid}")
def delete_polygon(sid: str, pid: str) -> dict[str, Any]:
    s = _get(sid)
    before = len(s["polygons"])
    s["polygons"] = [p for p in s["polygons"] if p["id"] != pid]
    if len(s["polygons"]) == before:
        raise HTTPException(status_code=404, detail="多边形不存在")
    with _LOCK:
        _save_session(s)
    return {"ok": True}


@app.delete("/api/sessions/{sid}/polygons")
def clear_polygons(sid: str) -> dict[str, Any]:
    s = _get(sid)
    s["polygons"] = []
    with _LOCK:
        _save_session(s)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

@app.get("/api/sessions/{sid}/export")
def export_session(sid: str) -> JSONResponse:
    s = _get(sid)
    cal = s.get("calibration")
    H = np.asarray(cal["H"]) if cal and cal.get("H") else None
    polygons = []
    all_ok = bool(s.get("image")) and bool(cal and cal.get("valid"))
    for idx, poly in enumerate(s["polygons"], start=1):
        m = _measure_payload(poly, cal)
        all_ok = all_ok and m["valid"]
        polygons.append({
            "id": poly["id"],
            "name": poly["name"],
            "index": idx,
            "image_points": [{"u": p["x"], "v": p["y"]} for p in poly["points"]],
            "world_points": (
                [{"x": round(w[0], 9), "y": round(w[1], 9)} for w in m["world_points"]]
                if m["world_points"] else None
            ),
            "area": (round(m["area"], 9) if m["area"] is not None else None),
            "perimeter": (round(m["perimeter"], 9) if m["perimeter"] is not None else None),
            "valid": m["valid"],
            "errors": m["errors"],
        })
    payload = {
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "schema": "photo-ruler/1.0",
        "session_id": sid,
        "image": s.get("image"),
        "calibration": (
            {
                "corners": [{"u": c["x"], "v": c["y"]} for c in cal["corners"]],
                "rect_width": cal["rect_width"],
                "rect_height": cal["rect_height"],
                "unit": cal["unit"],
                "homography_image_to_world": cal.get("H"),
                "valid": cal.get("valid", False),
                "errors": cal.get("errors", []),
            }
            if cal else None
        ),
        "all_measurements_valid": all_ok,
        "polygons": polygons,
    }
    return JSONResponse(
        payload,
        headers={"Content-Disposition": f'attachment; filename="measurements-{sid[:8]}.json"'},
    )


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# Static frontend (mounted last so /api routes win).
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))
