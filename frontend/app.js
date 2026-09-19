/* Planar photo ruler calibration — frontend.
 *
 * Coordinate systems:
 *   image  : raw photo pixels (u, v), the single source of truth for geometry
 *   world  : real-plane coords (x, y) after applying the 3x3 homography H
 *   screen : CSS pixels of the canvas element (devicePixelRatio handled here)
 *
 * Every stored point (corners, polygon vertices) is in image coordinates, so
 * zoom / pan / resize never change measurement geometry.
 */
"use strict";

/* ============================================================ geometry === */
const GEO = {
  DUP_TOL: 1e-7,
  COLINEAR_TOL: 1e-9,
  MIN_PIVOT: 1e-9,
  RESIDUAL_TOL: 1e-7,
  H_DET_TOL: 1e-8,
  DENOM_TOL: 1e-9,
  CONTACT_TOL: 1e-8,
  ZERO_AREA_TOL: 1e-10,
};

const sub = (a, b) => ({ x: a.x - b.x, y: a.y - b.y });
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

function findDuplicate(pts) {
  let diag = 0;
  for (let i = 0; i < pts.length; i++)
    for (let j = i + 1; j < pts.length; j++)
      diag = Math.max(diag, dist(pts[i], pts[j]));
  const ref = Math.max(diag, 1);
  for (let i = 0; i < pts.length; i++)
    for (let j = i + 1; j < pts.length; j++)
      if (dist(pts[i], pts[j]) < GEO.DUP_TOL * ref) return [i, j];
  return null;
}

function convexOrderedQuad(pts) {
  const dup = findDuplicate(pts);
  if (dup) return { ok: false, error: `角点 ${dup[0] + 1} 与角点 ${dup[1] + 1} 重合：四点必须彼此分离` };
  const crosses = [];
  let edgeScale = 0;
  for (let i = 0; i < 4; i++) {
    const a = pts[i], b = pts[(i + 1) % 4], c = pts[(i + 2) % 4];
    crosses.push((b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x));
    edgeScale = Math.max(edgeScale, dist(a, b));
  }
  edgeScale *= edgeScale;
  if (crosses.some((cr) => Math.abs(cr) < GEO.COLINEAR_TOL * Math.max(edgeScale, 1)))
    return { ok: false, error: "有三个或更多角点近似共线，四边形已退化" };
  if (!(crosses.every((c) => c > 0) || crosses.every((c) => c < 0)))
    return { ok: false, error: "四个角点没有构成凸四边形（顺序错乱或自交），请按 左上→右上→右下→左下 依次点击" };
  return { ok: true };
}

/** 8-point DLT homography (mirror of backend solve_homography). */
function solveHomography(imgPts, W, H) {
  const errors = [];
  if (!(isFinite(W) && isFinite(H) && W > 0 && H > 0))
    errors.push("已知矩形的宽和高必须为正数");
  const conv = convexOrderedQuad(imgPts);
  if (!conv.ok) errors.push(conv.error);
  if (errors.length) return { ok: false, errors };

  const worldPts = [
    { x: 0, y: 0 }, { x: W, y: 0 }, { x: W, y: H }, { x: 0, y: H },
  ];

  // Normalize both point sets (isotropic scaling).
  const norm = (pts) => {
    const cx = pts.reduce((s, p) => s + p.x, 0) / pts.length;
    const cy = pts.reduce((s, p) => s + p.y, 0) / pts.length;
    let d = 0;
    for (const p of pts) d += Math.hypot(p.x - cx, p.y - cy);
    d /= pts.length;
    if (d < 1e-12) return { T: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], s: 1 };
    const s = Math.SQRT2 / d;
    return { T: [[s, 0, -s * cx], [0, s, -s * cy], [0, 0, 1]], s };
  };
  const Ts = norm(imgPts), Td = norm(worldPts);
  const apply = (T, p) => ({ x: T[0][0] * p.x + T[0][1] * p.y + T[0][2],
                              y: T[1][0] * p.x + T[1][1] * p.y + T[1][2] });
  const ns = imgPts.map((p) => apply(Ts.T, p));
  const nd = worldPts.map((p) => apply(Td.T, p));

  const Arows = [];
  for (let i = 0; i < 4; i++) {
    const { x, y } = ns[i], { x: X, y: Y } = nd[i];
    Arows.push([-x, -y, -1, 0, 0, 0, X * x, X * y, X]);
    Arows.push([0, 0, 0, -x, -y, -1, Y * x, Y * y, Y]);
  }
  // Smallest singular vector of A (8x9): rank-revealing Gauss-Jordan on AᵀA
  // with complete pivoting.  The last pivoted column is the free (null) one.
  const AtA = Array.from({ length: 9 }, () => new Array(9).fill(0));
  for (const row of Arows)
    for (let i = 0; i < 9; i++)
      for (let j = 0; j < 9; j++) AtA[i][j] += row[i] * row[j];

  const M = AtA.map((r) => r.slice());
  const perm = [0, 1, 2, 3, 4, 5, 6, 7, 8];
  let pivotMag = 0;
  for (let k = 0; k < 8; k++) {
    let pr = k, pc = k, best = 0;
    for (let i = k; i < 9; i++)
      for (let j = k; j < 9; j++)
        if (Math.abs(M[i][j]) > best) { best = Math.abs(M[i][j]); pr = i; pc = j; }
    if (k === 0) pivotMag = best;
    // A genuine rank drop here (two near-null directions) means the four
    // point pairs do not determine a unique homography.
    if (best < GEO.MIN_PIVOT * (pivotMag || 1))
      return { ok: false, errors: ["透视变换不可逆：标定四点共线或构成退化四边形，无法从平面照片恢复唯一映射"] };
    if (pr !== k) [M[k], M[pr]] = [M[pr], M[k]];
    if (pc !== k) {
      M.forEach((row) => { const t = row[k]; row[k] = row[pc]; row[pc] = t; });
      [perm[k], perm[pc]] = [perm[pc], perm[k]];
    }
    // Eliminate column k in every other row, then normalize pivot row.
    for (let i = 0; i < 9; i++) {
      if (i === k || M[i][k] === 0) continue;
      const f = M[i][k] / M[k][k];
      M[i][k] = 0;
      for (let j = k + 1; j < 9; j++) M[i][j] -= f * M[k][j];
    }
    const pv = M[k][k];
    for (let j = k + 1; j < 9; j++) M[k][j] /= pv;
    M[k][k] = 1;
  }

  const hn = new Array(9);
  for (let k = 0; k < 8; k++) hn[perm[k]] = -M[k][8];
  hn[perm[8]] = 1;

  // Null-vector quality: |A h| / (|A| |h|) must be tiny.  This residual — not
  // the null eigen value (which is supposed to be ~0) — decides singularity.
  let maxA = 0;
  for (const row of Arows) maxA = Math.max(maxA, ...row.map(Math.abs));
  let maxRes = 0, normH = 0;
  for (const row of Arows) {
    let s = 0;
    for (let i = 0; i < 9; i++) s += row[i] * hn[i];
    maxRes = Math.max(maxRes, Math.abs(s));
  }
  for (const v of hn) normH += v * v;
  if (maxRes > GEO.MIN_PIVOT * maxA * Math.sqrt(normH))
    return { ok: false, errors: ["透视变换不可逆：标定四点共线或构成退化四边形，无法从平面照片恢复唯一映射"] };
  const h3x3 = [
    [hn[0], hn[1], hn[2]],
    [hn[3], hn[4], hn[5]],
    [hn[6], hn[7], hn[8]],
  ];
  // Denormalize: H = Td⁻¹ h Ts ; multiply by hand (3x3).
  const inv3 = (T) => {
    const [[a, b, tx], [, d, ty]] = T;
    return [[1 / a, 0, -tx / a], [0, 1 / d, -ty / d], [0, 0, 1]];
  };
  const mul3 = (A, B) => {
    const R = Array.from({ length: 3 }, () => [0, 0, 0]);
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++)
        for (let k = 0; k < 3; k++) R[i][j] += A[i][k] * B[k][j];
    return R;
  };
  let Hmat = mul3(inv3(Td.T), mul3(h3x3, Ts.T));
  const w00 = Hmat[2][2] || 1;
  Hmat = Hmat.map((row) => row.map((v) => v / w00));

  // DLT residual against the actual correspondences.
  let res = 0;
  for (let i = 0; i < 4; i++) {
    const p = imgPts[i], q = worldPts[i];
    const w = Hmat[2][0] * p.x + Hmat[2][1] * p.y + Hmat[2][2];
    const x = (Hmat[0][0] * p.x + Hmat[0][1] * p.y + Hmat[0][2]) / w;
    const y = (Hmat[1][0] * p.x + Hmat[1][1] * p.y + Hmat[1][2]) / w;
    res = Math.max(res, Math.hypot(x - q.x, y - q.y));
  }
  if (res > GEO.RESIDUAL_TOL * Math.max(W, H))
    return { ok: false, errors: [`透视校准残差过大（${res.toPrecision(3)} 单位），角点与已知矩形对应关系不一致，请检查点击顺序`] };

  // Determinant guard.
  const det =
    Hmat[0][0] * (Hmat[1][1] * Hmat[2][2] - Hmat[1][2] * Hmat[2][1]) -
    Hmat[0][1] * (Hmat[1][0] * Hmat[2][2] - Hmat[1][2] * Hmat[2][0]) +
    Hmat[0][2] * (Hmat[1][0] * Hmat[2][1] - Hmat[1][1] * Hmat[2][0]);
  let hNorm = 0;
  Hmat.forEach((r) => r.forEach((v) => (hNorm += v * v)));
  if (Math.abs(det) < GEO.H_DET_TOL * Math.max(Math.pow(hNorm, 1.5), 1e-30))
    return { ok: false, errors: ["透视矩阵行列式接近 0，变换不可逆：四点几乎共线或矩形在照片中退化"] };

  const row3Scale = Math.max(Math.hypot(Hmat[2][0], Hmat[2][1]), Math.abs(Hmat[2][2]));
  for (let i = 0; i < 4; i++) {
    const p = imgPts[i];
    const w = Math.abs(Hmat[2][0] * p.x + Hmat[2][1] * p.y + Hmat[2][2]);
    if (w < GEO.DENOM_TOL * row3Scale)
      return { ok: false, errors: [`角点 ${i + 1} 映射时分母接近 0（落在无穷远线上），此视角下透视不可求逆`] };
  }
  return { ok: true, H: Hmat };
}

function applyH(H, p) {
  const w = H[2][0] * p.x + H[2][1] * p.y + H[2][2];
  return {
    x: (H[0][0] * p.x + H[0][1] * p.y + H[0][2]) / w,
    y: (H[1][0] * p.x + H[1][1] * p.y + H[1][2]) / w,
    w,
  };
}

function segmentsCross(p, q, a, b) {
  const orient = (r, s, t) => (s.x - r.x) * (t.y - r.y) - (s.y - r.y) * (t.x - r.x);
  const o1 = orient(p, q, a), o2 = orient(p, q, b);
  const o3 = orient(a, b, p), o4 = orient(a, b, q);
  const tol = 1e-10 * Math.max(dist(p, q), dist(a, b), 1);
  if ([o1, o2, o3, o4].every((o) => Math.abs(o) < tol)) return false; // collinear: rare; endpoints handled by dup check
  return ((o1 > tol && o2 < -tol) || (o1 < -tol && o2 > tol)) &&
         ((o3 > tol && o4 < -tol) || (o3 < -tol && o4 > tol));
}

function polygonSelfIntersects(pts) {
  if (pts.length < 3) return { hit: false };
  const dup = findDuplicate(pts);
  if (dup) return { hit: true, error: `顶点 ${dup[0] + 1} 与顶点 ${dup[1] + 1} 重合` };
  const n = pts.length;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (j === i || (j + 1) % n === i || (i + 1) % n === j) continue;
      if (segmentsCross(pts[i], pts[(i + 1) % n], pts[j], pts[(j + 1) % n]))
        return { hit: true, error: `边 ${i + 1} 与边 ${j + 1} 相交` };
    }
  }
  return { hit: false };
}

function measurePolygon(H, imgPts) {
  if (imgPts.length < 3) return { valid: false, errors: ["多边形至少需要 3 个顶点"] };
  const row3Scale = Math.max(Math.hypot(H[2][0], H[2][1]), Math.abs(H[2][2]));
  const world = [];
  for (let i = 0; i < imgPts.length; i++) {
    const { x, y, w } = applyH(H, imgPts[i]);
    if (Math.abs(w) < GEO.DENOM_TOL * row3Scale)
      return { valid: false, errors: [`多边形顶点 ${i + 1} 的透视分母接近 0（位于无穷远线上），无法计算校正后坐标`] };
    world.push({ x, y });
  }
  const si = polygonSelfIntersects(world);
  if (si.hit) return { valid: false, errors: [`多边形自交（${si.error}），自交多边形没有唯一的面积，请调整顶点`] };
  let area2 = 0, perim = 0;
  for (let i = 0; i < world.length; i++) {
    const a = world[i], b = world[(i + 1) % world.length];
    area2 += a.x * b.y - b.x * a.y;
    perim += Math.hypot(b.x - a.x, b.y - a.y);
  }
  return { valid: true, area: Math.abs(area2) / 2, perimeter: perim, worldPoints: world };
}

/* ----------------------------------------------------- regions with holes */
function ringDiameter(pts) {
  let d = 0;
  for (let i = 0; i < pts.length; i++)
    for (let j = i + 1; j < pts.length; j++) d = Math.max(d, dist(pts[i], pts[j]));
  return d;
}

/** Closed-segment touch: proper crossings, endpoint tangency and collinear
 *  overlap all count (rings in a region may not share any point). */
function closedSegmentsTouch(p, q, a, b) {
  const orient = (r, s, t) => (s.x - r.x) * (t.y - r.y) - (s.y - r.y) * (t.x - r.x);
  const o1 = orient(p, q, a), o2 = orient(p, q, b);
  const o3 = orient(a, b, p), o4 = orient(a, b, q);
  const tol = GEO.CONTACT_TOL * Math.max(dist(p, q), dist(a, b), 1);
  if ([o1, o2, o3, o4].every((o) => Math.abs(o) < tol)) {
    const onSeg = (x, r, s) =>
      x.x >= Math.min(r.x, s.x) - tol && x.x <= Math.max(r.x, s.x) + tol &&
      x.y >= Math.min(r.y, s.y) - tol && x.y <= Math.max(r.y, s.y) + tol;
    return [p, q, a, b].some((z) => onSeg(z, p, q) && onSeg(z, a, b));
  }
  const span = (u, v) =>
    ((u > tol && v < -tol) || (u < -tol && v > tol) || (Math.abs(u) <= tol) || (Math.abs(v) <= tol));
  return span(o1, o2) && span(o3, o4);
}

function ringsTouch(a, b) {
  for (let i = 0; i < a.length; i++) {
    const p = a[i], q = a[(i + 1) % a.length];
    for (let j = 0; j < b.length; j++) {
      if (closedSegmentsTouch(p, q, b[j], b[(j + 1) % b.length])) return [i, j];
    }
  }
  return null;
}

function pointOnRingBoundary(pt, ring, tol) {
  for (let i = 0; i < ring.length; i++) {
    const a = ring[i], b = ring[(i + 1) % ring.length];
    const cross = (b.x - a.x) * (pt.y - a.y) - (b.y - a.y) * (pt.x - a.x);
    if (Math.abs(cross) > tol * Math.max(dist(a, b), 1)) continue;
    if (pt.x >= Math.min(a.x, b.x) - tol && pt.x <= Math.max(a.x, b.x) + tol &&
        pt.y >= Math.min(a.y, b.y) - tol && pt.y <= Math.max(a.y, b.y) + tol) return true;
  }
  return false;
}

function pointInRingStrict(pt, ring, tol) {
  if (pointOnRingBoundary(pt, ring, tol)) return "boundary";
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const a = ring[i], b = ring[j];
    if ((a.y > pt.y) !== (b.y > pt.y) &&
        pt.x < ((b.x - a.x) * (pt.y - a.y)) / (b.y - a.y) + a.x)
      inside = !inside;
  }
  return inside ? "inside" : "outside";
}

function ringSignedArea2(pts) {
  let a2 = 0;
  for (let i = 0; i < pts.length; i++) {
    const a = pts[i], b = pts[(i + 1) % pts.length];
    a2 += a.x * b.y - b.x * a.y;
  }
  return a2;
}

function ringPerimeter(pts) {
  let p = 0;
  for (let i = 0; i < pts.length; i++) p += dist(pts[i], pts[(i + 1) % pts.length]);
  return p;
}

/** Mirror of backend measure_region: all ring relations use world coords. */
function measureRegion(H, outerImg, holesImg) {
  holesImg = holesImg || [];
  if (outerImg.length < 3) return { valid: false, errors: ["外环至少需要 3 个顶点"] };
  const row3Scale = Math.max(Math.hypot(H[2][0], H[2][1]), Math.abs(H[2][2]));
  const errors = [];

  const toWorld = (pts, label) => {
    const out = [];
    for (let i = 0; i < pts.length; i++) {
      const { x, y, w } = applyH(H, pts[i]);
      if (Math.abs(w) < GEO.DENOM_TOL * row3Scale) {
        errors.push(`${label} 顶点 ${i + 1} 的透视分母接近 0（位于无穷远线上），无法计算校正后坐标`);
        return null;
      }
      out.push({ x, y });
    }
    return out;
  };

  const ringErrors = (pts, label) => {
    const out = [];
    if (pts.length < 3) { out.push(`${label} 至少需要 3 个顶点`); return out; }
    const si = polygonSelfIntersects(pts);
    if (si.hit) out.push(`${label} ${si.error}：自交或顶点重合的多边形没有唯一的面积，请调整顶点`);
    if (Math.abs(ringSignedArea2(pts)) / 2 <
        GEO.ZERO_AREA_TOL * Math.max(ringDiameter(pts) ** 2, 1))
      out.push(`${label} 面积为 0（顶点近似共线或退化），请调整顶点`);
    return out;
  };

  const worldOuter = toWorld(outerImg, "外环");
  if (worldOuter === null) return { valid: false, errors };
  errors.push(...ringErrors(worldOuter, "外环"));

  const worldHoles = [];
  for (let hi = 0; hi < holesImg.length; hi++) {
    const wh = toWorld(holesImg[hi], `孔洞 ${hi + 1}`);
    if (wh === null) continue;
    worldHoles.push(wh);
    errors.push(...ringErrors(wh, `孔洞 ${hi + 1}`));
  }
  if (errors.length) return { valid: false, errors };

  const tol = GEO.CONTACT_TOL * Math.max(ringDiameter(worldOuter), 1);

  // outer ring vs each hole
  for (let hi = 0; hi < worldHoles.length; hi++) {
    const wh = worldHoles[hi];
    const tag = `孔洞 ${hi + 1}`;
    const t = ringsTouch(worldOuter, wh);
    if (t) {
      errors.push(`${tag} 与外环相交或相切（孔洞边 ${t[1] + 1} 碰到外环边 ${t[0] + 1}），孔洞必须完全位于外环内部且互不接触`);
      continue;
    }
    let bad = -1;
    for (let vi = 0; vi < wh.length; vi++) {
      if (pointInRingStrict(wh[vi], worldOuter, tol) !== "inside") { bad = vi; break; }
    }
    if (bad >= 0) errors.push(`${tag} 顶点 ${bad + 1} 位于外环之外，孔洞必须完全位于外环内部`);
  }

  // hole vs hole
  let pairErrors = 0;
  for (let i = 0; i < worldHoles.length && pairErrors < 6; i++) {
    for (let j = i + 1; j < worldHoles.length && pairErrors < 6; j++) {
      const t = ringsTouch(worldHoles[i], worldHoles[j]);
      if (t) {
        errors.push(`孔洞 ${i + 1} 与孔洞 ${j + 1} 相交或相切（孔洞 ${i + 1} 边 ${t[0] + 1} 与孔洞 ${j + 1} 边 ${t[1] + 1} 接触），孔洞之间必须完全分离`);
        pairErrors++;
        continue;
      }
      let bad = null;
      for (let vi = 0; vi < worldHoles[j].length; vi++)
        if (pointInRingStrict(worldHoles[j][vi], worldHoles[i], tol) !== "outside") { bad = [j, vi]; break; }
      if (!bad)
        for (let vi = 0; vi < worldHoles[i].length; vi++)
          if (pointInRingStrict(worldHoles[i][vi], worldHoles[j], tol) !== "outside") { bad = [i, vi]; break; }
      if (bad) {
        const other = bad[0] === j ? i + 1 : j + 1;
        errors.push(`孔洞 ${bad[0] + 1} 与孔洞 ${other} 未完全分离（孔洞 ${bad[0] + 1} 顶点 ${bad[1] + 1} 落在另一孔洞内或边界上），孔洞不允许相互包含`);
        pairErrors++;
      }
    }
  }
  if (pairErrors >= 6) errors.push("……（还有更多孔洞冲突，未全部列出）");
  if (errors.length) return { valid: false, errors };

  const outerArea = Math.abs(ringSignedArea2(worldOuter)) / 2;
  const holeAreas = worldHoles.map((wh) => Math.abs(ringSignedArea2(wh)) / 2);
  const holePerimeters = worldHoles.map(ringPerimeter);
  return {
    valid: true,
    area: outerArea - holeAreas.reduce((s, a) => s + a, 0),
    perimeter: ringPerimeter(worldOuter) + holePerimeters.reduce((s, a) => s + a, 0),
    outerArea,
    holeAreas,
    holePerimeters,
    worldPoints: worldOuter,
    worldHoles,
  };
}

/* ============================================================ state ====== */
const els = {};
document.querySelectorAll("[id]").forEach((e) => (els[e.id] = e));

const UNITS = ["mm", "cm", "m", "in", "px"];
const UNIT_TO_MM = { mm: 1, cm: 10, m: 1000, in: 25.4, px: null };

const state = {
  sessionId: null,
  image: null,            // {filename, width, height}
  imgEl: null,
  corners: [],            // [{x,y}] image coords
  rectW: null,
  rectH: null,
  unit: "mm",
  H: null,
  calibValid: false,
  calibErrors: [],
  polygons: [],           // {id, name, points, holes:[{id,points,...}], area, outerArea, perimeter, valid, errors, worldPoints}
  selected: null,         // polygon id
  draft: [],              // in-progress ring points (image coords)
  draftTarget: null,      // null = drawing a new outer ring; polyId = drawing a hole into it
  mode: "pan",            // calib | draw | edit | pan
  displayUnit: "mm",
  view: { scale: 1, x: 0, y: 0 }, // screen = image*scale + (x,y)
  hoverImg: null,
  dragging: null,         // drag descriptor
  spaceDown: false,
};

const COLORS = ["#4da3ff", "#3ddc97", "#ffc24d", "#ff8f6b", "#c792ea",
  "#7cddff", "#a6e22e", "#ff6b6b", "#f4a261", "#8de0c2"];
const polyColor = (i) => COLORS[i % COLORS.length];

const $canvas = els.canvas;
const ctx = $canvas.getContext("2d");

/* ============================================================ API ======== */
async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: opts.body ? { "Content-Type": "application/json" } : {},
    ...opts,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch { /* ignore */ }
    throw new Error(detail);
  }
  return res.json();
}

function toast(msg, isError = false) {
  const t = els.toast || (() => {
    const d = document.createElement("div");
    d.className = "toast";
    d.id = "toast";
    document.body.appendChild(d);
    els.toast = d;
    return d;
  })();
  t.textContent = msg;
  t.classList.toggle("error", isError);
  t.classList.add("show");
  clearTimeout(t._timer);
  t._timer = setTimeout(() => t.classList.remove("show"), 2600);
}

/* ====================================================== view transforms == */
function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const r = $canvas.parentElement.getBoundingClientRect();
  $canvas.width = Math.max(1, Math.round(r.width * dpr));
  $canvas.height = Math.max(1, Math.round(r.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  render();
}
const screenSize = () => {
  const dpr = window.devicePixelRatio || 1;
  return { w: $canvas.width / dpr, h: $canvas.height / dpr };
};
const imgToScreen = (p) => ({ x: p.x * state.view.scale + state.view.x, y: p.y * state.view.scale + state.view.y });
const screenToImg = (p) => ({ x: (p.x - state.view.x) / state.view.scale, y: (p.y - state.view.y) / state.view.scale });

function clampImagePoint(p) {
  if (!state.image) return p;
  return {
    x: Math.min(Math.max(p.x, 0), state.image.width),
    y: Math.min(Math.max(p.y, 0), state.image.height),
  };
}

function fitView() {
  if (!state.image) return;
  const { w, h } = screenSize();
  const pad = 36;
  const s = Math.min((w - pad * 2) / state.image.width, (h - pad * 2) / state.image.height);
  state.view.scale = Math.min(s, 8);
  state.view.x = (w - state.image.width * state.view.scale) / 2;
  state.view.y = (h - state.image.height * state.view.scale) / 2;
  render();
}

function zoomAt(screenPt, factor) {
  if (!state.image) return;
  const before = screenToImg(screenPt);
  state.view.scale = Math.min(40, Math.max(0.02, state.view.scale * factor));
  state.view.x = screenPt.x - before.x * state.view.scale;
  state.view.y = screenPt.y - before.y * state.view.scale;
  updateZoomLabel();
  render();
}

/* ============================================================ rendering == */
const HANDLE_PX = 6;
const cornerColor = (i) => ["#ffc24d", "#4da3ff", "#3ddc97", "#ff8f6b"][i % 4];
const CORNER_NAMES = ["左上", "右上", "右下", "左下"];

function drawHandle(sp, color, filled = true, radius = HANDLE_PX) {
  ctx.beginPath();
  ctx.arc(sp.x, sp.y, radius, 0, Math.PI * 2);
  ctx.fillStyle = filled ? color : "#0f141b";
  ctx.fill();
  ctx.lineWidth = 2;
  ctx.strokeStyle = color;
  ctx.stroke();
}

function drawDashedLine(a, b, color) {
  ctx.save();
  ctx.setLineDash([6, 5]);
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.stroke();
  ctx.restore();
}

function render() {
  const { w, h } = screenSize();
  ctx.clearRect(0, 0, w, h);
  updateZoomLabel();
  if (!state.imgEl || !state.image) return;

  // image
  const tl = imgToScreen({ x: 0, y: 0 });
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(
    state.imgEl,
    tl.x, tl.y,
    state.image.width * state.view.scale,
    state.image.height * state.view.scale
  );
  // image border
  ctx.strokeStyle = "rgba(255,255,255,.18)";
  ctx.lineWidth = 1;
  ctx.strokeRect(tl.x, tl.y, state.image.width * state.view.scale, state.image.height * state.view.scale);

  // calibration quad
  if (state.corners.length > 0) {
    const scr = state.corners.map(imgToScreen);
    if (state.corners.length >= 2) {
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(scr[0].x, scr[0].y);
      for (let i = 1; i < scr.length; i++) ctx.lineTo(scr[i].x, scr[i].y);
      if (state.corners.length === 4) ctx.closePath();
      ctx.strokeStyle = state.calibValid ? "rgba(61,220,151,.95)" : "rgba(255,194,77,.95)";
      ctx.lineWidth = 2;
      ctx.stroke();
      if (state.corners.length === 4) {
        ctx.fillStyle = state.calibValid ? "rgba(61,220,151,.10)" : "rgba(255,194,77,.08)";
        ctx.fill();
      }
      ctx.restore();
    }
    state.corners.forEach((p, i) => {
      const sp = scr[i];
      drawHandle(sp, cornerColor(i), true, 8);
      ctx.font = "700 10px ui-monospace, monospace";
      ctx.fillStyle = "#0f141b";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(String(i + 1), sp.x, sp.y + 0.5);
    });
    // rubber line to next corner
    if (state.mode === "calib" && state.corners.length < 4 && state.hoverImg) {
      const inside = pointInImage(state.hoverImg);
      if (inside) drawDashedLine(scr[scr.length - 1], imgToScreen(state.hoverImg), "rgba(255,194,77,.7)");
    }
  }

  // polygons (outer ring + holes)
  state.polygons.forEach((poly, i) => {
    const color = polyColor(i);
    const selected = poly.id === state.selected;
    const scr = poly.points.map(imgToScreen);
    ctx.beginPath();
    ctx.moveTo(scr[0].x, scr[0].y);
    for (let k = 1; k < scr.length; k++) ctx.lineTo(scr[k].x, scr[k].y);
    ctx.closePath();
    // holes join the same path; "evenodd" fill leaves them cut out
    (poly.holes || []).forEach((hole) => {
      const hs = hole.points.map(imgToScreen);
      if (hs.length < 2) return;
      ctx.moveTo(hs[0].x, hs[0].y);
      for (let k = 1; k < hs.length; k++) ctx.lineTo(hs[k].x, hs[k].y);
      ctx.closePath();
    });
    ctx.fillStyle = poly.valid ? color + "26" : "rgba(255,107,107,.13)";
    ctx.fill("evenodd");
    ctx.lineWidth = selected ? 2.6 : 1.8;
    ctx.strokeStyle = poly.valid ? color : "#ff6b6b";
    ctx.stroke();

    // hole boundaries: dashed, red while the region is invalid
    (poly.holes || []).forEach((hole) => {
      const hs = hole.points.map(imgToScreen);
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(hs[0].x, hs[0].y);
      for (let k = 1; k < hs.length; k++) ctx.lineTo(hs[k].x, hs[k].y);
      ctx.closePath();
      ctx.setLineDash([7, 5]);
      ctx.lineWidth = selected ? 2.2 : 1.5;
      ctx.strokeStyle = poly.valid ? color : "#ff6b6b";
      ctx.stroke();
      ctx.restore();
    });

    if (state.mode === "edit" || selected) {
      poly.points.forEach((p, k) => {
        const active = k === state.dragging?.vertex && state.dragging?.polyId === poly.id &&
                       state.dragging?.ring === "outer";
        drawHandle(imgToScreen(p), poly.valid ? color : "#ff6b6b", active);
      });
      (poly.holes || []).forEach((hole, hi) => {
        hole.points.forEach((p, k) => {
          const active = state.dragging?.polyId === poly.id && state.dragging?.ring === "hole" &&
                         state.dragging?.holeIndex === hi && k === state.dragging?.vertex;
          drawHandle(imgToScreen(p), poly.valid ? color : "#ff6b6b", active, 5);
        });
      });
    }
    // label
    const cx = scr.reduce((s, p) => s + p.x, 0) / scr.length;
    const cy = scr.reduce((s, p) => s + p.y, 0) / scr.length;
    ctx.font = "600 11px -apple-system, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = "rgba(0,0,0,.65)";
    const label = poly.name;
    const tw = ctx.measureText(label).width;
    ctx.fillRect(cx - tw / 2 - 6, cy - 9, tw + 12, 18);
    ctx.fillStyle = color;
    ctx.fillText(label, cx, cy + .5);
  });

  // draft ring (white = new outer ring, amber = new hole)
  if (state.draft.length > 0) {
    const isHole = !!state.draftTarget;
    const draftColor = isHole ? "#ffc24d" : "rgba(255,255,255,.85)";
    const scr = state.draft.map(imgToScreen);
    ctx.beginPath();
    ctx.moveTo(scr[0].x, scr[0].y);
    for (let k = 1; k < scr.length; k++) ctx.lineTo(scr[k].x, scr[k].y);
    if (state.hoverImg && pointInImage(state.hoverImg))
      ctx.lineTo(imgToScreen(state.hoverImg).x, imgToScreen(state.hoverImg).y);
    ctx.strokeStyle = draftColor;
    ctx.lineWidth = 1.6;
    ctx.save();
    ctx.setLineDash([5, 4]);
    ctx.stroke();
    ctx.restore();
    // close hint line back to first point
    if (state.draft.length >= 2 && state.hoverImg && pointInImage(state.hoverImg))
      drawDashedLine(imgToScreen(state.hoverImg), scr[0],
                     isHole ? "rgba(255,194,77,.5)" : "rgba(255,255,255,.45)");
    state.draft.forEach((p, k) => {
      const sp = imgToScreen(p);
      // highlight first vertex when close enough to close
      if (k === 0 && state.hoverImg && dist(state.hoverImg, p) * state.view.scale < 12 && state.draft.length >= 3)
        drawHandle(sp, "#3ddc97", true, 8);
      else drawHandle(sp, isHole ? "#ffc24d" : "#ffffff");
    });
  }
}

const pointInImage = (p) =>
  state.image && p.x >= 0 && p.y >= 0 && p.x <= state.image.width && p.y <= state.image.height;

/* ====================================================== sidebar rendering */
function fmtNum(v) {
  if (v == null || !isFinite(v)) return "—";
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e7 || a < 1e-4) return v.toExponential(3);
  return String(Math.round(v * 1e4) / 1e4);
}

function convertValue(v, fromUnit, toUnit, isArea) {
  if (v == null) return null;
  if (fromUnit === toUnit) return v;
  const f = UNIT_TO_MM[fromUnit], t = UNIT_TO_MM[toUnit];
  if (f == null || t == null) return null; // px is only comparable to itself
  const linear = f / t;
  return v * (isArea ? linear * linear : linear);
}

function renderCorners() {
  const ol = els.cornerList;
  ol.innerHTML = "";
  for (let i = 0; i < 4; i++) {
    const li = document.createElement("li");
    const p = state.corners[i];
    if (p) li.classList.add("placed");
    if (state.mode === "calib" && i === state.corners.length) li.classList.add("active");
    li.innerHTML = `<span class="corner-dot" style="background:${cornerColor(i)}"></span>` +
      `<span>${CORNER_NAMES[i]}</span>` +
      `<span style="margin-left:auto">${p ? `${Math.round(p.x)},${Math.round(p.y)}` : "—"}</span>`;
    ol.appendChild(li);
  }
  const st = els.calibStatus;
  if (!state.image) {
    st.innerHTML = `<span class="dot dot-idle"></span><span>先上传照片，再依次点击矩形四角</span>`;
  } else if (state.corners.length < 4) {
    st.innerHTML = `<span class="dot dot-idle"></span><span>已标记 ${state.corners.length}/4 —— 下一点：${CORNER_NAMES[state.corners.length] || ""}</span>`;
  } else if (state.calibValid) {
    st.innerHTML = `<span class="dot dot-ok"></span><span>标定有效：${fmtNum(state.rectW)} × ${fmtNum(state.rectH)} ${state.unit} 的矩形已对齐，可绘制测量区域</span>`;
  } else {
    st.innerHTML = `<span class="dot dot-err"></span><span>${state.calibErrors[0] || "标定无效"}</span>`;
  }
}

function renderPolyList() {
  const list = els.polyList;
  list.innerHTML = "";
  if (state.polygons.length === 0) {
    list.innerHTML = `<p class="empty-text">完成标定并绘制闭合多边形后，这里实时显示校正后的面积与周长。</p>`;
    return;
  }
  state.polygons.forEach((poly, i) => {
    const color = polyColor(i);
    const card = document.createElement("div");
    card.className = "poly-card" + (poly.id === state.selected ? " selected" : "") + (poly.valid ? "" : " invalid");
    const areaConv = convertValue(poly.area, state.unit, state.displayUnit, true);
    const perConv = convertValue(poly.perimeter, state.unit, state.displayUnit, false);
    const areaUnit = state.displayUnit === "px" ? "px²" : `${state.displayUnit}²`;
    const nHoles = (poly.holes || []).length;
    const meta = poly.valid
      ? `${poly.points.length} 外环顶点${nHoles ? ` · ${nHoles} 个孔洞（面积已扣除，周长含孔洞边界）` : ""}`
      : "";
    card.innerHTML = `
      <div class="poly-card-head">
        <span class="poly-swatch" style="background:${color}"></span>
        <input type="text" value="${poly.name.replace(/"/g, "&quot;")}" data-id="${poly.id}" class="poly-name" />
        <button class="btn btn-ghost btn-mini poly-hole-btn" data-id="${poly.id}" title="在该外环内部绘制一个镂空孔洞">◔ 挖洞</button>
        <button class="poly-del" data-id="${poly.id}" title="删除该区域">🗑</button>
      </div>
      <div class="poly-metrics">
        <div class="metric"><span class="m-label">净面积（外环${nHoles ? "−孔洞" : ""}）</span>
          <span class="m-value ${poly.valid ? "" : "bad"}">${poly.valid ? fmtNum(areaConv) : "—"}</span>
          <span class="m-label m-unit">${areaUnit}</span></div>
        <div class="metric"><span class="m-label">总周长（外环+孔洞）</span>
          <span class="m-value ${poly.valid ? "" : "bad"}">${poly.valid ? fmtNum(perConv) : "—"}</span>
          <span class="m-label m-unit">${state.displayUnit}</span></div>
      </div>
      ${meta ? `<div class="poly-meta">${meta}</div>`
             : `<div class="poly-error">${(poly.errors || []).map((x) => `• ${x}`).join("<br>")}</div>`}
      <div class="hole-list"></div>
    `;
    card.addEventListener("click", (e) => {
      if (e.target.closest(".poly-del")) {
        deletePolygon(poly.id);
      } else if (e.target.closest(".poly-hole-btn")) {
        state.selected = poly.id;
        startHoleDraft(poly.id);
      } else if (e.target.closest(".poly-name")) {
        // allow editing; no selection change / mode switch
      } else {
        state.selected = poly.id;
        setMode("edit");
      }
    });
    card.querySelector(".poly-name").addEventListener("change", (e) => {
      poly.name = e.target.value.trim() || poly.name;
      syncPolygon(poly);
    });
    renderHoleRows(card.querySelector(".hole-list"), poly);
    list.appendChild(card);
  });
}

/** Per-hole metrics + delete buttons inside a region card. */
function renderHoleRows(container, poly) {
  container.innerHTML = "";
  (poly.holes || []).forEach((hole, hi) => {
    const row = document.createElement("div");
    row.className = "hole-row";
    const ha = convertValue(hole.area, state.unit, state.displayUnit, true);
    const hp = convertValue(hole.perimeter, state.unit, state.displayUnit, false);
    const areaUnit = state.displayUnit === "px" ? "px²" : `${state.displayUnit}²`;
    row.innerHTML = `
      <span class="hole-tag">孔洞 ${hi + 1}</span>
      <span class="hole-m">面积 <b>${hole.valid ? fmtNum(ha) : "—"}</b> ${areaUnit}</span>
      <span class="hole-m">周长 <b>${hole.valid ? fmtNum(hp) : "—"}</b> ${state.displayUnit}</span>
      <button class="hole-del" data-hi="${hi}" title="删除孔洞 ${hi + 1}">✕</button>`;
    row.querySelector(".hole-del").addEventListener("click", () => removeHole(poly, hi));
    container.appendChild(row);
  });
}

/** Update metric values inside existing cards without rebuilding the DOM
 *  (used while dragging, so the rename <input> keeps focus). */
function updatePolyCardMetrics() {
  const cards = els.polyList.querySelectorAll(".poly-card");
  cards.forEach((card, i) => {
    const poly = state.polygons[i];
    if (!poly) return;
    const vals = card.querySelectorAll(".m-value");
    const areaConv = convertValue(poly.area, state.unit, state.displayUnit, true);
    const perConv = convertValue(poly.perimeter, state.unit, state.displayUnit, false);
    if (vals[0]) {
      vals[0].textContent = poly.valid ? fmtNum(areaConv) : "—";
      vals[0].classList.toggle("bad", !poly.valid);
    }
    if (vals[1]) {
      vals[1].textContent = poly.valid ? fmtNum(perConv) : "—";
      vals[1].classList.toggle("bad", !poly.valid);
    }
    const units = card.querySelectorAll(".m-unit");
    if (units[0]) units[0].textContent = state.displayUnit === "px" ? "px²" : `${state.displayUnit}²`;
    if (units[1]) units[1].textContent = state.displayUnit;
    // live validity / conflict message + hole rows
    const errBox = card.querySelector(".poly-error");
    const metaBox = card.querySelector(".poly-meta");
    const nHoles = (poly.holes || []).length;
    if (poly.valid) {
      if (errBox) {
        const meta = document.createElement("div");
        meta.className = "poly-meta";
        meta.textContent = `${poly.points.length} 外环顶点${nHoles ? ` · ${nHoles} 个孔洞（面积已扣除，周长含孔洞边界）` : ""}`;
        errBox.replaceWith(meta);
      } else if (metaBox) {
        metaBox.textContent = `${poly.points.length} 外环顶点${nHoles ? ` · ${nHoles} 个孔洞（面积已扣除，周长含孔洞边界）` : ""}`;
      }
      card.classList.remove("invalid");
    } else if (errBox) {
      errBox.innerHTML = (poly.errors || []).map((x) => `• ${x}`).join("<br>");
      card.classList.add("invalid");
    }
    renderHoleRows(card.querySelector(".hole-list"), poly);
  });
}

function renderModeButtons() {
  els.modeCalib.classList.toggle("active", state.mode === "calib");
  els.modeDraw.classList.toggle("active", state.mode === "draw");
  els.modeEdit.classList.toggle("active", state.mode === "edit");
  els.modePan.classList.toggle("active", state.mode === "pan");
  const panning = state.spaceDown || state.dragging?.type === "pan";
  $canvas.className = `mode-${state.mode}` + (panning ? " panning" : "");
  const labels = { calib: "角点标定", draw: "绘制多边形 / 孔洞", edit: "编辑顶点", pan: "平移视图" };
  els.modeLabel.textContent = `模式：${labels[state.mode]}`;
  const targetName = state.draftTarget
    ? state.polygons.find((p) => p.id === state.draftTarget)?.name
    : null;
  const hints = {
    calib: "单击放置/拖动角点 · 右键拖动平移 · 滚轮缩放",
    draw: targetName
      ? `正在为「${targetName}」挖洞：单击添加顶点 · 双击或 Enter 闭合 · Esc 取消 · Backspace 撤销`
      : "单击添加顶点 · 双击或 Enter 闭合 · Esc 取消 · Backspace 撤销",
    edit: "拖动外环/孔洞顶点或整体 · 右键顶点删除 · 右键孔洞删除孔洞 · Del 删除区域",
    pan: "拖动平移 · 滚轮缩放",
  };
  els.hintLabel.textContent = state.image ? hints[state.mode] : "";
}

function refreshUI() {
  renderCorners();
  renderPolyList();
  renderModeButtons();
  updateUnitOptions();
  render();
}

function updateUnitOptions() {
  const sel = els.displayUnit;
  if (sel.options.length !== UNITS.length || sel.value !== state.displayUnit) {
    const prev = state.displayUnit;
    sel.innerHTML = UNITS.map((u) => `<option value="${u}">${u === "px" ? "标定单位" : u}</option>`).join("");
    sel.value = UNITS.includes(prev) ? prev : state.unit;
    state.displayUnit = sel.value;
  }
}

function updateZoomLabel() {
  els.zoomLevel.textContent = state.image ? Math.round(state.view.scale * 100) + "%" : "—";
}

/* ====================================================== calibration logic */
let calibSeq = 0;
async function recomputeCalibration({ push = true } = {}) {
  // Local live computation (instant while dragging).
  const dimsValid = isFinite(state.rectW) && isFinite(state.rectH) &&
    state.rectW > 0 && state.rectH > 0;
  if (state.corners.length === 4 && dimsValid) {
    const r = solveHomography(state.corners, state.rectW, state.rectH);
    state.calibValid = r.ok;
    state.H = r.ok ? r.H : null;
    state.calibErrors = r.errors || [];
  } else {
    state.calibValid = false;
    state.H = null;
    state.calibErrors = [];
  }
  recomputePolygonMetrics();

  const seq = ++calibSeq;
  if (push && state.corners.length === 4 && dimsValid) {
    try {
      const data = await api(`/api/sessions/${state.sessionId}/calibration`, {
        method: "PUT",
        body: JSON.stringify({
          corners: state.corners,
          rect_width: state.rectW,
          rect_height: state.rectH,
          unit: state.unit,
        }),
      });
      if (seq !== calibSeq) return; // a newer edit is already in flight
      // Adopt authoritative server answer.
      state.calibValid = data.calibration.valid;
      state.H = data.calibration.H;
      state.calibErrors = data.calibration.errors || [];
      mergeServerPolygons(data.polygons);
    } catch (err) {
      if (seq === calibSeq) toast("标定同步失败（本地结果仍显示）：" + err.message, true);
    }
  }
  els.modeDraw.disabled = !state.calibValid;
  els.modeEdit.disabled = state.polygons.length === 0;
  els.modeDrawAnother.disabled = !state.calibValid;
  els.clearPolys.disabled = state.polygons.length === 0;
  refreshUI();
}

function adoptServerPolygon(sp) {
  const local = state.polygons.find((p) => p.id === sp.id);
  if (!local) return;
  local.name = sp.name;
  local.points = sp.points.map((p) => ({ x: p.x, y: p.y }));
  local.holes = (sp.holes || []).map((h) => ({
    id: h.id,
    points: h.points.map((p) => ({ x: p.x, y: p.y })),
    area: h.area,
    perimeter: h.perimeter,
    valid: h.valid,
    worldPoints: h.world_points,
  }));
  local.area = sp.area;
  local.outerArea = sp.outer_area ?? null;
  local.perimeter = sp.perimeter;
  local.valid = sp.valid;
  local.errors = sp.errors;
  local.worldPoints = sp.world_points;
}

function mergeServerPolygons(serverPolys) {
  for (const sp of serverPolys) adoptServerPolygon(sp);
}

function measureLocal(poly) {
  if (state.calibValid && state.H) {
    return (poly.holes && poly.holes.length)
      ? measureRegion(state.H, poly.points, poly.holes.map((h) => h.points))
      : measurePolygon(state.H, poly.points);
  }
  return { valid: false, area: null, perimeter: null, errors: ["尚未完成透视校准"], worldPoints: null };
}

function applyLocalMeasure(poly, m) {
  poly.valid = m.valid;
  poly.area = m.area ?? null;
  poly.outerArea = m.outerArea ?? null;
  poly.perimeter = m.perimeter ?? null;
  poly.errors = m.errors || [];
  poly.worldPoints = m.worldPoints || null;
  if (poly.holes) {
    poly.holes.forEach((h, i) => {
      if (m.valid && m.holeAreas) {
        h.valid = true;
        h.area = m.holeAreas[i];
        h.perimeter = m.holePerimeters[i];
        h.worldPoints = m.worldHoles[i];
      } else {
        h.valid = false;
        h.area = null;
        h.perimeter = null;
        h.worldPoints = null;
      }
    });
  }
}

function recomputePolygonMetrics() {
  for (const poly of state.polygons) applyLocalMeasure(poly, measureLocal(poly));
}

let holeIdCounter = 0;
function newHoleId() {
  holeIdCounter += 1;
  return `hole-${Date.now().toString(36)}-${holeIdCounter}-${Math.random().toString(36).slice(2, 8)}`;
}

/* ============================================================ polygons === */
const polyBody = (poly) => ({
  points: poly.points,
  name: poly.name,
  holes: (poly.holes || []).map((h) => ({ id: h.id, points: h.points })),
});

async function addPolygon(points, name) {
  const payload = { points, name };
  let created;
  try {
    const r = await api(`/api/sessions/${state.sessionId}/polygons`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
    created = r.polygon;
  } catch (err) {
    toast("区域保存失败：" + err.message, true);
    return null;
  }
  state.polygons.push({
    id: created.id,
    name: created.name,
    points: created.points.map((p) => ({ x: p.x, y: p.y })),
    holes: (created.holes || []).map((h) => ({
      id: h.id,
      points: h.points.map((p) => ({ x: p.x, y: p.y })),
      area: h.area, perimeter: h.perimeter, valid: h.valid, worldPoints: h.world_points,
    })),
    area: created.area,
    outerArea: created.outer_area ?? null,
    perimeter: created.perimeter,
    valid: created.valid,
    errors: created.errors,
    worldPoints: created.world_points,
  });
  state.selected = created.id;
  recomputePolygonMetrics();
  els.modeEdit.disabled = false;
  els.modeDrawAnother.disabled = false;
  els.clearPolys.disabled = false;
  setMode("edit");
  refreshUI();
  return state.polygons[state.polygons.length - 1];
}

let syncTimer = null;
function syncPolygon(poly) {
  clearTimeout(syncTimer);
  // Never push an invalid candidate: the server keeps its last valid result,
  // and the live (red) preview is computed locally while dragging.
  if (!poly.valid) {
    showInvalidNotice(poly);
    return;
  }
  showNotice(null);
  syncTimer = setTimeout(async () => {
    let r;
    try {
      const res = await fetch(`/api/sessions/${state.sessionId}/polygons/${poly.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(polyBody(poly)),
      });
      if (res.status === 409) {
        const data = await res.json();
        // Only roll back if the local geometry is still the rejected candidate;
        // otherwise a newer edit is in progress and must not be overwritten.
        const cand = data.candidate;
        const sameGeom = cand &&
          cand.points.length === poly.points.length &&
          cand.points.every((p, k) => p.x === poly.points[k].x && p.y === poly.points[k].y) &&
          (cand.holes || []).length === (poly.holes || []).length &&
          (cand.holes || []).every((h, hi) =>
            h.points.length === poly.holes[hi].points.length &&
            h.points.every((p, k) => p.x === poly.holes[hi].points[k].x && p.y === poly.holes[hi].points[k].y));
        if (sameGeom && data.polygon) adoptServerPolygon(data.polygon);
        showNotice((cand?.errors || ["区域几何校验未通过"]).join("；") +
                   " —— 服务端保留最后一次有效结果", true);
        refreshUI();
        return;
      }
      if (!res.ok) {
        let detail = res.statusText;
        try { detail = (await res.json()).detail || detail; } catch { /* ignore */ }
        throw new Error(detail);
      }
      r = await res.json();
    } catch (err) {
      toast("区域更新失败：" + err.message, true);
      return;
    }
    const sp = r.polygon;
    poly.area = sp.area;
    poly.outerArea = sp.outer_area ?? null;
    poly.perimeter = sp.perimeter;
    poly.valid = sp.valid;
    poly.errors = sp.errors;
    poly.worldPoints = sp.world_points;
    poly.holes = (sp.holes || []).map((h) => ({
      id: h.id,
      points: h.points.map((p) => ({ x: p.x, y: p.y })),
      area: h.area, perimeter: h.perimeter, valid: h.valid, worldPoints: h.world_points,
    }));
    showNotice(null);
    refreshUI();
  }, 250);
}

function showInvalidNotice(poly) {
  showNotice(`「${poly.name}」当前状态非法：${(poly.errors || []).join("；")} —— 拖动调整后会实时重算，服务端结果未被覆盖`, true);
}

async function deletePolygon(id) {
  try {
    await api(`/api/sessions/${state.sessionId}/polygons/${id}`, { method: "DELETE" });
  } catch (err) {
    toast("删除失败：" + err.message, true);
    return;
  }
  state.polygons = state.polygons.filter((p) => p.id !== id);
  if (state.selected === id) {
    state.selected = state.polygons[0]?.id || null;
    if (!state.selected && state.mode === "edit") setMode(state.calibValid ? "draw" : "calib");
  }
  els.modeEdit.disabled = state.polygons.length === 0;
  els.modeDrawAnother.disabled = !state.calibValid;
  els.clearPolys.disabled = state.polygons.length === 0;
  refreshUI();
}

/* ====================================================== hit testing ====== */
function hitCorner(imgPt) {
  const TOL = 9 / state.view.scale;
  let best = -1, bestD = TOL;
  state.corners.forEach((p, i) => {
    const d = dist(p, imgPt);
    if (d < bestD) { bestD = d; best = i; }
  });
  return best;
}

function hitPolygonVertex(imgPt) {
  const TOL = 9 / state.view.scale;
  for (let i = state.polygons.length - 1; i >= 0; i--) {
    const poly = state.polygons[i];
    for (let k = 0; k < poly.points.length; k++)
      if (dist(poly.points[k], imgPt) < TOL) return { polyIndex: i, polyId: poly.id, ring: "outer", vertex: k };
    for (let hi = (poly.holes || []).length - 1; hi >= 0; hi--) {
      const hole = poly.holes[hi];
      for (let k = 0; k < hole.points.length; k++)
        if (dist(hole.points[k], imgPt) < TOL)
          return { polyIndex: i, polyId: poly.id, ring: "hole", holeIndex: hi, holeId: hole.id, vertex: k };
    }
  }
  return null;
}

function pointInPolygon(pt, pts) {
  let inside = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const a = pts[i], b = pts[j];
    if ((a.y > pt.y) !== (b.y > pt.y) &&
        pt.x < ((b.x - a.x) * (pt.y - a.y)) / (b.y - a.y) + a.x)
      inside = !inside;
  }
  return inside;
}

/** {polyIndex, ring:'outer'|'hole', holeIndex} or null.  The physical solid
 *  region is outer minus holes, so points inside a hole do not grab the body. */
function hitPolygonBody(imgPt) {
  for (let i = state.polygons.length - 1; i >= 0; i--) {
    const poly = state.polygons[i];
    const edgeHit = (pts, ring, holeIndex) => {
      const TOL = 6 / state.view.scale;
      for (let k = 0; k < pts.length; k++)
        if (distToSegment(imgPt, pts[k], pts[(k + 1) % pts.length]) < TOL)
          return { polyIndex: i, polyId: poly.id, ring, holeIndex };
      return null;
    };
    // holes first: hole edges/interiors take precedence over the outer body
    for (let hi = (poly.holes || []).length - 1; hi >= 0; hi--) {
      const hp = poly.holes[hi].points;
      const h = edgeHit(hp, "hole", hi);
      if (h) return h;
      if (pointInPolygon(imgPt, hp)) return { polyIndex: i, polyId: poly.id, ring: "hole", holeIndex: hi };
    }
    const o = edgeHit(poly.points, "outer", null);
    if (o) return o;
    if (pointInPolygon(imgPt, poly.points)) {
      const insideHole = (poly.holes || []).some((hh) => pointInPolygon(imgPt, hh.points));
      if (!insideHole) return { polyIndex: i, polyId: poly.id, ring: "outer", holeIndex: null };
    }
  }
  return null;
}

function distToSegment(p, a, b) {
  const abx = b.x - a.x, aby = b.y - a.y;
  const t = Math.min(1, Math.max(0, ((p.x - a.x) * abx + (p.y - a.y) * aby) / (abx * abx + aby * aby || 1)));
  return dist(p, { x: a.x + abx * t, y: a.y + aby * t });
}

/* ============================================================ pointer === */
function eventPos(e) {
  const r = $canvas.getBoundingClientRect();
  return { x: e.clientX - r.left, y: e.clientY - r.top };
}

$canvas.addEventListener("pointerdown", (e) => {
  if (!state.image) return;
  $canvas.setPointerCapture(e.pointerId);
  const sp = eventPos(e);
  const ip = screenToImg(sp);
  const panMode = state.spaceDown || state.mode === "pan" || e.button === 2;

  if (panMode) {
    state.dragging = { type: "pan", start: sp, view: { ...state.view } };
    renderModeButtons();
    return;
  }

  if (e.button !== 0) return;

  if (state.mode === "calib") {
    const ci = hitCorner(ip);
    if (ci >= 0) {
      state.dragging = { type: "corner", index: ci };
    } else if (state.corners.length < 4) {
      state.corners.push(clampImagePoint(ip));
      void recomputeCalibration({ push: false });
      if (state.corners.length === 4) void recomputeCalibration();
    }
    return;
  }

  if (state.mode === "draw") {
    if (!state.calibValid) return;
    // A double-click closes the ring; its second pointerdown must not add
    // a duplicate vertex at the same spot.
    if (state.draft.length && dist(ip, state.draft[state.draft.length - 1]) < 8 / state.view.scale)
      return;
    // clicking near first vertex closes
    if (state.draft.length >= 3 && dist(ip, state.draft[0]) < 12 / state.view.scale) {
      finishDraft();
      return;
    }
    state.draft.push(clampImagePoint(ip));
    render();
    return;
  }

  if (state.mode === "edit") {
    const hv = hitPolygonVertex(ip);
    if (hv) {
      state.selected = hv.polyId;
      state.dragging = {
        type: "vertex",
        ...hv,
      };
    } else {
      const hit = hitPolygonBody(ip);
      if (hit) {
        const poly = state.polygons[hit.polyIndex];
        state.selected = poly.id;
        const ringPts = hit.ring === "hole" ? poly.holes[hit.holeIndex].points : poly.points;
        state.dragging = {
          type: "body",
          polyId: poly.id,
          ring: hit.ring,
          holeIndex: hit.holeIndex,
          startImg: ip,
          orig: ringPts.map((p) => ({ ...p })),
        };
      } else {
        state.selected = null;
      }
    }
    refreshUI();
  }
});

$canvas.addEventListener("pointermove", (e) => {
  const sp = eventPos(e);
  const ip = screenToImg(sp);
  state.hoverImg = ip;
  els.cursorLabel.textContent = state.image
    ? `图像坐标 ${isFinite(ip.x) ? Math.round(ip.x) : "—"}, ${isFinite(ip.y) ? Math.round(ip.y) : "—"} px`
    : "图像坐标 —";

  const d = state.dragging;
  if (!d) { render(); return; }

  if (d.type === "pan") {
    state.view.x = d.view.x + (sp.x - d.start.x);
    state.view.y = d.view.y + (sp.y - d.start.y);
    render();
    return;
  }
  if (d.type === "corner") {
    state.corners[d.index] = clampImagePoint(ip);
    void recomputeCalibration({ push: false });
    return;
  }
  const poly = state.polygons.find((p) => p.id === d.polyId);
  if (!poly) return;
  const ringPts = d.ring === "hole" ? poly.holes[d.holeIndex].points : poly.points;
  if (d.type === "vertex") {
    ringPts[d.vertex] = clampImagePoint(ip);
  } else if (d.type === "body") {
    const dx = ip.x - d.startImg.x, dy = ip.y - d.startImg.y;
    const moved = d.orig.map((p) => clampImagePoint({ x: p.x + dx, y: p.y + dy }));
    if (d.ring === "hole") poly.holes[d.holeIndex].points = moved;
    else poly.points = moved;
  }
  applyLocalMeasure(poly, measureLocal(poly));
  if (!poly.valid) showInvalidNotice(poly); else showNotice(null);
  render();
  updatePolyCardMetrics();
});

$canvas.addEventListener("pointerup", (e) => {
  const d = state.dragging;
  state.dragging = null;
  $canvas.releasePointerCapture?.(e.pointerId);
  if (!d) { render(); return; }
  if (d.type === "corner") {
    void recomputeCalibration();
  } else if (d.type === "vertex" || d.type === "body") {
    const poly = state.polygons.find((p) => p.id === d.polyId);
    if (poly) syncPolygon(poly);
  }
  renderModeButtons();
  render();
});

$canvas.addEventListener("pointerleave", () => {
  state.hoverImg = null;
  render();
});

$canvas.addEventListener("dblclick", () => {
  if (state.mode === "draw" && state.draft.length >= 3) finishDraft();
});

$canvas.addEventListener("contextmenu", (e) => {
  e.preventDefault();
  if (!state.image) return;
  const ip = screenToImg(eventPos(e));
  if (state.mode === "edit") {
    const hv = hitPolygonVertex(ip);
    if (hv) {
      const poly = state.polygons[hv.polyIndex];
      const ring = hv.ring === "hole" ? poly.holes[hv.holeIndex] : poly;
      const ringName = hv.ring === "hole" ? `孔洞 ${hv.holeIndex + 1}` : "外环";
      if (ring.points.length <= 3) {
        if (hv.ring === "hole") {
          if (!confirm(`${ringName} 至少保留 3 个顶点，改为删除整个孔洞？`)) return;
          removeHole(poly, hv.holeIndex);
        } else {
          toast("闭合多边形至少保留 3 个顶点，已改为删除整个区域");
          void deletePolygon(poly.id);
        }
      } else {
        ring.points.splice(hv.vertex, 1);
        applyLocalMeasure(poly, measureLocal(poly));
        syncPolygon(poly);
        refreshUI();
      }
      return;
    }
    // right-click inside a hole body deletes that hole
    const hit = hitPolygonBody(ip);
    if (hit && hit.ring === "hole") {
      const poly = state.polygons[hit.polyIndex];
      if (confirm(`删除该区域的孔洞 ${hit.holeIndex + 1}？`)) removeHole(poly, hit.holeIndex);
    }
  }
});

$canvas.addEventListener("wheel", (e) => {
  if (!state.image) return;
  e.preventDefault();
  const factor = Math.exp(-e.deltaY * 0.0012);
  zoomAt(eventPos(e), factor);
}, { passive: false });

function finishDraft() {
  const pts = state.draft.map((p) => ({ ...p }));
  const targetId = state.draftTarget;
  state.draft = [];
  state.draftTarget = null;
  if (pts.length < 3) { toast("至少需要 3 个顶点"); state.draft = pts; state.draftTarget = targetId; return; }

  if (targetId) {
    const poly = state.polygons.find((p) => p.id === targetId);
    if (!poly) { setMode("edit"); refreshUI(); return; }
    const hole = { id: newHoleId(), points: pts, valid: false, area: null, perimeter: null, worldPoints: null };
    poly.holes = poly.holes || [];
    poly.holes.push(hole);
    state.selected = poly.id;
    applyLocalMeasure(poly, measureLocal(poly));
    if (poly.valid) {
      syncPolygon(poly);
    } else {
      // Keep the hole locally for adjustment; nothing invalid reaches the server.
      showInvalidNotice(poly);
    }
    setMode("edit");
    refreshUI();
    return;
  }

  // Instant local validity feedback, server re-checks on persist.
  const m = state.calibValid ? measurePolygon(state.H, pts) : { valid: false, errors: ["尚未标定"] };
  if (!m.valid) {
    showNotice(m.errors.join("；"), true);
    state.draft = pts;
    return;
  }
  void addPolygon(pts);
}

function removeHole(poly, holeIndex) {
  poly.holes.splice(holeIndex, 1);
  applyLocalMeasure(poly, measureLocal(poly));
  syncPolygon(poly);
  refreshUI();
}

/* ============================================================ keyboard == */
window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.code === "Space") { state.spaceDown = true; renderModeButtons(); e.preventDefault(); }
  if (e.key === "Escape") {
    state.draft = [];
    state.draftTarget = null;
    state.dragging = null;
    showNotice(null);
    render();
  }
  if (e.key === "Backspace" && state.mode === "draw" && state.draft.length) {
    state.draft.pop();
    render();
    e.preventDefault();
  }
  if (e.key === "Enter" && state.mode === "draw" && state.draft.length >= 3) finishDraft();
  if (e.key === "Delete" && state.mode === "edit" && state.selected) deletePolygon(state.selected);
  if (e.key === "1") setMode("calib");
  if (e.key === "2" && !els.modeDraw.disabled) setMode("draw");
  if (e.key === "3" && !els.modeEdit.disabled) setMode("edit");
  if (e.key === "4") setMode("pan");
  if (e.key === "f" || e.key === "F") fitView();
  if (e.key === "+" || e.key === "=") {
    const { w, h } = screenSize();
    zoomAt({ x: w / 2, y: h / 2 }, 1.2);
  }
  if (e.key === "-") {
    const { w, h } = screenSize();
    zoomAt({ x: w / 2, y: h / 2 }, 1 / 1.2);
  }
});
window.addEventListener("keyup", (e) => {
  if (e.code === "Space") { state.spaceDown = false; renderModeButtons(); }
});

/* ============================================================ modes ====== */
function setMode(m) {
  if (m === "draw" && !state.calibValid) {
    showNotice("请先完成有效的四点透视标定", true);
    return;
  }
  if (m === "edit" && state.polygons.length === 0) return;
  if (m !== "draw") {
    state.draft = [];
    state.draftTarget = null;
  }
  state.mode = m;
  showNotice(null);
  renderModeButtons();
  render();
}

/** Enter hole-drawing mode for a specific existing region. */
function startHoleDraft(polyId) {
  const poly = state.polygons.find((p) => p.id === polyId);
  if (!poly) return;
  state.selected = polyId;
  state.draftTarget = polyId;
  state.draft = [];
  setMode("draw");
  showNotice(`正在为「${poly.name}」绘制镂空孔洞：单击加点，双击/Enter 闭合；孔洞必须完全位于外环内且与其他孔洞分离`, false);
}

els.modeCalib.addEventListener("click", () => setMode("calib"));
els.modeDraw.addEventListener("click", () => { state.draftTarget = null; setMode("draw"); });
els.modeEdit.addEventListener("click", () => setMode("edit"));
els.modePan.addEventListener("click", () => setMode("pan"));
els.modeDrawAnother.addEventListener("click", () => { state.draftTarget = null; setMode("draw"); });

els.clearCorners.addEventListener("click", () => {
  state.corners = [];
  state.H = null;
  state.calibValid = false;
  state.calibErrors = [];
  recomputePolygonMetrics();
  els.modeDraw.disabled = true;
  setMode("calib");
  refreshUI();
  toast("已清除角点，可重新标记");
});

els.clearPolys.addEventListener("click", async () => {
  if (!state.polygons.length) return;
  if (!confirm(`确定删除全部 ${state.polygons.length} 个测量区域？`)) return;
  try {
    await api(`/api/sessions/${state.sessionId}/polygons`, { method: "DELETE" });
  } catch (err) { toast("清空失败：" + err.message, true); return; }
  state.polygons = [];
  state.selected = null;
  els.modeEdit.disabled = true;
  els.modeDrawAnother.disabled = !state.calibValid;
  els.clearPolys.disabled = true;
  setMode(state.calibValid ? "draw" : "calib");
  refreshUI();
});

/* ----------------------------------------------------------- zoom UI --- */
els.zoomIn.addEventListener("click", () => zoomCenter(1.25));
els.zoomOut.addEventListener("click", () => zoomCenter(1 / 1.25));
els.zoomFit.addEventListener("click", fitView);
els.zoomOne.addEventListener("click", () => zoomCenterTo(1));
function zoomCenter(f) { const { w, h } = screenSize(); zoomAt({ x: w / 2, y: h / 2 }, f); }
function zoomCenterTo(s) { const { w, h } = screenSize(); zoomAt({ x: w / 2, y: h / 2 }, s / state.view.scale); }

/* ------------------------------------------------------------- notice --- */
let noticeTimer = null;
function showNotice(msg, warn = false) {
  const n = els.notice;
  if (!msg) { n.classList.add("hidden"); return; }
  els.noticeText.textContent = msg;
  n.classList.toggle("warn", warn);
  n.classList.remove("hidden");
  clearTimeout(noticeTimer);
}
els.noticeClose.addEventListener("click", () => showNotice(null));

/* --------------------------------------------------------- dimension UI - */
els.rectW.addEventListener("input", () => {
  state.rectW = parseFloat(els.rectW.value);
  void recomputeCalibration();
});
els.rectH.addEventListener("input", () => {
  state.rectH = parseFloat(els.rectH.value);
  void recomputeCalibration();
});
els.rectUnit.addEventListener("change", () => {
  state.unit = els.rectUnit.value;
  if (!UNITS.includes(state.displayUnit) || state.displayUnit === "px")
    state.displayUnit = state.unit;
  void recomputeCalibration();
});
els.displayUnit.addEventListener("change", () => {
  state.displayUnit = els.displayUnit.value;
  renderPolyList();
});

/* ============================================================ upload ===== */
async function ensureSession() {
  let sid = null;
  try { sid = localStorage.getItem("pr_sid"); } catch { /* ignore */ }
  if (sid) {
    try {
      const s = await api(`/api/sessions/${sid}`);
      hydrateSession(s);
      return;
    } catch { /* expired session -> create new */ }
  }
  const r = await api("/api/sessions", { method: "POST" });
  state.sessionId = r.session_id;
  try { localStorage.setItem("pr_sid", r.session_id); } catch { /* ignore */ }
  setMode("calib");
  refreshUI();
}

function hydrateSession(s) {
  state.sessionId = s.session_id;
  try { localStorage.setItem("pr_sid", s.session_id); } catch { /* ignore */ }
  if (s.image) {
    state.image = s.image;
    if (s.calibration) {
      state.corners = s.calibration.corners.map((p) => ({ x: p.x, y: p.y }));
      state.rectW = s.calibration.rect_width;
      state.rectH = s.calibration.rect_height;
      state.unit = s.calibration.unit;
      state.H = s.calibration.H;
      state.calibValid = s.calibration.valid;
      state.calibErrors = s.calibration.errors || [];
      els.rectW.value = state.rectW;
      els.rectH.value = state.rectH;
      els.rectUnit.value = state.unit;
    }
    state.polygons = s.polygons.map((p) => ({
      id: p.id, name: p.name,
      points: p.points.map((q) => ({ x: q.x, y: q.y })),
      holes: (p.holes || []).map((h) => ({
        id: h.id,
        points: h.points.map((q) => ({ x: q.x, y: q.y })),
        area: h.area, perimeter: h.perimeter, valid: h.valid, worldPoints: h.world_points,
      })),
      area: p.area, outerArea: p.outer_area ?? null, perimeter: p.perimeter, valid: p.valid,
      errors: p.errors, worldPoints: p.world_points,
    }));
    state.selected = state.polygons[0]?.id || null;
    loadImage(s.image.filename, false);
  } else {
    resetUI();
  }
  els.modeDraw.disabled = !state.calibValid;
  els.modeEdit.disabled = state.polygons.length === 0;
  els.modeDrawAnother.disabled = !state.calibValid;
  els.clearPolys.disabled = state.polygons.length === 0;
  els.exportBtn.disabled = !s.image;
  state.displayUnit = state.unit;
  setMode(state.calibValid ? (state.polygons.length ? "edit" : "draw") : "calib");
  refreshUI();
}

function resetUI() {
  state.image = null;
  state.imgEl = null;
  state.corners = [];
  state.H = null;
  state.calibValid = false;
  state.calibErrors = [];
  state.polygons = [];
  state.draft = [];
  state.draftTarget = null;
  state.selected = null;
  els.exportBtn.disabled = true;
  els.emptyOverlay.classList.remove("hidden");
}

function loadImage(filename, keepView = false) {
  const img = new Image();
  img.onload = () => {
    state.imgEl = img;
    els.emptyOverlay.classList.add("hidden");
    els.exportBtn.disabled = false;
    if (!keepView) fitView();
    render();
  };
  img.src = `/api/images/${filename}`;
}

async function uploadFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    showNotice("请选择图像文件（JPG / PNG / WebP）", true);
    return;
  }
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(`/api/sessions/${state.sessionId}/image`, { method: "POST", body: fd });
  if (!res.ok) {
    let detail = "上传失败";
    try { detail = (await res.json()).detail || detail; } catch { /* ignore */ }
    showNotice(detail, true);
    return;
  }
  const data = await res.json();
  state.image = data.image;
  state.corners = [];
  state.H = null;
  state.calibValid = false;
  state.polygons = [];
  state.draft = [];
  state.draftTarget = null;
  state.selected = null;
  els.rectW.value = ""; els.rectH.value = "";
  state.rectW = null; state.rectH = null;
  els.modeDraw.disabled = true;
  els.modeEdit.disabled = true;
  els.modeDrawAnother.disabled = true;
  els.clearPolys.disabled = true;
  loadImage(data.image.filename);
  setMode("calib");
  refreshUI();
  toast("照片已上传，请标记已知矩形的四个角点");
}

els.fileInput.addEventListener("change", (e) => {
  const f = e.target.files?.[0];
  if (f) void uploadFile(f);
  e.target.value = "";
});

["dragenter", "dragover"].forEach((ev) =>
  els.dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    els.dropzone.classList.add("drag");
  })
);
["dragleave", "drop"].forEach((ev) =>
  els.dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    els.dropzone.classList.remove("drag");
  })
);
els.dropzone.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files?.[0];
  if (f) void uploadFile(f);
});
// Also allow dropping directly on the canvas to replace the photo.
$canvas.addEventListener("dragover", (e) => e.preventDefault());
$canvas.addEventListener("drop", (e) => {
  e.preventDefault();
  const f = e.dataTransfer.files?.[0];
  if (f) void uploadFile(f);
});

/* ============================================================ export ===== */
els.exportBtn.addEventListener("click", () => {
  if (!state.sessionId) return;
  const url = `/api/sessions/${state.sessionId}/export`;
  const a = document.createElement("a");
  a.href = url;
  a.download = `measurements-${state.sessionId.slice(0, 8)}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
});

/* ============================================================ boot ======== */
window.addEventListener("resize", resizeCanvas);

(async function boot() {
  resizeCanvas();
  renderModeButtons();
  try {
    await ensureSession();
  } catch (err) {
    showNotice("无法连接后端服务：" + err.message, true);
  }
})();
