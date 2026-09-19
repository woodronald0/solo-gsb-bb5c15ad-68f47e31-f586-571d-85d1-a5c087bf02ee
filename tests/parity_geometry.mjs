/* Parity test: frontend/app.js geometry vs the same synthetic camera used in
 * tests/test_geometry.py. Run: node tests/parity_geometry.mjs
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
let src = readFileSync(join(here, "..", "frontend", "app.js"), "utf8");

// Strip browser-only sections; keep pure geometry functions.
const start = src.indexOf("/* ============================================================ geometry");
const end = src.indexOf("/* ============================================================ state");
src = src.slice(start, end)
  .replace("\"use strict\";", "");

const fns = new Function(`${src}; return {
  solveHomography, applyH, measurePolygon, measureRegion, polygonSelfIntersects, convexOrderedQuad, dist
};`);
const G = fns();

// --- same synthetic camera as the Python test ---
const K = [[800, 0, 512], [0, 800, 384], [0, 0, 1]];
const r1 = [0.999, 0, -0.0499];
const r2 = [0.02, 1, 0.03];
const t = [0, 0, 12];
const P = [
  [K[0][0] * r1[0] + K[0][2] * r1[2], K[0][0] * r2[0] + K[0][2] * r2[2], K[0][2] * t[2]],
  [K[1][1] * r1[1] + K[1][2] * r1[2], K[1][1] * r2[1] + K[1][2] * r2[2], K[1][2] * t[2]],
  [r1[2], r2[2], t[2]],
];
const project = (x, y) => {
  const w = P[2][0] * x + P[2][1] * y + P[2][2];
  if (w <= 1) throw new Error("behind camera");
  return {
    x: (P[0][0] * x + P[0][1] * y + P[0][2]) / w,
    y: (P[1][0] * x + P[1][1] * y + P[1][2]) / w,
  };
};

const W = 100, H = 60;
const corners = [[0, 0], [W, 0], [W, H], [0, H]].map(([x, y]) => project(x, y));
let fails = 0;
const check = (name, cond, detail = "") => {
  console.log(`  ${cond ? "PASS" : "FAIL"}  ${name}  ${detail}`);
  if (!cond) fails++;
};

console.log("== JS homography ==");
const r = G.solveHomography(corners, W, H);
check("solve ok", r.ok, JSON.stringify(r.errors));
if (r.ok) {
  let err = 0;
  [[0, 0], [W, 0], [W, H], [0, H]].forEach(([x, y], i) => {
    const q = G.applyH(r.H, corners[i]);
    err = Math.max(err, Math.hypot(q.x - x, q.y - y));
  });
  check("residual ~0", err < 1e-4, `err=${err.toExponential(2)}`);

  const poly = [project(20, 10), project(70, 10), project(70, 50), project(20, 50)];
  const m = G.measurePolygon(r.H, poly);
  check("area == 2000", m.valid && Math.abs(m.area - 2000) < 1e-4, `area=${m.area}`);
  check("perimeter == 180", m.valid && Math.abs(m.perimeter - 180) < 1e-4, `p=${m.perimeter}`);
}

console.log("== JS invalid inputs ==");
check("duplicate", !G.solveHomography([{x:0,y:0},{x:100,y:0},{x:100,y:0},{x:0,y:60}], W, H).ok);
check("collinear", !G.solveHomography([{x:0,y:0},{x:50,y:0},{x:100,y:0},{x:150,y:0}], W, H).ok);
check("bowtie order", !G.solveHomography([corners[0], corners[2], corners[1], corners[3]], W, H).ok);
check("zero width", !G.solveHomography(corners, 0, H).ok);

const bt = [project(0, 0), project(100, 60), project(100, 0), project(0, 60)];
if (r.ok) {
  const m = G.measurePolygon(r.H, bt);
  check("self-intersect blocked", !m.valid && m.errors[0].includes("自交"), JSON.stringify(m.errors));
}
const si = G.polygonSelfIntersects([{x:0,y:0},{x:10,y:10},{x:10,y:0},{x:0,y:10}]);
check("bowtie detection", si.hit);

console.log("== JS regions with holes ==");
if (r.ok) {
  // outer [10,90] x [10,50] -> area 3200, perim 240; hole [20,30]^2 -> 100, p 40
  const outer = [project(10, 10), project(90, 10), project(90, 50), project(10, 50)];
  const hole = [project(20, 20), project(30, 20), project(30, 30), project(20, 30)];
  let m = G.measureRegion(r.H, outer, [hole]);
  check("region net area 3100", m.valid && Math.abs(m.area - 3100) < 1e-4, `area=${m.area}`);
  check("region perim 280", m.valid && Math.abs(m.perimeter - 280) < 1e-4, `p=${m.perimeter}`);
  check("outer area 3200", m.valid && Math.abs(m.outerArea - 3200) < 1e-4);
  check("hole area 100", m.valid && Math.abs(m.holeAreas[0] - 100) < 1e-4);
  check("world holes", m.valid && m.worldHoles.length === 1);

  const hole2 = [project(40, 20), project(50, 20), project(50, 30), project(40, 30)];
  m = G.measureRegion(r.H, outer, [hole, hole2]);
  check("two holes area 3000", m.valid && Math.abs(m.area - 3000) < 1e-4, `area=${m.area}`);
  check("two holes perim 320", m.valid && Math.abs(m.perimeter - 320) < 1e-4, `p=${m.perimeter}`);

  const tang = [project(20, 5), project(30, 5), project(30, 10), project(20, 10)];
  m = G.measureRegion(r.H, outer, [tang]);
  check("tangent rejected", !m.valid && m.errors[0].includes("相切"), JSON.stringify(m.errors));

  const outside = [project(92, 20), project(98, 20), project(98, 30), project(92, 30)];
  m = G.measureRegion(r.H, outer, [outside]);
  check("outside rejected", !m.valid && m.errors[0].includes("之外"), JSON.stringify(m.errors));

  const a1 = [project(20, 20), project(30, 20), project(30, 30), project(20, 30)];
  const a2 = [project(30, 25), project(40, 25), project(40, 35), project(30, 35)];
  m = G.measureRegion(r.H, outer, [a1, a2]);
  check("holes touch rejected", !m.valid && m.errors[0].includes("孔洞 1 与孔洞 2"), JSON.stringify(m.errors));

  const n1 = [project(15, 15), project(85, 15), project(85, 45), project(15, 45)];
  const n2 = [project(20, 20), project(30, 20), project(30, 30), project(20, 30)];
  m = G.measureRegion(r.H, outer, [n1, n2]);
  check("nested holes rejected", !m.valid && m.errors.join(" ").includes("包含"), JSON.stringify(m.errors));

  const degen = [project(20, 20), project(30, 20), project(40, 20)];
  m = G.measureRegion(r.H, outer, [degen]);
  check("degenerate hole rejected", !m.valid && m.errors[0].includes("面积为 0"), JSON.stringify(m.errors));

  const cross = [project(20, 20), project(40, 40), project(40, 20), project(20, 40)];
  m = G.measureRegion(r.H, outer, [cross]);
  check("self-cross hole rejected", !m.valid && m.errors.some((e) => e.includes("自交")), JSON.stringify(m.errors));
}

// identity-homography parity with the Python checks
const I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
let m = G.measureRegion(I, [{x:0,y:0},{x:10,y:0},{x:10,y:10},{x:0,y:10}],
                        [[{x:2,y:2},{x:4,y:2},{x:4,y:4},{x:2,y:4}]]);
check("identity area/perim", m.valid && m.area === 96 && m.perimeter === 48, JSON.stringify(m));
m = G.measureRegion(I, [{x:0,y:0},{x:10,y:0}], []);
check("outer <3 verts", !m.valid && m.errors[0].includes("外环"));
m = G.measureRegion(I, [{x:0,y:0},{x:10,y:0},{x:10,y:10}], [[{x:1,y:1},{x:2,y:1}]]);
check("hole <3 verts", !m.valid && m.errors.some((e) => e.includes("孔洞 1")), JSON.stringify(m.errors));

console.log(fails ? `\n${fails} FAILURES` : "\nALL JS PARITY TESTS PASSED");
process.exit(fails ? 1 : 0);
