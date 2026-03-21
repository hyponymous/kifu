// photo-pipeline.ts — pure computation functions for the photo→tsumego pipeline
// Extracted from proto-photo.html for reuse and testing.
// All functions take data in and return data out; no DOM or canvas access.
// OpenCV (cv) is assumed to be available as a global.

// ── Types ────────────────────────────────────────────────────────────────────

export interface Point {
  x: number;
  y: number;
}

export interface Circle extends Point {
  r: number;
}

export interface TPSPoint {
  x: number;
  y: number;
  target: number;
}

export interface TPSModel {
  norm: TPSPoint[];
  w: Float64Array | number[];
  a: number[];
  minX: number;
  scX: number;
  minY: number;
  scY: number;
}

export interface HoughLine { rho: number; theta: number; axis: 'row' | 'col' }
export interface Cluster { pos: number; count: number }

export interface Detection {
  rowPos: number[];
  colPos: number[];
  uniformRowPos: number[];
  uniformColPos: number[];
  step: number;
  rawCircles: Circle[];
  lineHPos: number[];
  lineVPos: number[];
  rowAngle: number;
  colAngle: number;
  harrisCorners: Point[];
  intersections: Point[][];
  medRadius?: number | null;
  circleMinR?: number;
  circleMaxR?: number;
  houghLines?: HoughLine[];
  houghRowCentroids?: Cluster[];
  houghColCentroids?: Cluster[];
  _houghEdges?: CvMat;  // debug: masked Canny edges fed to HoughLines
}

export interface ClassResult {
  stones: { r: number; c: number; color: string; mean?: number; rp?: number }[];
  threshold?: number;
}

export interface ElidedEdges {
  top: boolean;
  bottom: boolean;
  left: boolean;
  right: boolean;
}

export interface CylinderModel {
  R: number;
  yc: number;
}

export interface PiecewiseModel {
  pieces: CylinderModel[];
  breakpoints: number[];
  blendW: number;
}

export interface CombinedGrid {
  gridY: TPSPoint[];
  gridX: TPSPoint[];
  nRext: number;
  nCext: number;
  hasBorder: boolean;
}

export type ReadonlyMatrix<T> = readonly (readonly T[])[];

// GridBounds is a CvRect — returned by findGridBounds / cv.boundingRect
type GridBounds = CvRect;

// ── Multi-threshold Canny ───────────────────────────────────────────────────

export interface CannyStack {
  /** Per-pixel support count (CV_8U, 0..N) — how many threshold pairs detected an edge */
  support: CvMat;
  /** Individual binary edge maps per threshold pair */
  layers: CvMat[];
  /** Threshold pairs used */
  thresholds: readonly [number, number][];
}

/**
 * Default threshold pairs for multi-threshold Canny (sensitive → strict).
 * Soft edges (wood grain, noise) appear only at low thresholds; strong edges
 * (grid lines, stone outlines) survive across all levels.
 */
const DEFAULT_CANNY_THRESHOLDS: readonly [number, number][] = [
  [30, 60], [40, 90], [50, 125], [70, 175], [100, 250],
];

/**
 * Compute Canny edge detection at multiple threshold pairs and stack the
 * results into a per-pixel support map. Callers can threshold the support
 * map (e.g. `support >= 3`) to get a consensus edge map that suppresses
 * noise while retaining strong structural edges.
 *
 * The input should already be blurred if the caller needs blur — this
 * function does not apply any blur itself (caller-decides-blur pattern).
 */
function multiThresholdCanny(
  gray: CvMat,
  thresholds: readonly [number, number][] = DEFAULT_CANNY_THRESHOLDS,
): CannyStack {
  const support = cv.Mat.zeros(gray.rows, gray.cols, cv.CV_8U);
  const layers: CvMat[] = [];
  const ones = cv.Mat.ones(gray.rows, gray.cols, cv.CV_8U);

  for (const [lo, hi] of thresholds) {
    const edges = new cv.Mat();
    cv.Canny(gray, edges, lo, hi);
    layers.push(edges);
    // Increment support where this layer has edges:
    // support += (edges > 0) ? 1 : 0
    // edges is 0/255, so threshold to 0/1 first
    const binary = new cv.Mat();
    cv.threshold(edges, binary, 127, 1, cv.THRESH_BINARY);
    cv.add(support, binary, support);
    binary.delete();
  }

  ones.delete();
  return { support, layers, thresholds };
}

/**
 * Threshold a CannyStack's support map: pixels with support >= minSupport
 * become 255, others become 0. Returns a new CV_8U mat (caller must delete).
 */
function thresholdCannyStack(stack: CannyStack, minSupport: number): CvMat {
  const out = new cv.Mat();
  cv.threshold(stack.support, out, minSupport - 0.5, 255, cv.THRESH_BINARY);
  return out;
}

/**
 * Delete all mats owned by a CannyStack.
 */
function deleteCannyStack(stack: CannyStack): void {
  stack.support.delete();
  for (const layer of stack.layers) layer.delete();
}

// ── Input classification ─────────────────────────────────────────────────────

export type InputType = 'diagram' | 'photo';

const EDGE_DENSITY_THRESHOLD = 0.235;

/**
 * Classify whether the input image is a diagram (book page, app screenshot,
 * digital render) or a real-world board photo.
 *
 * Uses Canny edge density on a small downsampled version of the image.
 * Real board photos have dense edges from wood grain texture; diagrams —
 * even colored app screenshots with rendered wood — have smooth fills
 * between grid lines.
 *
 * Tested on 35 images: all diagrams ≤ 0.20, all photos ≥ 0.27.
 * Threshold set at 0.235 (midpoint of the gap).
 *
 * Possible future improvements if the gap narrows:
 * - Combine with median saturation (good for grayscale diagrams)
 * - Local variance in small patches (grain vs flat fill)
 * - Bayesian combination of multiple signals
 */
function classifyInputType(grayMat: CvMat): InputType {
  const targetSize = 200;
  const scale = targetSize / Math.max(grayMat.cols, grayMat.rows);
  const small = new cv.Mat();
  cv.resize(grayMat, small, new cv.Size(0, 0), scale, scale);

  const edges = new cv.Mat();
  cv.Canny(small, edges, 50, 125);

  const data = edges.data;
  let edgePixels = 0;
  for (let i = 0; i < data.length; i++) {
    if (data[i] > 0) edgePixels++;
  }
  const density = edgePixels / (small.cols * small.rows);

  edges.delete();
  small.delete();

  return density >= EDGE_DENSITY_THRESHOLD ? 'photo' : 'diagram';
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function median(arr: readonly number[]) {
  if (!arr.length) return 0;
  const s = arr.slice().sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
}

// ── Corner ordering & expansion ─────────────────────────────────────────────

function orderCorners(quad: CvMat) {
  const pts = Array.from({ length: 4 }, (_, i) => ({
    x: quad.data32S[i * 2],
    y: quad.data32S[i * 2 + 1],
  }));
  pts.sort((a, b) => (a.x + a.y) - (b.x + b.y));
  const [tl, , , br] = pts;
  const mid = [pts[1], pts[2]].sort((a, b) => (a.x - a.y) - (b.x - b.y));
  return [tl, mid[1], br, mid[0]]; // TL TR BR BL
}

function expandCorners(corners: readonly Point[], pixels: number, imgW: number, imgH: number) {
  const cx = corners.reduce((s, p) => s + p.x, 0) / 4;
  const cy = corners.reduce((s, p) => s + p.y, 0) / 4;
  return corners.map(p => {
    const dx = p.x - cx, dy = p.y - cy;
    const len = Math.sqrt(dx * dx + dy * dy) || 1;
    return {
      x: Math.max(0, Math.min(imgW - 1, p.x + (dx / len) * pixels)),
      y: Math.max(0, Math.min(imgH - 1, p.y + (dy / len) * pixels)),
    };
  });
}

// ── Hough-based quad refinement ─────────────────────────────────────────────

function refineQuadWithHough(corners: readonly Point[], edges: CvMat, rejectBadRefinement: boolean = true) {
  const xs = corners.map(c => c.x), ys = corners.map(c => c.y);
  const pad = 10;
  const x0 = Math.max(0, Math.min(...xs) - pad);
  const y0 = Math.max(0, Math.min(...ys) - pad);
  const x1 = Math.min(edges.cols, Math.max(...xs) + pad);
  const y1 = Math.min(edges.rows, Math.max(...ys) + pad);
  const cropW = x1 - x0, cropH = y1 - y0;
  if (cropW < 10 || cropH < 10) return corners;

  const roi = edges.roi(new cv.Rect(x0, y0, cropW, cropH));
  const local = corners.map(c => ({ x: c.x - x0, y: c.y - y0 }));

  const threshold = Math.round(Math.min(cropW, cropH) * 0.15);
  const lines = new cv.Mat();
  cv.HoughLines(roi, lines, 1, Math.PI / 180, threshold);
  roi.delete();

  if (lines.rows === 0) { lines.delete(); return corners; }

  const edgePairs = [[0,1],[1,2],[2,3],[3,0]];
  const edgeNames = ['top','right','bottom','left'];
  const maxDist = 0.15 * Math.max(cropW, cropH);

  function perpDist(px: number, py: number, rho: number, theta: number) {
    return Math.abs(px * Math.cos(theta) + py * Math.sin(theta) - rho);
  }

  function segAngle(p1: Point, p2: Point) {
    let a = Math.atan2(p2.y - p1.y, p2.x - p1.x);
    while (a < 0) a += Math.PI;
    while (a >= Math.PI) a -= Math.PI;
    return a;
  }

  function angleDiff(a: number, b: number) {
    let d = Math.abs(a - b) % Math.PI;
    return d > Math.PI / 2 ? Math.PI - d : d;
  }

  const matchedLines: ({ rho: number; theta: number } | null)[] = [];
  for (let ei = 0; ei < 4; ei++) {
    const [i0, i1] = edgePairs[ei];
    const p1 = local[i0], p2 = local[i1];
    const midX = (p1.x + p2.x) / 2, midY = (p1.y + p2.y) / 2;
    const edgeAngle = segAngle(p1, p2);

    let bestDist = Infinity, bestLine = null;
    for (let li = 0; li < lines.rows; li++) {
      const rho   = lines.data32F[li * 2];
      const theta = lines.data32F[li * 2 + 1];
      let lineAngle = theta - Math.PI / 2;
      while (lineAngle < 0) lineAngle += Math.PI;
      while (lineAngle >= Math.PI) lineAngle -= Math.PI;

      if (angleDiff(lineAngle, edgeAngle) > 15 * Math.PI / 180) continue;

      const dist = perpDist(midX, midY, rho, theta);
      if (dist < maxDist && dist < bestDist) {
        bestDist = dist;
        bestLine = { rho, theta };
      }
    }
    matchedLines.push(bestLine);
  }
  lines.delete();

  if (matchedLines.some(l => l === null)) {
    const missing = edgeNames.filter((_, i) => !matchedLines[i]);
    console.log(`[quad refine] no Hough match for: ${missing.join(', ')}; keeping original`);
    return corners;
  }

  const edgeForCorner = [[3,0],[0,1],[1,2],[2,3]];

  function intersectHoughLines(l1: { rho: number; theta: number }, l2: { rho: number; theta: number }) {
    const c1 = Math.cos(l1.theta), s1 = Math.sin(l1.theta);
    const c2 = Math.cos(l2.theta), s2 = Math.sin(l2.theta);
    const det = c1 * s2 - c2 * s1;
    if (Math.abs(det) < 1e-6) return null;
    return {
      x: (l1.rho * s2 - l2.rho * s1) / det,
      y: (c1 * l2.rho - c2 * l1.rho) / det,
    };
  }

  const refined: Point[] = [];
  for (let ci = 0; ci < 4; ci++) {
    const [e1, e2] = edgeForCorner[ci];
    const pt = intersectHoughLines(matchedLines[e1]!, matchedLines[e2]!);
    if (!pt) {
      console.log(`[quad refine] degenerate intersection at ${['TL','TR','BR','BL'][ci]}; keeping original`);
      return corners;
    }
    refined.push({ x: pt.x + x0, y: pt.y + y0 });
  }

  const names = ['TL','TR','BR','BL'];
  const parts = names.map((n, i) =>
    `${n} (${corners[i].x.toFixed(0)},${corners[i].y.toFixed(0)})→(${refined[i].x.toFixed(0)},${refined[i].y.toFixed(0)})`
  );
  console.log(`[quad refine] ${parts.join(' ')}`);

  // Reject refinement if it makes opposite sides significantly more unequal.
  // This prevents snapping to interior board lines when the actual border edge
  // is broken (e.g. stones occlude part of the border, dropping it below the
  // Hough threshold).
  function sideLengths(pts: readonly Point[]) {
    const [TL, TR, BR, BL] = pts;
    return {
      left:   Math.hypot(BL.x - TL.x, BL.y - TL.y),
      right:  Math.hypot(BR.x - TR.x, BR.y - TR.y),
      top:    Math.hypot(TR.x - TL.x, TR.y - TL.y),
      bottom: Math.hypot(BR.x - BL.x, BR.y - BL.y),
    };
  }
  function oppRatio(a: number, b: number) { return Math.min(a, b) / Math.max(a, b); }
  const orig = sideLengths(corners), ref = sideLengths(refined);
  const lrOrig = oppRatio(orig.left, orig.right), lrRef = oppRatio(ref.left, ref.right);
  const tbOrig = oppRatio(orig.top,  orig.bottom), tbRef = oppRatio(ref.top,  ref.bottom);
  if (rejectBadRefinement && (lrRef < lrOrig - 0.05 || tbRef < tbOrig - 0.05)) {
    console.log(`[quad refine] rejected: LR ${lrOrig.toFixed(3)}→${lrRef.toFixed(3)} TB ${tbOrig.toFixed(3)}→${tbRef.toFixed(3)}; keeping original`);
    return corners;
  }

  return refined;
}

// ── Board detection (core, no setStatus) ────────────────────────────────────

function findBoardCornersCore(src: CvMat, edges: CvMat, hintN: number, refineQuadFn: (corners: readonly Point[], edges: CvMat, rejectBadRefinement?: boolean) => readonly Point[] = refineQuadWithHough) {
  const imgArea = src.rows * src.cols;
  const dilated   = new cv.Mat();
  const kernel    = cv.Mat.ones(3, 3, cv.CV_8U);
  const contours  = new cv.MatVector();
  const hierarchy = new cv.Mat();

  cv.dilate(edges, dilated, kernel);
  cv.findContours(dilated, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);
  dilated.delete(); kernel.delete(); hierarchy.delete();

  let bestArea  = 0;
  let bestQuad  = null;

  for (let i = 0; i < contours.size(); i++) {
    const cnt  = contours.get(i);
    const area = cv.contourArea(cnt);
    if (area < imgArea * 0.04 || area > imgArea * 0.98) continue;

    const approx = new cv.Mat();
    const peri   = cv.arcLength(cnt, true);
    cv.approxPolyDP(cnt, approx, 0.02 * peri, true);

    if (approx.rows === 4 && area > bestArea) {
      const r  = cv.boundingRect(approx);
      const ar = r.width / r.height;
      if (ar > 0.35 && ar < 2.85) {
        bestArea = area;
        if (bestQuad) bestQuad.delete();
        bestQuad = approx.clone();
      }
    }
    approx.delete();
  }
  contours.delete();

  if (bestQuad) {
    const ordered = orderCorners(bestQuad);
    bestQuad.delete();
    const refined = refineQuadFn(ordered, edges);
    const refinedMat = cv.matFromArray(4, 1, cv.CV_32SC2,
      refined.flatMap(p => [Math.round(p.x), Math.round(p.y)]));
    const qr    = cv.boundingRect(refinedMat);
    refinedMat.delete();
    const quadW = (qr.width + qr.height) / 2;
    const refN      = hintN > 0 ? hintN : 19;
    const autoExpand = 2.0 * quadW / (refN - 1);
    const boardPct = Math.round(bestArea / imgArea * 100);
    const corners = expandCorners(refined, autoExpand, src.cols, src.rows);
    return { corners, boardPct };
  }
  return null;
}

// ── Grid fitting helpers ────────────────────────────────────────────────────

function clusterPositions(positions: readonly number[], tol: number = 8) {
  if (positions.length === 0) return [];
  const sorted = positions.slice().sort((a, b) => a - b);
  const merged = [];
  let groupSum = sorted[0], groupCount = 1;
  for (let i = 1; i < sorted.length; i++) {
    const groupMean = groupSum / groupCount;
    if (sorted[i] - groupMean <= tol) {
      groupSum += sorted[i];
      groupCount++;
    } else {
      merged.push({ pos: groupSum / groupCount, count: groupCount });
      groupSum = sorted[i];
      groupCount = 1;
    }
  }
  merged.push({ pos: groupSum / groupCount, count: groupCount });
  return merged;
}

function medianStep(positions: readonly number[]) {
  const clusters = clusterPositions(positions);
  if (clusters.length < 2) return null;
  const diffs = [];
  for (let i = 1; i < clusters.length; i++) diffs.push(clusters[i].pos - clusters[i - 1].pos);
  return median(diffs);
}

function nearestNeighborStep(points: readonly Point[]) {
  if (points.length < 2) return null;
  const nnDists = [];
  for (let i = 0; i < points.length; i++) {
    let minD = Infinity;
    for (let j = 0; j < points.length; j++) {
      if (i === j) continue;
      const d = Math.hypot(points[i].x - points[j].x, points[i].y - points[j].y);
      if (d < minD) minD = d;
    }
    nnDists.push(minD);
  }
  nnDists.sort((a, b) => a - b);
  return nnDists[Math.floor(nnDists.length / 2)];
}

function gridAlignmentScore(projections: readonly number[], step: number) {
  if (projections.length === 0 || step <= 0) return 1;
  let sumCos = 0, sumSin = 0;
  for (const p of projections) {
    const phase = ((p % step) + step) % step;
    const angle = (phase / step) * 2 * Math.PI;
    sumCos += Math.cos(angle);
    sumSin += Math.sin(angle);
  }
  const R = Math.hypot(sumCos, sumSin) / projections.length;
  return 1 - R;
}

function findGridAngle(points: readonly Point[], step: number, cx: number, cy: number) {
  function rotateAndScore(pts: readonly Point[], angleDeg: number) {
    const rad = angleDeg * Math.PI / 180;
    const cos = Math.cos(rad), sin = Math.sin(rad);
    const xs = [], ys = [];
    for (const p of pts) {
      const dx = p.x - cx, dy = p.y - cy;
      xs.push(cx + dx * cos - dy * sin);
      ys.push(cy + dx * sin + dy * cos);
    }
    return { score: gridAlignmentScore(xs, step) + gridAlignmentScore(ys, step), xs, ys };
  }

  let bestAngle = 0, bestScore = Infinity, bestXs: number[] = [], bestYs: number[] = [];
  for (let a = -15; a <= 15; a += 0.5) {
    const { score } = rotateAndScore(points, a);
    if (score < bestScore) { bestScore = score; bestAngle = a; }
  }

  for (let a = bestAngle - 0.5; a <= bestAngle + 0.5; a += 0.05) {
    const { score, xs, ys } = rotateAndScore(points, a);
    if (score < bestScore) { bestScore = score; bestAngle = a; bestXs = xs; bestYs = ys; }
  }

  if (bestXs.length === 0) {
    const result = rotateAndScore(points, bestAngle);
    bestXs = result.xs;
    bestYs = result.ys;
  }

  console.log(`[findGridAngle] angle=${bestAngle.toFixed(2)}° score=${bestScore.toFixed(4)}`);
  return { angle: bestAngle, rotatedXs: bestXs, rotatedYs: bestYs };
}

// ── Weighted least-squares quadratic fit ─────────────────────────────────────

function fitWeightedQuadratic(xs: readonly number[], ys: readonly number[], ws: readonly number[]) {
  const n = xs.length;
  if (n === 0) return [0, 0, 0];
  if (n === 1) return [ys[0], 0, 0];
  if (n === 2) {
    const b = (ys[1] - ys[0]) / (xs[1] - xs[0] || 1);
    return [ys[0] - b * xs[0], b, 0];
  }
  let sw = 0, sx = 0, sx2 = 0, sx3 = 0, sx4 = 0;
  let sy = 0, sxy = 0, sx2y = 0;
  for (let i = 0; i < n; i++) {
    const w = ws[i], x = xs[i], x2 = x * x, y = ys[i];
    sw   += w;
    sx   += w * x;
    sx2  += w * x2;
    sx3  += w * x * x2;
    sx4  += w * x2 * x2;
    sy   += w * y;
    sxy  += w * x * y;
    sx2y += w * x2 * y;
  }
  const A = [
    [sw,  sx,  sx2, sy],
    [sx,  sx2, sx3, sxy],
    [sx2, sx3, sx4, sx2y],
  ];
  for (let col = 0; col < 3; col++) {
    let maxRow = col;
    for (let row = col + 1; row < 3; row++)
      if (Math.abs(A[row][col]) > Math.abs(A[maxRow][col])) maxRow = row;
    [A[col], A[maxRow]] = [A[maxRow], A[col]];
    if (Math.abs(A[col][col]) < 1e-12) continue;
    for (let row = col + 1; row < 3; row++) {
      const f = A[row][col] / A[col][col];
      for (let k = col; k <= 3; k++) A[row][k] -= f * A[col][k];
    }
  }
  const c = [0, 0, 0];
  for (let row = 2; row >= 0; row--) {
    if (Math.abs(A[row][row]) < 1e-12) continue;
    c[row] = A[row][3];
    for (let k = row + 1; k < 3; k++) c[row] -= A[row][k] * c[k];
    c[row] /= A[row][row];
  }
  return c;
}

// ── fitGrid ─────────────────────────────────────────────────────────────────

function fitGrid(positions: readonly number[], hintN: number, stepHint: number | null = null, label: string = '', clusterTol: number = 8) {
  if (positions.length < 2) return null;

  const clusters = clusterPositions(positions, clusterTol);
  if (clusters.length < 2) return null;

  const allPos = clusters.map(c => c.pos);

  let refStep = stepHint;
  if (!refStep) {
    const diffs = [];
    for (let i = 1; i < allPos.length; i++) diffs.push(allPos[i] - allPos[i - 1]);
    refStep = median(diffs);
  }
  if (!refStep || refStep < 4) return null;

  const medCount = median(clusters.map(c => c.count));
  const countThr = Math.max(3, medCount * 0.3);
  let confident = clusters.filter(c => c.count >= countThr);
  if (confident.length < 2) return null;

  // Filter half-step tick mark artifacts.  If the median gap between
  // confident clusters is much less than refStep, tick marks from elided
  // edges are interleaving with real grid lines.  Classify each cluster
  // by its phase (mod refStep) and keep only the majority phase.
  if (refStep && confident.length > 4) {
    const gaps = [];
    for (let i = 1; i < confident.length; i++)
      gaps.push(confident[i].pos - confident[i - 1].pos);
    const medGap = median(gaps);
    if (medGap < refStep * 0.65) {
      const anchor = confident.reduce((a, b) => a.count > b.count ? a : b).pos;
      const phaseOf = (c: { pos: number; count: number }) => {
        const frac = (c.pos - anchor) / refStep;
        return Math.abs(frac - Math.round(frac)) < 0.25 ? 0 : 1;
      };
      let w0 = 0, w1 = 0;
      for (const c of confident) {
        if (phaseOf(c) === 0) w0 += c.count; else w1 += c.count;
      }
      const keepPhase = w0 >= w1 ? 0 : 1;
      const filtered = confident.filter(c => phaseOf(c) === keepPhase);
      if (filtered.length >= 2) {
        if (label) console.log(`[fitGrid ${label}] tick filter: ${confident.length}→${filtered.length} clusters (medGap=${medGap.toFixed(1)} refStep=${refStep.toFixed(1)})`);
        confident = filtered;
      }
    }
  }

  const inliers = [{ idx: 0, pos: confident[0].pos, count: confident[0].count }];
  const recentSteps = [];
  for (let i = 1; i < confident.length; i++) {
    const diff = confident[i].pos - confident[i - 1].pos;
    const localStep = (recentSteps.length >= 2 && !stepHint) ? median(recentSteps) : refStep;
    // Skip clusters too close to the previous — likely tick mark artifacts
    if (stepHint && diff < refStep * 0.6) continue;
    const nSteps = Math.max(1, Math.round(diff / localStep));
    const prevIdx = inliers[inliers.length - 1].idx;
    inliers.push({ idx: prevIdx + nSteps, pos: confident[i].pos, count: confident[i].count });
    if (nSteps === 1 && diff > localStep * 0.5 && diff < localStep * 1.5) {
      recentSteps.push(diff);
    }
  }

  if (inliers.length < 2) return null;

  const MODEL_TOL = 0.35;
  let fitPoints = inliers.map(p => ({ idx: p.idx, pos: p.pos, weight: p.count }));
  // All three are set on iter 0 and remain set; ! below is safe (loop invariant).
  let qCoeffs: number[] | undefined, evalModel: ((idx: number) => number) | undefined, binned: Map<number, { totalPos: number; totalWeight: number }> | undefined;

  for (let iter = 0; iter < 4; iter++) {
    qCoeffs = fitWeightedQuadratic(
      fitPoints.map(p => p.idx),
      fitPoints.map(p => p.pos),
      fitPoints.map(p => p.weight),
    );
    evalModel = (idx: number) => qCoeffs![0] + qCoeffs![1] * idx + qCoeffs![2] * idx * idx;

    binned = new Map();
    for (const pos of positions) {
      const guess = Math.round((pos - qCoeffs[0]) / Math.max(qCoeffs[1], refStep * 0.5));
      let bestIdx = guess, bestDist = Infinity;
      for (let tryIdx = guess - 3; tryIdx <= guess + 3; tryIdx++) {
        const dist = Math.abs(pos - evalModel(tryIdx)) / refStep;
        if (dist < bestDist) { bestDist = dist; bestIdx = tryIdx; }
      }
      if (bestDist < MODEL_TOL) {
        const prev = binned.get(bestIdx) || { totalPos: 0, totalWeight: 0 };
        prev.totalPos += pos;
        prev.totalWeight += 1;
        binned.set(bestIdx, prev);
      }
    }

    const newFitPoints = [];
    for (const [idx, b] of binned) {
      newFitPoints.push({ idx, pos: b.totalPos / b.totalWeight, weight: b.totalWeight });
    }
    newFitPoints.sort((a, b) => a.idx - b.idx);

    const prevIdxSet = new Set(fitPoints.map(p => p.idx));
    const newIdxSet = new Set(newFitPoints.map(p => p.idx));
    const converged = newIdxSet.size === prevIdxSet.size &&
      [...newIdxSet].every(idx => prevIdxSet.has(idx));

    fitPoints = newFitPoints;
    if (converged) break;
  }

  const populatedIndices = [...binned!.keys()].sort((a, b) => a - b); // safe: loop ran ≥1 iter
  if (populatedIndices.length < 2) return null;

  let bestRunStart = 0, bestRunLen = 1;
  let runStart = 0;
  for (let i = 1; i < populatedIndices.length; i++) {
    const gap = populatedIndices[i] - populatedIndices[i - 1];
    if (gap <= 2) {
      const runLen = i - runStart + 1;
      if (runLen > bestRunLen) { bestRunLen = runLen; bestRunStart = runStart; }
    } else {
      runStart = i;
    }
  }
  const lastRunLen = populatedIndices.length - runStart;
  if (lastRunLen > bestRunLen) { bestRunLen = lastRunLen; bestRunStart = runStart; }

  const idxFirst = populatedIndices[bestRunStart];
  const idxLast  = populatedIndices[bestRunStart + bestRunLen - 1];
  const effectiveN = idxLast - idxFirst + 1;

  const chainFirst = evalModel!(idxFirst); // safe: loop invariant (set on iter 0)
  const chainLast  = evalModel!(idxLast);
  const avgStep = effectiveN > 1 ? (chainLast - chainFirst) / (effectiveN - 1) : refStep;

  const n = hintN > 0 ? hintN : effectiveN;
  if (n < 2) return null;

  let startIdx;
  if (n <= effectiveN) {
    const skip = Math.floor((effectiveN - n) / 2);
    startIdx = idxFirst + skip;
  } else {
    const extra = n - effectiveN;
    startIdx = idxFirst - Math.floor(extra / 2);
  }
  const uniform = Array.from({ length: n }, (_, i) => evalModel!(startIdx + i)); // safe: loop invariant

  const snapped = uniform.map((expected, i) => {
    const idx = startIdx + i;
    const b = binned!.get(idx);
    return b ? b.totalPos / b.totalWeight : expected;
  });

  if (label) {
    const binnedSummary = populatedIndices.slice(bestRunStart, bestRunStart + bestRunLen)
      .map(idx => {
        const b = binned!.get(idx);
        return `${idx}:${b ? b.totalWeight : 0}`;
      });
    console.log(`[fitGrid ${label}] ${positions.length} votes → ${clusters.length} clusters → ${inliers.length} inliers → ${populatedIndices.length} model bins`);
    console.log(`[fitGrid ${label}] refStep=${refStep.toFixed(1)} model=[${qCoeffs!.map(c => c.toFixed(4))}]`); // safe: loop invariant
    console.log(`[fitGrid ${label}] run=[${idxFirst}..${idxLast}] effectiveN=${effectiveN} bins=${JSON.stringify(binnedSummary)} n=${n}`);
  }

  const binCounts = uniform.map((_, i) => {
    const b = binned!.get(startIdx + i);
    return b ? b.totalWeight : 0;
  });

  return { uniform, snapped, avgStep, chainFirst, chainLast, evalModel: evalModel!, startIdx, binCounts }; // safe: loop invariant
}

// ── Build 2D intersection grid ──────────────────────────────────────────────

function buildIntersections(rowPos: number[], colPos: number[], rowAngle: number, colAngle: number, W: number, H: number) {
  const rA = rowAngle * Math.PI / 180;
  const cA = colAngle * Math.PI / 180;
  const sinR = Math.sin(rA), cosR = Math.cos(rA);
  const sinC = Math.sin(cA), cosC = Math.cos(cA);
  const cx = W / 2, cy = H / 2;
  // Each intersection is where row-line r meets col-line c:
  //   row r: points where rotating by rowAngle gives Y = rowPos[r]
  //   col c: points where rotating by colAngle gives X = colPos[c]
  const det = Math.cos(rA - cA);
  return rowPos.map(rp =>
    colPos.map(cp => {
      const R = rp - cy;
      const C = cp - cx;
      return {
        x: cx + (R * sinC + C * cosR) / det,
        y: cy + (cosC * R - sinR * C) / det,
      };
    })
  );
}

// ── Find tight board bounds via edge contours ───────────────────────────────

function findGridBounds(grayMat: CvMat, cannyLo: number = 50, cannyHi: number = 125) {
  const W = grayMat.cols, H = grayMat.rows;
  const cx = W / 2, cy = H / 2;

  const edges = new cv.Mat();
  cv.Canny(grayMat, edges, cannyLo, cannyHi);

  // Small dilation to reconnect edge junctions suppressed by Canny's
  // non-maximum suppression (perpendicular grid lines lose connectivity
  // at intersections).  1px expansion won't bridge the gap to nearby text.
  const kernel = cv.Mat.ones(3, 3, cv.CV_8U);
  cv.dilate(edges, edges, kernel);
  kernel.delete();

  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();
  cv.findContours(edges, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
  edges.delete(); hierarchy.delete();

  let bestRect = null, bestArea = 0;

  // Prefer the largest contour whose bounding rect contains the image center
  for (let i = 0; i < contours.size(); i++) {
    const r = cv.boundingRect(contours.get(i));
    const area = r.width * r.height;
    if (area <= bestArea) continue;
    if (r.x <= cx && r.y <= cy &&
        r.x + r.width >= cx && r.y + r.height >= cy) {
      bestArea = area;
      bestRect = r;
    }
  }

  // Fallback: largest overall
  if (!bestRect) {
    for (let i = 0; i < contours.size(); i++) {
      const r = cv.boundingRect(contours.get(i));
      const area = r.width * r.height;
      if (area > bestArea) {
        bestArea = area;
        bestRect = r;
      }
    }
  }
  contours.delete();

  if (!bestRect) return new cv.Rect(0, 0, W, H);

  console.log(`[findGridBounds] ${W}×${H} → ${bestRect.width}×${bestRect.height} at (${bestRect.x},${bestRect.y})`);
  return bestRect;
}

// ── Grid feature extraction & model fitting ─────────────────────────────────
//
// detectGrid is split into two phases:
//   1. extractGridFeatures — Harris corners, HoughCircles, HoughLines, angle
//      estimation, step hint.  Pure feature extraction, no model decisions.
//   2. fitGridFromFeatures — vote pools, fitGrid, Harris refinement, edge
//      trimming, intersection construction.  All model-fitting logic.
//
// detectGrid itself is a thin wrapper that calls both in sequence.

export interface GridFeatures {
  W: number;
  H: number;
  estStep: number;

  // Harris corners (bounds-filtered)
  harrisCorners: Point[];
  nnStep: number | null;

  // HoughCircles
  rawCircles: Circle[];
  medRadius: number | null;
  radiusStep: number | null;
  bestSweepStep: number;

  // HoughLines (raw rho/theta pairs, pre-classification)
  linesMat: CvMat;          // caller must delete
  houghThr: number;
  houghEdgesClone: CvMat;   // debug visualization; caller must delete

  // Angle estimation
  rowAngle: number;
  colAngle: number;
  // Harris points rotated by each axis angle (parallel arrays with harrisCorners)
  rowRotXs: number[];
  rowRotYs: number[];
  colRotXs: number[];
  colRotYs: number[];

  // Step hint (derived from radiusStep / estStep / nnStep)
  stepHint: number;
  stepHintSrc: string;

  // Multi-threshold Canny stack (present when multiCannyEnabled)
  cannyStack?: CannyStack;
}

function extractGridFeatures(
  grayMat: CvMat, hintN: number, circleSens: number,
  gridBounds: GridBounds | null | undefined,
  houghBlurSize: number | undefined,
  useCirclesForAngle: boolean | undefined,
  multiCannyEnabled?: boolean,
  minCannySupport?: number,
): GridFeatures {
  const W = grayMat.cols, H = grayMat.rows;
  const refN = hintN > 0 ? hintN : 19;
  const estStep = W / (refN + 1);

  const inBounds = gridBounds
    ? (x: number, y: number) => x >= gridBounds.x && x <= gridBounds.x + gridBounds.width &&
                 y >= gridBounds.y && y <= gridBounds.y + gridBounds.height
    : () => true;

  // ── 1. Harris corners ──────────────────────────────────────────────────────
  const harrisMinDist = Math.max(5, W / 60);
  const cornersMat = new cv.Mat();
  const harrisMask = new cv.Mat();
  cv.goodFeaturesToTrack(grayMat, cornersMat, 500, 0.01, harrisMinDist,
                         harrisMask, 3, true, 0.04);
  harrisMask.delete();
  const harrisCorners: Point[] = [];
  for (let i = 0; i < cornersMat.rows; i++) {
    harrisCorners.push({ x: cornersMat.floatAt(i, 0), y: cornersMat.floatAt(i, 1) });
  }
  cornersMat.delete();

  if (gridBounds) {
    const before = harrisCorners.length;
    for (let i = harrisCorners.length - 1; i >= 0; i--) {
      if (!inBounds(harrisCorners[i].x, harrisCorners[i].y)) harrisCorners.splice(i, 1);
    }
    console.log(`[detectGrid] harrisCorners=${before} → ${harrisCorners.length} after bounds filter`);
  } else {
    console.log(`[detectGrid] harrisCorners=${harrisCorners.length}`);
  }

  const nnStep = nearestNeighborStep(harrisCorners);

  // ── 2. HoughCircles — sweep candidate steps ───────────────────────────────
  const STONE_R_FRAC = 0.5;
  const blurred2 = new cv.Mat();
  cv.GaussianBlur(grayMat, blurred2, new cv.Size(5, 5), 1.5);

  const sweepLo = W / 30, sweepHi = W / 8;
  const nCandidates = 6;
  let bestCircles: Circle[] = [], bestSweepStep = estStep;
  const targetMaxR = 25;

  for (let ci = 0; ci < nCandidates; ci++) {
    const candStep = sweepLo + (sweepHi - sweepLo) * ci / (nCandidates - 1);
    const candMaxR = candStep * 0.60;
    const candScale = candMaxR > targetMaxR ? targetMaxR / candMaxR : 1;
    const candMinR = Math.max(3, Math.floor(candStep * 0.35 * candScale));
    const candMaxRScaled = Math.max(candMinR + 1, Math.floor(candMaxR * candScale));
    const candMinDist = candStep * 0.6 * candScale;

    let circleInput;
    if (candScale < 1) {
      circleInput = new cv.Mat();
      cv.resize(blurred2, circleInput, new cv.Size(
        Math.round(W * candScale), Math.round(H * candScale)));
    } else {
      circleInput = blurred2.clone();
    }

    const circlesMat = new cv.Mat();
    cv.HoughCircles(circleInput, circlesMat, cv.HOUGH_GRADIENT, 1, candMinDist,
                    100, circleSens, candMinR, candMaxRScaled);
    circleInput.delete();

    const circles: Circle[] = [];
    for (let i = 0; i < circlesMat.cols; i++) {
      const x = circlesMat.data32F[i * 3] / candScale;
      const y = circlesMat.data32F[i * 3 + 1] / candScale;
      const r = circlesMat.data32F[i * 3 + 2] / candScale;
      if (x >= 0 && x <= W && y >= 0 && y <= H && inBounds(x, y)) circles.push({ x, y, r });
    }
    circlesMat.delete();

    if (circles.length > bestCircles.length) {
      bestCircles = circles;
      bestSweepStep = candStep;
    }
  }
  blurred2.delete();

  const rawCircles = bestCircles;
  const sortedR = rawCircles.map(c => c.r).sort((a, b) => a - b);
  const medRadius = sortedR.length ? sortedR[Math.floor(sortedR.length / 2)] : null;
  const radiusStep = medRadius ? medRadius / STONE_R_FRAC : null;
  console.log(`[detectGrid] circleSweep: bestStep=${bestSweepStep.toFixed(1)} circles=${rawCircles.length} medRadius=${medRadius?.toFixed(1) ?? 'null'} radiusStep=${radiusStep?.toFixed(1) ?? 'null'}`);

  // ── 3. HoughLines (with circle masking) ────────────────────────────────────
  const blurMat = new cv.Mat();
  const blurKernel = houghBlurSize ?? 3;
  cv.GaussianBlur(grayMat, blurMat, new cv.Size(blurKernel, blurKernel), 0);

  let edgesMat: CvMat;
  let cannyStack: CannyStack | undefined;
  if (multiCannyEnabled) {
    cannyStack = multiThresholdCanny(blurMat, DEFAULT_CANNY_THRESHOLDS);
    edgesMat = thresholdCannyStack(cannyStack, minCannySupport ?? 3);
  } else {
    edgesMat = new cv.Mat();
    cv.Canny(blurMat, edgesMat, 50, 125);
  }
  blurMat.delete();

  if (gridBounds) {
    const mask = cv.Mat.zeros(H, W, cv.CV_8U);
    const roi = mask.roi(gridBounds);
    roi.setTo(new cv.Scalar(255));
    roi.delete();
    cv.bitwise_and(edgesMat, mask, edgesMat);
    mask.delete();
  }

  if (rawCircles.length > 0) {
    const mask = cv.Mat.zeros(H, W, cv.CV_8U);
    for (const c of rawCircles) {
      cv.circle(mask, new cv.Point(Math.round(c.x), Math.round(c.y)),
                Math.round(c.r * 1.3), new cv.Scalar(255), -1);
    }
    const inv = new cv.Mat();
    cv.bitwise_not(mask, inv);
    cv.bitwise_and(edgesMat, inv, edgesMat);
    mask.delete(); inv.delete();
  }

  const houghEdgesClone = edgesMat.clone();

  const linesMat = new cv.Mat();
  const houghThr = Math.round(Math.min(W, H) * 0.15);
  cv.HoughLines(edgesMat, linesMat, 1, Math.PI / 180, houghThr);
  edgesMat.delete();

  // ── Derive grid angle from Hough line thetas ───────────────────────────────
  const allThetas: number[] = [];
  for (let i = 0; i < linesMat.rows; i++) {
    allThetas.push(linesMat.data32F[i * 2 + 1]);
  }

  const vDeviations: number[] = [], hDeviations: number[] = [];
  for (const theta of allThetas) {
    let dv = theta;
    if (dv > Math.PI / 2) dv = Math.PI - dv;
    const dh = Math.abs(theta - Math.PI / 2);
    if (dv < Math.PI / 4) {
      let dev = theta;
      if (theta > Math.PI / 2) dev = theta - Math.PI;
      vDeviations.push(dev);
    } else if (dh < Math.PI / 4) {
      hDeviations.push(theta - Math.PI / 2);
    }
  }

  const medDev = (arr: number[]) => {
    if (!arr.length) return null;
    const s = arr.slice().sort((a: number, b: number) => a - b);
    return s[Math.floor(s.length / 2)];
  };
  const vMed = medDev(vDeviations);
  const hMed = medDev(hDeviations);
  const houghColAngleDeg = vMed != null ? -vMed * 180 / Math.PI : (hMed != null ? -hMed * 180 / Math.PI : null);
  const houghRowAngleDeg = hMed != null ? -hMed * 180 / Math.PI : (vMed != null ? -vMed * 180 / Math.PI : null);
  console.log(`[detectGrid] houghAngle: vLines=${vDeviations.length} hLines=${hDeviations.length} vMed=${vMed?.toFixed(4) ?? 'null'} hMed=${hMed?.toFixed(4) ?? 'null'} colAngle=${houghColAngleDeg?.toFixed(2) ?? 'null'}° rowAngle=${houghRowAngleDeg?.toFixed(2) ?? 'null'}°`);

  // ── 4. Angle sweep — separate row and column angles ────────────────────────
  const cornerStep = radiusStep || estStep;
  const roughStep = nnStep || cornerStep;

  function sweepAxis(seedDeg: number, range: number, step: number, points: readonly Point[], cx: number, cy: number, axis: string) {
    let bestAngle = seedDeg, bestScore = Infinity, bestXs: number[] = [], bestYs: number[] = [];
    for (let a = seedDeg - range; a <= seedDeg + range; a += 0.05) {
      const r = a * Math.PI / 180;
      const cos = Math.cos(r), sin = Math.sin(r);
      const xs: number[] = [], ys: number[] = [];
      for (const p of points) {
        const dx = p.x - cx, dy = p.y - cy;
        xs.push(cx + dx * cos - dy * sin);
        ys.push(cy + dx * sin + dy * cos);
      }
      const score = axis === 'row' ? gridAlignmentScore(ys, step) : gridAlignmentScore(xs, step);
      if (score < bestScore) { bestScore = score; bestAngle = a; bestXs = xs; bestYs = ys; }
    }
    return { angle: bestAngle, score: bestScore, xs: bestXs, ys: bestYs };
  }

  let rowAngle: number, colAngle: number;
  let rowRotXs: number[], rowRotYs: number[];
  let colRotXs: number[], colRotYs: number[];
  if (houghRowAngleDeg != null) {
    const sweepStep = cornerStep || roughStep;
    const sweepPts = useCirclesForAngle && rawCircles.length >= 10
      ? rawCircles.map(c => ({ x: c.x, y: c.y }))
      : harrisCorners;
    const rowResult = sweepAxis(houghRowAngleDeg, 2, sweepStep, sweepPts, W / 2, H / 2, 'row');
    const colResult = sweepAxis(houghColAngleDeg!, 2, sweepStep, sweepPts, W / 2, H / 2, 'col'); // safe: houghColAngleDeg is non-null under the same condition as houghRowAngleDeg (both derive from hMed/vMed)
    rowAngle = rowResult.angle; rowRotXs = rowResult.xs; rowRotYs = rowResult.ys;
    colAngle = colResult.angle; colRotXs = colResult.xs; colRotYs = colResult.ys;
    console.log(`[findGridAngle] hough-seeded rowAngle=${rowAngle.toFixed(2)}° score=${rowResult.score.toFixed(4)} colAngle=${colAngle.toFixed(2)}° score=${colResult.score.toFixed(4)} sweepStep=${sweepStep.toFixed(1)} pts=${sweepPts.length}`);
  } else {
    const sweepStep = roughStep;
    const rowResult = sweepAxis(0, 15, sweepStep, harrisCorners, W / 2, H / 2, 'row');
    const colResult = sweepAxis(0, 15, sweepStep, harrisCorners, W / 2, H / 2, 'col');
    rowAngle = rowResult.angle; rowRotXs = rowResult.xs; rowRotYs = rowResult.ys;
    colAngle = colResult.angle; colRotXs = colResult.xs; colRotYs = colResult.ys;
    console.log(`[findGridAngle] sweep rowAngle=${rowAngle.toFixed(2)}° colAngle=${colAngle.toFixed(2)}°`);
  }

  // ── 5. Compute step hint ───────────────────────────────────────────────────
  const MIN_CIRCLES_RELIABLE = 8;
  const RATIO_MIN = 0.6, RATIO_MAX = 2.5;
  let stepHint: number;
  let stepHintSrc = 'fallback';
  if (radiusStep && rawCircles.length >= MIN_CIRCLES_RELIABLE) {
    const ratio = radiusStep / estStep;
    if (ratio >= RATIO_MIN && ratio <= RATIO_MAX) {
      stepHint = radiusStep;
      stepHintSrc = 'radiusStep';
    } else if (ratio < RATIO_MIN) {
      const doubled = radiusStep * 2;
      const dRatio = doubled / estStep;
      if (dRatio >= RATIO_MIN && dRatio <= RATIO_MAX) {
        stepHint = doubled;
        stepHintSrc = 'radiusStep×2';
      } else {
        stepHint = estStep;
      }
    } else {
      stepHint = estStep;
    }
  } else {
    stepHint = estStep;
  }
  console.log(`[detectGrid] circles=${rawCircles.length} medRadius=${medRadius?.toFixed(1) ?? 'null'} radiusStep=${radiusStep?.toFixed(1) ?? 'null'} nnStep=${nnStep?.toFixed(1) ?? 'null'} stepHint=${stepHint.toFixed(1)} (${stepHintSrc})`);

  return {
    W, H, estStep,
    harrisCorners, nnStep,
    rawCircles, medRadius, radiusStep, bestSweepStep,
    linesMat, houghThr, houghEdgesClone,
    rowAngle, colAngle,
    rowRotXs, rowRotYs, colRotXs, colRotYs,
    stepHint, stepHintSrc,
    cannyStack,
  };
}

function fitGridFromFeatures(
  features: GridFeatures,
  hintN: number,
  { forceRows, forceCols, gridBounds, skipTrimEdges }:
    { forceRows?: number; forceCols?: number; gridBounds?: GridBounds | null; skipTrimEdges?: boolean },
): Detection | null {
  const { W, H, estStep, harrisCorners, rawCircles, medRadius, bestSweepStep,
          linesMat, houghThr, houghEdgesClone,
          rowAngle, colAngle, rowRotXs, rowRotYs, colRotXs, colRotYs,
          stepHint } = features;

  const inBounds = gridBounds
    ? (x: number, y: number) => x >= gridBounds.x && x <= gridBounds.x + gridBounds.width &&
                 y >= gridBounds.y && y <= gridBounds.y + gridBounds.height
    : () => true;

  // ── 6. Build vote pools ────────────────────────────────────────────────────
  const hVotes: number[] = [];
  const vVotes: number[] = [];
  const rowAngleRad = rowAngle * Math.PI / 180;
  const colAngleRad = colAngle * Math.PI / 180;

  const angleTol = 5 * Math.PI / 180;
  let houghH = 0, houghV = 0;
  const rawHoughRows: number[] = [], rawHoughCols: number[] = [];
  const houghRowPositions: number[] = [], houghColPositions: number[] = [];
  const houghLines: HoughLine[] = [];
  for (let i = 0; i < linesMat.rows; i++) {
    const rho   = linesMat.data32F[i * 2];
    const theta = linesMat.data32F[i * 2 + 1];
    const hAngle = Math.PI / 2 - rowAngleRad;
    const vAngle = -colAngleRad;
    if (Math.abs(theta - hAngle) < angleTol || Math.abs(theta - Math.PI - hAngle) < angleTol) {
      const y = Math.abs(Math.sin(theta)) > 0.1 ? rho / Math.sin(theta) : null;
      if (y != null && y >= 0 && y <= H) {
        if (gridBounds && (y < gridBounds.y || y > gridBounds.y + gridBounds.height)) continue;
        rawHoughRows.push(y);
        houghLines.push({ rho, theta, axis: 'row' });
        houghH++;
      }
    } else if (Math.abs(theta - vAngle) < angleTol || Math.abs(theta - Math.PI - vAngle) < angleTol) {
      const x = Math.abs(Math.cos(theta)) > 0.1 ? rho / Math.cos(theta) : null;
      if (x != null && x >= 0 && x <= W) {
        if (gridBounds && (x < gridBounds.x || x > gridBounds.x + gridBounds.width)) continue;
        rawHoughCols.push(x);
        houghLines.push({ rho, theta, axis: 'col' });
        houghV++;
      }
    }
  }
  linesMat.delete();

  const houghClusterTol = Math.max(8, Math.round(stepHint * 0.3));
  const houghRowClusters = clusterPositions(rawHoughRows, houghClusterTol);
  const houghColClusters = clusterPositions(rawHoughCols, houghClusterTol);
  for (const c of houghRowClusters) {
    hVotes.push(c.pos);
    houghRowPositions.push(c.pos);
  }
  for (const c of houghColClusters) {
    vVotes.push(c.pos);
    houghColPositions.push(c.pos);
  }
  console.log(`[detectGrid] houghLines: ${houghH} row + ${houghV} col raw → ${houghRowClusters.length} row + ${houghColClusters.length} col centroids (thr=${houghThr}, clusterTol=${houghClusterTol})`);

  for (const c of rawCircles) {
    const dx = c.x - W / 2, dy = c.y - H / 2;
    const cosC = Math.cos(colAngleRad), sinC = Math.sin(colAngleRad);
    vVotes.push(W / 2 + dx * cosC - dy * sinC);
    const cosR = Math.cos(rowAngleRad), sinR = Math.sin(rowAngleRad);
    hVotes.push(H / 2 + dx * sinR + dy * cosR);
  }

  const filteredCorners = harrisCorners.filter(p =>
    !rawCircles.some(rc => Math.hypot(p.x - rc.x, p.y - rc.y) < rc.r * 1.3)
  );

  // ── 7. fitGrid ─────────────────────────────────────────────────────────────
  const clusterTol = Math.max(8, Math.round(stepHint * 0.2));

  const rowResult = fitGrid(hVotes, forceRows || hintN, stepHint, 'rows', clusterTol);
  if (!rowResult) { houghEdgesClone.delete(); return null; }
  const colResult = fitGrid(vVotes, forceCols || hintN, stepHint, 'cols', clusterTol);
  if (!colResult) { houghEdgesClone.delete(); return null; }

  // ── 8. Harris refinement ───────────────────────────────────────────────────
  const filteredRowRotY: number[] = [], filteredColRotX: number[] = [];
  for (let i = 0; i < harrisCorners.length; i++) {
    const p = harrisCorners[i];
    if (!rawCircles.some(rc => Math.hypot(p.x - rc.x, p.y - rc.y) < rc.r * 1.3)) {
      filteredRowRotY.push(rowRotYs[i]);
      filteredColRotX.push(colRotXs[i]);
    }
  }

  function refineModel(result: { evalModel: (idx: number) => number; startIdx: number; uniform: number[]; snapped: number[]; binCounts: number[] }, harrisCoords: readonly number[]) {
    const { evalModel, startIdx } = result;
    const n = result.uniform.length;
    const bins = new Map<number, { totalPos: number; count: number }>();
    for (const pos of harrisCoords) {
      let bestIdx = 0, bestDist = Infinity;
      for (let idx = startIdx; idx < startIdx + n; idx++) {
        const dist = Math.abs(pos - evalModel(idx)) / stepHint;
        if (dist < bestDist) { bestDist = dist; bestIdx = idx; }
      }
      if (bestDist < 0.3) {
        const prev = bins.get(bestIdx) || { totalPos: 0, count: 0 };
        prev.totalPos += pos;
        prev.count += 1;
        bins.set(bestIdx, prev);
      }
    }
    const fitIdxs: number[] = [], fitPos: number[] = [], fitW: number[] = [];
    for (let i = 0; i < n; i++) {
      const idx = startIdx + i;
      const hb = bins.get(idx);
      const existingW = result.binCounts[i];
      const existingPos = result.snapped[i];
      const harrisW = hb ? hb.count : 0;
      const harrisPos = hb ? hb.totalPos / hb.count : 0;
      const totalW = existingW + harrisW;
      if (totalW > 0) {
        fitIdxs.push(idx);
        fitPos.push((existingPos * existingW + harrisPos * harrisW) / totalW);
        fitW.push(totalW);
      }
    }
    if (fitIdxs.length < 3) return;
    const newCoeffs = fitWeightedQuadratic(fitIdxs, fitPos, fitW);
    const newEval = (idx: number) => newCoeffs[0] + newCoeffs[1] * idx + newCoeffs[2] * idx * idx;
    for (let i = 0; i < n; i++) {
      result.uniform[i] = newEval(startIdx + i);
      const idx = startIdx + i;
      const hb = bins.get(idx);
      const existingW = result.binCounts[i];
      const harrisW = hb ? hb.count : 0;
      if (harrisW > 0) {
        result.snapped[i] = (result.snapped[i] * existingW + (hb!.totalPos / hb!.count) * harrisW) / (existingW + harrisW); // non-null: harrisW > 0 implies hb exists
      }
    }
    result.evalModel = newEval;
  }
  refineModel(rowResult, filteredRowRotY);
  refineModel(colResult, filteredColRotX);

  // ── 9. Per-line quality trimming ───────────────────────────────────────────
  function lineQuality(linePositions: readonly number[], alongCoords: readonly number[], perpCoords: readonly number[], step: number) {
    const tol = step * 0.3;
    return linePositions.map((lp: number) => {
      const along: number[] = [];
      for (let i = 0; i < perpCoords.length; i++) {
        if (Math.abs(perpCoords[i] - lp) < tol) along.push(alongCoords[i]);
      }
      if (along.length < 2) return { n: along.length, medGap: 0 };
      along.sort((a, b) => a - b);
      const gaps: number[] = [];
      for (let i = 1; i < along.length; i++) gaps.push(along[i] - along[i - 1]);
      return { n: along.length, medGap: median(gaps) };
    });
  }

  function trimBadEdges(result: { uniform: number[]; snapped: number[]; startIdx: number; binCounts: number[] }, quality: { n: number; medGap: number }[], step: number, houghPositions: readonly number[], label: string) {
    const hasHough = (pos: number) =>
      houghPositions.some((hp: number) => Math.abs(hp - pos) < step * 0.3);
    const counts = result.binCounts;
    const interiorCounts = counts.slice(1, -1);
    const medVotes = interiorCounts.length > 0 ? median(interiorCounts) : 0;
    const isGood = (q: { n: number; medGap: number }, pos: number, idx: number) => {
      if (hasHough(pos)) return true;
      if (counts[idx] < medVotes * 0.25) return false;
      if (q.n < 3) return false;
      const ratio = q.medGap / step;
      const residual = Math.abs(ratio - Math.round(ratio));
      return residual < 0.3 && Math.round(ratio) >= 1;
    };

    let lo = 0, hi = result.uniform.length - 1;
    while (lo <= hi && !isGood(quality[lo], result.uniform[lo], lo)) lo++;
    while (hi >= lo && !isGood(quality[hi], result.uniform[hi], hi)) hi--;
    const trimmed = hi - lo + 1;

    if (lo > 0 || hi < result.uniform.length - 1) {
      console.log(`[trimEdges ${label}] ${result.uniform.length}→${trimmed} (lo=${lo} hi=${hi})`);
      result.uniform = result.uniform.slice(lo, hi + 1);
      result.snapped = result.snapped.slice(lo, hi + 1);
      result.startIdx += lo;
    }
    return result;
  }

  const rowQ = lineQuality(rowResult.snapped, rowRotXs, rowRotYs, stepHint);
  const colQ = lineQuality(colResult.snapped, colRotYs, colRotXs, stepHint);
  if (!forceRows && !skipTrimEdges) trimBadEdges(rowResult, rowQ, stepHint, houghRowPositions, 'row');
  if (!forceCols && !skipTrimEdges) trimBadEdges(colResult, colQ, stepHint, houghColPositions, 'col');

  // ── 10. Build result ───────────────────────────────────────────────────────
  const uniformRowPos = rowResult.uniform, rowPos = rowResult.snapped;
  const uniformColPos = colResult.uniform, colPos = colResult.snapped;

  const stepX = uniformColPos.length > 1
    ? (uniformColPos[uniformColPos.length - 1] - uniformColPos[0]) / (uniformColPos.length - 1) : estStep;
  const stepY = uniformRowPos.length > 1
    ? (uniformRowPos[uniformRowPos.length - 1] - uniformRowPos[0]) / (uniformRowPos.length - 1) : estStep;
  const step = (stepX + stepY) / 2;

  const colEnd = uniformColPos[uniformColPos.length - 1];
  const rowEnd = uniformRowPos[uniformRowPos.length - 1];
  if (colEnd > W + stepX * 0.5)
    console.warn(`[detectGrid] col overshoot: gridEnd=${colEnd.toFixed(1)} > W=${W} (by ${((colEnd - W) / stepX).toFixed(1)} steps)`);
  if (uniformColPos[0] < -stepX * 0.5)
    console.warn(`[detectGrid] col undershoot: gridStart=${uniformColPos[0].toFixed(1)} < 0 (by ${(-uniformColPos[0] / stepX).toFixed(1)} steps)`);
  if (rowEnd > H + stepY * 0.5)
    console.warn(`[detectGrid] row overshoot: gridEnd=${rowEnd.toFixed(1)} > H=${H} (by ${((rowEnd - H) / stepY).toFixed(1)} steps)`);
  if (uniformRowPos[0] < -stepY * 0.5)
    console.warn(`[detectGrid] row undershoot: gridStart=${uniformRowPos[0].toFixed(1)} < 0 (by ${(-uniformRowPos[0] / stepY).toFixed(1)} steps)`);

  const intersections = buildIntersections(rowPos, colPos, rowAngle, colAngle, W, H);

  return {
    rowPos, colPos, uniformRowPos, uniformColPos, step,
    rawCircles, medRadius,
    circleMinR: Math.max(3, Math.floor(bestSweepStep * 0.35)),
    circleMaxR: Math.max(5, Math.floor(bestSweepStep * 0.60)),
    rowAngle, colAngle, harrisCorners: filteredCorners, houghLines,
    houghRowCentroids: houghRowClusters, houghColCentroids: houghColClusters,
    lineHPos: [], lineVPos: [],
    intersections,
    _houghEdges: houghEdgesClone,
  };
}

function detectGrid(grayMat: CvMat, hintN: number, circleSens: number = 21, { forceRows, forceCols, gridBounds, skipTrimEdges, houghBlurSize, useCirclesForAngle, multiCannyEnabled, minCannySupport }: { forceRows?: number; forceCols?: number; gridBounds?: GridBounds | null; skipTrimEdges?: boolean; houghBlurSize?: number; useCirclesForAngle?: boolean; multiCannyEnabled?: boolean; minCannySupport?: number } = {}): Detection | null {
  const features = extractGridFeatures(grayMat, hintN, circleSens, gridBounds, houghBlurSize, useCirclesForAngle, multiCannyEnabled, minCannySupport);
  const result = fitGridFromFeatures(features, hintN, { forceRows, forceCols, gridBounds, skipTrimEdges });
  // Clean up cannyStack if present — it's not needed after grid fitting
  if (features.cannyStack) deleteCannyStack(features.cannyStack);
  return result;
}

// ── Preprocessing ───────────────────────────────────────────────────────────

function enhanceGray(grayMat: CvMat, usePercentileNorm: boolean = true, opts: { claheEnabled?: boolean } = {}) {
  // CLAHE (Contrast Limited Adaptive Histogram Equalization) works tile-by-tile,
  // avoiding the over-amplification of noise that global equalization causes on
  // real board photos with wood grain. Preferred over percentile norm for photos.
  // cv.createCLAHE is not available in all opencv-wasm builds (e.g. the Node.js
  // test build). Fall back silently to the standard path if it's missing.
  if (opts.claheEnabled && typeof (cv as unknown as Record<string, unknown>).createCLAHE === 'function') {
    const clahe = (cv as unknown as { createCLAHE: (clip: number, size: CvSize) => { apply: (src: CvMat, dst: CvMat) => void; delete: () => void } }).createCLAHE(2.0, new cv.Size(8, 8));
    const out = new cv.Mat();
    clahe.apply(grayMat, out);
    clahe.delete();
    return out;
  }
  if (!usePercentileNorm) {
    const out = new cv.Mat();
    cv.normalize(grayMat, out, 0, 255, cv.NORM_MINMAX, cv.CV_8U);
    return out;
  }
  const d      = grayMat.data;
  const sorted = new Uint8Array(d).sort();
  const lo     = sorted[Math.floor(sorted.length * 0.05)];
  const hi     = sorted[Math.floor(sorted.length * 0.95)];
  const range  = hi - lo || 1;
  const out    = new cv.Mat(grayMat.rows, grayMat.cols, cv.CV_8U);
  for (let i = 0; i < d.length; i++) {
    out.data[i] = Math.max(0, Math.min(255, Math.round((d[i] - lo) / range * 255)));
  }
  return out;
}

// ── Classification helpers ──────────────────────────────────────────────────

/** Grid extent for masking board-edge gradients in radialPower. */
interface GridExtent { left: number; right: number; top: number; bottom: number }

function radialPower(gx: Float32Array, gy: Float32Array, W: number, H: number, cx: number, cy: number, rIn: number, rOut: number, gradFloor: number, sinMask: number = 0, gridExtent: GridExtent | null = null) {
  const ir   = Math.ceil(rOut);
  const in2  = rIn * rIn, out2 = rOut * rOut;
  const sm2  = sinMask * sinMask;
  let radSum = 0, totSum = 0;
  for (let dy = -ir; dy <= ir; dy++) {
    for (let dx = -ir; dx <= ir; dx++) {
      const d2 = dx * dx + dy * dy;
      if (d2 < in2 || d2 > out2) continue;
      if (sm2 > 0) {
        const minA = Math.min(Math.abs(dx), Math.abs(dy));
        if (minA * minA < sm2 * d2) continue;
      }
      const px = cx + dx, py = cy + dy;
      if (px < 0 || px >= W || py < 0 || py >= H) continue;
      // Suppress board-edge gradients: skip ring pixels outside the grid extent
      if (gridExtent && (px < gridExtent.left || px > gridExtent.right || py < gridExtent.top || py > gridExtent.bottom)) continue;
      const idx  = py * W + px;
      const gxv  = gx[idx], gyv = gy[idx];
      const mag2 = gxv * gxv + gyv * gyv;
      if (mag2 < gradFloor) continue;
      const d   = Math.sqrt(d2);
      const dot = gxv * (dx / d) + gyv * (dy / d);
      radSum += dot * dot;
      totSum += mag2;
    }
  }
  return totSum > 0 ? radSum / totSum : 0;
}

function sampleDisc(gray: Uint8Array, W: number, H: number, cx: number, cy: number, r: number) {
  const ir = Math.ceil(r), r2 = r * r;
  let sum = 0, n = 0;
  for (let dy = -ir; dy <= ir; dy++) {
    for (let dx = -ir; dx <= ir; dx++) {
      if (dx * dx + dy * dy > r2) continue;
      const px = cx + dx, py = cy + dy;
      if (px < 0 || px >= W || py < 0 || py >= H) continue;
      sum += gray[py * W + px]; n++;
    }
  }
  return n ? sum / n : 128;
}

function sampleAnnulus(gray: Uint8Array, W: number, H: number, cx: number, cy: number, rIn: number, rOut: number) {
  const ir = Math.ceil(rOut);
  const in2 = rIn * rIn, out2 = rOut * rOut;
  let sum = 0, n = 0;
  for (let dy = -ir; dy <= ir; dy++) {
    for (let dx = -ir; dx <= ir; dx++) {
      const d2 = dx * dx + dy * dy;
      if (d2 < in2 || d2 > out2) continue;
      const px = cx + dx, py = cy + dy;
      if (px < 0 || px >= W || py < 0 || py >= H) continue;
      sum += gray[py * W + px]; n++;
    }
  }
  return n ? sum / n : 128;
}

function kmeans2(values: number[], seedLo: number, seedHi: number) {
  let c0 = seedLo, c1 = seedHi;
  const labels = new Int32Array(values.length);
  for (let iter = 0; iter < 20; iter++) {
    let changed = false;
    for (let i = 0; i < values.length; i++) {
      const lbl = Math.abs(values[i] - c0) <= Math.abs(values[i] - c1) ? 0 : 1;
      if (lbl !== labels[i]) { labels[i] = lbl; changed = true; }
    }
    if (!changed && iter > 0) break;
    let s0 = 0, n0 = 0, s1 = 0, n1 = 0;
    for (let i = 0; i < values.length; i++) {
      if (labels[i] === 0) { s0 += values[i]; n0++; }
      else                  { s1 += values[i]; n1++; }
    }
    const nc0 = n0 ? s0 / n0 : c0, nc1 = n1 ? s1 / n1 : c1;
    if (Math.abs(nc0 - c0) < 0.001 && Math.abs(nc1 - c1) < 0.001 && iter > 0) break;
    c0 = nc0; c1 = nc1;
  }
  return { labels, c0, c1 };
}

// ── Stone classification ────────────────────────────────────────────────────
// Returns { stones, grayCache, dbgInfo }
// useHoughW: whether to promote empty intersections with nearby circle detection to W

function classifyStones(grayMat: CvMat, rowPos: readonly number[], colPos: readonly number[], step: number, rawCircles: Circle[], intersections: ReadonlyMatrix<Point>, useHoughW: boolean = false, { rpThreshRatio = 0.85, gradFloor = 64, inputType = 'diagram' as InputType }: { rpThreshRatio?: number; gradFloor?: number; inputType?: InputType } = {}) {
  const W    = grayMat.cols, H = grayMat.rows;
  const gray = grayMat.data;
  const grayCache = { data: new Uint8Array(gray), W, H };

  const sortedR = rawCircles.map((c: Circle) => c.r).sort((a: number, b: number) => a - b);
  const medR    = sortedR.length ? sortedR[Math.floor(sortedR.length / 2)] : null;
  const stoneR  = medR ?? (step * 0.5);

  const ringIn  = stoneR * 0.85;
  const ringOut = stoneR * 1.05;

  const sobelX = new cv.Mat(), sobelY = new cv.Mat();
  cv.Sobel(grayMat, sobelX, cv.CV_32F, 1, 0, 3);
  cv.Sobel(grayMat, sobelY, cv.CV_32F, 0, 1, 3);
  const gx = sobelX.data32F, gy = sobelY.data32F;


  const sinMask    = stoneR / step;
  // Suppress board-edge gradients for photos: skip ring pixels outside grid extent.
  // The board edge creates a strong gradient that looks radial from corner intersections.
  // Pad by half ringOut: the board edge is roughly a stone radius beyond the
  // outermost grid line, so full ringOut would include it. Half ringOut keeps
  // most of an edge stone's ring while still clipping the board-edge gradient.
  const extentPad = ringOut * 0.5;
  const rpGridExtent: GridExtent | null = inputType === 'photo'
    ? { left: Math.round(colPos[0] - extentPad), right: Math.round(colPos[colPos.length - 1] + extentPad),
        top: Math.round(rowPos[0] - extentPad), bottom: Math.round(rowPos[rowPos.length - 1] + extentPad) }
    : null;
  // Diagrams have numbers/symbols at stone centers — use annulus to avoid them.
  // Real photos have clean stones — use the full disc for a stronger signal.
  const bodyAnnIn  = inputType === 'photo' ? 0 : stoneR * 0.35;
  const bodyAnnOut = stoneR * 0.85;
  const snapTol2 = (step * 0.35) ** 2;
  const snapRlo  = stoneR * 0.75, snapRhi = stoneR * 1.25;

  const pts   = [];
  const rpArr: number[] = [];
  const bdArr: number[] = [];
  for (let r = 0; r < rowPos.length; r++) {
    for (let c = 0; c < colPos.length; c++) {
      let cx = Math.round(intersections[r][c].x);
      let cy = Math.round(intersections[r][c].y);
      let bestD2 = snapTol2;
      for (const circ of rawCircles) {
        if (circ.r < snapRlo || circ.r > snapRhi) continue;
        const d2 = (circ.x - cx) ** 2 + (circ.y - cy) ** 2;
        if (d2 < bestD2) { bestD2 = d2; cx = Math.round(circ.x); cy = Math.round(circ.y); }
      }
      // 2D sweep: try nearby centers to handle off-center stone placement
      // (real boards — stones aren't snapped to exact grid intersections).
      // Sweep radius and step are proportional to the grid step size so the
      // search scales with image resolution / board size.
      const sweepR    = Math.max(1, Math.round(step * 0.1));
      const sweepStep = Math.max(1, Math.round(step * 0.04));
      let bestRP = radialPower(gx, gy, W, H, cx, cy, ringIn, ringOut, gradFloor, sinMask, rpGridExtent);
      let bestCx = cx, bestCy = cy;
      for (let sdy = -sweepR; sdy <= sweepR; sdy += sweepStep) {
        for (let sdx = -sweepR; sdx <= sweepR; sdx += sweepStep) {
          if (sdx === 0 && sdy === 0) continue;
          const rp = radialPower(gx, gy, W, H, cx + sdx, cy + sdy, ringIn, ringOut, gradFloor, sinMask, rpGridExtent);
          if (rp > bestRP) { bestRP = rp; bestCx = cx + sdx; bestCy = cy + sdy; }
        }
      }
      rpArr.push(bestRP);
      bdArr.push(sampleAnnulus(gray, W, H, bestCx, bestCy, bodyAnnIn, bodyAnnOut));
      pts.push({ r, c, cx: bestCx, cy: bestCy });
    }
  }
  sobelX.delete(); sobelY.delete();

  const N = pts.length;
  if (N === 0) return { stones: [], grayCache, dbgInfo: null };

  const minRP = Math.min(...rpArr), maxRP = Math.max(...rpArr);
  const { labels: rpLabels, c0: rpc0, c1: rpc1 } = kmeans2(rpArr, minRP, maxRP);
  const stoneLbl    = rpc1 >= rpc0 ? 1 : 0;
  const stoneRPCent = stoneLbl === 1 ? rpc1 : rpc0;
  const emptyRPCent = stoneLbl === 1 ? rpc0 : rpc1;
  const rpThresh = emptyRPCent + rpThreshRatio * (stoneRPCent - emptyRPCent);

  const stoneIndices = pts.map((_, i) => i).filter(i => rpArr[i] >= rpThresh);
  const stoneBodies  = stoneIndices.map(i => bdArr[i]);
  let bCent = 0, wCent = 255;
  let bwLabels = null;
  if (stoneBodies.length >= 2) {
    const minBd = Math.min(...stoneBodies), maxBd = Math.max(...stoneBodies);
    const { labels: bl, c0: bc, c1: wc } = kmeans2(stoneBodies, minBd, maxBd);
    bCent = bc <= wc ? bc : wc;
    wCent = bc <= wc ? wc : bc;
    const bLabel = bc <= wc ? 0 : 1;
    bwLabels = new Map(stoneIndices.map((gi, si) => [gi, bl[si] === bLabel ? 'B' : 'W']));
  } else if (stoneBodies.length === 1) {
    bwLabels = new Map([[stoneIndices[0], stoneBodies[0] < 128 ? 'B' : 'W']]);
  }

  const dbgInfo = { stoneRPCent, emptyRPCent, rpThresh,
                    bCent, wCent, stoneR, bodyAnnIn, bodyAnnOut, ringIn, ringOut, step };


  const circleNearby = new Set();
  if (useHoughW) {
    for (let i = 0; i < N; i++) {
      const { cx, cy } = pts[i];
      for (const circ of rawCircles) {
        if (circ.r < snapRlo || circ.r > snapRhi) continue;
        if ((circ.x - cx) ** 2 + (circ.y - cy) ** 2 <= snapTol2) {
          circleNearby.add(i); break;
        }
      }
    }
  }

  const stones = [];
  for (let i = 0; i < N; i++) {
    const { r, c, cx, cy } = pts[i];
    const isStone = rpArr[i] >= rpThresh;
    let color = isStone ? (bwLabels?.get(i) ?? '.') : '.';
    if (useHoughW && color === '.' && circleNearby.has(i)) color = 'W';

    stones.push({
      r, c, cx, cy, color,
      _rp:      rpArr[i],
      _body:    bdArr[i],
      _isStone: isStone,
      _circle:  circleNearby.has(i),
    });
  }

  return { stones, grayCache, dbgInfo };
}

// ── Perspective warp ────────────────────────────────────────────────────────

function rectifyBoard(src: CvMat, corners: readonly Point[], outW: number, outH: number) {
  const srcPts = cv.matFromArray(4, 1, cv.CV_32FC2, [
    corners[0].x, corners[0].y,
    corners[1].x, corners[1].y,
    corners[2].x, corners[2].y,
    corners[3].x, corners[3].y,
  ]);
  const dstPts = cv.matFromArray(4, 1, cv.CV_32FC2, [
    0,    0,
    outW, 0,
    outW, outH,
    0,    outH,
  ]);
  const M = cv.getPerspectiveTransform(srcPts, dstPts);
  const out = new cv.Mat();
  cv.warpPerspective(src, out, M, new cv.Size(outW, outH));
  srcPts.delete(); dstPts.delete(); M.delete();
  return out;
}

// ── Stone patch extraction (for digit detection) ───────────────────────────

/**
 * Extract preprocessed grayscale patches from stones for OCR.
 * Returns array of { index, imageData } for stones that have patches.
 * Polarity is normalized so digits are always dark-on-light.
 *
 * @param {cv.Mat} grayMat  - grayscale image of the board
 * @param {Array}  stones   - stone objects from classifyStones
 * @param {number} step     - grid step in pixels
 */
function extractStonePatches(grayMat: CvMat, stones: { r: number; c: number; color: string; cx?: number; cy?: number; _isStone?: boolean; _body?: number }[], step: number, opts: { rowPos?: number[]; colPos?: number[]; intersections?: Point[][]; indices?: number[]; blurRadius?: number } = {}) {
  const patches: { index: number; imageData: { data: Uint8ClampedArray; width: number; height: number } }[] = [];
  if (step < 25) return patches;

  const { indices, blurRadius = 0 } = opts;
  const indexSet = indices ? new Set(indices) : null;

  const patchSide = Math.round(step * 0.7);
  const halfPatch = Math.floor(patchSide / 2);
  const MIN_H = 64; // Tesseract needs ~30px+ to work; upscale small patches
  const W = grayMat.cols, H = grayMat.rows;

  for (let i = 0; i < stones.length; i++) {
    if (indexSet && !indexSet.has(i)) continue;
    const s = stones[i];
    if (!s._isStone && s.color !== 'W' && s.color !== 'B') continue;

    const cx = Math.round(s.cx!), cy = Math.round(s.cy!); // safe: cx/cy are set for all detected stones (caller contract)
    const x0 = cx - halfPatch, y0 = cy - halfPatch;
    const x1 = x0 + patchSide, y1 = y0 + patchSide;
    if (x0 < 0 || y0 < 0 || x1 > W || y1 > H) continue;

    // Extract center patch
    const patch = grayMat.roi(new cv.Rect(x0, y0, patchSide, patchSide));

    // Normalize polarity: if dark stone, invert so digits are dark-on-light
    const normalized = new cv.Mat();
    if (s._body! < 128) { // safe: _body is set for all classified stones (caller contract)
      cv.bitwise_not(patch, normalized);
    } else {
      patch.copyTo(normalized);
    }
    patch.delete();

    // Optional blur before thresholding (for re-detection variants)
    if (blurRadius > 0) {
      const kSize = blurRadius * 2 + 1;
      cv.GaussianBlur(normalized, normalized,
        new cv.Size(kSize, kSize), 0);
    }

    // Upscale small patches so Tesseract has enough pixels to work with
    let forThresh = normalized;
    if (patchSide < MIN_H) {
      forThresh = new cv.Mat();
      cv.resize(normalized, forThresh, new cv.Size(MIN_H, MIN_H), 0, 0, cv.INTER_CUBIC);
      normalized.delete();
    }

    // Otsu threshold — more robust than adaptive for small digit patches
    const binary = new cv.Mat();
    cv.threshold(forThresh, binary, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU);
    forThresh.delete();

    // Convert cv.Mat to ImageData-compatible RGBA for Tesseract
    const w = binary.cols, h = binary.rows;
    const rgba = new Uint8ClampedArray(w * h * 4);
    for (let p = 0; p < w * h; p++) {
      const v = binary.data[p];
      rgba[p * 4]     = v;
      rgba[p * 4 + 1] = v;
      rgba[p * 4 + 2] = v;
      rgba[p * 4 + 3] = 255;
    }
    binary.delete();

    patches.push({ index: i, imageData: { data: rgba, width: w, height: h } });
  }

  return patches;
}

// ── SGF generation ──────────────────────────────────────────────────────────

function inferBoardSize(visibleRows: number, visibleCols: number, elided: ElidedEdges | null) {
  if (!elided) return Math.max(visibleRows, visibleCols);
  const elidedCount = [elided.top, elided.bottom, elided.left, elided.right]
    .filter(Boolean).length;
  if (elidedCount === 0) return Math.max(visibleRows, visibleCols);
  // Corner tsumego (2 adjacent edges elided) → 19×19
  // Side tsumego (3 edges elided) → 19×19
  // Center tsumego (4 edges elided) → 19×19
  if (elidedCount >= 2) return 19;
  // 1 edge elided: the non-elided axis gives us the board size, assume square
  const rowElided = elided.top || elided.bottom;
  const colElided = elided.left || elided.right;
  if (rowElided && !colElided) return visibleCols; // cols span the full board
  if (colElided && !rowElided) return visibleRows; // rows span the full board
  return 19; // shouldn't reach here
}

function computeEdgeOffsets(visibleN: number, boardN: number, elidedLo: boolean, elidedHi: boolean) {
  // Returns the offset to add to the 0-based visible index to get the
  // absolute board coordinate.
  // elidedLo/elidedHi: whether the low (top/left) or high (bottom/right) edge is elided
  if (!elidedLo && !elidedHi) return 0;
  if (!elidedLo) return 0;                    // anchored at low edge
  if (!elidedHi) return boardN - visibleN;    // anchored at high edge
  // Both elided: center the visible region (rare, but handle it)
  return Math.floor((boardN - visibleN) / 2);
}

function generateSGF(stones: { r: number; c: number; color: string }[], nRows: number, elided: ElidedEdges | null, nCols: number, digitMap: Map<number, { number: number; confidence?: number }> | null) {
  const visibleRows = nRows;
  const visibleCols = nCols ?? nRows;
  const boardN = inferBoardSize(visibleRows, visibleCols, elided);
  const rowOff = elided ? computeEdgeOffsets(visibleRows, boardN, elided.top, elided.bottom) : 0;
  const colOff = elided ? computeEdgeOffsets(visibleCols, boardN, elided.left, elided.right) : 0;

  function coordOf(s: { r: number; c: number; color: string }) {
    return String.fromCharCode(97 + s.c + colOff)
         + String.fromCharCode(97 + s.r + rowOff);
  }

  const ab: string[] = [], aw: string[] = [];
  const moves = []; // { number, coord, color }

  for (let i = 0; i < stones.length; i++) {
    const s = stones[i];
    if (s.color === '.') continue;

    const digit = digitMap?.get(i);
    if (digit) {
      // Numbered stone → move node; color determined by parity (odd=B, even=W)
      const color = digit.number % 2 === 1 ? 'B' : 'W';
      moves.push({ number: digit.number, coord: coordOf(s), color });
    } else {
      // Unnumbered stone → setup property
      const coord = coordOf(s);
      (s.color === 'B' ? ab : aw).push(`[${coord}]`);
    }
  }

  let root = `SZ[${boardN}]`;
  if (ab.length) root += `AB${ab.join('')}`;
  if (aw.length) root += `AW${aw.join('')}`;

  // Sort moves by number and append as move nodes
  moves.sort((a, b) => a.number - b.number);
  const moveNodes = moves.map(m => `;${m.color}[${m.coord}]`).join('');

  return `(;${root}${moveNodes})`;
}

// ── Quadratic fit (unweighted) ──────────────────────────────────────────────

function fitQuadratic(xs: readonly number[], ys: readonly number[]) {
  const n = xs.length;
  let s4 = 0, s3 = 0, s2 = 0, s1 = 0;
  let sy2 = 0, sy1 = 0, sy0 = 0;
  for (let i = 0; i < n; i++) {
    const x = xs[i], y = ys[i], x2 = x * x;
    s4 += x2 * x2; s3 += x2 * x; s2 += x2; s1 += x;
    sy2 += x2 * y; sy1 += x * y; sy0 += y;
  }
  const A = [
    [s4, s3, s2, sy2],
    [s3, s2, s1, sy1],
    [s2, s1,  n, sy0],
  ];
  for (let col = 0; col < 3; col++) {
    let maxRow = col;
    for (let row = col + 1; row < 3; row++)
      if (Math.abs(A[row][col]) > Math.abs(A[maxRow][col])) maxRow = row;
    [A[col], A[maxRow]] = [A[maxRow], A[col]];
    if (Math.abs(A[col][col]) < 1e-12) continue;
    for (let row = col + 1; row < 3; row++) {
      const f = A[row][col] / A[col][col];
      for (let k = col; k <= 3; k++) A[row][k] -= f * A[col][k];
    }
  }
  const c = [0, 0, 0];
  for (let row = 2; row >= 0; row--) {
    if (Math.abs(A[row][row]) < 1e-12) continue;
    c[row] = A[row][3];
    for (let k = row + 1; k < 3; k++) c[row] -= A[row][k] * c[k];
    c[row] /= A[row][row];
  }
  return c;
}

function polyEval([a, b, c]: readonly number[], x: number) { return a * x * x + b * x + c; }

// ── preSnapToCircles ────────────────────────────────────────────────────────

function preSnapToCircles(rowPos: readonly number[], colPos: readonly number[], rawCircles: Circle[], intersections: ReadonlyMatrix<Point>, snapFrac: number = 0.45) {
  const nR = rowPos.length, nC = colPos.length;
  const stepY = nR > 1 ? (rowPos[nR-1] - rowPos[0]) / (nR - 1) : 50;
  const stepX = nC > 1 ? (colPos[nC-1] - colPos[0]) / (nC - 1) : 50;
  const snapTol = Math.min(stepX, stepY) * snapFrac;

  const medR = median(rawCircles.map(c => c.r));
  const goodCircles = medR > 0
    ? rawCircles.filter(c => c.r > medR * 0.6 && c.r < medR * 1.4)
    : [];

  let snappedCount = 0;
  const snappedIntersections = intersections.map((row, r) =>
    row.map((pt, c) => {
      const nearby = goodCircles.filter(gc =>
        Math.hypot(gc.x - pt.x, gc.y - pt.y) < snapTol);
      if (nearby.length) {
        snappedCount++;
        return { x: median(nearby.map(gc => gc.x)),
                 y: median(nearby.map(gc => gc.y)) };
      }
      return pt;
    })
  );

  const snappedRows = Array.from({ length: nR }, (_, r) =>
    median(snappedIntersections[r].map(p => p.y)));
  const snappedCols = Array.from({ length: nC }, (_, c) =>
    median(snappedIntersections.map(row => row[c].x)));

  console.log(`[preSnap] goodCircles=${goodCircles.length} snapTol=${snapTol.toFixed(1)} snapped=${snappedCount}/${nR * nC}`);

  return { snappedRows, snappedCols, snappedIntersections };
}

// ── collectOffsetSamples ────────────────────────────────────────────────────

function collectOffsetSamples(grayMat: CvMat, rowPos: readonly number[], colPos: readonly number[], cannyLo: number, cannyHi: number, rawCircles: Circle[], intersections: ReadonlyMatrix<Point>, uniformCoords: { rowY: number[]; colX: number[] } | null = null) {
  const nR = rowPos.length, nC = colPos.length;
  const W = grayMat.cols, H = grayMat.rows;
  const stepX = nC > 1 ? (colPos[nC-1] - colPos[0]) / (nC - 1) : W / 20;
  const stepY = nR > 1 ? (rowPos[nR-1] - rowPos[0]) / (nR - 1) : H / 20;
  const tol = Math.min(stepX, stepY) * 0.4;

  const medR = median(rawCircles.map(c => c.r));
  const goodCircles = medR > 0
    ? rawCircles.filter(c => c.r > medR * 0.6 && c.r < medR * 1.4)
    : [];

  const corners = new cv.Mat();
  const offsetMask = new cv.Mat();
  cv.goodFeaturesToTrack(grayMat, corners, 500, 0.01, Math.min(stepX, stepY) * 0.3,
                         offsetMask, 3, true, 0.04);
  offsetMask.delete();
  const harrisCorners: Point[] = [];
  for (let i = 0; i < corners.rows; i++) {
    const cx = corners.floatAt(i, 0), cy = corners.floatAt(i, 1);
    if (!goodCircles.some(gc => Math.hypot(cx - gc.x, cy - gc.y) < gc.r + 2)) {
      harrisCorners.push({ x: cx, y: cy });
    }
  }
  corners.delete();

  const halfStep = Math.min(stepX, stepY) * 0.5;
  const circleAt = new Map();
  for (const c of goodCircles) {
    let bestI = -1, bestJ = -1, bestDist = halfStep;
    for (let i = 0; i < nR; i++) {
      for (let j = 0; j < nC; j++) {
        const d = Math.hypot(c.x - intersections[i][j].x, c.y - intersections[i][j].y);
        if (d < bestDist) { bestDist = d; bestI = i; bestJ = j; }
      }
    }
    if (bestI >= 0) {
      const key = bestI + ',' + bestJ;
      if (!circleAt.has(key) || bestDist < Math.hypot(circleAt.get(key).x - intersections[bestI][bestJ].x, circleAt.get(key).y - intersections[bestI][bestJ].y)) {
        circleAt.set(key, c);
      }
    }
  }

  function nearestCorner(ey: number, ex: number) {
    let best = null, bestDist = tol;
    for (const c of harrisCorners) {
      const d = Math.hypot(c.x - ex, c.y - ey);
      if (d < bestDist) { bestDist = d; best = c; }
    }
    return best;
  }

  function detectHorizAt(ey: number, px: number) {
    const hw = Math.max(3, Math.round(stepX * 0.2));
    const hh = Math.round(stepY * 0.5);
    const x0 = Math.max(0, Math.round(px - hw));
    const y0 = Math.max(0, Math.round(ey - hh));
    const x1 = Math.min(W, Math.round(px + hw));
    const y1 = Math.min(H, Math.round(ey + hh));
    const sw = x1 - x0, sh = y1 - y0;
    if (sw < 3 || sh < 4) return null;

    const roi   = grayMat.roi(new cv.Rect(x0, y0, sw, sh));
    const edges = new cv.Mat();
    cv.Canny(roi, edges, cannyLo, cannyHi);
    roi.delete();

    let bestRow = -1, bestSum = 0;
    for (let r = 0; r < sh; r++) {
      let s = 0;
      const ay = y0 + r;
      for (let c = 0; c < sw; c++) {
        if (!edges.ucharAt(r, c)) continue;
        const ax = x0 + c;
        if (goodCircles.some(gc => Math.hypot(ax - gc.x, ay - gc.y) < gc.r)) continue;
        s += 255;
      }
      if (s > bestSum) { bestSum = s; bestRow = r; }
    }
    edges.delete();

    if (bestRow < 0 || bestSum < Math.max(1, sw * 0.1)) return null;
    const yDet = y0 + bestRow;
    return Math.abs(yDet - ey) < tol ? yDet - ey : null;
  }

  function detectVertAt(ex: number, py: number) {
    const hw = Math.round(stepX * 0.5);
    const hh = Math.max(3, Math.round(stepY * 0.2));
    const x0 = Math.max(0, Math.round(ex - hw));
    const y0 = Math.max(0, Math.round(py - hh));
    const x1 = Math.min(W, Math.round(ex + hw));
    const y1 = Math.min(H, Math.round(py + hh));
    const sw = x1 - x0, sh = y1 - y0;
    if (sw < 4 || sh < 3) return null;

    const roi   = grayMat.roi(new cv.Rect(x0, y0, sw, sh));
    const edges = new cv.Mat();
    cv.Canny(roi, edges, cannyLo, cannyHi);
    roi.delete();

    let bestCol = -1, bestSum = 0;
    for (let c = 0; c < sw; c++) {
      let s = 0;
      const ax = x0 + c;
      for (let r = 0; r < sh; r++) {
        if (!edges.ucharAt(r, c)) continue;
        const ay = y0 + r;
        if (goodCircles.some(gc => Math.hypot(ax - gc.x, ay - gc.y) < gc.r)) continue;
        s += 255;
      }
      if (s > bestSum) { bestSum = s; bestCol = c; }
    }
    edges.delete();

    if (bestCol < 0 || bestSum < Math.max(1, sh * 0.1)) return null;
    const xDet = x0 + bestCol;
    return Math.abs(xDet - ex) < tol ? xDet - ex : null;
  }

  const yDeltas    = Array.from({ length: nR }, (): number[] => []);
  const xDeltas    = Array.from({ length: nC }, (): number[] => []);
  const samplePoints  = [];
  const tpsYPoints = [];
  const tpsXPoints = [];

  for (let i = 0; i < nR; i++) {
    for (let j = 0; j < nC; j++) {
      const ey = intersections[i][j].y, ex = intersections[i][j].x;
      const circ = circleAt.get(i + ',' + j);
      const nomX = uniformCoords ? uniformCoords.colX[j] : ex;
      const nomY = uniformCoords ? uniformCoords.rowY[i] : ey;
      if (circ) {
        yDeltas[i].push(circ.y - ey);
        xDeltas[j].push(circ.x - ex);
        samplePoints.push({ x: circ.x, y: circ.y, source: 'circle' });
        tpsYPoints.push({ x: nomX, y: nomY, target: circ.y });
        tpsXPoints.push({ x: nomX, y: nomY, target: circ.x });
      } else {
        const corner = nearestCorner(ey, ex);
        if (corner) {
          yDeltas[i].push(corner.y - ey);
          xDeltas[j].push(corner.x - ex);
          samplePoints.push({ x: corner.x, y: corner.y, source: 'corner' });
          tpsYPoints.push({ x: nomX, y: nomY, target: corner.y });
          tpsXPoints.push({ x: nomX, y: nomY, target: corner.x });
        } else {
          const dy = (j === 0 || j === nC - 1) ? null : detectHorizAt(ey, ex);
          const dx = (i === 0 || i === nR - 1) ? null : detectVertAt(ex, ey);
          if (dy !== null) {
            yDeltas[i].push(dy);
            samplePoints.push({ x: ex, y: ey + dy, source: 'line-y' });
            tpsYPoints.push({ x: nomX, y: nomY, target: ey + dy });
          }
          if (dx !== null) {
            xDeltas[j].push(dx);
            samplePoints.push({ x: ex + dx, y: ey, source: 'line-x' });
            tpsXPoints.push({ x: nomX, y: nomY, target: ex + dx });
          }
        }
      }
    }
  }

  const circleCount = samplePoints.filter(p => p.source === 'circle').length;
  const cornerCount = samplePoints.filter(p => p.source === 'corner').length;
  const cannyCount = samplePoints.filter(p => p.source === 'line-y' || p.source === 'line-x').length;
  console.log(`[samples] harrisCorners=${harrisCorners.length} sources: circle=${circleCount} corner=${cornerCount} canny=${cannyCount}`);

  const yXs = [], yYs = [], xXs = [], xYs = [];
  for (let i = 0; i < nR; i++)
    if (yDeltas[i].length >= 2) { yXs.push(rowPos[i]); yYs.push(median(yDeltas[i])); }
  for (let j = 0; j < nC; j++)
    if (xDeltas[j].length >= 2) { xXs.push(colPos[j]); xYs.push(median(xDeltas[j])); }

  return { yXs, yYs, xXs, xYs, samplePoints, tpsYPoints, tpsXPoints };
}

// ── Separable quadratic fit ─────────────────────────────────────────────────

function fitSeparableQuadratic(yXs: readonly number[], yYs: readonly number[], xXs: readonly number[], xYs: readonly number[]) {
  return { yCoeffs: fitQuadratic(yXs, yYs), xCoeffs: fitQuadratic(xXs, xYs) };
}

// ── localGridOffsets ────────────────────────────────────────────────────────

function localGridOffsets(grayMat: CvMat, rowPos: readonly number[], colPos: readonly number[], cannyLo: number, cannyHi: number, lineThresh: number, rawCircles: Circle[]) {
  const TILES = 5;
  const nR = rowPos.length, nC = colPos.length;
  const W = grayMat.cols, H = grayMat.rows;

  const stepX = nC > 1 ? (colPos[nC-1] - colPos[0]) / (nC - 1) : W / 20;
  const stepY = nR > 1 ? (rowPos[nR-1] - rowPos[0]) / (nR - 1) : H / 20;
  const step  = (stepX + stepY) / 2;

  const ctrlRow = Array.from({ length: TILES }, (_, i) =>
    rowPos[0] + i * (rowPos[nR-1] - rowPos[0]) / (TILES - 1));
  const ctrlCol = Array.from({ length: TILES }, (_, i) =>
    colPos[0] + i * (colPos[nC-1] - colPos[0]) / (TILES - 1));

  const medR = median(rawCircles.map(c => c.r));
  const goodCircles = medR > 0
    ? rawCircles.filter(c => c.r > medR * 0.6 && c.r < medR * 1.4)
    : [];

  const tileHW  = Math.round(step * 2.5);
  const tileThr = Math.max(8, Math.round(lineThresh * (2 * tileHW) / W));

  const actualY: number[][] = [], actualX: number[][] = [];

  for (let ti = 0; ti < TILES; ti++) {
    actualY.push([]);
    actualX.push([]);

    for (let tj = 0; tj < TILES; tj++) {
      const iy = ctrlRow[ti], ix = ctrlCol[tj];
      const tx0 = Math.max(0, Math.round(ix - tileHW));
      const ty0 = Math.max(0, Math.round(iy - tileHW));
      const tx1 = Math.min(W, Math.round(ix + tileHW));
      const ty1 = Math.min(H, Math.round(iy + tileHW));
      const tw = tx1 - tx0, th = ty1 - ty0;

      if (tw < 20 || th < 20) {
        actualY[ti].push(iy); actualX[ti].push(ix); continue;
      }

      const tile    = grayMat.roi(new cv.Rect(tx0, ty0, tw, th));
      const blurred = new cv.Mat();
      const edges   = new cv.Mat();
      const lines   = new cv.Mat();
      cv.GaussianBlur(tile, blurred, new cv.Size(3, 3), 0);
      cv.Canny(blurred, edges, cannyLo, cannyHi);
      cv.HoughLines(edges, lines, 1, Math.PI / 180, tileThr);
      tile.delete(); blurred.delete(); edges.delete();

      const hDetected = [], vDetected = [];
      for (let i = 0; i < lines.rows; i++) {
        const rho   = lines.data32F[i * 2];
        const theta = lines.data32F[i * 2 + 1];
        if (Math.abs(Math.sin(theta)) > 0.95)      hDetected.push(ty0 + Math.abs(rho));
        else if (Math.abs(Math.cos(theta)) > 0.95) vDetected.push(tx0 + Math.abs(rho));
      }
      lines.delete();

      for (const c of goodCircles) {
        if (c.x >= tx0 && c.x <= tx1 && c.y >= ty0 && c.y <= ty1) {
          hDetected.push(c.y);
          vDetected.push(c.x);
        }
      }

      const hOffsets = [];
      for (const ey of rowPos.filter(y => y >= ty0 && y <= ty1)) {
        if (!hDetected.length) continue;
        const nearest = hDetected.reduce((b, y) =>
          Math.abs(y - ey) < Math.abs(b - ey) ? y : b, hDetected[0]);
        if (Math.abs(nearest - ey) < step * 0.4) hOffsets.push(nearest - ey);
      }
      const vOffsets = [];
      for (const ex of colPos.filter(x => x >= tx0 && x <= tx1)) {
        if (!vDetected.length) continue;
        const nearest = vDetected.reduce((b, x) =>
          Math.abs(x - ex) < Math.abs(b - ex) ? x : b, vDetected[0]);
        if (Math.abs(nearest - ex) < step * 0.4) vOffsets.push(nearest - ex);
      }

      actualY[ti].push(iy + median(hOffsets));
      actualX[ti].push(ix + median(vOffsets));
    }
  }

  return { actualY, actualX, ctrlRow, ctrlCol };
}

// ── RANSAC filters ──────────────────────────────────────────────────────────

function ransacFilter1D(xs: readonly number[], ys: readonly number[], coeffs: readonly number[], madMultiplier: number = 2.5) {
  const n = xs.length;
  if (n < 4) return { xs: Array.from(xs), ys: Array.from(ys) };
  const residuals = Array.from(xs, (x, i) => Math.abs(ys[i] - polyEval(coeffs, x)));
  const mad = median(residuals);
  if (mad < 0.5) return { xs: Array.from(xs), ys: Array.from(ys) };
  const filtXs = [], filtYs = [];
  for (let i = 0; i < n; i++)
    if (residuals[i] <= madMultiplier * mad) { filtXs.push(xs[i]); filtYs.push(ys[i]); }
  return { xs: filtXs, ys: filtYs };
}

function ransacFilter(pts: readonly TPSPoint[], predictFn: (pt: TPSPoint) => number, madMultiplier: number = 2.5) {
  if (pts.length < 4) return pts;
  const residuals = pts.map(pt => Math.abs(pt.target - predictFn(pt)));
  const mad = median(residuals);
  if (mad < 0.5) return pts;
  return pts.filter((_, i) => residuals[i] <= madMultiplier * mad);
}

// ── Piecewise cylinder dewarping ────────────────────────────────────────────

function evalCylinderDisp(R: number, yc: number, y: number) {
  return yc + R * Math.sin((y - yc) / R) - y;
}

function fitCylinder1D(xs: readonly number[], ys: readonly number[]) {
  const n = xs.length;
  if (n < 2) return { R: 1e6, yc: 0 };
  let yc = (xs[0] + xs[n - 1]) / 2;
  const halfSpan = (xs[n - 1] - xs[0]) / 2;
  const maxDisp = Math.max(...ys.map(Math.abs));
  let R = (maxDisp < 0.5 || halfSpan < 1)
    ? 1e6
    : Math.max(50, halfSpan * Math.sqrt(halfSpan / (6 * maxDisp)));
  for (let iter = 0; iter < 50; iter++) {
    let j00 = 0, j01 = 0, j11 = 0, r0 = 0, r1 = 0;
    for (let i = 0; i < n; i++) {
      const u = (xs[i] - yc) / R;
      const sinU = Math.sin(u), cosU = Math.cos(u);
      const res = yc + R * sinU - xs[i] - ys[i];
      const dR = sinU - u * cosU;
      const dYc = 1 - cosU;
      j00 += dR * dR; j01 += dR * dYc; j11 += dYc * dYc;
      r0 += dR * res; r1 += dYc * res;
    }
    const det = j00 * j11 - j01 * j01;
    if (Math.abs(det) < 1e-20) break;
    R   -= (j11 * r0 - j01 * r1) / det;
    yc  -= (j00 * r1 - j01 * r0) / det;
    if (!isFinite(R) || Math.abs(R) < 10) { R = 1e6; break; }
  }
  return { R: isFinite(R) ? R : 1e6, yc: isFinite(yc) ? yc : 0 };
}

function fitPiecewiseCylinders(xs: readonly number[], ys: readonly number[]) {
  const n = xs.length;
  if (n < 2) return { pieces: [{ R: 1e6, yc: 0 }], breakpoints: [], blendW: 1 };

  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => xs[a] - xs[b]);
  const sxs = order.map(i => xs[i]);
  const sys = order.map(i => ys[i]);

  const dataRange = sxs[n - 1] - sxs[0];
  const blendW = Math.max(5, dataRange * 0.03);

  const steps = Math.min(n - 1, 10);
  const cands = [];
  for (let i = 1; i < steps; i++)
    cands.push(sxs[Math.floor(i * n / steps)]);

  function segFit(lo: number, hi: number) {
    const segXs = sxs.slice(lo, hi), segYs = sys.slice(lo, hi);
    const cyl = fitCylinder1D(segXs, segYs);
    let ssr = 0;
    for (let i = 0; i < segXs.length; i++) {
      const r = evalCylinderDisp(cyl.R, cyl.yc, segXs[i]) - segYs[i];
      ssr += r * r;
    }
    return { cyl, ssr };
  }

  function evalBreaks(breaks: number[]) {
    const bounds = [0];
    for (const b of breaks) {
      const idx = sxs.findIndex(x => x >= b);
      bounds.push(idx < 0 ? n : idx);
    }
    bounds.push(n);
    const pieces = [];
    let totalSSR = 0;
    for (let k = 0; k < bounds.length - 1; k++) {
      const lo = bounds[k], hi = bounds[k + 1];
      if (hi - lo < 2) return null;
      const { cyl, ssr } = segFit(lo, hi);
      pieces.push(cyl);
      totalSSR += ssr;
    }
    return { pieces, ssr: totalSSR };
  }

  function bic(ssr: number, numPieces: number) {
    if (ssr <= 0) return -Infinity;
    const numParams = 2 * numPieces + (numPieces - 1);
    return n * Math.log(ssr / n) + numParams * Math.log(n);
  }

  let bestBIC = Infinity, bestResult = null;

  const res1 = evalBreaks([]);
  if (res1) {
    const b = bic(res1.ssr, 1);
    if (b < bestBIC) { bestBIC = b; bestResult = { ...res1, breakpoints: [] }; }
  }

  if (n >= 4) {
    for (const c of cands) {
      const res = evalBreaks([c]);
      if (!res) continue;
      const b = bic(res.ssr, 2);
      if (b < bestBIC) { bestBIC = b; bestResult = { ...res, breakpoints: [c] }; }
    }
  }

  if (n >= 6) {
    for (let a = 0; a < cands.length - 1; a++) {
      for (let b2 = a + 1; b2 < cands.length; b2++) {
        const res = evalBreaks([cands[a], cands[b2]]);
        if (!res) continue;
        const b = bic(res.ssr, 3);
        if (b < bestBIC) { bestBIC = b; bestResult = { ...res, breakpoints: [cands[a], cands[b2]] }; }
      }
    }
  }

  const result = {
    pieces: bestResult?.pieces ?? [{ R: 1e6, yc: 0 }],
    breakpoints: bestResult?.breakpoints ?? [],
    blendW,
  };
  console.log(`[piecewise] k=${result.pieces.length} breakpoints=${JSON.stringify(result.breakpoints)}`);
  return result;
}

function evalPiecewiseCylinders({ pieces, breakpoints, blendW }: PiecewiseModel, y: number) {
  let disp = evalCylinderDisp(pieces[0].R, pieces[0].yc, y);
  for (let j = 0; j < breakpoints.length; j++) {
    const t0 = (y - breakpoints[j] + blendW) / (2 * blendW);
    const t = Math.max(0, Math.min(1, t0));
    const ts = t * t * (3 - 2 * t);
    const nextDisp = evalCylinderDisp(pieces[j + 1].R, pieces[j + 1].yc, y);
    disp = disp + ts * (nextDisp - disp);
  }
  return disp;
}

function invertPiecewiseCylinder(model: PiecewiseModel, src: number) {
  let dst = src;
  for (let i = 0; i < 10; i++) {
    const f = dst + evalPiecewiseCylinders(model, dst) - src;
    if (Math.abs(f) < 0.01) break;
    const h = 0.5;
    const fp = 1 + (evalPiecewiseCylinders(model, dst + h) - evalPiecewiseCylinders(model, dst - h)) / (2 * h);
    if (Math.abs(fp) < 1e-10) break;
    dst -= f / fp;
  }
  return dst;
}

function mapPositionsThroughDewarp(positions: readonly number[], model: PiecewiseModel) {
  return positions.map(p => invertPiecewiseCylinder(model, p));
}

// ── TPS (Thin-plate spline) ─────────────────────────────────────────────────

const TPS_LAMBDA = 0.001;

function tpsKernel(r2: number) {
  return r2 < 1e-10 ? 0 : 0.5 * r2 * Math.log(r2);
}

function solveLinear(A: ArrayLike<number>[], b: ArrayLike<number>) {
  const N = b.length;
  const a = A.map(row => Float64Array.from(row));
  const rhs = Float64Array.from(b);
  for (let col = 0; col < N; col++) {
    let maxRow = col;
    for (let row = col + 1; row < N; row++)
      if (Math.abs(a[row][col]) > Math.abs(a[maxRow][col])) maxRow = row;
    [a[col], a[maxRow]] = [a[maxRow], a[col]];
    const tmp = rhs[col]; rhs[col] = rhs[maxRow]; rhs[maxRow] = tmp;
    if (Math.abs(a[col][col]) < 1e-12) return null;
    for (let row = col + 1; row < N; row++) {
      const f = a[row][col] / a[col][col];
      for (let k = col; k < N; k++) a[row][k] -= f * a[col][k];
      rhs[row] -= f * rhs[col];
    }
  }
  const sol = new Float64Array(N);
  for (let row = N - 1; row >= 0; row--) {
    sol[row] = rhs[row];
    for (let k = row + 1; k < N; k++) sol[row] -= a[row][k] * sol[k];
    sol[row] /= a[row][row];
  }
  return sol;
}

function fitTPS(pts: readonly TPSPoint[], lambda: number = TPS_LAMBDA) {
  const N = pts.length;
  if (N < 3) return null;

  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const p of pts) {
    if (p.x < minX) minX = p.x; if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y; if (p.y > maxY) maxY = p.y;
  }
  const scX = maxX - minX || 1, scY = maxY - minY || 1;
  const norm = pts.map(p => ({ x: (p.x - minX) / scX, y: (p.y - minY) / scY, target: p.target }));

  const size = N + 3;
  const A = Array.from({ length: size }, () => new Float64Array(size));
  const b = new Float64Array(size);

  for (let i = 0; i < N; i++) {
    b[i] = norm[i].target;
    A[i][i] = lambda;
    for (let j = 0; j < N; j++) {
      const dx = norm[i].x - norm[j].x, dy = norm[i].y - norm[j].y;
      A[i][j] += tpsKernel(dx * dx + dy * dy);
    }
    A[i][N] = 1; A[i][N+1] = norm[i].x; A[i][N+2] = norm[i].y;
    A[N][i] = 1; A[N+1][i] = norm[i].x; A[N+2][i] = norm[i].y;
  }

  const sol = solveLinear(A, b);
  if (!sol) return null;
  return { norm, w: sol.slice(0, N), a: [sol[N], sol[N+1], sol[N+2]], minX, scX, minY, scY };
}

function evalTPS({ norm, w, a, minX, scX, minY, scY }: TPSModel, x: number, y: number) {
  const xn = (x - minX) / scX, yn = (y - minY) / scY;
  let val = a[0] + a[1] * xn + a[2] * yn;
  for (let i = 0; i < norm.length; i++) {
    const dx = xn - norm[i].x, dy = yn - norm[i].y;
    val += w[i] * tpsKernel(dx * dx + dy * dy);
  }
  return val;
}

function invertTPS(tpsY: TPSModel, tpsX: TPSModel, sx: number, sy: number) {
  let dx = sx, dy = sy;
  const h = 0.5;
  for (let i = 0; i < 15; i++) {
    const fx = evalTPS(tpsX, dx, dy) - sx;
    const fy = evalTPS(tpsY, dx, dy) - sy;
    if (Math.abs(fx) < 0.01 && Math.abs(fy) < 0.01) break;
    const dXdx = (evalTPS(tpsX, dx + h, dy) - evalTPS(tpsX, dx - h, dy)) / (2 * h);
    const dXdy = (evalTPS(tpsX, dx, dy + h) - evalTPS(tpsX, dx, dy - h)) / (2 * h);
    const dYdx = (evalTPS(tpsY, dx + h, dy) - evalTPS(tpsY, dx - h, dy)) / (2 * h);
    const dYdy = (evalTPS(tpsY, dx, dy + h) - evalTPS(tpsY, dx, dy - h)) / (2 * h);
    const det = dXdx * dYdy - dXdy * dYdx;
    if (Math.abs(det) < 1e-10) break;
    dx -= (dYdy * fx - dXdy * fy) / det;
    dy -= (dXdx * fy - dYdx * fx) / det;
  }
  return { dx, dy };
}

// ── Dewarp image functions ──────────────────────────────────────────────────

function dewarpImage(colorMat: CvMat, yCoeffs: readonly number[], xCoeffs: readonly number[]) {
  const W = colorMat.cols, H = colorMat.rows;
  const mapXData = new Float32Array(W * H);
  const mapYData = new Float32Array(W * H);

  for (let y = 0; y < H; y++) {
    const srcY = y + polyEval(yCoeffs, y);
    for (let x = 0; x < W; x++) {
      mapXData[y * W + x] = x + polyEval(xCoeffs, x);
      mapYData[y * W + x] = srcY;
    }
  }

  const map1 = new cv.Mat(H, W, cv.CV_32FC1);
  const map2 = new cv.Mat(H, W, cv.CV_32FC1);
  map1.data32F.set(mapXData);
  map2.data32F.set(mapYData);
  const dewarped = new cv.Mat();
  cv.remap(colorMat, dewarped, map1, map2, cv.INTER_LINEAR, cv.BORDER_REPLICATE);
  map1.delete(); map2.delete();
  return dewarped;
}

function dewarpImageTPS(colorMat: CvMat, tpsY: TPSModel, tpsX: TPSModel, outW: number, outH: number) {
  const GRID = 25;

  const gMapX = new Float32Array(GRID * GRID);
  const gMapY = new Float32Array(GRID * GRID);
  for (let gi = 0; gi < GRID; gi++) {
    for (let gj = 0; gj < GRID; gj++) {
      const x = gj * (outW - 1) / (GRID - 1);
      const y = gi * (outH - 1) / (GRID - 1);
      gMapY[gi * GRID + gj] = evalTPS(tpsY, x, y);
      gMapX[gi * GRID + gj] = evalTPS(tpsX, x, y);
    }
  }

  const mapXData = new Float32Array(outW * outH);
  const mapYData = new Float32Array(outW * outH);

  for (let y = 0; y < outH; y++) {
    for (let x = 0; x < outW; x++) {
      const gf = x * (GRID - 1) / (outW - 1);
      const rf = y * (GRID - 1) / (outH - 1);
      const g0 = Math.max(0, Math.min(GRID - 2, Math.floor(gf)));
      const r0 = Math.max(0, Math.min(GRID - 2, Math.floor(rf)));
      const g1 = g0 + 1, r1 = r0 + 1;
      const dg = gf - g0, dr = rf - r0;
      mapXData[y * outW + x] =
        gMapX[r0*GRID+g0]*(1-dr)*(1-dg) + gMapX[r0*GRID+g1]*(1-dr)*dg +
        gMapX[r1*GRID+g0]*   dr *(1-dg) + gMapX[r1*GRID+g1]*   dr *dg;
      mapYData[y * outW + x] =
        gMapY[r0*GRID+g0]*(1-dr)*(1-dg) + gMapY[r0*GRID+g1]*(1-dr)*dg +
        gMapY[r1*GRID+g0]*   dr *(1-dg) + gMapY[r1*GRID+g1]*   dr *dg;
    }
  }

  const map1 = new cv.Mat(outH, outW, cv.CV_32FC1);
  const map2 = new cv.Mat(outH, outW, cv.CV_32FC1);
  map1.data32F.set(mapXData);
  map2.data32F.set(mapYData);
  const dewarped = new cv.Mat();
  cv.remap(colorMat, dewarped, map1, map2, cv.INTER_LINEAR, cv.BORDER_REPLICATE);
  map1.delete(); map2.delete();
  return dewarped;
}

function dewarpImagePiecewise(colorMat: CvMat, yModel: PiecewiseModel, xModel: PiecewiseModel) {
  const W = colorMat.cols, H = colorMat.rows;
  const mapXData = new Float32Array(W * H);
  const mapYData = new Float32Array(W * H);
  for (let y = 0; y < H; y++) {
    const srcY = y + evalPiecewiseCylinders(yModel, y);
    for (let x = 0; x < W; x++) {
      mapXData[y * W + x] = x + evalPiecewiseCylinders(xModel, x);
      mapYData[y * W + x] = srcY;
    }
  }
  const map1 = new cv.Mat(H, W, cv.CV_32FC1);
  const map2 = new cv.Mat(H, W, cv.CV_32FC1);
  map1.data32F.set(mapXData); map2.data32F.set(mapYData);
  const dewarped = new cv.Mat();
  cv.remap(colorMat, dewarped, map1, map2, cv.INTER_LINEAR, cv.BORDER_REPLICATE);
  map1.delete(); map2.delete();
  return dewarped;
}

function dewarpImageMesh(colorMat: CvMat, actualY: ReadonlyMatrix<number>, actualX: ReadonlyMatrix<number>, ctrlRow: readonly number[], ctrlCol: readonly number[]) {
  const nR = ctrlRow.length, nC = ctrlCol.length;
  if (nR < 2 || nC < 2) return colorMat.clone();

  const W = colorMat.cols, H = colorMat.rows;
  const y0 = ctrlRow[0], stepY = (ctrlRow[nR - 1] - y0) / (nR - 1);
  const x0 = ctrlCol[0], stepX = (ctrlCol[nC - 1] - x0) / (nC - 1);

  const mapXData = new Float32Array(W * H);
  const mapYData = new Float32Array(W * H);

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const cf = (x - x0) / stepX;
      const rf = (y - y0) / stepY;
      const c0 = Math.max(0, Math.min(nC - 2, Math.floor(cf)));
      const r0 = Math.max(0, Math.min(nR - 2, Math.floor(rf)));
      const c1 = c0 + 1, r1 = r0 + 1;
      const dc = cf - c0, dr = rf - r0;
      mapXData[y * W + x] =
        actualX[r0][c0] * (1-dr) * (1-dc) + actualX[r0][c1] * (1-dr) * dc +
        actualX[r1][c0] *    dr  * (1-dc) + actualX[r1][c1] *    dr  * dc;
      mapYData[y * W + x] =
        actualY[r0][c0] * (1-dr) * (1-dc) + actualY[r0][c1] * (1-dr) * dc +
        actualY[r1][c0] *    dr  * (1-dc) + actualY[r1][c1] *    dr  * dc;
    }
  }

  const map1 = new cv.Mat(H, W, cv.CV_32FC1);
  const map2 = new cv.Mat(H, W, cv.CV_32FC1);
  map1.data32F.set(mapXData);
  map2.data32F.set(mapYData);
  const dewarped = new cv.Mat();
  cv.remap(colorMat, dewarped, map1, map2, cv.INTER_LINEAR, cv.BORDER_REPLICATE);
  map1.delete(); map2.delete();
  return dewarped;
}

// ── buildCombinedGridPoints ─────────────────────────────────────────────────

function buildCombinedGridPoints(ptsY: readonly TPSPoint[], ptsX: readonly TPSPoint[], rowPos: readonly number[], colPos: readonly number[], pilotLambda: number, includeBorder: boolean = true, uniRowY: readonly number[] | null = null, uniColX: readonly number[] | null = null) {
  const nR = rowPos.length, nC = colPos.length;
  if (nR < 2 || nC < 2) return null;
  const stepY = (rowPos[nR - 1] - rowPos[0]) / (nR - 1);
  const stepX = (colPos[nC - 1] - colPos[0]) / (nC - 1);
  const uStepY = uniRowY && nR > 1 ? (uniRowY[nR - 1] - uniRowY[0]) / (nR - 1) : stepY;
  const uStepX = uniColX && nC > 1 ? (uniColX[nC - 1] - uniColX[0]) / (nC - 1) : stepX;
  const grpRowY = uniRowY || rowPos;
  const grpColX = uniColX || colPos;

  const rowYMeas = Array.from({ length: nR }, (): number[][] => []);
  for (const pt of ptsY) {
    let bestI = 0, bestD = Infinity;
    for (let i = 0; i < nR; i++) {
      const d = Math.abs(pt.y - grpRowY[i]);
      if (d < bestD) { bestD = d; bestI = i; }
    }
    if (bestD < uStepY * 0.4) rowYMeas[bestI].push([pt.x, pt.target]);
  }

  const colXMeas = Array.from({ length: nC }, (): number[][] => []);
  for (const pt of ptsX) {
    let bestJ = 0, bestD = Infinity;
    for (let j = 0; j < nC; j++) {
      const d = Math.abs(pt.x - grpColX[j]);
      if (d < bestD) { bestD = d; bestJ = j; }
    }
    if (bestD < uStepX * 0.4) colXMeas[bestJ].push([pt.y, pt.target]);
  }

  function linearFit(pairs: number[][]) {
    const n = pairs.length;
    if (n === 0) return null;
    if (n === 1) return () => pairs[0][1];
    let sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (const [x, y] of pairs) { sx += x; sy += y; sxx += x * x; sxy += x * y; }
    const det = n * sxx - sx * sx;
    if (Math.abs(det) < 1e-6) return () => sy / n;
    const b = (n * sxy - sx * sy) / det;
    const a = (sy - b * sx) / n;
    return (x: number) => a + b * x;
  }

  const rowYFns = rowYMeas.map(linearFit);
  const colXFns = colXMeas.map(linearFit);

  const pilotY = fitTPS(ptsY, pilotLambda);
  const pilotX = fitTPS(ptsX, pilotLambda);

  function extrapY(f0: ((x: number) => number) | null, f1: ((x: number) => number) | null, x_nom: number, y_nom: number, nomOffset: number) {
    if (f0 && f1) return 2 * f0(x_nom) - f1(x_nom);
    if (f0) return f0(x_nom) + nomOffset;
    if (pilotY) return evalTPS(pilotY, x_nom, y_nom);
    return y_nom;
  }
  function extrapX(f0: ((x: number) => number) | null, f1: ((x: number) => number) | null, y_nom: number, x_nom: number, nomOffset: number) {
    if (f0 && f1) return 2 * f0(y_nom) - f1(y_nom);
    if (f0) return f0(y_nom) + nomOffset;
    if (pilotX) return evalTPS(pilotX, x_nom, y_nom);
    return x_nom;
  }

  function actualY(rowIdx: number, x_nom: number, y_nom: number) {
    if (rowIdx === -1)   return extrapY(rowYFns[0],      rowYFns[1],      x_nom, y_nom, -uStepY);
    if (rowIdx === nR)   return extrapY(rowYFns[nR - 1], rowYFns[nR - 2], x_nom, y_nom, +uStepY);
    if (rowYFns[rowIdx]) return rowYFns[rowIdx](x_nom);
    if (pilotY)          return evalTPS(pilotY, x_nom, y_nom);
    return y_nom;
  }
  function actualX(colIdx: number, x_nom: number, y_nom: number) {
    if (colIdx === -1)   return extrapX(colXFns[0],      colXFns[1],      y_nom, x_nom, -uStepX);
    if (colIdx === nC)   return extrapX(colXFns[nC - 1], colXFns[nC - 2], y_nom, x_nom, +uStepX);
    if (colXFns[colIdx]) return colXFns[colIdx](y_nom);
    if (pilotX)          return evalTPS(pilotX, x_nom, y_nom);
    return x_nom;
  }

  const ys = includeBorder ? [grpRowY[0] - uStepY, ...grpRowY, grpRowY[nR - 1] + uStepY] : [...grpRowY];
  const xs = includeBorder ? [grpColX[0] - uStepX, ...grpColX, grpColX[nC - 1] + uStepX] : [...grpColX];
  const nRext = ys.length, nCext = xs.length;
  const rOff = includeBorder ? 1 : 0;

  const gridY = [], gridX = [];
  for (let ri = 0; ri < nRext; ri++) {
    const y_nom = ys[ri], rowIdx = ri - rOff;
    for (let ci = 0; ci < nCext; ci++) {
      const x_nom = xs[ci], colIdx = ci - rOff;
      gridY.push({ x: x_nom, y: y_nom, target: actualY(rowIdx, x_nom, y_nom) });
      gridX.push({ x: x_nom, y: y_nom, target: actualX(colIdx, x_nom, y_nom) });
    }
  }
  return { gridY, gridX, nRext, nCext, hasBorder: includeBorder };
}

// ── buildDetectionFromControlPoints ─────────────────────────────────────────

function buildDetectionFromControlPoints(combined: CombinedGrid, tpsY: TPSModel, tpsX: TPSModel, det1: Detection) {
  const { gridY, gridX, nRext, nCext, hasBorder } = combined;
  const rOff = hasBorder ? 1 : 0;
  const nRint = nRext - 2 * rOff, nCint = nCext - 2 * rOff;

  const dyRows = Array.from({ length: nRint }, (): number[] => []);
  const dxCols = Array.from({ length: nCint }, (): number[] => []);

  for (let ri = 0; ri < nRint; ri++) {
    for (let ci = 0; ci < nCint; ci++) {
      const k = (ri + rOff) * nCext + (ci + rOff);
      const { dx, dy } = invertTPS(tpsY, tpsX, gridX[k].target, gridY[k].target);
      dyRows[ri].push(dy);
      dxCols[ci].push(dx);
    }
  }

  const rowPos = dyRows.map(vals => median(vals));
  const colPos = dxCols.map(vals => median(vals));
  const intersections = buildIntersections(rowPos, colPos, 0, 0, 1, 1);
  return {
    rowPos, colPos,
    uniformRowPos: rowPos, uniformColPos: colPos,
    step: det1.step, rawCircles: [],
    lineHPos: [], lineVPos: [],
    rowAngle: 0, colAngle: 0, harrisCorners: [],
    intersections,
  };
}

// ── mapDetectionThroughWarp ─────────────────────────────────────────────────

function mapDetectionThroughWarp(det1: Detection, mapRows: (pos: number[]) => number[], mapCols: (pos: number[]) => number[]) {
  const rowPos = mapRows(det1.rowPos);
  const colPos = mapCols(det1.colPos);
  const intersections = buildIntersections(rowPos, colPos, 0, 0, 1, 1);
  return {
    rowPos, colPos,
    uniformRowPos: mapRows(det1.uniformRowPos),
    uniformColPos: mapCols(det1.uniformColPos),
    step: det1.step,
    rawCircles: [],
    lineHPos: [], lineVPos: [],
    rowAngle: 0, colAngle: 0, harrisCorners: [],
    intersections,
  };
}

// ── nearestIndex ────────────────────────────────────────────────────────────

function nearestIndex(arr: readonly number[], val: number) {
  let best = 0, bestDist = Infinity;
  for (let i = 0; i < arr.length; i++) {
    const d = Math.abs(arr[i] - val);
    if (d < bestDist) { bestDist = d; best = i; }
  }
  return best;
}

// ── Elided edge detection ───────────────────────────────────────────────────

function detectElidedEdges(grayMat: CvMat, detection: Detection, stones: { r: number; c: number; color: string }[]) {
  const { rowPos, colPos, step } = detection;
  const nRows = rowPos.length, nCols = colPos.length;

  // Stone shortcut: if any stone sits on an edge row/col, it's a real edge.
  const stoneOnEdge = { top: false, bottom: false, left: false, right: false };
  for (const s of stones) {
    if (s.color === '.') continue;
    if (s.r === 0)            stoneOnEdge.top    = true;
    if (s.r === nRows - 1)    stoneOnEdge.bottom = true;
    if (s.c === 0)            stoneOnEdge.left   = true;
    if (s.c === nCols - 1)    stoneOnEdge.right  = true;
  }

  // Compute Canny once
  const canny = new cv.Mat();
  cv.Canny(grayMat, canny, 50, 150);
  const W = grayMat.cols, H = grayMat.rows;

  const tickLen = Math.round(0.5 * step);
  const halfW = 1; // corridor half-width (±1px = 3px wide)
  const pixelThreshold = 0.20; // fraction of strip pixels to call a tick
  const voteThreshold  = 0.40; // fraction of lines that must have ticks

  function countTicksAlongEdge(perpPositions: number[], edgePos: number, dir: number, isVertical: boolean) {
    // perpPositions: positions along the edge (colPos for top/bottom, rowPos for left/right)
    // edgePos: position of the outermost row/col
    // dir: -1 means sample toward 0, +1 means sample toward W or H
    // isVertical: true if the tick lines run vertically (top/bottom edges)
    let votes = 0;
    for (const pos of perpPositions) {
      let count = 0, total = 0;
      for (let d = 2; d <= tickLen; d++) {
        const along = Math.round(edgePos + dir * d);
        if (along < 0 || along >= (isVertical ? H : W)) break;
        for (let off = -halfW; off <= halfW; off++) {
          const across = Math.round(pos + off);
          if (across < 0 || across >= (isVertical ? W : H)) continue;
          total++;
          const px = isVertical
            ? canny.ucharAt(along, across)
            : canny.ucharAt(across, along);
          if (px > 0) count++;
        }
      }
      if (total > 0 && count / total > pixelThreshold) votes++;
    }
    return votes;
  }

  const result = { top: false, bottom: false, left: false, right: false };

  if (!stoneOnEdge.top) {
    const votes = countTicksAlongEdge(colPos, rowPos[0], -1, true);
    result.top = votes / colPos.length > voteThreshold;
  }
  if (!stoneOnEdge.bottom) {
    const votes = countTicksAlongEdge(colPos, rowPos[nRows - 1], 1, true);
    result.bottom = votes / colPos.length > voteThreshold;
  }
  if (!stoneOnEdge.left) {
    const votes = countTicksAlongEdge(rowPos, colPos[0], -1, false);
    result.left = votes / rowPos.length > voteThreshold;
  }
  if (!stoneOnEdge.right) {
    const votes = countTicksAlongEdge(rowPos, colPos[nCols - 1], 1, false);
    result.right = votes / rowPos.length > voteThreshold;
  }

  canny.delete();
  return result;
}

// ── Build detection from forced grid in original image coords ────────────────

function buildDetectionFromGrid(gridPoints: { x: number; y: number; r: number; c: number }[], rectCorners: Point[], rectW: number, rectH: number) {
  // Forward perspective transform: original → rectified space
  const srcPts = cv.matFromArray(4, 1, cv.CV_32FC2, [
    rectCorners[0].x, rectCorners[0].y,
    rectCorners[1].x, rectCorners[1].y,
    rectCorners[2].x, rectCorners[2].y,
    rectCorners[3].x, rectCorners[3].y,
  ]);
  const dstPts = cv.matFromArray(4, 1, cv.CV_32FC2, [
    0, 0, rectW, 0, rectW, rectH, 0, rectH,
  ]);
  const M = cv.getPerspectiveTransform(srcPts, dstPts);
  srcPts.delete(); dstPts.delete();

  const flat = [];
  for (const p of gridPoints) flat.push(p.x, p.y);
  const srcMat = cv.matFromArray(gridPoints.length, 1, cv.CV_32FC2, flat);
  const dstMat = new cv.Mat();
  cv.perspectiveTransform(srcMat, dstMat, M);
  M.delete(); srcMat.delete();

  const rectPts = [];
  for (let i = 0; i < gridPoints.length; i++) {
    rectPts.push({ x: dstMat.data32F[i * 2], y: dstMat.data32F[i * 2 + 1] });
  }
  dstMat.delete();

  // Group by row/col and build detection object
  const fRows = new Map<number, number[]>(), fCols = new Map<number, number[]>();
  const intersections: Point[][] = [];
  for (let i = 0; i < rectPts.length; i++) {
    const { r, c } = gridPoints[i];
    const pt = rectPts[i];
    let rowArr = fRows.get(r);
    if (!rowArr) fRows.set(r, rowArr = []);
    let colArr = fCols.get(c);
    if (!colArr) fCols.set(c, colArr = []);
    rowArr.push(pt.y);
    colArr.push(pt.x);
  }

  const sortedR = [...fRows.keys()].sort((a, b) => a - b);
  const sortedC = [...fCols.keys()].sort((a, b) => a - b);
  const rowPos = sortedR.map(r => median(fRows.get(r)!)); // safe: sortedR keys come from fRows
  const colPos = sortedC.map(c => median(fCols.get(c)!)); // safe: sortedC keys come from fCols

  const rIdx = new Map(sortedR.map((r, i) => [r, i]));
  const cIdx = new Map(sortedC.map((c, i) => [c, i]));
  for (let ri = 0; ri < sortedR.length; ri++) intersections.push([]);
  for (let i = 0; i < rectPts.length; i++) {
    intersections[rIdx.get(gridPoints[i].r)!][cIdx.get(gridPoints[i].c)!] = rectPts[i]; // safe: keys come from sortedR/sortedC built from fRows/fCols
  }

  const gaps = [];
  for (let i = 1; i < colPos.length; i++) gaps.push(colPos[i] - colPos[i - 1]);
  for (let i = 1; i < rowPos.length; i++) gaps.push(rowPos[i] - rowPos[i - 1]);
  const step = gaps.length ? median(gaps) : 30;

  return {
    rowPos, colPos,
    uniformRowPos: rowPos, uniformColPos: colPos,
    step, rawCircles: [], medRadius: step * 0.4,
    circleMinR: Math.max(3, Math.floor(step * 0.35)),
    circleMaxR: Math.max(5, Math.floor(step * 0.60)),
    rowAngle: 0, colAngle: 0, harrisCorners: [], houghLines: [],
    houghRowCentroids: [], houghColCentroids: [],
    lineHPos: [], lineVPos: [],
    intersections,
  };
}

// ── Intersection patch extraction for ONNX classifier ───────────────────────

const ONNX_PATCH_SIZE = 32;

/**
 * Extract fixed-size grayscale patches centered at each board intersection.
 * Returns a flat array of Uint8Array patches (one per intersection, row-major).
 * Used as input for the ONNX stone classifier.
 */
function extractIntersectionPatches(
  grayMat: CvMat,
  intersections: ReadonlyMatrix<Point>,
  nRows: number,
  nCols: number,
  patchSize: number = ONNX_PATCH_SIZE,
): Uint8Array[] {
  const W = grayMat.cols, H = grayMat.rows;
  const half = Math.floor(patchSize / 2);
  const patches: Uint8Array[] = [];
  for (let r = 0; r < nRows; r++) {
    for (let c = 0; c < nCols; c++) {
      const { x, y } = intersections[r][c];
      const cx = Math.round(x), cy = Math.round(y);
      const x0 = cx - half, y0 = cy - half;
      if (x0 < 0 || y0 < 0 || x0 + patchSize > W || y0 + patchSize > H) {
        patches.push(new Uint8Array(patchSize * patchSize));
        continue;
      }
      const roi = grayMat.roi(new cv.Rect(x0, y0, patchSize, patchSize));
      let sized = roi;
      if (roi.cols !== patchSize || roi.rows !== patchSize) {
        sized = new cv.Mat();
        cv.resize(roi, sized, new cv.Size(patchSize, patchSize), 0, 0, cv.INTER_LINEAR);
        roi.delete();
      } else {
        sized = roi;
      }
      patches.push(new Uint8Array(sized.data));
      sized.delete();
    }
  }
  return patches;
}

// ── Exports ─────────────────────────────────────────────────────────────────

export {
  // helpers
  median,
  nearestIndex,

  // corner ordering
  orderCorners,
  expandCorners,

  // board detection
  refineQuadWithHough,
  findBoardCornersCore,

  // grid bounds
  findGridBounds,

  // grid detection
  extractGridFeatures,
  fitGridFromFeatures,
  clusterPositions,
  medianStep,
  nearestNeighborStep,
  gridAlignmentScore,
  findGridAngle,
  fitWeightedQuadratic,
  fitGrid,
  buildIntersections,
  detectGrid,

  // preprocessing
  enhanceGray,

  // classification
  radialPower,
  sampleDisc,
  sampleAnnulus,
  kmeans2,
  classifyStones,

  // perspective
  rectifyBoard,

  // stone patch extraction
  extractStonePatches,
  extractIntersectionPatches,
  ONNX_PATCH_SIZE,

  // SGF
  inferBoardSize,
  computeEdgeOffsets,
  generateSGF,

  // quadratic fit
  fitQuadratic,
  polyEval,

  // pre-snap
  preSnapToCircles,

  // offset sampling
  collectOffsetSamples,
  fitSeparableQuadratic,
  localGridOffsets,

  // RANSAC
  ransacFilter1D,
  ransacFilter,

  // piecewise cylinder
  evalCylinderDisp,
  fitCylinder1D,
  fitPiecewiseCylinders,
  evalPiecewiseCylinders,
  invertPiecewiseCylinder,
  mapPositionsThroughDewarp,

  // TPS
  TPS_LAMBDA,
  tpsKernel,
  solveLinear,
  fitTPS,
  evalTPS,
  invertTPS,

  // dewarp images
  dewarpImage,
  dewarpImageTPS,
  dewarpImagePiecewise,
  dewarpImageMesh,

  // combined grid points
  buildCombinedGridPoints,
  buildDetectionFromControlPoints,
  mapDetectionThroughWarp,

  // forced grid
  buildDetectionFromGrid,

  // edge detection
  detectElidedEdges,

  // multi-threshold Canny
  multiThresholdCanny,
  thresholdCannyStack,
  deleteCannyStack,
  DEFAULT_CANNY_THRESHOLDS,

  // input classification
  classifyInputType,
  EDGE_DENSITY_THRESHOLD,
};
