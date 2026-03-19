// run-pipeline.ts — shared photo pipeline used by e2e tests, ablation, etc.
// Consumers provide loaded image data; this module handles the pipeline logic.

import * as defaultFns from './photo-pipeline';
import { activeDefaults } from './pipeline-defaults';
import type { ClassifierMode } from './pipeline-defaults';
import type { Point, Detection, TPSModel, TPSPoint, ClassResult, ElidedEdges, CombinedGrid, InputType } from './photo-pipeline';

const { performance } = globalThis;
const DEFAULTS = activeDefaults();

export interface ImageData {
  colorMat: CvMat;
  grayMat: CvMat;
  width: number;
  height: number;
}

/** Synchronous stone classifier injectable (e.g. ONNX, pre-loaded before calling runPipeline). */
export type IntersectionClassifier = (
  patches: Uint8Array[],
  patchSize: number,
) => Array<'B' | 'W' | '.'>;

export interface PipelineOpts {
  fns?: Partial<typeof defaultFns>;
  cannyLo?: number;
  cannyHi?: number;
  tpsLambda?: number;
  ransacThr?: number;
  reDetect?: boolean;
  combinedGrid?: boolean;
  hintN?: number;
  forceRows?: number;
  forceCols?: number;
  skipTrimEdges?: boolean;
  rpThreshRatio?: number;
  gradFloor?: number;
  claheEnabled?: boolean;
  classifierMode?: ClassifierMode;
  houghBlurSize?: number;
  circleParam2?: number;
  /** When classifierMode is 'onnx', this must be provided (pre-loaded ONNX session wrapper). */
  classifyIntersections?: IntersectionClassifier;
  rectCorners?: Point[];
  forcedGrid?: { x: number; y: number; r: number; c: number }[];
  onStage?: (name: string, ms: number) => void;
  onIntermediate?: (name: string, data: Record<string, unknown>) => void;
}

export interface PipelineResult {
  inputType: InputType;
  nRows: number;
  nCols: number;
  grid: string[][];
  detectedIntersections: { r: number; c: number; x: number; y: number }[];
  rectCorners: Point[];
  rectW: number;
  rectH: number;
  classResult: ClassResult;
  finalDetection: Detection;
  elidedEdges: ElidedEdges;
}

/** Apply a 3x3 homography (row-major Float64Array) to a point */
export function applyHomography(H: Float64Array, x: number, y: number): Point {
  const w = H[6] * x + H[7] * y + H[8];
  return {
    x: (H[0] * x + H[1] * y + H[2]) / w,
    y: (H[3] * x + H[4] * y + H[5]) / w,
  };
}

/** Compute 3x3 inverse perspective from rectCorners / rectW / rectH */
export function computeInversePerspective(rectCorners: readonly Point[], rectW: number, rectH: number): Float64Array {
  const srcPts = cv.matFromArray(4, 1, cv.CV_32FC2, [
    0, 0, rectW, 0, rectW, rectH, 0, rectH,
  ]);
  const dstPts = cv.matFromArray(4, 1, cv.CV_32FC2, [
    rectCorners[0].x, rectCorners[0].y,
    rectCorners[1].x, rectCorners[1].y,
    rectCorners[2].x, rectCorners[2].y,
    rectCorners[3].x, rectCorners[3].y,
  ]);
  const M = cv.getPerspectiveTransform(srcPts, dstPts);
  const H = new Float64Array(9);
  for (let i = 0; i < 9; i++) H[i] = M.doubleAt(Math.floor(i / 3), i % 3);
  srcPts.delete(); dstPts.delete(); M.delete();
  return H;
}

export function runPipeline({ colorMat, grayMat, width, height }: ImageData, opts: PipelineOpts = {}): PipelineResult | null {
  const fns = { ...defaultFns, ...opts.fns };
  const cannyLo = opts.cannyLo ?? DEFAULTS.cannyLo;
  const cannyHi = opts.cannyHi ?? DEFAULTS.cannyHi;
  const tpsLambda = opts.tpsLambda ?? DEFAULTS.tpsLambda;
  const ransacThr = opts.ransacThr ?? DEFAULTS.ransacThr;
  const reDetect = opts.reDetect ?? DEFAULTS.reDetect;
  const combinedGrid = opts.combinedGrid ?? DEFAULTS.combinedGrid;
  const hintN = opts.hintN ?? 0;
  const forceRows = opts.forceRows ?? 0;
  const forceCols = opts.forceCols ?? 0;
  const skipTrimEdges = opts.skipTrimEdges ?? DEFAULTS.skipTrimEdges;
  const rpThreshRatio = opts.rpThreshRatio ?? DEFAULTS.rpThreshRatio;
  const gradFloor = opts.gradFloor ?? DEFAULTS.gradFloor;
  const claheEnabled = opts.claheEnabled ?? DEFAULTS.claheEnabled;
  const classifierMode = opts.classifierMode ?? DEFAULTS.classifierMode;
  const houghBlurSize = opts.houghBlurSize ?? DEFAULTS.houghBlurSize;
  const circleParam2 = opts.circleParam2 ?? DEFAULTS.circleParam2;
  const classifyIntersections = opts.classifyIntersections ?? null;
  const lockedRectCorners = opts.rectCorners ?? null;
  const forcedGrid = opts.forcedGrid ?? null;
  const onStage = opts.onStage ?? null;
  const onIntermediate = opts.onIntermediate ?? null;

  const toDelete: CvMat[] = [];
  const mat = (m: CvMat): CvMat => { toDelete.push(m); return m; };

  const timed = <T>(name: string, fn: () => T): T => {
    const s = performance.now();
    const result = fn();
    if (onStage) onStage(name, +(performance.now() - s).toFixed(2));
    return result;
  };

  try {
    // ── Input classification ──────────────────────────────────────────────
    const inputType = timed('classifyInput', () => fns.classifyInputType(grayMat));
    if (onIntermediate) onIntermediate('classifyInput', { inputType });

    // ── Board detection ───────────────────────────────────────────────────
    const { rectCorners, edges } = timed('boardDetection', () => {
      if (lockedRectCorners) {
        return { rectCorners: lockedRectCorners, edges: null as CvMat | null };
      }
      const blur = mat(new cv.Mat());
      const edges = mat(new cv.Mat());
      cv.GaussianBlur(grayMat, blur, new cv.Size(5, 5), 0);
      cv.Canny(blur, edges, cannyLo, cannyHi);
      const boardResult = fns.findBoardCornersCore(colorMat, edges, hintN, fns.refineQuadWithHough);
      const rectCorners: Point[] = boardResult ? boardResult.corners : [
        { x: 0, y: 0 }, { x: width - 1, y: 0 },
        { x: width - 1, y: height - 1 }, { x: 0, y: height - 1 },
      ];
      return { rectCorners, edges: edges as CvMat | null };
    });
    if (onIntermediate && edges) onIntermediate('boardDetection', { edges });

    // ── Rectification ─────────────────────────────────────────────────────
    const { rectified, rectW, rectH } = timed('rectification', () => {
      const dist2d = (a: Point, b: Point) => Math.hypot(b.x - a.x, b.y - a.y);
      const [TL, TR, BR, BL] = rectCorners;
      const naturalW = Math.min(Math.round(Math.max(dist2d(TL, TR), dist2d(BL, BR))), width);
      const naturalH = Math.min(Math.round(Math.max(dist2d(TL, BL), dist2d(TR, BR))), height);
      const rectScale = Math.min(1, 1200 / Math.max(naturalW, naturalH));
      const rectW = Math.round(naturalW * rectScale);
      const rectH = Math.round(naturalH * rectScale);
      const rectified = mat(fns.rectifyBoard(colorMat, rectCorners, rectW, rectH));
      return { rectified, rectW, rectH };
    });
    if (onIntermediate) onIntermediate('rectification', { rectified });

    // ── Enhance gray ──────────────────────────────────────────────────────
    const rectGray = timed('enhanceGray', () => {
      const rectGrayRaw = mat(new cv.Mat());
      cv.cvtColor(rectified, rectGrayRaw, cv.COLOR_RGBA2GRAY);
      const rectGrayBlurred = mat(new cv.Mat());
      cv.GaussianBlur(rectGrayRaw, rectGrayBlurred, new cv.Size(3, 3), 0);
      return mat(fns.enhanceGray(rectGrayBlurred, false, { claheEnabled }));
    });
    if (onIntermediate) onIntermediate('enhanceGray', { enhanced: rectGray });

    // ── Forced grid shortcut ────────────────────────────────────────────────
    // When forcedGrid is provided, skip grid detection / dewarp / re-detection
    // and go straight to classification on the rectified image.
    if (forcedGrid) {
      const forcedDetection = timed('forcedGrid', () =>
        fns.buildDetectionFromGrid(forcedGrid, rectCorners, rectW, rectH)
      );
      if (onIntermediate) onIntermediate('detectGrid', { detection: forcedDetection });

      const nRows = forcedDetection.rowPos.length;
      const nCols = forcedDetection.colPos.length;

      const classResult = timed('classification', () =>
        fns.classifyStones(rectGray, forcedDetection.rowPos, forcedDetection.colPos,
          forcedDetection.step, forcedDetection.rawCircles,
          forcedDetection.intersections, true,
          { rpThreshRatio, gradFloor })
      );
      if (onIntermediate) onIntermediate('classification', { classResult, finalDetection: forcedDetection });

      const elidedEdges = timed('elidedEdges', () =>
        fns.detectElidedEdges(rectGray, forcedDetection, classResult.stones)
      );
      if (onIntermediate) onIntermediate('elidedEdges', { elidedEdges });

      const grid: string[][] = Array.from({ length: nRows }, () => Array(nCols).fill('.'));
      for (const s of classResult.stones) {
        if (s.r < nRows && s.c < nCols) grid[s.r][s.c] = s.color;
      }

      return { inputType, nRows, nCols, grid, detectedIntersections: forcedGrid, rectCorners, rectW, rectH, classResult, finalDetection: forcedDetection, elidedEdges };
    }

    // ── Find tight board bounds ────────────────────────────────────────────
    const gridBounds = timed('findGridBounds', () => {
      const bounds = fns.findGridBounds(rectGray, cannyLo, cannyHi);
      if (onIntermediate) onIntermediate('findGridBounds', { bounds });
      return bounds;
    });

    // ── Grid detection ────────────────────────────────────────────────────
    const detection = timed('detectGrid', () => {
      return fns.detectGrid(rectGray, hintN, circleParam2, { forceRows, forceCols, gridBounds, skipTrimEdges, houghBlurSize });
    });
    if (!detection) return null;
    if (onIntermediate) onIntermediate('detectGrid', { detection });

    const nRows = detection.rowPos.length;
    const nCols = detection.colPos.length;

    // ── Pre-snap ──────────────────────────────────────────────────────────
    const { snappedRows, snappedCols, snappedIntersections } = timed('preSnap', () => {
      const uniformIntersections = fns.buildIntersections(
        detection.uniformRowPos, detection.uniformColPos,
        detection.rowAngle, detection.colAngle, rectGray.cols, rectGray.rows);
      return fns.preSnapToCircles(
        detection.uniformRowPos, detection.uniformColPos,
        detection.rawCircles, uniformIntersections);
    });

    // ── Offset sampling ───────────────────────────────────────────────────
    const { yCoeffs, xCoeffs, tpsYPoints, tpsXPoints,
            uniRowY, uniColX, uniOutW, uniOutH,
            uniStep, uniPad, stepX, stepY } = timed('offsetSampling', () => {
      const nR = snappedRows.length, nC = snappedCols.length;
      const stepY = nR > 1 ? (snappedRows[nR - 1] - snappedRows[0]) / (nR - 1) : 1;
      const stepX = nC > 1 ? (snappedCols[nC - 1] - snappedCols[0]) / (nC - 1) : 1;
      const uniStep = Math.max(stepX, stepY);
      const uniPad = uniStep;
      const uniRowY = Array.from({ length: nR }, (_, i) => uniPad + i * uniStep);
      const uniColX = Array.from({ length: nC }, (_, j) => uniPad + j * uniStep);
      const uniOutW = Math.round(2 * uniPad + (nC - 1) * uniStep);
      const uniOutH = Math.round(2 * uniPad + (nR - 1) * uniStep);

      const { yXs, yYs, xXs, xYs, tpsYPoints, tpsXPoints } = fns.collectOffsetSamples(
        rectGray, snappedRows, snappedCols, cannyLo, cannyHi,
        detection.rawCircles, snappedIntersections,
        { rowY: uniRowY, colX: uniColX });
      const { yCoeffs, xCoeffs } = fns.fitSeparableQuadratic(yXs, yYs, xXs, xYs);

      return { yCoeffs, xCoeffs, tpsYPoints, tpsXPoints,
               uniRowY, uniColX, uniOutW, uniOutH, uniStep, uniPad, stepX, stepY };
    });

    // ── TPS fit ───────────────────────────────────────────────────────────
    const { tpsY, tpsX, usedTPS, dewarped, tpsFitDetection } = timed('tpsFit', () => {
      const uni2snapY = (u: number) => snappedRows[0] + (u - uniPad) / uniStep * stepY;
      const uni2snapX = (u: number) => snappedCols[0] + (u - uniPad) / uniStep * stepX;

      const cleanY = fns.ransacFilter(tpsYPoints,
        (pt: TPSPoint) => uni2snapY(pt.y) + fns.polyEval(yCoeffs, uni2snapY(pt.y)), ransacThr);
      const cleanX = fns.ransacFilter(tpsXPoints,
        (pt: TPSPoint) => uni2snapX(pt.x) + fns.polyEval(xCoeffs, uni2snapX(pt.x)), ransacThr);

      let gridY: readonly TPSPoint[] = cleanY, gridX: readonly TPSPoint[] = cleanX;
      let combined: CombinedGrid | null = null;
      if (combinedGrid) {
        combined = fns.buildCombinedGridPoints(cleanY, cleanX,
          snappedRows, snappedCols, tpsLambda, false, uniRowY, uniColX);
        if (combined) { gridY = combined.gridY; gridX = combined.gridX; }
      }

      const tpsY = fns.fitTPS(gridY, tpsLambda);
      const tpsX = fns.fitTPS(gridX, tpsLambda);

      if (tpsY && tpsX) {
        const dewarped = mat(fns.dewarpImageTPS(rectified, tpsY, tpsX, uniOutW, uniOutH));
        const tpsFitDetection = combined
          ? fns.buildDetectionFromControlPoints(combined, tpsY, tpsX, detection)
          : detection;
        return { tpsY, tpsX, usedTPS: true, dewarped, tpsFitDetection };
      }
      const dewarped = mat(fns.dewarpImage(rectified, yCoeffs, xCoeffs));
      return { tpsY: null as TPSModel | null, tpsX: null as TPSModel | null, usedTPS: false, dewarped, tpsFitDetection: detection };
    });
    if (onIntermediate) onIntermediate('tpsFit', { dewarped });

    // ── Re-detection on dewarped image ────────────────────────────────────
    let finalDetection: Detection = tpsFitDetection;
    const dewarpedGray = timed('reDetection', () => {
      const dewarpedGrayRaw = mat(new cv.Mat());
      cv.cvtColor(dewarped, dewarpedGrayRaw, cv.COLOR_RGBA2GRAY);
      const dewarpedGray = mat(fns.enhanceGray(dewarpedGrayRaw, false, { claheEnabled }));

      if (reDetect && finalDetection === detection) {
        const detection2 = fns.detectGrid(dewarpedGray, hintN, circleParam2, { forceRows, forceCols, skipTrimEdges, houghBlurSize });
        if (detection2
          && detection2.rowPos.length === nRows
          && detection2.colPos.length === nCols) {
          finalDetection = detection2;
        }
      }
      return dewarpedGray;
    });
    if (onIntermediate) onIntermediate('reDetection', { dewarpedGray });

    // ── Classification ────────────────────────────────────────────────────
    const classResult = timed('classification', () => {
      if (classifierMode === 'onnx' && classifyIntersections) {
        const patches = fns.extractIntersectionPatches(
          dewarpedGray, finalDetection.intersections, nRows, nCols, fns.ONNX_PATCH_SIZE);
        const colors = classifyIntersections(patches, fns.ONNX_PATCH_SIZE);
        const stones = [];
        let idx = 0;
        for (let r = 0; r < nRows; r++) {
          for (let c = 0; c < nCols; c++) {
            const color = colors[idx++] ?? '.';
            if (color !== '.') {
              const pt = finalDetection.intersections[r][c];
              stones.push({ r, c, color, cx: Math.round(pt.x), cy: Math.round(pt.y) });
            }
          }
        }
        return { stones } as ClassResult;
      }
      return fns.classifyStones(dewarpedGray, finalDetection.rowPos, finalDetection.colPos,
        finalDetection.step, finalDetection.rawCircles,
        finalDetection.intersections, true,
        { rpThreshRatio, gradFloor });
    });

    if (onIntermediate) onIntermediate('classification', { classResult, finalDetection });

    // ── Elided edge detection ──────────────────────────────────────────────
    const elidedEdges = timed('elidedEdges', () =>
      fns.detectElidedEdges(dewarpedGray, finalDetection, classResult.stones)
    );
    if (onIntermediate) onIntermediate('elidedEdges', { elidedEdges });

    // Build stone grid
    const grid: string[][] = Array.from({ length: nRows }, () => Array(nCols).fill('.'));
    for (const s of classResult.stones) {
      if (s.r < nRows && s.c < nCols) {
        grid[s.r][s.c] = s.color;
      }
    }

    // ── Map back to original coordinates ──────────────────────────────────
    const detectedIntersections = timed('mapBack', () => {
      const H = computeInversePerspective(rectCorners, rectW, rectH);
      const result: { r: number; c: number; x: number; y: number }[] = [];
      for (let r = 0; r < nRows; r++) {
        for (let c = 0; c < nCols; c++) {
          const dpt = finalDetection.intersections[r][c];
          let rx: number, ry: number;
          if (usedTPS && tpsY && tpsX) {
            rx = fns.evalTPS(tpsX, dpt.x, dpt.y);
            ry = fns.evalTPS(tpsY, dpt.x, dpt.y);
          } else {
            rx = dpt.x + fns.polyEval(xCoeffs, dpt.x);
            ry = dpt.y + fns.polyEval(yCoeffs, dpt.y);
          }
          const orig = applyHomography(H, rx, ry);
          result.push({ r, c, x: orig.x, y: orig.y });
        }
      }
      return result;
    });

    return { inputType, nRows, nCols, grid, detectedIntersections, rectCorners, rectW, rectH, classResult, finalDetection, elidedEdges };
  } finally {
    toDelete.forEach(m => { try { m.delete(); } catch {} });
  }
}
