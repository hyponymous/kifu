// run-pipeline.ts — shared photo pipeline used by e2e tests, ablation, etc.
// Consumers provide loaded image data; this module handles the pipeline logic.
//
// The pipeline is decomposed into typed stage functions (stage*) with explicit
// input/output interfaces.  runPipeline chains them together, handling timing,
// intermediate callbacks, and Mat lifecycle via MatScope.

import * as defaultFns from './photo-pipeline';
import { activeDefaults } from './pipeline-defaults';
import { MatScope } from './mat-scope';
import type { ClassifierMode } from './pipeline-defaults';
import type { Point, Detection, TPSModel, TPSPoint, ClassResult, ElidedEdges, CombinedGrid, InputType } from './photo-pipeline';

const { performance } = globalThis;
const DEFAULTS = activeDefaults();

// ── Public types ────────────────────────────────────────────────────────────

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
  multiCannyEnabled?: boolean;
  minCannySupport?: number;
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

// ── Resolved configuration ──────────────────────────────────────────────────

type PipelineFns = typeof defaultFns;

interface ResolvedConfig {
  cannyLo: number;
  cannyHi: number;
  tpsLambda: number;
  ransacThr: number;
  reDetect: boolean;
  combinedGrid: boolean;
  hintN: number;
  forceRows: number;
  forceCols: number;
  skipTrimEdges: boolean;
  rpThreshRatio: number;
  gradFloor: number;
  claheEnabled: boolean;
  classifierMode: ClassifierMode;
  houghBlurSize: number;
  circleParam2: number;
  multiCannyEnabled: boolean;
  minCannySupport: number;
  classifyIntersections: IntersectionClassifier | null;
  lockedRectCorners: Point[] | null;
  forcedGrid: { x: number; y: number; r: number; c: number }[] | null;
}

function resolveConfig(opts: PipelineOpts): ResolvedConfig {
  return {
    cannyLo: opts.cannyLo ?? DEFAULTS.cannyLo,
    cannyHi: opts.cannyHi ?? DEFAULTS.cannyHi,
    tpsLambda: opts.tpsLambda ?? DEFAULTS.tpsLambda,
    ransacThr: opts.ransacThr ?? DEFAULTS.ransacThr,
    reDetect: opts.reDetect ?? DEFAULTS.reDetect,
    combinedGrid: opts.combinedGrid ?? DEFAULTS.combinedGrid,
    hintN: opts.hintN ?? 0,
    forceRows: opts.forceRows ?? 0,
    forceCols: opts.forceCols ?? 0,
    skipTrimEdges: opts.skipTrimEdges ?? DEFAULTS.skipTrimEdges,
    rpThreshRatio: opts.rpThreshRatio ?? DEFAULTS.rpThreshRatio,
    gradFloor: opts.gradFloor ?? DEFAULTS.gradFloor,
    claheEnabled: opts.claheEnabled ?? DEFAULTS.claheEnabled,
    classifierMode: opts.classifierMode ?? DEFAULTS.classifierMode,
    houghBlurSize: opts.houghBlurSize ?? DEFAULTS.houghBlurSize,
    circleParam2: opts.circleParam2 ?? DEFAULTS.circleParam2,
    multiCannyEnabled: opts.multiCannyEnabled ?? DEFAULTS.multiCannyEnabled,
    minCannySupport: opts.minCannySupport ?? DEFAULTS.minCannySupport,
    classifyIntersections: opts.classifyIntersections ?? null,
    lockedRectCorners: opts.rectCorners ?? null,
    forcedGrid: opts.forcedGrid ?? null,
  };
}

// ── Stage result types ──────────────────────────────────────────────────────

interface ClassifyInputResult {
  inputType: InputType;
  effectiveCannyLo: number;
  effectiveCannyHi: number;
}

interface BoardDetectionResult {
  rectCorners: Point[];
  edges: CvMat | null;
}

interface RectificationResult {
  rectified: CvMat;
  rectW: number;
  rectH: number;
}

interface PreSnapResult {
  snappedRows: number[];
  snappedCols: number[];
  snappedIntersections: Point[][];
}

interface OffsetSamplingResult {
  yCoeffs: number[];
  xCoeffs: number[];
  tpsYPoints: TPSPoint[];
  tpsXPoints: TPSPoint[];
  uniRowY: number[];
  uniColX: number[];
  uniOutW: number;
  uniOutH: number;
  uniStepX: number;
  uniStepY: number;
  uniPad: number;
  stepX: number;
  stepY: number;
}

interface TPSFitResult {
  tpsY: TPSModel | null;
  tpsX: TPSModel | null;
  usedTPS: boolean;
  dewarped: CvMat;
  tpsFitDetection: Detection;
}

interface ReDetectionResult {
  dewarpedGray: CvMat;
  finalDetection: Detection;
}

// ── Stage functions ─────────────────────────────────────────────────────────

function stageClassifyInput(
  fns: PipelineFns, grayMat: CvMat,
  cannyLo: number, cannyHi: number,
): ClassifyInputResult {
  const inputType = fns.classifyInputType(grayMat);
  // Photos need higher canny thresholds to suppress wood grain noise
  const effectiveCannyLo = inputType === 'photo' ? Math.max(cannyLo, 100) : cannyLo;
  const effectiveCannyHi = inputType === 'photo' ? Math.max(cannyHi, 250) : cannyHi;
  return { inputType, effectiveCannyLo, effectiveCannyHi };
}

function stageDetectBoard(
  fns: PipelineFns, scope: MatScope,
  colorMat: CvMat, grayMat: CvMat,
  effectiveCannyLo: number, effectiveCannyHi: number,
  hintN: number, lockedRectCorners: Point[] | null,
  width: number, height: number,
): BoardDetectionResult {
  if (lockedRectCorners) {
    return { rectCorners: lockedRectCorners, edges: null };
  }
  const blur = scope.track(new cv.Mat());
  const edges = scope.track(new cv.Mat());
  cv.GaussianBlur(grayMat, blur, new cv.Size(5, 5), 0);
  cv.Canny(blur, edges, effectiveCannyLo, effectiveCannyHi);
  const boardResult = fns.findBoardCornersCore(colorMat, edges, hintN, fns.refineQuadWithHough);
  const rectCorners: Point[] = boardResult ? boardResult.corners : [
    { x: 0, y: 0 }, { x: width - 1, y: 0 },
    { x: width - 1, y: height - 1 }, { x: 0, y: height - 1 },
  ];
  return { rectCorners, edges };
}

function stageRectify(
  fns: PipelineFns, scope: MatScope,
  colorMat: CvMat, rectCorners: readonly Point[],
  width: number, height: number,
): RectificationResult {
  const dist2d = (a: Point, b: Point) => Math.hypot(b.x - a.x, b.y - a.y);
  const [TL, TR, BR, BL] = rectCorners;
  const naturalW = Math.min(Math.round(Math.max(dist2d(TL, TR), dist2d(BL, BR))), width);
  const naturalH = Math.min(Math.round(Math.max(dist2d(TL, BL), dist2d(TR, BR))), height);
  const rectScale = Math.min(1, 1200 / Math.max(naturalW, naturalH));
  const rectW = Math.round(naturalW * rectScale);
  const rectH = Math.round(naturalH * rectScale);
  const rectified = scope.track(fns.rectifyBoard(colorMat, rectCorners, rectW, rectH));
  return { rectified, rectW, rectH };
}

function stageEnhanceGray(
  fns: PipelineFns, scope: MatScope,
  rectified: CvMat, claheEnabled: boolean,
): CvMat {
  const rectGrayRaw = scope.track(new cv.Mat());
  cv.cvtColor(rectified, rectGrayRaw, cv.COLOR_RGBA2GRAY);
  const rectGrayBlurred = scope.track(new cv.Mat());
  cv.GaussianBlur(rectGrayRaw, rectGrayBlurred, new cv.Size(3, 3), 0);
  return scope.track(fns.enhanceGray(rectGrayBlurred, false, { claheEnabled }));
}

function stagePreSnap(
  fns: PipelineFns, detection: Detection, rectGray: CvMat,
): PreSnapResult {
  const uniformIntersections = fns.buildIntersections(
    detection.uniformRowPos, detection.uniformColPos,
    detection.rowAngle, detection.colAngle, rectGray.cols, rectGray.rows);
  return fns.preSnapToCircles(
    detection.uniformRowPos, detection.uniformColPos,
    detection.rawCircles, uniformIntersections);
}

function stageOffsetSampling(
  fns: PipelineFns,
  rectGray: CvMat,
  snappedRows: readonly number[], snappedCols: readonly number[],
  effectiveCannyLo: number, effectiveCannyHi: number,
  detection: Detection,
  snappedIntersections: Point[][],
): OffsetSamplingResult {
  const nR = snappedRows.length, nC = snappedCols.length;
  const stepY = nR > 1 ? (snappedRows[nR - 1] - snappedRows[0]) / (nR - 1) : 1;
  const stepX = nC > 1 ? (snappedCols[nC - 1] - snappedCols[0]) / (nC - 1) : 1;
  // Preserve the grid's natural aspect ratio (stepY/stepX) in the output.
  // A square board photographed overhead has stepX ≈ stepY; a non-square board
  // (e.g. Japanese 15:14) or oblique angle has stepX ≠ stepY.  Forcing both to
  // the same value would distort circular stones into ellipses.
  const uniStepX = stepX;
  const uniStepY = stepY;
  const uniPad = Math.max(uniStepX, uniStepY);
  const uniRowY = Array.from({ length: nR }, (_, i) => uniPad + i * uniStepY);
  const uniColX = Array.from({ length: nC }, (_, j) => uniPad + j * uniStepX);
  const uniOutW = Math.round(2 * uniPad + (nC - 1) * uniStepX);
  const uniOutH = Math.round(2 * uniPad + (nR - 1) * uniStepY);

  const { yXs, yYs, xXs, xYs, tpsYPoints, tpsXPoints } = fns.collectOffsetSamples(
    rectGray, snappedRows, snappedCols, effectiveCannyLo, effectiveCannyHi,
    detection.rawCircles, snappedIntersections,
    { rowY: uniRowY, colX: uniColX });
  const { yCoeffs, xCoeffs } = fns.fitSeparableQuadratic(yXs, yYs, xXs, xYs);

  return { yCoeffs, xCoeffs, tpsYPoints, tpsXPoints,
           uniRowY, uniColX, uniOutW, uniOutH, uniStepX, uniStepY, uniPad, stepX, stepY };
}

function stageTpsFit(
  fns: PipelineFns, scope: MatScope,
  cfg: Pick<ResolvedConfig, 'ransacThr' | 'tpsLambda' | 'combinedGrid'>,
  detection: Detection,
  rectified: CvMat,
  snappedRows: readonly number[], snappedCols: readonly number[],
  offset: OffsetSamplingResult,
): TPSFitResult {
  const { yCoeffs, xCoeffs, tpsYPoints, tpsXPoints,
          uniRowY, uniColX, uniOutW, uniOutH, uniStepX, uniStepY, uniPad, stepX, stepY } = offset;

  const uni2snapY = (u: number) => snappedRows[0] + (u - uniPad) / uniStepY * stepY;
  const uni2snapX = (u: number) => snappedCols[0] + (u - uniPad) / uniStepX * stepX;

  const cleanY = fns.ransacFilter(tpsYPoints,
    (pt: TPSPoint) => uni2snapY(pt.y) + fns.polyEval(yCoeffs, uni2snapY(pt.y)), cfg.ransacThr);
  const cleanX = fns.ransacFilter(tpsXPoints,
    (pt: TPSPoint) => uni2snapX(pt.x) + fns.polyEval(xCoeffs, uni2snapX(pt.x)), cfg.ransacThr);

  let gridY: readonly TPSPoint[] = cleanY, gridX: readonly TPSPoint[] = cleanX;
  let combined: CombinedGrid | null = null;
  if (cfg.combinedGrid) {
    combined = fns.buildCombinedGridPoints(cleanY, cleanX,
      snappedRows, snappedCols, cfg.tpsLambda, false, uniRowY, uniColX);
    if (combined) { gridY = combined.gridY; gridX = combined.gridX; }
  }

  const tpsY = fns.fitTPS(gridY, cfg.tpsLambda);
  const tpsX = fns.fitTPS(gridX, cfg.tpsLambda);

  if (tpsY && tpsX) {
    const dewarped = scope.track(fns.dewarpImageTPS(rectified, tpsY, tpsX, uniOutW, uniOutH));
    const tpsFitDetection = combined
      ? fns.buildDetectionFromControlPoints(combined, tpsY, tpsX, detection)
      : detection;
    return { tpsY, tpsX, usedTPS: true, dewarped, tpsFitDetection };
  }
  const dewarped = scope.track(fns.dewarpImage(rectified, yCoeffs, xCoeffs));
  return { tpsY: null, tpsX: null, usedTPS: false, dewarped, tpsFitDetection: detection };
}

function stageReDetection(
  fns: PipelineFns, scope: MatScope,
  dewarped: CvMat, tpsFitDetection: Detection, origDetection: Detection,
  nRows: number, nCols: number,
  cfg: Pick<ResolvedConfig, 'reDetect' | 'claheEnabled' | 'hintN' | 'circleParam2' |
    'forceRows' | 'forceCols' | 'skipTrimEdges' | 'houghBlurSize' |
    'multiCannyEnabled' | 'minCannySupport'>,
  inputType: InputType,
): ReDetectionResult {
  const dewarpedGrayRaw = scope.track(new cv.Mat());
  cv.cvtColor(dewarped, dewarpedGrayRaw, cv.COLOR_RGBA2GRAY);
  const dewarpedGray = scope.track(fns.enhanceGray(dewarpedGrayRaw, false, { claheEnabled: cfg.claheEnabled }));

  let finalDetection = tpsFitDetection;
  if (cfg.reDetect && finalDetection === origDetection) {
    const detection2 = fns.detectGrid(dewarpedGray, cfg.hintN, cfg.circleParam2, {
      forceRows: cfg.forceRows, forceCols: cfg.forceCols,
      skipTrimEdges: cfg.skipTrimEdges, houghBlurSize: cfg.houghBlurSize,
      useCirclesForAngle: inputType === 'photo',
      multiCannyEnabled: cfg.multiCannyEnabled, minCannySupport: cfg.minCannySupport,
    });
    if (detection2
      && detection2.rowPos.length === nRows
      && detection2.colPos.length === nCols) {
      finalDetection = detection2;
    }
  }
  return { dewarpedGray, finalDetection };
}

function stageClassification(
  fns: PipelineFns,
  dewarpedGray: CvMat, finalDetection: Detection,
  nRows: number, nCols: number,
  cfg: Pick<ResolvedConfig, 'classifierMode' | 'classifyIntersections' | 'rpThreshRatio' | 'gradFloor'>,
  inputType: InputType,
): ClassResult {
  if (cfg.classifierMode === 'onnx' && cfg.classifyIntersections) {
    const patches = fns.extractIntersectionPatches(
      dewarpedGray, finalDetection.intersections, nRows, nCols, fns.ONNX_PATCH_SIZE);
    const colors = cfg.classifyIntersections(patches, fns.ONNX_PATCH_SIZE);
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
    { rpThreshRatio: cfg.rpThreshRatio, gradFloor: cfg.gradFloor, inputType });
}

function stageElidedEdges(
  fns: PipelineFns,
  dewarpedGray: CvMat, finalDetection: Detection,
  stones: ClassResult['stones'], inputType: InputType,
): ElidedEdges {
  if (inputType === 'photo') {
    return { top: false, bottom: false, left: false, right: false };
  }
  return fns.detectElidedEdges(dewarpedGray, finalDetection, stones);
}

function stageMapBack(
  fns: PipelineFns,
  finalDetection: Detection, nRows: number, nCols: number,
  rectCorners: readonly Point[], rectW: number, rectH: number,
  tpsY: TPSModel | null, tpsX: TPSModel | null, usedTPS: boolean,
  yCoeffs: readonly number[], xCoeffs: readonly number[],
): { r: number; c: number; x: number; y: number }[] {
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
}

// ── Helpers ─────────────────────────────────────────────────────────────────

function buildStoneGrid(nRows: number, nCols: number, classResult: ClassResult): string[][] {
  const grid: string[][] = Array.from({ length: nRows }, () => Array(nCols).fill('.'));
  for (const s of classResult.stones) {
    if (s.r < nRows && s.c < nCols) grid[s.r][s.c] = s.color;
  }
  return grid;
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

// ── Forced grid shortcut ────────────────────────────────────────────────────
// When forcedGrid is provided, skip grid detection / dewarp / re-detection
// and go straight to classification on the rectified image.

function runForcedGridPath(
  fns: PipelineFns,
  timed: <T>(name: string, fn: () => T) => T,
  onIntermediate: ((name: string, data: Record<string, unknown>) => void) | null,
  cfg: ResolvedConfig,
  inputType: InputType,
  rectGray: CvMat, rectCorners: Point[], rectW: number, rectH: number,
): PipelineResult {
  // forcedGrid is guaranteed non-null here
  const forcedGrid = cfg.forcedGrid!; // eslint-disable-line @typescript-eslint/no-non-null-assertion -- checked by caller
  const forcedDetection = timed('forcedGrid', () =>
    fns.buildDetectionFromGrid(forcedGrid, rectCorners, rectW, rectH)
  );
  if (onIntermediate) onIntermediate('detectGrid', { detection: forcedDetection });

  const nRows = forcedDetection.rowPos.length;
  const nCols = forcedDetection.colPos.length;

  const classResult = timed('classification', () =>
    stageClassification(fns, rectGray, forcedDetection, nRows, nCols, cfg, inputType)
  );
  if (onIntermediate) onIntermediate('classification', { classResult, finalDetection: forcedDetection });

  const elidedEdges = timed('elidedEdges', () =>
    stageElidedEdges(fns, rectGray, forcedDetection, classResult.stones, inputType)
  );
  if (onIntermediate) onIntermediate('elidedEdges', { elidedEdges });

  const grid = buildStoneGrid(nRows, nCols, classResult);

  return {
    inputType, nRows, nCols, grid,
    detectedIntersections: forcedGrid,
    rectCorners, rectW, rectH, classResult,
    finalDetection: forcedDetection, elidedEdges,
  };
}

// ── Pipeline orchestrator ───────────────────────────────────────────────────

export function runPipeline({ colorMat, grayMat, width, height }: ImageData, opts: PipelineOpts = {}): PipelineResult | null {
  const fns = { ...defaultFns, ...opts.fns };
  const cfg = resolveConfig(opts);
  const onStage = opts.onStage ?? null;
  const onIntermediate = opts.onIntermediate ?? null;

  const scope = new MatScope();
  const timed = <T>(name: string, fn: () => T): T => {
    const s = performance.now();
    const result = fn();
    if (onStage) onStage(name, +(performance.now() - s).toFixed(2));
    return result;
  };

  try {
    // ── Input classification ──────────────────────────────────────────
    const { inputType, effectiveCannyLo, effectiveCannyHi } =
      timed('classifyInput', () => stageClassifyInput(fns, grayMat, cfg.cannyLo, cfg.cannyHi));
    if (onIntermediate) onIntermediate('classifyInput', { inputType });

    // ── Board detection ───────────────────────────────────────────────
    const { rectCorners, edges } =
      timed('boardDetection', () => stageDetectBoard(fns, scope,
        colorMat, grayMat, effectiveCannyLo, effectiveCannyHi,
        cfg.hintN, cfg.lockedRectCorners, width, height));
    if (onIntermediate && edges) onIntermediate('boardDetection', { edges });

    // ── Rectification ─────────────────────────────────────────────────
    const { rectified, rectW, rectH } =
      timed('rectification', () => stageRectify(fns, scope, colorMat, rectCorners, width, height));
    if (onIntermediate) onIntermediate('rectification', { rectified });

    // ── Enhance gray ──────────────────────────────────────────────────
    const rectGray =
      timed('enhanceGray', () => stageEnhanceGray(fns, scope, rectified, cfg.claheEnabled));
    if (onIntermediate) onIntermediate('enhanceGray', { enhanced: rectGray });

    // ── Forced grid shortcut ──────────────────────────────────────────
    if (cfg.forcedGrid) {
      return runForcedGridPath(fns, timed, onIntermediate,
        cfg, inputType, rectGray, rectCorners, rectW, rectH);
    }

    // ── Find tight board bounds ───────────────────────────────────────
    const gridBounds = timed('findGridBounds', () => {
      const bounds = fns.findGridBounds(rectGray, effectiveCannyLo, effectiveCannyHi);
      if (onIntermediate) onIntermediate('findGridBounds', { bounds });
      return bounds;
    });

    // ── Grid detection ────────────────────────────────────────────────
    const detection = timed('detectGrid', () =>
      fns.detectGrid(rectGray, cfg.hintN, cfg.circleParam2, {
        forceRows: cfg.forceRows, forceCols: cfg.forceCols, gridBounds,
        skipTrimEdges: cfg.skipTrimEdges, houghBlurSize: cfg.houghBlurSize,
        useCirclesForAngle: inputType === 'photo',
        multiCannyEnabled: cfg.multiCannyEnabled, minCannySupport: cfg.minCannySupport,
      }));
    if (!detection) return null;
    if (onIntermediate) onIntermediate('detectGrid', { detection });

    const nRows = detection.rowPos.length;
    const nCols = detection.colPos.length;

    // ── Pre-snap ──────────────────────────────────────────────────────
    const { snappedRows, snappedCols, snappedIntersections } =
      timed('preSnap', () => stagePreSnap(fns, detection, rectGray));

    // ── Offset sampling ───────────────────────────────────────────────
    const offsetResult =
      timed('offsetSampling', () => stageOffsetSampling(fns, rectGray,
        snappedRows, snappedCols, effectiveCannyLo, effectiveCannyHi,
        detection, snappedIntersections));

    // ── TPS fit ───────────────────────────────────────────────────────
    const { tpsY, tpsX, usedTPS, dewarped, tpsFitDetection } =
      timed('tpsFit', () => stageTpsFit(fns, scope, cfg,
        detection, rectified, snappedRows, snappedCols, offsetResult));
    if (onIntermediate) onIntermediate('tpsFit', { dewarped });

    // ── Re-detection on dewarped image ────────────────────────────────
    const { dewarpedGray, finalDetection } =
      timed('reDetection', () => stageReDetection(fns, scope,
        dewarped, tpsFitDetection, detection, nRows, nCols, cfg, inputType));
    if (onIntermediate) onIntermediate('reDetection', { dewarpedGray });

    // ── Classification ────────────────────────────────────────────────
    const classResult =
      timed('classification', () => stageClassification(fns,
        dewarpedGray, finalDetection, nRows, nCols, cfg, inputType));
    if (onIntermediate) onIntermediate('classification', { classResult, finalDetection });

    // ── Elided edge detection ─────────────────────────────────────────
    const elidedEdges =
      timed('elidedEdges', () => stageElidedEdges(fns,
        dewarpedGray, finalDetection, classResult.stones, inputType));
    if (onIntermediate) onIntermediate('elidedEdges', { elidedEdges });

    // ── Build stone grid ──────────────────────────────────────────────
    const grid = buildStoneGrid(nRows, nCols, classResult);

    // ── Map back to original coordinates ──────────────────────────────
    const detectedIntersections =
      timed('mapBack', () => stageMapBack(fns,
        finalDetection, nRows, nCols,
        rectCorners, rectW, rectH,
        tpsY, tpsX, usedTPS,
        offsetResult.yCoeffs, offsetResult.xCoeffs));

    return {
      inputType, nRows, nCols, grid, detectedIntersections,
      rectCorners, rectW, rectH, classResult, finalDetection, elidedEdges,
    };
  } finally {
    scope.release();
  }
}
