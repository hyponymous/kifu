// run-pipeline.js — shared photo pipeline used by e2e tests, ablation, etc.
// Consumers provide loaded image data; this module handles the pipeline logic.

import * as defaultFns from './photo-pipeline.js';
import { activeDefaults } from './pipeline-defaults.js';
const { performance } = globalThis;
const DEFAULTS = activeDefaults();

/** Apply a 3x3 homography (row-major Float64Array) to a point */
export function applyHomography(H, x, y) {
  const w = H[6] * x + H[7] * y + H[8];
  return {
    x: (H[0] * x + H[1] * y + H[2]) / w,
    y: (H[3] * x + H[4] * y + H[5]) / w,
  };
}

/** Compute 3x3 inverse perspective from rectCorners / rectW / rectH */
export function computeInversePerspective(rectCorners, rectW, rectH) {
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

/**
 * Run the photo pipeline on pre-loaded image data.
 *
 * @param {{ colorMat, grayMat, width, height }} image
 * @param {object} [opts]
 * @param {object} [opts.fns] - override any photo-pipeline function
 * @param {number} [opts.cannyLo=50]
 * @param {number} [opts.cannyHi=125]
 * @param {number} [opts.tpsLambda=0.1]
 * @param {number} [opts.ransacThr=2.9]
 * @param {boolean} [opts.reDetect=true]
 * @param {boolean} [opts.combinedGrid=true]
 * @param {number} [opts.hintN=0]
 * @param {number} [opts.rpThreshRatio=0.85]
 * @param {number} [opts.gradFloor=64]
 * @param {Array<{x,y}>} [opts.rectCorners] - lock perspective corners (skip board detection)
 * @param {Array<{x,y,r,c}>} [opts.forcedGrid] - skip grid detection, use these points
 * @param {(name: string, ms: number) => void} [opts.onStage]
 * @param {(name: string, data: object) => void} [opts.onIntermediate]
 * @returns {{ nRows, nCols, grid, detectedIntersections } | null}
 */
export function runPipeline({ colorMat, grayMat, width, height }, opts = {}) {
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
  const lockedRectCorners = opts.rectCorners ?? null;
  const forcedGrid = opts.forcedGrid ?? null;
  const onStage = opts.onStage ?? null;
  const onIntermediate = opts.onIntermediate ?? null;

  const toDelete = [];
  const mat = m => { toDelete.push(m); return m; };

  const timed = (name, fn) => {
    const s = performance.now();
    const result = fn();
    if (onStage) onStage(name, +(performance.now() - s).toFixed(2));
    return result;
  };

  try {
    // ── Board detection ───────────────────────────────────────────────────
    const { rectCorners, edges } = timed('boardDetection', () => {
      if (lockedRectCorners) {
        return { rectCorners: lockedRectCorners, edges: null };
      }
      const blur = mat(new cv.Mat());
      const edges = mat(new cv.Mat());
      cv.GaussianBlur(grayMat, blur, new cv.Size(5, 5), 0);
      cv.Canny(blur, edges, cannyLo, cannyHi);
      const boardResult = fns.findBoardCornersCore(colorMat, edges, hintN, fns.refineQuadWithHough);
      const rectCorners = boardResult ? boardResult.corners : [
        { x: 0, y: 0 }, { x: width - 1, y: 0 },
        { x: width - 1, y: height - 1 }, { x: 0, y: height - 1 },
      ];
      return { rectCorners, edges };
    });
    if (onIntermediate && edges) onIntermediate('boardDetection', { edges });

    // ── Rectification ─────────────────────────────────────────────────────
    const { rectified, rectW, rectH } = timed('rectification', () => {
      const dist2d = (a, b) => Math.hypot(b.x - a.x, b.y - a.y);
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
      return mat(fns.enhanceGray(rectGrayBlurred, false));
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

      const grid = Array.from({ length: nRows }, () => Array(nCols).fill('.'));
      for (const s of classResult.stones) {
        if (s.r < nRows && s.c < nCols) grid[s.r][s.c] = s.color;
      }

      return { nRows, nCols, grid, detectedIntersections: forcedGrid, rectCorners, rectW, rectH, classResult, finalDetection: forcedDetection, elidedEdges };
    }

    // ── Find tight board bounds ────────────────────────────────────────────
    const gridBounds = timed('findGridBounds', () => {
      const bounds = fns.findGridBounds(rectGray, cannyLo, cannyHi);
      if (onIntermediate) onIntermediate('findGridBounds', { bounds });
      return bounds;
    });

    // ── Grid detection ────────────────────────────────────────────────────
    const detection = timed('detectGrid', () => {
      const circleSens = 24;
      return fns.detectGrid(rectGray, hintN, circleSens, { forceRows, forceCols, gridBounds, skipTrimEdges });
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
      const uni2snapY = u => snappedRows[0] + (u - uniPad) / uniStep * stepY;
      const uni2snapX = u => snappedCols[0] + (u - uniPad) / uniStep * stepX;

      const cleanY = fns.ransacFilter(tpsYPoints,
        pt => uni2snapY(pt.y) + fns.polyEval(yCoeffs, uni2snapY(pt.y)), ransacThr);
      const cleanX = fns.ransacFilter(tpsXPoints,
        pt => uni2snapX(pt.x) + fns.polyEval(xCoeffs, uni2snapX(pt.x)), ransacThr);

      let gridY = cleanY, gridX = cleanX;
      let combined = null;
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
      return { tpsY: null, tpsX: null, usedTPS: false, dewarped, tpsFitDetection: detection };
    });
    if (onIntermediate) onIntermediate('tpsFit', { dewarped });

    // ── Re-detection on dewarped image ────────────────────────────────────
    let finalDetection = tpsFitDetection;
    const dewarpedGray = timed('reDetection', () => {
      const dewarpedGrayRaw = mat(new cv.Mat());
      cv.cvtColor(dewarped, dewarpedGrayRaw, cv.COLOR_RGBA2GRAY);
      const dewarpedGray = mat(fns.enhanceGray(dewarpedGrayRaw, false));

      if (reDetect && finalDetection === detection) {
        const circleSens = 24;
        const detection2 = fns.detectGrid(dewarpedGray, hintN, circleSens, { forceRows, forceCols, skipTrimEdges });
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
    const grid = Array.from({ length: nRows }, () => Array(nCols).fill('.'));
    for (const s of classResult.stones) {
      if (s.r < nRows && s.c < nCols) {
        grid[s.r][s.c] = s.color;
      }
    }

    // ── Map back to original coordinates ──────────────────────────────────
    const detectedIntersections = timed('mapBack', () => {
      const H = computeInversePerspective(rectCorners, rectW, rectH);
      const result = [];
      for (let r = 0; r < nRows; r++) {
        for (let c = 0; c < nCols; c++) {
          const dpt = finalDetection.intersections[r][c];
          let rx, ry;
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

    return { nRows, nCols, grid, detectedIntersections, rectCorners, rectW, rectH, classResult, finalDetection, elidedEdges };
  } finally {
    toDelete.forEach(m => { try { m.delete(); } catch {} });
  }
}
