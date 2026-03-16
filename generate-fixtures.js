#!/usr/bin/env node
// generate-fixtures.js — runs the photo pipeline on images and writes fixture JSON
// Usage: node generate-fixtures.js fixtures/IMG_0976.jpg [...]

import './test-helpers/load-cv.js';
import { loadImage } from './test-helpers/load-image.js';
import { writeFileSync } from 'node:fs';
import { basename, dirname, join } from 'node:path';

const {
  findBoardCornersCore, rectifyBoard, detectGrid, enhanceGray,
  classifyStones, buildIntersections,
  preSnapToCircles, collectOffsetSamples, fitSeparableQuadratic,
  dewarpImage, dewarpImageTPS, polyEval,
  ransacFilter, fitTPS, evalTPS,
  buildCombinedGridPoints, buildDetectionFromControlPoints,
} = await import('./photo-pipeline.js');

const files = process.argv.slice(2);
if (files.length === 0) {
  console.error('Usage: node generate-fixtures.js <image> [...]');
  process.exit(1);
}

for (const filePath of files) {
  console.log(`\n=== Processing ${filePath} ===`);
  const toDelete = [];
  const mat = m => { toDelete.push(m); return m; };

  try {
    const { colorMat, grayMat, width, height } = await loadImage(filePath);
    mat(colorMat);
    mat(grayMat);

    // Edge detection
    const blur = mat(new cv.Mat());
    const edges = mat(new cv.Mat());
    cv.GaussianBlur(grayMat, blur, new cv.Size(5, 5), 0);
    cv.Canny(blur, edges, 50, 125);

    const hintN = 19;

    // Board detection
    const boardResult = findBoardCornersCore(colorMat, edges, hintN);
    const rectCorners = boardResult ? boardResult.corners : [
      { x: 0, y: 0 },
      { x: width - 1, y: 0 },
      { x: width - 1, y: height - 1 },
      { x: 0, y: height - 1 },
    ];

    const dist2d = (a, b) => Math.hypot(b.x - a.x, b.y - a.y);
    const [TL, TR, BR, BL] = rectCorners;
    const naturalW = Math.min(Math.round(Math.max(dist2d(TL, TR), dist2d(BL, BR))), width);
    const naturalH = Math.min(Math.round(Math.max(dist2d(TL, BL), dist2d(TR, BR))), height);
    const rectScale = Math.min(1, 1200 / Math.max(naturalW, naturalH));
    const rectW = Math.round(naturalW * rectScale);
    const rectH = Math.round(naturalH * rectScale);

    const rectified = mat(rectifyBoard(colorMat, rectCorners, rectW, rectH));

    // Grid detection
    const rectGrayRaw = mat(new cv.Mat());
    cv.cvtColor(rectified, rectGrayRaw, cv.COLOR_RGBA2GRAY);
    const rectGrayBlurred = mat(new cv.Mat());
    cv.GaussianBlur(rectGrayRaw, rectGrayBlurred, new cv.Size(3, 3), 0);
    const rectGray = mat(enhanceGray(rectGrayBlurred, false));

    const circleSens = 24;
    const detection = detectGrid(rectGray, hintN, circleSens);
    if (!detection) {
      console.error(`  Grid detection failed for ${filePath}, skipping.`);
      continue;
    }

    const nRows = detection.rowPos.length;
    const nCols = detection.colPos.length;
    console.log(`  Detected ${nRows}x${nCols} grid`);

    // TPS dewarp
    const uniformIntersections = buildIntersections(
      detection.uniformRowPos, detection.uniformColPos,
      detection.rowAngle, detection.colAngle, rectGray.cols, rectGray.rows);
    const { snappedRows, snappedCols, snappedIntersections } = preSnapToCircles(
      detection.uniformRowPos, detection.uniformColPos,
      detection.rawCircles, uniformIntersections);

    const nR = snappedRows.length, nC = snappedCols.length;
    const stepY = nR > 1 ? (snappedRows[nR - 1] - snappedRows[0]) / (nR - 1) : 1;
    const stepX = nC > 1 ? (snappedCols[nC - 1] - snappedCols[0]) / (nC - 1) : 1;
    const uniStep = Math.max(stepX, stepY);
    const uniPad = uniStep;
    const uniRowY = Array.from({ length: nR }, (_, i) => uniPad + i * uniStep);
    const uniColX = Array.from({ length: nC }, (_, j) => uniPad + j * uniStep);
    const uniOutW = Math.round(2 * uniPad + (nC - 1) * uniStep);
    const uniOutH = Math.round(2 * uniPad + (nR - 1) * uniStep);

    const { yXs, yYs, xXs, xYs, tpsYPoints, tpsXPoints } = collectOffsetSamples(
      rectGray, snappedRows, snappedCols, 50, 125,
      detection.rawCircles, snappedIntersections,
      { rowY: uniRowY, colX: uniColX });
    const { yCoeffs, xCoeffs } = fitSeparableQuadratic(yXs, yYs, xXs, xYs);

    const tpsLambda = 0.1;
    const ransacThr = 2.9;
    const uni2snapY = u => snappedRows[0] + (u - uniPad) / uniStep * stepY;
    const uni2snapX = u => snappedCols[0] + (u - uniPad) / uniStep * stepX;
    const cleanY = ransacFilter(tpsYPoints, pt => uni2snapY(pt.y) + polyEval(yCoeffs, uni2snapY(pt.y)), ransacThr);
    const cleanX = ransacFilter(tpsXPoints, pt => uni2snapX(pt.x) + polyEval(xCoeffs, uni2snapX(pt.x)), ransacThr);
    const combined = buildCombinedGridPoints(cleanY, cleanX, snappedRows, snappedCols, tpsLambda, false, uniRowY, uniColX);
    const { gridY, gridX } = combined ?? { gridY: cleanY, gridX: cleanX };
    const tpsY = fitTPS(gridY, tpsLambda);
    const tpsX = fitTPS(gridX, tpsLambda);

    let dewarped, finalDetection;
    if (tpsY && tpsX) {
      dewarped = mat(dewarpImageTPS(rectified, tpsY, tpsX, uniOutW, uniOutH));
      finalDetection = combined
        ? buildDetectionFromControlPoints(combined, tpsY, tpsX, detection)
        : detection;
    } else {
      dewarped = mat(dewarpImage(rectified, yCoeffs, xCoeffs));
      finalDetection = detection;
    }

    // Classify stones
    const dewarpedGrayRaw = mat(new cv.Mat());
    cv.cvtColor(dewarped, dewarpedGrayRaw, cv.COLOR_RGBA2GRAY);
    const dewarpedGray = mat(enhanceGray(dewarpedGrayRaw, false));

    if (finalDetection === detection) {
      const detection2 = detectGrid(dewarpedGray, hintN, circleSens);
      if (detection2
        && detection2.rowPos.length === nRows
        && detection2.colPos.length === nCols) {
        finalDetection = detection2;
      }
    }

    const classResult = classifyStones(dewarpedGray, finalDetection.rowPos, finalDetection.colPos,
      finalDetection.step, finalDetection.rawCircles,
      finalDetection.intersections, true);

    // Build fixture
    const stones = [];
    for (const s of classResult.stones) {
      if (s.color !== '.' && s.r < nRows && s.c < nCols) {
        stones.push([s.r, s.c, s.color]);
      }
    }

    const fixture = {
      image: filePath,
      boardRows: nRows,
      boardCols: nCols,
      stones,
    };

    const name = basename(filePath).replace(/\.[^.]+$/, '');
    const outDir = dirname(filePath);
    const outPath = join(outDir, `${name}.fixture.json`);
    writeFileSync(outPath, JSON.stringify(fixture, null, 2) + '\n');
    console.log(`  Wrote ${outPath} (${stones.length} stones)`);

  } catch (err) {
    console.error(`  Error processing ${filePath}:`, err.message);
  } finally {
    toDelete.forEach(m => { try { m.delete(); } catch {} });
  }
}
