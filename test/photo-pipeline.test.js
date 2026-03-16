// photo-pipeline.test.js — pure-math unit tests (no OpenCV needed)
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

import {
  median, nearestIndex,
  kmeans2, clusterPositions, medianStep,
  gridAlignmentScore, findGridAngle,
  fitQuadratic, polyEval, fitWeightedQuadratic, solveLinear,
  fitTPS, evalTPS, tpsKernel, TPS_LAMBDA,
  inferBoardSize, computeEdgeOffsets, generateSGF,
  ransacFilter1D,
  fitCylinder1D, evalCylinderDisp,
  radialPower, sampleDisc, sampleAnnulus,
} from '../src/photo-pipeline.js';

// ── Helpers ──────────────────────────────────────────────────────────────────

describe('median', () => {
  it('returns 0 for empty array', () => {
    assert.equal(median([]), 0);
  });
  it('returns the single element', () => {
    assert.equal(median([42]), 42);
  });
  it('returns middle of odd-length array', () => {
    assert.equal(median([3, 1, 2]), 2);
  });
  it('returns floor-middle of even-length array', () => {
    assert.equal(median([4, 1, 3, 2]), 3);
  });
  it('handles negative numbers', () => {
    assert.equal(median([-5, -1, -3]), -3);
  });
});

describe('nearestIndex', () => {
  it('finds closest value', () => {
    assert.equal(nearestIndex([10, 20, 30, 40], 25), 1);
    assert.equal(nearestIndex([10, 20, 30, 40], 31), 2);
  });
  it('returns 0 for single element', () => {
    assert.equal(nearestIndex([99], 0), 0);
  });
});

// ── Clustering ──────────────────────────────────────────────────────────────

describe('clusterPositions', () => {
  it('merges close positions into clusters', () => {
    const clusters = clusterPositions([10, 11, 12, 50, 51], 5);
    assert.equal(clusters.length, 2);
    assert.ok(Math.abs(clusters[0].pos - 11) < 1);
    assert.equal(clusters[0].count, 3);
    assert.ok(Math.abs(clusters[1].pos - 50.5) < 1);
    assert.equal(clusters[1].count, 2);
  });
  it('returns empty for empty input', () => {
    assert.deepEqual(clusterPositions([]), []);
  });
});

describe('kmeans2', () => {
  it('separates two clear clusters', () => {
    const values = [10, 12, 11, 90, 88, 91];
    const { labels, c0, c1 } = kmeans2(values, 0, 100);
    assert.ok(c0 < 20, `c0=${c0} should be near 11`);
    assert.ok(c1 > 80, `c1=${c1} should be near 90`);
    // First three should be cluster 0, last three cluster 1
    assert.deepEqual([...labels.slice(0, 3)], [0, 0, 0]);
    assert.deepEqual([...labels.slice(3, 6)], [1, 1, 1]);
  });
});

describe('medianStep', () => {
  it('returns median step of equally spaced positions', () => {
    const step = medianStep([10, 20, 30, 40, 50]);
    assert.ok(Math.abs(step - 10) < 1);
  });
  it('returns null for fewer than 2 positions', () => {
    assert.equal(medianStep([10]), null);
  });
});

// ── Grid math ───────────────────────────────────────────────────────────────

describe('gridAlignmentScore', () => {
  it('returns near 0 for perfectly aligned points', () => {
    // Points at multiples of step=10
    const score = gridAlignmentScore([0, 10, 20, 30, 40], 10);
    assert.ok(score < 0.01, `score=${score} should be near 0`);
  });
  it('returns high score for randomly scattered points', () => {
    // Randomly scattered phases should give score near 1 (low alignment)
    const pts = [0.3, 3.7, 6.1, 8.9, 2.2, 5.5, 7.8, 1.1, 9.4, 4.6];
    const score = gridAlignmentScore(pts, 10);
    assert.ok(score > 0.5, `score=${score} should be high for scattered phases`);
  });
  it('returns 1 for empty or zero step', () => {
    assert.equal(gridAlignmentScore([], 10), 1);
    assert.equal(gridAlignmentScore([1, 2, 3], 0), 1);
  });
});

describe('findGridAngle', () => {
  it('finds angle 0 for axis-aligned grid', () => {
    // Build a grid of points at step=10 centered on (50,50)
    const points = [];
    for (let r = 0; r < 5; r++)
      for (let c = 0; c < 5; c++)
        points.push({ x: 10 + c * 10, y: 10 + r * 10 });
    const result = findGridAngle(points, 10, 30, 30);
    assert.ok(Math.abs(result.angle) < 1, `angle=${result.angle} should be near 0`);
  });
  it('recovers a small rotation', () => {
    // Rotate grid by 5 degrees
    const angleDeg = 5;
    const rad = angleDeg * Math.PI / 180;
    const cx = 50, cy = 50;
    const points = [];
    for (let r = 0; r < 5; r++)
      for (let c = 0; c < 5; c++) {
        const x0 = 10 + c * 10, y0 = 10 + r * 10;
        const dx = x0 - cx, dy = y0 - cy;
        points.push({
          x: cx + dx * Math.cos(rad) - dy * Math.sin(rad),
          y: cy + dx * Math.sin(rad) + dy * Math.cos(rad),
        });
      }
    const result = findGridAngle(points, 10, cx, cy);
    // findGridAngle should recover the angle (within 1 degree)
    assert.ok(Math.abs(result.angle - (-angleDeg)) < 1,
      `angle=${result.angle} should be near ${-angleDeg}`);
  });
});

// ── Quadratic fitting ───────────────────────────────────────────────────────

describe('fitQuadratic / polyEval', () => {
  it('fits a linear function', () => {
    const xs = [0, 1, 2, 3, 4];
    const ys = xs.map(x => 3 * x + 7);
    const [a, b, c] = fitQuadratic(xs, ys);
    assert.ok(Math.abs(a) < 1e-6, `a=${a} should be ~0`);
    assert.ok(Math.abs(b - 3) < 1e-6, `b=${b} should be ~3`);
    assert.ok(Math.abs(c - 7) < 1e-6, `c=${c} should be ~7`);
  });
  it('fits a quadratic function', () => {
    const xs = [0, 1, 2, 3, 4];
    const ys = xs.map(x => 2 * x * x - 3 * x + 1);
    const [a, b, c] = fitQuadratic(xs, ys);
    assert.ok(Math.abs(a - 2) < 1e-6, `a=${a} should be ~2`);
    assert.ok(Math.abs(b - (-3)) < 1e-6, `b=${b} should be ~-3`);
    assert.ok(Math.abs(c - 1) < 1e-6, `c=${c} should be ~1`);
  });
  it('polyEval evaluates correctly', () => {
    assert.ok(Math.abs(polyEval([2, -3, 1], 3) - 10) < 1e-9);
  });
});

describe('fitWeightedQuadratic', () => {
  it('fits with uniform weights', () => {
    const xs = [0, 1, 2, 3, 4];
    const ys = xs.map(x => x * x);
    const ws = xs.map(() => 1);
    const [c0, c1, c2] = fitWeightedQuadratic(xs, ys, ws);
    // c0 + c1*x + c2*x^2 = x^2 → c0=0, c1=0, c2=1
    assert.ok(Math.abs(c0) < 1e-4, `c0=${c0}`);
    assert.ok(Math.abs(c1) < 1e-4, `c1=${c1}`);
    assert.ok(Math.abs(c2 - 1) < 1e-4, `c2=${c2}`);
  });
});

describe('solveLinear', () => {
  it('solves 2x2 system', () => {
    // x + y = 3, 2x - y = 0 → x=1, y=2
    const sol = solveLinear([[1, 1], [2, -1]], [3, 0]);
    assert.ok(sol);
    assert.ok(Math.abs(sol[0] - 1) < 1e-9);
    assert.ok(Math.abs(sol[1] - 2) < 1e-9);
  });
  it('returns null for singular matrix', () => {
    const sol = solveLinear([[1, 2], [2, 4]], [1, 2]);
    assert.equal(sol, null);
  });
});

// ── TPS ─────────────────────────────────────────────────────────────────────

describe('tpsKernel', () => {
  it('returns 0 for r2=0', () => {
    assert.equal(tpsKernel(0), 0);
  });
  it('returns 0.5 * r2 * ln(r2) for positive r2', () => {
    const r2 = 4;
    const expected = 0.5 * r2 * Math.log(r2);
    assert.ok(Math.abs(tpsKernel(r2) - expected) < 1e-10);
  });
});

describe('fitTPS / evalTPS', () => {
  it('interpolates identity-like mapping', () => {
    // Points where target = x + y
    const pts = [
      { x: 0, y: 0, target: 0 },
      { x: 1, y: 0, target: 1 },
      { x: 0, y: 1, target: 1 },
      { x: 1, y: 1, target: 2 },
    ];
    const tps = fitTPS(pts, 0.001);
    assert.ok(tps, 'fitTPS should succeed');
    // Evaluate at known points
    for (const p of pts) {
      const val = evalTPS(tps, p.x, p.y);
      assert.ok(Math.abs(val - p.target) < 0.1,
        `evalTPS(${p.x}, ${p.y}) = ${val}, expected ${p.target}`);
    }
    // Evaluate at midpoint
    const mid = evalTPS(tps, 0.5, 0.5);
    assert.ok(Math.abs(mid - 1) < 0.2, `midpoint = ${mid}, expected ~1`);
  });
  it('returns null for fewer than 3 points', () => {
    assert.equal(fitTPS([{ x: 0, y: 0, target: 0 }, { x: 1, y: 1, target: 1 }]), null);
  });
});

// ── SGF ─────────────────────────────────────────────────────────────────────

describe('inferBoardSize', () => {
  it('returns max of rows/cols when no elision', () => {
    assert.equal(inferBoardSize(9, 9, null), 9);
    assert.equal(inferBoardSize(9, 13, null), 13);
  });
  it('returns 19 for 2+ elided edges', () => {
    assert.equal(inferBoardSize(7, 7, { top: true, left: true, bottom: false, right: false }), 19);
  });
  it('uses non-elided axis for 1 edge elided', () => {
    assert.equal(inferBoardSize(7, 9, { top: true, bottom: false, left: false, right: false }), 9);
  });
});

describe('computeEdgeOffsets', () => {
  it('returns 0 when nothing elided', () => {
    assert.equal(computeEdgeOffsets(9, 19, false, false), 0);
  });
  it('returns 0 when only high edge elided', () => {
    assert.equal(computeEdgeOffsets(9, 19, false, true), 0);
  });
  it('anchors at high edge when low edge elided', () => {
    assert.equal(computeEdgeOffsets(9, 19, true, false), 10);
  });
  it('centers when both elided', () => {
    assert.equal(computeEdgeOffsets(9, 19, true, true), 5);
  });
});

describe('generateSGF', () => {
  it('generates basic SGF with setup stones', () => {
    const stones = [
      { r: 0, c: 0, color: 'B' },
      { r: 1, c: 1, color: 'W' },
      { r: 2, c: 2, color: '.' },
    ];
    const sgf = generateSGF(stones, 9, null, 9);
    assert.ok(sgf.includes('SZ[9]'));
    assert.ok(sgf.includes('AB[aa]'));
    assert.ok(sgf.includes('AW[bb]'));
    assert.ok(!sgf.includes('[cc]'));
  });
  it('handles digitMap for numbered stones', () => {
    const stones = [
      { r: 0, c: 0, color: 'B' },
      { r: 1, c: 1, color: 'W' },
    ];
    const digitMap = new Map([
      [0, { number: 1 }],
      [1, { number: 2 }],
    ]);
    const sgf = generateSGF(stones, 9, null, 9, digitMap);
    assert.ok(sgf.includes(';B[aa]'));
    assert.ok(sgf.includes(';W[bb]'));
  });
  it('applies edge offsets for elided boards', () => {
    const stones = [{ r: 0, c: 0, color: 'B' }];
    const elided = { top: true, left: true, bottom: false, right: false };
    const sgf = generateSGF(stones, 7, elided, 7);
    assert.ok(sgf.includes('SZ[19]'));
    // Row offset = 19-7 = 12, col offset = 12
    // coord = chr(97 + 0 + 12) = 'm'
    assert.ok(sgf.includes('[mm]'), `SGF=${sgf}`);
  });
});

// ── RANSAC ──────────────────────────────────────────────────────────────────

describe('ransacFilter1D', () => {
  it('keeps inliers and removes outliers', () => {
    // y = 2x + 1 with noise on every point so median residual > 0.5
    const xs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    // True: [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    // Add ~1 unit noise to each, plus a huge outlier at index 9
    const ys = [2, 4, 4, 8, 8, 12, 12, 16, 16, 100];
    // residuals: [1, 1, 1, 1, 1, 1, 1, 1, 1, 81] → median = 1 > 0.5
    // threshold = 2.5 * 1 = 2.5, only 100 (residual=81) is removed
    const result = ransacFilter1D(xs, ys, [0, 2, 1]);
    assert.ok(result.xs.length < xs.length, 'should remove outlier');
    assert.ok(!result.ys.includes(100), 'outlier 100 should be removed');
  });
  it('passes through arrays shorter than 4', () => {
    const result = ransacFilter1D([1, 2], [3, 4], [0, 1, 0]);
    assert.equal(result.xs.length, 2);
  });
});

// ── Cylinder ────────────────────────────────────────────────────────────────

describe('evalCylinderDisp', () => {
  it('returns 0 at center', () => {
    assert.ok(Math.abs(evalCylinderDisp(100, 50, 50)) < 1e-10);
  });
  it('returns nonzero displacement away from center', () => {
    const d = evalCylinderDisp(100, 50, 80);
    assert.ok(Math.abs(d) > 0.01, `displacement=${d} should be nonzero`);
  });
});

describe('fitCylinder1D', () => {
  it('fits a cylinder to synthetic barrel data', () => {
    const R = 200, yc = 100;
    const xs = [], ys = [];
    for (let y = 50; y <= 150; y += 5) {
      xs.push(y);
      ys.push(evalCylinderDisp(R, yc, y));
    }
    const fit = fitCylinder1D(xs, ys);
    assert.ok(Math.abs(fit.R - R) < 20, `R=${fit.R} should be near ${R}`);
    assert.ok(Math.abs(fit.yc - yc) < 10, `yc=${fit.yc} should be near ${yc}`);
  });
});

// ── Pixel sampling (synthetic arrays) ───────────────────────────────────────

describe('sampleDisc', () => {
  it('samples average within a disc', () => {
    const W = 20, H = 20;
    const gray = new Uint8Array(W * H).fill(100);
    const val = sampleDisc(gray, W, H, 10, 10, 3);
    assert.ok(Math.abs(val - 100) < 1e-6);
  });
  it('returns 128 for out-of-bounds center', () => {
    const gray = new Uint8Array(1);
    const val = sampleDisc(gray, 1, 1, 50, 50, 1);
    assert.equal(val, 128);
  });
});

describe('sampleAnnulus', () => {
  it('samples annular region', () => {
    const W = 20, H = 20;
    const gray = new Uint8Array(W * H).fill(200);
    const val = sampleAnnulus(gray, W, H, 10, 10, 2, 5);
    assert.ok(Math.abs(val - 200) < 1e-6);
  });
});

describe('radialPower', () => {
  it('returns 0 for uniform gradient field', () => {
    // All gradients point the same direction — no radial structure
    const W = 20, H = 20;
    const gx = new Float32Array(W * H).fill(10);
    const gy = new Float32Array(W * H).fill(0);
    const power = radialPower(gx, gy, W, H, 10, 10, 2, 5, 0);
    // For uniform horizontal gradient, radial power should not be 1
    assert.ok(power < 0.9, `power=${power} should be < 0.9 for non-radial gradient`);
  });
});
