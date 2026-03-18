#!/usr/bin/env node
// photo-pipeline-ablation.js — ablation studies and parameter sweeps
// Runs the photo pipeline with stage toggles and parameter overrides,
// records timing and accuracy to evals/ablation.eval.jsonl.
//
// Usage:
//   node photo-pipeline-ablation.js                    # run all experiments
//   node photo-pipeline-ablation.js baseline no-tps    # run named subset

import '../test/helpers/load-cv';
import { loadImage } from '../test/helpers/load-image';
import { runPipeline } from '../src/run-pipeline';
import { refineQuadWithHough } from '../src/photo-pipeline';
import type { Point, Circle, TPSPoint } from '../src/photo-pipeline';
import { defaults } from '../src/pipeline-defaults';
import { readdirSync, readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { execSync } from 'node:child_process';
import { performance } from 'node:perf_hooks';

// ── Fixture discovery ───────────────────────────────────────────────────────

const fixtureDir = 'fixtures';
const fixtureFiles = readdirSync(fixtureDir)
  .filter(f => f.endsWith('.fixture.json'))
  .map(f => ({
    name: f.replace('.fixture.json', ''),
    path: join(fixtureDir, f),
    data: JSON.parse(readFileSync(join(fixtureDir, f), 'utf8')),
  }));

if (fixtureFiles.length === 0) {
  console.error('No fixture files found. Run generate-fixtures.js first.');
  process.exit(1);
}

// ── Experiment definitions ──────────────────────────────────────────────────

// TODO: Move fns (DI function overrides) into pipeline-defaults.js so that
// stage ablations are also defined centrally and auto-generated here.

// Stage-off fns overrides for DI (manual until TODO above is done)
const FNS_EXPERIMENTS = [
  { name: 'no-quad-refine',      opts: { fns: { refineQuadWithHough: (corners: Point[]) => corners } } },
  { name: 'no-refine-rejection', opts: { fns: { refineQuadWithHough: (corners: Point[], edges: CvMat) => refineQuadWithHough(corners, edges, false) } } },
  { name: 'no-enhance-gray',     opts: { fns: { enhanceGray: (mat: CvMat, _: boolean) => mat } } },
  { name: 'no-pre-snap',         opts: { fns: { preSnapToCircles: (rows: number[], cols: number[], _circles: Circle[], intersections: Point[][]) => ({
    snappedRows: rows, snappedCols: cols, snappedIntersections: intersections }) } } },
  { name: 'no-grid-bounds',      opts: { fns: { findGridBounds: (gray: CvMat) => ({ x: 0, y: 0, width: gray.cols, height: gray.rows }) } } },
  { name: 'no-tps',              opts: { fns: { fitTPS: () => null } } },
  { name: 'no-ransac',           opts: { fns: { ransacFilter: (pts: TPSPoint[]) => pts } } },
];

// Auto-generate parameter experiments from pipeline-defaults.js.
// Groups variants that share an experiment name (e.g. cannyLo+cannyHi for 'canny-30-80').
function buildParamExperiments() {
  const byName = new Map();
  for (const [key, { variants }] of Object.entries(defaults)) {
    for (const [expName, value] of Object.entries(variants)) {
      if (!byName.has(expName)) byName.set(expName, {});
      byName.get(expName)[key] = value;
    }
  }
  return [...byName.entries()].map(([name, opts]) => ({ name, opts }));
}

const EXPERIMENTS = [
  { name: 'baseline', opts: {} },
  ...FNS_EXPERIMENTS,
  ...buildParamExperiments(),
];

// ── Grid error computation ──────────────────────────────────────────────────

function computeGridErrors(detected: { r: number; c: number; x: number; y: number }[], groundTruth: [number, number][], nRows: number, nCols: number) {
  if (groundTruth.length !== nRows * nCols) return null;
  const errors = [];
  for (const d of detected) {
    const idx = d.r * nCols + d.c;
    const [gx, gy] = groundTruth[idx];
    errors.push(Math.hypot(d.x - gx, d.y - gy));
  }
  errors.sort((a, b) => a - b);
  const mean = errors.reduce((s, e) => s + e, 0) / errors.length;
  const max = errors[errors.length - 1];
  const p95idx = Math.min(Math.floor(errors.length * 0.95), errors.length - 1);
  return { gridErrorMean: +mean.toFixed(2), gridErrorMax: +max.toFixed(2), gridErrorP95: +errors[p95idx].toFixed(2) };
}

// ── Git info ────────────────────────────────────────────────────────────────

function getGitInfo() {
  try {
    const gitHash = execSync('git rev-parse --short HEAD', { encoding: 'utf8' }).trim();
    const dirty = execSync('git diff HEAD', { encoding: 'utf8' }).length > 0;
    return { gitHash, dirty };
  } catch {
    return { gitHash: 'unknown', dirty: false };
  }
}

// ── Main ────────────────────────────────────────────────────────────────────

async function main() {
  const args = process.argv.slice(2);
  const selectedNames = args.length > 0 ? new Set(args) : null;
  const experiments = selectedNames
    ? EXPERIMENTS.filter(e => selectedNames.has(e.name))
    : EXPERIMENTS;

  if (experiments.length === 0) {
    console.error(`No matching experiments. Available: ${EXPERIMENTS.map(e => e.name).join(', ')}`);
    process.exit(1);
  }

  const git = getGitInfo();
  console.log(`Running ${experiments.length} experiments × ${fixtureFiles.length} fixtures (git: ${git.gitHash}${git.dirty ? ' dirty' : ''})\n`);

  const records = [];

  for (const experiment of experiments) {
    const fixtureResults: Record<string, any> = {};
    const matchRates: number[] = [];
    const gridErrorMeans: number[] = [];
    const timings: Record<string, any[]> = {};
    let failures = 0;

    process.stdout.write(`  ${experiment.name.padEnd(22)}`);
    const expStart = performance.now();

    for (const fixture of fixtureFiles) {
      const image = await loadImage(fixture.data.image);

      const timing: Record<string, any> = {};
      let result;
      try {
        result = runPipeline(image, {
          ...experiment.opts,
          onStage: (name, ms) => { timing[name] = ms; },
        });
      } finally {
        image.colorMat.delete();
        image.grayMat.delete();
      }

      const quarantined = !!fixture.data.quarantined;

      if (!result) {
        fixtureResults[fixture.name] = { matchRate: 0, mismatches: -1, timing, ...(quarantined ? { quarantined: true } : {}) };
        if (!quarantined) { matchRates.push(0); failures++; }
        continue;
      }

      // Check dimensions match
      const expectedRows = fixture.data.boardRows;
      const expectedCols = fixture.data.boardCols;
      if (result.nRows !== expectedRows || result.nCols !== expectedCols) {
        fixtureResults[fixture.name] = { matchRate: 0, mismatches: -1, timing, ...(quarantined ? { quarantined: true } : {}) };
        if (!quarantined) { matchRates.push(0); failures++; }
        continue;
      }

      // Compare stones
      const expectedGrid = Array.from({ length: expectedRows }, () => Array(expectedCols).fill('.'));
      for (const [r, c, color] of fixture.data.stones) {
        expectedGrid[r][c] = color;
      }

      let total = 0, matches = 0;
      for (let r = 0; r < expectedRows; r++) {
        for (let c = 0; c < expectedCols; c++) {
          total++;
          if (result.grid[r][c] === expectedGrid[r][c]) matches++;
        }
      }

      const matchRate = +(matches / total).toFixed(4);
      const mismatches = total - matches;
      if (!quarantined) matchRates.push(matchRate);

      const entry: Record<string, unknown> = { matchRate, mismatches, timing, ...(quarantined ? { quarantined: true } : {}) };

      // Grid errors
      if (fixture.data.intersections) {
        const gridErrors = computeGridErrors(
          result.detectedIntersections, fixture.data.intersections,
          expectedRows, expectedCols);
        if (gridErrors) {
          Object.assign(entry, gridErrors);
          gridErrorMeans.push(gridErrors.gridErrorMean);
        }
      }

      fixtureResults[fixture.name] = entry;

      // Accumulate timings
      for (const [stage, ms] of Object.entries(timing)) {
        if (!timings[stage]) timings[stage] = [];
        timings[stage].push(ms);
      }
    }

    const expMs = +(performance.now() - expStart).toFixed(0);

    // Aggregate
    const aggregate: Record<string, any> = {
      matchRateMean: matchRates.length > 0
        ? +(matchRates.reduce((s, v) => s + v, 0) / matchRates.length).toFixed(4)
        : 0,
      failures,
    };
    if (gridErrorMeans.length > 0) {
      aggregate.gridErrorMean = +(gridErrorMeans.reduce((s, v) => s + v, 0) / gridErrorMeans.length).toFixed(2);
    }

    const timingMean: Record<string, number> = {};
    for (const [stage, vals] of Object.entries(timings)) {
      timingMean[stage] = +(vals.reduce((s: number, v: number) => s + v, 0) / vals.length).toFixed(1);
    }
    aggregate.timingMean = timingMean;

    const record = {
      timestamp: new Date().toISOString(),
      gitHash: git.gitHash,
      dirty: git.dirty,
      experiment: experiment.name,
      overrides: experiment.opts,
      fixtures: fixtureResults,
      aggregate,
    };
    records.push(record);

    // Progress line
    const mr = (aggregate.matchRateMean * 100).toFixed(1) + '%';
    const ge = aggregate.gridErrorMean != null ? aggregate.gridErrorMean.toFixed(1) + 'px' : 'n/a';
    const total = Object.values(timingMean).reduce((s, v) => s + v, 0);
    const tm = total ? total.toFixed(0) + 'ms' : 'n/a';
    const fail = failures > 0 ? ` (${failures} failed)` : '';
    console.log(`match=${mr}  grid=${ge}  time=${tm}  [${expMs}ms]${fail}`);
  }

  // Write results
  mkdirSync('evals', { recursive: true });
  const evalPath = 'evals/ablation.eval.jsonl';
  const lines = records.map(r => JSON.stringify(r)).join('\n') + '\n';
  writeFileSync(evalPath, lines, { flag: 'a' });
  console.log(`\nWrote ${records.length} records to ${evalPath}`);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
