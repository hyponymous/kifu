// photo-pipeline-e2e.test.js — end-to-end tests using fixture ground truth
import './helpers/load-cv';
import { loadImage } from './helpers/load-image';
import { runPipeline } from '../src/run-pipeline';
import { describe, it, after } from 'node:test';
import assert from 'node:assert/strict';
import { readdirSync, readFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { execSync } from 'node:child_process';
import { createHash } from 'node:crypto';

// Discover fixture files
const fixtureDir = 'fixtures';
const fixtureFiles = readdirSync(fixtureDir)
  .filter(f => f.endsWith('.fixture.json'))
  .map(f => ({
    name: f.replace('.fixture.json', ''),
    path: join(fixtureDir, f),
    data: JSON.parse(readFileSync(join(fixtureDir, f), 'utf8')),
  }));

if (fixtureFiles.length === 0) {
  console.log('No fixture files found. Run generate-fixtures.js first.');
}

const MATCH_THRESHOLD = 0.85;

// ── Eval recording ──────────────────────────────────────────────────────────

type GridErrors = { gridErrorMean: number; gridErrorMax: number; gridErrorP95: number };
type EvalEntry = { matchRate: number; mismatches: number; gridErrors?: GridErrors | null };
const evalResults: Record<string, EvalEntry> = {};

function computeGridErrors(detected: { r: number; c: number; x: number; y: number }[], groundTruth: [number, number][], nRows: number, nCols: number) {
  // groundTruth is flat array of [x, y] in row-major order
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
  const p95 = errors[p95idx];

  return { gridErrorMean: +mean.toFixed(2), gridErrorMax: +max.toFixed(2), gridErrorP95: +p95.toFixed(2) };
}

function getGitInfo() {
  try {
    const gitHash = execSync('git rev-parse --short HEAD', { encoding: 'utf8' }).trim();
    const diff = execSync('git diff HEAD', { encoding: 'utf8' });
    const dirty = diff.length > 0;
    let diffHash = undefined;
    let diffFile = undefined;
    if (dirty) {
      diffHash = 'sha256:' + createHash('sha256').update(diff).digest('hex').slice(0, 16);
      const ts = Date.now();
      const diffDir = 'evals/diffs';
      mkdirSync(diffDir, { recursive: true });
      diffFile = `${diffDir}/${gitHash}-${ts}.patch`;
      writeFileSync(diffFile, diff);
    }
    return { gitHash, dirty, diffHash, diffFile };
  } catch {
    return { gitHash: 'unknown', dirty: false };
  }
}

function writeEvalRecord() {
  const git = getGitInfo();
  const fixtureScores: Record<string, object> = {};
  const matchRates = [];
  const gridErrorMeans = [];
  const gridErrorMaxes = [];

  for (const [name, data] of Object.entries(evalResults)) {
    fixtureScores[name] = { matchRate: data.matchRate, mismatches: data.mismatches };
    matchRates.push(data.matchRate);
    if (data.gridErrors) {
      Object.assign(fixtureScores[name], data.gridErrors);
      gridErrorMeans.push(data.gridErrors.gridErrorMean);
      gridErrorMaxes.push(data.gridErrors.gridErrorMax);
    }
  }

  const aggregate: { matchRateMean: number; gridErrorMean?: number; gridErrorMax?: number } = {
    matchRateMean: +(matchRates.reduce((s, v) => s + v, 0) / matchRates.length).toFixed(4),
  };
  if (gridErrorMeans.length > 0) {
    aggregate.gridErrorMean = +(gridErrorMeans.reduce((s, v) => s + v, 0) / gridErrorMeans.length).toFixed(2);
    aggregate.gridErrorMax = +Math.max(...gridErrorMaxes).toFixed(2);
  }

  const record = {
    timestamp: new Date().toISOString(),
    gitHash: git.gitHash,
    dirty: git.dirty,
    ...(git.diffHash ? { diffHash: git.diffHash } : {}),
    ...(git.diffFile ? { diffFile: git.diffFile } : {}),
    fixtures: fixtureScores,
    aggregate,
  };

  mkdirSync('evals', { recursive: true });
  const evalPath = 'evals/photo-pipeline.eval.jsonl';
  writeFileSync(evalPath, JSON.stringify(record) + '\n', { flag: 'a' });

  // Print summary table
  console.log('\n┌─────────────────────────────────────────────────────────────────────┐');
  console.log('│                    Photo Pipeline Eval Summary                     │');
  console.log('├──────────────────┬───────────┬───────┬───────────┬────────┬────────┤');
  console.log('│ Fixture          │ MatchRate │ Miss  │ GridMean  │ GridMax│ GridP95│');
  console.log('├──────────────────┼───────────┼───────┼───────────┼────────┼────────┤');
  for (const [name, data] of Object.entries(evalResults)) {
    const mr = (data.matchRate * 100).toFixed(1).padStart(6) + '%';
    const miss = String(data.mismatches).padStart(5);
    const gm = data.gridErrors ? data.gridErrors.gridErrorMean.toFixed(2).padStart(8) : '     n/a';
    const gx = data.gridErrors ? data.gridErrors.gridErrorMax.toFixed(2).padStart(6) : '   n/a';
    const gp = data.gridErrors ? data.gridErrors.gridErrorP95.toFixed(2).padStart(6) : '   n/a';
    console.log(`│ ${name.padEnd(16)} │ ${mr} │ ${miss} │ ${gm} │ ${gx} │ ${gp} │`);
  }
  console.log('├──────────────────┼───────────┼───────┼───────────┼────────┼────────┤');
  const amr = (aggregate.matchRateMean * 100).toFixed(1).padStart(6) + '%';
  const agm = aggregate.gridErrorMean != null ? aggregate.gridErrorMean.toFixed(2).padStart(8) : '     n/a';
  const agx = aggregate.gridErrorMax != null ? aggregate.gridErrorMax.toFixed(2).padStart(6) : '   n/a';
  console.log(`│ ${'AGGREGATE'.padEnd(16)} │ ${amr} │       │ ${agm} │ ${agx} │        │`);
  console.log('└──────────────────┴───────────┴───────┴───────────┴────────┴────────┘');
  console.log(`  Saved to ${evalPath} (git: ${git.gitHash}${git.dirty ? ' dirty' : ''})`);
}

// ── Tests ───────────────────────────────────────────────────────────────────

describe('photo pipeline e2e', () => {
  for (const fixture of fixtureFiles) {
    it(`matches fixture: ${fixture.name}`, async () => {
      const imagePath = fixture.data.image;
      const image = await loadImage(imagePath);
      const result = runPipeline(image);
      assert.ok(result, `Pipeline failed for ${imagePath}`);

      const expectedRows = fixture.data.boardRows;
      const expectedCols = fixture.data.boardCols;
      assert.equal(result.nRows, expectedRows,
        `Board rows: got ${result.nRows}, expected ${expectedRows}`);
      assert.equal(result.nCols, expectedCols,
        `Board cols: got ${result.nCols}, expected ${expectedCols}`);

      // Build expected grid
      const expectedGrid = Array.from({ length: expectedRows }, () =>
        Array(expectedCols).fill('.'));
      for (const [r, c, color] of fixture.data.stones) {
        expectedGrid[r][c] = color;
      }

      // Compare stones
      let total = 0, matches = 0;
      const mismatches = [];
      for (let r = 0; r < expectedRows; r++) {
        for (let c = 0; c < expectedCols; c++) {
          total++;
          if (result.grid[r][c] === expectedGrid[r][c]) {
            matches++;
          } else {
            mismatches.push(`(${r},${c}): got ${result.grid[r][c]}, expected ${expectedGrid[r][c]}`);
          }
        }
      }

      const matchRate = matches / total;
      const pct = (matchRate * 100).toFixed(1);
      if (mismatches.length > 0 && mismatches.length <= 10) {
        console.log(`  ${fixture.name}: ${pct}% match (${mismatches.length} mismatches: ${mismatches.join(', ')})`);
      } else if (mismatches.length > 10) {
        console.log(`  ${fixture.name}: ${pct}% match (${mismatches.length} mismatches, showing first 10: ${mismatches.slice(0, 10).join(', ')})`);
      }

      // Record eval data
      const evalData: EvalEntry = { matchRate, mismatches: mismatches.length };
      if (fixture.data.intersections) {
        evalData.gridErrors = computeGridErrors(
          result.detectedIntersections, fixture.data.intersections,
          expectedRows, expectedCols);
      }
      evalResults[fixture.name] = evalData;

      assert.ok(matchRate >= MATCH_THRESHOLD,
        `Match rate ${pct}% is below threshold ${MATCH_THRESHOLD * 100}% for ${fixture.name}`);
    });
  }

  after(() => {
    if (Object.keys(evalResults).length > 0) {
      writeEvalRecord();
    }
  });
});
