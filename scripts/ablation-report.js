#!/usr/bin/env node
// ablation-report.js — reads evals/ablation.eval.jsonl and prints comparison tables
//
//
// Usage:
//   node ablation-report.js                        # latest run per experiment
//   node ablation-report.js --fixture IMG_0983     # per-fixture drill-down
//   node ablation-report.js --timing               # show per-stage timing breakdown
//   node ablation-report.js --all                  # show all runs, not just latest

import { readFileSync } from 'node:fs';

const evalPath = 'evals/ablation.eval.jsonl';

let lines;
try {
  lines = readFileSync(evalPath, 'utf8').trim().split('\n').filter(Boolean);
} catch {
  console.error(`No ablation data found at ${evalPath}. Run: npm run ablation`);
  process.exit(1);
}

const records = lines.map(l => JSON.parse(l));

// Parse args
const args = process.argv.slice(2);
const fixtureFilter = args.includes('--fixture') ? args[args.indexOf('--fixture') + 1] : null;
const showTiming = args.includes('--timing');
const showAll = args.includes('--all');

// Get latest record per experiment (or all)
const byExperiment = new Map();
for (const r of records) {
  if (!showAll) {
    byExperiment.set(r.experiment, r);
  } else {
    if (!byExperiment.has(r.experiment)) byExperiment.set(r.experiment, []);
    byExperiment.get(r.experiment).push(r);
  }
}

if (fixtureFilter) {
  printFixtureDrillDown(fixtureFilter);
} else if (showTiming) {
  printTimingBreakdown();
} else {
  printSummaryTable();
}

function printSummaryTable() {
  const baseline = byExperiment.get('baseline');

  console.log('');
  console.log('Ablation Study Results');
  console.log('='.repeat(78));

  if (baseline) {
    console.log(`Baseline: git ${baseline.gitHash} @ ${baseline.timestamp}`);
  }
  console.log('');

  const header = padRow('Experiment', 'Match%', 'Fail', 'GridErr', 'Δ vs baseline');
  console.log(header);
  console.log('─'.repeat(78));

  const baseMatch = baseline?.aggregate?.matchRateMean;
  const baseFail = baseline?.aggregate?.failures ?? 0;
  const baseGrid = baseline?.aggregate?.gridErrorMean;

  for (const [name, record] of byExperiment) {
    const r = showAll ? record[record.length - 1] : record;
    const matchPct = (r.aggregate.matchRateMean * 100).toFixed(2) + '%';
    const fail = r.aggregate.failures ?? 0;
    const failStr = fail > 0 ? String(fail) : '-';
    const gridErr = r.aggregate.gridErrorMean != null ? r.aggregate.gridErrorMean.toFixed(2) + 'px' : 'n/a';

    let delta = '---';
    if (name !== 'baseline' && baseMatch != null) {
      const parts = [];
      const dFail = fail - baseFail;
      if (dFail > 0) parts.push(`+${dFail} FAIL`);
      const dMatch = r.aggregate.matchRateMean - baseMatch;
      if (Math.abs(dMatch) >= 0.0001) {
        parts.push((dMatch > 0 ? '+' : '') + (dMatch * 100).toFixed(2) + '%');
      }
      if (r.aggregate.gridErrorMean != null && baseGrid != null) {
        const dGrid = r.aggregate.gridErrorMean - baseGrid;
        if (Math.abs(dGrid) >= 0.01) {
          parts.push((dGrid > 0 ? '+' : '') + dGrid.toFixed(2) + 'px');
        }
      }
      if (dFail < 0) parts.push(`${dFail} fail`);
      delta = parts.join(', ') || '~same';
    }

    console.log(padRow(name, matchPct, failStr, gridErr, delta));
  }

  console.log('');
  console.log(`${byExperiment.size} experiments from ${evalPath}`);
}

function printFixtureDrillDown(fixtureName) {
  console.log('');
  console.log(`Fixture drill-down: ${fixtureName}`);
  console.log('='.repeat(78));
  console.log('');

  const header = padRow('Experiment', 'Match%', 'Mismatch', 'GridErr', 'Time');
  console.log(header);
  console.log('─'.repeat(78));

  for (const [name, record] of byExperiment) {
    const r = showAll ? record[record.length - 1] : record;
    const fix = r.fixtures?.[fixtureName];
    if (!fix) {
      console.log(padRow(name, 'n/a', 'n/a', 'n/a', 'n/a'));
      continue;
    }
    const failed = fix.mismatches < 0;
    const matchPct = failed ? 'FAIL' : (fix.matchRate * 100).toFixed(2) + '%';
    const mismatches = failed ? '-' : String(fix.mismatches);
    const gridErr = fix.gridErrorMean != null ? fix.gridErrorMean.toFixed(2) + 'px' : 'n/a';
    const time = fix.timing?.total ? fix.timing.total.toFixed(0) + 'ms' : 'n/a';
    console.log(padRow(name, matchPct, mismatches, gridErr, time));
  }
  console.log('');
}

function printTimingBreakdown() {
  console.log('');
  console.log('Per-stage timing breakdown (mean ms across fixtures)');
  console.log('='.repeat(100));
  console.log('');

  // Collect all stage names from baseline
  const baseline = byExperiment.get('baseline');
  const br = showAll && baseline ? baseline[baseline.length - 1] : baseline;
  const stages = br?.aggregate?.timingMean
    ? Object.keys(br.aggregate.timingMean).filter(s => s !== 'total')
    : [];

  const cols = ['Experiment', ...stages, 'Total'];
  const widths = cols.map(c => Math.max(c.length, 10));
  widths[0] = 22;

  // Header
  console.log(cols.map((c, i) => c.padEnd(widths[i])).join(' │ '));
  console.log(widths.map(w => '─'.repeat(w)).join('─┼─'));

  for (const [name, record] of byExperiment) {
    const r = showAll ? record[record.length - 1] : record;
    const tm = r.aggregate.timingMean || {};
    const vals = [name, ...stages.map(s => tm[s] != null ? tm[s].toFixed(1) : '-'), tm.total != null ? tm.total.toFixed(0) : '-'];
    console.log(vals.map((v, i) => String(v).padEnd(widths[i])).join(' │ '));
  }
  console.log('');
}

function padRow(col1, col2, col3, col4, col5) {
  return [
    String(col1).padEnd(22),
    String(col2).padStart(8),
    String(col3).padStart(8),
    String(col4).padStart(8),
    col5 != null ? '  ' + String(col5) : '',
  ].join(' │ ');
}
