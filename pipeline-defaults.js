// pipeline-defaults.js — central definition of pipeline tunables
//
// Each key maps to { active, variants }.
//   active:   the value used by the pipeline (and ablation baseline)
//   variants: { experimentName: value } — alternatives tested in ablation
//
// TODO: Extend this to cover fns (DI function overrides) so that stage
//       ablations like no-quad-refine, no-enhance-gray, etc. are also
//       defined here instead of hand-written in the ablation script.

export const defaults = {
  cannyLo:       { active: 50,    variants: { 'canny-30-80': 30, 'canny-70-180': 70 } },
  cannyHi:       { active: 125,   variants: { 'canny-30-80': 80, 'canny-70-180': 180 } },
  tpsLambda:     { active: 0.1,   variants: { 'tps-lambda-0.01': 0.01, 'tps-lambda-1.0': 1.0 } },
  ransacThr:     { active: 2.9,   variants: { 'ransac-1.5': 1.5, 'ransac-5.0': 5.0 } },
  reDetect:      { active: true,  variants: { 'no-re-detection': false } },
  combinedGrid:  { active: true,  variants: { 'no-combined-grid': false } },
  skipTrimEdges: { active: true,  variants: { 'trim-edges': false } },
  rpThreshRatio: { active: 0.85,  variants: { 'rp-thresh-0.70': 0.70, 'rp-thresh-0.95': 0.95 } },
  gradFloor:     { active: 64,    variants: { 'grad-floor-32': 32, 'grad-floor-128': 128 } },
};

/** Return a plain { key: active } object for use as pipeline opts. */
export function activeDefaults() {
  const out = {};
  for (const [k, v] of Object.entries(defaults)) out[k] = v.active;
  return out;
}
