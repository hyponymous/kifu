// pipeline-defaults.ts — central definition of pipeline tunables
//
// Each key maps to { active, variants }.
//   active:   the value used by the pipeline (and ablation baseline)
//   variants: { experimentName: value } — alternatives tested in ablation
//
// TODO: Extend this to cover fns (DI function overrides) so that stage
//       ablations like no-quad-refine, no-enhance-gray, etc. are also
//       defined here instead of hand-written in the ablation script.

export interface TunableVariants<T> {
  active: T;
  variants: Record<string, T>;
}

export type ClassifierMode = 'kmeans' | 'onnx';

export interface PipelineDefaults {
  cannyLo:        TunableVariants<number>;
  cannyHi:        TunableVariants<number>;
  tpsLambda:      TunableVariants<number>;
  ransacThr:      TunableVariants<number>;
  reDetect:       TunableVariants<boolean>;
  combinedGrid:   TunableVariants<boolean>;
  skipTrimEdges:  TunableVariants<boolean>;
  rpThreshRatio:  TunableVariants<number>;
  gradFloor:      TunableVariants<number>;
  claheEnabled:    TunableVariants<boolean>;
  classifierMode:  TunableVariants<ClassifierMode>;
  houghBlurSize:   TunableVariants<number>;
  circleParam2:    TunableVariants<number>;
  multiCannyEnabled: TunableVariants<boolean>;
  minCannySupport:   TunableVariants<number>;
}

export const defaults: PipelineDefaults = {
  cannyLo:        { active: 50,       variants: { 'canny-30-80': 30, 'canny-70-180': 70 } },
  cannyHi:        { active: 125,      variants: { 'canny-30-80': 80, 'canny-70-180': 180 } },
  tpsLambda:      { active: 0.1,      variants: { 'tps-lambda-0.01': 0.01, 'tps-lambda-1.0': 1.0 } },
  ransacThr:      { active: 2.9,      variants: { 'ransac-1.5': 1.5, 'ransac-5.0': 5.0 } },
  reDetect:       { active: true,     variants: { 'no-re-detection': false } },
  combinedGrid:   { active: true,     variants: { 'no-combined-grid': false } },
  skipTrimEdges:  { active: true,     variants: { 'trim-edges': false } },
  rpThreshRatio:  { active: 0.70,     variants: { 'rp-thresh-0.60': 0.60, 'rp-thresh-0.85': 0.85 } },
  gradFloor:      { active: 64,       variants: { 'grad-floor-32': 32, 'grad-floor-128': 128 } },
  // CLAHE is off by default to preserve existing diagram fixture behavior.
  // Enable via ablation ('clahe' variant) or for real-board photo inputs.
  claheEnabled:   { active: false,    variants: { 'clahe': true } },
  classifierMode: { active: 'kmeans', variants: { 'onnx-classifier': 'onnx' } },
  // Gaussian blur kernel applied inside detectGrid before Canny→HoughLines.
  // Real-board photos have wood grain (5–10px features) that survives the
  // default 3×3 blur and produces hundreds of spurious Hough lines.
  // Larger kernels suppress grain while preserving the long grid-line edges.
  houghBlurSize:  { active: 3,        variants: { 'hough-blur-5': 5, 'hough-blur-7': 7, 'hough-blur-9': 9 } },
  // HoughCircles accumulator threshold (param2). Lower = more circles detected,
  // higher = fewer false positives. Real stones (3D with specular highlights) need
  // a lower threshold than flat diagram circles.
  circleParam2:   { active: 24,       variants: { 'circle-p2-12': 12, 'circle-p2-16': 16 } },
  // Multi-threshold Canny: compute edges at several (lo, hi) pairs, use
  // per-pixel support count to separate structural edges from noise.
  // Off by default to preserve existing diagram fixture behavior.
  multiCannyEnabled: { active: false,  variants: { 'multi-canny': true } },
  // Minimum number of threshold layers that must agree for a pixel to be
  // considered an edge when multiCannyEnabled is true. Range: 1..5.
  minCannySupport:   { active: 3,      variants: { 'min-support-2': 2, 'min-support-4': 4 } },
};

export interface ActiveDefaults {
  cannyLo: number;
  cannyHi: number;
  tpsLambda: number;
  ransacThr: number;
  reDetect: boolean;
  combinedGrid: boolean;
  skipTrimEdges: boolean;
  rpThreshRatio: number;
  gradFloor: number;
  claheEnabled: boolean;
  classifierMode: ClassifierMode;
  houghBlurSize: number;
  circleParam2: number;
  multiCannyEnabled: boolean;
  minCannySupport: number;
}

/** Return a plain { key: active } object for use as pipeline opts. */
export function activeDefaults(): ActiveDefaults {
  return {
    cannyLo:        defaults.cannyLo.active,
    cannyHi:        defaults.cannyHi.active,
    tpsLambda:      defaults.tpsLambda.active,
    ransacThr:      defaults.ransacThr.active,
    reDetect:       defaults.reDetect.active,
    combinedGrid:   defaults.combinedGrid.active,
    skipTrimEdges:  defaults.skipTrimEdges.active,
    rpThreshRatio:  defaults.rpThreshRatio.active,
    gradFloor:      defaults.gradFloor.active,
    claheEnabled:   defaults.claheEnabled.active,
    classifierMode: defaults.classifierMode.active,
    houghBlurSize:  defaults.houghBlurSize.active,
    circleParam2:   defaults.circleParam2.active,
    multiCannyEnabled: defaults.multiCannyEnabled.active,
    minCannySupport:   defaults.minCannySupport.active,
  };
}
