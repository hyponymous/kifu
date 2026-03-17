// onnx-classifier.ts — lazy-loading ONNX stone classifier
//
// Architecture:
//   Input:  [N, 1, 32, 32] float32 tensor (grayscale patches, normalized 0–1)
//   Output: [N, 3] float32 logits → argmax → { 0: '.', 1: 'B', 2: 'W' }
//
// Usage:
//   const classifier = await loadOnnxClassifier('/models/stone-classifier.onnx');
//   const colors = classifier.classify(patches, 32);  // synchronous after load
//   // pass as opts.classifyIntersections to runPipeline

// Model path — override via loadOnnxClassifier(url) argument.
const DEFAULT_MODEL_URL = '/models/stone-classifier.onnx';

// Label order must match training script (scripts/train-classifier.py)
const LABELS: Array<'.' | 'B' | 'W'> = ['.', 'B', 'W'];

export interface OnnxClassifier {
  /**
   * Classify a batch of grayscale intersection patches.
   * @param patches  Array of Uint8Array patches (patchSize×patchSize, row-major)
   * @param patchSize  Side length of each square patch (default 32)
   * @returns Array of 'B' | 'W' | '.' in the same order as patches
   */
  classify(patches: Uint8Array[], patchSize: number): Array<'.' | 'B' | 'W'>;
}

/**
 * Lazy-load onnxruntime-web and create an OnnxClassifier.
 * The ONNX Runtime WASM bundle is fetched on first call and cached.
 *
 * @param modelUrl  URL to the .onnx model weights file
 */
export async function loadOnnxClassifier(modelUrl: string = DEFAULT_MODEL_URL): Promise<OnnxClassifier> {
  // Dynamic import so onnxruntime-web is not included in the main bundle
  // until actually needed (same lazy-load pattern as OpenCV.js).
  // onnxruntime-web is an optional peer dependency — install with:
  //   npm install onnxruntime-web
  // Using new Function to defer module resolution so the file compiles without
  // the package installed. Replace with a static import once onnxruntime-web
  // is added to package.json dependencies.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any -- deferred optional peer dep
  const ort: any = await new Function('return import("onnxruntime-web")')(); // unknown until installed

  const session = await ort.InferenceSession.create(modelUrl, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });

  return {
    classify(patches: Uint8Array[], patchSize: number): Array<'.' | 'B' | 'W'> {
      const N = patches.length;
      if (N === 0) return [];

      // Build [N, 1, patchSize, patchSize] float32 tensor, normalized 0–1
      const pixelsPerPatch = patchSize * patchSize;
      const inputData = new Float32Array(N * pixelsPerPatch);
      for (let i = 0; i < N; i++) {
        const patch = patches[i];
        const offset = i * pixelsPerPatch;
        for (let j = 0; j < pixelsPerPatch; j++) {
          inputData[offset + j] = patch[j] / 255.0;
        }
      }

      const inputTensor = new ort.Tensor('float32', inputData, [N, 1, patchSize, patchSize]);

      // ort.InferenceSession.run is async; we can't make this synchronous.
      // Callers must await the result at a higher level. This method is
      // intentionally kept synchronous-looking — actual sync execution requires
      // pre-running inference outside runPipeline and passing results via
      // forcedClassification (future work). For now, classify() throws if
      // called from a synchronous context without a pre-computed result.
      //
      // TODO: make runPipeline async and await classify() here, or introduce
      //       a forcedClassification opt to pass pre-computed results in.
      throw new Error(
        'OnnxClassifier.classify() is not yet integrated into runPipeline. ' +
        'See onnx-classifier.ts for integration notes.'
      );
      // The code below is the intended implementation once runPipeline is async:
      //
      // const feeds = { input: inputTensor };
      // const results = await session.run(feeds);
      // const logits = results.output.data as Float32Array;
      // return Array.from({ length: N }, (_, i) => {
      //   const base = i * 3;
      //   const argmax = logits[base] >= logits[base + 1] && logits[base] >= logits[base + 2] ? 0
      //                : logits[base + 1] >= logits[base + 2] ? 1 : 2;
      //   return LABELS[argmax];
      // });
    },
  };
}
