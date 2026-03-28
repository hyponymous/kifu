// opencv.d.ts — global cv shim for opencv.js loaded via <script> tag
// Provides minimal type declarations for the OpenCV functions used in this project.

interface CvMat {
  rows: number;
  cols: number;
  data: Uint8Array;
  data32S: Int32Array;
  data32F: Float32Array;
  ucharAt(row: number, col: number): number;
  floatAt(row: number, col: number): number;
  doubleAt(row: number, col: number): number;
  roi(rect: CvRect): CvMat;
  clone(): CvMat;
  copyTo(dst: CvMat, mask?: CvMat): void;
  setTo(value: CvScalar, mask?: CvMat): void;
  delete(): void;
}

interface CvRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

interface CvSize {
  width: number;
  height: number;
}

interface CvScalar {
  [index: number]: number;
}

interface CvMatVector {
  size(): number;
  get(i: number): CvMat;
  delete(): void;
}

interface CvModule {
  Mat: {
    new(): CvMat;
    new(rows: number, cols: number, type: number): CvMat;
    ones(rows: number, cols: number, type: number): CvMat;
    zeros(rows: number, cols: number, type: number): CvMat;
  };
  MatVector: { new(): CvMatVector };
  Size: { new(width: number, height: number): CvSize };
  Rect: { new(x: number, y: number, width: number, height: number): CvRect };
  Point: { new(x: number, y: number): { x: number; y: number } };
  Scalar: { new(v0: number, v1?: number, v2?: number, v3?: number): CvScalar };

  matFromArray(rows: number, cols: number, type: number, data: number[]): CvMat;
  matFromImageData(imageData: ImageData): CvMat;

  // Color conversion
  cvtColor(src: CvMat, dst: CvMat, code: number): void;
  COLOR_RGBA2GRAY: number;
  COLOR_GRAY2RGBA: number;

  // Image I/O
  imread(canvas: HTMLCanvasElement): CvMat;
  imshow(canvas: HTMLCanvasElement | string, mat: CvMat): void;

  // Edge/gradient
  Sobel(src: CvMat, dst: CvMat, ddepth: number, dx: number, dy: number, ksize?: number): void;

  // Filtering
  GaussianBlur(src: CvMat, dst: CvMat, ksize: CvSize, sigmaX: number): void;
  blur(src: CvMat, dst: CvMat, ksize: CvSize): void;
  medianBlur(src: CvMat, dst: CvMat, ksize: number): void;
  bilateralFilter(src: CvMat, dst: CvMat, d: number, sigmaColor: number, sigmaSpace: number): void;
  Canny(src: CvMat, dst: CvMat, threshold1: number, threshold2: number): void;

  // Morphology
  dilate(src: CvMat, dst: CvMat, kernel: CvMat, anchor?: { x: number; y: number }, iterations?: number): void;
  erode(src: CvMat, dst: CvMat, kernel: CvMat, anchor?: { x: number; y: number }, iterations?: number): void;
  getStructuringElement(shape: number, ksize: CvSize): CvMat;
  MORPH_ELLIPSE: number;
  MORPH_RECT: number;

  // Features
  goodFeaturesToTrack(image: CvMat, corners: CvMat, maxCorners: number, qualityLevel: number, minDistance: number, mask?: CvMat, blockSize?: number, useHarrisDetector?: boolean, k?: number): void;
  cornerHarris(src: CvMat, dst: CvMat, blockSize: number, ksize: number, k: number): void;
  HoughLines(src: CvMat, lines: CvMat, rho: number, theta: number, threshold: number): void;
  HoughCircles(src: CvMat, circles: CvMat, method: number, dp: number, minDist: number, param1?: number, param2?: number, minRadius?: number, maxRadius?: number): void;
  HOUGH_GRADIENT: number;

  // Contours
  findContours(image: CvMat, contours: CvMatVector, hierarchy: CvMat, mode: number, method: number): void;
  contourArea(contour: CvMat): number;
  arcLength(contour: CvMat, closed: boolean): number;
  approxPolyDP(curve: CvMat, approxCurve: CvMat, epsilon: number, closed: boolean): void;
  RETR_EXTERNAL: number;
  RETR_LIST: number;
  CHAIN_APPROX_SIMPLE: number;
  CHAIN_APPROX_NONE: number;
  boundingRect(contour: CvMat): CvRect;
  convexHull(src: CvMat, dst: CvMat): void;
  fitEllipse(points: CvMat): { center: { x: number; y: number }; size: { width: number; height: number }; angle: number };

  // Geometric transformations
  getPerspectiveTransform(src: CvMat, dst: CvMat): CvMat;
  warpPerspective(src: CvMat, dst: CvMat, M: CvMat, dsize: CvSize): void;
  perspectiveTransform(src: CvMat, dst: CvMat, M: CvMat): void;
  remap(src: CvMat, dst: CvMat, map1: CvMat, map2: CvMat, interpolation: number, borderMode?: number): void;
  resize(src: CvMat, dst: CvMat, dsize: CvSize, fx?: number, fy?: number, interpolation?: number): void;
  INTER_LINEAR: number;
  INTER_CUBIC: number;
  BORDER_REPLICATE: number;

  // Histogram
  equalizeHist(src: CvMat, dst: CvMat): void;
  createCLAHE(clipLimit?: number, tileGridSize?: CvSize): { apply(src: CvMat, dst: CvMat): void; delete(): void };

  // Thresholding
  threshold(src: CvMat, dst: CvMat, thresh: number, maxval: number, type: number): void;
  adaptiveThreshold(src: CvMat, dst: CvMat, maxValue: number, adaptiveMethod: number, thresholdType: number, blockSize: number, C: number): void;
  THRESH_BINARY: number;
  THRESH_OTSU: number;
  ADAPTIVE_THRESH_GAUSSIAN_C: number;

  // Drawing
  circle(img: CvMat, center: { x: number; y: number }, radius: number, color: CvScalar, thickness?: number): void;

  // Bitwise
  bitwise_and(src1: CvMat, src2: CvMat, dst: CvMat, mask?: CvMat): void;
  bitwise_not(src: CvMat, dst: CvMat): void;

  // Core
  normalize(src: CvMat, dst: CvMat, alpha: number, beta: number, normType: number, dtype?: number): void;
  NORM_MINMAX: number;
  add(src1: CvMat, src2: CvMat, dst: CvMat): void;
  subtract(src1: CvMat, src2: CvMat, dst: CvMat): void;
  addWeighted(src1: CvMat, alpha: number, src2: CvMat, beta: number, gamma: number, dst: CvMat): void;

  // Mat type constants
  CV_8U: number;
  CV_8UC1: number;
  CV_8UC4: number;
  CV_32F: number;
  CV_32FC1: number;
  CV_32FC2: number;
  CV_32SC1: number;
  CV_32SC2: number;
  CV_64FC1: number;
}

declare const cv: CvModule;
