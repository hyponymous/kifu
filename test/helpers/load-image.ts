// load-image.js — decodes an image file into cv.Mat objects for testing
// Requires globalThis.cv to be set (import load-cv.js first).
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const sharp = require('sharp');

export async function loadImage(filePath: string) {
  const img = sharp(filePath).rotate(); // auto-rotate per EXIF
  const { width, height } = await img.metadata();
  const rgba = await img.ensureAlpha().raw().toBuffer({ resolveWithObject: true });

  const w = rgba.info.width;
  const h = rgba.info.height;
  const colorMat = new cv.Mat(h, w, cv.CV_8UC4);
  colorMat.data.set(rgba.data);

  const grayMat = new cv.Mat();
  cv.cvtColor(colorMat, grayMat, cv.COLOR_RGBA2GRAY);

  return { colorMat, grayMat, width: w, height: h };
}
