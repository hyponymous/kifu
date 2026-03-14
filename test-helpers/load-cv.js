// load-cv.js — loads opencv-wasm (CJS) and sets globalThis.cv
// Must be imported (statically) before any dynamic import of photo-pipeline.js.
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { cv } = require('opencv-wasm');
globalThis.cv = cv;

export { cv };
