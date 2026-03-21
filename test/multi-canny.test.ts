// multi-canny.test.ts — unit tests for multiThresholdCanny
import './helpers/load-cv';
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  multiThresholdCanny,
  thresholdCannyStack,
  deleteCannyStack,
  DEFAULT_CANNY_THRESHOLDS,
} from '../src/photo-pipeline';

describe('multiThresholdCanny', () => {
  it('returns support map with correct dimensions', () => {
    // Create a 100x100 black image with a bright white cross
    const img = cv.Mat.zeros(100, 100, cv.CV_8U);
    // Draw horizontal and vertical white lines (strong edges)
    for (let x = 0; x < 100; x++) img.data[50 * 100 + x] = 255;
    for (let y = 0; y < 100; y++) img.data[y * 100 + 50] = 255;

    const stack = multiThresholdCanny(img, DEFAULT_CANNY_THRESHOLDS);

    assert.equal(stack.support.rows, 100);
    assert.equal(stack.support.cols, 100);
    assert.equal(stack.layers.length, DEFAULT_CANNY_THRESHOLDS.length);
    assert.deepStrictEqual(stack.thresholds, DEFAULT_CANNY_THRESHOLDS);

    deleteCannyStack(stack);
    img.delete();
  });

  it('strong edges have higher support than weak edges', () => {
    // Strong edge: white line on black background
    const img = cv.Mat.zeros(100, 100, cv.CV_8U);
    for (let x = 10; x < 90; x++) img.data[50 * 100 + x] = 255;

    const stack = multiThresholdCanny(img, DEFAULT_CANNY_THRESHOLDS);

    // Find max support value — should be high for the strong edge
    let maxSupport = 0;
    const data = stack.support.data;
    for (let i = 0; i < data.length; i++) {
      if (data[i] > maxSupport) maxSupport = data[i];
    }
    // Strong edge should be detected by multiple threshold pairs
    assert.ok(maxSupport >= 3, `max support ${maxSupport} should be >= 3`);

    deleteCannyStack(stack);
    img.delete();
  });

  it('blank image has zero support everywhere', () => {
    const img = cv.Mat.zeros(50, 50, cv.CV_8U);
    const stack = multiThresholdCanny(img, DEFAULT_CANNY_THRESHOLDS);

    let totalSupport = 0;
    const data = stack.support.data;
    for (let i = 0; i < data.length; i++) totalSupport += data[i];
    assert.equal(totalSupport, 0);

    deleteCannyStack(stack);
    img.delete();
  });

  it('thresholdCannyStack produces binary edge map', () => {
    const img = cv.Mat.zeros(100, 100, cv.CV_8U);
    for (let x = 10; x < 90; x++) img.data[50 * 100 + x] = 255;

    const stack = multiThresholdCanny(img, DEFAULT_CANNY_THRESHOLDS);
    const edges = thresholdCannyStack(stack, 3);

    // Result should be binary: only 0 and 255
    const data = edges.data;
    for (let i = 0; i < data.length; i++) {
      assert.ok(data[i] === 0 || data[i] === 255, `pixel ${i} = ${data[i]}`);
    }

    edges.delete();
    deleteCannyStack(stack);
    img.delete();
  });

  it('higher minSupport yields fewer edge pixels', () => {
    const img = cv.Mat.zeros(100, 100, cv.CV_8U);
    // Draw gradient edge (some areas strong, some weak)
    for (let x = 0; x < 100; x++) {
      const val = Math.round((x / 100) * 255);
      for (let y = 45; y < 55; y++) img.data[y * 100 + x] = val;
    }

    const stack = multiThresholdCanny(img, DEFAULT_CANNY_THRESHOLDS);
    const loEdges = thresholdCannyStack(stack, 1);
    const hiEdges = thresholdCannyStack(stack, 4);

    let loCount = 0, hiCount = 0;
    for (let i = 0; i < loEdges.data.length; i++) {
      if (loEdges.data[i] > 0) loCount++;
      if (hiEdges.data[i] > 0) hiCount++;
    }

    assert.ok(hiCount <= loCount, `hi-support ${hiCount} should be <= lo-support ${loCount}`);

    loEdges.delete();
    hiEdges.delete();
    deleteCannyStack(stack);
    img.delete();
  });
});
