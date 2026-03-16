import { encode }           from './encode.js';
import { decode, download } from './decode.js';
import { parse }            from './sgf-parser.js';
import { renderGoban, replayMain } from './goban.js';
import { runPipeline, computeInversePerspective, applyHomography } from './run-pipeline.js';
import { inferBoardSize, computeEdgeOffsets, fitTPS, evalTPS } from './photo-pipeline.js';

const board       = document.getElementById('board');
const gameInfo    = document.getElementById('game-info');
const statusEl    = document.getElementById('status');
const errorEl     = document.getElementById('error');
const sharingEl   = document.getElementById('sharing');
const shareLinkEl = document.getElementById('share-link');
const downloadBtn = document.getElementById('download-btn');
const qrModal     = document.getElementById('qr-modal');
const qrDisplay   = document.getElementById('qr-display');
const copyBtn     = document.getElementById('copy-btn');
const qrBtn       = document.getElementById('qr-btn');
const sgfPanel    = document.getElementById('sgf-panel');
const textarea    = document.getElementById('sgf-input');
const fileInput   = document.getElementById('file-input');
const dropZone    = document.getElementById('drop-zone');

let currentFilename  = 'game.sgf';
let encodeTimer      = null;
let constructMode    = false;
let constructStones  = new Map(); // `${r},${c}` → 'B' | 'W'
let constructColor   = 'B';
let constructCols    = 19;
let constructRows    = 19;

const constructEntryEl   = document.getElementById('construct-entry');
const constructToolbarEl = document.getElementById('construct-toolbar');

// ── Photo pipeline ────────────────────────────────────────────────────────

let cvReady = null;
function loadCV() {
  if (cvReady) return cvReady;
  cvReady = new Promise((resolve, reject) => {
    window.Module = { onRuntimeInitialized: resolve };
    const s = document.createElement('script');
    s.src = 'https://docs.opencv.org/4.x/opencv.js';
    s.async = true;
    s.onerror = reject;
    document.head.appendChild(s);
  });
  return cvReady;
}

function imageToMats(img) {
  const canvas = document.getElementById('photo-canvas');
  canvas.width  = img.naturalWidth  || img.width;
  canvas.height = img.naturalHeight || img.height;
  canvas.getContext('2d').drawImage(img, 0, 0);
  const colorMat = cv.imread(canvas);
  const grayMat  = new cv.Mat();
  cv.cvtColor(colorMat, grayMat, cv.COLOR_RGBA2GRAY);
  return { colorMat, grayMat, width: canvas.width, height: canvas.height };
}

function drawPhotoOverlay(canvas, { intersections, step }, grid, nRows, nCols) {
  const ctx = canvas.getContext('2d');

  // Grid lines
  ctx.strokeStyle = 'rgba(0,255,80,0.75)';
  ctx.lineWidth = 1.5;
  for (let r = 0; r < nRows; r++) {
    ctx.beginPath();
    let started = false;
    for (let c = 0; c < nCols; c++) {
      const pt = intersections[r]?.[c];
      if (!pt) continue;
      started ? ctx.lineTo(pt.x, pt.y) : ctx.moveTo(pt.x, pt.y);
      started = true;
    }
    ctx.stroke();
  }
  for (let c = 0; c < nCols; c++) {
    ctx.beginPath();
    let started = false;
    for (let r = 0; r < nRows; r++) {
      const pt = intersections[r]?.[c];
      if (!pt) continue;
      started ? ctx.lineTo(pt.x, pt.y) : ctx.moveTo(pt.x, pt.y);
      started = true;
    }
    ctx.stroke();
  }

  // Stone circles
  const radius = Math.max(3, step * 0.38);
  for (let r = 0; r < nRows; r++) {
    for (let c = 0; c < nCols; c++) {
      const pt = intersections[r]?.[c];
      const v = grid[r]?.[c];
      if (!pt || (v !== 'B' && v !== 'W')) continue;
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, radius, 0, Math.PI * 2);
      ctx.fillStyle = v === 'B' ? 'rgba(17,17,17,0.75)' : 'rgba(240,240,240,0.75)';
      ctx.fill();
      ctx.strokeStyle = v === 'B' ? 'rgba(255,255,255,0.35)' : '#333';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }
}

// ── Grid dot rendering ──────────────────────────────────────────────────
function drawGridDots(canvas) {
  if (!gridDots || !adjustGridMode || !lastPhotoResult) return;
  const ctx = canvas.getContext('2d');
  const step = savedOverlay ? savedOverlay.step : 30;
  const dotRadius = Math.max(3, step * 0.3);
  const nR = lastPhotoResult.nRows, nC = lastPhotoResult.nCols;

  // Build a lookup from (r,c) to dot for grid lines
  const dotAt = Array.from({ length: nR }, () => []);
  for (const dot of gridDots) dotAt[dot.r][dot.c] = dot;

  // Grid lines through dot positions
  ctx.strokeStyle = 'rgba(0,255,80,0.75)';
  ctx.lineWidth = 1.5;
  for (let r = 0; r < nR; r++) {
    ctx.beginPath();
    let started = false;
    for (let c = 0; c < nC; c++) {
      const d = dotAt[r][c];
      if (!d) continue;
      started ? ctx.lineTo(d.x, d.y) : ctx.moveTo(d.x, d.y);
      started = true;
    }
    ctx.stroke();
  }
  for (let c = 0; c < nC; c++) {
    ctx.beginPath();
    let started = false;
    for (let r = 0; r < nR; r++) {
      const d = dotAt[r][c];
      if (!d) continue;
      started ? ctx.lineTo(d.x, d.y) : ctx.moveTo(d.x, d.y);
      started = true;
    }
    ctx.stroke();
  }

  // Stone markers at dot positions
  const stoneRadius = Math.max(3, step * 0.38);
  for (const dot of gridDots) {
    const v = lastPhotoResult.grid[dot.r]?.[dot.c];
    if (v !== 'B' && v !== 'W') continue;
    ctx.beginPath();
    ctx.arc(dot.x, dot.y, stoneRadius, 0, Math.PI * 2);
    ctx.fillStyle = v === 'B' ? 'rgba(17,17,17,0.75)' : 'rgba(240,240,240,0.75)';
    ctx.fill();
    ctx.strokeStyle = v === 'B' ? 'rgba(255,255,255,0.35)' : '#333';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  // Control point dots (pinned / hovered only)
  for (let i = 0; i < gridDots.length; i++) {
    const dot = gridDots[i];
    const isPinned = pinnedDots.has(i);
    const isHovered = (i === hoveredDot);

    if (!isPinned && !isHovered) continue;

    if (isPinned) {
      const half = dotRadius * 0.85;
      ctx.fillStyle = 'rgba(124, 92, 191, 0.7)';
      ctx.strokeStyle = 'rgba(160, 130, 220, 0.9)';
      ctx.lineWidth = 1.5;
      ctx.fillRect(dot.x - half, dot.y - half, half * 2, half * 2);
      ctx.strokeRect(dot.x - half, dot.y - half, half * 2, half * 2);
    } else {
      ctx.beginPath();
      ctx.arc(dot.x, dot.y, dotRadius, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(0, 200, 80, 0.55)';
      ctx.strokeStyle = 'rgba(0, 255, 100, 0.8)';
      ctx.lineWidth = 1.5;
      ctx.fill();
      ctx.stroke();
    }
  }
}

function canvasFromEvent(e, canvas) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const clientX = e.touches ? e.touches[0].clientX : e.clientX;
  const clientY = e.touches ? e.touches[0].clientY : e.clientY;
  return {
    x: (clientX - rect.left) * scaleX,
    y: (clientY - rect.top) * scaleY,
  };
}

function findNearestDot(cx, cy) {
  if (!gridDots) return -1;
  const step = savedOverlay ? savedOverlay.step : 30;
  const hitRadius = Math.max(6, step * 0.45);
  let bestIdx = -1, bestDist = hitRadius;
  for (let i = 0; i < gridDots.length; i++) {
    const d = Math.hypot(gridDots[i].x - cx, gridDots[i].y - cy);
    if (d < bestDist) { bestDist = d; bestIdx = i; }
  }
  return bestIdx;
}

// ── TPS fit for grid adjustment ─────────────────────────────────────────
function fitGridTPS() {
  if (!gridDots || pinnedDots.size < 3) { lastTpsFit = null; return; }
  const ptsX = [], ptsY = [];
  for (const idx of pinnedDots) {
    const dot = gridDots[idx];
    ptsX.push({ x: dot.r, y: dot.c, target: dot.x });
    ptsY.push({ x: dot.r, y: dot.c, target: dot.y });
  }
  const tpsX = fitTPS(ptsX, 0.001);
  const tpsY = fitTPS(ptsY, 0.001);
  if (!tpsX || !tpsY) { lastTpsFit = null; return; }
  lastTpsFit = { tpsX, tpsY };
}

function applyTpsFit() {
  if (!lastTpsFit) return;
  for (let i = 0; i < gridDots.length; i++) {
    if (pinnedDots.has(i)) continue;
    gridDots[i].x = evalTPS(lastTpsFit.tpsX, gridDots[i].r, gridDots[i].c);
    gridDots[i].y = evalTPS(lastTpsFit.tpsY, gridDots[i].r, gridDots[i].c);
  }
}

function refitGrid() {
  fitGridTPS();
  applyTpsFit();
}

function redrawPhotoPreview() {
  if (!lastPhotoResult || !savedOverlay) return;
  const previewCanvas = document.getElementById('photo-preview');
  // Redraw rectified image
  const img = previewCanvas._rectifiedImg;
  if (img) {
    const ctx = previewCanvas.getContext('2d');
    ctx.putImageData(img, 0, 0);
  }
  if (adjustGridMode && gridDots) {
    // In adjust mode: draw only dots (with grid curves from TPS), no pipeline overlay
    drawGridDots(previewCanvas);
  } else {
    drawPhotoOverlay(previewCanvas, savedOverlay, lastPhotoResult.grid, lastPhotoResult.nRows, lastPhotoResult.nCols);
  }
}

function buildOverlayFromDots() {
  if (!gridDots || !lastPhotoResult) return savedOverlay;
  const nR = lastPhotoResult.nRows, nC = lastPhotoResult.nCols;
  const intersections = Array.from({ length: nR }, () => []);
  for (const dot of gridDots) {
    intersections[dot.r][dot.c] = { x: dot.x, y: dot.y };
  }
  return { intersections, step: savedOverlay.step };
}

// ── Reclassify after grid adjustment ────────────────────────────────────
async function reclassifyWithAdjustedGrid() {
  if (!gridDots || !lastPhotoResult || !lastPhotoFile) return;

  // Convert rectified-coord dots to original image coords via inverse perspective
  const H = computeInversePerspective(
    lastPhotoResult.rectCorners, lastPhotoResult.rectW, lastPhotoResult.rectH);
  const originalCoordDots = gridDots.map(dot => {
    const orig = applyHomography(H, dot.x, dot.y);
    return { x: orig.x, y: orig.y, r: dot.r, c: dot.c };
  });

  try {
    const img = new Image();
    img.src = URL.createObjectURL(lastPhotoFile);
    await new Promise((res, rej) => { img.onload = res; img.onerror = rej; });
    const mats = imageToMats(img);
    URL.revokeObjectURL(img.src);

    const previewCanvas = document.getElementById('photo-preview');

    const result = runPipeline(mats, {
      forceRows: lastPhotoResult.nRows,
      forceCols: lastPhotoResult.nCols,
      rectCorners: lastPhotoResult.rectCorners,
      forcedGrid: originalCoordDots,
      onIntermediate(name, data) {
        // Skip rectification display — corners are locked, image is cached
        if (name === 'detectGrid' && data.detection?.intersections) {
          savedOverlay = {
            intersections: data.detection.intersections.map(row =>
              row.map(pt => ({ x: pt.x, y: pt.y }))
            ),
            step: data.detection.step,
          };
        }
      },
    });

    mats.colorMat.delete();
    mats.grayMat.delete();

    if (!result) return;

    lastPhotoResult = result;
    const detectedEdges = result.elidedEdges
      ?? { top: false, bottom: false, left: false, right: false };
    // Preserve user overrides for edges
    currentElidedEdges = { ...detectedEdges, ...currentElidedEdges };

    // Rebuild gridDots from new overlay
    if (savedOverlay) {
      const newDots = [];
      for (let r = 0; r < result.nRows; r++) {
        for (let c = 0; c < result.nCols; c++) {
          const pt = savedOverlay.intersections[r]?.[c];
          if (pt) newDots.push({ x: pt.x, y: pt.y, r, c });
        }
      }
      gridDots = newDots;
    }

    // Redraw from cached rectified image — no canvas flash
    redrawPhotoPreview();
    renderPhotoBoard(result, currentElidedEdges, document.getElementById('photo-board-preview'));
    updateEdgeToggles(currentElidedEdges);
  } catch (e) {
    console.error('Reclassification error:', e);
  }
}

// ── Enter/exit adjust grid mode ──────────────────────────────────────────
function enterAdjustMode() {
  if (!savedOverlay || !lastPhotoResult) return;
  adjustGridMode = true;
  photoAdjustEl.textContent = 'Exit adjust';

  // Build gridDots from savedOverlay
  gridDots = [];
  const nR = lastPhotoResult.nRows, nC = lastPhotoResult.nCols;
  for (let r = 0; r < nR; r++) {
    for (let c = 0; c < nC; c++) {
      const pt = savedOverlay.intersections[r]?.[c];
      if (pt) gridDots.push({ x: pt.x, y: pt.y, r, c });
    }
  }
  // Auto-pin the four corners
  pinnedDots.clear();
  pinnedDots.add(0);                           // top-left
  pinnedDots.add(nC - 1);                      // top-right
  pinnedDots.add((nR - 1) * nC);               // bottom-left
  pinnedDots.add(nR * nC - 1);                 // bottom-right
  lastTpsFit = null;
  hoveredDot = -1;
  draggingDot = -1;

  const previewCanvas = document.getElementById('photo-preview');
  // _rectifiedImg was already cached during processPhoto (clean, no overlay)
  previewCanvas.classList.add('adjust-grid');
  attachGridListeners(previewCanvas);
  redrawPhotoPreview();
}

function exitAdjustMode() {
  adjustGridMode = false;
  photoAdjustEl.textContent = 'Adjust grid';
  hoveredDot = -1;
  draggingDot = -1;

  const previewCanvas = document.getElementById('photo-preview');
  previewCanvas.classList.remove('adjust-grid');
  detachGridListeners(previewCanvas);
  redrawPhotoPreview();
}

// ── Grid interaction handlers ────────────────────────────────────────────
let gridListenersAttached = false;
const gridHandlers = {};

function attachGridListeners(canvas) {
  if (gridListenersAttached) return;
  gridListenersAttached = true;

  gridHandlers.mousemove = e => {
    if (draggingDot >= 0) {
      const pos = canvasFromEvent(e, canvas);
      const dx = pos.x - dragStart.x;
      const dy = pos.y - dragStart.y;
      if (Math.abs(dx) > 1 || Math.abs(dy) > 1) didDrag = true;
      gridDots[draggingDot].x = dragDotStart.x + dx;
      gridDots[draggingDot].y = dragDotStart.y + dy;
      refitGrid();
      redrawPhotoPreview();
      return;
    }
    // Hover detection
    const pos = canvasFromEvent(e, canvas);
    const idx = findNearestDot(pos.x, pos.y);
    if (idx !== hoveredDot) {
      hoveredDot = idx;
      redrawPhotoPreview();
    }
  };

  gridHandlers.mousedown = e => {
    const pos = canvasFromEvent(e, canvas);
    const idx = findNearestDot(pos.x, pos.y);
    if (idx < 0) return;
    e.preventDefault();
    if (!pinnedDots.has(idx)) {
      pinnedDots.add(idx);
      refitGrid();
      redrawPhotoPreview();
    }
    draggingDot = idx;
    dragStart = pos;
    dragDotStart = { x: gridDots[idx].x, y: gridDots[idx].y };
    didDrag = false;
  };

  gridHandlers.mouseup = () => {
    if (draggingDot >= 0 && didDrag) {
      reclassifyWithAdjustedGrid();
    } else if (draggingDot >= 0 && !didDrag && pinnedDots.has(draggingDot)) {
      // Click without drag on pinned dot → unpin
      pinnedDots.delete(draggingDot);
      refitGrid();
      reclassifyWithAdjustedGrid();
    }
    draggingDot = -1;
    dragStart = null;
    dragDotStart = null;
  };

  gridHandlers.touchstart = e => {
    if (e.touches.length !== 1) return;
    const pos = canvasFromEvent(e, canvas);
    const idx = findNearestDot(pos.x, pos.y);
    if (idx < 0) return;
    e.preventDefault();
    if (!pinnedDots.has(idx)) {
      pinnedDots.add(idx);
      refitGrid();
      redrawPhotoPreview();
    }
    draggingDot = idx;
    dragStart = pos;
    dragDotStart = { x: gridDots[idx].x, y: gridDots[idx].y };
    didDrag = false;
  };

  gridHandlers.touchmove = e => {
    if (draggingDot < 0 || e.touches.length !== 1) return;
    e.preventDefault();
    const pos = canvasFromEvent(e, canvas);
    const dx = pos.x - dragStart.x;
    const dy = pos.y - dragStart.y;
    if (Math.abs(dx) > 1 || Math.abs(dy) > 1) didDrag = true;
    gridDots[draggingDot].x = dragDotStart.x + dx;
    gridDots[draggingDot].y = dragDotStart.y + dy;
    refitGrid();
    redrawPhotoPreview();
  };

  gridHandlers.touchend = e => {
    if (e.touches.length > 0) return;
    if (draggingDot >= 0 && didDrag) {
      reclassifyWithAdjustedGrid();
    } else if (draggingDot >= 0 && !didDrag && pinnedDots.has(draggingDot)) {
      pinnedDots.delete(draggingDot);
      refitGrid();
      reclassifyWithAdjustedGrid();
    }
    draggingDot = -1;
    dragStart = null;
    dragDotStart = null;
  };

  canvas.addEventListener('mousemove', gridHandlers.mousemove);
  canvas.addEventListener('mousedown', gridHandlers.mousedown);
  window.addEventListener('mouseup', gridHandlers.mouseup);
  canvas.addEventListener('touchstart', gridHandlers.touchstart, { passive: false });
  canvas.addEventListener('touchmove', gridHandlers.touchmove, { passive: false });
  canvas.addEventListener('touchend', gridHandlers.touchend);
}

function detachGridListeners(canvas) {
  if (!gridListenersAttached) return;
  gridListenersAttached = false;
  canvas.removeEventListener('mousemove', gridHandlers.mousemove);
  canvas.removeEventListener('mousedown', gridHandlers.mousedown);
  window.removeEventListener('mouseup', gridHandlers.mouseup);
  canvas.removeEventListener('touchstart', gridHandlers.touchstart);
  canvas.removeEventListener('touchmove', gridHandlers.touchmove);
  canvas.removeEventListener('touchend', gridHandlers.touchend);
}

function photoResultToSGF(result, elidedEdges) {
  const boardN = elidedEdges
    ? inferBoardSize(result.nRows, result.nCols, elidedEdges)
    : Math.max(result.nRows, result.nCols);
  const rowOff = elidedEdges
    ? computeEdgeOffsets(result.nRows, boardN, elidedEdges.top, elidedEdges.bottom) : 0;
  const colOff = elidedEdges
    ? computeEdgeOffsets(result.nCols, boardN, elidedEdges.left, elidedEdges.right) : 0;

  const ab = [], aw = [];
  for (let r = 0; r < result.nRows; r++) {
    for (let c = 0; c < result.nCols; c++) {
      const v = result.grid[r][c];
      if (v !== 'B' && v !== 'W') continue;
      const coord = String.fromCharCode(97 + c + colOff) + String.fromCharCode(97 + r + rowOff);
      (v === 'B' ? ab : aw).push(`[${coord}]`);
    }
  }
  let p = `SZ[${boardN}]`;
  if (ab.length) p += `AB${ab.join('')}`;
  if (aw.length) p += `AW${aw.join('')}`;
  return `(;${p})`;
}

function renderPhotoBoard(result, elidedEdges, container) {
  renderGoban(parse(photoResultToSGF(result, elidedEdges)), container, { forceFullBoard: true, jitter: 0 });
}

let lastPhotoFile      = null;
let lastPhotoResult    = null;
let savedOverlay       = null; // { intersections, step } in rectified coords
let photoRerunTimer    = null;
let currentElidedEdges = null; // user-overridable elided edges

// ── Adjust grid state ──────────────────────────────────────────────────
let adjustGridMode = false;
let gridDots = null;         // flat array of { x, y, r, c } in rectified coords
let pinnedDots = new Set();  // indices into gridDots
let lastTpsFit = null;       // { tpsX, tpsY } or null
let hoveredDot = -1;
let draggingDot = -1;
let dragStart = null;        // { x, y } canvas coords at drag start
let dragDotStart = null;     // { x, y } dot position at drag start
let didDrag = false;

const photoModalEl  = document.getElementById('photo-modal');
const photoStatusEl = document.getElementById('photo-status');
const photoDimsEl   = document.getElementById('photo-dims-row');
const photoEdgesEl  = document.getElementById('photo-edges-row');
const photoBodyEl   = document.getElementById('photo-modal-body');
const photoAcceptEl = document.getElementById('photo-accept-btn');
const photoAdjustEl = document.getElementById('photo-adjust-btn');

function openPhotoModal() {
  photoModalEl.hidden  = false;
  photoStatusEl.hidden = false;
  if (!lastPhotoResult) {
    photoDimsEl.hidden    = true;
    photoEdgesEl.hidden   = true;
    photoBodyEl.hidden    = true;
    photoAcceptEl.hidden  = true;
    photoAdjustEl.hidden  = true;
  }
}

function closePhotoModal() {
  if (adjustGridMode) exitAdjustMode();
  photoModalEl.hidden = true;
}

photoAdjustEl.addEventListener('click', () => {
  if (adjustGridMode) exitAdjustMode();
  else enterAdjustMode();
});

async function processPhoto(file, hintN = 0, { forceRows = 0, forceCols = 0 } = {}) {
  lastPhotoFile = file;
  openPhotoModal();
  photoStatusEl.textContent = 'Loading OpenCV\u2026';
  try {
    await loadCV();
    const img = new Image();
    img.src = URL.createObjectURL(file);
    await new Promise((res, rej) => { img.onload = res; img.onerror = rej; });
    const mats = imageToMats(img);
    URL.revokeObjectURL(img.src);
    photoStatusEl.textContent = 'Detecting board\u2026';

    // Reset adjust mode for new pipeline run
    if (adjustGridMode) exitAdjustMode();

    const previewCanvas = document.getElementById('photo-preview');
    savedOverlay = null;

    const result = runPipeline(mats, {
      hintN, forceRows, forceCols,
      onStage(name) { photoStatusEl.textContent = name; },
      onIntermediate(name, data) {
        if (name === 'rectification') {
          cv.imshow(previewCanvas, data.rectified);
          // Cache rectified image for adjust-mode redraws
          const ctx = previewCanvas.getContext('2d');
          previewCanvas._rectifiedImg = ctx.getImageData(0, 0, previewCanvas.width, previewCanvas.height);
        }
        if (name === 'detectGrid' && data.detection?.intersections) {
          savedOverlay = {
            intersections: data.detection.intersections.map(row =>
              row.map(pt => ({ x: pt.x, y: pt.y }))
            ),
            step: data.detection.step,
          };
        }
      },
    });

    mats.colorMat.delete();
    mats.grayMat.delete();

    if (!result) {
      photoStatusEl.textContent = 'No board detected';
      return;
    }

    lastPhotoResult = result;
    currentElidedEdges = result.elidedEdges
      ?? { top: false, bottom: false, left: false, right: false };
    photoStatusEl.textContent = '';
    photoStatusEl.hidden = true;

    if (savedOverlay) {
      drawPhotoOverlay(previewCanvas, savedOverlay, result.grid, result.nRows, result.nCols);
    }
    renderPhotoBoard(result, currentElidedEdges, document.getElementById('photo-board-preview'));

    document.getElementById('photo-hint-rows').value = result.nRows;
    document.getElementById('photo-hint-cols').value = result.nCols;
    updateEdgeToggles(currentElidedEdges);
    photoDimsEl.hidden    = false;
    photoEdgesEl.hidden   = false;
    photoBodyEl.hidden    = false;
    photoAcceptEl.hidden  = false;
    photoAdjustEl.hidden  = false;
  } catch (e) {
    photoStatusEl.textContent = 'Error: ' + e.message;
    console.error(e);
  }
}

document.getElementById('photo-btn').addEventListener('click', () => {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = 'image/*';
  input.onchange = () => { if (input.files[0]) processPhoto(input.files[0]); };
  input.click();
});

document.addEventListener('paste', e => {
  const file = [...(e.clipboardData?.files || [])].find(f => f.type.startsWith('image/'));
  if (file) { e.preventDefault(); processPhoto(file); }
});

function schedulePhotoRerun() {
  clearTimeout(photoRerunTimer);
  photoRerunTimer = setTimeout(() => {
    const rows = parseInt(document.getElementById('photo-hint-rows').value, 10);
    const cols = parseInt(document.getElementById('photo-hint-cols').value, 10);
    if (lastPhotoFile && rows >= 2 && cols >= 2) {
      processPhoto(lastPhotoFile, 0, { forceRows: rows, forceCols: cols });
    }
  }, 600);
}

document.getElementById('photo-hint-rows').addEventListener('input', schedulePhotoRerun);
document.getElementById('photo-hint-cols').addEventListener('input', schedulePhotoRerun);

function updateEdgeToggles(elided) {
  for (const btn of document.querySelectorAll('.edge-toggle')) {
    const edge = btn.dataset.edge;
    const isElided = elided[edge];
    btn.classList.toggle('elided', isElided);
    btn.textContent = btn.dataset.edge.charAt(0).toUpperCase()
      + btn.dataset.edge.slice(1)
      + (isElided ? ' \u2026' : ' \u2500');
  }
}

function refreshPhotoBoardPreview() {
  if (!lastPhotoResult) return;
  renderPhotoBoard(lastPhotoResult, currentElidedEdges, document.getElementById('photo-board-preview'));
}

for (const btn of document.querySelectorAll('.edge-toggle')) {
  btn.addEventListener('click', () => {
    const edge = btn.dataset.edge;
    currentElidedEdges = { ...currentElidedEdges, [edge]: !currentElidedEdges[edge] };
    updateEdgeToggles(currentElidedEdges);
    refreshPhotoBoardPreview();
  });
}

photoAcceptEl.addEventListener('click', () => {
  if (adjustGridMode) exitAdjustMode();
  closePhotoModal();
  const sgf = photoResultToSGF(lastPhotoResult, currentElidedEdges);
  textarea.value = sgf;
  render(sgf);
  scheduleHashUpdate(sgf, true);
});

document.getElementById('photo-cancel-btn').addEventListener('click', closePhotoModal);
photoModalEl.addEventListener('click', e => { if (e.target === photoModalEl) closePhotoModal(); });

// ── Init / routing ───────────────────────────────────────────────────────

async function init() {
  board.innerHTML = '';
  gameInfo.hidden   = true;
  sharingEl.hidden  = true;
  qrModal.hidden    = true;
  errorEl.hidden    = true;
  statusEl.hidden   = true;

  if (location.hash.length > 1) {
    sgfPanel.open   = false;
    statusEl.hidden = false;
    let sgf;
    try {
      sgf = await decode(location.hash.slice(1));
    } catch (e) {
      statusEl.hidden = true;
      showError(e.message);
      return;
    }
    statusEl.hidden = true;
    textarea.value  = sgf;
    render(sgf);
    shareLinkEl.href = location.href;
    shareLinkEl.textContent = location.href;
    if (!qrModal.hidden) renderQR();
  } else {
    textarea.value = '';
    sgfPanel.open  = true;
  }
}

window.addEventListener('popstate', init);

// ── Hash update (debounced) ───────────────────────────────────────────────

function scheduleHashUpdate(sgf, push = false) {
  clearTimeout(encodeTimer);
  if (!sgf) {
    history.replaceState(null, '', location.pathname);
    shareLinkEl.href = '';
    shareLinkEl.textContent = '';
    return;
  }
  encodeTimer = setTimeout(async () => {
    try {
      const fragment = await encode(sgf);
      (push ? history.pushState : history.replaceState).call(history, null, '', '#' + fragment);
      shareLinkEl.href = location.href;
      shareLinkEl.textContent = location.href;
      if (!qrModal.hidden) renderQR();
    } catch {}
  }, 400);
}

async function commitHash() {
  clearTimeout(encodeTimer);
  const sgf = textarea.value.trim();
  if (!sgf) return;
  try {
    const fragment = await encode(sgf);
    history.pushState(null, '', '#' + fragment);
    shareLinkEl.href = location.href;
    shareLinkEl.textContent = location.href;
  } catch {}
}

// ── Render ───────────────────────────────────────────────────────────────

function render(sgf, gobanOpts = {}) {
  errorEl.hidden = true;
  if (!sgf) {
    board.innerHTML  = '';
    gameInfo.hidden  = true;
    sharingEl.hidden = true;
    qrModal.hidden   = true;
    return;
  }

  let trees;
  try {
    trees = parse(sgf);
  } catch (e) {
    board.innerHTML  = '';
    gameInfo.hidden  = true;
    sharingEl.hidden = true;
    qrModal.hidden   = true;
    showError(e.message);
    return;
  }

  const props = trees[0].nodes[0].props;

  const lines = [
    props.GN?.[0],
    props.PB?.[0] && props.PW?.[0] && `${props.PB[0]} vs ${props.PW[0]}`,
    props.DT?.[0],
  ].filter(Boolean);

  if (lines.length) {
    gameInfo.innerHTML = lines.map(s => `<p>${esc(s)}</p>`).join('');
    gameInfo.hidden = false;
  } else {
    gameInfo.hidden = true;
  }

  const rawName   = props.GN?.[0]?.trim() || 'kifu';
  currentFilename = rawName.replace(/[/\\:*?"<>|]/g, '_').trim() + '.sgf';

  renderGoban(trees, board, gobanOpts);
  sharingEl.hidden = false;
}

// ── Textarea input ───────────────────────────────────────────────────────

let isPaste = false;
textarea.addEventListener('paste', () => { isPaste = true; });

textarea.addEventListener('input', () => {
  if (constructMode) exitConstruct();
  const sgf = textarea.value.trim();
  const push = isPaste;
  isPaste = false;
  render(sgf);
  scheduleHashUpdate(sgf, push);
});

// ── Download ─────────────────────────────────────────────────────────────

downloadBtn.addEventListener('click', async () => {
  await commitHash();
  download(textarea.value.trim(), currentFilename);
});

// ── QR code ───────────────────────────────────────────────────────────────

function renderQR() {
  const url = shareLinkEl.href;
  if (!url) { qrDisplay.innerHTML = ''; return; }
  try {
    const qr = qrcode(0, 'M');
    qr.addData(url);
    qr.make();
    qrDisplay.innerHTML = qr.createSvgTag({ cellSize: 4, margin: 4, scalable: true });
  } catch {
    qrDisplay.innerHTML = '';
  }
}

copyBtn.addEventListener('click', async () => {
  await commitHash();
  const textEl = copyBtn.querySelector('span');
  navigator.clipboard.writeText(shareLinkEl.href).then(() => {
    textEl.textContent = 'Copied!';
    setTimeout(() => { textEl.textContent = 'Copy'; }, 2000);
  });
});

qrBtn.addEventListener('click', async () => {
  await commitHash();
  qrModal.hidden = false;
  renderQR();
});

document.getElementById('qr-close').addEventListener('click', () => {
  qrModal.hidden = true;
});

qrModal.addEventListener('click', e => {
  if (e.target === qrModal) qrModal.hidden = true;
});

document.addEventListener('keydown', e => {
  if (e.key === 'Escape') {
    if (!qrModal.hidden) qrModal.hidden = true;
    else if (!photoModalEl.hidden) closePhotoModal();
  }
});

// ── File picker ───────────────────────────────────────────────────────────

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) readFile(fileInput.files[0]);
});

// ── Drag-and-drop ─────────────────────────────────────────────────────────

dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});
['dragleave', 'drop'].forEach(evt =>
  dropZone.addEventListener(evt, () => dropZone.classList.remove('dragover'))
);
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  if (e.dataTransfer.files[0]) readFile(e.dataTransfer.files[0]);
});

function readFile(file) {
  const reader = new FileReader();
  reader.onload = e => {
    textarea.value = e.target.result;
    sgfPanel.open  = true;
    const sgf = textarea.value.trim();
    render(sgf);
    scheduleHashUpdate(sgf, true);
  };
  reader.readAsText(file);
}

// ── Utilities ─────────────────────────────────────────────────────────────

function showError(msg) {
  errorEl.textContent = msg;
  errorEl.hidden = false;
}

function flash(btn, label, reset) {
  btn.textContent = label;
  setTimeout(() => { btn.textContent = reset; }, 2000);
}

function esc(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

// ── Construct mode ────────────────────────────────────────────────────────

function treeHasMoves(tree) {
  for (const node of tree.nodes) {
    if (node.props.B !== undefined || node.props.W !== undefined) return true;
  }
  return tree.variations.some(v => treeHasMoves(v));
}

function generateConstructSGF() {
  const ab = [], aw = [];
  for (const [key, color] of constructStones) {
    const [r, c] = key.split(',').map(Number);
    const coord = String.fromCharCode(97 + c) + String.fromCharCode(97 + r);
    (color === 'B' ? ab : aw).push(`[${coord}]`);
  }
  const sz = constructCols === constructRows ? constructCols : `${constructCols}:${constructRows}`;
  let p = `SZ[${sz}]`;
  if (ab.length) p += `AB${ab.join('')}`;
  if (aw.length) p += `AW${aw.join('')}`;
  return `(;${p})`;
}

function renderConstruct() {
  const sgf = generateConstructSGF();
  textarea.value = sgf;
  render(sgf, { forceFullBoard: true, onClick: onBoardClick, jitter: 0 });
  scheduleHashUpdate(sgf);
}

function onBoardClick(r, c) {
  const key = `${r},${c}`;
  if (constructStones.get(key) === constructColor) {
    constructStones.delete(key);
  } else {
    constructStones.set(key, constructColor);
  }
  renderConstruct();
}

function enterConstruct() {
  const sgf = textarea.value.trim();
  if (sgf) {
    let trees;
    try { trees = parse(sgf); } catch { return; }
    const tree = trees[0];
    const rootProps = tree.nodes[0]?.props ?? {};
    const szProp = rootProps.SZ?.[0] ?? '19';
    if (szProp.includes(':')) {
      const parts = szProp.split(':');
      constructCols = parseInt(parts[0], 10);
      constructRows = parseInt(parts[1], 10);
    } else {
      constructCols = constructRows = parseInt(szProp, 10) || 19;
    }
    if (treeHasMoves(tree) && !confirm('This will remove all moves, keeping only the current board position. Continue?')) {
      return;
    }
    const boardState = new Int8Array(constructCols * constructRows);
    replayMain(boardState, constructCols, constructRows, tree);
    constructStones.clear();
    for (let r = 0; r < constructRows; r++) {
      for (let c = 0; c < constructCols; c++) {
        const v = boardState[r * constructCols + c];
        if (v === 1) constructStones.set(`${r},${c}`, 'B');
        else if (v === -1) constructStones.set(`${r},${c}`, 'W');
      }
    }
  } else {
    constructCols = constructRows = 19;
    constructStones.clear();
  }
  constructMode = true;
  constructEntryEl.hidden   = true;
  constructToolbarEl.hidden = false;
  sgfPanel.open = false;
  renderConstruct();
}

function exitConstruct() {
  constructMode = false;
  constructEntryEl.hidden   = false;
  constructToolbarEl.hidden = true;
  render(textarea.value);
}

document.getElementById('construct-btn').addEventListener('click', enterConstruct);
document.getElementById('construct-close').addEventListener('click', exitConstruct);
document.getElementById('construct-clear').addEventListener('click', () => {
  constructStones.clear();
  renderConstruct();
});

const colorBtns = document.querySelectorAll('.color-btn');
document.getElementById('color-black').addEventListener('click', () => {
  constructColor = 'B';
  colorBtns.forEach(b => b.classList.remove('active'));
  document.getElementById('color-black').classList.add('active');
});
document.getElementById('color-white').addEventListener('click', () => {
  constructColor = 'W';
  colorBtns.forEach(b => b.classList.remove('active'));
  document.getElementById('color-white').classList.add('active');
});

init();
