import { encode }           from './encode';
import { decode, download } from './decode';
import { parse }            from './sgf-parser';
import { renderGoban, replayMain } from './goban';
import { runPipeline, computeInversePerspective, applyHomography } from './run-pipeline';
import { inferBoardSize, computeEdgeOffsets, fitTPS, evalTPS } from './photo-pipeline';
import type { Point, ElidedEdges, TPSModel, ReadonlyMatrix } from './photo-pipeline';
import type { PipelineResult } from './run-pipeline';
import type { RenderOptions } from './goban';
import type { GameTree } from './sgf-parser';

declare function qrcode(typeNumber: number, errorCorrectionLevel: string): any;

function getEl(id: string): HTMLElement {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Element #${id} not found`);
  return el;
}

function getCtx(canvas: HTMLCanvasElement): CanvasRenderingContext2D {
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('getContext("2d") returned null');
  return ctx;
}

const board       = getEl('board');
const gameInfo    = getEl('game-info');
const statusEl    = getEl('status');
const errorEl     = getEl('error');
const sharingEl   = getEl('sharing');
const shareLinkEl = getEl('share-link') as HTMLAnchorElement;
const downloadBtn = getEl('download-btn');
const qrModal     = getEl('qr-modal');
const qrDisplay   = getEl('qr-display');
const copyBtn     = getEl('copy-btn');
const qrBtn       = getEl('qr-btn');
const sgfPanel    = getEl('sgf-panel') as HTMLDetailsElement;
const textarea    = getEl('sgf-input') as HTMLTextAreaElement;
const fileInput   = getEl('file-input') as HTMLInputElement;
const dropZone    = getEl('drop-zone');

let currentFilename  = 'game.sgf';
let encodeTimer: ReturnType<typeof setTimeout> | undefined;
let constructMode    = false;
let constructStones  = new Map<string, string>(); // `${r},${c}` → 'B' | 'W'
let constructColor   = 'B';
let constructCols    = 19;
let constructRows    = 19;

const constructEntryEl   = getEl('construct-entry');
const constructToolbarEl = getEl('construct-toolbar');

const rectifiedImages = new WeakMap<HTMLCanvasElement, ImageData>();

// ── Photo pipeline ────────────────────────────────────────────────────────

let cvReady: Promise<void> | null = null;
function loadCV() {
  if (cvReady) return cvReady;
  cvReady = new Promise<void>((resolve, reject) => {
    (window as Window & { Module?: { onRuntimeInitialized: () => void } }).Module = { onRuntimeInitialized: resolve };
    const s = document.createElement('script');
    s.src = 'https://docs.opencv.org/4.x/opencv.js';
    s.async = true;
    s.onerror = reject;
    document.head.appendChild(s);
  });
  return cvReady;
}

function imageToMats(img: HTMLImageElement) {
  const canvas = document.getElementById('photo-canvas') as HTMLCanvasElement;
  canvas.width  = img.naturalWidth  || img.width;
  canvas.height = img.naturalHeight || img.height;
  getCtx(canvas).drawImage(img, 0, 0);
  const colorMat = cv.imread(canvas);
  const grayMat  = new cv.Mat();
  cv.cvtColor(colorMat, grayMat, cv.COLOR_RGBA2GRAY);
  return { colorMat, grayMat, width: canvas.width, height: canvas.height };
}

interface GridOverlay {
  intersections: Point[][];
  step: number;
}

interface GridDot {
  x: number;
  y: number;
  r: number;
  c: number;
}

function drawPhotoOverlay(canvas: HTMLCanvasElement, { intersections, step }: GridOverlay, grid: ReadonlyMatrix<string>, nRows: number, nCols: number) {
  const ctx = getCtx(canvas);

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
function drawGridDots(canvas: HTMLCanvasElement) {
  if (!gridDots || !adjustGridMode || !lastPhotoResult) return;
  const ctx = getCtx(canvas);
  const step = savedOverlay ? savedOverlay.step : 30;
  const dotRadius = Math.max(3, step * 0.3);
  const nR = lastPhotoResult.nRows, nC = lastPhotoResult.nCols;

  // Build a lookup from (r,c) to dot for grid lines
  const dotAt: (GridDot | undefined)[][] = Array.from({ length: nR }, () => []);
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

function canvasFromEvent(e: MouseEvent | TouchEvent, canvas: HTMLCanvasElement) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const clientX = 'touches' in e && e.touches.length ? e.touches[0].clientX : (e as MouseEvent).clientX;
  const clientY = 'touches' in e && e.touches.length ? e.touches[0].clientY : (e as MouseEvent).clientY;
  return {
    x: (clientX - rect.left) * scaleX,
    y: (clientY - rect.top) * scaleY,
  };
}

function findNearestDot(cx: number, cy: number) {
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
  const ptsX: { x: number; y: number; target: number }[] = [];
  const ptsY: { x: number; y: number; target: number }[] = [];
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
  if (!lastTpsFit || !gridDots) return;
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
  const previewCanvas = document.getElementById('photo-preview') as HTMLCanvasElement;
  // Redraw rectified image
  const img = rectifiedImages.get(previewCanvas);
  if (img) {
    const ctx = getCtx(previewCanvas);
    ctx.putImageData(img, 0, 0);
  }
  if (adjustGridMode && gridDots) {
    // In adjust mode: draw only dots (with grid curves from TPS), no pipeline overlay
    drawGridDots(previewCanvas);
  } else {
    drawPhotoOverlay(previewCanvas, savedOverlay, lastPhotoResult.grid, lastPhotoResult.nRows, lastPhotoResult.nCols);
  }
}

function buildOverlayFromDots(): GridOverlay | null {
  if (!gridDots || !lastPhotoResult || !savedOverlay) return savedOverlay;
  const nR = lastPhotoResult.nRows, nC = lastPhotoResult.nCols;
  const intersections: Point[][] = Array.from({ length: nR }, () => []);
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

    const previewCanvas = document.getElementById('photo-preview') as HTMLCanvasElement;

    const result = runPipeline(mats, {
      forceRows: lastPhotoResult.nRows,
      forceCols: lastPhotoResult.nCols,
      rectCorners: lastPhotoResult.rectCorners,
      forcedGrid: originalCoordDots,
      onIntermediate(name: string, data: Record<string, any>) {
        // Skip rectification display — corners are locked, image is cached
        if (name === 'detectGrid' && data.detection?.intersections) {
          savedOverlay = {
            intersections: data.detection.intersections.map((row: Point[]) =>
              row.map((pt: Point) => ({ x: pt.x, y: pt.y }))
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
      const newDots: GridDot[] = [];
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
    renderPhotoBoard(result, currentElidedEdges, getEl('photo-board-preview'));
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

  const previewCanvas = document.getElementById('photo-preview') as HTMLCanvasElement;
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

  const previewCanvas = document.getElementById('photo-preview') as HTMLCanvasElement;
  previewCanvas.classList.remove('adjust-grid');
  detachGridListeners(previewCanvas);
  redrawPhotoPreview();
}

// ── Grid interaction handlers ────────────────────────────────────────────
let gridListenersAttached = false;
const gridHandlers: Record<string, (e: Event) => void> = {};

function attachGridListeners(canvas: HTMLCanvasElement) {
  if (gridListenersAttached) return;
  gridListenersAttached = true;

  gridHandlers.mousemove = (e: Event) => {
    const me = e as MouseEvent;
    if (draggingDot >= 0) {
      const pos = canvasFromEvent(me, canvas);
      const dx = pos.x - dragStart!.x; // safe: dragStart/dragDotStart/gridDots are set in mousedown before draggingDot >= 0
      const dy = pos.y - dragStart!.y;
      if (Math.abs(dx) > 1 || Math.abs(dy) > 1) didDrag = true;
      gridDots![draggingDot].x = dragDotStart!.x + dx;
      gridDots![draggingDot].y = dragDotStart!.y + dy;
      refitGrid();
      redrawPhotoPreview();
      return;
    }
    // Hover detection
    const pos = canvasFromEvent(me, canvas);
    const idx = findNearestDot(pos.x, pos.y);
    if (idx !== hoveredDot) {
      hoveredDot = idx;
      redrawPhotoPreview();
    }
  };

  gridHandlers.mousedown = (e: Event) => {
    const me = e as MouseEvent;
    const pos = canvasFromEvent(me, canvas);
    const idx = findNearestDot(pos.x, pos.y);
    if (idx < 0) return;
    me.preventDefault();
    if (!pinnedDots.has(idx)) {
      pinnedDots.add(idx);
      refitGrid();
      redrawPhotoPreview();
    }
    draggingDot = idx;
    dragStart = pos;
    dragDotStart = { x: gridDots![idx].x, y: gridDots![idx].y }; // safe: gridDots is set in enterAdjustMode before listeners are attached
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

  gridHandlers.touchstart = (e: Event) => {
    const te = e as TouchEvent;
    if (te.touches.length !== 1) return;
    const pos = canvasFromEvent(te, canvas);
    const idx = findNearestDot(pos.x, pos.y);
    if (idx < 0) return;
    te.preventDefault();
    if (!pinnedDots.has(idx)) {
      pinnedDots.add(idx);
      refitGrid();
      redrawPhotoPreview();
    }
    draggingDot = idx;
    dragStart = pos;
    dragDotStart = { x: gridDots![idx].x, y: gridDots![idx].y }; // safe: gridDots is set in enterAdjustMode before listeners are attached
    didDrag = false;
  };

  gridHandlers.touchmove = (e: Event) => {
    const te = e as TouchEvent;
    if (draggingDot < 0 || te.touches.length !== 1) return;
    te.preventDefault();
    const pos = canvasFromEvent(te, canvas);
    const dx = pos.x - dragStart!.x; // safe: set in touchstart before draggingDot >= 0
    const dy = pos.y - dragStart!.y;
    if (Math.abs(dx) > 1 || Math.abs(dy) > 1) didDrag = true;
    gridDots![draggingDot].x = dragDotStart!.x + dx;
    gridDots![draggingDot].y = dragDotStart!.y + dy;
    refitGrid();
    redrawPhotoPreview();
  };

  gridHandlers.touchend = (e: Event) => {
    const te = e as TouchEvent;
    if (te.touches.length > 0) return;
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

function detachGridListeners(canvas: HTMLCanvasElement) {
  if (!gridListenersAttached) return;
  gridListenersAttached = false;
  canvas.removeEventListener('mousemove', gridHandlers.mousemove);
  canvas.removeEventListener('mousedown', gridHandlers.mousedown);
  window.removeEventListener('mouseup', gridHandlers.mouseup);
  canvas.removeEventListener('touchstart', gridHandlers.touchstart);
  canvas.removeEventListener('touchmove', gridHandlers.touchmove);
  canvas.removeEventListener('touchend', gridHandlers.touchend);
}

function photoResultToSGF(result: PipelineResult, elidedEdges: ElidedEdges | null) {
  const boardN = elidedEdges
    ? inferBoardSize(result.nRows, result.nCols, elidedEdges)
    : Math.max(result.nRows, result.nCols);
  const rowOff = elidedEdges
    ? computeEdgeOffsets(result.nRows, boardN, elidedEdges.top, elidedEdges.bottom) : 0;
  const colOff = elidedEdges
    ? computeEdgeOffsets(result.nCols, boardN, elidedEdges.left, elidedEdges.right) : 0;

  const ab: string[] = [], aw: string[] = [];
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

function renderPhotoBoard(result: PipelineResult, elidedEdges: ElidedEdges | null, container: HTMLElement) {
  renderGoban(parse(photoResultToSGF(result, elidedEdges)), container, { forceFullBoard: true, jitter: 0 });
}

let lastPhotoFile: File | null      = null;
let lastPhotoResult: PipelineResult | null    = null;
let savedOverlay: GridOverlay | null       = null; // { intersections, step } in rectified coords
let photoRerunTimer: ReturnType<typeof setTimeout> | undefined;
let currentElidedEdges: ElidedEdges | null = null; // user-overridable elided edges

// ── Adjust grid state ──────────────────────────────────────────────────
let adjustGridMode = false;
let gridDots: GridDot[] | null = null;         // flat array of { x, y, r, c } in rectified coords
let pinnedDots = new Set<number>();  // indices into gridDots
let lastTpsFit: { tpsX: TPSModel; tpsY: TPSModel } | null = null;       // { tpsX, tpsY } or null
let hoveredDot = -1;
let draggingDot = -1;
let dragStart: { x: number; y: number } | null = null;        // { x, y } canvas coords at drag start
let dragDotStart: { x: number; y: number } | null = null;     // { x, y } dot position at drag start
let didDrag = false;

const photoModalEl  = getEl('photo-modal');
const photoStatusEl = getEl('photo-status');
const photoDimsEl   = getEl('photo-dims-row');
const photoEdgesEl  = getEl('photo-edges-row');
const photoBodyEl   = getEl('photo-modal-body');
const photoAcceptEl = getEl('photo-accept-btn');
const photoAdjustEl = getEl('photo-adjust-btn');

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

async function processPhoto(file: File, hintN = 0, { forceRows = 0, forceCols = 0 } = {}) {
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

    const previewCanvas = document.getElementById('photo-preview') as HTMLCanvasElement;
    savedOverlay = null;

    const result = runPipeline(mats, {
      hintN, forceRows, forceCols,
      onStage(name: string) { photoStatusEl.textContent = name; },
      onIntermediate(name: string, data: Record<string, any>) {
        if (name === 'rectification') {
          cv.imshow(previewCanvas, data.rectified);
          // Cache rectified image for adjust-mode redraws
          const ctx = getCtx(previewCanvas);
          rectifiedImages.set(previewCanvas, ctx.getImageData(0, 0, previewCanvas.width, previewCanvas.height));
        }
        if (name === 'detectGrid' && data.detection?.intersections) {
          savedOverlay = {
            intersections: data.detection.intersections.map((row: Point[]) =>
              row.map((pt: Point) => ({ x: pt.x, y: pt.y }))
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
    renderPhotoBoard(result, currentElidedEdges, getEl('photo-board-preview'));

    (document.getElementById('photo-hint-rows') as HTMLInputElement).value = String(result.nRows);
    (document.getElementById('photo-hint-cols') as HTMLInputElement).value = String(result.nCols);
    updateEdgeToggles(currentElidedEdges);
    photoDimsEl.hidden    = false;
    photoEdgesEl.hidden   = false;
    photoBodyEl.hidden    = false;
    photoAcceptEl.hidden  = false;
    photoAdjustEl.hidden  = false;
  } catch (e) {
    photoStatusEl.textContent = 'Error: ' + (e as Error).message;
    console.error(e);
  }
}

getEl('photo-btn').addEventListener('click', () => {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = 'image/*';
  input.onchange = () => { if (input.files?.[0]) processPhoto(input.files[0]); };
  input.click();
});

document.addEventListener('paste', e => {
  const file = Array.from(e.clipboardData?.files || []).find(f => f.type.startsWith('image/'));
  if (file) { e.preventDefault(); processPhoto(file); }
});

function schedulePhotoRerun() {
  clearTimeout(photoRerunTimer);
  photoRerunTimer = setTimeout(() => {
    const rows = parseInt((document.getElementById('photo-hint-rows') as HTMLInputElement).value, 10);
    const cols = parseInt((document.getElementById('photo-hint-cols') as HTMLInputElement).value, 10);
    if (lastPhotoFile && rows >= 2 && cols >= 2) {
      processPhoto(lastPhotoFile, 0, { forceRows: rows, forceCols: cols });
    }
  }, 600);
}

getEl('photo-hint-rows').addEventListener('input', schedulePhotoRerun);
getEl('photo-hint-cols').addEventListener('input', schedulePhotoRerun);

function updateEdgeToggles(elided: ElidedEdges) {
  for (const btn of Array.from(document.querySelectorAll<HTMLElement>('.edge-toggle'))) {
    const edge = btn.dataset.edge as keyof ElidedEdges;
    const isElided = elided[edge];
    btn.classList.toggle('elided', isElided);
    const edgeName = btn.dataset.edge ?? '';
    btn.textContent = edgeName.charAt(0).toUpperCase() + edgeName.slice(1)
      + (isElided ? ' \u2026' : ' \u2500');
  }
}

function refreshPhotoBoardPreview() {
  if (!lastPhotoResult) return;
  renderPhotoBoard(lastPhotoResult, currentElidedEdges, getEl('photo-board-preview'));
}

for (const btn of Array.from(document.querySelectorAll<HTMLElement>('.edge-toggle'))) {
  btn.addEventListener('click', () => {
    if (!currentElidedEdges) return; // buttons are hidden until a photo result is available
    const edge = btn.dataset.edge as keyof ElidedEdges;
    currentElidedEdges = { ...currentElidedEdges, [edge]: !currentElidedEdges[edge] };
    updateEdgeToggles(currentElidedEdges);
    refreshPhotoBoardPreview();
  });
}

photoAcceptEl.addEventListener('click', () => {
  if (!lastPhotoResult) return; // button is hidden until a photo result is available
  if (adjustGridMode) exitAdjustMode();
  closePhotoModal();
  const sgf = photoResultToSGF(lastPhotoResult, currentElidedEdges);
  textarea.value = sgf;
  render(sgf);
  scheduleHashUpdate(sgf, true);
});

getEl('photo-cancel-btn').addEventListener('click', closePhotoModal);
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
    let sgf: string;
    try {
      sgf = await decode(location.hash.slice(1));
    } catch (e) {
      statusEl.hidden = true;
      showError((e as Error).message);
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

function scheduleHashUpdate(sgf: string, push = false) {
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

function render(sgf: string, gobanOpts: RenderOptions = {}) {
  errorEl.hidden = true;
  if (!sgf) {
    board.innerHTML  = '';
    gameInfo.hidden  = true;
    sharingEl.hidden = true;
    qrModal.hidden   = true;
    return;
  }

  let trees: GameTree[];
  try {
    trees = parse(sgf);
  } catch (e) {
    board.innerHTML  = '';
    gameInfo.hidden  = true;
    sharingEl.hidden = true;
    qrModal.hidden   = true;
    showError((e as Error).message);
    return;
  }

  const props = trees[0].nodes[0].props;

  const lines = [
    props.GN?.[0],
    props.PB?.[0] && props.PW?.[0] && `${props.PB[0]} vs ${props.PW[0]}`,
    props.DT?.[0],
  ].filter(Boolean) as string[];

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
  const textEl = copyBtn.querySelector('span') as HTMLElement; // HTML structure invariant: copy-btn always contains a span
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

getEl('qr-close').addEventListener('click', () => {
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
  if (fileInput.files?.[0]) readFile(fileInput.files[0]);
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
  const file = (e as DragEvent).dataTransfer?.files[0];
  if (file) readFile(file);
});

function readFile(file: File) {
  const reader = new FileReader();
  reader.onload = e => {
    textarea.value = (e.target as FileReader).result as string;
    sgfPanel.open  = true;
    const sgf = textarea.value.trim();
    render(sgf);
    scheduleHashUpdate(sgf, true);
  };
  reader.readAsText(file);
}

// ── Utilities ─────────────────────────────────────────────────────────────

function showError(msg: string) {
  errorEl.textContent = msg;
  errorEl.hidden = false;
}

function flash(btn: HTMLElement, label: string, reset: string) {
  btn.textContent = label;
  setTimeout(() => { btn.textContent = reset; }, 2000);
}

function esc(str: string) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

// ── Construct mode ────────────────────────────────────────────────────────

function treeHasMoves(tree: GameTree): boolean {
  for (const node of tree.nodes) {
    if (node.props.B !== undefined || node.props.W !== undefined) return true;
  }
  return tree.variations.some(v => treeHasMoves(v));
}

function generateConstructSGF() {
  const ab: string[] = [], aw: string[] = [];
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

function onBoardClick(r: number, c: number) {
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
    let trees: GameTree[];
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

getEl('construct-btn').addEventListener('click', enterConstruct);
getEl('construct-close').addEventListener('click', exitConstruct);
getEl('construct-clear').addEventListener('click', () => {
  constructStones.clear();
  renderConstruct();
});

const colorBtns = document.querySelectorAll('.color-btn');
const colorBlackBtn = getEl('color-black');
const colorWhiteBtn = getEl('color-white');
colorBlackBtn.addEventListener('click', () => {
  constructColor = 'B';
  colorBtns.forEach(b => b.classList.remove('active'));
  colorBlackBtn.classList.add('active');
});
colorWhiteBtn.addEventListener('click', () => {
  constructColor = 'W';
  colorBtns.forEach(b => b.classList.remove('active'));
  colorWhiteBtn.classList.add('active');
});

init();
