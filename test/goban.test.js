import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parse } from '../src/sgf-parser.js';
import { replayMain, calculateViewport } from '../src/goban.js';

// ── Helpers ───────────────────────────────────────────────────────────────────

function runBoard(size, sgf) {
  const tree = parse(sgf)[0];
  const board = new Int8Array(size * size);
  replayMain(board, size, size, tree);
  return board;
}

function runRect(cols, rows, sgf) {
  const tree = parse(sgf)[0];
  const board = new Int8Array(cols * rows);
  replayMain(board, cols, rows, tree);
  return board;
}

// Read board value by SGF coordinate string (e.g. 'pd' → col=15, row=3)
function at(board, cols, coord) {
  const c = coord.charCodeAt(0) - 97;
  const r = coord.charCodeAt(1) - 97;
  return board[r * cols + c];
}

const BLACK = 1, WHITE = -1, EMPTY = 0;

// ── Move placement ────────────────────────────────────────────────────────────

test('B and W moves place stones', () => {
  const b = runBoard(19, '(;B[pd];W[dp])');
  assert.equal(at(b, 19, 'pd'), BLACK);
  assert.equal(at(b, 19, 'dp'), WHITE);
});

test('pass B[] is ignored', () => {
  const b = runBoard(9, '(;B[])');
  assert.ok(b.every(v => v === 0));
});

test('pass B[tt] is ignored', () => {
  const b = runBoard(9, '(;B[tt])');
  assert.ok(b.every(v => v === 0));
});

// ── Setup properties ──────────────────────────────────────────────────────────

test('AB and AW place setup stones', () => {
  const b = runBoard(9, '(;AB[aa][bb]AW[cc])');
  assert.equal(at(b, 9, 'aa'), BLACK);
  assert.equal(at(b, 9, 'bb'), BLACK);
  assert.equal(at(b, 9, 'cc'), WHITE);
});

test('AE removes a stone', () => {
  const b = runBoard(9, '(;AB[aa];AE[aa])');
  assert.equal(at(b, 9, 'aa'), EMPTY);
});

test('setup stones in illegal position are not captured', () => {
  // AB/AW place stones without triggering capture logic, so a group
  // with zero liberties from setup remains on the board as-is.
  // White completely surrounds black at aa (corner): ba and ab are white.
  const b = runBoard(9, '(;AB[aa]AW[ba][ab])');
  assert.equal(at(b, 9, 'aa'), BLACK); // zero liberties but not removed
});

// ── Captures ──────────────────────────────────────────────────────────────────

test('single stone capture', () => {
  // White in corner (aa), black on both edges; B[ab] takes last liberty
  const b = runBoard(9, '(;AW[aa]AB[ba];B[ab])');
  assert.equal(at(b, 9, 'aa'), EMPTY); // captured
  assert.equal(at(b, 9, 'ab'), BLACK); // capturing stone remains
});

test('group capture', () => {
  // White group aa+ab, surrounded; B[ac] takes the last shared liberty
  const b = runBoard(9, '(;AW[aa][ab]AB[ba][bb];B[ac])');
  assert.equal(at(b, 9, 'aa'), EMPTY);
  assert.equal(at(b, 9, 'ab'), EMPTY);
  assert.equal(at(b, 9, 'ac'), BLACK);
});

test('one move captures two separate groups', () => {
  // White at ba and da each have only one liberty left (ca); B[ca] captures both
  const b = runBoard(9, '(;AW[ba][da]AB[aa][bb][ea][db];B[ca])');
  assert.equal(at(b, 9, 'ba'), EMPTY);
  assert.equal(at(b, 9, 'da'), EMPTY);
  assert.equal(at(b, 9, 'ca'), BLACK);
});

// ── Variations ────────────────────────────────────────────────────────────────

test('main line follows variations[0] only', () => {
  const b = runBoard(9, '(;(;B[aa])(;B[bb]))');
  assert.equal(at(b, 9, 'aa'), BLACK);
  assert.equal(at(b, 9, 'bb'), EMPTY);
});

// ── Rectangular boards ────────────────────────────────────────────────────────

test('rectangular board: stones placed within bounds', () => {
  // 5 cols × 9 rows; ea = col 4 (rightmost valid), ia = col 8 (out of bounds)
  const b = runRect(5, 9, '(;B[aa];W[ea])');
  assert.equal(at(b, 5, 'aa'), BLACK);
  assert.equal(at(b, 5, 'ea'), WHITE);
});

test('rectangular board: out-of-bounds column ignored', () => {
  // fa = col 5, which exceeds the 5-col board
  const b = runRect(5, 9, '(;B[fa])');
  assert.ok(b.every(v => v === 0));
});

test('rectangular board: out-of-bounds row ignored', () => {
  // aj = row 9, which exceeds the 9-row board
  const b = runRect(5, 9, '(;B[aj])');
  assert.ok(b.every(v => v === 0));
});

test('rectangular board: captures work across dimensions', () => {
  // White at aa (top-left corner of 5×9), surrounded by black at ba and ab
  const b = runRect(5, 9, '(;AW[aa]AB[ba];B[ab])');
  assert.equal(at(b, 5, 'aa'), EMPTY); // captured
  assert.equal(at(b, 5, 'ab'), BLACK);
});

// ── calculateViewport ─────────────────────────────────────────────────────────

function viewport(cols, rows, sgf) {
  const tree = parse(sgf)[0];
  const board = new Int8Array(cols * rows);
  replayMain(board, cols, rows, tree);
  const rootProps = tree.nodes[0].props;
  return calculateViewport(board, cols, rows, rootProps, tree);
}

// Heuristic A: game metadata forces full board
test('viewport: PB/PW forces full board', () => {
  const { vR0, vR1, vC0, vC1 } = viewport(19, 19,
    '(;GM[1]SZ[19]PB[Alice]PW[Bob]AB[jj])');
  assert.equal(vR0, 0); assert.equal(vR1, 18);
  assert.equal(vC0, 0); assert.equal(vC1, 18);
});

test('viewport: KM forces full board', () => {
  const { vR0, vR1, vC0, vC1 } = viewport(19, 19,
    '(;GM[1]SZ[19]KM[6.5]AB[jj])');
  assert.equal(vR0, 0); assert.equal(vR1, 18);
  assert.equal(vC0, 0); assert.equal(vC1, 18);
});

test('viewport: DT alone does not force full board', () => {
  const { vR0, vC0 } = viewport(19, 19,
    '(;GM[1]SZ[19]DT[2009-04-22]AB[jj])');
  // jj is interior — should be cropped despite DT
  assert.ok(vR0 > 0);
  assert.ok(vC0 > 0);
});

// Heuristic B: no setup stones → game → full board
test('viewport: pure move sequence forces full board', () => {
  const { vR0, vR1, vC0, vC1 } = viewport(19, 19,
    '(;GM[1]SZ[19];B[jj])');
  assert.equal(vR0, 0); assert.equal(vR1, 18);
  assert.equal(vC0, 0); assert.equal(vC1, 18);
});

// Heuristic C: per-edge snapping
test('viewport: corner stone snaps near edges, crops far edges', () => {
  // AB[dd] = col 3, row 3 — within K=4 of top and left, far from bottom and right
  const { vR0, vR1, vC0, vC1 } = viewport(19, 19,
    '(;GM[1]SZ[19]AB[dd])');
  assert.equal(vR0, 0);        // snapped to top edge
  assert.equal(vC0, 0);        // snapped to left edge
  assert.ok(vR1 < 18);         // bottom cropped
  assert.ok(vC1 < 18);         // right cropped
});

test('viewport: interior stone crops all four sides', () => {
  const { vR0, vR1, vC0, vC1 } = viewport(19, 19,
    '(;GM[1]SZ[19]AB[jj])');
  assert.ok(vR0 > 0);
  assert.ok(vR1 < 18);
  assert.ok(vC0 > 0);
  assert.ok(vC1 < 18);
});

test('viewport: stones near all four edges snap all sides', () => {
  // pd=col15,row3 and dp=col3,row15 — each within K=4 of their near edges
  const { vR0, vR1, vC0, vC1 } = viewport(19, 19,
    '(;GM[1]SZ[19]AB[pd][dp])');
  assert.equal(vR0, 0); assert.equal(vR1, 18);
  assert.equal(vC0, 0); assert.equal(vC1, 18);
});

test('viewport: empty board shows full board', () => {
  const { vR0, vR1, vC0, vC1 } = viewport(19, 19, '(;GM[1]SZ[19])');
  assert.equal(vR0, 0); assert.equal(vR1, 18);
  assert.equal(vC0, 0); assert.equal(vC1, 18);
});

// ── Compressed point lists (rectangle notation) ───────────────────────────────

test('AB compressed rect places all stones in rectangle', () => {
  // ao:bo = cols a-b, row o → ao and bo
  const b = runBoard(19, '(;AB[ao:bo])');
  assert.equal(at(b, 19, 'ao'), BLACK);
  assert.equal(at(b, 19, 'bo'), BLACK);
});

test('AW compressed rect places column strip', () => {
  // dp:ds = col d, rows p-s → dp, dq, dr, ds (4 stones)
  const b = runBoard(19, '(;AW[dp:ds])');
  assert.equal(at(b, 19, 'dp'), WHITE);
  assert.equal(at(b, 19, 'dq'), WHITE);
  assert.equal(at(b, 19, 'dr'), WHITE);
  assert.equal(at(b, 19, 'ds'), WHITE);
});

test('AW compressed 2x2 rect places all four corners', () => {
  // aa:bb = cols a-b, rows a-b → aa, ba, ab, bb
  const b = runBoard(9, '(;AW[aa:bb])');
  assert.equal(at(b, 9, 'aa'), WHITE);
  assert.equal(at(b, 9, 'ba'), WHITE);
  assert.equal(at(b, 9, 'ab'), WHITE);
  assert.equal(at(b, 9, 'bb'), WHITE);
});

test('AE compressed rect clears stones', () => {
  const b = runBoard(9, '(;AB[aa][ba][ab][bb];AE[aa:bb])');
  assert.equal(at(b, 9, 'aa'), EMPTY);
  assert.equal(at(b, 9, 'ba'), EMPTY);
  assert.equal(at(b, 9, 'ab'), EMPTY);
  assert.equal(at(b, 9, 'bb'), EMPTY);
});

test('SmartGo tsumego stone count: 4 black 8 white', () => {
  // AB[ao:bo][bp][cr] AW[bl][bn:cn][co][dp:ds]
  const b = runBoard(19, '(;AB[ao:bo][bp][cr]AW[bl][bn:cn][co][dp:ds])');
  const blacks = Array.from(b).filter(v => v === BLACK).length;
  const whites = Array.from(b).filter(v => v === WHITE).length;
  assert.equal(blacks, 4);
  assert.equal(whites, 8);
});
