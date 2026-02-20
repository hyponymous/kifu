import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parse } from './sgf-parser.js';
import { replayMain } from './goban.js';

// ── Helpers ───────────────────────────────────────────────────────────────────

function runBoard(size, sgf) {
  const tree = parse(sgf)[0];
  const board = new Int8Array(size * size);
  replayMain(board, size, tree);
  return board;
}

// Read board value by SGF coordinate string (e.g. 'pd' → col=15, row=3)
function at(board, size, coord) {
  const c = coord.charCodeAt(0) - 97;
  const r = coord.charCodeAt(1) - 97;
  return board[r * size + c];
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
