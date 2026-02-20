import { test } from 'node:test';
import assert from 'node:assert/strict';
import { encode } from './encode.js';
import { decode } from './decode.js';
import { MAX_BYTES } from './sgf-parser.js';

const SIMPLE = '(;FF[4]GM[1]SZ[19];B[pd];W[dp])';
const UNICODE = '(;C[日本語コメント])';

// ── Round-trip ────────────────────────────────────────────────────────────────

test('encode then decode returns original SGF', async () => {
  const fragment = await encode(SIMPLE);
  const result = await decode(fragment);
  assert.equal(result, SIMPLE);
});

test('round-trip preserves unicode content', async () => {
  const fragment = await encode(UNICODE);
  const result = await decode(fragment);
  assert.equal(result, UNICODE);
});

test('round-trip preserves a game with variations', async () => {
  const sgf = '(;FF[4];B[pd](;W[dp];B[pp])(;W[pp];B[dd]))';
  const result = await decode(await encode(sgf));
  assert.equal(result, sgf);
});

// ── encode ────────────────────────────────────────────────────────────────────

test('encode returns a non-empty string', async () => {
  const fragment = await encode(SIMPLE);
  assert.ok(fragment.length > 0);
});

test('encode output is valid base64url (no +, /, or = chars)', async () => {
  const fragment = await encode(SIMPLE);
  assert.doesNotMatch(fragment, /[+/=]/);
});

test('encode rejects invalid SGF', async () => {
  await assert.rejects(() => encode('not sgf'), /SGF parse error/);
});

test('encode rejects SGF exceeding MAX_BYTES', async () => {
  const big = '(;C[' + 'a'.repeat(MAX_BYTES) + '])';
  await assert.rejects(() => encode(big), /too large/);
});

// ── decode ────────────────────────────────────────────────────────────────────

test('decode rejects invalid base64url', async () => {
  await assert.rejects(() => decode('!!!'), /not valid base64url/);
});

test('decode rejects a fragment containing invalid SGF', async () => {
  // Compress a non-SGF string directly (bypassing encode's validation)
  const bytes = new TextEncoder().encode('this is not sgf');
  const stream = new CompressionStream('gzip');
  const writer = stream.writable.getWriter();
  await writer.write(bytes);
  await writer.close();
  const buf = await new Response(stream.readable).arrayBuffer();
  let binary = '';
  for (const b of new Uint8Array(buf)) binary += String.fromCharCode(b);
  const fragment = btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');

  await assert.rejects(() => decode(fragment), /SGF parse error/);
});

test('decode rejects fragment that decompresses beyond maxBytes', async () => {
  // Encode a small SGF, then decode with a tiny limit — no need to build a huge payload
  const fragment = await encode(SIMPLE);
  await assert.rejects(() => decode(fragment, 1), /exceeds limit/);
});
