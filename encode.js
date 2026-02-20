/**
 * Encodes an SGF string into a gzip-compressed, base64url-encoded URL fragment.
 *
 * encode(sgf)          → Promise<string>  pure: fragment without '#'
 * encodeToHash(sgf)    → Promise<string>  also sets location.hash
 */

import { parse } from './sgf-parser.js';

export async function encode(sgf) {
  parse(sgf); // validate before encoding

  const bytes = new TextEncoder().encode(sgf);
  const compressed = await gzip(bytes);
  return toBase64Url(compressed);
}

export async function encodeToHash(sgf) {
  const fragment = await encode(sgf);
  location.hash = fragment;
  return fragment;
}

// ── Internals ─────────────────────────────────────────────────────────────────

async function gzip(data) {
  const stream = new CompressionStream('gzip');
  const writer = stream.writable.getWriter();
  await writer.write(data);
  await writer.close();
  return new Uint8Array(await new Response(stream.readable).arrayBuffer());
}

function toBase64Url(bytes) {
  let binary = '';
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}
