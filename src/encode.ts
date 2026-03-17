/**
 * Encodes an SGF string into a gzip-compressed, base64url-encoded URL fragment.
 *
 * encode(sgf)          → Promise<string>  pure: fragment without '#'
 * encodeToHash(sgf)    → Promise<string>  also sets location.hash
 */

import { parse } from './sgf-parser';

export async function encode(sgf: string): Promise<string> {
  parse(sgf); // validate before encoding

  const bytes = new TextEncoder().encode(sgf);
  const compressed = await gzip(bytes);
  return toBase64Url(compressed);
}

export async function encodeToHash(sgf: string): Promise<string> {
  const fragment = await encode(sgf);
  location.hash = fragment;
  return fragment;
}

// ── Internals ─────────────────────────────────────────────────────────────────

async function gzip(data: Uint8Array): Promise<Uint8Array> {
  const stream = new CompressionStream('gzip');
  const writer = stream.writable.getWriter();
  // Fire-and-forget write/close so the readable side can drain concurrently.
  // Awaiting write() before a reader exists deadlocks once the internal buffer fills.
  writer.write(data as Uint8Array<ArrayBuffer>).then(() => writer.close());
  return new Uint8Array(await new Response(stream.readable).arrayBuffer());
}

function toBase64Url(bytes: Uint8Array): string {
  let binary = '';
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}
