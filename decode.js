/**
 * Decodes a base64url-encoded, gzip-compressed URL fragment back to SGF.
 *
 * decode(fragment)     → Promise<string>  pure: fragment without '#'
 * decodeFromHash()     → Promise<string>  reads location.hash
 * download(sgf, name)  → void             triggers browser file download
 */

import { parse, MAX_BYTES } from './sgf-parser.js';

export async function decode(fragment, maxBytes = MAX_BYTES) {
  let compressed;
  try {
    compressed = fromBase64Url(fragment);
  } catch {
    throw new Error('Invalid URL fragment: not valid base64url');
  }

  const decompressed = await gunzip(compressed, maxBytes);
  const sgf = new TextDecoder().decode(decompressed);

  parse(sgf); // validate — throws with a descriptive message if invalid
  return sgf;
}

export async function decodeFromHash() {
  const fragment = location.hash.slice(1);
  if (!fragment) throw new Error('No SGF data in URL fragment');
  return decode(fragment);
}

export function download(sgf, filename = 'game.sgf') {
  const safe = filename.replace(/[/\\:*?"<>|]/g, '_');
  const blob = new Blob([sgf], { type: 'application/x-go-sgf' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = safe;
  a.style.display = 'none';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ── Internals ─────────────────────────────────────────────────────────────────

async function gunzip(data, maxBytes) {
  const stream = new DecompressionStream('gzip');
  const writer = stream.writable.getWriter();
  await writer.write(data);
  await writer.close();

  const chunks = [];
  let totalLength = 0;
  const reader = stream.readable.getReader();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    totalLength += value.length;
    if (totalLength > maxBytes) {
      await reader.cancel();
      throw new Error(`Decompressed data exceeds limit of ${maxBytes} bytes`);
    }
    chunks.push(value);
  }

  const result = new Uint8Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.length;
  }
  return result;
}

function fromBase64Url(str) {
  const base64 = str.replace(/-/g, '+').replace(/_/g, '/');
  const padded = base64 + '=='.slice(0, (4 - (base64.length % 4)) % 4);
  const binary = atob(padded);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}
