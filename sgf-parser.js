/**
 * SGF (Smart Game Format) recursive-descent parser.
 *
 * Usage:
 *   import { parse } from './sgf-parser.js';
 *   const trees = parse(sgfString); // GameTree[]
 *
 * Types:
 *   GameTree = { nodes: Node[], variations: GameTree[] }
 *   Node     = { props: Record<string, string[]> }
 *
 * Throws an Error with line/column info on invalid input.
 */

export const MAX_NODES = 10_000;
export const MAX_BYTES = 1_000_000; // 1 MB

/**
 * Parse an SGF string into an array of game trees (a collection).
 * Most SGF files contain a single tree; check trees[0].
 */
export function parse(src) {
  if (src.length > MAX_BYTES) {
    throw new Error(`SGF too large: ${src.length} bytes (limit ${MAX_BYTES})`);
  }

  const st = { src, pos: 0, line: 1, col: 1, nodeCount: 0 };
  skipWS(st);

  const trees = [];
  while (st.pos < src.length) {
    if (src[st.pos] !== '(') throw err(st, `expected '('`);
    trees.push(parseTree(st));
    skipWS(st);
  }

  if (trees.length === 0) throw new Error('SGF parse error: empty input');
  return trees;
}

function parseTree(st) {
  eat(st, '(');
  skipWS(st);

  if (st.pos >= st.src.length || st.src[st.pos] !== ';') {
    throw err(st, `expected ';' to open first node`);
  }
  const nodes = [];
  while (st.pos < st.src.length && st.src[st.pos] === ';') {
    nodes.push(parseNode(st));
    skipWS(st);
  }

  const variations = [];
  while (st.pos < st.src.length && st.src[st.pos] === '(') {
    variations.push(parseTree(st));
    skipWS(st);
  }

  eat(st, ')');
  return { nodes, variations };
}

function parseNode(st) {
  eat(st, ';');
  if (++st.nodeCount > MAX_NODES) throw err(st, `too many nodes (limit ${MAX_NODES})`);
  skipWS(st);

  const props = {};
  while (st.pos < st.src.length && isUpper(st.src.charCodeAt(st.pos))) {
    const name = parseIdent(st);
    skipWS(st);
    if (st.pos >= st.src.length || st.src[st.pos] !== '[') {
      throw err(st, `expected '[' after property '${name}'`);
    }
    const values = [];
    while (st.pos < st.src.length && st.src[st.pos] === '[') {
      values.push(parseValue(st));
      skipWS(st);
    }
    props[name] = values;
  }

  return { props };
}

function parseIdent(st) {
  const start = st.pos;
  while (st.pos < st.src.length && isUpper(st.src.charCodeAt(st.pos))) advance(st);
  if (st.pos === start) throw err(st, 'expected property identifier');
  return st.src.slice(start, st.pos);
}

function parseValue(st) {
  eat(st, '[');
  let val = '';
  while (st.pos < st.src.length) {
    const ch = st.src[st.pos];
    if (ch === ']') { advance(st); return val; }
    if (ch === '\\') {
      advance(st); // consume backslash
      if (st.pos >= st.src.length) throw err(st, `unexpected end of input after '\\'`);
      const next = st.src[st.pos];
      if (next === '\r' || next === '\n') {
        advance(st); // soft line break — consume and discard
      } else {
        val += next;
        advance(st);
      }
    } else {
      val += ch;
      advance(st);
    }
  }
  throw err(st, 'unterminated property value');
}

// ── Primitives ───────────────────────────────────────────────────────────────

function advance(st) {
  const ch = st.src[st.pos++];
  if (ch === '\n') {
    st.line++;
    st.col = 1;
  } else if (ch === '\r') {
    st.line++;
    st.col = 1;
    if (st.pos < st.src.length && st.src[st.pos] === '\n') st.pos++; // \r\n pair
  } else {
    st.col++;
  }
}

function eat(st, ch) {
  if (st.pos >= st.src.length) throw err(st, `expected '${ch}', got end of input`);
  if (st.src[st.pos] !== ch) throw err(st, `expected '${ch}', got '${st.src[st.pos]}'`);
  advance(st);
}

function skipWS(st) {
  while (st.pos < st.src.length) {
    const ch = st.src[st.pos];
    if (ch !== ' ' && ch !== '\t' && ch !== '\n' && ch !== '\r') break;
    advance(st);
  }
}

function isUpper(code) { return code >= 65 && code <= 90; } // A–Z

function err(st, msg) {
  return new Error(`SGF parse error at line ${st.line}, col ${st.col}: ${msg}`);
}
