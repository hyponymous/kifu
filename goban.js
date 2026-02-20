const CELL = 30;
const MARGIN = 40;
const JITTER = 0.5; // max px of positional noise per stone
const COL_LETTERS = 'ABCDEFGHJKLMNOPQRST';
const HOSHI = {
  19: [[3,3],[3,9],[3,15],[9,3],[9,9],[9,15],[15,3],[15,9],[15,15]],
  13: [[3,3],[3,9],[6,6],[9,3],[9,9]],
   9: [[2,2],[2,6],[4,4],[6,2],[6,6]],
};

function svgEl(tag, attrs) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

function sgfCoord(ch) {
  return ch.charCodeAt(0) - 97; // 'a'=0, 'b'=1, …
}

// Deterministic pseudo-random offset in [-1, 1] based on board position + axis
function posJitter(r, c, axis) {
  const s = Math.sin(r * 127.1 + c * 311.7 + axis * 543.9) * 43758.5453;
  return (s - Math.floor(s)) * 2 - 1;
}

function floodFill(board, size, r, c, color) {
  const visited = new Uint8Array(size * size);
  const queue = [[r, c]];
  const cells = [];
  visited[r * size + c] = 1;
  let liberties = 0;
  while (queue.length) {
    const [row, col] = queue.shift();
    cells.push([row, col]);
    for (const [dr, dc] of [[-1,0],[1,0],[0,-1],[0,1]]) {
      const nr = row + dr, nc = col + dc;
      if (nr < 0 || nr >= size || nc < 0 || nc >= size) continue;
      const idx = nr * size + nc;
      if (visited[idx]) continue;
      visited[idx] = 1;
      const v = board[idx];
      if (v === 0) liberties++;
      else if (v === color) queue.push([nr, nc]);
    }
  }
  return { cells, liberties };
}

function checkCaptures(board, size, r, c, opponent) {
  for (const [dr, dc] of [[-1,0],[1,0],[0,-1],[0,1]]) {
    const nr = r + dr, nc = c + dc;
    if (nr < 0 || nr >= size || nc < 0 || nc >= size) continue;
    if (board[nr * size + nc] !== opponent) continue;
    const { cells, liberties } = floodFill(board, size, nr, nc, opponent);
    if (liberties === 0) {
      for (const [row, col] of cells) board[row * size + col] = 0;
    }
  }
}

function replayMain(board, size, tree) {
  for (const node of tree.nodes) {
    const { props } = node;
    for (const coord of (props.AB ?? [])) {
      const c = sgfCoord(coord[0]), r = sgfCoord(coord[1]);
      if (r >= 0 && r < size && c >= 0 && c < size) board[r * size + c] = 1;
    }
    for (const coord of (props.AW ?? [])) {
      const c = sgfCoord(coord[0]), r = sgfCoord(coord[1]);
      if (r >= 0 && r < size && c >= 0 && c < size) board[r * size + c] = -1;
    }
    for (const coord of (props.AE ?? [])) {
      const c = sgfCoord(coord[0]), r = sgfCoord(coord[1]);
      if (r >= 0 && r < size && c >= 0 && c < size) board[r * size + c] = 0;
    }
    for (const [prop, color] of [['B', 1], ['W', -1]]) {
      const val = props[prop]?.[0];
      if (val === undefined) continue;
      if (val === '' || val === 'tt') continue; // pass
      const c = sgfCoord(val[0]), r = sgfCoord(val[1]);
      if (r < 0 || r >= size || c < 0 || c >= size) continue;
      board[r * size + c] = color;
      checkCaptures(board, size, r, c, -color);
    }
  }
  if (tree.variations?.[0]) replayMain(board, size, tree.variations[0]);
}

export function renderGoban(trees, container, opts = {}) {
  const M  = opts.margin     ?? MARGIN;
  const jt = opts.jitter     ?? JITTER;
  const ss = opts.stoneScale ?? 0.49;

  container.innerHTML = '';
  try {
    const tree = trees?.[0];
    if (!tree) return;

    const rootProps = tree.nodes[0]?.props ?? {};
    if (rootProps.GM && rootProps.GM[0] !== '1') return;

    const szProp = rootProps.SZ?.[0] ?? '19';
    if (szProp.includes(':')) return;
    const size = parseInt(szProp, 10) || 19;
    if (size < 2 || size > 25) return;

    const board = new Int8Array(size * size);
    replayMain(board, size, tree);

    const R       = CELL * ss;
    const boardPx = (size - 1) * CELL;
    const totalW  = boardPx + M * 2;
    const totalH  = boardPx + M * 2;

    const svg = svgEl('svg', {
      viewBox: `0 0 ${totalW} ${totalH}`,
      width: totalW,
      height: totalH,
      style: 'max-width:100%;height:auto;display:block',
    });

    // Board background
    svg.appendChild(svgEl('rect', {
      x: 0, y: 0, width: totalW, height: totalH,
      fill: '#dcb483',
    }));

    // Grid lines
    for (let i = 0; i < size; i++) {
      svg.appendChild(svgEl('line', {
        x1: M + i * CELL, y1: M,
        x2: M + i * CELL, y2: M + boardPx,
        stroke: '#7a5230', 'stroke-width': 0.8,
      }));
      svg.appendChild(svgEl('line', {
        x1: M, y1: M + i * CELL,
        x2: M + boardPx, y2: M + i * CELL,
        stroke: '#7a5230', 'stroke-width': 0.8,
      }));
    }

    // Star points (hoshi)
    for (const [hr, hc] of (HOSHI[size] ?? [])) {
      svg.appendChild(svgEl('circle', {
        cx: M + hc * CELL,
        cy: M + hr * CELL,
        r: 3,
        fill: '#7a5230',
      }));
    }

    // Coordinate labels — anchored just outside the stone radius
    const lg      = opts.labelGap ?? 8; // visual clearance between stone edge and nearest text edge
    const colTopY = M - R - lg - 3;    // -3 for descenders hanging below baseline
    const colBotY = M + boardPx + R + lg + 7; // +7 for cap height above baseline
    const rowLX   = M - R - lg;
    const rowRX   = M + boardPx + R + lg;
    for (let i = 0; i < size; i++) {
      const colLabel = COL_LETTERS[i] ?? '';
      const rowLabel = String(size - i);
      const x = M + i * CELL;
      const y = M + i * CELL;

      const tTop = svgEl('text', { x, y: colTopY, 'text-anchor': 'middle', 'font-size': 10, fill: '#5a3a1a' });
      tTop.textContent = colLabel;
      svg.appendChild(tTop);

      const tBot = svgEl('text', { x, y: colBotY, 'text-anchor': 'middle', 'font-size': 10, fill: '#5a3a1a' });
      tBot.textContent = colLabel;
      svg.appendChild(tBot);

      const tLeft = svgEl('text', { x: rowLX, y: y + 4, 'text-anchor': 'end', 'font-size': 10, fill: '#5a3a1a' });
      tLeft.textContent = rowLabel;
      svg.appendChild(tLeft);

      const tRight = svgEl('text', { x: rowRX, y: y + 4, 'text-anchor': 'start', 'font-size': 10, fill: '#5a3a1a' });
      tRight.textContent = rowLabel;
      svg.appendChild(tRight);
    }

    // Stones
    for (let r = 0; r < size; r++) {
      for (let c = 0; c < size; c++) {
        const v = board[r * size + c];
        if (v === 0) continue;
        const cx = M + c * CELL + posJitter(r, c, 0) * jt;
        const cy = M + r * CELL + posJitter(r, c, 1) * jt;
        const attrs = { cx, cy, r: R };
        if (v === 1) {
          attrs.fill = '#1a1a1a';
        } else {
          attrs.fill = '#f5f5f0';
          attrs.stroke = '#888';
          attrs['stroke-width'] = 0.8;
        }
        svg.appendChild(svgEl('circle', attrs));
      }
    }

    container.appendChild(svg);
  } catch {
    // silently skip on any error (malformed SGF edge cases)
  }
}
