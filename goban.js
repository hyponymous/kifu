const CELL = 30;
const MARGIN = 40;
const JITTER = 0.5; // max px of positional noise per stone
const COL_LETTERS = 'ABCDEFGHJKLMNOPQRSTUVWXYZ';
// Star points for square boards only; rectangular boards get none.
// N < 5: none; 5–8: center if odd; 9–18: corners + center if odd; ≥19: full 3×3 grid.
function hoshiPoints(N) {
  if (N < 5) return [];
  const corner = N >= 13 ? 3 : 2;
  const far = N - 1 - corner;
  const mid = (N - 1) / 2;
  const hasCenter = Number.isInteger(mid);
  if (N <= 8) return hasCenter ? [[mid, mid]] : [];
  const corners = [[corner,corner],[corner,far],[far,corner],[far,far]];
  if (N <= 18) return hasCenter ? [...corners, [mid, mid]] : corners;
  // N >= 19: 3×3 grid (2×2 for even N)
  const pos = hasCenter ? [corner, mid, far] : [corner, far];
  return pos.flatMap(r => pos.map(c => [r, c]));
}

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

function floodFill(board, cols, rows, r, c, color) {
  const visited = new Uint8Array(cols * rows);
  const queue = [[r, c]];
  const cells = [];
  visited[r * cols + c] = 1;
  let liberties = 0;
  while (queue.length) {
    const [row, col] = queue.shift();
    cells.push([row, col]);
    for (const [dr, dc] of [[-1,0],[1,0],[0,-1],[0,1]]) {
      const nr = row + dr, nc = col + dc;
      if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) continue;
      const idx = nr * cols + nc;
      if (visited[idx]) continue;
      visited[idx] = 1;
      const v = board[idx];
      if (v === 0) liberties++;
      else if (v === color) queue.push([nr, nc]);
    }
  }
  return { cells, liberties };
}

function checkCaptures(board, cols, rows, r, c, opponent) {
  for (const [dr, dc] of [[-1,0],[1,0],[0,-1],[0,1]]) {
    const nr = r + dr, nc = c + dc;
    if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) continue;
    if (board[nr * cols + nc] !== opponent) continue;
    const { cells, liberties } = floodFill(board, cols, rows, nr, nc, opponent);
    if (liberties === 0) {
      for (const [row, col] of cells) board[row * cols + col] = 0;
    }
  }
}

export function replayMain(board, cols, rows, tree) {
  for (const node of tree.nodes) {
    const { props } = node;
    for (const coord of (props.AB ?? [])) {
      const c = sgfCoord(coord[0]), r = sgfCoord(coord[1]);
      if (r >= 0 && r < rows && c >= 0 && c < cols) board[r * cols + c] = 1;
    }
    for (const coord of (props.AW ?? [])) {
      const c = sgfCoord(coord[0]), r = sgfCoord(coord[1]);
      if (r >= 0 && r < rows && c >= 0 && c < cols) board[r * cols + c] = -1;
    }
    for (const coord of (props.AE ?? [])) {
      const c = sgfCoord(coord[0]), r = sgfCoord(coord[1]);
      if (r >= 0 && r < rows && c >= 0 && c < cols) board[r * cols + c] = 0;
    }
    for (const [prop, color] of [['B', 1], ['W', -1]]) {
      const val = props[prop]?.[0];
      if (val === undefined) continue;
      if (val === '' || val === 'tt') continue; // pass
      const c = sgfCoord(val[0]), r = sgfCoord(val[1]);
      if (r < 0 || r >= rows || c < 0 || c >= cols) continue;
      board[r * cols + c] = color;
      checkCaptures(board, cols, rows, r, c, -color);
    }
  }
  if (tree.variations?.[0]) replayMain(board, cols, rows, tree.variations[0]);
}

// Compute the visible viewport [vR0,vR1] × [vC0,vC1] for a replayed board.
// Starts from the stone bounding box with 2-cell padding, then applies
// heuristics to snap boundaries to real board edges:
//   A — game metadata (PB/PW/RE/KM) → force full board
//   B — no setup stones (AB/AW) → pure move sequence is a game → force full board
//   C — per-edge proximity: snap each edge independently when the bounding box
//       comes within K lines (corner hoshi + 1: 4 for ≥13, 3 for smaller boards)
export function calculateViewport(board, cols, rows, rootProps, tree) {
  let minR = rows, maxR = -1, minC = cols, maxC = -1;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (board[r * cols + c] !== 0) {
        if (r < minR) minR = r;
        if (r > maxR) maxR = r;
        if (c < minC) minC = c;
        if (c > maxC) maxC = c;
      }
    }
  }
  if (maxR < 0) { minR = 0; maxR = rows - 1; minC = 0; maxC = cols - 1; } // empty board

  const Kc = cols >= 13 ? 4 : 3, Kr = rows >= 13 ? 4 : 3;
  const hasGameMeta    = ['PB','PW','RE','KM'].some(p => rootProps[p]);
  const hasSetupStones = tree.nodes.some(n => n.props.AB?.length || n.props.AW?.length);
  const forceFullBoard = hasGameMeta || !hasSetupStones;

  return {
    vR0: forceFullBoard || minR <= Kr        ? 0        : Math.max(0,        minR - 2),
    vR1: forceFullBoard || maxR >= rows-1-Kr ? rows - 1 : Math.min(rows - 1, maxR + 2),
    vC0: forceFullBoard || minC <= Kc        ? 0        : Math.max(0,        minC - 2),
    vC1: forceFullBoard || maxC >= cols-1-Kc ? cols - 1 : Math.min(cols - 1, maxC + 2),
  };
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
    let cols, rows;
    if (szProp.includes(':')) {
      const parts = szProp.split(':');
      cols = parseInt(parts[0], 10);
      rows = parseInt(parts[1], 10);
    } else {
      cols = rows = parseInt(szProp, 10) || 19;
    }
    if (cols < 1 || cols > 25 || rows < 1 || rows > 25) return;

    const board = new Int8Array(cols * rows);
    replayMain(board, cols, rows, tree);

    let { vR0, vR1, vC0, vC1 } = calculateViewport(board, cols, rows, rootProps, tree);
    if (opts.forceFullBoard) { vR0 = 0; vR1 = rows - 1; vC0 = 0; vC1 = cols - 1; }

    const R      = CELL * ss;
    const boardW = (vC1 - vC0) * CELL;
    const boardH = (vR1 - vR0) * CELL;
    const totalW = boardW + M * 2;
    const totalH = boardH + M * 2;

    const svg = svgEl('svg', {
      viewBox: `0 0 ${totalW} ${totalH}`,
      width: '100%',
      style: 'display:block',
    });

    // Board background
    svg.appendChild(svgEl('rect', {
      x: 0, y: 0, width: totalW, height: totalH,
      fill: '#dcb483',
    }));

    // Grid lines
    for (let i = vC0; i <= vC1; i++) {
      svg.appendChild(svgEl('line', {
        x1: M + (i - vC0) * CELL, y1: M,
        x2: M + (i - vC0) * CELL, y2: M + boardH,
        stroke: '#7a5230', 'stroke-width': 0.8,
      }));
    }
    for (let i = vR0; i <= vR1; i++) {
      svg.appendChild(svgEl('line', {
        x1: M,          y1: M + (i - vR0) * CELL,
        x2: M + boardW, y2: M + (i - vR0) * CELL,
        stroke: '#7a5230', 'stroke-width': 0.8,
      }));
    }

    // Tick overhangs at interior crop boundaries — imply more board beyond
    const TICK = CELL / 2;
    const tStroke = { stroke: '#7a5230', 'stroke-width': 0.8 };
    if (vR0 > 0) { // top is interior: extend each column line upward
      for (let i = vC0; i <= vC1; i++) {
        const x = M + (i - vC0) * CELL;
        svg.appendChild(svgEl('line', { x1: x, y1: M, x2: x, y2: M - TICK, ...tStroke }));
      }
    }
    if (vR1 < rows - 1) { // bottom is interior: extend each column line downward
      for (let i = vC0; i <= vC1; i++) {
        const x = M + (i - vC0) * CELL;
        svg.appendChild(svgEl('line', { x1: x, y1: M + boardH, x2: x, y2: M + boardH + TICK, ...tStroke }));
      }
    }
    if (vC0 > 0) { // left is interior: extend each row line leftward
      for (let i = vR0; i <= vR1; i++) {
        const y = M + (i - vR0) * CELL;
        svg.appendChild(svgEl('line', { x1: M, y1: y, x2: M - TICK, y2: y, ...tStroke }));
      }
    }
    if (vC1 < cols - 1) { // right is interior: extend each row line rightward
      for (let i = vR0; i <= vR1; i++) {
        const y = M + (i - vR0) * CELL;
        svg.appendChild(svgEl('line', { x1: M + boardW, y1: y, x2: M + boardW + TICK, y2: y, ...tStroke }));
      }
    }

    // Star points (hoshi) — square boards only, rectangular boards get none
    if (cols === rows) {
      for (const [hr, hc] of hoshiPoints(cols)) {
        if (hr < vR0 || hr > vR1 || hc < vC0 || hc > vC1) continue;
        svg.appendChild(svgEl('circle', {
          cx: M + (hc - vC0) * CELL,
          cy: M + (hr - vR0) * CELL,
          r: 3,
          fill: '#7a5230',
        }));
      }
    }

    // For 1×n or n×1 boards, grid lines don't cross so intersections are invisible —
    // mark every intersection in the viewport with a hoshi dot
    if (cols === 1 || rows === 1) {
      for (let r = vR0; r <= vR1; r++) {
        for (let c = vC0; c <= vC1; c++) {
          svg.appendChild(svgEl('circle', {
            cx: M + (c - vC0) * CELL,
            cy: M + (r - vR0) * CELL,
            r: 3,
            fill: '#7a5230',
          }));
        }
      }
    }

    // Coordinate labels — prefer real board edges; fall back to top/left when
    // neither side of an axis touches a real edge (e.g. interior crop).
    const lg      = opts.labelGap ?? 8;
    const colTopY = M - R - lg - 3;
    const colBotY = M + boardH + R + lg + 7;
    const rowLX   = M - R - lg;
    const rowRX   = M + boardW + R + lg;

    const topIsEdge   = vR0 === 0,        botIsEdge   = vR1 === rows - 1;
    const leftIsEdge  = vC0 === 0,        rightIsEdge = vC1 === cols - 1;
    const showTop     = topIsEdge  || !botIsEdge;   // default to top
    const showBot     = botIsEdge;
    const showLeft    = leftIsEdge || !rightIsEdge; // default to left
    const showRight   = rightIsEdge;

    for (let i = vC0; i <= vC1; i++) {
      const x = M + (i - vC0) * CELL;
      const label = COL_LETTERS[i] ?? '';
      if (showTop) {
        const tTop = svgEl('text', { x, y: colTopY, 'text-anchor': 'middle', 'font-size': 10, fill: '#5a3a1a' });
        tTop.textContent = label;
        svg.appendChild(tTop);
      }
      if (showBot) {
        const tBot = svgEl('text', { x, y: colBotY, 'text-anchor': 'middle', 'font-size': 10, fill: '#5a3a1a' });
        tBot.textContent = label;
        svg.appendChild(tBot);
      }
    }

    for (let i = vR0; i <= vR1; i++) {
      const y = M + (i - vR0) * CELL;
      const label = String(rows - i);
      if (showLeft) {
        const tLeft = svgEl('text', { x: rowLX, y: y + 4, 'text-anchor': 'end', 'font-size': 10, fill: '#5a3a1a' });
        tLeft.textContent = label;
        svg.appendChild(tLeft);
      }
      if (showRight) {
        const tRight = svgEl('text', { x: rowRX, y: y + 4, 'text-anchor': 'start', 'font-size': 10, fill: '#5a3a1a' });
        tRight.textContent = label;
        svg.appendChild(tRight);
      }
    }

    // Stones
    for (let r = vR0; r <= vR1; r++) {
      for (let c = vC0; c <= vC1; c++) {
        const v = board[r * cols + c];
        if (v === 0) continue;
        const cx = M + (c - vC0) * CELL + posJitter(r, c, 0) * jt;
        const cy = M + (r - vR0) * CELL + posJitter(r, c, 1) * jt;
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

    // Markup — drawn on top of stones, from the last displayed node
    let dispTree = tree;
    while (dispTree.variations?.[0]) dispTree = dispTree.variations[0];
    const dispProps = dispTree.nodes[dispTree.nodes.length - 1]?.props ?? {};

    const mw = 1.5; // mark stroke-width
    for (const coord of (dispProps.TR ?? [])) {
      const mc = sgfCoord(coord[0]), mr = sgfCoord(coord[1]);
      if (mr < vR0 || mr > vR1 || mc < vC0 || mc > vC1) continue;
      const cx = M + (mc - vC0) * CELL, cy = M + (mr - vR0) * CELL;
      const v = board[mr * cols + mc];
      const col = v === 1 ? '#f5f5f0' : '#1a1a1a';
      const r = R * 0.52;
      svg.appendChild(svgEl('polygon', {
        points: `${cx},${cy - r} ${cx - r * 0.866},${cy + r * 0.5} ${cx + r * 0.866},${cy + r * 0.5}`,
        fill: 'none', stroke: col, 'stroke-width': mw,
      }));
    }
    for (const coord of (dispProps.SQ ?? [])) {
      const mc = sgfCoord(coord[0]), mr = sgfCoord(coord[1]);
      if (mr < vR0 || mr > vR1 || mc < vC0 || mc > vC1) continue;
      const cx = M + (mc - vC0) * CELL, cy = M + (mr - vR0) * CELL;
      const v = board[mr * cols + mc];
      const col = v === 1 ? '#f5f5f0' : '#1a1a1a';
      const hs = R * 0.42;
      svg.appendChild(svgEl('rect', {
        x: cx - hs, y: cy - hs, width: hs * 2, height: hs * 2,
        fill: 'none', stroke: col, 'stroke-width': mw,
      }));
    }
    for (const coord of (dispProps.CR ?? [])) {
      const mc = sgfCoord(coord[0]), mr = sgfCoord(coord[1]);
      if (mr < vR0 || mr > vR1 || mc < vC0 || mc > vC1) continue;
      const cx = M + (mc - vC0) * CELL, cy = M + (mr - vR0) * CELL;
      const v = board[mr * cols + mc];
      const col = v === 1 ? '#f5f5f0' : '#1a1a1a';
      svg.appendChild(svgEl('circle', {
        cx, cy, r: R * 0.5, fill: 'none', stroke: col, 'stroke-width': mw,
      }));
    }
    for (const coord of (dispProps.MA ?? [])) {
      const mc = sgfCoord(coord[0]), mr = sgfCoord(coord[1]);
      if (mr < vR0 || mr > vR1 || mc < vC0 || mc > vC1) continue;
      const cx = M + (mc - vC0) * CELL, cy = M + (mr - vR0) * CELL;
      const v = board[mr * cols + mc];
      const col = v === 1 ? '#f5f5f0' : '#1a1a1a';
      const d = R * 0.4;
      const ls = { stroke: col, 'stroke-width': mw, 'stroke-linecap': 'round' };
      svg.appendChild(svgEl('line', { x1: cx-d, y1: cy-d, x2: cx+d, y2: cy+d, ...ls }));
      svg.appendChild(svgEl('line', { x1: cx+d, y1: cy-d, x2: cx-d, y2: cy+d, ...ls }));
    }

    if (opts.onClick) {
      const g = svgEl('g', { style: 'cursor: crosshair' });
      for (let r = vR0; r <= vR1; r++) {
        for (let c = vC0; c <= vC1; c++) {
          const rect = svgEl('rect', {
            x: M + (c - vC0) * CELL - CELL / 2, y: M + (r - vR0) * CELL - CELL / 2,
            width: CELL, height: CELL, fill: 'transparent', 'pointer-events': 'all',
          });
          rect.addEventListener('click', () => opts.onClick(r, c));
          g.appendChild(rect);
        }
      }
      svg.appendChild(g);
    }

    container.appendChild(svg);
  } catch {
    // silently skip on any error (malformed SGF edge cases)
  }
}
