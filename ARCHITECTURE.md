# Architecture

## Overview

**kifu** is a static site for displaying and sharing Go game records (SGF files) via URL fragments. SGF data is compressed and base64url-encoded into the fragment — no server, no database, no user data stored anywhere. The page decodes the fragment client-side and renders the board.

An optional photo pipeline extracts board positions from images of Go diagrams (books, screenshots, apps) using OpenCV.js.

## Directory structure

```
src/                    Vite root — all browser-facing code
  index.html            App shell (markup + styles)
  main.js               App logic (extracted from inline script)
  encode.js             SGF text → compressed → base64url → URL fragment
  decode.js             URL fragment → base64url → decompress → validate → SGF
  sgf-parser.js         Recursive descent SGF parser/validator
  goban.js              Board renderer (SVG-based, handles markup: TR/SQ/CR/MA)
  photo-pipeline.js     Pure computation for all pipeline stages (~91 KB)
  run-pipeline.js       Pipeline orchestration with DI for overrides and callbacks
  pipeline-defaults.js  Default parameter values for the pipeline
  style.css             Styles
  favicon.svg           Favicon
  public/vendor/        Static assets copied as-is to dist/
    qrcode.js           Vendored QR code library (MIT)

dev/                    Dev-only HTML tools (not in prod builds)
  proto-photo.html      Interactive pipeline debugger: sliders, debug viz, hover panel
  fixture-editor.html   Ground-truth annotation editor for test fixtures

test/                   Node test runner tests
  helpers/
    load-cv.js          Loads opencv-wasm for Node tests
    load-image.js       Image loading utilities for tests
  *.test.js             Unit and e2e tests

scripts/                CLI tools
  photo-pipeline-ablation.js   Ablation experiment runner
  ablation-report.js           Ablation result analysis
  generate-fixtures.js         Fixture generation from images

fixtures/               Test fixture data (images + ground-truth JSON)
evals/                  Eval output (gitignored, append-only)
dist/                   Vite build output (gitignored)
```

## URL encoding format

```
https://hyponymous.github.io/kifu/#<base64url-encoded-compressed-sgf>
```

- Base64url alphabet (`+` → `-`, `/` → `_`, no padding) avoids fragment encoding issues
- Compression uses gzip via the browser's `CompressionStream`/`DecompressionStream` API (zero dependencies)
- Compression algorithm detected from magic bytes (gzip: `1f 8b`) — no version prefix needed
- Decompressed payload capped at 1 MB (zip bomb protection)

## SGF grammar

```
tree     = "(" node+ tree* ")"
node     = ";" property*
property = IDENT value+
value    = "[" content "]"
```

Tricky bits: escaped `\\` and `\]` in values, nested subtrees (variations), soft vs hard line breaks.

## Data flow: photo → construct → share

```
User uploads/pastes image
  → photo-pipeline.js  (OpenCV.js, lazy-loaded from CDN on first use)
  → { nRows, nCols, grid: 2D array of 'B'|'W'|'.' }
  → populate constructStones Map  (grid[r][c] → Map key "r,c" → 'B'|'W')
  → enterConstruct()  (user can fix misdetections by clicking)
  → generateConstructSGF() → encode() → URL fragment  (existing sharing flow)
```

## Photo pipeline

OpenCV.js loaded from CDN (~8 MB), only when needed.

### Step 1 — board detection

- Canny edge detection → dilate → `findContours`
- Pick the largest roughly-quadrilateral contour as the board boundary
- Expand detected corners outward by ~2 cell widths so edge stones aren't clipped
- Falls back to full image if no quad found
- **Hough quad refinement** (`refineQuadWithHough`): snaps each quad edge to the nearest Hough line (within 15 deg and 15% of crop size), then intersects the 4 matched lines for refined corners. Fixes shear from stone edges or text merging with the board border during dilation.

### Step 2 — rectification

- `getPerspectiveTransform` + `warpPerspective` to a rectangle
- Output capped at 1200 px on longest side

### Step 3 — grid detection

Pipeline order:

1. **Harris corners** — `goodFeaturesToTrack` on rectified image → grid intersection candidates
2. **HoughCircles radius sweep** — sweep 6 candidate steps from W/30 to W/8 with downsampling (target maxR <= 25 px); pick the candidate with the most detections. Eliminates dependency on estimated step.
3. **HoughLines with circle masking** — Canny edges with circle regions masked out for clean grid-only lines
4. **Grid angle** from Hough line thetas (median deviation from ideal axis)
5. **Angle sweep** — narrow +/-2 deg around Hough angle with Harris points, or full sweep as fallback
6. **Vote pools** — Hough line positions + circle centers only (Harris corners excluded from votes to prevent half-step contamination)
7. **`fitGrid`** — clusters → consecutive-diff index assignment with running median step → iterative quadratic model refit → re-bin → converge. Supports arbitrary board dimensions.
8. **Harris refinement** — after fitGrid locks topology, bin filtered Harris corners against the model, merge, refit for sub-pixel accuracy

Key functions:
- `clusterPositions`: merges nearby votes (tolerance 20% of step), returns `{pos, count}` with group-mean comparison (prevents chaining across adjacent lines)
- `lineQuality` + `trimBadEdges`: reject spurious edge lines using median gap check and vote count check (>= 25% of interior median). Hough-confirmed lines skip quality checks.

### Step 4 — per-intersection classification

Two-stage radial gradient coherence classifier.

Stone radius R from HoughCircles median; fallback to step * 0.5.

**Stage 1 — stone vs empty (k-means k=2 on `radial_power`):**

For each pixel in the ring annulus [0.85R, 1.05R], compute the fraction of gradient energy oriented radially:

```
radial_power = sum(dot(grad, r_hat)^2) / sum(|grad|^2)
```

- Stone borders → strongly radial gradients → high score
- Grid line crossings → axis-aligned → low score
- Hoshi dots → smaller than annulus → low score
- Optional **quad mask**: excludes ring pixels within +/-arcsin(R/step) of each cardinal axis to prevent adjacent stone bleed

**Stage 2 — black vs white (k-means k=2 on `center_body`, stones only):**

Mean brightness of the body annulus [0.35R, 0.85R]. Black stones are dark, white stones are bright. Uses annulus instead of center disc to avoid number text at the center.

**Line signal (optional):** For W-classified intersections, run HoughLines on a per-intersection Canny patch. If both H and V lines detected through the center, the grid cross is visible → reclassify as empty.

### Dewarping

Handles barrel distortion (phone lens) and page curl (book spine) by treating detected grid intersections as a calibration pattern.

**Warp models:**

- **Separable quadratic**: fits two independent 1-D quadratics (`dy(y)`, `dx(x)`) to a TILES x TILES mesh of local grid offsets. Low-order, robust to noisy measurements, extrapolates well.
- **Piecewise cylinder**: for non-uniform curl. Each segment fits the exact cylinder formula `src_y = yc + R*sin((y-yc)/R)` via Gauss-Newton. BIC auto-selects k in {1, 2, 3} segments. Cubic Hermite blending at breakpoints.

**TPS (Thin Plate Spline):**

Per-intersection combination: group line-y measurements by row, line-x by column; fit a 1-D linear curve per row and column; combine at each intersection for 2D control points. Circles override both axes. Pilot TPS provides fallback for sparse rows/columns and the border ring.

Coarse-then-fine: first pass gives rough intersections; one dewarp + re-detect iteration is usually sufficient.

## Eval and testing infrastructure

- **Unit tests** (`test/photo-pipeline.test.js`): test pipeline functions in isolation
- **E2e tests** (`test/photo-pipeline-e2e.test.js`): run full pipeline against fixture images, check stone accuracy and grid position error
- **Evals** (`evals/photo-pipeline.eval.jsonl`): per-commit pipeline metrics, appended automatically by a post-commit hook
- **Ablation** (`scripts/photo-pipeline-ablation.js`): stage toggles + parameter sweeps with reporting (`scripts/ablation-report.js`)
