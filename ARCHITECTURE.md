# Architecture

## Overview

**kifu** is a static site for displaying and sharing Go game records (SGF files) via URL fragments. SGF data is compressed and base64url-encoded into the fragment — no server, no database, no user data stored anywhere. The page decodes the fragment client-side and renders the board.

An optional photo pipeline extracts board positions from images using OpenCV.js. There are two distinct input categories — **diagrams** (book pages, screenshots, app renders) and **real-world board photos** — which share core grid-detection machinery but differ in preprocessing, tuning, and classification strategy.

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

### Diagrams vs real-world photos

The pipeline handles two categories of input:

- **Diagrams**: book pages, app screenshots, digital renders. High contrast, clean lines, flat printed circles for stones. The current pipeline is well-tuned for these.
- **Real-world board photos**: photos of physical boards with wooden texture, 3D stones (specular highlights, shadows), and variable lighting. Requires different preprocessing and tuning.

Both share the same core path — board detection → rectification → grid detection → classification — but diverge in parameters and (eventually) classification strategy. `pipeline-defaults.ts` defines all tunables with ablation variants for both cases.

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

## Real-world board photos

### Assumptions (current scope)

- There is a valid board in the image
- Overhead photo with roughly axis-aligned board
- 19×19 grid

### Approach

1. **Board isolation** — identify the board as quickly as possible and discard everything else. The board boundary is the highest-value signal; all downstream steps operate on the cropped/rectified board region only.
2. **Grid detection** — same `detectGrid` machinery as diagrams, but with different tuning (e.g. larger Gaussian blur to suppress wood grain, lower HoughCircles `param2` for 3D stones with specular highlights).
3. **Illumination normalization** — fit a low-order illumination model (bilinear or biquadratic surface) to empty-intersection brightness, then normalize each patch before classification. The grid provides a dense, evenly-spaced sampling of the board surface for this fit. Without this, lighting gradients across the board cause the checker shadow illusion — a white stone in shadow can appear darker than a black stone in direct light.
4. **Stone classification** — k-means brightness classifier works for well-lit overhead shots; ONNX CNN classifier planned for varied lighting (dim, warm, mixed).

### Key differences from diagrams

- **Wood grain noise**: real boards have visible grain → spurious Canny edges and Hough lines. Needs larger Gaussian blur before edge detection.
- **3D stones**: specular highlights and shadows differ from flat printed circles. HoughCircles may lock onto the inner highlight at ~half the true radius; `circleSweep` needs a lower accumulator threshold (`circleParam2`) and the step hint sanity check to catch half-radius detections.
- **Board vs background**: contour detection struggles when board and surface (e.g. wood floor) share similar color/tone.
- **Lighting variation**: white stones appear gray or translucent under warm/dim light, breaking global brightness classification.

### Alternative grid detection approaches (brainstorm)

The current vote-and-fit pipeline (Harris + HoughCircles + HoughLines → bin → fitGrid) works well for diagrams but struggles with real-world photos: wood grain, stone bowls, and floor produce too many spurious features, and the result is sensitive to getting the initial bounds right.

These approaches are not mutually exclusive — several could combine.

**Multi-threshold Canny tensor.** Instead of one (lo, hi) threshold pair, compute edges at 5–10 pairs and stack them into a volume. A pixel that's an edge across many thresholds is a "strong" edge (grid line); one that only appears at low thresholds is "weak" (wood grain). Grid line features would have high support across the stack; noise would not. This gives a soft edge strength measure while preserving Canny's directional/NMS benefits over raw gradient magnitude. Each threshold layer can be treated as an independent observation in a probabilistic model.

**Frequency-domain grid finding.** The board has a strong periodic signal (regularly-spaced grid lines). 2D FFT of the rectified image should show peaks at the grid spatial frequency. Peak location → step size and angle directly. Phase alignment → grid origin. Robust to local noise because it uses global periodic structure. Autocorrelation (spatial domain equivalent) is another way to extract the same signal.

**Vanishing point perspective recovery.** Instead of rectify-then-detect, detect the two families of converging lines first. Their vanishing points define the homography. This could replace contour-based board detection entirely and works naturally for angled shots. Barrel distortion can be calibrated from the grid lines themselves (straight lines that appear curved → solve for radial distortion coefficients).

**Local crawling / BFS grid discovery.** Seed from a high-confidence feature (detected stone, clear line intersection), then explore outward by following local edge/gradient evidence to adjacent intersections at the expected spacing. Build the grid incrementally. No global bounds needed — the grid grows organically and stops when evidence drops below threshold (board edge). Naturally handles partial boards, occlusion, and even perspective (the local step/angle adapts across the image). Could be formulated as RL or as a simpler greedy BFS.

**Bayesian grid inference.** Define a generative model: grid parameters (center, step, angle, N, perspective, distortion) → predicted image features (line positions, circle positions, edge density patterns). Invert: given observed features, infer the posterior over grid parameters. Key benefits: (1) calibrated uncertainty — occluded regions have wide posterior, clear regions have narrow, rather than the algorithm failing or guessing; (2) active learning — present low-confidence regions to the user for targeted input ("is there a stone here?", "does the grid extend this far?"), resolving ambiguity efficiently; (3) multi-threshold Canny layers and other feature sources are naturally incorporated as independent observations that tighten the posterior. Implementation could range from MAP estimation to MCMC to variational inference.

**RANSAC grid model.** Sample small subsets of detected features, fit a full grid model (origin, step, angle, dimensions), score by how many other features agree. The regularity constraint rejects spurious features that don't fit the periodic pattern. Lightweight version of Bayesian inference.

**Iterative re-bounding.** Run a first pass with loose or no bounds to get approximate grid geometry, then derive tight bounds from the detected grid's center and step, and re-run. Addresses the bootstrap problem where good bounds require knowing the grid, but finding the grid requires good bounds.

### Pipeline architecture (brainstorm)

The pipeline is a single long orchestrator (`runPipeline`) calling into a large bag of functions (`photo-pipeline.ts`, ~2700 lines). Adding new strategies (Bayesian, BFS, multi-threshold) on top of this will compound the complexity. Some structural directions to consider:

**Stage-based pipeline with explicit data flow.** Each stage is a function with a typed input and typed output. Stages compose linearly (or branch). The orchestrator just chains them — no ambient state. Makes it easy to swap, skip, or insert stages without touching the rest.

```
Input → ClassifyInput → FindBoard → Rectify → FindGrid → Dewarp → Classify → Output
```

**Strategy pattern for grid detection.** Grid detection isn't one algorithm — it's a family. Define an interface (`GridDetector`) and let different strategies implement it: the current vote-and-fit, a future Bayesian approach, a BFS crawler, etc. The orchestrator picks or combines strategies based on input type, confidence, or user preference.

**Confidence as a first-class data type.** Instead of stages returning "the answer," they return an answer with uncertainty attached. Downstream stages can use this: low-confidence grid → ask user for input, or try a different strategy. This is the Bayesian idea applied to the software architecture, not just the math.

**Separate feature extraction from model fitting.** Currently `detectGrid` does both — it extracts Harris corners, circles, Hough lines AND fits the grid model. If feature extraction were a separate step, multiple model-fitting strategies could share the same features. The multi-threshold Canny tensor idea fits here naturally as a richer feature extraction layer.

**Mat lifecycle management (implemented).** `MatScope` (`src/mat-scope.ts`) tracks CvMat allocations and bulk-deletes them on scope exit. Replaces the ad-hoc `toDelete[]`/`mat()` pattern. Already adopted in `run-pipeline.ts`, `proto-photo.html`, and `fixture-editor.html`. Individual functions in `photo-pipeline.ts` still manage Mats manually — these can be migrated incrementally as they're refactored into stages.
```

## Eval and testing infrastructure

- **Unit tests** (`test/photo-pipeline.test.js`): test pipeline functions in isolation
- **E2e tests** (`test/photo-pipeline-e2e.test.js`): run full pipeline against fixture images, check stone accuracy and grid position error
- **Evals** (`evals/photo-pipeline.eval.jsonl`): per-commit pipeline metrics, appended automatically by a post-commit hook
- **Ablation** (`scripts/photo-pipeline-ablation.js`): stage toggles + parameter sweeps with reporting (`scripts/ablation-report.js`)
