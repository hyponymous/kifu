# SGF Share Site — Implementation Plan

A static site that encodes SGF game/problem files into a URL fragment, allowing sharing and download with no server required.

## Concept

- SGF data is compressed and base64-encoded into the URL fragment (`#...`)
- The fragment is never sent to the server — purely client-side
- The page decodes the fragment and offers a file download
- No backend, no database, no user data stored anywhere

## URL Format

```
https://yoursite.com/#<base64url-encoded-compressed-sgf>
```

A version prefix (e.g. `v1:`) is optional — compression algorithm can be detected from magic bytes in the decoded data. A prefix is only needed if the overall format structure changes significantly.

Using base64url (URL-safe alphabet: `+` → `-`, `/` → `_`, no padding) to avoid any encoding issues in the fragment.

## Phases

### Phase 1: Download Only

**Encoding (to generate a shareable URL):**
1. Take SGF text input
2. Validate SGF (see below)
3. Compress with gzip (via `CompressionStream` API, built into modern browsers)
4. Base64url-encode the result
5. Set as the page fragment

**Decoding (on page load):**
1. Read fragment, strip version prefix
2. Base64url-decode
3. **Check decompressed size before expanding** — abort if > 1 MB (zip bomb protection)
4. Decompress with `DecompressionStream`
5. Validate SGF
6. Offer download via a `Blob` URL with a `.sgf` extension

### Phase 2: Rendering (future)

- Add an SGF viewer (e.g. integrate a library, or use the parser built in Phase 1)
- Since the parser is already built, rendering is mostly a UI problem

## SGF Validation

A full recursive descent parser is ~100–200 lines of JS and worth building upfront — you'll need it for rendering later anyway.

**SGF grammar (simplified):**
```
tree     = "(" node+ tree* ")"
node     = ";" property*
property = IDENT value+
value    = "[" content "]"
```

**Tricky bits to handle:**
- Escaped characters in values: `\\` and `\]`
- Nested subtrees (variations)
- Soft vs hard line breaks in text properties (`\` at end of line = soft break)

**Validation should reject:**
- Data that doesn't parse as valid SGF
- Decompressed payload over size limit (1 MB suggested)
- Any derived filename containing path separators or suspicious characters (if you ever take a filename from the data)

## Security Considerations

| Risk | Mitigation |
|------|-----------|
| Zip bomb | Cap decompressed size before fully expanding |
| Malformed SGF crashing parser | Wrap parser in try/catch; set a parse node limit |
| XSS (future rendering) | Sanitize all text content before inserting into DOM |
| Malicious file distribution | Low risk for SGF (niche plaintext format); full parser ensures only valid SGF is downloadable |

## Compression

Start with gzip (`CompressionStream`/`DecompressionStream`) — it's built into browsers with no dependencies.

**To swap algorithms later:** Most compression formats have magic bytes (gzip: `1f 8b`), so the algorithm can be detected from the decoded data itself without needing a version prefix.

A version prefix is still useful for **structural format changes** (e.g. adding metadata, supporting multiple files) rather than just swapping compression. If the format stays simple, magic byte detection is cleaner — a version prefix can always be added later as a new magic pattern if needed, and old URLs would still decode as the legacy format.

## File Structure

```
/
  index.html       # Single page — encoder UI + decoder logic
  encode.js        # SGF → compressed → base64url → fragment
  decode.js        # Fragment → base64url → decompress → validate → download
  sgf-parser.js    # Recursive descent SGF parser/validator
  style.css
```

Or bundle it all — it's a small enough project to keep simple.

## Open Questions / Future Work

- Add a drag-and-drop / file picker UI for encoding
- Support encoding multiple SGF files (as a collection)?
- Rendering with a board viewer
- Consider a short-link service integration for very long URLs

## Image → Tsumego

Extract a board position from a photo or scan and load it into the construct flow.
OpenCV.js loaded from CDN (`https://docs.opencv.org/4.x/opencv.js`, ~8 MB).
Prototype lives in `proto-photo.html` on the `photo-tsumego` branch.

### Pipeline

**Step 1 — rough board detection:**
- Canny edge detection → dilate → `findContours`
- Pick the largest roughly-quadrilateral contour → board boundary
- Expand detected corners outward by ~2 cell widths (generous, so edge stones aren't clipped)
- Falls back to full image if no quad found

**Step 2 — rectification:**
- `getPerspectiveTransform` + `warpPerspective` to a rectangle sized from detected quad edge lengths
- Output capped at 1200px on longest side

**Step 3 — grid detection (Harris corners + HoughCircles + HoughLines fusion):**
- `goodFeaturesToTrack` (Harris corners) on the rectified image → grid intersection candidates
- `HoughCircles` on the rectified image → stone circle centers
- Canny → circle-masked HoughLines → additional row/column votes (catches sparse edge lines)
- **Key insight:** corners, circles, and lines are complementary: corners detect empty intersections,
  circles detect stones, and HoughLines confirm lines even when corners/circles are sparse.
- `clusterPositions`: merge nearby votes (tolerance = 20% of step), returns `{pos, count}` with
  group-mean comparison (prevents chaining across adjacent lines)
- `fitGrid(positions, hintN)`: clusters → consecutive-diff index assignment with running median step
  (handles perspective-varying spacing) → iterative quadratic model refit → re-bin raw votes →
  converge. Supports arbitrary board dimensions (no 9/13/19 snapping).
- `lineQuality` + `trimBadEdges`: reject spurious edge lines using median gap check (must be near
  integer multiple of step) and vote count check (must be ≥25% of interior median). HoughLines-confirmed
  lines skip quality checks.

**Step 4 — per-intersection classification (2-stage radial gradient coherence):**

Stone radius R from HoughCircles median; fallback to step × 0.5 (measured: outer
radius ≈ 48px on a 93px grid step → fraction ≈ 0.52, rounded to 0.5).

**Stage 1 — stone vs empty (k-means k=2 on `radial_power`):**

For each pixel in the ring annulus [0.85R, 1.05R] around the intersection centre,
compute the fraction of gradient energy oriented radially:

  `radial_power = Σ dot(∇, r̂)² / Σ |∇|²`

where r̂ = (dx, dy)/|(dx, dy)| and pixels with |∇|² < 64 are skipped.

- Stone borders are circular → strongly radial gradients → high score.
- Grid line crossings are axis-aligned → not radial → low score.
- Hoshi dots are smaller than the annulus → negligible edge energy → low score.
- Numbered stones: text is inside the stone, border annulus is unaffected → robust.
- Works for stones without dark borders (app diagrams, web boards, real photos).

Seed k-means at (min, max); higher centroid = stone cluster.

**Stage 2 — B vs W (k-means k=2 on `center_body`, stones only):**

Mean brightness of the central disc (radius 30% of step). Black stones are dark,
white stones are bright. Only run on intersections classified as stone in Stage 1.

**Line signal (optional, checkbox "Lines"):**
For W-classified intersections: run HoughLines on a per-intersection Canny patch
(radius 40% of step — pulled inside cell boundary to avoid adjacent intersections).
If both H and V lines are detected through the centre, the grid cross is visible →
reclassify as empty. Uses rho position check to reject stone-border arcs.

**Removed:** ring brightness (`ring_delta`), hoshi post-pass, k-means k=3.
`circle-sens` slider remains (still needed for grid fitting / HoughCircles).

**Known failure modes:**

- *Empty → white* (Stage 1): When stones nearly fill the cell (2R ≈ step), the
  ring annulus [0.85R, 1.05R] overlaps adjacent stone borders. Adjacent stone edges
  at cardinal positions produce radially-oriented gradients relative to the empty
  centre, inflating radPow above the k-means decision boundary.

- *Numbered stones → wrong color / empty* (Stage 2 + line signal): Number text
  (white on black, or black on white) fills the center disc, pulling `center_body`
  toward mid-grey. Both black and white numbered stones land near the midpoint of
  the [B=12, W=210] centroids and can be misclassified. If misclassified as W,
  the line signal may then see number strokes as grid lines and reclassify as empty.

**Pending classifier improvements (each a checkbox in a Classifier group):**

- *Ring detect* (checkbox): For Stage 2, crop a per-intersection patch, run
  HoughCircles on the grayscale patch (which applies Canny internally). Black stones
  have 1 detectable circle (filled disc, single outer border). White stones have 2
  (hollow ring: outer border + inner border). Numbered stones retain their circular
  border regardless of text. Use circle count as primary B/W signal; fall back to
  `center_body` when no circle is found.

- *Quad mask* (checkbox): For Stage 1, exclude ring pixels within
  ±arcsin(R/step) of each cardinal axis before computing radPow. Adjacent stone
  bleed concentrates at cardinal angles; real stone borders contribute at all angles
  including the unmasked diagonal quadrants. Prevents the empty→stone false positive.

- *Body annulus* (checkbox): For Stage 2, replace the center disc with an annulus
  [0.35R, 0.85R]. Avoids the number text at the very centre while staying inside
  the stone material. Complements or replaces Ring detect.

- *Cap ringOut*: Set ringOut = min(R×1.05, step×0.48) to reduce (but not
  eliminate) adjacent stone bleed when cells are tightly packed.

### Code architecture

**Core modules:**
- `photo-pipeline.js` (~91 KB) — pure computation functions for all pipeline stages
- `run-pipeline.js` (~12 KB) — shared orchestration (`runPipeline()`) with dependency injection
  for function overrides, parameter customization, and callbacks (`onStage`, `onIntermediate`)

**Proto pages:**
- `proto-photo.html` — interactive UI: image upload, parameter sliders, debug visualizations,
  hover panel for per-intersection inspection
- `proto-fixture.html` — fixture editor: create/edit ground-truth annotations, dot overlay
  with pan/zoom, metric display, auto-load fixture JSON

**Test infrastructure:**
- `photo-pipeline.test.js` — unit tests for pipeline functions
- `photo-pipeline-e2e.test.js` — end-to-end tests against fixture images (23 fixtures in `fixtures/`)
- `photo-pipeline-ablation.js` — ablation experiment runner (stage toggles + parameter sweeps)
- `ablation-report.js` — ablation result analysis (summary tables, per-fixture drill-down, timing)
- `generate-fixtures.js` — fixture generation from images
- `test-helpers/load-image.js`, `test-helpers/load-cv.js` — shared test utilities
- `evals/photo-pipeline.eval.jsonl` — per-commit pipeline metrics (append-only, post-commit hook)
- `evals/ablation.eval.jsonl` — ablation study results

### Implementation order
- [x] Prototype scaffolding with contour-based detection (`proto-photo.html`)
- [x] HoughCircles stone detection in rectified image
- [x] HoughLines + HoughCircles fusion grid fitting (`fitGrid`, `detectGrid`)
- [x] Auto board-size detection from grid fitting
- [x] Per-intersection center-patch + border-ring classification (no circle detection in classify path)
- [x] RANSAC-style anchor voting in `fitGrid`; robust to outlier line detections
- [x] Circle-radius step estimation; validated against position-derived step
- [x] Contrast enhancement: global normalize + histogram equalisation blend
- [x] Percentile normalization preprocessing option (see below)
- [x] Dewarping for page curl / barrel distortion — bilinear mesh (see below)
- [x] Separable quadratic warp model: fit 1-D quadratics to tile mesh profiles
- [x] Piecewise cylinder warp: BIC-selected k∈{1,2,3} segments, Gauss-Newton fit, smooth blending
- [x] 2-stage radial gradient coherence classifier (see Step 4 above)
- [x] Optional line-signal checkbox for empty confirmation (HoughLines per patch)
- [x] Biased Stage 1 threshold: `emptyRPCent + 0.85*(stoneRPCent - emptyRPCent)` instead of k-means midpoint
- [x] Body annulus: replace center disc with [0.35R, 0.85R] annulus for Stage 2 (checkbox)
- [x] Quad mask: cardinal-masked radPow for Stage 1 (checkbox)
- [x] Skip detect2 for piecewise: map detect1 positions through inverse warp (Newton iteration)
- [x] Perspective-robust fitGrid: consecutive-diff index assignment with running median step, iterative
  quadratic refit, arbitrary board dimensions (no 9/13/19 snapping)
- [x] Intersections grid: `buildIntersections` computes 2D pixel-space intersection points from
  rotated row/col positions, used throughout for classification, sampling, snapping
- [x] Pre-snap grid to circles: 2D intersection→circle matching replaces independent 1D row/col snapping
- [x] HoughLines as additional grid votes: Canny → mask circles → HoughLines → filter by angle → add votes;
  Hough-confirmed lines skip quality trimming
- [x] Edge trimming: `lineQuality` (median gap) + `trimBadEdges` (vote count + gap ratio) reject text/noise
- [x] TPS output scaling: equalize grid step in both axes by stretching the compressed dimension
  (fixes residual perspective aspect ratio error)
- [x] Extract pipeline into ES module (`photo-pipeline.js` + `run-pipeline.js`)
- [x] Fixture editor (`proto-fixture.html`) with TPS dewarp, dot overlay, pan/zoom
- [x] Elided edge detection via tick marks on Canny edges
- [x] OCR move numbers on stones via Tesseract.js
- [x] Hough line clustering before voting to reduce noise
- [x] Tick mark artifact filtering from grid fitting
- [x] Automated e2e test suite against fixture images
- [x] Eval suite: per-commit pipeline metrics with post-commit hook
- [x] Ablation framework: stage toggles, parameter sweeps, reporting
- [x] DRY proto pages via `sharedRunPipeline` + `onIntermediate` callback
- [x] Debug visualizations and hover panel in proto-photo
- [-] Ring detect: abandoned — HoughCircles minDist=0 problem, edge-count approach had too many false positives
- [ ] Integrate into main app UI

### TPS improvements
- [x] Skip detect2 for TPS: map detect1 positions through inverse warp (2D Newton on evalTPS)
- [x] TPS stiffness λ slider: UI control (saved to localStorage) for regularization strength (already existed)
- [x] Dense grid inference (superseded — see below)
- **Grid anchoring note:** `fitGrid` anchors at `gridStart = bestAnchor + minK*step` (top-left inlier).
  Center anchoring would halve maximum error accumulation (9× vs 18× step_error for 19-line board) but
  doesn't fix the off-axis problem; not worth pursuing separately.
- **Grid overshoot fix (largely addressed):** Edge trimming (`trimBadEdges`) now removes spurious edge
  lines. HoughLines provides evidence for sparse edge lines. Remaining edge case: when no features at
  all exist near a true edge line (fully occluded by stones, no Hough detection).
- [x] Per-intersection combination: group line-y measurements by row, line-x by column; fit a 1D linear
  curve per row (`actual_y(col)`) and per column (`actual_x(row)`); combine at each intersection to get
  proper 2D control points `(actual_x_j(nominalRow_i), actual_y_i(nominalCol_j))`. Circles override both
  axes. Pilot TPS provides fallback for rows/columns with no data and for the one-step border ring.
  Replaces the pilot-TPS-then-resample approach, which still treated axes independently.
- [x] Use TPS control point post-images for classification grid positions: `buildDetectionFromControlPoints`
  inverts each interior control point through the TPS (`invertTPS`), takes per-row median `dy` and
  per-col median `dx`. Replaces `mapDetectionThroughTPS` which approximated with a fixed midCol/midRow
  reference point, losing 2D information.
- **2nd-derivative Lipschitz (future experiment):** The current Lipschitz constraint checks that adjacent
  per-step spacings are within ±20% of each other (1st-difference ratio). Under perspective distortion the
  spacing changes monotonically, so the 1st differences are roughly constant — the 2nd difference (change
  in change) is near zero. Could instead require `|d[i+1]/d[i] - d[i]/d[i-1]| < ε` (2nd-derivative bound)
  which would permit a steady linear drift in spacing (consistent perspective) while still rejecting sudden
  jumps. May allow looser per-step tolerance while remaining robust to noise.

- **sweepFromCenter (removed):** Replaced by consecutive-diff index assignment with running median
  step, which handles perspective-varying spacing natively. The old chain midpoint + centre-trim
  approach had problems with asymmetric detection.

- **Two-pass refinement (future experiment):** After the first `collectOffsetSamples` pass and initial TPS
  fit, map the uniform grid positions (or TPS control point pre-images) back to the original rectified
  image via `invertTPS`, then run a second `collectOffsetSamples` centred at those predicted positions.
  The patches would be better centred, giving more accurate measurements for the final TPS fit. Cost:
  one extra `collectOffsetSamples` pass (≈ same cost as the first). May help most on heavily distorted
  or perspective-skewed images where `uniformRowPos`/`uniformColPos` are far from the actual intersections.

### detectGrid pipeline (747f771)

**Hough quad refinement** (`refineQuadWithHough`, 671dc7a):
- After `findBoardCorners` finds the best quad via contour + `approxPolyDP`, snap each
  quad edge to the nearest Hough line (within 15° and 15% of crop size).
- Intersect the 4 matched lines to get refined corners. Fallback: return original if any
  edge has no match.
- Fixes shear when stone edges or nearby text merge with the board border during dilation.

**detectGrid pipeline** (747f771):
Reordered from `HoughCircles(estStep) → Harris → angle → HoughLines → fitGrid` to:

1. **Harris corners** — `minDistance = max(5, W/60)`, all corners kept (no circle filtering
   at detection time, since circles don't exist yet)
2. **HoughCircles radius sweep** — sweep 6 candidate steps from W/30 to W/8, each with
   downsampling (target maxR ≤ 25px) for speed. Pick the candidate yielding the most
   detections. Eliminates dependency on `estStep = W/(refN+1)` which was wrong for
   non-standard board sizes.
3. **HoughLines with circle masking** — Canny edges with circle regions masked out
   (restores the masking that was lost when circles ran after lines). Clean grid-only lines.
4. **Grid angle from Hough thetas** — classify each line as near-vertical or near-horizontal,
   take median deviation from ideal axis → `houghAngleDeg`.
5. **Angle sweep** — narrow ±2° around Hough angle with Harris points, or full sweep as fallback
6. **Vote pools** — Hough line positions + circle centers only (Harris corners excluded from
   votes to prevent half-step contamination from stone-edge corners not covered by the
   ~35 detected circles). Harris corners are used for angle finding and model refinement.
7. **fitGrid** — fits grid from Hough + circle votes
8. **Harris refinement** — after fitGrid locks in the grid topology, bin filtered Harris
   corners against the model, merge with existing data, refit the quadratic. Improves
   sub-pixel accuracy without risking half-step contamination.

### Intersection localization improvements (priority order)

1. **Harris corner detection for empty intersections (next up):**
   Replace the Canny strip projection (`detectHorizAt`/`detectVertAt`) with `cv.cornerHarris` on a
   circle-masked image. Harris directly finds where two edges meet — exactly what a grid intersection
   is — giving 2D positions in one shot instead of separate 1D x/y estimates. Pipeline becomes:
   stone on intersection → circle center (already works); empty intersection → Harris corner;
   neither → skip. Circle masking prevents stone edges from producing false corners.

2. **Fit a curve per row/col from first-pass detections:**
   Use circle centers + Canny/Harris measurements from pass 1 to fit a low-order curve (quadratic or
   cubic) for each row and column. The curves become the grid lines themselves, not just a dewarp
   correction. Naturally handles perspective distortion where grid lines aren't straight. Subsumes
   `preSnapToCircles` — circle centers become data points for the curve fit. RANSAC or median
   filtering for outlier rejection.

3. **Two-pass grid detection:**
   Pass 1 → dewarp → pass 2 on cleaner image. Infrastructure partially exists (`detection2`).
   With better pass 1 from items 1–2, pass 2 becomes very reliable. Main risk: instability /
   oscillation between passes.

4. **Distance-from-nearest-edge refinement:**
   `cv.distanceTransform` on circle-masked Canny image. Walk each expected intersection to the
   nearest zero along the curve's normal direction (requires item 2). More principled than strip
   projection but depends on distinguishing grid edges from other edges.

5. **Ellipse eccentricity for dewarping signal:**
   Stones are 3D circles → ellipses under perspective. Eccentricity and tilt encode camera angle.
   Requires `cv.fitEllipse` on contours (HoughCircles doesn't detect ellipses). Independent
   dewarping signal, most useful for heavily occupied boards where grid lines are occluded.
   High effort, defer.

6. **Other intersection localization methods considered:**
   - Line intersection geometry: compute where HoughLines cross algebraically (pure math, no image
     processing at the intersection). Depends on HoughLines accuracy.
   - Cross-correlation with a "+" template: match the 2D pattern of two lines crossing.
   - Saddle point detection: grid intersections have saddle-point intensity structure (Hessian with
     mixed eigenvalue signs). Very specific to grid intersections but may be noise-sensitive.

### Preprocessing pipeline

**Percentile normalization** (checkbox) — stretches the [P5, P95] brightness range to
[0, 255] instead of [min, max]. More robust than min-max: ignores stray dark marks and
bright hotspots at the extremes.

### Dewarping — grid-as-calibration warp

Real book photos have two sources of line curvature:
- **Barrel distortion** (phone lens) — uniform radial bowing, well-modelled by k1/k2
- **Page curl** (book spine) — smooth, low-frequency, physically cylindrical

Both are handled in one pass by treating detected grid intersections as a calibration
pattern:

1. Run the existing pipeline to get approximate intersection positions (curved mesh)
2. Compute a deformation map: detected position → ideal uniform-grid position
3. Apply the map via `cv.remap` to produce a flat, undistorted board image
4. Re-run grid detection and classification on the flattened image

**Warp model — separable quadratic (current):**

Page curl is a low-frequency, separable distortion: y-offsets depend primarily on y,
x-offsets primarily on x. We exploit this by fitting two independent 1-D quadratics
to the TILES×TILES mesh measurements:

1. `localGridOffsets` (TILES×TILES mesh) — runs HoughLines + fused HoughCircles per
   tile to measure where each control point actually sits vs its ideal position.
2. `fitSeparableQuadratic` — collapses the 2-D mesh into two 1-D offset profiles by
   taking the median offset across the orthogonal axis, then fits a quadratic
   `Δy(y) = a·y² + b·y + c` (and `Δx(x)`) via 3×3 normal-equations least squares.
3. `dewarpImage` — applies `src_y = y + Δy(y)`, `src_x = x + Δx(x)` to build the
   remap; called by `cv.remap` with `INTER_LINEAR`.

Advantages over raw bilinear mesh: constrained to smooth low-order deformations,
robust to noisy tile measurements, extrapolates well outside the board, and naturally
models the physical cylindrical shape of page curl.

**Piecewise cylinder model (implemented):**

For non-uniform curl (e.g. top half curls away, bottom is flat), the quadratic
can't fit. Piecewise cylinders model each region independently:

- Each segment fits the exact cylinder formula `src_y = yc + R·sin((y−yc)/R)`
  via Gauss-Newton (50 iterations). Guard: if R collapses below 10 px, reset to flat.
- **Gauss-Newton initialisation:** at R=1e6 the argument u=(y−yc)/R~10⁻³,
  so dR=sinU−u·cosU~u³~10⁻⁹ → Jacobian numerically zero → det<1e-20 → breaks
  immediately. Fix: estimate R from displacement magnitude before the loop:
  `R_init = halfSpan·√(halfSpan/(6·maxDisp))` (from the cylinder's cubic leading
  term Δy≈−(y−yc)³/(6R²)). Falls back to 1e6 for nearly flat pages (maxDisp<0.5px).
- **BIC** auto-selects k∈{1,2,3} pieces: penalises 2 params per cylinder + 1 per
  breakpoint. With n≈18 samples, requires SSR to drop >~50% to upgrade k=1→k=2.
- **Breakpoints** found by grid search over interior sample quantiles (≤10 candidates).
- **Smooth blending** at breakpoints: cubic Hermite (smooth-step) over
  `blendW = max(5px, 3% of data range)` so no visible seams.
- **RANSAC pilot filter** (`ransac-thr` slider) cleans 1-D (xs, ys) pairs using
  the global quadratic as a pilot before piecewise fitting.

**Coarse-then-fine:** first pass gives rough intersections (some error from distortion);
one dewarp + re-detect iteration is usually sufficient.

## Integration with main app

The photo pipeline lives on the `photo-tsumego` branch; the main app lives on `canon`.
Integration means merging the pipeline into the `canon` app so users can go from
photo → board position → edit → share in one flow.

### Branch state

Files unique to `photo-tsumego` (will be added to `canon`):
- `photo-pipeline.js`, `run-pipeline.js` — pipeline core
- `proto-photo.html` → move to `dev/proto-photo.html` (prototype, not user-facing)
- `proto-fixture.html` → move to `dev/fixture-editor.html` (rename: it's a proper tool, not a prototype)
- `fixtures/`, `test-helpers/`, `generate-fixtures.js` — test infrastructure
- `photo-pipeline.test.js`, `photo-pipeline-e2e.test.js`, `photo-pipeline-ablation.js`,
  `ablation-report.js` — tests and analysis
- `package-lock.json` — new devDependencies (`opencv-wasm`, `sharp`)

Files modified on `photo-tsumego`:
- `package.json` — added test/ablation scripts + devDependencies

Shared files (`index.html`, `goban.js`, `encode.js`, `decode.js`, `sgf-parser.js`,
`style.css`) are **unchanged** on `photo-tsumego` — merge should be clean.

### Data flow: photo → construct → share

```
User uploads/pastes image
  → photo-pipeline.js  (OpenCV.js, loaded on demand from CDN)
  → { nRows, nCols, grid: 2D array of 'B'|'W'|'.' }
  → populate constructStones Map  (convert grid[r][c] → Map key "${r},${c}" → 'B'|'W')
  → set constructRows, constructCols from nRows, nCols
  → enterConstruct()  (existing construct mode — user can fix misdetections by clicking)
  → generateConstructSGF()  → encode() → URL fragment  (existing sharing flow)
```

No new data formats needed. The pipeline's output maps directly to the construct
mode's `constructStones` Map. The user can correct errors before sharing.

### UI changes to index.html

1. **Add a "From photo" button** next to the existing "Edit position" button in
   `#construct-entry`. Clicking it opens a file picker (accept `image/*`).

2. **Paste support**: listen for `paste` events on the document; if the clipboard
   contains an image, treat it the same as a file upload.

3. **Processing overlay**: while the pipeline runs (~0.5–2 s), show a spinner or
   progress bar over the board area. The pipeline's `onStage` callback provides
   stage names for progress text.

4. **OpenCV.js lazy load**: only fetch the ~8 MB OpenCV.js bundle when the user
   actually clicks "From photo" or pastes an image. Cache the load promise so
   subsequent uses are instant.

5. **After pipeline completes**: call a new `enterConstructFromPhoto(result)` that:
   - Sets `constructRows = result.nRows`, `constructCols = result.nCols`
   - Populates `constructStones` from `result.grid`
   - Calls `renderConstruct()` to show the board
   - Enters construct mode so the user can fix errors

6. **No new pages or routes** — everything stays in `index.html`.

### What stays out of the main app

- `proto-photo.html` / `proto-fixture.html` — keep as dev tools, not linked from UI
- Parameter sliders, debug visualizations, hover panel — proto-only
- The main app calls `runPipeline()` with default parameters; no user-facing knobs
  (if accuracy needs tuning, do it in the pipeline defaults, not the UI)

### Merge plan

1. Ensure `photo-tsumego` is rebased on latest `canon` (shared files are unchanged,
   so this should be trivial)
2. Cherry-pick or merge --squash onto `canon` as a single commit (preserve
   `photo-tsumego` branch history — do not rewrite it)
3. Add the UI changes to `index.html` (button, paste handler, lazy OpenCV load,
   `enterConstructFromPhoto`)
4. Verify: upload an image → see board in construct mode → click to fix → share link works

## Future

### Photo pipeline
- [ ] Adaptive Gaussian kernels by image size (currently fixed; larger images may need wider kernels)
- [ ] Derive ordered moves from numbered diagram — OCR reads numbers, then sequence into an SGF
  move tree; may require human-in-the-loop for ambiguous numbers; consider Viterbi for path
  optimization (see https://claude.ai/share/55c34c9c-fb53-485d-8c1a-bd969b349a3e)
- [ ] Support multiple diagrams per image (detect and segment separate board regions)
- [ ] Stitch multiple diagrams into a single game record / SGF (requires move ordering across diagrams)
- [ ] HEIC support (iOS photo format; needs client-side conversion or server-side transcoding)
- [ ] Support real-world board images (3D wooden boards, lighting variation, stone glare/shadow)
- [ ] Support partial real-world boards (crop of a larger board, unknown board size)
- [ ] Cap ringOut at min(R×1.05, step×0.48) for tightly packed cells

### Site / UX
- [ ] Preview image for external shares (Open Graph meta tags for iMessage, Facebook, etc.)
- [ ] Randomize colors, orientation for fixture/training data (or generate all permutations)
- [ ] Scoring (territory + area scoring across rulesets: Japanese, Chinese, AGA)

### Codebase
- [ ] Write ARCHITECTURE.md
- [ ] Port to TypeScript

## Rendering Backlog

### SGF markup (on the displayed node)
- [x] TR — triangle
- [x] SQ — square
- [x] CR — circle
- [x] MA — X mark
- [ ] LB — text labels (letters, numbers, arbitrary strings)
- [ ] AR — arrows between two points
- [ ] LN — lines between two points
- [ ] TW / TB — territory marks (white/black area, e.g. final position)

### UX / viewer features
- [ ] Last-move indicator — small dot or ring on the most recently played stone
- [ ] Ko ban indicator — mark the intersection forbidden by the ko rule
- [ ] Move numbering — overlay sequence numbers on stones for the last N moves (or all moves)
