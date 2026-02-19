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
