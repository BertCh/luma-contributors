# Export the H3 face constants as supported read-only data

Status: draft, not filed. Target: `@luma.gl/shaders` 10.0.0-alpha.16. Tracking: DGGS-115;
proposal 11. This is a data-export request, separate from standalone uint64 arithmetic.

## Current public route and consumer

The published `dggs` export from `@luma.gl/shaders` provides WGSL source. Two functions embed
tables used by downstream extended H3 indexing:

| Current WGSL function | Exact data requested | Proposed public name (not available today) |
| --- | --- | --- |
| `dggs_h3_get_face_center_geo` | 20 rows indexed by face 0–19, each `[latitude, longitude]` in radians | `H3_FACE_CENTER_GEO` |
| `dggs_h3_get_face_axis_azimuths` | 20 rows indexed by face 0–19, each three axis azimuths in radians, preserving existing component order | `H3_FACE_AXIS_AZIMUTHS` |

`modules/analysis/src/gpu-spatial-analysis/cell-indexing/h3-index-extended-wgsl.ts` imports only
the public `dggs` export, then `parseDggsFaceTable` extracts `vec2f`/`vec3f` literals from its
source at module load. `createFaceBasisVectors` derives the Cartesian face centre and tangent
axes. `getFaceBasisVectors` also serves `cpu-point-to-cell-operation.ts`,
`cpu-cell-measures-operation.ts` and `h3-measures-fp64-wgsl.ts` in the same directory.

The parser already checks exactly 20 finite rows with the expected number of components and
throws on changes. This prevents silent malformed arrays; a harmless shader refactor can still
break module loading because source formatting is not a supported data contract. No private or
deep package import is proposed as a workaround.

## Requested contract

Export both tables through `@luma.gl/shaders` (names negotiable) as nested readonly numeric
tuples, freezing each row and the outer array at runtime. Preserve the canonical numeric
constants as JavaScript numbers, without rounding to float32 before export, and generate the
WGSL table literals from that same source. Keep existing WGSL function names and behavior.
Document face indexing, angle units, centre component order and azimuth component order.
Retain applicable H3 third-party attribution and licensing with the canonical data.

The two tables above are the concrete requirement of this consumer. Additional H3 tables can be
proposed with their own named consumer and shape; this request does not assume an unnamed
four-table contract.

## Acceptance and downstream deletion

1. Import the tables through the public package entry point in ESM and CommonJS consumers.
   Assert lengths 20×2 and 20×3, finite values and nested immutability.
2. Compare all 100 numeric components with an independently pinned H3 reference revision;
   record that revision and its attribution. Verify the emitted WGSL literals round to the same
   binary32 words as the exported constants. A comparison of two generated outputs alone is
   insufficient to establish correct canonical values.
3. Retain extended-index and cell-measure parity for all faces, pentagons and seam controls.
   Record installed versions and adapter for device tests; no GPU run was made for this draft.
4. Replace the two source-text parses with public table imports and delete `parseDggsFaceTable`
   only after the export and parity checks pass. Keep the domain face-basis derivation and its
   shared cache. No indexing algorithm is moved or re-exported.

Alternative: a supported typed table property on `dggs` could satisfy the same shape, precision,
immutability and public-entry-point contract. Parsing arbitrary WGSL formatting remains fragile.
