# w3-lib-padding (FX-18)

## 1. Done

All five new specs pass; no phantom-row defect was found in any examined site. One contract defect was found and fixed (concave hull `groupIds` length).

Per site: kernel, rows read, can a read outside owned rows happen, spec.

### Item 1. Line clip by polygon, polygon side
- Kernels: GPULineClipByPolygon takes polygon edges through `GPUSegmentIntersection` (`segment-intersection/segment-table.ts`) and containment of sub-piece midpoints through `GPUPointInPolygonJoin`.
- Rows read: segment table row `i` calls `ringOfVertex(i)` = `lastAtMost(0, RING_COUNT, i)`, then `vertex >= ringEndOf(ring)` returns SEGMENT_NONE. A row before `ringOffsets[0]` finds no ring and a trailing row is `>= ringEnd` of the last ring. Skipped rows get inverted bounds (invalid in the BVH). Ring end is clamped by `VERTEX_COUNT`. Point-in-polygon reads by feature -> polygon -> ring offsets.
- Outside read possible: no. Examined, safe, code unchanged.
- Spec: `test/gpu-spatial-analysis/line-clip/line-clip-padding-rows.spec.ts`, extended with a third test (inside and outside). The polygon positions get 2 leading and 3 trailing rows of NaN, +-3e6, 1.25 or 0.5, and ring offsets are shifted. It also pads both sides at once. Result equals the unpadded run. 4 of 4 pass. (The line side tests of the same file were already there.)
- I also fixed the existing spec's `const device: Device = await getWebGPUTestDevice()` to narrow after `requireTestDevice` (spec typecheck error, unrelated to the padding).

### Item 2a. Delaunay tessellation and Voronoi
- GPUDelaunayTessellation inputs are a flat point list (every row is a site) and one group id per point; there is no ring or line offset array. The exact-length check `ids.length === positions.length` exists. So there are no rows that belong to no ring or line on the input side. Output rows past `count` are written by the kernel and read only under `index < count`.
- GPUVoronoiDiagram takes a triangle buffer with capacity beyond `triangleCount`. `circumcenters` reads `triangles` for every capacity row but stores center.z = 0 and skips the invalid counter for `index >= min(count, capacity)`. `edge-keys` gives rows past the count the sentinel key. `classify-edges` uses `isValidCenter`, which is 0 for those rows. So a read of a stale row cannot reach the output. Examined, safe, code unchanged.
- Spec: `test/gpu-spatial-analysis/delaunay-tessellation/delaunay-padding-rows.spec.ts`, 3 tests. Delaunay with and without groups runs on output pre-filled with stale words (`0xffffffff`, set words, small indices) and equals the clean run (count, required, overflow, triangle prefix, group counts, offsets, converged). Voronoi runs with 1 to 5 stale triangle rows after the counted prefix (valid-looking triples, index 99, all ones) and equals the clean run (written, required, overflow, invalid, segments, site ids). 3 of 3 pass.

### Item 2b. Concave hull
- Kernels: `retain` (index < count gate, and for groups `converged[groupIds[...]]` only after the gate), shared `triangle-edge-adjacency` (live = triangle < min(count, capacity) and validity != 0; others get the sentinel key), `boundary` (validity gated; writes every row, reads edges only for valid rows), grouped `triangles-publish` and `edges-publish` (gated by `count`; slots cleared first by `clear-triangles` and `clear-edges`). Outside read possible: no. Safe.
- Defect found and fixed: `GPUGroupConcaveHull` validated `groupIds.length >= positions.length`, but its Delaunay stage requires `===`. A longer `groupIds` passed construction and threw later, in `getCommandNodes`, with the Delaunay stage's ID. Now the constructor requires equality: `group-concave-hull groupIds must have exactly N rows (one per position, as GPUDelaunayTessellation requires)`. Every in-repo caller (cluster-and-outline recipe, lenses expansion example, raster-terrain-expansion spec, kernel-audit cases) passes exactly one row per position.
- Files changed: `src/gpu-spatial-analysis/concave-hull/gpu-concave-hull.ts` (one validation line).
- Spec: `test/gpu-spatial-analysis/concave-hull/concave-hull-padding-rows.spec.ts`, 3 tests. GPUConcaveHull and GPUGroupConcaveHull on pre-filled stale outputs equal the clean run (valid triangles, valid boundary edges, converged); the new rejection case checks the message. 3 of 3 pass.

### Item 3. Join BVH builders
- Owned file used: none changed. Examined `createSpatialJoinBoundsNode` and `getPolygonEdgeLayoutNodes` in `spatial-join/spatial-join-passes.ts` (not changed), and `segment-intersection/segment-bvh.ts`.
- Bounds: one invocation per feature, rows `ringOffsets[polygonOffsets[featureOffsets[f]]]` to `ringOffsets[polygonOffsets[featureOffsets[f+1]]]`, end clamped by `VERTEX_COUNT`. Lines and points are read per feature or per row. Edge layout: a bit is set only for `vertex >= ringBegin && vertex < ringEnd`, ring >= 3 rows; a row before ring 0 is below `ringBegin` and gets no bit; a trailing row fails `vertex < ringEnd`. The BVH itself comes from `@luma.gl/graph` and sees only bounds. Segment BVH: skipped segments have inverted bounds and valid = 0. Outside read possible: no. Safe.
- Spec: `test/gpu-spatial-analysis/spatial-join/spatial-join-padding-rows.spec.ts`, 5 tests. GPUPointInPolygonJoin with polygon padding (a polygon with a hole), and GPUSpatialPredicateJoin `intersects` for points x polygons, lines x polygons, polygons x lines, points x lines, with padding on one or both sides (NaN, +-3e6, 1, 5). Same pairs and status as unpadded. 5 of 5 pass.

### Item 4. Loop removal and offset-curve components
- Spec (new file): `test/gpu-spatial-analysis/geometry-refinement/line-padding-rows.spec.ts`, 2 tests. GPULineLoopRemoval (a figure-eight line plus a straight line) and GPUOffsetCurveComponents (`original` and `raw` both padded), 2 leading and 3 trailing rows of NaN, +-3e6, 2; source-edge indices rebased by the leading count; status, offsets or components, positions and source edges equal the unpadded run. 2 of 2 pass. The kernels of both live in files I do not own; none needed a change (they pass through the fixed GPULineSplit and GPULineNoding).
- Not checked: the spec was not run against the pre-fix `flag-non-finite` (no git restore allowed), so I did not show that it fails without that fix.

## 2. Verified
- `npx vitest run --project headless` on the five new specs plus `concave-hull/`, `delaunay-groups.spec.ts` and all of `line-clip/`: 9 files, 37 tests passed, 0 failed.
- Source typecheck: `node scripts/typecheck-status.mjs 'concave-hull|line-clip|delaunay'`: 0 errors.
- Spec typecheck with a scratch tsconfig (`scratchpad/tsconfig.lib-padding.json`, the five new specs): errors in my files fixed (device narrowing); the last run printed no error for them.
- `node scripts/check-upstream-first.mjs`: no violations. `npx biome check --write` on every touched file: clean.
- Not run: `yarn check-contributor-contract` (the yarn script printed yarn's usage, not a result in this shell), the node project kernel-audit specs, the lenses and kepler specs, the full suites.

## 3. Not done
- Nothing skipped. Nothing was changed in `line-clip/**` source, `delaunay-tessellation/**` source, `spatial-join/**` or `segment-intersection/**`: all examined safe.

## 4. Work list for others
- None in `website/**` or `examples/**` (the constructor now rejects only a `groupIds` view longer than the positions, which no caller in the repository passes).
- Docs: `docs/api/**` has no sentence about the old "at least" rule, so no docs edit.

## 5. Roadmap (FX-18)
- Remaining: the padding rule is checked by spec for line clip (both sides), Delaunay, Voronoi, concave hull, join bounds and edge layout, loop removal and offset components. Still open: `getSnapOrigin` (polygon-coverage-topology, another session), and the spec for `GPUSegmentIntersection` alone with padded polygon positions (only exercised through line clip and joins).

## 6. Noticed
- `GPUGroupConcaveHull` accepted an invalid `groupIds` length at construction (fixed above).
- `spatial-join-passes.ts` `createSpatialJoinBoundsNode` for `polygons` guards `polygonEnd <= POLYGON_COUNT` and `ringEnd <= RING_COUNT`, but a malformed `featureOffsets` is skipped silently with empty bounds (a feature then never matches); this is by design for malformed input, not padding.
