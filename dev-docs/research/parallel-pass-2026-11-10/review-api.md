# Wave 2 API review (read-only)

Method: `diff -u` of `pre-w2` against the working tree, path by path. I read the source of every class whose meaning changed, its internal callers (shared publish kernels, lon-lat preparation, recipes, tessellation), its kepler, lenses, website and examples consumers, and the spec diffs. Scripts and intermediate lists are in this folder (`scan.py`, `*.diff`, `removed-names.txt`, `old-name.txt`, `pre-files.txt`). I ran no build, typecheck or spec.

## Summary

I found no wrong-number or crash defect in library code, kepler, lenses, the website runs or the examples. Every place that reads or sizes from a line-output `count`, or reads overlay `requiredCount`, now uses the new meaning, and every point-cloud raster consumer is consistent with the `'south'` default. The rename of `validatePackedView` is complete and correct. The real defects are in user-facing reference text (the website catalog, TSDoc and dev-docs), plus one stray file that would ship in the npm package.

## Confirmed defects (read on both sides), most severe first

1. **The website API catalog still describes the old geometry status layouts.** The w2-geometry work list (its section 4, the `WC/...` lines) was not applied, and w2-website does not claim it.
   - `website/src/catalog/entries/geometry-construction-overlay.ts`:
     - `:1599-1600` (GPULineSegmentize), `:1766-1767` (GPULineSmooth) and `:1855-1856` (GPULineChunk) still say `GPUCardinalityPort & GPUOverflowStatusPort & {lineCount?}` and "Written vertex count and overflow flag."
     - `status.lineCount` no longer exists, and `count` now counts lines.
     - A reader who follows the catalog either passes `lineCount`, which is ignored and stays unwritten, or reads `count` as vertices. Sizing a vertex buffer or iterating vertices from `count` then gives the line total, for example 400 instead of thousands, so the result is truncated.
   - `website/src/catalog/entries/geometry-processing.ts:848` (GPURectangleClip) says `{count, overflow}` "Count and overflow counters". It does not say that `count` now counts lines or rings, and it does not mention `vertexCount`.
   - Same file, overlay and buffer entries:
     - `:85-86` (GPUBufferSurface) and `:369-370` (GPUPolygonOverlay) say the status type is `GPUBoundedResultStatusPort` with "required counts". `requiredCount` was removed; the ring requirement is now `requiredRingCount`.
     - `:365` says the result has "ring offsets and source ids". Passing `geometry.sourceIds` now throws `output.geometry.sourceIds is not written` (`gpu-polygon-overlay.ts:337`).
   - Same file, `:2387` (GPULinearReferencing) says `{overflow, candidateCount?}`. Both fields were removed; the status is now `candidateOverflow` (required) plus an optional `requiredCandidateCount`. A plain-JS caller who passes `overflow` then crashes at construction (`validateSinglePackedView` on undefined), because `candidateOverflow` is required.
   - `inventory.json` was regenerated, but these hand-written entries were not.

2. **A sed backup file ships in the package source.** `modules/analysis/src/gpu-spatial-analysis/trajectory-encounters/clock-encounters.ts-E` is untracked (`git status` shows `??`).
   - It is byte-identical to the pre-wave `clock-encounters.ts`, including the removed `trackValid` in its `Pick`. This is the backup that macOS `sed -i -E` leaves behind.
   - `modules/analysis/package.json` has `"files": ["src", ...]`, so a publish would ship it.
   - It is not compiled (wrong extension), so nothing breaks at runtime.

3. **A TSDoc example imports a name the entry no longer exports.** `modules/analysis/src/gpu-spatial-analysis/geometry-measures/geodesic-wgsl.ts:61` imports `GPU_EARTH_MEAN_RADIUS_METERS` from `'@luma-contributors/analysis/gpu-spatial-analysis'`.
   - Wave item 7 removed that name from this entry; it now comes only from `analysis-core`.
   - Copying the example gives a broken import (a TS error, or `undefined` interpolated as `${undefined}` into the WGSL).
   - I scanned every `.ts`, `.md` and `.mjs` file under docs, website, examples, lenses, kepler and analysis/src for `import {…} from '…/(gpu-spatial-analysis|gpu-network|gpu-terrain)'` that names any of the 183 moved names. This is the only hit.

4. **The k-means rename is incomplete in dev-docs.** `dev-docs/upstream-workarounds.md:86` still describes "Local class of the same name ... `gpu-spatial-analysis/spatial-clustering/gpu-kmeans.ts` (`export class GPUKMeans`)". That file was deleted; the class is now `GPUSpatialKMeans` in `gpu-spatial-kmeans.ts`.
   - `dev-docs/roadmap.md:40`, `:321`, `:507`, `:767` and `:813` still list DUP-02 and the `validatePackedView` rename as open.
   - `:507` says "`GPUKMeans` seeds internally" about the local class.
   - These are register and plan text only, with no code effect.

5. **The operator catalog's contract string contradicts the new overlay type.** `modules/analysis/src/gpu-spatial-analysis/ontology/spatial-analysis-operators.ts:653` and `:676` declare `contract: 'GPUPolygonOverlayOutput & GPUGeneratedGeometryPort'` for GPUPolygonOverlay and GPUBufferSurface.
   - `GPUGeneratedGeometryPort` (`analysis-core/contracts/ports.ts:181-187`) requires `sourceIds` and a status with `requiredCount`. The overlay now rejects `sourceIds` and has no `requiredCount`.
   - The string predates the wave, but it is now false on both fields. The `status` lists right above it were updated correctly.

## Suspected (documentation; behaviour unaffected)

6. **The new comment in `analysis-core/contracts/ports.ts:86-88` may be too broad.** It says "Geometry contributors do not read an input `sourceIds`".
   - `GPUNearestFeatureJoin` documents that it reports `features.sourceIds[row]` as the feature ID (`spatial-join/gpu-nearest-feature-join.ts:108`, `:273`).
   - The overlay still accepts and validates operand `sourceIds` (`gpu-polygon-overlay.ts:169`, `:232`).
   - Whether a join counts as a "geometry contributor" is a matter of wording, so the comment could mislead a third-party author.

## Checked and found correct

- **Line-output `count` readers.**
  - Shared kernels: `createLineOffsetsPublishNode`, `createPolygonPublishNode` and `createLineClipPublishNode` are also used by `GPULonLatPolygonPreparation`. It now passes `vertexCount: status.count` and `requiredVertexCount: status.requiredCount` with the same stage word (4), so its output is unchanged. The dense-stage `count` it dropped was write-only.
  - Kepler: `catalog-platform4-geometry-common.ts` reads `count` as lines (`producedLineCount`). In `catalog-gpu-geometry.ts`, the rectangle-clip lines read `count` as the line loop bound, and segmentize reads only `overflow`.
  - Website: `shared/lines.ts` `allocateLineOutput` is used only by the four segmentize-family runs, and its buffers were swapped consistently. In `rectangle-clip.ts`, `pieceCount` is now `vertexCount`.
  - Examples:
    - `b3-common.ts` and `geometry-mode.ts` `importPathOutput`: all 10 call sites are segmentize, chunk, smooth, arcs or rectangle clip, and the vertex buffers moved to `vertexCount` and `requiredVertexCount`.
    - `geometry-tools-mode.ts`: the polygon clip reads vertices from `vertexCount`.
    - `point-pattern-mode.ts` and `point-patterns.compute.ts` never read `count`.
  - Lenses: the five line examples and lenses only capture views.
  - Kernels: `createLineGeometryCountNode` writes `min(total, LINE_CAPACITY)` and the unclamped total. The rectangle-clip tail node maps `totals[0..3]` correctly.
- **Overlay `requiredCount` and `sourceIds`.**
  - `building-structures.ts` and `building-tessellation.ts` split the geometry status (`requiredRingCount`) from the boundary status (`requiredCount`).
  - The removed `building-identity` kernel wrote only `geometry.sourceIds`, and nothing reads `overlays[i].geometry.sourceIds`.
  - The kepler, lenses (example, lens, operator pack), website and example consumers are all moved.
  - The A/B legacy harness keeps the old fields on purpose.
- **Linear referencing.**
  - Planar: the join's `candidateOverflow` folds in the BVH overflow (`gpu-nearest-feature-join.ts:1027-1028`), so routing the join's `overflow` to a transient drops no cause.
  - All consumers are updated, including `trajectory-twin-heading-signed-zero.spec.ts:165`.
- **Shared paths.** Only local variable names keep `leftLineIds`.
- **Point-cloud row direction.**
  - Every raster path goes through `getPointCloudRowDirection` (cell source, SMRF index, surfaces index).
  - `createPointCloudGridIndex` still defaults to `'north'`, but its only raster-mode callers pass the direction; the ICP, change and neighborhoods callers are non-raster.
  - In the examples (`lidar-ground`, `point-cloud-support`), every contributor shares one grid and no raster is read back or drawn.
  - The website removed the display flip consistently.
  - No lens or kepler consumer exists.
- **`validatePackedView` rename: all 409 files checked by script, not by sampling.**
  - 380 files imported the local function from `gpu-contributor-utils`, and each now has exactly as many `validateSinglePackedView` occurrences as it had `validatePackedView` ones.
  - No file imported both functions, no file re-exported either one, and no file imported it from anywhere else.
  - The 28 files that import from `@luma.gl/graph/gpu-core` are unchanged on this identifier, and none of them imported the local function before the wave. The definition imports the published function as `validatePackedGraphView`.
- **Barrels.**
  - My own parse of the pre-wave `contracts/index.ts` gives the same removed set as the author's list, plus the expected differences: `GPUStudyWindowSlabViews`, and the three new ports.
  - The 11 removed internal symbols are imported only by specs, through source paths.
  - No new deep import into lenses, kepler, website or examples (I compared before and after). `package.json` `exports` is unchanged and includes `./analysis-core`.
- **K-means.** Both renamed specs equal the old ones modulo the rename and formatting. `GPURasterEmbeddingKMeans` is untouched. No old symbol survives in code.
- **Specs.**
  - The deleted assertions are all alias, legacy-path or rename tests, and their replacements cover the new behaviour.
  - No `it.skip`, and no tolerance loosened: walking cost is still 4 ulp, and point-cloud `toBeCloseTo` precisions are unchanged.
  - The new numbers come from independent oracles: rectangle clip `39` from `clipLines`, and point-cloud bilinear and IDW from JS oracles.

## Out of the seven items, noted

- `modules/analysis/package.json` peer dependencies moved from `~10.0.0-alpha.10` to `~10.0.0-alpha.12` (the bump stream). `AGENTS.md` still says alpha.10.
- The overlay `combine-positions` and make-valid padding-row changes are behaviour changes (w2-ring-assembly). They are documented there and pinned by `polygon-overlay-padding-rows.spec.ts`.

## Risks I could not check without running code

- **Walking-cost fixture labels.** Before the wave, the spec asserted `parameters == legacy` bit for bit, and the earlier note said legacy was 1 ulp off in 8 Tobler cells. The new note says the parameters path is exact for Tobler and 1 ulp off for GRASS, in the same 8 cells. Both cannot be true unless the parameters path changed, or the earlier labels were swapped. This is within the 4-ulp bound either way.
- **Parity specs.** The `*.parity.spec.ts` files were edited (`validity:` to `mask:`, and line status) without being run.
- **Ground-threshold spec layout.** `gpu-pointcloud-ground-threshold.spec.ts` lays out `minimumSurface` north-up against the now-south grid. It passes only because points carry their own z.
- **Stale lens records and twins** (acknowledged by the authors; `lens-twins` is red):
  - `rectangle-clip~result` and `linear-referencing~result`;
  - the `gpu-raster.raster-walking-cost` and `raster-fractional-coverage` records;
  - the `spatial-k-means` records, which still contain `GPUKMeans`, and the website `lens/generated` copies.
- **Unverified pictures.** No browser check was run for the showcase and explorer scenes whose output changed (line helpers, overlay status, point-cloud).
