# Stream w2-geometry: STD-08 breaking flips (wave 2, step A)

## 1. Done

**Item 1. Segmentize family and rectangle clip: `count` / `requiredCount` count lines.**

- `GPULineGeometryOutput.status` is now `GPUCardinalityPort & GPUOverflowStatusPort & GPUApproximationStatusPort & GPUVertexCountStatusPort`:
  - `count` is the lines written, clamped to the line capacity. For rectangle-clip polygons it is the rings.
  - `requiredCount` is the unclamped line count.
  - `vertexCount` / `requiredVertexCount` hold the vertex totals.
  - `status.lineCount` is removed. It was now the same as `count`, and the contract bans synonyms.
  - A line truncated by the vertex capacity still counts as written. This is documented.
- Kernels (`line-segmentize/{line-segmentize-kernels,line-chunk-kernels,line-smooth-kernels,line-geometry-output}.ts`, `rectangle-clip/rectangle-clip-kernels.ts`):
  - The publish kernels write `overflow` plus the optional `vertexCount` / `requiredVertexCount` directly.
  - The new `createLineGeometryCountNode` writes `count` / `requiredCount`. These are constants, except for chunk, which reads `pieceTotal`.
  - Rectangle-clip lines publish four totals; the tail node writes all four scalars.
  - The wave 1 alias node and its transient (`createLineGeometryStatusAliasNodes`, `getLineGeometryRequiredCount`) are deleted.
  - Every kernel binds at most 8 storage buffers.
- Chunk `requiredCount` is exact, except when a single line needs more than `lineCapacity + 1` pieces. It is then a lower bound that still exceeds the capacity. This is documented and a spec pins it.
- `lon-lat-polygons/gpu-lon-lat-polygon-preparation.ts`: two call sites of the shared kernels were renamed (`requiredVertexCount`, `vertexCount`). Its behaviour is unchanged.
- Consumers moved:
  - kepler: `catalog-platform4-geometry-common.ts`, `catalog-gpu-geometry.ts` (2 rectangle clips).
  - lenses: the 5 example files and 5 lens files. `lineCount` captures became `vertexCount`.
  - tests: `line-test-utils.ts` (option `minimalStatus` replaces `vertexStatus`), the segmentize, chunk and rectangle-clip specs, 3 parity specs, `line-segmentize.node.spec.ts` (new last node `-line-count`), and the kernel-audit `lineGeometryOutput`.
  - docs: `operators-lines.md`, `operators-geometry.md`.
- New specs:
  - lines versus vertices at an undersized capacity (with and without optional views);
  - arcs `count = 3` while `vertexCount` is clamped;
  - chunk `requiredCount = pieceTotal` and the lower-bound case;
  - rectangle clip `requiredCount = 39` for the zigzag (from the f64 oracle), and polygons `count` = rings.

**Item 2. Overlay and buffer.**

- Decision: `status.requiredCount` is **removed**, not kept with a feature meaning.
  - Features are assembled from rings, so once rings are dropped the feature total is unknown.
  - When nothing is dropped, it equals `count`. Rule 8 says to omit it when it cannot be computed exactly.
  - The type is `Omit<GPUBoundedResultStatusPort, 'requiredCount'> & GPURingCountStatusPort & {...}`. Size `capacity.rings` from `requiredRingCount`.
- `geometry.sourceIds` is removed. The type is `Omit<GPUPackedGeometryPort<GPUPolygonGeometryPort>, 'sourceIds'>`, and the `source-ids` kernel is gone.
  - A view passed there now throws `output.geometry.sourceIds is not written: an overlay polygon has no single source row` (`validateRequirement`). Specs cover this.
- Workaround kept, see Noticed 1: the ring assembly still gets a transient `sourceIds`.
- Consumers moved:
  - `global-coordinates/gpu-spherical-polygon-overlay.ts`.
  - `urban-morphology/building-tessellation.ts`: the `building-identity` kernel is removed (it copied the building index; the index in `outputs` is the building row). The geometry status now carries `requiredRingCount`.
  - `urban-morphology/building-structures.ts`.
  - kepler `catalog-gpu-geometry.ts`.
  - lenses: polygon-overlay and buffer-surface example and lens files, `global-coordinates-examples.ts`, the operator pack `geometry.ts` (drops the `geometry.sourceIds` variable), `urban-ecology-operators.ts` text.
- Test fixtures: 13 files, plus the A/B harness, which keeps the legacy overlay on the old fields.
- Catalog: the overlay and buffer `status` lists, and five new `GPUSpatialAnalysisStatusKind` words.
- Docs: `operators-geometry.md`, `urban-morphology.md`.

**Item 3. Linear referencing.**

- Status is `{candidateOverflow (required), requiredCandidateCount?}`.
  - `overflow` is removed: both capacities are intermediate, and every query still gets its row.
  - `candidateCount` and the `requiredCount` misuse are removed.
  - `requiredCandidateCount` matches `\w*[rR]equired\w*Count` in the vocabulary spec. It is useful for sizing `candidateCapacity`, and the spec does that.
- Planar: the join writes `candidateOverflow` and `candidateCount` straight into the caller's views. The join's own `overflow` goes to a transient.
- Great circle: the segment-table and node-data kernels write `candidateOverflow` and zero `requiredCandidateCount`.
- Both copy nodes are deleted.
- Consumers moved: lenses (`linear-referencing.{example,lens}.ts`, `line-locate.lens.ts`), 7 kernel-audit cases, 6 spec files, `operators-lines.md`. The kepler binding is CPU-only, so nothing changed there.

**Item 4. Shared paths.** `leftLineIds` / `rightLineIds` are renamed to `leftSourceIds` / `rightSourceIds`, with no alias.

- Source: `gpu-shared-paths.ts`, `urban-morphology/building-adjacency.ts`.
- Lenses: `shared-paths.{example,lens}.ts`, `urban-ecology-examples.ts`, operator pack `lines.ts`.
- Tests: 2 kernel-audit files, the line-clip harness and spec, 4 urban-ecology specs.
- Docs: `operators-lines.md`.

**Item 5. Layout type in core.**

- `analysis-core/contracts/status.ts` gains `GPUVertexCountStatusPort`, `GPURingCountStatusPort` and `GPUGeneratedGeometryStatusPort`. They are exported from `contracts/index.ts`.
- `GPUGeneratedGeometryPort.status` is now `GPUGeneratedGeometryStatusPort`.
- Imported from core by: `GPULineGeometryOutput`, `GPULineSplitPieces`, `GPULineMergeOutput`, `GPUSharedPathsOutput`, `GPUPolygonOverlayOutput`.
- `ports.ts` comment now reads: "Geometry contributors do not read an input `sourceIds`: the `sourceIds` of generated geometry hold input feature rows (contributor contract rule 8), and a caller with its own feature IDs looks them up by that row."
- `GPUGeneratedGeometryPort` has a doc comment stating the rule.
- The `validateLineGeometryOutput` validator stays in `line-segmentize/`. It is specific to `GPULineGeometryOutput`, not the generic port.

**Item 6.** The catalog, kernel-audit and docs changes are listed under each item above.

## 2. Verified

- `node scripts/typecheck-status.mjs`: 0 errors (0 total), last cycle at 12:21:21 AM. This covers sources, lenses and kepler.
- Node project:
  - item 1: 8 files, 82 tests passed;
  - item 2: 13 files, 149 passed;
  - `overlay-face-labelling-executed` with overlay and buffer node specs, rerun after the workaround: 3 files, 82 passed;
  - item 3: 6 files, 51 passed;
  - item 4: 5 files, 152 passed;
  - vocabulary, ontology, geometry-output-contract, line-merge, line-split: 7 files, 39 passed;
  - doc-parameters and contracts: 3 files, 32 passed;
  - final batch: 13 files, 314 passed, 1 failed (`s3-geometry-pack` GPUShapeGenerator `latitude2`, not mine, see Noticed).
- Headless project:
  - item 1: 12 files, 279 passed;
  - item 2: 6 files, 281 passed, after the workaround (before it, 2 tessellation tests failed);
  - item 3: 3 files, 13 passed;
  - item 4: 5 files, 73 passed;
  - line-merge and line-split: 2 files, 15 passed;
  - kepler `catalog-gpu` and `catalog-platform4-geometry-gpu` with 3 urban-ecology specs: 5 files, 95 passed.
- `npx biome check --write` on all touched files: clean.
- `check-upstream-first`: no violations.
- `check-contributor-contract`: no new findings, exit 0.
- Lens record specs:
  - `lens-twins` reports stale twins for `linear-referencing~result` and `rectangle-clip~result`, because capture names changed. Regenerate in LN-REC2.
  - `record-structure` fails on 4 gpu-network records, which are not mine.
- Not run: `yarn typecheck-tests` (forbidden; test files are not in the watcher), the full suites, website and examples.

## 3. Not done

- Item 2: a per-feature source row for overlay and buffer is out of scope, as assigned.
- Wave 1 requests 5 and 6 (stop folding pair overflow into `overflow` for shared paths and line clip) and 7 (hull bit sets) were not in this assignment, so they are untouched.
- Lens records, twins and website view models are not regenerated (LN-REC2).

## 4. Website and examples work list

Paths: `WR` = `website/src/gpu/runs/entries/`, `WC` = `website/src/catalog/entries/`, `SC` = `examples/deck/spatial-analysis-showcase/src/scenes/`, `EX` = `examples/deck/spatial-analysis-explorer/modes/`.

**Line output: `count` is lines, `lineCount` is removed.** The buffers named `count` / `requiredCount` in these helpers keep holding vertices.

- `WR/geometry-construction-overlay/shared/lines.ts:143`:
  `status: {count: lineCount.view, overflow: overflow.view, vertexCount: count.view},`
  This covers the segmentize, arcs, smooth and chunk runs; nothing else in them changes.
- `WR/geometry-processing/rectangle-clip.ts:49`: `count: pieceCount.view,` becomes `count: builder.allocate('line-count', 'uint32', 1).view, vertexCount: pieceCount.view,`. The readout at line 125 still says vertices.
- `WR/geometry-processing/rectangle-clip.ts:77`: no change needed (unused; it now counts rings).
- `WC/geometry-construction-overlay.ts:1613`, `:1786`, `:1878`: `type: 'GPUCardinalityPort & GPUOverflowStatusPort & GPUVertexCountStatusPort'`.
- `WC/geometry-construction-overlay.ts:1614`, `:1787`, `:1879`: `description: 'Line count (count), vertex counts and overflow flag.'`.
- `WC/geometry-processing.ts:848`: `makeOutput('status', '{count, overflow, vertexCount?}', 'Line (or ring) count, vertex count and overflow.')`.
- `EX/geometry-mode.ts:344-351`, and the same block in `SC/geometry/b3-common.ts:434-441`: replace the status object with
  `{count: importGraphBuffer(graph, 'out-path-count', output.pathCount, 'uint32', 1), overflow: importGraphBuffer(graph, 'out-overflow', output.overflow, 'uint32', 1), vertexCount: importGraphBuffer(graph, 'out-count', output.count, 'uint32', 1), requiredVertexCount: importGraphBuffer(graph, 'out-total', output.requiredCount, 'uint32', 1)}`.
  - The `lineCount` option becomes unused. Its callers (`EX/geometry-mode.ts:1185`, `SC/geometry/b3-line-views-city.ts:295`, `SC/geometry/buffers-and-shapes.compute.ts:1447`) may drop `lineCount: ...`.
  - These helpers serve every other line-output site (airline-network, b3-line-views-city and far, buffers-and-shapes, geometry-mode).
- `EX/geometry-tools-mode.ts:491` (polygon rectangle clip; `clipCount` is read as vertices at :504 and :753): `count: countView,` becomes `count: importGraphBuffer(clipGraph, 'clip-ring-count', resources.createBuffer('clip-ring-count', 4), 'uint32', 1), vertexCount: countView,`.
- No change needed (unused `count`): `EX/point-pattern-mode.ts:510`, `SC/points/point-patterns.compute.ts:577`.
- Code snippets:
  - `SC/geometry/line-operations.scene.ts:626`: `status: {count, overflow, vertexCount}`.
  - `SC/geometry/line-operations.scene.ts:640`: `status: {count, overflow, vertexCount}` (replaces `lineCount`).
  - `SC/geometry/buffers-and-shapes.scene.ts:652`: `', requiredCount, lineCount'` becomes `', requiredCount, vertexCount'`.

**Overlay and buffer: no `status.requiredCount`, no `geometry.sourceIds`.**

- `WR/geometry-construction-overlay/shared/polygons.ts`:
  - delete line 228 and line 243, and drop the comma at the end of line 242;
  - line 247: `requiredRingCount: scalar('required-count').view,`.
- `WR/cells-index-join/spherical-polygon-overlay.ts`: delete line 94 and drop the comma at line 93; line 99: `requiredRingCount: status[0].view,`.
- `SC/earth/spherical-boundaries.compute.ts`: delete line 192 and drop the comma at line 191; line 197: `requiredRingCount: view('required', required, 'uint32', 1),`. The `sourceIds` buffer at lines 68, 130 and 158 may go.
- `SC/geometry/repair-overlay.compute.ts`: delete line 524 and drop the comma at line 523; line 528: `requiredRingCount: importView(`. `overlaySourceIds` (line 479) may go.
- `SC/geometry/block-relationships.compute.ts`:
  - delete line 1248 and drop the comma at line 1247;
  - line 1252: `requiredRingCount: importView(geometryStatus.requiredCount, 'uint32', 1),`;
  - line 2070: `['required', output.status.requiredRingCount, target.requiredCount],` (the tessellation outputs now carry `requiredRingCount`).
- `SC/geometry/footprint-shape-lab.compute.ts`: delete line 1245 and drop the comma at line 1244; line 1250: `requiredRingCount: output('buffer-required-count', 'uint32', 1, 4),`. The read at line 921 is already named `requiredRingCount`.
- `WC/geometry-construction-overlay.ts`:
  - `:85` and `:378`: `type: "Omit<GPUBoundedResultStatusPort, 'requiredCount'> & GPURingCountStatusPort"`;
  - `:86`: `'Count, overflow and the required ring count for capacity checks.'`;
  - `:379`: `'Count, overflow and the required ring count.'`;
  - `:374`: `'Result polygons with feature, polygon and ring offsets (no source rows).'`.

**Linear referencing: `status: {candidateOverflow, requiredCandidateCount?}`.**

- `WR/geometry-construction-overlay/linear-referencing.ts:48`: `status: {candidateOverflow: overflow.view}`.
- `EX/geometry-mode.ts:1391-1392`: `candidateOverflow: importGraphBuffer(graph, 'overflow', overflow, 'uint32', 1), requiredCandidateCount: importGraphBuffer(graph, 'candidate-count', candidateCount, 'uint32', 1)`.
- `SC/geometry/b3-line-views-city.ts:616-617`: `candidateOverflow: imp('overflow', overflow, 'uint32', 1), requiredCandidateCount: imp('candidate-count', candidateCount, 'uint32', 1)`.
- `SC/geometry/line-operations.scene.ts:671`: `status: {candidateOverflow, requiredCandidateCount}`.
- `WC/geometry-construction-overlay.ts:2416`: `type: '{candidateOverflow, requiredCandidateCount?}',`.

**Shared paths: `leftSourceIds` / `rightSourceIds`.** Only the property keys must change.

- `SC/geometry/block-relationships.compute.ts:1110-1111` and `:1388-1389`.
- `SC/geometry/county-linework.compute.ts:576` and `:582`.
- `WR/geometry-construction-overlay/building-adjacency-recipe.ts:77-78`.
- `WR/geometry-construction-overlay/shared-paths.ts:63-64`.

In each, `leftLineIds:` becomes `leftSourceIds:` and `rightLineIds:` becomes `rightSourceIds:`. The local variable names may stay.

**Generated:** `website/src/lens/generated/gpu-spatial-analysis.shared-paths.json`, the website lens view models, and `docs/lens-records/**` regenerate in LN-REC2.

## 5. Roadmap

- **STD-08**: replace "Remaining" with:

  > Done 2026-11-10 (wave 2): line output `count` counts lines (`lineCount` removed); overlay and buffer drop `requiredCount` (rings are `requiredRingCount`) and `geometry.sourceIds`; linear referencing has `candidateOverflow` and `requiredCandidateCount` and no `overflow`; shared paths `leftSourceIds` / `rightSourceIds`; `GPUVertexCountStatusPort`, `GPURingCountStatusPort` and `GPUGeneratedGeometryStatusPort` in `analysis-core/contracts`; `ports.ts` states the row rule. Remaining: offset curve and outline leave the fixed-stride layout (G7); per-feature source rows for overlay and buffer (L); the transient-reuse defect behind the overlay ring-assembly workaround.

  Delete the "To decide" sentence.
- **STD-04**: add to Done: "linear referencing reports only `candidateOverflow` (2026-11-10)". The remaining list keeps: shared paths and line clip still fold pair overflow into `overflow`.
- **11.6**: delete "the STD-08 flips (segmentize `count` to lines, overlay `requiredCount`, shared-paths names)" from the "Source, breaking" step.

## 6. Noticed

1. **Probable stale read of a reused transient.**
   - Removing `geometry.sourceIds` from the overlay means `GPUSegmentRingAssembly` runs without `polygons.sourceIds`, so its `polygon-groups` node is gone.
   - That alone makes `addBuildingTessellationRecipe` (simple enclosure: a clip overlay after a dissolve overlay) report `complete [1, 0]` instead of `[1, 1]` (`urban-ecology-advanced.spec.ts:233` and `:359`).
   - Extending the overlay status lifetime did not help. Giving the assembly a throwaway transient `sourceIds` restores the result exactly.
   - The `polygon-groups` node only writes that transient, so the dependence is on transient allocation in the graph's lifetime-based buffer reuse. Some kernel in the assembly or the overlay likely reads a transient row it did not write in this frame.
   - Kept the transient in `gpu-polygon-overlay.ts`, with a comment; it needs a dedicated hunt. Other ring-assembly callers without `sourceIds` may be affected.
2. `s3-geometry-pack.node.spec.ts` GPUShapeGenerator: "latitude-out -> wgsl:latitude2" no longer matches its kernel source (grid-generators changed today).
3. `record-structure.node.spec.ts` fails for 4 gpu-network records (node-count drift).
4. `GPULonLatPolygonPreparation.status.count` and `requiredCount` still count vertices, against rule 8. That class is not mine.
5. `GPUNearestFeatureJoin.status.candidateCount` is the synonym rule 9 bans. The joins stream could rename it to `requiredCandidateCount` as linear referencing now does.
6. Contract rule 8 ("geometry output carries the source row of each output feature") has no stated exception for union outputs (overlay, buffer). The wave 1 decision (provenance on boundary edges) should be added to `contributor-contract.md`.
