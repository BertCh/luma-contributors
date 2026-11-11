# Stream geometry-output: STD-08, STD-04 (line family), STD-03

Wave 1, additive only. Every new field is optional and defaults to today's behaviour. No export,
file or class was removed or renamed, and no existing output changed meaning.

## 0. Inventory (item 1)

Classes in my directories that have variable-length output, or output sized by data. "count" is
`status.count`. "src" is feature-level provenance. Line numbers have moved, so names are given
instead.

| Class | Layout | Offsets | `count` counts | `requiredCount` | Overflow reporting | Source rows |
| --- | --- | --- | --- | --- | --- | --- |
| `GPULineSegmentize`, `GPUGreatCircleArcs`, `GPULineSmooth`, `GPULineChunk` | `GPULineGeometryOutput` (`line-segmentize-types.ts`) | `lineOffsets`, lineCapacity+1, clamped to vertex capacity | **vertices** (`lineCount` counts lines) | vertices | `overflow` 0/1 (vertex or line capacity); `approximation` (segmentize, arcs) | `sourceLineIndices` per line (identity except chunk); per vertex `sourceRows`, `measures` |
| `GPURectangleClip` | `GPULineGeometryOutput`; for polygons `lineOffsets` are ring offsets and the input feature/polygon offsets are reused | as above | **vertices** | vertices | `overflow` 0/1 | `sourceLineIndices` (lines); polygons: implicit, ring index kept |
| `GPULineSplit` | `GPULineSplitPieces` = `GPUGeneratedGeometryPort<lines>` | `geometry.lineOffsets` | pieces (features) | pieces | `overflow`, `candidateOverflow` (pair list), `uncertainCount` | `sourceIds` = input line row; per vertex `vertexSourceEdges` |
| `GPULineClipByPolygon` | `GPULineSplitPieces` | as split | pieces | pieces | `overflow` (all causes), `candidateOverflow` (pair list), `uncertainCount` | `sourceIds` = line row. Polygons act as a union mask, so pieces have no polygon row |
| `GPUSharedPaths` | `GPUSharedPathsRuns` | `offsets` | runs (features) | runs | `overflow` includes pair overflow; **new** `candidateOverflow` | `leftLineIds`, `rightLineIds`, `forward` (two inputs, both carried) |
| `GPULineMerge` | `GPULineMergeOutput` | `chainOffsets` | chains (features) | was absent, **new** | was none (capacity always suffices), **new** `overflow` = 0 | inverse map `lineChains`/`lineOrders`/`lineReversed`; **new** `sourceIds` per chain |
| `GPULineSimplification` | row selection `GPUCompactOutput` + `lineCounts`/`lineStarts` | `lineStarts` into `ids` | kept vertex rows (the `ids` rows, which is the `GPUCompactOutput` meaning) | kept rows | `overflow` 0/1; `converged`, `iterationCount` | `ids` are input rows; lines are one-to-one |
| `GPUOffsetCurve` | fixed stride, `k` rows per input vertex, padded | implicit `i*k` | none | none | none (fixed size) | implicit (row / k) |
| `GPUOutlineGeometry` | fixed stride, render-only | implicit | none | none | none | implicit |
| `GPUGeometryCleanup` | compacted positions; republished line or ring offsets, index preserving | `lineOffsets` / `ringOffsets` | vertices | vertices | `overflow`; `collapsedCount` | identity (line/ring index kept) |
| `GPUCoverageSimplification` | positions + `ringOffsets` (ring count unchanged) | `ringOffsets` | none (`requiredCount` = vertices) | vertices | `overflow`; `topologyStatistics[4]` word 3 = candidate pair overflow (ad hoc) | identity (rings kept) |
| `GPUPolygonOverlay`, `GPUBufferSurface` | polygon port, four offset levels sharing one ring capacity | `featureOffsets`/`polygonOffsets`/`ringOffsets` | polygons (features) | **rings** | `overflow`, `candidateOverflow`, `uncertainCount`, `nodedSegmentRequiredCount`; boundary has its own status | `geometry.sourceIds` = **output feature index**; provenance only on `boundary.operandIds`/`sourceFeatureIds` |
| `GPUPolygonize`, `GPUMakeValid`, precision reduction | noded segments + directed edges + polygon port | polygon offsets | segments / edges per sub-port | per sub-port | `GPUTopologyStatusPort` (`overflow`, `candidateOverflow`, `uncertainCount`) | `polygons.sourceIds` = source group of the shell; noded segments carry feature, ring, edge and parameters |
| `GPULineNoding`, `GPUSnapRoundedNoding` | `GPUNodedSegmentPort` | none (one row per segment) | segments | segments | topology status | full per-segment provenance |
| `GPUCoverageDissolve` | ring-assembly output (+ optional polygons) | `ringOffsets` | rings | rings | `overflow` | polygons `sourceIds` = label |
| `GPUVoronoiDiagram` | segment list | none | segments | segments | bounded status | `siteIds` (two sites per edge) |
| `GPUConcaveHull`, `GPUGroupConcaveHull` | fixed Delaunay reservation with validity masks | slot per group | none | none | `converged` only | edges reference source positions |
| `GPUGroupConvexHull` | CSR per group | `offsets` + `counts` | none | none (`sizes`) | `overflow` **bit set** 1 or 2 | `vertexIndices` per vertex |
| `GPUMinimumBounds` | one row per group | none | none | none | `overflow` **bit set** | group row |
| `GPUOffsetExpansion` | compact owners | input offsets | child rows | child rows | `overflow` 0/1 | `owners` |
| Refinement classes (`GPUSubdividePolygons`, `GPUOffsetCurveComponents`, `GPULineLoopRemoval`, `GPUSegmentSnap`, skeletons) | ad hoc | ad hoc | ad hoc 4- or 8-word status arrays | none | ad hoc | varies. I did not re-read these, see review C4 |

Summary: there are seven line layouts, and `count` means vertices, pieces, runs, chains or kept
rows depending on the class. Polygon `requiredCount` counts rings. Overlay and buffer have no
feature provenance.

## 0b. Decision: the target layout (item 2)

1. One layout, `GPUGeneratedGeometryPort<Port>`, with `GPULineSplitPieces` as the reference
   implementation. `geometry` holds `positions` plus the offsets of its kind:
2. lines: `lineOffsets` (features to vertices). Polygons: `featureOffsets` to `polygonOffsets` to
   `ringOffsets` to vertices (GeoArrow, rings implicitly closed). Every offsets view has capacity+1
   rows, is GPU-written, and is padded flat past the count, so a truncated result is still a valid
   layout.
3. `status.count` = features written (clamped). `status.requiredCount` = unclamped features, and is
   omitted when it cannot be computed exactly. `status.overflow` is 0/1: some final capacity dropped
   rows.
4. Sub-levels use named fields: `vertexCount` and `requiredVertexCount`; polygons also have
   `ringCount` and `requiredRingCount`. Intermediate capacities use `candidateOverflow`; deliberate
   caps use `approximation`; uncertain predicates use `uncertainCount`.
5. Provenance from one input: `sourceIds`, aligned with features, holding the input feature row,
   with `0xffffffff` past `count`. From two inputs taken per pair: `leftSourceIds` and
   `rightSourceIds`. Shared paths' `leftLineIds`/`rightLineIds` are the same thing under older names.
6. Many-to-one outputs (merge, dissolve): `sourceIds` is the first (leader) input row, and the full
   membership is a per-input inverse map (`lineChains`, `lineOrders`).
7. Outputs over operand unions (polygon overlay, buffer surface, the polygon mask of line clip) are
   many-to-many, so they have no feature provenance. Their boundary carries it
   (`boundary.operandIds`, `sourceFeatureIds`). Feature pairs need a per-pair overlay, or a
   `GPULabelPoint` point joined to each operand with `GPUPointInPolygonJoin`.
8. Optional per-vertex columns: `sourceRows` / `vertexSourceEdges` (input segment) and `measures`.
9. Selections of input rows (simplification) stay `GPUCompactOutput`. Their `ids` are the
   provenance, so they are outside this layout.
10. Values are input rows. The port doc (`ports.ts`, "sourceIds ... replace the feature rows") says
    input `sourceIds` should be mapped through, but no geometry contributor does that. Wave 2
    decides.

## 1. Done

**STD-08: layout and provenance, additive (item 3)**

- `GPULineGeometryOutput` (segmentize, great-circle arcs, smooth, chunk, rectangle clip):
  - New optional `status.vertexCount` and `status.requiredVertexCount`, written by one
    post-publish node (`createLineGeometryStatusAliasNodes`).
  - When only `requiredVertexCount` is requested, the contributor sources the unclamped count from
    a transient (`getLineGeometryRequiredCount`).
  - `count`/`requiredCount` keep their vertex meaning and are documented as changing to lines in a
    breaking release.
  - Files: `line-segmentize/{line-segmentize-types,line-geometry-output,gpu-line-segmentize,gpu-great-circle-arcs,gpu-line-smooth,gpu-line-chunk}.ts`
    and `rectangle-clip/gpu-rectangle-clip.ts`.
- `GPULineMerge`:
  - New optional `sourceIds`: the chain's first line, `GPU_LINE_MERGE_NONE` past `count`.
  - New optional `status.requiredCount` (= count), `status.overflow` (= 0), `status.vertexCount`
    and `status.requiredVertexCount`.
  - All written by the existing offsets kernel. File: `line-merge/gpu-line-merge.ts`.
- `GPUPolygonOverlay` (and so `GPUBufferSurface`):
  - New optional `status.ringCount` and `status.requiredRingCount`.
  - `requiredCount` (rings) is documented as not the rule 8 meaning.
  - `geometry.sourceIds` (output index) is marked `@deprecated`.
  - The type, class and buffer docs now state the provenance rule.
  - Files: `polygon-overlay/gpu-polygon-overlay.ts`, `polygon-buffer/gpu-buffer-surface.ts`.
- Already on the target layout: line split, line clip, shared paths. Nothing was needed beyond
  item 4.

**STD-04 (item 4)**

- `GPUSharedPaths`: new optional `status.candidateOverflow` for the intersection pair list. Pair
  overflow still also sets `overflow`. File: `line-clip/gpu-shared-paths.ts`.
- `GPULinearReferencing`: new optional `status.candidateOverflow`, a copy of `overflow`. Both of its
  capacities, BVH leaf and candidate, are intermediate. File:
  `linear-referencing/gpu-linear-referencing.ts`.
- `approximation` on segmentize and arcs already existed. The other line contributors have no
  deliberate approximation.

**STD-03 (item 5)**

All 18 bare `assert` calls in my directories are replaced by `validateRequirement`,
`validateViewLength`, `validateCapacity`, `validateFiniteNumber` and `validatePowerOfTwo`, or by an
id-prefixed `throw` where TypeScript needs narrowing:

- `concave-hull/gpu-concave-hull.ts` (6, now through a shared `validateConcaveHullOutput`)
- `geometry-edit/gpu-geometry-cleanup.ts` (2)
- `geometry-measures/gpu-geometry-measures.ts` (3)
- `line-segmentize/line-segmentize-kernels.ts` (4)
- `polygon-triangulation/gpu-constrained-delaunay-triangulation.ts` (3)

**Docs**

- `docs/api/gpu-spatial-analysis/operators-lines.md`: line output status, linear referencing,
  line merge, shared paths.
- `docs/api/gpu-spatial-analysis/operators-geometry.md`: overlay and buffer status, and the
  provenance rule.

**Specs**

- New `modules/analysis/test/gpu-spatial-analysis/geometry-output-contract.node.spec.ts`: STD-03
  rejection messages for CDT, measures, cleanup and the line kernel builders.
- `concave-hull.spec.ts`: the existing `toThrow()` now checks the message; added rejections for
  output length, `groupCount` and the converged rows.
- `gpu-line-segmentize.spec.ts` + `line-test-utils.ts` (option `vertexStatus`): at an undersized
  capacity, `requiredVertexCount` matches the f64 oracle and `vertexCount` equals the capacity. Run
  with and without `requiredCount`.
- `gpu-line-merge.spec.ts`: `sourceIds` checked against a CPU reference (the input line that starts
  each chain, forward or reversed) on the fixture and on the fixture reversed. Also checks the
  status scalars and NONE padding.
- `polygon-overlay-snap-rounding.spec.ts` + `overlay-gpu-harness.ts`: `ringCount` is 1 and
  `requiredRingCount` is 3 at ring capacity 1, and both are 3 when sized.
- `gpu-linear-referencing.spec.ts`: `candidateOverflow` is 1 at an undersized candidate capacity
  and 0 when sized from `requiredCount`.
- `gpu-shared-paths.spec.ts` + `line-clip-harness.ts`: `candidateOverflow` is 1 only for pair
  overflow, not for run or vertex overflow.

## 2. Verified

- `node scripts/typecheck-status.mjs` on my paths: 0 matching errors (0 total) in the 18:37 and
  18:49 cycles. This covers sources only; test files are not in
  the watcher.
- `npx vitest run --project node` on `geometry-output-contract.node.spec.ts`,
  `line-merge/gpu-line-merge.node.spec.ts`, `line-segmentize/line-segmentize.node.spec.ts` and
  `linear-referencing/linear-referencing.node.spec.ts`: 4 files, 22 tests passed.
- `npx vitest run --project node` on `polygon-buffer.node`, `gpu-geometry-measures.node` and
  `geometry-cleanup-repeat.node`: 3 files, 28 tests passed.
- NOT completed: `overlay-face-labelling-executed.node.spec.ts` and
  `raster-terrain-expansion.node.spec.ts`. This contains the old CDT/concave-hull `toThrow()`
  checks, which still throw by construction. The batch ran over 20 minutes under load (load
  average about 25) and was stopped.
- Headless (`gpu-line-segmentize`, `gpu-line-merge`, `gpu-linear-referencing`, `concave-hull`):
  4 files, 30 tests passed.
- Headless (`polygon-overlay-snap-rounding`, `gpu-shared-paths`):
  - First run: 181 of 186 failed. My bug: the overlay publish kernel with the two ring views bound
    11 storage buffers, one over the device limit of 10.
  - Fixed by moving them to a separate `ring-status` node.
  - Rerun: 2 files, 186 tests passed.
- Headless `polygon-buffer.spec.ts` (after the fix): 17 passed.
- `npx biome check --write` on all 27 touched files: clean (2 reformatted).
- `yarn check-contributor-contract --all`: no bare-assert findings remain in my directories (15
  total elsewhere). The plain `yarn check-contributor-contract` exits 0.
  `node scripts/check-upstream-first.mjs`: no violations.
- Not run: `yarn typecheck-tests` (forbidden), the kernel-audit spec, the full suites, and the
  specs of other classes that share `line-test-utils.ts`. The fixture change there is opt-in, so
  their behaviour is unchanged. The disk was at 100% (about 105 MiB free) during the session.

## 3. Not done

- **Feature source rows for `GPUPolygonOverlay` and `GPUBufferSurface`.** The result is a Boolean
  combination of operand unions, so an output polygon can lie over several input features of each
  operand, and the union of adjacent features is one polygon. A source row per output feature is
  not defined. Making it defined needs group labels carried through snap noding, the winding
  labelling and ring assembly (review Q5/Q6, effort L), and that changes results. Left untouched,
  with the rule and a join-based workaround documented instead.
- **`GPUOffsetCurve` and `GPUOutlineGeometry` fixed-stride output.** Moving them to a compact
  `GPULineGeometryOutput` needs a count/scan/emit pipeline (review G7). That is a rework, not an
  additive field, so it was not started.
- **Group convex hull and minimum bounds `overflow` bit sets.** Making them 0/1 changes results.
  Adding detail fields needs vocabulary names agreed in `analysis-core` (another stream's
  directory). Listed under wave 2.
- **Coverage simplification `topologyStatistics[3]`.** This is a candidate overflow, but it is
  packed in an ad hoc array. Not changed.
- **`requiredLineCount` for `GPULineGeometryOutput`.** Only chunk differs from the input line
  count. Deferred to the wave 2 flip of `count`.

## 4. Wave 2 requests

Renames and meaning changes to finish:

1. `GPULineGeometryOutput.status.count` / `requiredCount`: from vertices to lines. Consumers must
   first read `vertexCount`/`requiredVertexCount`; add `requiredLineCount`.
   - lenses: `scenarios/gpu-spatial-analysis/{line-segmentize,great-circle-arcs,line-smooth,line-chunk,rectangle-clip}.example.ts`,
     `line-segmentize.lens.ts`, `great-circle-arcs.lens.ts`,
     `gpu-spatial-analysis/computation-pack/operators/lines.ts`.
   - kepler: `analysis/{catalog-bindings,catalog-gpu-bindings,catalog-gpu-geometry,catalog-platform4-geometry-lines}.ts`.
   - website: `gpu/runs/entries/geometry-construction-overlay/{line-segmentize,great-circle-arcs,line-smooth,line-chunk}.ts`,
     `gpu/runs/entries/geometry-processing/rectangle-clip.ts`,
     `catalog/entries/geometry-construction-overlay.ts`.
   - examples: `spatial-analysis-explorer/modes/{geometry-mode,geometry-tools-mode,point-pattern-mode}.ts`,
     `spatial-analysis-showcase/src/scenes/geometry/{b3-line-views-city,b3-line-views-far,line-operations.scene,great-circles.scene,buffers-and-shapes.compute,buffers-and-shapes.scene}.ts`,
     `scenes/flows/airline-network.compute.ts`, `scenes/points/point-patterns.compute.ts`.
2. `GPUPolygonOverlayOutput.status.requiredCount` (rings): make it optional and stop writing it.
   Consumers read `requiredRingCount`.
   - lenses: `polygon-overlay.{example,lens}.ts`, `buffer-surface.{example,lens}.ts`,
     `computation-pack/operators/geometry.ts`, `urban-morphology/urban-ecology-operators.ts`.
   - kepler: `catalog-gpu-bindings.ts`, `catalog-gpu-geometry.ts`.
   - website: `geometry-construction-overlay/{polygon-overlay,buffer-surface}.ts`,
     `geometry-construction-overlay/shared/polygons.ts`.
   - examples: `scenes/geometry/{repair-overlay,block-relationships,footprint-shape-lab}.compute.ts`.
3. `GPUPolygonOverlayOutput.geometry.sourceIds` (deprecated, output index): make it optional, then
   remove it. Same consumers as item 2. Callers that pass it to joins as feature IDs get the row
   index anyway.
4. `GPULinearReferencing.status.candidateCount`: remove (already deprecated).
   `status.overflow`: stop folding the intermediate capacities into it once consumers read
   `candidateOverflow`.
   - lenses: `linear-referencing.{example,lens}.ts`, `line-locate.example.ts`,
     `computation-pack/operators/lines.ts`.
   - kepler: `catalog-extra-bindings.ts`.
   - website: `geometry-construction-overlay/linear-referencing.ts`.
   - examples: `geometry-mode.ts`, `b3-line-views-city.ts`, `line-operations.scene.ts`.
   - Also: the vocabulary has no name for an unclamped candidate count. `requiredCount` is misused
     here. Proposal: a core `requiredCandidateCount`.
5. `GPUSharedPathsRuns.leftLineIds` / `rightLineIds`: rename to `leftSourceIds` /
   `rightSourceIds`, keeping deprecated aliases. Also stop folding pair overflow into `overflow`.
   - Consumers: shared-paths users in the kepler catalog (`catalog-platform4-geometry-lines.ts`)
     and `examples/.../noding-and-coverage.*`. Grep `GPUSharedPaths` again before changing.
6. `GPULineClipByPolygon`: stop folding pair overflow into `overflow`, as line split does.
7. `GPUGroupConvexHull` / `GPUMinimumBounds` `overflow` bit set: make it 0/1 and add named detail
   fields. Needs a core vocabulary name, for example `groupOverflow`.
8. Move the target layout into `analysis-core/contracts` (Core stream): a
   `GPUGeneratedGeometryStatusPort` = `GPUBoundedResultStatusPort & {vertexCount?, requiredVertexCount?, ringCount?, requiredRingCount?}`,
   and the validator now in `line-segmentize/line-geometry-output.ts`. Also decide whether
   provenance maps through input `sourceIds` (`ports.ts` doc versus every contributor).
9. Kernel audit: no new limits were added, so no case changed. A follow-up could add cases that
   request the new optional views (shared kernel-audit files).

## 5. Roadmap

- **STD-08**, replace the current state with: "Done 2026-11-10 (additive): target layout decided
  (generated-geometry port; `count` = features; named vertex and ring sub-counts; provenance rule
  for one-to-many, many-to-one and union outputs); `vertexCount`/`requiredVertexCount` on the
  segmentize family and rectangle clip; `sourceIds` and full status on line merge; `ringCount` and
  `requiredRingCount` on overlay and buffer; overlay `geometry.sourceIds` deprecated. Remaining:
  the breaking flips of wave 2 (line output `count` to lines, overlay `requiredCount`, shared-paths
  names); move the layout type into `analysis-core`; offset curve and outline to a compact layout
  (G7); per-feature buffer and per-pair overlay provenance (L)."
- **STD-04**, add to Done: "`candidateOverflow` on shared paths and linear referencing (2026-11-10)".
  Keep the rest, and add: "group hull and minimum bounds overflow bit sets; coverage
  simplification `topologyStatistics`".
- **STD-03**: remove `concave-hull`, `geometry-edit`, `geometry-measures`, `line-segmentize` and
  `polygon-triangulation` from the remaining list (18 calls).

## 6. Noticed

- `ports.ts` says input `sourceIds` replace feature rows in outputs, but line split, line clip,
  merge and overlay all write rows. Either the doc or the contributors is wrong.
- The disk was full (ENOSPC on the first Bash call; 105 MiB free afterwards). Builds and test runs
  may fail for space reasons.
- `GPUConstrainedDelaunayTriangulation` caps `maximumFlips` only by uint32. The bound comes from
  the flip loop, so a named limit with a reason should be added (the class is a GPU-12 rework
  target).
- Node vitest runs take over 10 minutes when batched with `raster-terrain-expansion.node.spec.ts`
  under current load.
