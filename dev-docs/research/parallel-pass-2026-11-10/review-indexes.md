# Review: brute-force searches replaced by indexes (uncommitted changes against 74dddcf5)

Scope: local kriging `sampleIndex`, OPTICS `neighborIndex`, fuzzy contiguity BVH, island attachment
merge, pair-statistics `gridSize` (cross-variogram, marked point statistics), network snapping
unindexed cap, and the urban-morphology segment BVH (visibility graph, viewpoint classifier).
Method: I read every diff and the helpers they call (`GPUNeighborSearch` kNN and radius semantics,
`createNeighborSearchDerivedParameterNodes`, `createCSRStages`, `getSlotGroupingNodes`, `GPUBVH`
d.ts, `pair-statistics-grid` lattice, `urban-ray-utils`). I checked float32 corner cases with two
CPU scripts that emulate f32 with `Math.fround`, in this folder: `kriging-tie.mjs` and
`fuzzy-slack.mjs`. I ran no spec, because of the concurrent full run.

## Defects, most severe first

### 1. Confirmed by derivation: fuzzy contiguity drops pairs at the tolerance and gives an asymmetric CSR (default path, any feature count)
`modules/analysis/src/gpu-spatial-analysis/spatial-weights/gpu-fuzzy-contiguity.ts:204`

The query box is `ownBox ± (TOLERANCE + slack)`, with `slack = max|own box coordinate| * 1e-5`. The
slack scales only with the probing feature's coordinates and not with `TOLERANCE`. When the
tolerance is much larger than the probing feature's coordinates, `boxMaximum + (T + slack)` rounds
down by up to half an ulp of `T`. A neighbour whose exact gap rounds to `<= T` (so the old dense
path, and the f32 distance test here, accept it) can then fall just outside the query box and be
pruned. The probe from the other side uses that feature's larger coordinates, so its slack covers the
rounding and it still finds the pair. The pair therefore appears in one row only. The class
documents a symmetric CSR, and this breaks it.

Concrete input (f32 emulation, `fuzzy-slack.mjs`): `tolerance = 100`. Feature A is the square
`[0.0861, 0.1861] x [0, 0.1]`. Feature B is the square `[100.18610382080078, 100.3] x [0, 0.1]`
(B.minX is the next f32 above `fl(0.1861 + 100)`).
- The edge gap is `fl(100.1861038 - 0.1861) = 100`, so the distance test accepts the pair.
- A's query maximum is `fl(0.1861 + fl(100 + 1.86e-6)) = 100.1860962 < B.minX`, so A does not list B.
- B's slack is about `1e-3`, so B lists A.

This path has no opt-in. The BVH replaced the dense table for every input, so this is a regression
even at 512 features or fewer.

Fix direction: make the slack `max(max|coordinate|, TOLERANCE) * 1e-5`, or test the expanded box in
a form that cannot round inward.

### 2. Confirmed by read: local kriging `'auto'` silently keeps an uncapped exhaustive scan above the cap (contract rule 3)
`modules/analysis/src/gpu-spatial-analysis/spatial-interpolation/gpu-local-kriging.ts:388-396`

With more than 4096 samples, `'auto'` falls back to `'brute-force'` in two cases:
- The scan already uses 8 bindings (external drift + `parameters` + `output.weights`).
- `queries * (neighborCount + 4) * 4 > 128 MiB`, for example 2M queries at `neighborCount 16`.

The work cap `GPU_LOCAL_KRIGING_MAXIMUM_BRUTE_FORCE_WORK` (validated at `:370`) applies only to an
explicit `'brute-force'`. So `mode: 'external-drift'`, `parameters`, `output.weights`, 1M samples and
1M queries constructs without error and runs 1e12 distance tests, one query per invocation over
every sample. Rule 3 says brute force is never the silent default above its cap.

### 3. Suspected (derived; depends on the GPU's `sqrt`/`length` rounding): local kriging grid candidates can differ from the scan on near-ties
`gpu-local-kriging.ts:72` (`LOCAL_KRIGING_INDEX_CANDIDATE_MARGIN = 4`)

The two paths rank by different keys:
- The index ranks by `(fl(dx^2 + dy^2), id)` and returns `neighborCount + 4` candidates.
- The kernel ranks by `(distance(), row)`, where ties go to the lower row.

`sqrt` maps about two adjacent f32 squared distances to one distance. A sample X whose `d^2` is one
ulp above that of five or more samples with higher rows can therefore have the same `distance()` as
them. The scan then keeps X (it has the lower row), while the index ranks it below all five and
drops it.

Concrete input (`kriging-tie.mjs`, f32 emulation): one support at `(0, 0)`, `neighborCount: 4`
(so k = 8).
- Rows 1 to 8 sit at `(±3, ±4)` and `(±4, ±3)`, all with `d^2 = 25`.
- Row 0 sits at `(3.000000238418579, 4)`, with `d^2 = 25.0000019` (25 + 1 ulp) and `sqrt` rounding to `5`.
- The scan keeps rows {0, 1, 2, 3}. The grid keeps rows {1, 2, 3, 4}. Weights and values differ.

The margin absorbs at most four such rows. The code comment claims the margin makes the retained
set equal the scan, and that does not hold in general. A robust fix is for the kernel to break ties
by the same `d^2` key, or to compare `d^2` instead of `distance()`.

### 4. Confirmed by read: pair-statistics `gridSize` lifts the row cap to 2^24 - 1, but single-invocation loops over all rows remain (contract rule 2)
- `dataframe-analysis/pair-statistics/gpu-cross-variogram.ts:203`: the finish pass runs one
  invocation per bin and loops over every row.
- `gpu-marked-point-statistics.ts:245`: the same finish pattern.
- `gpu-marked-point-statistics.ts:180-187`: the `moments` pass is `invocationCount: 1` over every row.

These loops were bounded at 16,384 rows before. With `gridSize` they reach millions of rows. For
example, `GPUCrossVariogram` with `lagCount: 1`, `gridSize: [64, 64]` and 16M rows runs one
invocation that loops 16M times; the marked moments pass with 5M rows does the same. On a real GPU
that risks a device timeout, and it is the rule 2 violation that the index was meant to remove.

Related, suspected: when `maximumDistance >= extent`, the lattice collapses to one cell
(documented). The grid pair pass then loops over every row per invocation, an O(n^2) pass at up to
2^24 rows, with no cap on cell occupancy.

### 5. Suspected (performance or timeout, not wrong values): the fuzzy contiguity BVH is in source order, not spatially sorted
`gpu-fuzzy-contiguity.ts:134` (`GPUBVH`, complete binary, "Source order defines stable leaf slots")

For spatially incoherent feature order (the new spec deliberately shuffles), internal boxes cover
most of the extent and each probe visits O(n) nodes. The traversal plus the exact
`getFeatureDistance` runs three times, in the count pass and the two emit passes
(`createCSRStages` re-runs `visit` per attribute, `:282`). With the cap raised from 512 to 65,536,
65,536 shuffled features cost about 3 x 65,536 x 131,071 node tests, with no cap.
`addUrbanObstacleBVH` already uses `spatialSort: true` (`getSegmentBVHNodes`). This class does not,
so the index does not meet rule 3's intent above small sizes.

### 6. Confirmed by read, low: an explicit kriging `sampleIndex: 'grid'` skips its limit checks (rule 10)
`gpu-local-kriging.ts:388-401`

The 128 MiB candidate-row check and the 8-binding check run only for `'auto'`. An explicit `'grid'`
whose `queries * k * 4` exceeds the device binding size, or that needs a ninth binding on an
8-binding device, passes construction and fails later in device validation. The contract asks for
`validateGPUStorageBindingByteLength` at the start of `getCommandNodes`.

### 7. Confirmed by read, low: the network snapping unindexed cap turns kepler's exact fallback into a CPU fallback
- `gpu-network/network-accessibility/gpu-network-snapping.ts:291`
- `modules/kepler/src/analysis/catalog-gpu-snapping.ts:146,149`

Kepler retries with the exhaustive scan after a BVH candidate overflow, and it always scans when
`maximumDistance` is not finite or is 0. Above `points * segments = 2^27`, that retry now throws.
`snapCatalogPoints` catches the error and quietly returns `engine: 'cpu-reference'`. No wrong
values result, but the GPU path disappears for large networks without a radius, and the fallback
comment "the exact scan is the fallback, never a wrong answer" is now only half true.

## Areas with no defect found (what I checked)
- **OPTICS grid path** (`gpu-optics.ts:540-958`).
  - Selection is by (reachability, row) over the seed list. When the seed list is empty, the
    selection is the lowest unvisited row past a monotone cursor (rows below the cursor can never
    return to UNVISITED).
  - Seed swap-remove followed by an atomic append: slots are distinct, and `seedBase` is read after
    the removal.
  - Each candidate row is relaxed by one thread, so there is no write race. Barriers are in uniform
    control flow, and `workgroupUniformLoad` is used correctly.
  - Relaxation re-tests with the scan's own distance and time predicates. The radius and time
    widening (1e-4) only adds candidates. Invalid rows enter as VISITED.
  - Overflow leaves `status = [0, complete, invalid]`, reported through `ports.status.candidateOverflow`.
  - 7 bindings per batch. The node budget is validated, and the xi cap stays at 4096. Great-circle
    distances use the same `GEODESIC_WGSL` haversine in the search and in the frontier.
- **Island attachment merge.**
  - The three-way ascending merge equals the old per-`other` scan: duplicates collapse to the first
    slot, self loops are kept only from the original, out-of-range IDs are dropped, and
    `NO_SLOT = 0xffffffff` terminates the merge correctly.
  - Inverse grouping uses key `rows` for "none", and `getSortKeyBits(rows)` covers that key.
  - 8 bindings in the weights emit pass.
  - The island nearest-representative search is still an all-rows scan, as documented.
- **Fuzzy contiguity, other than defects 1 and 5.**
  - `getFeatureDistance` equals the old symmetric distance.
  - Invalid features get empty leaf bounds (they never probe or match), and padded leaves are
    guarded by `other < ROWS`.
  - Leaves are in source order, so rows come out ascending.
  - Overflow and `requiredCount` come from `createCSRStages` with `initializesOverflow`.
- **Pair statistics, other than defect 4.**
  - Inclusion matches the dense `included()`: mask, finiteness of the second value and of time.
  - Bounds are rebuilt by atomics every frame, and the bounds keys are cleared every frame.
  - Cells are at least `radius * 1.00098` wide, with clamped edge columns and rows.
  - `neighbor <= focus` counts each pair once.
  - Marked shells: `getUpper(b - 1)` is bit-identical to the dense `lower(b)`, so each pair lands in
    exactly the dense shell, and the cumulative prefix equals the dense cumulative.
  - The `rows * bins * 6 * 4` partial table exactly fits 128 MiB at the cap.
  - The R^2-overflow difference from dense is documented.
- **Urban morphology.**
  - Sightline `castRay` with the BVH prunes at `nearest(1 + 1e-5) + 4 * tol` over padded boxes, and
    only "any hit" matters for visibility.
  - The classifier's horizontal-ray box test admits every segment that can change the winding
    number or lie within the tolerance, because the boxes are padded by `2 * tol + 1e-6 * largest`.
  - Non-finite segments are counted by a `GPUReduction` instead of the per-viewpoint loop.
  - 7 bindings per kernel.
  - The only theoretical miss is obstacles more than 1e30 to the right of a viewpoint (the
    `getBoxEntry` limit), which is not practical.
- **Kriging, other than defects 2, 3 and 6.**
  - The usable-sample predicate is shared by the mask pass and the kernel.
  - Centers are masked targets, so the bounds always cover the queries. NaN centers give empty rows.
  - The padding pass writes `0xffffffff`, which the kernel skips through `sample >= SAMPLE_COUNT`.
  - `searchRadius` is applied after the k-nearest search, which is correct.
  - The kNN output capacity is exact, so ignoring its overflow is safe.

## Test gaps that would hide a defect
- `local-kriging-sample-index.spec.ts` uses only integer lattice coordinates and 81 samples. Every
  tie is an exact `d^2` tie. There is no near-tie (`d^2` one ulp apart, as in defect 3).
- The kriging spec has no co-kriging or external-drift scenario on the grid path, so the shared
  usable-sample mask is untested with variable or drift exclusions.
- The kriging spec has no end-to-end `'auto'`-resolves-to-grid comparison: the >4096 case checks
  only the resolved enum.
- `fuzzy-contiguity-index.spec.ts`:
  - Tolerance is 0.1 and the gaps are 0.05, 0.08 and 0.15. No gap equals the tolerance, and no case
    has a tolerance large relative to the coordinates, so defect 1 cannot show.
  - It does not check that the CSR is symmetric.
  - 579 features cannot expose the unsorted-BVH cost (defect 5).
- `pair-statistics-grid-paths.spec.ts`:
  - The more-than-16,384-row test uses integer coordinates, far from the 3.5 cutoff.
  - No pair lies on a cell boundary or on a bin edge (marked shells).
  - No row count is large enough to exercise the single-invocation finish and moments loops
    (defect 4).
  - The "invalid distance" case uses only `maximumDistance: 0`; there is no case with
    `maximumDistance >= extent` (the one-cell collapse).
- `optics-grid-index.spec.ts` coordinates are on a 1/8 lattice, which is exact. Good for ties, but
  no pair distance falls within f32 rounding of `maximumDistance` with a non-representable radius.
  The great-circle case does cross the antimeridian.
- `urban-ecology-visibility-index.spec.ts` uses random coordinates, so no viewpoint shares a y with
  a wall vertex. The half-open winding rule at vertex y is not exercised against the BVH box test.
- No spec covers kepler snapping above the new 2^27 unindexed cap (defect 7).
