# fix-core report (wave 1, 2026-11-10)

Source review: `reviews/review-joins-geometry-raster.md`. Every finding below was verified with a spec that I
ran against the code before changing it.

## 1. Done

### Finding 1: line merge binds 10 storage buffers. Reproduced, fixed.
- New spec `GPULineMerge kernels fit the default eight storage buffers with every optional output` in
  `modules/analysis/test/gpu-spatial-analysis/line-merge/gpu-line-merge.spec.ts`. It uses
  `getWebGPUTestDevice('core')` and checks that `maxStorageBuffersPerShaderStage` is 8.
  - Before the fix it failed with `GPULineMerge: kernel merge-offsets-11111 binds 10 storage buffers, more than
    the device limit of 8`.
  - After the fix it passes. It checks the chains against shapely, every rule 8 status value and the
    `sourceIds` rows.
- `modules/analysis/src/gpu-spatial-analysis/line-merge/gpu-line-merge.ts`:
  - The `offsets` kernel is back to its exact 74dddcf5 form: same 5 bindings, same body, variant `'offsets'`.
  - Two new nodes are appended after every existing writer, after `line-info`:
    - `status-<bits>`: one thread, at most 6 bindings. It writes `requiredCount`, `overflow`, `vertexCount` and
      `requiredVertexCount`.
    - `source-ids`: `lineCount` threads, 3 bindings. It writes `NONE` past the chain count and each chain's
      leader line.
  - The existing outputs keep their values and their write order. The `getCommandNodes` doc lists the new nodes.
- Kernel-audit case file `modules/analysis/test/analysis-core/kernel-audit/cases-geometry.ts` (small in-place
  edit): the `GPULineMerge` "every output" case now also requests the four status scalars and `sourceIds`.

Binding audit. These are the kernels that gained a storage binding since 74dddcf5, counted with every optional
output requested. I read the code and the diffs to count them.

| Class / helper | Kernel that gained bindings | Max storage bindings |
| --- | --- | --- |
| `GPULineMerge` (before the fix) | `offsets` | **10 (over the limit)** |
| `GPULineMerge` (after the fix) | `offsets` / `status` / `source-ids` | 5 / 6 / 3 |
| Segmentize family and `GPURectangleClip` (`line-geometry-output.ts`) | `vertex-status` alias node | 4 |
| Segmentize publish kernels | `requiredCount` may now be a transient | unchanged (same as passing `requiredCount` before) |
| `GPUPolygonOverlay` (and `GPUBufferSurface` through it) | `ring-status` | 4 |
| `GPUSharedPaths` | `scalars` | 7 |
| `GPULinearReferencing` | `candidate-overflow` | 2 |
| Joins: `getCandidateTruncationNodes` | `candidate-truncated(-bvh)` / `candidate-overflow(-bvh)` | 4 / 3 |
| Joins: `getLeafOverflowStatusNodes` (nearest, great-circle, neighbors) | `leaf-overflow-status` | 3 |
| `GPUBufferSelection`, `GPUPointInPolygonJoin`, `GPUSpatialPredicateJoin` | pass-through only | no new kernel |
| `GPUZonalStatistics` | fill node for `candidateOverflow` | 1 |
| `GPURegionStatistics` | `status` | 3 |
| `GPUTerrainDepressionHierarchy`, `GPUTerrainSpikeRepair` (`createTerrainStatusPortNodes`) | `status-ports` | 3 |
| `GPUCostDistance` | `tie-finalize` | 5 |
| Raster relaxation status (`raster-relaxation.ts`) | status node | 4 |
| `GPUTerrainWatersheds` (`createTerrainIterationLimitNode`) | `iteration-limit` | 2 |
| `GPURasterWatershedSegmentation` | `iteration-status` | 2 |
| `GPUNetworkReachability` | `iteration-limit-reached` | 2 |
| `GPUNetworkCostMatrix` | converged finalize (+`limitReached`) | 3 |

Only line merge exceeded 8.

### Finding 2: the symmetric eigen solver stops before the first sweep at extreme scales. Reproduced, fixed.
- `modules/analysis/test/utils/symmetric-eigen-wgsl.spec.ts`:
  - New test `symmetric Jacobi eigen decomposition is invariant to the matrix scale`. It covers matrices at
    1e-25, 1e-12, 1e12 and 1e20: the reviewer's `[[s, s], [s, s]]`, a repeated-eigenvalue matrix, and random
    sizes 2 to 6. Inputs are rounded to float32. It requires at least one sweep for each case.
  - New test for a zero matrix: status OK, 0 sweeps, zero values, identity vectors.
  - The NONFINITE case now checks that the vectors are left untouched.
  - The binary64 reference now stops on a test relative to the Frobenius norm, so it is also scale-invariant.
    The comparison code was moved into a shared `expectEigenCasesMatchReference`.
- Before the fix:
  - The scale test failed. For example, case 0 (1e-25) returned `(1e-25, 1e-25)` instead of `(0, 2e-25)`.
  - The NONFINITE test failed: `vectors[0][0]` was 1.
- After the fix, all 4 tests pass.
- `modules/analysis/src/analysis-core/kernels/symmetric-eigen-wgsl.ts`:
  - The stopping test scales each entry by `ldexp(value, -frexp(largest).exp)`, where `largest` is the largest
    absolute lower-triangle entry.
  - Scaling by a power of two is exact, so well-scaled input makes the same stopping decisions bit for bit. The
    existing random-matrix test still passes unchanged.
  - Validation now runs in its own first pass. A NONFINITE return therefore writes nothing to `matrix`,
    `vectors` or `values`, as the doc says. The doc now spells this out.
  - The doc names one exception: subnormal-only input, which a GPU may flush to zero.
  - `getSymmetricEigenWGSL` has no consumers outside its spec, so nothing else changes.

### Finding 3: raster resistance overflow is not counted. Reproduced, fixed.
- `modules/analysis/test/gpu-spatial-analysis/urban-ecology/raster-resistance-connectivity.spec.ts`:
  - The copy of `DIRECTIONS` is gone. The reference adjacency now compares every pair of cells:
    - 4 neighbours: Manhattan distance 1.
    - 8 neighbours: Chebyshev distance 1.
  - It is a map keyed by unordered pair. Geographic edge lengths use the midpoint between the two cell centres.
  - `expectEdgesMatchReference` reads the GPU `edges` output (now requested in `runCase`) and checks:
    - every neighbouring pair appears exactly once, which is what the sparse solve expects;
    - no slot joins non-neighbours;
    - self slots have conductance 0;
    - every conductance matches the reference.
  - The existing cases now use this check.
  - New `adjacency-*` cases cover 5x3, 3x5, 6x1 and 1x6 grids, with non-square and sheared cells, for 4 and 8
    neighbours.
  - New overflow cases use R = 1e-35 in two neighbouring cells on 1e-4 cells. They cover east and south pairs
    (4 neighbours) and south-east and south-west pairs (8 neighbours).
- Before the fix the 4 overflow cases failed (`invalidCells` was 0, expected 2). The 8 adjacency cases passed:
  the slot table is correct, so they act as pins. After the fix all 21 tests pass, and the node spec passes 4 of
  4.
- Defined behaviour: a cell with data is invalid when its resistance is not positive, or when an edge between it
  and another positive resistance has a conductance that is not finite in float32. Both cells of such an edge are
  counted. With this rule the stage `resistance` count is nonzero whenever the solve reports invalid edges.
- `modules/analysis/src/gpu-spatial-analysis/landscape-ecology/raster-resistance-connectivity.ts`:
  - The invalid-cell count kernel now also reads `edges` and `conductances` (5 bindings). It checks the NaN bit
    pattern on the cell's own slots and on the incoming slots from its west, north, north-west and north-east
    neighbours. Because it reads what the edge kernel wrote, the count and the edges agree exactly; it does not
    recompute anything.
  - The docs of the props, the output and the function are updated.
- `docs/api/gpu-spatial-analysis/landscape-ecology.md`: one new bullet on the overflow rule, and the validation
  paragraph is updated.
- Results change only for grids where a conductance overflowed. Those already failed the solve; they now also
  count invalid cells.

### Finding 4: binary search with `first > last`. Reproduced (doc only), doc fixed.
- The code returns `first` because `high = max(first, last)`. The doc said `last`.
- I kept the code and corrected the doc in `binary-search-wgsl.ts`: a reversed range is empty, both functions
  return `first` without reading an element, and the result is always in `[first, max(first, last)]`. Changing
  the code would have changed results.
- `modules/analysis/test/utils/wgsl-words-search.spec.ts` adds reversed queries for f32 and u64 and pins all three
  key types to `[first, first]`. It passes (pin; nothing failed before, because the code already did this).

### Finding 5: walking cost regression check. Done. The outputs differ: this is a finding.
- I captured the output once:
  - I copied `git show 74dddcf5:.../gpu-raster-walking-cost.ts` to
    `scratchpad/walking-cost-74dddcf5/gpu-raster-walking-cost-74dddcf5.ts`, with its imports pointed at the
    current helpers. The helper changes since 74dddcf5 are additive.
  - I ran it once on the headless GPU through a temporary spec, which I then deleted.
  - Inputs were those of the spec: Tobler and GRASS on the 3 m grid, `maximumIterations: 64`. Both converged.
  - The raw capture is in `scratchpad/walking-cost-74dddcf5/capture.json`.
- New fixture `modules/analysis/test/geospatial-expansion/raster-walking-cost-74dddcf5-fixture.ts` holds the old
  kernel's cost bit patterns. I did not adjust them.
- `raster-walking-cost-parameters.spec.ts` now compares today's deprecated (no `parameters`) path against the
  fixture.
- **Finding:** on this Metal adapter the GRASS costs are bit-identical, but the Tobler costs are not. 8 of 35
  cells differ by 1 ulp (cells 1, 2, 8, 23, 24, 25, 33, 34). A 0-ulp assertion failed with exactly these distances.
- The spec asserts at most 4 ulp, documents the observed 1-ulp Tobler difference in a comment, and passes.
- Likely cause (not proven): the old kernel folded the compile-time literals (`FRICTION_WEIGHT = 1.0`, the affine
  coefficients), and the new kernel reads them at runtime from `parameters`. That probably lets the Metal compiler
  contract the expression or round `length` differently.
- The "bit-identical without `parameters`" claim does not hold for Tobler against the old kernel. It still holds
  between today's two paths. The owner of the walking cost should decide whether a 1-ulp change is acceptable.

### Finding 6: H3 boundary guarded `atan2`. Spec added.
- New test `H3 boundary longitude guards atan2 against a signed-zero second argument` in
  `modules/analysis/test/gpu-spatial-analysis/cell-indexing/h3-boundary-wgsl.spec.ts`. It calls
  `cellIndexH3BoundaryUnitVectorToLngLat` directly on vectors read from a storage buffer:
  - x = +0 and -0, each with y in {±1, ±0.6, ±1e-7};
  - seven ordinary-quadrant vectors, including x = ±1e-7.
  - Longitude and latitude are compared with `Math.atan2` (binary64) at the file's 3e-5 degree tolerance.
- It passes. With the guard temporarily replaced by a bare `atan2` (one run, then reverted), it failed on all six
  `-0` cases, for example `(-0, 1) longitude -90 vs 90`. So it does catch the FX-17 defect on this GPU.

## 2. Verified
- Before each fix I ran the failing spec as described above.
- Final runs:
  - `npx vitest run --project headless` on `gpu-line-merge.spec.ts`, `symmetric-eigen-wgsl.spec.ts`,
    `raster-resistance-connectivity.spec.ts` and `wgsl-words-search.spec.ts`: 4 files, 37 of 37 passed.
  - `npx vitest run --project headless` on `raster-walking-cost-parameters.spec.ts` and `h3-boundary-wgsl.spec.ts`:
    2 files, 7 of 7 passed.
  - `npx vitest run --project node` on `gpu-line-merge.node.spec.ts` and
    `raster-resistance-connectivity.node.spec.ts`: 6 of 6 passed.
  - `npx vitest run --project node` on `kernel-audit-geometry.node.spec.ts` and
    `kernel-audit-coverage-urban-ecology.node.spec.ts`: 44 of 44 passed.
  - `node scripts/typecheck-status.mjs 'line-merge|symmetric-eigen|binary-search|raster-resistance|walking-cost|h3-boundary|wgsl-words-search'`:
    0 errors. This was the 7:50 PM cycle, after my source edits; I did not re-check after the last spec-only
    edits.
  - `npx biome check --write` on every file I touched: clean.
- Not run:
  - the S2 covering specs (no item touched them);
  - `yarn check-contributor-contract`;
  - the full suites.
- Process note: one in-place replacement in `symmetric-eigen-wgsl.spec.ts` (a spec only I own) was made with a
  `python3` string replace, not the Edit tool. Every other edit of an existing file used Edit. The temporary
  capture spec was created and removed with `rm` (it is my own new file, not tracked).

## 3. Not done
- Nothing from the six items is left open.
- Out of scope, not attempted:
  - the reviewer's other test gaps for resistance (source or sink on a no-data cell, a barrier on the border);
  - dense-solver capacity 64;
  - a prepared BVH for the nearest join.

## 4. Wave 2 requests
- No renames.
- Walking-cost owner: decide whether the 1-ulp Tobler difference against 74dddcf5 is acceptable. If not, restore
  the old expression order or folding on the deprecated path. The fixture and the 4-ulp assertion are in place
  either way; tighten it to 0 if bit identity is restored.

## 5. Roadmap
- No roadmap rows were named for this stream. If the review findings are tracked as rows:
  - line merge binding limit: delete the row.
  - eigen scale-invariance: delete the row.
  - resistance overflow: delete the row.
  - binary-search doc: delete the row.
  - H3 atan2 spec: delete the row.
  - walking cost: "deprecated Tobler path differs from 74dddcf5 by 1 ulp in 8 of 35 cells; decide whether to
    accept it or restore bit identity".

## 6. Noticed
- `raster-walking-cost-parameters.spec.ts` was an untracked new file, and its comment "Same values ... identical
  results" only compares the two new paths. The new old-kernel check now covers the claim about the pre-change
  output.
- `GUARDED_ATAN2_WGSL` returns 0 for `(±0, ±0)`, while `Math.atan2` returns ±π for `(±0, -0)`. The reviewer judged
  this acceptable at the H3 poles; my spec does not pin it.
