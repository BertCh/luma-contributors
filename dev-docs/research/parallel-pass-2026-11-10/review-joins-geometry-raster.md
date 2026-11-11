# Review: joins, geometry status, raster parameters, atan2 guard, resistance edges, WGSL helpers

Scope: uncommitted changes against `74dddcf5` under `modules/analysis/src/` (slice listed in the brief) and their
specs. Method: I read the diffs and the new files. I ran no spec, because the full test run was in progress.
One scratch script, `reviews/eigen-check.mjs`, emulates the float32 stopping test of the Jacobi eigen solver.

## Confirmed defects

### 1. `GPULineMerge` offsets kernel binds up to 10 storage buffers (default device limit is 8)
- Where: `gpu-spatial-analysis/line-merge/gpu-line-merge.ts:571-597` (bindings), built from `optionalOutputs` at `:276-299`.
- What is wrong: the `offsets` kernel always binds 5 storage buffers: `leaderVertices`, `flagScan`, `vertexScan`,
  `chainOffsets` and `count`. Each new optional output adds one more in the same kernel: `requiredCount`,
  `overflow`, `vertexCount`, `requiredVertexCount` and `sourceIds`. With all five the kernel binds 10. With four it
  binds 9. WebGPU's default `maxStorageBuffersPerShaderStage` is 8.
- Input that fails: on a device with default limits (`getWebGPUTestDevice('core')`), any `GPULineMerge` that wires
  at least four of the five new optional outputs, for example the exact output set in
  `test/gpu-spatial-analysis/line-merge/gpu-line-merge.spec.ts:85-108`. `createWGSLKernelNode`
  (`analysis-core/kernels/wgsl-kernel-nodes.ts:328-333`) then throws
  `GPULineMerge: kernel ... binds 10 storage buffers, more than the device limit of 8`.
- How I know: I read the code and counted the bindings. The spec passes only because the default test device
  raises the limit. The repo already treats this as a bug: see the regression test
  `gpu-cell-set-outline.spec.ts:335-340`, which checks that kernels fit the default 8 buffers.
- The overlay change did split its ring status into its own node for this reason. Line merge needs the same fix:
  a separate status/sourceIds node after `offsets`.

### 2. Jacobi eigen solver stops before any rotation when squared entries overflow or underflow in f32
- Where: `analysis-core/kernels/symmetric-eigen-wgsl.ts:74-84`.
- What is wrong: the stopping test sums squares of raw entries, with no scaling first.
  - If entries are below about 1e-19 in magnitude, every square underflows to 0. Then `offDiagonal == 0.0` is true
    and the loop stops before the first sweep.
  - If entries are above about 1.8e19, the sums become `inf`. Then `inf <= tol^2 * inf` is true and the loop also
    stops.
  - Either way the function returns status `OK` (0) with the unrotated diagonal as the eigenvalues and the
    identity as the eigenvectors.
- Input that fails: `[[1e-25, 1e-25], [1e-25, 1e-25]]` returns eigenvalues `(1e-25, 1e-25)` and status 0. The true
  eigenvalues are `(0, 2e-25)`. Likewise `[[1e20, 1e20], [1e20, 1e20]]` returns `(1e20, 1e20)` instead of
  `(0, 2e20)`.
- How I know: I derived it by hand and checked it with the float32 emulation (`eigen-check.mjs` prints
  `stopsBeforeFirstSweep=true` for both scales). The rotations themselves would work at these scales; only the
  stopping test fails. A fix: scale by the largest absolute entry before the test, or compare the off-diagonal
  entries one at a time against the diagonal.
- Severity: medium for a library helper. Covariance or Gram matrices of tiny-unit or huge-unit data hit it silently.

## Suspected / minor

3. **Raster resistance: tiny positive resistance gives a NaN (invalid) edge but counts no invalid cell.**
   - Where: `landscape-ecology/raster-resistance-connectivity.ts:425-427`.
   - When `2 / (R1 + R2) / edgeLength` overflows f32, `conductance` stays NaN. The sparse solve then reports
     invalid edges and publishes no potentials, yet `invalidCellCount` is 0. This contradicts the documented rule
     that only zero or negative resistance is invalid.
   - Example: a unit grid with R = 1e-39 in two neighbouring cells (if denormals are kept), or cell size 1e-4 with
     R < about 6e-35.
   - Found by reading. Low severity; clamping the conductance to f32 max, or counting such cells, would make it
     consistent.
4. **Binary search with `first > last`.**
   - Where: `binary-search-wgsl.ts:84`.
   - It returns `first`, not `last` as the doc says ("or `last` when there is none"). Harmless for valid ranges;
     the doc is inexact. The spec query `[0, 5, 3, ...]` exercises this case.
5. **Eigen solver `NONFINITE` path writes part of the output.**
   - Where: `symmetric-eigen-wgsl.ts:61-71`.
   - The doc says "nothing is written", but the identity is already written into `vectors` (and the upper triangle
     of `matrix`) for rows before the bad entry.
6. **`atan2` guard and denormals (theoretical).**
   - `s2AnalysisAtan2` and the H3 guard test `x == 0.0`. If a GPU flushes a negative denormal `x` to -0 inside
     `atan2` but not in the comparison, the wrong-sign result would come back.
   - I cannot confirm this on any known adapter. Otherwise the guard is right in all four cases: `x = +-0` with
     `y > 0` gives +pi/2, with `y < 0` gives -pi/2, and with `y = 0` gives 0. In the winding loop, `y = 0, x = -0`
     cannot happen because of the `> 1e-12` guard. At the H3 poles, longitude 0 is acceptable.

## Areas checked with no defect found

- **BVH overflow flags (spatial joins).**
  - `getCandidateTruncationNodes` now ORs `bvhOverflow` into both `candidateTruncated` (every row) and
    `candidateOverflow`.
  - It is wired in:
    - point-in-polygon (`gpu-point-in-polygon-join.ts:587`)
    - predicate join, both the prepared BVH (`prepared.storage.overflow`) and the built one (`:963-989`, `:1135`)
    - planar nearest (`gpu-nearest-feature-join.ts:1024`)
    - great-circle nearest and neighbors mode, through `getLeafOverflowStatusNodes`
  - `getSpatialJoinCandidateNodes` has a single exit that always adds the truncation nodes.
  - Every new kernel binds at most 4 buffers. Nothing in the slice reads these flags back on the CPU (grep for
    `readAsync`/`mapAsync`).
- **Anti-join.**
  - `candidateTruncated` is always created for `how: 'anti'` (`:994-998`).
  - `unmatched-flags` keeps a row only if it has no match and is not truncated (`:1908-1912`). After a BVH overflow
    every row is withheld, and `overflow`/`pairsCandidateOverflow` already folded `bvhOverflow` (`:1600`). This is
    correct and conservative.
  - Neighbors-mode tie/neighbor capacity overflow is not part of `candidateTruncated`. That is documented, and it
    reaches `overflow`.
- **Pass-throughs.**
  - Zonal statistics: the length check, the polygons-only requirement, and a 0 fill for `feature-rows`.
  - Buffer selection: passes through to the nearest join with a matching length check.
  - Line-length-per-polygon: `candidateOverflow` = `GPUSpatialJoinCandidates` pair overflow, which includes BVH
    overflow (`gpu-spatial-join-candidates.ts:381`).
  - Region statistics: the status node reads the summary flags after the summary node.
- **Geometry status.**
  - The line-geometry `vertexCount`/`requiredVertexCount` aliases are added by all five users of
    `validateLineGeometryOutput`. The transient `requiredCount` only stands in for an optional view the caller
    could have supplied, so no publish kernel gets a new maximum binding count.
  - Overlay ring status is a separate appended node. It does not reorder existing writes.
  - The shared-paths `state` has 8 words, and the new word 5 is written next to word 4. The scalars kernel binds at
    most 7 buffers.
  - The linear-referencing `candidateOverflow` copy is appended after the writers in both branches.
  - The assert-to-validation conversions keep the same conditions. Concave hull's circumradius check still uses
    `Math.fround` through `validateFiniteNumber`.
- **Walking cost.**
  - The deprecated path packs the same float32 values that were literals before.
  - Prepare and relaxation kernels stay at 8 and 6 bindings or fewer.
  - `FRICTION_WEIGHT * 0.5` becomes `w * 0.5`, which is exact.
  - The GRASS slot mapping 8 to 12 matches the packer.
  - The midpoint-row ground scale matches `getRasterGroundScaleWGSL` and the row-0 north-edge convention.
- **Negative numbers in generated WGSL (`x--120`).**
  - In reprojection, every `-${...}` of a signed number is parenthesized. Fractional coverage no longer
    interpolates affine values.
  - I searched `raster-analysis` and `gpu-terrain` for other `-${…}` interpolations. The rest are non-negative
    integers (radius, distance), or `+${…}`/`*${…}`, which cannot form `--` (for example
    `gpu-terrain-stream-network.ts:348`).
- **Raster resistance edges.**
  - Slot layout: east (+1, 0), south (0, +1), south-east (+1, +1), south-west (-1, +1). Each unordered neighbour
    pair appears exactly once, with no duplicates or gaps.
  - Off-grid slots are `(cell, cell)` with conductance 0, and `other` is always below `cellCount`.
  - No-data (zero validity, NaN, +Inf) gives 0. Zero or negative resistance gives NaN.
  - Uniform edge length is `|A*(dc,dr)|`, exact for sheared or non-square cells. Geographic/mercator uses the
    midpoint row.
  - The maximum edge count of 4 x 524,288 slots equals the solve limit.
- **Dense solvers.**
  - All indices are below `capacity` (`size` is clamped by `min(size, N)`).
  - Cholesky's equilibration and its pivot ratio are as documented.
  - LU pivots are LAPACK-style sequential swaps applied consistently in the solve. The log-determinant correction
    for Ruiz scaling is correct.
  - A zero matrix with 0 passes is reported `SINGULAR`, and size 0 returns OK.
- **uint64 helpers.**
  - Add, subtract, carry and borrow are correct.
  - `multiplyUint32Wide`: worked by hand for `0xffffffff * 0xffffffff`, giving a high word of `0xfffffffe`.
  - Restoring division with `overflowBit` is exact.
  - Shifts are correct at 0, 32 and 63, and give 0 at 64 and above.
  - `int64ToFloat(INT64_MIN)` gives -2^63.
- **Binary search.** Correct for empty ranges, duplicates (lower and upper bound), and keys below the first or
  above the last element. The f32 ordered key and the u64 word order are consistent.
- **Sphere-distance helpers and the mask alias.** The antipode guard is correct. `resolveRasterInputMask` throws on
  both, and every consumer reads `props.mask`; isoband rings passes `mask` explicitly.
- **Cost distance / watershed `iterationLimitReached`.** Status word 3 is only ever 0 or 1, so `1u - x` cannot wrap.

## Test gaps that would hide a defect

- **Line merge** (`gpu-line-merge.spec.ts`): runs only on the raised-limit device, so it cannot catch defect 1. The
  status/sourceIds case needs a `getWebGPUTestDevice('core')` variant, as `gpu-cell-set-outline.spec.ts` has.
- **Walking cost "bit-identical without `parameters`"** (`raster-walking-cost-parameters.spec.ts:204-224`): compares
  the new `parameters` path with the new deprecated path. Both use the rewritten kernel, so it would not detect a
  change from the pre-change output. A golden array captured at `74dddcf5` would test the claim.
- **Symmetric eigen spec**: uses only O(1) magnitudes with capacity 6, so defect 2 passes. Fractional coverage has
  no test of the same kind for its deprecated path.
- **Raster resistance spec**:
  - `buildReferenceEdges` copies the implementation's slot table (`DIRECTIONS`), so a wrong direction table would
    be wrong in both. No assertion checks that the set of unordered edges equals the grid's 4- or 8-adjacency.
  - No case puts source or sink on a no-data cell, puts a barrier along the grid border, or uses a 1-row or
    1-column grid.
- **`atan2` guard**: the H3 boundary guard (`h3-boundary-wgsl.ts:158`) has no direct spec. The winding spec covers
  `-0` denominators with a positive numerator, but not explicitly a negative numerator over `-0`.
- **BVH overflow spec**: does not exercise line-length-per-polygon with `leafCapacity` (it cannot overflow there,
  because the leaf capacity is derived). It also does not exercise a prepared BVH for the nearest join; the
  predicate join is covered with `prepared`.
- **Dense solvers**: tested at a few capacities. No spec generates the largest capacity (64) to check that the
  private-memory arrays compile on the test adapter.
