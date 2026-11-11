# Wave 2 kernel review

Scope: the five kernel-level change sets of wave 2, compared against `pre-w2`. Read-only. No specs were run. One float32 emulation was run: `south-row.mjs` in this folder.

## Confirmed defects (most severe first)

### 1. The sweep missed `GPULineSplit`: every line-noding path still flags padding rows

- **Where:** `modules/analysis/src/gpu-spatial-analysis/line-split/gpu-line-split.ts:319-335` (`flag-non-finite`).
- **What is wrong:** the kernel runs over every row of `lines.positions`, with only `index < VERTEX_COUNT` as a guard. Any non-finite row sets `nonFiniteFlag`, including rows before `lineOffsets[0]` and rows from `lineOffsets[last]` on. The `incomplete` and `scalars` nodes (`:852-898`) then fold that flag into `pieces.status.overflow`.
- **Who inherits it:**
  - `GPULineNoding` (`geometry-topology/gpu-line-noding.ts:219-236`). Its `splitIncomplete` reaches `overflow` at `:549`.
  - Through `GPULineNoding`: `GPUPolygonize` without `snapRounding` (`gpu-polygonize.ts:244`), `GPUMakeValid` without `snapRounding`, and `GPUOffsetCurveComponents` (`gpu-offset-curve-components.ts:127`).
  - `GPUNetworkNoding`. `splitOverflow` reaches `overflow` at `gpu-network-noding.ts:758`.
  - `GPULineLoopRemoval`.
- **The authors' claim is wrong:** the sweep note says "`GPULineNoding` is bounded by `pieceVertexCount`: not affected". The flag is computed inside the `GPULineSplit` that `GPULineNoding` embeds, before any piece exists.
- **Failing input:** `GPUNetworkNoding` with `lines.positions` of 6 rows and `lineOffsets = [0, 4]`, where rows 4 and 5 are NaN (or unwritten capacity in a reused transient). Result: `output.status.overflow = 1` while the network is complete.
  - Same with `GPUPolygonize` (no `snapRounding`) on a fixed-capacity line buffer that is padded flat.
- **How I know:** read. Only the snap-rounded path was fixed (`isLineRow`).
- **Fix shape:** test the same membership as `isLineRow` (`lineOffsets[0] <= row < min(lineOffsets[last], rows)`) in `flag-non-finite`.

### 2. `GPUMakeValid` `ring-closure-positions` still copies unowned rows when ring offsets do not start at 0

- **Where:** `modules/analysis/src/gpu-spatial-analysis/geometry-topology/gpu-make-valid.ts:276-297`.
- **What is wrong:** for a closed row `index < ringOffsets[0]`, the binary search returns ring 0. If ring 0 has vertices, `source = index - 0 = index`, which is below `ringStart`. The test `source >= ringEnd` does not catch that, so the kernel copies `positions[index]`, a row no ring owns.
  - The new comment says this never happens ("never `positions[ringOffsets[last]]`, which no ring owns"), but leading rows are not covered.
- **Effect:**
  - With `snapRounding`: harmless. Those closed rows fall before `closedOffsets[0]`, and the noder now ignores them.
  - Without `snapRounding`: they go to `GPULineSplit`'s `flag-non-finite` (defect 1) and set `overflow`.
- **Failing input:** `GPUMakeValid` with no `snapRounding`, `ringOffsets = [2, 6]`, and positions rows 0 and 1 NaN. Result: `overflow = 1`.
- **How I know:** read and derived.
- **Fix:** add `|| source < ringStart` to the redirect test, or apply the padding rule to rows before `closedOffsets[0]`.

### 3. The `'south'` row mapping (now the default) puts points just below a row boundary in the row above

- **Where:** `modules/analysis/src/pointcloud-analysis/gpu-pointcloud-surfaces.ts:87-91`, and the same arithmetic in `pointcloud-grid-index.ts` (`south` branch, `ceil(coordinate.y) - 1`).
- **What is wrong:** `coordinate.y = H - southOffset.y` is rounded in f32 before `ceil`. When `southOffset.y` sits just below an integer `k`, the subtraction rounds up to `H - k`. The point then lands in row `H - k - 1` instead of `H - k`.
  - The north mapping `floor(offset)` has no such second rounding. So the default change makes cell membership differ from the documented half-open rule and from the mirror of the north result.
- **Failing input (emulated with `Math.fround`):**

  | `H` | `cellSize` | origin | point y | row produced | row expected |
  | --- | --- | --- | --- | --- | --- |
  | 1000 | 1 | 0 | 0.99999994 | 998 | 999 |
  | 4096 | 1 | 0 | 1.9999 | 4093 | 4094 |

  Uniform random y at `H = 1000`: 5 of 200,000 points are misassigned.
- **Scope:** Surfaces, Height (cell, bilinear, IDW), Ground, SMRF and the grid index all use the same arithmetic. No stage of a pipeline disagrees with another, so nothing is mirrored by only one stage; the error is in membership at boundaries only. Low severity.
- **Fix:** compute `row = H - 1 - floor(southOffset.y)` in integers, with the top-edge test kept on `southOffset.y < H`.

## Suspected defects (not proven)

### 4. `GPULineClipByPolygon` `segment-ranges` joins leading padding to line 0

- **Where:** `gpu-spatial-analysis/line-clip/gpu-line-clip-by-polygon.ts:367-418`.
- **What is wrong:** `lineOfVertex` returns 0 for rows before `lineOffsets[0]`, and `valid = index + 1 < lineEnd` is then true. A padding row therefore forms a segment with the next row, and the last padding row forms one with the first real vertex.
- **What saves it today:** pairs come from the segment table, which uses an exact `ringOfVertex`, so no pairs exist for these rows. Whether the phantom segment is still emitted as an inside or outside piece needs a read of the emit kernel, which I did not finish.
- **Reachable:** only with leading offsets.

### 5. The two sites listed as not fixed

- **`getSnapOrigin` reads row 0** (`coverage-topology-kernels.ts:45`, used by dissolve, simplification and contiguity weights).
  - This matters only when `ringOffsets[0] > 0`. With an all-empty input no vertex is quantised, so that case is harmless.
  - GeoArrow buffers built from offset 0 never reach it. A sliced Arrow column keeps non-zero offsets, and I did not verify that every upload path rebases them. Reachable from a caller only through such a slice.
- **`compact-flags` in `gpu-buffer-surface.ts:747`:** it does lack the `index >= begin` test. Its offsets are built internally from 0, so I agree it is not reachable.

## Checked and correct

### Padding-row fixes

- **Overlay `combine-positions`:**
  - Every combined row is written.
  - The padding vertex is `positions[ringOffsets[0]]`, which is the first vertex of the first non-empty ring because offsets are monotone. It is a vertex that already exists, so it adds no hot pixel and does not move the lattice bounds.
  - Empty rings become one-row lines with no segment.
  - Tail rows after a non-empty ring repeat its first vertex. That gives zero-length segments, which `segmentUsable` in the noder and `segmentIsUsable` in the segment table drop, so the noder does not count them.
  - Leading rows (`index < combinedOffsets[0]`) fall outside the noder's lines.
  - A left operand with tail rows shifts right ring 0 within its own line only, which was already true before.
  - When both operands are empty, `(0,0)` rows are one-row lines and the bounds stay finite (all zero).
  - No real result changes.
- **Noder `isLineRow`:**
  - Correct inequalities at both ends.
  - Padding rows stay NaN in `latPos`, and every later reader skips them: `mark-vertices` uses `lineOfVertex`, the segment table uses `ringOfVertex`, and the BVH scene bounds use only valid segments.
  - With no valid row, the bounds reduction publishes zeros, so the lattice is still usable.
- **Coverage `snap-range`:** `[ringOffsets[0], ringOffsets[last])` is correct. Both callers pass the `ringOffsets` that index the `positions` they pass.
- **Ring-assembly `hash`:** `counters` is only read, so `read` is the right declared access.
- **Loops:** no data-sized single-invocation loop was added.

### Dense solvers

- **`getCholeskyWGSL`:** the same subtraction order in the factor, forward and back loops, and division by the pivot in the same place.
  - The scale is a literal `1.0`, so `x * 1.0` is exact.
  - Differences: lower-triangle entries are now checked for finiteness up front, and pivots above 3e38 are now accepted. Both fail either way.
  - Status 3 cannot occur with a limit of 0, and every caller still tests a bool.
- **`getEquilibratedCholeskyWGSL`:** the same `1 / sqrt` and the same `(a * s_r) * s_c` order.
  - The limit test is now a ratio against `(1e-3)^2`, not `L_jj >= 1e-3`. This was disclosed, and it differs only within an ulp of the limit.
- **Kriging:**
  - `largestEntry` equals the old `max(1, max|gamma|)`.
  - The pivot choice is strict `>` in both versions. Elimination performs the same U operations.
  - The right-hand side is permuted first and then forward-substituted. This performs the same subtractions in the same order as the old in-place elimination of the augmented column.
  - The pivot-ratio test moved from `min >= tol * max` to `min / max < tol`, which differs only in rounding at the boundary.
  - Every non-zero status maps to `STATUS_UNCERTAIN`.
  - `SYSTEM_STRIDE` is 17 everywhere it is used.
- **Compile-time sizes and declarations:**
  - Capacities are compile-time constants; `count` is runtime and at most the capacity.
  - The `pivots` identity fill is overwritten for every column before any read.
  - No duplicate or missing helper in any combination: panel lag uses `p` and `p + 1`; 2SLS, IV and restricted regimes guard equal sizes; SUR uses a `Set`; the GWR, nonstationarity and GGWR grid paths leave out `IS_FINITE_VALUE_WGSL`.

### Raster parameters

- **Walking-cost packer against the shader:** `[a, b, d, e, northEdge, southEdge, frictionWeight, 0, c0..c3, threshold]` matches the shader reads in slots 0-6 and 8-12.
  - The row-0 edge is used as `mix(...)` start, consistently.
- **Fractional-coverage inverse affine:** `col = (e*x - b*y) / det` and `row = (-d*x + a*y) / det` match slots 2-6.
- **Callers that now build the buffer:** the lens runner (Tobler with Tobler) and the dasymetric recipe forward correctly.

### Point-cloud rows

- `getPointCloudCellSource` and the grid index use one convention.
- Bilinear and IDW work in the same row-down coordinate.
- The SMRF gradient uses only the magnitude, and its edge margins are symmetric.
- No module outside `pointcloud-analysis` consumes these rasters.

### Line counts

- `createLineGeometryCountNode` uses each class's own line capacity (input lines, pairs, output-offset capacity for chunk, rings for polygon clip).
- Vertex clamps still use `positions.length` (`CAPACITY`).
- The rectangle-clip line-tail words map correctly: count is word 1, requiredCount word 3, vertexCount word 0, requiredVertexCount word 2.
- Lon-lat preparation keeps vertex counts, as intended.

## What the new specs would not catch

- No padded input reaches `GPULineSplit`, `GPULineNoding`, `GPUNetworkNoding`, `GPULineLoopRemoval`, `GPUPolygonize` without `snapRounding`, or `GPUMakeValid` without `snapRounding` (defects 1 and 2).
- **`ringOffsets[0] > 0`:** this is tested only for the direct noder (`leading = 2`). Not for make-valid, overlay `combine-positions`, coverage (`getSnapOrigin` and the `snap-range` lower bound), or line clip.
- **Empty geometry:** no all-empty overlay (both operands with zero vertices), and no single-empty-ring operand on the right side only.
- **Coverage simplification:** the coverage padding spec covers dissolve only.
- **Solvers:** no case near the ill-conditioned boundary for the equilibrated ratio or the kriging pivot-ratio change.
- **Row boundaries:** point-cloud specs use small grids and cell centres, so the f32 misassignment in defect 3 (large `H`, points within `H·2^-24` cells below an edge) is never exercised.
- **Walking cost:** a GRASS `model.kind` with parameters packed for Tobler (zero coefficients) is accepted silently. No construction check or spec ties the packed coefficients to `model.kind`.
