# cells-ecology report (wave 1, 2026-11-10)

## 1. Done

### Item 1, FX-17 (polygon covering winding, atan2 audit)

Finding: the GPU result was wrong. A probe kernel on the headless test GPU gave
`atan2(1, +0) = +pi/2` (correct), `atan2(1, -0) = -pi/2` and `atan2(-1, -0) = +pi/2` (wrong: a
`-0.0` second argument flips the half-turn), and `atan2(-0, -1) = +pi` (the zero numerator's sign
is ignored). The winding denominator `dot(first, second) - dot(point, first) * dot(point, second)`
is `-0.0` when the ring vertices carry signed zeros, for example `dot((1,-0,0), (-0,1,-0)) = -0`.
In those cases interior points got winding pi instead of 2 pi and the cover dropped or split cells.

- `modules/analysis/src/gpu-spatial-analysis/cell-topology/s2-analysis-wgsl.ts`: new
  `s2AnalysisAtan2(y, x)` = `select(atan2(y, x), sign(y) * pi / 2, x == 0.0)`, the cap fix's
  approach as a helper, with a comment giving the probe result.
- `modules/analysis/src/gpu-spatial-analysis/cell-topology/gpu-s2-polygon-covering.ts`: the
  winding sum calls `s2AnalysisAtan2(numerator, denominator)`. Both traversals (serial and
  level-synchronous) share it. Results change only where the denominator is exactly zero.
- `modules/analysis/src/gpu-spatial-analysis/cell-indexing/h3-boundary-wgsl.ts`:
  `cellIndexH3BoundaryUnitVectorToLngLat` handles an exactly zero `x` the same way (the second
  atan2 there has a nonnegative length as x and is safe).
- New regression spec
  `modules/analysis/test/gpu-spatial-analysis/global-coordinates/s2-polygon-covering-winding.spec.ts`.
  It covers 8 base rings (an axis wedge, the octant, signed-zero variants of both, a nonconvex
  notched ring, the face-0 boundary through cube corners, a vertex on a face centre, a wedge with a
  hole) × 6 rotations (signed axis permutations with determinant +1, exact in float32) × exterior
  and interior × both refinements, at level 4. Each run is compared with a binary64 CPU port of the
  traversal (cell IDs in order, required count, visited count, status). The spec also asserts that
  the configurations really produce zero denominators: +0 with numerators of both signs, and -0
  with a positive numerator. With the old `atan2` call it fails at
  `axis-wedge-negative-zero-0-exterior serial: expected 199 to be 197`. With the fix it passes.
  Tiny-numerator-and-denominator cases (polygon vertices at cube corners and face centres) agree
  with the reference.

Other atan2 calls in my directories, checked:
- `cell-topology/s2-analysis-wgsl.ts`: line 21 (`s2AnalysisAngle`, already guarded by the cap fix);
  line 24 (triangle excess, denominator `1 + ...` is positive for cell triangles); lines 46–47
  (stable measures, positive denominators). No change.
- `cell-indexing/gpu-cell-geometry.ts:392/394`: already guarded (explicit `x == 0` branch; the
  second x is a nonnegative length).
- `cell-indexing/h3-measures-wgsl.ts:12,134,140,164`: denominators are positive for H3 cells. No
  change.

### Item 2, EC03-R (raster resistance adapter)

- New `modules/analysis/src/gpu-spatial-analysis/landscape-ecology/raster-resistance-connectivity.ts`:
  - `addRasterResistanceConnectivityRecipe`. A `RecipeBuilder` recipe; outputs are caller-owned,
    and it returns a `GPURecipeResult` with stages `resistance` (`invalidCount`) and `solve`
    (`converged`, `iterationCount`, `iterationLimitReached`, `invalidCount`), plus
    `estimatedCommandNodeCount` and `provenance`.
  - Input is a published `GPURasterBand`. Scale, offset, no-data and texture storage are resolved
    by `getRasterBufferBandNodes`, whose 'reuse-canonical' path is the published
    `GPURasterNeighborhood`.
  - Edges are built on the GPU: one kernel invocation per edge slot, with a fixed layout
    `cell * slotsPerCell + direction` (E, S for 4 neighbours; E, S, SE, SW for 8; default 8).
    Off-grid slots are zero-conductance self edges. Conductance is `2 / (R(a)+R(b)) / edgeLength`.
  - Edge length per `cellSizeMode`: in uniform mode it comes from the affine (exact for sheared or
    rotated grids). In web-mercator and geographic modes it uses the WGS84 ground cell size at the
    edge's midpoint row, and the grid must be axis-aligned.
  - The affine and georeference are read from a per-frame `parameters` view filled by
    `getRasterResistanceConnectivityParameterValues(grid)`.
  - The invalid-cell count uses `addRecipeCountNodes` (published `GPUReduction`). The sparse
    recipe's nodes are captured with `captureGraphCommandNodes` and added as builder stage
    `solve`, followed by a `solve-status` glue kernel.
  - No-data (zero validity, the no-data value, or a non-finite value including +Infinity) removes
    the cell, which then has NaN potential and NaN current.
  - Zero or negative resistance with data is invalid. Such cells are counted, the edges they share
    with data cells get NaN conductance, and the sparse solve then reports invalidEdges and
    publishes nothing.
  - Limits are checked once at construction, by name: ≤ 1,048,576 cells, ≤ 2,097,152 edge slots
    (524,288 cells with 8 neighbours), plus `maximumIterations`, `connectivityIterations`,
    `tolerance`, source/sink, neighbourhood, cell size mode with axis alignment, band, validity and
    parameters. The command-node budget is the sparse estimate plus an allowance of 16.
- `landscape-ecology/sparse-resistance-connectivity.ts`: additive. The limits are now exported
  constants (`SPARSE_RESISTANCE_CONNECTIVITY_MAXIMUM_VERTEX_COUNT`, `..._EDGE_COUNT`,
  `..._ITERATIONS`, `..._CONNECTIVITY_ITERATIONS`); internal aliases are kept and behaviour is
  unchanged.
- `landscape-ecology/index.ts`: exports the recipe, its packer, its constants and its types.
- Specs:
  - `modules/analysis/test/gpu-spatial-analysis/urban-ecology/raster-resistance-connectivity.spec.ts`
    (headless, 9 tests). Each grid is compared with a binary64 dense circuit solve: uniform R with
    4 and 8 neighbours; a no-data wall with a gap on a 2:1 anisotropic grid, 4/8; no-data holes
    including NaN and +Inf on a sheared affine, 4/8; and a geographic 8-neighbour grid. Every
    conductance slot, potential, node current, the effective resistance and the energy are
    checked. The gap cell carries the whole current (≈1). A closed wall gives
    `[0,0,0,0]` and all-NaN output. Zero and negative R give 2 invalid cells, 8 invalid edges and
    a blocked solve.
  - `.../urban-ecology/raster-resistance-connectivity.node.spec.ts` (4 tests): named-limit
    rejections (message starts with the ID and names the prop and limit); the largest accepted
    grids; the node-count estimate is at least the real node count at 4×4 and at the maxima; the
    stages; the packer.
- Operator catalog:
  - `ontology/spatiotemporal-review-catalog.ts`: entry for the new recipe in the post-261 list,
    family `coverage-landscape-ecology`. The original 261 stay unchanged because a spec pins them.
  - `ontology/all-entry-points/operator-annotations.ts`: limits for the raster and sparse recipes,
    and `tier: 'reference'` with limits for the dense `addResistanceConnectivityRecipe`.
  - `ontology/coverage-expansion-catalog.ts`: the family summary and limitation text no longer
    say the raster adapter is a follow-up.
- Kernel audit: `test/analysis-core/kernel-audit/cases-coverage-urban-ecology.ts` has 4 cases:
  1024×1024 with 4 neighbours, uniform; 1024×512 with 8 neighbours, geographic, both at 4096
  iterations and 1024 rounds with all optional outputs; and the two over-limit cases.
  `rejections-urban-ecology.ts` has the 2 matching rejections.
- Docs:
  - `docs/api/gpu-spatial-analysis/landscape-ecology.md`: new section
    "`addRasterResistanceConnectivityRecipe` — EC03-R" (formula, props, edge-length rule, slot
    layout, no-data/zero/negative rules, outputs and stages, limits, validation). The output table
    row and the limits sentence are updated, and the stale "follow-up" sentence is gone.
  - `docs/api/gpu-spatial-analysis/urban-ecology-lenses.md`: sections with marker and
    result/graph directives for the sparse and raster recipes. The sparse "pending" text is
    replaced.
- Lenses:
  - `modules/lenses/src/gpu-spatial-analysis/urban-morphology/urban-ecology-operators.ts`: pack
    operators with curated TeX formulas for `addSparseResistanceConnectivityRecipe` and
    `addRasterResistanceConnectivityRecipe`.
  - New example files `sparse-resistance-connectivity-recipe.example.ts` and
    `raster-resistance-connectivity-recipe.example.ts`.
  - Scenario cases in `urban-ecology-examples.ts`: a 3×3 lattice with a chord; and a 6×5 raster
    with a no-data wall, a gap and a high-R cell, 8 neighbours.
  - Registrations in `urban-ecology.lens.ts`: `gpu-spatial-analysis.sparse-resistance-connectivity-recipe`
    and `gpu-spatial-analysis.raster-resistance-connectivity-recipe`.
  - `modules/lenses/test/computation-docs/lens-coverage.node.spec.ts`: the EC03 pending entry is
    removed (only that).
  - No record added: by default the coverage spec does not require records
    (`LENS_COVERAGE_REQUIRE_RECORDS`). Nothing under `docs/lens-records/**` was touched.

## 2. Verified

All commands were run from the repository root.
- `npx vitest run --project headless .../global-coordinates/s2-polygon-covering-winding.spec.ts`:
  1/1 pass. With the call temporarily reverted to plain `atan2` it fails (above), and with the fix
  restored it passes again.
- A temporary headless probe kernel for atan2 signed-zero behaviour, deleted afterwards. Its
  results are quoted above.
- `npx vitest run --project headless` on s2-polygon-covering-winding, s2-covering-classification
  (recorded fixtures), s2-covering-level-synchronous, s2-cap-covering-level-synchronous and
  spherical-polygon-contains-rings: 5 files, 12/12 pass.
- `npx vitest run --project headless .../global-coordinates/global-coordinates.spec.ts modules/lenses/test/computation-lenses/urban-ecology.spec.ts`:
  2 files, 36/36 pass.
- `npx vitest run --project headless` on cell-indexing h3-boundary-wgsl, gpu-cell-geometry,
  gpu-cell-measures and cell-cover gpu-cell-cover: 4 files, 28/28 pass.
- `npx vitest run --project headless` on urban-ecology raster-resistance-connectivity,
  sparse-resistance-connectivity and urban-ecology-outputs: 3 files, 31/31 pass.
- `npx vitest run --project node .../global-coordinates/ .../urban-ecology/`: 9 files, 26/26 pass.
- `npx vitest run --project node .../analysis-core/kernel-audit-coverage-urban-ecology.node.spec.ts .../analysis-core/kernel-audit.node.spec.ts`:
  34/34 pass, including the 4 new raster cases and the rejections.
- `npx vitest run --project node modules/analysis/test/gpu-spatial-analysis/ontology/ modules/lenses/test/computation-docs/lens-coverage.node.spec.ts`:
  5 files, 50/50 pass.
- `npx vitest run --project node modules/lenses/test/computation-lenses/urban-ecology.node.spec.ts modules/lenses/test/computation-docs/lens-coverage.node.spec.ts`:
  31/31 pass. The node spec builds both new scenarios within core binding limits.
- Both new lens scenarios were run end to end on the headless GPU with a temporary spec (deleted
  afterwards).
  - Sparse: potentials `1.385,…,0`; status `[1,4,1,0]`.
  - Raster: converged in 17 iterations; the gap cell's node current is 1; the wall cells are NaN.
- `node scripts/check-upstream-first.mjs`: no violations.
- `yarn check-contributor-contract --all`: "no new findings". The only finding in my directories
  is the existing baseline `single-invocation-data-loop gpu-s2-polygon-covering.ts:304`.
- `node scripts/typecheck-status.mjs '<my paths>'`: 0 errors (0 total) at cycle 6:58:06 PM. That
  watcher covers `modules/*/src` only.
- `npx biome check --write` on every file I touched.

Not run:
- The test typecheck (`typecheck-tests` is forbidden here), so the new spec files have not been
  typechecked.
- The full suites, `yarn lint`, the build and `verify-package-exports`.
- The lens-coverage spec with `LENS_COVERAGE_REQUIRE_RECORDS=1`.

## 3. Not done

- Item 1: nothing outstanding in my directories. The other directories' atan2 calls are listed in
  section 6 and were not touched.
- Item 2: no lens records for the two new scenarios. They are not required by default; producing
  them needs the main session's capture run. I did not add the two elements to the explicit list
  in `modules/lenses/test/computation-lenses/urban-ecology.spec.ts`, because that file belongs to
  the records stream (see wave 2 requests).

## 4. Wave 2 requests

- Renames: none.
- `modules/lenses/test/computation-lenses/urban-ecology.spec.ts` (records stream): add
  `addSparseResistanceConnectivityRecipe` and `addRasterResistanceConnectivityRecipe` to
  `elements`, with assertions.
  - Sparse: `status` `[1, n, 1, 0]`, and `output-vertexCurrents` is 1 at vertices 0 and 8.
  - Raster: `output-solverStatus[0] = 1`, `output-vertexCurrents[21] ≈ 1` (the gap), and
    `output-potentials` NaN at cells 3, 9, 15, 27.
- `modules/lenses/test/computation-docs/lens-coverage.node.spec.ts`: two EC03 workarounds are now
  obsolete and can be deleted:
  - the `/^gpu-spatial-analysis\.(?:sparse|raster)-resistance-connectivity-recipe$/` clause in
    `isDroppedScenario`;
  - the `addRasterResistanceConnectivityRecipe` marker exemption in `collectSlotFailures`.
- Lens records: capture `docs/lens-records/gpu-spatial-analysis.sparse-resistance-connectivity-recipe.json`
  and `...raster-resistance-connectivity-recipe.json` in the main session's capture run.
- Website (website streams): the new recipe is not in `website/src/catalog/generated/inventory.json`
  or the catalog entries yet. It will show up after the dist rebuild and inventory regeneration.
- `modules/analysis/src/gpu-spatial-analysis/global-coordinates/gpu-spherical-polygon-contains.ts:271`
  (global-coordinates owner) uses the identical winding formula and needs the same guard, with a
  spec like mine.
- The terrain aspect calls `atan2(-eastGradient, -northGradient)` (see section 6) get a `-0.0` x
  whenever the north gradient is +0. On this GPU that reports the aspect of a pure east or west
  slope with the wrong sign. The terrain stream should verify and fix this.

## 5. Roadmap

- FX-17 (section 2.16, and the GPU-05/FX-17 summary row): replace with "Fixed in the S2 cap and
  polygon covering and in the H3 boundary longitude (`s2AnalysisAtan2`; the cause is
  `atan2(y, -0.0)` returning `-sign(y) pi/2` on the test GPU), with regression spec
  `s2-polygon-covering-winding.spec.ts`. Remaining: the atan2 calls in other directories listed in
  the cells-ecology report, chiefly `gpu-spherical-polygon-contains.ts:271`, the terrain aspect
  (`gpu-terrain-derivatives.ts:389`, `gpu-relief-shading.ts:535/704`) and
  `geodesic-kernels.ts:212`."
- EC03-R (section 2.10 and the summary row): replace with "`addRasterResistanceConnectivityRecipe`
  implemented (4/8 neighbours, `cellSizeMode`, at most 1,048,576 cells and 2,097,152 edge slots).
  The lens pack, formula and scenarios exist for the sparse and raster recipes. Remaining: lens
  records for both scenarios, the headless lens assertions in `urban-ecology.spec.ts`, and
  deleting the two obsolete EC03 workarounds in `lens-coverage.node.spec.ts`. Circuitscape
  multi-pair modes (pairwise, all-to-one, one-to-all) stay in GAP-05; merging the dense recipe
  stays DUP-01."

## 6. Noticed

The atan2 audit covers every WGSL `atan2(` in `modules/analysis/src` outside my directories.
"Zero x" means the second argument can be exactly zero. On this GPU only `-0.0` is wrong
(`+0.0` is correct), so the risk is highest where x is a negation or a product that can produce
`-0.0`.

| File:line | Can x be 0? | Owner |
| --- | --- | --- |
| gpu-terrain/terrain-analysis/gpu-terrain-derivatives.ts:389 `atan2(-eastGradient, -northGradient)` | yes, as -0 whenever northGradient = +0 (high risk) | gpu-terrain |
| gpu-terrain/terrain-illumination/gpu-relief-shading.ts:535, 704 (same expression) | yes, as -0 (high risk) | gpu-terrain |
| gpu-terrain/hydrology/terrain-flow-routing.ts:130, 131 | yes (flat facet) | gpu-terrain |
| gpu-terrain/terrain-illumination/gpu-solar-position.ts:248 | yes (rare geometry) | gpu-terrain |
| gpu-terrain/point-horizon/point-horizon-march.ts:657 (builtin path for x <= 0) | yes | gpu-terrain |
| raster-analysis/change-detection/gpu-change-detection.ts:388 | yes | raster-analysis |
| analysis-core/kernels/study-window-wgsl.ts:267, 291, 292, 293, 342, 386 | yes (dot, ratio, fan.x) | analysis-core |
| gpu-network/network-line-graph/gpu-network-line-graph.ts:782 `atan2(cross, projection)` | yes (right-angle turns) | gpu-network |
| dataframe-analysis/pair-statistics/gpu-ripley.ts:318, space-time-k-function-wgsl.ts:39 | yes (h = 0 on the boundary) | dataframe-analysis |
| dataframe-analysis/pair-statistics/gpu-variogram.ts:414 | yes | dataframe-analysis |
| gpu-spatial-analysis/global-coordinates/gpu-spherical-polygon-contains.ts:271 (same winding as the covering) | yes, -0 as in the covering (high risk) | global-coordinates |
| gpu-spatial-analysis/global-coordinates/gpu-spherical-convex-intersection.ts:208 | no (positive for valid triangles) | global-coordinates |
| gpu-spatial-analysis/global-coordinates/gpu-spherical-convex-intersection.ts:259, 262 | yes | global-coordinates |
| gpu-spatial-analysis/global-coordinates/gpu-spherical-arc-overlay.ts:212, 296 | yes (90° arcs, right-angle turns) | global-coordinates |
| gpu-spatial-analysis/linear-referencing/linear-referencing-kernels.ts:339, 522, 943 | yes | linear-referencing |
| gpu-spatial-analysis/linear-referencing/linear-referencing-kernels.ts:525 | no (nonnegative length) | linear-referencing |
| gpu-spatial-analysis/linear-referencing/gpu-line-locate.ts:355 | yes (only (0,0) guarded) | linear-referencing |
| gpu-spatial-analysis/trajectory-analysis/trajectory-metrics-kernels.ts:416, 422 | yes | trajectory-analysis |
| gpu-spatial-analysis/trajectory-interpolation/trajectory-interpolation-kernels.ts:75, 99 | yes (only (0,0) guarded) | trajectory-interpolation |
| gpu-spatial-analysis/spatial-interpolation/gpu-natural-neighbor.ts:286, 287 | yes | spatial-interpolation |
| gpu-spatial-analysis/geographic-distribution/gpu-rose-statistic.ts:323, 377 | yes | geographic-distribution |
| gpu-spatial-analysis/geographic-distribution/gpu-geographic-distribution.ts:573, 686 | yes | geographic-distribution |
| gpu-spatial-analysis/grid-generators/gpu-shape-generator.ts:469, 501 | yes | grid-generators |
| gpu-spatial-analysis/urban-morphology/isovists.ts:170 | yes | urban-morphology |
| gpu-spatial-analysis/shape-descriptors/shape-descriptors-kernels.ts:400 | yes (only both-zero guarded) | shape-descriptors |
| gpu-spatial-analysis/geometry-measures/geodesic-kernels.ts:212 `atan2(sinAlpha, -northTerm)` | yes, as -0 (high risk) | geometry-measures |
| gpu-spatial-analysis/geometry-measures/geodesic-kernels.ts:137, 159, 161, 174, 200, 204 | yes | geometry-measures |
| gpu-spatial-analysis/geometry-measures/geodesic-wgsl.ts:116, 147, 165 (only (0,0) guarded), 185 | yes | geometry-measures |
| gpu-spatial-analysis/geometry-measures/geodesic-wgsl.ts:117 | no (length) | geometry-measures |
| gpu-spatial-analysis/geometry-measures/rhumb-wgsl.ts:110 `atan2(deltaLambda, deltaPsi)` | yes (east-west rhumb lines) | geometry-measures |
| gpu-spatial-analysis/geometry-measures/geometry-measures-kernels.ts:116, 117 | possibly (cos(phi) at the poles) | geometry-measures |
| gpu-spatial-analysis/geometry-measures/geometry-measures-kernels.ts:1031 | yes | geometry-measures |
| gpu-spatial-analysis/geometry-measures/geometry-measures-kernels.ts:139, 1032 | no (sqrt or length) | geometry-measures |
| pointcloud-analysis/gpu-pointcloud-features.ts:125, gpu-pointcloud-icp.ts:364 | yes | pointcloud-analysis |
| geospatial-analysis/gpu-point-buffer.ts:134 | yes | geospatial-analysis |
| geospatial-analysis/gpu-point-buffer.ts:121 | no (length) | geospatial-analysis |
| gpu-spatial-analysis/minimum-bounds/gpu-minimum-bounds.ts:489 | yes | minimum-bounds |

Other notes:
- A single shared helper in `analysis-core/kernels` (an `atan2` that is safe for signed zeros)
  would be the contract-rule-11 home for this guard. Mine lives in `cell-topology` because that is
  where I am allowed to edit.
- `analysis-operator-catalog.node.spec.ts` pins the original operator list at 261. New
  `gpu-spatial-analysis` operators must go in the post-261 list
  (`spatiotemporal-review-catalog.ts`), not into a capability's `operators`. That file's name no
  longer describes its contents.
- The dataset label of `urban-ecology-illustrative/2` says "synthetic conductance networks and
  aligned population counts". It now also covers a synthetic resistance raster. I left it as it
  is so the existing records keep their dataset text.
