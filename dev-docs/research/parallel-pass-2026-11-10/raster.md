# Stream `raster` report (wave 1, 2026-11-10)

Paths are relative to `modules/analysis/src/raster-analysis` unless they start with `modules/`, `docs/` or `test/` (`test/` = `modules/analysis/test`).

## 0. Inventory (item 1)

Georeferencing ways (review C3): **1** per-frame packer from `grid.affine` (float64 `localOrigin` where world coordinates are read); **2** `grid.affine` baked into WGSL; **3** world `extent` in the packer settings, only width and height from the grid (can disagree with `grid.affine`); **4** pixel units only, affine ignored; **5** own grid type; **6** nothing (`cellCount`, `width`/`height`). `+csm` = honours `cellSizeMode`.

No-data input conventions (review C2): **a** published `GPURasterBand` (band `validity`, raw `noDataValue` sentinel, non-finite rejected); **b** NaN plus optional finite `noDataValue`, no validity view; **c** NaN plus optional input `validity` view (now `mask` where marked); **d** label sentinels (`0xfffffffe`, `GPU_RASTER_PATCH_UNRESOLVED`, `GPU_COST_DISTANCE_UNRESOLVED`); **-** no raster values read. Output: **V** NaN plus `output.validity`; **OV** NaN plus `outputValidity`; **OM** `outputMask`; **SC** NaN plus a per-cell status code; **N** NaN only; **-** none.

Status: **named** = only `GPUIterationStatusPort` / cardinality / overflow names; **adhoc** = a word or per-cell code with family-specific meaning.

| Class | File | Georef | No-data in | No-data out | Status (after this pass) |
| --- | --- | --- | --- | --- | --- |
| `GPURasterWalkingCost` | gpu-raster-walking-cost.ts | **1 +csm** (was 2) | a | V | named: `converged`, `iterationLimitReached`, `iterationCount` (port-typed, validated) |
| `GPUCostDistance` | cost-distance/gpu-cost-distance.ts | 1 +csm | a | N (`+Inf`, NaN unresolved) | named: `converged`, `iterationCount`, **`iterationLimitReached` (new)** |
| `GPUCostDistancePath` | cost-distance/gpu-cost-distance-path.ts | 1 | d (back-links) | - | named (`iterationLimitReached`, `overflow`, `requiredCount`) |
| `GPUDistanceField` | distance-field/gpu-distance-field.ts | 1 (+csm via friction note only) | `seedMask` (mask-like, contract name differs) | - | none |
| `GPURasterZonalStatistics` | raster-zonal-statistics/ | 1 +csm | a + `zoneMask`/`valueMask` | V | named `overflow` |
| `GPURasterSampling`, `GPURasterProfile` | raster-sampling/ | 1 (packer, `localOrigin`) | c, **`mask` (new)** + `noDataValue` | OV / N | cardinality + overflow (profile) |
| `GPUPolygonRasterization`, `GPURasterJoin` | polygon-rasterization/ | 1 + `spatialContext` | - | - | named (count, overflow) |
| `GPURasterExtremaPyramid` | raster-pyramid/ | 1 (size only) | a | V | none |
| `GPURasterReprojection` | gpu-raster-reprojection.ts | 2 (two grids + `CompiledProjection`) | a | V | none |
| `GPURasterTileReprojection` | gpu-raster-tile-reprojection.ts | 2 | a | V | none |
| `GPURasterFractionalCoverage` | gpu-raster-fractional-coverage.ts | **1** (was 2; optional `parameters`) | - | zone `validity` | named (count, requiredCount, overflow) |
| `GPURasterFractionalZonalStatistics` | same file | 2 (cell area baked) | a | V | per-zone validity |
| `GPURasterFractionalCategoricalStatistics` | gpu-raster-fractional-categorical-statistics.ts | 2 (cell area, CPU-checked determinant) | a | V | overflow, requiredCount |
| `GPURasterFractionalZonalDistribution` | gpu-raster-fractional-zonal-distribution.ts | 1 (probabilities) / 2 (area) | a | V + `quantileValidity` | adhoc `status` word |
| `GPUIsolines`, `GPUIsobands`, `GPUIsobandRings` | isolines/ | 3 | c, **`mask` (new)** + `noDataValue` | - | named (count, requiredCount, overflow, polylineOverflow) |
| `GPULineIntegralConvolution` | flow-texture/ | 3 | N | N | none |
| `GPUStreamlines` | flow-texture/ | 3 | N | N | named (converged, iterationLimitReached, overflow, requiredCount) |
| `GPUParticleAdvection` | particle-advection/ | 3 | N | N | adhoc |
| `GPURasterStretch` | raster-stretch/ | 4 | c, **`mask` (new)** + `noDataValue` + `regionMask` | N | statistics row |
| `GPURasterRankFilter` | gpu-raster-rank-filter.ts | 4 | a | V + `changed` | none |
| `GPURasterFillNoData` | gpu-raster-fill-nodata.ts | 4 | a | V + `filled` | adhoc `unresolvedCount` |
| `GPURasterReconstruction`, `GPURasterFillHoles` | gpu-raster-reconstruction.ts | 4 | raw float `mask` (a value bound, not a filter) / d | V / d | named (port-typed, validated) |
| `GPURasterSkeleton`, `GPURasterMedialAxis` | gpu-raster-skeleton.ts, raster-medial-axis.ts | 4 | d (`0xfffffffe`) | d | skeleton named (port-typed, validated); medial axis none |
| `GPURasterWatershedSegmentation` | gpu-raster-watershed-segmentation.ts, raster-watershed-forest.ts | 4 | a + d | d | named: `converged`, **`iterationLimitReached` (new)** |
| `GPURasterFocalPairs` | focal-pairs/ | 4 | a | per-cell bits | adhoc bits (documented) |
| `GPURasterTexture` | remote-sensing/gpu-raster-texture.ts | 4 | c (`validity`) | N | none |
| `GPURasterRegistration` | remote-sensing/gpu-raster-registration.ts | 4 (shift in pixels) | c (`referenceValidity`, `movingValidity`) | - | adhoc code (2 = ambiguous peak) |
| `GPURasterRegionGraph` | remote-sensing/gpu-raster-region-graph.ts | 4 | c / N | - | adhoc "pending" word (is `iterationLimitReached`) |
| `GPURasterPatchMetrics`, `GPURasterSieve` | raster-patches/ | 2 (metrics: affine baked for area and perimeter) / 4 (sieve) | d + `validity` on labels | d | `converged`, `overflow` |
| `GPURasterConservativeRemapping` | remote-sensing/gpu-raster-conservative-remapping.ts | 5 | c | SC | adhoc per-cell code |
| `GPURasterArithmetic`, `GPURasterCellStatistics`, `GPUWeightedOverlay` | raster-algebra/ | 6 | b | N / N / OV | none |
| `GPURasterConditional` | raster-algebra/ | 6 | b + `mask` | OM | none |
| `GPURasterReclassify` | raster-algebra/ | 6 | b + c, **`mask` (new)** | N | class counts |
| `GPURasterNDWI`, `NBR`, `EVI`, `SAVI` | gpu-raster-spectral-indices.ts | 6 (width, height) | a | OV | none |
| `GPUChangeDetection` | change-detection/ | 6 | NaN + `mask` (already contract name) | N | none |
| `GPURasterTemporal`, `GPURasterSpectralUnmixing`, `GPURasterSeasonalBreak` | remote-sensing/ | 6 | c | SC | adhoc per-cell code; unmixing `iterationCount` |
| 20 `raster-time-series` classes | raster-time-series/ | 6 | c (`validity`) | N / SC | adhoc per-cell `status` words (protected this wave) |
| 11 `raster-embeddings` classes | raster-embeddings/ | 6 | c (`validity`, `beforeValidity`, `afterValidity`) | N | adhoc (protected) |
| `GPULagrangianFlowMap`, `GPUFiniteTimeLyapunovExponent` | lagrangian-diagnostics/ | 3 | N | N | none (protected) |

Totals after this pass: way 1 (13 classes), way 2 (7), way 3 (8), way 4 (15), way 5 (1), way 6 (about 40, most in protected directories).

## 1. Done

- **STD-06 walking cost (item 2).** `gpu-raster-walking-cost.ts`: new `getGPURasterWalkingCostParameterValues(settings, grid)`, `GPU_RASTER_WALKING_COST_PARAMETER_LENGTH = 16`, `GPURasterWalkingCostSettings`, `GPURasterWalkingCostModel`. New optional props `parameters` (affine linear terms, row edges, `frictionWeight`, GRASS coefficients and threshold are read from it; a changed georeference does not recompile) and `cellSizeMode` (`'uniform'` default; `'web-mercator'`/`'geographic'` scale each step on WGS84 at the latitude between the two cell centres, axis-aligned grids only). Without `parameters` (deprecated) the existing `prepare` kernel packs the same values once, so every relaxation round has the same WGSL and no node is added (the 26,196-iteration budget is unchanged). Class TSDoc lists per-frame and compile-time props and the no-data rule. This also closes the review's SUSPECTED misuse (degrees of run on geographic grids), now opt-in correct.
- **STD-06 one path (item 3).** Target named in the table above and in docs: published `GPURasterBand` + `grid` from `analysis-core`, `cellSizeMode` where ground distance/slope/area is computed, affine from a `get...ParameterValues` packer. Moved additively: walking cost (above) and `GPURasterFractionalCoverage` (12 consumer files, the most-called baked-affine class): optional `parameters`, `getGPURasterFractionalCoverageParameterValues(grid)`, `GPU_RASTER_FRACTIONAL_COVERAGE_PARAMETER_LENGTH = 8`; without it a one-thread pack node writes the same values.
- **Correctness fix found on the way (changes results only where the old code failed to compile).** The baked WGSL in `GPURasterFractionalCoverage` and `GPURasterReprojection` printed numbers raw after a minus sign, so a negative origin or rotation term (any western- or southern-hemisphere grid) produced `x--120`, which WGSL rejects as the decrement token (checked with `wgsl_reflect`). Coverage now reads parameters; reprojection parenthesizes every number (`gpu-raster-reprojection.ts`, comment at the determinant). Spec: `test/geospatial-expansion/raster-georeference-parameters.spec.ts` (negative origins, legacy vs parameters identical, affine rewrite without recompiling). I did not run the spec against the old source.
- **STD-07 (item 4).** New `resolveRasterInputMask` and `getRasterInputMaskName` (`raster-band-utils.ts`, `@internal`). `mask` (contract input filter) added, `validity` kept as a `@deprecated` alias (both given throws `<id> mask and its deprecated alias validity must not both be given`; messages name whichever prop the caller used) on: `GPUIsolines`, `GPUIsobands`, `GPUIsobandRings` (forwards `mask`), `GPURasterSampling`, `GPURasterProfile`, `GPURasterStretch`, `GPURasterReclassify`. All of these also keep `noDataValue`, so NaN is not the only no-data marker. Walking cost and fractional coverage already take the published band.
- **STD-04 (item 5).** `RasterIterationStatusOutputs` is now `GPUIterationStatusPort & {converged}` (`raster-expansion-utils.ts`); walking cost, reconstruction, fill holes and skeleton call `validateGPUStatusPort`. `GPUCostDistance` gains optional `output.status.iterationLimitReached` (both the plain and the back-link finalize paths); `createRasterIterationFinalizeNode` (`cost-distance/raster-relaxation.ts`) takes it optionally (terrain callers unaffected). `GPURasterWatershedSegmentation` gains optional `iterationLimitReached` (both methods, one extra one-thread node only when requested) and an exported `GPURasterWatershedSegmentationStatus` type.
- **STD-03 (item 6).** The bare `assert` in `cost-distance/raster-relaxation.ts` (gate slot table) is `validateRequirement` with message `<operation> gate slot must be an integer in [0, n] (the group size)`; `@luma.gl/core` `assert` import removed.
- Specs: new `test/geospatial-expansion/raster-walking-cost-parameters.spec.ts` (legacy vs parameters bit-identical for Tobler and GRASS; float64 Bellman-Ford oracle; anisotropic affine plus friction weight rewritten on the same compiled graph; geographic `cellSizeMode` against `getRasterGroundScale`; rejection messages), `test/geospatial-expansion/raster-iteration-status.spec.ts` (watershed), `test/gpu-raster/cost-distance/gpu-cost-distance-status.spec.ts`, `test/gpu-raster/isolines/raster-input-mask.node.spec.ts`, plus appended cases in `test/gpu-raster/raster-sampling/raster-sampling.node.spec.ts`, `test/gpu-raster/raster-stretch/raster-stretch.node.spec.ts`, `test/gpu-raster/cost-distance/raster-relaxation-continuation.node.spec.ts`. Kernel-audit cases added in `test/analysis-core/kernel-audit/cases-expansion.ts` (walking cost with parameters in three cell size modes; fractional coverage with parameters and a negative-origin grid).
- Docs: `docs/api/gpu-raster/expansion.md` (C21 row, new "`GPURasterWalkingCost` georeference" section, fractional coverage paragraph, and the two generated parameter subsections via `UPDATE_DOC_PARAMETERS=1`; only this page was stale when I regenerated).

## 2. Verified

All with `npx vitest run`, my specs only:
- headless `raster-walking-cost-parameters.spec.ts` 3/3; `raster-iteration-status.spec.ts` 1/1; `raster-georeference-parameters.spec.ts` 2/2; `gpu-cost-distance-status.spec.ts` 1/1; `raster-terrain-expansion.spec.ts` 17/17 (rerun after all edits: 5 files 24/24); `gpu-cost-distance.spec.ts` + `gpu-cost-distance-continuation.spec.ts` 27/27; isolines, isobands, isoband rings, sampling, profile, stretch, reclassify headless specs 7 files 31/31.
- node: `raster-terrain-expansion.node.spec.ts` 3/3; `raster-relaxation-continuation.node.spec.ts` + `gpu-cost-distance.node.spec.ts` 19/19; `raster-sampling.node.spec.ts`, `raster-stretch.node.spec.ts`, `gpu-raster-profile.node.spec.ts` 22/22; `raster-input-mask.node.spec.ts` 2/2; isolines, isobands, raster-algebra, local-origin-precision node specs 26/26; `kernel-audit-expansion.node.spec.ts` 102 then 103/103; `kernel-audit-raster.node.spec.ts` 160/160; `kernel-audit-coverage-raster.node.spec.ts` 37/37; `raster-gate-pipelines`, `gpu-terrain-flow-flats.node`, `raster-terrain-expansion.node`, `kernel-audit-expansion` 118/118; `doc-parameters.node.spec.ts` 1/1 after regeneration.
- `node scripts/typecheck-status.mjs 'raster-analysis|geospatial-expansion|gpu-raster/|cases-expansion'`: 0 matching errors (0 total), cycle finished 6:49:40 PM, after all edits.
- `yarn check-contributor-contract`: no new findings. With `--all`, the `bare-assert` row for `raster-relaxation.ts` is gone; the raster rows left are root-file, multiple-classes, internal-export and single-invocation-data-loop, all from before this pass.
- `node scripts/check-upstream-first.mjs`: no violations.
- `npx biome check --write` on every file I touched.
- Not run: the parity specs (`*.parity.spec.ts`), terrain headless specs that use `raster-relaxation.ts` (the change there is an optional binding), lenses and website specs.

## 3. Not done

- STD-06 for the remaining baked or extent classes: `GPURasterReprojection` and `GPURasterTileReprojection` (affine still baked, now parenthesized; move both grids to a packer with float64 `localOrigin`); `GPURasterFractionalZonalStatistics`, `GPURasterFractionalCategoricalStatistics`, `GPURasterFractionalZonalDistribution`, `GPURasterPatchMetrics` (cell area baked, no `cellSizeMode` for areas); isolines, isobands, LIC, streamlines, particle advection, Lagrangian (way 3: the packer `extent` can disagree with `grid.affine`; an additive fix is an optional `extent` defaulting to the grid's, left for a session that can touch the protected Lagrangian files); the way-4 classes (rank filter, fill, focal pairs, texture, reconstruction, skeleton, watershed, region graph, registration) are correct in pixels, but none declares it or rejects geographic grids; way-6 classes take no grid at all (raster algebra, spectral indices, remote sensing, time series, embeddings).
- STD-07 remaining in non-protected directories: `GPURasterTexture` (`validity` input; the constructor uses a `readonly props` parameter property, so the alias needs a small constructor rewrite), `GPURasterRegistration` (`referenceValidity`/`movingValidity`), `GPURasterConservativeRemapping`, `GPURasterTemporal`, `GPURasterSpectralUnmixing`, `GPURasterSeasonalBreak` (input `validity`); `GPURasterArithmetic`, `GPURasterCellStatistics`, `GPUWeightedOverlay` (no row filter at all; NaN or the sentinel only); outputs `outputValidity` (spectral indices, sampling, overlay) and `outputMask` (conditional) are not yet `validity`; `GPUDistanceField.seedMask`. Protected this wave: every `validity` input in `raster-time-series`, `raster-embeddings`, Lagrangian.
- STD-04 remaining: `GPURasterFillNoData.unresolvedCount` (not a port name; not iterative), `GPURasterRegionGraph` "pending" word (should be `iterationLimitReached`), per-cell codes in registration, temporal, unmixing, seasonal break and remapping (same numbers mean different things), `GPURasterFractionalZonalDistribution.status`, the watershed forest's iteration count (`iterationCount` not published).
- Iteration-budget items from the review (G1, G2) were out of scope: walking cost still does whole-grid Jacobi rounds.

## 4. Wave 2 requests

Renames to complete (remove the alias):
- `validity` -> `mask` on `GPUIsolines`, `GPUIsobands`, `GPUIsobandRings`, `GPURasterSampling`, `GPURasterProfile`, `GPURasterStretch`, `GPURasterReclassify`. Consumers passing `validity:` to these classes, by `git grep`: `examples/deck/spatial-analysis-explorer/modes/contours-mode.ts`, `.../suitability-mode.ts`, `.../terrain-features-mode.ts`, `examples/deck/spatial-analysis-showcase/src/scenes/raster/burn-severity.compute.ts`, `.../spectral-texture-display.ts`; lenses and website files that name the classes (lenses scenarios under `modules/lenses/src/computation-lenses/scenarios/gpu-raster/`, website run entries under `website/src/gpu/runs/entries/`; 10 to 14 consumer files per class) need a prop-level grep at rename time. `website/src/catalog/generated/inventory.json` documents the `validity` props and needs regenerating after `yarn build`.
- Make `parameters` required on `GPURasterWalkingCost` and `GPURasterFractionalCoverage` and drop the compile-time pack path (consumers: `modules/lenses/src/computation-lenses/scenarios/gpu-raster/raster-walking-cost.example.ts`, `expansion-runner.ts`, `website/src/gpu/runs/entries/raster-bands-indices/walking-cost.ts`, `examples/deck/spatial-analysis-showcase/src/scenes/raster/habitat-connectivity.compute.ts`; for coverage, 12 files from `git grep -w GPURasterFractionalCoverage`).
- Lens records `docs/lens-records/gpu-raster.raster-walking-cost*.json` may need regenerating (LN-REC2) once examples pass `parameters`.
- Request to `core-wgsl` (owner of `analysis-core/raster-grid.ts`): add the `resolveGPURasterContext(id, grid, {cellSizeMode, rowDirection})` the review recommends, returning the packed georeference words (`a, b, d, e`, row edges, float64-subtracted `localOrigin`) and validating axis alignment. Walking cost and fractional coverage each pack their own words today; `getRasterGridGeoreference` is used for the edges.
- Operator catalog (`gpu-spatial-analysis/ontology/all-entry-points/entry-point-operator-details.ts`): `GPURasterWalkingCost.tests` should also list `raster-walking-cost-parameters.spec.ts`, and its coordinate model should say "planar meters, or WGS84 ground meters with `cellSizeMode`". I left the catalog untouched because the entry has no coordinate-model field to edit in place.

## 5. Roadmap

- STD-06: "Done 2026-11-10 for raster: `GPURasterWalkingCost` reads its affine, friction weight and GRASS coefficients from `getGPURasterWalkingCostParameterValues` and takes `cellSizeMode`; `GPURasterFractionalCoverage` reads its affine from parameters; negative-origin grids no longer break the reprojection and coverage WGSL. Remaining (raster): reprojection and tile reprojection affines; cell area baked in the three fractional aggregators and patch metrics, with no `cellSizeMode`; packer `extent` independent of `grid.affine` in isolines, isobands and the flow family; pixel-only classes do not declare their coordinate model; `parameters` becomes required in wave 2."
- STD-07: "Raster, 2026-11-10: `mask` (with deprecated `validity`) on isolines, isobands, isoband rings, sampling, profile, stretch, reclassify. Remaining (raster): texture, registration, remapping, temporal, unmixing, seasonal break, arithmetic, cell statistics, overlay; outputs `outputValidity` and `outputMask`; `seedMask`; the time-series and embedding families."
- STD-04: "Raster, 2026-11-10: iteration status port-typed on walking cost, reconstruction, fill holes, skeleton; `iterationLimitReached` on cost distance and watershed. Remaining (raster): fill no-data `unresolvedCount`, region graph pending word, per-cell codes in registration, temporal, unmixing, seasonal break and remapping, fractional distribution status."
- STD-03: remove `cost-distance/raster-relaxation.ts` from the list (no bare assert left in `raster-analysis`).

## 6. Noticed

- `GPURasterReprojection` and `GPURasterTileReprojection` interpolate JavaScript numbers into WGSL with `String(number)`. That is now safe for sign, but values such as `1e+21` depend on WGSL accepting that literal form. `getWGSLFloatLiteral` is the safer helper but rounds to f32 earlier.
- `GPURasterFractionalZonalStatistics` interpolates `cellArea` raw (`covered*${cellArea}`). That works only because the area is positive.
- The review's G12 float32 origin exposure remains in fractional coverage (origin and positions are both f32). A float64 `localOrigin` would need the zones to be origin-relative, which is a breaking change.
- `GPURasterWatershedSegmentation` plateau labels for a symmetric 1 x 6 ramp differ between FIFO and forest (expected and documented). My first spec draft assumed otherwise.
- `docs/api/gpu-raster/operations-analysis.md` "Conventions" still says "the validity output is `outputValidity`", which contradicts contract rule 7.
