# Fix review: callers (2026-11-10)

Scope `callers`: what the math-fix pass does to code that calls `@luma-contributors/analysis`
(`examples/**`, `modules/lenses/**`, `modules/kepler/**`, `website/src/**`, the recorded lens
files), and whether the declared behaviour changes are true and complete at the API surface. The
consolidated list of changes is [behaviour-changes.md](./behaviour-changes.md). Nothing was fixed
or regenerated.

Result in one paragraph: **no call site breaks.** No caller throws at construction and no exported
symbol, required prop, status layout or summary length changed. 87 call sites draw different numbers
on their own default path, and about 700 more construct a contributor whose result changes only for
particular inputs. No entry-point export was removed or renamed. 181 of the 410 recorded lens files
(in each of the two copies) belong to a contributor whose own source changed in code.

## 1. Coverage

Read in full:

- The 18 `## Resolution (2026-11-10)` sections and `README.md` from "Recurring causes" to the end.
- `git diff f5dcdc41 -- 'modules/analysis/src/**/index.ts'` (five barrels changed).
- `modules/analysis/src/raster-analysis/raster-grid/gpu-raster-grid.ts`, and the helper
  `resolveTerrainRowDirection` in `gpu-terrain/terrain-curvature/terrain-geomorphometry-utils.ts`.
- Every constructor call of the seven row-direction terrain classes outside `modules/analysis`
  (86 lines matched, display text included; 42 pass `rowDirection`), with the definition of the
  grid each one receives. In lenses the seven classes all take the Dixie DEM grid; the rows-north
  literal grids of the lens runners (`affine: [1, 0, 0, 0, 1, 0]`) go to raster and hydrology
  classes only.
- Every call of `addHotSpotAnalysisRecipe`, `GPUCostDistance`, `GPURasterReconstruction`,
  `GPUTerrainFlowFractions`, `GPUTerrainBranchingFlow`, `GPUTerrainTiledFlow`, `GPUModalSplit`,
  `GPUTrajectoryOutlierFlags`, `GPUUtilizationVolumeIsopleths`, `GPUSpaceTimeKernelDensity`,
  `GPUDatumGridTransform`, and the kepler bindings for noding, centrality, k-means and probit.

Read by script, one line per call (the props that decide the question, not the whole call):
`borderMode`, `rowDirection`, `grid`, `connectivity`, `distanceFloor`, `fixedPointScale`,
`initialization`, `seed`, `mode`, `tolerance`, `sliceCount`, `intensities`, `estimator` for the
contributors named in section 4.

Checked by script over the whole source diff (279 files): exported names per file before and
after, property lines inside `Props`, `Output` and `Status` types, removed lines that mention a
default, exported constants, and every added `validate*`, `throw` and `assertRecipe`.

Run (the six allowed spec files, all on the headless project, all pass):

| Spec | Tests | Why |
| --- | --- | --- |
| `modules/lenses/test/computation-lenses/statistics-coverage.spec.ts` | pass | asserts every status of every synthetic fit is 0 (probit status 4, pivot status 2, fold status) |
| `modules/lenses/test/computation-lenses/raster-terrain/terrain-processing.spec.ts` | pass | exact values of flow fractions, depression curves, catchment statistics, tiled flow |
| `modules/kepler/test/catalog-gpu.spec.ts` | pass | isochrone bands, noding, buffer |
| `modules/kepler/test/catalog-platform4-geometry-gpu.spec.ts` | pass | seeded k-means partition, minimum bounds, centrality, convex hull |
| `modules/lenses/test/computation-lenses/nested-interpolation-validation.spec.ts` | pass | statuses `[0, 0]` under the new fold assignment |
| `modules/lenses/test/computation-lenses/raster-terrain/coverage-tranche-one.spec.ts` | pass | exact fractional coverage sums after the axis-aligned division change |

The first pair ran 21 tests, the second 33, the third 17.

Not read:

- The call sites of contributors whose change is `result` for particular inputs (section 4.2).
  They are listed by file and line, not classified one by one.
- `.scene.ts` files of the showcase and `website/src/catalog/entries/*.ts` hold code as display
  text. They were searched for stale statements (section 6), not treated as callers.
- `examples/deck/kepler-compute` beyond its terrain, group statistics and column profile calls.
- The other 476 spec files under `modules/lenses/test` and `modules/kepler/test`. Lens specs run
  scenarios through runners, so a grep for a contributor name does not find every spec that
  depends on it.
- Twelve files under `modules/analysis/src` are untracked (`git status` shows `??`), among them
  `analysis-core/kernels/uint64-wgsl.ts`, `binary-search-wgsl.ts` and
  `spatial-regression/pairwise-sum-wgsl.ts`. They have no base version in the snapshot commit, so
  the export comparison does not cover them.
- `website/**` was edited by another session during the pass (45 files in the diff). Those edits
  were not reviewed.

## 2. Call sites that break

None.

| Risk | What was checked | Result |
| --- | --- | --- |
| Terrain `rowDirection` contradicts the grid | All 42 constructor calls that pass `rowDirection` (20 in the explorer, 22 in the showcase, one of them display text). Each receives a grid made by `makeGPURasterMetadata`, whose default is rows south (`affine[4] < 0`). 40 pass `'south'` with such a grid. Two pass `'north'` (`showcase .../hydrology/contours.compute.ts:195`, `.../hydrology/terrain-flow.compute.ts:134`) with `createSouthFirstRasterGrid` (`.../hydrology/b15-common.ts:234`), which builds the grid with `rowDirection: 'north'`. The rows-north grids of the explorer (`flippedGrid` in `contours-mode.ts:233` and `suitability-mode.ts:207`, `pixelGrid` in `terrain-features-mode.ts:171`) go to isolines, distance fields, sampling and rasterization, never to one of the seven classes | consistent everywhere |
| Terrain `rowDirection` left out on a rows-north grid | Lenses (`loadDixieDem` grid, `raster-lens-data.ts:35`), kepler (`analysis3-terrain-gpu.ts:74`), kepler-compute (`raster-terrain-zonal-gpu.ts:206`), website (`support/terrain.ts:37`, `demos/terrain-hydrology.ts:59`) | all rows south: the default is unchanged for them |
| `GPUCostDistance` on a sheared grid | 5 constructor calls and the packer calls | every grid is axis-aligned |
| `addHotSpotAnalysisRecipe` with an unreproducible `selfWeight` | 5 code calls. The showcase passes `statistic: selfWeight === 1 ? 'localGStar' : 'localG'` (`.../weights/hot-spots.compute.ts:1181`); the explorer passes `permutation` with the default `selfWeight`; lenses, kepler and website pass no `permutation` | consistent |
| `GPUSpaceTimeKernelDensity` `sliceCount` | `website/.../space-time-events/space-time-kernel-density.ts:63` passes 1 with an explicit `timeWindow` | valid |
| `GPURasterConservativeRemapping` extent limit, `h3-index-extended` table check, `directionEncoding`, `coordinateExtent` | new props are not passed anywhere; the limits are far from any caller's data | not reachable |
| Compile errors | No exported name was removed from an entry barrel, no prop became required, no union lost a member (`borderMode` gained `'one-sided'`) | none found; a typecheck was not run |

One indirect case to know about, not a break: `modules/kepler/src/analysis/analysis-bindings.ts:108`
passes a caller's props straight to `addHotSpotAnalysisRecipe`, so a kepler host that asks for
`permutation` with `selfWeight` between 0 and 1 now gets the recipe's construction error.

## 3. Declared changes checked at the API surface

Confirmed in source (the default, the throw and its message, or the prop exists as stated):

| Row | Where |
| --- | --- |
| `rowDirection` default from the affine, throw on contradiction, read-only field on seven classes | `terrain-geomorphometry-utils.ts:41`; `readonly rowDirection` in all seven class files |
| `borderMode` default `'one-sided'` in derivatives and curvature; `'clamp'` kept in relief shading and vector ruggedness | `gpu-terrain-derivatives.ts:222`, `:250`; `gpu-terrain-curvature.ts:325`, `:526`; `gpu-relief-shading.ts:457` |
| `GPURasterReconstruction` `connectivity` 8; `GPURasterFillHoles` 4 | `gpu-raster-reconstruction.ts:142`, `:300` |
| `GPUCostDistance` and its packer throw on a sheared grid | `raster-band-utils.ts:89`; message as declared |
| `GPUNeighborSearch` `distanceFloor` 1e-6, schema default, shared with `GPUNearestFeatureWeights` | `neighbor-search-parameters.ts:29`, `:56`, `:219`; `gpu-nearest-feature-weights.ts:240` |
| `GPUDatumGridTransform` default tolerance 5e-9 and the 4.8e-7 floor | `gpu-datum-grid-transform.ts:49`, `:256` |
| `addHotSpotAnalysisRecipe` default statistic and throw | `hot-spot-analysis-recipe.ts:568-578` |
| `GPUNetworkCoarsening` `fixedPointScale` optional, unit from the data when left out | `gpu-network-coarsening.ts:170`, `:327`, `:434` |
| `GPUSpaceTimeKernelDensity` `sliceCount` check | `space-time-kernel-density-math.ts:81` |
| `GPURasterConservativeRemapping` 2^31 limit | `gpu-raster-conservative-remapping.ts:106-111` |
| `h3-index-extended` table check | `h3-index-extended-wgsl.ts:42` (the message ends with one more sentence than the Resolution quotes) |
| `GPUContiguityWeights` `coordinateExtent` | `gpu-contiguity-weights.ts:122`, `:227` |
| `directionEncoding` on seven hydrology classes; unknown name throws | `terrain-flow-tracing.ts:51`; seven `Props` types |
| New optional outputs and props: `irradianceSkyViewFactor`, `localOrigin`, `parameters` (three fractional statistics classes), `validity` (`GPUChangeDetection`, `GPUCellGeometry`, `GPUHuffTradeAreas`), `emptyNeighborhood`, `intervalClosure`, `lagIntervalClosure`, `invalidRowCount`, `zeroWeightCycle`, `fittedScale` | each is an optional member of the `Props` or `Output` type |
| `GPUProbitRegression` status 4 for a fit pinned on the bound | `gpu-probit-regression.ts:481`. Status 4 existed before ("nonbinary or separated"); its meaning is wider, no new code |
| `GPUModalSplit` `0xffffffff`; `GPUUtilizationVolumeIsopleths` volume 1; `GPULineLocate` status `invalid` for NaN | `gpu-modal-split.ts:275`, `:319`; `gpu-utilization-volume-isopleths.ts:324`; `linear-referencing-kernels.ts:853` (the status constant; the test itself is at `:884-894` per the stream) |
| `GPUKMeans` default initialization is `'kmeans++'`, so the seeded change reaches callers that pass nothing | `gpu-kmeans.ts:216` |

Corrections to how two changes are described elsewhere:

- **The trajectory acceleration bit did not move.** `GPU_TRAJECTORY_OUTLIER_FLAGS` is unchanged
  and the kernel still sets `state |= 2u` (`gpu-trajectory-outlier-flags.ts`). What changed is
  which row gets the bit.
- **Probit status 4 is not a new status value.** It is the existing "separated" code, now also
  returned for a maximum beyond the bound, with NaN coefficients.

## 4. Call sites that change

### 4.1 On the caller's own default path (87 call sites)

These draw or record different numbers with no change to the caller. Paths are shortened:
`explorer/` is `examples/deck/spatial-analysis-explorer/modes/`, `showcase/` is
`examples/deck/spatial-analysis-showcase/src/scenes/`, `lenses/` is
`modules/lenses/src/computation-lenses/scenarios/`, `kepler/` is `modules/kepler/src/analysis/`,
`runs/` is `website/src/gpu/runs/entries/`, `demos/` is `website/src/gpu/demos/`.

| Change | Call sites | What the owner should decide |
| --- | --- | --- |
| Border ring of slope, aspect, hillshade (`GPUTerrainDerivatives` without `borderMode`) | explorer: `terrain-mode.ts:153`, `contours-mode.ts:374`, `cost-distance-mode.ts:180`, `drainage-mode.ts:278`, `hydrology-mode.ts:186`, `raster-zonal-mode.ts:148`, `relief-mode.ts:231`, `relief-visualization-mode.ts:517`, `suitability-mode.ts:363`, `terrain-features-mode.ts:380`, `visibility-mode.ts:195`. showcase: `hydrology/contours.compute.ts:195`, `hydrology/least-cost-paths.compute.ts:315`, `hydrology/terrain-flow.compute.ts:134`, `hydrology/watersheds.compute.ts:250`, `raster/site-suitability.compute.ts:321`, `terrain/relief-visualization.compute.ts:746`, `terrain/sun-and-shadow.compute.ts:554`. lenses: `gpu-terrain/terrain-derivatives.example.ts:40`, `gpu-terrain/solar-irradiance.example.ts:67`. website: `demos/terrain-hydrology.ts:90`, `runs/terrain-relief-light/relief-blend.ts:37`, `runs/terrain-relief-light/slope-aspect-hillshade.ts:30`, `runs/terrain-relief-light/support/horizon.ts:55` | Accept the one-sided border, or pass `borderMode: 'clamp'`. Where the same scene also draws `GPUReliefShading` (explorer `relief-mode.ts:355`, `relief-visualization-mode.ts:596`; showcase `relief-visualization.compute.ts`), the two hillshades now differ in the outer ring unless relief shading gets `borderMode: 'one-sided'` |
| Border ring of every curvature (`GPUTerrainCurvature` without `borderMode`) | explorer: `geomorphometry-mode.ts:412`, `relief-visualization-mode.ts:583`. showcase: `terrain/relief-visualization.compute.ts:575`. lenses: `gpu-terrain/terrain-curvature.example.ts:36`. website: `runs/terrain-relief-light/terrain-curvature.ts:39` | as above |
| Seeded k-means++ centres (`GPUKMeans`) | explorer: `clusters-mode.ts:498`. showcase: `earth/satellite-constellations.compute.ts:175`, `earth/hurricane-families.compute.ts:439`, `points/nature-clusters.compute.ts:558`, `regression/regionalization.compute.ts:526`. lenses: `gpu-spatial-analysis/k-means.example.ts:46`. kepler: `catalog-platform4-geometry-groups.ts:279` (seed 1, relabelled; its spec passes). website: `demos/k-means.ts:65` | Captions or presets written for the clusters of a particular seed need a look |
| Seeded Louvain visit order (`GPUNetworkLouvain`) | showcase: `flows/aviation-network-lab.compute.ts:369`, `:386` (seed 17). lenses: `gpu-network/network-louvain.example.ts:14` (no seed; changes if the graph has 12 or more vertices). website: `runs/network-routing-access/network-louvain.ts:61` (seed 4332) | Community label IDs, and colours keyed on them, can change |
| Network K envelope (`GPUNetworkKFunction`) | explorer: `network-k-function-mode.ts:208`. showcase: `networks/retail-network-k-function.compute.ts:384`. lenses: `gpu-network/network-k-function.example.ts:35`. website: `runs/network-routing-access/network-k-function.ts:67` | Text that reads the observed curve against the envelope |
| Conditional simulation draws (`GPUConditionalGaussianSimulation`) | showcase: `interpolation/meuse-conditional-simulation.compute.ts:222`. lenses: `gpu-spatial-analysis/conditional-gaussian-simulation.example.ts:17`. website: `runs/density-interpolation-regions/conditional-gaussian-simulation.ts:26` | last bits of every draw |
| Fold assignments (`GPUSpatialCrossValidation`, `GPUNestedInterpolationValidation`, `GPUSpatialCrossValidatedOLS`) | showcase: `interpolation/rainfall-interpolation.compute.ts:223`, `interpolation/meuse-validation.compute.ts:248`, `regression/health-regression.compute.ts:789`. lenses: `gpu-spatial-analysis/spatial-cross-validation.example.ts:17`, `.../nested-interpolation-validation.example.ts:15`, `.../spatial-cross-validated-ols.example.ts:17`. website: `runs/spatial-statistics/spatial-cross-validation.ts:62`, `.../nested-interpolation-validation.ts:83`, `:99`, `.../spatial-cross-validated-ols.ts:83` | Every fold map, every per-fold error and every selected model can change. Text that names which fold a block is in, or a selected bandwidth, needs a look |
| Scan replicates (`GPUExpectationBasedSpaceTimeScan`) | website: `runs/space-time-events/expectation-based-space-time-scan.ts:128` | p-values within Monte Carlo noise |
| Default sums (`GPUNetworkCoarsening` without `fixedPointScale`) | showcase: `flows/aviation-network-lab.compute.ts:405`, `:443`, `flows/flight-matrix.compute.ts:836`. lenses: `gpu-network/network-coarsening.example.ts:29`. website: `runs/network-routing-access/network-coarsening.ts:50` | Sums are now exact; pass `fixedPointScale: 65536` only to keep recorded numbers |
| Drive times and bands (`addDriveTimeCatchmentRecipe`) | explorer: `recipes-scenes-spatial.ts:871`. showcase: `networks/isochrones.compute.ts:657`. lenses: `gpu-network/drive-time-catchment-recipe.example.ts:25`. website: `runs/network-routing-access/drive-time-catchment.ts:50` | Demand on a facility's own road gets a shorter time and can change band. The caption "about 44.8 seconds, in band 0" in `docs/api/gpu-network/operators.md` was flagged by the stream |
| Isochrone raster bands `(a, b]`, `'max'` mode (`GPUNetworkIsochrones`) | explorer: `isochrones-mode.ts:349`, `:436`. showcase: `networks/isochrones.compute.ts:748` (offers `'max'` mode), `networks/transit-reachability.compute.ts:318`. lenses: `gpu-network/network-isochrones.example.ts:33`. kepler: `catalog-gpu-network.ts:1047` (its spec passes). website: `runs/network-routing-access/network-isochrones.ts:60` | Contours move by one float32 step; the showcase `'max'` picture changes materially |
| Percentile and equalized stretch (`GPURasterStretch`) | explorer: `contours-mode.ts:440` (percentile by default). showcase: `raster/burn-severity.compute.ts:403` (percentile by default). website: `runs/raster-bands-indices/contrast-stretch.ts:49` | Equalized pictures are now step functions of the bin |
| Swept-span counts in `topologyStatistics` (`GPUCoverageSimplification`) | explorer: `coverage-mode.ts:278` (reads it at `:322`). showcase: `geometry/noding-and-coverage.compute.ts:1455` (reads it at `:1470`). lenses: `gpu-spatial-analysis/coverage-simplification.example.ts:54` | A label that calls words 0 and 1 "crossings" now also counts swept spans |
| Reconstruction floods through corners (`GPURasterReconstruction` without `connectivity`) | lenses: `gpu-raster/expansion-runner.ts:114` through `raster-reconstruction.example.ts:17` | The uniform 3 x 3 mask gives the same values in fewer iterations; the recorded kernel and iteration count change |
| Mode sentinel (`GPUModalSplit`) | website: `runs/trajectory-mobility/trajectory-modal-split.ts:71` | The run already treats `0xffffffff` as "no mode" (`NONE`); a zero-duration tripleg is no longer drawn as mode 0 or counted in the bars |
| Acceleration flag rows (`GPUTrajectoryOutlierFlags`) | website: `runs/trajectory-mobility/trajectory-outlier-flags.ts:97` (drives `maximumAcceleration` at `:116`) | The synthetic jumps are flagged on different rows; the caption and the catalog summary describe the old two-step rule |
| Volume of empty cells (`GPUUtilizationVolumeIsopleths`) | website: `runs/trajectory-mobility/trajectory-utilization-volume-isopleths.ts:38` | The picture is the same (`value > 0 ? 1 - value : 0` gives 0 for both 0 and 1); a pointer readout of an empty cell's level reads 1 instead of 0 |
| Whittaker curves (`GPURasterWhittakerSmoother`) | website: `runs/raster-time-series/whittaker-smoother.ts:61`, `runs/raster-time-series/phenology-metrics.ts:77` | Last digits at small `lambda`, materially at large `lambda` |

Counts: explorer 20, showcase 27, lenses 15, kepler 2, website 23.

### 4.2 Only for particular inputs (not classified one by one)

Construction sites of contributors whose change is `result` or `status` for some inputs. Each
needs its scene's data to decide; the condition is in [behaviour-changes.md](./behaviour-changes.md).
The contributors with a `result`, `status`, `default` or `seeded` row have about 790 construction
sites outside `modules/analysis` (explorer 106, showcase 254, kepler-compute 8, lenses 182, kepler
44, website 187, other 6); 87 are in 4.1 and the sites in section 5 were read and are unaffected.

The ones most likely to show:

| Change | Condition | Call sites |
| --- | --- | --- |
| `GPUMarkedPointStatistics` same-type K times `n / (n - 1)` | same-type sets, no intensities (none of the three passes intensities) | showcase `points/earthquake-pair-structure.compute.ts:389`; lenses `gpu-dataframe/marked-point-statistics.example.ts:19`; website `runs/dataframe-pointcloud-records/marked-point-statistics.ts:52` |
| `GPUNetworkInhomogeneousK` rises 1 to 20 percent | `correction: 'ang'`, real edge lengths | showcase `networks/retail-network-intensity.compute.ts:710`; lenses `gpu-network/network-inhomogeneous-k.example.ts:13`; website `runs/network-routing-access/network-inhomogeneous-k.ts:51` |
| Street profile facade lengths | real coordinates | showcase `geometry/street-enclosure.compute.ts:105`; lenses `gpu-spatial-analysis/street-profiles-recipe.example.ts:14`; website `runs/geometry-construction-overlay/street-profiles-recipe.ts:36` |
| `GPUGeodesicPairs` rhumb midpoints | `'rhumb'` metric | showcase `earth/ocean-drifters-vs-model.compute.ts:311` (offers rhumb) |
| Point horizon distances and azimuths | Web Mercator grids | explorer `visibility-mode.ts:326`; showcase `terrain/horizon.compute.ts:412`, `:511`; lenses `gpu-terrain/point-horizon-profile.example.ts:40`, `point-horizon-visibility.example.ts:41`; website `runs/terrain-relief-light/point-horizon-profile.ts:61`, `point-horizon-visibility.ts:59` |
| `GPUNetworkNoding` loops listed once, unused weights NaN | self-loops; readers of unused slots | explorer `noding-mode.ts:176`; showcase `geometry/noding-and-coverage.compute.ts:680`; lenses `gpu-network/network-noding.example.ts:57`; website `runs/network-routing-access/network-noding.ts:51`. Kepler (`catalog-gpu-network.ts:266`) does not read the CSR outputs |
| `GPUTerrainFlowFractions` no-data rows 0; direct `GPUTerrainBranchingFlow` needs `validity` | DEM with no-data cells | showcase `hydrology/terrain-conditioning.compute.ts:342`, `:370`, `:441`, `:462`, `:487`, `:512`, `hydrology/terrain-boundaries.compute.ts:358`, `:382`; website `runs/terrain-hydrology-flow/terrain-flow-fractions.ts:38`, `:50`, `terrain-branching-flow.ts:72`, `:84`. None of these scenes has no-data cells or passes `validity`, so they are unaffected today and would give a valid outlet at a no-data cell if their DEM gained one |
| `GPUProbitRegression` status 4 with NaN coefficients | a row at the bound | showcase `regression/meuse-lime-model-form.compute.ts:269`; kepler `catalog-platform5-regression.ts:576` (hands any non-zero status to the CPU engine); lenses `gpu-spatial-analysis/probit-regression.example.ts:17`; website `runs/regression-weights/probit-regression.ts:47` |
| `GPUGeographicallyTemporallyWeightedRegression` AICc | every fit | website `runs/space-time-models/geographically-temporally-weighted-regression.ts:88` (changes if the run shows the AICc) |
| `GPUCompositeScore` component | negatively correlated indicators with `'pca'` | explorer `regression-mode.ts:1225`; showcase `statistics/vulnerability-index.compute.ts:320`; lenses `gpu-dataframe/composite-score.example.ts:44`; website `runs/dataframe-pointcloud-records/composite-score.ts:71` |
| `GPUGroupStatistics` and `GPUKeyJoin` means and sums | digits below 1/65536 | explorer 3, showcase 12, kepler-compute 2, lenses 2, website 3 call sites (`group-statistics-mode.ts:603`, `:760`, `zone-events-mode.ts:404`, `statistics/group-statistics.compute.ts:386` and others) |
| `GPUColumnProfile` histogram bins | values on a bin edge | explorer 1, showcase 4, kepler-compute 4, lenses 1, kepler 1 (`modules/kepler/src/exploration/gpu-column-statistics.ts:164`), website 1 |
| `GPURegionMask`, `GPURegionStatistics` lasso boundary | a point exactly on the lasso | explorer `lasso-mode.ts:411`, `:436`; showcase `points/lasso-explorer.compute.ts:505`, `:530`; website `runs/density-interpolation-regions/region-mask.ts:27` |
| `GPUIsolines`, `GPUIsobands`, `GPUIsobandRings` saddles | asymmetric saddle cells | explorer `contours-mode.ts:464`, `:507`, `:627`; showcase `hydrology/contours.compute.ts:228`, `:270`, `:365`; lenses three examples; website `runs/terrain-hydrology-flow/isolines.ts:40`, `:61`, `isobands.ts:38`, `isoband-rings.ts:65` |

## 5. Call sites read and found unaffected

- All 42 `rowDirection` calls and every default-`rowDirection` call of the seven terrain classes
  (section 2).
- Terrain border default: kepler `analysis3-terrain-gpu.ts:114`, `:138` and kepler-compute
  `raster-terrain-zonal-gpu.ts:277`, `:301` pass `'nodata'`; showcase
  `terrain/terrain-basics.compute.ts:311`, `:381`, `terrain/landforms-products.ts:212` and
  `raster/wildfire-terrain.compute.ts:395` pass the mode from scene state.
- `GPUNeighborSearch` floor: every inverse-distance caller passes `distanceFloor` (explorer
  `spatial-weights-mode.ts:1531`, `trade-areas-mode.ts:253`; showcase
  `weights/b4-weights-core.ts:423`, `weights/spatial-weights.compute.ts:390`,
  `regression/trade-areas.compute.ts:304`, `time/emerging-hot-spots.compute.ts:584`; lenses
  `huff-trade-areas.example.ts:83`).
- `GPURasterReconstruction`: showcase `raster/missing-observations.compute.ts:635` and website
  `runs/raster-bands-indices/morphological-reconstruction.ts:40` pass `connectivity`.
- `GPUDatumGridTransform`: website `runs/dataframe-pointcloud-records/datum-grid-transform.ts:52`
  passes `tolerance: 3e-9`.
- `GPUNetworkCentrality` zero-weight rule: kepler `catalog-platform4-geometry-network.ts:168`
  asks for closeness only, and the path-count phases run only with a betweenness output.
- `GPUCostDistance`, `addHotSpotAnalysisRecipe`, `GPUSpaceTimeKernelDensity` (section 2).

## 6. Stale lens records and stale text

### 6.1 Recorded lens files

`docs/lens-records/*.json` and `website/src/lens/generated/*.json` hold the same 410 scenario ids.
No spec compares a record with a live run (`capture-records.spec.ts` only writes, and only with
`VITE_CAPTURE_LENS_RECORDS=1`), so stale records fail nothing; they show old kernel text, old node
lists and old values.

Method: a record is matched to contributors by the class names in its text, by the classes its
`.example.ts` and `.lens.ts` name, and by its id. Tier 1 means a matched contributor's own source
file (the class file, or a kernel file that carries its operation name) has a code change in the
diff, comments excluded. Tier 2 means the record reaches changed code only through a shared
import, so its kernel text may differ while its values may not; this tier is an upper bound,
because `wgsl-common.ts` and `philox-random.ts` changed and are imported widely. Tier 1 can include
a contributor whose only change is a new optional prop.

- **Tier 1: 181 records** in each folder (865 of the 1,849 files under `docs/lens-records/twins/`
  belong to them).
- Tier 2: 195 more records (870 twin files).
- Not reached by any changed file: 34.

Records the streams named themselves: `gpu-terrain.terrain-depression-hierarchy` (it still lists
the scratch `terrain-depression-hierarchy-curve-compensation`, which no longer exists),
`gpu-raster.raster-reconstruction`, the isoline and isoband records, the cross-validation trio,
`conditional-gaussian-simulation`, the network K, Louvain, centrality, isochrones, analytics
columns, noding, coarsening, inhomogeneous K and drive-time records, `cell-table-compare`,
`region-mask`.

Tier 1 by domain:

- `gpu-dataframe` (9): `composite-score`, `cross-variogram`, `key-join`, `marked-point-statistics`, `ripley`, `ripley-envelope`, `spatial-correlogram`, `variogram`, `variogram-model-fit`
- `gpu-network` (15): `adjacency-matrix`, `drive-time-catchment-recipe`, `network-accessibility`, `network-analytics-columns`, `network-centrality`, `network-coarsening`, `network-cost-matrix`, `network-cost-weights`, `network-inhomogeneous-k`, `network-isochrones`, `network-k-function`, `network-louvain`, `network-noding`, `network-snapping`, `track-quality-recipe`
- `gpu-raster` (18): `change-detection`, `cost-distance`, `cost-distance-path`, `isoband-rings`, `isolines`, `line-integral-convolution`, `raster-arithmetic`, `raster-conditional`, `raster-conservative-remapping`, `raster-fill-holes`, `raster-fractional-categorical-statistics`, `raster-fractional-coverage`, `raster-fractional-zonal-distribution`, `raster-fractional-zonal-statistics`, `raster-reconstruction`, `raster-reprojection`, `raster-stretch`, `raster-tile-reprojection`
- `gpu-spatial-analysis` (117): `azp-regions`, `buffer-surface`, `building-adjacency-recipe`, `building-tessellation-recipe`, `cell-aggregation`, `cell-compaction`, `cell-cover`, `cell-geometry`, `cell-grid-path`, `cell-level-selection`, `cell-measures`, `cell-pyramid`, `cell-rollup`, `cell-set-outline`, `cell-table-compare`, `cell-topology`, `class-assignment`, `classification-fit`, `cluster-and-outline-recipe`, `conditional-gaussian-simulation`, `contiguity-weights`, `coverage-dissolve`, `coverage-simplification`, `coverage-validity`, `drainage-network-recipe`, `emerging-hot-spots`, `fleet-dwell-recipe`, `focal-statistics`, `fuzzy-contiguity`, `gamma-statistic`, `generalized-geographically-weighted-regression`, `global-permutation-test`, `global-spatial-statistics`, `group-convex-hull`, `group-geometry`, `hot-spot-analysis`, `hot-spot-analysis-recipe`, `huff-trade-areas`, `inhomogeneous-point-pattern`, `interpolation-diagnostics-recipe`, `k-means`, `kriging`, `landscape-metrics-recipe`, `line-merge`, `line-noding`, `line-precision-reduction`, `local-geary`, `local-kriging`, `local-moran`, `local-permutation-test`, `location-allocation`, `make-valid`, `map-coloring`, `max-p-regions`, `minimum-bounds`, `multiple-testing-adjustment`, `nearest-feature-weights`, `neighborhood-summary`, `nested-interpolation-validation`, `offset-curve-components`, `ordinary-least-squares`, `period-comparison-recipe`, `point-density`, `point-to-cell`, `polygon-overlay`, `polygon-precision-reduction`, `polygonize`, `probit-regression`, `rate-cluster-map-recipe`, `region-mask`, `region-partition-evaluation`, `region-statistics`, `regional-raster-change-recipe`, `restricted-regime-regression`, `seemingly-unrelated-regression`, `segment-ring-assembly`, `segregation`, `shape-generator`, `skater-regions`, `space-time-hot-spots-recipe`, `spatial-clustering`, `spatial-cross-validated-ols`, `spatial-cross-validation`, `spatial-empirical-bayes-rates`, `spatial-error-gm`, `spatial-error-maximum-likelihood`, `spatial-lag`, `spatial-lag-impacts`, `spatial-lag-maximum-likelihood`, `spatial-markov`, `spatial-panel-error-maximum-likelihood`, `spatial-panel-lag-maximum-likelihood`, `spatial-panel-regression`, `spatial-pearson`, `spatial-rate-recipe`, `spatial-regime-regression`, `spatial-regression-diagnostics`, `spatial-regression-recipe`, `spatial-two-stage-least-squares`, `spatial-weights-algebra`, `spatial-weights-minimum-spanning-tree`, `spatial-weights-summary`, `spatial-weights-traces`, `spatial-weights-transform`, `spatial-weights-transpose`, `spatial-weights-validation`, `spherical-polygon-overlay`, `street-profiles-recipe`, `study-window-preparation`, `trajectory-encounters`, `trajectory-metrics`, `trajectory-playhead`, `trajectory-resample`, `trajectory-restriction`, `transition-diagnostics`, `transition-matrix`, `ward-regions`
- `gpu-terrain` (22): `geomorphons`, `point-horizon-profile`, `point-horizon-visibility`, `relief-blend`, `relief-shading`, `solar-irradiance`, `solar-shadow-mask`, `terrain-cast-shadow`, `terrain-catchment-summary`, `terrain-curvature`, `terrain-depression-hierarchy`, `terrain-derivatives`, `terrain-flow-distance`, `terrain-flow-fractions`, `terrain-height-above-drainage`, `terrain-horizon`, `terrain-spike-repair`, `terrain-stream-network`, `terrain-stream-order`, `terrain-tiled-flow`, `terrain-vector-ruggedness`, `terrain-watersheds`

Tier 2 by domain:

- `gpu-dataframe` (12): `bivariate-classification`, `class-breaks`, `color-scale`, `column-profile`, `column-quantiles`, `group-statistics`, `inequality`, `point-pattern-indices`, `ripley-distance-functions`, `space-time-variogram`, `temporal-reduction`, `variable-standardization`
- `gpu-network` (17): `alternative-routes`, `edge-bundling`, `facility-scenario-recipe`, `flow-aggregation`, `map-matching`, `network-cost-table`, `network-disruption-recipe`, `network-kernel-density`, `network-line-graph`, `network-neighborhood`, `network-path-extraction`, `network-reachability`, `network-reachability.chained`, `network-service-areas`, `network-statistics`, `network-structure`, `network-subgraph-filter`
- `gpu-raster` (28): `distance-field`, `isobands`, `particle-advection`, `polygon-rasterization`, `raster-cell-statistics`, `raster-extrema-pyramid`, `raster-fill-no-data`, `raster-focal-pairs`, `raster-join`, `raster-medial-axis`, `raster-patch-metrics`, `raster-profile`, `raster-rank-filter`, `raster-reclassify`, `raster-region-graph`, `raster-registration`, `raster-sampling`, `raster-seasonal-break`, `raster-sieve`, `raster-skeleton`, `raster-spectral-unmixing`, `raster-temporal`, `raster-texture`, `raster-walking-cost`, `raster-watershed-segmentation`, `raster-zonal-statistics`, `streamlines`, `weighted-overlay`
- `gpu-spatial-analysis` (114): `affine-transform`, `areal-interpolation`, `assuncao-reis-standardization`, `bounds-filter`, `buffer-selection`, `canonical-geometry-equality`, `canonical-polygon-equality`, `catchment-accessibility`, `cell-flow-topology-recipe`, `change-of-support-recipe`, `concave-hull`, `constrained-delaunay-triangulation`, `continuous-spatial-sample`, `dasymetric-redistribution-recipe`, `delaunay-tessellation`, `dot-density`, `empirical-bayes-rates`, `excess-risk-recipe`, `exposure-by-zone-recipe`, `fleet-dwell-zone-events-recipe`, `geodesic-destination`, `geodesic-pairs`, `geographic-distribution`, `geographically-weighted-regression`, `geographically-weighted-regression-nonstationarity-test`, `geometry-cleanup`, `geometry-measures`, `geometry-orientation`, `geometry-predicates`, `geometry-validity`, `great-circle-arcs`, `grid-generator`, `group-concave-hull`, `hierarchical-density-clustering`, `hilbert-keys`, `instrumental-variables-covariance`, `inverse-distance-weighting`, `island-attachment`, `isovists-recipe`, `kernel-density`, `knox-test`, `line-chunk`, `line-clip-by-polygon`, `line-density`, `line-length-per-polygon`, `line-locate`, `line-loop-removal`, `line-segmentize`, `line-simplification`, `line-smooth`, `line-split`, `linear-referencing`, `lisa-markov`, `local-collinearity`, `local-outlier-factor`, `local-rbf`, `local-spatial-heteroscedasticity`, `local-variability-recipe`, `lon-lat-polygon-preparation`, `losh-residual-permutation-test`, `mantel-test`, `maximum-inscribed-circle`, `minimum-clearance`, `multiscale-geographically-weighted-regression`, `multivariate-local-geary`, `multivariate-local-geary-permutation-test`, `natural-neighbor`, `nearest-feature-join`, `neighbor-search`, `normalize-geometry`, `normalize-polygons`, `optics`, `pair-gather`, `patch-proximity-recipe`, `point-in-polygon-join`, `point-pattern-envelope`, `points-in-polygons-choropleth-recipe`, `polygon-triangulation`, `population-support-alignment-recipe`, `pycnophylactic`, `quantile-lisa`, `random-points-in-polygon`, `random-points-on-line`, `rank-markov`, `raster-zonal-distribution-recipe`, `ratio-concave-hull`, `rectangle-clip`, `resistance-connectivity-recipe`, `rose-statistic`, `s2-cap-covering`, `s2-cell-measures`, `s2-polygon-covering`, `segment-intersection`, `segregation-counterfactual-recipe`, `segregation-decomposition-recipe`, `shape-descriptors`, `shared-paths`, `similar-locations`, `spatial-balanced-sample`, `spatial-interaction`, `spatial-join-candidates`, `spatial-join-prepared`, `spatial-predicate-join`, `spatial-scan-statistic`, `straight-line-catchments-recipe`, `subdivide-polygons`, `surface-preparation-recipe`, `trajectory-kalman`, `vertex-snap`, `visibility-graph-recipe`, `voronoi-diagram`, `zonal-distribution-recipe`, `zonal-statistics`, `zone-events`
- `gpu-terrain` (24): `local-dominance`, `multi-scale-relief`, `profile-peaks`, `simple-local-relief`, `solar-position`, `terrain-branching-flow`, `terrain-breach`, `terrain-contours`, `terrain-critical-points`, `terrain-cumulative-viewshed`, `terrain-dem-coregistration`, `terrain-flow`, `terrain-flow-field`, `terrain-hydrologic-indices`, `terrain-line-of-sight`, `terrain-peak-prominence`, `terrain-peak-snap`, `terrain-rgb-decode`, `terrain-ruggedness`, `terrain-summits`, `terrain-topographic-position`, `terrain-viewshed`, `terrain-weiss-landforms`, `texture-shading`

Not reached:

- `gpu-dataframe` (3): `calendar-buckets`, `scalar-scale`, `time-window-filter`
- `gpu-network` (6): `adjacency-matrix-order`, `network-facility-optimization`, `network-flow`, `network-point-routing`, `network-rational-access`, `network-time-routing`
- `gpu-raster` (4): `raster-evi`, `raster-nbr`, `raster-ndwi`, `raster-savi`
- `gpu-spatial-analysis` (21): `bounded-facility-allocation`, `convex-straight-skeleton`, `dorling-cartogram`, `h3-cell-identities`, `label-point`, `lattice-weights`, `local-frame-projection`, `offset-curve`, `offset-expansion`, `outline-geometry`, `pick-region-mask`, `s2-cell-hierarchy`, `segment-snap`, `spherical-arc-overlay`, `spherical-convex-contains`, `spherical-convex-intersection`, `spherical-polygon-contains`, `track-similarity`, `track-similarity-search`, `trajectory-closest-approach`, `vector-straight-skeleton`

### 6.2 Text that restates an old rule

Seen while checking callers; not a complete search.

| File | What it says | Now |
| --- | --- | --- |
| `website/src/catalog/entries/terrain-relief-light.ts:76-85` | `rowDirection` default `'south'`; `borderMode` type `GPURasterBorderMode \| undefined`, default `'clamp'` | default from `grid.affine[4]`; `'one-sided' \| GPURasterBorderMode`, default `'one-sided'` |
| `website/src/catalog/entries/terrain-relief-light.ts:165-168` | curvature `borderMode` type `'clamp' \| 'nodata'` | `'one-sided' \| 'clamp' \| 'nodata'`, default `'one-sided'` |
| `website/src/catalog/entries/trajectory-mobility.ts:121` and the header of `runs/trajectory-mobility/trajectory-outlier-flags.ts` | a fix is flagged "only when both steps around it exceed a speed or acceleration limit" | true for speed; the acceleration bit follows the displaced-fix rule |
| `modules/lenses/src/gpu-raster/computation-pack/operators/isolines.ts:48` | saddles are joined when the mean of the four corners is at or above the level | the bilinear decider |
| `modules/lenses/src/gpu-dataframe/computation-pack/operators/indicators.ts:17` | 64 power iterations starting at `1/sqrt(d)` | `d + 1` starts, the largest Rayleigh quotient kept (named by the dataframe stream) |
| `modules/lenses/src/gpu-spatial-analysis/computation-pack/operators/cells.ts:312` | fallback cell measures are unreliable beyond resolution 5 | usable to resolution 10 (named by the cells-joins stream) |
| `modules/analysis/src/gpu-spatial-analysis/ontology/spatiotemporal-review-catalog.ts:588` | "recurse and Fauchald and Tveraa definitions" for `GPUTrajectoryRecursions` | named by the trajectories stream, left alone because it feeds generated records |

## 7. Undeclared API changes

None from the math-fix streams at the entry points. Detail:

- **Entry barrels.** Five changed. Four add a name: `getGPUCellSumScale`
  (`gpu-spatial-analysis/cell-aggregation`), `GPUClassIntervalClosure`
  (`.../distribution-dynamics`), `GPU_CONTIGUITY_WEIGHTS_MAXIMUM_SNAP_CELLS`
  (`.../spatial-weights`), `GPUTerrainFlowDirectionEncoding` (`gpu-terrain/hydrology`). The fifth,
  `pointcloud-analysis/index.ts`, replaces `export * from './pointcloud-plane-icp'` with the two
  names that file exported before, so that the new `POINT_CLOUD_PLANE_ICP_BLOCK_WORDS` stays
  internal. Nothing is removed. All four additions are declared.
- **Removed exports.** One file lost exported names: `createNetworkAnalyticsTopology` and
  `NetworkAnalyticsTopologyProps` in `analysis-core/network-topology.ts`. Declared
  (luma-official-03). The module is not in an entry barrel and nothing outside
  `modules/analysis/src` named them (the remaining mentions are in `dev-docs`). Renames and
  removals were forbidden in this pass; this one is of a module-internal helper, and the lead
  should confirm it is accepted.
- **New module-level exports** (such as `resolveTerrainRowDirection`,
  `validateRasterGridOrthogonal`, `ISOLINES_SADDLE_WGSL`, `getPivotLimitedCholeskyWGSL`): none is
  reachable from an entry barrel.
- **Props.** Inside exported `Props` and `Output` types: 24 new optional members and three widened
  `borderMode` unions. No member became required and none was removed.
- **Exported constants.** Only additions; no status layout, parameter length or summary length
  constant changed value.
- **Two optional props in the diff that no Resolution lists**, both from the other session's
  parallel pass (`dev-docs/research/parallel-pass-2026-11-10/fix-indexes.md`, `fix-contract.md`),
  not from the math streams: `GPUNetworkSnappingProps.maximumUnindexedWork` and
  `GPULocalKrigingProps.sampleIndexChunkQueryCount`. The same pass changed validations in
  `trajectory-estimation/prepare-trajectory-observations.ts` and
  `pointcloud-analysis/gpu-pointcloud-statistical-filter.ts`. Seen, not reviewed.
- **Behaviour declared under an ID that a caller may not connect with the class**:
  `GPUContiguityWeights` snap anchoring is under polygon-topology-03 in the spatial-statistics
  report; the swap-or-not shuffle change reaches `GPUSpatialScanStatistic` (core-numerics-02); the
  tile-sum change reaches `GPUSpaceTimeHawkes` and `GPUSinkhornDivergence` (regression-16).

## 8. Rows of `behaviour-changes.md` not confirmed in source

Every `default`, `throws` and `output` row was opened in source except these, which are taken from
the Resolution text:

- `GPUPolygonPrecisionReduction` transient names `<id>-ring-closure-*` (polygon-topology extra-D).
- `GPUCoverageDissolve` and `GPUCoverageSimplification` `status.overflow` for the snap range
  (polygon-topology-03): the output existed; the new meaning was not traced.
- `GPUTerrainDepressionHierarchy` scratch view ids (terrain-hydrology-02b).
- `getGPUAdjacencyMatrixFixedWeight` rounding (network-flows-07).
- `GPURegionPartitionEvaluation` reaching `GPUMaxPRegions` (E5).
- The `GPURasterStretch` lens scenario using the linear mode only (raster-imagery, "Not done").

All `result`, `seeded` and most `status` rows are the streams' statements. Their arithmetic is the
other four reviewers' scope; here only the caller side was checked.
