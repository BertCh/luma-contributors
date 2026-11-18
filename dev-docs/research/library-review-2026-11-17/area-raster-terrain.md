# Raster analysis and GPU terrain

Five public operations have concrete wrong-result cases worth testing first. Geographic zonal area can become zero for a valid cell. Fractional coverage can erase a polygon whose vertices are exactly representable. A horizon sweep can change when the same terrain is cropped with its advertised halo. Reprojection can move a sample between identical grids. Terrain derivatives can report the wrong slope on a sheared grid. The first two have especially small regression fixtures.

The main ergonomic issue is composition. A caller has to know which classes accept texture bands, which require buffer bands, which return reusable validity, and whether an output allocated with spare capacity can feed the next class. Much of the machinery to improve this already exists in the repository.

## What I checked

`R` means `modules/analysis/src/raster-analysis`, `T` means `modules/analysis/src/gpu-terrain`, and `C` means `modules/analysis/src/analysis-core`. `RT` and `TT` mean `modules/analysis/test/gpu-raster` and `modules/analysis/test/gpu-terrain`. The raster specs are under `gpu-raster`, despite the source directory name.

| Scope | Coverage |
|---|---|
| Source census | 195 raster TypeScript files, 50,758 lines. 119 terrain files, 38,671 lines. Also `C/raster-grid.ts`, 391 lines. |
| Public class census | 86 raster and 46 terrain exported `GPU…` classes after excluding 24 `…Operation` wrappers. I inspected the constructor port declarations for the 132 classes. The grouped matrix below covers them. |
| Files read in full | `C/raster-grid.ts`, `R/gpu-raster-reprojection.ts`, `R/gpu-raster-tile-reprojection.ts`, `R/gpu-raster-fill-nodata.ts`, `R/gpu-raster-watershed-segmentation.ts`, `R/raster-grid/gpu-raster-grid.ts`, `R/raster-expansion-utils.ts`, `T/terrain-processing/gpu-terrain-flow-fractions.ts`. |
| Deep samples | The large horizon, zonal-statistics, cost-distance and relaxation implementations. Also fractional clipping and aggregation, derivatives, embedding overviews, reconstruction, both time-series chunk planners, and recent tiled-flow planning, boundary solving and scheduling. I checked the installed public luma.gl raster and halo types against their local callers. |
| Specs and history | Inventoried 218 raster and 189 terrain specs. Read selected fixtures and assertions for affine precision, padded views, fractional coverage, embedding overviews, halo stages and tiled-flow boundaries. Checked the contributor contract, relevant previous review findings, roadmap rows and recent domain history. |
| Verification limit | Read-only review. I used small arithmetic probes for the affine, clipping, latitude and digital-line examples. I did not compile, run repository tests, execute GPU code, or establish parity against external GIS software. `UNVERIFIED` below identifies predicted device results. I did not review the implementation of every wrapper or every spec. |

The census used these commands, plus a syntax-only Node probe over TypeScript prop declarations. The probe did not invoke the compiler or typechecker.

```sh
rg --files modules/analysis/src/{raster-analysis,gpu-terrain} -g '*.ts' | xargs wc -l | sort -nr
rg --no-filename '^export class GPU' modules/analysis/src/{raster-analysis,gpu-terrain} |
  rg -v '^export class \w+Operation\b' | wc -l
rg --files modules/analysis/test/{gpu-raster,gpu-terrain} -g '*.spec.ts'
rg -l 'readonly requiredHalo' modules/analysis/src/{raster-analysis,gpu-terrain}
rg -l 'implements GPUCommandNodeProducer, GPURasterHaloStage' modules/analysis/src/{raster-analysis,gpu-terrain}
rg -l 'validateRasterBuffersDistinct\(' modules/analysis/src/{raster-analysis,gpu-terrain}
rg -l "from .*raster-time-series-chunks'" modules/analysis/src/raster-analysis -g 'gpu-*.ts'
rg -l "from .*raster-time-series-chunks-b'" modules/analysis/src/raster-analysis -g 'gpu-*.ts'
git log -12 --format='%h %s' -- modules/analysis/src/{raster-analysis,gpu-terrain}
```

## Findings

### RT-01 Geographic zonal area can collapse to zero on a one-row window [P0] [correctness]

- **Evidence.** The public packer writes two absolute latitudes into a `Float32Array`. Both the CPU twin and WGSL recover cell height by subtracting those latitudes and dividing by grid height. The API documentation explicitly says the separately packed `cellSizeY` is unused. For a one-row area grid starting at 60° with row step `-0.000001°`, `Math.fround(60) === Math.fround(59.999999)`. The recovered height is exactly zero. The shared spherical-area function already uses a numerically stable thin-cell formula if it receives the actual height. `R/raster-zonal-statistics/gpu-raster-zonal-statistics.ts:77,255,296,315` and `C/spherical-cell-area.ts:30,79`.
- **Why it matters.** A valid high-resolution cell contributes zero geographic weight. Reading a small window can change an aggregate even when its cell values, mask and resolution agree with the larger raster.
- **Proposed change.** Carry the signed row step independently through the zonal parameter layout. Use it directly in both area implementations and validate axis alignment for the geographic mode. Add one-row north-going and south-going fixtures, plus a window-versus-full-grid fixture. **M**. Preserve or version the public layout for callers that upload their own parameter words.
- **Confidence.** High. The CPU arithmetic is reproducible. GPU output `UNVERIFIED`.

### RT-02 Fractional clipping erases representable polygons at large pixel indices [P0] [correctness]

- **Evidence.** `snapToLattice` uses `4.76837158203125e-7 * max(abs(nearest), 1)` and applies it even to the identity affine. Both clipping routes subtract the cell origin **after** snapping. On a one-row identity grid, a unit-height rectangle from `x = 2²⁰ + 0.125` to `2²⁰ + 0.375` covers exactly one quarter of a cell. Both coordinates are distinct, exactly representable float32 values. The tolerance is `0.5`, so both snap to `2²⁰`. The indexed route drops nonpositive coverage. `R/fractional-coverage/raster-fractional-coverage-index.ts:116,119,385,401` and `R/gpu-raster-fractional-coverage.ts:147,326,333`.
- **Why it matters.** Translating a polygon across an otherwise identical raster can remove its tuple and all its zonal support. This is much larger than the boundary slivers the snap was introduced to address.
- **Proposed change.** Preserve exact identity-grid coordinates. For general affines, evaluate the boundary decision in cell-local coordinates with an explicit absolute subpixel error bound. Test indexed and brute-force routes with both the translated quarter-cell rectangle and the existing decimal-cell boundary case. **M**, compatible API.
- **Confidence.** High for the snap and resulting degenerate polygon. Device output `UNVERIFIED`.

### RT-03 The horizon sweep can follow a different ray after halo tiling [P0] [correctness]

- **Evidence.** `GPUTerrainHorizon` advertises `maximumRadius` as its seamless tile halo. Its sweep follows digital lines using `F(a) = floor((slopeFixed * a + 32768) / 65536)`, with `a` counted from the input window. At a 22.5° line with `slopeFixed = 27146`, `F(5) − F(4) = 0`, while `F(4) − F(3) = 1`. A core pixel at global major index 4 becomes local index 3 when the window starts at 1. The next sampled cell therefore changes even with a full radius-two halo. Geographic sweep geometry also uses the window’s `HEIGHT / 2` as its reference row. `T/terrain-illumination/gpu-terrain-horizon.ts:674,719` and `T/terrain-illumination/terrain-horizon-sweep.ts:38,66,461,563,676`.
- **Why it matters.** One tall blocker in the differing cell is enough for the full-grid and tiled horizon to disagree. Supplying more surrounding samples does not restore the digital line’s phase.
- **Proposed change.** Anchor the sweep phase in overview-level global pixel coordinates and share the reference latitude across windows. The published metadata already carries a level-zero origin. Until that contract is implemented, advertise seamless sweep tiling only for configurations that preserve the geometry. Add an offset-window test with an oblique sector and a single blocker. **M**, changes affected tile results. `node_modules/@luma.gl/graph/src/gpu-raster/types.ts:75`.
- **Confidence.** High for the differing cell sequences. Horizon values on hardware `UNVERIFIED`.

### RT-04 Same-grid nearest reprojection can read column 7 for destination column 4 [P0] [correctness]

- **Evidence.** The context packs float32 axes alongside a determinant computed from the original float64 axes. It checks that both determinants are nonzero. Reprojection subsequently inverts the **packed** axes using the **separately rounded** determinant. Take identical 8×1 point grids with affine `[1, 1, 0, 1, 1 + 0.55 × 2⁻²³, 0]`. The packed axes have determinant `2⁻²³`, while the uploaded determinant is approximately `0.55 × 2⁻²³`. At `(4,0)` the source column evaluates to about `7.27`, which nearest sampling rounds to 7. The current spec covers collapsed axes and axes already exactly one float32 step apart. `C/raster-grid.ts:365,374,381`, `R/gpu-raster-reprojection.ts:95,297,336` and `RT/raster-context.node.spec.ts:81,103`.
- **Why it matters.** Identical-grid reprojection is a useful identity invariant. Here it fails on a grid that passes construction, with a finite in-bounds result and no invalid flag.
- **Proposed change.** For same-CRS transforms, compose the source inverse with the destination affine in float64 on the host before packing. Carry origin differences in that composition. For the projection path, use a consistent inverse representation or reject affines for which the supported precision cannot identify a source cell. Add the identity fixture with distinct values in columns 4 and 7. **M**, compatible for ordinary inputs. Revisit the published packed layout explicitly.
- **Confidence.** High for the packed arithmetic. Device sampling `UNVERIFIED`.

### RT-05 Terrain derivatives accept shear and compute an incorrect slope magnitude [P0] [correctness]

- **Evidence.** In uniform mode the constructor’s alignment check accepts shear. Grid resolution is reduced to the lengths of the two pixel axes. The output kernel divides the two directional derivatives by those lengths and takes their Euclidean norm. With affine `[1, 1, 0, 0, -1, 0]` and the plane `z = worldX`, the components have magnitudes `1` and `1/√2`. The computed interior slope is about **50.77°**. The plane’s slope is **45°**. `T/terrain-analysis/gpu-terrain-derivatives.ts:218,431,436,448`, `C/raster-grid.ts:84` and `R/raster-band-utils.ts:61,89`.
- **Why it matters.** This affects slope magnitude even when the caller never requests aspect or hillshade. The earlier orthogonality fix covered D8 hydrology consumers and provides a directly reusable validator. `dev-docs/research/math-correctness-review-2026-11-10/terrain-hydrology.md:413`.
- **Proposed change.** Call `validateRasterGridOrthogonal` in derivatives and its parameter packer as an immediate fix. Full shear support can apply the inverse-transpose affine to the pixel gradient. Add the plane above as an interior-cell oracle. **S** for rejection, **M** for full support. Rejection narrows previously accepted input.
- **Confidence.** High for the analytic result. GPU output `UNVERIFIED`.

### RT-06 Two terrain classes structurally advertise a halo they explicitly cannot guarantee [P1] [ergonomics]

- **Evidence.** The published `GPURasterHaloStage` has one required member, `requiredHalo`. The assembler reads it directly. I counted 23 source files declaring that member and 21 explicitly implementing the stage. The other two are Weiss landforms and geomorphons. Both explain in their source why their radius can be insufficient, yet both satisfy the TypeScript type. Existing CPU specs encode full-versus-cropped Weiss classes **5 and 10** and geomorphon direction digits **2 and 1**. `node_modules/@luma.gl/graph/src/gpu-raster/gpu-raster-tile-halo.ts:32,404`, `T/topographic-position/gpu-terrain-weiss-landforms.ts:181,198`, `T/geomorphons/gpu-geomorphons.ts:194,207` and `TT/halo-stage.node.spec.ts:205,225`.
- **Why it matters.** A caller can pass either instance to the halo assembler using normal typed code. The numeric field validates and the assembled output can contain seams.
- **Proposed change.** Rename the advisory value to `localRadius` or `searchRadius`. Offer a method returning a published `GPURasterHaloStage` only when the configuration has a complete bound. Migrate Weiss local mode first. **M**, removing the old public field is breaking and merits a deprecation cycle.
- **Confidence.** High. **KNOWN(UPS-03)** is under-scoped here. Its documentation action has already been taken, while structural assignability still defeats the warning. `dev-docs/roadmap.md:439`.

### RT-07 Building an embedding pyramid as documented changes the weights of source pixels [P1] [correctness]

- **Evidence.** The class recommends building coarser levels from the previous level. Each output is normalized, and the next level accepts vectors and a mask. The previous level’s `validCount` and `meanLength` cannot be supplied. For a 3×1 raster of unit vectors `[east, east, north]`, direct scale-four aggregation gives `(2,1)/√5`. Two scale-two levels give `(1,1)/√2`. Masks and heterogeneous vector concentrations introduce the same discrepancy away from odd edges. `R/raster-embeddings/gpu-raster-embedding-pyramid-level.ts:38,80,169,240,255`. The sampled oracle checks individual level outputs. `RT/raster-embeddings/rts03-embedding-rasters.spec.ts:624`.
- **Why it matters.** Similarity and classification can depend on which overview levels happened to be materialized en route to the requested resolution.
- **Proposed change.** Correct the chaining guidance immediately. For an exact composable pyramid, carry vector sums and valid counts, including support from a child whose vectors cancel exactly. Normalize only the displayed level. Add odd-edge, masked and cancellation cases comparing direct and cascaded aggregation. **M**, additive ports.
- **Confidence.** High for the aggregation algebra. GPU output `UNVERIFIED`.

### RT-08 Morphological reconstruction publishes one round beyond `maximumIterations` [P1] [correctness]

- **Evidence.** The prop describes the number of rounds before stopping unconverged. The implementation requests `maximumIterations + 1` to test the fixed point. That test writes `next`, swaps buffers and publishes the swapped result. For dilation with marker `[1,0,0,0]`, mask `[1,1,1,1]` and a budget of one, the first iterate is `[1,1,0,0]`. The verification round advances it to `[1,1,1,0]`. `R/gpu-raster-reconstruction.ts:65,176,183,220,225,229,243`.
- **Why it matters.** The caller receives a partial result with a larger propagation radius than requested. This also affects composed operations using reconstruction.
- **Proposed change.** Preserve the budgeted buffer for publication. Let the extra pass compare against it and update convergence only. Document whether `iterationCount` includes verification, and assert both the count and the partial values. **S**, changes budget-exhausted output.
- **Confidence.** High for the generated node sequence. Device output `UNVERIFIED`.

### RT-09 Buffer-only input types prevent otherwise natural raster pipelines [P1] [ergonomics]

- **Evidence.** The port census found 12 contributors naming `GPURasterBufferBand` directly, plus tile reprojection through its source-tile type. Rank filter, fill nodata and reprojection already call `getRasterBufferBandNodes`, whose texture branch uses the published `GPURasterTextureToBuffer` with calibration and output validity. In contrast, categorical statistics genuinely binds raw `uint32` category IDs and validates unit calibration. `R/gpu-raster-rank-filter.ts:36,96`, `R/gpu-raster-fill-nodata.ts:51,130`, `R/gpu-raster-reprojection.ts:41,286`, `R/gpu-raster-tile-reprojection.ts:36`, `R/raster-band-utils.ts:196,234` and `R/gpu-raster-fractional-categorical-statistics.ts:250,419`.
- **Why it matters.** A texture-backed DEM accepted by terrain derivatives cannot be passed directly to rank filtering or reprojection through the typed API. The caller ends up taking responsibility for conversion, channel selection, mask propagation and temporary storage.
- **Proposed change.** Widen rank filter, fill nodata and reprojection to `GPURasterBand` first. Validate texture ownership, dimensions and channel before graph mutation. Exercise a selected RGBA channel with raw nodata, nontrivial calibration and an explicit mask. Keep integer label paths exact. **M**, additive API.
- **Confidence.** High. This identifies three concrete migrations where the conversion path already exists.

### RT-10 Zonal siblings assign different physical meanings to area [P1] [consistency]

- **Evidence.** Whole-cell zonal statistics supports unit weights or authalic-sphere square metres. Fractional categorical statistics supports coordinate units squared, geographic ground area and Web Mercator ground area. Fractional zonal distribution additionally accepts caller-provided `cellAreas`. Fractional **scalar** zonal statistics exposes `coveredArea`, `validArea` and density integrals but always multiplies by `abs(determinant)`. There is no ground-area option. `R/raster-zonal-statistics/gpu-raster-zonal-statistics.ts:57,415`, `R/gpu-raster-fractional-categorical-statistics.ts:105,153`, `R/gpu-raster-fractional-zonal-distribution.ts:94,108,257` and `R/gpu-raster-fractional-coverage.ts:435,439,612`.
- **Why it matters.** A longitude/latitude coverage tuple stream can yield square metres in one consumer and degrees squared in its sibling. An analyst integrating a per-square-metre density cannot obtain the corresponding physical total from the scalar API directly.
- **Proposed change.** Add the existing `cellSizeMode` and `cellAreas` options to fractional scalar statistics with the distribution contributor’s mutual-exclusion rule. State units on every area and integral output. Preserve the planar default. Add a shared one-zone fixture spanning two latitudes. **M**, additive.
- **Confidence.** High. The differences are explicit in the types and equations.

### RT-11 Tiled reprojection has no construction-time command-node budget [P1] [performance]

- **Evidence.** Tile count is explicitly unlimited. Encoding creates a full-source mosaic, then emits at least one route kernel per tile, followed by reprojection. Even 32,768 overlapping one-cell source tiles produce more than 32,768 nodes after clear and sampling. The repository contract requires a construction-time upper bound when node count grows with a prop. `R/gpu-raster-tile-reprojection.ts:39,47,54,137,152,170`, `C/kernels/gpu-contributor-utils.ts:495,506` and `dev-docs/contributor-contract.md:26`.
- **Why it matters.** The source and destination can both fit comfortably while the generated graph exceeds the library’s intended compile and encode budget. A storage-size preflight cannot identify that case.
- **Proposed change.** Compute `estimatedCommandNodeCount` before creating graph resources, including per-tile conversion allowances. Call `validateCommandNodeBudget` and identify `sources.length` in the rejection. Preserve documented last-tile-wins ordering in any later batched route. **S** for the guard, observable for oversized plans.
- **Confidence.** High. The route count is a source-level lower bound.

### RT-12 Raster outputs with spare capacity cannot be wrapped by the advertised composition helper [P2] [ergonomics]

- **Evidence.** `requireRasterLength` accepts capacity at least equal to the pixel count. The reprojection spec uses 53-element value and validity views for a 51-pixel grid. `makeGPURasterBand`, documented for feeding one contributor’s output into the next, requires both lengths to equal the pixel count. Raster algebra already borrows a bounded prefix while preserving byte offsets. `R/raster-expansion-utils.ts:55`, `RT/raster-context.node.spec.ts:113`, `R/raster-grid/gpu-raster-grid.ts:73,85` and `R/raster-algebra/raster-algebra-input.ts:217`.
- **Why it matters.** A caller who pools buffers or allocates to a high-water mark cannot use the natural output-to-band workflow without rebuilding views manually.
- **Proposed change.** Expose a graph-aware `makeGPURasterBandFromPrefix` that bounds both values and validity without copying. Reuse the algebra implementation and test nonzero byte offsets. **S**, additive.
- **Confidence.** High.

### RT-13 The horizon sweep constructs and stringifies the full march schedule [P2] [performance]

- **Evidence.** With default step growth, the public distance builder appends one JavaScript number for every integer through `maximumRadius`, then copies them to a `Float32Array`. Construction does this before selecting the algorithm. Encoding converts the entire array to WGSL literals before entering the sweep branch. That branch passes only `maximumRadius` to its kernel. `T/terrain-illumination/gpu-terrain-horizon.ts:289,296,303,712,734,926,988`.
- **Why it matters.** A sweep on a small grid with radius ten million incurs ten million host entries, a 40 MB typed array and a large temporary string for a schedule the sweep never consumes. This cost precedes GPU execution.
- **Proposed change.** Select the algorithm before materializing distances. Build the literal string only for march. Preserve the public `stepDistances` property through lazy materialization if compatibility requires it. Bound the useful march schedule by the grid extent before constructing shader text. **S–M**, compatible for normal callers.
- **Confidence.** High. Allocation sizes follow from the loop. I did not run the large allocation.

### RT-14 Raster families use conflicting rules for shared buffers and chunk topology [P2] [consistency]

- **Evidence.** `validateRasterBuffersDistinct` compares graph-buffer identities. It rejects disjoint writable views of one handle and cannot itself recognize two imports of the same physical buffer. It appears in 39 files including its definition. Terrain processing already checks physical storage, logical ranges and actual co-bound dispatches. In time series, nine GPU modules import one chunk helper and thirteen import the other. The first requires every participating field to be a vector once any field is one. The second compares chunk topology and can admit a packed field alongside a matching one-chunk vector. `R/raster-band-utils.ts:132`, `T/terrain-processing/terrain-view-validation.ts:21,53,108`, `R/raster-time-series/raster-time-series-chunks.ts:70,85` and `R/raster-time-series/raster-time-series-chunks-b.ts:70,80,141`.
- **Why it matters.** Whether a pooled allocation or a single-chunk adapter works can depend on which analysis is next. Graph-handle equality also gives weaker physical-alias diagnostics. The topology difference is demonstrated at helper level.
- **Proposed change.** Consolidate logical-range and per-dispatch binding checks using the existing physical-buffer-aware helpers. Specify packed versus one-chunk compatibility once. Migrate HANTS and temporal resampling as the first pair, keeping each operation’s actual dispatch bindings explicit. **M**, broadens some valid inputs and rejects previously missed aliases. `R/raster-time-series/gpu-raster-hants.ts:19` and `R/raster-time-series/gpu-raster-temporal-resample.ts:20`.
- **Confidence.** High for the validators. Downstream device handling of physical aliasing `UNVERIFIED`.

### RT-15 Four backlog rows describe capabilities that have already partly landed [P2] [docs]

- **Evidence.** **KNOWN(TE06)** says resident-only, but D8 has bounded tile scratch and a coarse boundary forest, with a spec asserting a boundary solve above the old resident ceiling. Branching remains resident and host paging and independent filling remain open. **KNOWN(R7.2)** says only steepest descent exists alongside a reference to the actual marker-controlled forest implementation. **KNOWN(R-REPROJ)** says source-tile routing is unimplemented, although ordered mosaic routing and a double-single projection path exist. **KNOWN(R-RELAX)** asks for `settledAt` with continuation, both present in the cost-distance API. `dev-docs/roadmap.md:160,161,168,169`, `T/terrain-processing/gpu-terrain-tiled-flow.ts:110,142`, `TT/terrain-processing/terrain-tiled-boundary.node.spec.ts:122,129`, `R/gpu-raster-watershed-segmentation.ts:77,89`, `R/gpu-raster-tile-reprojection.ts:45`, `R/gpu-raster-reprojection.ts:90` and `R/cost-distance/gpu-cost-distance.ts:188,268`.
- **Why it matters.** These rows can send the next contributor to implement an existing algorithm while the remaining limitation goes unaddressed.
- **Proposed change.** Rewrite each row around its outstanding acceptance case. For TE06, distinguish host paging, independent depression filling and branching transport. For reprojection, distinguish bounded mosaic assembly from source-window selection and filtering. For R-RELAX, identify a missing continuation case or close the row. **S**, documentation only.
- **Confidence.** High.

## Cross-cutting patterns

### Raster data model and contributor port matrix

The published model is a borrowed scalar band plus separate grid metadata. A band carries a native `float32`, `uint32` or `sint32` sample, optional source-domain nodata, scale, offset and a `uint32` validity view. A texture band selects one channel of an `r32*` or `rgba32*` texture. The grid supplies dimensions, a six-coefficient affine, area/point interpretation, optional CRS, overview level and level-zero tile origin. A band alone does not identify its grid. The local conversion helper can produce calibrated float32 values and explicit validity through public luma.gl primitives. `node_modules/@luma.gl/graph/src/gpu-raster/types.ts:9,18,33,42,67` and `R/raster-band-utils.ts:196,206,234`.

In the matrix, `B` accepts a published texture **or** buffer band, `Bb` requires a buffer band, `F` and `U` are raw float32 and uint32 buffer views, and `V` is an explicit uint32 validity plane. `G` means published grid metadata. The `GPU` class prefix is omitted, and braces expand names. Outputs are the principal caller-visible formats. Status, counts, class codes and binary masks are identified separately from validity.

| Contributors | Grid and input | Principal output and validity |
|---|---|---|
| `Raster{FillNoData,RankFilter}` | G, Bb | F values, V, U filled/changed indicators |
| `RasterFractionalCoverage` | G, packed F2 polygon positions and U topology | Compact U zone/cell IDs, F fractions, U zone validity and overflow |
| `RasterFractional{CategoricalStatistics,ZonalStatistics,ZonalDistribution}` | G, coverage tuples and Bb. Categorical requires uncalibrated U labels | Per-zone F area/statistics, U validity, category IDs or quantile validity as applicable |
| `Raster{Reprojection,TileReprojection}` | Source and destination G, Bb or ordered Bb tiles | F values and V |
| `Raster{NDWI,NBR,EVI,SAVI}` | Width/height, two or three named Bb inputs | F index and V |
| `RasterReconstruction`, `Raster{FillHoles,Skeleton,MedialAxis}` | G, F or B marker/mask for reconstruction, U binary rasters for the others | Reconstruction F and V. Binary results U, medial radius F, iterative status where applicable |
| `RasterWalkingCost`, `RasterWatershedSegmentation` | G, Bb elevation/friction plus U sources, or Bb surface plus U markers | F costs and V, or U labels with unresolved codes and status |
| `CostDistance`, `CostDistancePath`, `DistanceField` | G, B friction and U sources, U back-links, or F2 seeds/U seed mask | F costs or distances, U allocation/links/status or compact path. Distance field optionally writes `r32float` |
| `RasterFocalPairs`, `ScalarFieldRidges` | G, two B inputs or one B scalar field | F local metrics and U paired counts/status, or U ridges and V |
| `Isolines`, `Isobands`, `IsobandRings`, `PolygonRasterization`, `RasterJoin` | G, F/B plus mask and thresholds, or packed feature geometry/zone inputs | F geometry, U classes/IDs/offsets/counts, compact status or F aggregates |
| `Raster{Arithmetic,Conditional,Reclassify,CellStatistics}`, `WeightedOverlay` | Optional G on legacy raw-view paths. F/B operands or band-sequential F/B arrays, U mask | F values/statistics, U predicates/classes/counts. Overlay exposes V. Arithmetic has no separate output V |
| `Raster{Stretch,Sampling,Profile,ContrastEqualization,SpectralFilter,ExtremaPyramid}` | G, F/B plus mask for stretch/sampling/profile, B for the others | F values, samples or min/max levels. Sampling, equalization and filtering expose V. Stretch/profile use counts or compact status |
| `Raster{PatchMetrics,Sieve,ZonalStatistics}` | G, U patch labels and mask, or B values plus U zones/dictionary | F/U patch or zone metrics, U labels/counts and status |
| `Raster{ConservativeRemapping,Registration,Texture,RegionGraph,SpectralUnmixing}` | Remapping has its own origin/spacing grids. Registration, texture and region graph take G with raw F/U views. Unmixing takes shaped F records and endmembers | F coverage, shifts, features, weights or abundances. U mask/count/status semantics vary by operation |
| `LineIntegralConvolution`, `Streamlines`, `ParticleAdvection`, `LagrangianFlowMap`, `FiniteTimeLyapunovExponent` | LIC takes G. The others use explicit field/seed dimensions or spatial context. Raw F2 velocity, positions or seeds | F scalar fields, F2 positions, compact lines or mutable particle state. Flow map and FTLE expose V. LIC optionally writes `r32float` |
| `RasterEmbedding{Quantize,Dequantize,PyramidLevel,Projection,Similarity,Change,Covariance,ZoneMean,KMeans,KNearestClassifier,LinearProbe,NearestPixels}` | Pixel count/dimensions or source width/height. F vector records or packed U words, with U masks depending on operation | F vectors, scores, moments or centroids. U labels, counts, packed vectors or validity depending on operation |
| `ChangeDetection`, `Raster{Temporal,SeasonalBreak,BestAvailablePixel,BFASTMonitor,COLDMonitor,ClimateAnomaly,ClimateRunAccumulator,EOFProjection,GeometricMedianComposite,HANTS,HarmonicRegression,LaggedCorrelation,MedoidComposite,PhenologyMetrics,RollingMean,SnapshotCovariance,StandardizedIndex,StandardizedIndexCalibration,StreamingAccumulator,TemporalResample,WhittakerSmoother}` | Cell/slice/band dimensions, raw observations or chunked vector views. Per-observation masks, QA and temporal axes depend on method | F statistics, fitted series, composites or accumulator state. U counts, selections, packed state or status. There is no family-wide output V |
| `Geomorphons`, `Terrain{Derivatives,Curvature,Ruggedness,VectorRuggedness,TopographicPosition,WeissLandforms}` | G and B elevation | F measures or U landform classes, V. Derivatives optionally writes float hillshade texture |
| `LocalDominance`, `MultiScaleRelief`, `SimpleLocalRelief`, `TextureShading`, `ReliefShading`, `Terrain{Horizon,CastShadow}`, `ReliefBlend` | G and B elevation, except `ReliefBlend` takes an array of raw F layers | F relief/light and V, optional display textures on several operators. Horizon also supports packed U `unorm16` |
| `SolarIrradiance`, `SolarShadowMask`, `SolarPosition` | G and F/packed U horizon for the first two. Position takes F2 geographic positions | F energy/visibility/angles, optional float textures. Irradiance exposes V. Position emits U daylight |
| `Terrain{Contours,CriticalPoints,PeakProminence,PeakSnap,Summits,FlowField,SpikeRepair}`, `ProfilePeaks` | G and B elevation. `ProfilePeaks` takes raw F profiles, U validity and offsets | F geometry/measures/flow vectors or U classes, peak masks and compact status. Spike repair exposes V |
| `PointHorizon{Profile,Visibility}`, `Terrain{CumulativeViewshed,Viewshed,LineOfSight}` | G, B elevation, F observer/target geometry or parameters. Several accept reusable extrema levels | F angle/distance/clearance or U visibility/sample counts. Viewshed optionally writes `r32uint`/`rgba32uint` |
| `TerrainRGBDecode` | G, `rgba8unorm` texture or packed U buffer, optional validity/nodata RGB | F elevation and V |
| `Terrain{Flow,HydrologicIndices,HeightAboveDrainage}` | G, B elevation plus raw U/F routing, accumulation or stream inputs as applicable | F surface, accumulation, indices or HAND. U D8 directions, cell classes, drainage IDs or convergence status |
| `Terrain{FlowDistance,StreamNetwork,StreamOrder,Watersheds}` | G, raw U D8 directions, streams or pour points | F distance with V, compact network geometry, or U orders/basin labels with status |
| `Terrain{Breach,DepressionHierarchy,DEMCoregistration,CatchmentSummary,FlowFractions,BranchingFlow,TiledFlow}` | G, raw F elevation/values/fractions or U routing and validity. Tiled flow takes explicit per-tile views and owned rectangles | F modified surfaces, differences, summaries, fractions or accumulation. U modification masks, IDs, per-cell resolution or per-pit status. Coregistration exposes V |

For API documentation, I would generate this inventory from the exported declarations and hand-author only semantics a type cannot convey, especially units, nodata handling, band order and the meaning of partial results. The distinctions are visible in `R/gpu-raster-spectral-indices.ts:16,37`, `R/raster-algebra/gpu-raster-arithmetic.ts:46`, `R/raster-embeddings/gpu-raster-embedding-similarity.ts:81`, `T/terrain-decode/gpu-terrain-rgb-decode.ts`, and `T/terrain-processing/gpu-terrain-tiled-flow.ts:74`.

### Iteration and partial results

| Family | What the caller can actually infer |
|---|---|
| Cost distance | Continuation has caller-owned working values, a restart protocol and cumulative iteration count. During continued execution, public costs for passable cells stay NaN until convergence. `R/cost-distance/gpu-cost-distance.ts:164,268`. |
| Terrain flow | Filling, flat resolution and accumulation expose separate convergence flags. A partial upstream stage can invalidate downstream routing. `T/hydrology/gpu-terrain-flow.ts:234,242,247,443`. |
| Tiled flow | The resident route uses an iteration budget. Tiled D8 uses bounded forest passes and ignores that budget. A stalled cycle can leave both `converged` and `iterationLimitReached` zero. `T/terrain-processing/gpu-terrain-tiled-flow.ts:127,133`. |
| Watersheds | Terrain drainage uses D8 pointer jumping with a default of 32 doubling rounds. Raster marker segmentation uses FIFO steps or forest rounds and marks every non-marker valid cell unresolved on exhaustion. `T/hydrology/gpu-terrain-watersheds.ts:119,130` and `R/gpu-raster-watershed-segmentation.ts:64,89,329`. |
| Distance field | Exact and jump-flood are scheduled algorithms. Jump-flood’s `0`, `1` or `2` refinements specify an exact pass count. A convergence flag would misdescribe the algorithm. `R/distance-field/gpu-distance-field.ts:97,100,119`. |

One documentation rule would help across all five families. For each iterative contributor, state **what the budget counts, what stops execution, which output is usable on exhaustion, and how to resume**. The general contract currently promises publication of partial work, while individual APIs sometimes preserve working values privately or deliberately suppress unproven labels. `dev-docs/contributor-contract.md:24` and `R/raster-watershed-forest.ts:274`.

## Keep

- **Keep the two public entry points.** Terrain callers have a coherent vocabulary around surfaces, drainage, visibility and landforms. Raster callers have algebra, sampling, spectral operations and image segmentation. Their two watershed operations answer different questions: “which marker wins this image cell?” and “where does this D8 flow path drain?” Document that distinction prominently in both entry points. `modules/analysis/package.json:44,54`, `R/gpu-raster-watershed-segmentation.ts:77` and `T/hydrology/gpu-terrain-watersheds.ts:119`.
- **Keep one direction decoder and shared D8 metrics.** Whitebox, TauDEM and ESRI encodings meet in one terrain decoder. Flow fractions already reuse the raster D8 grid helper and terrain routing equations. That is the right boundary for sharing. `T/hydrology/terrain-flow-tracing.ts:72,100`, `T/terrain-processing/gpu-terrain-flow-fractions.ts:23,27` and `R/cost-distance/raster-grid-utils.ts:130`.
- **Keep domain-specific focal semantics.** Paired regression needs valid paired observations and exposes their count. GDAL-style ruggedness has a deliberate edge extrapolation and invalid-neighbour replacement rule. Share window addressing and halo tests where they agree, while preserving those semantics. `R/focal-pairs/gpu-raster-focal-pairs.ts:45,83,117` and `T/topographic-position/gpu-terrain-ruggedness.ts:65,70,274`.
- **Keep the published band model and explicit conversion.** Its raw-domain nodata, calibration, channel and mask rules are a sound basis for interoperability. The local helper already uses public upstream operators. `node_modules/@luma.gl/graph/src/gpu-raster/types.ts:18,42` and `R/raster-band-utils.ts:196,234`.
- **Keep whole-versus-window oracle fixtures.** The halo specs demonstrate exactly how a plausible local result can differ from the global answer. The tiled-flow spec separately checks that a bounded execution plan is selected above the resident ceiling. Both styles deserve wider use. `TT/halo-stage.node.spec.ts:205,225` and `TT/terrain-processing/terrain-tiled-boundary.node.spec.ts:122`.

## Future

1. **Make a GIS window a first-class composition case.** Provide analysis-side adapters from published tile metadata to the operator-specific ownership window. Reprojection uses source-grid half-open bounds, tiled flow uses local `[column,row,width,height]`, and isolines uses exclusive bounds over marching **cells**. For rasterio/GDAL-style callers, explicitly distinguishing sample ownership from marching-cell ownership would remove a recurring off-by-one decision. The trade-off is a small adapter surface while existing tuple APIs remain compatible. `R/gpu-raster-tile-reprojection.ts:32`, `T/terrain-processing/gpu-terrain-tiled-flow.ts:74` and `R/isolines/gpu-isolines.ts:365,373`.
2. **Make pipeline equivalence a regression criterion.** Use small fixtures for whole versus cropped, direct versus cascaded overview, buffer versus texture, area versus point placement, and exact-size versus padded views. Each fixture should compare validity as well as values. This adds a modest fixture burden and directly covers RT-01 through RT-09 without needing large GPU benchmarks.
3. **Organize the raster root by family while preserving exports.** Raster has 17 root TypeScript files including its index. Terrain has one. Move fractional coverage and its reducers together, reprojection and mosaic together, and morphology together. Move genuinely shared *analysis-domain* grid metrics out of `cost-distance` when their callers span both trees. Preserve both public barrels and use upstream public primitives for generic raster operations. The cost is internal import churn that can be spread across ordinary feature changes. `R/index.ts:10`, `T/terrain-processing/gpu-terrain-flow-fractions.ts:23` and `dev-docs/contributor-contract.md:11`.
4. **Publish a compatibility table organized around user tasks.** For “read a window, compute a slope, aggregate by polygon, build an overview,” list accepted band storage, mask convention, area or distance units, coordinate frame, tile equivalence and partial-result policy. Include multi-band layout explicitly: named spectral bands, band-sequential stacks and pixel-major embedding vectors require different preparation. Generate type facts and maintain a small set of hand-reviewed semantic statements. `R/gpu-raster-spectral-indices.ts:37`, `R/raster-algebra/gpu-raster-cell-statistics.ts:83` and `R/raster-embeddings/gpu-raster-embedding-pyramid-level.ts:38`.

## Top 5

1. **RT-01** — Preserve signed row spacing so a valid geographic cell cannot acquire zero area when read as a window.
2. **RT-02** — Fix fractional snapping and lock in translation invariance for both clipping routes.
3. **RT-03** — Anchor the horizon sweep to global overview coordinates and verify a shifted tile with an oblique blocker.
4. **RT-04** — Compose same-CRS affines before float32 packing and enforce the identical-grid sampling invariant.
5. **RT-05** — Reject unsupported shear in terrain derivatives until the operator applies the full affine gradient transform.

## Questions for the lead

- Are the numeric layouts produced by public `getGPU…ParameterValues` packers treated as stable wire formats for hand-authored buffers? That determines whether RT-01 and RT-04 need parallel versioned layouts.
- Should the contributor contract permit explicit exceptions for withholding uncertified partial output? Marker watersheds and continued cost distance already make defensible but different choices.
- For physical-area APIs, should the project standardize on one named Earth model or expose the choice? That decision affects comparisons between whole-cell and fractional zonal summaries.
- Can the next compatibility release deprecate advisory `requiredHalo` fields? The existing counterexamples make that a higher-value migration than another documentation warning.