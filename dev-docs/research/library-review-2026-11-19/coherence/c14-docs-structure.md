# Finding the right analysis operation

## Summary

The reference often answers a GIS question well once the reader reaches the right section. Finding that section is the weak point.  
Three ordinary lookups can select a different calculation from the one requested: bivariate local Moran, rhumb-line navigation, and PySAL empirical-Bayes smoothing. The first two can silently retain the wrong mode in JavaScript.  
The spatial index promises every contributor and recipe, but names only **165 of 311**. All **120** API pages in the staging-aware census have a table-of-contents entry, yet only **94** are reachable by following links in API page bodies from the landing page.  
Of 60 sampled class targets, **18** contained a TypeScript example using that class. Four spectral-index targets contained only recording directives. The current website Markdown transform emits those directives as text.  
Of 15 ordinary-language tasks, **one** returned a result when passed verbatim to the public catalog search. Even the short query `change point` returns only `GPUPointCloudChange`.  
**The highest-value change is one checked task-to-contract index.** Each row should identify the question, qualified export, required mode, result shape and canonical section. Use those rows to produce the landing-page chooser, tool cross-reference and catalog search aliases.

## Census

I read the R19 working tree at HEAD `85a29fcc`. For files in `.typecheck/r19/staging-owned.txt`, I also read the newer working copy in `luma-contributors-ki`, whose base HEAD was `9e93cbc7`. **KI** below means that newer copy. Counts labelled **overlay** substitute only listed files and add the four listed API pages present only in KI. This is a review projection, not a claim about a merged build. `D/` means `docs/api/`, `S/` means `modules/analysis/src/`, and `Xref` means `D/gpu-spatial-analysis-cross-reference.md`.

### Pages and public class headings

| Measure | R19 | Overlay |
|---|---:|---:|
| Markdown API pages | 116 | 120 |
| Headings outside code fences | 1,581 | 1,587 |
| Pages with no `ts` or `typescript` fence | 54 | 58 |
| Pages absent from the matching table of contents | 0 | 0 |
| Public classes with an exact class-name heading | 402 | 402 |
| Additional classes named in a grouped or descriptive heading | 81 | 81 |
| Classes named only in a Parameters heading | 7 | 7 |
| Classes in no heading | 271 | 271 |
| Classes with 0 / 1 / 2 / 3 non-Parameters heading occurrences | 278 / 402 / 78 / 3 | 278 / 403 / 77 / 3 |

`COUNT` — `rg --files docs/api -g '*.md' | wc -l` gave 116. For classes I used Node and the installed TypeScript API to enumerate `getExportsOfModule` from the 11 `package.json#exports` source entries, follow aliases, and deduplicate defining class declarations. I counted whole class-name tokens in Markdown headings, outside fences. A grouped heading can qualify several classes. The result was **761** distinct classes. The earlier **289** in `KNOWN(DOC-108)` is historical. Today **278** lack a non-Parameters class heading, and the new qualified catalog tables do provide alternate destinations. DST-09 and DST-10 below test what is *at* those destinations.

The seven Parameters-only names are `GPUAZPRegions`, `GPUMaxPRegions`, `GPUWardRegions`, `GPURandomPointsOnLine`, `GPURasterEmbeddingProjection`, `GPURasterStandardizedIndex` and `GPURasterStandardizedIndexCalibration`.

### Sixty-class entry score

I stratified across all nine owning entries and selected direct producers, grouped entries, infrastructure, CPU wrappers, small single-purpose pages and large family references. Selection was purposeful, including operations encountered in the discovery exercise. These percentages should not be extrapolated to 761 classes.

I scored what a reader gets **at the destination**: the class section and its children, the named family section for a dedicated page, or the actual catalog row for a shared-form target. Earlier prose on the same page does not rescue a deep link unless the entry directs the reader there. A point means I found usable evidence for that item. For layouts, one usable primary port earns credit; the scores do not certify every optional port. An explicit absence earns credit where applicable.

| Entry item | Code | Classes with credit / 60 |
|---|---|---:|
| Says what it computes or provides | Q | 53 |
| Gives a formula, algorithmic definition or reference | F | 32 |
| Identifies construction props or settings a caller can act on | P | 51 |
| Gives a primary input element type and its layout | I | 21 |
| Gives a primary output layout and how much to read, including unused slots where relevant | O | 35 |
| Explains a status, validity, missing-result or failure condition | S | 35 |
| States a concrete limit or supported domain | L | 43 |
| Contains a TypeScript usage fence naming the class | E | 18 |
| Identifies the CPU form, pairs the forms, or explicitly states there is no twin | X | 5 |
| Links another operation for this task or its next step | R | 4 |

Here are the individual scores. Names have the `GPU` prefix unless another prefix or a complete unprefixed name is shown. A dash means zero at the selected destination.

| Selected reference | Classes and credits |
|---|---|
| Shared fitting kernels | `AdamStep` QFPOSL · `FittingMinibatchOrder` QFPIOL · `FittingGradientSums` QFPOL · `FittingCoordinateEncoder` QFPOSL |
| Catalog infrastructure and forms | `ParameterBuffer` QP · `ContributorOperation` QP · `AnalysisGeographicExtentAccumulator` QPS · `CPUPointToCellOperation` X · `CPUAlternativeRoutesOperation` X · `CPURasterFillHolesOperation` X |
| Computation record | `SnapshotTap` QPIOLE · `TransferMask` QFPIO |
| Geospatial | `PointBuffer` QFPIOLE · `DatumGridTransform` QFPIOSLE |
| Point cloud, KI | `PointCloudNeighborhoods` QPISL · `PointCloudGround` QPSL · `PointCloudSMRF` QPOSL · `PointCloudICP` QPOSL · `PointCloudChange` QPOSL · `PointCloudVoxelMerge` QPSL · `PointCloudHeight` QPSL |
| Dataframe, KI | `GroupStatistics` QFPIOSLE · `KeyJoin` QFPIOSLE · `ColumnProfile` QFPIOSLE · `ColumnQuantiles` QPOS · `ClassBreaks` QFPL · `ScalarScale` QPISL · `CalendarBuckets` QPIOSLE · `PeriodIndex` QFPIOSLE |
| Network, KI | `NetworkReachability` QFPOSLE · `NetworkServiceAreas` QPOSL · `NetworkIsochrones` QFPIOSLR · `NetworkSnapping` QPSL · `NetworkPathExtraction` QPOSL · `AlternativeRoutes` QFPOSL |
| Terrain | KI operators: `TerrainDerivatives` QFPL · `TerrainViewshed` QFPL · `TerrainLineOfSight` QPL · `TerrainRGBDecode` Q · `TerrainSpikeRepair` Q. Dedicated HAND page: `TerrainHeightAboveDrainage` QFPIOSLX |
| Raster | `RasterNDWI` – · `RasterNBR` – · `RasterEVI` – · `RasterSAVI` – · KI `RasterEmbeddingNearestPixels` QFPO · `RasterEmbeddingChange` QFPL · `RasterEmbeddingDequantize` QFPSLE · `RasterFillNoData` QFPIOSL |
| Spatial statistics, KI | `EmergingHotSpots` QFPIOSLE · `EmpiricalBayesRates` QFPIOSLER · `SpatialEmpiricalBayesRates` QFPOER · `LocalGeary` QPL · `GlobalSpatialStatistics` QFPIOSLE · `HotSpotAnalysis` QFPIOSLE |
| Spatial joins, tracks, lines and density, KI | `PointInPolygonJoin` QFPOSL · `SpatialPredicateJoin` QFPIOSLEX · `TrackSimilaritySearch` QFPOSL · `LineLocate` QFPOSLE · `PointDensity` QFPIOER |

Two distinctions matter. `GPURasterNDWI` has a good formula and band discussion **above** its empty class heading. `GPUPointInPolygonJoin` shares a long section with `GPUNearestFeatureJoin`; the latter's constructor example does not earn an example point for the former. Counting headings or fences page-wide misses both problems.

### Fifteen discovery journeys

For each row I started at `D/README.md`, tried the family landing or tool cross-reference, checked the sidebar route, and ran `queryGPUAnalysisOperators({search: phrase})` over the source ontology. The number in the last column is the result count for the **literal first-column phrase**. Text after it records a separately executed retry. The task labels are deliberately questions a developer might bring, not export names.

| User task | Landing-page route and result | TOC route or dead end | Catalog result and useful retry |
|---|---|---|---|
| which points fall in which polygon | Spatial → Spatial joins → `GPUPointInPolygonJoin`. Xref:39 explains smallest-row ownership on overlap. Use `GPUSpatialPredicateJoin` for all matching pairs. | Spatial → Operator Families → Spatial joins | 0. `point in polygon` and `point-in-polygon` also give 0. `PointInPolygon` finds the join. |
| drive-time areas | Network → Operators → `GPUNetworkIsochrones`. Xref:261 also names the demand-join recipe. | Network → Operators → Isochrones or drive-time recipe | 0. `isochrone` gives 3, including raster `GPUCostDistance`. `DriveTime` finds the recipe. |
| hot spots over time | Xref:95 → `GPUEmergingHotSpots` for an existing cube, or `addSpaceTimeHotSpotsRecipe` to construct one. | Spatial → Operator Families → Spatial statistics | 0. `hot spots` also gives 0. `emerging hot` gives both targets. |
| smooth rates for small populations | Spatial → Spatial statistics → pooled or neighborhood empirical Bayes. The family page gives the parity switch that Xref:88 omits. | Spatial statistics for shrinkage; Rate mapping for composed outputs | 0. `empirical bayes` gives 0. `EmpiricalBayes` gives both producers. |
| viewshed | Terrain → Operators → `GPUTerrainViewshed`. The same family distinguishes single-observer and cumulative visibility. | Terrain → Operators | **2**, single and cumulative viewshed. |
| fill holes in a raster | Raster → expansion → `GPURasterFillHoles` for enclosed background. Raster → Nodata filling → `GPURasterFillNoData` for missing measurements. | Nodata filling is explicit; binary hole filling sits under Raster, terrain and geometry expansion. | 0. `fill holes` gives 0. `fillholes` finds binary filling; `nodata` includes interpolation. |
| extract the ground from a point cloud | Point cloud → contributor reference → `GPUPointCloudGround` or `GPUPointCloudSMRF`. The introductory comparison distinguishes the methods. | Point cloud analysis → Point cloud analysis | 0. `ground` gives 10, including both classifiers. |
| snap GPS traces to roads | Xref:265 → `GPUMapMatching`. Network snapping is the separate operation for independent point locations. | Network → Operators → Map matching. Map-matched edge volumes consumes matches. | 0. `GPS` gives 3, including `GPUMapMatching`. |
| where has vegetation declined | Raster → Calibrated spectral indices identifies NDVI. For a time series, Raster change monitoring describes the seasonal regression slope. A single index image does not establish decline. | Raster analysis → Calibrated spectral indices **and** Raster change monitoring | 0. `NDVI` gives 0, correctly for the analysis-only catalog. The needed NDVI owner is upstream. |
| how high above the nearest drainage | Terrain → Operators → Height above drainage. The dedicated page clarifies that “nearest” here follows flow; branching flow can have several first streams. | Terrain → Height above drainage over D-infinity flow | 0. `height above` gives HAND and point-cloud height. |
| find an alternative route | Network → Bounded alternative routes → `GPUAlternativeRoutes`, with edge-distinct simple paths, budgets and `complete`. | Network → Alternative routes | 0. `alternative route` gives 0. `AlternativeRoutes` works. |
| compare two tracks | Spatial → Track similarity measures → `GPUTrackSimilarity`; select a measure and pair list. Bounded search is a separate choice. | Spatial → Track similarity | 0. `track similarity` gives 0. `TrackSimilarity` gives the measure and search classes. |
| find a change point | No API-body path to the dedicated page from the root. The TOC led to the single-change test and the multiple-change operation. | Spatial → Change points and the expectation-based scan → `GPUPettittTest` or `GPULocationChangePoints` | 0. `change point` gives **only `GPUPointCloudChange`**. `Pettitt` finds the time-series test. |
| summarize land cover by polygon | Raster → categorical composition → `GPURasterFractionalCategoricalStatistics`. The page distinguishes observed area and missing area. | Raster → fractional categorical statistics | 0. `land cover` gives 0. `fractional` includes the desired operation among 7. |
| find similar pixels | The raster landing has no API-body link path to Embedding rasters. The sidebar led to `GPURasterEmbeddingNearestPixels` and its missing-match sentinel. | Raster → Embedding rasters → nearest pixels | 0. Removing `find` works: `similar pixels` gives exactly the intended operator. `nearest pixels` gives 0. |

The first-result material is worth preserving. For example, the terrain example has **62 physical lines** inside the shared lifecycle fence and **26** inside its terrain fence, including imports, blank lines and checks. The example reads a number at `D/README.md:133`. A returning reader looking for an operation, however, reaches the family chooser only at line 270. A top-of-page “Find an operation” link would serve both readers without duplicating the lifecycle.

## Proposed convention

Every public task mapping should identify **one qualified export, the props needed to select the advertised calculation, its result shape, and a link to the section that explains how to read it**. Every class or form should have one canonical contract target; grouped sections are appropriate when the reader can distinguish the members there. Check the public binding and mode, inspect the rendered target, and require the task to be reachable from a family chooser.

## Findings

All recommended changes below are **ADDITIVE**. **No existing TypeScript call sites need to change**. The proposed navigation moves retain the old routes and fragments. Costs are implementation estimates.

| ID | Kind | Reach | Change | Cost | Where | One line |
|---|---|---|---|---|---|---|
| DST-01 | W | ORDINARY | ADDITIVE | S | Xref:84 | The bivariate local Moran lookup selects a univariate producer and advertises an unavailable permutation switch. |
| DST-02 | W/S | ORDINARY | ADDITIVE | S | Xref:56,57,224 | The rhumb-line setting in two lookup tables and the reference does not select rhumb navigation. |
| DST-03 | W | ORDINARY | ADDITIVE | S | Xref:88,324 | The PySAL smoothing lookup omits the option that changes the numerical answer. |
| DST-04 | W | ORDINARY | ADDITIVE | S | Xref:185 | A line-substring request is mapped to a point-returning operation. |
| DST-05 | S | ORDINARY | ADDITIVE | S | Xref:293,329; `coverage-expansion.md:15` | Three tasks lead readers to an entry that lacks the named export. |
| DST-06 | D | ORDINARY | ADDITIVE | S | Spatial README:143,225; Xref:190 | The lookup says concave hulls are unbuilt and sends polygon subdivision elsewhere. |
| DST-07 | S | ORDINARY | ADDITIVE | M | Four family Catalog sections; `website/build/markdown-plugin.ts:102` | The website renders the catalog and recording directives as text. |
| DST-08 | D | ORDINARY | ADDITIVE | M | Spatial `operators.md:10` | The “every contributor and recipe” index omits 146 public exports in its stated scope. |
| DST-09 | E | ORDINARY | ADDITIVE | S | `analysis-catalog.md:443` | All 125 shared-form rows link to their own table instead of the operation contract. |
| DST-10 | E | ORDINARY | ADDITIVE | M | `spectral-indices.md:164` and 31 peers | A real class heading can contain no prose, layout, example or link to the contract. |
| DST-11 | E | ORDINARY | ADDITIVE | M | Ontology all-entry query:188,222 | Familiar task phrases and even spaced class names miss supported operations. |
| DST-12 | E | ORDINARY | ADDITIVE | S | `D/README.md:270`; raster `operations.md:16` | The sidebar is complete, but 26 pages are unreachable through API page bodies. |
| DST-13 | S | ORDINARY | ADDITIVE | S | `analysis-catalog.md:14`; Markdown rewrite:154 | Links to repository evidence are rewritten into site routes that do not serve those files. |
| DST-14 | D | ORDINARY | ADDITIVE | S | Spatial `operators.md:19,22`; Xref:93 | Seven valid fragments send a class-name link to a different section, despite an available named heading. |
| DST-15 | C | ORDINARY | ADDITIVE | M | `docs/table-of-contents.json`, KI:32,64,103,140 | Fourteen page paths retain work-batch names; several also obscure the task in the visible title. |

### DST-01: Bivariate local Moran resolves to univariate local Moran

**Where.** Xref:84. `S/gpu-spatial-analysis/spatial-autocorrelation/gpu-local-moran.ts`, KI:78,199,473; `permutation-inference/gpu-local-permutation-test.ts`, KI:120,329,691.

**EXECUTED.** With four-row transient views on a null device, I declared `GPULocalMoran` with `quadrantGating: 'analytic'`. I repeated with `quadrantGating: 'permutation'` and a second variable. Both declarations produced **8 nodes, 8 recorded kernels and identical WGSL source**. Resolved `GPULocalMoranProps` contains `values` and `quadrantGating`, with no `secondValues`. The accepted gating choices are `'analytic'` and `'none'`.

**Now.** A JS caller following the row can write `new GPULocalMoran({...props, secondValues: y, quadrantGating: 'permutation'})`. A fresh TS object literal exposes the problem; the JS declaration succeeds.

**Wanted.** Give `esda.Moran_Local` and `esda.Moran_Local_BV` separate rows. For the latter show `GPULocalPermutationTest` with `statistic: 'localBivariateMoran'`, `values: x` and `secondValues: y`. Say which output is the observed statistic and which is the permutation probability. **SOURCE GPU confirmation:** hold `x` and non-self weights fixed, reverse a nonconstant `y`, and compare observed local statistics for the two runs. The host result above establishes the bad selection without asserting a device result.

**Value and cost.** PySAL users select the intended variable and inference procedure. S.

### DST-02: `model: 'rhumb'` is ignored

**Where.** Xref:56,57,224; `D/gpu-spatial-analysis/README.md:119`; `operators-lines.md` R19:367, KI:375. Both `GPUGeodesicPairs` and `GPUGeodesicDestination` resolve the model from `spatialContext` at their constructors. This extends `KNOWN(DOC-102)` to two different contributors and a different retired prop.

**EXECUTED.** One-row host declarations with valid output views produced the following for **both** classes:

| Props supplied | Observed result |
|---|---|
| `model: 'rhumb'` alone | Constructor error ending in `does not support planar coordinates` |
| `model: 'rhumb'` plus a longitude/latitude great-circle context | Instance `model === 'sphere'` |
| Longitude/latitude context with `metric: 'rhumb'` | Instance `model === 'rhumb'` |

**Wanted.** Print `spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'rhumb'}` in every rhumb row. Use `metric: 'ellipsoidal'` for the ellipsoidal distance row. Include the context's distance unit. **SOURCE GPU confirmation:** compare a constant-latitude journey from `(0°,60°)` to `(90°,60°)` with the spherical rhumb distance `Rπ/4` and the distinct great-circle distance.

**Value and cost.** Navigation users select the advertised path model on their first declaration. S.

### DST-03: The PySAL smoothing row needs both a mode and an output

**Where.** Xref:88,324; compare the correct short instruction at `D/gpu-spatial-analysis/README.md:256`, the full contract in KI `operators-statistics.md:2314`, and `S/gpu-spatial-analysis/rate-smoothing/gpu-empirical-bayes-rates.ts:271`.

**EXECUTED.** I ran the host oracle in both copies with events `[10,11,9]` and populations `[100,100,100]`. The prior variance was `−0.0009333333`. With the contributor default, `clampPriorVariance: true`, smoothed rates were `[0.10,0.10,0.10]`. With `false`, the PySAL-compatible answer was `[0.10,−0.04,0.24]`, rounded. Standardized scores were `[0,1.224745,−1.224745]` in both cases. These are three different things a reader could take away as “the rates.”

**Wanted.** Put `smoothing.Empirical_Bayes → GPUEmpiricalBayesRates({clampPriorVariance: false, …}).output.smoothedRates` in the lookup, expressing `output` as a caller-supplied port in the example. Give GeoDa-style clamped smoothing and Assunção–Reis standardization separate rows. For `Moran_Rate`, identify the subsequent Moran calculation. **SOURCE GPU confirmation:** submit the same three rows with both settings and read `smoothedRates`.

**Value and cost.** Analysts reproducing a published calculation can distinguish parity from the library default. S.

### DST-04: Slicing a line and locating an event have different result shapes

**Where.** `D/gpu-spatial-analysis/README.md:122`; Xref:185; KI `operators-lines.md:112,197`; `S/gpu-spatial-analysis/linear-referencing/gpu-line-locate.ts:104,142` and `line-segmentize/gpu-line-chunk.ts:119,132`.

**EXECUTED.** The resolved `GPULineLocateOutput` members are `positions, segmentIndices, tangents, angles, statuses, vertexMeasures`. The props take `eventMeasures`. The substring packer `getGPULineChunkParameterValues({startMeasure: 1, endMeasure: 3})` returned `[0,1,3,0]` as documented.

**Now.** `lineSliceAlong → GPULineLocate` suggests locating the endpoints will recover the sliced geometry. Any interior bend is lost that way.

**Wanted.** Split the row into `along → GPULineLocate` and `lineSliceAlong → GPULineChunk({mode: 'substring', …})`. Show input measure units and the line-offset output. **SOURCE GPU confirmation:** slice `[(0,0),(2,0),(2,2)]` from 1 to 3. The expected line is `[(1,0),(2,0),(2,1)]`; two endpoint locations cannot represent it.

**Value and cost.** Geometry users choose an output that retains the original path. S.

### DST-05: Print the owning import beside the task result

**Where.** `D/gpu-spatial-analysis/coverage-expansion.md:15` says datum-grid transforms live in `gpu-project`. Xref:18 defines the package suffix convention; Xref:293 marks NDVI `(raster)`; Xref:329 gives the drive-time recipe with no owning suffix. The spectral reference at `D/gpu-raster/spectral-indices.md:7` also mentions NDVI without its import. KI spatial `recipes.md:102` correctly gives the network owner.

**EXECUTED.** I checked the installed ESM namespace bindings:

| Export | Entry inferred from the lookup | Observed | Entry with function binding |
|---|---|---|---|
| `GPUDatumGridTransform` | `@luma.gl/graph/gpu-project` | `undefined` | `@luma-contributors/analysis/geospatial-analysis` |
| `GPURasterNDVI` | `@luma-contributors/analysis/raster-analysis` | `undefined` | `@luma.gl/graph/gpu-raster` |
| `addDriveTimeCatchmentRecipe` | `@luma-contributors/analysis/gpu-spatial-analysis` | `undefined` | `@luma-contributors/analysis/gpu-network` |

**Wanted.** Put the full import on these rows and qualify individual members of mixed-package recipe chains. `KNOWN(DOC-101)` repaired invalid *package paths*. These probes resolve valid packages and find the requested binding absent.

**Value and cost.** A reader can paste the import without first learning which package owns a neighboring operation. S.

### DST-06: The “not built” concave-hull and subdivision guidance is stale

**Where.** Spatial README:143,225; Xref:190. Compare `D/gpu-geometry-refinement.md`, KI:13,18, and `D/gpu-raster/expansion.md`, KI:57. The latter is an especially unlikely place to find the alpha-hull contract.

**EXECUTED.** `GPURatioConcaveHull` and `GPUSubdividePolygons` are function bindings on the spatial entry. `computeRatioConcaveHullOnCPU(Float32Array.of(0,0,2,0,0,2), {ratio:1})` returned `converged=[1]`, `triangleValidity=[1]`, `edgeValidity=[1,1,1]`. One input triangle has one area face and three boundary edges.

**Wanted.** Offer the implemented choices: `GPUConcaveHull` and `GPUGroupConcaveHull` for an alpha/circumradius criterion, `GPURatioConcaveHull` for ratio erosion, and `GPUSubdividePolygons` for bounded polygon subdivision. State the restricted geometry and mode. KI's ratio-hull page already explains why parallel erosion can differ from GEOS largest-first erosion; link to it without promising exact Shapely equivalence.

**Value and cost.** Users can evaluate four shipped operators before moving data to another library. S.

### DST-07: Family catalog and example directives render as literal text

**Where.** `D/gpu-spatial-analysis/README.md:96`, network README:106, raster README:78, terrain README KI:115; `website/build/markdown-plugin.ts:102`; `website/src/routes/docs/$.tsx:32`. `KNOWN(LENS-06)` concerns scenario coverage. This finding is about the actual docs renderer.

**EXECUTED.** I called the website's Markdown transform in memory with `::lens-catalog{entry=gpu-spatial-analysis}`. Its HTML was `<p>::lens-catalog{entry=gpu-spatial-analysis}</p>`. The route inserts `page.html` directly. The overlay contains **4 catalog directives and 1,537 individual lens directives**. Parsing the pages with the renderer's Markdown stages found **1,137 paragraphs beginning with a lens directive across 58 pages**; adjacent directives can share a paragraph.

**Wanted.** Make the four Catalog sections ordinary links to the relevant catalog view first. Then render each recording directive as a card or an ordinary link to its recorded example. Add one host render assertion for a catalog and one for an individual recording, checking the resulting destination and accessible label.

**Value and cost.** Family browsing works without understanding the site's authoring syntax. S for four links, M for the renderer.

### DST-08: The spatial index omits nearly half its advertised surface

**Where.** `D/gpu-spatial-analysis/operators.md:3,10,16`; spatial README:63. The missing names are listed under **Departures by name**.

**COUNT.** For the spatial barrel, I resolved class declarations with an instance `getCommandNodes` and functions matching `^add\w+Recipe$`. I compared their export spellings with whole-word tokens in `operators.md`. **311 in scope, 165 present, 146 absent**: 118 producer classes and 28 recipes. A mere mention counted as present, so this gives the index the benefit of the doubt.

**Now.** A reader following “lists them all” cannot find `GPUPettittTest`, `GPUQuantileLISA`, `GPUTrackSimilaritySearch` or `addSpatialRateRecipe`. The catalog contains the operations.

**Wanted.** Immediately label the current table “Selected operator families.” Generate the complete task index from the qualified public inventory plus editorial task rows, and compare both membership and destinations in a host check. Include a short explanation of the input or decision that separates siblings.

**Value and cost.** Browsing users see the same available operations as callers who already know an export name. S for the label, M for the index.

### DST-09: The operation-forms index has 125 self-links

**Where.** `D/analysis-catalog.md:430`; tables beginning at lines 443, 532, 558, 577 and 593. Examples are `CPUPointToCellOperation` at 452, `CPUAlternativeRoutesOperation` at 544 and `CPURasterFillHolesOperation` at 612.

**COUNT.** All **125** “Shared forms contract” links point back to the heading of the table containing that row. They cover **250 class names**, distributed across spatial **79 pairs**, network **16**, terrain **11**, dataframe **6** and raster **13**.

**Now.** Clicking the only reference on the alternative-routes row moves the reader to the top of the network table. The actual page explains the route prefix, padding and differences between omitted CPU budgets and GPU defaults.

**Wanted.** Keep the shared lifecycle link once above each table. Give every row a link to its domain contract and, where exported, the pure CPU function. For example, send the alternative-routes pair to `gpu-network/alternative-routes.md`. Use `KNOWN(DOC-108)`'s qualified identity to verify that following a row leaves the index and reaches its operation.

**Value and cost.** All 250 names gain useful destinations without adding 250 reference pages. S.

### DST-10: Thirty-two class targets contain no contract text

**Where.** The complete list of classes and pages appears below. A simple reproduction is `D/gpu-raster/spectral-indices.md:164`: the entire `GPURasterNDWI` section consists of a marker and a graph-recording directive. The formulas, bands and usage appear at lines 24, 67 and 92.

**COUNT.** For each named non-Parameters heading, I took the body through the next heading of equal or lesser depth, removed headings, author comments and directives, and retained classes for which **every** such section was empty. This gave **32** on the overlay. The test at `modules/analysis/test/gpu-spatial-analysis/ontology/catalog-reference-targets.node.spec.ts:295` checks class target coverage. An author can satisfy a named-heading check with an example marker.

**Wanted.** Place the canonical class anchor at the actual contract, or put one sentence and a direct “Inputs, outputs and limits” link at the existing class heading. Keep the recording next to it. Reject a canonical target with neither contract content nor an explicit contract link.

**Value and cost.** Search and deep links lead to information that lets the reader construct the operation. M for all 32.

### DST-11: Catalog search penalizes the vocabulary used by the docs

**Where.** `D/analysis-catalog.md:256,285`; `S/gpu-spatial-analysis/ontology/all-entry-points/analysis-operator-catalog.ts:188,222`; historical spatial query `ontology/spatial-analysis-operators.ts:1046,1070`.

**EXECUTED.** `queryGPUAnalysisOperators({search:'change point'}).map(o => o.exportName)` returned `['GPUPointCloudChange']`. `queryGPUAnalysisOperators({search:'Pettitt'})` found the time-series test. `similar pixels` found the nearest-pixel operator, while `find similar pixels` returned `[]`. Lowercasing and substring matching across concatenated fields explains the differences. The fifteen trials above give successes as well as failures.

**Wanted.** Keep the existing filter's matching semantics. Add a task-search function or docs search index that tokenizes CamelCase, spaces and hyphens, indexes approved GIS aliases, and ranks exact task matches before incidental substrings. Use the existing question taxonomy as a browse facet. Include an upstream result card for tasks such as NDVI without adding upstream classes to the analysis export catalog.

**Value and cost.** Users can search with a GIS term instead of guessing the constructor spelling. M.

### DST-12: A complete sidebar masks missing paths from the family pages

**Where.** `D/README.md:270`; `D/gpu-raster/README.md:72,133`; `D/gpu-raster/operations.md:16`; spatial README:63; `docs/table-of-contents.json`, KI:20,48. `KNOWN(CAT-09)` and `KNOWN(DOC-06)` concern sidebar membership. That check passes here.

**EXECUTED.** I parsed Markdown links with `remark-parse`, `remark-gfm` and `remark-rehype`, resolved local paths, and traversed from the API root. **94/120** overlay pages were reachable through API body links. From the four family landing pages, the reachable family pages were **38/60 spatial, 11/15 network, 16/22 raster and 7/8 terrain**. The traversal allowed intermediate pages in other families. It excluded sidebar links and external sites.

**Now.** The raster “Choose a page” route cannot reach Embedding rasters. The dedicated page for change points cannot be reached from the API root by this route at all. Both appear in the TOC.

**Wanted.** Put a task chooser before the long first-run examples, with direct links to each family chooser. Generate family chooser links from the same records as the sidebar. Check reachability from each family's own landing page separately from global TOC membership.

**Value and cost.** The documentation works when read as Markdown, through package links, or one page at a time. S.

### DST-13: Three evidence links become nonexistent documentation routes

**Where.** `D/analysis-catalog.md:14`, KI `gpu-spatial-analysis/vecchia-likelihood.md:107`, KI `pointcloud-analysis.md:1032`; `website/build/markdown-plugin.ts:154`; `website/src/docs.ts:18,28`.

**EXECUTED.** The actual transform converted the validation-catalog link to `href="/docs/../examples/deck/kepler-compute/VALIDATION-CATALOG"`. The docs loader enumerates `docs/**/*.md`; that example file has no docs route. The other two escaping Markdown targets are the spatiotemporal research README and the Kepler example README's point-cloud catalog. All three files being present in the checkout does not make their emitted URLs valid.

**Wanted.** Distinguish docs routes from repository paths during rewriting. Publish a small docs destination for user-facing example contracts, or resolve repository evidence through the site's configured source base. A host check can normalize every emitted local `href` against registered routes or served assets. This does not require settling the repository-owner decision in `KNOWN(ERG-12)` to repair the three user-facing destinations.

**Value and cost.** Users following validation and worked-example links reach the promised evidence. S.

### DST-14: Fragment existence is weaker than correct destination

**Where.** Spatial `operators.md:19,22`; Xref:93. Correct destinations are KI `operators-statistics.md:380,1070` and `operators-joins.md:785`.

**COUNT.** Seven local class links across the two lookup pages land at a heading that omits the class name even though the same destination page has a non-Parameters heading naming it. The spatial index sends `GPULocalGeary` to hot spots/local Moran, `GPUSpatialPearson` to permutation tests, `GPUGammaStatistic` to additional statistics, and `GPUNearestFeatureWeights` to point-in-polygon/nearest joins. Xref sends the first three statistics to additional statistics. The permutation section even mentions Pearson to explain that it is a different statistic.

**Wanted.** Point the seven links to the class or named shared heading. Keep contextual comparisons as secondary links. Extend `KNOWN(DOC-107)`'s fragment-existence check with a qualified class-to-preferred-target comparison, allowing an explicit family target where one genuinely serves several classes.

**Value and cost.** Users who click a known operator arrive at its contract on the first click. S.

### DST-15: Page groups preserve the order of implementation work

**Where.** `docs/table-of-contents.json`, KI:32,64,71,73,74,76,103–106,120,140,151,156; `D/gpu-raster/expansion.md`, KI:57–61; `D/gpu-spatial-analysis/coverage-expansion.md:12`.

**COUNT.** Nine API filenames contain `expansion` and five contain `mobNN`. Both the terrain and raster sidebar pages have the visible title **Expansion computation lenses**. The spatial sidebar category has **50 immediate items**, including an “Operator Families” group, while many operator families are siblings outside that group. The raster expansion table puts concave hulls, walking costs, binary reconstruction, reprojection and constrained Delaunay triangulation in adjacent numbered rows.

**Wanted.** Use task names in the sidebar. Move examples into their task sections and retain links to old routes. Keep historical capability identifiers intact; changing those identifiers is the separate `KNOWN(STD-13)` decision.

**Value and cost.** Readers choose by data and desired output without knowing when an operation was developed. M.

## Departures by name

**The 32 empty named contract targets.** These are departures at the class anchor, not claims that the algorithm is undocumented elsewhere.

| Page and first affected line | Every affected class, with `GPU` prefix omitted |
|---|---|
| `gpu-network/point-process.md:21` | `NetworkInhomogeneousK` |
| `gpu-raster/focal-pairs.md:120` | `RasterFocalPairs` |
| `gpu-raster/remote-sensing.md`, KI:131 | `RasterConservativeRemapping`, `RasterTexture`, `RasterTemporal`, `RasterRegistration`, `RasterSpectralUnmixing`, `RasterSeasonalBreak` |
| `gpu-raster/spectral-indices.md:164` | `RasterNDWI`, `RasterNBR`, `RasterEVI`, `RasterSAVI` |
| `gpu-spatial-analysis/coverage-statistics.md`, KI:735 | `InstrumentalVariablesCovariance`, `RestrictedRegimeRegression`, `ProbitRegression`, `LocalCollinearity`, `FuzzyContiguity`, `IslandAttachment`, `SpatialCrossValidation`, `SpatialCrossValidatedOLS` |
| `gpu-spatial-analysis/global-coordinates.md`, KI:489 | `S2PolygonCovering`, `SphericalPolygonContains`, `SphericalPolygonOverlay`, `SphericalArcOverlay` |
| `gpu-spatial-analysis/point-process.md`, KI:182 | `InhomogeneousPointPattern`, `PointPatternEnvelope`, `SpatialBalancedSample`, `OPTICS` |
| `gpu-terrain/terrain-processing.md:168` | `TerrainFlowFractions`, `TerrainDepressionHierarchy`, `TerrainCatchmentSummary`, `TerrainDEMCoregistration` |

**The 146 omissions from the spatial index.** For compactness each producer name below has the literal prefix `GPU`; each recipe stem expands to `add<stem>Recipe`. The grouping is editorial. Membership is the executed barrel-minus-index comparison.

| Group | Missing producer names |
|---|---|
| Geometry and cells | `CanonicalGeometryEquality`, `CanonicalPolygonEquality`, `ConcaveHull`, `ConstrainedDelaunayTriangulation`, `ConvexStraightSkeleton`, `GroupConcaveHull`, `H3CellIdentities`, `HealpixCellAggregation`, `HealpixPointToCell`, `LineLoopRemoval`, `LinePrecisionReduction`, `MaximumInscribedCircle`, `NormalizeGeometry`, `NormalizePolygons`, `OffsetCurveComponents`, `PolygonPrecisionReduction`, `RatioConcaveHull`, `S2CapCovering`, `S2CellHierarchy`, `S2CellMeasures`, `S2PolygonCovering`, `SegmentSnap`, `SphericalArcOverlay`, `SphericalConvexContains`, `SphericalConvexIntersection`, `SphericalPolygonContains`, `SphericalPolygonOverlay`, `SubdividePolygons`, `VectorStraightSkeleton` |
| Weights, sampling and local statistics | `AdaptiveBandwidthKernelDensity`, `AnalysisContiguityStatistics`, `AnalysisSpacing`, `CategoryPairCounts`, `ChunkedSpatialWeightsAlgebra`, `ChunkedSpatialWeightsTransform`, `ChunkedSpatialWeightsTranspose`, `ChunkedSpatialWeightsValidation`, `ColocationQuotient`, `ContinuousSpatialSample`, `DorlingCartogram`, `FuzzyContiguity`, `GeographicalDetector`, `InhomogeneousPointPattern`, `IslandAttachment`, `JacquezTest`, `LOSHResidualPermutationTest`, `LocalCategoryDependence`, `LocalKnox`, `LocalSpatialHeteroscedasticity`, `MultivariateLocalGeary`, `MultivariateLocalGearyPermutationTest`, `NearRepeatKnox`, `OPTICS`, `PointPatternEnvelope`, `QuantileLISA`, `RasterFalseDiscoveryRateMask`, `RedcapSingleLinkage`, `SpatialBalancedSample`, `SpatialEntropy` |
| Regression, interpolation and validation | `AreaOfApplicability`, `ConditionalGaussianSimulation`, `ConformalPredictionIntervals`, `EnsembleScores`, `GaussianPredictiveScores`, `GaussianRandomField`, `GeneralizedGeographicallyWeightedRegression`, `GeoConformalPredictionIntervals`, `GeographicallyTemporallyWeightedRegression`, `InstrumentalVariablesCovariance`, `LocalCollinearity`, `LocalKriging`, `LocalRBF`, `MultiscaleGeographicallyTemporallyWeightedRegression`, `MultiscaleGeographicallyWeightedRegression`, `NaturalNeighbor`, `NestedInterpolationValidation`, `ProbitRegression`, `RestrictedRegimeRegression`, `SeeminglyUnrelatedRegression`, `SpatialCrossValidatedOLS`, `SpatialCrossValidation`, `SpatialPanelErrorMaximumLikelihood`, `SpatialPanelLagMaximumLikelihood`, `SpatialPanelRegression`, `SpatialRegimeRegression`, `VecchiaFisherScoring`, `VecchiaLogLikelihood`, `VecchiaSpaceTimeLogLikelihood` |
| Movement, time and other workflows | `BoundedFacilityAllocation`, `BrownianBridgeUtilization`, `ExpectationBasedSpaceTimeScan`, `LocationChangePoints`, `MobilityIndicators`, `ModalSplit`, `PettittTest`, `PopulationSupportAlignment`, `RankMarkov`, `SinkhornDivergence`, `SpaceTimeCrossCorrelation`, `SpaceTimeHawkes`, `SpaceTimeKernelDensity`, `SpaceTimeKriging`, `SpaceTimeLocalMoran`, `SpaceTimeRelativeRisk`, `SpatialInteraction`, `TrackSimilaritySearch`, `TrajectoryHampel`, `TrajectoryKalman`, `TrajectoryLocations`, `TrajectoryOutlierFlags`, `TrajectoryRecursions`, `TrajectoryRestriction`, `TrajectorySegments`, `TrajectorySplit`, `TrajectoryStaypoints`, `TrajectoryThinning`, `TrajectoryTrackRelativeTimes`, `UtilizationVolumeIsopleths` |
| Recipes | `BuildingAdjacency`, `BuildingTessellation`, `BuildingTessellationReference`, `CellFlowTopology`, `ContiguityOrderCorrelogram`, `DasymetricRedistribution`, `DrainageNetwork`, `ExcessRisk`, `ExposureByZone`, `InterpolationDiagnostics`, `Isovists`, `LandscapeMetrics`, `LocalBuildingTessellation`, `LocalVariability`, `PatchProximity`, `PopulationSupportAlignment`, `RasterResistanceConnectivity`, `RasterZonalDistribution`, `RegionalRasterChange`, `ResistanceConnectivity`, `SegregationCounterfactual`, `SegregationDecomposition`, `SparseResistanceConnectivity`, `SpatialRate`, `StreetProfiles`, `SurfacePreparation`, `VisibilityGraph`, `ZonalDistribution` |

**The 26 pages unreachable by API-body links from the root.** All are in the sidebar. Filename stems below are relative to `docs/api` or the named subdirectory.

| Directory | Page stems |
|---|---|
| API root | `computation-views`, `gpu-geometry-refinement` |
| Network | `flow-models-mob10`, `network-time-geography-mob09`, `point-process` |
| Raster | `contrast-equalization`, `lagrangian-diagnostics-mob06`, `raster-change-monitoring`, `raster-eof-analysis` |
| Spatial | `analysis-experience`, `change-points-and-expectation-based-scan`, `colocation-category-dependence-adaptive-density`, `coverage-expansion`, `gaussian-random-field`, `geographical-detector-redcap-entropy`, `geometry-refinement-lenses`, `gpu-analysis-contiguity-statistics`, `healpix-cells-dggs-json`, `home-range-and-recursion-mob07`, `interpolation-lenses`, `movement-coverage`, `point-process`, `sinkhorn-divergence`, `space-time-hawkes`, `spatial-frames` |
| Terrain | `terrain-processing` |

## Quick wins

- [ ] Correct the four calculation mappings in DST-01 through DST-04. Put the mode and output directly beside the external tool name.
- [ ] Print the three verified import owners, replace the two stale geometry availability claims, and repair the seven misleading fragments.
- [ ] Give the four family Catalog sections ordinary working links. Put **Find an operation** beside **Run your first computation** at the top of the API landing page.
- [ ] Relabel the spatial index as selected coverage until its public-set difference is empty.
- [ ] Replace the 125 self-links with domain references. Start with alternative routes, point-to-cell and raster hole filling as three distinct fixtures.
- [ ] Move the four spectral-index anchors to their contracts, or link them explicitly to **Formula and band contracts**, **Usage** and **Calibration, masks, and denominators**.
- [ ] Repair the validation catalog, spatiotemporal review and Kepler point-cloud destinations. Verify emitted HTML and the receiving route together.

## Larger proposals

**Use a single task record for navigation.** Start with the fifteen journeys above. Each record needs a user question, GIS/tool aliases, qualified export, required mode, output description, section target and any material parity caveat. A task may offer two clearly distinguished answers, such as one owner versus every containing polygon, or filling a binary hole versus interpolating a missing measurement. Generate the task chooser and both projections of the cross-reference from these records. S for the initial records, M to cover the existing tables.

**Use one operation-entry template.** In reading order:

1. **Question and result.** One sentence with the analysis unit and what one output row means.
2. **Import and execution forms.** Copyable package import, CPU function or explicit availability, and links to named graph forms.
3. **Definition and applicability.** Formula or reference, units, coordinate model, and the mode needed for external-tool parity.
4. **Inputs and settings.** Port, required/optional, element type, lanes/order/stride, rows, units, mask policy and construction-time versus per-frame settings.
5. **Outputs and status.** For each port, give its shape, readable length, tail/sentinel policy, and which status must be checked before use. State when fixed-size output or absence of a status makes a field inapplicable.
6. **Smallest checkable example.** Use the shared lifecycle; give concrete input, expected result, and the line that reads it.
7. **Limits and alternatives.** State the limit at which the answer changes, becomes incomplete or is refused. Link the neighboring operation and explain when to choose it.

`D/gpu-raster/fractional-zonal-distribution.md:13,47,85,99` already approximates this order. The HAND page is a good model for explaining meaning: it distinguishes multiple valid downstream streams, missing drainage and unfinished routing, and explicitly answers the CPU-twin question.

**Organize the sidebar by task within data family.** Suggested spatial groups are **Join and select**, **Measure and construct geometry**, **Build neighborhoods**, **Test spatial patterns**, **Estimate and validate**, **Analyze change over time**, and **Analyze tracks**. Keep an alphabetical export index for an engineer arriving with a class name. The existing twelve-question ontology is a source of browse labels; the direct data-kind and package indexes remain useful.

| Existing pages | Recommended destination |
|---|---|
| Network `flow-models-mob10`, `map-matched-edge-volumes-mob05`, `network-time-geography-mob09` | Rename to `origin-destination-flow-models`, `map-matched-edge-volumes`, `network-time-geography`. |
| Raster `lagrangian-diagnostics-mob06`; spatial `home-range-and-recursion-mob07` | Rename to `flow-map-and-particle-diagnostics` and `home-ranges-and-recursions`. |
| Network `expansion` | Split the chooser into **Network structure and resilience**, **Routes**, and **Network topology**; link the existing detailed pages. |
| Raster `expansion` | Move raster reconstruction and hole filling under **Raster morphology**, walking costs under **Cost surfaces**, drainage under **Terrain hydrology**, hulls and triangulation under **Geometry**. |
| Spatial `coverage-expansion` | Merge its useful task descriptions into the family choosers. |
| Spatial `interpolation-expansion`, `statistical-expansion`, `expansion-recipes` | Use **Interpolation and simulation**, distribute statistics by question, and merge recipe entries into the task groups with an execution-form label. |
| Raster, terrain and spatial `expansion-lenses` | Attach recording links to the corresponding operation; retain a browsable recordings index. |

Changing titles and adding canonical destinations is additive. I recommend keeping existing URLs and fragment aliases when moving content. This gives the pre-alpha library a coherent navigation scheme without requiring any package rename, constructor migration or capability-ID decision.

## Not read

I read the coverage-expansion overview, nodata-filling guide, HAND guide, website docs registry and docs route in full. I inspected the renderer, qualified class-target checker, ontology search, relevant source declarations and selected constructor paths. I enumerated all API files, headings, rendered local fragment targets, public classes, scoped spatial producers and recipes. I sampled the long family references, both cross-reference presentations, recording sections and the selected 60 class destinations. I checked candidates against the earlier synthesis, area reports, API/docs findings, direction report and decision list.

I did not review all 761 classes for semantic accuracy, run a website build, visit a deployed site, fetch any external reference, or execute a GPU shader. The numeric smoothing evidence is a host-oracle result. The Moran and geodesic probes establish what the host declares. The proposed device probes distinguish those observations from numerical GPU qualification. No files were written.