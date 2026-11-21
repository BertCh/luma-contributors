# Entry points, barrels and import paths

## Summary

Analysis has **11 public entry points**. The reported 55 is the workspace total after wildcard expansion: 11 analysis, 43 lenses and one kepler. The 200 analysis barrels expose 5,416 distinct declarations.

The family imports work well once the reader knows the path. All 112 named bindings in the 56 explicit analysis imports I found in API code fences resolved from the stated entry. The private testing barrel stays outside every public entry.

The most valuable change is to make `import` and `require` agree on object identity and behavior. Today a valid two-chunk vector becomes one apparent chunk, recording silently loses kernel metadata, and a valid operation fails lowering when producer and consumer use different module formats. All three reproduced without a GPU.

The biggest first-use gap is finding the import in the first place. Twenty-two of 70 family pages with a local class heading omit the family package path. The dedicated spatial ontology entry works, but 69 code imports still obtain its names through a barrel marked for removal.

Two small export changes would remove disproportionate friction: publish the named types used by public signatures, and put the space-time variogram fitter and bin-lag helper beside the dataframe contributor that produces their input.

## Census

`S/` means `modules/analysis/src/`, `D/` means `docs/api/`. Entry names below follow `@luma-contributors/analysis`. Counts are from the r19 source tree. Runtime probes used Node 22.23.4 and the existing workspace outputs. I checked source mechanisms separately because implementers were editing the checkout.

| Entry | Classes | Functions | Other values | Types | Total | Same declaration at another entry | Lenses / kepler source imports |
|---|---:|---:|---:|---:|---:|---:|---:|
| `.` | 4 | 22 | 3 | 63 | 92 | 92, all of `computation-record` | 0 / 0 |
| `computation-record` | 4 | 22 | 3 | 63 | 92 | 92, root | 97 / 1 |
| `gpu-spatial-analysis` | 434 | 491 | 770 | 1,322 | 3,017 | 196, all of `ontology` | 270 / 103 |
| `gpu-spatial-analysis/ontology` | 1 | 61 | 40 | 94 | 196 | 196, spatial | 0 / 4 |
| `gpu-network` | 75 | 65 | 62 | 197 | 399 | 0 | 49 / 8 |
| `gpu-terrain` | 68 | 60 | 87 | 198 | 413 | 1, `GPURasterCellSizeMode` | 47 / 2 |
| `dataframe-analysis` | 48 | 112 | 105 | 162 | 427 | 0 | 47 / 10 |
| `raster-analysis` | 113 | 90 | 163 | 266 | 632 | 0 | 99 / 14 |
| `pointcloud-analysis` | 11 | 26 | 57 | 38 | 132 | 0 | 0 / 6 |
| `geospatial-analysis` | 2 | 2 | 3 | 3 | 10 | 0 | 0 / 2 |
| `analysis-core` | 6 | 92 | 54 | 144 | 296 | 1, `GPURasterCellSizeMode` | 182 / 69 |
| **Distinct declarations** | **761** | **959** | **1,304** | **2,392** | **5,416** | | |

**How I counted.** `rg --files modules/analysis/src -g index.ts | wc -l` returned **200**. I read all 200 with the TypeScript parser, derived the 11 starting files from `package.json#exports`, called `getExportsOfModule` and followed aliases to their declaring symbols. There are **5,706 entry/name pairs**, **5,413 distinct spellings**, and **5,416 originating declarations**. Of the barrels, 173 use only named re-exports, 10 only stars, and 17 both. The source scan found 2,171 export declarations in those barrels, 318 of them stars.

- **Overlap by name.** Four spellings resolve to *different* declarations across entries: `addVisibilityGraphRecipe`, `GPUVisibilityGraphCSRProps`, `GPUVisibilityGraphCSROutput` and `GPUVisibilityGraphCSRStatus`. One declaration has two public spellings within one entry, `validateUnitVector` and `assertUnitVector`. The latter is the compatibility alias already covered by `KNOWN(DGGS-112)`.
- **Consumer check.** I parsed static imports and re-exports in lenses and kepler `src`, examples, website, tests and scripts. The last column counts declarations, including type imports. Across all 116 API Markdown pages I found **56 explicit static analysis imports on 30 pages, naming 112 bindings, with zero wrong entry/name pairs**. This measures import resolution. It does not certify the rest of each example.
- **Types with classes.** Of **737 visible constructor parameter occurrences** directly naming a local, non-generic type, **736** name a publicly reachable declaration. The exception is `CPUTerrainCatchmentSummaryOperationProps`. Separately, **eight leaf-exported types** directly named as a public function argument or return type have no public entry. They are listed under ENT-08.
- **Private boundary.** Every export reached through a non-testing `index.ts` reaches at least one public entry. The testing barrel has **32 names**, none reachable through a public entry. The package explicitly excludes `src/testing` and `dist/testing`. I found **65 lenses-test and eight kepler-test imports** of `/testing`, plus **14 lenses-test imports** of `/test/...`. These are workspace test dependencies.
- **Implementation details.** I found zero public originating declarations explicitly marked `@internal`. Lack of a documentation hit alone is weak evidence for removal: the same filter catches CPU twins. ENT-13 identifies three shader generators with stronger evidence and distinguishes them from shader exports used by consumers.
- **Family versus dependency.** The seven domain entries follow their source folder names. They are family boundaries, with substantial differences in eager dependency cost. The documentation routes for `raster-analysis`, `dataframe-analysis` and `geospatial-analysis` are respectively `gpu-raster`, `gpu-dataframe-analysis` and `geospatial`. Three more entries serve cross-cutting roles: `analysis-core`, `computation-record` and ontology. The root forwards records.

## Proposed convention

Give every supported declaration one canonical import path by user task. Put each contributor’s named inputs, outputs, statuses, documented limits and direct preparation/readback helpers at that path, with explicit aliases where another family consumes the same contract.

Extend the existing manifest-based declaration inventory to check those relationships, documentation import lines, workspace-only paths, and a small set of behavioral probes that exchange real objects between `import` and `require`.

## Findings

| ID | Kind | Reach | Change | Cost | Where | One line |
|---|---|---|---|---|---|---|
| ENT-01 | W | ORDINARY | BREAKING | S | Geographic viewport bounds | Ordinary packed XY positions can silently produce plausible bounds for the wrong points. |
| ENT-02 | W | UNUSUAL | ADDITIVE | M | Core graph views | A two-chunk ESM vector becomes one apparent chunk in CommonJS. |
| ENT-03 | W | UNUSUAL | ADDITIVE | M | Core and computation record | A recorder can be enabled and still lose every kernel created through the other module format. |
| ENT-04 | S | UNUSUAL | ADDITIVE | M | Operation lowering | The published lowering rejects a genuine operation created through the other export condition. |
| ENT-05 | S | ORDINARY | ADDITIVE | S | Local kriging | Two documented route-selection limits cannot be imported. |
| ENT-06 | C | ORDINARY | ADDITIVE | S | Space-time variograms | The dataframe producer’s lag helper, fitter and six model types live exclusively at the spatial entry. |
| ENT-07 | E | ORDINARY | ADDITIVE | S | Terrain, network, spatial, raster | Four named constructor, output or status contracts stop at their leaf files. |
| ENT-08 | E | ORDINARY | ADDITIVE | S | Five entries and core | Eight directly exposed function-signature types require type extraction or a private source path. |
| ENT-09 | D | ORDINARY | ADDITIVE | S | API navigation | Twenty-two family pages omit the canonical import path; the two entry tables also omit real entries. |
| ENT-10 | E | ORDINARY | BREAKING | M | Spatial ontology | Documentation and 69 code imports still teach or exercise a deprecated route with a much larger load. |
| ENT-11 | T | ORDINARY | ADDITIVE | S | Workspace path mapping | Production TypeScript resolution accepts plausible package paths that fail in an installed consumer. |
| ENT-12 | M | ORDINARY | ADDITIVE | S | Recipe imports | One capacity-planning import traverses a 111-module barrel and loads Arrow. |
| ENT-13 | C | UNUSUAL | BREAKING | S | Spatial regression | Three internal-use shader generators are public API with zero identified public-import callers. |
| ENT-14 | C | ORDINARY | ADDITIVE | S | Visibility graph | Three identically spelled public types describe different contracts at two entries. |

### ENT-01: Geographic viewport bounds silently reinterpret XY as XYZ

**Where.** `S/geospatial-analysis/viewport-bounds/geographic-viewport-bounds.ts:18`, `D/geospatial.md:13`. The family page’s export table lists the two GPU classes and mentions the viewport source folder at line 22. It gives no callable signature for this public host helper.

**Evidence — EXECUTED.** Through the public entry, `getGeographicViewportBounds([Float32Array.of(10,20,11,21,12,22)])` returned `[9.45,11.6,21.55,20.4]`. The same call with `2` returned `[9.9,19.9,12.1,22.1]`, the padded extent of the three XY points. Six words pass both stride checks. The default drops the last latitude and introduces a longitude of 21.

**Now.** `getGeographicViewportBounds([packedXY])` type-checks and can return an extent that excludes a real vertex.

**Wanted.** Require `positionSize: 2 | 3` and add the function to the geospatial export table with one XY example and the `[west, south, east, north] | null` result. Choosing a different default leaves the same ambiguity for the other layout.

**In-repository callers that change.** **Eight one-argument calls out of 13 actual calls**: four in `modules/analysis/test/geospatial-analysis/geographic-viewport-bounds.node.spec.ts`, two in `modules/kepler/test/trips-browser-fixture.node.spec.ts`, one in `modules/kepler/test/resident-layers.node.spec.ts`, and one in `examples/deck/kepler-compute/src/source-publication.ts:530`. Counted call expressions from the `rg` results, excluding a synthetic source string.

**Value and cost.** GIS callers get bounds for the declared layout. **S.**

### ENT-02: Mixed-module graph views lose their chunk boundaries

**Where.** `S/analysis-core/kernels/gpu-contributor-utils.ts:174` and `:181`, exposed at `S/analysis-core/index.ts:20`. `rg -o 'instanceof Graph(VectorView|DataView)' modules/analysis/src` found 62 vector and 19 data-view checks across 42 files. The demonstrated failure is in the shared chunk reader.

**Evidence — EXECUTED.** With an ESM `GPUCommandGraph` on a null device, I imported two one-row `float32` chunks as vector `two`. I passed the *same vector* to both public versions of `getGraphViewChunks`.

| Reader | Returned IDs | `validateGraphViewsBelongToGraph` on the owning graph |
|---|---|---|
| ESM | `['two-chunk-0', 'two-chunk-1']` | Accepted |
| CommonJS | `['two']` | `Cannot read properties of undefined (reading 'graph')` |

An empty vector similarly returned chunk counts `0` and `1`. The two upstream `GraphVectorView` constructors compared unequal.

**Now.** `require('@luma-contributors/analysis/analysis-core').getGraphViewChunks(vector)` can return the vector itself as a purported `GraphDataView`.

**Wanted.** Use a cross-format view discriminator based on the published vector/data shape at the shared boundary and use it consistently in consumers that accept both kinds. Pin the two ordered chunk IDs, empty-vector case and ownership validation in the packed consumer.

**In-repository callers that change.** None. **Value and cost.** Graph owners can safely pass chunked inputs to a CommonJS adapter or plug-in. **M.**

### ENT-03: Kernel recording is local to the module format

**Where.** `S/computation-record/computation-recorder.ts:65`, `S/analysis-core/kernels/wgsl-kernel-nodes.ts:586`, `:732` and `:761`, `S/computation-record/describe-command-graph.ts:128`. The packed checker compares exported names at `scripts/check-packed-consumer.mjs:341`.

**Evidence — EXECUTED.** I called `new GPUComputationRecorder().recordKernels(graph)` through ESM. On that graph I declared two one-invocation, zero-binding nodes with `createWGSLKernelNode`, one per format. ESM reported recording enabled and returned the ESM node’s record. CommonJS reported recording disabled, returned its node normally, and the ESM reader returned `undefined` for that node. No compilation or shader execution was involved.

**Now.** `getWGSLKernelNodeRecord(graph, node.id)` can silently omit a kernel from a graph whose recorder was enabled before declaration.

**Wanted.** Make both entry conditions use one instance of the recording registry, or store a versioned shared registry on the graph. Test both writer/reader directions and assert a real node’s ID, source and dispatch.

**In-repository callers that change.** None. **Value and cost.** Lens and provenance consumers retain the execution evidence they requested. **M.**

### ENT-04: The public lowering rejects an operation from the same installed package

**Where.** `S/analysis-core/operations/gpu-contributor-operation.ts:208`, `S/analysis-core/operations/index.ts:7`. Direct subclasses by entry are spatial 80, network 17, raster 13, terrain 11 and dataframe 7.

**Evidence — EXECUTED.** I constructed a CommonJS `GPUContributorOperation` named `probe`, with no ports and a contributor factory returning an empty node list. The CommonJS lowering reached the context’s `lower()` callback. The ESM `GPU_CONTRIBUTOR_OPERATION_LOWERING` threw `probe is not a GPUContributorOperation` before reading any port. The type string was identical.

**Now.** A host that supplies `lowerings: [GPU_CONTRIBUTOR_OPERATION_LOWERING]` can reject operations from a dependency solely because it used `require`.

**Wanted.** Share the operation implementation across entry conditions, or use a versioned cross-format brand and validate the lowering contract. Keep rejecting objects that merely claim the operation type string. Add a mixed-format lowering fixture.

**In-repository callers that change.** None. **Value and cost.** All 128 concrete operation subclasses can cross a module-format boundary. **M**, partly shared with ENT-03.

### ENT-05: Local kriging documents four limits and exports only two of them

**Where.** `D/gpu-spatial-analysis/interpolation-expansion.md:42`, `S/gpu-spatial-analysis/spatial-interpolation/gpu-local-kriging.ts:81`–`:97`, `S/gpu-spatial-analysis/spatial-interpolation/index.ts:199`.

**Evidence — EXECUTED.** A namespace lookup on the spatial entry returned `32` and `64` for the neighbor and support limits, then `undefined` for both `GPU_LOCAL_KRIGING_BRUTE_FORCE_MAXIMUM_SAMPLE_COUNT` and `GPU_LOCAL_KRIGING_MAXIMUM_BRUTE_FORCE_WORK`. Named imports from current source produced TS2724 and TS2305. The defining values are `4096` and `2 ** 32`.

**Now.** Following the page to write `sampleCount <= GPU_LOCAL_KRIGING_BRUTE_FORCE_MAXIMUM_SAMPLE_COUNT` requires copying a number the library already names.

**Wanted.** Export both constants from the spatial-interpolation barrel alongside the existing two limits. Add a public-entry assertion for their values. This is separate from the network event-limit decision `KNOWN(NET-110)`.

**In-repository callers that change.** None. **Value and cost.** Callers can plan or explain a kriging route using the library’s thresholds. **S.**

### ENT-06: A dataframe variogram cannot import its fitter from the producer’s entry

**Where.** `S/dataframe-analysis/pair-statistics/index.ts:97`, `S/gpu-spatial-analysis/spatial-interpolation/index.ts:25`, `D/gpu-spatial-analysis/interpolation-expansion.md:92`. The r19 dataframe guide names the fitter at `D/gpu-dataframe-analysis.md:1014`. The KI guide still describes the producer at `:1897` and omits the fitter reference there.

**Evidence — EXECUTED.** From the workflow description I would write:

```ts
import {
  GPUSpaceTimeVariogram,
  getGPUSpaceTimeVariogramBinLags,
  fitSpaceTimeVariogramModel
} from '@luma-contributors/analysis/dataframe-analysis';
```

The class resolves. The other two names produced TS2724 and TS2305. Runtime lookup returned `undefined` for both at dataframe and `function` for both at spatial. All nine model-related declarations originate in `dataframe-analysis/pair-statistics/space-time-variogram-model.ts`.

**Wanted.** Publish `getGPUSpaceTimeVariogramBinLags`, `fitSpaceTimeVariogramModel`, `evaluateSpaceTimeVariogramModel`, `SpaceTimeVariogramFit`, `SpaceTimeVariogramFitOptions`, `SpaceTimeVariogramInput`, `SpaceTimeVariogramModel`, `SpaceTimeVariogramModelType` and `SpaceTimeVariogramStructure` from dataframe. Keep the spatial names as explicit aliases for kriging consumers. Give the dataframe producer page the single import above.

**In-repository callers that change.** None. **Value and cost.** A user can go from empirical bins to a fitted model through the entry that produced the bins. **S.**

### ENT-07: Four class and port contracts stop one barrel too soon

**Where and evidence — SOURCE, EXECUTED.** A virtual consumer importing each exact name from its owner received TS2724 in all four cases.

| Missing export | Defining source | Public declaration that uses it |
|---|---|---|
| `CPUTerrainCatchmentSummaryOperationProps` | `S/gpu-terrain/terrain-processing/cpu-terrain-catchment-summary-operation.ts:58` | Constructor at `:307`. The documented CPU `output.status` adds fields beyond the GPU operation props. |
| `GPUNetworkIsochronesOutput` | `S/gpu-network/network-isochrones/gpu-network-isochrones.ts:210`, same line in KI | `GPUNetworkIsochronesProps.output` at `:311` |
| `GPUSpatialScanStatisticOutput` | `S/gpu-spatial-analysis/scan-statistics/gpu-spatial-scan-statistic.ts:117`, KI `:122` | `GPUSpatialScanStatisticProps.output` at `:224` in both copies |
| `GPURasterPatchLabelsStatus` | `S/raster-analysis/raster-patches/patch-labels.ts:57` | Exported `GPURasterPatchLabels.status` at `:53` |

**Now.** `ConstructorParameters<typeof CPUTerrainCatchmentSummaryOperation>[0]` works. The obvious named import fails, and TypeScript suggests `TerrainCatchmentSummaryOperationProps`, which omits the CPU-specific status mapping.

**Wanted.** Add the four `export type` bindings to `terrain-processing/index.ts`, `network-isochrones/index.ts`, `scan-statistics/index.ts` and `raster-patches/index.ts`, respectively. The expected-scan sibling already exports `GPUExpectationBasedSpaceTimeScanOutput` from the same barrel.

**In-repository callers that change.** None. **Value and cost.** Typed adapters can declare the complete contract without extracting it through a constructor or optional property. **S.**

### ENT-08: Eight public function-signature types have no supported import

**Where and evidence — COUNT, EXECUTED.** I inspected directly named local argument and return types on public functions, including overloads. These eight are exported by their defining files and absent from every package entry. Virtual consumer imports produced TS2305 or TS2724.

| Type to expose | Public functions that directly name it | Defining source |
|---|---|---|
| `CPUPolygonRows` | `computeLabelPointOnCPU`, `computeShapeDescriptorsOnCPU`, `computeMaximumInscribedCircleOnCPU` | `S/gpu-spatial-analysis/shared/cpu-polygon-rows.ts:6` |
| `GPUMapColoringPriority` | `computeMapColoringOnCPU` | `S/gpu-spatial-analysis/map-coloring/gpu-map-coloring.ts:66`, KI `:67` |
| `RasterGridAffine` | `getRasterResistanceConnectivityParameterValues` | `S/analysis-core/contracts/raster-grid.ts:54`. Also used by public core context options at `:305`. |
| `CPURasterOperationBand` | `computeRasterWatershedSegmentationOnCPU`, `computeTerrainPeakProminenceOnCPU` | `S/raster-analysis/raster-grid/raster-operation-band.ts:34`, KI `:35` |
| `CPUCompositeDimensions` | `computeRasterMedoidCompositeOnCPU` | `S/raster-analysis/raster-time-series/cpu-composite-observations.ts:8` |
| `VariogramModelFitDiagnostics` | `fitVariogramModel` | `S/dataframe-analysis/pair-statistics/variogram-model.ts:93` |
| `PublishNodeProps` | `createPublishNode` | `S/analysis-core/kernels/wgsl-kernel-nodes.ts:921` |
| `WGSLDispatchGate` | `createWGSLDispatchGate` | `S/analysis-core/kernels/wgsl-kernel-nodes.ts:160` |

**Now.** A geometry adapter needs `type PolygonRows = Parameters<typeof computeLabelPointOnCPU>[0]` to name the input shared by three functions. A wrapper around `fitVariogramModel(..., {diagnostics: true})` cannot import the name of its return type. The widening-overload issue was being corrected during this review. The remaining export omission builds on `KNOWN(PTS-113)`.

**Wanted.** Export each type at its function’s entry. Give `CPURasterOperationBand` a canonical raster home plus a terrain type alias, and `RasterGridAffine` a canonical core home plus a spatial type alias. The three file-private scalar-format unions used by `getCPUOperation{Float32,Int32,Uint32}Rows` can stay implementation-local unless the project chooses public format names.

**In-repository callers that change.** None. I also checked the KI versions of the map-coloring, maximum-inscribed-circle and watershed CPU signatures. They still name the listed types. **Value and cost.** Library consumers can publish stable wrapper declarations using ordinary package imports. **S.**

### ENT-09: Correct imports are missing where readers land

**Where.** `D/README.md:270`, `modules/analysis/README.md:28`, `D/gpu-raster/raster-patches.md:8`, `D/gpu-raster/focal-pairs.md:1`. `KNOWN(DOC-101)` corrected two wrong paths. This finding concerns pages that give no owning path at all.

**Evidence — COUNT, EXECUTED.** Among 70 pages in six family directories with a heading naming a class exported by that family, **22** omit the literal canonical family specifier. The result was identical after substituting KI copies for staging-owned pages. Five of the 19 qualifying raster pages omit it. The patch page names `@luma.gl/graph/gpu-raster` for upstream classes. A plausible newcomer import, `import {GPURasterPatchMetrics} from '@luma-contributors/analysis/gpu-raster'`, fails with `ERR_PACKAGE_PATH_NOT_EXPORTED`.

The API index lists nine of 11 entries, omitting root and ontology. The package README lists ten, omitting root. The root really does export the 92 record names. `typeof root.GPUPointDensity` was `undefined`.

**Wanted.** Generate a short `Import from` line from each class’s manifest owner. Show all 11 entries in both entry tables, labeling `.` as the `computation-record` alias and the ontology path as the catalog import. Preserve the existing documentation URLs and inbound links.

**In-repository callers that change.** None. **Value and cost.** Someone arriving from search can import the class directly from the page they found. **S.**

### ENT-10: Catalog users are directed through a deprecated, expensive barrel

**Where.** `S/gpu-spatial-analysis/index.ts:53`, `D/gpu-spatial-analysis/analysis-experience.md:74`, `D/gpu-spatial-analysis/concepts.md:25` in both r19 and KI, `modules/kepler/src/curated/curated-analysis-mapping.ts:5`.

**Evidence — COUNT, EXECUTED.** The parent re-exports all 196 ontology names under an ordinary `// @deprecated` comment. In fresh Node processes, requiring the dedicated entry loaded **31 analysis files, zero luma.gl files and zero Arrow files**. Requiring the spatial parent loaded **1,193 analysis files, 396 luma.gl files and 134 Arrow files** in the existing build. These are eager CommonJS module counts, not bundle-size or timing estimates.

**Now.** The concepts example imports `queryGPUSpatialAnalysisCapabilities` through the parent. It succeeds, disguising both the cost and the pending removal.

**Wanted.** Update users to the existing `/gpu-spatial-analysis/ontology` path and remove `export * from './ontology/index'` from the parent before alpha. Keep the catalog available independently of the GPU implementation.

**In-repository callers that change.** **69 named code imports**: 33 in kepler source, 26 in kepler tests, 10 in the Kepler example. There is also one named docs import, and two qualified namespace references in `modules/analysis/test/gpu-spatial-analysis/ontology/spatial-analysis-package-ownership.node.spec.ts:17` and `:26`. The 70 named code/docs import declarations span 56 files.

**Value and cost.** Catalog-only hosts avoid loading the analysis implementation and retain the supported path across releases. **M.**

### ENT-11: Workspace resolution admits private imports into production code

**Where.** `tsconfig.json:41`, `S/testing/index.ts:5`, `modules/analysis/package.json:75`.

**Evidence — EXECUTED.** From a synthetic `modules/kepler/src` importer, `ts.resolveModuleName` with the checkout configuration resolved `@luma-contributors/analysis/gpu-spatial-analysis/spatial-interpolation` to its source `index.ts`. Node’s package resolver returned `ERR_PACKAGE_PATH_NOT_EXPORTED`. The same experiment resolved `/testing` locally and rejected it at the package boundary.

**Now.** An IDE can accept `import {GPULocalKriging} from '@luma-contributors/analysis/gpu-spatial-analysis/spatial-interpolation'` in a sibling package. The import fails after packing. I found no such static production import in the sampled consumers. The 87 private imports counted above were all in tests.

**Wanted.** Derive the production TypeScript aliases from the 11 manifest keys. Retain `/testing` and `/test/*` in test resolution. Add a negative consumer fixture for the spatial-interpolation deep path. Keep the private testing barrel and package exclusions as they stand.

**In-repository callers that change.** No production import edits found. The test configuration must continue resolving its 87 existing declarations. **Value and cost.** Contributors catch an invalid dependency at authoring time. **S.**

### ENT-12: A capacity helper loads the entire spatial-weights family

**Where.** `S/gpu-spatial-analysis/recipes/recipe-utils.ts:38`, `S/gpu-spatial-analysis/spatial-weights/index.ts:22`, `S/gpu-network/recipes/drive-time-catchment-recipe.ts:53`, KI `:57`.

**Evidence — EXECUTED.** In fresh processes, requiring the built spatial-weights barrel and reading `getGPUSpatialWeightsCapacityPlan` loaded **111 analysis modules and 134 Arrow modules**. Reading the identical function from its defining module loaded **26 analysis modules and zero Arrow modules**. The source edge in `recipe-utils.ts` imports this one runtime function plus two types from the barrel. The network recipe reaches it in both copies.

**Now.** `require('@luma-contributors/analysis/gpu-network')` reaches polygon-contiguity Arrow preparation through this chain, even for a caller selecting a routing contributor. Other dependency paths may also bring in Arrow.

**Wanted.** Import `getGPUSpatialWeightsCapacityPlan` internally from `../spatial-weights/spatial-weights`. Keep type-only imports explicit. Use defining-module imports for this kind of cross-family implementation dependency. Preserve the public family path for users.

**In-repository callers that change.** One internal import declaration; no public callers. **Value and cost.** Recipe loading sheds an avoidable dependency edge without a new package entry. **S.**

### ENT-13: Three regression shader generators have no identified public consumer

**Where.** `S/gpu-spatial-analysis/spatial-regression/index.ts:109`, `:110`, `:267`. The same three exports remain in KI at `:109`, `:110`, `:268`.

**Evidence — COUNT, SOURCE.** Of nine spatial runtime exports ending in `WGSL`, three have neither an identifier mention under `docs/` nor a named import in the examined consumers: `getCholeskyWGSL`, `getSymmetricEigenvalueRangeWGSL` and `getSpatialWeightsTracesProbeWGSL`. Their defining files are `spatial-regression-solve.ts:38`, `:125` and `spatial-weights-traces-parameters.ts:375` respectively, the last at KI `:379`. The Cholesky generator documents caller-supplied shader declarations. Regression implementations import these helpers directly. The eigenvalue test also imports its leaf directly.

**Now.** Auto-import offers `getCholeskyWGSL` from the same entry as `GPUPointDensity`. Exporting that generator makes generated function names and shader dependencies part of the user-facing surface.

**Wanted.** Remove these three names from the public regression barrel and retain their internal exports. Keep the documented spatial multiplier helpers and the density hexagon shader constants; the latter have real example imports. This extends the public-surface decision in `KNOWN(ERG-09)` with three separately identified names.

**In-repository callers that change.** **Zero public-import call sites** for the three removals. `rg` found the identifiers in 24 analysis source files, including the barrel; their defining-module imports can stay. **Value and cost.** Maintainers can change solver implementation details without a public API migration. **S.**

### ENT-14: Visibility graph status has two incompatible public meanings

**Where.** `S/gpu-network/visibility-graph/visibility-graph.ts:64`, `:71`, `:87`, and `S/gpu-spatial-analysis/urban-morphology/visibility-graph-csr.ts:75`, `:86`, `:98`. Both barrels export all three spellings.

**Evidence — EXECUTED.** Resolving `GPUVisibilityGraphCSRStatus` with the type checker gave five spatial fields: `count`, `requiredCount`, `overflow`, `candidateOverflow`, `invalidCount`. The network type adds `approximation`, `converged`, `iterationLimitReached` and `zeroWeightCycle`. The network implementation itself aliases the imported spatial types to `GPUVisibilityGraphAdjacency*` at line 44.

**Now.** Auto-importing `GPUVisibilityGraphCSROutput` chooses between an adjacency result and a result with betweenness, closeness and integration fields. The existing qualified catalog identity fixes recipe lookup `KNOWN(ERG-08, ERG-103)`; it does not disambiguate TypeScript type suggestions.

**Wanted.** Add `GPUVisibilityGraphAdjacencyCSR{Props,Output,Status}` at spatial and `GPUNetworkVisibilityGraphCSR{Props,Output,Status}` at network, and use the qualified names in examples. Keep the existing type names as aliases. No implementation duplication is needed.

**In-repository callers that change.** None required for aliases. The network lens example at `modules/lenses/src/computation-lenses/scenarios/gpu-network/visibility-graph-recipe.example.ts:6` is a good first migration. **Value and cost.** Adapter authors can recognize the intended result before inspecting its fields. **S.**

## Departures by name

- **Duplicated public declarations.** All 92 names from `computation-record` are forwarded by `S/index.ts:5`. All 196 names from `gpu-spatial-analysis/ontology` are forwarded by `S/gpu-spatial-analysis/index.ts:55`. `GPURasterCellSizeMode` is shared by core and terrain. These three sets account for every cross-entry re-export of the same declaration.
- **Homonyms.** `addVisibilityGraphRecipe`, `GPUVisibilityGraphCSRProps`, `GPUVisibilityGraphCSROutput`, `GPUVisibilityGraphCSRStatus`. The two recipes have distinct catalog identities. The three type pairs are ENT-14.
- **Deep-only contracts to surface.** The two local kriging constants in ENT-05, four constructor/port types in ENT-07, and eight function-signature types in ENT-08. Each has a defining source export. A source export alone grants no installed package import path.
- **Implementation classes to keep private.** Excluding testing, the six leaf-exported classes outside every entry are `RecipeBuilder`, `WardActiveGraph`, `RasterSpectralIndex`, `GPULocalPermutationReplay`, `GPUSnapRoundedNoding` and `PhiloxStream`. I found no reason to expose those classes to close a public-signature gap.
- **Family guide pages missing their owning specifier.** Spatial: `gpu-dorling-cartogram`, `coverage-statistics`, `geodesic-precision`, `movement-semantics`, `expansion-lenses`, `spatial-validation-advanced`, `gpu-analysis-contiguity-statistics`, `geometry-refinement-lenses`, `global-coordinates`. Network: `flow-models-mob10`, `point-process`, `network-time-geography-mob09`, `network-movement-coverage`. Raster: `lagrangian-diagnostics-mob06`, `tile-reprojection`, `expansion-lenses`, `raster-patches`, `focal-pairs`. Terrain: `terrain-processing`, `expansion-lenses`, `operators`, `height-above-drainage`. These are page basenames under their respective API sections.

## Quick wins

- [ ] Export the two local kriging limits and twelve named types in ENT-05, ENT-07 and ENT-08. Assert their availability through public entry imports.
- [ ] Publish the nine space-time variogram declarations at dataframe and add the producer-to-fit import block.
- [ ] Generate the family import line for the 22 listed pages. Complete both entry tables and document `getGeographicViewportBounds`.
- [ ] Replace the one runtime spatial-weights barrel import in `recipe-utils.ts`.
- [ ] Add the six unambiguous visibility type aliases.
- [ ] Keep test-only aliases in test configuration and check a negative production deep import.

## Larger proposals

1. **Keep the seven current domain paths.** They match their source folders, and the census shows substantial real usage. Generate import guidance from the manifest; a wholesale rename to match three documentation routes would impose migrations without resolving the missing contracts. Keep `analysis-core` for contributor authors, `computation-record` for records, and its documented root alias. Keep `testing` private.
2. **Use one supported catalog route.** Migrate the counted imports, then remove the parent’s deprecated ontology re-export before alpha. The existing `/gpu-spatial-analysis/ontology` entry already delivers the dependency separation. Adding another `/catalog` spelling would require a second migration decision.
3. **Make interoperability a release property.** Keep separate ESM and CommonJS consumer checks and add tests that pass vectors, operations and recorded graphs *between* them. A shared implementation can resolve local class and registry identity. Upstream view objects also need a cross-format discriminator. The three host reproductions provide acceptance criteria.
4. **Require viewport layout before alpha.** Eight call-site edits buy an unambiguous public contract. Remove the three unused regression WGSL exports in the same pre-alpha cleanup; their public-import migration count is zero.

## Not read

I parsed the full text of all 200 barrels and followed the manifest exports. I read the manifest, root and selected family barrels, named leaf declarations, recording registry, lowering, chunk reader, viewport helper, affected guide passages and packed export checks directly. I checked all seven staging-owned barrel counterparts and the KI copies relevant to individual findings. For example, r19’s scoring-rules page still suggests `./scoring-rules/index`, but the KI page already corrects that instruction, so I excluded it from the findings.

I did not read every contributor body, every consumer call or every API page line by line. The hidden-signature census covers direct named arguments and returns; deeply nested conditional and mapped types could expose more. Existing `dist` module-load counts describe native CommonJS and exclude bundler tree-shaking. I ran no build, test suite, browser, network request or GPU computation, and wrote no files.