# Operator catalog, ontology, coverage and API docs

The catalog resolves all **554 records** to exports in their declared packages. The most important gap is in the reverse direction. Two public packages export different implementations of `addVisibilityGraphRecipe`, but the catalog and its completeness test identify them by name alone. A second discrepancy affects **63 operators** that can be found by name but disappear when a caller browses their capability families.

I also reproduced a wrong result in the public validity classifier, found a generated Parameters section advertising settings the caller cannot supply, and verified that the forest inference page understates the supported row count by a factor of 256.

## What I checked

Reviewed the working tree at `bdc715ea` on 2026-11-17. I made no file changes and ran no builds, typechecks, tests, GPU workloads, or documentation generators. The export census followed the current source barrels with `typescript.createSourceFile` for syntax parsing only. It did not load `dist` or construct a TypeScript program.

`O/` below means `modules/analysis/src/gpu-spatial-analysis/ontology/`, `T/` means `modules/analysis/test/gpu-spatial-analysis/ontology/`, and `D/` means `docs/api/`. Line references use those prefixes.

I read the all-entry catalog assembly, ontology barrel, fitting-kernel catalog, package-ownership spec, all-entry catalog spec, parameter regeneration spec, and catalog API page in full. I sampled the original catalog and overrides, capability and kind records, annotations, recently added AX code, coverage rows, docs parser and snippet checker. I traced individual claims into the contributor implementation. I searched the ontology specs, `scripts`, and `website/scripts` for catalog checks. I did not independently validate all 554 semantic descriptions, 209 annotation keys, external numerical references, or rendered lens cards.

### Reconciliation

“Heading” counts distinct catalog names occurring as whole identifiers in Markdown headings. It excludes generated `… parameters` headings. “Mention” includes prose, code, and markers. “Matrix symbols” counts distinct catalog names in the final **LIB entry point** column. “Matrix rows” counts rows naming the package. One row names two local packages.

| Analysis entry point | Exported `GPU*` classes | Exported `add*Recipe` | Catalog records | API heading | API mention | Matrix symbols | Matrix rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| `analysis-core` | 6 | 0 | 5 | 0 | 1 | 0 | 0 |
| `computation-record` | 4 | 0 | 4 | 1 | 4 | 0 | 0 |
| `dataframe-analysis` | 41 | 0 | 40 | 34 | 40 | 7 | 7 |
| `geospatial-analysis` | 2 | 0 | 2 | 2 | 2 | 1 | 2 |
| `gpu-network` | 58 | 5 | 45 | 33 | 45 | 11 | 13 |
| `gpu-spatial-analysis` | 353 | 37 | 315 | 290 | 315 | 81 | 80 |
| `gpu-terrain` | 57 | 0 | 46 | 46 | 46 | 10 | 10 |
| `pointcloud-analysis` | 11 | 0 | 11 | 11 | 11 | 10 | 7 |
| `raster-analysis` | 99 | 0 | 86 | 82 | 86 | 20 | 19 |
| **Total** | **631** | **42** | **554** | **499** | **550** | **140** | **138 package-row incidences** |

The differences have identifiable causes.

- **Classes.** Of 631 `GPU*` classes, 502 are catalog records. The other 129 are 128 `…Operation` classes and the `GPUContributorOperation` base. I found no missing `getCommandNodes` contributor class. Operation-form discovery already has a home in `KNOWN(CPU-05 / INT-03)`. `dev-docs/roadmap.md:351`, `modules/analysis/src/analysis-core/operations/gpu-contributor-operation.ts:94`.
- **Functions.** The 42 package-qualified recipe exports have 41 distinct names. The catalog contains those 41 names plus 11 other functions, giving `502 + 41 + 11 = 554`. The 11 are `addClockEncounters`, `addTrajectorySortCheck`, `getHealpixCellBoundary`, `preparePolygonContiguityTopology`, `computePolygonContiguityOnCPU`, `aggregateInteger64OnGpu`, `aggregateInteger64Range`, `aggregateDecimal128OnGpu`, `aggregateDecimal128Column`, `scanWideNumericColumnOnGpu`, and `diagnoseFloat32Narrowing`. The catalog intentionally includes some CPU preparation and readback operations. `O/reused-analysis-catalog.ts:7`, `O/spatiotemporal-review-catalog.ts:902`.
- **Coverage.** The matrix headline is arithmetically correct. It has **144 distinct rows, 137 implemented, one partial, six excluded**. All 145 distinct local package/symbol references in the final column resolve. Five are public helpers outside the catalog: `addRasterRegionGraphTiles`, `fitSpaceTimeVariogramModel`, `prepareGPUNetworkFlowDemands`, `prepareGPUNetworkRestrictionStates`, and `prepareGPUTrajectoryObservations`. The matrix measures assessed reference functions, so the 414 catalog records outside that column are not 414 implementation defects. `dev-docs/coverage-matrix.md:7`, `:80`, `:119`, `:184`, `:189`, `:198`.
- **Docs.** Of 554 records, 499 have a non-Parameters API heading, another 51 have an API mention, and four fitting helpers have their documentation under `modules/analysis/docs/` instead. Those four do have real headings. `modules/analysis/docs/fitting-kernels.md:16`, `:29`, `:95`, `:107`.

<details>
<summary>All 55 catalog records without a non-Parameters heading in docs/api</summary>

| Entry | Names |
|---|---|
| `analysis-core` | `GPUAdamStep`, `GPUFittingCoordinateEncoder`, `GPUFittingGradientSums`, `GPUFittingMinibatchOrder`, `GPUParameterBuffer` |
| `computation-record` | `GPUComputationRecorder`, `GPUSnapshotArena`, `GPUSnapshotTap` |
| `dataframe-analysis` | `aggregateDecimal128Column`, `aggregateDecimal128OnGpu`, `aggregateInteger64OnGpu`, `aggregateInteger64Range`, `diagnoseFloat32Narrowing`, `scanWideNumericColumnOnGpu` |
| `gpu-network` | `GPUAlternativeRoutes`, `GPUMapMatchedEdgeVolumes`, `GPUNetworkFacilityOptimization`, `GPUNetworkFlow`, `GPUNetworkPointRouting`, `GPUNetworkRationalAccess`, `GPUNetworkStronglyConnectedComponents`, `GPUNetworkStructure`, `GPUNetworkTimeRouting`, `addFacilityScenarioRecipe`, `addNetworkDisruptionRecipe`, `addTrackQualityRecipe` |
| `gpu-spatial-analysis` | `GPUAZPRegions`, `GPUAreaOfApplicability`, `GPUConditionalGaussianSimulation`, `GPULocalRBF`, `GPULocationAllocation`, `GPUMaxPRegions`, `GPUNaturalNeighbor`, `GPUNestedInterpolationValidation`, `GPUPickRegionMask`, `GPURandomPointsOnLine`, `GPURegionMask`, `GPURegionStatisticsReadback`, `GPURoseStatistic`, `GPUSpatialPanelLagMaximumLikelihood`, `GPUTrackSimilaritySearch`, `GPUTrajectoryKalman`, `GPUTrajectoryRestriction`, `GPUTransitionDiagnostics`, `GPUWardRegions`, `addClockEncounters`, `addDasymetricRedistributionRecipe`, `addExcessRiskRecipe`, `addFleetDwellZoneEventsRecipe`, `addSpatialRateRecipe`, `addTrajectorySortCheck` |
| `raster-analysis` | `GPURasterEmbeddingCovariance`, `GPURasterEmbeddingProjection`, `GPURasterStandardizedIndex`, `GPURasterStandardizedIndexCalibration` |

Some have substantial prose already. For example, `D/gpu-network/map-matched-edge-volumes-mob05.md:3` describes the operator directly. `D/gpu-raster/raster-embeddings.md:148` explains the projection, and line 153 has its generated settings heading. Adding a canonical anchor can resolve these cases without commissioning new pages.

</details>

<details>
<summary>Read-only commands, including the complete family, heading, mention, and matrix differences</summary>

```sh
wc -l modules/analysis/src/gpu-spatial-analysis/ontology/{*.ts,*/*.ts}
git log --format='%h %ad %s' --date=short --stat -- modules/analysis/src/gpu-spatial-analysis/ontology
rg -n 'getGPUAnalysisOperator|GPU_SPATIAL_ANALYSIS_(OPERATORS|CAPABILITIES)|/ontology' \
  modules/lenses/src modules/kepler/src website/scripts
rg -n 'ANALYSIS_DOC_PAGES|UPDATE_DOC_PARAMETERS|findClassHeading' modules/analysis/test/analysis-core
```

This source-only probe prints the individual names for all four differences, including all 63 missing family members and all 414 records outside the matrix symbol column.

```sh
python3 -c '
from pathlib import Path
import re
O = Path("modules/analysis/src/gpu-spatial-analysis/ontology")
def read(name): return (O / name).read_text()
def words(lines): return set(re.findall(r"\b[A-Za-z_]\w*\b", "\n".join(lines)))
files = ["spatial-analysis-catalog.ts", "coverage-expansion-catalog.ts", "kepler-analysis-catalog.ts",
         "../spatial-validation/spatial-validation-descriptors.ts",
         "all-entry-points/entry-point-capabilities.ts", "fitting-kernel-catalog.ts"]
families = {n for f in files for a in re.findall(r"\boperators:\s*\[([^]]*)\]", read(f))
            for n in re.findall(r"\x27(\w+)\x27", a)}
st = read("spatiotemporal-review-catalog.ts").split("const SPATIOTEMPORAL_REVIEW_OPERATOR_OVERRIDES:")[1].split("\n};")[0]
catalog = families | set(re.findall(r"^  (\w+):", st, re.M)) | set(re.findall(r"exportName: \x27(\w+)\x27", read("reused-analysis-catalog.ts")))
docs = [p.read_text() for p in Path("docs/api").rglob("*.md")]
heads = words(l for d in docs for l in d.splitlines() if re.match(r"^#{1,6}\s", l) and not re.search(r" parameters\b", l, re.I))
rows = [l.split("|") for l in Path("dev-docs/coverage-matrix.md").read_text().splitlines()
        if re.match(r"^\| [^|]+ \| (Implemented|Excluded|Partial|Planned) \|", l)]
for label, names in [("family omissions", catalog-families), ("heading omissions", catalog-heads),
                     ("no API mention", catalog-words(docs)), ("outside matrix symbol column", catalog-words(row[-2] for row in rows))]:
    print(label, len(names), *sorted(names))
'
```

</details>

### Data model, history and consumers

A **capability** is a stable family with a domain, stage, data kinds, coordinate models, output bound, member names and optional evidence. An **operator** has an export name, package, primary family, kinds, semantics, status fields, readback behavior, tier, limits and evidence. The public data contains names and metadata, without constructor references. The all-entry catalog assembles **261 original operators + 226 entry-point operators + 54 later spatial operators + 9 reused operations + 4 fitting helpers**. There are **92 capability families**. `O/spatial-analysis-ontology.ts:210`, `:302`, `:313`, `O/all-entry-points/analysis-operator-catalog.ts:63`.

The path is historical. The first catalog was spatial. Commit `93228e74` added the all-entry layer there, while tests explicitly preserved the original 261 operators and 49 families for existing consumers. The dedicated ontology subpath exists, and the spatial barrel already labels its re-export deprecated. There is no present-day model requirement for a dataframe or point-cloud descriptor to live under spatial analysis. `T/analysis-operator-catalog.node.spec.ts:76`, `modules/analysis/src/gpu-spatial-analysis/index.ts:53`, `modules/analysis/package.json:34`.

The directory has **27 files and 20,561 lines**. Some of that is executable AX logic. The largest metadata files are `operator-annotations.ts` **3,347**, `entry-point-operator-details.ts` **1,797**, `analysis-operator-kind-records.ts` **1,173**, `spatial-analysis-catalog.ts` **1,136**, and `spatial-analysis-operators.ts` **1,041**. `analysis-data-roles.ts`, at **2,037** lines, is mainly interpretation logic and should be counted separately when estimating a migration. Nine commits touched the directory from November 7–16. Four touched `operator-annotations.ts` from November 10–16. The November 16 landing alone changed 17 ontology files, with 3,131 insertions and 650 deletions. The checked-in descriptions explicitly identify manually copied facts, including the first sentence of class documentation and the file said to enforce each limit. `O/all-entry-points/entry-point-operator-details.ts:7`, `O/all-entry-points/operator-annotations.ts:15`, `O/analysis-experience/analysis-operator-kind-records.ts:46`.

| Consumer | Actual dependency |
|---|---|
| Kepler | The main binding registry still enumerates the legacy array. Point-cloud and network discovery already call the all-entry API. `modules/kepler/src/analysis/catalog-bindings.ts:1056`, `modules/kepler/src/analysis/catalog-pointcloud-entries.ts:16`, `modules/kepler/src/analysis/catalog-network-categorical-discovery.ts:26`. |
| Website | The inventory extractor reads current public symbols and enriches them from the **all-entry** catalog. The website checker independently indexes the resulting inventory by bare symbol name. `website/scripts/extract-inventory.ts:139`, `:145`, `website/scripts/check-catalog.ts:74`. |
| Lenses | I found no direct ontology import in `modules/lenses/src`. Its coverage spec enumerates barrel contributors and recipes, or uses pack operator keys. It is an independent inventory worth preserving as a cross-check. `modules/lenses/test/computation-docs/lens-coverage.node.spec.ts:487`. |
| Docs | The catalog guide documents the lookup API. Parameters blocks come from TypeScript packers and TSDoc. The sidebar comes from a separate JSON file. `D/analysis-catalog.md:49`, `modules/analysis/test/analysis-core/doc-parameters.ts:90`, `website/src/docs.ts:33`. |

### Eight-page reader check

“Complete” means the example supplies its input data, executes the operation, and shows how to obtain the result. An existing device may be an explicit function argument. This is a source-reading assessment.

| API page | Runnable example | Props, outputs, status and limits |
|---|---|---|
| `operation-forms-and-cpu-twins.md` | **Yes.** The pure CPU example supplies its data. The graph version takes a device, runs, reads the result and cleans up. | Named inputs, output formats, status codes, numerical differences and ownership. Best onboarding model. `D/operation-forms-and-cpu-twins.md:55`, `:101`, `:149`, `:179`. |
| `gpu-network/visibility-graph.md` | Numerical worked example and lens, no callable TypeScript example. | Good expected values, approximation, convergence, and import guidance. No full props or output-allocation table. `D/gpu-network/visibility-graph.md:5`, `:12`, `:19`. |
| `gpu-raster/spectral-filter.md` | Fragment. `device`, `graph`, `input`, and output views are free names. | Precise units, periodic boundary, dimension limit, filter choices and invalid-count output. Settings table covers the packer. `D/gpu-raster/spectral-filter.md:7`, `:23`, `:32`, `:66`. |
| `gpu-spatial-analysis/gaussian-random-field.md` | Lens and numerical explanation, no callable TypeScript example. | Clear alternate inputs, output formats, approximation and method-specific limits. The generated table covers settings. `D/gpu-spatial-analysis/gaussian-random-field.md:15`, `:30`, `:43`, `:116`. |
| `forest-inference.md` | Importer commands, no complete inference call. | Excellent view-format and validity tables. Its published storage and sizing instructions conflict with the implementation. `D/forest-inference.md:18`, `:31`, `:43`, `:66`. |
| `geospatial.md` | Two construction fragments. Both depend on existing graph views. | The datum-grid example explains status codes and precision. The buffer example identifies output length and per-frame radius. No full props table. `D/geospatial.md:24`, `:33`, `:59`, `:82`. |
| `pointcloud-analysis.md` | No complete contributor invocation. | The props/output/status-bit/limit bullets are especially easy to use. Some generated sections appear under the wrong contributor. `D/pointcloud-analysis.md:72`, `:77`, `:79`, `:149`, `:352`. |
| `gpu-terrain/terrain-processing.md` | Worked descriptions and lenses, no complete TypeScript invocation. | Strong family-level explanation of hydrology outputs, convergence, device bounds and measured cases. Constructor inputs are dispersed across prose. `D/gpu-terrain/terrain-processing.md:15`, `:30`, `:74`, `:113`. |

## Findings

### CAT-01 Valid uint32 values disappear from result tallies [P0] [correctness]

- **Evidence.** The contributor contract says “`validity` (`uint32`, nonzero is valid).” The public classifier constructs classes for `0` and `1` and checks `tally.has(value)`. An extracted pure-function probe on `Uint32Array.of(0, 1, 2, 0xffffffff)` returned **one valid, one invalid, two unrecognized**. The contract gives **three valid, one invalid**. Kepler consumes both the tally and its encoding score. `dev-docs/contributor-contract.md:42`, `O/analysis-experience/analysis-interpretation.ts:383`, `:453`, `:483`, `modules/kepler/src/presentation/result-presentation.ts:604`.
- **Why it matters.** A caller can pass a contract-valid buffer and receive wrong class counts and an encoding score of 0.5. This is a CPU result reachable through the public API.
- **Proposed change.** For `validity-flag` with `contributor-integer`, normalize every in-range uint32 to zero or one before lookup. Retain strict handling of significance codes, strings, fractions and negative numbers. Add the four-value fixture to both tally and encoding tests. **S, nonbreaking correction.**
- **Confidence.** High.

### CAT-02 Bare export names collapse two distinct public recipes [P1] [architecture]

- **Evidence.** The network recipe dispatches CSR requests to `addVisibilityGraphNetworkCSRRecipe`. The spatial recipe supplies adjacency and degree in CSR mode. The catalog maps only `operator.exportName`, the completeness test asks only `getGPUAnalysisOperator(name)`, and another assertion *requires* bare names to be unique. The contract checker accepts a textual occurrence anywhere in the ontology. `modules/analysis/src/gpu-network/visibility-graph/visibility-graph.ts:136`, `modules/analysis/src/gpu-spatial-analysis/urban-morphology/visibility-graph.ts:118`, `O/all-entry-points/analysis-operator-catalog.ts:81`, `T/analysis-operator-catalog.node.spec.ts:55`, `:73`, `scripts/check-contributor-contract.mjs:1194`.
- **Why it matters.** A network-package catalog query cannot discover the network recipe. Its spatial record even describes the network implementation in its limitation. The website repeats the same key choice. This is the only missing package-qualified producer/recipe found in the barrel comparison. `O/all-entry-points/operator-annotations.ts:2443`, `website/scripts/check-catalog.ts:74`.
- **Proposed change.** Add lookup by `{package, exportName}`, register both implementations, and use the pair for export coverage, inventory joins, and evidence anchors. Preserve the historical spatial result for the one-argument lookup. **M, additive with the compatibility lookup.**
- **Confidence.** High. `KNOWN(STD-13)` covers general registry consistency, but the current row misses the identity collision and its effect on the existing gate.

### CAT-03 Capability browsing omits 63 registered members [P1] [correctness]

- **Evidence.** `getGPUAnalysisCapabilityCatalog()` returns existing family objects. The operator catalog separately appends newer operators without updating those objects. `GPUAnalysisSpacing` declares `spatial-neighborhoods`, whose `operators` list ends at `GPUMapColoring`. Across **92 families**, the arrays enumerate **491 of 554** names. The missing members are exactly the **54** spatiotemporal-review entries and **nine** reused entries. The spec checks that the family ID exists, never that it contains the operator. `O/all-entry-points/analysis-operator-catalog.ts:64`, `:70`, `:87`, `O/spatiotemporal-review-catalog.ts:172`, `:895`, `O/spatial-analysis-catalog.ts:385`, `T/analysis-operator-catalog.node.spec.ts:100`.
- **Why it matters.** Two natural routes through the same public catalog give different inventories. A family picker silently drops operations that a direct lookup finds.
- **Proposed change.** Build the **all-entry** family member lists by grouping finalized records on `primaryCapability`. Keep the frozen historical membership in the legacy API. Assert equality in both directions using qualified identities. **S–M, additive membership correction.**
- **Confidence.** High.

### CAT-04 Public filters turn missing metadata into false answers [P1] [correctness]

- **Evidence.** All **226** records built by `makeEntryPointOperator` receive `status: []` and `readback: 'none'`. Yet `GPUKShortestPaths` explicitly publishes `candidateOverflow`, `converged`, and `iterationLimitReached`. `GPUSnapshotArena.read()` explicitly calls `buffer.readAsync()`. Separately, `GPUGaussianRandomField` accepts a raster grid and produces a sampled field, but its record has no `surface` on either side. `O/all-entry-points/analysis-operator-catalog.ts:38`, `:47`, `:124`, `modules/analysis/src/gpu-network/k-shortest-paths/gpu-k-shortest-paths.ts:635`, `modules/analysis/src/computation-record/gpu-snapshot-arena.ts:145`, `O/spatiotemporal-review-catalog.ts:191`, `D/gpu-spatial-analysis/gaussian-random-field.md:15`.
- **Why it matters.** A status filter misses a contributor whose contract explicitly provides the requested status. Readback-sensitive discovery gives an incorrect answer for the snapshot arena. A surface-kind search misses a documented grid workflow.
- **Proposed change.** Add operator-specific status and readback to the entry detail type. Populate them from public output declarations and explicit adapter metadata. Record grid and point modes for the Gaussian field. Represent unresolved facts explicitly and test these three cases through public queries. **M, corrections additive; exposing an `unknown` readback value needs a type migration.**
- **Confidence.** High.

### CAT-05 The public kind guard rejects `time-series` [P1] [correctness]

- **Evidence.** `time-series` is one of the 15 canonical kinds. `isGPUSpatialAnalysisDataKind` determines membership by scanning historical capability inputs and outputs. Those mention 14 of the 15. The public kind-query validator uses the canonical list and accepts `time-series`. The guard spec loops over the same historical capabilities as its implementation. `O/spatial-analysis-ontology.ts:6`, `O/spatial-analysis-operators.ts:1035`, `O/analysis-experience/analysis-operator-kind-supplements.ts:54`, `T/spatial-analysis-queries.node.spec.ts:217`, `T/analysis-operator-catalog.node.spec.ts:201`.
- **Why it matters.** A UI validating a saved temporal filter can reject a query the catalog itself supports.
- **Proposed change.** Implement the guard directly against `GPU_SPATIAL_ANALYSIS_DATA_KINDS`, use it in both paths, and iterate the canonical tuple in the guard test. **S, nonbreaking correction.**
- **Confidence.** High.

### CAT-06 A lookup exposes writable records shared with every other caller [P1] [architecture]

- **Evidence.** `exportName`, `primaryCapability`, `package`, and `tier` are writable properties in the public type. The lookup returns the actual object stored in a map keyed once at module initialization. The path index caches against array identity. Thus an ordinary TypeScript assignment to a returned record’s `exportName` can leave the old lookup key pointing at a newly named object. `O/spatial-analysis-ontology.ts:314`, `:343`, `O/all-entry-points/analysis-operator-catalog.ts:81`, `:95`, `:100`, `O/analysis-experience/analysis-operator-kind-supplements.ts:135`.
- **Why it matters.** A host decorating a descriptor can alter another consumer’s queries in the same process. The catalog already has a working local pattern for protecting shared records. `O/analysis-experience/analysis-operator-kind-records.ts:1160`.
- **Proposed change.** Publish deeply readonly descriptor types and freeze finalized records, nested evidence and limit records, and catalog arrays once during construction or generation. Add a mutation-isolation regression. **S–M, breaking for callers that edit returned descriptors.**
- **Confidence.** High.

### CAT-07 Generated Parameters can advertise private implementation settings [P1] [docs]

- **Evidence.** The generator assigns a packer to a class if the class’s **source file imports** the packer or its length constant. `GPUAnalysisContiguityStatistics` imports the column-quantiles packer and calls it with fixed `{quantiles: [0.5]}`. Its generated public section nevertheless lists caller settings for `quantiles`, `interpolation`, `filterRange`, and `quantileCount`. The preceding manual paragraph has to explain the internal fixed median. `modules/analysis/test/analysis-core/doc-parameters.ts:145`, `:183`, `modules/analysis/src/gpu-spatial-analysis/spatial-weights/gpu-analysis-contiguity-statistics.ts:75`, `:240`, `D/gpu-spatial-analysis/analysis-experience.md:21`, `:29`.
- **Why it matters.** The generated section is reproducible and can pass the sync check while documenting knobs the API never exposes.
- **Proposed change.** Declare the public packer-to-port relationship explicitly, including shared packers. Generate only from that relationship. Add a negative fixture for a composite that calls a child’s packer internally. **M, nonbreaking.**
- **Confidence.** High.

### CAT-08 Generated blocks can stay under the wrong operator indefinitely [P2] [docs]

- **Evidence.** The generator prefers headings with backticked class names, then falls back to text mentions. Once a marker exists anywhere, it skips ownership resolution. Refreshing uses the nearest preceding heading to determine depth. Consequently, `GPUPointCloudHeight` parameters sit under `GPUPointCloudSurfaces`, and `GPUPointCloudNeighborhoods` parameters sit under `GPUPointCloudChange`. Both intended owners already have headings. `modules/analysis/test/analysis-core/doc-parameters.ts:445`, `:500`, `:555`, `D/pointcloud-analysis.md:72`, `:139`, `:149`, `:177`, `:299`, `:352`.
- **Why it matters.** Reading the page in order gives the wrong association. Regeneration preserves the mistake.
- **Proposed change.** Give each documented operator a canonical page/anchor and resolve blocks against it on every run. Accept plain and backticked headings. Check that each qualified operator’s block is unique and belongs to its declared section. **S–M, nonbreaking.**
- **Confidence.** High.

### CAT-09 The sidebar and docs checks cover different subsets of the API [P2] [testing]

- **Evidence.** There are **107** API Markdown pages. The TOC lists **98**, all of which exist. The snippets/Parameters allowlist lists **66**. Across all pages there are **184** `ts` fences, of which **17 on 10 pages** fall outside the snippet checker. The checker also supplies unknown free values as `any`. The website loads pages by glob but builds its sidebar from the TOC. `docs/table-of-contents.json:75`, `:91`, `:131`, `modules/analysis/test/analysis-core/doc-program.ts:21`, `modules/analysis/test/analysis-core/doc-snippets.node.spec.ts:222`, `:244`, `website/src/docs.ts:18`, `:33`.
- **All nine sidebar omissions.** `forest-inference.md`, `gpu-network/k-shortest-paths.md`, `gpu-network/visibility-graph.md`, `gpu-raster/contrast-equalization.md`, `gpu-raster/raster-embedding-codecs.md`, `gpu-raster/raster-patches.md`, `gpu-raster/spectral-filter.md`, `gpu-spatial-analysis/gaussian-random-field.md`, `reused-analysis-preparation.md`.
- **Why it matters.** Direct links can open pages a reader cannot find in the sidebar. A typo in an undeclared name can also survive the fragment checker. Only one of the eight sampled pages supplied a complete execution example.
- **Proposed change.** Discover API pages from disk for checking and give each page a sidebar category or explicit omission reason. Distinguish checked fragments from complete examples. In complete examples permit only declared inputs, such as a typed `device` function argument. **M, nonbreaking.**
- **Confidence.** High. `KNOWN(STD-13)` understates the problem as docs grouping. Navigation and test coverage have independently drifted.

### CAT-10 Published sizing guidance still describes superseded execution paths [P1] [docs]

- **Evidence.** Forest inference says “All views are single packed buffers” and gives **16,777,216** as the row maximum. The implementation accepts aligned chunked feature/output records, applies traversal-work limits per batch, and exposes `GPU_FOREST_INFERENCE_MAXIMUM_ROW_COUNT = 0xffffffff`. The catalog already has the updated value. The coverage matrix also gives Dorling a 1,024-row ceiling, gives SCCs a blanket 1,024-vertex ceiling, and describes the raster resistance adapter as a follow-up although it is exported. `D/forest-inference.md:31`, `:74`, `modules/analysis/src/dataframe-analysis/forest-inference/gpu-forest-inference.ts:114`, `modules/analysis/src/dataframe-analysis/forest-inference/forest-inference-parameters.ts:24`, `O/all-entry-points/operator-annotations.ts:295`, `dev-docs/coverage-matrix.md:223`, `:231`, `:256`, `O/kepler-analysis-catalog.ts:50`, `modules/analysis/src/gpu-network/network-structure/gpu-network-structure.ts:85`, `modules/analysis/src/gpu-spatial-analysis/landscape-ecology/index.ts:35`.
- **Why it matters.** A user sizing batches from the forest page can discard a supported execution path. For SCCs, the old number still applies to an explicit exact route, which makes deleting the limit as misleading as leaving it unqualified.
- **Proposed change.** Correct these rows now. Separate total rows, per-batch work, per-binding storage, and mode-specific limits in the descriptor. Generate the numerical cells and preserve hand-written qualifications. Remove the obsolete KG03 re-check too: the row already lists both 1,024 and 8,192. `dev-docs/coverage-matrix.md:232`, `:254`. **S for corrections, M for generation, nonbreaking.**
- **Confidence.** High. `KNOWN(STD-13)` is stale about the location of the Dorling defect. The ontology’s Dorling description has been corrected. The matrix still needs the update.

### CAT-11 Hand-written annotations can be silently bypassed, and real limits can be omitted [P2] [consistency]

- **Evidence.** `GPUFittingCoordinateEncoder` has a chunking limitation in `OPERATOR_ANNOTATIONS`, backed by its implementation. The fitting catalog constructs its records separately and never reads that table, so lookup drops the limitation. Of the **209** annotation keys, this is the public one bypassed by assembly. A second key, `GPUSnapRoundedNoding`, is explicitly internal. In another family, the `GPUNetworkInhomogeneousK` record lists event, pair and radius maxima but omits enforced node, edge and event-by-node maxima. The limit spec checks positivity, reason and file existence. `O/all-entry-points/operator-annotations.ts:904`, `:2048`, `:2468`, `O/fitting-kernel-catalog.ts:91`, `modules/analysis/src/analysis-core/kernels/gpu-fitting-coordinate-encoder.ts:107`, `modules/analysis/src/gpu-network/gpu-network-inhomogeneous-k.ts:137`, `T/analysis-operator-catalog.node.spec.ts:127`.
- **Why it matters.** A correct edit in the advertised maintenance file can have no public effect. An existing source path is weak evidence that a record describes the checks in that file.
- **Proposed change.** Give each public descriptor one construction path. Make public limits reference a named constant or documented derived expression, with applicability by mode. Keep internal annotations separately identified. Test that every public annotation contributes to exactly one final record. **M, nonbreaking.**
- **Confidence.** High.

### CAT-12 The catalog guide promises fields and describes consumers inaccurately [P2] [docs]

- **Evidence.** The guide says each operator has a one-line summary. None of the **261** original descriptors gets a `summary` from `makeOperator`. The guide says `analysis-core` exports no contributors, although it exports the four fitting producers. It also says the website inventory enumerates the historical arrays, while the extractor calls the all-entry API. `D/analysis-catalog.md:18`, `:24`, `:75`, `O/spatial-analysis-operators.ts:900`, `:925`, `modules/analysis/src/analysis-core/index.ts:56`, `O/fitting-kernel-catalog.ts:32`, `website/scripts/extract-inventory.ts:145`.
- **Why it matters.** An application following the guide gets empty descriptions for nearly half the inventory. Free-text search also lacks those summaries. The incorrect consumer description makes the historical compatibility burden look larger than it is.
- **Proposed change.** Populate original summaries from public class/function TSDoc where suitable, with editorial overrides for unclear first sentences. Correct the guide’s package and consumer descriptions. **S for guide corrections, M for summaries, nonbreaking.**
- **Confidence.** High.

## Cross-cutting patterns

**Identity, membership and presentation need separate owners.** The package export determines identity. An operator declares one primary capability. The family member list can then be derived. A docs anchor and a lens scenario can share that identity without sharing a display title. CAT-02, CAT-03, CAT-08 and CAT-09 are four consequences of reconstructing these relationships independently.

**Completeness tests need an independent comparison.** The guard test iterates the same historical capability set as the guard. The package test throws away the package half of the identity. The docs sync test compares generated output while trusting the marker’s existing location. Each can stay green for the specific defect it ought to catch.

**A size limit needs a subject and a condition.** The annotation file already has **58 literal `name` fields with parenthesized qualifications**, such as `positions.length (csr)`. Those strings encode distinctions a caller needs to query: total versus chunk, dense versus indexed, and default versus explicitly selected mode. `O/all-entry-points/operator-annotations.ts:2449`, `:2461`.

## Keep

- **Keep discovery free of constructors.** The catalog can be consumed by documentation and host UIs without instantiating a contributor or loading its GPU implementation. `O/spatial-analysis-operators.ts:939`.
- **Keep the independent barrel check and explicit kind exceptions.** Checking actual exports catches a different failure from checking descriptor consistency. The kind spec also distinguishes a legitimate empty side from unexplained missing metadata. `T/analysis-operator-catalog.node.spec.ts:65`, `T/analysis-experience/analysis-kind-completeness.node.spec.ts:45`.
- **Keep curated reference coverage.** A single contributor can cover several distinct upstream operations or modes. The separate local-kriging rows demonstrate why reference versions, subsets and oracle evidence deserve editorial ownership. `dev-docs/coverage-matrix.md:121`, `:126`, `:260`.
- **Copy the variogram lifecycle example.** It shows actual inputs, checks result status before selecting a model, handles the missing result, and makes ownership visible in `finally`. `D/operation-forms-and-cpu-twins.md:134`, `:149`, `:159`.

## Future

1. **Introduce a neutral, metadata-only catalog entry.** Add `@luma-contributors/analysis/catalog`, backed by an allowed source directory such as `analysis-core/catalog`. Keep `gpu-spatial-analysis/ontology` as a compatibility re-export. Leave the historical arrays and their order intact. The cost is maintaining two import paths during migration.
2. **Pilot sibling `*.meta.ts` descriptors in network routing, forest inference and model fitting.** Derive package, name, public props/output types, optionality and source location from public barrels and declarations. Let the sibling descriptor own logical kinds, modes, tier, reference, packer-to-port mapping, and evidence anchors. A sibling file also accommodates recipes and CPU functions, and avoids bringing shader implementations into metadata consumers. The trade-off is one small authoring file per contributor or closely related family. The existing `spatial-validation-descriptors.ts` is a local precedent. `modules/analysis/src/gpu-spatial-analysis/spatial-validation/spatial-validation-descriptors.ts:5`.
3. **Generate one checked-in manifest, then generate consumer views from it.** Use qualified IDs for lookup indexes, capability membership, docs anchors, and website inventory joins. Generate props/outputs/limits tables and sidebar membership. Keep tutorial prose, worked examples, host-specific bindings, and reference-function coverage claims authored. During migration, compare generated and existing public records and review differences family by family. This adds a generation step and removes several hand-edited joins.
4. **Give evidence a scope.** Record separately what a constructor enforces, what a test exercises, and what a benchmark measured. For matrix rows, store references to qualified operators and evidence alongside the manually assessed upstream subset. Generate counts and validate links. This retains the value of the matrix without turning a new export into an automatic parity claim.

## Top 5

1. Fix the validity tally and pin the non-binary uint32 case.
2. Introduce package-qualified identity and register the network visibility recipe.
3. Derive all-entry capability membership and restore the 63 missing members.
4. Correct public discovery facts, starting with `time-series`, K-shortest-path status, snapshot readback and Gaussian-field grid mode.
5. Establish canonical docs anchors and public packer ownership. Use forest inference and contiguity statistics as the first two regression cases.

## Questions for the lead

- Can the new qualified lookup preserve the current spatial result for bare `addVisibilityGraphRecipe` while requiring qualified IDs for new consumers and persisted documents?
- Should the four public fitting helpers appear in the main API sidebar, or should the catalog explicitly link to their package-local reference? Their current documentation already exists, so this is a publishing decision.
- Is the intended guarantee of `operator.status` and `operator.limits` exhaustive or best-effort? The public query filters and sizing guide currently invite callers to treat an empty result as a definitive answer.