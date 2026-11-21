# Public API, user documentation and operator catalog (2026-11-19)

The first CPU computation now works as printed. The network quick start now seeds the intended node, and population alignment now produces a mask that a real quantile consumer interprets correctly.  
Of 34 findings from November 17, 24 are **FIXED** and 10 are **PARTLY**. The remaining work includes incomplete journeys, unpinned factual corrections and catalog metadata that still gives a wrong answer.  
Three ordinary-use failures deserve attention before publication. An overflow search silently omits six line operators, two documented package paths fail to resolve, and a documented spherical measurement option selects the planar constructor path.  
All 498 public command-node producers have `getCommandNodes(graph)`. All 116 API pages are discoverable, but only 472 of 761 public classes appear in a class heading, and 58 pages contain no TypeScript fence.  
The catalog covers every public contributor and recipe by entry-qualified identity. Its 555 records contain no CPU-twin linkage, and 242 describe status as confirmed absent, including the six counterexamples.  
The next three priorities are to correct those public discovery and first-run failures, make examples and references checkable against the package a reader can import, and give each operator one qualified record connecting its forms, status and documentation.

## Method

The area lead (`gpt-6.2-astra`) read all six reports in full, including their coverage limits. Six slice reviewers (`gpt-6.1-sol`) covered release notes and errors, first-run workflows, catalog and coverage, public API vocabulary, nonspatial reference coverage, and spatial reference and composition.

The comparison starts at `bdc715ea`. The observed HEAD was `bd8279a49a924bc55c29c807bb46929de69adc56`. The review distinguished committed changes from working-tree changes. For example, the November 19 working roadmap already contains review findings, and segregation-decomposition preflight in the working source is ahead of one of those rows. The census below uses the slices’ common snapshot. During lead verification the API pages grew from 37,332 to 37,364 lines and the spec inventory from 2,498 to 2,504. Line citations identify the version read. Website reader observations use `git show 73e8f638:website/scripts/extract-inventory.ts`.

Read in full across the team were the two READMEs, first-result walkthrough, family quick starts, ten package-local guide pointers, contributor and release contracts, authoring walkthrough, analysis catalog guide, coverage matrix, catalog assembly and freeze logic, summary generator, Parameters collector and its fixtures, public vocabulary and audit discovery checks, and the pertinent November 17 reports. The spatial reviewer also read the population CPU implementation, recipe result, prepared snapshot implementation and discard fixtures in full. All public barrels, API pages, headings, TypeScript fences and ontology records were enumerated mechanically. The large operator tables, experience rules, implementation bodies and numerical specs were sampled. The 59 pages in the spatial API directory received a census, not a line-by-line factual review.

The lead read the shared diff and path histories, including `git diff bdc715ea..HEAD -- docs/api modules/analysis/src/gpu-spatial-analysis/ontology dev-docs/coverage-matrix.md scripts/generate-spatial-catalog-summaries.mjs`, the documentation-check changes, the breaking-change ledger and the later optional-color change. The lead independently opened the source behind every reported wrong discovery result and both documentation findings promoted to visible failures. Lead executions included both catalog filters, Node package resolution, three linear-referencing constructor variants, the literal CPU variogram, both network seed fixtures, population alignment composed with quantiles, and the short-array comparator. Other explicitly attributed CPU and declaration probes below were executed by slice reviewers. Null-device probes stopped at construction or graph declaration.

Evidence labels distinguish `EXECUTED-CPU`, `SOURCE` and `COUNT`. A test named as a pin contains an inspected assertion and an identified regression it would catch. No test, typecheck, build, browser, GPU program, release gate or network lookup ran. The November 18 separate-step gate results remain dated evidence. This review did not reproduce a test failure on each parent revision.

For compact citations, `S/` expands to `modules/analysis/src/`, `T/` to `modules/analysis/test/`, `D/` to `docs/api/`, and `O/` to `modules/analysis/src/gpu-spatial-analysis/ontology/`. `CR` denotes `modules/analysis/test/gpu-spatial-analysis/ontology/catalog-regressions.node.spec.ts`.

## Census

### Public surface

`rg --files modules/analysis/src -g '*.ts' | wc -l` gave **1,772** files, with **565,068** newline-delimited lines, versus **1,710 / 541,841** at `bdc715ea`. `rg --files modules/analysis/test -g '*.spec.ts' -g '*.spec.js' | wc -l` gave **2,498** at the slice snapshot. The public inventory parsed the eleven source barrels identified by `modules/analysis/package.json#exports`, followed named and star re-exports, resolved inherited and mapped members, and deduplicated originating declarations. The old inventory used `git ls-tree -r bdc715ea -- modules/analysis/src` and `git cat-file --batch` with the same predicates. Public declarations originate in 1,072 files.

| Owning entry | Public GPU classes | Producers, old → current | Concrete GPU / CPU operation classes | Public `compute*OnCPU` | Qualified recipes | Catalog records |
|---|---:|---:|---:|---:|---:|---:|
| `analysis-core` | 6 | 4 → 4 | 0 / 0 | 0 | 0 | 5 |
| `computation-record` | 4 | 2 → 2 | 0 / 0 | 0 | 0 | 4 |
| `dataframe-analysis` | 41 | 34 → 34 | 7 / 7 | 7 | 0 | 40 |
| `geospatial-analysis` | 2 | 2 → 2 | 0 / 0 | 0 | 0 | 2 |
| `gpu-network` | 58 | 41 → 41 | 17 / 17 | 17 | 5 | 46 |
| `gpu-spatial-analysis` | 353 | 272 → 272 | 80 / 80 | 83 | 37 | 315 |
| `gpu-terrain` | 57 | 46 → 46 | 11 / 11 | 11 | 0 | 46 |
| `pointcloud-analysis` | 11 | 11 → 11 | 0 / 0 | 0 | 0 | 11 |
| `raster-analysis` | 99 | 80 → 86 | 13 / 13 | 13 | 0 | 86 |
| **Distinct total** | **631** | **492 → 498** | **128 / 128** | **131** | **42** | **555** |

Root and ontology re-exports add no distinct class. The 761 public classes comprise 631 GPU-prefixed, 128 CPU-prefixed and two other classes. `GPUSnapRoundedNoding` and `GPULocalPermutationReplay` explain the difference from the source count of 633 GPU classes. Public pure CPU functions rose from 129 to 131. `computeNetworkFlowPhasesOnCPU` and `computeAZPRegionsTableOnCPU` explain the difference from 133 source declarations. The 42 recipes have 41 names because two entries export `addVisibilityGraphRecipe`.

| Producer contract | November 17 | Current |
|---|---:|---:|
| Directly declares `implements GPUCommandNodeProducer` | 491 | 497 |
| One visible constructor argument | 491 | 498 |
| One graph argument and readonly node-array return | 489 | 496 |
| Additional visible optional composition argument | 1 | 0 |
| Accepts `props.output` | 471/492 | 488/498 |
| Point-cloud producers accepting `props.output` | 0/11 | 11/11 |
| Instance member named `output` | 2 | 3 |
| Exposes `estimatedCommandNodeCount` | 171 | 177 |
| Owns a `destroy()` method | 7 | 7 |

All 498 producers expose the graph method. `GPURasterNDWI`, `GPURasterNBR`, `GPURasterEVI` and `GPURasterSAVI` inherit it. `GPURasterEmbeddingKMeans` and `GPURasterEmbeddingNearestPixels` now implement it directly. The two mutable return declarations are `GPUWideNumericScan` and `GPULonLatPolygonPreparation`. The ten without `props.output` are `GPUSnapshotTap`, `GPUTransferMask`, `GPUSpatialJoinPrepared`, `GPUNetworkAccessibility`, `GPUNetworkAnalyticsColumns`, `GPURasterTexture`, `GPURasterStreamingAccumulator`, `GPURasterStandardizedIndexCalibration`, `GPURasterClimateRunAccumulator` and `GPUPointBuffer`. These include handles and accumulators. The three instance `output` owners are `GPUAnalysisSpacing`, `GPUSpatialWeightsMinimumSpanningTree` and `GPURasterExtremaPyramid`. See `S/raster-analysis/spectral-indices/raster-spectral-index.ts:81`, `S/raster-analysis/raster-embeddings/gpu-raster-embedding-kmeans.ts:107` and `S/dataframe-analysis/column-profile/gpu-wide-numeric-scan.ts:107`.

The seven lifecycle owners are `GPULocalFrameProjection`, `GPUConditionalGaussianSimulation`, `GPUSpatialJoinPrepared`, `GPUStudyWindowPreparation`, `GPUTerrainSpikeRepair`, `GPURasterReprojection` and `GPURasterTileReprojection`. `GPUSpaceTimeKrigingProps` gets `output` through its `Rows` intersection. `GPUPointCloudVoxelMergeProps` gets it through a mapped type. A literal search of class text misses both. See `S/gpu-spatial-analysis/spatial-interpolation/gpu-space-time-kriging.ts:105` and `S/pointcloud-analysis/operations/gpu-pointcloud-voxel-merge.ts:79`.

### Vocabulary, validation and layout

The comparable November 17 traversal counts directly authored fields of source aliases named `GPU*Props` or `GPU*Output`, once where written. In this table `a/b` means occurrences / optional occurrences. Unchanged entries reproduced on both revisions. Recipe `outputs` and producer `output` have different contracts.

| Measure | November 17 → current |
|---|---|
| Props plus output aliases | 930 → 942. Public props 559 → 559, public outputs 356 → 368. Explicit `any` in these public aliases 2 → 0. |
| Documented fields | Props 3,869/4,943 → 3,859/4,930. Outputs 1,541/1,666 → 1,583/1,716. Declaration JSDoc on 415/559 props and 320/368 current outputs. |
| Row count | `rowCount` 9/2, `rows` 2/0, `count` 4/3, `length` 1/0, `cellCount` 34/0. |
| Capacities | 45 spellings, 135/76 occurrences. `leafCapacity` 23/23, `candidateCapacity` 17/11, bare `capacity` 12/0, `intersectionCapacity` 12/2, `pairCapacity` 6/3. |
| Maximums | 96 `maximum[A-Z]` spellings, 224/169. No non-coordinate short `max[A-Z]` prop. `maximumIterations` 57/51, `maximumNeighborCount` 6/6 versus `maximumNeighbors` 5/4, `maximumVertexCount` 2/2 versus `maximumVertices` 3/3. |
| Masks and validity | Input `mask` 139/135, `validity` 15/14 → 14/14. Output `validity` 27/11 → 28/12, `outputValidity` 19/17, `outputMask` 6/6, `mask` 4/2 → 3/2. |
| IDs and rows | Input `…Ids` 34/19 → 33/19, `…Rows` 11/10 → 12/10. Output `…Ids` 26/16 → 23/14, `…Rows` 23/21 → 29/25. |
| Positions | `positions` 97/5, `points` 7/1, `queryPositions` 2/1, `referencePositions` 2/0, `coordinates` 1/0. |
| Writable results | Props `output` 483/26 → 495/26, `outputs` 10/9, `views` 1/0. |
| Status | Output `status` 177/88 → 188/90. Of the current 188, 157 resolve to named objects and 31 to views. `localStatus` 3/1, `statuses` 1/1. |

`maximumNeighbors` occurs on neighborhood summary, local permutation test, expectation-based space-time scan, multivariate local Geary permutation test and quantile LISA. `maximumVertices` occurs on convex straight skeleton, subdivide polygons and vector straight skeleton. `rows` means a grid dimension for grid generator and line density. Bare `count` is a view on segment ring assembly, cell set outline, cell compaction and trajectory locations. `rowIds` still occurs on trajectory split and thinning. `points` occurs on point buffer, geometry validity, cluster-and-outline and points-in-polygons choropleth recipes, buffer selection, point-in-polygon join and raster join. `coordinates` belongs to the fitting coordinate encoder. A migration needs the role of each field, particularly whether its values can index another input.

The 19 output aliases spelling `outputValidity` are Geomorphons, LocalDominance, MultiScaleRelief, ReliefBlend, SimpleLocalRelief, TerrainDerivatives, TerrainCurvature, TerrainRGBDecode, TerrainSpikeRepair, ReliefShading, SolarIrradiance, TerrainCastShadow, TerrainHorizon, TextureShading, TerrainRuggedness, TerrainTopographicPosition, TerrainVectorRuggedness, TerrainWeissLandforms and RasterSampling, each with the `GPU` prefix. The spectral base also exposes the spelling outside this alias denominator. The population migration demonstrates why a name count alone cannot establish polarity. Matched alignment has diagnostic code **0** and keep-mask **1**.

Of the 31 view-valued status declarations, 24 are `uint32`, four are `uint32x4`, and three permit a data or vector view. The four vector-word layouts belong to population support alignment, resistance connectivity, sparse resistance connectivity and visibility graph. The three data/vector unions belong to point-cloud features, ground and height. Inline props can add other layouts outside this alias count. In particular, network KDE exposes a **per-query** code and `output.ports.status.invalidCount`. Depression hierarchy and OPTICS expose packed summaries **and** named status ports. See `S/gpu-network/gpu-network-kernel-density.ts:51`, `S/gpu-terrain/terrain-processing/gpu-terrain-depression-hierarchy.ts:85`, and `S/gpu-spatial-analysis/spatial-clustering/gpu-optics.ts:166`.

Of 498 producer constructors, 413 directly call a named core limit helper, 492 directly call a `validate…` or context resolver, and 295 contain a direct throw. The six outside the second predicate are `GPULocalFrameProjection`, `GPUChunkedSpatialWeightsValidation` and the four spectral classes. They delegate construction. The permitted contributor-contract inventories returned zero unregistered contributors and zero bare-assert findings in their checked scope. Across class JSDoc, 103 mention chunks, 21 explicitly mention packed restrictions, ten mention both and **384/498 are unclassified by these predicates**. The census does not infer support for every port from one chunked input.

**Error-message audit.** For each family, the reviewer sorted `.ts` files having a throw or `/^validate[A-Z]/` call, selected indices `floor((n−1) × p)` for `p = 0, .25, .5, .75, 1`, and followed the first qualifying site to its formatter. Eligible file counts were spatial 579, network 86, raster 145, terrain 78, dataframe 88, point cloud 19, geospatial 3 and core/record 64. For geospatial the five positions came from its 47 qualifying sites. Of 40 messages, **37 identified the contributor or function, 39 the correct subject, 38 the accepted relation, and 34 all three**. Removing two pure argument errors and one structured record issue gives 34/37 jointly.

| Five selected sites per group | Identifiers or subjects sampled |
|---|---|
| Spatial | Analysis spacing context, population alignment, convexity hull scratch, spatial-lag ML weights status, zonal output alias |
| Network | Adjacency matrix order, snapping output range, K-function parameter target, Louvain node count, visibility `metricsMode` |
| Raster | Change-detection parameter target, isoline tile width, embedding chunk dimensions, medoid cell count, spectral buffer alias |
| Terrain | Geomorphons target, local dominance target, peak prominence grid, CPU depression-hierarchy width, summed-area scratch |
| Dataframe | Calendar target, scalar scale target, Ripley envelope radii, point-pattern target, standardization length |
| Point cloud | Change support, ICP distance, surfaces ground class, search-grid origin, voxel mean XYZ triples |
| Geospatial | Datum mask, datum width, vertical references, point-buffer product, viewport coordinates |
| Core/record | Capacity source count, device limit, chunk gather length, CPU trajectory rows, record `derivedFrom` reference |

The misleading cases are identifiable. A 65,536² raster gives “grid dimensions must be positive integers” although both are. The product exceeds `uint32`. A point-cloud origin `[1e40,0,0]` gives “must hold finite numbers” although `1e40` is finite in JavaScript and overflows float32. Point buffer validates `points.length * verticesPerRing` under the name `offsets.length`. The first two were executed by the release reviewer. See `S/raster-analysis/raster-band-utils.ts:43`, `S/pointcloud-analysis/operations/pointcloud-grid-index.ts:40`, `S/geospatial-analysis/point-buffer/gpu-point-buffer.ts:138`.

### Documentation, catalog and audit coverage

| Inventory | November 17 | Slice snapshot |
|---|---:|---:|
| API pages / distinct sidebar entries / scanner-eligible pages | 107 / 98 / 66 | **116 / 116 / 116**, one sidebar entry per page |
| `ts` or `typescript` fences / discoverable | 192 / 167 before opt-outs | **203 / 203**, zero opt-outs |
| Pages with / without such a fence | — | 58 / 58 |
| Generated Parameters blocks | — | 220 for 220 classes, on 43 pages, including 34 explicit shared-packer relationships |
| Distinct public classes in actual headings | — | 472/761, 555 occurrences, 45 combined headings |
| Actual class headings with no corresponding export | — | 0 |
| Ontology source / specs | — | 29 files / 21,265 lines and 23 files / 14,386 lines |

The 289 classes without an actual class heading consist of the following exact sets. Some have substantial prose or grouped tables under other titles.

- **32 producers.** Recording has `GPUSnapshotTap`. Spatial has `GPUTransitionDiagnostics`, `GPURandomPointsOnLine`, `GPURoseStatistic`, `GPULocationAllocation`, `GPURegionMask`, `GPUPickRegionMask`, `GPULocalRBF`, `GPUNaturalNeighbor`, `GPUConditionalGaussianSimulation`, `GPUAZPRegions`, `GPUWardRegions`, `GPUMaxPRegions`, `GPUSpatialPanelLagMaximumLikelihood`, `GPUTrackSimilaritySearch`, `GPUTrajectoryKalman`, `GPUTrajectoryRestriction`, `GPUNestedInterpolationValidation`, `GPUAreaOfApplicability`. Network has `GPUNetworkFlow`, `GPUNetworkTimeRouting`, `GPUNetworkPointRouting`, `GPUNetworkFacilityOptimization`, `GPUNetworkRationalAccess`, `GPUAlternativeRoutes`, `GPUMapMatchedEdgeVolumes`, `GPUNetworkStructure`, `GPUNetworkStronglyConnectedComponents`. Raster has `GPURasterEmbeddingCovariance`, `GPURasterEmbeddingProjection`, `GPURasterStandardizedIndexCalibration`, `GPURasterStandardizedIndex`.
- **250 operation forms.** The set is every public concrete CPU/GPU operation class except the six `CPU` and `GPU` pairs for `CoregionalizationFitOperation`, `KShortestPathsOperation` and `PopulationSupportAlignmentOperation`. It includes `CPULocationAllocationOperation` and `GPULocationAllocationGraphOperation`.
- **Seven other classes.** `GPUSnapshotArena`, `GPUComputationRecorder`, `GPUParameterBuffer`, `GPUContributorOperation`, `GPURegionStatisticsReadback`, `AnalysisGeographicExtentAccumulator` and `FlowFieldViewportTracker`.
- **Reverse difference.** `GPURaster` at `D/gpu-raster/operations-analysis.md:7` is an umbrella title. Experimental lens entries in `D/computation-views.md:8` are explicitly attributed. Neither is a missing analysis export.

The following syntax-only inventory reproduces page rows and prints the missing names by entry. Run as `node` input from the repository root. It counts named class tokens outside fences and generated Parameters headings. The slice also checked local export aliases and inherited producers.

```js
const fs = require('fs'), path = require('path'), ts = require('typescript');
const manifest = require('./modules/analysis/package.json');
const cache = new Map(), headings = new Set(), sidebar = [];
function classes(file) {
  file = path.resolve(file);
  if (cache.has(file)) return cache.get(file);
  const result = new Set();
  cache.set(file, result);
  if (!fs.existsSync(file)) return result;
  const ast = ts.createSourceFile(file, fs.readFileSync(file, 'utf8'), 99, true);
  for (const s of ast.statements) {
    if (ts.isClassDeclaration(s) && s.name &&
        s.modifiers?.some(m => m.kind === ts.SyntaxKind.ExportKeyword)) result.add(s.name.text);
    if (!ts.isExportDeclaration(s) || !s.moduleSpecifier) continue;
    const base = path.resolve(path.dirname(file), s.moduleSpecifier.text);
    const child = classes(fs.existsSync(base + '.ts') ? base + '.ts' : path.join(base, 'index.ts'));
    if (!s.exportClause) for (const name of child) result.add(name);
    else if (ts.isNamedExports(s.exportClause))
      for (const e of s.exportClause.elements)
        if (child.has((e.propertyName || e.name).text)) result.add(e.name.text);
  }
  return result;
}
function visit(items) {
  for (const e of items) {
    if (typeof e !== 'string') visit(e.items);
    else if (e.startsWith('api/')) sidebar.push(e.slice(4) + '.md');
  }
}
visit(require('./docs/table-of-contents.json'));
for (const file of fs.readdirSync('docs/api', {recursive: true}).filter(f => f.endsWith('.md'))) {
  let fenced = false, snippets = 0;
  for (const line of fs.readFileSync('docs/api/' + file, 'utf8').split('\n')) {
    if (/^\s*```/.test(line)) {
      if (!fenced && /^\s*```(?:ts|typescript)\s*$/.test(line)) snippets++;
      fenced = !fenced;
    } else if (!fenced && /^#{1,6}\s/.test(line) && !/ parameters\s*$/i.test(line)) {
      for (const n of line.match(/\b(?:GPU|CPU)[A-Z]\w*\b|\bAnalysisGeographicExtentAccumulator\b|\bFlowFieldViewportTracker\b/g) || [])
        headings.add(n);
    }
  }
  console.log('page', file, sidebar.filter(x => x === file).length, snippets);
}
const all = new Set();
for (const [entry, target] of Object.entries(manifest.exports)) {
  const names = classes(target.import.replace('./dist/', 'modules/analysis/src/').replace(/\.js$/, '.ts'));
  names.forEach(n => all.add(n));
  console.log(entry, [...names].filter(n => !headings.has(n)));
}
console.log('heading tokens outside public classes', [...headings].filter(n => !all.has(n)));
```

The catalog has **555 qualified records, 554 names, 92 families, zero duplicate qualified identities and zero missing or extra finalized family memberships**. Its 502 class records cover all 498 producers plus other registered classes. It includes all 42 recipes and 11 other functions. Concrete named graph operations, pure twins and the `GPUContributorOperation` base are outside the contributor registration rule. No catalog record in the reconciliation failed to resolve to its public export. The frozen historical projection contains 49 families and 261 records. Default queries exclude seven helpers by design.

| Field | Classified | Unknown or unspecified | Confirmed empty or none |
|---|---:|---:|---:|
| Status | 91 nonempty lists | 222 | 242 |
| Readback | 8 explicit | 225 | 322 |
| Tier | 324 scale, 30 reference | 201 | — |
| Topology | 120 fixed, 4 dynamic-parameters | 431 | — |
| Cardinality / semantics / determinism | 353 / 118 / 89 specified | 202 / 437 / 466 | — |
| Summary / `cpuTwin` | 555 / 0 populated | — | — |

There are 220 public annotation keys, 348 numeric limits on 177 records and 163 records with limitation text. All inspected reference paths exist. Entry-point construction gives 226 records family-level coordinate models and output bounds. Historical construction gives 261 records representative family scale evidence. A family measurement cannot establish each member’s independently tested scale. See `O/all-entry-points/analysis-operator-catalog.ts:45` and `O/spatial-analysis-operators.ts:874`.

All 14 wave-named IDs survive. The `coverage-` suffixes are `movement`, `geometry-refinement`, `global-coordinates`, `local-prediction`, `statistical-contracts`, `point-processes`, `urban-morphology`, `landscape-ecology`. The `expansion-` suffixes are `local-variability`, `regression`, `footprints`, `h3-identities`, `facility-allocation`, `workflows`. The “never wave-named” assertion exempts original families at `T/gpu-spatial-analysis/ontology/analysis-operator-catalog.node.spec.ts:116`.

The coverage matrix has **144 assessment rows** comprising 137 Implemented, one Partial and six Excluded. All **146 distinct local entry/symbol references** in the final LIB column resolve. Five referenced helpers are outside the catalog: `addRasterRegionGraphTiles`, `fitSpaceTimeVariogramModel`, `prepareGPUNetworkFlowDemands`, `prepareGPUNetworkRestrictionStates` and `prepareGPUTrajectoryObservations`. Conversely, **414 catalog identities** are outside the matrix’s assessed symbol set. The matrix is a survey of named workflows. Expanding it to 555 rows would change its purpose.

This leaf probe prints the complete named differences, including the 414 names and all unknown status/readback/tier names. Pipe through the prescribed stdin esbuild command. Published upstream rows are checked against the installed barrels separately.

```ts
import {readFileSync} from 'node:fs';
import {getGPUAnalysisOperatorCatalog as catalog} from './modules/analysis/src/gpu-spatial-analysis/ontology/all-entry-points/analysis-operator-catalog';
const refs = new Set<string>();
for (const line of readFileSync('dev-docs/coverage-matrix.md', 'utf8').split('\n')) {
  if (!/^\| [^|]+ \| (Implemented|Partial|Excluded|Planned) \|/.test(line)) continue;
  for (const m of line.split('|').at(-2)!.matchAll(/`([^`]+)`\s*:\s*([^;]+)/g))
    for (const s of m[2].matchAll(/`([^`]+)`/g))
      refs.add(`@luma-contributors/analysis/${m[1]}:${s[1]}`);
}
const key = r => `${r.package}:${r.exportName}`;
const records = catalog(), keys = new Set(records.map(key));
console.log('outside matrix', records.filter(r => !refs.has(key(r))).map(key));
console.log('outside catalog', [...refs].filter(r => !keys.has(r)));
for (const field of ['status', 'readback', 'tier'])
  console.log(field, records.filter(r => r[field] === 'unknown').map(key));
```

The shared kernel audit discovers 538 qualified identities across eight barrels but deduplicates to 537 names. Its 3,094 registered cases also cover 537 names. Both visibility recipe cases import the spatial implementation. Computation-record is absent from discovery: `GPUSnapshotTap` is copy-only and `GPUTransferMask` does have WGSL. Four chunked spatial-weights entries, Algebra, Transform, Transpose and Validation, use separate checks. Registration proves neither the execution of a case nor coverage of each optional output. See `T/analysis-core/kernel-audit/analysis-contributors.ts:31`, `T/analysis-core/kernel-audit/cases-coverage-urban-ecology.ts:205`, and `T/computation-record/gpu-transfer-mask.spec.ts:14`.

## Follow-through on the 2026-11-17 findings

The table covers the ERG, CAT and DOC findings in `dev-docs/research/library-review-2026-11-17/area-api-ergonomics.md`, `area-ontology-catalog.md` and `area-docs-devdocs.md`, including their overlap with `verification.md`. ERG-03 also closes the duplicated parameter-buffer claim CORE-03. It is counted once here.

| ID | November 17 claim | Verdict | Current implementation, pin and regression caught |
|---|---|---|---|
| ERG-01 | Alignment diagnosis acted as an inverted keep-mask | FIXED | `S/gpu-spatial-analysis/landscape-ecology/cpu-population-support-alignment-operation.ts:77` and `gpu-population-support-alignment.ts:258` emit separate `alignmentCode` and `matchedMask`. `T/gpu-spatial-analysis/urban-ecology/cpu-population-support-alignment-operation.node.spec.ts:42` feeds the mask to real quantiles. Substituting the code selects the missing row. |
| ERG-02 | Discarding an encoder consumed a prepared rebuild | FIXED | `S/gpu-spatial-analysis/spatial-join/prepared-geometry-snapshot.ts:116,164` compares and commits generations in graph work. `T/analysis-core/prepared-rebuild-discard.spec.ts:20` moves a square and checks resulting rows or bounds after discard and partial encodings. An encode-time acknowledgement leaves the old geometry. SOURCE for device behavior. |
| ERG-03 | Parameter buffer accepted the wrong typed-array bits | FIXED | `S/analysis-core/kernels/gpu-contributor-utils.ts:74,121` checks before allocation/write. `T/analysis-core/core-buffers.node.spec.ts:218,229,254` checks that backing calls never occur on mismatches and that a valid byte offset works. |
| ERG-04 | Arrow advertised as optional despite eager imports | FIXED | `modules/analysis/package.json:91` requires Arrow. `T/analysis-core/release-package.node.spec.ts:82` rejects restoring optional metadata. The packed consumer links declared peers, so that gate alone would miss it. |
| ERG-05 | Network guide seeded node zero and omitted breaks | FIXED | `D/gpu-network/README.md:58`, `T/analysis-core/first-run-docs.node.spec.ts:75` and the device fixture `network-first-run.spec.ts:21` distinguish the actual unreachable node, costs, bands and counts. CPU replay below. |
| ERG-06 | Quick starts omitted allocation and readback | PARTLY | `D/operation-forms-and-cpu-twins.md:246` gives a complete shared lifecycle. Terrain and dataframe now construct real inputs and outputs. Four journeys lack a literal complete, device-backed example. Statistical filtering lacks a construction fence. |
| ERG-07 | Class-breaks CPU and GPU forms disagreed on inputs | FIXED | Shared `S/dataframe-analysis/column-classification/class-breaks-input.ts:57` validates declarations. CPU reads actual input format. `T/gpu-dataframe/column-classification/cpu-class-breaks-operation.node.spec.ts:132,169` catches a float default on integers and acceptance of inconsistent format or `rowCount`. |
| ERG-08 | Two visibility recipes collapsed to one | FIXED | `O/all-entry-points/analysis-operator-catalog.ts:83,139,167` indexes both. `CR:55,156` discriminates network metrics from spatial adjacency, checks ambiguous bare lookup and both qualified results. |
| ERG-09 | Internal switches appeared in public props | PARTLY | All eight cited declarations removed. `T/analysis-core/contributor-contract-checker.node.spec.ts:376,406,442` exercises aliases, mapped or inherited types and erased overloads. Three baselined internal declaration sites and two undocumented public WGSL authoring helpers remain. |
| ERG-10 | Two prepared handles used an undocumented `any` frame | FIXED | `S/gpu-spatial-analysis/spatial-join/gpu-spatial-join-prepared.ts:121` and `study-window/gpu-study-window-preparation.ts:95` preserve frame generics. `D/gpu-spatial-analysis/concepts.md:351` explains them. `T/analysis-core/prepared-frame-types.node.spec.ts:51` expects errors for void/mismatched frames. Reintroducing `any` erases a required error. |
| ERG-11 | Recipe implementation was mistaken for result compatibility | FIXED for the cited recipe | `S/gpu-spatial-analysis/landscape-ecology/segregation-counterfactual.ts:266` exposes `outputs`, `intermediates`, `status.stages` and `packedStatus`. `T/gpu-spatial-analysis/urban-ecology/population-support-alignment-result.node.spec.ts:12` checks omitted, partial, full and offset overrides, including buffer identity. Wider STD-11 remains. |
| ERG-12 | Distributed README lost its doc links | PARTLY | `modules/analysis/README.md:10,31,47` uses repository destinations. `T/analysis-core/first-run-review.node.spec.ts:69` has file/anchor typo counterexamples. All 26 repository links still rely on a provisional repository identity. Public reachability is unresolved. |
| CAT-01 | Legal nonzero validity values disappeared from tallies | FIXED | `O/analysis-experience/analysis-interpretation.ts:452` normalizes validity. `CR:29` uses 2 and `0xffffffff`, with a categorical control that would catch normalizing all code schemes. |
| CAT-02 | Bare-name identity lost a public recipe | FIXED | `O/all-entry-points/analysis-operator-catalog.ts:139,167`, `CR:156`. Both qualified identities resolve. An ambiguous bare all-entry lookup throws and identifies both entries. Historical spatial lookup remains spatial. |
| CAT-03 | Family browsing omitted registered members | FIXED | `O/all-entry-points/analysis-operator-catalog.ts:127` derives membership from final records. `CR:196` catches a reversion to historical lists. The executed qualified difference is empty in both directions. |
| CAT-04 | Unknown metadata became a definite negative answer | PARTLY | `CR:86,99,108` pins unknowns, K-shortest status, snapshot readback and Gaussian surface kinds. Historical defaulting at `O/spatial-analysis-operators.ts:863` still produces the six false negatives in CAT-101. |
| CAT-05 | Public kind guard rejected `time-series` | FIXED | `O/spatial-analysis-operators.ts:988` checks the canonical tuple. `CR:174` iterates it. Lead lookup returned `true`. |
| CAT-06 | Consumers could mutate shared catalog records | FIXED | `O/freeze-catalog.ts:5`, `O/spatial-analysis-operators.ts:884`, `CR:74,181` address nested reused values as well as records. Shallow freezing would permit an evidence or limit mutation. |
| CAT-07 | Imported child packers appeared as public settings | FIXED | `T/analysis-core/doc-parameters.ts:196,241` requires an owner or explicit public relationship. `doc-parameters.node.spec.ts:150` excludes the contiguity statistic’s private median while retaining genuine shared packers. |
| CAT-08 | Generated blocks stayed under the wrong heading | FIXED | `T/analysis-core/doc-parameters.ts:596,649` resolves the heading again after removing old blocks. `doc-parameters.node.spec.ts:179` starts with duplicate/misplaced Height blocks and checks relocation and idempotence. |
| CAT-09 | Sidebar and checker subsets differed | PARTLY | All nine old omissions are closed. `T/analysis-core/doc-parameters.node.spec.ts:141,145` checks equality and duplicates. Free names in ordinary fragments still become `any` at `doc-snippets.node.spec.ts:277`. |
| CAT-10 | Published sizing claims described superseded routes | PARTLY | Forest, Dorling, natural breaks, raster resistance and SCC statements cited previously are corrected. `T/analysis-core/documentation-facts.node.spec.ts:65,166` pins forest and matrix SCC. No discriminating matrix wording pin was established for Dorling, resistance or natural breaks. The network API still repeats the old SCC restriction. |
| CAT-11 | Fitting annotations bypassed and network K limits absent | FIXED | Final annotation application `O/all-entry-points/analysis-operator-catalog.ts:117`. `CR:121,127,139` catches bypass, requires all 220 public keys to reach one identity and compares three network K bounds to source constants. |
| CAT-12 | Catalog guide promised absent summaries and misstated readers | FIXED | `O/spatial-analysis-operators.ts:845` assigns summaries to all 261 historical records. `T/gpu-spatial-analysis/ontology/catalog-summaries.node.spec.ts:15,33` catches missing summaries or search ignoring descriptions. Generated and checked-in maps matched. Individual guide sentences have weaker protection. |
| DOC-01 | CPU fit silently truncated mismatched bin arrays | FIXED | `S/dataframe-analysis/pair-statistics/cpu-variogram-model-fit-operation.ts:27` validates lengths. Its `cpu-variogram-model-fit-operation.node.spec.ts:394` exercises shorter and longer inputs through both forms. Restoring `Math.min` would return a fit. The new error still omits the offending name. |
| DOC-02 | README stopped before a first result | FIXED | Root and package READMEs link the first result before the package inventory. `T/analysis-core/first-run-docs.node.spec.ts:103` compares literal examples to executable source, and `point-density-first-run.spec.ts:22` checks actual grid counts in a fresh realm. Lead executed the CPU fence. |
| DOC-03 | Roadmap assigned work that had shipped | PARTLY | `T/analysis-core/roadmap-reconciliation.node.spec.ts:190,212` protects selected migrations and priorities. UF-10, STD-09, the packed-status decision and one STD-11 clause still have stale premises. See Roadmap rows. |
| DOC-04 | Reference understated available implementations | PARTLY | `D/gpu-raster/raster-embedding-codecs.md:87`, `dev-docs/coverage-matrix.md:255`, `D/analysis-catalog.md:22` now match. `T/analysis-core/documentation-facts.node.spec.ts:153,166` catches codec and matrix SCC reversions. The fitting sentence lacks an equivalent pin and `D/gpu-network/expansion.md:44` contradicts the corrected matrix. |
| DOC-05 | Local bridge restriction attributed to upstream executor | FIXED | `dev-docs/contributor-contract.md:174`, `S/analysis-core/operations/gpu-contributor-operation.ts:217`, and `cpu-operation-rows.ts:27` distinguish executor snapshots, local single-chunk lowering and helper packing. `T/analysis-core/cpu-callback-copies.node.spec.ts:10` observes the callback and distinguishes the copies. |
| DOC-06 | Publication, TOC and snippet inventories differed | FIXED | `T/analysis-core/doc-program.ts:17` discovers pages on disk. `doc-parameters.node.spec.ts:145` rejects missing, duplicate and nonexistent pages. `doc-snippets.node.spec.ts:58,84` pins both fence spellings. |
| DOC-07 | Ten package guides were outside publication | FIXED | The ten `modules/analysis/docs/*.md` files are now pointers to canonical API pages. `T/analysis-core/doc-publication.node.spec.ts:65,96,120` checks their destinations, navigation and active references. No active ontology file points to the former path. |
| DOC-08 | Setup and integration docs described an older checkout | FIXED in scope | Both READMEs, required peers and installed target agree on alpha.16. `T/analysis-core/first-run-docs.node.spec.ts:40` checks the literal target in both READMEs. `documentation-facts.node.spec.ts:112,132` checks setup and heap behavior. |
| DOC-09 | Authoring rules lacked a worked change | FIXED | `dev-docs/authoring-walkthrough.md` follows coregionalization through implementation, docs, catalog and tests. `T/analysis-core/authoring-walkthrough.node.spec.ts:97,124,140` distinguishes mutable dimensions, incorrect ownership and wrong format. The slice reproduced signed covariance and a zero budget. |
| DOC-10 | Retention guidance conflated executable inputs and citations | PARTLY | `dev-docs/README.md:29` and `dev-docs/research/README.md:44` distinguish dependencies and historical citations. `T/analysis-core/upstream-inventory-tools.node.spec.ts:22` uses maintained tools under `scripts/`. The research index at `dev-docs/research/README.md:26` still describes the upgrade inventory scripts as part of the dated directory. |

## Findings

Final grades comprise **one W, two S, nine C, five T and two M**, totaling **19**. Reach is **17 ORDINARY, one UNUSUAL and one EXTREME**. Related symptoms with the same correction share an ID.

| ID | Kind | Reach | Evidence | Where | What goes wrong | Proposed change |
|---|---|---|---|---|---|---|
| CAT-101 | W | ORDINARY | EXECUTED-CPU, SOURCE | `O/spatial-analysis-operators.ts:863`, `D/analysis-catalog.md:79,109` | Overflow discovery silently omits six public operators with overflow ports. The record asserts absence. | Populate the six mode-aware statuses, default unaudited historical status to unknown, and pin both filters. `KNOWN(STD-13)` |
| DOC-101 | S | ORDINARY | EXECUTED-CPU | `D/gpu-spatial-analysis-cross-reference.md:18`, `modules/analysis/package.json:49,54` | Following the entry-point guidance for raster or dataframe fails during module resolution. | Use the two public suffixes and check prose entry-point lists against the export map. |
| DOC-102 | S | ORDINARY | EXECUTED-CPU, SOURCE | `D/gpu-spatial-analysis-cross-reference.md:47,215,216`, `S/gpu-spatial-analysis/linear-referencing/gpu-linear-referencing.ts:377` | The documented spherical option enters planar validation and throws. Adding planar capacity lets the call proceed in native units. | Replace retired `coordinateSystem` examples with typed `spatialContext` contracts for each operator. |
| DOC-103 | C | ORDINARY | SOURCE | `D/gpu-spatial-analysis-cross-reference.md:32`, `D/gpu-dataframe-analysis.md:174` | “One overflow flag” omits stage-specific diagnostics. The dataframe convention suggests `output.status.code` for fits whose actual public output is a per-model status vector. | Show aggregate, final-output and candidate loss separately. Show model-vector access beside scalar status bags. `KNOWN(STD-04)` |
| DOC-104 | C | ORDINARY | SOURCE, EXECUTED-CPU | `D/gpu-network/expansion.md:39`, `D/gpu-dataframe-analysis.md:389,534`, `D/pointcloud-analysis.md:133` | General network structure inherits exact-mode bounds in prose. Rational quantiles and classification are described as single-binding or packed-only. The shared point-cloud introduction incorrectly includes all eleven in a packed-only rule. | Name limits by route and port. A slice declared 33,554,433 rational-quantile rows across two chunks, producing 14 nodes. `KNOWN(GPU-13, STD-14)` |
| DOC-105 | C | ORDINARY | SOURCE | `D/gpu-spatial-analysis/interpolation-expansion.md:88`, `D/gpu-spatial-analysis/urban-morphology.md:52,62` | Kriging prose implies all queries scan all samples. Tessellation prose carries reference output/limits into the local route. | Document automatic index cutoff, explicit scan cap, local certification and reference dissolved-ring output. `KNOWN(Space-time kriging, GPU-11)` |
| DOC-106 | C | ORDINARY | EXECUTED-CPU, SOURCE | `CHANGELOG.md:123,268`, `S/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-segments.ts:236` | Migration omits a required trips candidate-loss output, a conditional tripleg output, and numerical suppression by mobility indicators when supplied staypoint loss is set. | Add an old/new output map and the consumer’s suppression behavior. The intended constructor refusal already has a test. |
| DOC-107 | C | ORDINARY | COUNT, SOURCE | `D/gpu-spatial-analysis-cross-reference.md:52,82,126`, `D/gpu-spatial-analysis/operators.md:19` | Two K-means links use the obsolete `#gpukmeans` anchor. At least five locally implemented dataframe operators send readers to upstream references. The Ripley summary also omits prepared polygon windows. | Link each symbol to its owning local reference. Validate local fragments. |
| DOC-108 | C | ORDINARY | COUNT | `T/analysis-core/doc-parameters.ts:189,290`, `D/gpu-network/expansion.md:39` | Reference discovery has no qualified class-to-section relation for 289 public classes. A class can have a synchronized packer or a prose mention without a findable class section. | Give every class a canonical anchor, shared-form section or explicit infrastructure entry. `KNOWN(STD-12, STD-13)` |
| DOC-109 | C | ORDINARY | SOURCE | `dev-docs/coverage-matrix.md:30,254`, `D/gpu-spatial-analysis/interpolation-expansion.md:158` | Matrix attribution says alpha.10 and gives an unqualified small-fit CPU speed advantage. Installed target is alpha.16, and the cited history distinguishes serial, cooperative and transfer comparisons. | Date each upstream assessment and comparison. Link the route, shape and transfer boundary. `KNOWN(IN01, LR-22)` |
| ERG-101 | C | ORDINARY | EXECUTED-CPU, SOURCE | `S/dataframe-analysis/pair-statistics/cpu-variogram-model-fit-operation.ts:27`, `dev-docs/contributor-contract.md:102,162` | The first CPU failure says only “invalid model or bin topology”. Authoring guidance also describes `complete` as an invented synonym and says wrappers perform no validation despite shipped completeness predicates and preflight refusals. | Name the offending array and lengths. Document approved completeness meaning and the distinction between operation preflight and algorithm behavior. |
| ERG-102 | C | EXTREME | EXECUTED-CPU, SOURCE | `S/raster-analysis/raster-band-utils.ts:43`, `S/pointcloud-analysis/operations/pointcloud-grid-index.ts:40`, `S/geospatial-analysis/point-buffer/gpu-point-buffer.ts:138` | Three boundary diagnostics misidentify the rejecting range, representation or subject. | State the actual product cap, float32 representability and product expression. |
| DOC-110 | T | ORDINARY | EXECUTED-CPU, SOURCE | `T/analysis-core/doc-snippets.node.spec.ts:277`, `T/analysis-core/doc-program.ts:82,102`, `tsconfig.json:41` | An implicit misspelled API can become `any`. A source wildcard can resolve unpublished deep imports. Zero bad explicit fenced imports today does not exercise either rejection. | Require explicit API imports, declare fragment inputs, and validate package specifiers against `exports`. |
| DOC-111 | T | ORDINARY | COUNT, SOURCE | `T/analysis-core/first-run-review.node.spec.ts:16`, `D/pointcloud-analysis.md:54` | Four of the six first-run journeys do not have a literal complete example tied to a device-run fixture. | Use a shared lifecycle and small per-family examples with independent expected outputs. |
| ERG-103 | T | ORDINARY | COUNT, SOURCE | `T/analysis-core/kernel-audit/analysis-contributors.ts:31`, `cases-coverage-urban-ecology.ts:205` | Name-only coverage hides the missing network visibility recipe case. The shared audit also misses `GPUTransferMask`. | Key by package and export, enumerate computation-record, and identify copy-only exceptions. |
| ERG-104 | T | ORDINARY | COUNT, SOURCE | `T/analysis-core/analysis-vocabulary.node.spec.ts:30,173` | Vocabulary checks omit point cloud and computation record and miss referenced or mapped members. Point-cloud ICP `iterations` and SMRF `selection` illustrate the blind spot. | Compare resolved public members across all entries and name semantic exceptions. |
| ERG-105 | T | UNUSUAL | EXECUTED-CPU | `T/gpu-raster/raster-time-series/raster-time-series-test-utils.ts:92` | The comparison helper reports zero maximum error for an empty or shortened actual array. Nine test/helper files reference it. | Require equal lengths before numerical comparison and pin mismatched-NaN handling separately. |
| CAT-102 | M | ORDINARY | SOURCE, COUNT | `O/spatial-analysis-operators.ts:828`, `O/all-entry-points/analysis-operator-catalog.ts:27,117`, `O/coverage-expansion-catalog.ts:505` | Several construction paths own defaults and apply annotations. One path now distinguishes unknown while another silently emits `[]`. | Finalize metadata once per qualified identity and project the historical view afterward. `KNOWN(STD-13)` |
| ERG-106 | M | ORDINARY | SOURCE, COUNT | `scripts/check-contributor-contract.mjs:459`, `T/analysis-core/kernel-audit/analysis-contributors.ts:31`, `T/analysis-core/analysis-vocabulary.node.spec.ts:173` | Contract, audit and docs checks independently resolve different public surfaces. A passing count has a different denominator in each. | Share a lightweight entry-qualified declaration inventory while keeping domain-specific checks separate. |

### CAT-101 — six false negatives in public status discovery

The lead imported `getGPUAnalysisOperator`, `queryGPUAnalysisOperators` and the historical `queryGPUSpatialAnalysisOperators` from their source files. For each of `GPULineSegmentize`, `GPUGreatCircleArcs`, `GPULineSmooth`, `GPULineChunk`, `GPURectangleClip` and `GPUOffsetCurve`, the probe read the record and tested membership in both `{status:'overflow'}` results. **All six printed `status: [], all: false, legacy: false`.** Expected membership is `true` for all six, with compact mode identified for offset curve.

The shared output contract declares `count` and `overflow` at `S/gpu-spatial-analysis/line-segmentize/line-segmentize-types.ts:49,63`. The publisher is wired at `line-segmentize/gpu-line-segmentize.ts:294`, `gpu-great-circle-arcs.ts:158`, `gpu-line-smooth.ts:188`, `gpu-line-chunk.ts:207`, `rectangle-clip/gpu-rectangle-clip.ts:429`, and `outline-geometry/gpu-offset-curve.ts:360`, under `S/gpu-spatial-analysis/`. `O/spatial-analysis-operators.ts:863` converts absent annotation into `[]`. The filter at `O/all-entry-points/analysis-operator-catalog.ts:202` then gives the ordinary caller no indication of uncertainty. This is a CPU query result. It requires no shader prediction.

### DOC-101 — two published-path instructions fail locally

From the checkout, the lead called Node’s `import.meta.resolve` on `@luma-contributors/analysis/gpu-raster` and `@luma-contributors/analysis/gpu-dataframe`. **Both returned `ERR_PACKAGE_PATH_NOT_EXPORTED`.** Replacing the suffixes with `raster-analysis` and `dataframe-analysis` resolved both entry modules. The former names appear in prose at `D/gpu-spatial-analysis-cross-reference.md:18`, on a page with zero TypeScript fences. Independent enumeration of 48 analysis `from` occurrences in fenced TypeScript found ten valid public paths and zero bad specifiers. Correctly checked imports and a failing published instruction coexist on the same documentation surface.

### DOC-102 — documented spherical selection enters planar validation

The lead supplied valid transient views for two line vertices, two line offsets, one query, its search radius, measure output and candidate-loss status to `GPULinearReferencing`. No graph was compiled. Input `{coordinateSystem:'spherical'}` printed **`linear-referencing planar needs candidateCapacity`**. Adding `candidateCapacity:1` printed the resolved context **`{"coordinateSpace":"planar","metric":"native","units":"native"}`**. Replacing the obsolete option with `{spatialContext:{coordinateSpace:'longitude-latitude',metric:'great-circle'}}` selected the requested geographic context without the planar capacity. The guide promises the first call selects spherical measurement. The constructor reads `props.spatialContext` at `S/gpu-spatial-analysis/linear-referencing/gpu-linear-referencing.ts:377`. Geometry measures also reads the current context contract at `S/gpu-spatial-analysis/geometry-measures/gpu-geometry-measures.ts:127,273`.

**GPU probe wanted for the numerical consequence.** Use a line from `(0°,0°)` to `(1°,0°)`, a query at `(0.5°,0°)`, sufficient radius and capacity, and read the along-line measure. Compare explicit geographic great-circle metres, about 55.6 km, with planar native coordinates, about 0.5 coordinate units. The constructor failure and wrong selected context are verified. The device values are predictions.

## Refuted or downgraded

- An empty catalog query result for an explicitly `unknown` record is documented policy. CAT-101 concerns six records falsely marked *known empty*.
- Point-in-polygon top-level `status.overflow` really does aggregate leaf, candidate and final overflow. `matches.status.overflow` diagnoses final storage and `candidateOverflow` isolates intermediate loss. The short reference loses diagnostic detail. No incorrect aggregate result was demonstrated. `S/gpu-spatial-analysis/spatial-join/gpu-point-in-polygon-join.ts:175,200,598`.
- The suggestion that changing both READMEs back to alpha.13 would pass the documentation suite is refuted by `T/analysis-core/first-run-docs.node.spec.ts:40`. One release-package test considered in isolation has weaker assertions.
- Clearing a six-line operator search or registering 537 audit names does not prove a GPU computation failed. The former is a CPU discovery defect, the latter a test coverage gap.
- `GPUSpaceTimeKriging` and `GPUPointCloudVoxelMerge` both have `props.output`. Their inherited or mapped declarations defeat a simple grep. The earlier eleven point-cloud exceptions are closed.
- `GPUTransferMask` has independent device coverage. The missing case is in the *shared kernel audit*. `GPUSnapshotTap` needs a copy-only classification.
- Exact-name changelog comparison yielded 118 changed shape/result names absent from the text. A contributor-level migration often covers several such names. Only the independently traced trajectory omission is carried as a finding.
- The release record’s absence in this checkout and a public-access field in `package.json` establish neither remote publication nor its absence. The repository identity remains a maintainer decision.
- Documentation claims about device values, performance and upstream URLs received source or historical attribution only. No GPU defect or dead external URL is inferred from those claims.

## Tests

### First-run paths and independent values

The root README, package README and API index now link the same CPU and GPU starting point. The CPU block imports the public function, makes its own data and returns a result. The complete point-density example goes from adapter selection to destruction. The mixed variogram function explicitly takes an existing device, even with a CPU stage. The package README has 28 links, including 26 repository URLs, one upstream URL and one shipped notice. The first-run reviewer checked 140 local/repository targets including 26 anchors with zero local failures in that sample. The wider API anchor scan found the two K-means exceptions among 558 links with fragments.

The lead executed the literal CPU variogram fence at `D/operation-forms-and-cpu-twins.md:69`, substituting only the direct source import. Twenty distances `h=(i+1)*0.2`, semivariances `0.2+1.5*(1-exp(-3*h/2.2))`, and pair counts `10+i` gave exponential row **`[0.2000000774860382,1.5,2.200000286102295,2.7552409681032286e-13]`**, with statuses **`[0,0,0]`**. The independent target is nugget 0.2, partial sill 1.5, range 2.2 and near-zero residual. Removing the last pair count produced `variogram-model-fit invalid model or bin topology`. The old silent truncation is gone.

Line counts below use nonblank, non-comment physical lines and include the same 31-line reconstructed scaffold each time. They measure one explicit reconstruction from the docs, not a minimum API cost. The scaffold creates a WebGPU device, allocates and imports buffers, creates typed views, compiles, encodes, submits, reads the requested prefixes/status, then destroys resources. The network and point-cloud imports had to be assembled from documented class and entry names.

| Workflow | Nov 17 guide / complete | Current landing example / reconstructed complete | Reconstructed inputs, outputs and oracle |
|---|---:|---:|---|
| Point density | 12 / 42 | linked complete program 66 / 41 | Five points on a 2×2 grid, one outside bounds. Read `values` and scalar overflow. Hand-counted `[2,0,1,1]`, `0`. Program and device fixture are source-linked. |
| Network reachability | 23 / 44 | 10 data lines / 53 | CSR offsets `[0,0,1,1]`, neighbor `[2]`, weight `[2]`, start `[1]`, breaks `[0,2,5,10]`, limit 10. Lead CPU run gave costs `[Infinity,0,2]`, bands `[4294967295,0,1]`, counts `[1,1,0,0]`. Old four-slot starts gave `[0,0,2]`, `[0,0,1]`, `[2,1,0,0]`. GPU caller also reads convergence and iteration limit. |
| Terrain derivatives | 21 / 44 | 25-line constructor / 43 | Nine zero elevations, 30-unit cells, actual band constructed via `makeGPURasterBand`, 8-float parameters, slope and hillshade. Center expected 0 and about 0.7071 at 45° illumination. |
| Distance field | 16 / 43 | 25-line constructor / 43 | 3×3 unit cells, one center seed, exact mode, radius 5. Expected distances `[√2,1,√2,1,0,1,√2,1,√2]`, allocations all zero, inside-radius flags all one. |
| Time-window filter | 17 / 44 | 24-line constructor / 43 | Times `[0,10,20]`, inclusive window `[5,15]`. Read compact rows through `count`, expected `[1]`, required count 1 and overflow 0. Unwritten tail has no result meaning. |
| Point-cloud statistical filter | no TS / 42 | no filter construction fence / 41 | Squared distances `[1,1,100,0]`, counts `[1,1,1,0]`, capacity 1, two standard deviations. Expected means `[1,1,10,-1]`, validity `[1,1,1,0]`. GPU confirmation remains outstanding. |

All six have catalog records and audit registrations. Point density and time filter support chunked inputs on relevant row ports. Network CSR, the terrain band in this example, distance seeds and point-cloud neighbor slots use packed views. The six do not share one output convention. Each of the two graph statuses in the network example is necessary to distinguish partial reachability.

None of these six has a dedicated public result-layout function or decoder. A public-barrel scan for `^decode|^get.*Layout$` found 20 names. Spatial covers spacing statistics, geometry validity codes, region statistics, scan statistic layout, segregation layout, space-time inference metadata, GWR and multiscale GWR status, contiguity statistics, segment-snap issues and population alignment codes. Network covers coarsening summary, network statistics, subgraph-filter counts, K-shortest layout and K-shortest decoding. Terrain covers horizon unorm16. Raster covers extrema pyramid and geometric-median-composite layouts. Dataframe has `decodeNarrowingError`. Point cloud has none under this naming rule. Parameter packers, metadata builders and the reachability maximum-row helper serve different purposes.

### Accuracy of documented facts

The selected sample contains **62 explicit claims, 53 fully supported, eight misses and one partially true**, for a strict hit rate of **85.5%**. It deliberately includes changed contracts, old high-risk prose and least-read spatial families. It is not a random estimate for the entire reference.

| Sample and documentary anchor | Source anchors | Checked facts | Full matches |
|---|---|---|---:|
| Population, `D/gpu-spatial-analysis/landscape-ecology.md:393`, `recipes.md:63` | `S/gpu-spatial-analysis/landscape-ecology/cpu-population-support-alignment-operation.ts:61,77`, `segregation-counterfactual.ts:254` | Codes, keep polarity, full uint32 ID range, nonfinite/negative copy, packed stage aliases, packed nonempty support | 6/6 |
| Prepared handles and recipes, `D/gpu-spatial-analysis/concepts.md:351`, `recipes.md:33,56` | `S/gpu-spatial-analysis/spatial-join/prepared-geometry-snapshot.ts:116,164`, `recipes/prepared-recipe.ts:50,153` | Typed frame, discard generation, request counter, preparation graph, final caller-mutation boundary | 5/5 |
| Joins and weights, `D/gpu-spatial-analysis/operators-joins.md:36,146,669`, `operators-weights.md:116` | `S/gpu-spatial-analysis/spatial-join/cpu-feature-join.ts:442,454`, `gpu-nearest-feature-weights.ts:224`, `spatial-weights/gpu-contiguity-weights.ts:316` | Feature rows, multipart predicate subset, two loss stages, neighbor rows, default 4× budget | 5/5 |
| Diagnostics and hot spots, `D/gpu-spatial-analysis/recipes.md:113`, `operators-statistics.md:2174` | `S/gpu-spatial-analysis/spatial-regression/gpu-spatial-regression-diagnostics.ts:670`, `emerging-hot-spots/emerging-hot-spot-parameters.ts:195,218` | Weights-loss code, BH/final-slice defaults, 0.95 confidence/ten-slice trend default | 3/3 |
| Grid, point processes, global coordinates and contiguity, `D/gpu-spatial-analysis/operators-geometry.md:946`, `point-process.md:10,78`, `global-coordinates.md:54,94`, `analysis-experience.md:9` | `S/gpu-spatial-analysis/grid-generators/gpu-grid-generator.ts:45,428`, `point-process/translation-point-pattern.ts:65`, `global-coordinates/spherical-limits.ts:24`, `cell-topology/gpu-s2-cap-covering.ts:121,204`, `spatial-weights/gpu-analysis-contiguity-statistics.ts:240` | Hex orientation/height, 4,096 events/256 radii, overlap default/range, polygon/stride bounds, S2 levels, private median | 6/6 |
| Cross-reference and route summaries, `D/gpu-spatial-analysis-cross-reference.md:18,32,47`, `interpolation-expansion.md:88` and `urban-morphology.md:62` in the spatial directory | DOC-101, DOC-102, DOC-103, DOC-105 | **Misses:** package suffixes, retired spherical option, omitted separate loss diagnostics, universal kriging scan. **Partial:** 64 buildings/128 samples true for reference tessellation only. | 0/5 |
| Network/terrain, `D/gpu-network/expansion.md:44,100,107`, `D/gpu-terrain/height-above-drainage.md:32,40`, `operators.md:507` | `S/gpu-network/network-structure/gpu-network-strongly-connected-components.ts:123`, `gpu-network-structure.ts:85`, `S/gpu-terrain/hydrology/gpu-terrain-height-above-drainage.ts:70,200`, `terrain-decode/gpu-terrain-rgb-decode.ts:301` | SCC node budget and three scalars, **miss:** general structure bounds, HAND defaults/sentinel/range, Terrarium and Mapbox formulas | 7/8 |
| Fitting, `D/analysis-core/fitting-kernels.md:21,54,151` | `S/analysis-core/kernels/gpu-fitting-gradient-sums.ts:149`, `adam-step-parameters.ts:21,44`, `gpu-fitting-coordinate-encoder.ts:55,366` | Zero mask, pairwise reduction, Adam defaults and packed epsilon, Fourier 8,192, harmonic degree 39 | 6/6 |
| Dataframe, `D/gpu-dataframe-analysis.md:389,534`, `D/variable-standardization.md:13,29,34` | `S/dataframe-analysis/column-classification/gpu-column-quantiles.ts:198,213`, `variable-standardization/gpu-variable-standardization.ts:230,349,357` | **Misses:** one total binding for rational quantiles, packed-only classification inputs. Nonzero mask, singleton sample NaN and constant zero-MAD behavior match. | 3/5 |
| Point cloud, `D/pointcloud-analysis.md:64,73,133` | `S/pointcloud-analysis/operations/gpu-pointcloud-statistical-filter.ts:129,182,257`, `gpu-pointcloud-height.ts:99,115,219` | Retained slots, 1–64 capacity, output polarity. **Miss:** all eleven require packed views. | 3/4 |
| Geospatial, `D/geospatial.md:27,43,57,81` | `S/geospatial-analysis/datum-grid-transform/gpu-datum-grid-transform.ts:342,356`, `point-buffer/gpu-point-buffer.ts:223,230` | Zero mask, first covering grid/nodata, clockwise-from-north bearing, latitude clamp | 4/4 |
| Recording and raster, `D/computation-record.md:224,266,292,311`, `D/gpu-raster/raster-patches.md:34` | `S/computation-record/gpu-snapshot-arena.ts:84,154`, `gpu-transfer-mask.ts:100`, `submit-timed-command-graph.ts:43`, `S/raster-analysis/raster-patches/gpu-raster-sieve.ts:282,297` | 64 MiB default, single map/independent copies, nonzero selection, absent timing, sieve size/tie rule | 5/5 |

**Five weakest contributor checks for the public promises examined here**, ranked by the gap a reader can encounter:

1. `GPULinearReferencing`. Correct constructor behavior can pass while the cross-reference prescribes a property it ignores. A strict, explicitly imported geographic example would discriminate it.
2. `GPULineSegmentize`. Catalog path existence and a structurally valid `status: []` pass while a query omits the operator. A literal independent expectation for optional and required public status names would catch it.
3. `GPUNetworkStructure`. The source-backed SCC wording check targets the matrix. Restoring the same blanket bounds in the API guide passes that check. Exercise exact and auto declarations and name the two routes in the guide.
4. `GPUPointCloudStatisticalFilter`. The guide has no filter construction fence. A four-row fixture with a zero-support row and two multipliers would check retained slots, exclusion from moments and polarity independently.
5. `GPUColumnQuantiles`. The packed-only and rational-binding statements survive despite a multi-chunk declaration exceeding one binding. Pin per-port documentation and keep separate numerical cases for exact rational ranks.

There are discriminating tests beyond twin parity. The population fixture’s real quantile consumer, the three-node reachability graph, the hand-counted density grid and the closed-form variogram each distinguish a plausible wrong implementation. A further CPU alignment replay gave codes `[0,1]`, mask `[1,0]`, selected count 1 and median 10 for source IDs `[7,9]`, targets `[7,8]`, values `[10,99]`. Duplicate targets `[7,7]` gave codes `[2,2]`, mask `[0,0]`, zero selected and NaN median. The coregionalization walkthrough’s `[1,2,1]` symmetric triangle projects to four `1.5` entries and residual 1 by eigenvalue clipping. An explicit zero budget on `[2,-1,3]` yielded zero iterations and objective 15.

The optional-color correction at `73e8f638` has stronger pins than repeatability alone. `T/gpu-dataframe/column-classification/gpu-color-scale.spec.ts:711` compares requested subsets with the complete run and requires a nonzero total. `gpu-bivariate-classification.spec.ts:399` requires positive reference counts and equal IDs/counts without colors. Both would catch repeatable zeros. Those assertions were read. The unrelated raster comparator demonstrably returns zero for `([], [42])` and `([1], [1,42])`, and NaN for `([NaN],[1])`. The verified false acceptance is unequal length.

## Maintainability

**Catalog ownership.** Historical `makeOperator`, all-entry `makeEntryPointOperator`, coverage-family overrides and final annotation application each resolve part of the same contract. Summary and kind maps use additional keys. This split explains how full registration and a wrong status answer coexist. `O/all-entry-points/operator-annotations.ts` has 3,447 lines, `O/analysis-experience/analysis-data-roles.ts` 2,037, `O/all-entry-points/entry-point-operator-details.ts` 1,812 and `O/analysis-experience/analysis-interpretation.ts` 1,573. Preserve the cheap metadata-only import and the historical projection. Separate descriptor assembly from requirement/default/presentation code. No dead public contributor was established.

**Public surface tooling.** The contributor-contract checker is 2,074 lines. It resolves aliases and erased overloads that the vocabulary and audit checks do not. There are 365 inspected alias-validation call sites, including 182 literal arrays, six literal records and 177 expressions. Anonymous arrays can give positions in an error where a named port map could identify the view. Reuse qualified declaration traversal in development tooling. The inventory has no reason to live in a GPU runtime package or upstream luma.gl.

The eight removed public switches were `GPUCellAggregationProps.extraCounts`, `GPUCellRollupProps.extraCounts`, `GPUGeometryCleanupProps.forceSequentialRepeatRemoval`, `GPUSpatialScanStatisticProps.batchReplicates`, `GPUExpectationBasedSpaceTimeScanProps.batchReplicates`, `GPULocalKrigingProps.sampleIndexChunkQueryCount`, shared GWR `fittedScale`, and `GPUSpatialLagProps.rowStart`. The remaining internal declarations are study-window `slabs` and `packed` at `S/analysis-core/contracts/study-window.ts:43,58` and `GPUContributorOperation.createContributor` at `S/analysis-core/operations/gpu-contributor-operation.ts:176`. The two window members reach eight props types. `readPointCloudAttribute` and `writePointCloudAttribute` generate WGSL for authors. Lack of a user guide is insufficient grounds to delete them.

**Reference and release maintenance.** The dataframe, network operators, terrain operators and raster analysis reference pages each exceed 1,500 lines, as does spatial `operators-statistics.md`. Generated Parameters, worked examples, comparisons and changing hardware limits share these pages. The same SCC limit appears correctly in a matrix and incorrectly in a family guide. Prefer a canonical mode-specific paragraph with stable anchors and short references from overviews. Keep the ten former package guide paths as pointers.

**Changelog cross-check.** `git diff --name-only --diff-filter=AM bdc715ea..HEAD -- modules/analysis/src` yielded 624 added/modified TypeScript paths, including 41 `index.ts` paths. A syntax comparison found 697 changed existing named exported declarations, including internal and body-only changes. All eleven analysis-affecting breaking-change subsections were read. The ledger covers row/ID migration, compact versus candidate loss, typed parameter writes, alignment mask/code, Arrow, point-cloud `output`, raster graph methods, status renames, solver budgets and returned recipe shape. `CHANGELOG.md:123,200,212,277,299,321,349` gives migration points. DOC-106 is the supported omission: a slice added only `output.trips.status.candidateOverflow` to the formerly valid compact result shape and moved from the intended refusal to acceptance. Supplying staypoint loss also requires tripleg propagation. `T/gpu-spatial-analysis/trajectory-staypoints/cpu-trajectory-segments-operation.node.spec.ts:16` pins the refusal. The recipe-preparation summary at `CHANGELOG.md:184` should retain the qualification in `S/gpu-spatial-analysis/recipes/prepared-recipe.ts:44`: final caller-graph collisions, frozen state and allocation failures have no general rollback.

## Roadmap rows

Verdicts apply to the library or documentation clause named. The evidence does not close separate host, device-performance or publication gates.

| Row ID | Verdict | Evidence | Replacement current-limit text if narrowed |
|---|---|---|---|
| UF-01 | NARROW | `modules/analysis/README.md:10`, `modules/analysis/package.json`; package URLs and configured origin identify different repositories. | Confirm the canonical public repository and branch, then update the manifest, 26 README links and validation constants together. |
| UF-10 | NARROW | `dev-docs/authoring-walkthrough.md`, `dev-docs/release-package.md` and `dev-docs/contributor-contract.md` exist. | Authoring and release guides exist. Canonical URL, current artifact qualification and publication remain under LR-04. Delete the duplicate “no guide” task. |
| LR-04 and summary | NARROW, retain H | `modules/analysis/package.json:91`, `scripts/check-packed-consumer.mjs:339`, `T/analysis-core/release-package.node.spec.ts:82`. No local latest-release pointer was available to this review. | Set the canonical repository identity, retain a checked tarball tied to its source, complete the local release qualification and publish those bytes. Keep dated separate gate passes distinct from artifact identity. |
| LR-06, catalog/docs completion | CLOSE delivered clauses | `O/spatial-analysis-operators.ts:845`, `T/analysis-core/documentation-facts.node.spec.ts:153`, summary equality and ten canonical guides. | — |
| LR-08 / LR-24 | KEEP | `S/gpu-spatial-analysis/spatial-join/gpu-spatial-predicate-join.ts:679`, `D/gpu-spatial-analysis/operators-joins.md:36,58,81`. Multipart predicates and nearest modes have explicit subsets. | — |
| LR-10 | NARROW | `S/gpu-spatial-analysis/recipes/prepared-recipe.ts:138,153`; working `S/gpu-spatial-analysis/landscape-ecology/segregation-decomposition.ts:70,88,142` preflights randomization before caller declarations. | Preserve the recipe-index preparation guarantee and exclusions for final graph mutation. Date or remove the segregation randomization example against the integration revision. Audit domain recipes separately. |
| LR-17 / STD-12 | NARROW | Contract baseline retains 16 raster and five network root files and seven multi-class files. Ten package guides now have canonical destinations. `T/analysis-core/doc-publication.node.spec.ts:65`. | Retain the named source-layout exceptions. For docs, assign class/entry anchors or explicit grouped ownership to the 289 classes, and split the oversized references without breaking anchors. |
| LR-22 / KI-03 | KEEP | `dev-docs/roadmap.md:64,82`, `T/analysis-core/first-run-docs.node.spec.ts:40`. Integration target is alpha.16 and peers accept `~10.0.0-alpha.16`. | — |
| KI-01 and LR discriminating-test rule | KEEP | Optional-color subset pins and the raster short-array counterexample distinguish useful assertions from nominal coverage. `T/gpu-dataframe/column-classification/gpu-color-scale.spec.ts:711`, `T/gpu-raster/raster-time-series/raster-time-series-test-utils.ts:92`. | — |
| CPU-05 / INT-03 | KEEP | Zero of 555 records carries `cpuTwin`. The reader at `73e8f638:website/scripts/extract-inventory.ts:17,410` still recognizes forms by filename. Qualified joins there already use package identity. | — |
| STD-04 | NARROW, retain H for loss propagation | `S/gpu-network/gpu-network-kernel-density.ts:51,57`, `S/gpu-terrain/terrain-processing/gpu-terrain-depression-hierarchy.ts:85,91`, `S/gpu-spatial-analysis/spatial-clustering/gpu-optics.ts:166,181`. | All three cited legacy APIs expose `output.ports.status`. Resolve remaining producer-to-consumer loss, packed compatibility and per-row/per-model code documentation separately. |
| STD-05 | NARROW | `S/pointcloud-analysis/operations/gpu-pointcloud-height.ts:111`, neighborhoods parameter view and surface parameter exports. | Point-cloud parameter and grid-parameter views exist. Document which dimensions, capacities and view shapes require reconstruction; retain the other parameter-adoption work. |
| STD-07 | KEEP | Nineteen `outputValidity` aliases and the spectral base remain. `D/pointcloud-analysis.md:73` and `D/gpu-spatial-analysis/landscape-ecology.md:400` illustrate different roles. | — |
| STD-09 | NARROW | `S/analysis-core/index.ts:46` exports `validateSinglePackedView`. `T/analysis-core/gpu-contributor-utils.node.spec.ts:157` exercises the helper. | The packed-view validator is public. Retain only remaining ownership and compatibility work. |
| STD-11 | NARROW | `S/gpu-spatial-analysis/landscape-ecology/segregation-counterfactual.ts:255,266` uses `RecipeBuilder` and returns the canonical population result. Working row `dev-docs/roadmap.md:395` groups it among thirteen that do neither. | Separate inclusion in the prepared-declaration matrix, use of `RecipeBuilder`, and public result compatibility. Population alignment satisfies the latter two. Check other domain recipes and optional branches individually. |
| STD-13 | NARROW | 555 qualified records, family difference zero, 14 historical IDs and six false status answers. `O/all-entry-points/analysis-operator-catalog.ts:117,127`, `dev-docs/coverage-matrix.md:231`. | Structural registration, final family membership and the Dorling matrix correction are complete. Correct known-empty status, establish canonical documentation targets and specify a versioned migration for 14 historical IDs. |
| STD-14 / known chunked input | KEEP | `S/dataframe-analysis/column-classification/gpu-column-quantiles.ts:198,213` accepts chunked rows. Other scratch/auxiliary ports retain separate binding limits. | — |
| STD-15 | NARROW | Authored output occurrences/optional: `minimums` 4/3, `minimum` 1/1, `counts` 33/19, `cellCounts` 1/1, `validCount` 5/2. | Decide equivalent meanings before renaming. Distinguish grouped statistic vectors, raster cell counts, per-sample valid counts and inherited compact `status.count`. |
| Decision STD-04 / STD-09 | NARROW | All three named ports exist, and the packed validator is public. KDE status is per-query. | Decide retention or deprecation of each legacy layout. Remove the already-settled publication and port-existence decisions. |
| Decision STD-07 / STD-15 | NARROW | All six add-only raster classes migrated. Two mutable return declarations, iteration names, validity vocabulary and statistic spellings remain. | Decide coordinated treatment of validity, condition results, iteration/cardinality and statistic names, plus the two mutable return types. Remove `addToGraph` migration from the decision. |
| DUP-02 | KEEP | `GPUSpatialKMeans` is publicly implemented and distinct from embedding K-means. `D/gpu-spatial-analysis/operators-statistics.md:1562`. | — |
| DUP-01, kriging compatibility | KEEP | `S/gpu-spatial-analysis/spatial-interpolation/kriging-compatibility.md:12,30` records different duplicate, exact-hit, model and neighbor contracts. | — |
| GPU-13 | KEEP code limits, REPRIORITISE documentation before alpha | `S/dataframe-analysis/column-classification/gpu-column-quantiles.ts:198,213`; class breaks still enforces 2²⁴, exact natural breaks has separate 1,024/8,192 routes. | — |
| EMB-02, streaming codec widths | CLOSE delivered clause | `D/gpu-raster/raster-embedding-codecs.md:87`, `S/raster-analysis/raster-embeddings/raster-embedding-chunks.ts:115`. Streaming width up to 1,024 and chunking ship. The 128-wide reference covariance route is separate. | — |
| NW-SCC, matrix/API correspondence | REPRIORITISE before alpha | `dev-docs/coverage-matrix.md:255` is route-specific, `D/gpu-network/expansion.md:44` is stale. | — |
| GPU-11 / GE01–GE08 / UR01–UR05, urban tessellation | KEEP implementation limits, REPRIORITISE reference correction before alpha | `S/gpu-spatial-analysis/urban-morphology/building-tessellation.ts:130` and `building-tessellation-local.ts:44`. Local supports up to one million buildings with bounded enclosure/certification work. Dissolved rings remain on the 64-building reference route. | — |
| Known visibility graph | CLOSE identity/discovery clause, KEEP numerical limits | `O/all-entry-points/analysis-operator-catalog.ts:83,139`; `S/gpu-network/visibility-graph/visibility-graph.ts:63`. Distinct network metrics and spatial adjacency are published. | — |
| Known space-time kriging | KEEP row, correct API page | `S/gpu-spatial-analysis/spatial-interpolation/gpu-space-time-kriging.ts:217`. Automatic index above 4,096 samples and explicit scan cap are distinct. | — |
| Known space-time variogram | NARROW | `S/dataframe-analysis/pair-statistics/gpu-space-time-variogram.ts:32,86` and `cpu-variogram-model-fit-operation.ts:49,152`. | Keep required index topology, deprecated cross-variogram default and eight-float layout. CPU fit exposes per-model status 3 and bracket diagnostics, without a separate convergence boolean. |
| Known CPU arithmetic / CPU output tails | KEEP | `D/operation-forms-and-cpu-twins.md:32,192` qualifies precision. CPU path-extraction tails and GPU unwritten compact tails have distinct contracts. | — |
| Known CPU budgets | NARROW | `S/dataframe-analysis/pair-statistics/cpu-coregionalization-fit-operation.ts:54`. Explicit zero budget executed zero iterations. | Twins may omit GPU implementation ceilings. Explicit algorithm budgets bind where documented. State defaults, candidate limits and convergence rules per operation. |
| IN01 | NARROW | `dev-docs/coverage-matrix.md:254`, `D/gpu-spatial-analysis/interpolation-expansion.md:158`. The later record quotes 0.7 ms warm at 256 bins versus 2.3–2.5 ms CPU round trips; small fits had no consistent winner. | Keep residency guidance. Attribute serial and cooperative timings to the exact implementation, size, transfer boundary and dated device run. |
| AX defaults, interpretation and remaining host qualification | KEEP open host gates | `O/analysis-experience/analysis-interpretation.ts:452`, `CR:29` settle a library interpretation defect. They do not satisfy the AX-04/20/33/40/50/60/70 screenshot, latency, hand-off or browser gates at `dev-docs/roadmap.md:291`. | — |

## Next steps

### Before the alpha

| Rank | Action | Who needs it | Cost and completion evidence |
|---:|---|---|---|
| 1 | **Finish catalog status semantics.** Correct six false absences, distinguish omitted historical metadata from audited empty lists, and check each filter with explicit independent names. | Users selecting operators for bounded pipelines and maintainers of catalog-driven hosts | **S** for the six records and pins, **M** to audit the remaining known-empty records. Both qualified and legacy queries must find all six. |
| 2 | **Finish public instructions and migration notes.** Correct the two entry suffixes, geographic context and units, route-specific limits, status-vector examples, K-means anchors and trajectory migration. | First-time package users, users sizing workloads, and callers upgrading a loss-bearing pipeline | **S** for the corrections, **M** for fixtures. The printed imports must resolve, the geographic declaration must select the intended route, and both old/new trajectory calls must have named outcomes. |
| 3 | **Connect docs checks to the published API.** Validate prose entry lists, explicit package imports, canonical fragments and first-run fixtures. Give fragments typed inputs and complete examples an independent expected output. | Documentation authors and release reviewers | **M**. Include negative fixtures for an unknown constructor, a resolvable private subpath, a stale option and an anchor on an existing page. |
| 4 | **Finish the release record.** Set the public repository identity, update the 26 package README links together, and retain the source identity, tarball and qualification evidence for the release candidate. | Maintainer and installed-package readers | **S** for metadata and documentation after the maintainer decision. Release qualification is separate work at iteration end. |
| 5 | **Repair misleading tests.** Require equal raster comparison lengths and register entry-qualified kernel-audit cases for network visibility and transfer masks. | Raster, network and recording maintainers | **S**. A shortened array must fail and removing the network recipe case must produce a qualified missing identity. |

### After

| Family or surface | Action | Who needs it | Cost |
|---|---|---|---:|
| Catalog and analysis experience | **Consolidate** descriptor finalization while retaining metadata-only imports. **Connect** each public operator to its concrete CPU/GPU forms, pure function, guide and qualified alternatives. **Freeze** the 14 historical capability IDs until a versioned migration exists. | Discovery clients, mixed-graph authors and historical catalog readers | M, M, S |
| API and core tooling | **Consolidate** qualified declaration inventory across contract, docs and audit checks. **Finish** role-based vocabulary decisions for validity, counts and status. Preserve documented per-row codes and packed compatibility until callers have a migration. | Contributor authors and integrators composing families | M, M |
| Spatial and network documentation | **Connect** multipart row space, application IDs, candidate loss and final loss with small composition examples. **Finish** canonical anchors for 18 spatial and nine network producers. **Freeze** kriging consolidation pending compatibility fixtures. | Join, routing and interpolation users | M, M, S |
| Raster, terrain and geospatial documentation | **Finish** literal terrain and distance journeys with closed-form grids. **Consolidate** layout/units, validity and route-specific bound tables around owner pages. Remove repeated unconditional bounds from overview prose. | Raster pipeline and terrain users, especially users switching physical units or grid routes | M, M |
| Dataframe and point-cloud documentation | **Finish** time-filter and statistical-filter examples with compact tails and unsupported rows. **Connect** per-port chunking, parameter views and result decoders where callers otherwise reconstruct layouts. | Users feeding selections, grids and neighborhood results into the next operator | M, M |
| Recording, testing and release docs | **Finish** copy-only versus compute audit classifications. **Connect** migration notes to required-port and compatibility fixtures. Delete duplicate settled roadmap clauses while retaining dated release evidence. | Recorder authors and release maintainers | S, M |

## Limits of this report

This is a review of the library as a caller discovers, imports, configures and composes it. The factual sample covers 62 claims and the error sample covers 40 selected sites. The remaining prose, most WGSL bodies, most numerical tests, every branch of analysis experience, all 348 catalog limits and all 697 changed named source declarations did not receive individual semantic verification. Unknown catalog values and absent class headings have explicit denominators. Neither is counted as a numerical defect.

No clean installation, public repository, registry publication, deployed page, browser journey, performance result or device output was independently qualified. Source-backed device fixtures are identified where they discriminate a fix. The unreviewed implementation rounds have stronger evidence here for the named fixes and later color-output change, but this area cannot certify those rounds library-wide or assign a newly observed defect to a particular round without history supporting that attribution. The separate staging pass and concurrent working-tree edits require a final reconciliation against the exact release revision.