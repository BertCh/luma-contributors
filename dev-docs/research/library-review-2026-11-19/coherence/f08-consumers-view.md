# What consumers had to work around

## Summary

Kepler has **229 executable bindings and 229 hand-written descriptors**. Every registered binding resolves to a hand-written descriptor. The generic fallback accounts for zero of them.  
The five lens packs maintain another **427 operator or recipe entries**. Four packs alone assign meanings to **1,604 named captures**.  
The most consequential failures happen before dispatch. A missing measurement becomes a measured zero, a clip ignores parts of its polygon, and a planar packing path turns two distinct vertices into one.  
The metadata also changes what users can run. Kepler blocks a CPU clustering job at 6,324 rows that its own runner completed at 8,000. Its discovery panel omits 176 analysis catalog records even after both of its supplemental discovery passes.  
The most valuable pre-alpha change is a **qualified, mode-specific declaration of the actual ports and parameters**, shared by constructors, packers, result readers and consumer bindings. Start with line geometry, scoring and grid operations, where existing adapters supply executable counterexamples.  
Keep graph ownership, display text, host memory budgets and publication with the consumer. The analysis package can remove substantial duplication by describing the data crossing that boundary.

## Census

`A` = `modules/analysis/src`, `K` = `modules/kepler/src`, `L` = `modules/lenses/src`, `E` = `examples/deck/kepler-compute/src`. Paths refer to the `r19` checkout unless marked `ki`.

| Measured variant | Lenses | Kepler | Examples |
|---|---:|---:|---:|
| `rg -n "from '@luma-contributors/analysis"` matches / matching files | 791 / 555 | 219 / 157 | 735 / 486 |
| Actual static analysis imports in JS/TS / importing files | 791 / 555 | 219 / 157 | 564 / 388 |
| Direct `new` expressions for named analysis imports / distinct imported class names | 621 / 386 | 217 / 178 | 977 / 375 |
| `.createDataView()` / `.readAsync()` calls across the root | 78 / 1 | 19 / 9 | 115 / 148 |
| `.readScalar()`, `.readUint32()`, `.readInt32()`, `.readFloat32()` or `.readFloat64()` | 0 | 580 | 10 |
| Function declarations named `pack*`, `unpack*`, `decode*`, `read*Status`, `read*Summary` or `get*Status` | 7 | 11 | 85 |
| Direct `as any` or `as unknown` assertions across the root | 2 | 17 | 149 |

The second through last rows used `rg --files` and `ts.createSourceFile` over 1,045, 295 and 1,962 JS/TS files respectively. Constructor counts resolve local import aliases. Method and assertion counts are inventories, including legitimate host work. Examples also contain prose and snippets, hence the difference between the first two rows. The most frequently constructed imports were `GPUParameterBuffer` in lenses and Kepler, and `GPUNeighborSearch` and `GPUTerrainDerivatives` in examples.

| Registry or table | Executed count | What the count covers |
|---|---:|---|
| `getGPUAnalysisOperatorCatalog()` | 557 | 502 GPU-prefixed records. 179 records have `limits`, four have `forms`, zero have a `parameters` schema or a named `ports` schema. |
| `KEPLER_CATALOG_BINDINGS` | 229 | 95 GPU, 72 GPU-or-CPU, 62 CPU-reference. Their contributor-name lists mention 239 distinct names. |
| Binding input / output / parameter declarations | 277 / 299 / 1,124 | Parameter appearances comprise 316 enums, 282 numbers, 181 integers, 53 booleans, 276 strings and 16 number lists. |
| `KEPLER_OPERATION_DESCRIPTORS` | 229 | `resolveKeplerOperationDescriptor(b).source === 'hand-written'` for all 229. `wc -l modules/kepler/src/descriptors/descriptors-*.ts` gives 11,592 lines across 27 files. |
| Kepler `analysis/catalog-*.ts` | 100 files, 45,805 lines | Includes runners, interchange and publication as well as declarations. This is the maintenance surface, **not** an estimate of deletable lines. |
| Lens operator tables | 260 spatial, 54 raster, 46 terrain, 26 dataframe, 41 network | Respectively 222, 54, 46, 26 and 36 GPU-prefixed keys. The other entries include recipes. |
| Lens `variables` entries | 1,045 spatial, 273 raster, 193 terrain, 93 dataframe | Counted `Object.keys(entry.variables).length` per entry. The network operator table uses another shape. |
| Host limits and allocations | 13 grid-cap IDs, 16 CPU-work-cap IDs, 643 literal `session.output(` sites in 69 Kepler files | Counted `Object.keys(KEPLER_GRID_CELL_CAPS)`, `Object.keys(KEPLER_CPU_WORK_CAPS)` and `rg -o 'session\.output\(' modules/kepler/src`. |

For reproducibility, the registry census is equivalent to the following over the imported TypeScript source. All runtime probes in this report ran through esbuild to Node stdout. No GPU execution was simulated as evidence of a shader result.

```ts
const bindings = KEPLER_CATALOG_BINDINGS;
const operators = getGPUAnalysisOperatorCatalog();
const visible = completeNetworkCategoricalCatalogEntries(
  includeSpacetimeCatalogEntries(listCatalogEntries({lensScenarios: []}))
);
const missing = operators.filter(
  operator => !visible.some(entry => entry.contributorName === operator.exportName)
);
console.log(bindings.length, operators.length, visible.length, missing.length);
// 229 557 383 176
console.log(bindings.filter(b => resolveKeplerOperationDescriptor(b).source === 'derived').length);
// 0
```

## Proposed convention

An author declares each supported mode once under `{package, exportName}`, including parameter constraints and resolved units, named input/output paths, formats, valid lengths, row identities, missingness and status meanings. A host binding references that declaration and supplies its own data mapping, tighter resource limits and presentation. A checker should instantiate each described mode with caller-owned views, compare the declaration with its public props, and round-trip a valid row, a missing row and an empty result through the binding.

## Findings

| ID | Kind | Reach | Change | Cost | Where | One line |
|---|---|---|---|---|---|---|
| UCO-01 | W | ORDINARY | ADDITIVE | S | `E/analysis-catalog-lens-mapping.tsx:49` | A missing mapped value becomes a measured zero. |
| UCO-02 | W | ORDINARY | ADDITIVE | S | `K/analysis/catalog-geometry.ts:585` | Convex clip silently ignores holes and all but the first clip polygon. |
| UCO-03 | W | ORDINARY | ADDITIVE | S | `K/analysis/catalog-platform4-geometry-lines.ts:257` | Raw planar packing collapses vertices, and an independent area reader loses translated area. |
| UCO-04 | W | UNUSUAL | ADDITIVE | M | `L/computation-lenses/user-input.ts:35` | An embedded run can read another active run’s mapped data. |
| UCO-05 | W | UNUSUAL | ADDITIVE | S | `K/analysis/catalog-trajectory-common.ts:155` | Numeric and text track IDs can silently join two trajectories. |
| UCO-06 | W | ORDINARY | ADDITIVE | S | `L/gpu-raster/computation-pack/create-raster-pack.ts:62` | Ten raster operators can get another operator’s measurement type. |
| UCO-07 | D | ORDINARY | ADDITIVE | S | `E/analysis-sheet.tsx:275` | The form labels geographic grid distances “CRS units” while the runner measures metres. |
| UCO-08 | W | UNUSUAL | ADDITIVE | S | `K/analysis/catalog-platform4-geometry-common.ts:126` | Empty output can become a positive line count or a polygon made from a hole. |
| UCO-09 | W | ORDINARY | ADDITIVE | S | `K/analysis/catalog-raster-embeddings-bindings.ts:115` | PCA provenance advertises a CPU eigensolver as a dispatched GPU contributor. |
| UCO-10 | C | ORDINARY | BREAKING | M | `K/analysis/catalog-bindings.ts:1005` | Discovery drops 176 records and conflates the two visibility recipes. |
| UCO-11 | S | ORDINARY | ADDITIVE | S | `K/descriptors/default-parameters.ts:316` | A stale CPU work formula blocks runnable DBSCAN inputs. |
| UCO-12 | S | UNUSUAL | ADDITIVE | S | `K/analysis/catalog-platform4-geometry-common.ts:25` | Independent coordinate collectors hit the JS argument limit inside admitted sizes. |
| UCO-13 | E | ORDINARY | ADDITIVE | M | `K/analysis/catalog-validation-scores.ts:37` | Three column-based operations insist on a feature collection. |
| UCO-14 | E | ORDINARY | ADDITIVE | S | `L/computation-lenses/user-input.ts:131` | A one-column quantile scenario offers sixteen roles and discovers required ones by running. |
| UCO-15 | C | ORDINARY | ADDITIVE | S | `K/analysis/catalog-bindings.ts:557` | Form defaults share mutable arrays, and admission cannot express a nonzero parameter. |
| UCO-16 | E | ORDINARY | ADDITIVE | S | `K/analysis/catalog-validation-prediction.ts:334` | Conformal summaries hand-decode three conditions and omit neighbor truncation. |
| UCO-17 | E | ORDINARY | ADDITIVE | M | `E/analysis-ols-outputs.ts:30` | Even a consumer that knows the operator must reproduce output formats and allocation arithmetic. |

### UCO-01: Preserve missing measurements at the first conversion

**Where.** `E/analysis-catalog-lens-mapping.tsx:35,49`, called from `E/analysis-catalog-panel.tsx:881`. Compare `K/analysis/catalog-lens.ts:92` and `L/computation-lenses/scenarios/gpu-dataframe/column-quantiles.run.ts:23`.

**EXECUTED.** `readLensUserData` on one native `real` field containing `[null, 0, NaN, undefined, 12]`, mapped to `poverty150Pct`, returned `[0, 0, null, null, 12]`. Expected `[null, 0, null, null, 12]`. The downstream quantile scenario already converts null to `NaN`. `Number(valueAt(...))` destroys the distinction before it gets there. This differs from KNOWN(SEAM-103), which concerns rendering an already missing output.

**Wanted.** Use the nullable numeric reader already implemented in `mapCatalogLensUserData`, or share its scalar conversion at both entry points. Accept a finite number according to the field contract and preserve null. **Callers:** one panel path, no signature change. **Value/cost:** analysts retain the correct quantile population, S.

### UCO-02: Validate the entire clip operand

**Where.** `K/analysis/catalog-bindings.ts:294,310,332`, `catalog-geometry.ts:585`, `catalog-gpu-geometry.ts:588`, `catalog-gpu-pack.ts:314`. The registered contributor is `GPURectangleClip`.

**EXECUTED.** I called `runCatalogOperation(binding, {}, {features, clip})`. A line `[[3,3],[4,4]]` wholly inside the hole of a `[0,10]²` square with a `[2,8]²` hole came back unchanged. Its only diagnostic was `planar-edges`. A line wholly inside the *second* polygon of a two-square MultiPolygon produced zero rows and `feature-outside-clip`. Both routes take `[0]?.[0]` before testing the clip. Independently, `getCatalogRectangle([[0,0],[1,1],[1,0],[0,1],[0,0]])` accepted the crossed ring and returned its bounding box.

**Wanted.** Declare `polygonCount === 1`, `ringCount === 1`, convex simple shell for this binding. Reject a clip outside that domain before either engine runs. Require consecutive axis-aligned edges for the rectangle fast path. **Callers:** one binding, two runners, no valid-call migration. **Value/cost:** a GIS user gets a correct intersection or a specific refusal, S.

### UCO-03: Use a local frame before narrowing coordinates

**Where.** `K/analysis/catalog-platform4-geometry-lines.ts:257,319` (`GPULineChunk`), `catalog-platform4-geometry-common.ts:57`, `catalog-gpu-pack.ts:79,249,285`, `catalog-gpu-geometry.ts:947,1113,1265,1489` (buffer, overlay and topology-repair area publication).

**EXECUTED.** The line-chunk input packer turned `[[20000000,0],[20000001,0]]` into `[20000000,0,20000000,0]`. Using the existing `makeLocalFrame('planar', line, {shift:'center'})` first produced `[-0.5,0,0.5,0]`. Independently, `getCatalogPolygonArea` returned `1` for a unit square at the origin and `0` for the same square translated by `[1e8,1e8]`. The sibling `measureRingSignedArea(ring,'planar')` returned `1` for the translated square. These are host results. This adds two distinct mechanisms to KNOWN(SEAM-106).

**Wanted.** Give a packed planar input a shared origin and inverse transform. Apply the inverse on line readback. Reuse the translated shoelace calculation for ring areas. Keep the geographic great-circle path in its declared angular coordinates. **Callers:** the existing public calls retain their shape. **Value/cost:** projected-coordinate users retain metre differences and translation-invariant areas, S.

### UCO-04: Scope mapped input to the run that requested it

**Where.** `L/computation-lenses/user-input.ts:34,42,85`, `L/computation-lenses/scenarios/gpu-dataframe/dataframe-lens-support.ts:69`, `K/analysis/catalog-lens.ts:133`.

**EXECUTED.** I paused a mapped run after its loader entered, then called the ordinary `loadDataFrameLensData('f-chicago-tracts')`. Before, during and after the mapped run, `(rowCount, first poverty150Pct)` was **`(791,27.6) → (1,99) → (791,27.6)`**. The ordinary loader should have returned `(791,27.6)` all three times. The queue serializes mapped calls against one another. Ordinary scenario loads bypass it.

**Wanted.** Pass an optional run-scoped dataset resolver through the scenario options. The default resolver loads the embedded data. The mapped resolver closes over that invocation’s input and accessed roles. **Callers:** the three dataframe dataset-loader branches and the mapped-run wrapper can adopt it without changing existing calls. **Value/cost:** simultaneous analyses cannot consume one another’s rows, M.

### UCO-05: Preserve track identity before grouping

**Where.** `K/analysis/catalog-trajectory-common.ts:129,148,155,200`, `catalog-trajectory-bindings.ts:164,169`, `K/profile/native-dataset-profile.ts:178`.

**EXECUTED.** Two Point features with IDs `7` and `"7"`, times `0` and `1`, produced `trackIds = ["7"]`, `trackOffsets = [0,2]`. Expected two tracks with `[0,1,2]`, or a diagnostic that the binding cannot publish these identities distinctly. Grouping uses `String(identity)` before constructing the track map.

**Wanted.** Use typed identity in the trajectory packer and a stable row-to-track index. The cheapest compatible repair is to reject collisions when projecting IDs into the existing string output column. **Callers:** three preparation call sites cover 13 registered operations and native profiling. **Value/cost:** fixes on different tracks cannot become a synthetic movement segment, S.

### UCO-06: Resolve measurement kind by operator and port

**Where.** `L/gpu-raster/computation-pack/create-raster-pack.ts:58`, `L/computation-views/registry/view-registry.ts:85`, `L/computation-views/hosts/lens-controller.ts:363`. I checked **both** `r19` `L/computation-views/views/value-strip.ts:60` and the newer **`ki`** `value-strip.ts:65`. Both look up the bare capture name in `pack.variableKinds`.

**EXECUTED.** `getDeclaredVariableKind({packs:[getRasterViewPack()]}, 'values')` returned `{level:'ratio', extent:'extensive'}`. `pack.operators.GPURasterFillHoles.variableKinds.values` returned `{level:'nominal'}`. Comparing every raster operator’s declaration with the pack-wide result identified ten disagreements on two names. `Object.assign(variableKinds, kinds)` makes family registration order decide the answer.

**Wanted.** Resolve `(record operator, capture/port name)` before any pack-wide fallback. Expose a pack-wide kind only for names on which every declaring operator agrees. **Callers:** existing per-operator maps can supply the correction without changing records. **Value/cost:** categorical masks and spectral indices reach renderers with the intended scale, S.

### UCO-07: Display the unit in which the runner interprets the number

**Where.** `E/analysis-sheet.tsx:275`, `E/workflow/workbench-parameter-form.tsx:387`. I checked both checkouts for `K/analysis/catalog-platform3-b-bindings.ts` and `catalog-platform3-b-grids.ts`. The newer **`ki`** copies still assign `'crs'` at binding lines `150,290,480,582,705` and create a local metric frame at grid line `123`.

**EXECUTED.** `preparePointGrid` for `(0°,0°)` and `(0.01°,0°)` with `cellSize:100` produced **12 columns** under `EPSG:4326` and **one column** for the same numbers under `'planar'`. The geographic result is consistent with a roughly 1.1 km span and 100 m cells. Both form renderers print `CRS units`, which for the input geographic CRS are degrees. Across the registry, 129 parameter declarations in 93 bindings use `'crs'`. Some, such as the unprojected lattice, genuinely use the source units.

**Wanted.** Declare the unit per parameter and coordinate mode, e.g. `{geographic:'metres', planar:'coordinate-units'}`, or explicitly `source-coordinate-units`. Render the resolved unit. Start with the five grid bindings named below. **Callers:** two form renderers, no numerical default changes. **Value/cost:** analysts can enter a search radius or cell size on the intended scale, S.

### UCO-08: Decode the emitted prefix and preserve shell ownership

**Where.** `K/analysis/catalog-platform4-geometry-common.ts:126`; `catalog-platform4-geometry-lines.ts:176,265,303` (`GPULineSmooth`, `GPULineChunk`); `catalog-gpu-pack.ts:262`; `catalog-gpu-geometry.ts:863,1452` (overlay and `GPUMakeValid`).

**EXECUTED.** With one input line and a host readback stub supplying `count=0, overflow=0`, `runCatalogLineGeometry` returned `lineCount=1` because it uses `producedLineCount || lineCount`. Separately, `readCatalogPolygons` given a two-vertex shell followed by a three-vertex hole returned a polygon with the hole as its shell. Expected zero lines and zero polygons. These fixtures exercise the actual readers. They do not establish that a particular shader emits either fixture.

**Wanted.** Return the reported count, including zero. Add a shared offset-geometry decoder that reads only the valid prefix and rejects or skips the whole polygon when its exterior is absent. The rectangle-clip sibling already distinguishes an empty shell at `catalog-gpu-geometry.ts:663`. **Callers:** two line runner uses and two polygon reader uses. **Value/cost:** an empty or structurally unusable result stays empty, S.

### UCO-09: Distinguish CPU stages from GPU dispatches

**Where.** `K/analysis/catalog-raster-embeddings-bindings.ts:26,31,102,115`, `raster-embeddings.ts:342,347`, `catalog-bindings.ts:1031`; `A/raster-analysis/raster-embeddings/raster-embedding-principal-components.ts:36`.

**EXECUTED.** `listCatalogEntries({lensScenarios:[]})` reported `getRasterEmbeddingPrincipalComponents` as `engine:'gpu'`, with bindings `raster-embedding-pca` and `raster-embedding-projection`. Both runners call the ordinary CPU eigensolver after covariance readback and push its name into the contributor list returned as `dispatchedContributors`.

**Wanted.** Describe each stage as `{identity, execution:'cpu'|'gpu'}` and derive the dispatch list from GPU stages actually submitted. Keep the CPU stage in provenance. **Callers:** two binding rows share one implementation. **Value/cost:** performance and reproducibility records identify where the work took place, S.

### UCO-10: Enumerate the qualified catalog once

**Where.** `K/analysis/catalog-bindings.ts:944,1005,1055`, `catalog-pointcloud-entries.ts:13`, `catalog-network-categorical-discovery.ts:16`, `catalog-lens.ts:100`; `E/analysis-catalog-spacetime.ts:37`, `analysis-catalog-panel.tsx:185,1045`. Both actual lens declarations exist at `L/computation-lenses/scenarios/gpu-network/index.ts:15` and `scenarios/gpu-spatial-analysis/urban-ecology.lens.ts:39`.

**EXECUTED.** The fully supplemented panel produced 383 entries. **176 canonical records had no matching name at all**, including 159 GPU-prefixed records. The missing counts were raster 64, terrain 40, dataframe 35, network 17, spatial 12, recording 4 and core 4. Examples include `GPURasterArithmetic`, `GPUTerrainFlowFractions`, `GPUWideNumericScan` and `GPUKShortestPaths`. In addition, two qualified `addVisibilityGraphRecipe` records became one entry. Passing a network lens scenario made the **spatial** “Building and street analysis” entry runnable by that scenario. The React key is also `entry.contributorName`. KNOWN(ERG-08) fixed analysis lookup; this consumer loses the identity again.

**Wanted.** Start from `getGPUAnalysisOperatorCatalog()`. Join bindings, lens scenarios and UI keys on `{package, exportName}`. Register upstream contributors as qualified extensions. Remove the 22-name network supplement and the point-cloud, categorical and spacetime discovery passes. Distinguish discoverable from runnable with the existing fields.

**In-repository callers that change.** `rg '\blistCatalogEntries\('` over code found **13 calls**, one production panel and 12 test calls. The panel’s list key and the name-based joins must migrate. Tests selecting a unique name can retain that convenience. **Value/cost:** every family becomes discoverable and a user selecting a recipe gets the right computation, M. The change breaks the assumption of one row per bare name.

### UCO-11: Derive admission from the runner’s current limit

**Where.** `K/descriptors/default-parameters.ts:294,316,336`, `applicability.ts:2349`, `K/analysis/catalog-platform3-b-clusters.ts:70`. `modules/kepler/test/descriptors/applicability.node.spec.ts:1784` explicitly expects the obsolete rejection.

**EXECUTED.** With an 8,000-row point profile, `evaluateKeplerOperationApplicability(...,{executionEngine:'cpu'})` returned `blocked` and “at most 6,324 on the CPU engine.” Running that same `density-clustering` binding on 8,000 planar points spaced two units apart, `epsilon:1`, `minimumPoints:2`, returned **8,000 point rows, 8,000 noise points, zero clusters**. The current CPU implementation uses spatial buckets and admits up to 20,000 points. The form still asserts `rows² ≤ 4e7`.

**Wanted.** Remove the copied matrix limit from `density-clustering` and audit the identical entry for `curated-clusters-and-footprints` against its additional stages. Put each engine’s hard admission predicate next to the engine. A host may separately impose and label an interactive budget. **Callers:** two cap entries and one pinned rejection test. **Value/cost:** valid analyses become available on CPU-only machines, S.

### UCO-12: Iterate vertices without spreading them into a function call

**Where.** `K/analysis/catalog-gpu-pack.ts:88`, `catalog-platform4-geometry-common.ts:25`, `catalog-platform3-b-common.ts:134`, `catalog-platform3-a-shared.ts:110`, `catalog-extra-geometry.ts:41`. Related per-ring uses occur at `catalog-geometry.ts:244` and `catalog-platform3-a-measure.ts:153`.

**EXECUTED.** One 150,000-vertex LineString caused `RangeError: Maximum call stack size exceeded` in each of `createCatalogFrame('EPSG:4326', ...)`, `collectCatalogPositions`, `collectAllPositions` and the platform-3-A `collectPositions`. Multiple feature specs allow 200,000 vertices, including `catalog-binding-types.ts:260`. The problem is `positions.push(...part)`, independent of the published vertex bound.

**Wanted.** Use a common iterator over feature, part, ring and vertex. Stream frame bounds directly or append with a loop when a packed array is required. **Callers:** five collector implementations can share traversal; their public signatures can stay. **Value/cost:** a valid coastline or road does not fail according to the JS engine’s argument limit, S.

### UCO-13: Declare numeric columns independently of display geometry

**Where.** `K/analysis/catalog-validation-scores.ts:25,37,83,225`, `catalog-validation-common.ts:152`, `catalog-data-input.ts:63`, `K/descriptors/descriptors-validation.ts:436,495`.

**EXECUTED.** A valid one-row native table with `observed`, `predicted`, `variance` and `pvalue` passed `makeCatalogTableInput`. All three bindings rejected the table at input admission with `points input must be a FeatureCollection`. `GPUGaussianPredictiveScores` consumes numeric forecast columns, `GPUEnsembleScores` consumes members and row-pair indices, and `GPURasterFalseDiscoveryRateMask` consumes indexed p-values. None of these binding calculations reads a point coordinate. Geometry is used when publishing the result.

**Wanted.** Add a table input route for `validation-gaussian-scores`, `validation-ensemble-scores` and `validation-fdr`. Describe numeric roles, row order and optional passthrough geometry separately. **Callers:** three bindings, existing feature callers continue to work. **Value/cost:** a forecast or p-value table can be scored without manufacturing GIS features, M.

### UCO-14: Declare the columns a scenario actually reads

**Where.** `L/computation-lenses/user-input.ts:70,127,141`, `K/analysis/catalog-lens.ts:59,70`, `E/analysis-catalog-lens-mapping.tsx:61`, `L/computation-lenses/scenarios/gpu-dataframe/column-quantiles.run.ts:23`.

**EXECUTED.** `listLensUserInputRoles` for the Chicago tracts dataset returned **16** names, from `tract` to `y`. The `GPUColumnQuantiles` scenario reads **one**, `poverty150Pct`. A host learns a missing required role by attempting the run. Kepler also repeats the same three-dataset eligibility rule already exported by lenses as `getLensUserInputDatasetId`.

**Wanted.** Give this scenario a mapping such as `{port:'values', role:'poverty150Pct', required:true, missing:'exclude'}`. Project the mapping form from the scenario’s declared bindings to analysis ports. Reuse `getLensUserInputDatasetId` immediately. **Callers:** one mapping form and one duplicated eligibility predicate; scenario authors add their actual mappings. **Value/cost:** the first quantile run asks for one field with a clear purpose, S for this path.

### UCO-15: Make parameter specifications safe to reuse

**Where.** `K/analysis/catalog-bindings.ts:557,568,592`, `catalog-raster-embeddings-bindings.ts:169`, `lagrangian-diagnostics.ts:36`; parameter factories at `catalog-platform3-a-shared.ts:60`, `catalog-platform3-b-common.ts:58` and `catalog-trajectory-common.ts:49`.

**EXECUTED.** In an isolated Node process, changing `getCatalogDefaultParameters(modalSplit).modeSpeedBreaks[0]` from `1.5` to `222` made both a new defaults request and `normalizeCatalogParameters(binding,{})` return `222`. Sixteen list defaults share this exposure for JavaScript callers. Separately, `normalizeCatalogParameters(ftle,{timeStep:0})` succeeded, then the runner immediately refused “a finite nonzero time step” before GPU allocation. For IDW, `cellSize:-1` yielded only `Parameter out of range`, despite the spec containing the parameter name and minimum.

**Wanted.** Copy list values when materializing editable defaults or normalized invocations. Add a nonzero numeric constraint and use the same spec for parsing, error formatting and the form. For example, `idw-grid: cellSize must be at least 0.000001; received -1`. **Callers:** central defaults and normalization functions, one FTLE parameter declaration. **Value/cost:** editing one form cannot alter the next run’s defaults, and invalid input is explained at entry, S.

### UCO-16: Publish status meaning once and count every relevant bit

**Where.** `K/analysis/catalog-validation-prediction.ts:204,316,334`, `A/gpu-spatial-analysis/spatial-validation/index.ts:17`. I checked the newer **`ki`** `gpu-conformal-prediction-intervals.ts:42–50` and `gpu-geo-conformal-prediction-intervals.ts`; the public truncation bit remains `16`.

**EXECUTED.** Applying the binding’s summary expressions to status words `[16,17,8]` gave `unbounded_rows=1`, `invalid_query_rows=0`, `no_calibration_rows=1`. Two rows carry the exported `GPU_CONFORMAL_STATUS_NEIGHBORS_TRUNCATED` bit. The diagnostic table has no corresponding metric. The raw row status does survive.

**Wanted.** Export a pure `decodeConformalStatus(word)` beside the existing constants and derive the diagnostic counters from it. Add `neighbors_truncated_rows` to the geographic binding. **Callers:** the two conformal bindings share one summary builder. **Value/cost:** analysts can see when local calibration was bounded without deciphering integer columns, S.

### UCO-17: Expose output requirements without taking ownership of buffers

**Where.** `A/gpu-spatial-analysis/ontology/spatial-analysis-ontology.ts:413`, `K/analysis/catalog-gpu-session.ts:88`, `E/analysis-ols-outputs.ts:30,52,86`. I checked both OLS implementations; the newer **`ki`** `A/gpu-spatial-analysis/spatial-regression/gpu-ordinary-least-squares.ts:297,302,308` still enforces coefficient, summary and status lengths.

**Docs-first probe.** From `docs/api/analysis-catalog.md:18,40`, I tried the documented qualified lookup as the starting point for wiring a line operation:

```ts
const op = getGPUAnalysisOperator({
  package: '@luma-contributors/analysis/gpu-spatial-analysis',
  exportName: 'GPULineChunk'
});
console.log(op?.inputs, op?.outputs);
// ['lines', 'parameters'] ['lines']
```

**EXECUTED.** Inspecting that record gave `undefined` for `ports` and `parameters`. The public type supplies neither property. The lookup answers which operation to use. It cannot tell the caller to allocate `lineOffsets` with a terminal entry or which status views qualify the prefix. OLS shows the cost in a different family. Its example separately budgets `3 * 16 * 4` bytes, allocates three coefficient buffers of length 16, then re-imports each with active length `predictorCount + 1`. It already reuses the exported summary length.

**Wanted.** Add `describeOutputs(identity, mode, sizes)` returning named paths, format, capacity, active-length rule and associated status. For OLS, declare the distinction between maximum reserved coefficients and fitted coefficients. Let callers continue allocating, importing and submitting. **Callers:** optional adoption at 643 literal Kepler `session.output(` sites; 643 is an upper bound on opportunities, not a promised deletion count. **Value/cost:** hosts can budget and wire an operation from the same rules its constructor enforces, M for the first families. This makes KNOWN(DIR-109) testable on actual consumer code.

## Departures by name

- **Bare-name discovery:** `listCatalogEntries`, `listPointCloudCatalogEntries`, `completeNetworkCategoricalCatalogEntries`, `includeSpacetimeCatalogEntries`, `groupCatalogLensScenarios` and the panel React key. The two `addVisibilityGraphRecipe` exports are the observed collision. The `missing` expression in the census enumerates every omitted record.
- **Wrong raster variable-kind lookup:** `GPURasterNDWI`, `GPURasterNBR`, `GPURasterEVI`, `GPURasterSAVI`, `GPURasterFillNoData`, `GPURasterFillHoles` on `values`; `GPURasterRankFilter`, `GPURasterFractionalZonalStatistics`, `GPURasterReprojection`, `GPURasterTileReprojection` on `input`.
- **String-coerced trajectory groups:** `GPUTrajectoryHampel`, `GPUTrajectoryOutlierFlags`, `GPUTrajectorySplit`, `GPUTrajectoryThinning`, `GPUTrajectoryTrackRelativeTimes`, `GPUTrajectoryStaypoints`, `GPUTrajectorySegments`, `GPUTrajectoryLocations`, `GPUTrajectoryRecursions`, `GPUMobilityIndicators`, `GPUModalSplit`, `GPUBrownianBridgeUtilization`, `GPUUtilizationVolumeIsopleths`. All inherit the same preparation.
- **Traced grid unit declarations:** `idw-grid` (`cellSize`, `searchRadius`), `kernel-density-grid` (`cellSize`, `bandwidth`, `padding`), `point-density-grid` and `line-density-grid` (`cellSize`), `kriging-grid` (`cellSize`, `range`, `searchRadius`). The other `'crs'` occurrences require classification by actual frame.
- **Shared list defaults:** `network-service-area`, `linear-referencing`, `pointcloud-ground`, `pointcloud-normal-scale`, `network-multi-origin-service-area`, `network-isochrone-polygons`, `network-travel-time-percentiles`, `network-k-function`, `regression-mgtwr` (two lists), `recipe-zonal-distribution`, `trajectory-recursions`, `trajectory-modal-split`, `curated-dasymetric-redistribution`, `curated-zonal-exposure`, `curated-drive-time-bands`.
- **Geometry workarounds:** `createCatalogFrame`, `collectCatalogPositions`, `collectAllPositions`, both local `collectPositions` implementations, `packCatalogParts`, `getCatalogRingArea`, `getCatalogRectangle`, `readCatalogPolygons`, `runCatalogLineGeometry`. The evidence above distinguishes incorrect results from repeated work.
- **Mixed execution:** `raster-embedding-pca` and `raster-embedding-projection`. **Column-only scoring behind feature admission:** `validation-gaussian-scores`, `validation-ensemble-scores`, `validation-fdr`. **Manual conformal decoding:** `validation-conformal`, `validation-geo-conformal`.
- **Missing invocation metadata:** all 557 inspected ontology records lack a named port and parameter schema. The complete affected identity list is `getGPUAnalysisOperatorCatalog().map(({package:p,exportName}) => [p,exportName])`. The 229 bindings and 427 pack entries are overlapping projections with host-specific content, not 656 independently removable tables.

## Quick wins

- [ ] Preserve null before native numeric conversion. Pin `[null,0,12] → [null,0,12]`.
- [ ] Reject multipart, holed or non-simple clip operands that the selected runner cannot honor.
- [ ] Use the existing centered frame for planar line chunks and the existing translated area calculation for planar rings.
- [ ] Trust an emitted zero count. Drop the polygon when its exterior is unusable.
- [ ] Read raster variable kinds from the selected operator. Pin `GPURasterFillHoles.values` to nominal.
- [ ] Remove the obsolete DBSCAN matrix cap and replace the test that requires the false rejection.
- [ ] Copy list defaults, name invalid parameters in errors and declare FTLE’s nonzero time step.
- [ ] Replace vertex argument spreads with iteration, remove the duplicate lens eligibility predicate and publish the conformal truncation count.

## Larger proposals

**Recommended descriptor.** Extend the existing qualified catalog with an optional versioned invocation contract. A missing contract should remain explicitly undisclosed. A practical first record would describe `GPULineChunk` separately in chunk and range modes.

| Descriptor field | What its consumer needs to know |
|---|---|
| `identity`, `mode`, `stages[]` | Qualified export, mode discriminant, and actual CPU/GPU stage identities. |
| `parameters[]` | Name, type, default or derivation, constraints, mode conditions, resolved unit, and whether a change requires rebuilding. |
| `inputs[]` | Public prop path, format, component count, row/key space, optionality, missing-value policy, accepted multiplicity and coordinate frame. |
| `outputs[]` | Public prop path, format, capacity expression, active-length expression, source-row mapping, optionality and measurement kind. |
| `status[]` | Associated output or stage, scalar versus per-row scope, enum versus bitset, meanings, and conditions under which a result can be published. |
| `limits[]` | Backend, mode, measured quantity and predicate. Keep analytical limits distinct from host-selected budgets. |

**Ten changes with the largest observed opportunities for removing consumer code**, ordered by reusable surface rather than speculative deleted LOC:

1. **Declare parameters beside the implementation.** Project shared ranges, defaults, units and apply policy into the 1,124 binding parameter appearances. Keep host labels and data-derived choices local.
2. **Publish output plans.** Start with OLS and line geometry. Replace repeated format, terminal-offset and byte arithmetic across the 69 Kepler files allocating session outputs.
3. **Publish named port roles, row spaces and missingness.** Bind lens capture aliases to ports. Four packs currently assign meanings at 1,604 capture sites.
4. **Project the qualified registry into discovery.** Replace the spatial-first loop and four supplements. Keep an extension map for operators owned by upstream packages.
5. **Supply common bounded-result readers.** Start with line/polygon offsets, active counts, validity and source indices. These are reusable across Kepler, lenses and examples without a new graph executor.
6. **Share frame-aware geometry preparation.** Stream vertices, choose a shared origin, retain identities and return the inverse transform. This replaces five independent traversal implementations and the lossy line pack.
7. **Expose executable, backend-specific limit predicates.** Let admission and execution call the same rule. Start with the 13 grid and 16 CPU-work records.
8. **Describe analytical column roles separately from transport.** Let native tables, feature properties and lens datasets bind the same numeric role; attach geometry only when requested.
9. **Export typed status decoders and stage execution metadata.** Conformal and mixed PCA are small first cases. Consumers can generate diagnostic columns and truthful dispatch lists.
10. **Provide scenario bindings over declared ports.** A lens supplies its dataset, field-to-port mapping and educational material. Reuse that mapping for forms and run-scoped loading, replacing dataset-wide role guesses.

I recommend **adding the invocation contract incrementally and changing bare-name discovery before alpha**. Moving all 11,592 descriptor lines into analysis would also move application choices such as menu prominence, remedial UI copy and map styling. An acceptance test for the first slice is smaller and more discriminating: use only the new contract to construct the line output views, decode zero lines, display the geographic grid unit, and bind the quantile scenario’s one nullable input.

## Not read

I read the catalog guide before tracing the API, the lens user-input and Kepler lens adapters, the raster pack merger, and the relevant discovery, geometry, trajectory, scoring, parameter and readback paths. I enumerated all registered bindings and the five pack tables at runtime. I sampled individual implementations by repeated adapter pattern and by disagreements that could be tested with a small host fixture.

I checked candidates against the synthesis, earlier area findings, direction report and decision list. For findings involving owned files, I inspected the newer `ki` copies of the grid bindings and runner, value-strip reader, conformal contributors and OLS contributor. I did not read every example scene, all 229 runner bodies, every lens formula, or every GPU kernel. No shader result, browser rendering, byte-saving estimate or full verification gate is claimed.