# One operation, several forms

## Summary

The best-supported swap is **between the two graph operation classes**. All 128 CPU graph operations have a GPU graph operation, a GPU contributor and a public array function. For 127 pairs, the graph constructors literally name the same props type. That is a strong basis for the alpha API.

The most consequential exception is graph output validation. Thirteen CPU operations let two ports use one destination name. In an executed multiple-testing example, the adjusted p-values disappeared and the graph returned only the count. The GPU form rejected the same declaration.

Two numerical examples also defeat substitution. Changing the backend of a bounded Sinkhorn calculation changes whether the target is reachable. Changing a completed CPU network calculation from a sparse cost table to a dense matrix changes whether a node is inside the same inclusive cutoff.

I ran 26 pure CPU functions on published examples. All 25 lens examples matched their recorded output on the selected column. A second column on the straight-skeleton example exposed stale edge provenance. The standalone variogram walkthrough worked as printed.

The most valuable pre-alpha change is **one entry-qualified description of each operation’s forms, ports and semantic options**, used by both graph constructors, the catalog, the family page and a parity test. The checkout acquired the first four real catalog links during this review. Extend those.

## Census

`S/` means `modules/analysis/src/`, `T/` means `modules/analysis/test/`. Paths marked **KI** refer to the newer copy in `../luma-contributors-ki/`. I resolved public exports from `modules/analysis/package.json`, followed TypeScript aliases, and overlaid `.typecheck/r19/staging-owned.txt` before inspecting declarations. The producer test was the presence of `getCommandNodes` on an exported class instance. A class merely declared in a source file did not qualify.

| Public entry | GPU contributors | GPU graph operations | CPU graph operations | `compute*OnCPU` | `add*Recipe` | Final catalog records |
|---|---:|---:|---:|---:|---:|---:|
| `gpu-spatial-analysis` | 272 | 80 | 80 | 83 | 39 | 317 |
| `gpu-network` | 41 | 17 | 17 | 17 | 5 | 46 |
| `gpu-terrain` | 46 | 11 | 11 | 11 | 0 | 46 |
| `dataframe-analysis` | 34 | 7 | 7 | 7 | 0 | 40 |
| `raster-analysis` | 86 | 13 | 13 | 13 | 0 | 86 |
| `pointcloud-analysis` | 11 | 0 | 0 | 0 | 0 | 11 |
| `geospatial-analysis` | 2 | 0 | 0 | 0 | 0 | 2 |
| `analysis-core` | 4 | 0 | 0 | 0 | 0 | 5 |
| `computation-record` | 2 | 0 | 0 | 0 | 0 | 4 |
| **Total** | **498** | **128** | **128** | **131** | **44** | **557** |

The catalog grew from 555 to 557 while I was reading. The final two records are `addBuildingTessellationReferenceRecipe` and `addLocalBuildingTessellationRecipe`. All 44 entry-qualified recipe exports are now represented, including the two different `addVisibilityGraphRecipe` exports. `addRasterRegionGraphTiles` is a separate tiled builder.

| Comparison | COUNT |
|---|---:|
| CPU and GPU graph operations use the same named props type | 127/128 |
| GPU graph operation constructs the matching GPU contributor | 128/128 |
| GPU graph operation adds top-level props beyond the direct contributor | 102/128 |
| Array result and GPU graph `output` have the same declared leaf paths | 65/128 |
| Leaf paths agree after removing `status.` or `ports.status.` prefixes and deduplicating aliases | 36 more |
| Other result-shape differences | 27 |
| Pure function has one / two / three / four / five declared arguments, including optional arguments | 13 / 92 / 16 / 5 / 2 |
| First pure argument is a typed array / a named input object | 27 / 101 |
| Catalog-selected documentation page names the pure function / CPU operation / both / neither | 17 / 23 / 15 / 103 |
| Exact pure function name occurs anywhere under `docs/api` | 25/128 |
| Exact CPU operation name occurs anywhere under `docs/api` | 128/128 |
| Final catalog links to both graph forms / to a pure function | 4 / 0 |

The output-path census compares names, not values. It includes optional members and union branches. Some GPU results live in an auxiliary props branch, and some numbers under `output` size a buffer. A path difference identifies what a caller must translate. It does not by itself establish a missing computation.

### Operation matrix

Each stem in the following table has all four forms: `GPU<stem>`, `GPU<stem>Operation`, `CPU<stem>Operation`, `compute<stem>OnCPU`. The one spelling exception is `GPULocationAllocationGraphOperation`, as permitted by `dev-docs/contributor-contract.md:177`. Columns give the pure-result versus graph-output comparison.

| Entry | Identical declared paths | Only status placement or alias differs | Other differences, detailed below |
|---|---|---|---|
| Spatial | PettittTest, PointToCell, CellMeasures, S2CapCovering, S2PolygonCovering, HealpixPointToCell, LabelPoint, LocationAllocation, MultipleTestingAdjustment, PolygonTriangulation, EnsembleScores, ShapeDescriptors, MantelTest, ConditionalGaussianSimulation, VecchiaFisherScoring, VecchiaLogLikelihood, VecchiaSpaceTimeLogLikelihood, WardRegions, SkaterRegions, AZPRegions, MaxPRegions, RestrictedRegimeRegression, SpatialLagImpacts, TrackSimilarity, BrownianBridgeUtilization, ModalSplit, MaximumInscribedCircle, NormalizeGeometry, NormalizePolygons, ConvexStraightSkeleton, LineLoopRemoval, OffsetCurveComponents, VectorStraightSkeleton, PopulationSupportAlignment, ContinuousSpatialSample, SpatialBalancedSample, PointPatternEnvelope, TrajectoryKalman, TrajectoryRestriction, NestedInterpolationValidation, SpatialInteraction, SphericalArcOverlay | LocationChangePoints, TransitionDiagnostics, RankMarkov, GeodesicPairs, MapColoring, MinimumBounds, NeighborhoodSummary, ConstrainedDelaunayTriangulation, SinkhornDivergence, SpaceTimeHawkes, OPTICS, SpatialWeightsTraces, ContiguityWeights, SpatialWeightsValidation, SpatialWeightsAlgebra, MobilityIndicators, TrajectoryRecursions, TrajectoryStaypoints, **SegmentSnap**, RatioConcaveHull | DelaunayTessellation, LocalFrameProjection, LineSimplification, OrdinaryLeastSquares, SeeminglyUnrelatedRegression, SpatialRegressionDiagnostics, MultiscaleGeographicallyWeightedRegression, SpatialErrorGM, ProbitRegression, SpatialTwoStageLeastSquares, SpatialErrorMaximumLikelihood, SpatialLagMaximumLikelihood, SpatialPanelLagMaximumLikelihood, TrajectorySplit, TrajectoryThinning, TrajectoryLocations, TrajectorySegments, SpatialCrossValidation |
| Network | NetworkTimeRouting, NetworkPointRouting, NetworkFlow, NetworkRationalAccess, AlternativeRoutes, RadiationFlowModel, NetworkTravelTimePercentiles | NetworkCostMatrix, NetworkServiceAreas, NetworkCentrality, NetworkCostTable, NetworkReachability, KShortestPaths | NetworkLouvain, MapMatching, NetworkPathExtraction, NetworkStructure |
| Terrain | TerrainBranchingFlow, TerrainBreach, TerrainFlowDistance, TerrainFlow, TerrainPeakProminence, SolarPosition | TerrainDepressionHierarchy, TerrainStreamOrder, TerrainWatersheds | TerrainCatchmentSummary, ProfilePeaks |
| Dataframe | ClassBreaks, ColumnQuantiles, Inequality, VariogramModelFit, RipleyEnvelope, SpaceTimeDiggleTest | CoregionalizationFit | — |
| Raster | RasterFractionalZonalDistribution, RasterFractionalZonalStatistics, RasterGeometricMedianComposite, RasterMedoidComposite | RasterWalkingCost, CostDistance, RasterRegionGraph, RasterWatershedSegmentation, RasterReconstruction, RasterFillHoles | CostDistancePath, Streamlines, Isolines |

The three other public array functions matter for discovery:

| Contributor | Array function | Named graph pair | Difference |
|---|---|---|---|
| `GPUSpatialPredicateJoin` | `computeSpatialPredicateJoinOnCPU` | Absent | Direct array reference. `S/gpu-spatial-analysis/spatial-join/cpu-feature-join.ts:351` |
| `GPUNearestFeatureJoin` | `computeNearestFeatureJoinOnCPU` | Absent | Direct array reference. Same file, line 600 |
| `GPUContiguityWeights` | `computePolygonContiguityOnCPU` | Already has the `ContiguityWeights` pair and pure function | Additional prepared-topology route. `S/gpu-spatial-analysis/spatial-weights/prepare-polygon-contiguity-topology.ts:70,319` |

That leaves **368 contributors** without a corresponding public function under these conventions, and **370 without named graph forms**. The following read-only census expands the matrix to one row per contributor, including those 368. It also prints the actual prop names, function arguments, output fields, source location and each pure function’s documented differences in defaults, limits, precision and stopping behavior. This preserves distinctions such as “the CPU drops a GPU storage ceiling” versus “the CPU ignores an explicitly requested algorithm parameter.”

```js
// Run with node on stdin from the checkout. No compilation, emit or diagnostics.
const fs = require('node:fs');
const path = require('node:path');
const ts = require('typescript');
const root = 'modules/analysis/src/';
const owned = new Set(fs.readFileSync('.typecheck/r19/staging-owned.txt', 'utf8').trim().split(/\r?\n/));
const entries = Object.keys(require('./modules/analysis/package.json').exports)
  .filter(e => e !== '.' && !e.endsWith('/ontology')).map(e => e.slice(2));
const options = {target: ts.ScriptTarget.ESNext, module: ts.ModuleKind.ESNext,
  moduleResolution: ts.ModuleResolutionKind.Bundler, strict: true, skipLibCheck: true};
const host = ts.createCompilerHost(options);
const read = host.readFile;
host.readFile = f => {
  const p = path.relative(process.cwd(), path.resolve(f));
  return read(owned.has(p) ? '../luma-contributors-ki/' + p : f);
};
const program = ts.createProgram(entries.map(e => root + e + '/index.ts'), options, host);
const checker = program.getTypeChecker();
const follow = s => s?.flags & ts.SymbolFlags.Alias ? checker.getAliasedSymbol(s) : s;
const type = s => checker.getTypeOfSymbolAtLocation(s, s.valueDeclaration || s.declarations[0]);
const fields = t => checker.getPropertiesOfType(checker.getNonNullableType(t))
  .map(s => s.name + (s.flags & ts.SymbolFlags.Optional ? '?' : ''));
function describe(s) {
  if (!s?.valueDeclaration) return;
  const d = s.valueDeclaration, t = type(s);
  const sig = (ts.isClassDeclaration(d) ? t.getConstructSignatures() : t.getCallSignatures())[0];
  if (!sig) return;
  const first = sig.parameters[0], output = first && type(first).getProperty('output');
  return {name: s.name, file: d.getSourceFile().fileName,
    line: d.getSourceFile().getLineAndCharacterOfPosition(d.getStart()).line + 1,
    arguments: sig.parameters.map(p => [p.name, checker.typeToString(type(p)), fields(type(p))]),
    output: fields(output ? type(output) : sig.getReturnType()),
    notes: ts.displayPartsToString(s.getDocumentationComment(checker))};
}
const rows = [], bySymbol = new Map(), recipes = [];
for (const entry of entries) {
  const file = program.getSourceFile(root + entry + '/index.ts');
  const xs = new Map(checker.getExportsOfModule(checker.getSymbolAtLocation(file))
    .map(s => [s.name, follow(s)]));
  for (const [name, s] of xs) {
    if (/^add.*Recipe$|^addRasterRegionGraphTiles$/.test(name)) recipes.push([entry, name, s]);
    if (!s.valueDeclaration || !ts.isClassDeclaration(s.valueDeclaration) ||
        !name.startsWith('GPU') || !checker.getDeclaredTypeOfSymbol(s).getProperty('getCommandNodes')) continue;
    const row = {entry, stem: name.slice(3), xs, s, recipes: new Set()};
    rows.push(row);
    bySymbol.set(s, row);
  }
}
for (const [entry, name, recipe] of recipes) {
  const seen = new Set();
  function visit(s) {
    if (!s || seen.has(s)) return;
    seen.add(s);
    for (const d of s.declarations || []) {
      if (!d.getSourceFile().fileName.includes(root) ||
          !(ts.isFunctionDeclaration(d) || ts.isMethodDeclaration(d) || ts.isVariableDeclaration(d))) continue;
      function walk(n) {
        if (ts.isNewExpression(n))
          bySymbol.get(follow(checker.getSymbolAtLocation(n.expression)))?.recipes.add(entry + '/' + name);
        if (ts.isCallExpression(n)) {
          visit(follow(checker.getSymbolAtLocation(n.expression)));
          for (const a of n.arguments) if (ts.isIdentifier(a)) visit(follow(checker.getSymbolAtLocation(a)));
        }
        ts.forEachChild(n, walk);
      }
      if (d.body || d.initializer) walk(d.body || d.initializer);
    }
  }
  visit(recipe);
}
for (const {entry, stem, xs, s, recipes} of rows) {
  console.log(JSON.stringify({entry, contributor: describe(s),
    gpuGraph: describe(xs.get(`GPU${stem}GraphOperation`) || xs.get(`GPU${stem}Operation`)),
    cpuGraph: describe(xs.get(`CPU${stem}Operation`)), arrays: describe(xs.get(`compute${stem}OnCPU`)),
    preparedArrays: stem === 'ContiguityWeights' ? describe(xs.get('computePolygonContiguityOnCPU')) : undefined,
    reachableFromRecipes: [...recipes]}));
}
```

Recipe membership here means a constructor is reachable through a local function, including conditional stages. A recipe can also declare WGSL directly. In particular, the common preparation helper can reach `GPUSpatialJoinPrepared` while checking caller resources. Its presence in a row does not establish that every invocation executes a spatial join.

### Published examples executed

I ran the pure functions through the permitted `esbuild --bundle --platform=node --format=esm --loader=ts` → `node --input-type=module -` pipeline. For the lens pages, I expanded the checked-in record with `expandLensRecordEnvelope`, restored the recorded typed arrays, and used the options in the scenario that produced the example. This extra step was necessary because most pages show graph inputs or captured output without an array-function invocation.

The input and result columns below are excerpts. I compared the **entire selected column**, including length, using exact integers and a float tolerance of `2e-5 * max(1, abs(expected))`, with matching NaNs. The lengths range from one to 791 elements. These are comparisons with published captures, not fresh GPU runs.

| CPU function stem | Published input or setting | Selected result obtained |
|---|---|---|
| TrajectoryKalman | positions `[0,0,2,0]`, smoothing on, `times` renamed to `timestamps` | `filteredState=[0,0,0,0,1.3333334,0,.6666667,0]` |
| TrajectoryRestriction | positions `[-2,0,2,0]`, captured `times,offsets` | `relationships=[1,0,2,0,4294967295,4294967295,0,0]` |
| NetworkTimeRouting | state vertices `[0,1,2]`, recorded `edgeCosts` passed as `costs` | `arrivals=[3,4,9]` |
| NetworkFlow | four nodes, first edges `[0,1,0,2]` | `flow=[3,2,1,0,2,3]` |
| NetworkLouvain | four nodes, four levels, four iterations | `communities=[1,1,3,3]` |
| NetworkRationalAccess | costs `[0,0,∞,∞]`, four cycles | `assignments=[5,5,0,0]` |
| NetworkPointRouting | edge costs `[10,20,30]` | `costs=[5,40,∞]` |
| NetworkStructure | offsets start `[0,2,3,4]` | `strongComponents=[0,0,0,0]` |
| NormalizePolygons | positions start `[4,2,0,2]` | `positions=[0,0,0,2,4,2,4,0]` |
| NormalizeGeometry | positions start `[4,0,0,0]` | `children=[0,1]` |
| SegmentSnap | tolerance `.2`, maximum six vertices per line | `counts=[4,4,0,0]` |
| LineLoopRemoval **KI** | crossing line starts `[0,0,4,4]`, capacities 16 | `lineOffsets=[0,3]` |
| MaximumInscribedCircle **KI** | polygon, positional tolerance `.01`, capacity and iterations 128 | `circles=[2,1,1,.012944174]` |
| ConvexStraightSkeleton **KI** | four vertices, offset `.5`, capacity eight | `status=[5,5,4,0]` |
| VectorStraightSkeleton **KI** | eight vertices, offset `.25`, capacity 32 | `status=[12,12,8,0]`, see FRM-07 for provenance |
| OffsetCurveComponents **KI** | original and raw line sets, distance `.5`, tolerance `.001` | `status=[1,3,1,3,0,0,0,0]` |
| RatioConcaveHull **KI** | ratio `.1`, holes off, eight iterations | `converged=[1]` |
| TerrainBreach | elevations start `[2,5,0,NaN]`, depth limit four | `cutDepth=[0,3.5,0,NaN,NaN,NaN,NaN,NaN,NaN]` |
| TerrainBranchingFlow | recorded `injection` passed as `values`, transport mode | `values=[8,2,0,6,8,0,0,0,0]` |
| TerrainDepressionHierarchy | elevations start `[0,2,1,NaN]`, capacity three | `status=[3,3,0]` |
| TerrainCatchmentSummary | 3×3 grid, two basins, two bins | 12 `statistics` values, starting `[3,2,1,3,2,.6666667,6,6.5,4]` |
| SpatialCrossValidation **KI** | three folds, seed 17, buffer `.25` | `counts=[16,8,0,16,8,0,16,8,0]` |
| RankMarkov **KI** | three periods, geographic rank | `mobility=[1.75,1.5]` |
| PopulationSupportAlignment | target IDs start `[10,30,50,20]` | 12 `populations`, starting `[6,4,NaN,NaN]` |
| MultipleTestingAdjustment | captured p-values, Benjamini–Hochberg, minimum family zero | 791 `adjustedPValues`, starting `[NaN,.66339934,.87350893,.99569887]` |
| VariogramModelFit | literal 20-bin exponential example in `docs/api/operation-forms-and-cpu-twins.md`, **KI** page | fitted row `[.20000008,1.5,2.2000003,2.76e-13]`, model status codes `[0,0,0]` |

## Proposed convention

A contributor author should give the direct contributor, both graph classes and the pure function one qualified operation identity, one vocabulary for semantic inputs and results, and one meaning for an explicitly supplied algorithm option. Device limits, CPU reference algorithms, prepared resources and recipe-specific names may differ, but the identity should record those differences and the family page should display them.

A checker can resolve every registered export in both directions, compare graph port declarations, and run the same named-input fixture through the pure function and both graph forms with every requested result and status qualified by its documented tolerance.

## Findings

| ID | Kind | Reach | Change | Cost | Where | One line |
|---|---|---|---|---|---|---|
| FRM-01 | W | ORDINARY | ADDITIVE | S | CPU graph output declarations | Duplicate destination names silently remove an output in 13 operations. |
| FRM-02 | W | ORDINARY | BREAKING | M | SinkhornDivergence | An explicit blur radius changes the transport problem only on GPU. |
| FRM-03 | W | ORDINARY | ADDITIVE | M | NetworkCostTable, NetworkCostMatrix, NetworkReachability | Completed sparse and dense CPU searches disagree at the same inclusive cutoff. |
| FRM-04 | W | ORDINARY | ADDITIVE | S | MultipleTestingAdjustment | Mutating the original props changes the method executed by an already constructed CPU stage. |
| FRM-05 | W | ORDINARY | ADDITIVE | S | TerrainCatchmentSummary | Swapping CPU for GPU drops three requested graph outputs. |
| FRM-06 | C | ORDINARY | BREAKING | S | DrainageNetwork recipe | A geographic flow stage feeds distance and reach stages using planar units. |
| FRM-07 | D | ORDINARY | ADDITIVE | S | VectorStraightSkeleton lens | The published example still attributes four arcs to different edges than the staged CPU reference. |
| FRM-08 | S | ORDINARY | ADDITIVE | S | TrajectoryRestriction, TrajectoryKalman | Graph-compatible trajectory keys fail in the restriction array function. |
| FRM-09 | S | ORDINARY | ADDITIVE | S | TerrainCatchmentSummary | The pure function treats unrelated properties as mandatory raster inputs. |
| FRM-10 | C | ORDINARY | ADDITIVE | S | ClassBreaks | A graph configuration can produce three CPU classes yet fail GPU admission. |
| FRM-11 | E | ORDINARY | ADDITIVE | S | NetworkLouvain | Five named diagnostics require hand-decoding or recreating values outside a graph. |
| FRM-12 | C | ORDINARY | ADDITIVE | S | PointToCell, HealpixPointToCell, ClassBreaks | An empty CPU batch changes from a defined result to an exception when put in a graph. |
| FRM-13 | E | ORDINARY | ADDITIVE | M | Catalog and family pages | Only four of 128 graph pairs are linked, and no pure function is linked. |
| FRM-14 | T | ORDINARY | ADDITIVE | S | Public export test | Three complete CPU export losses can pass the current inventory check. |
| FRM-15 | E | ORDINARY | ADDITIVE | S | Recipe argument maps | Sixteen recipes directly rename stage arguments without a common way to recover the mapping. |

### FRM-01: A count can overwrite the analysis result

**Where.** `S/analysis-core/operations/gpu-contributor-operation.ts:126–155` validates the GPU contract. The first CPU counterexample is `S/gpu-spatial-analysis/multiple-testing/cpu-multiple-testing-adjustment-operation.ts:51,66,83`. All affected CPU constructors are listed under Departures.

**EXECUTED.** Of 128 CPU constructors, 114 build their declaration using the corresponding GPU wrapper. I exercised all 14 others. Thirteen multi-output CPU constructors accepted duplicate destinations. Their GPU counterparts all rejected them. The remaining single-output CPU constructor accepted an empty destination that the GPU rejected.

```ts
output: {adjustedPValues: 'count', testedCount: 'count'}
```

With `pValues=[.02,.20]` and `method='bonferroni'`, the CPU declaration became a single `uint32[1]` port. Execution returned `{count:[2]}`. The expected adjusted values `[.04,.40]` vanished before graph planning could detect a conflict.

**Wanted.** Validate the original logical-port-to-destination map for nonempty names, duplicate outputs and input/output collisions **before** constructing a JavaScript object keyed by destination. Share this validation without importing GPU-only size ceilings into the CPU backend. **Callers:** no valid declarations change. **Value and cost:** prevents silent column loss across three entries, S.

### FRM-02: Bounded Sinkhorn becomes full transport on CPU

**Where.** `S/gpu-spatial-analysis/optimal-transport/cpu-sinkhorn-divergence-operation.ts:51,113,288`; `docs/api/gpu-spatial-analysis/sinkhorn-divergence.md:61–62`; `T/gpu-spatial-analysis/optimal-transport/sinkhorn-divergence-operation.spec.ts:116`. This is separate from `KNOWN(PTS-01)`, which concerned stopping before all three plans had settled.

**EXECUTED.** I used the page’s own counterexample: source `[1,0,0]`, target `[0,0,1]`, unit 3×1 grid, `blurRadius:1`, `maximumIterations:10`, tolerance `.0001`. CPU returned divergence `4`, cross marginal error `0`, `converged=1`, `iterationLimitReached=0`, `iterationCount=2`, `approximation=0`. The page predicts cross error `1` and exhaustion. The CPU validates the radius and then performs a full blur. The existing comparison test deliberately excludes CPU for this case.

**Wanted.** Honor an explicit radius in both `computeSinkhornDivergenceOnCPU` and the CPU graph operation. Keep full support when the radius is omitted. **SOURCE GPU probe:** use those three cells, request `marginalError` and all convergence ports, and enable CPU in the existing `unreachable-target` comparison.

**Callers.** Two explicit `blurRadius:0` CPU fixtures currently depend on full support, at `cpu-sinkhorn-divergence-operation.node.spec.ts:198,241`. One graph-test selection changes at line 116 above. The direct call search found four TypeScript files, all in the implementation or tests. **Value and cost:** makes a backend change preserve the user’s transport domain, M.

### FRM-03: A node is reachable in the sparse CPU table and unreachable in the dense CPU matrix

**Where.** `S/gpu-network/network-cost-table/cpu-network-cost-table-operation.ts:54,146`; **KI** `S/gpu-network/network-accessibility/cpu-network-cost-matrix-operation.ts:49` and `S/gpu-network/network-reachability/cpu-network-reachability-operation.ts:54`; **KI** `docs/api/operation-forms-and-cpu-twins.md:36`. `KNOWN(NET-124)` concerned passing a producer’s search horizon downstream. Here both searches use the same horizon and finish.

**EXECUTED.** CSR offsets `[0,1,2,2]`, neighbors `[1,2]`, float32 weights `[.1,.4]`, start node `0`, float32 cutoff `.5`:

| Pure CPU operation | Result |
|---|---|
| NetworkCostTable | Row 0 targets `[0,1,2]`, costs `[0,.10000000149,.5]`, `complete=1` |
| NetworkCostMatrix | Row 0 `[0,.10000000149,Infinity]`, `converged=1` |
| NetworkReachability | Node 2 unreachable with the same cutoff |

The sparse implementation rounds each edge addition to float32 before comparing. The dense and reachability implementations compare the binary64 sum of the float32 inputs. That sum exceeds `.5`. The library-wide statement that a CPU twin rounds only at output is incorrect for the table.

**Wanted.** Document the arithmetic at the inclusive cutoff on all three pages. Provide a common explicit `costPrecision: 'float32-per-edge' | 'float64'` for CPU routing consumers that need to reproduce table membership. Include this three-node fixture in the shared examples. **Callers:** none for the additive option. **Value and cost:** a GIS developer can switch between sparse and dense outputs without losing a boundary destination, M.

### FRM-04: A constructed CPU stage can change statistical methods

**Where.** `S/gpu-spatial-analysis/multiple-testing/cpu-multiple-testing-adjustment-operation.ts:52–85`; **KI** `docs/api/operation-forms-and-cpu-twins.md:30–34`. `KNOWN(CORE-08)` fixed six other producers. The additional evidence here is a changed statistical answer from the same CPU graph callback.

**EXECUTED.** Construct with a caller-owned props object, p-values `[.02,.20]` and `method:'bonferroni'`. Execute once, assign `props.method='none'`, and execute the same operation on the same input. Results were `[.04,.40]` and `[.02,.20]`. The constructor validated the first method. The callback reads the original object again at line 79. It also rereads output destinations at line 83.

**Wanted.** Capture `rowCount`, `method`, `minimumFamilySize` and resolved destination names into constructor locals used by both declaration and execution. **Callers:** none following the documented construction contract. **Value and cost:** a reused graph retains its declared statistical procedure, S.

### FRM-05: Substituting the GPU catchment-summary operation discards diagnostics

**Where.** `S/gpu-terrain/terrain-processing/gpu-terrain-catchment-summary-operation.ts:21,37,59`; `cpu-terrain-catchment-summary-operation.ts:58,306`; `docs/api/gpu-terrain/terrain-processing.md:69`. The page already acknowledges the pending GPU mapping. The new evidence is that this is the **only** divergent named constructor type, and that switching an existing props variable silently drops destinations.

**EXECUTED.** I used a single props variable with four ordinary outputs and `output.status={converged:'ok',invalidCount:'bad',iterationLimitReached:'lim'}`. CPU contract keys were `[stats,h,pr,ps,ok,bad,lim]`. GPU contract keys were `[stats,h,pr,ps]`. An inline GPU object literal can get an excess-property diagnostic. A structurally compatible variable passes through.

**Wanted.** Add the three optional GPU names, declare their one-row `uint32` vectors and forward their resolved views to the contributor. Use `TerrainCatchmentSummaryOperationProps` on both constructors. **Callers:** none. **Value and cost:** preserves a caller’s completeness checks when changing one class name, S.

### FRM-06: The drainage recipe accepts two units systems in one result

**Where.** `S/gpu-spatial-analysis/recipes/drainage-network-recipe.ts:22,58,95,131`; `S/gpu-terrain/hydrology/gpu-terrain-flow-distance.ts:65`; `docs/api/gpu-spatial-analysis/expansion-recipes.md:168–182`; `examples/deck/spatial-analysis-showcase/src/scenes/hydrology/terrain-boundaries.compute.ts:336`. The page describes a planar recipe. The props type also admits a geographic flow stage, and the showcase forwards a grid-dependent mode.

**EXECUTED host construction.** On a 2×1 equatorial grid with affine `[1,0,0,0,-1,.5]`, I passed `flow.cellSizeMode='geographic'` and inspected the constructed flow, distance and stream-network children. Their settings were respectively `'geographic'`, `undefined`, `undefined`. The flow-distance CPU reference for the same one-cell eastward step returned `[1,0]` under the default and `[111319.4921875,0]` under geographic mode.

**Wanted.** Pass the flow stage’s `cellSizeMode` to both length-producing stages and document the result units at the recipe output. **SOURCE GPU probe:** route a descending two-cell equatorial DEM, read recipe distance and reach length, and compare them to the two standalone contributors with geographic mode.

**Callers.** `rg 'addDrainageNetworkRecipe\(' modules examples website` found ten TypeScript occurrences outside the declaration, including two website snippets. No signature edits are needed. Nonuniform callers get different numeric values, including the grid-dependent showcase path. **Value and cost:** keeps lengths and catchment summaries interpretable in one coordinate system, S.

### FRM-07: The straight-skeleton example’s green status conceals stale provenance

**Where.** `docs/api/gpu-spatial-analysis/geometry-refinement-lenses.md:140–142`; `docs/lens-records/gpu-spatial-analysis.vector-straight-skeleton.json:1`; **KI** `S/gpu-spatial-analysis/geometry-refinement/cpu-vector-straight-skeleton-operation.ts:284,320`. `KNOWN(D089)` is the underlying edge-provenance repair. The remaining issue is the public result capture.

**EXECUTED.** I replayed the page’s 10×10 square with its 3–7 square hole, offset `.25`, event budget 16 and capacity 32. The published capture and staged CPU reference agree on the 12 geometry rows and `status=[12,12,8,0]`. For the arc from `(8.5,1.5)` to `(1.5,1.5)`, the capture says source edges `[6,7]`, and the staged CPU returns `[7,0]`. The other three final pairs change from `[0,1],[1,2],[2,3]` to `[1,6],[2,5],[3,4]`.

**Wanted.** Recapture the published lens after the staged fix and compare geometry **and source edges** on its 12 live arcs. **SOURCE GPU probe:** run this existing example and read `sourceEdges` for rows 8–11. I did not establish the current WGSL result. **Callers:** none. **Value and cost:** the provenance shown to users describes the geometry the example actually computes, S.

### FRM-08: Adjacent trajectory twins disagree on the names of the same two arrays

**Where.** `S/gpu-spatial-analysis/trajectory-estimation/gpu-trajectory-kalman-operation.ts:25,41`; `trajectory-restriction/cpu-trajectory-restriction-operation.ts:28,179,184`; `docs/api/gpu-spatial-analysis/movement-coverage.md:51–73`.

**EXECUTED first use.** The page’s graph example uses `timestamps` and `trackOffsets`. Both graph forms accept that pair. The Kalman array function accepts those names. Applying the same vocabulary to the neighboring restriction example gives:

```ts
computeTrajectoryRestrictionOnCPU({positions, timestamps, trackOffsets, parameters}, {});
```

The declared input requires `times,offsets`. JavaScript throws `Cannot read properties of undefined (reading 'length')`. Changing only those keys to `times:timestamps, offsets:trackOffsets` yields the eight relationship words in the example table. Going from the record to Kalman requires the reverse translation.

**Wanted.** Accept the same exclusive pair of aliases in the pure input as in the graph form. Normalize once to `timestamps,trackOffsets` and issue an operation-specific error on an incomplete or mixed pair. **Callers:** none. **Value and cost:** one packed trajectory can feed either pure function, S.

### FRM-09: Extra properties make a valid catchment input fail

**Where.** `S/gpu-terrain/terrain-processing/cpu-terrain-catchment-summary-operation.ts:90–93`.

**EXECUTED.** I passed the expanded published catchment record, which contains the correctly shaped `elevation`, `streams`, `basinIds` and `flowDirections`, plus the captured outputs. The pure function threw `statistics must contain one value per grid cell`. Passing the same four input arrays explicitly produced the 12 expected statistics. `Object.entries(input)` validates every enumerable property as a cell-sized raster. TypeScript structurally accepts the wider object in a variable.

**Wanted.** Check only the four declared input arrays and optional `validity`. Extra record columns should have no effect. **Callers:** none. **Value and cost:** users can pass a wider raster record without copying or destructuring it to remove unrelated columns, S.

### FRM-10: CPU success can certify an impossible class-breaks configuration

**Where.** `S/dataframe-analysis/column-classification/gpu-class-breaks.ts:287–302`, `gpu-class-breaks-operation.ts:35–38`, `cpu-class-breaks-operation.ts:127,312`. `KNOWN(DPC-120)` covered the raw contributor’s number-of-bins constraint. The remaining gap is admission across graph forms.

**EXECUTED.** Values `[0,1,2,3,4,5]`, approximate natural breaks, requested and maximum class count `3`, bin count `2`. The CPU operation produced breaks `[0,2,4,5]`, class count `3`. The GPU graph wrapper constructed. Resolving its views and constructing its contributor then threw `class-breaks maximumClassCount 3 must not exceed naturalBreaksBinCount 2 in approximate natural-breaks mode`.

**Wanted.** Check this relationship in common graph preflight when approximation is explicitly selected. Document that the pure reference computes exact Fisher–Jenks breaks, including the consequence for comparison with binned GPU results. **Callers:** none with a configuration runnable on both graph backends. **Value and cost:** a CPU dry run can catch an unusable GPU configuration before graph execution, S.

### FRM-11: Louvain exposes five diagnostics only after writing a graph adapter

**Where.** `S/gpu-network/network-optimization/cpu-network-louvain-operation.ts:29,61,253–268`.

**EXECUTED.** The documented four-node example returned keys `communities,objectiveTrace,summary`, with `summary=[1,1,2,3]`. The graph adapter turns the same result into named `iterationCount=3`, `converged=1`, `iterationLimitReached=0`, `invalidCount=0`, `approximation=1`. Two flags are Boolean expressions and one is a constant. They cannot all be recovered by looking up an identically named field or a single summary offset.

**Wanted.** Add `result.status` with those five one-row arrays, retain the packed summary, and use the same decoder in the CPU graph adapter. **Callers:** none. **Value and cost:** array users can check convergence and qualify heuristic results with the same code as graph users, S.

### FRM-12: Moving an empty batch into a graph changes its meaning

**Where.** `S/gpu-spatial-analysis/cell-indexing/cpu-point-to-cell-operation.ts:62,119`; `healpix-cells/cpu-healpix-point-to-cell-operation.ts:43,105`; `S/dataframe-analysis/column-classification/cpu-class-breaks-operation.ts:127,312`. The graph validation is in `S/analysis-core/operations/gpu-contributor-operation.ts`.

**EXECUTED.** Empty XY arrays gave `cells=[]`, `validity=[]` in both the quadbin and HEALPix order-one pure functions. With `rowCount:0`, both CPU and GPU graph declarations rejected `outputs.cells.length` as outside `[1,4294967295]`. Empty exact class breaks gave `breaks=[NaN,NaN,NaN]`, `classCount=[0]`, `classCounts=[0,0]` in the pure function. The CPU graph path rejected `values.length`.

**Wanted.** Declare empty-input support per form. Permit zero-length point outputs and short-circuit these stages where the executor supports empty vectors. Publish the exact empty class-breaks result or the required caller guard. `NetworkFlow` already documents zero edges and empty edge-flow outputs on all three forms. **Callers:** none. **Value and cost:** empty selections become a predictable GIS operation, S for these three cases and their documentation.

### FRM-13: The catalog has a working linkage model, populated for four pairs

**Where.** `S/gpu-spatial-analysis/ontology/spatial-analysis-ontology.ts:418`; `ontology/all-entry-points/spatial-operation-forms.ts:7`; `analysis-operator-catalog.ts:121,162`. This updates `KNOWN(CPU-05 / INT-03)` and adds an execution-form use case for `KNOWN(ERG-106)`.

**EXECUTED and COUNT.** Final `getGPUAnalysisOperatorCatalog()` returned 557 rows. Four had `forms.cpu` and `forms.gpu`: ConstrainedDelaunayTriangulation, DelaunayTessellation, MinimumBounds and PolygonTriangulation. None linked the pure function. Of the 128 evidence pages selected by contributor identity, 103 named neither the pure function nor the CPU graph class, even though every CPU graph class is mentioned somewhere under `docs/api`. Looking up the contributor and searching the entire documentation tree give materially different answers.

**Wanted.** Extend the existing `forms` record with qualified `arrays`, `variants` and recipe-stage relationships. Populate the other 124 graph pairs, all 131 array functions, and each form’s supported inputs, changed defaults and ignored or unsupported options. Render the table at the contributor’s evidence page. Preserve the qualified export identity for both visibility recipes. **Callers:** none. **Value and cost:** a user can discover which swap is available before writing the graph or allocating a buffer, M.

### FRM-14: The export test can pass after losing three twins

**Where.** `T/analysis-core/mixed-execution/operation-exports.node.spec.ts:25,44,46,60,70,75`.

**COUNT and EXECUTED inventory simulation.** The test starts from exports matching `^CPU(\w+)Operation$`, checks siblings only for those names, compares pure names case-insensitively, and requires at least 125. I removed `CPUKShortestPathsOperation`, `CPUCoregionalizationFitOperation` and `CPUPopulationSupportAlignmentOperation` from an in-memory copy of the public-name inventory. The count was `125`; all visited sibling-name checks still succeeded. No files were altered and no Vitest run was needed to demonstrate the blind spot.

**Wanted.** Assert exact, case-sensitive equality of the qualified expected and observed form sets in both directions, with explicit rows for partial and prepared-only forms. **Callers:** none. **Value and cost:** losing the first export in a family becomes as detectable as losing its GPU sibling, S.

### FRM-15: Recipe parameter names need a visible mapping to their stages

**Where.** `S/gpu-spatial-analysis/recipes/zonal-distribution-recipe.ts:169`, `cluster-and-outline-recipe.ts:322`, **KI** `change-of-support-recipe.ts`, **KI** `raster-summary-recipes.ts`; `landscape-ecology/landscape-patch-metrics.ts:63`; **KI** `S/gpu-network/recipes/drive-time-catchment-recipe.ts`. Full named results appear below.

**COUNT.** In the 44 public recipe functions, a TypeScript AST walk found **24 distinct direct argument renames in 16 recipes**. The test was an object property in a contributor construction whose value was `props.<path>` and whose final property name differed. It excludes local variables, computed expressions and spreads, so this is a lower bound. For example, `probabilities` feeds `GPUGroupStatistics.percentiles`, while `zoneCount` feeds `GPURasterZonalStatistics.zoneCapacity`. The latter distinction matters when reasoning about cardinality and allocation.

**Wanted.** Keep domain names such as `facilities` and `demand`, and publish a `recipe path → stage prop` table including units and count-versus-capacity meaning. Use identical names when both levels refer to the same setting and disambiguation adds nothing. **Callers:** none for the mapping. **Value and cost:** developers can carry a working standalone stage into a recipe and locate its controls, S.

## Departures by name

### Graph destination validation

The following are all 14 exceptions found by tracing the 128 CPU constructors. Names below omit the surrounding `CPU…Operation`. The number is the class declaration line in its `cpu-<name>-operation.ts`.

| Entry and directory under `S/` | Constructors and lines |
|---|---|
| `gpu-spatial-analysis/multiple-testing`, `spatial-regression` | MultipleTestingAdjustment 51; SpatialLagImpacts 116 **KI** |
| `dataframe-analysis/composite-indicators`, `pair-statistics` | Inequality 254; CoregionalizationFit 286 |
| `raster-analysis/cost-distance` | CostDistancePath 103 **KI**; RasterWalkingCost 233 **KI** |
| `raster-analysis/raster-zonal-statistics` | RasterFractionalZonalDistribution 359; RasterFractionalZonalStatistics 208 |
| `raster-analysis/remote-sensing` | RasterRegionGraph 262 **KI** |
| `raster-analysis/raster-time-series` | RasterGeometricMedianComposite 106; RasterMedoidComposite 80 |
| `raster-analysis/raster-watershed` | RasterWatershedSegmentation 321 **KI** |
| `raster-analysis/raster-morphology` | RasterReconstruction 113; RasterFillHoles 90 |

SpatialLagImpacts is the empty-name case. The other thirteen are the duplicate-destination cases.

### Output names beyond status placement

These are all 27 rows in the last column of the operation matrix. The direction is graph declaration → pure result.

| Operations | Difference |
|---|---|
| DelaunayTessellation | `groups.triangleCounts`, `groups.triangleOffsets`, `groups.status.*` become `groupTriangleCounts`, `groupTriangleOffsets`, `group*` fields. |
| LocalFrameProjection | One primary `output` view becomes a result with `output` and `validity`. |
| LineSimplification | Pure result includes `importance` and flattens status. |
| OrdinaryLeastSquares, SpatialRegressionDiagnostics, SpatialErrorGM | `status.code` becomes `status`. |
| SpatialErrorMaximumLikelihood, SpatialLagMaximumLikelihood | `status.code, status.converged` become `status, converged`. |
| SpatialTwoStageLeastSquares | Same `status.code` change, with an additional pure `robust` bundle. |
| SeeminglyUnrelatedRegression | Pure result also nests `spatial.parameters,status`. |
| MultiscaleGeographicallyWeightedRegression | Pure result flattens the status group and includes `inference`. |
| ProbitRegression | Pure result includes `spatialDiagnostics` and `spatialLag`. |
| SpatialPanelLagMaximumLikelihood | Pure result includes `hausman.tests,status`. |
| SpatialCrossValidation | Pure result includes `diagnostics`. |
| TrajectorySplit, TrajectoryThinning | `positions,timestamps` become `outputPositions,outputTimestamps`. Thinning also changes `trackOffsets` to `outputTrackOffsets`. |
| TrajectoryLocations | `status.count` becomes `locationCount`. |
| TrajectorySegments | Nested `triplegs` and `trips` columns and status groups become top-level names such as `triplegIds`, `triplegCount`, `tripIds`, `tripFinishRows`. |
| NetworkLouvain | Five named status outputs are absent from the pure result, FRM-11. |
| MapMatching | `traversedEdges.edges,counts,status.*` become `traversedEdges,traversedCounts,traversedOverflow,traversedCandidateOverflow`. |
| NetworkPathExtraction | Graph `output.capacity` is a sizing field. The pure result also has an `edges` bundle with its own geometry and counts. |
| NetworkStructure | Pure result includes `strongComponentsStatus`. |
| TerrainCatchmentSummary | Three additional pure status arrays, FRM-05. |
| ProfilePeaks | `peaks.ids`, `peaks.status.*` become top-level names. `peaks.capacity` is a sizing field. |
| CostDistancePath | `path.ids`, `path.status.*` become top-level names. |
| Streamlines | `lines.ids`, `lines.status.*` flatten, and `candidates.positions,spans` become `candidatePositions,candidateSpans`. |
| Isolines | `polylines.vertices,polylineOffsets,polylineLevels,polylineClosed,polylineCount,vertexCount` flatten. `polylines.vertexCapacity` sizes storage. |

The other 36 affected stems are named in the middle column of the matrix. SegmentSnap is counted there because the graph exposes both `candidateOverflow` and `status.candidateOverflow` for the same logical field.

### Props, defaults, limits and execution variants

- **Direct contributor → graph.** The three top-level capability keys dropped by the named graph props in the 128-pair comparison are `GPUTerrainFlow.flatContinuation`, `GPUCostDistance.continuation`, and `GPURipleyEnvelope.studyWindow`. The last one changes the spatial domain: both named operations use rectangles, while the direct contributor accepts a prepared polygon window. See `S/gpu-terrain/hydrology/gpu-terrain-flow-operation.ts:19,39`, **KI** `S/raster-analysis/cost-distance/gpu-cost-distance-operation.ts:12`, and **KI** `S/dataframe-analysis/pair-statistics/gpu-ripley-envelope-operation.ts:17,44`.
- **Direct contributor output → GPU graph output.** Four declared types lose leaves: RankMarkov (`homogeneity` from a wider shared type, unsuitable for the one-condition case), ContiguityWeights (`distances` and generic CSR metadata/status), SpatialWeightsAlgebra (generic CSR metadata/status), and TerrainCatchmentSummary (three real diagnostics). The generic CSR fields include `information`, `spatialContext` and status fields such as `count`, `uncertainCount`, `iterationCount` and `approximation`. Type membership alone does not prove a kernel writes every optional field.
- **Graph-only sizing.** The 102 graph forms that add props mostly ask for dimensions previously carried by views. Examples span `rowCount` for RankMarkov, `polygonCount,indexCapacity` for both polygon triangulators, `observationCount,queryCount` for ConditionalGaussianSimulation, `nodeCount,edgeCount,reverseEdgeCount,pivotCount` for NetworkCentrality, and `timestampFormat,rowCount,trackCount,pairCapacity,compactCapacity` for TrajectoryRestriction. The census code emits the exact per-operation key lists, including optionality. Starting from a direct contributor is a larger change than selecting the CPU or GPU graph class.
- **High-arity array functions.** All seven taking four or more arguments are `MapColoring(offsets,neighbors,parameters,priority,maximumIterations?)`, `ProfilePeaks(values,offsets,parameters,options,validity?)`, `SinkhornDivergence(source,target,parameters,options)`, `TerrainFlowDistance(flowDirections,options,streams?,transitionCosts?)`, `ClassBreaks(values,parameters,options,mask?)`, `CostDistancePath(backLinks,target,grid,capacity)`, and `RasterGeometricMedianComposite(values,parameters,options,mask?)`. These are particularly good candidates for named-object overloads.
- **Different omitted budgets.** SpaceTimeHawkes defaults to 100 GPU iterations and 1,000 CPU iterations. VariogramModelFit defaults to 32 and 60 refinements. Explicit budgets bind both in these two operations. Several search twins instead run to completion when the CPU budget is omitted, including TrajectoryStaypoints, TrajectoryRecursions, TrajectoryLocations, MaximumInscribedCircle and NetworkCostTable. The distinction should appear beside the option in the public forms table.
- **Explicit options with backend-dependent force.** Besides Sinkhorn’s `blurRadius`, MinimumBounds accepts `maximumHullVertices` on the CPU graph form but computes the full hull. ConstrainedDelaunayTriangulation ignores `maximumFlips` in the CPU solve. TerrainStreamOrder’s CPU queue completes without `maximumIterations` and reports `iterationCount=0`. The three Vecchia functions use all usable CSR neighbors on CPU, without the GPU’s `neighborCount` truncation. NormalizeGeometry can finish nesting that the GPU’s depth budget interrupts. ClassBreaks runs exact CPU natural breaks even when approximate GPU execution is selected. These are disclosed in source comments. They are material when the user selects an option deliberately.
- **Representation and hardware limits with legitimate differences.** PettittTest admits 131,071 CPU slices versus 512 on GPU. RankMarkov drops the GPU’s 64-location and 65,536-period caps. SpatialLagImpacts drops the GPU ceiling of 256 on `maximumPower` but still honors the requested truncation. DelaunayTessellation counts inserted sites on CPU versus insertion/flip rounds on GPU. NetworkFlow uses exact integer costs internally on CPU and can admit larger capacities. Reporting the scope and meaning of the difference is preferable to imposing all GPU ceilings on CPU.
- **Chunked and tiled identities.** `GPUChunkedSpatialWeightsValidation`, `GPUChunkedSpatialWeightsTranspose`, `GPUChunkedSpatialWeightsTransform` and `GPUChunkedSpatialWeightsAlgebra` correspond to the four unprefixed spatial-weight contributors. Only Validation and Algebra have named graph/CPU pairs. `GPUTerrainTiledFlow` is the tiled flow route. `addRasterRegionGraphTiles` is a builder for `GPURasterRegionGraph`, and `GPUIsolines` takes an optional `tile`. A user cannot discover these three forms by applying one prefix or suffix.
- **Prepared identities.** `GPUSpatialJoinPrepared` feeds PointInPolygonJoin, NearestFeatureJoin, SpatialPredicateJoin and SpatialJoinCandidates. `GPUStudyWindowPreparation` is relevant to KernelDensity, InhomogeneousPointPattern, Variogram, SpatialCorrelogram, Ripley, RipleyEnvelope, PointPatternIndices and RipleyDistanceFunctions. ConditionalGaussianSimulation exposes `reusePreparation`, and RasterConservativeRemapping exposes `preparedWeights`. Polygon contiguity has a separate public `preparePolygonContiguityTopology` → `computePolygonContiguityOnCPU` route. A forms index needs relationships between exports as well as a list of names.

### Recipe argument renames

These are all 24 direct mappings from the stated AST test. Names omit `add…Recipe` and `GPU`. The left side is the recipe prop.

| Recipe | Recipe prop → stage prop |
|---|---|
| HotSpotAnalysis, PointsInPolygonsChoropleth | `preparationParameters → LonLatPolygonPreparation.parameters` in each |
| SpatialRegression | `olsParameters → OrdinaryLeastSquares.parameters` |
| SpaceTimeHotSpots | `calendarParameters → CalendarBuckets.parameters` |
| PeriodComparison | `classBreaksParameters → ClassBreaks.parameters`, `colorScaleParameters → ColorScale.parameters` |
| ClusterAndOutline | `maximumVerticesPerHull → GroupConvexHull.maximumVerticesPerGroup`, `hullCapacity → GroupConvexHull.vertexCapacity`, `geometryParameters → GroupGeometry.parameters`, `clusteringParameters → SpatialClustering.parameters` |
| ChangeOfSupport | `preparationParameters → LonLatPolygonPreparation.parameters`, `sourceValues → SpatialLag.values` |
| StraightLineCatchments | `facilityCount → DistanceField.seedCount`, `facilities.sourceIds → seedIds`, `facilities.positions → seedPositions` |
| FleetDwell | `zones → PointInPolygonJoin.polygons` |
| ZonalDistribution, RasterZonalDistribution | `probabilities → GroupStatistics.percentiles` in each |
| ExposureByZone, RegionalRasterChange | `zoneCount → RasterZonalStatistics.zoneCapacity` in each |
| DasymetricRedistribution | `sourceTotals → SpatialLag.values` |
| LandscapeMetrics | `countBoundary → RasterPatchMetrics.countRasterBorder` |
| DriveTimeCatchment | `facilities → NetworkSnapping.positions`, `demand → NetworkSnapping.positions` in separate stages |

## Quick wins

- [ ] Validate destination names before constructing destination-keyed output objects in the 14 listed CPU constructors. Pin the two-output p-value example.
- [ ] Capture MultipleTestingAdjustment’s validated method, sizes and output names once.
- [ ] Share the catchment-summary graph props and forward its three GPU status views. Validate only declared arrays in the pure helper.
- [ ] Accept canonical trajectory keys in `computeTrajectoryRestrictionOnCPU` and name both expected keys in errors.
- [ ] Give pure Louvain a named status decoder used by its graph adapter.
- [ ] Put the three-node `.1 + .4` cutoff example on the routing pages and qualify the library-wide float64 statement.
- [ ] Check the approximate natural-breaks class/bin relationship before backend selection.
- [ ] Replace the 125-export floor with exact qualified-set checks. Include public partial forms explicitly.
- [ ] Refresh straight-skeleton provenance after GPU verification and compare the live edge pairs as well as status and geometry.

## Larger proposals

1. **Extend the four-row forms catalog before publishing.** This is additive. Start with the 128 complete pairs, then the two direct-only joins, prepared contiguity, and the named chunked/tiled forms. Render each family’s actual function signature, differences in defaults and limits, and a runnable CPU invocation. A reader should be able to answer “can I swap this operation?” from its own page.
2. **Use common logical result paths for new forms.** For existing functions, add a named result view or decoder first. Renaming all existing fields would affect 63 pure results, including 36 where status placement is the only issue. Start with Louvain, trajectory segments and path extraction, where an ordinary caller otherwise has to reconstruct useful structure.
3. **Distinguish a bounded calculation from an unbounded reference.** For Sinkhorn I recommend the breaking fix in FRM-02: an explicit radius should bind. For algorithms such as exact Fisher–Jenks that intentionally provide a reference for an approximate GPU route, identify the reference behavior in the forms record and example. A dropped hardware ceiling and an ignored scientific option need different descriptions.
4. **Forward the drainage recipe’s units before alpha.** Passing through the existing setting costs less than introducing a second units option. Document the numeric change for geographic and Web Mercator users.

## Not read

I inspected public entry barrels, resolved declarations for all 498 producers and 128 graph pairs, read the pure functions’ difference notes, queried the live catalog, examined the first-round synthesis and relevant area/decision reports, and scanned the public recipes. I read the implementations behind the reported counterexamples and followed the 25 lens examples into their scenario inputs. I used the newer KI copies for staging-owned paths, including the network references, several geometry references, raster constructors, recipe sources and central forms page cited above.

I did not independently prove every algorithm’s stated tolerance or every default in every implementation. The per-operation census includes the authors’ documented differences; the findings explicitly distinguish those declarations from executed behavior. I did not execute shaders, run Vitest, rebuild declarations or write files. The GPU probes identified above remain unverified on this review machine.