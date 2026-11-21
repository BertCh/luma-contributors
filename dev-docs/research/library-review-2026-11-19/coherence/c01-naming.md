# Names that change the answer

## Summary

I counted **761 public classes, 7,852 class–constructor-property pairs, and 509 distinct string literals directly accepted by constructor fields**. The GPU/CPU class prefixes are consistent. The costly inconsistencies are in settings that look portable between operations.

- `percentiles: [.95]` means the 95th percentile in grouped statistics and the 0.95th percentile in network travel times. Both are valid inputs. On a two-departure CPU fixture the distinction was **5 versus 90**.
- Copying the documented inverse-distance option from nearest-feature weights into neighbor search produced **binary weights**. The object-spread version passed TypeScript.
- `ties: 'lowest-id'` selects the lowest **feature row**. With equally distant features carrying IDs `[99, 1]`, the CPU nearest join returned **99**, with both loss flags clear.
- The most valuable pre-alpha change is to give shared settings one public spelling **and one unit**, starting with quantiles, distance weights and feature identity. Check the resolved TypeScript type, packed word and reported metadata together. Each of those three surfaces looked plausible in isolation.

## Census

**COUNT.** I followed the 11 entries in `modules/analysis/package.json`, including the root and nested ontology entry, through their TypeScript exports. The program reached 199 source `index.ts` files. There are 200 on disk. I deduplicated re-exports, resolved inherited constructors, mapped types, intersections and each branch of discriminated unions. For the three `(device, props)` constructors I counted `props`. Counts below are for r19. `KI` identifies the newer staging checkout where applicable.

| Public class variant | Classes |
|---|---:|
| `GPU*` with `getCommandNodes` | 498 |
| Other `GPU*` | 133 |
| `CPU*` | 128 |
| `AnalysisGeographicExtentAccumulator`, `FlowFieldViewportTracker` | 2 |
| **Total** | **761** |

| Defining family | Classes | Producers | Class–prop pairs |
|---|---:|---:|---:|
| Spatial analysis, including ontology | 430 | 270 | 4,518 |
| Raster | 113 | 86 | 1,091 |
| Network | 75 | 41 | 843 |
| Terrain | 68 | 46 | 670 |
| Dataframe | 48 | 34 | 490 |
| Point cloud | 11 | 11 | 118 |
| Analysis core | 10 | 6 | 84 |
| Computation record | 4 | 2 | 18 |
| Geospatial | 2 | 2 | 20 |

There are **1,297 distinct top-level prop names**. Of the 7,852 class–prop pairs, **668** directly accept string literals, across **393 classes**. They admit **1,743 class–prop–literal triples**. Of the 509 distinct values, 354 are lowercase alphanumeric, 129 are lowercase hyphenated, eight are uppercase codes, 17 are mixed case, and one is `kmeans++`. The census found five `criterion: 'queen'` class fields and zero `'Queen'`. It found 42 class–prop occurrences of `'none'`, zero `'off'`, and zero string `'false'`. These examples are already consistent.

A separate recursive walk through package-authored constructor structures, including nested outputs and array element types, found **2,319 field names and 603 literal values**. Extending that walk to all **1,018 exported `*Props`, `*Settings` and `*Options` types** found **3,040 field names and 785 literals**. Those broader counts include port discriminants and metadata. They matter here because `weightKind`, the DBSCAN minimum and raster stretch percentiles live in packers rather than constructors.

This is the reproducible core of the census. Run with `node` from the checkout to print the class names, resolved direct prop names and literals, including classes with no props. For the recursive count I additionally followed unions, array and tuple type arguments, and locally declared data members, with a visited-type set. I stopped at class instances and external implementation members.

```js
const ts = require('typescript');
const entries = Object.values(require('./modules/analysis/package.json').exports);
const roots = entries.map(entry =>
  entry.types.replace('./dist/', 'modules/analysis/src/').replace('.d.ts', '.ts')
);
const options = ts.convertCompilerOptionsFromJson(
  require('./tsconfig.json').compilerOptions, process.cwd()
).options;
const program = ts.createProgram(roots, options);
const checker = program.getTypeChecker();
const seen = new Set();
const branches = type => type.isUnion() ? type.types : [type];

for (const root of roots) {
  const module = checker.getSymbolAtLocation(program.getSourceFile(root));
  for (const entry of checker.getExportsOfModule(module)) {
    const symbol = entry.flags & ts.SymbolFlags.Alias ? checker.getAliasedSymbol(entry) : entry;
    const decl = symbol.valueDeclaration;
    if (!decl || !ts.isClassDeclaration(decl) || seen.has(entry.name)) continue;
    seen.add(entry.name);
    const props = new Map();
    for (const sig of checker.getTypeOfSymbolAtLocation(symbol, decl).getConstructSignatures()) {
      const arg = sig.parameters.find(p => p.name === 'props') ?? sig.parameters[0];
      if (!arg) continue;
      const type = checker.getNonNullableType(checker.getTypeOfSymbolAtLocation(arg, decl));
      for (const branch of branches(type)) {
        for (const field of checker.getPropertiesOfType(branch)) {
          if (!props.has(field.name)) props.set(field.name, new Set());
          const value = checker.getTypeOfSymbolAtLocation(field, field.valueDeclaration ?? decl);
          for (const item of branches(value)) {
            if (item.isStringLiteral()) props.get(field.name).add(item.value);
          }
        }
      }
    }
    console.log(entry.name, [...props].map(([name, values]) => [name, ...values]));
  }
}
```

### Synonyms and false synonyms

Counts are **all public classes / producers** with the exact top-level prop. A constructor and its GPU and CPU operation forms count separately when all three expose the prop. Packer fields are identified separately in the findings.

| Concept | Spellings and counts | Proposed spelling by meaning |
|---|---|---|
| Distance and support | `radius` 8/8, `distance` 8/6, `searchRadius` 3/3, `maximumDistance` 8/6, `maximumRadius` 7/7, `bandwidth` 4/4, `gaussianCutoff` 7/5, bare `cutoff` 0/0 | `searchRadius` for spatial inclusion, `bandwidth` for kernel scale, `cellRadius` for an integer footprint, `indexCellSizeHint` for performance only. A Gaussian cutoff expressed in bandwidths needs that unit in its name or type. |
| Requested neighbors and storage | `k` 5/5, `neighborCount` 16/8, `maximumNeighborCount` 6/6, `maximumNeighbors` 9/7, `neighborhoodCapacity` 2/2 | `neighborCount` for requested neighbors, `maximumNeighborCount` for an algorithmic upper bound, `neighborCapacity` for allocated slots per query, `clusterCount` for k-means. |
| Grid resolution | `cellSize` 8/6, `resolution` 8/6, `spacing` 4/2 | `resolution` for a DGGS level, `binsPerAxis` for a matrix dimension, `metersPerPixel` for ground resolution, `cellSize` for a step in declared coordinate units. |
| Data and weights | `weights` 101/67, `values` 131/94, bare `weight` and `field` both 0/0 | `values` for observations, `weights` for weights, `weightType` for their construction rule. A CSR weights object and an observation column need distinct types. |
| Coordinates and geometries | `positions` 222/138, `points` 5/5, `coordinates` 1/1, `xy` 0/0, `queryPositions` 2/2, `queries` 14/10, `referencePositions` 4/2 | `positions` and `queryPositions` for spatial coordinate views, `points` for a point-geometry port. Qualify embedding vectors, network query pairs and coordinate-encoder features separately. |
| Identity and indexing | Bare `ids` 0/0, `rows` 2/2, `indices` 1/1. Suffixed `*Ids` appears on 41 classes, `*Rows` on 23, `*Indices` on seven. | `…Ids` for identifiers supplied or preserved by the caller, `…Rows` for positions that index a named input. |
| Cardinality | `count` 6/4, `rowCount` 120/8, `cellCount` 42/34, `capacity` 38/11, `length` 1/0 | `<entity>Count` for logical cardinality, `<entity>Capacity` for allocated storage. State whether a capacity is per row or total. |

The earlier direct-declaration census already identified `maximumNeighbors`, `maximumVertices` and validity aliases. I have not counted those as new findings. Resolving constructor types adds two actionable inherited `maximumNeighbors` producers, `GPUSpaceTimeLocalMoran` and `GPULOSHResidualPermutationTest`, to the five directly identified earlier. `KNOWN(ERG-104)`.

## Proposed convention

Name each public input for its **role, unit and index space**. Reuse the same spelling and finite set of option values wherever those three agree. Use the reference algorithm’s searchable name for a class that implements that algorithm, and lowercase hyphenated strings for library-defined multiword options.

A checker can compare entry-qualified constructor fields and packer settings against a small table of shared concepts. For each shared option, test one nondefault value through both sibling entry points, plus an unknown value through the JavaScript boundary. For units and identity, use a fixture where the two interpretations give different answers.

## Findings

Paths below are relative to `modules/analysis/src` unless prefixed with `docs/`, `modules/analysis/test/` or `examples/`. Costs are implementation estimates.

| ID | Kind | Reach | Change | Cost | Where | One line |
|---|---|---|---|---|---|---|
| NAM-01 | W | ORDINARY | ADDITIVE | S | Group statistics, raster stretch, network time percentiles | The same `percentiles` input has two valid scales. |
| NAM-02 | W | ORDINARY | ADDITIVE | S | Neighbor search, nearest-feature weights | Reusing inverse-distance settings produces binary weights. |
| NAM-03 | W | ORDINARY | BREAKING | S | Nearest-feature join | `'lowest-id'` can select the largest application ID. |
| NAM-04 | W | ORDINARY | ADDITIVE | M | Raster flow, isolines, dot density | Four-number `Extent` tuples encode either bounds or signed pixel steps. |
| NAM-05 | W | ORDINARY | ADDITIVE | S | Raster sampling and cell statistics | Copying `'strict'` into cell statistics selects ignore-nodata. |
| NAM-06 | W | UNUSUAL | ADDITIVE | S | H3 point indexing and cell measures | Copying `'fp64'` selects the ordinary H3 indexing kernel. |
| NAM-07 | W | UNUSUAL | ADDITIVE | S | Geospatial point buffer | The library’s longitude/latitude token selects planar buffer arithmetic. |
| NAM-08 | W | ORDINARY | ADDITIVE | S | Key join and pair gather | Reusing `how: 'inner'` selects a left key join. |
| NAM-09 | W | UNUSUAL | ADDITIVE | S | Spherical overlays and weights algebra | An unknown overlay operation selects symmetric difference. |
| NAM-10 | S | ORDINARY | BREAKING | S | Analytic inference | Two public types offer permutation-only tails that always throw. |
| NAM-11 | C | ORDINARY | ADDITIVE | S | Point-pattern indices | `maximumDistance` names a grid-size hint, with no distance limit. |
| NAM-12 | E | ORDINARY | ADDITIVE | S | Neighbor-search parameter packer | Omitting one `radius` means unbounded kNN or an empty distance band. |
| NAM-13 | C | ORDINARY | BREAKING | S | Spatial and embedding k-means | `k` changes counted entity across five producers. |
| NAM-14 | E | ORDINARY | BREAKING | M | Density clustering and interpolation | Three class names omit the algorithm or computed noun a GIS user searches for. |
| NAM-15 | C | ORDINARY | ADDITIVE | S | Similar locations and standardization | The z-score option changes spelling between families. |
| NAM-16 | C | ORDINARY | ADDITIVE | S | Density kernels and spatial weights | A documented quartic/bisquare alias works in only one direction across families. |

### NAM-01: `percentiles` has a 100-fold unit collision

**Where.** `dataframe-analysis/group-statistics/gpu-group-statistics.ts:252`, `raster-analysis/raster-stretch/raster-stretch-parameters.ts:57`, `gpu-network/network-time-geography/gpu-network-travel-time-percentiles.ts:66` and its GPU/CPU operation forms. The existing adapter effectively supplies the desired vocabulary at `gpu-spatial-analysis/recipes/zonal-distribution-recipe.ts:169`, mapping `probabilities` to input `percentiles`, then mapping output `percentiles` to `quantiles` at line 185.

**EXECUTED.** The network CPU twin on a one-edge network with departure times `[0,1]` and respective travel times `[5,90]`, given `percentiles: [.95,95]`, returned destination times `[5,90]`. The stretch packer given `[.02,.98]` wrote fractions `[.0002,.0098]` within float32 rounding. Given `[2,98]` it wrote `[.02,.98]`. Group statistics explicitly asks for fractions in `[0,1]`. This is independent of the rank-rounding fix in `KNOWN(NET-04)`.

**Now.** A user following `docs/api/gpu-dataframe-analysis.md:835` can keep `const percentiles = [.95]` and pass it unchanged to the network CPU API. No range error distinguishes the two requests. The KI dataframe guide still uses `percentiles: fractions.importToGraph(graph)` at line 851.

**Wanted.** Expose `probabilities` for the grouped input and `quantiles` for the output values, following the existing recipe. Reserve `percentiles` for the 0–100 scale. Preserve the old group names as aliases with their original units. Put `.95 ↔ 95` in the three API tables. **Callers:** none required for aliases. **Value/cost:** cross-family users get the intended order statistic, S.

### NAM-02: The nearest-neighbor weight setting silently becomes binary

**Where.** `gpu-spatial-analysis/spatial-join/gpu-nearest-feature-weights.ts:128`, `gpu-spatial-analysis/neighbor-search/neighbor-search-parameters.ts:133`, `:173`, `:224`. The two names remain in KI, including `docs/api/gpu-spatial-analysis/operators-joins.md:795` and `operators-weights.md:369`.

**EXECUTED.** This was my first-use probe from the join documentation. TypeScript reported zero diagnostics for the spread. On both checkouts the packer wrote weight code `0` and the metadata helper returned `'binary'`. The intended results were `1` and `'raw'`.

```ts
const weighting = {weightType: 'inverse-distance'} as const;
const settings = {bounds: [0, 0, 4, 4] as const, ...weighting};
getGPUNeighborSearchParameterValues(settings);
```

Changing just the prop to `weightKind` throws `GPUNeighborSearch unknown weightKind inverse-distance`. The working pair is `weightKind: 'inverseDistance'`.

**Wanted.** Accept `weightType: 'binary' | 'inverse-distance' | 'kernel'` in the neighbor packer and metadata helper. Normalize both historical spellings once. Reject conflicting choices if both keys are present. Retain each contributor’s supported subset. **Callers:** none required. **Value/cost:** GIS users can reuse one documented weighting choice and get matching bytes and metadata, S.

### NAM-03: `'lowest-id'` breaks ties on source row

**Where.** `gpu-spatial-analysis/spatial-join/nearest-types.ts:47`, `gpu-nearest-feature-join.ts:339`, `cpu-feature-join.ts:582`. KI retains the rule at `nearest-types.ts:47` and `cpu-feature-join.ts:698`. `KNOWN(GEO-02)` fixed the downstream rows/IDs wiring. The public tie selector still promises the other identity.

**EXECUTED.** Query `(0,0)`, targets `(-1,0)` and `(1,0)`, target `sourceIds=[99,1]`. On both checkouts `computeNearestFeatureJoinOnCPU` returned `neighborRows=[0]`, `neighborIds=[99]`, `neighborDistances=[1]`, `overflow=0`, `candidateOverflow=0`. A lowest-application-ID rule would return row `1`, ID `1`. The CPU twin implements the default one-neighbor rule. The GPU selector is documented and generated with that same row ordering. A confirming GPU probe uses these three points, `k: 1`, and reads both row and ID.

**Wanted.** Rename the option to `ties: 'lowest-row' | 'all'`. Preserve ordering and results. **Callers:** `rg` found **17 quoted-literal lines in 11 consumer, test and website files** outside implementation, including the nearest-facility scene, explorer control, lens choices and parity fixtures. The showcase already labels the option “Keep the lowest feature rows.” **Value/cost:** users can predict which facility wins a tie from the option alone, S.

### NAM-04: An extent can be two corners or one corner and two steps

**Where.** Bounds tuples occur in `raster-analysis/isolines/isolines-parameters.ts:32`, `isobands-parameters.ts:33`, `gpu-network/network-isochrones/network-isochrones-parameters.ts:22` and `gpu-spatial-analysis/trajectory-home-range/gpu-brownian-bridge-utilization.ts:73`. Placement tuples occur in `raster-analysis/lagrangian-diagnostics/gpu-lagrangian-flow-map.ts:61`, `particle-advection/particle-advection-parameters.ts:27`, `flow-texture/line-integral-convolution-parameters.ts:27` and `:34`, `flow-texture/streamlines-parameters.ts:32` and `:39`, and `gpu-spatial-analysis/dot-density/dot-density-parameters.ts:21`.

**EXECUTED.** For a 2×2 raster covering `[100,200,120,220]`, the isolines packer encoded origin `[100,200]`, step `[10,10]`. Passing the same tuple as `fieldExtent` to the flow-map packer encoded origin `[100,200]`, step **`[120,220]`**. Both succeeded. No shader execution is necessary to see the placement change.

**Wanted.** Keep bounds as `extent`. Add `fieldPlacement`, `outputPlacement`, `gridPlacement` and `maskPlacement` accepting `{origin: [x,y], cellSize: [dx,dy]}`. Encode to the current layout and retain signed Y steps. Update examples to pass the object. **Callers:** none required with old tuple aliases. **Value/cost:** raster developers can move a georeferenced grid between families without changing its scale or row direction, M.

### NAM-05: `'strict'` can request the opposite missing-data policy

**Where.** `raster-analysis/raster-sampling/raster-sampling-parameters.ts:20`, `raster-algebra/local-operations-parameters.ts:48`, `raster-algebra/weighted-overlay-parameters.ts:82`, `focal-pairs/gpu-raster-focal-pairs.ts:149`, `gpu-raster-rank-filter.ts:133`, `gpu-raster-reprojection.ts:48` and `gpu-raster-tile-reprojection.ts:48`. KI still documents both vocabularies in `docs/api/gpu-raster/operations-analysis.md:712` and `:1165`.

**EXECUTED.** `getGPURasterCellStatisticsParameterValues({noDataPolicy:'strict'})` returned `[0,1,0,0]`, exactly the result for `'ignore'`. `'propagate'` returned `[1,1,0,0]`. Sampling documents `'strict'` as requiring every participating sample to be valid. A JavaScript user copying it into cell statistics loses that requirement. A typed direct literal is caught.

**Wanted.** Use `'propagate'` as the common spelling for invalidating a result on a missing participant. Accept `'strict'` as an alias for that behavior. Keep `'renormalize'` for interpolation that redistributes the surviving weights. Validate the token before packing or generating a branch. **Callers:** none required. **Value/cost:** raster pipelines retain their missing-data policy across stages, S.

### NAM-06: Copying H3 measurement precision selects the ordinary indexing kernel

**Where.** `gpu-spatial-analysis/cell-indexing/gpu-point-to-cell.ts:89`, `:232`, `:302`, `gpu-cell-measures.ts:124`, and `gpu-spatial-analysis/cell-topology/gpu-s2-cell-measures.ts:53`. The option tuples also appear through the point-to-cell and cell-measures GPU/CPU operation forms. The KI cell guide still recommends `'fp64'` for measures at line 453.

**EXECUTED.** I declared three one-row H3 indexing graphs on a CPU null device at resolution 12. `precision:'float32'` produced workload variant `'h3'`. **`precision:'fp64'` also produced `'h3'`.** `'extended'` produced `'h3-extended'`. This establishes host selection, independently of any assertion about a cell key.

**Wanted.** Accept one shared user-facing token, `'double-single'`, for the high-precision GPU routes. Keep `'extended'` and `'fp64'` as aliases where applicable. Reject unsupported precision values at construction and document the CPU twin’s binary64 behavior separately. **Callers:** none required. **Value/cost:** a precision request survives the index-to-measure workflow, S.

### NAM-07: `GPUPointBuffer` accepts the shared coordinate-space token and builds a planar body

**Where.** `geospatial-analysis/point-buffer/gpu-point-buffer.ts:90`, `:235`, `:259`, compared with `analysis-core/contracts/spatial-context.ts:45` and `:366`. **175 producers** accept `spatialContext`. Only `GPUPointBuffer` has the top-level prop `coordinateSystem`.

**EXECUTED / SOURCE.** On a null device, `coordinateSystem:'longitude-latitude'` constructed successfully and emitted a workload labelled `'longitude-latitude'`. The body selection at line 259 tests only `coordinateSystem === 'geographic'` and otherwise inserts the planar body. This spelling can occur naturally when adapting `context.coordinateSpace` in JavaScript. **GPU confirmation:** buffer one equatorial point at a radius of 1,000 metres with four vertices. The northward geographic offset should be about `0.009°`. The selected planar body uses the radius directly.

**Wanted.** Immediately reject unknown `coordinateSystem` values. Accept the shared `spatialContext` with explicit supported planar and spherical-metre combinations. Derive the workload label from the resolved mode. **Callers:** none required for existing valid modes. **Value/cost:** geographic users can no longer obtain planar offsets under a geographic-looking workload label, S.

### NAM-08: The common join-mode prop is ignored by key join

**Where.** `dataframe-analysis/key-join/gpu-key-join.ts:202`, `:256`, `:344`, `gpu-spatial-analysis/pair-gather/gpu-pair-gather.ts:292`, `gpu-spatial-analysis/spatial-join/gpu-spatial-predicate-join.ts:415`. Two producers use `how`, one uses `kind`. Key join and pair gather share both `'left'` and `'inner'`, but have opposite defaults.

**EXECUTED.** With valid keys and a row-aligned `rightRows` output, `new GPUKeyJoin({...ports, ...{how:'inner'}}).kind` returned **`'left'`**. The spread passed the TypeScript probe. With the recognized `kind:'inner'`, the same configuration requires the compact `output.rows` port. The spelling error suppresses that diagnostic too.

**Wanted.** Accept `how` on `GPUKeyJoin`, resolve it before validating mode-specific outputs and retain `kind` as an alias. Reject disagreement between the two. Keep the existing per-operation defaults explicit in the docs. **Callers:** none required. **Value/cost:** dataframe and spatial joins can share join settings without silently changing row selection, S.

### NAM-09: Unknown spherical overlay names select symmetric difference

**Where.** `gpu-spatial-analysis/global-coordinates/gpu-spherical-arc-overlay.ts:155`, `:255`, `gpu-spherical-polygon-overlay.ts:160`, and `cpu-spherical-arc-overlay-operation.ts:451`. KI retains the fallbacks at arc lines 154 and 264 and polygon line 357. Geometry uses `'symmetric-difference'`; weights algebra uses `'symmetricDifference'` on its ordinary, chunked, GPU-operation and CPU-operation forms.

**EXECUTED.** With both operands declaring `referenceInside: true`, arc overlay returned `referenceInside === true` for `'union'` and **`false` for `'Union'`**, on both copies. The CPU operation explicitly rejects an unknown operation. This is an observable host result before graph submission. The smallest GPU follow-up is the same nonempty polygon on both sides: union retains it, symmetric difference removes it.

**Wanted.** Validate against an explicit operation set before computing reference membership. Use `'symmetric-difference'` across set operations, accepting the historical weights alias. **Callers:** none required. **Value/cost:** a spelling error cannot become a valid, different geometry operation, S.

### NAM-10: Analytic inference advertises tails it cannot run

**Where.** `analysis-core/contracts/inference.ts:13`, `gpu-spatial-analysis/global-spatial-statistics/gpu-global-spatial-statistics.ts:218`, `:1525`, and `dataframe-analysis/pair-statistics/point-pattern-indices-parameters.ts:78`, `:133`. KI global statistics still declares the broad type at line 221 and rejects the options at line 1561.

**EXECUTED.** Both `new GPUGlobalSpatialStatistics({...ports, alternative:'folded'})` and `getGPUPointPatternIndicesParameterValues({...settings, alternative:'directed'})` passed the TypeScript probe. Both entry points rejected **both** `'folded'` and `'directed'` at runtime. The point-pattern packer accepted `'lesser'` and encoded `2`. The LOSH residual test already demonstrates the narrower declaration at `gpu-spatial-analysis/spatial-autocorrelation/gpu-losh-residual-permutation-test.ts:32`.

**Wanted.** Export `GPUAnalyticAlternative = 'two-sided' | 'greater' | 'lesser'` and use it on the two rejecting surfaces. Keep the broader union for permutation methods. **Callers:** one intentional invalid-input call site in the global-statistics Node spec at line 322 needs an explicit invalid-input cast. The point-pattern rejection test already uses one. I inspected 123 direct call expressions in the files selected by `rg` for the two entry points. **Value/cost:** autocomplete and static checking agree with the accepted scientific choices, S.

### NAM-11: `maximumDistance` can mean “do not limit the distance”

**Where.** `dataframe-analysis/pair-statistics/point-pattern-indices-parameters.ts:55` and `gpu-point-pattern-indices.ts:310`, compared with `variogram-parameters.ts:30`, `spatial-correlogram-parameters.ts:40`, `ripley-parameters.ts:55`, `ripley-distance-parameters.ts:41` and `gpu-space-time-variogram.ts:40`.

**EXECUTED / SOURCE.** The point-pattern packer accepts `maximumDistance:1` and stores `1` in word four. Its nearest-neighbor search uses it to size cells. Variogram calls the same field the largest accepted lag. In the API guide these meanings occur on the same page. **GPU confirmation:** use two unmasked points `(0,0)` and `(3,0)`, a covering rectangle, and `maximumDistance:1`. Read `nearestNeighborDistances` from point-pattern indices and pair counts from a one-bin variogram. The traced implementations predict `[3,3]` and zero pairs respectively.

**Wanted.** Expose `indexCellSizeHint` on `GPUPointPatternIndicesSettings`. Document it as a performance setting and preserve the numerical meaning of the old alias. **Callers:** none required. **Value/cost:** point-pattern users can distinguish tuning the index from censoring nearest-neighbor distances, S.

### NAM-12: An optional `radius` has opposite defaults in two search modes

**Where.** `gpu-spatial-analysis/neighbor-search/neighbor-search-parameters.ts:127`, `:223`, and `neighbor-search-kernels.ts:86`, `:100`, `:107`, `:121`. The API table documents both interpretations at `docs/api/gpu-spatial-analysis/operators-weights.md:481`.

**EXECUTED / SOURCE.** `getGPUNeighborSearchParameterValues({bounds:[0,0,2,2]})[4]` returned `Infinity`. The packer does not know the mode. In kNN this disables the maximum distance. In radius mode the generated lattice validity test requires a finite positive radius. **GPU confirmation:** use two distinct points one unit apart, complete output capacity and the same packed buffer. Compare `mode:'knn', k:1` with `mode:'radius'`; inspect CSR offsets.

**Wanted.** Add mode-aware packing. For `{mode:'radius'}`, require a positive finite `searchRadius`. For `{mode:'knn'}`, accept optional `maximumDistance` with omitted meaning unbounded. Both can write the existing word. Keep the current packer overload for existing callers. **Callers:** none required. **Value/cost:** a newcomer cannot accidentally pack an empty distance-band search using the new signature, S.

### NAM-13: Four `k` props count neighbors and the fifth counts clusters

**Where.** `gpu-spatial-analysis/neighbor-search/gpu-neighbor-search.ts:236`, `space-time-tests/gpu-jacquez-test.ts:127`, `spatial-join/gpu-nearest-feature-join.ts:333`, `spatial-join/gpu-nearest-feature-weights.ts:117`, `spatial-clustering/gpu-spatial-kmeans.ts:205`. Its closest sibling, `raster-analysis/raster-embeddings/gpu-raster-embedding-kmeans.ts:43`, already uses `clusterCount`.

**COUNT.** Five producers expose bare `k`. Eight expose `neighborCount`. In the nearest-feature weights adapter, `k` is the number kept while `slotCapacity` is the number of available slots. Replacing both with a generic maximum would destroy information. The staged nearest-feature and neighbor-search copies retain `k`.

**Wanted.** Rename required `GPUSpatialKMeans.k` to `clusterCount`. Offer `neighborCount` on the four neighbor APIs and use it in examples. Keep `k` in mathematical prose, including “k nearest neighbors.” **Callers:** `rg -o '\bnew GPUSpatialKMeans\('` found **47 constructor spellings** across code and embedded examples. The four neighbor aliases require no immediate migrations. **Value/cost:** application settings name the quantity they control and the two k-means APIs agree, S.

### NAM-14: The clustering exports hide the reference algorithm

**Where.** `gpu-spatial-analysis/spatial-clustering/gpu-spatial-clustering.ts:256`, `gpu-hierarchical-density-clustering.ts:164`, `gpu-optics.ts:229`, `spatial-clustering-parameters.ts:48`, and `gpu-spatial-analysis/areal-interpolation/gpu-pycnophylactic.ts:174`. The DBSCAN name is absent from the class export in both checkouts. The repository’s own cross-reference maps these names at `docs/api/gpu-spatial-analysis-cross-reference.md:131`, `:132`, `:161`.

**COUNT / EXECUTED.** Of the DBSCAN, HDBSCAN and OPTICS algorithm families, only `GPUOPTICS` can be found by typing the reference algorithm after `GPU`. DBSCAN also calls its self-inclusive sample threshold `minimumPoints`. The other two call it `minimumSamples`. The DBSCAN packer with `epsilon:1`, valid bounds and `minimumSamples:4` threw `minimumPoints must be an integer in [1, 16777216]`; `minimumPoints:4` packed `4`. `KNOWN(PTS-124)` concerns the HDBSCAN default. This proposal concerns discoverability and spelling.

**Wanted.** Rename the classes to `GPUDBSCAN`, `GPUHDBSCAN` and `GPUPycnophylacticInterpolation`. Match their props and public packer names. Accept `minimumSamples` in DBSCAN. Keep descriptions of supported variants and self-inclusion alongside the option. The abbreviation rule in `AGENTS.md:177` applies to variables; DBSCAN and HDBSCAN are established algorithm names that users already search for.

**Callers:** the code-file `rg` census found **57**, **47** and **34** `new` spellings respectively, including embedded examples. The DBSCAN packer has **59 declaration/call-text matches in 33 TypeScript files**. Imports, ontology export names, docs headings, lenses and website catalog records also need migration. **Value/cost:** users coming from PostGIS, scikit-learn or Tobler can find the implementation in autocomplete, M.

### NAM-15: Z-score standardization changes spelling between siblings

**Where.** `dataframe-analysis/variable-standardization/gpu-variable-standardization.ts:48`, `dataframe-analysis/composite-indicators/composite-score-parameters.ts:69`, and `gpu-spatial-analysis/similar-locations/gpu-similar-locations.ts:110`, `:211`. KI similar locations retains `'zscore'` at line 144 and the rejection at line 247.

**EXECUTED.** The standardization guide uses `method:'z-score'` at `docs/api/variable-standardization.md:14`. On both copies, `new GPUSimilarLocations({...ports, standardization:'z-score'})` threw `standardization must be 'zscore' or 'rank'`. The default divisor is a separate choice: similar locations uses population deviation.

**Wanted.** Accept `'z-score'` in similar locations and use it as the documented spelling. Preserve `'zscore'` as an alias and state the existing divisor. **Callers:** none required. **Value/cost:** dataframe and spatial users can reuse the name of the transformation, S.

### NAM-16: The quartic kernel alias stops at the family boundary

**Where.** `gpu-spatial-analysis/point-density/gpu-kernel-density.ts:332`, `gpu-adaptive-bandwidth-kernel-density.ts:214`, `gpu-space-time-kernel-density.ts:255`, `:267`, `gpu-space-time-relative-risk.ts:95`; `gpu-spatial-analysis/neighbor-search/neighbor-search-parameters.ts:97`, and `spatial-weights/gpu-spatial-weights-transform.ts:74`, `:302`. The chunked transform also exposes this kernel at `gpu-chunked-spatial-weights-transform.ts:52`. KI retains the transform restriction at line 364.

**EXECUTED.** `getGPUSpatialWeightsKernelValue('bisquare', .5)` returned `0.52734375`, exactly `15/16 × (1 − .5²)²`; `'quartic'` threw `unknown kernel quartic`. The neighbor packer accepted `'bisquare'` as code `3` and rejected `'quartic'`. Density explicitly canonicalizes `'bisquare'` to `'quartic'`. Even within space-time density the spatial and temporal axes accept different alias sets.

**Wanted.** Accept `'quartic'` and `'bisquare'` for this kernel shape in density and spatial weights, on both axes where implemented. Share the name resolver and preserve each operator’s normalization constant, bandwidth convention and physical dimensions. **Callers:** none required. **Value/cost:** users can select the same shape across PySAL-style weighting and density estimation without guessing its alias, S.

## Departures by name

The findings name the affected consumers for the proposed behavioral changes. Four further groups matter when applying the convention mechanically.

- **Positions versus geometries.** Of the five producer `points` fields, `GPUPointBuffer` and `GPURasterJoin` take a bare coordinate view. `GPUGeometryValidity`, `GPUBufferSelection` and `GPUPointInPolygonJoin` use a geometry port. The bare-view declarations are `geospatial-analysis/point-buffer/gpu-point-buffer.ts:78` and `raster-analysis/polygon-rasterization/gpu-raster-join.ts:137`. Likewise `GPUSpaceTimeKriging`, `GPULocalRBF`, `GPUNaturalNeighbor` and `GPUConditionalGaussianSimulation` use `queries` for coordinate views; `GPULinearReferencing` and `GPUNeighborSearch` already use `queryPositions`. `GPUNetworkPointRouting.queries` contains integer pairs, and the two raster embedding query APIs take feature vectors. A blanket `queries → queryPositions` replacement would introduce errors.
- **Three quantities called `resolution`.** `GPUCellAggregation`, `GPUCellRollup`, `GPUCellCover` and `GPUPointToCell` take a discrete grid level. `GPUAdjacencyMatrix` takes bins per axis at `gpu-network/adjacency-matrix/gpu-adjacency-matrix.ts:140`. `GPUMultiScaleRelief` takes ground metres per pixel at `gpu-terrain/relief-visualization/gpu-multi-scale-relief.ts:55`. On the host, `getGPUMultiScaleReliefRadii` with feature sizes `4…16` and scaling factor `1` changed from radii `[1,…,8]` to `[0,…,4]` when `resolution` changed from `1` to `2`. These last two are the candidates for `binsPerAxis` and `metersPerPixel`.
- **Opaque operand names.** Exactly two producers expose both `a` and `b`: `GPURasterArithmetic` at `raster-analysis/raster-algebra/gpu-raster-arithmetic.ts:55` and `GPURasterConditional` at `gpu-raster-conditional.ts:63`. They mean `left`/`right` in arithmetic and `whenTrue`/`whenFalse` in conditional selection. Give the conditional branches priority. Reversing them produces a perfectly valid opposite classification.
- **Option casing.** The 17 mixed-case direct literals belong to `GPULocalPermutationTest` (eight statistic names), `GPUGlobalPermutationTest` (three), `GPUSpatialPredicateJoin` (`coveredBy`, `containsProperly`), and the spatial-weights algebra forms (`spaceTime`, `symmetricDifference`, `higherOrder`, `selfWeight`, with a subset on the chunked form). There is even a possessive difference between global `leesL` and local `localLeeL`. The eight uppercase literals are `BJC`, `CLC`, `HC0`, `HC1`, `HC2`, `HC3`, `HAC`, `CLUSTER`. Preserve recognizable reference codes explicitly. For new library-defined multiword options, hyphenation already has 129 precedents.

Class word order alone is a poor migration criterion. Six producers match the explicit leading verbs `Make`, `Normalize`, `Subdivide`, `Pick` or `Polygonize`: `GPUMakeValid`, `GPUNormalizeGeometry`, `GPUNormalizePolygons`, `GPUSubdividePolygons`, `GPUPickRegionMask` and `GPUPolygonize`. `MakeValid` and `Polygonize` are useful GIS search terms. I would spend the pre-alpha rename budget on the missing algorithm names in NAM-14 before imposing noun order on those six.

## Quick wins

- [ ] Resolve inverse-distance spelling once for both neighbor packing and metadata. Test the spread-object example and assert **both** code `1` and transform `'raw'`.
- [ ] Add the grouped `probabilities` input alias. Put the two-departure `[5,90]` example beside the 0–1 / 0–100 conversion.
- [ ] Validate spherical overlay operations, point-buffer coordinate modes and H3 indexing precision before generating nodes. Test the sibling spelling as well as an unrelated invalid token.
- [ ] Accept `how` on key join and `'strict'` on the cell-statistics packer. Test the nondefault behavior so a fallback cannot pass.
- [ ] Add `indexCellSizeHint`, the mode-aware neighbor packer, `'z-score'` and the quartic/bisquare aliases. Keep current numeric layouts.
- [ ] Extend the vocabulary test behind `KNOWN(ERG-104)` with **shared-settings fixtures**. A declaration snapshot alone would miss the type-correct weight and join examples.

## Larger proposals

1. **Rename `'lowest-id'` before alpha.** Changing the algorithm to choose an application ID would also require a policy for absent or duplicate IDs. The deterministic row rule already exists on both backends and the showcase describes it correctly. Rename the selector.
2. **Migrate the three algorithm exports together with documentation and catalog identity.** A permanent second class name would give catalog and audit tooling two public identities for one implementation. Update imports, props, packers, headings and catalog records in one change. Keep serialized scenario IDs stable where they are independent of the export name.
3. **Introduce named raster placement objects alongside the existing tuple packers.** This is the largest shape change worth doing here. Changing only `fieldExtent` to another tuple-typed name would still let the compiler accept `[minX,minY,maxX,maxY]`. Named `origin` and `cellSize` members make that particular mistake visible at the call site.

## Not read

The export and constructor census covered the public package surface. I read the nearest tie-mode declaration, DBSCAN settings packer, point-pattern settings packer and isolines packer in full. I traced selected constructors, metadata builders, CPU twins and generated-branch selections where the census showed a collision. I sampled the API index, cross-reference and family guides from the user’s side first, then checked the candidates against the synthesis, area reports and decision register. For staging-owned findings I inspected the KI copy; several host probes were repeated against both copies.

I did not read every shader, lens, test, catalog description or recipe body. I did not verify external reference packages, run a build, or execute a GPU kernel. GPU predictions above are marked `SOURCE` with a confirming probe. The reported `rg` migration counts are source-text counts in the live checkout and include embedded examples where indicated.