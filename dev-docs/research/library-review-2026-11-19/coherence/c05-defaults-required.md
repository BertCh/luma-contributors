# Defaults and required props — the minimal correct call

## Summary

I counted all 498 public command-node producers and checked 94 minimal constructor expressions against their TypeScript signatures. All 94 type-checked. Several still failed at construction because their required inputs or outputs were declared optional.  
The most consequential default is in `GPUQuantileLISA`. With no variation in either the observations or the permutation distribution, increasing the number of permutations can turn every tested row “significant.” The caller cannot select a conservative tail.  
A shared `significanceLevel: 0.01` can silently become `0.05` in multivariate Geary, and selecting Imhof relief shading can silently produce aspect shading. Both callers have already specified what they want.  
I found 45 producers for which `{}` satisfies the output type despite an explicit runtime requirement to select an output. Conversely, a point-cloud caller must repeat a stride that its input views already determine exactly.  
The most valuable pre-alpha change is to make **one selection determine both the accepted props and the computation**. Start with inference tails, resident parameter packers and algorithm selection. A minimal-call test should check the resolved settings and one discriminating result as well as TypeScript acceptance.

## Census

`S` below means `modules/analysis/src`. `KI` identifies the newer staging-owned copy in `luma-contributors-ki`. Counts are from the R19 working tree during the review. Line numbers can move with concurrent edits.

I used `rg --files modules/analysis/src`, the package export map and a read-only Node probe using the installed TypeScript API. The census selected the exported class behind each family-barrel symbol, kept classes whose instance type had `getCommandNodes`, and inspected the first constructor signature. It excluded graph-operation wrappers that do not themselves implement that method. The essential selection was:

```js
const instance = checker.getDeclaredTypeOfSymbol(symbol);
if (!instance.getProperty('getCommandNodes')) continue;
const signature = checker.getTypeOfSymbolAtLocation(symbol, declaration).getConstructSignatures()[0];
const props = checker.getTypeOfSymbolAtLocation(signature.parameters[0], declaration);
const required = checker.getPropertiesOfType(props)
  .filter(property => !(property.flags & ts.SymbolFlags.Optional));
```

| Public family | Producers | Minimal calls checked | Required top-level count or capacity¹ | Required `parameters` | Optional `parameters` |
|---|---:|---:|---:|---:|---:|
| Spatial analysis | 272 | 15 | 96 | 86 | 18 |
| Raster analysis | 86 | 15 | 49 | 44 | 10 |
| Terrain | 46 | 15 | 4 | 28 | 0 |
| Network | 41 | 15 | 12 | 10 | 4 |
| Dataframe | 34 | 15 | 25 | 22 | 1 |
| Point cloud | 11 | 11 | 4 | 1 | 9 |
| Analysis core | 4 | 4 | 3 | 2 | 0 |
| Geospatial | 2 | 2 | 0 | 1 | 0 |
| Computation record | 2 | 2 | 0 | 0 | 0 |
| **Total** | **498** | **94** | **193** | **194** | **42** |

¹ Required common top-level key matching `/(Count|Capacity)$|^capacity$/`. A count can legitimately describe a model, allocation or logical prefix. The 193 is the population inspected for candidates, not a count of redundant props. `testing` has no public package entry.

| Convention measured | COUNT result |
|---|---|
| Top-level constructor `seed` | **3 required**, **4 optional**. Required on `GPUConditionalGaussianSimulation`, `GPUSpatialBalancedSample`, `GPUContinuousSpatialSample`. Optional on `GPUNetworkLouvain`, `GPUMultivariateLocalGearyPermutationTest`, `GPUSpatialCrossValidation`, `GPUNestedInterpolationValidation`. Other producers can receive seeds through resident parameters. |
| Top-level `maximumPermutations` | **12 required**, **1 optional**. Multivariate Geary is the optional case and uses the value as the actual number of draws. |
| Implicit time or JavaScript randomness | `rg -n 'Math.random\(\|Date.now\(' modules/analysis/src` with the two alternatives searched as a regex returned **0** matches. This measures those two sources of randomness, not device determinism. |
| Public `getGPU*ParameterValues` functions with a non-union, all-optional first `settings` or `options` object | **45** permit zero arguments, **6** require one, **21** require two, **1** requires three. Required later arguments such as a raster grid explain the last two groups. Five of the six one-argument functions successfully packed `{}`. |
| Empty output objects | **45** public producers matched a literal “needs/requires at least one output” guard and accepted `const empty: ConstructorParameters<typeof C>[0]['output'] = {}`. The latter check produced **0** TypeScript diagnostics. This is a lower bound because constructors use other wording and helper validators. |

**Minimal-call inventory.** I selected prominent first-run and operator-page examples, then filled each large family to 15 in ascending `SHA256("2026-11-19/" + className)` order. I took every producer in smaller families. The anchors were the first five names shown for dataframe, network, spatial and terrain, and the first three for raster. In the notation below, `Name(a,b)` means `new GPUName({a,b})`, with each shorthand variable declared at the corresponding public prop type. `output` is the caller-owned, typed output, including its required nested views. These are constructor calls, not allocation or submission examples.

<details>
<summary>All 94 minimal constructor expressions</summary>

```text
dataframe-analysis
ColumnQuantiles(values,parameters,quantileCount,output)
ClassBreaks(values,parameters,maximumClassCount,output)
ColumnProfile(columns,histogramBinCount,output)
GroupStatistics(keys,columns,output)
TimeWindowFilter(timestamps,window,output)
CellPeriodKeys(cells,periods,periodCount,fill,output)
SpaceTimeVariogram(positions,values,times,parameters,gridSize,lagCount,temporalLagCount,output)
PeriodIndex(temporalContext,periods,output)
SpaceTimeKFunction(positions,times,parameters,gridSize,distanceClassCount,timeClassCount,output)
CalendarBuckets(timestamps,parameters,output)
MarkedPointStatistics(positions,parameters,radiusCount,output)
WideNumericScan(id,words,resultWordCount,output,phaseBodies)
TemporalReduction(cellIds,timestamps,values,parameters,cellCount,bucketCount,output)
ScalarScale(values,domain,range,parameters,maximumDomainCount,output)
VariableStandardization(values,method,output)

gpu-network
NetworkLouvain(nodeCount,parameters,output,weights)
NetworkCostMatrix(topology,startNodes,output)
NetworkAccessibility(costs,opportunityWeights,parameters)
NetworkCostWeights(costs,parameters,output)
NetworkReachability(topology,startNodes,output)
NetworkNeighborhood(topology,startNodes,hops,output)
KShortestPaths(topology,sources,targets,routeCount,output)
NetworkTravelTimePercentiles(stateVertices,transitions,costs,vertexCount,initialState,departureCount,percentiles,window,output)
NetworkRationalAccess(costs,demand,supply,parameters,output)
AlternativeRoutes(topology,sources,targets,routeCount,output)
NetworkPointRouting(turns,costs,queries,fractions,output)
NetworkTimeRouting(stateVertices,transitions,costs,endpoints,departure,output)
NetworkNoding(lines,intersectionCapacity,tolerance,output)
NetworkServiceAreas(topology,facilities,output)
NetworkKFunction(positions,topology,maximumDistance,networkLength,parameters,output)

gpu-spatial-analysis
PointDensity(positions,bounds,gridSize,output)
QuantileLISA(values,weights,quantileParameters,parameters,maximumPermutations,output)
SpaceTimeHawkes(positions,times,parameters,output)
OrdinaryLeastSquares(predictors,response,predictorCount,output)
PointPatternEnvelope(observed,simulations,reference,integrationWeights,simulationCount,nullModel,fittedParameters,output)
RasterFalseDiscoveryRateMask(cellCount,pValues,parameters,output)
Segregation(groupCount,groupCounts,output)
LocalCollinearity(weights,predictors,predictorCount,output)
LocalFrameProjection(frame,positions,output,bounds)
TrajectoryTrackRelativeTimes(timestamps,trackOffsets,output)
MinimumBounds(output,positions,labels,groupCount)
SpatialRegressionDiagnostics(weights,predictors,response,residuals,predictorCount,output)
CoverageValidity(positions,featureOffsets,polygonOffsets,ringOffsets,parameters,output)
MinimumClearance(geometry,output)
CellGeometry(family,cells,output)

gpu-terrain
TerrainFlow(grid,elevation,parameters,output)
TerrainHeightAboveDrainage(grid,elevation,streams,output)
TerrainDerivatives(grid,elevation,parameters,output)
TerrainContours(grid,elevation,levels,output)
TerrainViewshed(grid,elevation,parameters,output)
TerrainRuggedness(grid,elevation,output)
TerrainFlowFractions(grid,elevation,flowDirections,parameters,routing,output)
TerrainStreamNetwork(grid,flowDirections,streams,output)
TerrainSpikeRepair(grid,elevation,output)
TerrainBreach(grid,elevation,pitCells,maximumSteps,maximumCutDepth,elevationDecrement,output)
TerrainLineOfSight(grid,elevation,pairs,parameters,output)
PointHorizonProfile(maximumDistance,cellSize,grid,elevation,observers,parameters,output)
TerrainBranchingFlow(grid,fractions,values,mode,output)
TerrainWatersheds(grid,flowDirections,output)
TerrainFlowField(grid,elevation,parameters,output)

raster-analysis
DistanceField(grid,parameters,output)
ParticleAdvection(velocities,fieldWidth,fieldHeight,parameters,uint32Parameters,state)
RasterZonalStatistics(grid,zones,values,zoneCapacity,output)
RasterEmbeddingSimilarity(pixelCount,dimensions,vectors,queryCount,queries,output)
RasterClimateRunAccumulator(cellCount,chunkSliceCount,values,parameters,state)
Streamlines(velocities,fieldWidth,fieldHeight,gridWidth,gridHeight,seedColumns,seedRows,stepsPerDirection,parameters,uint32Parameters,output)
RasterConditional(cellCount,parameters,output)
RasterEmbeddingZoneMean(pixelCount,dimensions,zoneCount,vectors,output)
PolygonRasterization(grid,parameters,positions,featureOffsets,polygonOffsets,ringOffsets,crossingCapacity,output)
ScalarFieldRidges(input,grid,parameters,output)
RasterCOLDMonitor(cellCount,sliceCount,bandCount,values,parameters,dateAxis,output)
RasterReconstruction(grid,marker,mask,output)
RasterContrastEqualization(grid,input,parameters,output)
RasterFillNoData(grid,input,maximumDistance,output)
RasterEmbeddingProjection(pixelCount,dimensions,vectors,componentCount,mean,components,output)

pointcloud-analysis
PointCloudVoxelMerge(partitionPolicy,parts,output)
PointCloudICP(source,target,maximumDistance,output)
PointCloudChange(reference,comparison,cores,normals,alignment,radius,halfDepth,output)
PointCloudSMRF(positions,classifications,grid,output)
PointCloudNeighborhoods(positions,radius,capacity,output)
PointCloudSurfaces(positions,classifications,grid,output)
PointCloudFeatures(positions,indices,counts,capacity,output)
PointCloudNormalScale(positions,radii,capacity,parameters,output)
PointCloudGround(positions,classifications,grid,minimumSurface,coverage,radii,initialThreshold,slope,maximumThreshold,output)
PointCloudStatisticalFilter(squaredDistances,counts,capacity,output)
PointCloudHeight(positions,grid,terrain,coverage,output)

analysis-core
FittingMinibatchOrder(rowCount,parameters,output)
FittingCoordinateEncoder(rowCount,coordinates,output,encoding,maximumDegree)
AdamStep(gradients,loss,parameters,maximumIterations,output)
FittingGradientSums(rowCount,parameterCount,gradients,output)

geospatial-analysis
DatumGridTransform(positions,samples,mask,grids,operation,output)
PointBuffer(points,parameters,offsets,verticesPerRing,coordinateSystem)

computation-record
TransferMask(id,source,relation,target)
SnapshotTap(id,arena,views)
```

I checked the dense `weights` branch of Louvain, the labelled branch of minimum bounds, and `encoding: 'spherical-harmonics'` for the coordinate encoder. An in-memory TypeScript source containing the 94 expressions returned `calls 94 diagnostics []`. Runtime probes below use actual graph views on a null device or execute CPU functions directly.

</details>

## Proposed convention

**The minimal typed call for a selected mode must either compute that mode or identify an unsatisfied requirement before graph construction.** Derive counts from exact view shapes, require scientific choices that change the interpretation of an answer, and give CPU and GPU forms the same resolved algorithm settings unless the caller explicitly selects another profile. Check this with one omitted-options fixture per mode, comparing the resolved settings, packed words, selected graph nodes and a discriminating CPU or device result; generate the docs’ default, unit, range and mode applicability from those same settings.

## Findings

| ID | Kind | Reach | Change | Cost | Where | One line |
|---|---|---|---|---|---|---|
| DEF-01 | W | ORDINARY | BREAKING | M | Quantile LISA | Permutation count can determine significance on a constant column. |
| DEF-02 | W | ORDINARY | ADDITIVE | S | Multivariate local Geary | A shared 1% significance setting silently packs 5%. |
| DEF-03 | W | ORDINARY | BREAKING | M | Relief shading | A packed request for Imhof swing selects aspect shading in the default graph. |
| DEF-04 | W | ORDINARY | ADDITIVE | S | Neighbor search | Omitting a radius packs a value that invalidates radius search. |
| DEF-05 | W | ORDINARY | ADDITIVE | S | Analysis default resolver | A clustering-radius decision made from eight sampled points reports `sampled: false`. |
| DEF-06 | C | ORDINARY | ADDITIVE | S | Class breaks | CPU and GPU natural-break defaults can put the same observation in different classes. |
| DEF-07 | S | ORDINARY | BREAKING | S | Class breaks | A two-class equal-interval request fails because the default also compiles six-class box plots. |
| DEF-08 | C | ORDINARY | BREAKING | S | Network accessibility and cost weights | Selecting exponential or power decay while omitting its rate produces constant weights. |
| DEF-09 | E | ORDINARY | ADDITIVE | M | 45 producers | The public output type accepts a selection that the constructor rejects. |
| DEF-10 | E | ORDINARY | ADDITIVE | M | Geometry, raster, network and kriging | Required choices among input sources are invisible in the minimal type. |
| DEF-11 | E | ORDINARY | ADDITIVE | S | Point cloud and OLS | The caller repeats a stride already fixed by exact input lengths. |
| DEF-12 | S | ORDINARY | ADDITIVE | M | Point-pattern envelope | An unused integration weight can invalidate the default statistic. |
| DEF-13 | C | ORDINARY | BREAKING | S | Space-time Hawkes | Changing backend silently changes the stopping budget tenfold. |
| DEF-14 | E | ORDINARY | ADDITIVE | S | Spatial sampling | Two samplers require a scalar seed even when resident words wholly determine the draw. |
| DEF-15 | E | ORDINARY | ADDITIVE | S | Five parameter packers | A mandatory `{}` carries no information. |
| DEF-16 | S | ORDINARY | ADDITIVE | S | Vector ruggedness | Two publicly accepted border modes always throw. |

### DEF-01 — Make the Quantile LISA inference rule an explicit choice

**Where and evidence.** `S/gpu-spatial-analysis/spatial-autocorrelation/gpu-quantile-lisa.ts:390`, KI `:431`; `S/analysis-core/contracts/inference.ts:9`; `modules/kepler/src/analysis/weighted-quantile-lisa.ts:72,158`. Both copies of `docs/api/gpu-spatial-analysis/quantile-lisa.md` correctly describe the legacy rule. The new issue is the absence of a safe constructor choice and its downstream effect on both point and polygon Kepler recipes.

**Now.** Take four equal values on a four-node cycle, choose the final of two quantile classes, and use `significanceLevel: 0.05`. Every row has two selected neighbors. **EXECUTED** enumeration of the ordered conditional draws for one focus row gave `[2,2,2,2,2,2]`, all equal to the observation. **SOURCE** the finalizer computes `(min(upper, P-upper)+1)/(P+1)`. At `P=9` this is `0.1`; at `P=99` it is `0.01`. The same map crosses the significance threshold solely because the caller spent more on identical draws. Both inclusive upper-tail and conservative inclusive two-sided inference give `1` here.

**Wanted.** Require `alternative: 'greater' | 'two-sided' | 'geoda-legacy'`. Use the existing inclusive tail machinery for the first two choices. Record the choice in Kepler recipe provenance and show it alongside the p-value. **GPU probe:** submit that cycle twice at 9 and 99 draws; verify the conservative choices give four p-values of `1`, and the explicitly legacy choice reproduces the old values.

**In-repository callers and value.** `rg` followed by a TypeScript call-expression count found **29 executable `new GPUQuantileLISA(...)` sites**: 24 analysis, 2 lenses, 1 Kepler, 1 showcase and 1 website runner. Shared test-props builders cover several sites with one edit. A separate website catalog string also shows the old call. Cost M gives map authors a defensible inference choice while preserving intentional GeoDa comparisons.

### DEF-02 — Multivariate Geary silently discards the common significance setting

**Where and evidence.** `S/gpu-spatial-analysis/spatial-autocorrelation/multivariate-geary-parameters.ts:16,30,52`; `gpu-multivariate-local-geary-permutation-test.ts:71,325`, KI `:84,419`. Compare `permutation-inference/permutation-parameters.ts:36,65`. I ran the Geary packer from **both** copies. KI fixes float32 endpoint validation; the naming and fallback persist. This adds a settings-reuse failure to `KNOWN(STD-05)`.

```ts
const settings = {seed: [1, 0] as const, significanceLevel: 0.01};
const words = getGPUMultivariateLocalGearyPermutationTestParameterValues(settings);
```

**Now.** The variable shares the valid `seed` property, so structural assignment permits it. **EXECUTED:** decoding word 2 gave `0.05000000074505806`. Replacing `significanceLevel` with `significance` gave `0.009999999776482582`. Fresh-literal excess-property checks do not protect a settings panel or a spread settings object. A second surprise is that `maximumPermutations` defaults to 999 and is the **actual** number of draws. The three resident words contain no active count.

**Wanted.** Accept `significanceLevel`, deprecate `significance`, and reject conflicting values. Accept a nonnegative safe-integer `seed` alongside the full two-word representation, preserving all 64-bit keys. Expose `permutationCount` as the preferred name for this fixed draw count. Document the default together as “999 draws, seed words `[0,0]`, significance level 0.05, inclusive equal tails.”

**In-repository callers and value.** No existing callers need edits with aliases. Cost S prevents a fivefold threshold error when reusing settings.

### DEF-03 — Honor an Imhof shading request in the default graph

**Where and evidence.** `S/gpu-terrain/terrain-illumination/gpu-relief-shading.ts:227,238,341,609,634`; `docs/api/gpu-terrain/operators.md:1240`, also checked in KI.

```ts
const settings = getGPUReliefShadingParameterValues({lightWeighting: 'imhof-swing'}, grid);
new GPUReliefShading({grid, elevation, parameters, output: {hillshade}});
```

**Now.** **EXECUTED:** the packer produced weighting code `2` and swing `65°`. A 3×3 null-device graph with the default constructor contained `gradient-x`, `gradient-y`, `hillshade`. Adding `imhofSwing: true` produced a fourth node, `hillshade-swing`. **SOURCE:** the base shader interprets any nonzero weighting code as aspect weighting. The detailed docs mention the second switch; the packer accepts the complete-looking request without it.

**Wanted.** Have the ordinary hillshade graph implement every public weighting code, preferably with a uniform branch in the existing pass. Retain explicit specialization only if a request outside the specialization gives a visible invalid result. **GPU probe:** use a tilted 3×3 plane, one oblique light and `65°`; compare the center at code 2 with the Imhof oracle and with code 1. Flat terrain cannot distinguish them.

**In-repository callers and value.** **26** executable constructor sites: 15 omit the switch in plain object literals, 8 set it true, 1 passes shorthand and 2 forward props. Zero syntax edits are required by the proposed default. The 15 omitted-switch sites are the direct behavior-audit set, and the graph-node snapshots need review. Cost M makes the packed style authoritative.

### DEF-04 — Give radius search a mode-aware parameter packer

**Where and evidence.** `S/gpu-spatial-analysis/neighbor-search/neighbor-search-parameters.ts:124,223`, `neighbor-search-kernels.ts:86,107,223`, `gpu-neighbor-search.ts:157,243`. KI retains the constructor mode at `:157`. The generated radius row in `docs/api/gpu-spatial-analysis/operators-weights.md:481`, KI `:527`, labels it optional and explains the two meanings in prose.

**Now.** **EXECUTED:** `getGPUNeighborSearchParameterValues({bounds:[0,0,2,2]})` packed `[0,0,2,2,Infinity,0,1,≈1e-6,1,0,0,0]`. `Infinity` is a useful unrestricted kNN cutoff. **SOURCE:** the radius lattice requires a finite positive radius, so the same omitted value disqualifies the lattice before any distance comparison. `invalidCount` is optional.

**Wanted.** Add a mode-aware overload with `{mode:'radius', bounds, radius}` or `{mode:'knn', bounds, radius?}` and use it in the first examples. Require a positive finite packed radius in the radius branch. **GPU probe:** self-join `[(0,0),(1,0)]` with space for both nonself slots and read offsets and `invalidCount`, first with the current omitted radius, then with radius `1.1`.

**In-repository callers and value.** No migration is needed to add the overload. Cost S moves an empty-result diagnosis from readback to the parameter call.

### DEF-05 — Preserve sampling provenance when a default rule declines to suggest a radius

**Where and evidence.** `S/gpu-spatial-analysis/ontology/analysis-experience/analysis-default-rules.ts:720,738,748`; `analysis-experience-types.ts:329`. Two rejection branches hard-code `sampled: false`. The successful branch propagates `spacing.sampled`.

**Now.** **EXECUTED:** on the 100 points of a 10×10 unit lattice, `computeGPUAnalysisSpacingStatistics(points, {coordinateModel:'planar', neighborRank:3, maximumSampleSize:8})` returned `sampled:true, sampleSize:8`. `resolveGPUAnalysisDefault('k-distance-knee', profile)` then returned `value:undefined, sampled:false`, explaining a measured bend of `0.387`. The full 100 points produced a bend of `0.96` and failed the separate range-to-median test. The fallback can be the right outcome while its reported evidence is wrong.

**Wanted.** Once the resolver has inspected a sampled curve, propagate `spacing.sampled` on every return and include “8 of 100 points” in the fallback explanation. Cover both the bend and the range-to-median rejection paths.

**In-repository callers and value.** No signature changes. Cost S lets a GIS interface distinguish a weak sample from a full-data diagnostic.

### DEF-06 — Identify which natural-breaks algorithm produced the breaks

**Where and evidence.** `S/dataframe-analysis/column-classification/gpu-class-breaks.ts:148,189,200`, `cpu-class-breaks-operation.ts:117,128`; `modules/analysis/test/gpu-dataframe/column-classification/natural-breaks-mirror.ts:322`. `KNOWN(DPC-120)` concerns the class cap imposed by the bins. This case uses four classes and 2,048 bins.

**Now.** **EXECUTED** the public CPU twin and the repository’s host mirror of the default GPU algorithm on `[174,15,79,177,157,114,36,56,179,166,104,100000]`, with `methods:['natural-breaks']`, `maximumClassCount:4`, and four requested classes. The CPU inner edges were **`[79,157,100000]`**. The histogram/refinement mirror gave **`[104,157,100000]`**. The observation `79` changes class. Neither output array identifies the algorithm. The CPU twin computes exhaustive Fisher–Jenks even when the GPU form uses its default approximation.

**Wanted.** Expose the resolved `naturalBreaksMode` and backend algorithm in result/provenance metadata. Put the above class-membership example alongside the first natural-breaks call. **GPU probe:** submit the same twelve values in default and explicit exact modes and read the three inner edges. The mirror result is host evidence, not a device observation.

**In-repository callers and value.** No existing callers need change for additive metadata. Cost S lets a map legend or saved analysis explain a changed class after a backend switch.

### DEF-07 — Derive the default class-break method set from the declared capacity

**Where and evidence.** `S/dataframe-analysis/column-classification/gpu-class-breaks.ts:129,272,282,307`, `cpu-class-breaks-operation.ts:49,57`. The API example says omitted `methods` means every method in both `docs/api/gpu-dataframe-analysis.md:346` and KI.

**Now.** **EXECUTED:** for values `[0,2,4]` I packed `{method:'equal-interval',classCount:2}` and supplied `maximumClassCount:2`. The GPU constructor threw `class-breaks box-plot needs maximumClassCount of at least 6`. The CPU twin threw the equivalent error. Adding `methods:['equal-interval']` constructed successfully and the CPU returned `[0,2,4]`. The caller had already chosen a valid method and sized its buffers correctly.

**Wanted.** Omitted `methods` should compile every method compatible with the declared capacity and layout. Exclude box plot below six classes and packed-only methods for incompatible chunked input. Preserve strict rejection for an explicitly requested incompatible method. Expose the resulting method list.

**In-repository callers and value.** **88** direct `GPUClassBreaks` constructions, **9** GPU operation constructions and **9** CPU operation constructions were found. Zero successful calls require a source edit: the recommended change expands previously rejected combinations. It changes the documented default, so I grade it BREAKING. Cost S makes binary, tercile and quartile classification usable with the obvious call.

### DEF-08 — Require the rate when the caller selects a rate-dependent decay

**Where and evidence.** `S/gpu-network/network-accessibility/gpu-network-accessibility.ts:74,87`, `network-accessibility-passes.ts:34,37`; `S/gpu-network/network-cost-weights/gpu-network-cost-weights.ts:86,105,435`. Both R19 and KI network docs state that `beta` defaults to zero.

**Now.** **EXECUTED:** `{threshold:10, decay:'exponential'}` and `{costLimit:10, decay:'exponential'}` both packed `[10,1,0,1]`. Selecting `'power'` packed `[10,2,0,1]`. **SOURCE:** at positive costs `1` and `9`, both selected formulas give weights `[1,1]`. The zero rate is a valid intentional setting, but selecting a distance-decay function without supplying a rate is indistinguishable from requesting equal weights.

**Wanted.** Use a discriminated settings union. Require a finite nonnegative `beta` for `'exponential'` and `'power'`. Keep `{threshold}` and `{costLimit}` as valid binary catchments. An explicit `beta:0` remains valid. **GPU probe:** read the two neighbor weights for costs `[1,9]` with omitted and explicit `beta:1`.

**In-repository callers and value.** Of **51** direct packer calls, **2 inline literals** select a rate-dependent mode without `beta`, both in Node tests (`gpu-network-accessibility.node.spec.ts:325`, `gpu-network-cost-weights.node.spec.ts:90`). Six calls forward settings variables and require type review, including the explorer and showcase settings builders. Cost S prevents a real modeling choice from silently becoming a no-op.

### DEF-09 — Express “select at least one output” in the output type

**Where and evidence.** See the 45-class inventory below. Representative sites are `S/gpu-terrain/terrain-analysis/gpu-terrain-derivatives.ts:106,198` and `S/dataframe-analysis/column-classification/gpu-color-scale.ts:175`, KI `:193`.

**Now.** **COUNT:** `{}` was assignable to all 45 output types. **EXECUTED:** the first-run 3×3 terrain grid, valid elevation band and eight-word parameter view, with `new GPUTerrainDerivatives({grid,elevation,parameters,output:{}})`, threw `terrain-derivatives requires at least one output`. `output:{slope}` is the concise useful call.

**Wanted.** Encode at least one of the **actual terminal choices** in each public output type. Preserve the runtime guard for JavaScript. Where a nested object supplies the choice, encode its required leaf too; an empty `curvatures` object should not qualify as a requested curvature. Add one positive minimal example per output group.

**In-repository callers and value.** No valid call needs a new output. Negative fixtures that deliberately exercise the runtime guard should mark the expected type error. Cost M across the 45 declarations gives authors a usable completion list and catches the mistake before allocating buffers.

### DEF-10 — Encode required alternative inputs where the constructor already enforces them

**Where and evidence.** The following are separate, executed counterexamples; each declaration accepts the minimal call shown.

| Public API | Typed call with valid indicated resources | Observed constructor or packer requirement |
|---|---|---|
| `GPUDistanceField` | `new GPUDistanceField({grid,parameters,output})` | `requires seedPositions or seedMask` at `S/raster-analysis/distance-field/gpu-distance-field.ts:193` |
| `GPUHilbertKeys` | `new GPUHilbertKeys({output:{keys}})` | `needs positions, or minima and maxima` at `S/gpu-spatial-analysis/hilbert-keys/gpu-hilbert-keys.ts:172` |
| `GPUGeometryValidity` | `new GPUGeometryValidity({output:{issueCode}})` | `requires exactly one of polygons, lines and points` at `S/gpu-spatial-analysis/geometry-validity/gpu-geometry-validity.ts:186`, KI `:221` |
| `GPUGaussianRandomField` | `new GPUGaussianRandomField({parameters,output})` | `grid or positions must be supplied exclusively` at `S/gpu-spatial-analysis/spatial-interpolation/gpu-gaussian-random-field.ts:169` |
| `GPUNetworkAccessibility` | `new GPUNetworkAccessibility({costs,opportunityWeights,parameters})` | `needs at least one of cumulative, gravity, or catchment` at `S/gpu-network/network-accessibility/gpu-network-accessibility.ts:306` |
| `getGPULocalKrigingParameterValues` | `getGPULocalKrigingParameterValues({})` | Requires a covariance structure at `S/gpu-spatial-analysis/spatial-interpolation/prediction-covariance.ts:218,263`. The runtime message mentions `coregionalizationModels` even for an ordinary univariate caller. |

**Wanted.** Use the same union pattern already present in `GPUNetworkLouvainProps`. Preserve “one **or both**” for distance-field seed inputs and “exactly one” for the random-field domain. For kriging, make `{covariance}` the visible minimal univariate branch and retain the separate multi-structure branch. Say `supply covariance or coregionalizationModels` on omission.

**In-repository callers and value.** No valid source combination changes. The explicit invalid kriging fixture at `covariance-parameters.node.spec.ts:95` should assert the type error as well. Cost M makes six first-use requirements discoverable without running the program.

### DEF-11 — Infer a stride when two exact view lengths determine it

**Where and evidence.** `S/gpu-spatial-analysis/spatial-regression/gpu-ordinary-least-squares.ts:219,233`, KI `:277`; `S/pointcloud-analysis/operations/gpu-pointcloud-features.ts:138,142`, `gpu-pointcloud-statistical-filter.ts:132,140`, `gpu-pointcloud-neighborhoods.ts:259,268`, also `:268` in KI. The first-run guide passes `capacity:1` separately from the neighborhood views.

**Now.** **EXECUTED:** three responses and six predictor scalars gave a constructor error with omitted `predictorCount`, a length error with `1`, and a three-row OLS contributor with `2`. Three point-cloud counts and six squared distances likewise failed with omitted capacity or `1`, and constructed with `2`. In either example the caller is asked to supply the only integer the constructor will accept.

**Wanted.** Default OLS `predictorCount` to `predictors.length / response.length`. Infer point-cloud `capacity` from `indices.length / counts.length`, `squaredDistances.length / counts.length`, or the corresponding exact output slots per query in `GPUPointCloudNeighborhoods`. Validate integral divisibility and the existing named limits once. Preserve an explicit count for a zero-row or otherwise ambiguous layout. A padded output that merely has to be large enough does **not** justify this inference.

**In-repository callers and value.** None must change. Cost S removes duplicated shape information from four producers. The 23 public producers requiring `predictorCount` are candidates for further inspection, not 23 verified instances of this defect.

### DEF-12 — Require envelope inputs only in the statistics that use them

**Where and evidence.** `S/gpu-spatial-analysis/point-process/gpu-point-pattern-envelope.ts:39,100,125,214`; `cpu-point-pattern-envelope-operation.ts:103,159,173`. I read both newer KI forms and executed the CPU twin from both trees. `docs/api/gpu-spatial-analysis/point-process.md:72,78`, KI `:84`, acknowledges the extreme-rank placeholders.

**Now.** For `observed:[3]`, `simulations:[1,2]`, `reference:[0]`, two replicates, and a fixed null, the default maximum-deviation statistic with `integrationWeights:[1]` gave `status=0`, statistic `3`, p-value `1/3`. Changing only the unused weight to `0` gave **`status=1`, statistic `NaN`, p-value `NaN`**. For extreme rank, the same zero weight gave `status=0`, statistic `1`, p-value `2/3`, yet the caller still has to allocate both unused vectors. These are **EXECUTED** CPU results.

**Wanted.** Maximum deviation requires a reference. Integrated squared deviation requires a reference and positive integration weights. Extreme rank requires neither. Infer `simulationCount` from `simulations.length / observed.length` for exact, nonempty views. Preserve `nullModel` and `fittedParameters`, which carry scientific information the views cannot supply. **GPU probe:** reuse this one-ordinate fixture and compare maximum deviation with the weight omitted, zero and one.

**In-repository callers and value.** Existing valid calls continue to work. Cost M across the contributor, operation and twin removes two allocations in rank mode and prevents irrelevant data from invalidating a valid maximum-deviation analysis.

### DEF-13 — Use one default Hawkes iteration budget on both backends

**Where and evidence.** `S/gpu-spatial-analysis/space-time-hawkes/gpu-space-time-hawkes.ts:76,251` uses `100`; `cpu-space-time-hawkes-operation.ts:25,83,286` uses `1000`. The difference is documented at `docs/api/gpu-spatial-analysis/space-time-hawkes.md:131`. It is distinct from `KNOWN(PTS-11)`, which concerned the iterate used to report likelihood.

**Now.** **EXECUTED:** eight events at times `0…7`, positions `[2,4,2,5,3,5,4,3,1,3,0,1,4,3,7,4]`, parameter words `[30,8,8,15,1e-5,.05,.5,1,2]`. With no options the CPU twin converged on update **128**. With `{maximumIterations:100}` the identical CPU arithmetic stopped at **100**, unconverged. Fitted log likelihood rounded to `-35.2095795` in both cases. A comparison of the scalar fit alone misses the difference.

**Wanted.** Share a default of `100` for both execution forms. Keep `1000` and larger supported CPU budgets available explicitly. Log the resolved budget beside `iterationCount`. **GPU probe:** run the eight events with default and explicit 100 and check that their status and estimates agree on the same device. Float32 need not converge on the CPU’s 128th update.

**In-repository callers and value.** **7 of 18** direct CPU function calls omit the options argument, all in `cpu-space-time-hawkes-operation.node.spec.ts`; two are intentionally invalid inputs. Of **9** CPU-operation constructor sites, 4 give an explicit literal budget and 5 forward props. No syntax migration is required. Any caller relying on 1,000 updates should pin it. Cost S makes a CPU reproduction use the same stopping rule.

### DEF-14 — A resident random seed should satisfy the seed requirement

**Where and evidence.** `S/gpu-spatial-analysis/point-process/gpu-spatial-balanced-sample.ts:62,148,187,281`; `gpu-continuous-spatial-sample.ts:80,158,342`; `cpu-continuous-spatial-sample-operation.ts:78,143`. The staging-owned continuous GPU operation still forwards both sources in KI at `gpu-continuous-spatial-sample-operation.ts:91,96`.

**Now.** **EXECUTED:** on a line from `(0,0)` to `(2,0)`, spacing 1, capacity 2, resident words `[17,0]`, the continuous CPU twin rejected an omitted scalar seed. Scalar seeds `3` and `9` both returned positions `[0.93040454,0,1.93040454,0]` and status `[2,2,2,0]`. The balanced GPU constructor likewise rejected omission with a resident view. With scalar `3` it published both `seed:3` and `parameterSource:'resident-view'` in provenance.

**Wanted.** Permit `{parameters}` or `{seed, replicate?}`. On the resident branch, describe the seed source as resident and let the caller record the uploaded words when replay is needed. Keep the scalar mandatory for these sampling designs when there is no resident seed. A mandatory **effective** replay seed does not require two seed inputs.

**In-repository callers and value.** None must change. Cost S removes a dummy argument and prevents a saved analysis from recording an inactive seed as the one that generated the sample.

### DEF-15 — Let genuinely defaultable packers be called with zero arguments

**Where and evidence.** **COUNT:** 45 matching packers already allow this. Of the six requiring a single all-optional object, five successfully packed `{}`. The sixth is the missing kriging union in DEF-10.

| Packer | Source under `S` | EXECUTED result for `{}` |
|---|---|---|
| `getGPULineChunkParameterValues` | `gpu-spatial-analysis/line-segmentize/gpu-line-chunk.ts:78` | `[0,0,3.39999995e38,0]` |
| `getGPULineLocateParameterValues` | `gpu-spatial-analysis/linear-referencing/gpu-line-locate.ts:81` | `[1,0,0,0]` |
| `getGPUTerrainViewshedToleranceParameterValues` | `gpu-terrain/terrain-analysis/gpu-terrain-viewshed.ts:189` | `[0,0,0,0]` |
| `getGPULineIntegralConvolutionUint32ParameterValues` | `raster-analysis/flow-texture/line-integral-convolution-parameters.ts:98` | `[0,0,0,0]` |
| `getGPUStreamlinesUint32ParameterValues` | `raster-analysis/flow-texture/streamlines-parameters.ts:99` | `[0,2,0,0]` |

**Now and wanted.** Calling each with zero arguments in JavaScript threw while reading a property of `undefined`. Give the five functions `settings: Settings = {}` and assert byte-for-byte equality between `f()` and `f({})`. Keep the separate grid or topology argument wherever it carries information.

**In-repository callers and value.** None. Cost S makes the no-options convention predictable for geometry, terrain and raster users.

### DEF-16 — Advertise only the vector-ruggedness border modes the constructor accepts

**Where and evidence.** `S/gpu-terrain/topographic-position/gpu-terrain-vector-ruggedness.ts:157,159,220`; the published dependency defines four `GPURasterBorderMode` members at `node_modules/@luma.gl/graph/src/gpu-raster/gpu-raster-neighborhood.ts:24`. `GPUTerrainDerivatives` and `GPUReliefShading` expose the broader vocabulary.

**Now.** **EXECUTED** on the same 3×3 band with an eight-word parameter view. `borderMode:'clamp'` and `'nodata'` constructed. Both `'reflect'` and `'constant'` threw `terrain-vector-ruggedness borderMode must be clamp or nodata`. All four are accepted by the public prop type. The existing prose correctly describes the two supported modes.

**Wanted.** Declare `borderMode?: 'clamp' | 'nodata'`, default `'clamp'`. Keep the runtime guard for JavaScript. This tightens the type for two calls that already always fail, and preserves the separately justified edge policy of scalar ruggedness.

**In-repository callers and value.** **15** direct vector-ruggedness construction sites to check for unnecessarily wide annotations; zero successful mode literals change. Cost S makes a shared terrain settings panel surface the real choices.

## Departures by name

**Empty output selection, DEF-09.** The following are all 45 matches of the measured guard pattern. Each name is the public class with the `GPU` prefix omitted. The number is the guard line in its implementation file, which can be located uniquely with `rg -l 'export class GPU<ClassName>\b' modules/analysis/src`. A slash gives the corresponding checked KI line for a staging-owned implementation.

| Family | Public classes and guard lines |
|---|---|
| Dataframe, 4 | `CalendarBuckets:327`, `ColorScale:175/193`, `BivariateClassification:142/160`, `Inequality:273` |
| Spatial, 9 | `CellMeasures:243`, `GeographicDistribution:422/426`, `GroupGeometry:247`, `ShapeDescriptors:325`, `FocalStatistics:148`, `TrajectoryMetrics:363`, `TrajectoryPlayhead:278`, `MobilityIndicators:319`, `ZonalStatistics:587` |
| Terrain, 25 | `Geomorphons:233`, `TerrainFlow:554/576`, `TerrainHeightAboveDrainage:230`, `TerrainHydrologicIndices:171`, `PointHorizonProfile:169`, `SimpleLocalRelief:173`, `MultiScaleRelief:268`, `LocalDominance:252`, `ReliefBlend:236`, `TerrainDerivatives:198`, `TerrainViewshed:356`, `TerrainCurvature:305`, `TerrainSummits:249`, `ProfilePeaks:378`, `SolarPosition:126`, `TerrainHorizon:503`, `TerrainCastShadow:319`, `SolarIrradiance:355`, `SolarShadowMask:189`, `ReliefShading:399`, `TextureShading:275`, `TerrainRuggedness:105`, `TerrainVectorRuggedness:196`, `TerrainTopographicPosition:197`, `TerrainWeissLandforms:290` |
| Raster, 7 | `ChangeDetection:253`, `RasterJoin:242`, `RasterReclassify:127`, `RasterCellStatistics:151`, `RasterExtremaPyramid:371`, `RasterStretch:219`, `RasterZonalStatistics:546` |

**Permutation count vocabulary.** The required `maximumPermutations` constructors are `GPULocalPermutationTest`, `GPUGlobalPermutationTest`, `GPUQuantileLISA`, `GPUSpatialScanStatistic`, `GPUSpaceTimeLocalMoran`, `GPUKnoxTest`, `GPUNearRepeatKnox`, `GPULocalKnox`, `GPUJacquezTest`, `GPUMantelTest`, `GPULOSHResidualPermutationTest` and `GPUGeographicallyWeightedRegressionNonstationarityTest`. The verified naming departure is `GPUMultivariateLocalGearyPermutationTest`: its optional maximum is also its requested draw count. No change to the twelve is implied.

**Other departures.** DEF-10 names each of the six executed alternative-input failures. DEF-11 names the four verified shape-derived counts. DEF-14 names both constructors requiring an inactive seed. DEF-15 lists all five redundant settings arguments in the measured packer population. The remaining findings identify individual producers or the two sibling decay packers at their implementation sites.

## Quick wins

- [ ] Accept `significanceLevel` in both Geary entry paths, verify the packed float is 0.01 for a shared settings variable, and document its fixed 999-draw default.
- [ ] Add the mode-aware neighbor-search packer and make the radius example use it.
- [ ] Propagate sampled provenance on both knee-rejection branches. Pin the 8-of-100 lattice case.
- [ ] Publish the resolved natural-breaks algorithm and the twelve-value class-membership example.
- [ ] Infer exact point-cloud and OLS strides, retaining explicit overrides and existing length checks.
- [ ] Allow a resident seed to satisfy the two sampling constructors and the continuous CPU twin.
- [ ] Default the five empty settings arguments to `{}` and narrow the vector-ruggedness border union.

## Larger proposals

1. **Require an explicit Quantile LISA tail before alpha.** A documentation-only change costs less, but the documentation already explains the all-ties failure. A required choice reaches saved recipes and executable callers. Preserve the old rule by name.
2. **Make selected algorithms executable with their minimal inputs.** Start with the Imhof switch, the compatible class-break method set and the three envelope alternatives. For resident controls that can change after compilation, either compile the selectable branch or make unsupported selection externally visible. Silently substituting a different statistic or rendering is the costly behavior.
3. **Move the existing runtime choices into the public types.** Start with the six verified input alternatives and the 25 terrain output types. Retain runtime checks. Test one valid minimal call, one impossible selection and one meaningful result per mode. This can land family by family.
4. **Separate numerical reference profiles from ordinary backend defaults.** Use the same Hawkes stopping budget for ordinary CPU and GPU execution. Keep a higher explicitly selected CPU budget for reference work. For natural breaks, record the algorithm actually run because equal work budgets alone cannot reconcile the two algorithms.

## Not read

I read the first-run docs and selected family reference sections before constructing the examples. I inspected the public signatures of all producers, type-checked the 94 calls above in memory, read the small inference and settings packers in full where cited, and followed the relevant constructor, CPU and shader branches for the findings. I checked the synthesis, area reports, API/catalog report, direction report, decision list and the relevant staging-owned copies. The differently validated float32 significance endpoints already fixed in KI are excluded from the findings.

I did not read all 498 implementations, every nested output layout, every recipe setter or every catalog entry. The 45-output count is deliberately limited to a reproducible guard pattern. The host mirror is identified separately from the public CPU twin. I ran no GPU computations, browser, build, test suite or network request, and wrote no files.