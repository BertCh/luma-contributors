# First run from install to a number

## Summary

All five tasks can be expressed with public imports. I constructed and added all five programs to real `GPUCommandGraph` instances on a null device. I did not compile shaders, submit work, or observe GPU output.

The most consequential first-run error is in the local Moran example. Its `pValues` buffer belongs to Gi*, while the final writer of its `zScores` buffer is Moran. Both numbers look plausible. The next two are GIS interpretation errors. Searching forward from a facility answers the opposite question on a one-way road, and assigning one polygon to each point undercounts a GeoPandas spatial join when polygons overlap.

Of the five requested tasks, only terrain has an end-to-end program in the indexed docs. The shared lifecycle now exists and works for constructing all five. It takes 60 nonblank lines before adding a task, while each contributor declaration below takes two or three.

**The highest-value change is to publish these five small, executable examples with discriminating inputs and named results.** An asymmetric road, two overlapping polygons, and two distinct p-values teach more than a longer API inventory. The smallest package addition I recommend is a direction-explicit facility-search helper. Device creation, graph submission, and typed buffer imports already have public upstream APIs.

## Census

I scanned `README.md`, `modules/analysis/README.md`, and TypeScript fences under `docs/api`. For paths in `.typecheck/r19/staging-owned.txt`, I used the corresponding `luma-contributors-ki` file. **KI** below denotes that newer copy. Unmarked paths are in **r19**. `D` means `docs/api`, `S` means `modules/analysis/src`, and `T` means `modules/analysis/test`.

For each class I counted fences matching `new <Class>(` and, separately, those containing that constructor plus `runFirstComputation(` or `.submit(`. The example count is the number of `examples/**/*.compute.ts` files containing the constructor. It excludes indirect uses through recipes.

| Public class | Constructor fences | Fences connected to submission | Direct compute examples | Setup / operation / readback lines in my program |
|---|---:|---:|---:|---:|
| `GPUKernelDensity` | 1 | 0 | 1 | 59 / 3 / 15 |
| `GPUNetworkServiceAreas` | 1 | 0 | 1 | 61 / 3 / 15 |
| `GPUTerrainDerivatives` | 3 | 1 | 9 | 60 / 2 / 12 |
| `GPULocalMoran` | 2 | 0 | 0 | 61 / 2 / 12 |
| `GPUSpatialPredicateJoin` | 3 | 0 | 2 | 62 / 3 / 16 |

The line counts use the literal shared fence at `D/README.md:38` plus the task block below, excluding blank lines. Imports, fixtures, allocations, submission, cleanup and outer control are setup. `graph.add(...)` is the operation. The two typed read implementations, invocation of the deferred reader, and each task’s result callback are readback. Thus **49 setup and 11 readback lines are identical** in every column. These are code counts, not measured completion times.

I also imported the nine production domain/core/record entries through the built package paths and deduplicated exported constructors by identity:

```js
const producers = new Set(entries.flatMap(entry => Object.values(entry))
  .filter(value => typeof value === 'function' &&
    typeof value.prototype?.getCommandNodes === 'function'));
```

Result: **498 producers**, including all five above. Searching those exports for `^run[A-Z]`, `^create.*Device$`, `createAnalysisGraph`, and `createNetworkCentralityContinuation` returned **two** names: `dataframe-analysis/runWideNumericKernel` and `gpu-network/createNetworkCentralityContinuation`. Neither runs these five tasks. `runFirstComputation` is docs code. `createNullWebGPUDevice`, `readFloat32`, `readUint32` and `submitGraph` exist under `testing`. The production `getAnalysisRequiredLimits` helps choose device limits. It does not create a device.

### The five programs

The common prefix is **the entire [Shared first-run lifecycle fence](./docs/api/README.md#shared-first-run-lifecycle)**, `D/README.md:38–99`. Concatenate that literal fence with **any one** block below to get a complete browser TypeScript program. No placeholders, test imports, existing graph, dataset, or external data requests are required. In this unpublished checkout the linked setup is `nvm use`, `yarn install`, a current `yarn build`, then a localhost Vite module page. The exact HTML entry and Vite command are at KI `D/operation-forms-and-cpu-twins.md:252–262`. I did not run installation, build, or Vite.

The common concepts I had to learn were the WebGPU adapter and device, the physical `Buffer` and its usage bits, the graph import handle, the typed view and its logical row count, `graph.add`, compilation, command encoding, submission, asynchronous byte readback, and destruction order. A point has two float components but occupies **one** `float32x2` row. A returned array must survive disposal of the resources that produced it.

**1. Kernel density of points on a grid.** The linked walkthrough computes binned counts. The cross-reference correctly redirected me to KDE. From the density page I could supply the grid, planar coordinates, bandwidth, kernel, index bounds, and normalization. I opened the parameter packer and the exact gather to verify row placement and investigate precision. For one point at the northwest cell center, the expected raster is `[3/π, 0, 0, 0]` points per square coordinate unit. The host kernel evaluator returned `[0.954929658551372, 0, 0, 0]`.

```ts
import {GPUKernelDensity, getGPUKernelDensityParameterValues} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
console.log(await runFirstComputation('kde', ({graph, float32, uint32}) => {
  const grid = makeGPURasterMetadata({width: 2, height: 2, cellSize: 1, origin: [0, 2]});
  const points = float32('xy', Float32Array.of(0.5, 1.5));
  const positions = graph.createDataView(points.handle, {format: 'float32x2', length: 1});
  const parameters = float32('parameters', getGPUKernelDensityParameterValues({bandwidth: 1}, grid));
  const values = float32('density', new Float32Array(4).fill(NaN));
  const invalid = uint32('invalid', Uint32Array.of(99));
  graph.add(new GPUKernelDensity({positions, grid, parameters: parameters.view,
    indexBounds: [0, 0, 2, 2], indexGridSize: [2, 2], kernel: 'quartic', edgeCorrection: 'none',
    output: {values: values.view, status: {invalidCount: invalid.view}}}));
  return async () => {
    if ((await invalid.read())[0] !== 0) throw Error('Excluded points');
    return values.read();
  };
}));
```

**2. Distance from each node to its nearest facility.** The physical road is `0 → 1 → 2`, with costs 2 and 5 seconds. Facility row 0 is at node 2. I could get the CSR layout from the network quick start. I had to open `gpu-network-service-areas.ts` and its CPU twin to settle the search direction, the exact output field for distance, and the meaning of the assignment. The following is the **incoming** CSR of that road. The CPU twin returned seconds `[7, 5, 0]`, facility rows `[0, 0, 0]`, and convergence `1`.

```ts
import {GPUNetworkServiceAreas} from '@luma-contributors/analysis/gpu-network';
console.log(await runFirstComputation('to-facility', ({graph, float32, uint32}) => {
  const topology = {
    offsets: uint32('offsets', Uint32Array.of(0, 0, 1, 2)).view,
    neighbors: uint32('neighbors', Uint32Array.of(0, 1)).view,
    weights: float32('seconds', Float32Array.of(2, 5)).view
  };
  const facilities = uint32('facilities', Uint32Array.of(2));
  const assignments = uint32('assignments', new Uint32Array(3).fill(99));
  const costs = float32('costs', new Float32Array(3).fill(NaN));
  const done = uint32('done', Uint32Array.of(0));
  graph.add(new GPUNetworkServiceAreas({topology, facilities: facilities.view,
    maximumIterations: 4, maximumLabelIterations: 4,
    output: {assignments: assignments.view, costs: costs.view, status: {converged: done.view}}}));
  return async () => {
    if ((await done.read())[0] !== 1) throw Error('Incomplete service areas');
    return {seconds: await costs.read(), facilityRows: await assignments.read()};
  };
}));
```

**3. DEM slope and hillshade.** This was the one task I could finish from the API index alone. I changed its flat tile to a plane rising 30 m for every 30 m east, illuminated from the west at 45°. At the center the expected slope is **45 degrees**, and the unit surface normal points at the light, giving **hillshade 1**. I subsequently opened `gpu-terrain-derivatives.ts` to check the azimuth convention, derivative scale, and validity write. Those GPU expectations are `SOURCE`.

```ts
import {GPUTerrainDerivatives, getGPUTerrainDerivativesParameterValues} from '@luma-contributors/analysis/gpu-terrain';
import {makeGPURasterBand, makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
console.log(await runFirstComputation('slope', ({graph, float32}) => {
  const grid = makeGPURasterMetadata({width: 3, height: 3, origin: [0, 90], cellSize: 30});
  const heights = float32('meters', Float32Array.of(0,30,60, 0,30,60, 0,30,60));
  const elevation = makeGPURasterBand(heights.view, grid, {id: 'dem'});
  const light = {azimuthDegrees: 270, altitudeDegrees: 45};
  const parameters = float32('parameters', getGPUTerrainDerivativesParameterValues(light, grid));
  const slope = float32('slope', new Float32Array(9).fill(NaN));
  const shade = float32('shade', new Float32Array(9).fill(NaN));
  graph.add(new GPUTerrainDerivatives({grid, elevation, parameters: parameters.view,
    output: {slope: slope.view, hillshade: shade.view}}));
  return async () => [(await slope.read())[4], (await shade.read())[4]];
}));
```

**4. Local Moran’s I with p-values.** I chose raw two-sided **analytic** p-values, no multiplicity adjustment, and a four-row directed ring with one unit-weight neighbor per row. This tiny graph makes the wiring checkable. It is not a substantive use of the normal approximation. I opened `gpu-local-moran.ts` for the mandatory `zScores` output and gating behavior, the weights declaration for an uploaded CSR, and the independent host test oracle to distinguish the two statistics in the guide. The oracle gave local I `[0.502174, -0.045652, 0.110870, -1.219565]` and p `[0.285049, 0.907456, 0.160967, 0.285049]`.

```ts
import {GPULocalMoran, getGPUSpatialAutocorrelationParameterValues} from '@luma-contributors/analysis/gpu-spatial-analysis';
console.log(await runFirstComputation('local-i', ({graph, float32, uint32}) => {
  const weights = {
    offsets: uint32('offsets', Uint32Array.of(0,1,2,3,4)).view,
    neighbors: uint32('neighbors', Uint32Array.of(1,2,3,0)).view,
    weights: float32('weights', Float32Array.of(1,1,1,1)).view
  };
  const values = float32('values', Float32Array.of(1,2,4,8));
  const parameters = float32('parameters', getGPUSpatialAutocorrelationParameterValues({}));
  const result = (id: string) => float32(id, new Float32Array(4).fill(NaN));
  const [z, localI, p] = ['z', 'I', 'p'].map(result);
  graph.add(new GPULocalMoran({weights, values: values.view, parameters: parameters.view,
    output: {zScores: z.view, localI: localI.view, pValues: p.view}}));
  return async () => ({localI: await localI.read(), analyticP: await p.read()});
}));
```

**5. Spatial join with a count for each polygon.** I used GeoPandas `sjoin(predicate='within')` semantics: a point strictly inside two polygons contributes once to **each**. The cross-reference initially led me to exclusive assignment. I opened the predicate-join output type, final status writer, point-in-polygon CPU twin, and predicate CPU twin to resolve that discrepancy. Putting polygons on the left and using `contains` yields one CSR row per polygon. Differencing offsets gives counts without downloading and regrouping two columns of pair IDs. The CPU predicate joined polygon rows `[0,1]` to point rows `[0,0]`, giving counts `[1,1]`.

```ts
import {GPUSpatialPredicateJoin} from '@luma-contributors/analysis/gpu-spatial-analysis';
console.log(await runFirstComputation('join', ({graph, float32, uint32}) => {
  const xy = (id: string, ...coordinates: number[]) => graph.createDataView(
    float32(id, Float32Array.from(coordinates)).handle, {format: 'float32x2', length: coordinates.length / 2});
  const words = (id: string, ...values: number[]) => uint32(id, Uint32Array.from(values));
  const polygons = {kind: 'polygons' as const, positions: xy('rings', 0,0,2,0,2,2,0,2, 1,0,3,0,3,2,1,2),
    featureOffsets: words('features',0,1,2).view, polygonOffsets: words('polygons',0,1,2).view,
    ringOffsets: words('vertices',0,4,8).view};
  const points = {kind: 'points' as const, positions: xy('point',1.5,1)};
  const offsets = words('offsets',99,99,99), neighbors = words('neighbors',99,99);
  const weights = float32('weights', new Float32Array(2));
  const overflow = words('loss',99), uncertain = words('uncertain',99);
  graph.add(new GPUSpatialPredicateJoin({left: polygons, right: points, predicate: 'contains', candidateCapacity: 2,
    output: {weights: {offsets: offsets.view, neighbors: neighbors.view, weights: weights.view},
      status: {overflow: overflow.view, uncertainCount: uncertain.view}}}));
  return async () => {
    if ((await overflow.read())[0] || (await uncertain.read())[0]) throw Error('Incomplete or uncertain join');
    const ends = await offsets.read();
    return {counts: ends.slice(1).map((end,i) => end - ends[i]), pointRows: await neighbors.read(ends[2])};
  };
}));
```

For the host construction check I bundled the programs in memory with esbuild, replaced only device acquisition with the repository’s null device, and stopped after the construction callback. **All five reached the end of `graph.add`.** I repeated the Moran and predicate declarations against their KI implementations. This exercises constructor validation, view ownership and node declaration. It is not a TypeScript compiler run or a numerical GPU test.

## Proposed convention

A first-run page should take literal inputs through public imports to a **displayed, named value** and its applicable completeness evidence. Its fixture should distinguish the advertised answer from the nearest plausible wrong answer. A checker can extract the fence, typecheck the public imports, run the unchanged example on a device, and compare its named outputs with an independent expectation.

## Findings

All recommended changes below are **ADDITIVE**. Existing callers can keep their current ports and behavior. “Zero callers” refers to mandatory application migrations, not the documentation or regression fixtures that should change.

| ID | Kind | Reach | Change | Cost | Where | One line |
|---|---|---|---|---|---|---|
| FST-01 | W | ORDINARY | ADDITIVE | S | KI statistics guide:311 | The Moran example leaves Gi* p-values beside Moran z-scores. |
| FST-02 | W | ORDINARY | ADDITIVE | S | KI network operators:327 | The nearest-facility headline omits travel direction. |
| FST-03 | W | ORDINARY | ADDITIVE | S | Cross-reference:39,43,325 | Exclusive assignment is presented as a route to an all-matches polygon count. |
| FST-04 | W | ORDINARY | ADDITIVE | S | Cross-reference:84 | A JavaScript caller can ask for permutation gating and silently get analytic gating. |
| FST-05 | W | ORDINARY | ADDITIVE | S | KI joins guide:302 | The displayed pair-join snippet can return a plausible partial join with its only overflow flag clear. |
| FST-06 | W | UNUSUAL | ADDITIVE | S | KDE packer:171 | Finite, ordered grid corners can become an empty grid during packing. |
| FST-07 | W | UNUSUAL | ADDITIVE | S | Terrain derivatives:98,445 | Nonfinite light settings pass the packer and bypass the validity condition. |
| FST-08 | C | ORDINARY | ADDITIVE | M | KI point-in-polygon CPU twin:127 | Merely adding singleton feature offsets changes `featureCounts`. |
| FST-09 | E | ORDINARY | ADDITIVE | S | KI point-in-polygon CPU twin:89 | Asking the public CPU function for counts materializes the full membership matrix. |
| FST-10 | D | ORDINARY | ADDITIVE | S | KI density guide:238 | A normalized kernel does not guarantee that sampled cell values times cell area sum to one. |
| FST-11 | E | ORDINARY | ADDITIVE | S | KI point-density walkthrough:320 | Output decoding depends on allocation order. |
| FST-12 | E | ORDINARY | ADDITIVE | S | API index:79 | The shared example only works with a device it creates and destroys. |
| FST-13 | E | ORDINARY | ADDITIVE | S | KI CPU walkthrough:72 | The first CPU calculation succeeds without showing the user a number. |
| FST-14 | T | ORDINARY | ADDITIVE | S | API index:106 | The terrain example is insensitive to slope units, cell scale and light azimuth. |

### FST-01: The statistics example mixes two analyses in one result

**Where.** `D/gpu-spatial-analysis/operators-statistics.md:268–269`, KI `:311–312`. Both copies wire `zScores` to both contributors, but `pValues` only to `GPUHotSpotAnalysis`. Neither line requests `localI`.

**Evidence.** `EXECUTED`: the two contributors accepted the shared destination and lowered on the same null graph. For the four-row fixture above, the host oracles gave row-zero Gi* `(z,p) = (-1.453631, 0.146049)` and Moran `(1.069045, 0.285049)`. `SOURCE`: KI `gpu-local-moran.ts:680` writes the latter z-score. The smallest device probe is the same two declarations and a read of row zero from both shared buffers.

**Now.** `console.log(zScores[0], pValues[0])` after the displayed graph can present `(1.069045, 0.146049)` as an inferential result. There is no I to read.

**Wanted.** Give each statistic its own `{zScores, pValues}`. Show `output: {localI: moranI, zScores: moranZ, pValues: moranP}` on the Moran line. Assert both p-values on the same asymmetric fixture.

**Callers, value and cost.** Zero application callers. An analyst can associate each p-value with its statistic. S.

### FST-02: A fully converged search can answer the reverse travel question

**Where.** `D/gpu-network/operators.md:317`, KI `:256–257,327–341`; `S/gpu-network/network-analysis/gpu-network-service-areas.ts:423–429`. The implementation passes `facilities` as the reachability **starts**.

**Evidence.** `EXECUTED`, KI `computeNetworkServiceAreasOnCPU`: for `0 → 1` costing 2, `1 → 2` costing 5, facility `[2]`, the outgoing CSR gave `[Infinity, Infinity, 0]` and `converged=[1]`. Its transpose gave `[7,5,0]` and `converged=[1]`. The KI text includes an outward-search example farther down. The headline still promises the natural inward interpretation.

**Now.** `new GPUNetworkServiceAreas({topology: roads, facilities, ...})` looks like the right answer to “how long does it take a resident to reach a hospital?”

**Wanted.** Put “facility → node: outgoing CSR; node → facility: incoming CSR” on the constructor example. Show `costs` and `status.converged`. For distance alone, show multi-source `GPUNetworkReachability` with only costs and convergence. Use service areas when the winning facility row is also wanted.

**Callers, value and cost.** Zero. Routing users can choose the direction before trusting a converged answer. S.

### FST-03: GeoPandas users are directed to an exclusive count

**Where.** `D/gpu-spatial-analysis-cross-reference.md:39,43,325`; KI `D/gpu-spatial-analysis/operators-statistics.md:2880–2893`; KI `D/gpu-spatial-analysis/operators-joins.md:679–693`.

**Evidence.** `EXECUTED`: with the overlapping rectangles and interior point in program 5, the predicate twin gave two pairs and counts `[1,1]`. The single-point assignment twin gave `[1,0]`, with no loss. Moving the point to `(0,1)` on the outer left edge gave zero strict pairs but assignment counts `[1,0]` under the half-open default.

**Now.** The lookup row for `GeoSeries.within` and “Count points in polygon” names `GPUPointInPolygonJoin`. Its notes mention the smallest-row winner, yet the `sjoin`/`groupby` row also directs readers to exclusive zonal aggregation.

**Wanted.** Put two questions in the lookup: “Return every strict point–polygon pair, then count” → `GPUSpatialPredicateJoin`; “Assign each point to one coverage zone” → `GPUPointInPolygonJoin` or `GPUZonalStatistics`. Show `[1,1]` versus `[1,0]` and name the boundary rule.

**Callers, value and cost.** Zero. GIS users can reproduce their intended join cardinality. S.

### FST-04: The advertised permutation switch silently selects analytic gating

**Where.** `D/gpu-spatial-analysis-cross-reference.md:84`; KI `gpu-local-moran.ts:78,320,473,558`; KI `D/gpu-spatial-analysis/operators-statistics.md:290–296,933`.

**Evidence.** `EXECUTED`: I constructed and lowered `GPULocalMoran` with `quadrantGating` set to `'analytic'`, `'permutation'`, and `'none'` from JavaScript. All three succeeded. The generated WGSL for the first two was identical, SHA-256 prefix `267b3c63d547`. `'none'` produced `6d4a52bed6b5`. The declared TypeScript union admits only `'analytic' | 'none'`.

**Now.** `quadrantGating: 'permutation'` is a natural reading of the lookup row. That row also maps bivariate `Moran_Local_BV` to the one-value `GPULocalMoran`.

**Wanted.** Reject unknown gating strings at construction with the two allowed literals. Move `Moran_Local_BV` to `GPULocalPermutationTest` with `statistic: 'localBivariateMoran'` and `secondValues`. Explain that permutation gating uses the permutation test’s significance mask and ungated quadrants. A single option does not perform that test.

**Callers, value and cost.** Zero supported callers. Configuration-driven applications get a visible error before displaying the wrong inference. S.

### FST-05: The pair-join example checks the wrong level of overflow for complete counts

**Where.** KI `D/gpu-spatial-analysis/operators-joins.md:301–308,335–341,505–506`; KI `S/gpu-spatial-analysis/spatial-join/gpu-spatial-predicate-join.ts:2029–2045`. Builds on `KNOWN(DOC-103)` with a different constructor example and an executed missing-match case.

**Evidence.** `EXECUTED`: one point inside two polygons, pair capacity 2, candidate capacity 1. The predicate twin returned `{count:1, requiredCount:1, overflow:0, candidateOverflow:1}`. With candidate capacity 2 it returned two pairs. `SOURCE`: the GPU writer separately assigns `pairsOverflow` and `pairsCandidateOverflow`. The device probe is the same two-capacity experiment with both scalar destinations.

**Now.** `output: {pairs: {leftRows, rightRows, status: {count, overflow}}}` leaves no visible candidate-loss signal in the example. Even `requiredCount === count` cannot certify the count.

**Wanted.** Wire the aggregate `output.status.overflow` or the pair port’s separate `candidateOverflow` in both examples. Check it before printing counts. For uncertified classifications, also expose `uncertainCount`.

**Callers, value and cost.** Zero. Users can distinguish one genuine match from one surviving candidate. S.

### FST-06: KDE validates binary64 geometry and can upload a zero-height window

**Where.** `S/gpu-spatial-analysis/point-density/gpu-kernel-density.ts:171–200,1189–1203`; KI `D/gpu-spatial-analysis/operators-density.md:231–237,282–285`. This adds a concrete host failure to `KNOWN(PTS-139)`.

**Evidence.** `EXECUTED`: a one-cell grid with `origin:[500000,6000000]`, `cellSize:0.1`, `bandwidth:1` passed the public packer. Its Y endpoints both became `6000000`, and the default window height became zero. Independently, bandwidths `1e-50` and `1e40` passed and packed to `0` and `Infinity`. `SOURCE`: the gather requires `all(gridEnd != gridStart)`. The smallest GPU probe compares this one cell with the same geometry translated to local coordinates before float32 conversion.

**Now.** `getGPUKernelDensityParameterValues(settings, grid)` returns nine words for inputs it can already tell the shader cannot use.

**Wanted.** Validate finite packed corners, distinct packed endpoints, ordered packed rectangle bounds, and a positive finite packed bandwidth before returning the target. For collapsed geometry, name the axis and tell the caller to subtract a common origin from positions, grid, index bounds and window **before** converting to float32.

**Callers, value and cost.** Zero valid callers. Projected-coordinate users get an actionable diagnostic at setup. S.

### FST-07: Terrain validity does not include the light used to compute hillshade

**Where.** `S/gpu-terrain/terrain-analysis/gpu-terrain-derivatives.ts:98–103,169–170,445–447,469–477`.

**Evidence.** `EXECUTED`: packing `{azimuthDegrees:NaN}` on the documented grid returned `[30,30,1,NaN,45,90,0,0]`. `SOURCE`: `isValid` tests the gradients and cell sizes. The lighting expression reads the nonfinite angle afterward, and `outputValidity` publishes `isValid`. The type documentation promises NaN and validity zero for invalid settings. The GPU probe is a valid flat tile with `hillshade` and `outputValidity`, changing only azimuth from 315 to NaN. Check the validity bit independently of backend NaN arithmetic.

**Now.** A missing light value can leave a pixel marked usable.

**Wanted.** Reject nonfinite packed light angles in the public settings packer. On a graph that publishes hillshade, include finite light settings in the per-encoding validity condition, since callers can write directly into a parameter buffer.

**Callers, value and cost.** Zero valid callers. Terrain consumers can use the published validity word for a shaded result. S.

### FST-08: Singleton multipart encoding changes the meaning of `featureCounts`

**Where.** KI `S/gpu-spatial-analysis/spatial-join/cpu-point-in-polygon-join-operation.ts:119–131`, `gpu-point-in-polygon-join.ts:995,1409,1441`; KI joins guide `:121–136`. This concerns the new KI multipart support beyond `KNOWN(LR-08)`.

**Evidence.** `EXECUTED`: for the same single point inside two polygons, omitting point `featureOffsets` gave `featureCounts=[1,0]`. Adding `featureOffsets:Uint32Array.of(0,1)` gave `[1,1]`. In both runs `pointFeatureIds=[0]`, membership was `[1,1]`, and overflow was zero. The KI guide describes the multipart behavior. Its consequence for equivalent singleton geometries is easy to miss.

**Now.** Converting `Point` rows to one-part `MultiPoint` rows can change an aggregate without changing any locations or logical feature identities.

**Wanted.** Add explicitly named `assignedFeatureCounts` and `matchedFeatureCounts` with representation-independent meanings on the GPU and CPU forms. Retain the existing field for compatibility and remove it from first-run examples. Pin the single-point and singleton-multipart representations to the same expectations for the new fields.

**Callers, value and cost.** Zero mandatory migrations. Data ingestion can normalize geometry representation without changing count semantics. M.

### FST-09: The new public CPU point join materializes dense membership for a counts-only user

**Where.** KI `S/gpu-spatial-analysis/spatial-join/index.ts:83`; KI `cpu-point-in-polygon-join-operation.ts:39–51,89–102`. The counts-only implementation exists but is internal.

**Evidence.** `EXECUTED`: 128 outside points, 64 polygons, candidate and match capacities both 1. The public function returned `count=0`, `candidateCount=0`, and a **32,768-byte** `featurePolygonMembership`. The allocation is `4 × pointFeatures × polygons` regardless of matches. At 100,000 points and 1,000 polygons that field alone would be 400 MB. I did not allocate it.

**Now.** `computePointInPolygonJoinOnCPU(input, options).featureCounts` pays for every possible membership. The nearby GPU guide describes dense membership as explicitly requested.

**Wanted.** Expose `computePointInPolygonCountsOnCPU` with a return type containing the rows, counts and diagnostics it actually returns. Link it beside the full CPU join. This addresses allocation; the existing CPU implementation still examines polygon bounds.

**Callers, value and cost.** Zero. CPU-first users can count without allocating a dense matrix. S.

### FST-10: The probability-density description invites an incorrect raster sum

**Where.** KI `D/gpu-spatial-analysis/operators-density.md:238–243`; `S/gpu-spatial-analysis/point-density/kernel-density-math.ts:65–82`.

**Evidence.** `EXECUTED`: one unit-weight point at the center of one unit-area cell, quartic bandwidth `0.1`, no edge correction. The complete kernel support fits inside the cell. The host kernel evaluator gives `95.4929658551372` at its center. Multiplying the published sample by cell area still gives `95.4929658551372`. The continuous probability within that cell is 1. The device probe is program 1 with one cell, bandwidth `0.1`, and `normalization:'probability'`.

**Now.** “Multiply by a cell area to get expected weight per cell” sits beside a statement about integration to one. The later cell-center qualification does not quantify the distinction.

**Wanted.** Say “Values sample a continuous density at cell centers. Multiplying by cell area gives a midpoint approximation to cell mass. Its sum can differ substantially from one when bandwidth is small relative to cell size.” Use the one-cell example.

**Callers, value and cost.** Zero. A data developer can distinguish a density surface from per-cell probabilities. S.

### FST-11: The primary GPU walkthrough decodes an output by its position in the cleanup list

**Where.** `D/operation-forms-and-cpu-twins.md:316–317`, KI `:320–325`; the API-index helper already demonstrates named reads at `D/README.md:54–73`.

**Evidence.** `EXECUTED` host decoder replay: insert a weights buffer containing ones before the density output and use a fixture with actual counts `[0,2,1,1]`. The walkthrough expressions reading `buffers[2]` and `buffers[3]` decode `[1,1,1,1]` and overflow `0`. The displayed guard passes. The original unmodified fixture is correctly indexed.

**Now.** Adding an ordinary input can redirect both readbacks without a type error.

**Wanted.** Retain a named `{buffer, view}` or typed `read` closure at allocation. Decode `values.read()` and `overflow.read()`. Keep the array solely for destruction. Add an unrelated allocation before the outputs in the regression fixture.

**Callers, value and cost.** Zero. Readers can adapt the first GPU example without depending on allocation order. S.

### FST-12: A deck.gl user cannot borrow the host device through the shared first-run function

**Where.** `D/README.md:79–98`; `examples/deck/spatial-analysis-showcase/src/engine/deck-host.ts:336–355`; `examples/deck/spatial-analysis-shared/graph-buffers.ts:47–74`.

**Evidence.** `COUNT`: all five programs use the identical 60-line prefix. The callback receives graph and allocation functions. The outer function always creates a device and destroys it. The showcase obtains Deck’s `device` in `onLoad` and uses it across scenes. `EXECUTED`: on a null graph, two public `graph.importGPUData` calls on borrowed `GPUData` over one buffer gave **the same graph buffer handle**, byte offsets `0` and `8`, and the requested one-row `float32x2` format.

**Now.** Calling `runFirstComputation` inside Deck creates another device. Simply replacing the creation line with Deck’s device would then run `device.destroy()` on the host device.

**Wanted.** Split the **example** into `runFirstComputationOnDevice(device, id, construct)` and a standalone wrapper that creates and destroys its own device. Show the two-line Deck call beside it. Document `graph.importGPUData(id, data)` for an existing column. The point-density guide already mentions that method in prose.

**Callers, value and cost.** Zero package callers. Deck users can share resident inputs and keep ownership of their device. S.

### FST-13: The CPU first run verifies a number without displaying it

**Where.** `README.md:8`, `modules/analysis/README.md:10`, KI `D/operation-forms-and-cpu-twins.md:57–63,72–89`; `T/analysis-core/first-run-docs.node.spec.ts:22–37`. This is the remaining user-visible step after `KNOWN(DOC-02)`.

**Evidence.** `EXECUTED`: the literal TypeScript fence, bundled and run in memory, emitted **no stdout**. Appending `console.log([...exponentialParameters], cpuFit.status[1])` printed `[0.2000000774860382,1.5,2.200000286102295,2.7552409681032286e-13] 0`. The documented workspace command runs assertions in Vitest.

**Now.** A newcomer sees a passing test or a quiet application, then has to inspect the last local variable.

**Wanted.** Print `{model:'exponential', nugget, partialSill, effectiveRange, status}` in the literal example and supply one workspace command that executes that example and displays the result. Keep the numerical assertions in the test.

**Callers, value and cost.** Zero. The first computation ends with an identifiable model and number on screen. S.

### FST-14: A flat DEM cannot test the interpretation of slope or illumination

**Where.** `D/README.md:106–138`, KI `D/gpu-terrain/README.md:58–96`, `T/analysis-core/family-first-run.spec.ts:30–39`. This extends `KNOWN(DOC-111)`: an executable terrain example now exists.

**Evidence.** `SOURCE`: changing slope degrees to percent, changing horizontal cell scale, or rotating the light in azimuth leaves the documented flat-center answers unchanged. The nonzero hillshade and poisoned slope already detect an omitted write. They do not detect those three interpretation errors. Program 3 is the minimal GPU probe. Change azimuth from 270° to 90° on the same east-rising plane. Center slope should stay 45°, while center shade should move from approximately 1 to approximately 0.

**Now.** `console.log(0, 0.70710678)` gives no evidence that the caller arranged the DEM or supplied the light direction correctly.

**Wanted.** Use the ramp in the user-facing example and verify both illuminations in its device fixture. Retain the flat tile as a separate flat-gradient test.

**Callers, value and cost.** Zero. Readers can recognize reversed columns, wrong slope units, or a misinterpreted azimuth. S.

## Departures by name

- **Missing a first-run result for the requested task:** `GPUKernelDensity`, `GPUNetworkServiceAreas`, `GPULocalMoran`, `GPUSpatialPredicateJoin`. `GPUTerrainDerivatives` has one and needs a discriminating input.
- **The obvious result name can identify a different calculation:** `GPULocalMoran` beside `GPUHotSpotAnalysis`; `GPUNetworkServiceAreas` for inward versus outward cost; `GPUPointInPolygonJoin` and `GPUZonalStatistics` in the GeoPandas count lookup.
- **First-run completeness evidence falls short:** the `GPUSpatialPredicateJoin` pair-only snippets and the `GPUTerrainDerivatives` combined lighting validity.
- **The public packing or result path can establish the problem before submission:** `getGPUKernelDensityParameterValues`, `getGPUTerrainDerivativesParameterValues`, KI `computePointInPolygonJoinOnCPU`.
- **Lifecycle departures are in examples:** `runPointDensityExample` couples readback to allocation order; `runFirstComputation` couples graph execution to device ownership; the `computeVariogramModelFitOnCPU` walkthrough lacks a display step. None requires changing all 498 producers.

## Quick wins

- [ ] Fix the Moran and Gi* output destinations. Correct both misleading entries in the PySAL lookup.
- [ ] Put outgoing and incoming CSR side by side with the three-node road and expected costs.
- [ ] Put exclusive ownership and all-pairs containment side by side with the overlapping rectangles.
- [ ] Read aggregate loss in the predicate-join snippet. Add a one-candidate negative control.
- [ ] Validate the float32 values actually packed for KDE and terrain lighting.
- [ ] Expose the existing KI counts-only CPU path with an accurate result type.
- [ ] Use named output reads and a borrowed-device variant of the common example.
- [ ] Print the CPU fit. Use the east-rising DEM for the first terrain number.

## Larger proposals

**Publish five runnable entry modules under one example, with the docs displaying their actual source.** This makes the existing package build and Vite setup serve all five tasks. The common source should have exactly one owner. An integration fixture can check each module’s displayed result against the values above. This is a narrower implementation of `KNOWN(DIR-101, DOC-111)` and costs M. The showcase already contains the techniques worth extracting: `rail-support-planning.compute.ts:136–150` selects forward or reversed CSR, `:306–320` withholds incomplete results, `nature-kernel-density.compute.ts:396` updates bandwidth without recompiling, and `spatial-analysis-shared/resources.ts:76–104` disposes resources in reverse order.

**Add one public convenience function where domain knowledge changes the answer.** The examples contain two separate CPU CSR reversal implementations, `rail-support-planning.compute.ts:44` and `rail-support-allocation.compute.ts:59`. My recommended first package addition is:

```ts
type PackedRoadTopology = {
  offsets: Uint32Array;
  neighbors: Uint32Array;
  weights: Float32Array;
};

function makeFacilitySearchTopologyOnCPU(
  outgoing: PackedRoadTopology,
  options: {travel: 'to-facility' | 'from-facility'}
): PackedRoadTopology;
```

Document the ownership explicitly: `'from-facility'` borrows the input arrays; `'to-facility'` returns newly allocated transposed arrays with each edge’s weight carried to its new slot. Both go through the existing upload path. Test one-way edges, isolated nodes, unequal reverse-edge costs, and parallel edges. This adds no device, graph, submission, or buffer owner. M including replacing the two example implementations.

**Keep execution convenience in the shared example initially.** A borrowed-device function and the existing public `GPUData` import solve the measured repetition across all five programs. If these journeys later need package-level facades, expose domain-specific graph builders that take the caller’s graph and output views, as in `KNOWN(DIR-103)`. For this round, the concrete package needs are direction-aware road preparation and unambiguous polygon counts. There is no measured need for a second graph vocabulary.

## Not read

I read the three landing READMEs, the shared first-run lifecycle and its terrain program, the relevant linked operator sections, and the first CPU and GPU walkthroughs. I sampled constructors, parameter packers, output declarations, generated shader branches, public barrels, test oracles, and example engine helpers along the five paths above. I checked candidates against the prior synthesis, matching area-report entries, `area-api-docs-catalog.md`, `area-direction.md`, and `NEEDS-DECISION.md`.

I did not audit every operator on the long reference pages, every KDE approximation route, complete routing or predicate kernels, the general correctness of permutation inference, or the rest of the 498 producers. The CPU and null-device observations are executed evidence. Every numerical GPU expectation in this report remains a proposed device assertion. No files were written.