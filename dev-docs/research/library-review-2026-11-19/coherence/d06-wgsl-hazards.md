# WGSL portability sweep across analysis kernels

## Summary

The highest-value fix is in raster embedding similarity. The docs recommend using zone means as queries. An empty zone produces NaN, yet the default cosine kernel can select that zone as the best match. A single valid pixel and two queries expose the problem.

The next two fixes are independent of device-specific rounding. `GPURasterTexture` emits a quantizer whose lower bound and width describe different intervals. `GPUTrackSimilarity` squares planar distances and tolerances, although both the requested distance and the correct threshold decision can be represented without squaring.

The sweep also found three one-shot weak atomic exchanges, a two-writer race on malformed resident CSR, and an indirect-dispatch count that wraps to zero. The race requires malformed input. The dispatch arithmetic is reproducible without allocating a large dataset.

I ran the public CPU twins, parameter packers, null-device kernel builders, and shader reflection where applicable. All claims about shader execution are **SOURCE**. Each finding includes a device probe and an expected result.

## Census

I scanned the r19 checkout. Locations below are relative to `modules/analysis/src/` unless they start with `docs/` or `test/`. **KI** identifies a file for which I also checked the newer staging-owned copy. Counts are from one snapshot of a checkout under active edits.

`rg --files modules/analysis/src`, filtered to `.ts`, returned **1,818 files**. A Node pass counted **633** declarations matching `export class (GPU\w+)`. For each pattern below I counted occurrences, matching files, and exported GPU classes declared in those files. The last column understates reach when the shader lives in a separate helper. The counts measure review coverage, not defects.

| Pattern counted | Occurrences | Files | GPU classes in those files |
|---|---:|---:|---:|
| `\bworkgroupBarrier\s*\(` | 464 | 115 | 61 |
| `\bstorageBarrier\s*\(` | 13 | 8 | 5 |
| `\batomicCompareExchangeWeak\s*\(` | 11 | 11 | 2 |
| `\btextureLoad\s*\(` | 1 | 1 | 1 |
| `\bselect\s*\(` | 3,864 | 686 | 367 |
| `\bgetScaledQuotient\s*\(` | 141 | 48 | 30 |
| `\bvar\s+\w+\s*:\s*array\s*<` | 652 | 152 | 94 |
| `\bloop\s*\{` | 220 | 121 | 54 |

Three narrower censuses support actionable comparisons.

- **Weak exchange.** `rg -n 'atomicCompareExchangeWeak\(' modules/analysis/src` found **8 sites with a local retry and 3 one-shot sites**. The three belong to **2 public classes**, `GPUTerrainFlow` and `GPUTerrainStreamOrder`. WGS-12 names all variants.
- **Midpoints.** `rg -nF 'lower + (upper - lower) * 0.5' modules/analysis/src` found **2 sites in 2 public classes**, `GPUClassificationFit` and `GPUClassBreaks`. Both operate on data-derived endpoints.
- **Bindings.** I built the non-blocked cases in `test/analysis-core/kernel-audit/cases-coverage-interpolation.ts` on the null device and compared recorded bindings with `new WgslReflect(source).entry.compute[0].resources`. **48 chunk-kernel records checked, 0 missing bindings, 0 build or reflection failures.** A one-binding-unused control did expose a difference between the two builders, detailed in WGS-15.

## Proposed convention

When a kernel turns arithmetic into a bin, label, match, winner, or dispatch count, give it a tiny fixture with an independently known **discrete** answer at the boundary of its accepted domain. For storage writes, establish a unique owner or retry a weak exchange while the expected value still holds. The kernel audit can check generated names, reflected bindings, and unhandled weak exchanges mechanically. Device tests should assert the actual bin, ID, mask, or count.

## Findings

| ID | Kind | Reach | Change | Cost | Where | One line |
|---|---|---|---|---|---|---|
| WGS-01 | W | ORDINARY | ADDITIVE | S | Raster embedding similarity | An empty query zone can win a cosine search. |
| WGS-02 | W | ORDINARY | ADDITIVE | S | Raster GLCM texture | Independently rounded origin and width change contrast from 4 to 9. |
| WGS-03 | W | EXTREME | ADDITIVE | M | Track similarity | Squaring representable distances changes a Fréchet decision. |
| WGS-04 | W | EXTREME | ADDITIVE | M | Similar locations | Changing units can turn a distinct nearest row into an ID tie. |
| WGS-05 | W | EXTREME | ADDITIVE | S | Classification fit and maximum breaks | Opposite finite endpoints lose their exactly representable midpoint. |
| WGS-06 | W | EXTREME | ADDITIVE | S | Embedding projection | An accepted stretch can lose neutral gray at its midpoint. |
| WGS-07 | W | EXTREME | ADDITIVE | S | Quadrat indices | One-bin repair cannot recover a quotient seed six bins away. |
| WGS-08 | W | EXTREME | ADDITIVE | S | Focal regression and histogram matching | A subnormal reciprocal can erase a finite slope or endpoint bin. |
| WGS-09 | W | EXTREME | ADDITIVE | M | Ensemble scores | A perfect forecast can lose its zero variogram score. |
| WGS-10 | W | EXTREME | ADDITIVE | S | Network isochrones | Conversion and unsigned addition precede the sample cap. |
| WGS-11 | W | EXTREME | ADDITIVE | S | Active-count dispatch | A valid maximum count can generate zero workgroups. |
| WGS-12 | C | UNUSUAL | ADDITIVE | S | Terrain hydrology | Spurious weak-exchange failure consumes convergence budget. |
| WGS-13 | M | UNUSUAL | ADDITIVE | M | Potential path area and local outlier factor | Resident CSR offsets can exceed the slot budget or give a slot two writers. |
| WGS-14 | S | ORDINARY | ADDITIVE | S | WGSL kernel builders | Two legal input names generate duplicate module declarations. |
| WGS-15 | C/T | UNUSUAL | ADDITIVE | S | Chunk bindings and embedding decisions | Two independent compiler-sensitive cases need discriminating audit fixtures. |

The recommended changes require **zero in-repository caller edits**. They correct results, improve declaration-time diagnostics, or add tests. No public signature, output layout, or default needs to change.

### WGS-01: An empty query zone can beat a valid query

**Where.** `raster-analysis/raster-embeddings/gpu-raster-embedding-zone-mean.ts:402` writes missing means. `gpu-raster-embedding-similarity.ts:305–335` chooses the winner. `docs/api/gpu-raster/raster-embeddings.md:58,193` recommends this composition and describes empty zones. The **KI** page retains both statements at lines 64 and 204.

**Now.** This is the wiring a reader can reasonably take from the page, using already allocated views:

```ts
new GPURasterEmbeddingSimilarity({
  pixelCount: 1, dimensions: 1, queryCount: 2,
  vectors: pixels, queries: zoneMeans, parameters,
  output: {similarity, maximum, bestQuery, matches}
}).getCommandNodes(graph);
```

**Evidence and GPU probe.** Use pixel `[1]`, query rows `[-1, NaN]`, and threshold `0`. **SOURCE:** only the pixel is checked for finite components. The cosine `select` can take the zero-denominator fallback for the invalid query. **EXECUTED**, binary32 branch model: scores `[-1,0]`, winner `1`. Expected scores `[-1,NaN]`, winner `0`, maximum `-1`, match `0`. Read all four outputs together.

There is a second failure in the same winner reduction using entirely finite inputs. With `metric: 'dot'`, pixel `[1]`, and queries `[-3.3e38,-3.2e38]`, neither score beats the initializer `-3.0e38`. The model retained `-3.0000000054977558e38`. The public packer accepted threshold `-3.1e38`. The correct winner is query `1` and the match is `0`.

**Wanted.** Validate each query's components in the kernel. Initialize the winner from the first eligible score. If none qualify, write NaN maximum, `0xffffffff` best query, and zero matches. Keep the existing convention for genuinely zero vectors. This adds to KNOWN(RT-126), which concerns completeness of the upstream zone mean. **Value and cost:** GIS users can pass a zone table directly into search, S.

### WGS-02: The GLCM quantizer mixes two coordinate intervals

**Where.** `raster-analysis/remote-sensing/gpu-raster-texture.ts:69–75,128,142`; `docs/api/gpu-raster/remote-sensing.md:19–21`.

**Now.** From the documented finite increasing range contract, use `range: [10000000.49,10000002.51]`, `levels: 4`, `radius: 1`, `offset: [1,0]`, and a two-cell row `[10000000,10000002]`.

**Evidence.** **EXECUTED**, real constructor and null-device kernel generation. It accepted the range and emitted `getScaledQuotient(value - 10000000.0, 2.0199999809265137, 1.0, 0)`. The origin came from the rounded lower endpoint. The width came from the difference of the original doubles. A host bin probe put the second pixel in bin **3**. Using the supplied endpoints consistently gives bin **2**. Rounding both endpoints first also gives bin **2**.

**GPU probe and wanted.** Read `features[0]` and `pairCount[0]`. **SOURCE:** the emitted bins imply contrast `9` for one directed pair. Expect `4` and `1`. Round both endpoints once, then validate and emit the width derived from those endpoints. Document that ranges use float32 endpoints. **Value and cost:** texture features stop depending on a host-only part of the range, S.

### WGS-03: Squaring a tolerance changes the Fréchet yes/no answer

**Where.** `gpu-spatial-analysis/track-similarity/gpu-track-similarity.ts`, **KI** lines 753–769, and `track-similarity-dynamic-programs.ts:155–156,193–217`. `GPUTrackSimilaritySearch` also delegates its retained Fréchet pairs to this metric, **KI** `gpu-track-similarity-search.ts:461`.

**Now.** Compare two single-vertex planar tracks `(0,0)` and `(d,0)` with `frechetEpsilon = d/2`. Both distances and tolerances are normal float32 inputs.

**Evidence.** **EXECUTED** `computeTrackSimilarityOnCPU` with `measures: ['frechet','frechetWithin']`:

| `d` | CPU `frechet` | CPU `frechetWithin` | CPU `pairStatus` |
|---|---:|---:|---:|
| `2^-80` | `8.271806125530277e-25` | 0 | 0 |
| `2^70` | `1.1805916207174113e21` | 0 | 0 |

**SOURCE:** the planar GPU route compares `dot(offset,offset)` with `epsilon*epsilon`. In binary32 the two squares become `0,0` in the first fixture and overflow in the second. The first strict rejection already loses the distinction.

**GPU probe and wanted.** Send the two singleton pairs to `GPUTrackSimilarity`. Assert the actual distances and both decision words against the table. Compare scaled norms to the tolerance in a common exponent frame, and carry that representation through the coupling bounds. **Value and cost:** distance and decision agree across units and with the CPU twin, M.

### WGS-04: Z-score ranking can erase a whole attribute

**Where.** `gpu-spatial-analysis/similar-locations/gpu-similar-locations.ts:504–511`; **KI** lines 555–565 and 679–690. The newer copy improves subtraction around an anchor and still squares the unscaled residuals.

**Now.** Use one attribute `[0,2*d,d]`, reference mask `[1,0,0]`, exclude the reference, and request two nearest IDs. Changing `d` should preserve both standardized distances and their ordering.

**Evidence and GPU probe.** **EXECUTED**, binary32 model of the moment and distance steps: `d=1` gave distances `[2.4494897,1.2247449]` for rows 1 and 2. `d=2^-80` gave `[0,0]`. **SOURCE:** the latter variance becomes zero and the shader writes inverse deviation zero. Run both cases and assert `topIds == [2,1]` plus the two corresponding distances. Checking only the result count misses this.

**Wanted.** Normalize centered residuals by a common power of two before accumulating squares. Use that frame when computing standardized differences. **Value and cost:** nearest-location results survive a change in attribute units, M.

### WGS-05: Two midpoint calculations overflow on opposite finite endpoints

**Where.** `gpu-spatial-analysis/distribution-dynamics/gpu-classification-fit.ts:369`, **KI** `:372`; `dataframe-analysis/column-classification/gpu-class-breaks.ts:1206,1255`.

**Now.** Let `a=2^127`. The median of `[-a,a]` is zero. So is the only interior edge for `method: 'maximum-breaks', classCount: 2`.

**Evidence.** **EXECUTED** `computeClassBreaksOnCPU` with the public parameter packer: breaks `[-1.7014118346046923e38,0,1.7014118346046923e38]`, counts `[1,1]`. **SOURCE:** both kernels use `lower + (upper-lower)*0.5`. The rounded-operation model produced `Infinity` for the midpoint. This extends KNOWN(DPC-111) to the maximum-gap route and the separately exported median. These endpoints are widely separated, so the documented adjacent-value limitation does not apply.

**GPU probe and wanted.** Run maximum breaks on the two values and classification fit with both rows in class 0. Read the interior edge and both class and pooled medians. Expect zero in all three slots. Branch on endpoint signs before subtracting. For maximum breaks, also order gaps in a scaled frame so multiple overflowing gaps remain distinguishable. **Value and cost:** table summaries and map classification retain representable boundaries, S.

### WGS-06: A valid projection stretch loses its midpoint

**Where.** `raster-analysis/raster-embeddings/gpu-raster-embedding-projection.ts:59–73,290`.

**Now and evidence.** **EXECUTED** `getGPURasterEmbeddingProjectionParameterValues` with three ranges `[-2^127,2^127]`. The packer returned six finite float32 endpoints with the requested signs. **SOURCE:** `(score-low)/(high-low)` forms the overflowing difference at runtime. The mathematically correct result for score zero is `0.5` in every channel. KNOWN(RT-121) fixed the different case of a zero-width stretch.

**GPU probe and wanted.** One zero pixel, zero mean, three identity components, and those packed ranges. Read RGB and assert `[0.5,0.5,0.5]` alongside the zero scores. Scale score and endpoints by a common power of two before subtracting when the span is wide. **Value and cost:** a valid neutral score renders at the center of the color stretch, S.

### WGS-07: The quadrat correction assumes the initial bin is nearly right

**Where.** `dataframe-analysis/pair-statistics/quadrat-index-wgsl.ts:33–52` and `gpu-point-pattern-indices.ts:856,859`.

**Now and evidence.** **EXECUTED** the public point-pattern packer with `bounds: [0,0,2**127,1]`, `maximumDistance: 1`. It accepted the finite extent. Split it into eight columns and put one point at `(3*2^125,0.5)`. The correct zero-based column is **6**. **SOURCE:** the seed uses `offset/extent`. Its exact product comparisons can correct only one bin. In a host model of reciprocal lowering with subnormal flush, the seed was `0` and the corrected column was **1**.

**GPU probe and wanted.** Read the eight quadrat counts with one row and the one point. Expect `[0,0,0,0,0,0,1,0]`. Use the existing scaled quotient for the seed or binary-search the existing exact product comparison. KNOWN(PTS-109) fixed rounding at ordinary interior edges. This fixture tests the recovery distance of that fix. **Value and cost:** the point enters its correct spatial cell on reciprocal-flushing backends, S.

### WGS-08: Two ordinary divisions share a backend-dependent failure mode

**Where.** `raster-analysis/focal-pairs/raster-focal-pairs-wgsl.ts:94,119`; `raster-analysis/raster-contrast-equalization/gpu-raster-contrast-equalization.ts:298–302`.

**Now and evidence.** These probes isolate reciprocal lowering. A denominator of `2^127` is finite and normal. Its reciprocal is subnormal. **EXECUTED**, a multiply-by-reciprocal model with subnormal flush returned `0` for `2^127 / 2^127`. This is a conditional backend result. A denominator merely exceeding `1e19` does not establish it.

| GPU probe | Expected | What to read |
|---|---|---|
| Focal OLS, 2×1 grid, radius 1, default clipped border, predictor and response each `[-2^63,2^63]` | Centered cross-product and predictor squares are both `2^127`. Slope `1`, intercept `0`, residual sum `0`, status `0`. | Read the slope **and** status. A finite wrong slope can pass the nonfinite check. |
| Histogram matching, 256 bins, input domain `[0,2^127]`, input at the upper endpoint | Endpoint goes into bin `255`. The reciprocal-flush model put it in bin `0`. | Read the generated `getBin` result in a one-lane probe, or inspect the endpoint histogram before CDF mapping. |

**EXECUTED:** the public contrast-equalization packer accepted that input domain. **SOURCE:** both routes divide by the full-scale denominator. Use `getScaledQuotient` for the focal ratios and normalized histogram coordinate. **Value and cost:** regression coefficients and histogram bins retain their finite answers, S.

### WGS-09: A perfect ensemble can lose its zero variogram score

**Where.** `gpu-spatial-analysis/scoring-rules/gpu-ensemble-scores.ts:401–417`.

**Now and evidence.** **EXECUTED** `computeEnsembleScoresOnCPU` with one member `[-2^127,2^127]`, identical observations, one pair `[0,1]`, and the default order `0.5`. It returned `variogramScore: Float32Array [0]`. **SOURCE:** the shader subtracts the components before applying the fractional power. The difference overflows even though its square root is representable. Subtracting the two resulting terms can then lose the exact zero.

**GPU probe and wanted.** Submit the same single pair and require finite score `0`. Scale the pair endpoints before subtraction, accumulate forecast and observation terms in a common exponent frame, then restore the final difference. **Value and cost:** probabilistic verification preserves the basic perfect-forecast invariant, M.

### WGS-10: An overlong road can receive fewer samples than its cap

**Where.** `gpu-network/network-isochrones/network-isochrones-passes.ts:245–249`; the expression is unchanged in **KI**.

**Now.** A road from `(-2^30,0)` to `(2^30,0)`, buffer radius `1`, has spacing `0.5`. Set `maximumSamplesPerEdge: 3`. Three samples include the midpoint. The kernel computes `clamp(u32(ceil(length/spacing)) + 1u, 2u, MAXIMUM_SAMPLES)`. The conversion is already out of range. On a saturating conversion the addition wraps, selecting only two endpoints.

**Evidence and GPU probe.** **EXECUTED** the repository’s `splatIsochroneRasterOnCPU` oracle with one directed edge of cost 2, source cost 0, grid 3×1, extent `[-1.5,-0.5,1.5,0.5]`, zero walk cost, and that cap. It returned `[1,1,1]`. **SOURCE:** endpoints alone cannot paint the center. Require the three oracle values. This isolates sample-count arithmetic from KNOWN(NET-108) and KNOWN(NET-109).

**Wanted.** Compare to `MAXIMUM_SAMPLES-1` before the integer conversion or addition. **Value and cost:** the cap preserves the intended sample positions on long edges, S.

### WGS-11: The indirect-dispatch packer wraps a valid count to zero

**Where.** `analysis-core/kernels/wgsl-kernel-nodes.ts:79–92,130–145`, `createWGSLActiveCountDispatch`.

**Now and evidence.** **EXECUTED** the real builder on a null device with one count word, `maximumItemCount: 0xffffffff`, `workgroupSize: 256`. It accepted the configuration and recorded `MAXIMUM_X=65535`, `MAXIMUM_Y=257`. Unsigned emulation of `(invocationCount + 255u) / 256u` gave **0**. The correct ceiling is **16,777,216** workgroups.

**GPU probe and wanted.** Run only the one-invocation update kernel, with the resident count set to `0xffffffff`. Read its three dispatch words without submitting the downstream workload. Expect `[65535,257,1]`. Use `n/d + select(0u,1u,n%d != 0u)` for each ceiling division. This is an accepted internal-helper configuration. A public contributor reaching the same limit with ordinary device buffer sizes was not established. **Value and cost:** high-count callers cannot silently dispatch zero work, S.

### WGS-12: Three terrain finalizers spend a round on spurious weak-exchange failure

**Where.** `gpu-terrain/hydrology/terrain-flow-passes.ts:416` for D8, `terrain-flow-routing.ts:392` for weighted flow, and `gpu-terrain-stream-order.ts:642`. The **KI** `gpu-terrain-flow.ts` still routes through the flow finalizers. A sibling implementation retries at `gpu-terrain/terrain-processing/gpu-terrain-branching-flow.ts:277–281`.

**Now and evidence.** **SOURCE:** all three return `atomicCompareExchangeWeak(...).exchanged` immediately. A false result with `old_value == SENTINEL` means ownership is still available. The frontier treats this as failed progress. At a small iteration budget a resolvable cell can remain pending. The existing unresolved status remains significant.

**GPU probe and wanted.** Use a one-cell headwater with a one-round budget. In a test variant inject one permitted spurious failure on the first attempt, then call the real atomic. Expect the accumulation or stream order to finalize in that round. Retry the CAS while the returned old value equals the sentinel. Stop immediately when another invocation has finalized the cell. **Value and cost:** convergence budgets measure dependency depth more reliably, S.

### WGS-13: Malformed resident CSR can create two writers or exceed the slot budget

**Where.** `gpu-network/network-time-geography/gpu-network-potential-path-area.ts:216–225`, **KI** lines 229–237; `gpu-spatial-analysis/outlier-detection/gpu-local-outlier-factor.ts:237,261,285`, **KI** lines 248, 274, 298.

**Now and evidence.** The trigger requires malformed offsets. **EXECUTED** offset enumeration for `[0,2,1,2]` gave `(slot,owner)` pairs `[(0,0),(1,0),(1,2)]`. **SOURCE:** the potential-path kernel dispatches one invocation per node. Owners 0 and 2 both write `edgeSlack[1]` and `edgeInPrism[1]`. For budget `2`, edge weight `1`, reverse cost `0`, and forward costs `0` and `10`, those owners write `(1,1)` and `(-9,0)`. This conflict follows from the addresses. Repeated runs returning the same bytes would not establish safe ownership. The **KI** clear pass fixes KNOWN(NET-107)’s stale output but leaves these concurrent stores.

The three local-outlier-factor loops separately trust resident end offsets as iteration bounds. A one-slot allocation can therefore request many more loop iterations than there are slots.

**GPU probe and wanted.** For potential path area, use the four offsets above and assert a malformed-topology result before scatter. For outlier factor, use a one-slot table ending at `3` and instrument slot visits. Expect at most one visit. Validate monotonicity and slot capacity once on the GPU before scatter, or consume an already validated topology. Clamp read-loop bounds to the declared slot capacity. **Value and cost:** a malformed upstream GPU table cannot race output or monopolize a dispatch, M.

### WGS-14: Legal binding names can generate an illegal module

**Where.** `analysis-core/kernels/wgsl-kernel-nodes.ts:377–382,425–436`; `analysis-core/kernels/graph-chunk-groups.ts:243–245,291–297`. Existing keyword tests are in `test/analysis-core/wgsl-generated-identifiers.node.spec.ts:35,91`, relative to `modules/analysis/`.

**Now and evidence.** **EXECUTED** the public `createWGSLKernelNode` with two one-word views named `values` and `valuesOffset`. The builder accepted both. Its recorded source contained both declarations:

```wgsl
const valuesOffset: u32 = 0u;
@group(0) @binding(1) var<storage, read> valuesOffset: array<u32>;
```

The chunk builder similarly generates a private offset with that name. Checking whether each name is individually legal cannot detect the collision.

**GPU probe and wanted.** Compile the recorded two-binding module and inspect compilation messages. The cheaper permanent test is a host assertion that the builder rejects it before graph declaration, naming the kernel and both colliding names. Check uniqueness across bindings, textures, generated offsets, and fixed module-scope names. **Value and cost:** contributor authors get a useful error at the call site, S.

### WGS-15: Two compiler-sensitive cases deserve separate audit fixtures

**A. Unused storage bindings.** `analysis-core/kernels/wgsl-kernel-nodes.ts:438–445` adds static references for all bindings. `analysis-core/kernels/graph-chunk-groups.ts:303–309,363–390` initializes private offsets and submits all bindings, even if the entry point references only one storage variable.

**EXECUTED:** build both routes with `values` and `weights` and body `values[valuesOffset + index] = 1u;`. Reflected entry resources were `[values,weights]` for the regular builder and `[graphChunkGroupParameters,values]` for the chunk builder. The 48 production interpolation records sampled above all passed. **GPU probe:** compile and submit this one-word control through each builder and require `values[0] == 1` with no bind-group validation error. Add the regular builder’s phony address references to the chunk builder. **Value and cost:** optional routes have consistent binding behavior, S.

**B. Near-boundary class labels.** `raster-analysis/raster-embeddings/gpu-raster-embedding-similarity.ts:320–335` and `gpu-raster-embedding-linear-probe.ts:232–247` feed accumulated dot products directly into discrete decisions. **EXECUTED**, a round-after-each-operation model for weights `[0.5,0.5,0.5]` gave `0` for input `[0.5,2^-26,-0.5]` and `7.450580596923828e-9` for the same bands reordered `[0.5,-0.5,2^-26]`. The exact score is `2^-27` in both orders.

**GPU probe:** for dot similarity use threshold `2^-28`. For the classifier, give class 0 zero weights and bias `2^-28`, and class 1 the weights above with zero bias. Read the **mask and label**, then permute the bands. The exact-arithmetic answers are match `1` and class `1`. **SOURCE:** source-order rounding alone can cross this margin. A Metal-specific reassociation was not observed here. Document the decision precision and pin the chosen policy with this fixture. **Value and cost:** callers learn whether a near-tie label is portable. The test and docs are S. Guaranteeing the exact decision is a separate design choice.

## Departures by name

| Convention under review | Confirmed departures or test gaps |
|---|---|
| Winner must identify an eligible candidate | `GPURasterEmbeddingSimilarity` |
| A numeric domain must retain consistent endpoints and bins | `GPURasterTexture`, `GPURasterEmbeddingProjection`, `GPUPointPatternIndices`, `GPURasterContrastEqualization` |
| A finite quotient, distance, midpoint, or score must survive intermediate scaling | `GPUTrackSimilarity`, its delegated Fréchet search in `GPUTrackSimilaritySearch`, `GPUSimilarLocations`, `GPUClassificationFit`, `GPUClassBreaks` maximum-breaks route, `GPURasterFocalPairs`, `GPUEnsembleScores` |
| Cap before narrowing or unsigned addition | `GPUNetworkIsochrones`, `createWGSLActiveCountDispatch` |
| Weak exchange retries locally while ownership is available | `GPUTerrainFlow` D8 and weighted routes, `GPUTerrainStreamOrder` |
| Resident ranges establish a finite visit budget and unique write ownership | `GPUNetworkPotentialPathArea` and `GPULocalOutlierFactor`, on malformed CSR |
| Generated names and static resource use agree with the builder interface | `createWGSLKernelNode`, `createGraphChunkGroupKernelNode` |
| Boundary tests read the final decision | `GPURasterEmbeddingSimilarity`, `GPURasterEmbeddingLinearProbe` |

The eight local-retry comparators in the atomic census were `trajectory-counts-wgsl.ts`, `reconcile-patches.ts`, `spatial-clustering-kernels.ts`, `hierarchical-density-clustering-kernels.ts`, `ward-active-graph.ts`, `raster-region-boruvka.ts`, `gpu-terrain-branching-flow.ts`, and `terrain-peak-prominence-tree.ts`.

## Quick wins

- [ ] **First:** fix the empty-query winner and the mixed GLCM range. Both have two-row fixtures and externally visible wrong answers.
- [ ] Use sign-aware midpoint arithmetic in the two named classification kernels. Run the maximum-breaks GPU fixture against the existing CPU oracle.
- [ ] Fix projection stretch scaling, quadrat seeding, focal ratios, and histogram endpoint normalization. Keep each discrete or exact result as a separate fixture.
- [ ] Cap isochrone samples before integer conversion. Replace addition-based ceiling division in the active-count packer.
- [ ] Retry the three weak exchanges. Inject one spurious failure in the test so passing does not depend on a particular GPU producing one.
- [ ] Validate the generated WGSL namespace, retain chunk binding references, and add the near-tie mask and label fixtures.

## Larger proposals

1. **Scale before squaring in the affected metric families.** Start with planar track similarity because its public CPU twin already supplies both the distance and the decision oracle. Then address z-score distances and variogram terms. Each change is M. A blanket replacement of every `dot` would make unrelated kernels more expensive.
2. **Offer a validated resident-CSR path.** A shared GPU validation pass can establish monotonic offsets, terminal capacity, and row ownership for several consumers. Per-kernel clamps are cheaper for read-only walks. They cannot establish exclusive scatter ownership. I recommend validation before a producer performs scatter. Cost M for the two routes reviewed.
3. **Decide whether embedding classifications promise reproducibility at a near tie.** Documenting ordinary float32 accumulation and its margin sensitivity is additive and S. A guaranteed reproducible decision would need a specified summation and tie rule, plus testing on more than one backend. I recommend documenting the current precision first and measuring the cost before making that guarantee.

## Not read

I read `gpu-spatial-analysis/shared` and `raster-analysis/focal-pairs` in full. Elsewhere I selected kernel bodies from the library-wide searches, then followed their constructors, parameter packers, output writes, CPU twins, or staging copies where relevant. That included **all 11 weak exchanges**, the **sole direct `textureLoad`**, both named midpoint sites, and the generated-source paths used above.

The direct texture load in `gpu-terrain/terrain-decode/gpu-terrain-rgb-decode.ts:353` is paired with mip and dimension checks at lines 169–175. I did not establish an out-of-bounds texture read, a divergent barrier on valid input, or a reachable function-scope array over 8,192 bytes in the variants I inspected. The 652 array declarations and 477 barrier calls were lexical leads. I did not expand every template specialization, follow every integer shift and signed hash, or inspect all consumer and test bodies. No GPU, browser, network, build, or test runner was used.