# Points, interpolation, clustering, regionalization, and space-time

## What I checked

**Two public CPU operations returned wrong results on small, valid inputs.** Sinkhorn divergence reported convergence with a result about 2.46 times the independently calculated value. Location-allocation produced different assignments and objectives for dense and sparse representations of the same one-entry cost matrix. Both GPU implementations contain the corresponding decision logic. I would fix these first.

I inventoried all 19 assigned directories at `69305eb4`. They contain **195 source TypeScript files and 272 specs**, including 116 `*.node.spec.ts` files. In the references below, `S/` means `modules/analysis/src/gpu-spatial-analysis/`, `T/` means `modules/analysis/test/gpu-spatial-analysis/`, and `C/` means `modules/analysis/src/analysis-core/`.

- **Read in full** `AGENTS.md`, `dev-docs/contributor-contract.md`, `S/change-points/cpu-pettitt-test-operation.ts`, `S/optimal-transport/sinkhorn-divergence-parameters.ts`, `S/space-time-hawkes/cpu-space-time-hawkes-operation.ts`, `S/scan-statistics/scan-draw-interval.ts`, and `S/point-density/point-density-tiles.ts`.
- **Sampled deeply** the large spatial-scan and location-allocation implementations, the recently added analysis-spacing contributor, the CPU and GPU Sinkhorn updates, and the graph construction and output publication paths for REDCAP, SKATER, Max-p, AZP, Ward, Huff, and space-time cross-correlation. I compared local kriging with the Vecchia likelihood and Fisher scoring paths, and sampled KDE, DBSCAN, HDBSCAN, k-means, permutation stopping, and their oracles. I checked targeted roadmap rows and the earlier library, math, and scalability reviews.
- **Verification** used extracted pure CPU functions for the two reproductions, a separate two-cell primal calculation for Sinkhorn, and float32 arithmetic probes. I ran no repository build, compiler, test suite, or GPU workload and modified no files. Emerging hot spots, point process, geographic distribution, dot density, and line density received lighter coverage. I did not establish full device parity or audit every random draw transitively.

The source census found **58** files declaring `implements GPUCommandNodeProducer`, **20** mentioning `estimatedCommandNodeCount`, **20** mentioning `validateCommandNodeBudget`, **22** referencing `PHILOX_PURPOSES`, and **zero** containing `Math.random(`. These are file counts. Fixed-size producers do not all need a node estimate. The census can be reproduced with:

```python
from pathlib import Path
import re

families = """point-density point-process spatial-clustering spatial-regionalization
redcap-regions spatial-interpolation scan-statistics emerging-hot-spots space-time-hawkes
space-time-statistics space-time-tests change-points location-allocation optimal-transport
catchment-accessibility geographic-distribution dot-density line-density analysis-spacing""".split()
root = Path("modules/analysis")
source = [p.read_text() for f in families
          for p in (root / "src/gpu-spatial-analysis" / f).rglob("*.ts")]
specs = [p for f in families
         for p in (root / "test/gpu-spatial-analysis" / f).rglob("*.spec.ts")]
print(len(source), len(specs), sum(p.name.endswith(".node.spec.ts") for p in specs))
for pattern in [r"implements\s+GPUCommandNodeProducer", "estimatedCommandNodeCount",
                "validateCommandNodeBudget", "PHILOX_PURPOSES", r"Math\.random\("]:
    print(pattern, sum(bool(re.search(pattern, text)) for text in source))
```

## Findings

### PTS-01 Sinkhorn declares the divergence converged before its self-costs settle [P0] [correctness]

- **Evidence.** The CPU stopping error uses only `targetOfCross - nextTarget`. The two self-solves take damped updates, then the same loop exits on `error <= tolerance`. Both self-solves enter the published divergence. The GPU likewise writes the residual only under `if (job == 0u)` and disarms the whole iteration gate from that residual. `S/optimal-transport/cpu-sinkhorn-divergence-operation.ts:216,224,230,255`; `S/optimal-transport/gpu-sinkhorn-divergence.ts:582,589,650`.
- **Reproduction.** On a unit-spaced `2 × 1` grid, source `[99, 1]`, target `[1, 0]`, fixed epsilon `1`, tolerance `1e-10`, and budget `1000`, the CPU returned divergence `0.0007405065`, `converged=1`, `iterationCount=2`, and marginal error `5.06e-16`. The independent primal result is `0.0003010437`. It needs just one unknown: the off-diagonal self-coupling mass satisfies `x² = exp(-2)(0.99−x)(0.01−x)`. The cross-cost is `0.01` and the target self-cost is zero.
- **Why it matters.** Even a caller who checks convergence, the reported error, and the iteration limit has no indication that the published divergence is inaccurate.
- **Proposed change.** Require all three residuals to meet tolerance at the final epsilon before disarming. Preserve `marginalError`’s existing meaning if consumers depend on it. Add the two-cell case to the existing dense-primal oracle spec and assert all three components. **M, nonbreaking correctness fix.** `T/optimal-transport/cpu-sinkhorn-divergence-operation.node.spec.ts:48`.
- **Confidence.** High on CPU. GPU stopping logic confirmed in source, device result **UNVERIFIED**.

### PTS-02 Location-allocation treats a valid finite distance as unreachable [P0] [correctness]

- **Evidence.** The public API reserves positive infinity for unreachable costs and promises equivalent dense and sparse results for finite coverage thresholds. Internally, `LARGE_COST` is `1e30`. The sparse finalizer accepts only `cost < assignedCost`, while the dense finalizer also accepts equality. The greedy coverage step compares its initial sentinel with the threshold. `S/location-allocation/gpu-location-allocation.ts:167,181,1129,1313,1510,1556`; `S/location-allocation/cpu-location-allocation-operation.ts:25,173,189,262`.
- **Reproduction.** With one demand row, one facility, unit weight, swaps off, and cost `Math.fround(1e30)`, CPU p-median selected the facility in both representations. Dense returned assignment `0` and objective `1.0000000150474662e30`. CSR returned assignment `0xffffffff` and objective `0`. There is a second failure with an ordinary cost: set covering with cost `1` and threshold `Math.fround(1e30)` selected **zero** facilities in both representations.
- **Why it matters.** Changing the storage representation changes the answer. The existing location-allocation oracle also uses `LARGE_COST`, so parity against it can preserve the defect. `T/location-allocation/location-allocation-oracle.ts:9,75`.
- **Proposed change.** Represent reachability explicitly through candidate scoring, swaps, and final assignment. Use the same tie rule in both finalizers. Test finite values below, at, and above the old sentinel plus an actual unreachable pair. **M, nonbreaking correctness fix.**
- **Confidence.** High on CPU. GPU branches confirmed in source, device result **UNVERIFIED**.

### PTS-03 Huff probabilities retain old edges when a resident CSR shrinks [P0] [correctness]

- **Evidence.** The output promises “0 for slots past the last row.” The writer dispatches once per demand row and writes only `offsets[index] .. offsets[index + 1]`. The shared CSR contract permits `offsets[rows] <= neighbors.length`. The packed-buffer adapter also preserves the old output when it needs scratch storage. `S/catchment-accessibility/gpu-huff-trade-areas.ts:72,354,357,363`; `S/spatial-weights/spatial-weights.ts:31`; `S/geographic-distribution/spatial-analysis-packed-kernel.ts:68,121`.
- **Why it matters.** One row with two equal-weight, equally attractive facilities gives slot 1 probability `0.5`. On resubmission with offsets changed from `[0,2]` to `[0,1]`, there is no writer to replace that value with the promised zero. A consumer exporting the capacity-sized result can attribute demand to a removed edge.
- **Proposed change.** Clear the probability output before scatter, or dispatch a tail clear from the terminal offset to capacity. Add a two-submission test using the same output view and check every slot. **S, nonbreaking.**
- **Confidence.** High on the missing write. Device reproduction **UNVERIFIED**.

### PTS-04 Two independently optional outputs can be bound without acquiring a writer [P1] [correctness]

- **Evidence.** `GPUSpaceTimeCrossCorrelation` offers `testedCount?` as the family size, but `getUsedOutput()` drops it unless `adjustedPValues` exists. Its only writer is inside `if (output.adjustedPValues)`. The sampled rig always supplies both. REDCAP accepts the shared `GPUIterationStatusPort`, which includes `iterationCount`, yet its status bindings and component invocation forward only convergence, the limit flag, and invalid count. `S/space-time-statistics/gpu-space-time-cross-correlation.ts:52,188,325`; `T/space-time-statistics/space-time-cross-correlation.spec.ts:121`; `S/redcap-regions/gpu-redcap-single-linkage.ts:116,213,382,392`; `C/contracts/status.ts:55`.
- **Why it matters.** A consumer can request either scalar through a valid typed API and read a previous frame’s value. Zero-initialized fixtures make this easy to miss.
- **Proposed change.** Count finite raw p-values when only `testedCount` is requested. Have the REDCAP component pass publish the number of executed rounds and document its counting rule. Test each optional port alone in a prefilled view, including a zero-result case. **M, additive behavior for existing ports.**
- **Confidence.** High on both binding paths. Device results **UNVERIFIED**.

### PTS-05 Four regionalizers expand caller budgets into command graphs without a node estimate [P1] [performance]

- **Evidence.** SKATER accepts `maximumRegionCount` with a lower bound of two, allocates state for `maximumRegionCount - 1` steps, and constructs nodes for each step. The gate is attached *inside* that construction loop. Ward, AZP, and Max-p also expand round budgets into graph nodes. None of these four files references `estimatedCommandNodeCount` or `validateCommandNodeBudget`. Their sibling minimum-spanning-tree contributor uses both. `S/spatial-regionalization/gpu-skater-regions.ts:241,337,762,769`; `gpu-ward-regions.ts:464`, `gpu-azp-regions.ts:864`, `gpu-max-p-regions.ts:763,1024`, and `gpu-spatial-weights-minimum-spanning-tree.ts:216` in the same directory.
- **Why it matters.** `maximumRegionCount: 100_000` on a tiny SKATER input requests at least 99,999 step nodes before any device-side convergence decision. The shared command-node limit is 32,768. `C/kernels/gpu-contributor-utils.ts:495`.
- **Proposed change.** Clamp effective SKATER cuts to the number attainable by the input while preserving the documented log layout. Estimate all four contributors’ transitive nodes at construction and use `validateCommandNodeBudget`. Add cheap constructor cases for tiny inputs and oversized budgets. **M. Oversized requests would fail early or use the attainable bound.**
- **Confidence.** High. `KNOWN(PLAT-04)` covers upstream GPU-gated continuation. These contributors can enforce the existing admission limit independently of that work. `dev-docs/roadmap.md:77`.

### PTS-06 A one-zone scan window still triggers an all-pairs nearest-zone search [P1] [performance]

- **Evidence.** The scan admits work by `zones * listLength`. With 65,536 zones and `maximumWindowZones: 1`, list length is two. The builder launches one invocation per zone, then performs four full-zone radix-selection passes per invocation before the final gather. That is **17,179,869,184 separation-key evaluations** before gathering, even with this smallest window. `S/scan-statistics/scan-statistic-parameters.ts:78,84`; `S/scan-statistics/scan-statistic-nearest-zones.ts:45,73,78,101`; `S/scan-statistics/gpu-spatial-scan-statistic.ts:394`.
- **Why it matters.** The memory check admits a configuration whose candidate-discovery cost is quadratic. Reducing the public window limit gives the caller little control over that cost.
- **Proposed change.** Add an indexed exact-nearest route for small windows, preserving `(separation key, zone ID)` order and the extra lookahead zone. Give the full-scan route an explicit work estimate. Verify planar ties and great-circle boundaries against the current implementation. **M–L, additive route.**
- **Confidence.** High on the operation count. Device timing **UNVERIFIED**. `KNOWN(GPU-05)` is under-scoped here: its remaining-work description names large-radius work and snapping, whereas this cost occurs at the smallest scan window. `dev-docs/roadmap.md:429`.

### PTS-07 Poisson scan significance depends on the units of the baseline [P1] [correctness]

- **Evidence.** The API accepts finite nonnegative baselines “with any total scale.” The scoring kernel computes `windowWeight * totalCases / totalWeight` and the published expected count repeats that order. The existing two-zone fixture uses cases `[10,0]`, baseline `[1,1]`, and expects `10 log(2)`. Scaling both baseline entries to `1e38` keeps their float32 sum finite, but `Math.fround(Math.fround(1e38) * 10)` overflows. The correct expected count is still `5`. `S/scan-statistics/gpu-spatial-scan-statistic.ts:168,803,808,1514`; `T/scan-statistics/scan-overflow-review.spec.ts:73,80,108`.
- **Why it matters.** A change of units can put a valid cluster through the zero-likelihood branch. The existing oracle already gives the expected result without requiring another statistical package.
- **Proposed change.** Form a scaled baseline fraction before multiplying by total cases, and reuse it for both scoring and publication. Normalize before accumulation as well if supporting the documented full scale range. Parameterize the existing fixture over baseline scales. **M, nonbreaking.**
- **Confidence.** High on the arithmetic counterexample. Exact GPU output **UNVERIFIED**.

### PTS-08 Early-stopped permutation tests can describe 10 draws as 150 [P1] [ergonomics]

- **Evidence.** `GPUInferenceMetadata` describes how a result’s p-values were produced. Its `permutationCount` is populated from `Math.min(settings.permutationCount, maximumPermutations)` for Knox, Mantel, and Jacquez. The stopping fixture requests 150, sets a budget of 191, and expects exactly 10 draws. `C/contracts/inference.ts:30,34`; `S/space-time-tests/space-time-parameters.ts:187`; `S/space-time-tests/gpu-knox-test.ts:257`, `gpu-mantel-test.ts:297`, `gpu-jacquez-test.ts:241`; `T/space-time-tests/gpu-space-time-stopping.spec.ts:186,191`.
- **Why it matters.** A legend or exported methods record built through the documented metadata API can misstate the amount of inference supporting the p-value. The executed count already exists on the output.
- **Proposed change.** Expose `requestedPermutationCount` and the stopping rule in settings-derived metadata. Add a result decoder taking the published count for the final inference record, with no implicit readback. Assert the final metadata in the existing 150-to-10 fixture. **S–M, additive with a documented transition for `permutationCount`.**
- **Confidence.** High.

### PTS-09 Huff raises attractiveness to a power before normalizing the row [P1] [correctness]

- **Evidence.** The parameter packer accepts any positive finite float32 exponent. The shader stores `pow(value, exponent)` in a float32 array, sums weighted powers, then divides. For equal weights, attractiveness `[10000,20000]`, and exponent `10`, both powers exceed float32 range. The mathematical probabilities are well conditioned: `[1/1025, 1024/1025]`. `S/catchment-accessibility/gpu-huff-trade-areas.ts:57,73,279,307,333,361,394`.
- **Why it matters.** The probabilities, winning facility, and expected-demand calculation all consume the same unnormalized intermediates. Expressing attractiveness in different units can affect all three outputs.
- **Proposed change.** Calculate row shares from `exponent * log(attractiveness) + log(weight)` with the row maximum subtracted before exponentiation. Reuse the normalization for reverse demand aggregation and preserve zero-weight behavior. Add a scale-invariance fixture checking the winner and expected demand as well as probabilities. **M, nonbreaking.**
- **Confidence.** High on the range failure. Specific GPU values **UNVERIFIED**.

### PTS-10 Max-p calls successful improvement rounds a move count [P2] [ergonomics]

- **Evidence.** Max-p publishes status word 5 as `improvementMoveCount` by copying AZP status word 0. AZP documents that word as the number of rounds applying at least one move. Its CPU spec explicitly exercises two independent moves in one round. `S/spatial-regionalization/gpu-max-p-regions.ts:185,1328,1340`; `S/spatial-regionalization/gpu-azp-regions.ts:236`; `T/spatial-regionalization/cpu-azp-regions-operation.node.spec.ts:41`.
- **Why it matters.** Consumers profiling progress or comparing starts can interpret one reported unit as one reassigned row. Batched moves make those quantities materially different.
- **Proposed change.** Add `improvementIterationCount` as the preferred name at packed offset 5 and deprecate the old name. Add a separate row-move counter only if a consumer needs it. **S, nonbreaking.**
- **Confidence.** High.

### PTS-11 Hawkes publishes a likelihood and fitted parameters from different EM iterates [P2] [ergonomics]

- **Evidence.** The shader computes `S_LOG_LIKELIHOOD` at the entering parameters, writes four updated parameters, and then checks `change <= tolerance`. The per-event probabilities are documented at the published estimates. The CPU twin explicitly documents the entering-iterate likelihood. `S/space-time-hawkes/gpu-space-time-hawkes.ts:111,117,642,665,669`; `S/space-time-hawkes/cpu-space-time-hawkes-operation.ts:54`.
- **Why it matters.** With a one-iteration budget, the caller can receive updated parameters alongside the likelihood of the starting guess. A nonzero convergence tolerance also permits a difference at successful termination. That pairing is awkward for comparing fits.
- **Proposed change.** Offer `fittedLogLikelihood` at the published estimates and document legacy `logLikelihood` as the entering-iterate diagnostic. Reuse the final responsibility evaluation when both fitted diagnostics are requested. Test a one-iteration fit with a visible parameter change. **M, additive.**
- **Confidence.** High on the chosen iterates.

### PTS-12 Two roadmap descriptions would send implementers after the wrong work [P2] [docs]

- **Evidence.** The RBF row says degree above one is unimplemented, while local RBF supports degree two and has an independent SciPy quintic/degree-two fixture. The REDCAP row calls the implementation first-order single linkage without identifying its different partition step. The implementation documents a six-node example where its cut yields within-region SSD `13.26` and the SSD partition yields `5.85`. `dev-docs/roadmap.md:141,400`; `S/spatial-interpolation/gpu-local-rbf.ts:89`; `T/spatial-interpolation/local-rbf-coordinate-scale.spec.ts:70`; `S/redcap-regions/gpu-redcap-single-linkage.ts:137,143`.
- **Why it matters.** A roadmap reader could duplicate shipped RBF work or select the existing REDCAP class expecting SSD cuts.
- **Proposed change.** Record degree two as shipped and retain the actual global-RBF/kriging gaps. Describe REDCAP’s tree and cut separately and point SSD users to the documented MST-to-SKATER route. **S, documentation only.**
- **Confidence.** High. `KNOWN(IN05 / IN02)` and `KNOWN(STX-08)` are stale in these specific respects.

## Cross-cutting patterns

- **Convergence needs an output-level definition.** For Sinkhorn it must cover all components of the reported scalar. For Hawkes the iterate attached to each diagnostic matters. K-means already estimates graph size, gates the expensive work after convergence, and publishes a convergence flag. A short per-contributor table covering *what converged, which iterate is published, what is counted, and what happens at the cap* would make the differences usable. `S/spatial-clustering/gpu-spatial-kmeans.ts:109,294,518,1438`; see PTS-01, PTS-05, PTS-10, and PTS-11.
- **Missing labels are more uniform than valid labels.** DBSCAN, HDBSCAN, and k-means document compact or fixed cluster IDs and `0xffffffff` for noise. REDCAP emits the smallest member row as the region label. Partition evaluation ignores labels at or above `labelCapacity`. A REDCAP result `[0,0,2,2]` has two regions, but passing `labelCapacity: 2` excludes the second. Document capacity as *one past the largest accepted label* and identify representative-row encodings on producer outputs. `S/spatial-clustering/gpu-spatial-clustering.ts:81`; `S/spatial-clustering/gpu-hierarchical-density-clustering.ts:109`; `S/spatial-clustering/gpu-spatial-kmeans.ts:141`; `S/redcap-regions/gpu-redcap-single-linkage.ts:107`; `S/spatial-regionalization/gpu-region-partition-evaluation.ts:142,146`.
- **Index adoption is substantial, but candidate eligibility is algorithm-specific.** Fixed, adaptive, and space-time KDE and DBSCAN already call `createPointNeighborGridIndexNodes`. The larger remaining work is in traversal, tie preservation, and filtering usable observations before top-k selection. The RBF compatibility note gives a concrete case where indexing raw positions changes the answer. `KNOWN(PLAT-03)` should distinguish builder adoption from traversal work. `S/point-density/gpu-kernel-density.ts:889`, `gpu-adaptive-bandwidth-kernel-density.ts:372`, `gpu-space-time-kernel-density.ts:544`; `S/spatial-clustering/gpu-spatial-clustering.ts:631`; `S/spatial-interpolation/kriging-compatibility.md:41`; `dev-docs/roadmap.md:406`.
- **The sampled stochastic paths use registered Philox purposes.** K-means names its seeding purpose and documents the `(row, round)` counter. Max-p names its seed-order purpose. Jacquez reuses the Knox shuffle purpose. The census supports retaining this mechanism. It does not establish independence of every pair of streams. `S/spatial-clustering/gpu-spatial-kmeans.ts:801,813`; `S/spatial-regionalization/gpu-max-p-regions.ts:701`; `S/space-time-tests/gpu-jacquez-test.ts:327`.

## Keep

- **Keep the independent mathematical oracles.** The Sinkhorn dense-primal fixture, SciPy RBF fixture, and spacing oracle searching the full 200,000-point population test properties that a CPU/GPU translation can get wrong in the same way. The highest-value additions are asymmetric inputs, changes of units, and repeated submissions through existing rigs. `T/optimal-transport/cpu-sinkhorn-divergence-operation.node.spec.ts:48`; `T/spatial-interpolation/local-rbf-coordinate-scale.spec.ts:70`; `T/analysis-spacing/gpu-analysis-spacing.spec.ts:46`.
- **Keep separate public kriging classes.** Raster kriging coalesces duplicates and gives exact hits zero variance. Local covariance kriging preserves individual observations and can smooth an exact hit with measurement error. The compatibility table tells a caller which statistical model they are selecting. I would use that format for the KDE and clustering families before merging any public classes. `S/spatial-interpolation/kriging-compatibility.md:14`.
- **Copy HDBSCAN’s pairing of construction bounds and execution gates.** It validates its graph estimate and separately gates work during execution. Regionalizers can adopt the former immediately. `S/spatial-clustering/gpu-hierarchical-density-clustering.ts:348,354,556`.

## Future

1. **Add a small contract fixture for optional outputs and graph reuse.** Run each selected scalar port alone against a prefilled view, then resubmit with fewer live rows or edges. Start with Huff, REDCAP, and cross-correlation. This adds device-test time but catches a class of stale-output defects that numerical oracles on fresh buffers miss. See PTS-03 and PTS-04.
2. **Make candidate selection reusable at the level of its semantics.** Share indexed nearest/radius traversal where the caller can specify eligibility, distance ordering, tie order, and overflow behavior. Keep method-specific classes for fixed KDE, adaptive KDE, and space-time KDE. The trade-off is a more explicit internal interface in exchange for fewer independent full scans and fewer accidental changes in statistical meaning. `S/spatial-interpolation/kriging-compatibility.md:41`; `S/scan-statistics/scan-statistic-nearest-zones.ts:56`.
3. **Split the spatial scan implementation along its existing numerical boundaries.** Keep one public contributor, with internal units for candidate discovery, null draws, likelihood evaluation, and winner publication. Give observed and published expected counts a shared calculation. The immediate cost is moving shader assembly and fixtures; the benefit is that scale, boundary, and tie tests can exercise each calculation directly. `S/scan-statistics/gpu-spatial-scan-statistic.ts:803,1107,1507,1514`.
4. **Define a conditioning-set contract for the Vecchia family.** Both GPU APIs describe `neighborCount` as the largest conditioning set. If a caller supplies a wider CSR, the likelihood path delegates to kriging’s nearest-k retention, Fisher scoring takes the first k usable slots, and the CPU twin uses all usable slots. I would document over-width rows as a precondition violation and expose a diagnostic, or deliberately support them through one shared selection rule. A diagnostic costs a check; a shared selection rule may change results for callers already exceeding the declared width. I did not classify this as a valid-input correctness defect. `S/spatial-interpolation/gpu-vecchia-log-likelihood.ts:65,69,168`; `S/spatial-interpolation/gpu-local-kriging.ts:218`; `S/spatial-interpolation/gpu-vecchia-fisher-scoring.ts:85,276`; `S/spatial-interpolation/cpu-vecchia-fisher-scoring-operation.ts:45`.

## Top 5

1. Gate Sinkhorn convergence on all three solves and land the independent two-cell regression.
2. Remove the finite reachability sentinel from location-allocation and assert identical dense/CSR answers.
3. Give every promised Huff probability slot a writer on graph reuse.
4. Implement the independently bindable cross-correlation and REDCAP status outputs and test each alone.
5. Bound regionalizer graph construction before expanding caller-supplied round budgets.

## Questions for the lead

- Can the contract checker or a shared test fixture cover independently optional outputs across streams? The two silent scalar omissions and the Huff tail have small, reusable reproductions.
- Does the inference/catalog stream already distinguish requested, maximum, and executed permutation counts? Knox, Mantel, and Jacquez should use the same result metadata convention.
- Should an over-width Vecchia CSR be rejected, or should it have defined truncation semantics? That decision determines whether the next change belongs in input diagnostics or in a shared conditioning-set selector.