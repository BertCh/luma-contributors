# Consumer-side review of the 2026-11-10/11 wave 2 changes

Read-only review. Baseline: `scratchpad/pre-w2` (same layout). The scripts are in `scratchpad/reviews-w2/s/`:

- `cmp2.cjs`: compares decoded captures for every scenario. It expands `rle:` and `same:` before comparing. Its output is `diffs2.json` and `summary2.txt`.
- `dump.cjs`, `maxdiff.cjs`: per-record dumps.
- `imports.cjs`: checks catalog example imports against the entry barrels.

Problems are listed most serious first. "Confirmed" means I read both the code and the text, or decoded the values. "Suspected" means I inferred it.

## Confirmed

### C1. The GPUMarkedPointStatistics lens formula is stale, and its own re-captured record contradicts it

- **Where:** `modules/lenses/src/gpu-dataframe/computation-pack/operators/pair-statistics.ts:413-416`.
- **What the lens says:** the tex is `K(r) = A/(n_a n_b) Σ_{i∈a, j∈b, i≠j} …` and the ascii says "cross-type K".
- **What the kernel does now:** it normalises a same-type set by `area / (n (n - 1))` (`gpu-marked-point-statistics.ts:92`, `:262-268`). This is behaviour-changes row "same-type K grows by n/(n-1)".
- **Evidence in the record:** in `gpu-dataframe.marked-point-statistics`, `result` changed in 6 of 8 entries, from 13.6719 / 2.08612 / 1.08797 to 18.2292 / 2.40884 / 1.45063. That is exactly ×4/3, so the scenario is same-type with n = 4. The formula shown beside it gives the old number.
- **What a reader sees:** the lens page, its twins and the website view model all show a formula that does not reproduce the displayed value.
- **Status:** this text was not among the 15 rewritten descriptions.

### C2. The GPUNetworkAnalyticsColumns lens text states the old degree rule

- **Where:** `modules/lenses/src/gpu-network/computation-pack/network-pack-structure.ts:14` (`gpu-network.analytics.normalize`).
- **What the lens says:** "The raw degree is the outgoing CSR row length."
- **What the kernel does now:** it counts live slots only, i.e. neighbours `< NODE_COUNT` (`gpu-network-analytics-columns.ts:72`, `:299-326`). This is network-paths-06: a row `[1, 0xffffffff, 7]` had degree 3 and now has degree 1.

### C3. The space-time variogram run now draws only 7 of its 9 tick segments

- **Where:** `website/src/gpu/runs/entries/dataframe-pointcloud-records/space-time-variogram.ts`, the draw layer `{layer: 'lines', …, count: 7}`.
- **Cause:** the fix changed the tick loops to `COLUMNS = T + 1` and `ROWS = L + 1`. They now emit 5 time ticks (lags 0, 2, 4, 6, 8) and 4 distance ticks (bins 4, 8, 12, 16), which is 9 segments. The old count, 4 + 3 = 7, was kept.
- **What a reader sees:** the last time tick and the bin-16 distance tick are missing.
- **The rest of the fix is right:** the parameter layout is 8 floats with `bounds`, the output is `(L + 1) × (T + 1)` spatial-major (`gpu-space-time-variogram.ts:109-111`, `:386`), and the caption mentions the zero-lag row and column.

### C4. Catalog: three line contributors still describe `status` with the removed `lineCount` and say `count` is vertices

- **Where:** `website/src/catalog/entries/geometry-construction-overlay.ts:1599`, `:1766`, `:1855` (line-segmentize, line-smooth, line-chunk).
- **What the catalog says:** type `GPUCardinalityPort & GPUOverflowStatusPort & {lineCount?}`, description "Written vertex count and overflow flag."
- **What the library does now:** `status.count` counts lines and `status.vertexCount` counts vertices. For example, `gpu-line-segmentize.ts:94` says "count lines (the input line count); `vertexCount` … count vertices".
- **The records agree with the library:** line-segmentize `count` went from 1476 to 223, line-smooth from 2052 to 223, line-chunk from 873 to 403, and `lineCount` was removed while `vertexCount` was added.
- **Effect:** a host that follows the catalog reads the line count as a vertex count.

### C5. Catalog defaults that are now wrong

- **GPUTerrainCastShadow, `website/src/catalog/entries/terrain-relief-light.ts:1507-1511`:** `rowDirection` default `'south'`. The class now defaults to the sign of `grid.affine[4]` and throws on a contradiction (behaviour-changes item 1 lists GPUTerrainCastShadow). The derivatives entry at `:76` was fixed; this one was missed.
- **GPUDatumGridTransform, `website/src/catalog/entries/dataframe-pointcloud-records.ts:2962-2966`:** `tolerance` default `'1e-5'`. The library default is 5e-9 degrees (`gpu-datum-grid-transform.ts:49`, geometry-measures-05). A reader who copies 1e-5 gets back the old behaviour of skipping shifts below 1.1 m.

### C6. The lens record recapture is wrongly described in `w2-lenses-records.md` §1 item 5

- **"110 of the 410 old records have changed captures": not true of the values.**
  - After decoding `rle:` and `same:`, 73 scenarios differ, plus the k-means rename.
  - 3 of the 73 only gained a `parameters` snapshot: raster-fractional-coverage, raster-fractional-zonal-statistics and raster-walking-cost.
  - Of the rest, 6 changed only output names (rename only). 5 differ only in unused capacity past the written count. The remaining 59 changed written values.
  - The 37 extra ids in `captures-changed.txt` have identical decoded values. Examples: `raster-reconstruction`, `kernel-density`, `terrain-spike-repair`, `network-coarsening`, `make-valid` and `street-profiles-recipe`.
  - The author's list therefore probably counts changes to `same:` reference indices or to run-length segments.
  - **Consequence:** `numeric-captions.txt` (135 captions "to check") is built on that over-count.
- **The baseline predates the math pass.** 406 of the 410 old records carry `recordedAt` 2026-11-07, which is before commit 93228e74 ("Library convergence pass", 11-10) and before the whole 11-10 math pass. Changes from 93228e74 are therefore mixed in, and neither `behaviour-changes.md` nor the wave 2 reports cover them. See S3.
- **"The old set is a subset of the new one" is false as written.** `gpu-spatial-analysis.k-means` is gone and `spatial-k-means` replaced it. That is intended, and `website/src/catalog/entries/spatial-statistics.ts:1899` was updated. No other scenario was lost; the two EC03 ids are new.

### C7. The re-captured records now carry GPU timings, so a recapture is no longer byte-reproducible

- **Before:** 135 records had `metrics`, with `timestamped` false.
- **Now:** 182 records carry `timestamped: true` with per-node `gpuBegin`, `gpuEnd` and `cpuEncodeMs` in ms, all on network and raster records.
- **Example:** `docs/lens-records/gpu-network.network-flow.json` has `"gpuEnd":0.0588…`.
- **Documentation:** the report mentions this, but I found no record-format document that declares these fields as allowed non-reproducible content.
- **The other checks are clean:**
  - `adapter` is `apple / metal-3`, as before.
  - No absolute paths, user names, `file://` or `localhost` appear in any record or twin.
  - The capture tails of 5 records (drainage-network-recipe, buffer-surface, building-tessellation-recipe, shared-paths, region-partition-evaluation) differ only past the written count. That is stale pool memory: the new drainage `lengths[4..8]` hold the old `vertexOffsets` integers as denormals. It was already non-deterministic before and is not a regression.

### C8. Minor stale lens text

These were not rewritten.

- **Raster conditional, `modules/lenses/src/gpu-raster/computation-pack/operators/algebra.ts:81` ascii:** says "output = chosen if the condition value and chosen branch are neither NaN nor the optional noDataValue". The kernel now never applies the sentinel to per-frame constants (`gpu-raster-conditional.ts:53-54`, `:176`; raster-imagery-01). The lens values (0.4 and 1, sentinel 0) are unaffected; the rule is wrong.
- **Spatial empirical Bayes rates, `…/statistics-summaries.ts` ascii:** says "only the structure of the weights counts". Zero-weight slots are no longer neighbours, and self slots count once (`gpu-spatial-empirical-bayes-rates.ts:74-76`, `:109`).
- **Variogram doc comment, `modules/lenses/src/gpu-dataframe/computation-pack/operators/pair-statistics.ts:18`:** gives the robust divisor as `0.457 + 0.494/N`. The kernel adds `0.045/N²` (`gpu-variogram.ts:497`; dataframe-05).
- **Shape generator kernel comment, `modules/analysis/src/gpu-spatial-analysis/grid-generators/gpu-shape-generator.ts:463`:** says "The shared half-angle destination: asin(sin(lat) cos(d) + cos(lat) sin(d) cos(b))…". It names the old form in front of the new call. This is not lens text, but readers of the bound source see it.

## Suspected

### S1. Kepler `addKeplerHotSpots`: correct, and it cannot fire on a default

- **The condition matches the recipe exactly.**
  - `selfWeight ?? 1`.
  - The default statistic is `localG` when the self weight is 0, else `localGStar`.
  - `localGStar` requires 1, `localG` requires 0, and other statistics pass. This matches `analysis-bindings.ts:126-137` against `hot-spot-analysis-recipe.ts:574-585`.
- **It throws before the recipe adds any node.**
- **No current default reaches it:**
  - The only caller, `recipe-bindings.ts:813`, passes no `permutation`.
  - `point-hotspots.ts` uses GPUHotSpotAnalysis directly.
  - No parser defaults `selfWeight` to a value other than 1.
- **Edge case:** a host that sets `permutation.statistic: 'localG'` and leaves `selfWeight` out gets 1 and throws. That is the recipe's rule too.
- **Note:** in `recipe-bindings.ts:807` a GPUParameterBuffer is created before the call. I did not verify that the surrounding `catch` destroys it on a throw; it is not reachable today.
- **Verdict:** no problem found.

### S2. The point-cloud row-direction fix looks right

- `PointCloudRasterGrid` now defaults to `'south'` (row 0 at max y), matching the top-down raster drawing. Removing `flipColumns` in `pointcloud-surfaces.ts` and `pointcloud-smrf.ts` is consistent.
- `pointcloud-height.ts` and `pointcloud-ground.ts` pass the same `createCloudGrid` (now explicit `'south'`) to every contributor.
- **Side effect:** the south index treats `y == origin.y` as outside and `y == top` as inside (`pointcloud-grid-index.ts:120-127`). A point on the bottom edge of the cloud drops out. This is library behaviour, but the runs' clouds touch `y = 0`.
- **Not verified:** whether any visible point sits exactly on that edge.

### S3. Records that changed with no stated reason

All are small, at last-digit level. They probably come from 93228e74 (see C6) or from shared helpers, but no report names them. Values are old → new at the largest difference.

| Record | Output | Index | Old | New | Relative change |
| --- | --- | --- | --- | --- | --- |
| `geographically-weighted-regression` | interceptCoefficient | 555 | 9.245169 | 9.245161 | 8e-7 |
| `geographically-weighted-regression` | noVehiclePctCoefficient | 204 | −0.0219324 | −0.0219328 | 1.8e-5 |
| `geographically-weighted-regression-nonstationarity-test` | zSimulated | 0 | 12.947428 | 12.947433 | — |
| `spatial-scan-statistic` | zoneStatistics | 12 | 342.80853 | 342.80939 | 2.5e-6 |
| `spatial-scan-statistic` | statistics | 0 | 500.66739 | 500.66791 | — |
| `probit-regression` | diagnosticTests | 0 | 3.5325925 | 3.5325902 | — |
| `trajectory-metrics` | trackLengths | 4 | 25458.6074 | 25458.6055 | — |
| `fleet-dwell-recipe` | stopCentroids | 7 | 3711.68408 | 3711.68359 | — |
| `group-statistics` | variances | 16 | 77.334190 | 77.334175 | — |

Notes:

- The GWR rows matter because behaviour-changes lists only `localConditionNumber` for GWR, and `w2-adoption` says the shared dense solve gives "identical results". Across GWR, the largest relative change, 6e-4, is on near-zero coefficients. `localR2` also changed.
- For group-statistics, behaviour-changes lists only means, sums and z-scores, not `variances`.
- The probit row is not in the regression-16 list.

The larger changes can all be traced to stated causes:

- **Emerging hot spots** (one location, row 19): trend S −21 → −22, z −1.381 → −1.447, p 0.167 → 0.148. This is plausibly a Gi tie broken by spatial-statistics-07.
- **Restricted regime** Chow F: 2.002e-4 → 1.959e-4 (regression-02).
- **Panel lag:** iterations 3 → 2.
- **Drive-time:** demand 0 rose from 79.49 to 231.79 s, which is one-way honouring (network-flows-03). Increases are not listed in behaviour-changes, as the author noted.
- **Coverage simplification:** `[4,0,10,0]` → `[6,0,10,0]`.
- **Terrain:** exactly the 251 border cells of a 64 × 64 grid changed.
- **Seeded draws:** k-means, K envelope, folds and the nested validation.
- **p floors:** the spatial correlogram's 8 `pValues` are all 1.17549e-38. That is constant and at the floor, but it is the documented `2^-126` floor (all z > 14), not a failure.

No output became all-NaN, empty or zero in the written range.

## Checks that passed

- **Shape generator formula** (`geometry-formulas.ts:297-…`). It matches `geodesicDestination` (`geodesic-wgsl.ts:168-193`), including:
  - the half-angle `Δs`, the clamp, and `cos((φ + asin(sin φ'))/2)`;
  - the fallback when that cosine is at most 1e-3;
  - `atan2` with `sinPhi2`, and `p' = c + 180/π (Δλ, Δφ)`;
  - the units: `δ = d/R` in metres over the radius, and the bearing converted to degrees and back.
  - Only the tex omits the fallback branch and the inner clamp.
- **SKATER gain** (`statistics-regions.ts:209-226`). It is exactly `count*remainder*squaredDifference/regionCount` with `difference = childSum/count − (regionSum − childSum)/remainder` (`gpu-skater-regions.ts:780-813`). The size guard and the lowest-edge tie (`atomicMin(min(index, twin))`) match the text. The non-candidate rule for NaN or Inf matches. The kernel also clamps the gain at ≥ 0 before keying, which the text does not mention.
- **The 15 rewritten descriptions match their kernels:**
  - terrain one-sided border, `(-3f0+4f1-f2)/(2h)` and "clamp halves";
  - curvature one-sided;
  - Louvain Philox Fisher-Yates;
  - cross-validation rank mod foldCount;
  - coarsening 64-bit sums;
  - catchment `t ≤ B[k]` with `(a, b]`, per-direction `cost ≥ 0`, the same-road `ahead` cost and the +Infinity for unreached demand;
  - isochrone `(a, b]` via next-float breaks, and max mode taking the cheaper direction;
  - coverage simplification swept spans;
  - reconstruction 8-connectivity;
  - the isolines bilinear decider;
  - composite score closed form for d = 2, otherwise Jacobi;
  - cell fallback accuracy;
  - conditional simulation with an open uniform on (0, 1).
  - The reachability rule cited by the catchment text (`count of thresholds < cost`) is also `(a, b]`.
- **Twins and view models.** 412 records, 412 view models and twins for 412 scenario ids (1,855 files). Ids match one to one, with no stale k-means twin. Every twin carries its record's `runId`. View models carry the new values: drive-time 231.79, marked point 18.229, nested validation 0.5043. The new catchment ascii is in the drive-time view model.
- **Source text.** 405 records carry it. It is off only for the 2 named terrain scenarios (42 and 36 sources without text) and absent in 5 that have no sources. The two EC03 records converge: raster `[1, 17, 1, 0]` within the 64 bound, sparse `[1, 4, 1, 0]`.
- **Resistance run.** It uses source and sink as cell indices in the recipe's layout. It reads `diagnostics[0]`, the effective resistance, only when `solverStatus[0] == 1` and only for the applied patch. `source` and `neighborhood` are compile-time and applied through `settle`, so a moving pointer only re-picks a patch. The barrier resistance is written per frame. The caption and the catalog say "Modeled scenario", and the catalog's conductance wording matches `2/(Ra+Rb)/edgeLength`.
- **Walking cost.** It packs `[a, b, d, e, N, S, frictionWeight, 0, c0..c3, threshold]` per frame with the same model, and `frictionWeight` was removed from `rebuildOn`.
- **Outlier flags.** The header, caption and catalog summary now state the displaced-fix rule. The wording "rises by more than maximumAcceleration" mixes speed with acceleration but is otherwise right.
- **Rectangle clip and shared lines.** They map `count` to lines and `vertexCount` to vertices, and the readout "vertices kept" reads `vertexCount`.
- **Sampled website captions and readouts:** relief-blend, slope-aspect-hillshade, terrain-curvature, Louvain, network K, coarsening, drive-time, isochrones, contrast-stretch, utilization isopleths, cross-validation, modal split, marked point and analytics columns. None states an old number or rule.
- **Catalog example imports.** All 463 named imports in catalog `example` strings resolve against the entry barrels (`imports.cjs`). No `GPUKMeans`, `getGPUKMeansParameterValues` or `GPU_KMEANS` remains outside generated files. Every `symbols: [...]` name exists.
- **Pre-existing, not from today:** the catalog names types that the library never exported: `GPUDrainageNetworkOutput`, `GPUSphericalPolygonOverlayOutput`, `GPURecipeIntermediates` and `GPURecipeStatus`.

## Not checked without running code

- Whether the view models were generated from a stale lenses `dist`, as the author warns. The values match the records, but I could not check whether any pack text in the view models comes from the old `dist`.
- The pictures: no screenshots were read. This covers the tick omission in C3, the point-cloud orientation, the pointer behaviour of the resistance run, and whether the lake's NaN cells render transparent.
- Whether the kepler throw path in `recipe-bindings.ts` destroys the parameter buffer.
- That the S3 deltas really come from 93228e74 and not from a defect; I would need the 11-10 tree to re-capture.
- That the two terrain records without source text say so on the website.
- The other 125 numeric doc captions in `numeric-captions.txt`.
