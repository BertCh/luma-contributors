# Exploratory and descriptive spatiotemporal statistics, 2022-2026: what to add next

Research date: 2026-11-09. This report covers exploratory and descriptive space-time statistics for a
browser WebGPU (WGSL) analysis library that is float32 only, uses bounded loops, and has every operation
pinned to a reference implementation.

**Citation rule.** Each claim carries the URL of a page that was actually fetched. PDFs were downloaded
and read with `pdftotext`. Anything not confirmed from a fetched page is marked **UNVERIFIED**.

## 0. What the library already has (checked in the source tree)

I grepped `modules/analysis/src/gpu-spatial-analysis` so as not to recommend work that already exists:

- `space-time-tests/`: the global Knox test (one spatial pair list, one time threshold, a Baker-modified
  Monte Carlo reference, swap-or-not permutations, Besag-Clifford stopping) and the Mantel test. There is
  no Jacquez test, no multi-band near-repeat table, no local Knox, and no Diggle space-time K test.
- `scan-statistics/`: circular kNN zones with Poisson, Bernoulli and space-time-permutation models, up to
  32 time buckets, at most 64 clusters and at most 99,999 replicates. There is no expectation-based or
  Bayesian scan, no subset scan and no flexible scan.
- `spatial-autocorrelation/`: local bivariate Moran (`localBivariateMoran`), local join count and
  bivariate local join count (the GeoDa colocation map), and spatial cross-correlation.
- `multiple-testing/`: Holm, Benjamini-Hochberg and Benjamini-Yekutieli.
- `point-process/`: inhomogeneous point pattern statistics and pointwise and rank envelopes. The spatial
  Ripley and pair-correlation functions live in `dataframe-analysis/pair-statistics/gpu-ripley.ts`.
- `spatial-clustering/`: DBSCAN, OPTICS, OPTICS-xi, HDBSCAN and k-means.
- `emerging-hot-spots/`, `distribution-dynamics/` and `trajectory-*`.

Nothing in the source matches space-time K, STKDE, ST-DBSCAN, CUSUM, change points, PELT, LISTA,
colocation mining or time-series clustering.

---

## 1. Ranked candidates (value to an analyst relative to effort)

| Rank | Candidate | Effort | Risk | Main reuse |
|---|---|---|---|---|
| 1 | Near-repeat Knox table, local Knox and Jacquez kNN test | S | Low | Knox pair list, permutation engine, kNN search, sort |
| 2 | Gridded space-time kernel density (STKDE) and space-time relative risk | S-M | Low | point-density KDE, grid index, sort by time |
| 3 | Space-time K and pair correlation (STIKhat/PCFhat), Diggle D(s,t) test, LISTA | M | Medium (edge correction, float32 sums) | gpu-ripley, inhomogeneous K, envelopes, Knox permutations |
| 4 | Per-location change-point detection (exact optimal partitioning ≡ PELT; SegNeigh) | M | Low-Medium | temporal reductions, cube layout |
| 5 | Expectation-based space-time scan and Fast Subset Scan (LTSS), with Bayesian Fast Subset Sums as a follow-up | M | Medium | scan kNN zone tables, workgroup sort, Philox, harmonic baseline |
| 6 | ST-DBSCAN / ST-OPTICS / ST-HDBSCAN (time-filtered neighbour graph) | S | Low (semantics must match oracle) | existing DBSCAN/OPTICS/HDBSCAN |
| 7 | Time-series cross-correlation per location (lags, prewhitening, neighbour averaging) | S | Low | weights, temporal reductions |
| 8 | Differential and lagged bivariate local Moran as recipes | S | Low | localBivariateMoran, local Moran |
| 9 | Spatially adaptive FDR (LAWS/STRAW) for local space-time p-values | M | Medium (no pinned code) | BH sort, kernel smoothing over weights |
| 10 | Time-series clustering of cube locations (k-means on values, k-medoids on correlation, Fourier), optionally spatially constrained | M | Medium | k-means, AZP/regionalization |
| 11 | Prospective point surveillance: Shiryaev-Roberts `stcd`, prospective local Knox | M-L | High | Knox, sort |
| 12 | Nonparametric self-exciting triggering histogram (MISD / Marsan-Lengliné EM) | M-L | High (no pinned oracle) | pair lists, fixed-count EM |

The detailed cards follow in rank order.

---

### 1. Near-repeat Knox table, local Knox and Jacquez kNN test (S)

**Questions answered.** Near repeat asks whether events that follow an event are over-represented at
each combination of spatial band and temporal band. This is the standard crime-science near-repeat
table. The Jacquez test asks whether events are k-nearest neighbours in space and in time more often
than chance. Local Knox scores which events drive the interaction.

**Sources.**
- The NearRepeat R package uses the signature
  `NearRepeat(x, y, time, sds, tds, ..., method = "manhattan", nrep = 999)`. It returns observed pair
  counts per (spatial band, temporal band), Knox ratios against the mean and the median of the
  simulations, and p-values. It is GPL-3, version 0.1.1, and is on GitHub only, not CRAN
  (https://rdrr.io/github/wsteenbeek/NearRepeat/man/NearRepeat.html,
  https://rdrr.io/github/wsteenbeek/NearRepeat).
- pointpats 2.6.0 lists `Knox`, `KnoxLocal`, `mantel`, `jacquez` and `modified_knox` under space-time
  interaction tests (https://pysal.org/pointpats/api.html). It is BSD-3 as part of PySAL
  (https://zenodo.org/records/18446136).
- The Jacquez test statistic counts pairs that are k-nearest neighbours in both space and time, and its
  significance comes from randomization. Malizia and Mack (2012, Stat Med) note that it is susceptible
  to population-shift bias. Both points come from the search summary only. The Jacquez (1996) paper
  itself was not fetched: **UNVERIFIED**.
- The adjusted Knox for heterogeneous risk is Briz-Redón, Martínez-Ruiz and Montes (2022), European
  Journal of Criminology 19(4). Per the abstract seen in search, the adjustment reduces the number of
  significant intervals. The full text was paywalled and the record page served only a bot challenge:
  **UNVERIFIED** beyond the abstract.
- Prospective local Knox: Rogerson's (2001) local Knox CUSUM was criticised by Marshall et al. (2007).
  Piroutek et al. (2014) fixed it by counting only past events and calibrating by randomization
  (Allévius and Höhle, survey chapter, read from the PDF text: https://arxiv.org/pdf/1711.08960).

**Data-parallel formulation.**
- *Pair histogram.* Enumerate spatial pairs within `max(sds)` once with `GPUNeighborSearch` in radius
  mode. For each pair, the spatial band is fixed. For each permutation, the temporal band is
  `bin(|t_π(i) − t_π(j)|)`. Each (permutation block, pair block) accumulates an `S×T` integer histogram
  in workgroup memory and then writes partials. The counts are exact u32 values, the same design as the
  existing Knox. The output is `P × S × T` counts, from which the Knox ratio and p-value come.
  `S×T ≤ 64` cells fits workgroup memory easily.
- *Manhattan distance* only changes the pair predicate. The grid index must use the L1 radius.
- *Local Knox* is a per-event row histogram, `n × T` (or `n × S × T` when small), built from the same
  pair list.
- *Jacquez.* Spatial kNN comes from the existing kNN search. Temporal kNN is computed on the sorted
  times: kNN in 1-D is a bounded window of `2k` around the rank, so it has no data-dependent loop.
  Precompute the `n × k` temporal neighbour table over time ranks. Under permutation π, the statistic is
  `Σ_i Σ_{j∈kNN_s(i)} [π(j) ∈ kNN_t(π(i))]`. Membership is a bounded scan of `k` entries. Tie handling
  must match pointpats: `jacquez` docs show 12 vs 13 for the same example between doc versions
  (https://pysal.org/pointpats/api.html context, per search), so it must be pinned to an exact
  pointpats version.

**Capacity.** The pair-list limit is the same as the existing Knox, around 10^7 pairs. Jacquez scales
to the kNN index limit, about 10^6 points with `k ≤ 32`.

**Oracles.** NearRepeat (GPL, oracle only) and pointpats 2.6.0 `jacquez`/`KnoxLocal` (BSD-3).

**Risk.** Low. The main traps are the band edge conventions (`include.lowest`/`right` follow R `cut()`)
and ties.

---

### 2. Gridded space-time kernel density and space-time relative risk (S-M)

**Question answered.** Where and when is event intensity high, as a 3-D voxel cube that can be animated
over time? The relative-risk variant asks where the case density exceeds the control density.

**Sources.**
- *Parallel STKDE* (Saule, Panchananam, Hohl, Tang and Delmelle; arXiv 1705.09366, read from the PDF):
  - The voxel-based (VB) gold standard costs Θ(per voxel × points).
  - The point-based (PB) algorithm costs Θ(GxGyGt + n·Hs²·Ht).
  - PB-SYM exploits the separable product kernel `Ks(x,y)·Kt(t)`.
  - With domain decomposition, 16 threads reach a speedup of about 14.9.
  - The authors note that an earlier GPU voxel-based implementation was slower than their sequential PB
    (https://arxiv.org/pdf/1705.09366).
- *GPU STKDE* (Eaglin, Cho and Ribarsky, HICSS 2017, read from the PDF): an oct-tree on the GPU, run on
  a GTX 770. 6.3M voxels took 0.636 s on the GPU vs 3.425 s on the CPU, and 0.387 s with coalescing. One
  million points at 10M voxels took about 0.73-0.94 s. Sequential code took about 23 minutes
  (https://webpages.charlotte.edu/icho1/data/eaglin17.pdf).
- *SWS* (Chan, Ip, U, Choi and Xu, PVLDB 15(4) 2022, read from the PDF): with an Epanechnikov
  (polynomial) temporal kernel, a sliding window of running moment sums cuts STKDV from O(XYTn) to
  O(XY(T+n)) without extra space, a 1.71x-24x speedup (https://www.vldb.org/pvldb/vol15/p814-chan.pdf).
  LIBKDV/Rlibkdv packages exist. Their licences were not checked: **UNVERIFIED**.
- *sparr* 2.3-16 (2025-03-20, GPL ≥2) provides:
  - `spattemp.density` (fixed bandwidth, isotropic spatial smoothing, per Fernando and Hazelton 2014),
    with `sedge`/`tedge` "uniform" edge correction;
  - `spattemp.risk` with time-static or time-varying controls;
  - asymptotic tolerance contours;
  - the bandwidth selectors `OS.`, `NS.`, `LSCV.`, `LIK.` and `BOOT.spattemp`
  (https://cran.r-project.org/web/packages/sparr/refman/sparr.html).

**Data-parallel formulation (gather, because WGSL has no float atomics).**
- Sort points by time, and give each grid cell a time-sorted run of points (the existing grid index plus
  sort).
- One invocation per voxel `(x, y, t_q)`. Visit the grid cells within `hs`. In each cell, binary-search
  the time-sorted run for `[t_q − ht, t_q + ht]`. Sum `Ks·Kt`. Both loops are bounded by compile-time
  caps on cells (from `hs / cellSize`) and points per cell.
- *SWS-style variant for polynomial kernels.* Per pixel, keep the moment sums
  `Σ Ks·{1, t, t²}` over the window. On the GPU this becomes a per-pixel prefix scan along t. Float32
  cancellation in `t²` moments is a real risk: centre t on the window and avoid it unless profiling
  demands it.
- *Edge correction* ("uniform"): divide by the kernel mass inside the window per voxel. Precompute this
  as a separable 2-D mask convolution times a 1-D temporal factor.
- *Relative risk:* two densities and a log ratio. Tolerance contours need the asymptotic variance
  formula from Fernando and Hazelton (2014) (**UNVERIFIED**, paper not fetched).

**Reuse.** Point-density KDE kernels, grid index, sort and temporal reductions.

**Capacity.** About 256×256×64 ≈ 4M voxels, and points up to around 10^6. This is bounded by
points-per-cell caps, so report truncation as other indexed operations do.

**Risk.** Low for fixed bandwidth. Adaptive bandwidth and bandwidth selection (LSCV/LIK) make up the
M part. Bandwidth selection needs repeated evaluations, which is fine with a fixed candidate grid.

---

### 3. Space-time K and pair correlation, the Diggle D(s,t) test and LISTA (M)

**Questions answered.**
- STIK: is there space-time clustering beyond what the first-order intensity explains, and at which
  (u, v) scales?
- The Diggle test: is there space-time interaction, that is, is `K(s,t) ≠ K_s(s)·K_t(t)`?
- LISTA: which events sit in locally anomalous space-time second-order structure?

**Sources.**
- *stpp* 2.0-8 (2024-06-27, GPL-3). Functions:
  - `STIKhat(xyt, s.region, t.region, dist, times, lambda, correction="isotropic", infectious=FALSE)`
  - `PCFhat(..., ks="box", hs, kt="box", ht, correction=...)`
  - `ASTIKhat`, `KLISTAhat` and `LISTAhat`
  - Corrections: isotropic, border, modified.border, translate, none.

  Source: https://cloud.r-project.org/web/packages/stpp/refman/stpp.html
- *stopp* 1.0.0 (2025-07-08, GPL-2|3), D'Angelo and Adelfio, JSS 113(10), 2025. It covers local
  first- and second-order characteristics in Euclidean space and on linear networks, and imports stpp,
  splancs, sparr and stlnpp (https://cran.r-project.org/web/packages/stopp/index.html,
  https://www.jstatsoft.org/article/view/v113i10, https://arxiv.org/abs/2408.15052v1).
- *LISTA local tests:* Siino, Rodríguez-Cortés, Mateu and Adelfio (2018), "Testing for local structure
  in spatiotemporal point pattern data", Environmetrics 29(5-6)
  (https://ideas.repec.org/a/wly/envmet/v29y2018i5-6ne2463.html).
- *Diggle test:* `splancs::stmctest(pts, times, poly, tlimits, s, tt, nsim)` permutes times and uses a
  sum of standardized residuals of D(s,t). Diggle, Chetwynd, Häggkvist and Morris (1995), SMMR 4
  (https://rdrr.io/cran/splancs/man/stmctest.html). The surveillance package wraps it as `stKtest`,
  version 1.26.1 (https://surveillance.r-forge.r-project.org/pkgdown/reference/index.html).
- *Distributed space-time K* (Spark, R-tree index, edge-weight cache): Wang, Gui et al., FGCS 2020
  (https://arxiv.org/abs/1912.04753). No GPU space-time K paper was found (see section 7).

**Data-parallel formulation.**
- *Pair histogram over (u-band, v-band).* For pairs within `(u_max, v_max)`, accumulate
  `w_ij / (λ_i λ_j)` into a `U×V` grid, where `w_ij` is the spatial isotropic or translate edge weight
  times the temporal edge weight.
- *Float sums.* There are no float atomics, so each workgroup reduces its own pair block into a private
  `U×V` grid. Partials are summed in fixed order (a deterministic tree), the same pattern as Knox.
  Compensated (Kahan) summation in the final reduction protects float32.
- *PCF.* Replace the indicator with `ks(u−d_ij)·kt(v−|Δt|)` across a bounded number of bins.
- *Diggle D-test.* `D(s,t) = K(s,t) − K_s(s)K_t(t)`. Under time permutation, `K_s` is fixed and
  `K_t` is fixed (times are only relabelled), so only `K(s,t)` changes. Each permutation is one
  `U×V` integer histogram, exactly the near-repeat kernel from card 1 with cumulative bands.
  This makes it close to free once card 1 exists.
- *LISTA.* Per-event rows `n×U×V`, bounded by `U×V ≤ 64`. The local test compares each event's
  LISTA to the global function via permutation.
- *Envelopes.* Reuse the existing pointwise, rank and DCLF envelopes over simulated Poisson or
  time-permuted patterns.

**Reuse.** gpu-ripley (isotropic edge weights), inhomogeneous K (`λ` at points), envelope operations
and the Knox pair list.

**Capacity.** Brute force works to a few thousand points. Indexed pairs within `u_max` work to about
10^7 pairs.

**Risk.**
- Matching stpp's exact temporal edge correction and `infectious=TRUE` semantics. Pin stpp 2.0-8 and
  run it as the oracle only, because it is GPL.
- Float32 precision in `1/(λλ)` sums when the intensity spans many orders of magnitude. Scale `λ`
  beforehand.

---

### 4. Per-location change-point detection (M)

**Question answered.** At each location of a space-time cube, when did the mean, variance, slope or
Poisson count rate change, and how many times?

**Sources.**
- ArcGIS Pro *Change Point Detection* (added in 2.9; `min_seg_len` and SLOPE added in 3.0)
  (https://pro.arcgis.com/en/pro-app/3.5/tool-reference/space-time-pattern-mining/space-time-pattern-mining-toolbox-history-pro.htm).
  How it works:
  - It runs PELT when the number of change points is automatic, and SegNeigh when the number is
    defined.
  - Costs are Normal likelihoods for mean shift, standard deviation and slope, and Poisson for count.
  - The sensitivity value of 1 corresponds to minimizing BIC. For slope, the default 0.5 corresponds to
    AIC.
  - A minimum segment length applies.
  - It runs independently per location.
  - References: Auger and Lawrence (1989); Killick, Fearnhead and Eckley (2012, JASA); Killick and
    Eckley (2014, JSS `changepoint`)
  (https://doc.esri.com/en/arcgis-pro/latest/tool-reference/space-time-pattern-mining/how-change-point-detection-works.html).
- *Spatial panel change points* (not recommended to implement, see section 3): the Two-Way MOSUM
  (Li, Chen, Wang and Wu; arXiv 2208.13074, Annals of Statistics 2024 per search) aggregates MOSUM by
  the ℓ² norm over spatial-temporal moving regions (https://arxiv.org/abs/2208.13074v2).

**Data-parallel formulation.**
- One workgroup or one invocation per location.
- *Exactness.* PELT is exact optimal partitioning plus pruning, so the unpruned optimal partitioning DP
  `F(t) = min_{s<t} F(s) + C(s+1..t) + β` gives the identical optimum with O(T²) bounded loops. That is
  fine for T ≤ 512.
- *Segment costs.* Each is O(1) from per-location prefix sums of `y`, `y²`, `t·y` and `t²` (and, for
  Poisson, `Σy` and `Σ y·log`).
- *SegNeigh with K change points* is O(K·T²), with K a compile-time cap.
- *Backtracking* is bounded by K or T.
- *Float32.* Use prefix sums relative to a per-location mean, and pairwise or Kahan prefix sums.
- *Penalty.* Map Esri's sensitivity to β on the CPU. The exact formula is shown only as an image on
  that page: **UNVERIFIED**. Expose β directly and validate against R `changepoint::cpt.mean/var/meanvar`
  with `method="PELT"`. The version and licence of the R package were not fetched: **UNVERIFIED**.

**Capacity.** Up to about 10^5-10^6 locations × T ≤ 512. The whole problem is embarrassingly parallel.

**Risk.** Low-medium. Matching the penalty conventions and the variance-change cost to the oracle
needs care.

---

### 5. Expectation-based space-time scan and Fast Subset Scan, then Bayesian Fast Subset Sums (M)

**Question answered.** Which recent space-time region has counts above the expected baseline?
Prospective surveillance asks this every day. It also covers irregularly shaped clusters, without the
conditional-margin blind spot.

**Why it matters for this library.** The library ships the space-time permutation scan. The survey
chapter states that permutation and prospective scans condition on the marginal counts. Tango et
al. (2011) show they "fail to detect" a uniform increase in all regions. Neill (2009) shows that
conditional scans have low power and are slower to detect outbreaks affecting many regions. Survey
text: https://arxiv.org/pdf/1711.08960. The expectation-based variants exist for the Poisson, Gaussian,
negative binomial and ZIP distributions (same source).

**Sources.**
- *Fast subset scan.* Neill (2012), JRSS-B 74(2):337-360, read from the PDF. Kulldorff's statistic
  and the expectation-based Poisson statistic satisfy the LTSS property with priority `c_i/b_i`.
  Sorting by priority and scanning the top-j prefixes finds the exact unconstrained optimum in
  O(N log N). The proximity-constrained version (a centre plus its k−1 nearest neighbours) detected
  outbreaks 2 days faster (https://www.cs.nyu.edu/~neill/papers/jrssb2012.pdf).
- *Fast Bayesian scan / Fast Subset Sums.* Neill (2011), Stat Med 30(5), read from the PDF. It uses a
  hierarchical prior over (centre, k, a uniformly random subset of the k-neighbourhood). The total
  posterior over 2^k subsets is computed in O(k) per neighbourhood, so O(N·kmax) neighbourhoods
  overall. The paper reports runtime scaling quadratically with kmax
  (https://www.cs.nyu.edu/~neill/papers/StatMed2011.pdf).
- *scanstatistics* 1.1.2 (2025-10-16, GPL ≥3). Space-time anomaly detection focused on prospective
  surveillance, with Monte Carlo inference
  (https://cran.r-project.org/web/packages/scanstatistics/index.html). It is the oracle for EB
  Poisson, negative binomial and ZIP scans, and the Bayesian negative binomial scan. The function list
  comes from the search summary; check it against the refman.
- *SaTScan* v10.3.3 (September 2025). Registration is required and the source is on GitHub
  (https://www.satscan.org/download.html). The source licence was not confirmed: **UNVERIFIED**.
  *rsatscan* 1.0.10 (2026-01-26) defaults to the 10.3 parameter set
  (https://cran.r-universe.dev/rsatscan/NEWS).
- *The prospective-scan dispute.* Correa, Assunção and Costa (Stat Med 2015) argue that SaTScan
  prospective adjusted p-values do not control type I error. Kulldorff and Kleinman reply that they do
  when interpreted correctly, and recommend recurrence-interval signal rules
  (https://pmc.ncbi.nlm.nih.gov/articles/PMC4357279). Whatever ships should report recurrence
  intervals and say which repeated-analysis adjustment, if any, is applied.

**Data-parallel formulation.**
- *Baseline.* `b_{i,t}` comes from the existing temporal reductions or the harmonic model per
  location. A moving average over the last W buckets is the simplest.
- *Per (centre c, temporal window w ≤ Wmax).* Gather the kNN zone list (existing table, k ≤ 64 or 128)
  with `C_i = Σ_{t∈w} c_it` and `B_i = Σ b_it`. Run a workgroup bitonic sort by `C_i/B_i` (a fixed
  network, so bounded). Take prefix sums of C and B, evaluate `F = C log(C/B) + B − C` for `C > B` on
  each prefix, and reduce to the maximum.
- *Bounds.* Everything is bounded by kmax and Wmax.
- *Inference.* Monte Carlo replicates draw Poisson counts from the baseline with Philox. Each replicate
  repeats the same kernel, and the maximum F is compared, so it reuses the batch machinery of the
  existing scan.
- *Fast Subset Sums (Bayesian).* Per centre and k, compute the per-location likelihood ratios
  `LR_i = ∫…` (closed-form Gamma-Poisson). Then compute `Π (1+LR_i)/2` over the neighbourhood in log
  space (log-sum-exp in float32), and accumulate per-location posterior maps by gather.

**Capacity.** Locations ≤ about 10^5 × kmax ≤ 64 × Wmax ≤ 32.

**Risk.** Medium. The baseline estimation choice drives results. Float32 log-likelihoods are fine with
log-space accumulation.

---

### 6. ST-DBSCAN, ST-OPTICS and ST-HDBSCAN (S)

**Question answered.** Which events form dense clusters in both space and time, such as outbreaks,
protests or animal-group episodes?

**Sources.**
- `st-dbscan` 0.2.3 (2025-02-19, MIT): `ST_DBSCAN(eps1, eps2, min_samples)`, with chunked sparse
  matrices through `fit_frame_split`. It cites Cakmak et al. (HANIMOB'21)
  (https://pypi.org/project/st-dbscan/).
- `st-clustering` 0.1.0 (2023-03-07, MIT) offers ST-DBSCAN, ST-OPTICS, ST-HDBSCAN, ST-Agglomerative,
  ST-Spectral and ST-Affinity Propagation on top of scikit-learn (https://pypi.org/project/st-clustering/).
- How `eps1` and `eps2` combine is not stated on either PyPI page. It is presumably the conjunction of
  a spatial and a temporal threshold, as in Birant and Kut (2007): **UNVERIFIED**, so read the source
  before pinning.

**Data-parallel formulation.** The existing spatial radius pair list gains a time predicate
`|Δt| ≤ eps2`, a filter pass with stream compaction. The existing DBSCAN, OPTICS and HDBSCAN kernels
then run on that neighbour graph unchanged. For OPTICS and HDBSCAN, the oracle's combined metric must
be reproduced: ST-OPTICS uses `eps2` as a hard time filter on the spatial reachability, but this is
**UNVERIFIED**.

**Capacity.** Same as the existing indexed clustering, about 10^6 points.

**Risk.** Low computationally. The risk is semantic (metric conventions).

---

### 7. Time-series cross-correlation per location (S)

**Question answered.** Does variable B lead or lag variable A at each location, and by how much?

**Source.** ArcGIS Pro *Time Series Cross Correlation*, added in 3.3 and documented through 3.6:
- Pearson cross-correlation for each lag up to `10·log10(T/2)`.
- Optional prewhitening: OLS AR(1..5) plus a seasonal model with the FFT-estimated period, chosen by
  AICc, with the same coefficients applied to the secondary series.
- Optional neighbour averaging with equal, bisquare or Gaussian weights.
- p-values only when filtering is on and neighbours are off.
- The page gives the standard deviation as "the square root of the number of time steps", which looks
  inverted, and states that there is "no correction" for multiple testing.

Sources:
https://pro.arcgis.com/en/pro-app/3.5/tool-reference/space-time-pattern-mining/how-time-series-cross-correlation-works.htm,
https://pro.arcgis.com/en/pro-app/3.6/tool-reference/space-time-pattern-mining/time-series-correlation.htm.

**Formulation.** One invocation per (location, lag). Prefix-sum Pearson. AR fits are small fixed-size
normal equations (≤ 6×6) per location. Neighbour averaging is a weighted gather. Add BH/BY across
locations × lags, which ArcGIS lacks. This is a differentiator.

**Oracle.** Any textbook CCF, for example `statsmodels.tsa.stattools.ccf`. Version and licence were not
fetched: **UNVERIFIED**. ArcGIS is proprietary, so it serves only as a behavioural reference.

---

### 8. Differential and lagged bivariate local Moran (S, recipes)

**Question answered.** Is a location's change over time related to its neighbours' change? This is the
differential Moran. Does the neighbours' past value relate to the focal present value? This is the
lagged bivariate Moran.

**Sources.**
- GeoDa 1.22 (GPL) lists the global and local differential Moran's I, local join count with a
  colocation map, univariate and multivariate local Geary, the Time Editor, and an Averages Chart with
  diff-in-diff tests (https://geodacenter.github.io).
- The GeoDa workbook warns that a lagged bivariate LISA (`y_t` vs `W·y_{t−1}`) can show High-High
  through temporal inertia without diffusion, and that it ignores in-situ correlation
  (https://geodacenter.github.io/workbook/6c_local_multi/lab6c.html).
- esda 2.10.0 (2026-06-19, BSD-3) changed `Moran_Local` `p_sim` to use the new crand path, which
  matters for oracle pinning (https://zenodo.org/records/8283741/latest).

**Formulation.** Compute `Δy` per location, then run the existing global and local Moran with
permutation. Lagged bivariate = the existing `localBivariateMoran` with `x = y_t` and `y = y_{t−1}`. Ship
these as documented recipes with the GeoDa caveats.

---

### 9. Spatially adaptive FDR for local space-time p-values (M)

**Question answered.** How can the many local tests (hot spots per bin, LISA per location × time) be
corrected without throwing away power where the signals cluster spatially?

**Sources.**
- LAWS (Cai, Sun and Xia, JASA 117(539):1370-1383, 2022) estimates local sparsity, builds spatially
  adaptive weights for the p-values, then thresholds. It controls FDR asymptotically under mild
  dependence (https://ideas.repec.org/a/taf/jnlasa/v117y2022i539p1370-1383.html).
- STRAW (Wang, Yan and Li, arXiv 2309.15699, 2023) generalises LAWS: weights
  `((1−π̂₁)/π̂₁)^{1/k}` with k chosen from data, π̂₁ from Nadaraya-Watson smoothing of 1−Lfdr over
  neighbours, then a BH-type cutoff. LAWS is the special case k=1
  (https://ar5iv.labs.arxiv.org/html/2309.15699).
- e-values: Ignatiadis, Wang and Ramdas (arXiv 2409.19812, v4 2025). Every FDR procedure can be
  recovered as e-BH on compound e-values (https://arxiv.org/abs/2409.19812). For this library,
  Benjamini-Yekutieli already gives arbitrary-dependence control, so e-values add little in practice
  (see section 3).
- Cluster-level inference: TFCE (Smith and Nichols, NeuroImage 44(1) 2009) scores each voxel by its
  cluster-like support without a cluster-forming threshold (https://wrap.warwick.ac.uk/38213).

**Formulation.**
- LAWS/STRAW: kernel-smooth over the existing weights graph (a space-time neighbourhood = spatial
  neighbours × ±L bins), form the weighted p-values, then run the existing BH sort. The k-grid search
  in STRAW is a fixed 19-value loop.
- TFCE on a lattice needs connected components per threshold step. That is iterative label
  propagation, so a readback or a fixed iteration cap with a convergence flag. Treat it as L and defer
  it.

**Risk.** Medium. No pinned reference code was found, so the oracle would be the authors' code, if
public (**UNVERIFIED**), or a re-implementation from the paper.

---

### 10. Time-series clustering of cube locations (M)

**Question answered.** Which locations share a temporal profile?

**Source.** ArcGIS *Time Series Clustering* (added in 2.2; Fourier in 2.5):
- Value: k-means with Euclidean distance.
- Profile (Correlation): k-medoids (PAM) with `1−r`, switching to CLARA above 10,000 locations.
- Profile (Fourier): k-means on basis weights.
- k chosen from 2-10 by pseudo-F.
- Explicitly no spatial constraint.
- References: TSclust, Ramsay and Silverman, Kaufman and Rousseeuw, Lloyd, and k-means++.

Source: https://pro.arcgis.com/en/pro-app/3.5/tool-reference/space-time-pattern-mining/learnmoretimeseriesclustering.htm.

The newest research direction is MDL regionalization of spatial time series (Weng and Kirkley, arXiv
2605.05008, May 2026). It is contiguous, nonparametric, log-linear in the number of series, and has no
code listed (https://arxiv.org/abs/2605.05008).

**Formulation.**
- Value and Fourier: reuse k-means. The Fourier projection is a small dense matmul.
- Correlation k-medoids: an n² distance matrix, brute force to a few thousand locations, with PAM swap
  iterations at a fixed iteration cap.
- Pseudo-F: a reduction.
- Spatially constrained variant: feed the correlation distance to the existing AZP/regionalization.

**Risk.** Medium. PAM has data-dependent swaps; cap them and report convergence.

---

### 11. Prospective point surveillance: the Shiryaev-Roberts `stcd` (M-L)

**Sources.**
- `surveillance::stcd(x, y, t, radius, epsilon, areaA, areaAcapBk, threshold, cusum=FALSE)` implements
  Assunção and Correa (2009, CSDA 53(8)). It is marked experimental, and two area arguments are
  "currently ignored" (https://rdrr.io/cran/surveillance/man/stcd.html).
- The surveillance package is at 1.26.1
  (https://surveillance.r-forge.r-project.org/pkgdown/reference/index.html).

**Formulation.** The statistic for event n sums likelihood ratios over candidate change events `k ≤ n`
and candidate centres. The (n, k) pairs parallelise, but the alarm is a sequential first-passage. The
whole computation is O(n²) pairs, with brute-force capacity of a few thousand events.

**Risk.** High. The oracle is experimental and has ignored arguments. The value is mainly for
surveillance users who already have SaTScan.

---

### 12. Nonparametric self-exciting triggering histogram (M-L)

**Question answered.** What is the empirical space-time triggering shape (near-repeat decay) after
separating the background? This is descriptive Hawkes analysis.

**Sources.**
- Reinhart's review of self-exciting spatio-temporal point processes (Statistical Science 33(3) 2018)
  covers earthquakes, disease and crime (https://arxiv.org/abs/1708.02647).
- Mohler et al. (JASA 2011) "Self-Exciting Point Process Modeling of Crime" and the Marsan-Lengliné
  (2008) histogram EM (MISD) were seen only in search snippets: **UNVERIFIED**.
- Fox, Schoenberg and Gordon (AOAS 2016) extend MISD to a spatially varying background. The Project
  Euclid page did not render: **UNVERIFIED**.

**Formulation.** The E-step is the branching probability `p_ij` for pairs within `(u_max, v_max)` (the
pair list), with row normalisation per event. The M-step is a `U×V` histogram of `Σ p_ij` divided by
bin volume and n, plus the background. Run a fixed iteration count (for example 50) with a convergence
readback. Float32 is acceptable.

**Risk.** High. There is no pinned oracle package. It would need a careful CPU re-implementation as its
own oracle.

---

## 2. Reference software status, 2025-2026

| Software | Version / date | Licence | Space-time content | Oracle suitability |
|---|---|---|---|---|
| esda | 2.10.0 (2026-06-19); 2.8.1 in PySAL 26.01 | BSD-3 | local Moran, BV Moran, Geary, LOSH; crand change in 2.10 | Pin exact version (crand p_sim changed) — https://zenodo.org/records/8283741/latest, https://zenodo.org/records/18446136 |
| pointpats | 2.6.0 | BSD-3 (PySAL) | Knox, KnoxLocal, mantel, jacquez, modified_knox | Good — https://pysal.org/pointpats/api.html |
| giddy | 2.3.8 (2025-12-05, in PySAL 26.01) | BSD-3 | Markov family, dynamic LISA | Already used — https://zenodo.org/records/18446136 |
| PySAL meta | 26.01 (2026-02-01) | BSD-3 | bundles the above | https://zenodo.org/records/18446136 |
| stpp | 2.0-8 (2024-06-27) | GPL-3 | STIKhat, PCFhat, ASTIKhat, KLISTAhat, LISTAhat | Oracle only — https://cloud.r-project.org/web/packages/stpp/refman/stpp.html |
| stopp | 1.0.0 (2025-07-08) | GPL-2/3 | local 1st/2nd order, networks, Poisson ST models | Oracle only — https://cran.r-project.org/web/packages/stopp/index.html |
| sparr | 2.3-16 (2025-03-20) | GPL ≥2 | spattemp.density/risk, selectors, tolerance | Oracle only — https://cran.r-project.org/web/packages/sparr/refman/sparr.html |
| splancs | not shown | not shown (**UNVERIFIED**) | stmctest (Diggle 1995) | https://rdrr.io/cran/splancs/man/stmctest.html |
| surveillance | 1.26.1 | not shown on page (GPL per CRAN, **UNVERIFIED**) | knox, stKtest, stcd, twinstim, CUSUM, Farrington | https://surveillance.r-forge.r-project.org/pkgdown/reference/index.html |
| scanstatistics | 1.1.2 (2025-10-16) | GPL ≥3 | EB/Bayesian/permutation space-time scans | Oracle only — https://cran.r-project.org/web/packages/scanstatistics/index.html |
| rflexscan | 1.2.0 (2025-11-28) | GPL-3 | flexible (Tango-Takahashi) and circular scan | Oracle only — https://cran.r-project.org/web/packages/rflexscan/index.html |
| SaTScan | 10.3.3 (Sep 2025) | **UNVERIFIED** (source on GitHub) | all classic scans | Behavioural oracle — https://www.satscan.org/download.html |
| rsatscan | 1.0.10 (2026-01-26) | GPL-3 (per search) | SaTScan driver | https://cran.r-universe.dev/rsatscan/NEWS |
| NearRepeat | 0.1.1, GitHub only | GPL-3 | Knox band table | https://rdrr.io/github/wsteenbeek/NearRepeat |
| st-dbscan / st-clustering | 0.2.3 (2025-02) / 0.1.0 (2023-03) | MIT | ST-DBSCAN, ST-OPTICS, ST-HDBSCAN | Good — https://pypi.org/project/st-dbscan/, https://pypi.org/project/st-clustering/ |
| GeoDa | 1.22 | GPL | differential Moran, local join count, Time Editor | Behavioural — https://geodacenter.github.io |
| ArcGIS Pro STPM | docs to 3.6; newest toolbox change 3.5 | proprietary | EHSA, local outlier, change points, TS clustering, cross-correlation, forecasting | Spec only — https://doc.esri.com/en/arcgis-pro/latest/tool-reference/space-time-pattern-mining/space-time-pattern-mining-toolbox-history-pro.html |
| scikit-mobility | 1.3.1 | BSD-3 (per search) | mobility measures | https://www.github.com/scikit-mobility/scikit-mobility/releases |
| sf / stars / sftime / gstat, spatstat, CrimeStat | not fetched | **UNVERIFIED** | — | — |

**ArcGIS notes.**
- The Space Time Pattern Mining toolbox history lists these tools: Change Point Detection (2.9),
  Time Series Clustering (2.2), Time Series Cross Correlation (3.3), Local Outlier Analysis (1.2),
  Curve Fit, Exponential Smoothing and Forest-based Forecast (2.6), Evaluate Forecasts by Location
  (2.6), Fill Missing Values (2.0), Describe/Subset Space Time Cube (3.0), and Make Space Time Cube
  Layer (3.5) (https://pro.arcgis.com/en/pro-app/3.5/tool-reference/space-time-pattern-mining/space-time-pattern-mining-toolbox-history-pro.htm).
- Local Outlier Analysis is a space-time Anselin Local Moran:
  - temporal neighbours look backward only;
  - first-slice bins get no index;
  - it uses permutations and FDR;
  - it classifies each location as Never Significant, Only High-High, …, or Multiple Types.

  Source: https://doc.esri.com/en/arcgis-pro/latest/tool-reference/space-time-pattern-mining/learnmorelocaloutlier.html.

  This maps directly onto the existing local Moran with a space-time weights graph. It is S as a recipe.

---

## 3. Not worth doing (and why)

- **Spatiotemporal co-location pattern mining** (participation index over time slices, sub-prevalent
  or duration patterns). It is Apriori-style candidate generation with combinatorial, data-dependent
  growth. There is no pinned reference implementation, and the literature is a scatter of variants
  (e.g. dynamic co-location, https://arxiv.org/abs/1812.11542). The question "do A and B co-occur in
  space and time" is better answered by a cross-type space-time Knox or K (cards 1 and 3) and the
  existing bivariate local join count.
- **Flexible scan (Tango-Takahashi).** It enumerates connected subsets, which grows exponentially with
  K. rflexscan 1.2.0 exists as a GPL oracle (https://cran.r-project.org/web/packages/rflexscan/index.html),
  but Fast Subset Scan (card 5) finds irregular clusters exactly and in bounded work.
- **Two-way MOSUM and high-dimensional spatial change-point inference.** The theory is heavy
  (111 pages) and there is no reference implementation (https://arxiv.org/abs/2208.13074v2).
  Per-location exact change points (card 4) plus spatial FDR (card 9) give analysts most of the value.
- **"Spatiotemporal Moran's I" indices** (Gao et al. 2019, GSIS, and similar). Many incompatible
  definitions exist, and the formula was not visible on the fetched page
  (https://geos3.netlify.app/publication/gao-2019-measuringa). Use differential and lagged bivariate
  Moran (card 8) and per-location cross-correlation (card 7) instead.
- **e-value multiple testing.** Benjamini-Yekutieli, which already exists, controls FDR under
  arbitrary dependence. e-BH needs e-values that permutation tests don't naturally produce
  (https://arxiv.org/abs/2409.19812).
- **Esri forecasting tools** (forest-based, exponential smoothing, curve fit). These are predictive,
  not exploratory, and out of scope.
- **Full multivariate Bayesian scan (MBSS)** with learned event-type models. It needs labelled
  training data and priors. The Fast Subset Sums core in card 5 is the reusable part.
- **Naive voxel-based STKDE.** Saule et al. show that point-based or gather-with-index is
  asymptotically better (https://arxiv.org/pdf/1705.09366).
- **TFCE and cluster-mass permutation on irregular areal units** for now. Connected components per
  threshold need data-dependent iteration. Revisit for raster cubes only.

---

## 4. Published GPU or parallel implementations

| Method | Work | Hardware | Reported numbers | Parallel formulation |
|---|---|---|---|---|
| STKDE | Eaglin, Cho, Ribarsky, HICSS 2017 | GTX 770 | 6.3M voxels 0.636 s (0.387 s coalesced) vs 3.425 s CPU; ~23 min sequential | Oct-tree on GPU, per-voxel gather — https://webpages.charlotte.edu/icho1/data/eaglin17.pdf |
| STKDE | Saule et al., arXiv 1705.09366 | 16-thread CPU | ~14.9x with domain decomposition; PB vs VB orders of magnitude | Point-based scatter, separable kernel, replicated buffers — https://arxiv.org/pdf/1705.09366 |
| STKDV | Chan et al., PVLDB 2022 | CPU | 1.71-24x; O(XYTn) → O(XY(T+n)) | Sliding window moment sums — https://www.vldb.org/pvldb/vol15/p814-chan.pdf |
| Space-time K | Wang, Gui et al., FGCS 2020 | Spark | not quantified on abstract page | ST R-tree, KDB partitioning, edge-weight cache — https://arxiv.org/abs/1912.04753 |
| Knox, Jacquez, Diggle, space-time scan, ST-DBSCAN | none found | — | — | Searches returned only CPU, Spark or MapReduce work, or spatial-only GPU K and Gi* (search summaries; not fetched individually) |

---

## 5. Most surprising findings

1. **There is effectively no published GPU work** for space-time K, Knox, Jacquez, the Diggle test or
   space-time scans. The only GPU space-time statistic found is a 2017 STKDE. The large published
   speedups are algorithmic: PB-SYM, and SWS's O(XY(T+n)). A WebGPU library with exact permutation
   inference across these tests would be first of its kind. That claim rests on negative search
   results, so treat it as "not found" rather than proven absent.
2. **The library's space-time permutation scan inherits a documented blind spot.** Tests conditional
   on the margins cannot see a uniform rise across all regions (Tango et al. 2011, via
   https://arxiv.org/pdf/1711.08960). The expectation-based and fast subset scans (card 5) fix this and
   share most of the existing kNN-zone machinery.
3. **ArcGIS Time Series Cross Correlation applies no multiple-testing correction**, and its page gives
   the cross-correlation standard deviation as √T. Change Point Detection is plain per-location
   PELT/SegNeigh with textbook likelihood costs
   (https://pro.arcgis.com/en/pro-app/3.5/tool-reference/space-time-pattern-mining/how-time-series-cross-correlation-works.htm,
   https://doc.esri.com/en/arcgis-pro/latest/tool-reference/space-time-pattern-mining/how-change-point-detection-works.html).
   Both are cheap to match and easy to exceed with FDR.
