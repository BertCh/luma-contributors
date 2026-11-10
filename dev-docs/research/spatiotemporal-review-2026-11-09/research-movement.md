# Movement, mobility and flow analytics: research scout (2022 to 2026)

Date of research: 2026-11-09. Scope: what a current movement toolkit has that the library does not, and
how each gap maps onto bounded WGSL compute.

How to read the citations:
- A URL next to a claim means I opened that page in this session.
- **UNVERIFIED (snippet)** means the claim comes only from a search-engine summary. I did not open the
  primary page, or the page was blocked (Springer, PMC, MDPI and some university repositories returned
  redirects, 403 errors or bot checks).
- Licences are given only where the page I opened states them. Otherwise they are marked UNVERIFIED.

---

## 0. Executive summary

The library already covers the "trajectory algebra" layer: resampling, Kalman/RTS smoothing,
MEOS-style restriction, encounters, zone events, DTW/LCSS/Fréchet and HMM map matching. It also covers
a strong network layer. **The biggest gap is the semantic preprocessing layer that every mobility
toolkit is built on: positionfixes, then staypoints, then locations, then triplegs and trips.** Almost
every mobility indicator people ask for (radius of gyration, home and work, OD matrices from real
traces, entropy, motifs, visitation) is defined on staypoints and locations, not on raw fixes.
trackintel (MIT) and scikit-mobility (BSD) are both permissive, deterministic oracles for this layer.

The second gap is animal movement ecology: Brownian-bridge utilisation distributions, recursion and
residence analysis, first passage time, step-selection random steps and HMM state decoding. These are
per-track scans and grid-by-segment integrals, which fit the platform well. Most reference packages
are GPL-3 R packages, so they can serve as test oracles but their code cannot be copied. The exception
is `recurse`, which is MIT.

The third gap is Lagrangian diagnostics on time-varying fields. FTLE, connectivity matrices and
residence time are cheap once particle advection exists, and lcs-parcels (MIT) splits the work the same
way: it emits seed particles and ingests the advected positions.

### Ranked candidates (analyst value over effort)

| Rank | Candidate | Value | Effort | Risk | Oracle (licence) |
|---|---|---|---|---|---|
| 1 | Staypoints, locations, triplegs, trips (trackintel semantics) | Very high | M | Medium | trackintel (MIT), MovingPandas `TrajectoryStopDetector` (BSD-3) |
| 2 | Mobility indicator pack on staypoints (radius of gyration, k-radius, jump lengths, waiting times, location entropies, home/work FREQ/OSNA, tracking quality, speed-based mode, modal split) | High | S | Low | scikit-mobility 1.3.1 (BSD), trackintel (MIT) |
| 3 | Cleaning and generalisation pack (speed/acceleration outlier and Hampel filters, gap and stop splitters, MinDistance/MinTimeDelta, Top-Down Time Ratio) | High | S–M | Low to medium | MovingPandas 0.23 (BSD-3), PTRAIL (BSD-3) |
| 4 | Lagrangian diagnostics on time-varying fields (time interpolation of fields, FTLE, connectivity matrix, residence time) | High (ocean/atmosphere users) | S–M | Low | lcs-parcels (MIT), OceanParcels Lagrangian_diags tutorial (licence UNVERIFIED) |
| 5 | Brownian bridge and dynamic BBMM utilisation distributions, plus UD volume isopleths | High (ecology) | M | Medium | R `move` 4.2.x (GPL-3 per UNVERIFIED snippet; oracle only) |
| 6 | Recursion, residence time and first passage time | Medium–high | S | Low | `recurse` 1.4.0 (MIT) |
| 7 | Extra similarity measures on the existing DP kernels (EDR, ERP, Hausdorff, SSPD), plus lower bounds for top-k | Medium | S | Low | TrajSimiMeasures (licence UNVERIFIED), MovingPandas 0.23 distances (BSD-3) |
| 8 | Radiation model and flow smoothing (Guo & Zhu kernel flow smoothing) on the existing OD aggregation | Medium | S–M | Low | scikit-mobility `Radiation` (BSD) |
| 9 | Time geography: network potential path area from two shortest-path trees, departure-time-window travel-time percentiles over FIFO routing | Medium | S–M | Low | NetworkX / pgRouting for the trees; r5r `time_window` semantics as a specification |
| 10 | HMM behavioural state decoding (Viterbi and forward-backward with given parameters), step lengths and turning angles, random steps for SSF | Medium (ecology) | M | Medium | moveHMM 1.12, momentuHMM 1.5.8, amt 0.3.1.0 (all GPL-3, oracle only) |
| 11 | Moving clusters and convoys (CMC on the resampled grid), pairwise following and leadership by time-lagged correlation | Medium | M–L | High | No clean open-source oracle found; a CPU reference would have to be written from the paper |
| 12 | Interchange: GeoArrow trajectories with per-vertex time lists, MF-JSON, MobilityDuck in DuckDB-Wasm as an in-browser oracle | Medium (enabler) | S | Low | MobilityDuck (licence UNVERIFIED), MEOS 1.3 |

### Most surprising findings

1. **GPU speedups for classical similarity measures are modest, and learned embeddings dominate kNN.**
   In the PVLDB 2024 efficiency study, batched DTW on 512 pairs ran only about 4.5 times faster on a V100
   than on a CPU (28.11 s vs 6.21 s). Single-pair GPU DTW was *slower* than CPU (146 s vs 36 s on
   Porto). For kNN queries, TrajCL embeddings were reported as 984 times faster than the fastest
   non-learned measure (Fréchet) on GPU. https://arxiv.org/html/2311.00960v3
   Implication: the library's pruned top-k with lower bounds is the right lever. Raw DP throughput is
   not. Anti-diagonal parallelism inside one pair pays off only for long sequences.
2. **MobilityDuck ships precompiled DuckDB-Wasm extension packages.** The MEOS temporal algebra, which
   is this library's reference semantics, can therefore run inside a browser test harness next to the
   WebGPU kernels. https://arxiv.org/html/2510.07963v1

---

## 1. Trajectory preprocessing

### 1.1 Stop / stay-point detection and location clustering (rank 1)

**Question answered.** Where and when did each object stop, and which stops are the same place?

**Methods and sources**
- *trackintel* runs a staged model of positionfixes, staypoints, triplegs, trips and tours.
  `generate_staypoints(method='sliding', distance_metric='haversine', dist_threshold=100,
  time_threshold=5.0, gap_threshold=15.0, include_last=False, exclude_duplicate_pfs=True)` is the
  sliding-window stay-point method in the Li et al. lineage. `generate_triplegs(method=
  'between_staypoints' | 'overlap_staypoints', gap_threshold=15)`, `generate_locations(method='dbscan',
  epsilon=100, num_samples=1, distance_metric='haversine', agg_level='user'|'dataset')`,
  `merge_staypoints(max_time_gap='10min')`, `generate_trips(gap_threshold=15)` and
  `generate_tours(max_dist=100, max_time='1D')`.
  https://trackintel.readthedocs.io/en/latest/modules/preprocessing.html. Licence MIT
  (https://github.com/mie-lab/trackintel). The release version was not shown on the pages I opened.
- *Infostop* (Aslak & Alessandretti, arXiv 2003.14370, 2020) works in two steps. Step 1 labels stays
  per user with r1 (roaming radius, 20 m), t_min (10 min) and t_max (24 h). Step 2 builds a network of
  stays linked within r2 across all users and partitions it with Infomap. The paper reports that DBSCAN
  merges nearby overlapping destinations and that Infostop scales better.
  https://ar5iv.arxiv.org/html/2003.14370. The package is at version 0.2.0 (changelog 2026-10-28) and
  now runs Infomap with `num_trials=50` by default, so labels differ from 0.1.9.
  https://github.com/ulfaslak/infostop (licence not stated on the page: UNVERIFIED).
- Bertè et al. 2024 (arXiv 2407.11579) trains classifiers (LightGBM, RF) for stop probability on
  incomplete data. It reports no comparison with Infostop or DBSCAN. https://arxiv.org/html/2407.11579v1
- *CB-SMoT* (Palma et al., ACM SAC 2008) runs DBSCAN on one-dimensional speed. Later work criticises its
  global speed parameter. **UNVERIFIED (snippet)**.
- MovingPandas has `TrajectoryStopDetector`, which moved to `n_processes` in 0.22.
  https://github.com/movingpandas/movingpandas/releases

**Data-parallel formulation (bounded)**
1. *Per-point forward extent.* For each fix i, a thread scans forward up to a compile-time cap W fixes
   (for example 256) within the same track. It finds the last j with dist(i, j) ≤ d and checks
   t_j − t_i ≥ T_min and gap < gap_threshold. This is a bounded loop with an early exit. It gives
   `end[i]` and `isStay[i]`.
2. *Greedy chaining without sequential iteration.* trackintel's sliding method is greedy: after a stay
   from i to end[i], the next anchor is end[i] + 1, and otherwise i + 1. Define
   `next[i] = isStay[i] ? end[i] + 1 : i + 1` and find which anchors lie on the chain from each track
   start by pointer doubling: ceil(log2(maxTrackLength)) passes, unrolled at compile time. Mark
   reachable anchors with a segmented scan. This reproduces the sequential greedy result exactly,
   provided W is not exceeded. Write an overflow flag when it is.
3. *Staypoint attributes* (centroid, start, end, duration) come from a segmented reduction over the
   [i, end[i]] ranges.
4. *Locations.* Run DBSCAN over staypoints using the existing grid or neighbour-index join (eps = 100 m),
   then connected components by bounded label-propagation passes or a union-find with bounded hooking
   rounds. Raise a convergence flag if the round budget is hit. With trackintel's `num_samples=1`
   default, DBSCAN reduces to connected components of the eps-graph, which makes the GPU result
   deterministic.
5. *Triplegs and trips* fall out as the complement intervals between staypoints (segmented scan), plus
   gap splitting.

**Reuses:** per-track scans, the neighbour index, zone dwell machinery, and the playhead and resampling
helpers for time handling.

**Capacity:** about 1M fixes for steps 1 to 3 (W × N distance tests). Location clustering on about 100k
staypoints.

**float32:** store time as float32 seconds relative to a per-track epoch (int64 or two-float base on the
CPU). Use local projected metres for the distance tests.

**Effort:** M. **Risk:** medium. The parity edge cases are `include_last`, duplicate fixes and the
gap_threshold rule. Use trackintel as the primary oracle. Infomap (step 2 of Infostop) is not
recommended, because it is a stochastic multi-trial optimisation and 0.2.0 deliberately changed its
outputs.

### 1.2 Cleaning, splitting and generalisation (rank 3)

**What MovingPandas offers (0.21 to 0.23):**
- `ValueChangeSplitter` and nanosecond handling (0.21). `OutlierCleaner` now returns valid
  trajectories (0.21).
- Multiprocess splitters (0.22).
- Fréchet, DTW and LCSS distances (0.23.0, released 2026-08-09).
- Sources: https://github.com/movingpandas/movingpandas/releases and
  https://pypi.org/project/movingpandas/ (BSD-3, 0.23.0).

**Generalizers:** `DouglasPeuckerGeneralizer` (via shapely/GEOS), `MinDistanceGeneralizer`,
`MinTimeDeltaGeneralizer`, and `TopDownTimeRatioGeneralizer`, which is Meratnia & de By 2004 using the
synchronised Euclidean distance (SED) to the time-ratio projection.
https://movingpandas.readthedocs.io/en/main/api/trajectorygeneralizer.html

**PTRAIL** (SoftwareX 2022, BSD-3) adds:
- Kinematic features: distance, speed, acceleration, bearing, bearing rate.
- Hampel, kinematic and temporal filters, and duplicate removal.
- Linear, cubic, kinematic and random-walk interpolation.

https://ar5iv.labs.arxiv.org/html/2108.13202

**SQUISH-E** (Muckell et al., GeoInformatica 2014) uses a priority queue keyed by removal error.
**UNVERIFIED (snippet)**.

**Data-parallel formulation**
- *Speed and acceleration outlier removal* is not one pass. MovingPandas' cleaner recomputes after
  removals. Use a fixed number K of passes (unrolled), each a map plus segmented compaction, and flag
  any track still dirty after K passes.
- *Hampel filter:* a fixed window median per point (bounded sort of 2w + 1 values in registers).
- *Gap, stop and value-change splitters:* a flag map followed by a segmented scan that assigns new
  track IDs.
- *MinDistance and MinTimeDelta:* these are sequential "keep if far from the last kept" rules. Use the
  same pointer-doubling trick as for stays. `next[i]` is the first j > i beyond the tolerance, found by
  a bounded forward search.
- *Top-Down Time Ratio:* a level-synchronous recursion. At each level, compute for every active
  segment the max-SED point with a segmented arg-max, then split where the maximum exceeds the
  tolerance. Use a fixed depth cap L (for example 24, unrolled) and flag segments that still need
  splitting. Douglas-Peucker takes the same kernel with perpendicular distance.
- *Not recommended:* SQUISH-E's priority-queue order, because it is inherently sequential.

**Capacity:** about 1M fixes. **Effort:** S (filters, splitters), M (TD-TR and DP with depth caps).
**Risk:** low. TD-TR parity depends on tie-breaking and the CRS distance used by the oracle (shapely
`Point.distance`), so compare in projected coordinates.

### 1.3 Mode detection

trackintel's `predict_transport_mode(method='simple-coarse')` classifies on speed thresholds: slow below
15 km/h, motorized below 100 km/h, fast above that. `calculate_modal_split` aggregates by count,
distance or duration. https://trackintel.readthedocs.io/en/latest/modules/analysis.html

This is a trivial segmented reduction on triplegs. Learned mode classifiers belong in an ML stack, not
here. **Effort:** S (bundle with rank 2).

### 1.4 Smoothing beyond the constant-velocity Kalman filter

MovingPandas' smoothing is a Kalman smoother (the optional `smoothing` extra).
https://pypi.org/project/movingpandas/

The cheap next steps are a constant-acceleration state and a speed-gated innovation test. Both slot into
the existing per-track Kalman/RTS scan. Effort is S, but value is low. A dedicated oracle beyond
MovingPandas or a hand-checked filter was not found.

---

## 2. Trajectory similarity and clustering at scale

### 2.1 Evidence on GPU similarity

**Chang, Tanin, Cong, Jensen & Qi, "Trajectory Similarity Measurement: An Efficiency Perspective"**
(arXiv 2311.00960v3, June 2024; PVLDB vol. 17 p. 2293 per the search listing).
https://arxiv.org/html/2311.00960v3
- Implements DTW, ERP, Fréchet, Hausdorff, STEDR, CDDS and SAR on CUDA (Numba). Also seven learned
  measures, including TrajCL and NEUTRAJ.
- Batched speedups are modest: DTW about 4.5x, Hausdorff about 3x. Single-pair GPU runs are slower than
  CPU.
- Appendix A.2 parallelises DP measures by anti-diagonal wavefront. Each core owns a column (with
  round-robin interleaving when columns exceed cores), keeps a 3-row rolling buffer, and synchronises
  once per anti-diagonal over |T| + |T′| + 1 slots.
- Code: https://github.com/changyanchuan/TrajSimiMeasures (licence UNVERIFIED).

**SoftDTW-CUDA-Torch** (Shapira Weber & Freifeld, arXiv 2602.17206, Feb 2026, MIT).
https://arxiv.org/html/2602.17206v1
- One kernel launch per anti-diagonal, tiled in blocks of 256 threads, with a single-launch fast path
  when max(N, M) ≤ 1024.
- A "fused" mode recomputes distances in-kernel: memory drops from O(BNM) to O(B(N + M)), at about
  10–15x the run time.

**Other GPU work:**
- cuDTW++ (Euro-Par 2020) uses warp intrinsics and reports more than 90% of peak on Volta.
  **UNVERIFIED (snippet)**.
- Gudmundsson & Valladares (IEEE TPDS 2015), GPU subtrajectory clustering under continuous Fréchet,
  up to 11x. **UNVERIFIED (snippet)**.
- TKSimGPU / FastTopK top-k Hausdorff on GPU. **UNVERIFIED (snippet)**.
- RAPIDS cuSpatial 25.04 still lists `directed_hausdorff_distance` among its spatial functions.
  https://docs.rapids.ai/api/cuspatial/stable/api_docs/spatial/

**Survey:** Hu et al., arXiv 2303.05012 (2023). It organises measures as non-learned vs learned, free
space vs road network, and standalone vs distributed. https://arxiv.org/abs/2303.05012

### 2.2 Recommended additions (rank 7)

- **EDR, ERP:** same DP shape as LCSS/DTW. Swap the cell recurrence and the boundary initialisation.
- **Discrete Hausdorff:** an enumeration measure (a max of mins). It is the cheapest measure, with no DP
  dependency, so use one thread per (pair, row) and a reduction.
- **SSPD** (symmetric segment-path distance): a mean of point-to-polyline minimum distances. Also an
  enumeration, so trivially parallel.
- **Continuous Fréchet decision problem** (free-space diagram reachability at a threshold ε): a bounded
  DP over cells with interval propagation. It answers "is the Fréchet distance ≤ ε" for within-distance
  joins. The full optimisation needs a parametric search, which is data-dependent and should stay a
  CPU-driven binary search over the GPU decision kernel with a fixed iteration count. Effort M.
- **Lower bounds for pruned top-k:**
  - Bounding-box or endpoint bounds for Fréchet and Hausdorff.
  - LB_Keogh-style envelopes for banded DTW.
  - Count-filter bounds for LCSS and EDR.
  - These are enumeration kernels that feed the existing pruned top-k.

**Formulation choice:** keep one workgroup per pair, using the anti-diagonal inside the workgroup with
a rolling 3-row buffer in workgroup memory. Batch many pairs per dispatch. Given the study above, the
throughput win comes from batching and pruning, not from intra-pair parallelism.

**Capacity:** thousands of pairs × sequences up to about 1–2k points (workgroup memory bound).

**Oracle:** tslearn (BSD) for DTW and LCSS is already used. MovingPandas 0.23 has
Fréchet/DTW/LCSS (BSD-3). For EDR and ERP, a small hand-checked CPU reference is safer than
TrajSimiMeasures until its licence is verified.

**Effort:** S per measure. **Risk:** low.

### 2.3 Learned similarity (t2vec, TrajCL and successors)

- TrajCL (Chang, Qi, Liang & Tanin; ICDE 2023 per the repo) uses contrastive learning with four
  augmentations and a dual-feature self-attention encoder. It reports up to 56% higher accuracy after
  fine-tuning. https://arxiv.org/abs/2210.05155v3
- An earlier study reports that t2vec is "orders of magnitude more scalable than classical models".
  https://dbis.ipd.kit.edu/2652.php
- What a compute library should provide instead of training:
  - Tokenisers: grid-cell or H3 sequences, and map-matched road-segment sequences (map matching
    already exists).
  - Fixed-length resampling (already exists).
  - A batched cosine or L2 kNN over embedding vectors.

  That kNN is a dense matrix-product-plus-top-k kernel, effort S. The model itself would run in an ML
  runtime such as ONNX or WebGPU ML.

### 2.4 Sub-trajectory clustering (TRACLUS) and DBSCAN over trajectories

- TRACLUS (Lee, Han & Whang, SIGMOD 2007): MDL partitioning followed by density clustering of line
  segments using perpendicular, parallel and angular distances. **UNVERIFIED (snippet)** for the paper.
- `traclus-python` 1.0.2 (Apache-2.0, April 2023) is a thesis port. Its author warns it is "not
  guaranteed bug-free". https://pypi.org/project/traclus-python/
- On GPU:
  - MDL partitioning is a greedy per-track scan. Pointer doubling applies, but the MDL cost function
    makes the forward search window-bounded.
  - Segment DBSCAN is an all-pairs segment-distance kernel plus components. It is feasible up to about
    20k segments.
- **Verdict:** medium value, M effort, and a weak oracle (high parity risk). Defer it. DBSCAN over whole
  trajectories using an existing measure's distance matrix plus connected components is S, and more
  useful.

---

## 3. Collective movement patterns (rank 11)

**Convoys.** Jeung, Yiu, Zhou, Jensen & Shen define a convoy as at least m objects that are
density-connected (DBSCAN with e and m) for k consecutive time points.
https://ar5iv.labs.arxiv.org/html/1002.0963
- CMC interpolates virtual points, runs DBSCAN per time step, and intersects candidates with clusters
  sharing at least m members.
- The CuTS family filters with Douglas-Peucker-simplified trajectories and distance bounds, then
  refines with CMC. It runs 3.9–33.1x faster than CMC.
- Evolving convoys (Aung & Tan, SSDBM 2010). **UNVERIFIED (snippet)**.

**Flocks.** Calderon Romero, Tsotras, Bakalov & Vieira, "Scalable Processing of Moving Flock Patterns",
SSTD 2025 (CC BY).
https://www.cs.ucr.edu/~acald013/public/Javeriana/Research/sstd25-17-4.pdf
- A flock is at least μ entities inside a disk of diameter ε for δ consecutive instants. The problem is
  NP-hard.
- BFE costs O(2n²) per time step. Its pipeline is: find pairs within ε, compute two candidate centres
  per pair, find disks, prune non-maximal disks, then temporally join partial flocks.
- The SSTD 2025 work parallelises by spatial partitioning (distributed, not GPU).
- An earlier GPU flock paper (Fort, Sellarès & Valladares, IJGIS 2014): **UNVERIFIED (snippet)**.
  The repository page was bot-blocked.
- A GPU "loosely moving congestion pattern" method reports more than 65x over sequential.
  **UNVERIFIED (snippet)**. MDPI returned 403.

**Leadership and following.**
- Fernández-Gracia et al. (PLOS Complex Systems 2025) infer leader-follower networks from lag-time
  asymmetry using a Kolmogorov-Smirnov distance. https://arxiv.org/abs/2412.03990
- Time-lagged directional correlation of velocities is the standard trajectory-based method.
  **UNVERIFIED (snippet)**.

**Formulation on the existing resampled time grid (all objects on common ticks):**
1. Per tick: grid neighbour query, then connected components of the e-graph. The existing neighbour
   index applies. Components with fewer than m members are dropped.
2. *Pairwise convoy (m = 2) and co-membership runs:* for each candidate pair (bounded thousands), a
   per-tick bit "same component" followed by a segmented scan for run lengths ≥ k. This is exact and
   cheap. It generalises the existing within-distance encounters to density-connected encounters.
3. *Moving clusters / CMC for m > 2:* for consecutive ticks, histogram the (component_t,
   component_{t+1}) label pairs with sort plus run-length encoding to get the overlap counts. Chain
   overlaps ≥ m with a segmented scan.
   - This gives the "moving cluster" variant. Exact CMC candidate-set intersection, where member sets
     shrink, is not reproduced.
   - Exact convoys need the candidate set semantics. Do these on the CPU after the GPU produces
     per-tick components.
4. *Following:* for each pair, compute the lagged velocity correlation for lags −L..L (unrolled) over
   sliding windows.

**Capacity:** about 10k objects × about 1k ticks for components. Pairs bounded to thousands.

**Effort:** M for components, co-membership and following. L for exact convoys and flocks.

**Risk:** high for exact parity. No open-source convoy or flock implementation was found to pin.

**Recommendation:** ship per-tick density clusters, moving-cluster chaining and pairwise co-membership
runs. Do not attempt exact maximal flocks.

---

## 4. Mobility indicators and flows

### 4.1 Indicator pack (rank 2)

**scikit-mobility measures:**
- Individual: `radius_of_gyration`, `k_radius_of_gyration`, `jump_lengths`, `waiting_times`,
  `random_entropy`, `uncorrelated_entropy`, `real_entropy`, `home_location`,
  `max_distance_from_home`, `number_of_locations`, `number_of_visits`, `location_frequency`,
  `frequency_rank`, `recency_rank`, `distance_straight_line`, `individual_mobility_network`.
- Collective: `visits_per_location`, `homes_per_location`, `mean_square_displacement`,
  `random_location_entropy`, `uncorrelated_location_entropy`, `visits_per_time_unit`.
- Source: https://scikit-mobility.github.io/scikit-mobility/reference/measures.html
- Latest release 1.3.1 (June 2022), BSD. It also has Gravity, Radiation, EPR/DITRAS, `LocationAttack`
  privacy risk and `FlowDataFrame`. https://pypi.org/project/scikit-mobility/

**trackintel analysis:**
- `radius_gyration(method='count'|'duration')`, `jump_length`.
- `location_identifier(method='FREQ'|'OSNA')` with `pre_filter_locations` thresholds.
- `temporal_tracking_quality(granularity=…)`, `create_activity_flag`.

https://trackintel.readthedocs.io/en/latest/modules/analysis.html

**Formulation:** all of these are segmented reductions over per-user staypoint or location tables
(sort by user, then segmented sum, max and count), plus small per-user histograms for entropy.
- `real_entropy` (Lempel-Ziv) needs, for each position, the shortest substring not seen before. That is
  O(n²) per user, bounded by a cap on visits per user. Treat it with care: critiques argue that the
  estimator and the Fano bound overestimate predictability. **UNVERIFIED (snippets)** for those
  critiques.

**Capacity:** millions of staypoints. **Effort:** S. **Risk:** low.

### 4.2 OD extraction, flow smoothing and flow models (rank 8)

- OD from traces = origin and destination staypoint locations of each trip (rank 1 output), mapped to
  zones with the existing point-in-polygon, then the existing OD aggregation.
- **Radiation model** (Simini, González, Maritan & Barabási, Nature 2012): parameter-free; the flux
  depends on m_i, n_j and s_ij, the population within radius r_ij of i.
  https://arxiv.org/abs/1111.0586
  GPU: for each origin, sort destinations by distance (segmented sort), take an exclusive prefix sum of
  population to get s_ij, then evaluate the closed-form flux. It is an O(Z² log Z) bounded kernel.
  About 5k zones (25M pairs) is realistic. Oracle: scikit-mobility `Radiation`. Effort S.
- **Universal visitation law** (Schläpfer et al., Nature 593, 2021): visitors scale with the inverse
  square of frequency × distance.
  https://ideas.repec.org/a/nat/nature/v593y2021i7860d10.1038_s41586-021-03480-9.html
  This is a 2-D histogram (distance × frequency per location) over staypoint visits. Effort S, as a
  diagnostic.
- **Flow smoothing** (Guo & Zhu, IEEE TVCG 20(12), 2014): a p-size kNN neighbourhood of each location,
  where the weighted size sums to p. A flow's neighbours are the flows whose origin and destination both
  fall in those neighbourhoods. Each flow is re-estimated with a Gaussian kernel. Flow selection then
  removes duplicates.
  https://vis.cs.ucdavis.edu/vis2014papers/TVCG/papers/2043_20tvcg12-guo-2346271.pdf
  GPU: kNN (existing) plus a sorted prefix to reach p, then a per-flow sum over neighbour-flow pairs.
  It is bounded with a cap on neighbours. Effort M. No packaged oracle was found, so a hand reference
  would be needed.
- **Deep Gravity** (Simini, Barlacchi, Luca & Pappalardo, Nat. Commun. 2021) is a neural network over
  39 OSM features. **UNVERIFIED (snippet)**; PMC was bot-blocked. This is out of scope beyond feature
  extraction.
- **Mobility motifs** (Schneider et al., J. R. Soc. Interface 2013): 17 daily networks cover up to 90%
  of people. https://dspace.mit.edu/handle/1721.1/89420
  GPU: per user-day, build the visited-location sequence (at most about 6 locations), canonicalise the
  directed graph (bounded: for at most 6 nodes, enumerate permutations or use a precomputed canonical
  hash table) and histogram. Effort S–M. Value is medium for transport analysts.

---

## 5. Animal movement ecology

### 5.1 Brownian bridge and dynamic BBMM (rank 5)

**Method.** Horne et al. 2007 (Ecology 88): a Brownian bridge between fixes, with location error and a
maximum-likelihood Brownian motion variance. **UNVERIFIED (snippet)**.

**Dynamic BBMM** (Kranstauber et al. 2012) estimates the variance in a sliding window with
behavioural change-point analysis. The R `move` API (4.2.6 docs) is:
`brownian.bridge.dyn(object, raster=1, dimSize=10, location.error, margin=11, window.size=31, ext=.3, …)`.
`time.step` defaults to the shortest interval divided by 15. The first and last segments are omitted.
https://bartk.gitlab.io/move/reference/brownian.bridge.dyn.html
Licence GPL-3 per a snippet: **UNVERIFIED**.

**GPU formulation:**
1. Variance estimation:
   - Per window (bounded at 31 fixes), evaluate the leave-one-out likelihood on a fixed grid of
     variance values (for example 64 candidates). Arg-max replaces the optimiser.
   - The change-point step compares at most (window − 2·margin)/2 split models per window by BIC. This
     is bounded and unrolled.
   - Parity with `move` will need the same optimiser tolerance. Pin by comparing UD volumes, not raw
     variances.
2. UD raster: for every (cell, segment) pair, sum over bounded sub-steps a Gaussian with
   bridge-interpolated mean and variance. Restrict each segment to its 4σ bounding box and use an atomic
   or gather pattern.
3. Volume isopleths: sort cell probabilities, take a prefix sum, and map to a cumulative volume. This is
   the same as `getVolumeUD` and reuses existing sort and scan.

**Capacity:** about 100k fixes × about 1M cells (gather per cell over a segment index).
**Effort:** M. **Risk:** medium (variance optimiser parity).

### 5.2 AKDE and continuous-time models

- ctmm 1.3.0 (2025-07-26, GPL-3) fits continuous-time stochastic process models (OU, OUF) and AKDE.
  https://cran.csail.mit.edu/web/packages/ctmm/index.html
- The vignette says model selection (`ctmm.select`) comes first. The exact weighting algorithm is
  "prohibitively slow" for about 10k–100k points, and the fast one needs a time grid `dt`.
  https://cran.r-project.org/web/packages/ctmm/vignettes/akde.html
- **Verdict:** NOT worth reproducing. It is an iterative likelihood optimisation with a model-selection
  loop. The GPU could only take a fitted model's bandwidth and evaluate the KDE, which the existing
  point-density kernel already does.

### 5.3 Recursion, residence time, first passage time (rank 6)

- `recurse` (Bracis, Bildstein & Mueller, Ecography 2018) provides `getRecursions(radius)`: revisits,
  residence time and time between visits. It also supports polygons, and
  `calculateIntervalResidenceTime` for user-specified intervals.
  https://ftp.fau.de/cran/web/packages/recurse/readme/README.html
- CRAN 1.4.0 (2024-07-18), **MIT**: a rare permissive ecology oracle.
  https://cran.r-project.org/web/packages/recurse/index.html
- First passage time (Fauchald & Tveraa, Ecology 2003): the time to exit a circle of radius r, with
  variance of log FPT against r peaking at the area-restricted-search scale. **UNVERIFIED (snippet)**.

**GPU formulation:**
- *Recursions:* for each query point i, loop over the track's segments (bounded by a per-track cap, or
  use a spatial index of segments). Compute the entry and exit crossings of the circle analytically per
  segment, then count visits with segmented scans over the inside/outside flags. Residence time is the
  sum of inside intervals.
- *FPT:* for each point and each radius in a fixed list, do bounded forward and backward searches until
  the track leaves the circle. Interpolate the crossing time.

**Capacity:** about 100k points × about 10k segments per track with an index. **Effort:** S.
**Risk:** low.

### 5.4 HMM behavioural states, step-selection (rank 10)

- moveHMM 1.12 (2025-10-06, GPL-3): https://cran.r-project.org/web/packages/moveHMM/index.html
- momentuHMM 1.5.8 (2025-11-25, GPL-3): https://cran.r-project.org/web/packages/momentuHMM/index.html
- amt 0.3.1.0 (2026-01-30, GPL-3): https://cran.r-project.org/web/packages/amt/index.html.
  `random_steps(n_control=15)` followed by `fit_clogit`, per the vignette: **UNVERIFIED (snippet)**.

**GPU formulation:**
- Step length and turning angle are maps.
- With fitted parameters, gamma or Weibull step densities and von Mises angle densities feed the
  existing HMM forward-backward and Viterbi kernels from map matching (at most 4 states, bounded).
- Parameter fitting stays on the CPU (numerical optimiser). The GPU can evaluate the log-likelihood for
  a batch of parameter vectors per optimiser iteration.
- *SSF random steps:* draw n control steps per observed step from the fitted distributions (counter
  RNG), then sample rasters at the endpoints and along the paths (existing raster sampling). The
  conditional logistic fit stays on the CPU.

**Effort:** M. **Risk:** medium. RNG parity is impossible, so validate distributions statistically.

### 5.5 T-LoCoH

Lyons, Turner & Getz 2013 (Movement Ecology 1:2) use a time-scaled distance
TSD² = Δx² + s²·v_max²·Δt², kNN hulls, and hull sorting into isopleths. **UNVERIFIED (snippet)**;
Springer redirected to a login.

The kNN in TSD space is easy on the GPU, but hull unions into isopleths are polygon overlay, which is
heavy. **Verdict:** NOT recommended (low value over effort; R-Forge only).

---

## 6. Time geography (rank 9)

- Jansen & Kuijpers, "Visit Probability in Space-Time Prisms for Moving Object Data", TIME 2025
  (LIPIcs 355, paper 17, CC BY 4.0). It formalises visit probability as a probability space over
  velocity-bounded trajectories. The potential path area is the prism's spatial projection.
  https://drops.dagstuhl.de/storage/00lipics/lipics-vol355-time2025/LIPIcs.TIME.2025.17/LIPIcs.TIME.2025.17.pdf
- Other network prisms with uncertain anchors, kinetic prisms (with acceleration bounds) and
  probabilistic potential path trees (Downs & Horner 2012): **UNVERIFIED (snippet)**.
- r5r `time_window` computes one result per departure minute and reports percentiles of travel time;
  accessibility at the p-th percentile is computed from p-th percentile times.
  https://cran.nics.utk.edu/cran/web/packages/r5r/vignettes/time_window.html
  For non-transit modes on a static network the window is degenerate, so the useful analogue is the
  library's FIFO time-dependent routing.

**Formulation:**
- *Planar PPA:* the ellipse test |x − a| + |x − b| ≤ v·(t_b − t_a) is a per-cell map. It is trivial,
  and it chains into the existing encounter and dwell checks.
- *Network PPA / potential path tree:* one forward shortest-path tree from anchor a and one reverse
  tree to b (existing shortest-path code). An edge (u, v) is in the prism if
  d(a, u) + len + d(v, b) ≤ budget. Partial edges are clipped by interpolation.
  Oracle: NetworkX `single_source_dijkstra` on both directions.
- *Probabilistic visit density:* reuse the Brownian bridge kernel of rank 5, with the bridge clipped to
  the prism.
- *Departure-time accessibility:* run the FIFO time-dependent one-to-many routing for D departure times
  (unrolled batch dimension). Take per-destination percentiles by bounded sort over D (at most 60), then
  cumulative opportunities.

**Capacity:** network size as in the existing routing. D ≤ 60. **Effort:** S (PPA), M (departure
window). **Risk:** low.

---

## 7. Moving-object data management (rank 12, enabler)

- **MEOS / MobilityDB 1.3.0rc1** (2026-02-19), alongside 1.2.1:
  - New types: `tgeometry` and `tgeography` (discrete or step interpolation), and `tcbuffer`, `tpose`
    and `trgeometry` (linear interpolation; experimental).
  - Supports PostgreSQL 18 and PostGIS 3.6.0.
  - The binary format changed between 1.2 and 1.3.
  - https://lists.osgeo.org/pipermail/mobilitydb-dev/2026-February/000139.html
  - Implication: pin oracle tests to one MEOS version. Moving rigid bodies (`trgeometry`) are a
    possible future extension of the restriction operations.
- **MobilityDuck** (Hoang, Pham, Hoang & Zimányi, arXiv 2510.07963, Oct 2025):
  - Wraps MEOS in DuckDB with types stored as BLOBs.
  - Supported functions include `atTime`, `atGeometry`, `valueAtTimestamp`, `eDwithin` and
    `tDwithin`.
  - Precompiled packages exist for Linux, macOS and **DuckDB-Wasm**. Many MEOS functions are not yet
    implemented.
  - https://arxiv.org/html/2510.07963v1. Repo https://github.com/MobilityDB/MobilityDuck (licence
    UNVERIFIED).
- **OGC API – Moving Features – Part 1: Core** (OGC 22-003r3, document dated 2024-10-25):
  - Resources: `/collections/{id}/items` for moving features, `…/tgsequence` for temporal geometry
    sequences, `…/tproperties` for temporal properties, plus `{queryType}` geometry queries.
  - https://docs.ogc.org/is/22-003r3/22-003r3.pdf
  - MF-JSON (OGC 19-045r3) is version 1.0 per the search listing: **UNVERIFIED (snippet)**.
- **GeoArrow** format 0.2 supports XY, XYZ, XYM and XYZM, with all coordinates as `double`. It says
  nothing about time semantics. https://geoarrow.org/format.html
  - In practice (Lonboard/deck.gl), a trajectory is a GeoArrow LineString plus a parallel list column of
    timestamps with the same nesting.
    https://developmentseed.org/lonboard/latest/examples/air-traffic-control/
  - Recommendation: accept both LineString + list<timestamp> and XYM-with-epoch-seconds. Convert on the
    CPU to float32 offsets from a per-track float64/int64 epoch, and document the round-trip error.
- **H3:** `gridPathCells` returns `gridDistance + 1` adjacent cells. It can fail near pentagons or for
  distant cells, and its intermediate cells are not stable across library versions.
  https://h3geo.org/docs/api/traversal
  This makes H3 tokenisation of trajectories (for foundation models or space-time indexing) unsuitable
  as a pinned-oracle feature unless the endpoints only are tokenised. A plain fixed-resolution
  grid-cell tokeniser is safer.

---

## 8. Flow fields and transport (rank 4)

- **FTLE recipe** (OceanParcels Lagrangian_diags tutorial):
  1. Seed a regular grid and advect for T.
  2. Take finite-difference Jacobians of the flow map from neighbour separations.
  3. Form C = JᵀJ.
  4. FTLE = (1/T)·ln √λ_max.
  https://lagrangian-diags.readthedocs.io/en/latest/tutorials/FTLE_method.html (licence UNVERIFIED)
- **lcs-parcels** (MIT, GEOMAR, ©2026) provides FTLE, hyperbolic LCS (shrink lines) and elliptic LCS
  (vortex boundaries). It "contains no Parcels code": it emits a seed set and ingests advected
  positions. https://lcs-parcels.readthedocs.io/en/latest/
  This is exactly the split this library would use. lcs-parcels is a clean MIT oracle for FTLE given
  identical positions.
- Parcels v4 (alpha, unstructured grids) and OceanTracker connectivity: **UNVERIFIED (snippet)**.
- **Browser particle visualisation:** Mapbox webgl-wind (ISC per the repo sidebar) renders "up to 1
  million wind particles at 60fps" from a single snapshot field. https://github.com/mapbox/webgl-wind
  No 2024–2026 WebGPU wind-tracer reference was found. The library's existing particle advection is
  ahead of the public baseline.

**What to add:**
1. *Time-varying fields:* a ring of 2–3 time slices as textures or buffers, linear in time, in the RK4
   advection step. This is the prerequisite for everything else. Effort S–M.
2. *FTLE:* seed grid, then advect (existing), then a per-cell 2×2 Jacobian and closed-form eigenvalue.
   Effort S. Backward FTLE uses negative time.
3. *Connectivity matrix:* region ID at the start and end per particle, then a 2-D histogram, then row
   normalisation. Effort S. Reuses the region mask.
4. *Residence time / exposure:* per particle, a bounded sum over steps of the inside-region flag.
   Effort S.
5. Elliptic and hyperbolic LCS extraction (ridge and curve tracing) is NOT recommended. It is a
   data-dependent curve integration and is better done on the CPU from the FTLE raster.

**Capacity:** 1M particles × several hundred steps. **Risk:** low.

---

## 9. Trajectory foundation models and privacy (descriptive)

- **Survey/tutorial:** Yang, Sun, Cheng, Lin, Torp & Hu, arXiv 2511.20729 (Nov 2025).
  https://arxiv.org/html/2511.20729v1
  - Tokenisations: raw GPS points at fixed intervals, grid-cell IDs, map-matched road segments, text
    and POIs, and image patches.
  - Models: START (ICDE 2023), LightPath (KDD 2023), RED (PVLDB 2024), TrajRL (ICDE 2025), MM-Path
    (KDD 2025), Path-LLM (WWW 2025), UniTR (AAAI 2025), GTR (ICML 2025).
  - Primitives needed: map matching, grid transformation, masking and top-k retrieval.
- **What a compute library should provide:** resampling (exists), map matching (exists), a grid-cell
  tokeniser, a road-segment sequence exporter, and batched embedding kNN. Do not train models.
- **Privacy:** Shaham, Ghinita & Shahabi (arXiv 2202.12342, 2022) publish DP OD matrices with
  intermediate stops over a density-aware partitioning. https://arxiv.org/abs/2202.12342v1
  Hierarchical TopDown DP OD (Boninsegna & Silvestri 2024) and HDX Private-OD: **UNVERIFIED
  (snippet)**.
  - The GPU-relevant part is small: contribution bounding (cap trips per user, as a segmented rank),
    then Laplace or geometric noise, then cell suppression.
  - Privacy accounting and RNG security belong outside a WebGPU kernel. If offered at all, offer it as
    a CPU post-step on GPU-aggregated counts.

---

## 10. Published GPU implementations (summary)

| Work | Year | What | Verification |
|---|---|---|---|
| Chang et al., TrajSimiMeasures (PVLDB 17) | 2024 | 7 CUDA (Numba) measures; anti-diagonal DP | Opened arXiv 2311.00960v3 |
| SoftDTW-CUDA-Torch | 2026 | Tiled anti-diagonal kernels; fused distance; MIT | Opened arXiv 2602.17206 |
| cuSpatial 25.04 | 2025 | Directed Hausdorff, haversine, quadtree joins | Opened RAPIDS docs |
| cuDTW++ (Euro-Par) | 2020 | Warp-intrinsic subsequence DTW | UNVERIFIED (snippet) |
| Gudmundsson & Valladares (TPDS) | 2015 | GPU continuous-Fréchet subtrajectory clustering, up to 11x | UNVERIFIED (snippet) |
| Fort, Sellarès, Valladares (IJGIS) | 2014 | GPU flock reporting | UNVERIFIED (snippet) |
| Hu & Chen PDCLUR (IJGI) | 2021 | GPU loose congestion patterns, >65x | UNVERIFIED (snippet) |
| Dauch et al. (Computers & Fluids); Conti et al. (JCP) | 2018; 2012 | GPU FTLE | UNVERIFIED (snippet) |
| Mapbox webgl-wind | — | 1M particles at 60 fps (WebGL) | Opened repo |

No published WebGPU (WGSL) implementation of any trajectory-analysis method was found. This library
appears to be first.

---

## 11. Not worth doing on this platform

| Method | Why not |
|---|---|
| ctmm model fitting and AKDE weights | Iterative likelihood optimisation and model selection; GPL-3; the vignette itself flags scaling limits. Use the existing KDE with a supplied bandwidth instead. |
| Infostop Infomap step | Stochastic multi-trial community detection; 0.2.0 changed labels. Use DBSCAN or connected components (trackintel default) instead. |
| SQUISH-E | Priority-queue removal order is inherently sequential. TD-TR or DP with a depth cap covers the need. |
| Exact maximal flocks; exact evolving convoys | NP-hard (flocks); O(2n²) disks per tick; candidate-set semantics; no open oracle. Ship moving clusters and pairwise co-membership instead. |
| T-LoCoH isopleths | Polygon-union heavy; R-Forge-only oracle. |
| Training t2vec, TrajCL or foundation models; Deep Gravity training | Belongs in an ML runtime. Provide tokenisers and embedding kNN only. |
| Elliptic and hyperbolic LCS curve extraction | Data-dependent curve tracing; do it on the CPU from the GPU FTLE raster. |
| Real-entropy predictability as a headline metric | O(n²) per user is feasible, but the estimator is contested. Offer it only next to the uncorrelated and random entropies. |
| H3 path tokenisation with pinned outputs | `gridPathCells` intermediate cells are not version-stable. |
| Differential privacy release | Accounting and secure randomness belong outside WGSL. Only the counting is GPU work. |
| Transit-style time windows (r5r) | Out of scope (no transit). The non-transit analogue is the FIFO departure batch in §6. |
| TRACLUS (for now) | The only Python oracle self-declares as not bug-free. MDL partitioning parity is fragile. Revisit with a hand-checked reference. |

---

## 12. Suggested sequencing

1. Rank 3 cleaning and splitters, then rank 1 staypoints, locations and triplegs. Together they unlock
   rank 2 indicators, trace-derived OD (rank 8) and motifs.
2. Rank 4 time-varying fields with FTLE, connectivity and residence time. Small and independent.
3. Rank 6 recursion and FPT, then rank 5 BBMM/dBBMM. These open the ecology audience.
4. Rank 7 additional measures and lower bounds for the existing top-k.
5. Rank 9 time-geography items reuse routing. Rank 11 collective patterns last.
