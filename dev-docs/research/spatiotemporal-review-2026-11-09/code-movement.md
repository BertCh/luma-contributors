# Movement and time-dependent flow code review

Scope: trajectory contributors in `modules/analysis/src/gpu-spatial-analysis` (trajectory-*, track-similarity, fleet-dwell recipe), `gpu-network` (map matching, time routing, flow aggregation, edge bundling, track-quality recipe), vector fields (`raster-analysis` particle advection, flow texture, field sampling, `gpu-terrain` terrain flow field), the trajectory port in `analysis-core/contracts/ports.ts`, and a short look at `modules/kepler/src`.

Method: I read the code end to end (no tests or builds were run). Paths are relative to `modules/analysis/src` unless they say otherwise. Tags: **CONFIRMED** means I traced the code path. **SUSPECTED** means the mechanism is in the code but I did not verify the runtime consequence.

Roadmap rows already covered and not repeated here: NW06/X04 (FIFO only, no transit) and NW-KSP. The coverage matrix rows are B-MV01/02 and MV01–MV03. Section 3 adds one precision correction to MV01 and a note on what NW06's evidence actually tests.

---

## 1. Inventory

| Symbol | File | Operation | Data / time model | Capacity limits | Reference oracle | Spec |
| --- | --- | --- | --- | --- | --- | --- |
| `GPUTrajectoryMetrics` | trajectory-analysis/gpu-trajectory-metrics.ts:227 | Length, duration, average and maximum speed, per-row speed, heading and acceleration, speed-threshold dwell "stops", stop centroids | Shared port (`positions` f32x2, `trackOffsets`, `timestamps` f32, f32+`timestampsLow` double-single, or u32x2 Int64 words). Planar or lon/lat great-circle | Stop capacity is set at compile time. The centroid pass dispatches `2*ceil(rows/2)` segments of 256 threads each | In-repo oracle; geodesic-trajectory.py (pyproj/haversine) | test/gpu-spatial-analysis/trajectory-analysis/* (27 tests) |
| `GPUTrajectoryResample` | trajectory-interpolation/gpu-trajectory-resample.ts:132 | Fixed `sampleCount` per track by time, arc length or shared clock | Shared port; f32 or words (no double-single). Planar or slerp | `trackCount*sampleCount < 2^32`. Arc length is a serial loop per track | In-repo oracle | trajectory-interpolation/* (14) |
| `GPUTrajectoryPlayhead` | trajectory-interpolation/gpu-trajectory-playhead.ts:179 | Per-track position, heading and speed at a playhead, with `maximumGap` status | Shared port; f32 or words plus an f32 fraction | One thread per track (binary search) | In-repo oracle | same |
| `GPUTrajectoryEncounters`, `addClockEncounters` | trajectory-encounters/gpu-trajectory-encounters.ts:206, clock-encounters.ts | Discrete same-bucket within-distance pairs (MobilityDB `tdwithin` preview) | Dense clock table; planar or great-circle | `columns*rows*bucketCount <= 2^25-1`; `trackCount*bucketCount < 2^31`; full sort of `hitCapacity` every encoding | parity-wp2-trajectory-encounters.py | trajectory-encounters/* (17) |
| `GPUTrajectoryClosestApproach` | trajectory-encounters/gpu-trajectory-closest-approach.ts:167 | Exact CPA, first time, duration within distance (linear motion) | Shared port; planar only | At most 4096 space-time cells per segment; entries <= 8.4M and hits <= 5.6M on a default device | trajectory-closest-approach.py | same |
| `GPUTrajectoryKalman` | trajectory-estimation/gpu-trajectory-kalman.ts:52 | Constant-velocity Kalman filter and RTS smoother (isotropic or full covariance) | **Own model:** `positions`, `times` f32 seconds, `offsets`. No word times. Planar meters | Serial per track | MovingPandas 0.22.4 KalmanSmootherCV (one 4-point fixture), NumPy | trajectory-estimation spec (8), network-movement-advanced.spec.ts |
| `prepareGPUTrajectoryObservations` | trajectory-estimation/prepare-trajectory-observations.ts:35 | Host-side WGS84 to local ENU and bigint epoch to f32 seconds | CPU | CPU only | pyproj | same |
| `GPUTrajectoryRestriction` | trajectory-restriction/gpu-trajectory-restriction.ts:77 | atTime/atGeometry clipping (rectangle, polygon <=128 vertices), ever/always, pairwise continuous `tdwithin` | **Own model:** f32 `times`, `offsets`. Planar | Pairwise: <=4096 pairs **baked into WGSL**, O(nA·nB) per thread; `intervalsPerSegment <= 256` | MovingPandas clip, MobilityDB 1.2.1 | network-movement-advanced.spec.ts, trajectory-estimation spec |
| `GPUZoneEvents` | trajectory-zones/gpu-zone-events.ts | Enter/exit events, dwell, visits, sparse visit table | Shared port; f32 or words; planar | Dense `trackCount*zoneCount` scratch even for the sparse table; full sort of `candidateCapacity` | parity-wp2-zone-events.py | trajectory-zones/* (26) |
| `GPUTrackSimilarity` | track-similarity/gpu-track-similarity.ts | Hausdorff, discrete Fréchet, Fréchet decision, DTW (sum of distances), LCSS, max distance | **Lines, no time** (`linesA/B`, `pairA/B`); planar or great-circle | Shorter track <= `maximumFrechetVertexCount` (<=2048, default 256); pairs < 2^24 | Shapely, similaritymeasures, tslearn | track-similarity/* (22) |
| `GPUTrackSimilaritySearch` | track-similarity/gpu-track-similarity-search.ts | Threshold search with an admissible bound, survivor compaction, exact top-k | Lines; **planar only** (no `spatialContext`) | topK <= 1024; top-k kernel is O(T·P·K²) | tslearn, similaritymeasures | trajectory-estimation spec, network-movement-advanced |
| `addFleetDwellRecipe`, `addFleetDwellZoneEventsRecipe` | recipes/fleet-dwell-recipe.ts:270, :530 | Stops to point-in-polygon to per-zone stats; zone-events variant | Shared port; planar | Inherits metrics and zone limits | None directly | recipes/fleet-dwell-recipe.spec.ts (2) |
| `addTrackQualityRecipe` | gpu-network/recipes/track-quality-recipe.ts | Per-row invalid, speed and gap flags; optional similarity and map matching | Shared port | One serial thread per track | None | geospatial-expansion/recipe-review*.ts (1 case) |
| `GPUMapMatching` | gpu-network/map-matching/gpu-map-matching.ts | Newson–Krumm HMM: grid candidates, bounded A* or UBODT routes, Viterbi | `positions` + `trackOffsets`, **no timestamps**; planar or great-circle | K <= 8; route node budget 8–128; grid <= 2^24 cells; Viterbi serial per track; `routes` scratch `points*K*(K+1)*4` B not validated | Composite NetworkX/Shapely HMM (parity-wp3), Chicago subset | map-matching/* (20) |
| `GPUNetworkTimeRouting`, `prepareGPUNetworkRestrictionStates` | gpu-network/network-optimization/gpu-network-time-routing.ts | FIFO earliest arrival (affine or piecewise) on a CPU-expanded restriction automaton | f32 relative departure | <=256 states, <=4096 transitions, **1 thread** | NetworkX time-expanded (no Valhalla run) | network-movement-advanced.spec.ts |
| `GPUFlowAggregation` | gpu-network/flow-aggregation/gpu-flow-aggregation.ts:283 | OD aggregation into top-K flows and zone totals, optional instant time window | Per-row origin/destination positions or IDs | Zones <= 65535; power-of-two `pairCapacity` | In-repo oracle | flow-aggregation/* (24) |
| `GPUEdgeBundling` | gpu-network/edge-bundling/gpu-edge-bundling.ts:239 | KDEEB visual bundling | Node positions and edges; no time | Iterations <= 64; points per edge <= 64 | In-repo oracle | edge-bundling/* (15) |
| `GPUParticleAdvection` | raster-analysis/particle-advection/gpu-particle-advection.ts:142 | RK2 particles with Philox respawn and trails | Steady field per frame, planar f32 | Field <= 65535 per axis | particle-advection-cpu.ts, parity-wp4-advection.py | gpu-raster/particle-advection (10) |
| `GPULineIntegralConvolution`, `GPUStreamlines` | raster-analysis/flow-texture | LIC and Jobard–Lefer streamlines | Steady field | Compile-time step counts and iterations | CPU oracles, parity-wp4 | gpu-raster/flow-texture (16) |
| `getFieldSamplingWGSL` | raster-analysis/vector-field-common/vector-field-sampling.ts | Bilinear cell-centre sampling with NaN as no data | Planar | n/a | CPU mirror | (via the above) |
| `GPUTerrainFlowField` | gpu-terrain/terrain-flow-field/gpu-terrain-flow-field.ts:140 | Uniform wind projected onto the DEM tangent plane | Raster frame | One kernel | terrain-flow-field-oracle.ts, parity-wp5 | gpu-terrain/terrain-flow-field (9) |

Also relevant: TD-TR exists as `GPULineSimplification` with `metric: 'time-ratio'` (gpu-spatial-analysis/line-simplification, f32 timestamps only).

---

## 2. Data model and time model

**Shared port.** `GPUTrajectoryGeometryPort` in analysis-core/contracts/ports.ts:56-64 is `positions` f32x2, `trackOffsets` (`trackCount + 1`) and `timestamps` (f32 or u32x2 Int64 words). The track ID is implicit: it is the index into `trackOffsets`. Optional `sourceIds` exist. Rows must be sorted by track and then time. **This ordering is never validated.** The resample, playhead and zone binary searches silently give wrong answers on unsorted input (CONFIRMED by reading; the precondition is documented). Metrics treats backward steps as speed 0.

**The model is not shared across contributors** (CONFIRMED):

| Contributor | Positions | Time | Track delimiter |
| --- | --- | --- | --- |
| Metrics, Resample, Playhead, Zones, ClosestApproach, Encounters | shared port | f32 \| words (Metrics also double-single) | `trackOffsets` |
| Kalman (gpu-trajectory-kalman.ts:34) | planar f32x2 | f32 seconds only | `offsets` |
| Restriction | planar f32x2 | f32 only | `offsets` |
| MapMatching | f32x2 | none | `trackOffsets` |
| TrackSimilarity / Search | lines (`lineOffsets`) | none | `lineOffsets` |
| TD-TR (GPULineSimplification) | lines | f32 only | `lineOffsets` |
| Kepler trip animation (modules/kepler/src/analysis/trip-animation.ts) | n/a | f32 ms relative to a host origin, with an error budget | n/a |

Kepler does not import any trajectory contributor. It only uses `GPUTimeWindowFilter` with word timestamps for selection (`selection/gpu-time-selection.ts`).

**Words mode.** dataframe-analysis/time-window-filter/time-words.ts subtracts two Int64 times exactly with a borrow and only then rounds to f32. Differences are therefore exact below 2^24 units and relatively accurate (about 2^-23) above that. That is excellent for differences between consecutive rows. It does not help quantities measured from a distant reference. Examples:
- resample clock offset `clockStart - trackFirst`, and `getProgress(row, trackStart)` (trajectory-interpolation-kernels.ts:191-212, :654-656)
- zone event times, relative to the track's first sample
- closest-approach cell boxes, measured from row 0 of the whole buffer (closest-approach-kernels.ts:152-153); the CPA itself is measured from the segment row and stays exact

**1 Hz data over three weeks.** About 1.8e6 s or 1.8e9 ms.

- *f32 relative seconds or milliseconds.* The ULP at the end of the window is 0.125 s (or 128 ms). Samples stay distinct, but each step duration carries up to ±0.125 s of error, so step speeds can be off by up to about 12.5% late in the window. 10 Hz data is unusable there (step 0.1 s < ULP).
- *f32 over one year.* 3.15e7 s has a ULP of 2 s, so 1 Hz samples collapse into duplicates. Metrics then reports speed 0 and "slow" zero-distance steps; Kalman sees `delta = 0` or `delta = 2`.
- *Words.* Consecutive-row metrics are exact. Resample and zone times relative to a track start three weeks earlier are quantized to 128 ms. The position error is at most speed × 64 ms, which is negligible for vehicles.
- *Kalman and Restriction.* Both only accept f32 times relative to one global epoch, so they have the f32 problem above regardless of how the data arrived (see F2).

**Coordinates.** All coordinates are f32. On lon/lat the quantum is about 1.7 m at |lon| ≥ 128° (2^-16°) and about 0.42 m in latitude at 32–64°. The code warns about this (`warnGPUShortLongitudeLatitudeDistance`) and recommends `createGPULocalFrame`.

Geometry support by contributor:
- *Great circle:* Metrics, Resample (slerp), Playhead, Encounters, TrackSimilarity, MapMatching.
- *Planar only:* ClosestApproach, Zones, Restriction, Kalman, SimilaritySearch, fleet-dwell recipes, particle advection and flow texture.
- *Geodesic (ellipsoidal):* nothing. The only ellipsoid code is the host WGS84 to ENU transform in `prepareGPUTrajectoryObservations`. It uses one tangent plane for all data, so tracks spanning hundreds of kilometres accumulate curvature error. That is documented.

**Gap model is inconsistent** (CONFIRMED). A maximum-gap concept exists in Playhead (`maximumGap` and a `gap` status), Kalman (`maximumGap` reset), Restriction (unknown support) and track quality (a flag only). It does **not** exist in Resample (clock mode included), Encounters, ClosestApproach, Zones or Metrics. All of those interpolate straight across any gap (see F1).

---

## 3. Correctness findings

**F1 — Clock resample and clock encounters interpolate across data gaps (CONFIRMED).**
- *Where:* trajectory-interpolation-kernels.ts:694-697. A sample is absent only when `targetProgress` is outside `[0, total]` (the whole track's span). The surrounding code then interpolates between the bracketing rows whatever the interval is.
- *Scenario:* an AIS vessel reports at 08:00 at A and then at 20:00 at B, 200 km away. Every clock bucket between those times gets a straight-line position. `addClockEncounters` (clock-encounters.ts) reports "encounters" with any ship that crossed that line, and `bucketCounts` counts time the vessel was never observed.
- *Contradiction:* clock-encounters.ts says "a pair only encounters while both tracks exist", which is not what the code does for gappy data.
- *Same behaviour elsewhere:* ClosestApproach treats the gap segment as linear motion (see F7). Zones dwell counts the gap time as inside.
- *Reference:* MovingPandas also interpolates linearly, but MobilityDB with sequence sets does not interpolate across gaps. The repo's own Restriction contributor explicitly refuses to bridge gaps (gpu-trajectory-restriction.ts:74-75), so the library contradicts itself.

**F2 — Kalman times lose precision against one global epoch; the "bigint epoch" claim is weaker than MV01 implies (CONFIRMED).**
- *Where:* prepare-trajectory-observations.ts:86-90. The integer quotient and remainder protect only the f64 step. The result goes into a `Float32Array`. The kernel then computes `delta = time - previousTime` in f32 (gpu-trajectory-kalman.ts:155).
- *Scenario:* 10 Hz GNSS with `epoch` = the dataset start, three weeks of data. Late rows have deltas of 0 or 0.125 s instead of 0.1 s, so the process covariance and velocity are wrong. Per-track epochs are not supported: there is one `epoch` argument.
- *Restriction:* same issue. It only takes f32 `times`.
- *Correction to MV01:* the bigint protection ends at epoch subtraction. The kernel precision is f32 relative to that epoch, not per step.

**F3 — Stop detection is per-step speed, not MovingPandas' diameter-based stay points; stationary GPS jitter splits stops (CONFIRMED mechanism).**
- *Where:* trajectory-metrics-kernels.ts:200-206. A step is "slow" when `distance < threshold * deltaTime`.
- *Scenario:* a parked vehicle at 1 Hz with 3–5 m multipath jitter and `stopSpeedThreshold` = 2 m/s. Each 5 m jump breaks the run, so a 30-minute stop becomes many short dwells, most below `stopMinimumDuration`. MovingPandas `TrajectoryStopDetector(max_diameter, min_duration)` reports one stop.
- *Second case:* raising the threshold makes slow driving in congestion register as a stop.
- *Lon/lat:* the documented f32 quantum (about 1.7 m) adds jitter on its own.
- *Fleet dwell:* the stops variant of the fleet-dwell recipe inherits this.

**F4 — Track quality flags the valid point after a speed outlier (CONFIRMED).**
- *Where:* track-quality-recipe.ts:98-102. The speed flag is the speed of the step ending at a row.
- *Scenario:* rows 0..4 at 1 m/s, with row 2 displaced 500 m. Step 1→2 is fast, so row 2 is flagged (correct). Step 2→3 is fast too, so row 3 is also flagged, although it is valid.
- *Why it matters:* a cleaner that drops flagged rows removes a good point. MovingPandas `OutlierCleaner` evaluates the point, not the incoming step.
- *Also:* the first row of a track is never speed-tested, so an outlier at row 0 is invisible.
- *Tests:* the only recipe test (recipe-review.spec.ts:127) covers just invalid times and thresholds.

**F5 — Map matching cannot move backward on a one-way edge; stationary jitter causes breaks (CONFIRMED mechanism; the break count depends on the data).**
- *Where:* gpu-map-matching.ts:1435 and :1448-1453. A same-edge transition is allowed only when `toFraction >= fromFraction`. Otherwise the route must leave the edge and loop back through the network, and that loop must fit within `routeFactor * straight + routeSlack`. With a 2 m GPS displacement the bound is 3·2 + 50 = 56 m, far less than a city block loop.
- *Scenario:* a car stopped on a one-way street whose fixes jitter backwards along the street. That transition becomes infeasible, producing a break (`breaks = 1`) or a jump to another candidate.
- *Reference:* Newson & Krumm §4 drop points within 2σ of the previous one to avoid exactly this. That preprocessing is absent.
- *Tests:* the parity fixture states "Break/restart semantics within a track ... are deliberately not covered" (test/oracles/parity-wp3-map-matching.py, docstring).

**F6 — Viterbi scores are not renormalized per step (CONFIRMED mechanism, SUSPECTED impact).**
- *Where:* gpu-map-matching.ts:1570. `score = fromScore + transition - |d|/beta` accumulates over the whole unbroken stretch in f32.
- *Effect:* a stretch of 100k points reaches magnitudes of roughly 1e5–1e6, where the f32 ULP is 0.008–0.06. Candidate comparisons whose true difference is smaller become ties, broken by index. Reference implementations subtract the per-step maximum.
- *Impact:* small, except on very long tracks that never break.

**F7 — Closest approach: one long gap segment raises a global overflow (CONFIRMED).**
- *Where:* closest-approach-kernels.ts:169-172. A segment whose padded space-time box covers more than 4096 cells is dropped and sets the global `entryOverflow`.
- *Scenario:* `timeCellSize` = 60 s, `cellSize` = 500 m, and one track has a 12-hour gap during which it moved 5 km. That segment covers 721 time cells × 121 spatial cells, about 87k cells.
- *Effect:* the segment is dropped and the whole batch is flagged "pairs may be missing". With F1, there is no way to say "skip gap segments".

**F8 — Encounters with great-circle clock samples drop samples near the antimeridian unless the bounds wrap (SUSPECTED).**
- *Where:* resample slerp keeps longitude continuous with the segment start, so a crossing yields values such as 181° (trajectory-interpolation-kernels.ts:81-90). Encounters treats samples outside `bounds` as absent and folds longitudes only when the bounds are exactly [-180, 180] (gpu-trajectory-encounters.ts, `bounds` doc).
- *Effect:* with regional bounds such as [170, -60, 190, 60], samples past 180 are folded nowhere and lost.
- *Not verified:* I did not check whether the positions kernel folds in non-wrap mode.

**F9 — TrackSimilarity treats trajectories as untimed polylines (CONFIRMED, by design but relevant).**
- DTW, Fréchet and LCSS run over vertex indices. LCSS's "temporal" constraint is an index band (`lcssBandRadius`), not a time δ as in Vlachos et al. and MovingPandas-style LCSS.
- Comparing a 1 Hz track with a 0.1 Hz track by DTW is dominated by the sampling ratio unless both are resampled first, and nothing enforces that.
- `GPUTrackSimilaritySearch` has no `spatialContext` (gpu-track-similarity-search.ts:24-56 and :273), so lon/lat input is measured as planar degrees. That is silently wrong; it does not throw.

**F10 — Search DTW bound is weak (CONFIRMED, performance not correctness).**
- *Where:* gpu-track-similarity-search.ts:204. The bound is `max(start distance, end distance)`.
- *Why weak:* for sum-of-distances DTW, `start + end` is admissible whenever both tracks have at least 2 vertices, and LB_Keogh or LB_Kim give much tighter bounds.
- *Effect:* survivors stay high and the expensive dynamic program runs more often.

**F11 — Time routing semantics are simpler than Valhalla's.** This is a note, not a bug.
- *Affine model:* `arrival = t + base + slope*t` uses the absolute relative time `t` (gpu-network-time-routing.ts:156), so travel time grows without bound with `t`. It is not a periodic time-of-day profile.
- *Evidence:* the oracle provenance says "no pgRouting/Valhalla server installed" (test/oracles/network-movement-advanced.py:127). NW06's evidence is a NetworkX time-expanded graph. The coverage matrix states this honestly, but "Valhalla semantics" is not tested against Valhalla.

**F12 — Particle advection and flow textures assume a steady, planar field (CONFIRMED).**
- Fields are sampled from one `velocities` buffer per frame (vector-field-sampling.ts). There is no time-varying field and no interpolation between two time slices; the caller can swap buffer contents per frame, but the RK2 midpoint then samples the new field at the old time.
- Lon/lat fields with u, v in m/s are not converted to degrees per second (no `cos(lat)` scaling), so wind on a geographic grid streams east-west at the wrong rate unless the caller pre-scales it.

Edge cases checked and handled correctly:
- *Duplicate timestamps:* resample, playhead and metrics resolve them to the last row and never divide by zero. Closest approach drops `dt ≤ 0` segments. Restriction marks a zero-duration segment with movement as invalid (status 4), and that propagates to `unknownCount` for the track. Kalman with `delta = 0` updates twice, which is mathematically fine.
- *Metrics stop run capacity:* `ceil(rows/2)` is a correct upper bound on slow runs (gpu-trajectory-metrics.ts:421).
- *Restriction pairwise CPA:* evaluating the distance at closest approach before subtracting the radius is numerically sound (trajectory-pairwise-kernels.ts:66-75).

---

## 4. Scaling walls

| Wall | Location | Effect |
| --- | --- | --- |
| Restriction pairwise: <=4096 pairs **compiled into the WGSL as a const array**, O(nA·nB) nested loop on one thread per pair | trajectory-pairwise-kernels.ts:37, :45-47; gpu-trajectory-restriction.ts:136-142 | Two 1 Hz day-long tracks (86,400 rows each) mean 7.5e9 iterations on one thread, so a GPU timeout is likely. Any change of pairs recompiles. A two-pointer merge over the time-sorted segments would be O(nA+nB). |
| Similarity search top-k: one thread per A-track scans all P pairs for each of K slots and checks the K previous picks | gpu-track-similarity-search.ts:319-339 | O(T·P·K²). Example: T = 4096, P = 200k, K = 10 gives about 8e10 operations. |
| LCSS bound O(n·m) serial per pair | gpu-track-similarity-search.ts:194-202 | 2048 × 2048 vertices is 4M operations per thread. |
| Fréchet, DTW and LCSS cap on the shorter track | gpu-track-similarity.ts (cap 2048, default 256) | Raw 1 Hz tracks must be resampled or simplified first. |
| Map matching scratch: `routes = points*K*(K+1)*4` B, `candidates = points*K*12` B; only the edge count is validated | gpu-map-matching.ts:555-560 (validation only at :472) | With K = 8 that is 288 B per point, so about 466k points fill a 128 MiB binding. No explicit check or chunking; how it fails is SUSPECTED (device validation error). |
| Map matching Viterbi: forward and backward passes are one thread per track | gpu-map-matching.ts:1538 | One long track serializes; there is no intra-track parallel Viterbi (for example a max-plus scan). |
| Map matching route budget <= 128 nodes, K <= 8 | gpu-map-matching.ts:384-385 | Samples more than about 1 km apart on a 100 m grid need the UBODT table. Dense urban junctions with 8 directed candidates miss parallel roads. |
| Time routing: 1 thread, <=256 states, <=4096 transitions, Bellman–Ford | gpu-network-time-routing.ts:57-61, :120 | A reference toy, not a router. Not on the roadmap as a scale item (NW06 only covers semantics). |
| Encounters lattice × buckets <= 2^25-1 | gpu-trajectory-encounters.ts:369-374 | A 20 km city at 50 m cells is 160k cells, so at most about 209 buckets per encoding (3.5 h at a 1-minute clock). Long windows must be tiled by the caller. |
| Closest approach: <=4096 cells per segment, global overflow | closest-approach-kernels.ts:169 | See F7. |
| Zones: dense `trackCount*zoneCount` scratch (initial inside, dwell, visits, cell offsets, first and last times) even when only `visitTable` is wanted | gpu-zone-events.ts:545, :678-691 | 100k tracks × 10k zones is 1e9 cells, which cannot be allocated. The docs admit this (it bounds output, not memory). |
| Resample arc length: sequential per-track loop | trajectory-interpolation-kernels.ts:532-547 | One 1M-row track is one serial thread. A segmented scan (published `GPUScan`) would fix it. |
| Metrics centroids: segmented reduction over `2*runCapacity` = about `rowCount` segments, one 256-thread workgroup each, most of them empty "gap" segments | gpu-trajectory-metrics.ts:562-601; sorted-segment-sums.ts:298 | 10M rows means 10M workgroups (2.56e9 invocations) just for centroids. Running segments only over the compacted stop list would scale with the stop count instead. |
| Kalman, track quality and restriction (non-pairwise): one serial thread per track | — | These scans are inherently sequential, but parallelism collapses for few long tracks. |
| Flow aggregation <= 65535 zones (u32 pair key) | flow-aggregation-pairs.ts:9 | Rules out road-edge OD or link-to-link matrices on real networks. |
| Pair enumeration without an index | TrackSimilarity, SimilaritySearch, Restriction pairwise | Pairs are caller-supplied. Only Encounters (grid with a time axis) and ClosestApproach (hashed space-time grid) generate candidates with an index. |

---

## 5. Composability map

Edge types: **GPU** = no readback, **GPU+glue** = a small kernel the user must write (not provided), **CPU** = host work or recompile required.

```
raw port (positions, trackOffsets, ts f32|words)
 ├─GPU→ Metrics → stops → PointInPolygonJoin → GroupStatistics      (fleet-dwell recipe)
 ├─GPU→ ZoneEvents → visitTable / dense dwell
 ├─GPU→ Playhead (render)                       [kepler does not use it]
 ├─GPU→ ClosestApproach (planar)
 ├─GPU→ Resample(clock) → Encounters → pairs(ids, partners, count)
 │                                          ├─GPU→ TrackSimilarity(activePairCount) on raw lines
 │                                          ├─GPU (sentinel pairs)→ SimilaritySearch
 │                                          └─CPU (readback + recompile)→ Restriction pairwise
 ├─GPU+glue→ Resample(time|arc) → TrackSimilarity   (needs uniform lineOffsets k*S; clock output has NaN rows, so unusable)
 ├─CPU→ Kalman / Restriction   (f32 seconds + `offsets`; words input needs host prep or a conversion kernel)
 │        Kalman out f32x4 → GPU+glue gather → f32x2 for any downstream
 ├─GPU→ TD-TR (GPULineSimplification, f32 ts only; words → CPU or glue)
 ├─GPU→ MapMatching → matchedEdges per point
 │        ├─GPU→ GroupAggregation(matchedEdges)  = point counts per edge (biased by sampling rate)
 │        ├─ ✗  traversed edge sequence: not produced, so no edge volumes or link flows
 │        └─ ✗  FlowAggregation 'ids' zones ≤ 65535, so edge-level OD is impossible on real networks
 └─GPU+glue→ first/last row gather → FlowAggregation (trip OD)       (no adapter; no trip segmentation)
TerrainFlowField ─GPU→ ParticleAdvection / LIC / Streamlines (raster-local frame)
prepareGPUNetworkRestrictionStates (CPU) → GPUNetworkTimeRouting (1 thread)
```

Points where the pipeline forces CPU work (CONFIRMED):
1. Restriction pairwise pairs are compile-time WGSL constants.
2. Kalman and Restriction accept f32 times only; `prepareGPUTrajectoryObservations` and its WGS84 to ENU step run on the host.
3. The time-routing restriction automaton is built on the host.
4. Bounds and capacities are compile-time: encounter `bounds`, `cellSize` and `bucketCount`; map-matching grid `bounds` and `cellSize`.
5. Track-quality flags have no compaction-to-clean-trajectory path. That one is GPU+glue: `GPUCompaction` plus offsets rebuilt with a scan.

Map matching cannot feed network flow aggregation in any meaningful sense: it emits the matched edge per GPS point, not the route between points (outputs at gpu-map-matching.ts:139-180).

---

## 6. Absent operations and extension points

S/M/L is the implementation size given the existing kernels.

| Operation | Status | Cheap path from existing kernels | Size |
| --- | --- | --- | --- |
| Stay-point / stop detection (diameter + duration, Li 2008 / MovingPandas) | Partial: speed-threshold dwell in `GPUTrajectoryMetrics` (F3) | Per-row "radius run" via a two-pointer per track (serial per track, like Kalman), or windowed max-distance with a segmented scan; reuse the Metrics run, compaction and gather nodes | M |
| Trip segmentation (gap, stop, temporal splitters) | Absent | Per-row break flags (gap from `timeDifferenceRows`, stop runs from Metrics `runStartFlags`/`runEndFlags`), then published `GPUScan`, then new `trackOffsets` plus a `segment → source track` map. Positions stay zero-copy and the output is a new `GPUTrajectoryGeometryPort` | S |
| Gap-aware interpolation (F1) | Absent in Resample, Encounters, ClosestApproach, Zones | Port Playhead's `maximumGap` test into the resample, closest-approach segment and zone crossing kernels | S |
| TD-TR | **Present** (`GPULineSimplification`, `'time-ratio'`) but f32 timestamps only | Add a words path using `TIME_WORDS_WGSL` | S |
| SQUISH / STTrace (online, priority queue) | Absent | Per-track serial heap in a fixed buffer; or approximate it with Visvalingam rounds using SED importance | M |
| Speed, acceleration and turn outlier cleaning | Flags only (track-quality), with the F4 off-by-one | Point-centred test (`min(in, out)` speed), then `GPUCompaction`, then rebuild offsets; iterate N rounds | S |
| Whole-trajectory clustering | Absent | Resample (time or arc length) to a dense `2·S` vector, then N-dimensional k-means in gpu-vector-search (not verified to accept that layout); or a similarity matrix plus HDBSCAN on precomputed distances (not supported: planar f32x2 only) | S–M |
| TRACLUS (sub-trajectory) | Absent | MDL partition (serial per track) plus segment-distance DBSCAN; would need a line-segment neighbour index | L |
| Co-movement (flock, convoy, swarm) | Absent | Clock resample, then per-bucket `GPUSpatialClustering` / DBSCAN on each bucket's samples (bucket as a key), then cluster-intersection chains across buckets (sort and join on `(bucket, cluster)` member sets). Encounters already provides the pair graph per bucket | M (flock/convoy), L (swarm) |
| Home range: KDE utilization distribution | Partial: `GPUKernelDensity` on points, plus a percent-volume contour (needs a sort of cell values) | Feed raw fixes per track (masking per animal), then a CDF threshold via sort and isolines | S |
| Brownian bridge movement model | Absent | Per-segment Gaussian bridge integrated over raster cells: a splat kernel per segment × cell window (like line density) with a time-weighted variance | M |
| T-LoCoH | Absent | Space-time kNN (published grid index plus a time axis), then hull union per point (concave-hull and overlay exist) | L |
| Time geography: space-time prisms, potential path area | Absent | Planar: PPA of a segment is an ellipse given `vmax`; rasterize per segment (like line density). Network PPA: two bounded Dijkstras (cost-distance or network routing) intersected | S (planar), M (network) |
| OD matrix from trajectories, and OD flow clustering | OD aggregation exists (`GPUFlowAggregation`); extraction is absent | First/last row gather (after trip segmentation), then FlowAggregation. Flow clustering (e.g. Zhu & Guo) needs 4-D OD-vector kNN | S / L |
| Edge traversal volumes from map matching | Absent (F5 area) | Store the A* predecessor per (point, from, to) route or re-run one search per chosen transition, then emit an edge-sequence CSR, then `GPUGroupAggregation` per edge | M |
| Space-time density of trajectories (STKDE / space-time cube) | Absent; `GPULineDensity` is spatial only | Line density with a time-window mask per slice; or a 3-D splat with a time axis like the Encounters grid | S (slice loop), M (true STKDE) |
| Continuous-time movement models (ctmm, OU/OUF) | Absent | Kalman CV exists; OU/OUF needs a different transition matrix plus likelihood-based parameter fitting (variogram fitting exists for spatial data) | L |
| Semantic enrichment (stops to POI, map-matched road attributes) | Partial: fleet-dwell recipe joins stops to zones | Nearest-feature join for stop centroids; matchedEdges then a gather of edge attributes | S |
| Lagrangian diagnostics (FTLE, residence time) | Absent | Seed a particle per grid cell and integrate N RK2 steps without respawn: a variant of `GPUParticleAdvection`, or of `GPUStreamlines` tracing with fixed time instead of normalized steps. Then a flow-map gradient kernel for FTLE, or an exit-time reduction for residence time | M |
| Time-varying vector fields in advection | Absent (F12) | Two velocity buffers plus a per-frame blend factor in `sampleField`; RK2 midpoint at t + h/2 | S |
| Geographic (lon/lat) advection | Absent (F12) | Scale `u` by `1/(R cos φ)` in the step, or require a local frame | S |

---

## 7. Top recommendations, ranked

1. **Unify the gap model (F1, F7).** Add `maximumGap` (per-frame parameter) to Resample (clock and time modes), ClosestApproach (skip gap segments instead of a global overflow), Encounters (via resample NaN) and Zones (unknown dwell). Playhead, Kalman and Restriction already have it. This is the most visible correctness risk for real AIS and GPS data. **S–M.**
2. **One trajectory port everywhere.** Make Kalman and Restriction accept `GPUTrajectoryGeometryFields` with word timestamps and per-row relative differences, as Metrics does. Add a words path to TD-TR. This removes the host epoch step (F2) and lets map-matched or cleaned outputs flow into Kalman without CPU work. Also validate (or optionally check on the GPU) the sort-order precondition. **M.**
3. **Trip segmentation and cleaning primitives.** A splitter that turns per-row break flags into new offsets, a point-centred outlier test (fixes F4), and diameter-based stay points (F3). These are the foundation for OD extraction and the most-used MovingPandas features. **S + S + M.**
4. **Remove the two quadratic single-thread walls.** Use a two-pointer time merge for Restriction pairwise, and take pairs from a GPU view (Encounters output) instead of compile-time constants. Use a segmented sort or select for SimilaritySearch top-k instead of O(T·P·K²). **M.**
5. **Map matching: route reconstruction and robustness.** Emit the traversed edge sequence (enables edge volumes and network flow), drop points within 2σ of the previous fix (Newson–Krumm preprocessing, F5), renormalize Viterbi scores per step (F6), and validate the `routes` and `candidates` binding sizes or chunk by track. **M.**
6. **Make SimilaritySearch context-aware** (throw on lon/lat or support great-circle), and use `start + end` (or LB_Kim) for the DTW bound (F9, F10). **S.**
7. **Scale fixes.** Make the Zones sparse path not allocate dense `trackCount × zoneCount` scratch, compute Metrics centroids over compacted stops only, and replace the serial arc-length loop with a segmented scan. **M.**
8. **Vector fields.** Two-slice time interpolation and a lon/lat velocity scale (F12), then FTLE and residence time on top. **S then M.**
9. **Test gaps.**
   - Map-matching breaks and restarts and stationary jitter (explicitly not covered).
   - Track quality: only one test, which covers neither outliers nor gaps.
   - Kalman MovingPandas parity: a single 4-point fixture.
   - Long-time-span precision tests: f32 versus words at weeks to a year.
   - Gap behaviour in clock encounters.
   - Fleet dwell: 2 tests.
10. **Kepler.** Trips use f32 ms relative to an origin with an error budget, and do not use `GPUTrajectoryPlayhead` or word times. Wiring Playhead (words mode) into trip rendering would remove the error budget for multi-week domains. **S–M**, adapter only.
