# Float32 at real-world magnitudes

## Summary

The highest-value change is to subtract a nearby origin **before** narrowing coordinates, times, or observations, and keep that origin out of the calculation until publication. Several contributors already do this.

- The public polygonization diagnostic lost an entire face after translation to projected coordinates. All four segments still came back classified as `closed-ring`. This result was **executed on the host**.
- Three public raster packers gave incompatible instructions for locating the same point on the same grid. At 0.85 m spacing, the modeled sampling column was 20 and the distance-field seed column was 19, even with `localOrigin` supplied.
- A 255-row, one-cluster example produced modeled inertia **254 instead of 0.9961**. The library already has the fix behind `centerCoordinates: true`. Its default favors compatibility with an unpublished release.
- Two observations suffice to turn perfect negative spatial correlation into zero. A 256-slice pressure series overstated EOF variance by **76.6%** before its 96-bit accumulator received the first product.
- Fifteen findings follow. Host outputs and packer bytes are executed results. All device-output claims are **SOURCE** predictions backed by an executed float32 transcription. Metal may fuse or reassociate operations.

## Census

I followed relative `export *` and named re-exports from the 11 entries in `modules/analysis/package.json`, using `typescript.createSourceFile` in Node without running the compiler. I deduplicated class names and counted classes declaring a `getCommandNodes` method directly. This found **761 public classes, 494 with a directly declared method**. The latter has a narrower definition than the earlier census of 498 producers.

| Measured source cohort | Variant | Public classes |
|---|---|---:|
| `flow-texture`, `particle-advection`, `lagrangian-diagnostics` | Advances an absolute float32 position during integration | 4 — `GPUStreamlines`, `GPULineIntegralConvolution`, `GPUParticleAdvection`, `GPULagrangianFlowMap` |
| Same three directories | Consumes a flow map or scalar field without integrating particles | 2 — `GPUFiniteTimeLyapunovExponent`, `GPUScalarFieldRidges` |
| `distance-field` and `raster-sampling` | Packs `f32(1 / f32(step))` | 1 — `GPUDistanceField` |
| Same two directories | Packs `f32(1 / step)` from the host number | 2 — `GPURasterSampling`, `GPURasterProfile` |
| Contributors on the EOF and lagged-correlation reference page | Computes an absolute temporal mean before subtracting it | 2 — `GPURasterSnapshotCovariance`, `GPURasterEOFProjection` |
| Same page | Scales and subtracts the first paired observation before reducing | 1 — `GPURasterLaggedCorrelation` |
| Shared edge-association implementation | Uses absolute edge-end means for a centered cross-product | 2 — `GPUSpatialPearson`, `GPUGammaStatistic` |
| Public producers whose defining file declares `centerCoordinates?:` | Offers data-relative accumulation as an opt-in | 1 — `GPUSpatialKMeans` |

The source searches used to select candidates gave these counts. They count files, including helpers, rather than defects.

```sh
rg --files modules/analysis/src -g '*.ts' | wc -l
# 1817
rg -l 'sumSquares|sumSquared|sumOfSquares|sumXX|sumYY|sumXY|twiceArea|shoelace' \
  modules/analysis/src -g '*.ts' | wc -l
# 31
rg -l 'timeWordsDifferenceToF32|localOrigin|centerCoordinates' \
  modules/analysis/src -g '*.ts' | wc -l
# 25
```

**First use from the docs.** The flow-map page says the release and slice dates use the velocity field’s time unit. This is the code I would write for two observations ten seconds apart. It compiles, and the last two timestamp slots become identical.

```ts
const t = 1_700_000_000;
const parameters = getGPULagrangianFlowMapParameterValues({
  fieldExtent: [0, 0, 100, 100],
  startTime: t, timeStep: 10,
  sliceStartTime: t, sliceEndTime: t + 10
});
```

The executed result was `[0, 0, 100, 100, 1700000000, 10, 1700000000, 1700000000]`. No graph or readback was necessary to establish the information loss. [Flow-map reference](/Users/robertchristie/Documents/GitHub/luma-contributors-r19/docs/api/gpu-raster/lagrangian-diagnostics-mob06.md:10)

## Proposed convention

For any observable defined by differences, compute in a shared local frame, take the difference before narrowing where the API has wider inputs, and publish exact identities and cardinalities as integers. Check this with paired fixtures that translate coordinates, dates, or values while preserving the mathematical answer. Assert the selected row, cell, face, support count, or residual in its own units, alongside any float output.

## Findings

Source paths below are relative to `modules/analysis/src`. `R` is `raster-analysis`, `S` is `gpu-spatial-analysis`, `N` is `gpu-network`, `P` is `pointcloud-analysis`, and `D` is `dataframe-analysis`. **KI** identifies the newer copy in `luma-contributors-ki`. Implementation lines otherwise refer to `luma-contributors-r19`. For numerical corrections, **ADDITIVE** means the documented operation and existing call signature are preserved.

| ID | Kind | Reach | Change | Cost | Where | One line |
|---|---|---|---|---|---|---|
| FLT-01 | W | ORDINARY | ADDITIVE | S | Polygonization diagnostics | A translated square loses its face while all four edges remain classified as closed. |
| FLT-02 | W | ORDINARY | ADDITIVE | M | Distance field, sampling, profile | The same grid edge can belong to different columns in two contributors. |
| FLT-03 | W | ORDINARY | ADDITIVE | S | Lagrangian flow-map time | Two real slice dates pack to one timestamp and suppress temporal interpolation. |
| FLT-04 | W | ORDINARY | BREAKING | S | Spatial k-means | The default can put a centroid at the minority point in a 254-to-1 cluster. |
| FLT-05 | W | ORDINARY | ADDITIVE | M | Spatial Pearson and Gamma | An unrepresentable intermediate mean erases an exactly representable association. |
| FLT-06 | W | ORDINARY | ADDITIVE | M | Snapshot covariance and EOF projection | The snapshot matrix can contain a false temporal mode before eigendecomposition. |
| FLT-07 | W | ORDINARY | ADDITIVE | M | Particle advection, flow map, streamlines | Repeated quarter-metre steps can produce zero displacement. |
| FLT-08 | W | ORDINARY | ADDITIVE | S | Brownian bridge utilization | Rounding a cell center before evaluating the Gaussian can inflate density by 172%. |
| FLT-09 | W | ORDINARY | ADDITIVE | M | Trajectory Kalman | A moving prediction stops during a gap in observations. |
| FLT-10 | W | ORDINARY | ADDITIVE | S | Line integral convolution | Moving the grid origin changes a scalar texture value by 160%. |
| FLT-11 | W | ORDINARY | ADDITIVE | M | Raster embedding covariance | A two-pass covariance still loses low-contrast variation in a million-pixel image. |
| FLT-12 | W | UNUSUAL | ADDITIVE | M | Network time routing and percentiles | Two positive-duration edges become a zero-duration journey with valid, converged status. |
| FLT-13 | W | UNUSUAL | ADDITIVE | M | Raster temporal resampling | Distinct Int64 acquisitions alias and the exact first observation is replaced by the second. |
| FLT-14 | W | UNUSUAL | ADDITIVE | M | Point-cloud neighborhoods | A rounded squared-distance tie changes both the support count and the retained point. |
| FLT-15 | E | UNUSUAL | ADDITIVE | S | Key join | Exact integer aggregate counts exist internally but the public count destination must be float32. |

### FLT-01: Polygonization can report closed edges and zero faces

**Where.** `analysis-core/ring-assembly/segment-polygonization-diagnostics.ts:195`, `:291`, `:354`. The diagnostic is publicly forwarded by `S/index.ts:8`.

**EXECUTED.** I called `getSegmentPolygonizationDiagnostics({segments})` with the four edges of a square. A side of `1/16` at `[0,0]` returned one face of area `0.00390625`. Adding `10_000_000` to every X and Y returned `closedRings: []`, `closedRingSegmentIndices: [0,1,2,3]`, four `'closed-ring'` classifications and no unused segments. Face-count error was **100%**. With side `1.0625`, the face survived but area changed from `1.12890625` to `1.125`, about **−0.346%**. All input numbers are exact binary64 values.

**Now → wanted.** `diagnostics.closedRings.length` can contradict the edge classifications. Subtract the first ring vertex from both endpoints before each shoelace product. The GPU ring assembler already does this in both its short- and long-ring paths, checked in KI at `analysis-core/ring-assembly/gpu-segment-ring-assembly.ts:1651` and `:1688`. This host finding occurs in binary64 itself, before any GPU upload.

**Callers and value.** No caller edits. Survey and overlay users recover small bounded faces. **S**.

### FLT-02: Raster packers disagree at the same physical edge

**Where.** `R/distance-field/distance-field-parameters.ts:87`, `R/distance-field/distance-field-kernels.ts:229`, `R/raster-sampling/raster-sampling-parameters.ts:94`, `R/raster-sampling/raster-sampling-utils.ts:188`. The shared sampler feeds `GPURasterSampling` at `gpu-raster-sampling.ts:229` and `GPURasterProfile` at `gpu-raster-profile.ts:643`.

**EXECUTED packers, SOURCE device result.** I passed the same 21-column grid, X origin `10_000_000`, cell width `0.85`, and `localOrigin: [10_000_000,0]` to all three public packers. For local X `17`, sampling and profile packed reciprocal `1.1764706373214722`, yielding `floor(20) = 20`. Distance field packed `1.1764705181121826`, yielding `floor(19.999998092651367) = 19`. The error is one column, **5% of the expected index**.

There is also a case with no decimal input ambiguity. For a 41 m cell all three packers emitted `0.024390242993831635`. The float32 product at exactly 41 m was `0.9999999403953552`. All three therefore model column **0 instead of 1**. On the same two-cell X grid, the executed public `computePointCloudVoxelMeans` put the point in voxel **1**.

**Now → wanted.** A user can seed one cell and sample its neighbor with the same metadata. Define ownership from one canonical packed step and correct the candidate integer against its bounding edges. Preserve the documented special handling of the outermost sampling edge. Merely standardizing which reciprocal gets packed leaves the 41 m case. This extends **KNOWN(PTS-109)** to three raster producers, including calls that already use a local origin.

**GPU probe.** Use two cells with values `[10,20]`, width 41, and a seed/query at local X `41`. Read the sampled value and the cell with zero seed distance. Expected `20` and column `1`. Then run the 0.85 m fixture to check agreement.

**Callers and value.** No caller edits. Grid overlays retain cell identity through seed, sample and profile operations. **M**.

### FLT-03: A flow-map packer can erase a whole velocity slice

**Where.** `R/lagrangian-diagnostics/gpu-lagrangian-flow-map.ts:109`, `:428`, `:459`. The packer comment at `:78` discusses release times far from the slices. The failure also occurs when release equals the first slice exactly.

**EXECUTED packer, SOURCE device result.** Use the first-use example above. With one supported field cell, velocity `[0,0]` at the first slice and `[2,0]` at the second, the midpoint velocity is `1` and a ten-second step travels `10`. Equal packed endpoints select inverse span `0`. The float32 transcription traveled `0`, **100% displacement error**.

**Wanted.** Add `timeOrigin?: number` to the packer settings. Subtract it from all three instants in binary64, leaving `timeStep` alone. Show `timeOrigin: sliceStartTime` in the first example. Reject distinct source slice times that become identical after packing, with an error naming both fields and `timeOrigin`. Preserve intentionally equal source times.

**GPU probe.** One seed at `[50,50]` in the example’s 100-unit cell. Read final X. The rebased run should give `60`, with validity `1`.

**Callers and value.** No edits for representable inputs. Previously collapsed calls can supply one additional field. Time-varying fields retain their second observation. **S**.

### FLT-04: The k-means precision switch is needed at 255 rows

**Where.** `S/spatial-clustering/gpu-spatial-kmeans.ts:219`, `:314`, `:948`, `:965`, `:1169`. The existing regression at `modules/analysis/test/gpu-spatial-analysis/spatial-clustering/gpu-clustering-extras.spec.ts:584` uses 60,000 points and explicitly enables centering.

**EXECUTED model, SOURCE device result.** Set `k: 1`, X to `10_000_001` for the first 254 rows and `10_000_002` for the last, and Y to zero. I transcribed the actual one-chunk reduction, including its 16 lane slices and serial combination. The default center was `10_000_002`. The mathematical center is `10_000_001 + 1/255`, which rounds to `10_000_001`. Modeled inertia was **254**, against **254/255**, a **25,400% overestimate**. For 255 identical rows, the same model gave inertia `255` against exactly zero.

**Wanted.** Default `centerCoordinates` to `true`. Keep explicit `false` for callers reproducing historical arithmetic. The existing data-relative path computes final distances before adding the origin back to the published center.

**GPU probe.** Use `initialization: 'first-valid'`, two iterations, and request centers, squared distances and inertia. Inspect the minority point’s distance as well as the summed objective.

**In-repository callers that change.** `rg 'new GPUSpatialKMeans\b'` found 47 text matches. Parsing the matching TypeScript and JavaScript found **42 actual constructor expressions** — 34 tests, five examples, one lens, one kepler site and one website site. Eighteen pass an object literal with no option or spread. Twenty-four forward a spread or another props value, so their effective setting depends on the caller. No signature edits are required. Pin `false` in the existing plain-versus-centered control and review default-dependent numerical and command-count fixtures.

**Value and cost.** GIS callers get meaningful distances and inertia from the default constructor. **S, BREAKING** because the default and resulting graph change.

### FLT-05: Two directed edges can lose perfect negative correlation

**Where.** `S/spatial-autocorrelation/gpu-additional-spatial-statistics.ts:638`, `:673`, `:689`. I checked the staging-owned KI copy at `:702`, `:737`, `:755`, `:827`. Both `GPUSpatialPearson` and the `'cross-product'` form of `GPUGammaStatistic` use this path.

**EXECUTED model, SOURCE device result.** Use two unit-weight edges, `0→1` and `1→0`. Both value columns are `[10_000_000,10_000_001]`. The edge-end pairs are therefore `(a,b)` and `(b,a)`. Their exact means are `10_000_000.5`. The modeled means round to `10_000_000`, giving cross-products `0` and `0`. Pearson is **0 instead of −1** and Gamma is **0 instead of −0.25**, both **100% magnitude error**.

**Wanted.** Accumulate offsets from one eligible value for each edge end and retain the means as offsets through the cross-product pass. The staged fix for **KNOWN(D063)** protects the final Pearson denominator against product overflow. Here all products are small and the numerator has already become zero.

**GPU probe.** Read both statistics on the two-edge graph, then subtract `10_000_000` from both input columns and repeat. Both runs should give `−1` and `−0.25`.

**Callers and value.** No caller edits. Edge association retains its sign under a change of baseline. **M**.

### FLT-06: Fixed-point snapshot covariance accumulates the wrong anomalies exactly

**Where.** `R/raster-time-series/gpu-raster-snapshot-covariance.ts:194`, `:202`, `:241`. `GPURasterEOFProjection` independently rebuilds the same absolute mean in `gpu-raster-eof-projection.ts:209`, `:215`, `:221`.

**EXECUTED model and public eigensolver, SOURCE device result.** For one cell containing 256 alternating pressure readings `100000` and `100000.125` Pa, the true mean is `100000.0625`. The modeled mean was `100000.0078125`. The trace of the anomaly Gram matrix was **1.765625 instead of 1**, **+76.5625%**. All input samples and the correct mean are representable.

A two-slice fixture shows what a user sees downstream. For `[10_000_000,10_000_001]`, I passed the model’s matrix `[0,0,0,1]` and the reference matrix `[.25,-.25,-.25,.25]` to the public `getGPURasterSnapshotEOFDecomposition`. It returned temporal mode `[0,1]` versus `[0.70710677,-0.70710677]` and singular value `1` versus `0.70710677`. This is upstream of **KNOWN(RT-120)**.

**Wanted.** Store the temporal mean of `value − firstValue` and construct anomalies as `(value − firstValue) − meanOffset` in both contributors. Keep the existing fixed-point cross-cell accumulator.

**GPU probe.** One cell, two slices, weight one. Decode `covarianceWords` through the public matrix decoder and check all four entries before invoking the eigensolver.

**Callers and value.** No caller edits. EOF users recover the temporal contrast that generated the data. **M**.

### FLT-07: A valid moving particle can travel zero metres

**Where.** `R/particle-advection/gpu-particle-advection.ts:352`, `:354`, `:380`; `R/lagrangian-diagnostics/gpu-lagrangian-flow-map.ts:467`, `:472`; `R/flow-texture/gpu-streamlines.ts:461`, `:491`, `:499`. The streamline twin retains double-precision integration state at `R/flow-texture/cpu-streamlines-operation.ts:213`.

**EXECUTED model and CPU twin, SOURCE device result.** In a constant field, start at X `10_000_000` and advance `0.25` m four times. Rounding after every state update gives total displacement `0`; integrating in local coordinates gives `1` m, **100% error**. The final displacement is representable even in world coordinates. For streamlines I also ran the public CPU twin with one 100 m wide cell, seed `8`, step `.25`, and eight steps each way. It emitted 17 points with translated X offsets spanning **28 through 32**. The shader recurrence at that origin repeats the seed X, for a modeled span of zero.

**Wanted.** Use a shared local frame for integration. For multi-frame `GPUParticleAdvection`, retain the local position in the persistent buffer across encodings. For planar flow-map and streamline packers, accept `localOrigin` and subtract it from every relevant extent, seed or spawn bound consistently. Document reconstruction on readback. The existing geographic flow-map warning covers a separate projected-geographic route. **KNOWN(RT-124)** covers velocity support at the landing.

**GPU probe.** Four particle frames with reset, drop and age expiry disabled; four flow-map steps in one dispatch; one streamline seed as above. Read displacement, alongside validity or age.

**Callers and value.** An optional local-frame interface preserves existing signatures. Callers choosing it convert positions once. Tracking and flow visualization retain cumulative movement. **M**.

### FLT-08: Brownian bridge density depends on the map origin

**Where.** `S/trajectory-home-range/gpu-brownian-bridge-utilization.ts:289`, `:317`, `:322`; CPU twin `cpu-brownian-bridge-utilization-operation.ts:93`.

**EXECUTED CPU twin and model, SOURCE device result.** Use two stationary fixes at `[O,.5]`, times `[0,1]`, location errors `[.5,.5]`, zero motion variance, one midpoint, and a single cell with extent `[O,0,O+1,1]`. The public CPU twin returned **0.46839866** at both `O=0` and `O=10_000_000`. The shader transcription rounded the translated cell center `O+.5` to `O`, giving **1.27323949**, **+171.83%**. Every input coordinate and extent endpoint is exactly representable. The point is within the kernel cutoff in both frames.

**Wanted.** Construct the center offset from the grid minimum. Subtract that minimum from each fix before interpolating the bridge mean or forming cutoff bounds. Evaluate the Gaussian using the resulting local difference. This fixture needs no new prop.

**GPU probe.** Run the two one-cell fixtures and compare `output.density[0]` with `0.46839866`.

**Callers and value.** No caller edits. Home-range density retains its physical meaning after reprojection to a translated planar frame. **S**.

### FLT-09: Kalman prediction loses velocity during a measurement gap

**Where.** `S/trajectory-estimation/gpu-trajectory-kalman.ts:317`, `:323`, `:329`; full covariance `trajectory-kalman-full-covariance.ts:136`. The smoother also consumes published states at `gpu-trajectory-kalman.ts:395` and `trajectory-kalman-full-covariance.ts:197`.

**EXECUTED CPU twin and model, SOURCE device result.** At integer seconds `0…8`, give measured X coordinates `[O,O+1]` followed by seven missing fixes. Use isotropic parameters `[q,r,v,maximumGap]=[0,1,1,0]`. At `O=10_000_000`, the public twin published X offsets `[0,1,1,1,2,2,2,3,3]`, with prediction status `2` on all seven missing rows. Its last position is exactly representable. The shader recurrence rounds the first updated offset to `1` and then repeatedly adds velocity `1/3` to the absolute position. The final modeled offset is **1 instead of 3**, **−66.7%**.

**Wanted.** Keep position state relative to the first valid fix of each initialized track segment. Preserve local filtered states for backward smoothing when requested. Add the origin only when publishing an absolute position. Velocity and covariance retain their existing units.

**GPU probe.** Use the nine observations above, reading final X, velocity and status. Repeat with diagonal full-covariance parameters equivalent to the isotropic case.

**Callers and value.** No caller edits. Gap prediction keeps working at projected northings. **M**.

### FLT-10: Translating the LIC grid changes the texture itself

**Where.** `R/flow-texture/gpu-line-integral-convolution.ts:315`, `:334`, `:338`, `:342`.

**EXECUTED model, SOURCE device result.** Use an 8×1 field and output grid with unit cells, constant eastward velocity, noise `[0,0,1,0,1,0,0,0]`, four steps each way, `stepLength=.25`, and ripple disabled. At column 3 the binary64 Hann-weighted answer is **0.30729490**. At X origin `10_000_000`, the center `O+3.5` rounds to `O+4`, then every quarter-step returns to that coordinate. All eight off-center reads select column 4. The modeled result is **0.79999995**, **+160.34%**. Every landing has field support.

**Wanted.** Trace in output-cell coordinates and sample the velocity field through a transform between the two local grids. Compute the difference between their origins in the host packer before narrowing. The output is scalar, so no world-coordinate reconstruction is needed.

**GPU probe.** Read `values[3]` at origins `0` and `10_000_000` on the stated grid.

**Callers and value.** No caller edits. A map translation preserves the rendered texture. **S** for the planar transform and fixture.

### FLT-11: Two-pass embedding covariance acquires variance from its mean error

**Where.** `R/raster-embeddings/gpu-raster-embedding-covariance.ts:64`, `:208`, `:231`, `:258`, `:288`. The class promises precision for small variation relative to a large mean.

**EXECUTED model, SOURCE device result.** Use one dimension and a 1024×1024 image alternating `f32(.3)` and `f32(.30001)`. With the default 256 ranges, I rounded each addition in the range sums, final sum, deviations, products and covariance reduction. Sample covariance was **2.63753967e−11** against binary64 **2.49189796e−11**, **+5.84%**. An even smaller fixture uses 256 pixels alternating `f32(.1)` and `f32(.1000001)`. The modeled result was **4.44994790e−14** against **2.35454358e−15**, **+1,790%**. Both references use the actual uploaded float32 values.

**Wanted.** Retain a per-component anchor and mean offset through the covariance pass. Continue publishing `output.mean` in the caller’s units. Compute covariance from the retained offsets, independently of rounding the public mean.

**GPU probe.** Use the 256-pixel fixture, default `rangeCount`, one band, all pixels usable. Read its one covariance entry and repeat after subtracting the first component value on the host.

**Callers and value.** No caller edits. PCA can distinguish low-contrast features without an artificial variance floor. **M**.

### FLT-12: Relative departure times can still erase every edge duration

**Where.** `N/network-optimization/gpu-network-time-routing.ts:78`, `:292`, `:330`; `N/network-time-geography/gpu-network-travel-time-percentiles.ts:284`, `:356`. The lower-level contract asks for nonnegative relative departure times.

**EXECUTED CPU twin and model, SOURCE device result.** Choose a valid relative clock of `16_777_216` seconds since the start of a service year. Route `0→1→2` with affine `[base,slope]=[1,0]` on both edges, one departure, percentile `[50]`. The public percentile twin returned times **`[0,1,2]`**, reached counts `[1,1,1]`, status `[1,1]`. The float32 recurrence gives `f32(16777216+1) = 16777216` on **both** edges. Modeled travel times are **`[0,0,0]`**, **100% error** at both destinations. The same effect occurred in the model for 30-second edges at Unix time `1700000000`; the relative-clock example already satisfies the lower-level contract.

**Wanted.** Relax elapsed durations relative to each departure and let the percentile stage consume those durations directly. For time-dependent costs, retain the clock origin needed to evaluate the model. Rebasing an affine duration requires transforming its intercept; piecewise schedules require rebasing both sample times and arrivals. Simply subtracting an epoch from `window` would change those models.

**GPU probe.** Read travel times, reached counts and status on the three-state fixture. Add an alternative direct edge of duration `1.5` to check route ordering after the arithmetic fix.

**Callers and value.** No caller edits for an internal elapsed-time path. Routing users retain short links on long timetable clocks. **M**.

### FLT-13: Distinct Int64 acquisitions can select the wrong exact observation

**Where.** Staging-owned files, checked in **KI**: `R/raster-time-series/raster-irregular-date-axis.ts:79`, `:120` and `gpu-raster-temporal-resample.ts:219`, `:225`. R19 also advances by comparing fractional years.

**EXECUTED KI public helper, SOURCE device result.** Let `t=1700000000000n` milliseconds. Supply acquisition times `[t,t+1000n,t+2n*31556952000n]` and values `[7,11,13]`. The KI helper returned exactly **`[-1,-1,1]`**. The mathematical first grid point is exactly `−1` year. The `<=` scan advances across both observations and its exact-match branch selects the second. Modeled result **11 instead of 7**, **+57.14%**, with computed status `0`.

**Wanted.** Offer regular target times as Int64 words when acquisition identity matters. Compare target and acquisition words first and convert the elapsed numerator and denominator only for interpolation. Document the collision limit of the fractional-year route. **KNOWN(RT-130)** concerns interpolation outside the bracket; **KNOWN(RT-131/D115)** concerns gap gating; **KNOWN(D116)** concerns helper endpoints. This fixture returns a different observed value even when the helper and shader agree and the exact-match branch takes precedence.

**GPU probe.** One cell, three acquisitions, one output point at `gridStart: -1`, positive `gridStep`, generous `maximumGap`. Read value and status together.

**Callers and value.** Existing callers can retain fractional grids. An optional word-time grid preserves exact acquisitions in mixed-cadence records. **M**.

### FLT-14: Radius membership and nearest-row identity are rounded together

**Where.** `P/operations/gpu-pointcloud-neighborhoods.ts:344`, `:452`, `:464`. Checked the staging-owned **KI** copy at `:445`, `:446`, `:457`. Its props specify an inclusive Euclidean radius and smallest-row-ID tie break.

**EXECUTED KI packer and model, SOURCE device result.** Query `[10_000_000,10_000_000,0]`. Give the lower-ID candidate displacement `[10000,1,0]` and the higher-ID candidate `[10000,0,0]`. The public packer emits radius `10000`, squared radius `100000000`. Exact squared distances are `100000001` and `100000000`; both float32 keys are `100000000`. With `capacity: 1`, the model counts **2 instead of 1**, selects the outside point and sets truncation. The squared-distance error is only about **0.000001%**; the count error is **100%**. Subtracting an origin leaves these displacement vectors unchanged.

**Wanted.** Keep the fast float32 comparison for separated keys. For an equality at the cutoff or retained-list boundary, resolve the comparison from the coordinate differences using a higher-precision squared norm before breaking ties by row ID. Keep `squaredDistances` as float32 output. If rounded-key membership is the intended contract, name it explicitly in the radius and `counts` documentation.

**GPU probe.** Three positions, one query row, no grid, capacity one. Expect support `1`, the higher candidate ID, status `0`. Run both serial and cooperative selection.

**Callers and value.** No signature changes. Radius users can trust support counts and the ID at the truncation boundary. **M**.

### FLT-15: Key-join count outputs have no exact integer destination

**Where.** `D/key-join/gpu-key-join.ts:100`, `:113`, `:288`, `:328`, `:1056`; `D/key-join/key-join-distinct.ts:122`. `output.matchCounts` already has a `uint32` path for the count of matching rows, but cannot represent a per-column finite or distinct count.

**EXECUTED.** I constructed a null-device graph with one left row and **16,777,217** declared right rows, without allocating or submitting the large dataset. A `count-distinct` aggregate with a float32 destination was accepted. With a uint32 destination it threw `key-join aggregates[0].output must be packed, uint32-aligned float32 GPU data`. The implemented publication cast gives `f32(16777217) = 16777216`, an error of **one row**, or **0.00000596%**. The TSDoc acknowledges rounding for `count-distinct`; the issue is that a caller needing the integer has no output for it. This extends **KNOWN(DPC-13)** beyond whether the operation exists.

**Wanted.** Make `GPUKeyJoinAggregate` a discriminated union allowing `GraphDataView<'uint32'>` as `output` for `'count'` and `'count-distinct'`. Select integer publication from the destination format. Preserve float32 destinations for existing callers.

**GPU probe.** Exercise the finish kernel with one preaccumulated integer word `0x01000001`; require that exact uint32 word back. A separate end-to-end large join can qualify capacity on a suitable device.

**Callers and value.** No existing caller edits. Dataframe users can reconcile finite and distinct counts exactly. **S**.

## Departures by name

| Convention departure in the audited paths | Public classes or functions |
|---|---|
| Absolute coordinates in area or spatial-state arithmetic | `getSegmentPolygonizationDiagnostics`, `GPUSpatialKMeans` by default, `GPUParticleAdvection`, planar `GPULagrangianFlowMap`, `GPUStreamlines`, `GPULineIntegralConvolution`, `GPUBrownianBridgeUtilization`, `GPUTrajectoryKalman` |
| Absolute time or rounded date used before an elapsed calculation or identity comparison | `getGPULagrangianFlowMapParameterValues`, `GPUNetworkTimeRouting`, `GPUNetworkTravelTimePercentiles`, `GPURasterTemporalResample` |
| Absolute mean reused for small residuals | `GPUSpatialPearson`, `GPUGammaStatistic` cross-product, `GPURasterSnapshotCovariance`, `GPURasterEOFProjection`, `GPURasterEmbeddingCovariance` |
| Rounded reciprocal or squared-distance key decides discrete membership | `GPUDistanceField`, `GPURasterSampling`, `GPURasterProfile`, `GPUPointCloudNeighborhoods` |
| Exact internal integer forced through a float output | `GPUKeyJoin` finite and distinct aggregate counts |

**Cleared for the specific failure mechanism inspected.** `GPUSegmentRingAssembly` uses a ring origin in both reduction paths, including KI. `GPUGeometryMeasures` forms planar edge terms in a feature-local frame. `GPUPointCloudFeatures` subtracts a neighborhood origin before its covariance pass. `GPURasterLaggedCorrelation` scales and shifts both paired series. `GPURasterFocalPairs` centers on a real pair and corrects residual sums. `GPUKernelDensity` uses a data-relative origin for its automatic bandwidth spread. `GPUClassBreaks` uses a local origin in both exact natural-breaks cost paths. `GPUTrajectoryStaypoints` and `GPUTrajectoryLocations` accumulate centroid offsets from an anchor, including the KI locations implementation. The planar `GPULinearReferencing` projection computes distance from a local foot offset. These clearances apply to the named calculations.

## Quick wins

- [ ] Anchor the host polygonization shoelace sum and assert agreement between face count and closed-edge classifications on the translated square.
- [ ] Add `timeOrigin` and a collapsed-slice check to flow-map packing. Put the ten-second epoch example next to the temporal interpolation formula.
- [ ] Compute Brownian bridge centers, bounds and Gaussian offsets relative to the grid minimum.
- [ ] Compute LIC trajectories in the output grid’s local frame. Pin the scalar texture value after translation.
- [ ] Permit integer key-join aggregate destinations. Test `0x01000001` directly in the publication path.

## Larger proposals

1. **Flip `GPUSpatialKMeans.centerCoordinates` before alpha.** This is the one recommended default change. The accurate implementation and a device fixture already exist. Preserve explicit `false`, update the old-route control, and add the 255-row inertia fixture.
2. **Adopt one raster edge-ownership calculation.** Resolve the packed step, sign and boundary correction together for sampling, profiles and seed scatter. Test an integer-width edge where both reciprocal routes fail, a decimal-width edge where the routes disagree, and a negative row step.
3. **Keep local state through the full calculation.** A corrected packer cannot recover a particle displacement discarded on every frame. A wider final sum cannot recover a residual formed around a rounded absolute mean. Give integration and reduction kernels separate local working state where their public outputs must remain absolute.
4. **Test discrete decisions separately from float tolerances.** Check face count, selected cell, selected row, radius support and exact observation identity. For model outputs, compare displacement, covariance or inertia in its own units. A relative tolerance applied to a `10_000_000` m world position can conceal the entire effect being measured.

## Not read

I read the polygonization diagnostic, distance-field parameter packer and voxel-means helper in full. I traced the relevant packing, selection, reduction and publication sections of the named kernels, plus the CPU twins used in the probes. I compared staging-owned implementation candidates with KI and checked the earlier synthesis, area findings, catalog/direction reports and decision list for overlap. The census covers public barrels; the numerical review samples arithmetic paths selected from source searches.

I did not audit every regression solver, spherical predicate, terrain reducer, count conversion, generated operation wrapper or upstream luma.gl kernel. I ran no browser, GPU, build or test suite and wrote no files. The listed GPU probes remain the verification step for the shader findings.