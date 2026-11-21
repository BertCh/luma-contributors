# The host/GPU border: preserve what each word means

## Summary

The most urgent defect is in `GPUSpaceTimeVariogram`. A valid positions view starting eight bytes into a buffer generates a shader that starts reading sixteen bytes in. I reproduced the incorrect address in the generated WGSL.

Five public packers overwrite the next record in a caller-owned parameter array. Seventeen others report success after silently dropping the last word of their own record. Both defects can be tested without a GPU.

Several packers preserve finite scalar values while changing the operation. A wrapped longitude query becomes a single meridian, distinct raster cells acquire zero width, and three different transfer functions disagree on the midpoint of an interval after its endpoints round together.

Validation also fails in the other direction. The phenology packer rejects its documented default for **50 of 99** ordinary percentage thresholds.

The most valuable library-wide change is to **encode and validate one complete record before copying its declared prefix to the caller’s target**. For predicates and grids, validation must include the comparisons and derived quantities the shader actually uses.

## Census

I enumerated `rg --files modules/analysis/src -g '*.ts'`, parsed the 1,817 files with the installed TypeScript syntax parser, selected implemented functions matching `^get\w*ParameterValues$`, and followed named and star exports from the 11 entries in `modules/analysis/package.json`. Overload declarations counted once. Of 215 source functions, **213 are public**. The two internal functions are `getRegionalizationSeedParameterValues` and `getTerrainSweepSourceParameterValues`.

| Declared return | Public packers | Public classes with an exactly matching `get<Class>ParameterValues` name |
|---|---:|---:|
| `Float32Array` | 181 | 170 |
| `Uint32Array` | 30 | 19 |
| `Int32Array` | 1 | 1 |
| `Uint32Array \| Float32Array` | 1 | 1 |
| **Total** | **213** | **191** |

The class column counts exact name matches. Shared packers such as the permutation packer and secondary packers such as the word-time packer explain why it is smaller. There are 89 source packers in spatial analysis, 49 in raster, 28 in terrain, 23 in dataframe, 12 in network, 11 in point cloud, two in core and one in geospatial. The two internal functions belong to spatial analysis and terrain.

**172 public packers accept a parameter named `target`.** I selected indexed writers without a local length check or recognized target validator and ran each with a target one word short. Seventeen returned the short array. Separately, `rg -n '(target|destination)\.fill\(0\)' modules/analysis/src` located five public packers whose whole-view clear I confirmed with a sentinel after the record.

**Layout and size controls.** `rg -l 'var<uniform>' modules/analysis/src -g '*.ts'` excluding `testing` found 13 source files with explicit uniform declarations. I compared writers and WGSL readers for bounds filtering, group statistics, raster time series, raster embeddings and graph chunk groups. For bounds filtering I captured the actual host uniform using a null device. View byte offsets `[16,264,12,4]` and three bounds rows produced **32 bytes**, words `[4,2,3,1,0,3,0,0]`. The two `vec4<u32>` readers use those component offsets and read length at `[1][1]`. In a separate constructor probe, `GPUPointBuffer` rejected `65,536 × 65,536` output rows as `4294967296 > 4294967295`, before allocation. PCK-01 and PCK-08 show where comparable protections fail.

**Known findings rechecked.** `KNOWN(DPC-108)`’s time-window and composite-indicator follow-ups **have landed in r19**. The float and word window packers rejected `Number.MIN_VALUE` as a positive fade. Composite score rejected both that epsilon and weight `1e60`. Inequality rejected that positive Palma share. Space-time K rejected X bounds `[1e7,1e7+0.25]`. PCK-06 and PCK-13 concern additional failures beyond those repairs. `KNOWN(CORE-102)` concerns neighbor membership in a normalized index. PCK-05 reproduces an independent loss in the **output raster placement bytes**, before an index exists.

All hexadecimal values below are 32-bit words in buffer order. **EXECUTED** means actual TypeScript host code bundled into Node, including null-device graph construction where stated. **SOURCE** identifies inferred GPU behavior and its confirming probe. Paths under `src/` and `test/` are relative to `modules/analysis/`.

## Proposed convention

A packer must validate the encoded record, including exact integer fields, enabled/disabled sentinels, endpoint order and the derived ranges used by its WGSL reader, then publish exactly the declared number of words in one successful update. Every generated binding offset must use the unit of its WGSL array element. A checker can exercise these rules with short and padded subarrays, adjacent floating-point values, `2**24+1`, `2**32`, and recorded shader source for views with nonzero binding prefixes.

## Findings

| ID | Kind | Reach | Change | Cost | Where | One line |
|---|---|---|---|---|---|---|
| PCK-01 | W, S | ORDINARY | ADDITIVE | S | Space-time variogram | A scalar-word offset indexes an array of two-word positions. |
| PCK-02 | W | ORDINARY | ADDITIVE | S | Five parameter packers | Packing one record zeroes the following record. |
| PCK-03 | W | UNUSUAL | ADDITIVE | S | Bounds filter | Float32 rounding changes a wrapped query into a single meridian. |
| PCK-04 | W | ORDINARY | BREAKING | S | Network subgraph filter | Nearest rounding changes membership at ordinary decimal brush boundaries. |
| PCK-05 | W | ORDINARY | ADDITIVE | S | Five interpolation and density packers | A valid projected raster reaches the GPU with zero cell width. |
| PCK-06 | W, D | UNUSUAL | ADDITIVE | S | Split timestamps | A row outside an exactly packed inclusive window can acquire the endpoint’s identical bytes. |
| PCK-07 | W | UNUSUAL | ADDITIVE | S | Four raster, terrain and dataframe packers | Ordered transfer endpoints round together and select different endpoint policies. |
| PCK-08 | W | UNUSUAL | ADDITIVE | S | Raster reclassify | A break table that fits the device loses its last active break. |
| PCK-09 | W | UNUSUAL | ADDITIVE | S | Space-time density | Three requested time slices can encode one evaluation time and a zero-duration correction window. |
| PCK-10 | W | EXTREME | ADDITIVE | S | Point buffer and density | Positive radii and bandwidths arrive as zero. |
| PCK-11 | W | EXTREME | ADDITIVE | S | Streamlines | The cell width passes validation and its packed reciprocal is infinity. The CPU twin drops the line. |
| PCK-12 | W | EXTREME | ADDITIVE | S | Network K function | A safe integer simulation count silently wraps to zero. |
| PCK-13 | W | EXTREME | ADDITIVE | S | Space-time K and local co-kriging | Finite input fields produce zero or infinite normalization quantities. |
| PCK-14 | S | ORDINARY | ADDITIVE | S | Raster phenology | Half the ordinary hundredths tested fail with the documented default second-season threshold. |
| PCK-15 | C, E | ORDINARY | ADDITIVE | S | Seventeen target writers | A one-word-short destination is returned successfully. |
| PCK-16 | C, E | ORDINARY | ADDITIVE | M | Six reusable target writers | A rejected update can destroy the last valid parameter record. |

### PCK-01: An eight-byte positions prefix advances the variogram by two points

**Where.** `src/dataframe-analysis/pair-statistics/empirical-variogram-nodes.ts:88` and `:114`, reached from `gpu-space-time-variogram.ts:171`. `src/analysis-core/kernels/wgsl-kernel-nodes.ts:429` emits the offset. The installed upstream `graph-data-view-utils.ts:92` returns an offset in 32-bit components. The legacy cross-variogram route takes an earlier return.

**Evidence — EXECUTED.** I constructed the public contributor with three `float32x2` rows in a view starting at byte 8, recorded its command nodes, and obtained:

```text
binding range {offset: 0, size: 32}
getViewElementOffset(positions) = 2
const positionsOffset: u32 = 2u;
positions: array<vec2f>
let point = positions[positionsOffset + index];
```

Row zero therefore addresses byte 16. The intended address is byte 8. The declaration also supplies `'vec2f'` where the local `WGSLKernelElementType` union at `wgsl-kernel-nodes.ts:242` permits scalar and atomic types only. This is a new defect in the complete-case pass added for `KNOWN(DPC-112)`.

**Now.** A caller can use `graph.createDataView(buffer, {format: 'float32x2', byteOffset: 8, length: 3})` and get a mask computed from another row’s position.

**Wanted.** Bind positions as `f32` and read `positions[positionsOffset + 2u*index]` and the following word. This also supports scalar-aligned packed views. **SOURCE GPU probe:** prefix one unused point, then use logical points `[(0,0),(100,0),(1,0)]`, values `[0,9,2]`, equal times, and bounds including only X from 0 to 1. Require one pair of semivariance 2. The first good row currently reads the excluded middle point when building its mask.

**In-repository callers that change.** None. **Value and cost.** GIS callers can slice a positions buffer without changing the observations in a variogram. S.

### PCK-02: Five packers erase the next parameter record

**Where.** `src/gpu-network/network-isochrones/network-isochrones-parameters.ts:107`, `src/gpu-spatial-analysis/spatial-interpolation/kriging-parameters.ts:127`, `prediction-covariance.ts:288` and `:526`, `gaussian-random-field-parameters.ts:62`.

**Evidence — EXECUTED.** For each packer I obtained its normal length, supplied `new Float32Array(length + 1).fill(77)`, packed valid settings, and inspected word `length`.

| Packer | Record words | Following word before → after |
|---|---:|---|
| `getGPUNetworkIsochronesParameterValues` | 12 | `429a0000 → 00000000` |
| `getGPUKrigingParameterValues` | 12 | `429a0000 → 00000000` |
| `getGPULocalKrigingParameterValues` | 8 for one univariate structure | `429a0000 → 00000000` |
| `getGPUSpaceTimeKrigingParameterValues` | 28 | `429a0000 → 00000000` |
| `getGPUGaussianRandomFieldParameterValues` | 16 | `429a0000 → 00000000` |

**Now.** `pack(settings, arena.subarray(firstWord))` can erase every subsequent record. An oversized target is documented as legal.

**Wanted.** Clear only `fill(0, 0, requiredWordLength)`, or build that exact-length record and `target.set(record)`. Test an interior subarray with distinct prefix and suffix sentinels.

**In-repository callers that change.** None. **Value and cost.** Callers can update one record in an upload arena safely. S.

### PCK-03: Narrow rounding reverses the meaning of a wrapped bounds query

**Where.** `src/gpu-spatial-analysis/bounds-filter/gpu-bounds-filter.ts:115` and `:319`, `bounds-filter-mask-node.ts:66`. The public rule is in `docs/api/gpu-spatial-analysis/operators-joins.md:896`.

**Evidence — EXECUTED.**

```ts
getGPUBoundsFilterParameterValues({
  minX: 10.0000001, maxX: 10, minY: -1, maxY: 1, wrapX: true
});
// [10, -1, 10, 1], words 41200000 bf800000 41200000 3f800000
```

**Now.** The caller requested `x >= 10.0000001 || x <= 10`. The shader recovers `wrapped` solely from `boxMin.x > boxMax.x`, which is false for the emitted record. This adds to `KNOWN(GEO-04)` even when the input and query already use the same longitude branch.

**Wanted.** Before writing, require a requested strict inversion to survive encoding. Name `minX`, `maxX` and their converted values in the error. Preserve deliberately equal unwrapped endpoints. **SOURCE GPU probe:** query the one-point feature box `[0,0,0,0]` with the settings above. Its intersection mask should be 1.

**In-repository callers that change.** None for rejection of the misencoded input. **Value and cost.** Longitude brushing avoids replacing nearly the whole world with a meridian. S.

### PCK-04: A decimal brush endpoint can add or remove a representable row

**Where.** `src/gpu-network/network-subgraph-filter/network-subgraph-filter-parameters.ts:87`, and the **KI** copy of `gpu-network-subgraph-filter.ts:429` and `:533`. Both copies use a half-open attribute range and a closed edge-time range. The example in `docs/api/gpu-network/operators.md:465` supplies JavaScript numbers directly.

**Evidence — EXECUTED.** Replacing the example’s integer limits with decimal readings gives:

```ts
getGPUNetworkSubgraphFilterParameterValues(
  {nodeColumnCount: 1, edgeColumnCount: 0, hasEdgeTimes: true},
  {nodeRanges: [[0, 0.7]], edgeTimeWindow: [0, 0.1]}
);
```

The upper attribute word is `3f333333` (`0.699999988079071`). A row with exactly that float32 value satisfies `< 0.7` and fails `< packedMax`. The time word is `3dcccccd` (`0.10000000149011612`). An edge with that time fails `<= 0.1` and passes `<= packedEnd`. Neither example loses precision in the **data**.

**Wanted.** Specify that JS limits select representable input rows according to the documented comparisons. For `min <= x < max`, encode both endpoints toward positive infinity. For `start <= t <= end`, encode start upward and end downward. Preserve the documented infinite endpoint. **SOURCE GPU probe:** one isolated node with the attribute above and a separate one-edge graph with the time above. Expect node mask 1 and edge mask 0 respectively.

**In-repository callers that change.** **BREAKING** for users relying on nearest-rounded endpoints. `rg 'getGPUNetworkSubgraphFilterParameterValues\('` found five executable TypeScript calls excluding the declaration: three in the network tests, one lens, one website run. No signature edits. Review those five expectations and add decimal fixtures.

**Value and cost.** Interactive brushes select the same rows as the displayed inequalities. S.

### PCK-05: Five output-grid packers lose the entire X span at a projected origin

**Where.** `src/gpu-spatial-analysis/spatial-interpolation/spatial-interpolation-parameters.ts:40` and `:129`, `kriging-parameters.ts:128`, `src/gpu-spatial-analysis/point-density/gpu-kernel-density.ts:174` and `:194`, `gpu-adaptive-bandwidth-kernel-density.ts:111`, `space-time-kernel-density-math.ts:97`. Readers include `gpu-inverse-distance-weighting.ts:723`, `gpu-kriging.ts:561` and `gpu-kernel-density.ts:1194`.

**Evidence — EXECUTED.** I passed all five public packers an area raster with `width: 2, height: 2, affine: [0.1,0,10000000,0,-1,2]`. IDW, kriging, kernel density, adaptive kernel density and space-time kernel density each wrote the same first four words:

```text
4b189680 40000000 4b189680 00000000  →  [10000000, 2, 10000000, 0]
```

The host extent is 0.2 wide. The shader derives `(endX-startX)/2 = 0` from those bytes. This happens with four cells and a normal bandwidth of 1.

**Now.** A projected 10-centimeter raster packs successfully. Density explicitly tests `gridEnd != gridStart` before producing a value.

**Wanted.** Check that both signed spans preserve direction and that each encoded cell step is finite and nonzero. Give the caller the failing axis and suggest subtracting a common tile origin from grid and points. **SOURCE GPU probe:** one point at the origin, positive unit bandwidth, the 2×2 grid above, then repeat in tile-local coordinates. The local run supplies the finite reference.

**In-repository callers that change.** None for rejecting collapsed placement. **Value and cost.** A projected GIS caller gets an actionable error before launching interpolation or density. S.

### PCK-06: Exactly representable window boundaries do not protect rounded timestamp rows

**Where.** `src/dataframe-analysis/time-window-filter/time-window-parameters.ts:68` and `:114`, `time-window-sorted-nodes.ts:113`. `docs/api/gpu-dataframe-analysis.md:211` recommends `splitTimestamps` for float64 input.

**Evidence — EXECUTED.**

```ts
const end = 1 + 2 ** -25;
const window = getGPUTimeWindowFilterParameterValues({start: end, end});
const row = splitTimestamps([end + Number.EPSILON]);
```

The accepted window endpoint is `1.0000000298023224`. The row is `1.0000000298023226`, strictly later. Both encode as **`3f800000 33000000`**. In the other direction, the window packer rejects `{start: 0.1, end: 0.7}` because the low lanes are inexact. The two public helpers have different loss policies for values subsequently compared for equality.

**Wanted.** Add `splitTimestamps(values, {requireExact: true})`, with an error identifying the first inexact row. On the time-window page, state that default inclusion is exact for the **encoded pairs** and show the strict option for membership against original doubles. Retain the Int64 route for integer timestamps. **SOURCE GPU probe:** the singleton window and row above, in sorted and unsorted modes. Both lanes compare equal although the original row is outside.

**In-repository callers that change.** None for an optional strict mode. **Value and cost.** Callers who need exact original-time membership can establish it before upload. S.

### PCK-07: Four transfer functions accept endpoints that collapse onto their midpoint

**Where.** `src/raster-analysis/raster-algebra/weighted-overlay-parameters.ts:109` and `gpu-weighted-overlay.ts:336`; `src/dataframe-analysis/column-classification/bivariate-classification-parameters.ts:65` and **KI** `gpu-bivariate-classification.ts:429`; `src/gpu-terrain/terrain-illumination/gpu-relief-shading.ts:223`, `:260` and `:946`; `src/raster-analysis/raster-time-series/gpu-raster-best-available-pixel.ts:77` and `:309`.

**Evidence — EXECUTED.** I supplied `[99.999999,100.000001]` as an overlay input range, a bivariate alpha domain and two relief elevation stops. All wrote `42c80000` (`100`) twice. Overlay also wrote `degenerate=1`. For best-available-pixel, cloud distances `[10,10.0000001]` both wrote `41200000`. Adding `Number.MIN_VALUE` to `10` in its validator leaves exactly `10`.

**Now.** At input `100`, the requested linear fraction is 0.5. **SOURCE:** overlay chooses its upper step, bivariate classification chooses full opacity, and relief shading chooses the first stop color. For cloud distance `10`, the documented minimum scores zero, but the packed maximum comparison is already true.

**Wanted.** Preserve each operator’s existing rule for **deliberately** equal endpoints. Reject an originally ordered active ramp whose encoded order disappears. Require the cloud maximum to exceed its minimum in both representations. **GPU probe:** one cell of value 100, black-to-white stops or zero-to-one opacity, and a cloud-only score at distance 10. Expected fractions are `0.5, 0.5, 0.5, 0`.

**In-repository callers that change.** None for rejection of misencoded ranges. **Value and cost.** Classification, imagery scoring and cartography retain the caller’s transfer function. S.

### PCK-08: The last active raster break disappears from a supported table

**Where.** `src/raster-analysis/raster-algebra/reclassify-parameters.ts:41` and `:48`; `gpu-raster-reclassify.ts:111`, `:266`.

**Evidence — EXECUTED.** `getGPURasterReclassifyParameterValues({breakCount: 16_777_217})` returned `[16_777_216,0,0,0]`, first word `4b800000`. I also constructed the public class with that many breaks and a single output cell. The null graph created one command node and retained `maximumBreakCount === 16_777_217`. The break binding needs **67,108,868 bytes** and the device binding limit was **134,217,728**.

**Now.** Shape validation and parameter packing give different answers about the same table. **SOURCE GPU probe:** use ascending break values `0..16_777_216` and one input equal to the last break, with left closure. The expected class is `16_777_217`. The shader receives an active count of `16_777_216`.

**Wanted.** Immediately require `Math.fround(breakCount) === breakCount` within the supported uint32 domain. For full support, add an exact uint32 active-count route. A test of the existing host failure needs only the four-word packer.

**In-repository callers that change.** None for the guard or an optional exact count. **Value and cost.** Large tables retain their final class boundary. S for the guard.

### PCK-09: A three-slice density grid can evaluate the same instant three times

**Where.** `src/gpu-spatial-analysis/point-density/space-time-kernel-density-math.ts:76` and `:97`, `gpu-space-time-kernel-density.ts:611` and `:688`. `GPUSpaceTimeRelativeRisk` uses the same parameter view. The documented slice formula is in `docs/api/gpu-spatial-analysis/space-time-density-and-k-function.md:63`.

**Evidence — EXECUTED.** For `firstSliceTime: 0.7, sliceInterval: 1e-9, sliceCount: 3`, both controls remain positive and finite where required. The temporal words were `3f800000 3f333333 3089705f 3f333333 3f333333`. Replaying the shader’s slice expression gave `[0.699999988079071, 0.699999988079071, 0.699999988079071]`. The default correction window also has identical endpoints. These are relative times already.

**Wanted.** Derive the default window from the encoded controls using the shader’s arithmetic. Refuse a multi-slice grid whose adjacent slice times cease to advance, with the failing step and guidance to move the time origin. Preserve the documented one-slice case. **SOURCE GPU probe:** one cell, one event, three slices and uniform temporal edge correction. Compare against the equivalent grid recentered to zero.

**In-repository callers that change.** None for refusal of a collapsed grid. **Value and cost.** A time cube cannot silently repeat a slice or acquire a zero-duration window through packing. S.

### PCK-10: Positive point-buffer radius and density bandwidth become zero

**Where.** `src/geospatial-analysis/point-buffer/gpu-point-buffer.ts:63` and `:229`; `src/gpu-spatial-analysis/point-density/gpu-kernel-density.ts:171`, `:195` and `:1202`. The adaptive and space-time writers delegate at `gpu-adaptive-bandwidth-kernel-density.ts:111` and `space-time-kernel-density-math.ts:97`.

**Evidence — EXECUTED.** `getGPUPointBufferParameterValues(Number.MIN_VALUE)` returned `00000000 00000000`. The fixed, adaptive and space-time density packers each wrote `00000000` into bandwidth slot 4 for that same positive input. Point buffer checks encoded finiteness and then checks positivity on the original double.

**Wanted.** Require positive finite `Math.fround(radius)` or `Math.fround(bandwidth)` before publication. **SOURCE GPU probe:** a three-vertex planar buffer around one point reads zero for every radial offset. A one-cell fixed KDE reads a zero bandwidth at its validity gate. Both settings should fail on the host.

**In-repository callers that change.** None. **Value and cost.** Four public entry points consistently reject a range error at its source. S.

### PCK-11: Streamline cell size is checked before its reciprocal overflows

**Where.** `src/raster-analysis/flow-texture/streamlines-parameters.ts:72` and `:82`, `gpu-streamlines.ts:377`, `cpu-streamlines-operation.ts:120`.

**Evidence — EXECUTED, including the public CPU twin.** A grid cell width of `1e-40` encoded as `000116c2`, positive and finite. Its reciprocal in slot 8 encoded as **`7f800000`**. I ran `computeStreamlinesOnCPU` with one cell, one seed, vertical unit velocity, one step each way, `minimumPoints: 1`, and matching field and occupancy widths.

| X cell width | Line count | Position count | Overflow |
|---:|---:|---:|---:|
| `1` | 1 | 3 | 0 |
| `1e-40` | 0 | 0 | 0 |

Only the X scale changed. No horizontal motion was requested.

**Wanted.** Check the actual reciprocal words for finiteness and positivity, or use a scaled quotient in both implementations. **SOURCE GPU probe:** repeat the same one-cell CPU fixture and read the three counts.

**In-repository callers that change.** None for rejecting an infinite reciprocal. **Value and cost.** An invalid auxiliary word can no longer look like a successful empty streamline result. S.

### PCK-12: The network simulation count wraps while the seed remains exact

**Where.** `src/gpu-network/network-k-function/network-k-function-parameters.ts:41` and `:46`, `gpu-network-k-function.ts:716`, `:1080` and `:1111`.

**Evidence — EXECUTED.** `{seed: 2**32+1, activeSimulations: 2**32}` encoded as **`[1,1,0,0]`**. The seed is correctly split. The count passes `Number.isSafeInteger` and wraps to zero in a single uint32 slot.

**Wanted.** Require `activeSimulations` to be a uint32 before assignment. If an instance supplies its compiled simulation limit, also check against that limit. **SOURCE GPU probe:** compile with one simulation, provide the record above, and inspect its simulation count or curve. The three consumers all evaluate `min(0u, SIMULATION_COUNT)`.

**In-repository callers that change.** None. **Value and cost.** A configuration error becomes distinguishable from a deliberate request to disable simulations. S.

### PCK-13: Normalization checks miss arithmetic performed on the encoded fields

**Where.** `src/dataframe-analysis/pair-statistics/space-time-k-function-nodes.ts:171`, `:580` and `:686`; `src/gpu-spatial-analysis/spatial-interpolation/prediction-covariance.ts:312`, `:324`, `:386` and `:397`.

**Evidence — EXECUTED.**

- Space-time K accepted `bounds: [0,0,1e-25,1e-25]`. Both upper endpoints encoded as `15f79688`, distinct from zero. The product of the encoded widths rounds to **zero**. With unit spatial bounds and `timeWindow: [-3e38,3e38]`, the two finite time words were `ff61b1e6 7f61b1e6`, and the encoded subtraction rounds to **infinity**. These inputs pass the repaired `KNOWN(DPC-108)` endpoint tests.
- Local co-kriging with unit spherical covariance and `coregionalization: [1e20,0,0,1]` returned `7f800000` in both normalization slot 0 and coefficient slot 8. All four supplied loadings are finite. **SOURCE:** the reader’s `scale > 0` accepts positive infinity.

**Wanted.** Validate the actual float32 differences, area and required volume in space-time K. Validate each computed `L Lᵀ` coefficient and its normalization scale before publishing local kriging parameters. Preserve intentionally zero-loading covariance. **GPU probes:** two included K events with the tiny frame, and one co-kriging query with the loading matrix above. Inspect the K value and solve status alongside the parameter record.

**In-repository callers that change.** None for rejection of unusable normalizers. **Value and cost.** Statistical callers get the failed physical quantity or coefficient in the error. S.

### PCK-14: The documented phenology default fails for half the hundredths

**Where.** `docs/api/gpu-raster/raster-time-series.md:334`, `src/raster-analysis/raster-time-series/gpu-raster-phenology-metrics.ts:108`, `src/analysis-core/kernels/gpu-contributor-utils.ts:605`.

**Evidence — EXECUTED.** From the settings table alone, I would write:

```ts
getGPURasterPhenologyMetricsParameterValues({
  startFraction: 0.7, endFraction: 0.2, minimumAmplitude: 0
});
```

It throws `secondSeasonMinimumFraction must be a finite number in [0.7, 1]`. The default is exactly the requested larger fraction, `0.7`. The validator rounds the value to `0.699999988079071` and compares it to the unrounded lower bound. A sweep with equal start and end fractions `i/100` for `i=1..99` gave **49 accepted, 50 rejected**. No bytes were published for the rejected calls.

**Wanted.** Check the requested relation in double precision, then compare `packedSecond >= max(packedStart, packedEnd)`. Derive the omitted fourth word from those same packed fractions. Its expected word for the example is `3f333333`.

**In-repository callers that change.** None. **Value and cost.** Ordinary phenology settings work without a caller having to guess which decimal rounds upward. S.

### PCK-15: Seventeen public writers return truncated records successfully

**Where.** The complete named list and function-entry lines are below. `KNOWN(CORE-101)` addressed short targets in local-frame projection. These are separate parameter writers.

**Evidence — EXECUTED.** For each named function I obtained the default record length `n`, called it with valid settings and `new Float32Array(n-1).fill(77)`, and inspected the return. All seventeen returned the truncated target. For example:

```ts
getGPUPointBufferParameterValues(1, new Float32Array(1)); // [1], needs [1, 0]
getGPURasterHarmonicRegressionParameterValues({period: 1}, new Float32Array(0)); // []
```

The temporal-resample packer returned `[0,1]` instead of `[0,1,maximumGap]` in **both r19 and KI**. Passing such an array to a contributor may produce a later shape error. Uploading it into an existing full-length buffer can instead leave an old final setting in place.

**Wanted.** Check the destination length before the first indexed write. Report the packer ID, required words and supplied words. Pin short, exact and padded targets for every named writer.

**In-repository callers that change.** None for well-sized targets. **Value and cost.** An incomplete update fails where the parameter array was produced. S.

### PCK-16: Six rejected updates mutate the previously valid record

**Where.** `src/raster-analysis/raster-algebra/weighted-overlay-parameters.ts:80`; `src/gpu-spatial-analysis/spatial-interpolation/kriging-parameters.ts:127`, `prediction-covariance.ts:288` and `:526`, `gaussian-random-field-parameters.ts:62`; `src/raster-analysis/raster-embeddings/gpu-raster-embedding-projection.ts:65`.

**Evidence — EXECUTED.** I filled each destination with `77`, induced a validation error, caught it, and counted changed words:

| Writer | Rejected setting | Words changed |
|---|---|---:|
| Weighted overlay | Second layer has `weight: NaN` | 24 of 24 |
| Kriging | Sheared grid | 12 of 12 |
| Local kriging | Negative covariance range | 8 of 8 |
| Space-time kriging | Negative metric anisotropy | 28 of 28 |
| Gaussian random field | Point origin X `1e100` | 16 of 16 |
| Raster embedding projection | Third stretch `[1,0]` | 4 of 6 |

For overlay, the first layer’s weight became `40000000` (`2`) and the second layer’s word became zero. The caller received an exception and a mixed record.

**Wanted.** Use the fixed composite-score pattern: construct an exact-length candidate, validate it, then `target.set(candidate)`. For a hot path where allocation matters, finish validation before a bounded write pass. Test byte equality of the whole destination after failure.

**In-repository callers that change.** None. **Value and cost.** Interactive callers can retain and retry their last valid configuration after a rejected edit. M for the six writers and regressions.

## Departures by name

PCK-01 through PCK-14 and PCK-16 name the affected writers, consuming classes and shared readers at the finding. For **PCK-15**, each name below is the public function `getGPU<Name>ParameterValues`. Each line number is the function entry in its corresponding `gpu-*.ts` file.

| Directory under `src/` | Names and lines |
|---|---|
| `geospatial-analysis/point-buffer` | `PointBuffer:58` |
| `gpu-spatial-analysis/multiple-testing` | `RasterFalseDiscoveryRateMask:57` |
| `raster-analysis/raster-time-series` | `RasterBFASTMonitor:63`, `RasterClimateRunAccumulator:44`, `RasterGeometricMedianComposite:50`, `RasterHANTS:58`, `RasterHarmonicRegression:62`, `RasterLaggedCorrelation:41`, `RasterSnapshotCovariance:60`, `RasterStandardizedIndex:41`, `RasterStandardizedIndexCalibration:47`, `RasterStreamingAccumulator:62`, `RasterTemporalResample:51`, `RasterWhittakerSmoother:57` |
| `raster-analysis/raster-embeddings` | `RasterEmbeddingChange:50`, `RasterEmbeddingProjection:53`, `RasterEmbeddingSimilarity:52` |

**Packers cleared for specific boundary cases I executed:**

- `getGPUPointCloudGridParameterValues` and `getGPUGridGeneratorParameterValues` rejected a positive cell size that rounds to zero. `getGPUAdamStepParameterValues` rejected a beta that rounds to 1.
- `getGPUStreamlinesUint32ParameterValues`, `getGPUParticleAdvectionUint32ParameterValues` and `getGPULineIntegralConvolutionUint32ParameterValues` rejected seed `2**32` and preserved `0xffffffff`. The first two also preserved count/frame `16_777_217` exactly.
- `getGPUTimeWindowFilterWordParameterValues` encoded `[-1n,4294967296n]` into endpoint words `[ffffffff,ffffffff,00000000,00000001]`. Both time-window packers rejected a positive fade rounding to zero.
- `getGPUCompositeScoreParameterValues`, `getGPUInequalityParameterValues` and `getGPUSpaceTimeKFunctionParameterValues` passed the scoped DPC-108 regressions described in the census.
- `getGPUGaussianRandomFieldParameterValues` preserved raw Philox words `ffffffff` and `12345678` in its mixed float buffer. `getGPUBoundsFilterParameterValues` preserved deliberate signed-zero endpoint bits `[80000000,80000000,00000000,00000000]`. Their separate departures are above.
- `getGPUPermutationParameterValues` still packs `1-2**-30` as significance level 1 in r19. I ran the **staging-owned KI copy**, which correctly rejected it with `significanceLevel must be in (0, 1) after float32 rounding`. I have excluded it from new findings.

## Quick wins

- [ ] Fix the sole `type: 'vec2f'` positions binding in the empirical variogram. Add the nonzero-prefix graph-record assertion and the three-observation device oracle.
- [ ] Bound the five whole-view clears and add required-length checks to the seventeen demonstrated short writers. Reuse the same short/exact/padded fixture.
- [ ] Fix the phenology default and retain the 99-case hundredths sweep. It requires no device.
- [ ] Check encoded order for wrapped bounds, active transfer ramps and raster corners. Include the input and converted endpoints in the error.
- [ ] Check actual reciprocals, covariance coefficients, normalizers and positive distances before copying the candidate record.
- [ ] Reject inexact float counts and out-of-range uint32 counts. Use both `16_777_217` and `4_294_967_296` in host fixtures.

## Larger proposals

1. **Make comparison policy part of a parameter field’s contract.** A finite continuous coefficient can round to nearest. A count needs an exact integer. A half-open brush needs directed endpoint rounding. A stored flag may depend on two words retaining their order. Start with the five network-filter call sites in PCK-04, then use the same tests for new predicate packers. This is the one recommended change here that deliberately alters membership for previously accepted decimal inputs.
2. **Keep compact records and add exact routes where valid inputs exceed them.** An optional uint32 active-count view would support every device-addressable reclassification table. An optional local-origin placement would preserve valid projected rasters. Adding these alongside today’s formats costs more than host guards, but lets a caller complete the computation after receiving the error.
3. **Make parameter fixtures exercise publication as well as numbers.** Run each packer against a nonzero-offset destination with distinct leading and trailing sentinels, then repeat with an invalid final field. For generated WGSL, record the actual binding offset and array type. This would catch PCK-01, PCK-02, PCK-15 and PCK-16 without a browser.

## Not read

The census covered the full source tree and package export barrels. I read the complete parameter schema, time-window float packer, network subgraph and isochrone packers, weighted-overlay packer, streamline packers, and bounds-mask writer/reader. Elsewhere I read the complete candidate functions and the relevant constructor, WGSL or CPU-twin paths. I selected these by typed-array writes, computed reciprocals/products, scalar-to-vector addressing, open and closed comparisons, optional destinations and post-conversion range checks.

I did not inspect every constructor body, every shader generated indirectly by luma.gl, every third-party upload path or every input combination of the 213 public packers. Seventeen is the demonstrated short-target count, not an upper bound. I checked staging ownership for the findings and used the newer copies where indicated. No GPU, browser, compiler gate or test suite ran. The generated WGSL, packed bytes, CPU results and null-device size checks above are the verification available from this review.