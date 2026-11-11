# Fix review: closing pass, fields scope (quantile rank, Diggle test, COLD, Louvain, flow packers, sky view, published histogram)

Date: 2026-11-11. Independent review of the closing-pass changes named below. I wrote none of the
code. There is no commit to diff against, so I read the current tree. Paths are relative to
`modules/analysis` unless they start with `dev-docs/`, `docs/` or `scripts/`.

Method. The authors' accounts were read as claims: the "Closing pass (2026-11-11)" subsections of
`dataframe.md`, `raster-imagery.md`, `raster-fields.md`, `terrain-surface.md`, section 9 of
`shader-compiler-findings.md` and the closing section of `behaviour-changes.md`. Then:

- the changed kernels, packers, specs and generators were read in full;
- the splancs 2.01-45 source (`stmctest.S`, `stkhat.S`, `stkhat.f`, `cncvwt.f`, `iplace.f`,
  `weight.f`, `stsecal.f`, `stdiagn.S`; tarball SHA-256 `8bccf1d6...4b76d3`, as the author
  states), pyxccd 1.1.0 `cold.c`, `misc.c`, `defines.h`, rvt-py 2.2.1 `rvt/vis.py` and the
  published `@luma.gl/graph` 10.0.0-alpha.12 `gpu-core/gpu-histogram.js` were read against them;
- seven probe specs were run (six on the test GPU in headless Chromium, one in node) from a
  scratch folder that is now deleted, and Python probes were run in the pinned environments (`venv314`: NumPy 2.5.4;
  `venv312`: pyxccd 1.1.0). The Python scripts are kept in
  `/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-luma-contributors/ba3fe269-f86e-4664-86ec-632935ba200a/scratchpad/review-fields/`.

"CONFIRMED" means run on a concrete input (GPU where the row says so) or derived by hand on one.

Summary: no blocker. Two major findings, both about how far a reference agreement reaches (the
Diggle statistic on date-valued times; the pyxccd agreement for small breaks). Five minor ones.
The quantile rank arithmetic, the splancs port, the COLD stability rule, the Louvain literal, the
flow-packer callers, the sky-view convention test and the histogram counts all checked out.

## 1. Findings

### frc-fields-01 (major, CONFIRMED on the GPU): on integer-valued times `GPUSpaceTimeDiggleTest` does not return the splancs statistic

- Where: `src/dataframe-analysis/pair-statistics/space-time-k-function-wgsl.ts:63`
  (`getSpaceTimeTemporalWeight`: weight 1 when `lag <= radius`), and
  `gpu-space-time-diggle-test.ts:566` and `:574` (a pair at exactly `maximumDistance` or
  `maximumTimeLag` is counted), with the same rules in the observed pass
  (`space-time-k-function-nodes.ts:390`). Class documentation `gpu-space-time-diggle-test.ts:221-227`.
- What is wrong. splancs `stkhat.f` sets `vij = 2` when `(ti - tlow) .le. tj` or
  `(thigh - ti) .le. tj`, and drops a pair when `tj .ge. tmax` (and `sj .ge. smax`). The author
  found both differences and documented them as "exact ties only, a set of measure zero for
  continuous data that integer-valued times can hit". For the data this test is used on (event
  dates) ties are the normal case, and the natural `maximumTimeLag` is a whole number of days. The
  class says `U` equals the port "to float32 accuracy"; the only committed tie case is two events
  with `D = 0` under both rules, so nothing pins the size.
- Concrete input (GPU run and the port, same scene): 60 events, positions uniform in
  `[2, 8] x [2, 6]` (NumPy `default_rng(3)`, rounded to float32), times integer days 1 to 29,
  window `[0, 10] x [0, 8]`, time limits `[0, 30]`, `maximumDistance` 1.5 in 3 classes (every
  spatial weight is 1).
  - `maximumTimeLag` 8.5 in 5 classes (no lag on a class break; 81 of 3,540 ordered pairs have a
    lag equal to the distance to a time limit): kernel `U` = 43.9529, splancs port `U` = 52.2608
    (16% lower here). `K_t(8.5)` 17.585 against 18.271.
  - `maximumTimeLag` 7 in 7 classes (214 ordered pairs at lag exactly 7, 74 edge ties): kernel
    `U` = 21.1089, port 24.6711. `K_t(7)` 15.432 against 13.729. Each rule alone moves `U` to
    16.05 (closed window) and 30.58 (last class inclusive); they partly cancel.
  - By hand: three events at `t` = 1, 2, 3 in `[0, 10]`, all within one distance and one lag
    class (lag limit 4). splancs weights sum to 8, `K_t` = 10 / 6 * 8 = 13.33; here 7 and 11.67.
- What is still right. Every labelling gets the same function, so the Monte Carlo p-value is a
  valid test under either rule. Lags on interior class breaks are binned as splancs bins them
  (checked on the GPU, see section 2). With continuous times the fixture agreement stands.
- Suggested fix. Give this class the splancs rules: weight 2 when `radius <= lag`, and drop pairs
  with `lag >= maximumTimeLag` or `distance >= maximumDistance`. A compile-time flag or a second
  function beside `getSpaceTimeTemporalWeight` keeps `GPUSpaceTimeKFunction` (reference stpp) as
  it is. Add an integer-time scene to `space-time-diggle-splancs-port.py` and its spec. If the
  conventions are kept instead, replace "to float32 accuracy" and "measure zero" in the class
  documentation and in `docs/api/gpu-spatial-analysis/space-time-density-and-k-function.md` with
  the measured size for date-valued times, and add the row to `behaviour-changes.md`.

### frc-fields-02 (major, documentation; CONFIRMED on the GPU and with pyxccd): the pyxccd agreement on break dates does not hold for small breaks

- Where: `docs/api/gpu-raster/raster-change-monitoring.md:69` ("Supplying pyxccd's value
  reproduces its break dates") and `:89`; class documentation
  `src/raster-analysis/raster-time-series/gpu-raster-cold-monitor.ts:154-157` ("the dates of
  abrupt breaks and the segment bounds are equal, except that a segment can start a few
  observations later").
- What is wrong. The statement rests on eight series. One of them, `small-break`, has a
  disturbance a quarter the size of the others and passes. Drawing more series from the same
  family (150 dates, 8 to 24 day revisit, noise 40, pyxccd's bands, thresholds, variogram floor
  and year; `cold_detect` at `lam=0` against the monitor's binary64 twin) gives:

  | Disturbance | Series | Break lists differ | Only segment starts differ |
  | --- | --- | --- | --- |
  | full size, 8 to 24 day revisit | 45 | 0 | 7 |
  | half size | 35 | 0 | 11 |
  | quarter size (the `small-break` size) | 40 | 18 | 10 |
  | quarter size, alternating dense and sparse revisit | 40 | 17 | 0 |

  Seven of these series were then run through `GPURasterCOLDMonitor` on the test GPU: the kernel
  returned the twin's segments in all seven. Examples (true break, pyxccd, kernel):
  break at 62: pyxccd none, kernel 67. Break at 74: pyxccd none, kernel 74. Break at 82: pyxccd
  84, kernel 82. Break at 71 (sparse revisit): pyxccd 71, kernel none.
- Cause. With the twin's `every-observation-refit` toggle the agreement rises from 22 to 32 of 40
  (and 23 to 33), so the refit schedule accounts for most of it. That difference is in the table,
  but its stated effect is on magnitudes and on "a drifting series", not on whether a small abrupt
  break is found.
- Also: the class documentation gives the tolerances of the spec as the agreement (coefficients
  to 0.2, RMSE to 0.1, magnitudes to 25) where `docs/api` gives the measured 0.07, 0.02 and 20.
- Suggested fix. Scope the two sentences to breaks several times the noise, state the measured
  rate for breaks of about 2.5 times the noise in the refit row, and keep one such disagreeing
  series in the pyxccd fixture as a pinned difference so the spec does not suggest parity there.

### frc-fields-03 (minor, documentation; CONFIRMED): the list of differences from pyxccd omits how the initialization window advances

- Where: `docs/api/gpu-raster/raster-change-monitoring.md:39-42` ("otherwise the window shifts.
  This is pyxccd's test") and the table at `:63-75`; kernel
  `gpu-raster-cold-monitor.ts:438` (`var end = start + minimumObservations`, then extended to
  the span).
- What is wrong. The stability test is pyxccd's (see section 2). The window is not. pyxccd moves
  both ends by one observation (`i_start++; i++`, `cold.c` 1.1.0 lines 2973 and 3280), so the
  window keeps its observation count and only grows; the kernel rebuilds the shortest window from
  the new start. With irregular dates the two test different windows.
- Concrete input. The twin with pyxccd's advance rule differs from the kernel's rule in 4 of 240
  random series. Sparse-revisit series, break at 68: first segment starts at slice 7 (kernel, GPU
  run), 2 (pyxccd's advance rule), 0 (pyxccd, after its lookback). Another, break at 63: the
  segment after the break starts at 66 (kernel, GPU run), 65 (advance rule), 63 (pyxccd).
- Not sized: pyxccd also requires `conse` further observations after the window
  (`while (i <= end - adj_conse)`, line 2911); the kernel initializes a window that reaches the
  end of the series.
- Suggested fix: one more row in the table. Implementing pyxccd's advance is a small change to
  the loop, but it is a behaviour change and needs its own decision.

### frc-fields-04 (minor, documentation; CONFIRMED): "just below an integer" describes 6 of the 22 NumPy differences

- Where: `src/dataframe-analysis/column-classification/gpu-column-quantiles.ts:127-130`,
  `docs/api/gpu-dataframe-analysis.md:290-291`, `test/oracles/dataframe-quantile-rational-rank.py:23-26`.
- What is wrong. In 16 of the 22 cases NumPy's float64 product lands just above the integer, and
  it is `higher` (and `midpoint`) that is one rank off. All 22 are at 101 rows.
- Concrete input, by hand and in NumPy 2.5.4: 101 rows, `q = 0.55`: `100 * 0.55` is
  `55.00000000000001`, so `numpy.quantile(x, 0.55, method='higher')` returns `x[56]` (560 on
  `x = 0, 10, ..., 1000`); the exact rank is 55. The documented case is the other direction:
  `q = 0.29` is `0.28999999999999998`, `100 * q` rounds to `28.999999999999996`, `lower` returns
  `x[28]` (280), exact rank 29 (290).
- Suggested fix: "lands just below or just above an integer", with one example of each.

### frc-fields-05 (minor; CONFIRMED in node): the streamlines packer still packs a `gridExtent` cell size that rounds to zero or is infinite

- Where: `src/raster-analysis/flow-texture/streamlines-parameters.ts:67-70`.
- What is wrong. The closing fix added `validateNonZeroFiniteNumber` for `fieldExtent` only. The
  older check of `gridExtent` tests the double (`!(cellWidth > 0)`), and its message does not
  follow the contract form (no contributor ID, prop or limit).
- Concrete input: `gridExtent: [0, 0, 1e-50, 1e-50]` packs cell size 0 and reciprocal `Infinity`;
  `[0, 0, Infinity, Infinity]` and `[0, 0, 1e39, 1e39]` pack cell size `Infinity` and reciprocal 0.
  All three return without throwing.
- Suggested fix: validate the float32 value (finite and above 0) with a named-limit helper. Not
  checked: non-finite origins in any of the three packers.

### frc-fields-06 (minor; CONFIRMED by reading and NumPy): section 6 of the upstream proposal keeps a superseded block that contradicts its own table

- Where: `dev-docs/upstream-proposal-math-review.md:160-165`.
- What is wrong. The block headed "The committed test as first described (partly superseded ...)"
  still says "On the test GPU today the literal form should pass every case". The table above it
  shows the literal form wrong for bins of width 41 and 83. A maintainer reading to the end meets
  both statements.
- Two wordings to tighten before it is sent (the facts are right, see section 2):
  - "`1 / 2e38` is subnormal ..., so the product is 0": a subnormal is not zero under IEEE; this
    needs flush-to-zero. Say so, since it is already labelled an inference.
  - The case for "the formula is the cause" is stronger than the text: an IEEE evaluation of the
    published expression is wrong at edge 27 of 49 bins (as stated) and also at edges 251, 253,
    502, 506 and 511 of 1000 bins of width 1 (NumPy float32), which the test GPU gets right.
- Suggested fix: delete the superseded block; add the flush-to-zero qualifier.

### frc-fields-07 (minor; CONFIRMED by a run): the fixture runner reports a fixture as not written

- Where: `test/oracles/GENERATORS.md:55`.
- What is wrong. `node scripts/run-oracle-fixture.mjs rts04-change-monitoring-numpy` prints
  `NOT WRITTEN rts04-cold-pyxccd.json (listed in GENERATORS.md for rts04-change-monitoring-numpy.py)`.
  The prose under that generator's heading names the pyxccd fixture in backticks, and the runner
  reads every backticked `.json` under a heading as an expected output. The fixture itself is
  `IDENTICAL`.
- Suggested fix: name the pyxccd generator there, not its fixture file.

## 2. Checked and found correct

**Exact quantile rank (item 1).**

- `multiplyUint32Wide`, `divideUint64ByUint32`, `shiftLeftUint64` and the four rank functions
  were derived by hand (no partial sum overflows; the restoring division carries bit 32; the
  quotient of the fraction is in `[2^24, 2^26)`, one guard bit and a sticky bit, ties to even,
  carry into the exponent) and then run on the GPU on 32,270 triples against BigInt arithmetic
  written apart from the WGSL: 0 failures. The set held counts and denominators at 1, 2, 3,
  `2^16`, `2^24`, `2^31 - 1`, `2^31`, `2^32 - 2`, `2^32 - 1`; numerator 0, 1, the denominator and
  the denominator minus 1; 342 exact halves for `nearest`; 353 fractions that round up to 1.0;
  remainders with 25 significant bits over a power of two (exact float32 ties); and 30,000
  random triples of every bit length.
- All five methods take the rank from the integer functions; `linear` uses
  `min(lower + 1, n - 1)` and the fraction above. The generator's reference is Python integers
  and `fractions.Fraction`, not the kernel's formula.
- NumPy one rank off: verified by hand for 101 rows and `q = 0.29` (see frc-fields-04) and the
  count of 22 in 34,560 was recounted from the fixture.
- Default path: a line diff of `column-quantiles-nodes.ts` against the author's pre-change copy
  shows only insertions guarded by `rationalProbabilities`; the float32 branch, its bindings and
  its variant names are unchanged. 101 rows at `p = 0.59` (58 against 59) and 16,777,216 rows at
  `p = 0.95` (fraction 0.25) reproduce.
- Validation: the packer refuses a zero denominator, a non-integer or negative term and a term
  above `2^32 - 1` with `validateCapacity` and a message naming the term and the limit. A
  numerator above the denominator is packed and gives NaN, as documented and as the float path
  treats a probability outside `[0, 1]`. The view itself is per-frame data, so the class checks
  its format and length only. The row limits (`2^31 - 1`; `2^24` with a filter output) are
  checked in the constructor and the binding size at `getCommandNodes`.
- Documentation names the exact path and the inexact one in the class, in
  `order-statistic-wgsl.ts` and in `docs/api/gpu-dataframe-analysis.md`.

**Diggle test (item 2).**

- The port against the source. `stmctest` sums `kst - outer(ks, kt)` over the whole grid for the
  data and for each `sample(times)`, never calls `stsecal`, and returns `t0` and `t` (it prints a
  rank and forms no p-value). `ksthat`: class by `iplace` (first break `>=` the value), pairs
  dropped at `>= smax` and `>= tmax`, `hs` and `ht` accumulated independently, `hst` when both,
  the three normalisations, `weight` equal to `cncvwt` with no cap. The port matches each. The
  author's statement that the first pass misdescribed `stmctest` is right.
- The kernel's spatial weight for a rectangle (arc of each side minus the corner overlap) was
  rederived by hand. The two-event tie case (`kt` 10 here, 15 in splancs, `D` 0 in both) is right.
- p-value: `(1 + #{U_p >= U}) / (P + 1)` in the kernel. Ties between the observed labelling and a
  permutation that reproduces it are counted: on the GPU, 9 configurations of 3 to 5 events with
  199 permutations (7 to 65 permutations equal to the observed labelling in each) gave the same
  `greaterCount`, `lesserCount` and equal-count as the binary64 reference, although the observed
  row and the permuted rows are summed from different expressions.
- Lags on interior class breaks: 14 combinations of `maximumTimeLag` and class count with integer
  lags on every break (up to 2,330 ordered pairs on a break) gave `K_t` and `K` equal to the
  closed-above rule of `iplace` on the GPU.

**COLD (item 3).**

- The rule. pyxccd 1.1.0 `cold.c:3226-3259`:
  `mini_rmse = max(min_rmse, rmse)`, `v_start` and `v_end` are the first and last residual of the
  window over `mini_rmse`, `v_slope` is the slope times the span over `mini_rmse` (and over
  `SLOPE_SCALE`, which undoes the scaling of the stored slope), `v_dif = fabs(v_slope) +
  max(fabs(v_start), fabs(v_end))`, unstable when the sum of squares exceeds the threshold. The
  kernel at `gpu-raster-cold-monitor.ts:457-465` is the same expression with the same
  normalisation. The initial fit is four coefficients with `rmse = norm / sqrt(n - 4)` in both
  (`misc.c`, `auto_ts_fit_float`, case 4).
- The author's numbers. With the sum in the twin, the fixture series start at (1, 91), (2, 79),
  86 and 103 where pyxccd has (0, 90), (0, 70), 85 and 100; with the larger residual every start
  is pyxccd's except slice 2 for 0 on `two-breaks`. So the fixture separates the two rules on 5
  of 8 series.
- The gap test (`>` against `>=`) and the span test agree for whole-day dates.

**Louvain visit order (item 4).** The literal is `0u, 1u, ..., (n - 1)u` for every size; the
seed is validated as a uint32 and printed as a decimal `u32` literal. Run on the GPU for 1, 2, 3,
5, 13, 31 and 32 vertices by seeds 0, `2^31` and `2^32 - 1`: every shader compiles and returns
valid contiguous communities on a ring. The declaration is 214 characters at 32 vertices and
3,028 at 512.

**Flow packers (item 5).** Every `fieldExtent` and `outputExtent` passed in
`modules/lenses/src`, `modules/kepler/src` (none), `website/src` and `examples/` was read: cell
sizes come from a grid affine, a terrain's cell size, a constant, or viewport bounds divided by a
raster size (deck.gl makes no viewport of zero size). None passes a zero or placeholder extent.

**Sky view (item 6).** `rvt/vis.py` was read independently: direction `i` has angle
`i * 360 / n`, `numpy.roll` brings the neighbour at `(row - round(cos * r), column -
round(sin * r))`, so 0 looks at row 0, 90 at column 0, and the weight peaks where the angle
equals the main direction; 315 is the north-east on a north-up array. The kernel's
`(n - d) mod n` mapping is that. The spec compares the heaviest and lightest main direction for
walls north, west, south, east, north-east and north-north-east, and values for all main
directions on walls at 4 and 8 sectors and on an asymmetric terrain: an east-west mirror fails
the east and west walls, a north-south mirror the north and south walls, and any rotation every
wall.

**Published `GPUHistogram` (item 7).** NumPy 2.5.4 `histogram` returns one per bin and two in the
last for all 11 cases, on float32 and float64 input. The published expression is as quoted. The
two rounding models of the proposal reproduce every listed edge in NumPy float32, and the IEEE
evaluation is wrong at edge 27 of 49 bins. `[0, 123]` in 3 bins with 0, 41, 82, 123 gives
2, 1, 1 under both models. The wide domain: NumPy gives bins 8, 8, 4, 12.

**General.**

- Fixtures: all five generators report `IDENTICAL` (`dataframe-quantile-rational-rank`,
  `space-time-diggle-splancs-port`, `rts04-change-monitoring-numpy`, `terrain-sky-view-rvt` on
  `venv314`; `rts04-cold-pyxccd` on `venv312`).
- Specs: the eight changed headless specs pass (76 tests) and the three node specs pass (11).
- Oracles. `rts04-change-monitoring-numpy.py` is a twin of the kernel's definition and its
  `score` line was changed with the kernel, so it is no evidence for the stability rule; the
  evidence is the pyxccd fixture. The Louvain order and `space-time-diggle-test.spec.ts` also
  compare with TypeScript twins; the independent evidence there is the pair-label property and
  the splancs port.
- Contract: `node scripts/check-contributor-contract.mjs` reports no new findings; the typecheck
  watcher shows 0 errors for the changed source files; no bare `assert`; SPDX licence and
  copyright lines are on the 15 new or changed spec, generator and helper files checked.

## 3. Not checked

- No run of R. The splancs evidence is my reading of the source against the port. The
  surveillance `stKtest` p-value formula and the stpp temporal edge rule were not read from
  source (my recollection is that stpp treats the window as closed, which would make
  frc-fields-01 a genuine conflict between the two references).
- The `stsecal` port (`get_standard_error`) was read against `stsecal.f` only as far as the pair
  weights and the constants; its diagonal formula was not rederived.
- The radix selection above `2^25` rows, and any second adapter, for the quantile path.
- Whether the spec files typecheck (the watcher covers source only).
- pyxccd: no real Landsat series; Tmask, the lookback and the tail were read, not exercised one
  at a time. The random-series comparison used `lam=0` only.
- The other tables of section 9 of `shader-compiler-findings.md` were not rerun; only the
  Louvain change was reviewed.
- `docs/api/gpu-terrain/operators.md` and the sky-view numbers for 16 sectors were not rechecked.
- The float32 default path of `GPUColumnQuantiles` was compared by reading and by the existing
  specs, not by regenerating the shader text of all 42 kernels.
- The website and example callers were read, not run.
