# Math correctness review: trajectories and lines

Scope id: `trajectories-lines`. Date: 2026-11-10. Reviewer scope: every `trajectory-*` directory,
`track-similarity`, and the line family (`line-segmentize`, `linear-referencing`,
`line-simplification`, `line-clip`, `line-split`, `line-merge`) under
`modules/analysis/src/gpu-spatial-analysis`. About 30,000 lines.

Paths are relative to `modules/analysis/src/gpu-spatial-analysis` unless they say otherwise.
Scratch scripts are in
`/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-luma-contributors/9972295f-91de-4373-ba95-d2f799611a84/scratchpad/trajectories-lines/`
(`confirm.mjs`, `arc-length.mjs`). No spec was run. No reference library (MovingPandas, trackintel,
scikit-mobility, Shapely, R `move`, `adehabitatLT`) was run; where a finding depends on what a
reference does, it is marked SUSPECTED and says so.

## 1. Coverage

An unread file is not a verified file.

**Read in full (kernel bodies and parameter packers).**

- `trajectory-analysis/trajectory-metrics-kernels.ts`, `trajectory-analysis/trajectory-common.ts`
- `trajectory-staypoints/movement-common.ts`, `gpu-trajectory-staypoints.ts`,
  `gpu-trajectory-segments.ts`, `gpu-trajectory-locations.ts`
- `trajectory-cleaning/gpu-trajectory-hampel.ts`, `gpu-trajectory-outlier-flags.ts`,
  `gpu-trajectory-thinning.ts`, `gpu-trajectory-track-relative-times.ts`
- `trajectory-estimation/gpu-trajectory-kalman.ts`, `trajectory-kalman-full-covariance.ts`,
  `prepare-trajectory-observations.ts`
- `trajectory-home-range/gpu-brownian-bridge-utilization.ts`,
  `brownian-bridge-variance-estimation.ts`, `gpu-utilization-volume-isopleths.ts`
- `trajectory-mobility/gpu-mobility-indicators.ts`, `gpu-modal-split.ts`
- `trajectory-recursion/gpu-trajectory-recursions.ts`
- `trajectory-interpolation/trajectory-interpolation-kernels.ts`, `trajectory-clock-parameters.ts`,
  `trajectory-playhead-parameters.ts`
- `trajectory-encounters/closest-approach-kernels.ts`, `encounter-kernels.ts`
- `trajectory-restriction/trajectory-restriction-kernels.ts`,
  `trajectory-restriction-pair-merge-kernels.ts`
- `track-similarity/track-similarity-dynamic-programs.ts`, `track-similarity-edit-programs.ts`,
  `track-similarity-search-bounds.ts`, `track-similarity-search-great-circle-bounds.ts`
- `line-simplification/line-simplification-kernels.ts`, `visvalingam-kernels.ts`,
  `line-simplification-parameters.ts`
- `line-segmentize/line-segmentize-kernels.ts`

**Read in part.**

- `track-similarity/gpu-track-similarity.ts`: lines 66-135, 150-345, 606-1030 (every kernel; not
  the constructor or the status kernel). `gpu-track-similarity-search.ts`: 150-558.
- `linear-referencing/linear-referencing-kernels.ts`: 94-283, 290-664, 790-977 (planar and
  spherical projection, arc bounds, locate; not the segment node or the scatter plumbing).
- `line-segmentize/line-chunk-kernels.ts`, `line-smooth-kernels.ts`: kernel bodies, bindings skipped.
- `trajectory-zones/zone-events-kernels.ts`: 34-66, 100-500, 630-1050 (edges, probe, crossing,
  initial state, event time, gap, dwell, finalize). Not the sort, gather or table kernels.
- `trajectory-cleaning/gpu-trajectory-split.ts`: 36-365.
- `trajectory-analysis/gpu-trajectory-metrics.ts`: 40-300 (documentation and validation).
- `trajectory-analysis/trajectory-sort-check.ts`: 60-112.
- `trajectory-interpolation/gpu-trajectory-resample.ts`: 30-135 (documentation).
- `trajectory-encounters/gpu-trajectory-encounters.ts`: 99-300 (lattice sizing and its proof).
- `line-split/gpu-line-split.ts`: 76-170, 395-480. `line-clip/gpu-line-clip-by-polygon.ts`: 150-215
  and the sub-piece midpoint lines. `line-merge/gpu-line-merge.ts`: 60-135, 225-300.
- `line-simplification/gpu-line-simplification.ts`, `linear-referencing/gpu-line-locate.ts`,
  `gpu-linear-referencing.ts`, `line-segmentize/gpu-great-circle-arcs.ts`,
  `trajectory-zones/gpu-zone-events.ts`: documentation and selected lines by search only.

**Not opened.**

- `trajectory-encounters/clock-encounters.ts`, `gpu-trajectory-closest-approach.ts`
- `trajectory-interpolation/gpu-trajectory-playhead.ts`
- `trajectory-restriction/gpu-trajectory-restriction.ts`
- `trajectory-analysis/trajectory-metrics-parameters.ts`
- `line-clip/gpu-shared-paths.ts`
- `line-merge/list-ranking.ts`
- `line-segmentize/gpu-line-chunk.ts`, `gpu-line-segmentize.ts`, `gpu-line-smooth.ts`,
  `balanced-search.ts`, `line-geometry-output.ts`, `line-segmentize-parameters.ts`,
  `line-segmentize-types.ts`
- every `index.ts`

The topology kernels of line split, line clip, line merge and shared paths (event sort, unique
flags, piece tables, pointer jumping) were not verified line by line. Their predicates come from
`GPUSegmentIntersection` and `GPUPointInPolygonJoin`, which are outside this scope.

Dependencies read outside the scope: `geometry-measures/geodesic-wgsl.ts` (1-220),
`analysis-core/kernels/sphere-distance-wgsl.ts`, `analysis-core/contracts/earth.ts`.
`TIME_WORDS_WGSL` (`dataframe-analysis/time-window-filter/time-words.ts`) was not opened; the exact
Int64 subtraction is taken on trust.

## 2. Findings

| ID | Severity | Status | File:line | What the kernel computes | What the reference says | Input that gives a wrong output | Suggested fix |
| --- | --- | --- | --- | --- | --- | --- | --- |
| trajectories-lines-01 | wrong-result | CONFIRMED (script) | `trajectory-cleaning/gpu-trajectory-outlier-flags.ts:241-243` (constant doc `:45-46`, class doc `:104-105`) | Acceleration bit is set when the scalar accelerations ending at row `i` and at row `i + 1` both exceed the limit | The bit is documented as "the fix is displaced". A displaced fix has two fast steps of about equal speed, so the acceleration between them is about 0 | 1 Hz track at 10 m/s, row 5 displaced 500 m sideways, `maximumAcceleration: 50`, speed test off: no row gets the acceleration bit | Pair the acceleration into the fix with the one after it (`a_i` and `a_{i+2}`), or use the change of the velocity vector across the fix |
| trajectories-lines-02 | wrong-edge-case | CONFIRMED (script) | `trajectory-mobility/gpu-modal-split.ts:264-272` | `mode` starts at 0 and is only replaced when `speed < break`; a speed at or above the last finite break keeps 0 | trackintel returns no category there; the prop doc defines classes only below the last break | `modeSpeedBreaks: [15/3.6, 100/3.6]`, tripleg at 60 m/s: mode 0, the slowest class | Fall through to the last class, or require the last break to be `Infinity`, or write `0xffffffff` |
| trajectories-lines-03 | precision | CONFIRMED (script, worst case) | `trajectory-interpolation/trajectory-interpolation-kernels.ts:541-546` | Cumulative arc length is a plain f32 running sum | The line family sums the same quantity with a Neumaier-compensated f32 sum (`line-segmentize/line-segmentize-kernels.ts:195-213`) | 86,400 steps of 13.9 m: plain sum 1,199,909 against 1,200,960 exact (1,051 m low). The compensated sum is 0.4 m off | Use the compensated prefix of the line family for `spacing: 'arc-length'` |
| trajectories-lines-04 | doc-mismatch | CONFIRMED for the code; SUSPECTED for `move` | `trajectory-home-range/gpu-utilization-volume-isopleths.ts:319` (doc `:80-85`, `docs/api/gpu-spatial-analysis/home-range-and-recursion-mob07.md:56-58`) | `volume` includes the cell's own mass, and is 0 for a zero-density cell | Documented as "the volume of `getVolumeUD` in `move`", which as recalled is `1 - cumsum(sort(values))[rank]`: own mass excluded, zero cells get 1 | Densities `0.5, 0.3, 0.15, 0.05, 0, 0`: local volumes `0.50, 0.80, 0.95, 1.00, 0, 0`; `move` `0, 0.50, 0.80, 0.95, 1, 1`. `volume <= 0.95` selects the two empty cells | Write 1 for zero-density cells and either match `move` or drop the claim. `inside` is correct and is the safe output |
| trajectories-lines-05 | doc-mismatch | SUSPECTED (reference from memory) | `trajectory-recursion/gpu-trajectory-recursions.ts:72-76`, `:394-421` | First passage time is the time from the fix forward until the track leaves the circle | Fauchald and Tveraa (2003) and `adehabitatLT::fpt` use the time between the first crossing backward and the first crossing forward | Straight track at constant speed `v`, radius `r`: kernel `r / v`, two-sided definition `2 r / v` | Add the backward leg, or keep the forward value and stop citing Fauchald and Tveraa for it |
| trajectories-lines-06 | doc-mismatch | SUSPECTED (reference from memory) | `trajectory-home-range/brownian-bridge-variance-estimation.ts:45`, `:65-66` | The likelihood variance adds the predicted fix's own error `delta[i]^2` | Horne et al. (2007), and the R packages `BBMM` and `move`, use `T a (1 - a) s^2 + (1 - a)^2 d^2 + a^2 d^2` without it | Random walk, true variance 15 m²/s, location error 20 m: with the term 15.6, without it 28.5 | Keep the formula (it is the better estimator) and say it departs from Horne, or add an option |
| trajectories-lines-07 | doc-mismatch | SUSPECTED (reference from memory) | `trajectory-cleaning/gpu-trajectory-split.ts:104-109`, `:250-256` | Mode `'stop'` emits the maximal runs of rows outside every staypoint | Documented as MovingPandas 0.23 `StopSplitter`, whose segments run from the last fix of one stop to the first fix of the next, both included | Stay on rows 3..8: kernel segment before it is rows 0..2; MovingPandas would give rows 0..3 | Say that it is the trackintel `between_staypoints` rule, or include the two boundary rows |
| trajectories-lines-08 | wrong-edge-case | CONFIRMED (by hand) for the status word | `linear-referencing/linear-referencing-kernels.ts:876-888` | A NaN event measure passes both range tests (`< 0`, `> total`) and keeps status `ok` | An unusable measure should be `invalid` | `eventMeasures = NaN` on a valid line: status 0 (`ok`); the position is whatever `clamp(NaN)` gives on the device | Test `!(measure >= 0.0 || measure < 0.0)` first and write `invalid` with NaN outputs |
| trajectories-lines-09 | inconsistent | CONFIRMED (by reading) | `trajectory-restriction/trajectory-restriction-kernels.ts:146` against `trajectory-zones/zone-events-kernels.ts:282-300` | Restriction decides "on the polygon boundary" with a tolerance `|cross| <= 1e-6 * max(1, |edge|^2)`; zone events use exact orientation with simulation of simplicity | One boundary rule for one question (contract section 11) | A point 0.05 m from a 100 km zone edge is "on the boundary" in restriction (inside when closed, outside when open) and strictly inside or outside in zone events | Use the exact predicates in restriction as well |

### trajectories-lines-01: the acceleration flag does not mark a displaced fix

The kernel does what its formula says: it flags row `i` when `|v_i - v_{i-1}| / dt_i` and
`|v_{i+1} - v_i| / dt_{i+1}` both exceed the limit, with `v` the scalar step speed. The formula
does not do what its label says. At a displaced fix both adjacent steps are fast, by about the same
amount, so `v_{i+1} - v_i` is near 0. The two large accelerations are at rows `i` and `i + 2`.

`confirm.mjs` replicates lines 211-245 with `Math.fround`:

- Displaced fix at row 5 (500 m sideways, 1 Hz, 10 m/s), speed limit 100, acceleration limit 50:
  flags `0 0 0 0 0 1 0 0 0 0 0`. Only the speed bit. With the speed test off: all zero.
- Position jump (rows 5 onward shifted 500 m, so no single fix is displaced): row 5 gets the
  acceleration bit.

So with `maximumSpeed: 0` a GPS spike is not flagged at all, and with both tests on the
acceleration bit marks the first good row after a jump. The spec
(`test/gpu-spatial-analysis/trajectory-cleaning/trajectory-cleaning.spec.ts:327-345`) pins the
kernel with one fast step in a ramp, and computes the expected flags with the same formula. No
independent reference is involved: MovingPandas has no acceleration cleaner, and the oracle
(`test/oracles/movement-cleaning.py`) records only isolated speed spikes.

The speed rule on the same lines (234-240) is right and was checked by hand, including both
endpoint cases.

### trajectories-lines-02: speeds above the last break land in class 0

`var mode = 0u;` then a loop that assigns only on `speed < SPEED_BREAKS[candidate]`. With the
trackintel default (`Infinity` last, emitted as the f32 maximum) every finite speed is classified.
With a finite last break the fastest triplegs silently join the slowest class, and so does a NaN
speed. `confirm.mjs`: breaks `[4.17, 27.8]`, speeds 27.8, 60 and 250 give mode 0. The modal split
then adds their distance to class 0.

### trajectories-lines-03: arc length is a plain f32 sum

`arc-length.mjs` sums a constant step in f32 both ways. A constant step is the worst case because
every addition rounds the same way; real tracks with varying steps lose less.

| Track | Exact | Plain f32 sum | Neumaier f32 sum |
| --- | --- | --- | --- |
| 86,400 steps of 13.9 m | 1,200,960 | 1,199,909 (-0.09 %) | 1,200,960.4 |
| 1,000,000 steps of 13.9 m | 13,899,999.6 | 13,982,225 (+0.59 %) | 13,899,318 (-0.005 %) |
| 5,000,000 steps of 1.4 m | 6,999,999.9 | 7,321,324.5 (+4.6 %) | 7,010,881 (+0.16 %) |

The error is not uniform along the track (it changes each time the running total crosses a power of
two), so arc-length samples move relative to each other. It only affects `spacing: 'arc-length'`
of `GPUTrajectoryResample`. Track lengths of `GPUTrajectoryMetrics` come from the shared segment
sum and are not affected.

### trajectories-lines-04: isopleth volume convention

Two separate points.

1. Zero-density cells get volume 0 (line 319, documented at line 82). The value is otherwise
   monotone: the less dense the cell, the larger the volume, up to 1. A zero cell should be 1. The
   usual idiom with a volume raster, `volume <= level`, then selects every empty cell.
   `confirm.mjs` shows it. This is CONFIRMED from the code.
2. The claim "the volume of `getVolumeUD` in `move`" could not be checked (R is not installed, and
   the doc page says so). From memory `move` computes `1 - cumsum(sort(values))[rank]`, which is
   the mass of strictly denser cells: 0 for the densest cell and 1 for empty cells. The local value
   includes the cell itself. On a fine grid the difference is one cell's mass. SUSPECTED.

`output.inside` implements "smallest set of densest cells whose mass reaches the level" correctly.

### trajectories-lines-05, -06, -07: three definitions that depart from the cited reference

All three kernels compute exactly what their own documentation states. The departure is from the
paper or package named next to it. I could not run the references, so these need a second opinion
from someone who can.

- **First passage time.** The forward-only time is the physicist's first passage time. The
  ecology definition cited (Fauchald and Tveraa 2003, and `adehabitatLT::fpt`) is the time to
  cross the circle: backward leg plus forward leg. Values here are about half of theirs. The
  censoring flag and the interpolation of the exit time on the leaving segment are right.
- **Brownian motion variance.** Adding `delta[i]^2` is statistically sound: the observed fix has
  its own error. With errors on every fix the local estimator recovers the true variance and the
  Horne form is biased high (`confirm.mjs`: 15.6 against 28.5, truth 15). But results will be
  lower than `BBMM` and `move` for the same input whenever the location error is not 0. The
  density kernel (`gpu-brownian-bridge-utilization.ts:312-314`) correctly leaves the term out,
  because it describes the true position.
- **Stop split.** The kernel is the trackintel tripleg rule. The spec case
  (`trajectory-cleaning.spec.ts:127-174`) is "a reference from the definition"; the MovingPandas
  oracle does not record a stop split.

### trajectories-lines-09: two boundary rules

Restriction's `inside()` returns the closed or open answer for any point within a band around an
edge. The band is `1e-6 * max(1 / |d|, |d|)` wide for an edge of length `|d|`: 2 µm for a 0.5 m
edge, 1 mm for a 1 km edge, 0.1 m for a 100 km edge. It is defensible on its own. It is listed
because the sibling contributor answers the same question exactly, and a caller who runs both on
one zone can get different answers for a fix near a long edge.

### Status of earlier findings (spatiotemporal review, `code-movement.md`)

| Earlier finding | Status in the working tree |
| --- | --- |
| F1 clock resample interpolates across gaps | Fixed as an option. `maximumGap` exists in resample (`trajectory-interpolation-kernels.ts:788-796`), closest approach (`closest-approach-kernels.ts:251-258`) and zones (`zone-events-kernels.ts:637-749`). Omitted or 0 still interpolates, which is documented |
| F2 Kalman and restriction only take f32 times | Fixed. Both take Int64 words (`trajectory-kalman-full-covariance.ts:30-37`, `trajectory-restriction-kernels.ts:61-93`). Kalman steps are exact. Restriction interval end points are still f32 relative to `timeOrigin` |
| F3 stops by step speed | `GPUTrajectoryStaypoints` added (verified below). `GPUTrajectoryMetrics` unchanged and documented |
| F7 one long gap overflows closest approach | Fixed when `maximumGap` is given (rows skipped before the cell cap) |
| F8 encounters near the antimeridian | Still present, documented (`gpu-trajectory-encounters.ts:150-155`): only bounds of exactly `[-180, 180]` wrap |
| F9 similarity search is planar only | Fixed. The search resolves a spatial context and has great-circle bounds |
| Precision: resample progress measured from the track start in f32 | Still present: `trajectory-interpolation-kernels.ts:704`, `:750` |
| Precision: zone event times measured from the track start in f32 | Still present: `trajectory-zones/zone-events-kernels.ts:484-486` |
| Precision: closest-approach boxes measured from row 0 | Still present: `closest-approach-kernels.ts:170-171`. Only candidate generation; the pair test is measured from the segment row |

## 3. Verified correct

Each line says what was compared. "By hand" means the WGSL was traced against the reference
recurrence or formula.

- `GPUTrajectoryMetrics` (steps, headings, accelerations, dwell, centroids): by hand against its
  own documentation. Heading is `atan2(dy, dx)`; the great-circle branch gives
  `atan2(cos b, sin b) = pi/2 - b`, the same convention. Dwell row count `b - a + 2` and the
  centroid segment offsets are monotone and disjoint.
- `GPUTrajectoryStaypoints`: by hand against trackintel `_generate_staypoints_sliding_user`
  (gap test first, `distance >= threshold`, `duration >= threshold`, restart at the breaking fix,
  `include_last`). Li et al. (2008) use strict `>`; the code follows trackintel and says so.
  Pointer doubling: the pass count `bitLength(rowCount - 1)` reaches every chain distance, and the
  race only adds true members.
- `GPUTrajectorySegments` triplegs: by hand against trackintel `between_staypoints` (runs outside
  staypoints, split at gaps, at least two rows, not all one coordinate). Trip state machine: the
  new-trip rule matches trackintel (first of user, gap, last activity of a run). Origin and
  destination handling was read but not derived case by case; the trackintel fixture pins it.
- `GPUTrajectoryLocations`: min-label hooking with path halving converges to connected components
  of the inclusive epsilon graph (DBSCAN with `min_samples = 1`). The convergence test (one stable
  hook round) implies one label per component, equal to its lowest row.
- `GPUTrajectoryHampel`: by hand against the `hampel` package 1.x (`window_size // 2`,
  `1.4826 * MAD`, strict `>`, ends pass through).
- `GPUTrajectoryOutlierFlags` speed bit: by hand, interior and both endpoint rules.
- `GPUTrajectoryThinning`: by hand against MovingPandas `MinDistanceGeneralizer` and
  `MinTimeDeltaGeneralizer` (`>= tolerance` from the last kept row, first and last kept).
- `GPUTrajectorySplit` `'gap'` and `'value-change'`: by hand against MovingPandas
  `ObservationGapSplitter` (strict `>`) and `ValueChangeSplitter` (segment repeats the first row
  of the next; `len > 1`).
- `GPUTrajectoryKalman`, both covariance modes: every matrix written out in column-major order and
  checked. Transition `[[1, dt], [0, 1]]`; process noise `q [[dt^3/3, dt^2/2], [dt^2/2, dt]]`;
  gain `P H^T / S`; Joseph update `(I - K H) P (I - K H)^T + K R K^T`; symmetry enforced on store;
  RTS gain `P F^T P_pred^-1`, state and covariance corrections. The 4x4 block layout of the full
  mode and the pivoted inverse are right. Word times difference exactly per step.
- `prepareGPUTrajectoryObservations`: WGS84 constants are consistent (`b = a (1 - f)`,
  `e^2 = 2f - f^2`); the quotient and remainder split is right.
- `GPUBrownianBridgeUtilization`: variance `T a (1 - a) s^2 + (1 - a)^2 d_a^2 + a^2 d_b^2`, the
  isotropic normal `exp(-r^2 / 2v) / (2 pi v)`, duration weighting and the 7-sigma reach bound, by
  hand against Horne et al. (2007).
- `GPUUtilizationVolumeIsopleths` `inside` and the three-level prefix sum: by hand. Ties are in
  cell order because published `GPUSort` descending radix is stable (checked in its source).
- `GPUMobilityIndicators`: by hand against scikit-mobility 1.3.1 source as recalled. Radius of
  gyration (haversine to the arithmetic mean of coordinates), random and uncorrelated entropy, and
  the Lempel-Ziv real entropy including the end rule (`lambda = n - i + 1` when the match reaches
  the last element) and the first and last terms 1 and 2. Home and work against trackintel FREQ;
  tracking quality against `temporal_tracking_quality`.
- `GPUModalSplit` accumulation and `speed < bound` class rule: against trackintel, apart from
  finding 02.
- `GPUTrajectoryRecursions` visits and residence time: the quadratic for the closed disc, the
  visit count (a segment starting outside that reaches the disc starts a visit) and the window
  rule, by hand.
- `GPUTrajectoryResample` and `GPUTrajectoryPlayhead` kernels: upper-bound search, fraction,
  duplicate timestamps, gap rule (strictly longer, strictly inside), by hand.
- `GPUTrajectoryEncounters`: the half-shell visits each adjacent cell pair once, including the
  wrapped lattice of three or more columns. The great-circle search limit
  `asin(sin(cell) cos(L))` was derived and is conservative.
- `GPUTrajectoryClosestApproach` pair kernel: relative motion quadratic in half-`b` form, clamped
  closest time, root interval, and the padded boxes (half the distance each) with the
  minimum-corner rule that tests each segment pair once.
- `GPUZoneEvents`: crossing parameter `cross(a - p, e) / cross(d, e)`, ray parity, visit count
  `seed + enters`, dwell sums and the gap-observed time, by hand.
- `GPUTrajectoryRestriction`: polygon and rectangle interval boundaries, even-odd rule, the
  pairwise closest-approach form (distance at closest approach before subtracting the radius), and
  the two-pointer merge invariant.
- `GPUTrackSimilarity`: the anti-diagonal layout (up, up-left, left read from the right diagonals,
  carried row across strips) for discrete Fréchet, DTW, LCSS, EDR and ERP, each against its
  recurrence (Eiter and Mannila; `similaritymeasures.dtw`; tslearn `lcss` and `sakoe_chiba_mask`;
  Chen 2005; Chen and Ng 2004). Hausdorff with the Taha-Hanbury break is unchanged by the break.
  Densify sampling matches GEOS. SSPD matches Besse et al.
- `GPUTrackSimilaritySearch` bounds: all admissible. Endpoint bound for Fréchet and for
  sum-of-distances DTW with any band; box bounds for Hausdorff and SSPD; `max(n, m) - min(possible)`
  for EDR; reverse triangle inequality for ERP; the great-circle fallbacks.
- `GPULineSimplification` Douglas-Peucker and time ratio: segment distance (clamped, as GEOS),
  importance `min(distance, parent)`, keep when `importance > tolerance`, first maximum on ties.
  Visvalingam is a documented approximation of the heap order; its area and floor rule are right.
- `GPULinearReferencing`: planar projection, measure, side sign; spherical arc projection
  (`atan2` on the arc plane, cross-track angle, end-point fallback), arc latitude bounds
  (`asin(sqrt(upZ^2 + tangentZ^2))` at `atan2(tangentZ, upZ)`), lowest-row tie rule.
- `GPULineLocate`: upper-bound search, fraction, left-positive offset in both models, apart from
  finding 08.
- `GPULineSegmentize`, `GPUGreatCircleArcs`: `ceil(length / maximum)` pieces as PostGIS and GEOS;
  compensated prefix; longitude unwrapping. `GPULineChunk`: `ceil(length / chunk)` pieces and the
  interior-vertex searches. `GPULineSmooth`: Chaikin corner cutting, `n` to `2n` vertices, open
  ends kept, rings wrapped.
- `GPULineSplit` event key (dominant coordinate along the segment, ordered f32 key) and
  `GPULineClipByPolygon` classification rule (collinear overlap is boundary, midpoint otherwise):
  read and consistent with Shapely `intersection` and `difference`. Not verified beyond that.

## 4. Standards consistency

**One convention, used consistently.**

- Sphere distance. Every contributor in scope calls `geodesicHaversineDistance` or
  `geodesicCentralAngle` of `GEODESIC_WGSL`, with the radius from `getGPUGreatCircleRadius` of the
  context (mean Earth radius 6,371,008.8 m by default). No second Earth radius. trackintel and
  scikit-mobility use 6,371,000 m; the staypoint doc tells the caller to set `sphereRadius`.
- Gap rule. `maximumGap` means "a step strictly longer is a gap, 0 is no limit" in staypoints,
  segments, split, resample, playhead, closest approach, zones, Kalman and restriction.
- Missing index. `0xffffffff` everywhere.
- Offsets. Positive is left of the direction of travel in linear referencing (signed offset) and
  line locate (planar and spherical).

**Two conventions for one thing.**

- Direction angles. Radians counter-clockwise from east in `stepHeadings`
  (`trajectory-analysis/gpu-trajectory-metrics.ts:97-104`) and playhead `headings`; degrees
  counter-clockwise from +x in line locate `angles` (`linear-referencing/gpu-line-locate.ts:103-107`,
  `:355`); degrees clockwise from north in `geodesicInitialBearingDegrees`. Each is documented.
- Missing float value. NaN (resample, restriction, Kalman, similarity, line locate); 0 (metrics
  stop columns, staypoint durations and centroids past the count, mobility outputs); -1 (zone
  `firstEnterTimes`, `zone-events-kernels.ts:1022`; unmatched distance in linear referencing,
  `linear-referencing-kernels.ts:134`).
- Polygon boundary rule: finding 09.
- Summation. Neumaier prefix in the line family; plain running sums in resample arc length
  (finding 03), zone gap prefix (`zone-events-kernels.ts:742-747`), Brownian bridge total duration,
  and DTW and ERP (documented there).
- Sphere radius literal. `getWGSLFloatLiteral(radius)` in most kernels;
  `radius.toExponential(10)` in `trajectory-staypoints/gpu-trajectory-locations.ts:215`; the raw
  number in `track-similarity/gpu-track-similarity.ts:657` and
  `track-similarity-search-great-circle-bounds.ts:22`. Same f32 value in each case.

**Private copies of something `analysis-core` has.**

- Finite test `fn finite(value)`: six copies (`trajectory-estimation/gpu-trajectory-kalman.ts:231`,
  `trajectory-kalman-full-covariance.ts:49`, `trajectory-restriction/gpu-trajectory-restriction.ts:403`,
  `trajectory-restriction-kernels.ts:135`, `trajectory-restriction-pair-merge-kernels.ts:157`,
  `track-similarity/gpu-track-similarity-search.ts:302`). Same bit test as `IS_FINITE_VALUE_WGSL`.
- NaN construction: `bitcast<f32>(0x7fc00000u | (index & 0u))` in resample, encounters, zones and
  similarity; `var nanBits = 0x7fc00000u` in restriction and the search. `GET_NAN_WGSL` exists and
  Kalman uses it. Same value.
- Owner-of-row binary search: `findTrack` in `trajectory-metrics-kernels.ts:131`,
  `closest-approach-kernels.ts:134` and `:238`, and zone events; `findLine` in
  `line-segmentize-kernels.ts:28`; `lineOfVertex` in `line-split/gpu-line-split.ts:417`. Integer
  searches, same results. `analysis-core/kernels/binary-search-wgsl.ts` exists.
- Ordered float key: inline in `gpu-line-split.ts:472` and the search top-k. Same mapping as
  `GET_ORDERED_KEY_WGSL`.
- 4x4 Gauss-Jordan inverse in `trajectory-kalman-full-covariance.ts:72-84`. Its singularity test is
  an exact zero pivot. I did not compare it with `dense-solve-wgsl.ts`; the library review tracks
  the solver copies.
- Space-time cell hash in `closest-approach-kernels.ts:58-61` (the 73856093 / 19349663 / 83492791
  hash), not `hash-uint32.ts`. It only buckets; every use compares cell coordinates. Not a random
  number source.

**Literals where a named constant exists.**

- WGS84 spheroid in `trajectory-estimation/prepare-trajectory-observations.ts:19-23`, against
  `GPU_WGS84_SEMI_MAJOR_AXIS_METERS` and `GPU_WGS84_FLATTENING` in `contracts/earth.ts`. Same values.
- `PI` and degrees per radian in `linear-referencing-kernels.ts:318-319` and
  `gpu-line-locate.ts:355`, against the `GEODESIC_` constants. Same values.
- The f32 maximum is spelled `3.402823466e+38`, `3.402823e38` and `3.4e38` (14 places in scope).

## 5. luma.gl official implementations

No contributor in this scope re-implements a published class one for one. The published building
blocks (`GPUCompaction`, `GPUSort`, `GPUScan`) are used, with arguments that keep their meaning.

Closest relatives in `@luma.gl/graph/geospatial`:

- `GPUPairwisePointLinestringNearest` against the planar path of `GPULinearReferencing`. The
  official class is pairwise (each point against its own geometry) and returns the nearest point,
  linestring and segment. The local class searches all lines through a BVH and adds measure,
  fraction and side. Where they overlap the math agrees: same clamped projection, and the same tie
  rule (first segment wins: official strict `<`, local lowest row). The official kernel also
  normalizes exponents with `frexp` and `ldexp` so that extreme magnitudes do not overflow the
  squared length; the local kernel does not. Not relevant for map coordinates. Delegation is not
  possible without losing the cross-line search.
- `GPUPairwisePointSegmentDistance` uses the same clamped point-to-segment formula as the planar
  `pointSegmentMetric` of track similarity and `lineSegmentDistance` of Douglas-Peucker. Those need
  it inside a kernel, and the entry point publishes a contributor, not a WGSL function.
- `GPUHaversineDistance` against `GEODESIC_WGSL`. Three differences of convention, none an error:
  the official default radius is 6371 (kilometres, the cuSpatial convention) and the local default
  is 6,371,008.8 metres; the official kernel switches to an equirectangular formula below 1e-4 rad
  and the local one uses the `atan2` form at every distance; the official kernel subtracts
  longitudes directly and the local one takes the difference across the antimeridian first. A
  caller comparing a local trajectory length with `GPUHaversineDistance` output will see a factor
  of 1000.0014 unless the radii are aligned.
- `gpu-timeseries` (`GPUTimeSeries`, `GPUTimeSeriesEnvelope`) is for uniformly sampled series and
  does not overlap the Hampel filter or resampling.

Candidates for upstream proposals rather than local copies: a published WGSL point-to-segment and
great-circle function (today each consumer inlines one), and a segmented compensated prefix sum.

## 6. Test oracles

**Independent oracle (a reference library produced the expected values).**

- Staypoints, triplegs, trips, locations, home and work, tracking quality, modal split: trackintel
  (`test/oracles/movement-trackintel.py`).
- Mobility indicators: scikit-mobility 1.3.1 (`movement-scikit-mobility.py`).
- Gap split, value-change split, thinning, time-ratio simplification, Hampel, speed outliers on
  isolated spikes: MovingPandas 0.23 and `hampel` (`movement-cleaning.py`).
- Kalman: NumPy forward and backward solve, and a two-observation Gaussian calculation.
- Track similarity: Shapely (Hausdorff, Fréchet), `similaritymeasures`, tslearn
  (`track-similarity-dtw-frechet.py`, `track-similarity-edit-measures.py`).
- Trajectory metrics in great-circle mode: pyproj (`geodesic-trajectory.py`).
- Encounters, closest approach, zone events: `parity-wp2-trajectory-encounters.py`,
  `trajectory-closest-approach.py`, `parity-wp2-zone-events.py`.
- Line simplification: `parity-wp2-line-simplification.py`, `parity-wp2-round8-simplification.py`;
  Visvalingam against `simplification.cutil` at stated tolerances.
- Line clip, shared paths, line split: Shapely fixtures. Spherical lines:
  `parity-wp2-round8-spherical-lines.py`.

**Only a copy of the contributor's own formula.**

- Brownian bridge density, variance estimator, volume isopleths, recursions and first passage time
  (`trajectory-home-range/home-range-recursion-mob07.spec.ts`): binary64 CPU versions of the same
  formulas plus hand cases. The documentation states this. Findings 04, 05 and 06 are the places
  where the formula itself is in question.
- Outlier acceleration bit and split mode `'stop'`: expected values computed from the definition
  in the spec (findings 01 and 07).
- Resample, playhead, line segmentize, line chunk, line smooth, linear referencing (planar),
  line merge: in-repository TypeScript oracles. I did not check whether each is written
  independently of the kernel.

**Nothing found.**

- Modal split with a finite last break (finding 02), NaN measures in line locate (finding 08),
  long-track arc length (finding 03).

## Resolution (2026-11-10)

Source paths are relative to `modules/analysis/src/gpu-spatial-analysis`, spec paths to
`modules/analysis/test/gpu-spatial-analysis`. Every spec named below was run on the test GPU
(headless Chromium, Metal). Findings 02, 03, 04, 08 and 09 were also run against the old kernel
first and failed there; finding 01 was not (the kernel was changed before the first run).

| ID | Outcome | What changed | Spec that pins it | Behaviour change a caller sees |
| --- | --- | --- | --- | --- |
| trajectories-lines-01 | fixed (code follows the documentation, roadmap 12.3) | `trajectory-cleaning/gpu-trajectory-outlier-flags.ts:235-262`, call at `:279`. Row `i` gets bit 2 when the signed acceleration into it exceeds the limit and the acceleration two rows later is below minus the limit. First row: the fall only. Last row: the rise only. Second and second-to-last row: the one side that exists, unless the acceleration between the row's two steps contradicts it. Class and constant documentation, `docs/api/gpu-spatial-analysis/movement-semantics.md` | `trajectory-cleaning/trajectory-cleaning.spec.ts:376` and the test after it. Hand-built tracks: one displaced fix, two consecutive, a sustained jump, displaced first and last row, displaced second and second-to-last row. The old test, which computed its expectation with the kernel's formula, is removed | A single displaced fix now gets the bit (before: never). The first good row after a sustained jump no longer gets it (before: flagged). Two consecutive displaced fixes: no bit on any row (before: the first of the two) |
| trajectories-lines-02 | fixed | `trajectory-mobility/gpu-modal-split.ts:275-292` (predict), `:317-320` (aggregate). Prop documentation and `movement-semantics.md` | `trajectory-mobility/mobility-indicators.spec.ts:358-440` (two tests, hand-built triplegs) | With `modeSpeedBreaks`, a speed at or above a finite last break is the last mode (before: mode 0). A tripleg of zero duration or non-finite length gets `0xffffffff` in `output.predictedModes` and is left out of `output.modalSplit` (before: mode 0). A caller's `modes` value of `0xffffffff` is left out of the split (before: counted in the last class) |
| trajectories-lines-03 | fixed | `trajectory-interpolation/trajectory-interpolation-kernels.ts:541-556`, with the correction term from the new `ADDITION_ROUNDING_ERROR_WGSL` (now in `analysis-core/kernels/addition-rounding-error-wgsl.ts`, re-exported at `line-segmentize/line-segmentize-kernels.ts:77`). The oracle `trajectory-interpolation/trajectory-interpolation-oracle.ts` emulated the plain f32 sum; its arc length is now a binary64 sum rounded per row | `trajectory-interpolation/gpu-trajectory-resample.spec.ts:102` (closed form, 86,400 equal steps) | `spacing: 'arc-length'` samples of long tracks move to the right place: on the test track the interior samples were 11 to 44 rows off and are now within 0.03 rows |
| trajectories-lines-04 | fixed (empty cells), reworded (`move` claim) | `trajectory-home-range/gpu-utilization-volume-isopleths.ts:324` writes 1 for a cell without mass. Output and class documentation, `docs/api/gpu-spatial-analysis/home-range-and-recursion-mob07.md` | `trajectory-home-range/home-range-recursion-mob07.spec.ts:356`; the 70,000-cell test in the same file had the old 0 in its reference and now expects 1 | `output.volume` of a zero, negative or NaN cell is 1 (before: 0), also when the whole surface is empty. `output.inside` is unchanged. Whether a cell's own mass is counted in `move::getVolumeUD` stays unverified; the parity claim is removed |
| trajectories-lines-05 | fixed (documentation only) | `trajectory-recursion/gpu-trajectory-recursions.ts:72-78`, `:107-124`; `home-range-and-recursion-mob07.md`. The value is described as the forward leg only; the Fauchald and Tveraa citation now names the difference | none (no formula change) | none |
| trajectories-lines-06 | fixed (documentation only) | `trajectory-home-range/brownian-bridge-variance-estimation.ts:40-54`, `:139-156`; `home-range-and-recursion-mob07.md`. The predicted fix's own error term is stated, and that estimates are lower than `BBMM` and `move` for a location error above 0 | none (no formula change) | none |
| trajectories-lines-07 | fixed (documentation only) | `trajectory-cleaning/gpu-trajectory-split.ts:104-115`; `movement-semantics.md`. Mode `'stop'` is described as the trackintel tripleg rule and no longer as `StopSplitter` | none (no formula change) | none |
| trajectories-lines-08 | fixed | `linear-referencing/linear-referencing-kernels.ts:884-894`. Status constant, class documentation, `docs/api/gpu-spatial-analysis/operators-lines.md` | `linear-referencing/gpu-linear-referencing.spec.ts:663` | A NaN event measure (after the per-frame scale and offset) gives status `invalid`, NaN position and tangent, segment index `0xffffffff` (before: status `ok`). Infinite measures still clamp |
| trajectories-lines-09 | fixed (converged on the exact predicates, contract section 11) | `trajectory-restriction/trajectory-restriction-kernels.ts:143-163` (the polygon test is `exactProbeCrossing`), `:189-250` (one classification pass). `geometryBoundary` documentation, `docs/api/gpu-network/network-movement-coverage.md` | `trajectory-restriction/trajectory-restriction-boundary.spec.ts:167` (six hand-read cases, closed and open) and `:102` (128-vertex ring crossed through two vertices) | A point within `1e-6 * edge length` of a non-axis-aligned polygon edge is no longer "on the boundary": it is inside or outside by its exact side. Before, such a point was inside for `'closed'` and outside for `'open'`. Axis-aligned edges were not affected by the band (its bounding-box test excluded them), so the reviewer's 100 km example holds for diagonal edges only. Rectangle restriction is unchanged |

### Not in the report: the compensated sums were not compensated

Found while fixing finding 03. The first fix used the float Neumaier term the brief names,
`(sum - nextSum) + segmentLength`. The spec still failed by exactly the plain-sum error (11.357
rows). The same was then measured for the line family's own prefix
(`line-segmentize/line-segmentize-kernels.ts`, used by `GPULineLocate`, `GPULinearReferencing`,
`GPULineSegmentize`, `GPULineChunk`, `GPUGreatCircleArcs` and `GPULonLatPolygonPreparation`): on a
line of 86,400 equal segments the last vertex measure was 16.432958602905273 off, which is bit for
bit what a Node replication gives with the correction term deleted, against 0.108 with it. The
shader compiler on this GPU reassociates floating-point expressions and folds the term to zero.

- Fix: `ADDITION_ROUNDING_ERROR_WGSL` reads the rounding error of an f32 addition off the integer
  significands. It now lives in `analysis-core/kernels/addition-rounding-error-wgsl.ts` (moved there
  and made signed by the lead); `line-segmentize/line-segmentize-kernels.ts:77` re-exports it, and
  the line prefix uses it at `:214` and `:290`, the resample arc length at
  `trajectory-interpolation/trajectory-interpolation-kernels.ts:558`.
  `gpu-terrain/terrain-processing/gpu-terrain-depression-hierarchy.ts:308` holds an older private
  function for the same purpose.
- Spec: `linear-referencing/gpu-linear-referencing.spec.ts:695` (largest error under 0.25 at 1.2
  million).
- Behaviour change: vertex measures of long lines are closer to the true length (16 units in 1.2
  million on the test line).
- Not done, outside this scope: every other float Kahan or Neumaier sum in the library is suspect
  on this GPU until measured. A search for the words finds them in `spatial-regression` (12 files),
  `geometry-measures`, `shape-descriptors`, `space-time-hawkes`, `change-points` and
  `gpu-network/map-matching`. None of these was run.

### Time precision spots listed under "Status of earlier findings"

| Spot | Outcome | Why |
| --- | --- | --- |
| Resample progress from the track start (`trajectory-interpolation-kernels.ts`) | deferred: documented in the class documentation of `GPUTrajectoryResample` | The sample target and `output.sampleTimes` are float32 by contract, so exact step differences only improve the interpolation fraction, not the instant sampled. A real fix carries the target as Int64 words |
| Zone event times from the track start (`trajectory-zones/zone-events-kernels.ts:484-486`) | deferred: documented on `events.eventTimes` | The output is a float32 time relative to the track start. `eventRows` and `eventPositions` keep full resolution |
| Closest-approach boxes from row 0 (`trajectory-encounters/closest-approach-kernels.ts:170-171`) | not reproduced as a defect: documented | The box is padded by `1e-6` of the time, about eight times the two float32 roundings, so rounding can add candidates but not drop a pair |

### From `luma-official.md`

| Item | Outcome | What changed |
| --- | --- | --- |
| luma-official-02, order of work 5, for `trajectory-encounters/encounter-kernels.ts:143` | fixed (pinned) | New `trajectory-encounters/encounter-grid-coordinate-pin.node.spec.ts` compares the local `getCoordinate` text with the published `GPUGridIndex` kernel text. No numeric change |

### Left alone, for the lead

- `ontology/spatiotemporal-review-catalog.ts:588` still says "recurse and Fauchald and Tveraa
  definitions" for `GPUTrajectoryRecursions`. It is a shared file and the wording feeds generated
  records, so it was not edited.
- `website/src/gpu/runs/entries/trajectory-mobility/trajectory-modal-split.ts` reads
  `predictedModes`, which can now hold `0xffffffff`; `trajectory-outlier-flags.ts` there drives
  `maximumAcceleration`, whose bit now means something else. Both are out of bounds for this pass.
- The audit cases of `GPUTrajectoryRestriction` and of the line family
  (`cases-coverage-network-movement.ts`, `cases-geometry.ts`) are listed in `KERNEL_AUDIT_CASES` but
  no spec file runs them. The 128-vertex restriction kernel was compiled and run on the real GPU
  instead.
- Rectangle restriction classifies a computed crossing point by comparing its rounded coordinates
  with the box, so the start-open and end-open bits at a box crossing depend on rounding. Not part
  of finding 09 and not changed.
- `linear-referencing-kernels.ts` calls `atan2(tangent.x, tangent.y)` for the spherical bearing; a
  tangent with a north component of negative zero is the case the `atan2` session is fixing
  elsewhere. Not touched.

### Upstream gaps

- luma.gl publishes no cell-coordinate WGSL snippet for `GPUGridIndex`; the encounter scan keeps a
  text copy, now pinned by a spec.
- luma.gl publishes no compensated or exact segmented prefix sum; a public one would need an error
  term that survives fast-math compilers.

### Review follow-up (2026-11-10)

From `fix-review-geometry.md` and the coordinator's follow-up. Each new spec was run against the
kernel before the change and failed there.

| Item | Outcome | What changed | Spec that pins it | Behaviour change a caller sees |
| --- | --- | --- | --- | --- |
| fr-geometry-09, float32 crossing detection in `GPUTrajectoryRestriction` | fixed. The suspicion is confirmed on the GPU: a segment that passes 1e-8 of an edge length outside a polygon vertex was reported as touching the closed polygon | `trajectory-restriction/trajectory-restriction-kernels.ts:234-323`. One job loop holds the only call of `exactOrientSigns` (`:276`). Edge jobs decide whether the segment meets each edge from the exact sides of the two fixes against the edge and of the edge's two ends against the segment (`:277-305`); float32 only places a crossing that exists, and a vertex on the segment is placed by its own projection so both edges that share it give one parameter. Probe jobs are the exact even-odd test (`:306-321`), with the rule of `exactProbeCrossing` written out, because calling that helper as well would be a second call site | `trajectory-restriction/trajectory-restriction-boundary.spec.ts:294` (two grazing fixtures, found by a search over float32 evaluation orders and checked in the spec with integer arithmetic) and `:320` (a segment through a vertex) | A segment that misses the polygon near a vertex no longer reports a touch (`ever` 1 before, 0 now, closed geometry). A segment through a vertex reports one instant, `[5, 5]` (before: a sliver `[4.9999995, 5]` from two unequal parameters) |
| fr-geometry-06, bare `atan2` with a second argument that can be -0.0 | fixed with `guardedAtan2` (contract section 11) in every heading and bearing of these directories | `trajectory-interpolation/trajectory-interpolation-kernels.ts:79` (planar) and `:104` (great-circle); `trajectory-analysis/trajectory-metrics-kernels.ts:426` (great-circle twin of the already guarded `:419`); `linear-referencing/gpu-line-locate.ts:356`; `linear-referencing/linear-referencing-kernels.ts:961` (bearing), `:345` (latitude peak of an arc for its bounds) and `:529` (projection angle) | New `trajectory-interpolation/trajectory-twin-heading-signed-zero.spec.ts`, four tests. Three failed before the change: playhead planar heading -pi/2 for a step due north, line-locate angle -90 for a line due north, and no match at all for a query at the high point of an arc that starts at latitude -0. The great-circle headings and the projection angle at `:529` did not fail on a built input and are guarded on the same rule | `GPUTrajectoryPlayhead` `headings` and `GPULineLocate` `angles`: +90 degrees for a step due north from x = +0 to x = -0 (before: -90). `GPULinearReferencing` with a great-circle context: an arc of more than 90 degrees that starts at latitude -0.0 is found near its highest latitude (before: its bounds stopped at the end points, so a query there had no match) |
| `ontology/spatiotemporal-review-catalog.ts:588` | fixed | The summary says "forward-only first passage time" and keeps Fauchald and Tveraa as the origin ("forward leg only") | `ontology/analysis-operator-catalog.node.spec.ts` still passes | none |
| NumPy second checks | added | New `test/oracles/trajectories-lines-numpy-checks.py` and `fixtures/trajectories-lines-numpy-checks.json` (NumPy 2.0.2, system Python 3.9.6), listed in `GENERATORS.md`. Arc-length resample: `cumsum` and `interp` over the float32 positions give sample rows `86400 k / 7`, the closed form of the spec. Volume isopleths: a stable `argsort` and `cumsum` give `0.5, 0.8, 0.95, 1, 1, 1` and the mask `1, 1, 1, 0, 0, 0` | `trajectory-interpolation/gpu-trajectory-resample.spec.ts` and `trajectory-home-range/home-range-recursion-mob07.spec.ts` compare the GPU with the fixture as well | none |
| fr-geometry-10, stale references in this Resolution | fixed | The three references to `line-segmentize-kernels.ts:74-125`, `:262` and `:338` above now name the shared file and the current lines | | |

Closing pass (2026-11-11): the NumPy second checks were rerun on the pinned set (NumPy 2.5.4), every value equal, fixture replaced; see [fixtures-pinned-rerun.md](./fixtures-pinned-rerun.md).

Not found: a segment that crosses an edge and was missed by the float32 test under every
evaluation order (the search of four million random grazing segments found none); the exact
decision covers it all the same. The rectangle variant is unchanged.
