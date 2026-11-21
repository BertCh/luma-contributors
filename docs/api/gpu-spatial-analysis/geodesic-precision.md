# Geodesic precision (CR01)

Import from `@luma-contributors/analysis/gpu-spatial-analysis`.

See also the [global coordinate guide](global-coordinates.md).

## GPUGeodesicPairs

One invocation measures each packed float32 longitude/latitude pair. Context, output
columns and `maximumIterations` are compile-time; input coordinates are per-frame.
Requesting every optional column uses two bounded dispatches, keeping each within
WebGPU's portable eight-storage-binding budget without readback or scratch.
Construction rejects overlapping output byte ranges across the entire output set,
including columns written by different dispatches. Disjoint ranges in one buffer
remain accepted across separate dispatches; outputs co-bound within one dispatch must also
have disjoint aligned storage-binding ranges. Output/input buffer separation is checked.
The default iteration limit is 16, with accepted integer limits of 1–200. The named
maximum bounds per-invocation work, not the number of pairs.

Request `output.status.converged`, `output.status.approximation` and/or
`output.status.iterationLimitReached` as **per-pair uint32 columns**, not aggregate
scalars. Their meanings are:

- `converged`: ellipsoidal inverse resolved outside the antipodal exclusion. This
  port is accepted only for the ellipsoidal metric.
- `approximation`: ellipsoidal inverse fell back to the context sphere, or spherical
  bearing/midpoint conditioning is outside the measured envelope (within 0.02
  radians, about 1.146 degrees, of an antipode). A numerically stopped float32
  iteration inside that region is also excluded. Rhumb rows report zero.
- `iterationLimitReached`: inverse exhausted its iteration budget. A singular row
  rejected before iteration reports zero. A conditioning exclusion can therefore
  have `converged=0`, `approximation=1`, `iterationLimitReached=0`.

The 0.02-radian exclusion measures the small angular gap **to the antipode directly**
from the stored coordinates on both routes; it does not subtract a rounded central angle
from pi. For `(0°,0°) → (180°,latitude)`, the independent cutoff oracle is
`abs(latitude) * pi / 180`. Stored latitudes `1.1459144353866577°` and
`1.1459163427352905°` give gaps `0.019999979843629` and `0.020000013133143`
radians. At budget 32 their status triples are `[0,1,0]` and `[1,0,0]`, respectively.
Their float32-published CPU distances are `19887694 m` (sphere fallback) and
`19877222 m` (ellipsoid); GPU distance assertions allow the measured 12 m envelope.
The longitude supplement is formed from canonical stored endpoints before a near-half-turn
subtraction can round it. For `(1.1459152698516846°,0°) → (180°,0°)`, the equatorial gap
is `0.019999994407791207` radians: budget 32 reports `[0,1,0]` and sphere distance
`19887694 m` on both routes. Integer-grid subtraction for large longitude endpoints prevents
backend reassociation from recreating the cancelled supplement.
This fixes model selection and changes distances for previously misclassified boundary pairs.

Fallback rows publish sphere distance and bearings; these **do not satisfy the
ellipsoidal error envelope**. Use `CPUGeodesicPairsOperation` or the pure
`computeGeodesicPairsOnCPU` for a float64 exact ellipsoidal inverse at antipodes.
The twin has the same output columns. Without an explicit `maximumIterations`, its exact inverse
has no iteration budget, so `iterationLimitReached` is zero and valid ellipsoidal rows converge.
An explicit budget (1–200) bounds auxiliary-longitude convergence using float32 updates,
including reduced-latitude arithmetic and the GPU stopping criterion, before the exact solve.
This matters at the stopping boundary: `(0,20)` to `(170,60)` falls back at budget two and
converges at three; the sphere and ellipsoid distances are 11,073,359.15 m and 11,091,265.03 m.
Exhaustion publishes the context-sphere distance and bearings
with `[converged, approximation, iterationLimitReached] = [0,1,1]`. The pure helper accepts this
budget as its third argument. Antipodal conditioning can trigger fallback before exhaustion.
The twin also marks spherical midpoint/bearing conditioning through `approximation`.

For a one-iteration ellipsoidal inverse from `(0,0)` to `(1,0)`, the first auxiliary-longitude
change is `f * pi / 180`, about `5.85e-5` radians. Work remains at that budget, so the status
triple is `[0,1,1]`. Coincident points instead give `[1,0,0]`; equatorial antipodes give
`[0,1,0]` because singularity is detected before iteration. These distinctions apply to full
status words, including zero values, on every execution.
The stopping test scales with the iterated quantity. For a `0.01°` equatorial inverse,
the first change is about `5.85176e-7` radians, above the relative threshold `8.40567e-11`.
A 2,000 m northward direct trip from `(0,0)` changes by about `5.27660e-7`, above its
`2.4e-7` threshold. Both report `[0,1,1]` after one iteration even though their changes
are below an absolute `1e-6`.
At `0.001°`, the inverse change is about `5.85176e-8`, still above its relative threshold
`8.40567e-12`; retaining the coefficient `4.8e-7` as an absolute tolerance would stop too early.
For a 19,500,000 m direct trip from `(0,0)` at bearing `85.75°`, float32 sigma changes from
`3.0675757` to `3.067575`: change `7.15256e-7` is within the scaled threshold `7.36218e-7`,
so one iteration reports `[1,0,0]`. This accepts a larger change than the unfinished 2,000 m
trip, which requires a scale-dependent stopping decision.

Midpoints are spherical even in ellipsoidal mode. Within 1e-6 radians of an
antipode the great-circle path is nonunique or too ill-conditioned: both midpoint
components are NaN on GPU and CPU, with `approximation=1`. At a coincident pole,
longitude zero is the midpoint convention; distance and initial bearing are zero.
Coincident ellipsoidal pairs also have final bearing zero. Spherical coincident
pairs retain final bearing 180 degrees. Pole longitude does not distinguish positions.

## GPUGeodesicDestination

The same compile-time/per-frame distinction and iteration limits apply. Distance
and initial bearing are per-frame columns. `status.converged` (ellipsoidal only)
and `status.iterationLimitReached` are per-row uint32 columns. The optional
`status.approximation` is one for an unconverged result or a nonzero-distance
arrival with `abs(cos(latitude)) <= 0.03` (about 1.72 degrees from a pole).
This conditioning guard reserves a normalized position-error budget of 2e-6
(12.8 meters at Earth's radius): division by 0.03 stays below 0.004 degrees.
Closer to a pole the arrival azimuth is excluded even if iteration converges;
small phase errors amplify into large bearing errors well before the terminal
latitude rounds to 90 degrees. This guard is a conservative policy for the
measured envelope, not an all-input error proof. Position bounds still apply when the iteration converges. Direct results are
published even when their iteration budget is exhausted; only converged results
are within the measured envelope. Zero-distance destinations preserve the origin.
The sphere retains its existing 180-degree arrival-bearing convention for zero
distance; the ellipsoidal model retains the supplied bearing modulo a turn.

The sphere direct solution keeps the established half-angle change for ordinary
paths, using horizontal and vertical components at polar origins or arrivals and
for angular distances of at least pi/4. The polar component region starts at
80 degrees latitude at either endpoint. Long paths otherwise cancel in both the
latitude reconstruction and the longitude denominator. Both direct models derive nonzero-distance arrival azimuths
before rounding destination coordinates. The inverse keeps colatitude-plus-gap
arithmetic except within 0.001 degrees of a pole, where it normalizes from the exact
colatitude. Converting a pole to rounded radians first loses relative precision. Zero-axis `atan2` uses the shared guarded helper under the
geodesic prefix, including negative zero.

## Measurements and acceptance bounds

These are finite deterministic regression envelopes, **not an all-input proof**.
Inputs are exactly float32-representable, canonical longitudes in [-180, 180] and
latitudes in [-90, 90], with WGS84 or a sphere of radius 6371008.8 meters. There are
528 pairs per model (poles, antipodes, seam identities, seeded global samples and
samples across the exclusion boundary), 208 WGS84 direct cases and 398 sphere
direct cases (0 through 20,000,000 meters, including exact poles, near-terminal
poles and the independent review's long-path and near-meridian neighborhoods). Bearings compare
modulo 360; midpoint/destination errors measure physical position, since longitude
at a pole is not unique. Coincident inverse bearings use the conventions above.

Regenerate with `/usr/bin/python3 modules/analysis/test/oracles/cr01-geodesic-envelopes.py`.
The committed fixture records pyproj 3.6.1 / PROJ 9.3.0 `Geod.inv` and `Geod.fwd`
results, rounding **inputs before** calling PROJ. Oracle data never comes from the
CPU twin or a Vincenty test oracle. The script separately measures a nominal IEEE
float32 transcription with NumPy 2.0.2. Its results are recorded under `measurement`
in the fixture; that transcription is **not a GPU run**.

| Quantity | Nominal float32 maximum | Headless acceptance bound |
| --- | ---: | ---: |
| Resolved WGS84 inverse distance | 4.885 m | 12 m |
| Resolved WGS84 initial / final azimuth | 0.001100 / 0.001048 degrees | 0.004 degrees |
| Sphere distance, including antipodes | 3.062 m | 12 m |
| Resolved sphere initial azimuth | 0.000276 degrees | 0.004 degrees |
| Resolved spherical midpoint | 51.010 m | 128 m |
| WGS84 direct position | 2.101 m | 12 m |
| Sphere direct position | 1.072 m | 12 m |
| Resolved WGS84 / sphere direct azimuth | 0.000013 / 0.000062 degrees | 0.004 degrees |

345 of the 528 nominal WGS84 inverse rows resolve and 183 are explicit fallbacks.
The polar 0.852-meter path that previously had about 0.0154 degrees azimuth error
is included. The headless spec also checks direct arrival azimuths to 0.004 degrees,
iteration budgets, CPU/GPU operation ports, exact pole identities and untouched
padding rows. The unfiltered nominal direct azimuth error reaches 180 degrees on
WGS84 and 64.262 degrees on the sphere at terminal poles; these rows must set
`approximation=1` and are excluded from the azimuth envelope. Hardware measurements
are still required.

**GPU certification pending:** Chromium cannot run in this staging sandbox.
`cr01-geodesic-envelopes.spec.ts` must pass on the landing GPU before these bounds
can be called hardware-certified. Fast math and GPU transcendental accuracy are
not covered by the NumPy measurements. The existing safeguarded geometry-measures
inverse/area regressions must also run because they share the corrected kernels.

Kilometer outputs scale meter bounds by 1/1000. Arbitrary radii, custom ellipsoids,
unbounded longitudes, nonfinite coordinates, out-of-range latitudes, rhumb paths
and direct distances beyond 20,000,000 meters have no new certified envelope here.
No new public contributor class is introduced.


## Compatibility regressions from the lead GPU run

The lead's first run passed 182 headless tests and failed five existing cases.
The follow-up restores latitude-difference arithmetic for same-hemisphere polar
pairs, short-move half-angle arithmetic for ordinary sphere destinations, ordinary
reduced-latitude cosine arithmetic outside the millidegree polar region, and the
sphere's zero-distance arrival convention. The existing distance and bearing
acceptance tolerances are unchanged. A focused headless case repeats the reported
polar pair, 250 km destination, zero-distance bearing, New York–London length and
city area, including both serial and cooperative geometry paths.

The city polygon's old area oracle used decimal latitudes before float32 rounding:
463546956.6654968 square meters, versus PROJ's 463539890.41503906 square meters for
the actual float32 inputs. The test now uses the latter independent reference,
with its original 5e-6 relative tolerance. Short-edge area samples also retain
unrounded local longitude and sine-latitude changes, so a globally rounded
midpoint cannot quantize away a sub-meter bulge. The local latitude inversion is
limited to auxiliary-sphere angles below 0.01 radians, origin latitudes below
85 degrees and flattening at most 0.005; other paths retain the general solution.
For the city case the nominal float32 area is 463539872 square meters (relative
error 3.973e-8 against PROJ). This is an arithmetic-model measurement; the updated
headless suite still requires the lead's rerun.


## Independent review regressions

All three review findings are confirmed. The WGS84 `[0, 0]`, f32 `0.0001`-degree
bearing, 10,001,976-meter path has PROJ arrival bearing 132.695609 degrees, versus
130.186707 degrees in the nominal float32 model. Its position error is only
0.925 meters, but the old 0.0001-degree pole cutoff incorrectly certified its
azimuth. The conditioning guard now requires `approximation=1`. The fixture adds
54 nearby WGS84 paths, with PROJ determining which arrivals must be flagged.

The sphere `[0, 88]`, 135-degree, 19,900,000-meter path previously had 50.129 meters
of nominal position error. The long-path component branch reduces that error to
0.106 meters. The fixture adds 244 north/south, east/west and distance variants,
including the component-region boundaries and a polar arrival from 44 degrees.
They retain the 12-meter position bound; the original short-path regressions remain.
A separate nominal float32 sweep of 34,368 sphere paths (integer latitudes from
-89 to 89, bearings every 15 degrees, and distances 1,000, 100,000, 1,000,000,
5,000,000, 10,000,000, 15,000,000, 19,900,000 and 20,000,000 meters) measured a
maximum 3.574-meter position error against PROJ, with no rows exceeding 12 meters.
That sweep measures NumPy arithmetic, not the GPU.

The CPU twin now returns exact longitude zero when a midpoint's horizontal vector
is zero, including negative signed-zero components from negative pole longitudes.
A Node regression failed before the fix; a headless operation regression compares
both models and both execution paths at negative-longitude north/south poles.
The two direct review cases and all neighbors are in the headless PROJ fixture
checks, with padding and convergence/status assertions retained.


## Interpolation endpoints and output aliases

The antipodal interpolation guard applies only to interior fractions. Fractions
zero and one preserve the known input endpoint, retaining continuous longitude
across the antimeridian. This keeps `GPULineChunk` equal-measure substrings at the
start or end of a nearly antipodal line finite: `[0, 0]` measures on
`[0, 0] -> [179.99998474121094, 0]` emit two `[0, 0]` vertices. Interior fractions
inside the existing 1e-6-radian antipodal exclusion still return NaNs.
Headless regressions cover the actual substring, both fractions, exact antipodes,
interior rejection, antimeridian continuity and untouched dispatch padding.

Exact longitude half-turn interpolation aligns interior longitude with the known unwrapped
endpoint branch across a pole; float32 sin(pi) must not select an opposite-sign meridian and
create a 360-degree final jump. The arcs and segmentize polar-branch tests require the lead's
GPU rerun.

The split-dispatch alias regression requests all seven geodesic-pair outputs and
places distinct `converged` and `iterationLimitReached` views over the same bytes.
Construction now rejects both exact and partial overlaps before any dispatch is
created. Its Node regressions failed before the fix and pass afterward. Adjacent
ranges remain accepted, and the existing positive pair-count limit is retained.

## Planar measurement units

`GPUGeometryMeasures` rejects known angular planar frames when `lengths` or `areas`
are requested, including group columns, `metric: 'none'`, and omitted result units.
Project positions into a planar linear frame before requesting those measurements.
The check resolves both explicit angular frame units and recognized geographic CRS identifiers.
Coordinate-only columns such as bounds remain accepted with `metric: 'none'`;
unlabelled XY remains ambiguous, and valid linear frames retain their existing units.
