var e={slug:`api/gpu-spatial-analysis/geodesic-precision`,title:`Geodesic precision (CR01)`,html:`<h1 id="geodesic-precision-cr01">Geodesic precision (CR01)</h1>
<p>See also the <a href="/luma-contributors/docs/api/gpu-spatial-analysis/global-coordinates">global coordinate guide</a>.</p>
<h2 id="gpugeodesicpairs">GPUGeodesicPairs</h2>
<p>One invocation measures each packed float32 longitude/latitude pair. Context, output
columns and <code>maximumIterations</code> are compile-time; input coordinates are per-frame.
Requesting every optional column uses two bounded dispatches, keeping each within
WebGPU's portable eight-storage-binding budget without readback or scratch.
Construction rejects overlapping output byte ranges across the entire output set,
including columns written by different dispatches. Disjoint ranges in one buffer
remain accepted; output/input buffer separation is also checked.
The default iteration limit is 16, with accepted integer limits of 1–200. The named
maximum bounds per-invocation work, not the number of pairs.</p>
<p>Request <code>output.status.converged</code>, <code>output.status.approximation</code> and/or
<code>output.status.iterationLimitReached</code> as <strong>per-pair uint32 columns</strong>, not aggregate
scalars. Their meanings are:</p>
<ul>
<li><code>converged</code>: ellipsoidal inverse resolved outside the antipodal exclusion. This
port is accepted only for the ellipsoidal metric.</li>
<li><code>approximation</code>: ellipsoidal inverse fell back to the context sphere, or spherical
bearing/midpoint conditioning is outside the measured envelope (within 0.02
radians, about 1.146 degrees, of an antipode). A numerically stopped float32
iteration inside that region is also excluded. Rhumb rows report zero.</li>
<li><code>iterationLimitReached</code>: inverse exhausted its iteration budget. A singular row
rejected before iteration reports zero. A conditioning exclusion can therefore
have <code>converged=0</code>, <code>approximation=1</code>, <code>iterationLimitReached=0</code>.</li>
</ul>
<p>Fallback rows publish sphere distance and bearings; these <strong>do not satisfy the
ellipsoidal error envelope</strong>. Use <code>CPUGeodesicPairsOperation</code> or the pure
<code>computeGeodesicPairsOnCPU</code> for a float64 exact ellipsoidal inverse at antipodes.
The twin has the same output columns. Its exact inverse has no iteration budget,
so <code>iterationLimitReached</code> is always zero and valid ellipsoidal rows converge.
It still marks spherical midpoint/bearing conditioning through <code>approximation</code>.</p>
<p>Midpoints are spherical even in ellipsoidal mode. Within 1e-6 radians of an
antipode the great-circle path is nonunique or too ill-conditioned: both midpoint
components are NaN on GPU and CPU, with <code>approximation=1</code>. At a coincident pole,
longitude zero is the midpoint convention; distance and initial bearing are zero.
Coincident ellipsoidal pairs also have final bearing zero. Spherical coincident
pairs retain final bearing 180 degrees. Pole longitude does not distinguish positions.</p>
<h2 id="gpugeodesicdestination">GPUGeodesicDestination</h2>
<p>The same compile-time/per-frame distinction and iteration limits apply. Distance
and initial bearing are per-frame columns. <code>status.converged</code> (ellipsoidal only)
and <code>status.iterationLimitReached</code> are per-row uint32 columns. The optional
<code>status.approximation</code> is one for an unconverged result or a nonzero-distance
arrival with <code>abs(cos(latitude)) &#x3C;= 0.03</code> (about 1.72 degrees from a pole).
This conditioning guard reserves a normalized position-error budget of 2e-6
(12.8 meters at Earth's radius): division by 0.03 stays below 0.004 degrees.
Closer to a pole the arrival azimuth is excluded even if iteration converges;
small phase errors amplify into large bearing errors well before the terminal
latitude rounds to 90 degrees. This guard is a conservative policy for the
measured envelope, not an all-input error proof. Position bounds still apply when the iteration converges. Direct results are
published even when their iteration budget is exhausted; only converged results
are within the measured envelope. Zero-distance destinations preserve the origin.
The sphere retains its existing 180-degree arrival-bearing convention for zero
distance; the ellipsoidal model retains the supplied bearing modulo a turn.</p>
<p>The sphere direct solution keeps the established half-angle change for ordinary
paths, using horizontal and vertical components at polar origins or arrivals and
for angular distances of at least pi/4. The polar component region starts at
80 degrees latitude at either endpoint. Long paths otherwise cancel in both the
latitude reconstruction and the longitude denominator. Both direct models derive nonzero-distance arrival azimuths
before rounding destination coordinates. The inverse keeps colatitude-plus-gap
arithmetic except within 0.001 degrees of a pole, where it normalizes from the exact
colatitude. Converting a pole to rounded radians first loses relative precision. Zero-axis <code>atan2</code> uses the shared guarded helper under the
geodesic prefix, including negative zero.</p>
<h2 id="measurements-and-acceptance-bounds">Measurements and acceptance bounds</h2>
<p>These are finite deterministic regression envelopes, <strong>not an all-input proof</strong>.
Inputs are exactly float32-representable, canonical longitudes in [-180, 180] and
latitudes in [-90, 90], with WGS84 or a sphere of radius 6371008.8 meters. There are
528 pairs per model (poles, antipodes, seam identities, seeded global samples and
samples across the exclusion boundary), 208 WGS84 direct cases and 398 sphere
direct cases (0 through 20,000,000 meters, including exact poles, near-terminal
poles and the independent review's long-path and near-meridian neighborhoods). Bearings compare
modulo 360; midpoint/destination errors measure physical position, since longitude
at a pole is not unique. Coincident inverse bearings use the conventions above.</p>
<p>Regenerate with <code>/usr/bin/python3 modules/analysis/test/oracles/cr01-geodesic-envelopes.py</code>.
The committed fixture records pyproj 3.6.1 / PROJ 9.3.0 <code>Geod.inv</code> and <code>Geod.fwd</code>
results, rounding <strong>inputs before</strong> calling PROJ. Oracle data never comes from the
CPU twin or a Vincenty test oracle. The script separately measures a nominal IEEE
float32 transcription with NumPy 2.0.2. Its results are recorded under <code>measurement</code>
in the fixture; that transcription is <strong>not a GPU run</strong>.</p>
<table>
<thead>
<tr>
<th>Quantity</th>
<th align="right">Nominal float32 maximum</th>
<th align="right">Headless acceptance bound</th>
</tr>
</thead>
<tbody>
<tr>
<td>Resolved WGS84 inverse distance</td>
<td align="right">4.885 m</td>
<td align="right">12 m</td>
</tr>
<tr>
<td>Resolved WGS84 initial / final azimuth</td>
<td align="right">0.001100 / 0.001048 degrees</td>
<td align="right">0.004 degrees</td>
</tr>
<tr>
<td>Sphere distance, including antipodes</td>
<td align="right">3.062 m</td>
<td align="right">12 m</td>
</tr>
<tr>
<td>Resolved sphere initial azimuth</td>
<td align="right">0.000276 degrees</td>
<td align="right">0.004 degrees</td>
</tr>
<tr>
<td>Resolved spherical midpoint</td>
<td align="right">51.010 m</td>
<td align="right">128 m</td>
</tr>
<tr>
<td>WGS84 direct position</td>
<td align="right">2.101 m</td>
<td align="right">12 m</td>
</tr>
<tr>
<td>Sphere direct position</td>
<td align="right">1.072 m</td>
<td align="right">12 m</td>
</tr>
<tr>
<td>Resolved WGS84 / sphere direct azimuth</td>
<td align="right">0.000013 / 0.000062 degrees</td>
<td align="right">0.004 degrees</td>
</tr>
</tbody>
</table>
<p>345 of the 528 nominal WGS84 inverse rows resolve and 183 are explicit fallbacks.
The polar 0.852-meter path that previously had about 0.0154 degrees azimuth error
is included. The headless spec also checks direct arrival azimuths to 0.004 degrees,
iteration budgets, CPU/GPU operation ports, exact pole identities and untouched
padding rows. The unfiltered nominal direct azimuth error reaches 180 degrees on
WGS84 and 64.262 degrees on the sphere at terminal poles; these rows must set
<code>approximation=1</code> and are excluded from the azimuth envelope. Hardware measurements
are still required.</p>
<p><strong>GPU certification pending:</strong> Chromium cannot run in this staging sandbox.
<code>cr01-geodesic-envelopes.spec.ts</code> must pass on the landing GPU before these bounds
can be called hardware-certified. Fast math and GPU transcendental accuracy are
not covered by the NumPy measurements. The existing safeguarded geometry-measures
inverse/area regressions must also run because they share the corrected kernels.</p>
<p>Kilometer outputs scale meter bounds by 1/1000. Arbitrary radii, custom ellipsoids,
unbounded longitudes, nonfinite coordinates, out-of-range latitudes, rhumb paths
and direct distances beyond 20,000,000 meters have no new certified envelope here.
No new public contributor class is introduced.</p>
<h2 id="compatibility-regressions-from-the-lead-gpu-run">Compatibility regressions from the lead GPU run</h2>
<p>The lead's first run passed 182 headless tests and failed five existing cases.
The follow-up restores latitude-difference arithmetic for same-hemisphere polar
pairs, short-move half-angle arithmetic for ordinary sphere destinations, ordinary
reduced-latitude cosine arithmetic outside the millidegree polar region, and the
sphere's zero-distance arrival convention. The existing distance and bearing
acceptance tolerances are unchanged. A focused headless case repeats the reported
polar pair, 250 km destination, zero-distance bearing, New York–London length and
city area, including both serial and cooperative geometry paths.</p>
<p>The city polygon's old area oracle used decimal latitudes before float32 rounding:
463546956.6654968 square meters, versus PROJ's 463539890.41503906 square meters for
the actual float32 inputs. The test now uses the latter independent reference,
with its original 5e-6 relative tolerance. Short-edge area samples also retain
unrounded local longitude and sine-latitude changes, so a globally rounded
midpoint cannot quantize away a sub-meter bulge. The local latitude inversion is
limited to auxiliary-sphere angles below 0.01 radians, origin latitudes below
85 degrees and flattening at most 0.005; other paths retain the general solution.
For the city case the nominal float32 area is 463539872 square meters (relative
error 3.973e-8 against PROJ). This is an arithmetic-model measurement; the updated
headless suite still requires the lead's rerun.</p>
<h2 id="independent-review-regressions">Independent review regressions</h2>
<p>All three review findings are confirmed. The WGS84 <code>[0, 0]</code>, f32 <code>0.0001</code>-degree
bearing, 10,001,976-meter path has PROJ arrival bearing 132.695609 degrees, versus
130.186707 degrees in the nominal float32 model. Its position error is only
0.925 meters, but the old 0.0001-degree pole cutoff incorrectly certified its
azimuth. The conditioning guard now requires <code>approximation=1</code>. The fixture adds
54 nearby WGS84 paths, with PROJ determining which arrivals must be flagged.</p>
<p>The sphere <code>[0, 88]</code>, 135-degree, 19,900,000-meter path previously had 50.129 meters
of nominal position error. The long-path component branch reduces that error to
0.106 meters. The fixture adds 244 north/south, east/west and distance variants,
including the component-region boundaries and a polar arrival from 44 degrees.
They retain the 12-meter position bound; the original short-path regressions remain.
A separate nominal float32 sweep of 34,368 sphere paths (integer latitudes from
-89 to 89, bearings every 15 degrees, and distances 1,000, 100,000, 1,000,000,
5,000,000, 10,000,000, 15,000,000, 19,900,000 and 20,000,000 meters) measured a
maximum 3.574-meter position error against PROJ, with no rows exceeding 12 meters.
That sweep measures NumPy arithmetic, not the GPU.</p>
<p>The CPU twin now returns exact longitude zero when a midpoint's horizontal vector
is zero, including negative signed-zero components from negative pole longitudes.
A Node regression failed before the fix; a headless operation regression compares
both models and both execution paths at negative-longitude north/south poles.
The two direct review cases and all neighbors are in the headless PROJ fixture
checks, with padding and convergence/status assertions retained.</p>
<h2 id="interpolation-endpoints-and-output-aliases">Interpolation endpoints and output aliases</h2>
<p>The antipodal interpolation guard applies only to interior fractions. Fractions
zero and one preserve the known input endpoint, retaining continuous longitude
across the antimeridian. This keeps <code>GPULineChunk</code> equal-measure substrings at the
start or end of a nearly antipodal line finite: <code>[0, 0]</code> measures on
<code>[0, 0] -> [179.99998474121094, 0]</code> emit two <code>[0, 0]</code> vertices. Interior fractions
inside the existing 1e-6-radian antipodal exclusion still return NaNs.
Headless regressions cover the actual substring, both fractions, exact antipodes,
interior rejection, antimeridian continuity and untouched dispatch padding.</p>
<p>The split-dispatch alias regression requests all seven geodesic-pair outputs and
places distinct <code>converged</code> and <code>iterationLimitReached</code> views over the same bytes.
Construction now rejects both exact and partial overlaps before any dispatch is
created. Its Node regressions failed before the fix and pass afterward. Adjacent
ranges remain accepted, and the existing positive pair-count limit is retained.</p>`};export{e as default};