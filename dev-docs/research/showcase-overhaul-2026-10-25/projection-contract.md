# GPUSinusoidalProjection scene contract

The `sinusoidal-projection` scene is a contract inspection of the public
`GPUSinusoidalProjection`, not a map-projection recommendation or a route-distance analysis.

## Sources and conventions

- The contributor implementation is [`gpu-sinusoidal-projection.ts`](../../../modules/graph/src/geospatial/gpu-sinusoidal-projection.ts). It specifies cuSpatial-style `origin - point` signs, midpoint-latitude cosine on x, and an optional longitude-delta wrap into `[-180°, 180°)`.
- RAPIDS' upstream projection implementation spells out the `40,000 km` circumference convention and coordinate equations: [cuSpatial sinusoidal projection source](https://raw.githubusercontent.com/rapidsai/cuspatial/branch-25.04/cpp/include/cuspatial/detail/projection/sinusoidal_projection.cuh).
- RAPIDS documents the limitation that this projection is approximate near its origin and that error increases farther away: [cuSpatial spatial API notes](https://docs.rapids.ai/api/libcuspatial/stable/group__spatial__join).
- The graph's constant is `111.111 km / degree` (`40,000 / 360`). Other spatial-analysis operators in this repository use the mean-Earth radius, approximately `111.19508 km / degree`; the per-degree convention differs by approximately `0.0755%`. That comparison isolates the sphere constant, and is not a bound on the full projection's error.
- Input coordinates come from the bundled frozen OpenFlights `locations` column (3,257 rows). The airport CSV row indices align with that array; selected origins are ORD 2033, HND 1071, SYD 2589, with LAX 1501 as the fixed probe.

## Observable behavior

Each hub variant compiles two actual projection contributors: the default unwrapped path and the wrapped path. GPU kernels split their interleaved xy outputs into scalar columns, `GPUColumnProfile` computes the four output extents/counts, and a bounded GPU gather reads both coordinate pairs for the hub and LAX. The point layer binds one of those scalar GPU outputs for color while preserving original geographic positions. Thus changing wrap changes color/readouts but never moves markers on the map. The selected origin and fixed LAX probe are independently marked.

The CPU reference is maintained separately in [`projection-contract-reference.py`](./projection-contract-reference.py) and its output [`projection-contract-reference.json`](./projection-contract-reference.json). It records scalar formula values using the same float32 inputs. At LAX, the reference x coordinate is about 23,571.784 km raw and -9,295.128 km wrapped from HND, and 29,953.889 km raw and -10,046.111 km wrapped from SYD. ORD's LAX x remains about 2,672.197 km under both settings. Use a tolerance around 0.02 km when comparing GPU browser readbacks to the reference; small trigonometric/backend differences are expected.

## Interpretation limits

The output is a local coordinate approximation, not global metric distance, not an equal-area result, and not evidence of flight activity or airport density. The map encodes a coordinate value on source locations. OpenFlights is a frozen community snapshot, not a current network. The raw and wrapped modes are distinct longitude conventions; the no-wrap default intentionally matches cuSpatial. Do not infer that the wrapped result is universally preferable: it changes which branch of longitude the formula uses.

## Wave 11 numerical acceptance

The production harness passed all 12 combinations of three origins, raw/wrapped longitude,
and x/y display. Each verifies 3,257 finite GPU rows, min/max extents, zero origin coordinates,
and all raw/wrapped LAX coordinates against the independent formula within 0.02 km. Report:
`/tmp/luma-showcase-overhaul-20261025/projection-wave11.json`; no page or console errors.
The saved screenshot was reviewed. Its initial world framing cropped the far reference marker,
so the three step cameras and initial view were widened to zoom 0.55. This camera-only change
passed scoped Biome; corrected rendering awaits the next bundle. Numerical acceptance does not
certify this formula as a global distance or equal-area projection.
