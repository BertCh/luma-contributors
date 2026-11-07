# Pacific boundary logic implementation

Date: 2026-10-25. Scene: `earth/pacific-boundary-logic`.

## Investigation

The scene asks whether Pacific region boundaries need a longitude seam. Two synthetic analyst
windows, A (`[145,-15], [-150,-15], [-150,55], [145,55]`) and B
(`[170,-35], [-115,-35], [-115,30], [170,30]`), both cross ±180°. They are not tectonic,
warning, or hazard boundaries. Their unit-vector vertices are checked before upload: cross-product
orientation against the normalized vertex center is positive, each edge is a minor arc, all edges
support the same convex half-spaces, and every vertex is strictly on the positive side of the center
plane. No closing duplicate is supplied. Display edges are slerp-densified and longitudes are
unwrapped around 180° only for screen rendering.

## Evidence and methods

- The existing `poopdeck-earthquakes` bytes contain 77,231 rows; the scene selects all 77,215 rows
  with type 0 and converts their original float32 longitude/latitude values into float32 unit XYZ.
  Every selected row enters GPU containment. Every selected row is also drawn with a disclosed 1:1
  muted display mark; the accepted subset is brighter. Four exact source-corner probes per selected
  window are appended after the archive query rows. They expose boundary status but do not enter
  earthquake counts. The observation window ends on 2024-12-30; this is not a claim of complete
  global ComCat coverage.
- `GPUSphericalConvexContains` runs for A and B, open and closed boundary rules, and ordinary and
  complementary regions. Readouts count only archived rows by returned `contains` and `status`.
  Status 2 marks the tolerance band; the exact-vertex probe count is reported separately. The
  tolerance is `1e-6` in unit-vector coordinates.
- `GPUSphericalConvexIntersection` computes paired A/B results at capacities 3 and 16 with
  tolerance `1e-6` and optional four-value measures. A successful result reports steradians,
  intersection area using IUGG mean radius 6,371,008.8 m, source and target fractions, and union
  area. Any nonzero status withholds both geometry and measures; capacity-3 reports written and
  required counts. A zero-vertex status-0 result is a valid empty/contact case, never an invalid
  result or an assumed zero substituted for unavailable area.
- `GPUSphericalArcOverlay` executes intersection, union, difference and symmetric difference for
  ordinary A and complementary A. Both operands use the original four-vertex input rings, with
  only reference membership changing. The anchor `[1,0,0]` is 0°E, 0°N, away from both boundaries;
  its operand and result membership are stated in the readout. Output capacity is 64 vertices and
  65 ring offsets. The selected map outline is built from GPU readback rings, then slerp-densified
  solely for rendering. Archive membership displayed in this mode is a CPU Boolean combination of
  the two GPU containment arrays, explicitly separate from arc-overlay point classification.

Natural Earth 1:110m coastlines and a graticule orient the abyss-style Pacific map. Window edges
remain cyan and coral, archived events stay muted, accepted events are vivid, and computed output
geometry has a separate vivid color. There are six authored chapters with substantive text
alternatives and live evidence tokens. The scene offers method, window, boundary, complement,
operation and bounded/full capacity controls.

## Contributor contract boundaries

The three contributors answer different questions. Convex containment uses oriented great-circle
half-spaces for a convex open-hemisphere polygon. Convex intersection uses oriented 3D planes and
requires each input to satisfy its convex open-hemisphere contract; it does not handle a
complementary or more-than-hemisphere region. Arc overlay uses spherical arc arrangement and
reference-anchored parity, including complementary regions larger than a hemisphere. This scene
makes no claim of algorithmic parity with S2 or any other geometry implementation. Numerical
status codes and output counts remain visible; nonzero results are not rendered as complete
geometry.

## Research references

- [S2 basic types: points, edges, loops, and regions](https://s2geometry.io/devguide/basic_types.html)
  describes spherical points as unit vectors and loops as oriented boundary edges whose inside is
  defined by a region convention. It motivates the unit-vector representation and the need to state
  which side of an unchanged boundary is the region.
- [S2 overview](https://s2geometry.io/about/overview.html) explains S2's sphere-based geographic
  model. It supports the spherical framing, not a claim that luma's contributors use S2 algorithms.
- [USGS ComCat documentation](https://earthquake.usgs.gov/data/comcat/) describes the ComCat
  earthquake catalog and its data sources. The scene reuses the bundled exported snapshot and does
  not assert that its M4+ subset is a complete catalog.

## Verification scope and limitations

Direct Biome formatting, owned-file TypeScript compilation, and scoped showcase story, cartography,
and WGSL static checks passed. The wider showcase TypeScript command still reports errors in the
workspace's `cells/cell-neighborhoods.compute.ts` and `raster/grid-alignment-frame.ts`, outside this
file list. No GPU or browser execution was performed. Consequently, rendered geometry, GPU
status/count values, and steradian measures remain runtime evidence to confirm in the coordinated
WebGPU pass. The displayed map projection can distort shape and scale; all contributor inputs and
spherical measures use unit XYZ sphere geometry. Coastline lines are cartographic context, not
operand geometry. Source dataset metadata and the existing `spherical-boundaries` story were not
edited.
