# Cell topology and route-editing investigations

Research/design record, 2026-10-25. Authorized follow-on work for the existing Astra planner and
Luna implementer after the repeated-survey studies. Parent coordinates production builds and
GPU access. New files are reserved under `cells/cell-neighborhoods.*` and
`geometry/route-editing.*`; existing shared source datasets remain read-only.

## Questions and evidence

The cell investigation asks whether an index path can stand in for a real journey. Real source
locations provide geographic anchors; the computation concerns the grid, not transport service.
Cell identities, directed edges and shared vertices describe topology. Physical cell area and
perimeter vary geographically. A minimum-hop path has a discrete meaning that differs from a
measured distance or observed route. Endpoint choices and resolution must make these differences
visible, with invalid and truncated results explicitly separated from a genuine empty answer.

The route investigation asks which edits preserve a route and which change its meaning. Use
an identified original CTA shape and an explicitly described serialization or displacement
scenario. Reversing stored vertex order can preserve geometric coverage while changing direction;
removing repeats differs from moving coordinates. Snapping to vertices differs from snapping to
segments. Precision reduction, loop erasure and polygon clipping can remove real detail. Never
claim the published source is corrupt merely because the investigation creates a stress case.

## Primary-source reading and consequences

- [H3 grid traversal](https://h3geo.org/docs/api/traversal/) defines distance in adjacent-cell hops.
  It documents failure for incompatible resolutions, distant cells and pentagonal distortion.
  Path ordering can change between versions; check length, adjacency and endpoints rather than
  demanding one particular tied path. A failed grid calculation says nothing about whether a
  real journey is possible.
- [H3 vertex identities](https://h3geo.org/docs/api/vertex/) distinguishes five topological vertices
  for pentagons from six for ordinary hexagons. Display canonical shared identities as topology,
  not separate observations and not as cell identifiers.
- [H3 indexing](https://h3geo.org/docs/api/indexing/) distinguishes grid centers from geographic
  polygon centroids. Keep the map's center marks labeled as grid centers.
- [H3 spherical measurements](https://h3geo.org/docs/api/misc/) provides independent cell-area and
  edge-length references. Report spherical units and measure each actual cell, not resolution-wide
  average area.
- [Shapely precision reduction](https://shapely.readthedocs.io/en/stable/reference/shapely.set_precision.html)
  explains that rounding can remove duplicates or collapse geometries. Distinguish pointwise
  rounding from valid-output behavior; count and display what each chosen operation removes.
- [Shapely snapping](https://shapely.readthedocs.io/en/stable/reference/shapely.snap.html) warns that
  tolerance alone does not guarantee validity. The showcase should inspect actual output topology
  and avoid claiming that snapping is a universally safe repair.
- [Shapely canonical geometry and oriented bounds](https://shapely.readthedocs.io/en/stable/reference/shapely.Polygon.html)
  distinguishes coordinate normalization from the minimum-area oriented envelope. Bounds summarize
  extent; they do not recover the route or its travel direction.

## Local contributor contract audit

`GPUH3CellIdentities` writes six slots per cell for either directed-edge IDs or canonical vertex
IDs; the deleted pentagon slot is zero. Edge/vertex IDs cannot be passed to cell-only operations.
`GPUCellGridPath` reports the true path length, independently of bounded stored capacity. Some
pentagon-crossing pairs retain a valid distance but emit no path; display this separate status.
`GPUCellMeasures` supports ten geometric boundary segments, including distortion vertices, with
square-kilometre area and kilometre perimeter available explicitly. The contributor's documented
float32 fallback becomes unreliable on distorted cells above resolution five; this investigation
will stay at resolutions three through five instead of presenting unsupported fine-scale accuracy.

`GPULineLoopRemoval` implements chronological erasure on noded open lines, retaining source-edge
provenance. It is not full GEOS offset-curve component selection. `GPUSubdividePolygons` offers
convex and rectilinear partitions with explicit capacity/work status; any use should explain a
real clipping region rather than append an unrelated demonstration. `GPUMinimumBounds` requires
planar coordinates and reports overflow. Use a named local projected frame and disclose its
measurement approximation.

`GPULocalCollinearity` is a regression diagnostic for predictors, not a line-editing operation.
It has been referred to the statistics planner and excluded from this geometry scope.

## Validation plan

H3 checks will use independent h3-js operations for exact edge/vertex IDs, grid distances and
path invariants, plus cell-area/perimeter tolerances appropriate to the documented precision.
Route checks will use independent CPU geometry calculations or Shapely where semantic contracts
match, preserving the distinction between canonical representation, precision change, snapping,
clipping and chronological loop erasure. Browser checks must inspect the actual rendered output,
all error/capacity statuses, and graph rebuilds on a stable production preview.

The cell investigation is source-ready and awaiting coordinated production validation. Both CTA
investigations are implemented and undergoing source review before GPU acceptance.

## Endpoint audit

The existing OpenFlights source has named airport rows ORD 2033, MDW 1703, JFK 1247, LAX 1501,
and SYD 2589. Its `locations` column is the shipped float32 coordinate source. The
[provider documentation](https://openflights.org/data.php) identifies the historical nature of
its route data; this story uses airport coordinates only and makes no current-service claim.
Independent h3-js calculation over those exact shipped coordinates gives:

| ORD destination | Resolution 3 hops | Resolution 4 hops | Resolution 5 hops |
| --- | ---: | ---: | ---: |
| MDW | 1 | 1 | 2 |
| JFK | 11 | 27 | 75 |
| LAX | 23 | 66 | 164 |
| SYD | undefined | undefined | undefined |

This supports a default ORD–JFK comparison at resolution four, storage-capacity choices of
16/64/256 cells, and a distant Sydney case that demonstrates an undefined local grid path.
The ORD cell areas are respectively 12,564.609499, 1,785.025829 and 255.350519 km². These are
reference calculations for the actual indexed cells, not average areas for a whole resolution.

## Cell implementation and independent check

`cell-neighborhoods` is now implemented with an actual GPU pipeline from float32 source airport
coordinates through H3 keys, bounded grid path, per-cell measures, directed-edge identities and
canonical vertices. h3-js decodes GPU-produced keys for map presentation only. A separate boundary
focus highlights the first adjacent pair and its shared canonical vertices; airport labels and
camera framing distinguish local, regional, continental and distant cases. A consistent destination
colour agrees with the legend. The sixth edge slot is inspected even for pentagons because their
deleted direction need not be the final slot.

The durable `scripts/check-cell-neighborhoods-reference.mjs` runs fourteen scenarios: four airport
destinations at each of three resolutions, plus two constrained-capacity cases. Test-only
Playwright instrumentation observes the production SummaryReader's mapped output bytes; no
production debug interface or expected-result table is added. The test verifies exact endpoint
keys, hop distance and required count, every actual path-cell adjacency, complete-path endpoints,
all directed-edge and vertex identities, exact edge/pentagon counts, and spherical area/perimeter
against independent h3-js calculations. A 0.3% relative tolerance accommodates the documented
resolution-five boundary fallback; ordinary-cell error should be much smaller. GPU execution and
visual acceptance are pending the next coordinated production build and slot.

Production wave 10 results: all fourteen numerical cases passed, with zero browser errors. Every
endpoint key, hop count, stored-path adjacency, directed-edge identity and canonical vertex identity
agreed with h3-js. The largest relative area error was 0.00019791 (about 0.0198%) on the resolution-five
Los Angeles path. Required/stored counts were 67/16 and 165/64 for the two bounded cases; Sydney
returned undefined distance at all three resolutions. All five story steps passed the production
smoke harness. Evidence is in `/tmp/cell-neighborhoods-reference/` and
`/tmp/cell-neighborhoods-production/`.

Screenshot review then caught a separate presentation defect: adding `0x` to H3 vertex keys silently
decoded them into the wrong region, while prefixed directed edges were rejected. The numerical
identities were correct. Production display helpers now use canonical bare-hex keys; they were
extracted to `cell-neighborhoods-geography.ts` so `check-cell-neighborhoods-identities.mjs` tests the
actual functions used by the map. It verifies both shared vertices against independent H3 coordinates
within 1e-7 degrees and checks Chicago bounds, as well as reciprocal directed-edge endpoints. This
CPU regression passes. Compact airport labels and SDF/automatic glyph handling address clipping and
outline warnings. The corrected boundary map still requires a rebuilt production visual check.

## CTA route scope and source audit

Parent approved two separate inquiries so that editing intent and polygon windowing remain
readable. `route-editing` asks which operations preserve a source route and which deliberately
change it. `route-windowing` asks which portion of that route lies inside the actual Loop
community boundary, with coarse bounds and polygon partitioning kept distinct from membership.

The selected CTA Red Line shape is row 16,958, route-table row 124 (`id: Red`), direction 0.
Its 96 original stored vertices occupy source rows 49,371 through 49,466, inclusive. The first
coordinate is (−87.67289734, 42.01906967) and the last (−87.62442017, 41.72238541). An independent
source check finds no consecutive duplicate coordinates. The derived source already describes
its clipping to the city bounding box plus 1 km; the investigation does not call it a damaged
or full-resolution shape. Any reversal, duplicate insertion, revisit or coordinate perturbation
is an explicit scenario, never silently substituted for the source.

[CTA's GTFS publication](https://www.transitchicago.com/developers/gtfs/) is the source service.
The [GTFS schedule specification](https://gtfs.org/documentation/schedule/reference/) assigns
meaning to ordered shape points and explains why traveled-distance references matter when a
vehicle retraces or crosses an alignment. Thus canonical geometric equality cannot establish
identical travel order. The [GTFS best practices](https://gtfs.org/documentation/schedule/schedule-best-practices/)
also distinguish shape alignment from stop placement. Genuine station coordinates can be snap
references without being guaranteed locations on the represented centerline; a tolerance is an
editing choice, not evidence that the result is more accurate.

The Loop boundary is Chicago Community Area feature row 31 (`areaId: 32`), polygon part 32,
ring 32, with 236 source vertices in the range `[5441, 5677)`. Red Line source vertices 52–53 lie
inside this polygon. Both data sources will be transformed into the same named local metric
frame before planar operations. Source geographic coordinates remain available for orientation.
No service frequency, current operating condition or journey recommendation is inferred.

The editing story will distinguish reversal/canonical equality, repeated-point cleanup, precision
rounding, snapping to genuine reference vertices, and chronological loop erasure on an explicit
simulated revisit. Segment snapping inserts reference vertices near segments; it is not projection
onto a reference line and has no GEOS topology-safety heuristic. The windowing story will expose
minimum bounds as coarse summaries, convex versus rectilinear subdivision as bounded computational
representations, and actual inside/outside clipped pieces with completeness flags and conservation
checks. Neither subdivision nor a bounding rectangle establishes point-in-polygon membership.

## Independent CTA geometry reference and review findings

`scripts/validate-cta-geometry.py` reads the unchanged binary source assets, reproduces the declared
local projection and float32 storage, then supplies those exact input coordinates to independent
Shapely 2.1.2 / GEOS operations in double precision. This isolates algorithm comparison from
coordinate-system differences. The report `cta-geometry-validation.json` records input checksums,
source coordinates and expected results. It is a development reference, not shipped runtime data.

- Retained Red Line: 96 vertices, 35,223.888754 m stored length, simple and open.
- Loop boundary: 4,297,981.241583 m²; inside line length 2,231.938909 m and outside length
  32,991.949845 m sum to the original length.
- Route convex hull: 11 vertices; diameter 33,228.296726 m; minimum rotated rectangle area
  69,208,194.245133 m².
- There are 33 genuine rail station references linked to this route. Strict-tolerance vertex
  snapping affects 0, 2, 5, 8 and 16 source vertices at 0, 5, 20, 50 and 100 m respectively.
- Precision grids of 1 and 10 m leave 96 rows; 50 and 100 m reduce the count to 92 and 82. The
  coarse settings are explicit stress scenarios rather than an accuracy recommendation.

Review found a draft error where lengths and areas were computed after unprojecting native XY
back to degrees. The implementer is correcting all measurement paths before production testing.
Review also required lazy computation imports, an actual predicate comparison of reversed stored
order with canonical equality, native spatial contexts, and a camera centered on the northern
derived revisit instead of downtown. Count labels distinguish removed vertices from positional
row differences after compaction; an index comparison must not be called a moved-vertex count.

The durable route-windowing checker observes actual production readback and uses GEOS to verify
both clipping modes, minimum bounds, partition union, overlap and source-area conservation for
convex and rectilinear variants. The separate editing checker covers fourteen cases using GEOS
cleanup/precision and independent nearest-reference/insertion calculations. Their source and
syntax checks are complete; GPU execution and visual acceptance remain pending.

## Wave-eleven production follow-up

The corrected H3 display was inspected on the fifth step in production. Both shared-vertex marks
now appear at the first adjacent Chicago-area cell boundary, and reciprocal directed-edge keys
are decoded in the readout. No browser/GPU errors occurred. The independent helper regression
also passes actual production helper code with GPU-format low/high word identities. Visual
artifact: `/tmp/cell-identities-wave11/step5.png`.

CTA route editing passed all 14 independent GEOS/direct-reference cases, including exact emitted
coordinates, counts, statuses and canonical/predicate results. Artifact:
`/tmp/route-editing-reference-wave11/`. Its final revisit screenshot revealed low-contrast outlined
endpoint labels on the light basemap; these will become solid dark labels. The separate
route-windowing study stopped before GPU work because its guard incorrectly expected GeoJSON
`areaId`, while the shipped feature uses `id: 32, name: 'Loop'`. The source type and guard now use
that actual property; its subdivision and clipping tests remain pending the next production build.

## Declared affine frame scenario

The final editing step applies an explicit (+64 m east, −32 m north) translation with
`GPUAffineTransform`, followed by its inverse on the GPU. Source, shifted and recovered paths
are rendered separately. `GPUGeometryMeasures` measures both source and shifted line lengths;
readback also measures coordinate recovery and retains the original line offsets and row order.
This is a controlled coordinate-frame scenario, not observed movement or a correction established
by external control points. It extends the existing investigation rather than adding a new story.

The independent `check-route-affine-reference.mjs` computes float32 arithmetic over the retained
96 input rows. Source length is 35,223.888754 m; shifted length is 35,223.888757 m; maximum inverse
recovery error is 0.000007629 m. The production editing checker now adds a fifteenth case checking
every translated/recovered coordinate, both measured lengths (5 cm tolerance for float32
reductions), unchanged offsets, and the translation view. Existing fourteen-mode readback layouts
are unchanged; only translation appends recovered XY rows and two length scalars. Source/type/
CPU-reference checks pass; this added GPU mode awaits the next coordinated production build.

The [official Shapely affine-transformation reference](https://shapely.readthedocs.io/en/stable/manual.html#shapely.affinity.translate),
reviewed 2026-10-25, defines translation as dimension-wise offsets in a Cartesian frame. The
scenario uses that definition; the fixed vector is declared, not estimated. Rotation, shear,
nonuniform scaling and alternate transform origins are separate capabilities and are not claimed
as exercised by this translation case. The local independent GEOS environment remains Shapely
2.1.2; the live documentation reviewed here is 2.2.0.

### Final CTA acceptance: frozen wave15

All 17 final cases passed on the production preview: 15 editing cases (including the declared
translation and inverse) and two windowing modes, each checking inside and outside clips. The
tested bundle SHA-256 was `5c13c77a7d5dd844146b724ae5c70d49272c63080f028da1228e99f5e22fac65`.
All four served scene/compute chunks matched the build manifest before and after the run. No
browser or GPU console errors were reported, and both browser sessions were closed afterward.

The durable [acceptance packet](cta-acceptance-wave15/acceptance.json) records exact commands,
checker and reference hashes, tolerances, limitations and raw-report hashes. Full parsed editing
and windowing GPU outputs and the independent GEOS checks are retained beside it. Editing uses
independent GEOS/direct references and exact float32 translation/inverse arithmetic. Windowing
compares polygon union, overlap, clips, diameter and minimum rectangle with Shapely 2.1.2 / GEOS.
The maximum clip Hausdorff error was approximately 0.00010535 metres; clip length error was
approximately 0.00009136 metres. Both partitions had zero measured union difference and overlap.

Four saved desktop maps were visually reviewed: declared translation, eight-row revisit stress,
convex subdivision and rectilinear subdivision. Both subdivision modes return the same 94-piece
geometry for this boundary and vertex limit. This fixture establishes conservation and capacity
completeness for both execution paths, but does not demonstrate distinct partition shapes. The
affine check covers translation and its inverse only; rotation, skew, scale and alternate origins
remain outside this bounded acceptance. These controlled scenarios make no claim about observed
movement or malformed CTA source data.
