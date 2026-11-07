# Chicago route workbench

## Plan versus delivered

| Planned capability | Delivered in the isolated scene | Evidence / boundary |
| --- | --- | --- |
| Real Chicago streets and turns | Reproducible connected extract from `chicago-walk`; directed links, turn pairs and source polylines are retained | CPU extract check: 61 vertices, 128 directed edges; 65,554 source edges excluded by the local bound |
| Point-on-edge inquiry | Direct `GPUNetworkPointRouting`; three named local query pairs plus adjustable origin and destination fractions | GPU status 0/1/2 is rendered as cost, invalid data, or incomplete incumbent |
| Departure-time consequences | Direct `GPUNetworkTimeRouting` in affine-duration and piecewise-arrival modes | Both variants are compiled and computed; choice selects the featured arrival summary. Piecewise values are a hypothetical arrival envelope, not timetable or forecast data |
| Alternatives and overlap | Direct `GPUAlternativeRoutes`, route costs and edge arrays publish GPU route masks; shared directed links are counted from returned paths | Three routes requested, 100,000 expansion limit and 1,024 frontier limit; incomplete results are explicitly incumbents |
| Bounded cost-table reuse | Direct `GPUNetworkCostTable` writes CSR output; a downstream GPU lookup consumes that table for the currently selected endpoints | Readout distinguishes cached-pair hit, outside-bound miss, overflow and nonconvergence |
| Visible result and pending states | Computed alternative edge IDs drive map masks over actual OSM path vertices; fractions drive ring markers | Current generation must have both route and summary readbacks before route overlays or computed tooltips are valid |
| Public option behavior | Every UI control is a parameter write; all graph sizes, model variants and solver capacities are compile-time constants shown in the code sample and this note | No control advertises a rebuild badge. `timeModel` selects the featured result while both models run |

## Implementation

The scene is registered by `chicago-route-workbench.scene.ts`; its heavy code is lazy-loaded from
`chicago-route-workbench.compute.ts`. It consumes only the bundled `chicago-walk` network. The
workbench locates the closest vertex to Millennium Park in the dataset's local metric projection,
then expands breadth-first through outgoing edges. Candidate nodes are kept only if their induced
directed links fit `GPUNetworkPointRouting`'s 128-edge ceiling. The final subgraph is checked and
trimmed to that bound. The current byte bundle yields 61 vertices and 128 directed links, with
65,554 source directed links omitted. The extract remains connected by construction, but it is a
local workbench graph rather than a citywide routing network.

The graph uses independent topology and line geometry. CSR `source`/`target` IDs and explicit
`[incomingEdge, outgoingEdge]` turns define legal transitions. OSM path vertices are projected to
the dataset's local meter frame and drawn as line segments. A segment-to-edge index lets GPU route
flags select the original line geometry without inventing straight connector geometry. Query
markers are interpolated along the selected link's polyline at the requested fraction.

All costs shown as minutes use the bundle's `edgeTravelTime` values (length divided by 4.8 km/h).
Point-routing costs apply the adjustable nonnegative affine slope at the selected departure offset.
The affine time contributor receives `[base duration, slope]` per transition. The piecewise contributor
receives three transition-major samples at 08:00, 14:00 and 20:00, with hypothetical arrival offsets
of 1.12×, 1.04× and 1.18× the bundle's duration. The arrivals are monotone for this local data and
are interpolated by the contributor. No measured congestion, signal delay, transit schedule or
traffic forecast is represented.

The cost table uses the same fixed base walk minutes as alternatives and exposes an adjustable
minute limit. Its CSR is consumed by a GPU lookup kernel for the selected time-route endpoints.
Disabling reuse sets the contributor's limit to zero, so the table contains only zero-cost
self-pairs; a missing non-self pair is reported as a bounded miss. Output overflow and convergence
are checked before a lookup can be described as definitive.

Summary readers are asynchronous. A monotonically increasing query generation invalidates old
results when controls change, and the layer mask stays hidden until both the alternative result and
solver summary for the same generation are valid. Route costs, path lengths, completion flags,
shared-link counts and the selected arrival are CPU summaries of GPU-written buffers; line geometry
and route masks remain GPU resident. The `ParameterBuffer.importToGraph` pattern is not needed here:
buffers are application-owned and imported once per graph through the public graph-buffer helper.

## Provenance and citations

- **Bundled source:** `examples/deck/spatial-analysis-showcase/public/data/chicago-walk/manifest.json`
  records the bundled network, EPSG:4326 coordinates, ODbL 1.0, edge lengths in metres and travel
  times in seconds. The corresponding catalog entry links to OpenStreetMap's copyright page.
- **Reproduction and selection:** `examples/deck/spatial-analysis-showcase/scripts/data/chicago-walk/README.md`
  and `build.py` identify a BBBike Chicago OSM extract and the reproducible pedestrian-network
  builder. The builder filters selected inaccessible/private and non-routing feature tags, constructs
  both directions, simplifies the graph, keeps its largest strongly connected component and derives
  travel time at 4.8 km/h. Its source bounding box is approximately 5.4 by 5.0 km around the Loop.
- **Licence and attribution:** OpenStreetMap's [copyright and licence page](https://www.openstreetmap.org/copyright/sp)
  identifies ODbL 1.0. The [OSM Foundation attribution guidance](https://osmfoundation.org/wiki/Attribution_Guidelines)
  recommends making the source recognizable to users of the produced work. The scene displays
  “© OpenStreetMap contributors” and the data panel retains the catalog source link.
- **Pedestrian feature semantics:** the [OpenStreetMap `footway` key reference](https://wiki.openstreetmap.org/wiki/Key%3Afootway)
  describes sidewalks and crossings as distinct pedestrian ways, matching the bundled builder's
  separation of sidewalk, crossing, path/pedestrian, street and steps classes.

## Remaining limits and verification

- The 61-node, 128-directed-edge extract is constrained by the point-routing API. Links outside it
  are excluded explicitly; a failed route is not a statement about full-city pedestrian access.
- The alternative-route search has exponential worst-case behavior and is bounded. `complete = 0`
  means the displayed candidates are not certified top-k. The cost table is also bounded by its
  configured cost limit and capacity; neither overflow nor nonconvergence is interpreted as
  infeasibility.
- Time results use two illustrative FIFO duration assumptions and a three-sample arrival envelope.
  They are not measured travel conditions. Clock readouts are local time-of-day offsets from 08:00;
  route and table costs are minutes.
- A standalone CPU structural check reproduced 61 vertices, 128 directed edges and 65,554 omitted
  edges from the bundled bytes. Browser/GPU execution and numerical path references remain for the
  parent review; no runtime GPU result is claimed here.
- Scoped formatting passed. The showcase TypeScript command reported no errors in either owned
  file, but exited nonzero on concurrent workspace failures in `earth/spherical-boundaries`,
  `geometry/repair-overlay`, `regression/county-panel` and `statistics/county-regionalization`
  (option value types, a missing result variant, invalid call arity, an incorrect `CommandEncoder`
  import, stale result-shape fields and invalid `labels` scene fields). `nvm use` could not run
  because `nvm` is unavailable in this shell; Node was v21.7.1 against the repository's `.nvmrc`
  v22.23.4.

## Planner review corrections and independent reference

The final query labels refer to the east, north and west boundaries of the retained extract.
They do not claim to reach Millennium Park, the Riverwalk or Union Station when those landmarks
lie outside the graph. Selection uses the closest retained polyline to each extremal boundary
midpoint. Static point-on-link costs use the bundled walk minutes; departure-time effects belong
to the two separately labelled FIFO routing models. Affine slope is minutes per departure-minute,
not a percentage of the base duration.

`chicago-route-reference.mjs` independently reconstructs the graph and solves CPU path references.
At fractions 0.2/0.8, east-to-north has a fractional-point cost about 4.864 minutes and static
vertex alternatives about 4.464, 5.070 and 5.252 minutes. East-to-west has point cost 4.196 and
vertex alternatives 3.927, 5.303 and 5.417. North-to-west has point cost 5.036 and alternatives
4.880, 5.065 and 5.671. Different point and vertex endpoints explain those differences.
The 12-minute cost table contains all 3,721 directed pairs, while the 2-minute table has 1,671.
These are independent numerical targets; production GPU assertions are pending.

## Pre-first-sample waiting extension

The piecewise contributor exposes one waiting policy: before an edge's first sample it returns that sample's arrival; after its last sample it holds the last duration constant. The previous samples began at zero, so nonnegative departures never exposed the first branch. `firstSampleMinute` now shifts both departure and arrival samples by 0, 15 or 60 minutes, retaining the default zero and the same FIFO travel-duration shape. The interface labels this as hypothetical availability on the actual street extract, not a real street opening rule or transit timetable.

The numerical harness adds three independent time-translation checks: departure 0 with first sample 15, departure 30 with first sample 60, and departure 60 at the first sample 60. Each must equal the separately reconstructed zero-origin piecewise arrival plus its sample-clock shift. The corresponding initial wait is 15, 30 and 0 minutes. These source-ready cases are pending a fresh production build; the earlier 14-case wave11 acceptance does not cover them.
