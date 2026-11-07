# Route footpoints implementation ledger

Date: 2026-10-25

## Delivered

The `joins/route-footpoints` story has seven chapters: establish the source question, preserve multipart boundaries, inspect the nearest footpoint, compare distance questions, classify against a synthetic workframe, run a bounded query, and interpret the offset distribution. Its inputs and visible results are built by `route-footpoints.compute.ts`; scene metadata is in `route-footpoints.scene.ts`.

The route select defaults to Blue and offers Red, Blue, Brown (`Brn`) and Green (`G`). Each route ID is looked up in `cta-transit` `properties.routes`; stop rows are selected only where `stopMode === 1` and the route index occurs between the corresponding `stopRouteOffsets` entries in `stopRoutes`. Direction 0 source shape parts come from `shapeRoute`, `shapeDirection`, `shapePathOffsets` and `shapeVertices`. Each station is paired with every part for the selected route. Part boundaries stay separate: no endpoint stitching and no implicit line closure.

The derived archive stores source longitude/latitude vertices as float32. The scene uses the dataset's projected local metre coordinates and supplies `transit.getProjection(origin).spatialContext` to planar contributors. This is the documented local Mercator-scaled metre frame, not an azimuthal-equidistant frame. The archive describes shape paths as clipped to the city bounding box plus one kilometre; a finite nearest result therefore refers to available clipped geometry and does not prove the uncut route has no continuation. If a paired feature has no reachable segment, GPU NaN distances and `0xffffffff` line/segment indices remain distinct and are reported as no segment.

| Contributor | Delivered input/output and map/readout use |
| --- | --- |
| `GPUPairwisePointLinestringNearest` | One query row per route-assigned rail station; each feature row contains all route parts. Caller-owned outputs are float32 distances and positions plus uint32 local part and segment indices. GPU positions render as footpoints; a GPU-result-driven link joins the selected source station to its footpoint. The offset histogram uses finite GPU distances. |
| `GPUPairwisePointDistance` | Computes station-to-returned-footpoint distance in the same graph and station-to-first-source-route-vertex reference distance. The footpoint result is shown beside the nearest-linestring distance as a parity check; the first-vertex reference is explicitly a different question. |
| `GPUPairwisePointSegmentDistance` | After the nearest summary arrives, the returned local part/segment indices select that exact archived source segment. The small paired endpoint arrays are gathered on the CPU from those indices, uploaded, and compiled asynchronously as a second graph. That contributor's distance is shown beside the linestring result, and the selected source segment is drawn. CPU does not search for a nearest segment. `GraphSwap` manages the asynchronous comparison graph outside `encode`. |
| `GPUHaversineDistance` | Uses station and paired first-route-vertex longitude/latitude arrays. The units select is passed to the contributor constructor with a matching sphere radius (6371008.8 m, or its kilometre conversion). The readout shows the actual output unit selected (metres to one decimal or kilometres to six decimals) and a normalized metre comparison. The contributor result remains float32. |
| `GPUPairwisePointInPolygon` | Classifies each actual station against one per-row copy of a source-anchored synthetic ±400 m square centered on the active subset's first station. Station fill binds the uint32 category output directly with palette order outside, inside, boundary and uncertain. The matching square is visible and labeled synthetic. |
| `GPUPointSpatialQuery` | Radius, bounds or polygon query against the active route's station rows. Radius and bounds are centered at the first station in that route subset; the polygon uses the same source-anchored synthetic workframe. Matching row IDs, clamped count, required count and overflow are read from the GPU output. A dashed support circle or rectangle is visible. Compact IDs draw as pink rings only after a current, non-overflow summary; zero-result and overflow summaries draw no rings. Capacity equals the active station count. |

Every point and segment layer passes an explicit `instanceCount`: route source segments, all active station rows, query compact rows, returned nearest positions, and the current selected link/segment. The selected station layer draws one instance when the active subset is nonempty. Nearest-footpoint drawing waits for the current GPU summary; the point and segment comparison remains driven by actual GPU nearest output positions and indices. The CPU gathers only the returned source segment endpoints for the second GPU distance check.

Route, query predicate, radius and haversine unit changes build their requested graph variant through `GraphSwap`; deep-link initial state is passed into the first build. The station selector is a zero-based active-subset row; requested values are normalized to that route's valid row range and the selection readout reports the subset size and source row. Layers hide while a different keyed variant is requested or its identity is stale. Summary results are guarded by the current variant identity and requested option key. Readouts and charts clear when a request or selected station changes, including the footpoint parity readout and highlight. Segment graph commits also check current route identity and selected row, explicitly disposing stale results. Readers are stopped on replacement or destroy, and failed builds dispose their resources. One imported graph view is reused per buffer in each graph.

The map uses the actual archived route paths and assigned station positions over the Chicago street context. Official route-color accents are read from the archive manifest. No link exaggeration is applied. Point and segment comparison geometry comes from GPU result positions/indices. The cartouche and narrative state that a station-to-route offset is not entrance distance, accessibility, network distance, travel time, source-error diagnosis, or evidence about service after the archive vintage. There is no facility/model pairing in this scene; it compares a parent-merged CTA rail station point with archived route-shape geometry.

## Research references and source assumptions

- CTA feed source and archive provenance: https://www.transitchicago.com/developers/gtfs/
- GTFS `stops.txt` and `shapes.txt` semantics: https://gtfs.org/documentation/schedule/reference/
- Geospatial family comparison: https://docs.rapids.ai/api/cuspatial/stable/api_docs/spatial/
- Local bundled data: `cta-transit` manifest; sample service date `2026-09-16`; `stopMode` category 1 is rail, parent platforms are merged, and `stopRoutes` stores route-property indices. The archive has no station names in the used columns, so readouts use source row IDs.
- The archive is derived data under CTA's developer terms. The scene credits CTA and does not describe the archive as a live feed.

## Deferred variant

Raw-binary64 coordinate-encoding variants are not delivered. The archived coordinates are already float32; a synthetic shared 1e9 m origin-shift experiment can demonstrate cancellation in local-f32 geometry and the uint32x4 raw-binary64/double-single nearest, point-distance, segment-distance and polygon paths, but it should follow GPU validation of this primary scene. Such a demo would preserve offsets in the archived float32 coordinates, not recover additional location accuracy. Haversine output remains float32 even when its raw input coordinates are float64.

## Verification status

- Implementation status: code and scoped scene checks are ready; GPU validation remains pending.
- Direct Biome formatting of `route-footpoints.compute.ts` and `route-footpoints.scene.ts`: passed.
- `node examples/deck/spatial-analysis-showcase/scripts/check-story-steps.mjs joins`: passed. This checks story metadata structure, not rendered output.
- `node examples/deck/spatial-analysis-showcase/scripts/check-cartography.mjs --strict joins`: passed. This checks declared cartography rules, not rendered output.
- No GPU graph compilation, browser, numerical comparison, root build, install, lint wrapper or TypeScript build was run for this reviewer-fix pass. GPU validation is pending for buffer binding and actual layer visibility, f32 distance parity, invalid rows, graph-swap races, route deep links, and actual query counts/overflow on device.
