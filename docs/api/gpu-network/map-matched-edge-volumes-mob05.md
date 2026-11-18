# Edge volumes of map-matched tracks

`GPUMapMatchedEdgeVolumes` turns the `output.traversedEdges` of `GPUMapMatching` into traversal
counts per network edge and, optionally, the busiest edge-to-edge movements.

`GPUMapMatching` lists the edges each vehicle passed, including the intermediate edges between two
fixes, in travel order. Between sub-tracks (after a break or an unmatched fix) it writes one
`GPU_MAP_MATCHING_NONE`. It also inserts this separator when a hop's intermediate edges could not
be recovered, preventing a false turn between the two retained endpoint edges. This contributor
reads that list and its status:

- `output.edgeVolumes` (`edgeCount` rows): how many times each edge was traversed. Each listed edge
  entry counts once, with an integer atomic add, so the result does not depend on thread order. A
  loop that passes an edge twice counts twice. A fix on the same edge as the previous fix adds
  nothing, because the route lists the edge once. This is the per-edge count of the matched path
  (Valhalla `trace_attributes` edges, a pgRouting route table), not a count of GPS fixes.
- `output.transitions` (optional): a `GPUFlowAggregationOutput` whose zones are the edges. An edge
  followed by another edge of the same sub-track is one movement. The top-K list is sorted by
  count, `zoneOutCounts[e]` counts traversals of `e` that continued on another edge and
  `zoneInCounts[e]` those that entered it from another edge. No movement is invented across a
  separator. This inherits the flow aggregation limit of 65535 zones, so it needs `edgeCount <= 65535`.
  `edgeVolumes` alone has no such limit.
- `output.status.candidateOverflow` (required): 1 when a track needed more than
  `traversedEdgeCapacity` entries or the forwarded traversal `overflow`/`candidateOverflow` word
  is nonzero. The flag resets each encoding. With grid or scoring candidate loss, counts are
  conditional on the retained match: another road can win, increasing some edge counts and
  decreasing others. Lower bounds apply only to omissions from an otherwise fixed traversal,
  such as list capacity or failed recovery after an exact match. This replaces `output.truncated`.
  A requested `transitions.status.candidateOverflow` combines traversal loss with pair-table loss,
  so it can be checked when the transition output is consumed separately. Allocate this nested
  flag and the volume flag on separate physical buffers; their publication pass reads one while
  writing the other.

| Prop | Meaning |
| --- | --- |
| `traversedEdges` | Forward `{edges, counts, status}` from matching; `status.candidateOverflow` is required and `status.overflow` is also accepted. Shapes are `trackCount * traversedEdgeCapacity` and `trackCount`. Fixed external sequences instead explicitly assert `completeness: 'known-complete'`; never use that assertion for unchecked matching output. |
| `trackCount`, `traversedEdgeCapacity`, `edgeCount` | Compile-time shapes |
| `transitionPairCapacity`, `maximumProbeCount` | Compile-time, with `output.transitions` |

Named props and nested port/status maps are captured once at construction, including inherited
getters. Later caller mutation cannot change the validated graph; view contents remain live.

Verified end to end: tracks are generated along known paths of a 4 x 4 grid, matched on the GPU,
and the edge volumes and movements are compared with plain counting over the ground-truth edge
lists (including intermediate edges, opposite directions, a repeated edge and a break). Valhalla
and pgRouting were not run; the semantics above follow their documented output.

{/* lens:GPUMapMatchedEdgeVolumes */}
