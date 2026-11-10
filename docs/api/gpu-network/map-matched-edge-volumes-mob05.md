# Edge volumes of map-matched tracks

`GPUMapMatchedEdgeVolumes` turns the `output.traversedEdges` of `GPUMapMatching` into traversal
counts per network edge and, optionally, the busiest edge-to-edge movements.

`GPUMapMatching` lists the edges each vehicle passed, including the intermediate edges between two
fixes, in travel order. Between sub-tracks (after a break or an unmatched fix) it writes one
`GPU_MAP_MATCHING_NONE`. This contributor reads that list:

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
- `output.truncated` (optional): 1 when a track needed more than `traversedEdgeCapacity` entries;
  only the stored entries are counted.

| Prop | Meaning |
| --- | --- |
| `traversedEdges` | `{edges, counts}` of the matching, `trackCount * traversedEdgeCapacity` and `trackCount` rows |
| `trackCount`, `traversedEdgeCapacity`, `edgeCount` | Compile-time shapes |
| `transitionPairCapacity`, `maximumProbeCount` | Compile-time, with `output.transitions` |

Verified end to end: tracks are generated along known paths of a 4 x 4 grid, matched on the GPU,
and the edge volumes and movements are compared with plain counting over the ground-truth edge
lists (including intermediate edges, opposite directions, a repeated edge and a break). Valhalla
and pgRouting were not run; the semantics above follow their documented output.

{/* lens:GPUMapMatchedEdgeVolumes */}
