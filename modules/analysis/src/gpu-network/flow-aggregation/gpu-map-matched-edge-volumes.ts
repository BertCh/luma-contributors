// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createFillNode,
  createWGSLKernelNode,
  getWGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {GPUFlowAggregation, type GPUFlowAggregationOutput} from './gpu-flow-aggregation';
import {GPU_FLOW_AGGREGATION_MAXIMUM_ZONE_COUNT} from './flow-aggregation-pairs';

const OPERATION = 'GPUMapMatchedEdgeVolumes';

/** The `output.traversedEdges` of `GPUMapMatching`, as read by {@link GPUMapMatchedEdgeVolumes}. */
export type GPUMapMatchedEdgeVolumesTraversedEdges = {
  /** `trackCount * traversedEdgeCapacity` CSR edge rows, `0xffffffff` between sub-tracks. */
  edges: GraphDataView<'uint32'>;
  /** Entries each track needs; entries past the capacity were not stored. */
  counts: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUMapMatchedEdgeVolumes}.
 *
 * Compile-time: `trackCount`, `traversedEdgeCapacity`, `edgeCount`, `transitionPairCapacity`,
 * `maximumProbeCount` and which outputs exist. Per-frame: the contents of `traversedEdges`.
 */
export type GPUMapMatchedEdgeVolumesProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'map-matched-edge-volumes'`. */
  id?: string;
  /** Traversed-route output of `GPUMapMatching` (its `traversedEdgeCapacity` must be `traversedEdgeCapacity`). */
  traversedEdges: GPUMapMatchedEdgeVolumesTraversedEdges;
  /** Number of tracks of the matching. */
  trackCount: number;
  /** Edge rows reserved per track: `GPUMapMatchingProps.traversedEdgeCapacity`. */
  traversedEdgeCapacity: number;
  /** Number of network edges (CSR rows). Entries at or above it are ignored. */
  edgeCount: number;
  /**
   * Hash table slots of the transition aggregation, a power of two: the most distinct
   * `(edge, next edge)` pairs retained. Required with `output.transitions`.
   */
  transitionPairCapacity?: number;
  /**
   * Forwarded to the transition aggregation's `GPUHashIndex`. Defaults to `transitionPairCapacity`,
   * lowered so that `rows * maximumProbeCount` stays below 2^32.
   */
  maximumProbeCount?: number;
  output: {
    /**
     * `edgeCount` rows: how many times matched tracks traversed each edge. A track that passes
     * the same edge twice (a loop) counts twice; a fix on the same edge as the previous fix adds
     * nothing because the traversed route lists it once. Overwritten each encoding.
     */
    edgeVolumes: GraphDataView<'uint32'>;
    /**
     * Optional turning movements: a {@link GPUFlowAggregationOutput} whose zones are edges. A row
     * is the move from an edge to the next edge of the same sub-track; flows are sorted by count
     * descending, `ids` hold `edge * edgeCount + nextEdge` pair keys, `zoneOutCounts[e]` are the
     * traversals of `e` that continued onto another edge and `zoneInCounts[e]` the ones that
     * entered `e` from another edge. Requires `edgeCount <= 65535` (the flow aggregation pair key).
     */
    transitions?: GPUFlowAggregationOutput;
    /** Optional one-row flag: 1 when a track needed more than `traversedEdgeCapacity` entries. */
    truncated?: GraphDataView<'uint32'>;
  };
};

/**
 * Edge volumes of map-matched tracks: how many traversals each network edge received, and which
 * edge-to-edge movements the tracks made.
 *
 * `GPUMapMatching` lists the edges each vehicle passed, including the intermediate edges no fix
 * landed on, separated by `GPU_MAP_MATCHING_NONE` between sub-tracks. Each listed edge is one
 * traversal, counted into `edgeVolumes` with an integer atomic add (exact and independent of
 * thread order). With `output.transitions` the same list is also read as flows: an edge followed
 * by another edge of its sub-track is one movement of a {@link GPUFlowAggregation} whose zones
 * are the edges, which gives the busiest turning movements and per-edge in and out counts without
 * inventing an adjacency across a break. The last edge of a sub-track starts no movement.
 *
 * This is the traversal semantics of Valhalla `trace_attributes` and the per-edge count of a
 * pgRouting route table: one count per edge entry of the matched path, not per GPS fix.
 */
export class GPUMapMatchedEdgeVolumes implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUMapMatchedEdgeVolumesProps;
  /** Rows of the traversed-route list: `trackCount * traversedEdgeCapacity`. */
  readonly rowCount: number;

  constructor(props: GPUMapMatchedEdgeVolumesProps) {
    this.id = props.id ?? 'map-matched-edge-volumes';
    this.props = props;
    const {id} = this;
    validateCapacity(id, 'trackCount', props.trackCount);
    validateCapacity(id, 'traversedEdgeCapacity', props.traversedEdgeCapacity);
    validateCapacity(id, 'edgeCount', props.edgeCount);
    this.rowCount = props.trackCount * props.traversedEdgeCapacity;
    // Row indices are u32 in the kernel.
    validateCapacity(id, 'trackCount * traversedEdgeCapacity', this.rowCount, 1, 0xfffffffe);
    validatePackedView(props.traversedEdges.edges, ['uint32'], `${id} traversedEdges.edges`);
    validatePackedView(props.traversedEdges.counts, ['uint32'], `${id} traversedEdges.counts`);
    validateViewLength(id, 'traversedEdges.edges', props.traversedEdges.edges, this.rowCount);
    validateViewLength(id, 'traversedEdges.counts', props.traversedEdges.counts, props.trackCount);
    validatePackedView(props.output.edgeVolumes, ['uint32'], `${id} output.edgeVolumes`);
    validateViewLength(id, 'output.edgeVolumes', props.output.edgeVolumes, props.edgeCount);
    if (props.output.truncated) {
      validatePackedView(props.output.truncated, ['uint32'], `${id} output.truncated`);
      validateViewLength(id, 'output.truncated', props.output.truncated, 1);
    }
    if (props.output.transitions) {
      validateCapacity(
        id,
        'edgeCount with output.transitions',
        props.edgeCount,
        1,
        GPU_FLOW_AGGREGATION_MAXIMUM_ZONE_COUNT
      );
      if (props.transitionPairCapacity === undefined) {
        throw new Error(`${id} output.transitions requires transitionPairCapacity`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.output.edgeVolumes, props.output.truncated],
      [props.traversedEdges.edges, props.traversedEdges.counts]
    );
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, rowCount} = this;
    const {edges, counts} = props.traversedEdges;
    validateGraphViewsBelongToGraph(id, graph, [
      edges,
      counts,
      props.output.edgeVolumes,
      props.output.truncated
    ]);
    const capacity = props.traversedEdgeCapacity;
    const hasTransitions = Boolean(props.output.transitions);
    const truncated =
      props.output.truncated ?? createTransientView(graph, `${id}-truncated`, 'uint32', 1);
    const originZones = hasTransitions
      ? createTransientView(graph, `${id}-origin-edges`, 'uint32', rowCount)
      : undefined;
    const destinationZones = hasTransitions
      ? createTransientView(graph, `${id}-destination-edges`, 'uint32', rowCount)
      : undefined;
    const nodes: GPUCommandNode<Parameters>[] = [
      createFillNode<Parameters>(graph, {
        id: `${id}-clear-volumes`,
        operation: OPERATION,
        view: props.output.edgeVolumes,
        type: 'u32',
        value: '0u'
      }),
      createFillNode<Parameters>(graph, {
        id: `${id}-clear-truncated`,
        operation: OPERATION,
        view: truncated,
        type: 'u32',
        value: '0u'
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-count`,
        operation: OPERATION,
        variant: hasTransitions ? 'count-and-transitions' : 'count',
        invocationCount: rowCount,
        bindings: [
          {name: 'edges', view: edges, type: 'u32', access: 'read'},
          {name: 'counts', view: counts, type: 'u32', access: 'read'},
          getWGSLKernelBinding('volumes', props.output.edgeVolumes, 'atomic'),
          getWGSLKernelBinding('truncated', truncated, 'atomic'),
          ...(originZones && destinationZones
            ? [
                {
                  name: 'origins',
                  view: originZones,
                  type: 'u32' as const,
                  access: 'read_write' as const
                },
                {
                  name: 'destinations',
                  view: destinationZones,
                  type: 'u32' as const,
                  access: 'read_write' as const
                }
              ]
            : [])
        ],
        // A row is a stored entry of a track that is an edge; its movement goes to the next entry
        // when that is an edge too (not a separator and not past the stored count).
        body: `let track = index / ${capacity}u; let slot = index - track * ${capacity}u;
let stored = min(counts[countsOffset + track], ${capacity}u);
if (slot == 0u && counts[countsOffset + track] > ${capacity}u) { atomicMax(&truncated[truncatedOffset], 1u); }
let edge = edges[edgesOffset + index];
let isEdge = slot < stored && edge < ${props.edgeCount}u;
var next = 0xffffffffu;
if (isEdge && slot + 1u < stored) { let candidate = edges[edgesOffset + index + 1u]; if (candidate < ${props.edgeCount}u) { next = candidate; } }
if (isEdge) { atomicAdd(&volumes[volumesOffset + edge], 1u); }
${
  hasTransitions
    ? `origins[originsOffset + index] = select(0xffffffffu, edge, isEdge && next != 0xffffffffu);
destinations[destinationsOffset + index] = next;`
    : ''
}`
      })
    ];
    if (hasTransitions && originZones && destinationZones) {
      nodes.push(
        ...new GPUFlowAggregation({
          id: `${id}-transitions`,
          zones: {kind: 'ids', zoneCount: props.edgeCount},
          originZoneIds: originZones,
          destinationZoneIds: destinationZones,
          pairCapacity: props.transitionPairCapacity!,
          maximumProbeCount:
            props.maximumProbeCount ??
            Math.min(props.transitionPairCapacity!, Math.floor(0xffffffff / rowCount)),
          output: props.output.transitions!
        }).getCommandNodes(graph)
      );
    }
    return nodes;
  }
}
