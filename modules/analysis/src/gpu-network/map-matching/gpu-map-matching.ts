// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getHashSize,
  getForwardDeclarations,
  getRouteSearchDeclarations,
  getAccessDeclarations,
  getRoutesBody,
  getPreviousFixWGSL,
  getRouteTableDeclarations,
  getRouteTableBody,
  getRouteEdgesDeclarations,
  getRouteEdgesBody,
  HOP_WORD_COUNT
} from './map-matching-search-wgsl';
import {
  getViterbiDeclarations,
  getForwardBody,
  getBackwardBody,
  getSumDeclarations,
  getHopsBody
} from './map-matching-viterbi-wgsl';
import type {Device} from '@luma.gl/core';
import {snapshotMapMatchingConstructionProps} from './map-matching-construction';
import {
  resolveGPUSpatialContext,
  warnGPUShortLongitudeLatitudeDistance,
  GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  getGPUGreatCircleRadius
} from '../../analysis-core/contracts/spatial-context';
import {
  createTransientView,
  getViewBindingRange,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createFillNode,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  getStorageBindingElementLimit,
  validateCapacity,
  validateMaximumCount,
  validateSinglePackedView,
  validateFiniteNumber,
  validateNonZeroFiniteNumber,
  validateRequirement,
  validateGPUStorageBindingByteLength,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  validateDisjointWritableViews,
  validateGraphChunkBindingRanges
} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {SCALED_QUOTIENT_WGSL} from '../../analysis-core/kernels/scaled-quotient-wgsl';

import type {GPUNetworkAdjacency} from '../../analysis-core/contracts/network-topology';
import {
  getGPUNetworkTopologyViews,
  validateGPUNetworkTopology
} from '../../analysis-core/contracts/network-topology';
import type {
  GPUSpatialContext,
  GPUSpatialContextSupport
} from '../../analysis-core/contracts/spatial-context';
import {getNetworkMetricWGSL} from '../network-spatial-context';
import {
  getWorkgroupCounterKernelSource,
  WORKGROUP_COUNTER_WORKGROUP_SIZE,
  type WorkgroupCounter
} from '../network-workgroup-counters';

const MAP_MATCHING_SPATIAL_CONTEXT_SUPPORT: GPUSpatialContextSupport = Object.freeze({
  ...GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  measure: 'distance'
});

const OPERATION = 'GPUMapMatching';
/** Bits of `-1.0f`: an out-of-range node ID and a negative (invalid) edge length. */
const INVALID_EDGE_WORD = 0xbf800000;
/** Bounds the candidate and Viterbi private arrays and the quadratic per-fix route table. */
export const GPU_MAP_MATCHING_MAXIMUM_CANDIDATE_COUNT = 8;
/** Bounds five private search tables, their hash and the optional route reconstruction array. */
export const GPU_MAP_MATCHING_MAXIMUM_ROUTE_NODE_BUDGET = 128;
/** A bounded edge rectangle inserts at most this many cells per axis. */
const MAP_MATCHING_MAXIMUM_EDGE_CELL_SPAN = 16;
/** Every grid counter and its prefix must fit uint32 even when every edge spans 16 by 16 cells. */
const MAP_MATCHING_MAXIMUM_EDGE_COUNT = Math.floor(
  0xffffffff / MAP_MATCHING_MAXIMUM_EDGE_CELL_SPAN ** 2
);
/** Bounds dense grid scratch and preserves the existing float32-exact grid indexing envelope. */
const MAP_MATCHING_MAXIMUM_GRID_CELL_COUNT = 2 ** 24;

/** Value written to `matchedEdges` for a point with no candidate edge or no feasible match. */
export const GPU_MAP_MATCHING_NONE = 0xffffffff;

/**
 * Largest `topology` edge count of {@link GPUMapMatching} on `device`: the edge table holds 12
 * bytes per edge in one storage binding, so 11184810 on a default device (128 MiB bindings).
 * Larger bindings remain capped at 16777215 edges so the grid's uint32 count scan cannot wrap.
 */
export function getGPUMapMatchingMaximumEdgeCount(device: Device): number {
  return Math.min(MAP_MATCHING_MAXIMUM_EDGE_COUNT, getStorageBindingElementLimit(device, 12));
}

/**
 * Largest `entryCapacity` of {@link GPUMapMatching} on `device`: the edge-cell table holds 4 bytes
 * per entry in one storage binding, so 33554432 on a default device (128 MiB bindings).
 */
export function getGPUMapMatchingMaximumEntryCapacity(device: Device): number {
  return getStorageBindingElementLimit(device, 4);
}

/**
 * Largest point count of {@link GPUMapMatching} on `device` for `candidateCount` candidates per
 * point (default 8). The per-point scratch is the binding that limits it: the route table takes
 * `candidateCount * (candidateCount + 1) * 4` bytes per point (288 for 8 candidates, so 466033
 * points on a default device with 128 MiB bindings), the candidate table 12 bytes per candidate,
 * and, with {@link GPUMapMatchingOutput.traversedEdges}, the hop table 20 bytes.
 */
export function getGPUMapMatchingMaximumPointCount(
  device: Device,
  candidateCount: number = 8,
  hasTraversedEdges: boolean = false
): number {
  return Math.min(
    getStorageBindingElementLimit(device, candidateCount * (candidateCount + 1) * 4),
    getStorageBindingElementLimit(device, candidateCount * 12),
    getStorageBindingElementLimit(device, (candidateCount + 1) * 4),
    getStorageBindingElementLimit(device, 8),
    hasTraversedEdges ? getStorageBindingElementLimit(device, HOP_WORD_COUNT * 4) : Infinity
  );
}

/** Number of `float32` words in the per-frame parameter view of {@link GPUMapMatching}. */
export const GPU_MAP_MATCHING_PARAMETER_LENGTH = 8;

/**
 * Per-frame settings of {@link GPUMapMatching}. Every distance is in the context's length unit
 * (native planar units, or meters/kilometers for great-circle). `routeFactor` is unitless.
 */
export type GPUMapMatchingSettings = {
  /**
   * Spatial context of the positions; when it is longitude/latitude, a distance under 10 float32
   * quanta (about 17 m) logs a warning (see Precision of longitude/latitude positions). Does not
   * change the packed values.
   */
  spatialContext?: GPUSpatialContext;
  /** GPS noise standard deviation in the context's length unit. Must be positive. */
  sigma: number;
  /** Scale in the context's length unit of `exp(-|straight - route| / beta)`. Must be positive. */
  beta: number;
  /**
   * Candidate search radius in the context's length unit. Edges farther away are not candidates. Values above
   * `2 * cellSize` are clamped to `2 * cellSize` (on longitude/latitude, to the context's length unit equivalent of
   * `2 * cellSize` degrees of latitude).
   */
  searchRadius: number;
  /**
   * A route, including the partial edges at both ends, may be at most
   * `routeFactor * straightDistance + routeSlack` long; longer routes are infeasible. Defaults to 3.
   */
  routeFactor?: number;
  /**
   * Additive route allowance in the context's length unit, see `routeFactor`. Defaults to 50 (50 m
   * in meters; pass for example 0.05 with a kilometer context).
   */
  routeSlack?: number;
};

/** Packs {@link GPUMapMatchingSettings} into the `float32` words read by {@link GPUMapMatching}. */
export function getGPUMapMatchingParameterValues(settings: GPUMapMatchingSettings): Float32Array {
  const spatialContext = settings.spatialContext;
  const values = [
    settings.sigma,
    settings.beta,
    settings.searchRadius,
    settings.routeFactor ?? 3,
    settings.routeSlack ?? 50
  ];
  const words = new Float32Array(GPU_MAP_MATCHING_PARAMETER_LENGTH);
  const names = ['sigma', 'beta', 'searchRadius', 'routeFactor', 'routeSlack'] as const;
  for (let index = 0; index < names.length; index++) {
    const name = names[index];
    const value = Math.fround(values[index]);
    validateFiniteNumber('GPUMapMatching', name, value, 0);
    if (index < 2)
      validateRequirement(
        'GPUMapMatching',
        value > 0,
        `${name} must be positive after float32 conversion; received ${values[index]}`
      );
    words[index] = value;
  }
  warnGPUShortLongitudeLatitudeDistance('GPUMapMatching', spatialContext, words[0]);
  warnGPUShortLongitudeLatitudeDistance('GPUMapMatching', spatialContext, words[2]);
  return words;
}

/** Caller-owned outputs of {@link GPUMapMatching}. */
export type GPUMapMatchingOutput = {
  /** CSR row of the matched edge per point, or {@link GPU_MAP_MATCHING_NONE}. */
  matchedEdges: GraphDataView<'uint32'>;
  /** Optional matched position along the edge from source (0) to target (1), or -1. */
  matchedFractions?: GraphDataView<'float32'>;
  /** Optional matched distance along the edge from its source, in the context's length unit, or -1. */
  matchedOffsets?: GraphDataView<'float32'>;
  /** Optional distance from the point to its matched edge in the context's length unit, or -1. */
  snapDistances?: GraphDataView<'float32'>;
  /** Optional snapped position, or the raw point when it is unmatched. */
  snappedPositions?: GraphDataView<'float32x2'>;
  /**
   * Per point: 0, or a break where a new matched sub-track starts. 1 is an ordinary break: no
   * feasible transition from any candidate of the previous point, or no candidate at all. 2 is a
   * budget break: no transition was found, but a route search into the point hit
   * `routeNodeBudget` and dropped nodes, so the missing route may be the budget's doing rather
   * than the data's (raise `routeNodeBudget` or sample more densely). First points of tracks are
   * not breaks.
   */
  breaks?: GraphDataView<'uint32'>;
  /**
   * Optional log-likelihood per track: the sum of the best final score of each unbroken stretch.
   * Scores are renormalized per step in `float32` and the per-step offsets use the shared 96-bit
   * fixed-point sum, so the value is accurate to about one `float32` rounding of its own magnitude
   * however long the track is.
   */
  trackLogLikelihoods?: GraphDataView<'float32'>;
  /**
   * Optional traversed route: the edges the vehicle passed along, including the intermediate
   * edges that no fix landed on. Requires {@link GPUMapMatchingProps.traversedEdgeCapacity}.
   */
  traversedEdges?: GPUMapMatchingTraversedEdges;
  /** Optional one-row count of points with a matched edge. */
  matchedCount?: GraphDataView<'uint32'>;
  /** Optional one-row count of points flagged in `breaks` (ordinary and budget breaks). */
  breakCount?: GraphDataView<'uint32'>;
  /** Optional GPU-written status words. */
  status?: {
    /** One-row flag: grid, route-search or traversal-recovery loss made results incomplete. */
    candidateOverflow?: GraphDataView<'uint32'>;
    /**
     * Final matched-row capacity overflow. Matched outputs have one slot per input point, so this
     * word is always zero. Intermediate grid/search loss is candidateOverflow.
     */
    overflow?: GraphDataView<'uint32'>;
    /**
     * Optional one-row count of the network nodes the route searches discovered but could not keep
     * in their `routeNodeBudget` tables during transition scoring, saturating at `0xffffffff`.
     * 0 means no scoring search dropped a node (up to f32 rounding); traversal status separately
     * reports recovery loss.
     */
    routeOverflow?: GraphDataView<'uint32'>;
  };
};

/**
 * Caller-owned traversed-route output of {@link GPUMapMatching}: a compact list of CSR edge rows
 * per track, in travel order.
 *
 * Within a track the list holds the matched edge of the first matched fix, then, per consecutive
 * matched fixes of one unbroken sub-track, the edges of the shortest route between them (found by
 * a fresh bounded search over the same network and bound that the model used) and the second fix's
 * edge. A fix on the edge of the previous fix and not behind it adds nothing, so consecutive
 * entries differ except after a loop. Between sub-tracks (after a break or a fix with no match)
 * the list holds one {@link GPU_MAP_MATCHING_NONE} entry, so a consumer that counts traversals
 * skips it and does not invent an adjacency across the break. An unrecovered hop also inserts
 * a separator and raises status.candidateOverflow. Dropped fixes
 * ({@link GPUMapMatchingProps.minimumFixSpacingSigmas}) contribute nothing. Hops are processed one
 * thread per track, serially, like the Viterbi passes.
 */
export type GPUMapMatchingTraversedEdges = {
  /**
   * `trackCount * traversedEdgeCapacity` rows; track `t` owns rows
   * `t * traversedEdgeCapacity` to `(t + 1) * traversedEdgeCapacity - 1`, of which the first
   * `min(counts[t], traversedEdgeCapacity)` are written. Other rows are not touched.
   */
  edges: GraphDataView<'uint32'>;
  /**
   * Entries the route of each track needs, including {@link GPU_MAP_MATCHING_NONE} separators and
   * entries past the capacity that were not stored, so `counts[t] > traversedEdgeCapacity` tells
   * how much capacity the track needed.
   */
  counts: GraphDataView<'uint32'>;
  /** Optional GPU-written status words. */
  status?: {
    /**
     * One-row flag: 1 only when a track needed more than `traversedEdgeCapacity` entries.
     */
    overflow?: GraphDataView<'uint32'>;
    /** One-row flag: grid/scoring/recovery loss made traversal incomplete. Missing hops
     * insert a NONE separator before the destination edge, so consumers never invent a turn. */
    candidateOverflow?: GraphDataView<'uint32'>;
  };
};

/**
 * A precomputed bounded cost table for {@link GPUMapMatchingProps.routeTable}: the output views of
 * a `GPUNetworkCostTable` built over the same network.
 */
export type GPUMapMatchingRouteTable = {
  /** `nodeCount + 1` row offsets of the table. */
  offsets: GraphDataView<'uint32'>;
  /** Target node per entry, ascending within each row. */
  targets: GraphDataView<'uint32'>;
  /** Network distance per entry in the context's length unit, aligned with `targets`. */
  costs: GraphDataView<'float32'>;
  /** Retained table's builtCostLimit, in the context's length unit. Rebuild when topology, edge lengths or the certified cutoff increase; never bind a mutable build-input cutoff. */
  costLimit: GraphDataView<'float32'>;
  /** Optional status words of the table. */
  status?: {
    /**
     * One-row completeness flag (`GPUNetworkCostTable` `output.status.complete`): nonzero means
     * converged without capacity overflow. Zero or omitted uses A* for every transition, so an
     * unfinished table never proves that a route is missing. A* reports its own budget limits.
     */
    complete?: GraphDataView<'uint32'>;
  };
};

/** Properties for {@link GPUMapMatching}. */
export type GPUMapMatchingProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'map-matching'`. */
  id?: string;
  /**
   * Coordinates of `positions` and `topology.nodePositions`. Omitted means native planar coordinates.
   * `{coordinateSpace: 'longitude-latitude', metric: 'great-circle'}` takes degrees and measures
   * every distance (emission, straight-line and route lengths, `sigma`, `beta`, `searchRadius`,
   * `routeSlack`) in the context's length unit (meters by default, or kilometers).
   * Local planar frames must be azimuthal-equidistant, not Lambert equal-area.
   */
  spatialContext?: GPUSpatialContext;
  /** GPS positions of every track, concatenated in time order. */
  positions: GraphDataView<'float32x2'>;
  /** Track-to-position offsets with `trackCount + 1` rows. */
  trackOffsets: GraphDataView<'uint32'>;
  /**
   * Road network. `nodePositions` are in the same coordinates as `positions`; the CSR row is the
   * edge ID and edge length is measured from the node positions, so `weights` are not read. Edges
   * with an out-of-range target are ignored.
   */
  topology: GPUNetworkAdjacency & {nodePositions: GraphDataView<'float32x2'>};
  /**
   * Per-frame parameters: `GPU_MAP_MATCHING_PARAMETER_LENGTH` `float32` words written with
   * {@link getGPUMapMatchingParameterValues}.
   */
  parameters: GraphDataView<'float32'>;
  /**
   * Compile-time candidates kept per point, 1 to 8. Defaults to 8. A candidate is a directed edge
   * row, so the two directions of a two-way road take two slots: the default keeps the 4 nearest
   * two-way roads (review B-17).
   */
  candidateCount?: number;
  /**
   * Compile-time edge grid cell size, in coordinate units (degrees on longitude/latitude). Choose
   * about the largest search radius.
   */
  cellSize: number;
  /**
   * Compile-time bounds of the edge grid, in coordinate units (nodes and points outside are
   * clamped). On longitude/latitude the maximum longitude may exceed 180 for a grid that crosses
   * the antimeridian.
   */
  bounds: {
    minimum: readonly [number, number];
    maximum: readonly [number, number];
  };
  /**
   * Compile-time capacity of the edge-cell table, at most {@link getGPUMapMatchingMaximumEntryCapacity}
   * (33554432 on a default device). Defaults to `16 * edgeCount + 256`, capped at that maximum;
   * entries beyond the capacity set `status.candidateOverflow`.
   */
  entryCapacity?: number;
  /**
   * Compile-time size of each route search's node table, 8 to 128. Defaults to 64. When a search
   * discovers more nodes, the table keeps the open nodes with the best A* keys and drops the rest,
   * so a route can be overestimated or missed; `status.candidateOverflow`, `status.routeOverflow` and budget breaks
   * (`breaks` value 2) report it. A search needs about as many nodes as the street blocks between
   * two samples: on a 100 m street grid, 64 covers samples up to about 800 m apart and 128 up to
   * about 1.1 km.
   */
  routeNodeBudget?: number;
  /**
   * Drops fixes closer than this many `sigma` to the previous retained fix of the track (Newson
   * and Krumm 2009, section 4: 2 is their value), measured with the `sigma` and the distance of
   * the frame's parameters. Without it a fix that jitters backward on a one-way edge has no
   * feasible transition and breaks the track. A dropped fix is never a candidate source or target
   * and is unmatched in every output (`matchedEdges` is {@link GPU_MAP_MATCHING_NONE}, `breaks`
   * is 0); the first fix of a track is always kept. Compile-time; omitted or 0 keeps every fix.
   */
  minimumFixSpacingSigmas?: number;
  /**
   * Compile-time number of edge rows reserved per track by `output.traversedEdges`, required with
   * it. The route of a long track needs one entry per edge passed plus one per sub-track break.
   */
  traversedEdgeCapacity?: number;
  /**
   * Optional precomputed cost table of a static network (UBODT, Yang and Gidofalvi 2018). A route
   * search whose bound `routeFactor * straight + routeSlack - tail` is within the table's
   * `costLimit` reads its network costs from the table row of the source edge's target node by
   * binary search, with no search and no node budget, so it is exact and never counts in
   * `status.routeOverflow`. Searches with a larger bound, or without a nonzero table `complete`,
   * run the bounded A* search as without a table, so there is no cliff and no extra result flag.
   *
   * Build the table with `GPUNetworkCostTable` over the same node numbering, with `weights` equal to
   * the edge lengths this contributor measures from the node positions (`networkDistance` of the
   * edge endpoints, in the context's length unit) and `costLimit` at least the
   * typical bound. Results equal the A*-only results up to f32 rounding of the summed route costs
   * (the table is the f32 fixpoint of the relaxations, A* sums along its own settle order: a few
   * `2^-24` relative per hop of the route), and are identical wherever A* was exact. Omitted, the
   * node list is unchanged.
   */
  routeTable?: GPUMapMatchingRouteTable;
  /** Caller-owned output. */
  output: GPUMapMatchingOutput;
};

/**
 * Hidden Markov model map matching (Newson and Krumm 2009) of GPS tracks onto a road network.
 *
 * Per point, up to `candidateCount` distinct edges within `searchRadius` are found through a uniform
 * edge grid built each encoding (integer atomics and a prefix sum) and kept in order of distance,
 * ties by edge row. Emission is Gaussian in the perpendicular distance (`sigma`). The transition
 * from a candidate of one point to a candidate of the next is exponential in
 * `|straight distance - route distance|` (`beta`), where the route runs along the first edge from
 * its position to the target node, over the network, then along the second edge to its position.
 * Network distances come from one bounded A* search per source candidate (Euclidean heuristic toward
 * the next point's candidate edges, so the search does not settle the whole disc that plain Dijkstra
 * would): the search stops at `routeFactor * straight + routeSlack` and keeps at most
 * `routeNodeBudget` nodes in a private table. A full table keeps the open nodes with the best keys,
 * so the search can still overestimate or miss a route when samples are far apart for the budget;
 * every such search is counted (`status.candidateOverflow`, `status.routeOverflow`), never silent. The route cost is the
 * edge length, and every edge row is directed: list both directions of a two-way road (each then
 * takes a candidate slot). CSR rows past `offsets[nodeCount]` are not part of the network.
 *
 * Viterbi runs with one thread per track, so tracks are parallel and the points of one track are
 * sequential. Where no candidate of a point is reachable from any candidate of the previous one,
 * the model restarts there (a break) instead of failing the track; a break where a route search
 * into the point dropped nodes is a budget break (`breaks` value 2). Results are deterministic
 * (integer atomics only, ties by index).
 *
 * Distances are planar by default. With a great-circle `spatialContext` positions are longitude and
 * latitude degrees and every distance is in the context's length unit: edge lengths and straight-line
 * distances are haversine, the closest point of an edge is found in the local equirectangular frame
 * of the GPS point, the A* heuristic is the great-circle distance (still consistent), and the
 * candidate window widens in longitude by `1 / cos` of the highest latitude it reaches and wraps
 * at +-180. Edges are straight segments between nodes, so they should be road-length. For a
 * network that crosses the antimeridian give `bounds` across it (for example longitudes 179.9 to
 * 180.1): longitudes are unwrapped around the grid center.
 *
 * Cap rationale: eight candidates bounds serial per-fix insertion selection and the K-by-K
 * Viterbi transitions; the storage-backed candidate/route tables alone do not make those stages
 * cooperative. Increasing K needs a measured cooperative selection/transition design, not just
 * larger private arrays. A route budget of 128 keeps the search tables, hash and reconstruction
 * below 8 KiB of private storage; the next power-of-two budget (256) exceeds it even at K=8.
 * A supplied routeTable already bypasses bounded searches when its cost limit covers the hop.
 * CPUMapMatchingOperation accepts more candidates; omitted CPU route budgets are uncapped and
 * explicit budgets bind. GPU defaults remain until a larger GPU model preserves candidate ties,
 * restart behavior and drop statuses.
 */
export class GPUMapMatching implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUMapMatchingProps;
  /** Number of GPS points. */
  readonly pointCount: number;
  /** Number of tracks. */
  readonly trackCount: number;
  /** Number of network nodes. */
  readonly nodeCount: number;
  /** Number of directed edges. */
  readonly edgeCount: number;
  /** Candidates per point. */
  readonly candidateCount: number;
  /** Edge grid width in cells. */
  readonly gridWidth: number;
  /** Edge grid height in cells. */
  readonly gridHeight: number;
  /** Resolved spatial context. */
  readonly spatialContext: GPUSpatialContext;

  constructor(props: GPUMapMatchingProps) {
    props = snapshotMapMatchingConstructionProps(props);
    this.id = props.id ?? 'map-matching';
    this.props = props;
    const {id, props: p} = {id: this.id, props};
    const {output} = props;
    this.spatialContext = resolveGPUSpatialContext(
      id,
      props.spatialContext,
      MAP_MATCHING_SPATIAL_CONTEXT_SUPPORT
    );
    const geodesicRadius = getGPUGreatCircleRadius(this.spatialContext);
    if (geodesicRadius !== undefined) {
      validateNonZeroFiniteNumber(id, 'spatialContext.sphereRadius', geodesicRadius);
    }
    const topologyShape = validateGPUNetworkTopology(id, props.topology, {
      nodePositions: true
    });
    validateSinglePackedView(props.positions, ['float32x2'], `${id} positions`);
    validateSinglePackedView(props.parameters, ['float32'], `${id} parameters`);
    for (const [name, view] of [
      ['trackOffsets', props.trackOffsets],
      ['output.matchedEdges', output.matchedEdges],
      ['output.breaks', output.breaks],
      ['output.matchedCount', output.matchedCount],
      ['output.breakCount', output.breakCount],
      ['output.status.overflow', output.status?.overflow],
      ['output.status.candidateOverflow', output.status?.candidateOverflow],
      ['output.status.routeOverflow', output.status?.routeOverflow],
      ['output.traversedEdges.edges', output.traversedEdges?.edges],
      ['output.traversedEdges.counts', output.traversedEdges?.counts],
      ['output.traversedEdges.status.overflow', output.traversedEdges?.status?.overflow],
      [
        'output.traversedEdges.status.candidateOverflow',
        output.traversedEdges?.status?.candidateOverflow
      ]
    ] as const) {
      if (view) {
        validateSinglePackedView(view, ['uint32'], `${id} ${name}`);
      }
    }
    for (const [name, view] of [
      ['output.matchedFractions', output.matchedFractions],
      ['output.matchedOffsets', output.matchedOffsets],
      ['output.snapDistances', output.snapDistances],
      ['output.trackLogLikelihoods', output.trackLogLikelihoods]
    ] as const) {
      if (view) {
        validateSinglePackedView(view, ['float32'], `${id} ${name}`);
      }
    }
    if (output.snappedPositions) {
      validateSinglePackedView(
        output.snappedPositions,
        ['float32x2'],
        `${id} output.snappedPositions`
      );
    }
    this.pointCount = props.positions.length;
    this.trackCount = props.trackOffsets.length - 1;
    this.nodeCount = topologyShape.nodeCount;
    this.edgeCount = topologyShape.edgeCount;
    validateMaximumCount(id, 'topology edges', this.edgeCount, MAP_MATCHING_MAXIMUM_EDGE_COUNT);
    this.candidateCount = props.candidateCount ?? 8;
    if (this.trackCount < 1) {
      throw new Error(`${id} trackOffsets needs at least one track`);
    }
    if (this.pointCount < 1 || this.edgeCount < 1 || this.nodeCount < 1) {
      throw new Error(`${id} needs at least one point, node and edge`);
    }
    if (props.parameters.length !== GPU_MAP_MATCHING_PARAMETER_LENGTH) {
      throw new Error(`${id} parameters must contain ${GPU_MAP_MATCHING_PARAMETER_LENGTH} rows`);
    }
    validateCapacity(
      id,
      'candidateCount',
      this.candidateCount,
      1,
      GPU_MAP_MATCHING_MAXIMUM_CANDIDATE_COUNT
    );
    validateCapacity(
      id,
      'routeNodeBudget',
      props.routeNodeBudget ?? 64,
      8,
      GPU_MAP_MATCHING_MAXIMUM_ROUTE_NODE_BUDGET
    );
    validateFiniteNumber(id, 'cellSize', props.cellSize);
    validateRequirement(
      id,
      Math.fround(props.cellSize) > 0,
      'cellSize must be positive in float32'
    );
    const [minX, minY] = props.bounds.minimum;
    const [maxX, maxY] = props.bounds.maximum;
    for (const [name, value] of [
      ['minimum.x', minX],
      ['minimum.y', minY],
      ['maximum.x', maxX],
      ['maximum.y', maxY]
    ] as const) {
      validateFiniteNumber(id, `bounds.${name}`, value);
    }
    if (!(maxX > minX) || !(maxY > minY)) {
      throw new Error(`${id} bounds must have a positive extent`);
    }
    this.gridWidth = Math.max(1, Math.ceil((maxX - minX) / props.cellSize));
    this.gridHeight = Math.max(1, Math.ceil((maxY - minY) / props.cellSize));
    validateMaximumCount(
      id,
      'cellSize grid cells',
      this.gridWidth * this.gridHeight,
      MAP_MATCHING_MAXIMUM_GRID_CELL_COUNT
    );
    if (props.entryCapacity !== undefined) {
      validateCapacity(id, 'entryCapacity', props.entryCapacity);
    }
    if (props.minimumFixSpacingSigmas !== undefined) {
      if (
        !(props.minimumFixSpacingSigmas >= 0) ||
        !Number.isFinite(props.minimumFixSpacingSigmas)
      ) {
        throw new Error(`${id} minimumFixSpacingSigmas must be a non-negative number`);
      }
      validateFiniteNumber(id, 'minimumFixSpacingSigmas', props.minimumFixSpacingSigmas, 0);
    }
    if (output.traversedEdges) {
      // A traversed route needs a capacity; the views are sized from it.
      validateCapacity(id, 'traversedEdgeCapacity', props.traversedEdgeCapacity ?? 0);
      const {edges, counts} = output.traversedEdges;
      const overflow = output.traversedEdges.status?.overflow;
      if (edges.length !== this.trackCount * props.traversedEdgeCapacity!) {
        throw new Error(
          `${id} output.traversedEdges.edges must contain trackCount * traversedEdgeCapacity rows`
        );
      }
      if (counts.length !== this.trackCount) {
        throw new Error(`${id} output.traversedEdges.counts must contain ${this.trackCount} rows`);
      }
      if (overflow && overflow.length !== 1) {
        throw new Error(`${id} output.traversedEdges.status.overflow must contain 1 rows`);
      }
      const candidateOverflow = output.traversedEdges.status?.candidateOverflow;
      if (candidateOverflow && candidateOverflow.length !== 1)
        throw new Error(`${id} output.traversedEdges.status.candidateOverflow must contain 1 rows`);
    }
    for (const [name, view, length] of [
      ['matchedEdges', output.matchedEdges, this.pointCount],
      ['matchedFractions', output.matchedFractions, this.pointCount],
      ['matchedOffsets', output.matchedOffsets, this.pointCount],
      ['snapDistances', output.snapDistances, this.pointCount],
      ['snappedPositions', output.snappedPositions, this.pointCount],
      ['breaks', output.breaks, this.pointCount],
      ['trackLogLikelihoods', output.trackLogLikelihoods, this.trackCount],
      ['matchedCount', output.matchedCount, 1],
      ['breakCount', output.breakCount, 1],
      ['status.overflow', output.status?.overflow, 1],
      ['status.candidateOverflow', output.status?.candidateOverflow, 1],
      ['status.routeOverflow', output.status?.routeOverflow, 1]
    ] as const) {
      if (view && view.length !== length) {
        throw new Error(`${id} output.${name} must contain ${length} rows`);
      }
    }
    void p;
    const {routeTable} = props;
    if (routeTable) {
      for (const [name, view] of [
        ['routeTable.offsets', routeTable.offsets],
        ['routeTable.targets', routeTable.targets],
        ['routeTable.status.complete', routeTable.status?.complete]
      ] as const) {
        if (view) {
          validateSinglePackedView(view, ['uint32'], `${id} ${name}`);
        }
      }
      validateSinglePackedView(routeTable.costs, ['float32'], `${id} routeTable.costs`);
      validateSinglePackedView(routeTable.costLimit, ['float32'], `${id} routeTable.costLimit`);
      if (routeTable.offsets.length !== this.nodeCount + 1) {
        throw new Error(`${id} routeTable.offsets must contain nodeCount + 1 rows`);
      }
      if (routeTable.costs.length !== routeTable.targets.length) {
        throw new Error(`${id} routeTable.costs length must equal routeTable.targets length`);
      }
      for (const [name, view] of [
        ['costLimit', routeTable.costLimit],
        ['status.complete', routeTable.status?.complete]
      ] as const) {
        if (view && view.length !== 1) {
          throw new Error(`${id} routeTable.${name} must contain exactly one row`);
        }
      }
    }
    const outputs = getOutputs(props);
    validateGraphOutputsDisjointFromInputs(id, outputs, getInputs(props));
    validateDisjointWritableViews(
      id,
      Object.fromEntries(outputs.map((view, index) => [`output ${index}`, view]))
    );
  }

  /**
   * Returns `clear-edge-table` and `edge-table`, the edge grid (`cell-count`, scan, `cell-fill`),
   * `candidates`, `routes`, the `forward` and `backward` Viterbi passes, `route-status`, and the
   * optional output nodes.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, pointCount, trackCount, nodeCount, edgeCount, candidateCount} = this;
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [...getInputs(props), ...getOutputs(props)]);
    validateGraphChunkBindingRanges(
      graph,
      id,
      Object.fromEntries(
        [...getInputs(props), ...getOutputs(props)].map((view, index) => [`view ${index}`, view])
      )
    );
    validateMaximumCount(
      id,
      'topology edges',
      edgeCount,
      getGPUMapMatchingMaximumEdgeCount(graph.device)
    );
    // Per-point scratch (candidates, routes, scores, hops) must fit one storage binding.
    validateMaximumCount(
      id,
      'positions rows',
      pointCount,
      getGPUMapMatchingMaximumPointCount(
        graph.device,
        candidateCount,
        Boolean(output.traversedEdges)
      )
    );
    if (output.traversedEdges) {
      validateMaximumCount(
        id,
        'trackCount * traversedEdgeCapacity',
        trackCount * props.traversedEdgeCapacity!,
        getStorageBindingElementLimit(graph.device, 4)
      );
    }
    const maximumEntryCapacity = getGPUMapMatchingMaximumEntryCapacity(graph.device);
    if (props.entryCapacity !== undefined) {
      validateCapacity(id, 'entryCapacity', props.entryCapacity, 1, maximumEntryCapacity);
    }
    const budget = props.routeNodeBudget ?? 64;
    const hasFixFilter = (props.minimumFixSpacingSigmas ?? 0) > 0;
    const {traversedEdges} = output;
    const cellCount = this.gridWidth * this.gridHeight;
    const entryCapacity =
      props.entryCapacity ?? Math.min(16 * edgeCount + 256, maximumEntryCapacity);
    // Check aligned caller bindings and every row-proportional scratch plane before declaring
    // any resource or node. A rejected graph can then be reused with a smaller batch or grid.
    for (const [name, byteLength] of Object.entries({
      edgeTable: edgeCount * 12,
      grid: cellCount * 4,
      entries: entryCapacity * 4,
      candidates: pointCount * candidateCount * 12,
      routes: pointCount * candidateCount * (candidateCount + 1) * 4,
      scores: pointCount * (candidateCount + 1) * 4,
      backPointers: pointCount * candidateCount * 4,
      points: pointCount * 4,
      ...(traversedEdges ? {hops: pointCount * HOP_WORD_COUNT * 4} : {}),
      ...Object.fromEntries(
        [...getInputs(props), ...getOutputs(props)]
          .filter(view => view !== undefined)
          .map((view, index) => [`view ${index}`, getViewBindingRange(view).size])
      )
    }))
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        name,
        byteLength,
        'reduce point count, entryCapacity or grid cells'
      );
    const [minX, minY] = props.bounds.minimum;
    const maxX = props.bounds.maximum[0];
    const geodesicRadius = getGPUGreatCircleRadius(this.spatialContext);
    const transient = <Format extends 'uint32' | 'float32'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, `${id}-${name}`, format, Math.max(length, 1));
    const read = (name: string, view: GraphDataView, type: WGSLKernelBinding['type'] = 'u32') =>
      ({name, view, type, access: 'read'}) as WGSLKernelBinding;
    const write = (name: string, view: GraphDataView, type: WGSLKernelBinding['type'] = 'u32') =>
      ({name, view, type, access: 'read_write'}) as WGSLKernelBinding;
    const common = `${SCALED_QUOTIENT_WGSL}
const POINT_COUNT: u32 = ${pointCount}u;
const NODE_COUNT: u32 = ${nodeCount}u;
const EDGE_COUNT: u32 = ${edgeCount}u;
const K: u32 = ${candidateCount}u;
// Route rows: K route lengths, then the number of nodes the route search dropped (an exact f32).
const ROUTE_STRIDE: u32 = ${candidateCount + 1}u;
const NONE: u32 = 0xffffffffu;
// Back pointer of a restart where a route search into the point dropped nodes: a budget break.
const BUDGET_BREAK: u32 = 0xfffffffeu;
// Predecessor of a fix dropped by the minimum fix spacing (NONE marks the first fix of a track).
const DROPPED_FIX: u32 = 0xfffffffeu;
// Score rows: K scores, then the offset subtracted from them (per-step renormalization).
const SCORE_STRIDE: u32 = ${candidateCount + 1}u;
const NEG: f32 = -1.0e30;
const GRID_WIDTH: u32 = ${this.gridWidth}u;
const GRID_HEIGHT: u32 = ${this.gridHeight}u;
const CELL_COUNT: u32 = ${cellCount}u;
const ENTRY_CAPACITY: u32 = ${entryCapacity}u;
const GRID_MINIMUM: vec2<f32> = vec2<f32>(${toLiteral(minX)}, ${toLiteral(minY)});
const CELL_SIZE: f32 = ${toLiteral(props.cellSize)};
const MAX_EDGE_SPAN: u32 = ${MAP_MATCHING_MAXIMUM_EDGE_CELL_SPAN}u;
fn getCellX(x: f32) -> u32 {
  return u32(clamp(floor(getScaledQuotient(x - GRID_MINIMUM.x, CELL_SIZE, 1.0, 0)), 0.0, f32(GRID_WIDTH - 1u)));
}
fn getCellY(y: f32) -> u32 {
  return u32(clamp(floor(getScaledQuotient(y - GRID_MINIMUM.y, CELL_SIZE, 1.0, 0)), 0.0, f32(GRID_HEIGHT - 1u)));
}
${getNetworkMetricWGSL(geodesicRadius)}
${getGridXWGSL(geodesicRadius !== undefined, minX, maxX)}`;
    const nodeAccess = `fn readNode(node: u32) -> vec2<f32> {
  return vec2<f32>(nodePositions[nodePositionsOffset + node * 2u], nodePositions[nodePositionsOffset + node * 2u + 1u]);
}`;
    const edgeTableAccess = `
fn edgeSource(edge: u32) -> u32 { return edgeTable[edgeTableOffset + edge * 3u]; }
fn edgeTarget(edge: u32) -> u32 { return edgeTable[edgeTableOffset + edge * 3u + 1u]; }
fn edgeLength(edge: u32) -> f32 { return bitcast<f32>(edgeTable[edgeTableOffset + edge * 3u + 2u]); }
fn isEdgeValid(edge: u32) -> bool { return edgeLength(edge) >= 0.0; }`;
    const kernel = (
      variant: string,
      bindings: WGSLKernelBinding[],
      invocationCount: number,
      body: string,
      declarations = ''
    ) =>
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-${variant}`,
        operation: OPERATION,
        variant,
        bindings,
        invocationCount,
        declarations: `${common}
${bindings.some(binding => binding.name === 'nodePositions') ? nodeAccess : ''}
${declarations}`,
        body
      });

    const edgeTable = transient('edge-table', 'uint32', edgeCount * 3);
    const cellCounts = transient('cell-counts', 'uint32', cellCount);
    const cellEnds = transient('cell-ends', 'uint32', cellCount);
    const cursors = transient('cursors', 'uint32', cellCount);
    const entries = transient('entries', 'uint32', entryCapacity);
    const candidates = transient('candidates', 'uint32', pointCount * candidateCount * 3);
    const pointStarts = transient('point-starts', 'uint32', pointCount);
    // Previous retained fix per point (NONE for the first of a track, DROPPED_FIX for a dropped one).
    const predecessors = transient('predecessors', 'uint32', pointCount);
    const routes = transient(
      'routes',
      'float32',
      pointCount * candidateCount * (candidateCount + 1)
    );
    const scores = transient('scores', 'float32', pointCount * (candidateCount + 1));
    const backPointers = transient('back-pointers', 'uint32', pointCount * candidateCount);
    const chosen = transient('chosen', 'uint32', pointCount);
    const breaks = output.breaks ?? transient('breaks', 'uint32', pointCount);
    // All fixed matched-row outputs fit. This scratch/port reports intermediate loss only.
    const overflow =
      output.status?.candidateOverflow ?? transient('candidate-overflow', 'uint32', 1);

    // With a route table: `route-bounds` stores, per (point, fromSlot) row, the route bound in
    // slot 0 and the search allowance after the source edge's tail in the trailing word when the
    // table can answer the row (allowance within its cost limit, complete table), else -1.
    // `route-table` then writes the whole row from the table and sets the trailing word to 0, which
    // the A* `routes` kernel reads as "resolved". Rows with -1 fall through to A*.
    const getRouteTableNodes = (): GPUCommandNode<Parameters>[] => {
      const {routeTable} = props;
      if (!routeTable?.status?.complete) {
        return [];
      }
      const tableComplete = routeTable.status.complete;
      return [
        kernel(
          'route-bounds',
          [
            hasFixFilter ? read('predecessors', predecessors) : read('pointStarts', pointStarts),
            read('points', props.positions, 'f32'),
            read('parameters', props.parameters, 'f32'),
            read('candidates', candidates),
            read('edgeTable', edgeTable),
            write('routes', routes, 'f32'),
            read('tableCostLimit', routeTable.costLimit, 'f32'),
            read('tableComplete', tableComplete)
          ],
          pointCount * candidateCount,
          `let point = index / K;
  let fromSlot = index % K;
  let base = routesOffset + index * ROUTE_STRIDE;
  routes[base + K] = -1.0;
  ${getPreviousFixWGSL(hasFixFilter)}
  if (tableComplete[tableCompleteOffset] == 0u) { return; }
  let fromEdge = getCandidateEdge(previous, fromSlot);
  if (fromEdge == NONE) {
    return;
  }
  let straight = networkDistance(readPoint(point), readPoint(previous));
  let bound = straight * parameters[parametersOffset + 3u] + parameters[parametersOffset + 4u];
  let tail = (1.0 - getCandidateFraction(previous, fromSlot)) * edgeLength(fromEdge);
  let remaining = bound - tail;
  if (!(remaining >= 0.0) || !(remaining <= tableCostLimit[tableCostLimitOffset]) ||
      edgeTarget(fromEdge) >= NODE_COUNT) {
    return;
  }
  routes[base] = bound;
  routes[base + K] = remaining;`,
          `${edgeTableAccess}
${getAccessDeclarations()}`
        ),
        kernel(
          'route-table',
          [
            read('candidates', candidates),
            read('edgeTable', edgeTable),
            write('routes', routes, 'f32'),
            read('tableOffsets', routeTable.offsets),
            read('tableTargets', routeTable.targets),
            read('tableCosts', routeTable.costs, 'f32'),
            ...(hasFixFilter ? [read('predecessors', predecessors)] : [])
          ],
          pointCount * candidateCount,
          getRouteTableBody(hasFixFilter),
          `${edgeTableAccess}
${getRouteTableDeclarations()}`
        )
      ];
    };

    const nodes: GPUCommandNode<Parameters>[] = [];
    if (output.status?.overflow)
      nodes.push(
        createFillNode<Parameters>(graph, {
          id: `${id}-clear-output-overflow`,
          operation: OPERATION,
          view: output.status.overflow,
          type: 'u32',
          value: '0u'
        })
      );
    // Source node, target node and length of every CSR row; -1 length marks an invalid edge. Every
    // word starts as the bits of -1.0 (an out-of-range node and a -1 length), so the rows past
    // offsets[NODE_COUNT], which the per-node kernel never writes, are invalid edges instead of
    // stale rows of an earlier frame or of an aliased transient.
    nodes.push(
      createFillNode<Parameters>(graph, {
        id: `${id}-clear-edge-table`,
        operation: OPERATION,
        view: edgeTable,
        type: 'u32',
        value: `${INVALID_EDGE_WORD}u`
      }),
      kernel(
        'edge-table',
        [
          read('adjacency', props.topology.offsets),
          read('edgeTargets', props.topology.neighbors),
          read('nodePositions', props.topology.nodePositions, 'f32'),
          write('edgeTable', edgeTable)
        ],
        nodeCount,
        `let end = min(adjacency[adjacencyOffset + index + 1u], EDGE_COUNT);
  let start = readNode(index);
  for (var edge = adjacency[adjacencyOffset + index]; edge < end; edge++) {
    let targetNode = edgeTargets[edgeTargetsOffset + edge];
    var length = -1.0;
    if (targetNode < NODE_COUNT) {
      length = networkDistance(start, readNode(targetNode));
    }
    edgeTable[edgeTableOffset + edge * 3u] = index;
    edgeTable[edgeTableOffset + edge * 3u + 1u] = targetNode;
    edgeTable[edgeTableOffset + edge * 3u + 2u] = bitcast<u32>(length);
  }`
      ),
      createFillNode<Parameters>(graph, {
        id: `${id}-clear-counts`,
        operation: OPERATION,
        view: cellCounts,
        type: 'u32',
        value: '0u'
      }),
      createFillNode<Parameters>(graph, {
        id: `${id}-clear-cursors`,
        operation: OPERATION,
        view: cursors,
        type: 'u32',
        value: '0u'
      }),
      createFillNode<Parameters>(graph, {
        id: `${id}-clear-overflow`,
        operation: OPERATION,
        view: overflow,
        type: 'u32',
        value: '0u'
      })
    );
    const cellRange = `let a = readNode(edgeSource(index));
  let b = readNode(edgeTarget(index));
  let gridXA = getGridX(a.x);
  let gridXB = getGridXNear(b.x, a.x, gridXA);
  let cellX0 = getCellX(min(gridXA, gridXB));
  let cellX1 = getCellX(max(gridXA, gridXB));
  let cellY0 = getCellY(min(a.y, b.y));
  let cellY1 = getCellY(max(a.y, b.y));`;
    nodes.push(
      kernel(
        'cell-count',
        [
          read('edgeTable', edgeTable),
          read('nodePositions', props.topology.nodePositions, 'f32'),
          write('cellCounts', cellCounts, 'atomic<u32>')
        ],
        edgeCount,
        `if (!isEdgeValid(index)) {
    return;
  }
  ${cellRange}
  for (var cellY = cellY0; cellY <= min(cellY1, cellY0 + MAX_EDGE_SPAN - 1u); cellY++) {
    for (var cellX = cellX0; cellX <= min(cellX1, cellX0 + MAX_EDGE_SPAN - 1u); cellX++) {
      atomicAdd(&cellCounts[cellCountsOffset + cellY * GRID_WIDTH + cellX], 1u);
    }
  }`,
        edgeTableAccess
      ),
      ...new GPUScan({
        id: `${id}-cell-scan`,
        input: cellCounts,
        output: cellEnds,
        mode: 'inclusive'
      }).getCommandNodes(graph),
      kernel(
        'cell-fill',
        [
          read('edgeTable', edgeTable),
          read('nodePositions', props.topology.nodePositions, 'f32'),
          read('cellEnds', cellEnds),
          write('cursors', cursors, 'atomic<u32>'),
          write('entries', entries),
          write('overflow', overflow, 'atomic<u32>')
        ],
        edgeCount,
        `if (!isEdgeValid(index)) {
    return;
  }
  ${cellRange}
  if (cellX1 - cellX0 >= MAX_EDGE_SPAN || cellY1 - cellY0 >= MAX_EDGE_SPAN) {
    atomicStore(&overflow[overflowOffset], 1u);
  }
  for (var cellY = cellY0; cellY <= min(cellY1, cellY0 + MAX_EDGE_SPAN - 1u); cellY++) {
    for (var cellX = cellX0; cellX <= min(cellX1, cellX0 + MAX_EDGE_SPAN - 1u); cellX++) {
      let cell = cellY * GRID_WIDTH + cellX;
      let start = select(0u, cellEnds[cellEndsOffset + cell - 1u], cell > 0u);
      let slot = start + atomicAdd(&cursors[cursorsOffset + cell], 1u);
      if (slot < ENTRY_CAPACITY) {
        entries[entriesOffset + slot] = index;
      } else {
        atomicStore(&overflow[overflowOffset], 1u);
      }
    }
  }`,
        edgeTableAccess
      ),
      // Per point: the K nearest distinct edges within the radius, sorted by (distance, edge row).
      kernel(
        'candidates',
        [
          read('points', props.positions, 'f32'),
          read('nodePositions', props.topology.nodePositions, 'f32'),
          read('edgeTable', edgeTable),
          read('cellEnds', cellEnds),
          read('entries', entries),
          read('parameters', props.parameters, 'f32'),
          write('candidates', candidates)
        ],
        pointCount,
        `let point = vec2<f32>(points[pointsOffset + index * 2u], points[pointsOffset + index * 2u + 1u]);
  ${getCandidateRadiusWGSL(geodesicRadius !== undefined)}
  var bestEdge: array<u32, ${candidateCount}>;
  var bestFraction: array<f32, ${candidateCount}>;
  var bestDistance: array<f32, ${candidateCount}>;
  for (var slot = 0u; slot < K; slot++) {
    bestEdge[slot] = NONE;
    bestFraction[slot] = 0.0;
    bestDistance[slot] = 0.0;
  }
  let cellY0 = getCellY(point.y - latitudeExtent);
  let cellY1 = getCellY(point.y + latitudeExtent);
  for (var windowIndex = 0u; windowIndex < WINDOW_COUNT; windowIndex++) {
  let longitudeRange = getLongitudeRange(getGridX(point.x), longitudeExtent, windowIndex);
  if (longitudeRange.x > longitudeRange.y) {
    continue;
  }
  let cellX0 = getCellX(longitudeRange.x);
  let cellX1 = getCellX(longitudeRange.y);
  for (var cellY = cellY0; cellY <= cellY1; cellY++) {
    for (var cellX = cellX0; cellX <= cellX1; cellX++) {
      let cell = cellY * GRID_WIDTH + cellX;
      let start = select(0u, cellEnds[cellEndsOffset + cell - 1u], cell > 0u);
      let end = min(cellEnds[cellEndsOffset + cell], ENTRY_CAPACITY);
      for (var entry = start; entry < end; entry++) {
        let edge = entries[entriesOffset + entry];
        let origin = readNode(edgeSource(edge));
        let segmentEnd = readNode(edgeTarget(edge));
        let fraction = networkSegmentFraction(point, origin, segmentEnd);
        let distance = networkSegmentDistance(point, origin, segmentEnd, fraction);
        if (!(distance <= radius)) {
          continue;
        }
        var isKnown = false;
        for (var slot = 0u; slot < K; slot++) {
          isKnown = isKnown || bestEdge[slot] == edge;
        }
        if (isKnown) {
          continue;
        }
        // Insertion into the sorted list; the last slot is dropped when full.
        var position = K;
        for (var slot = 0u; slot < K; slot++) {
          if (position == K && (bestEdge[slot] == NONE || distance < bestDistance[slot] ||
              (distance == bestDistance[slot] && edge < bestEdge[slot]))) {
            position = slot;
          }
        }
        if (position < K) {
          for (var slot = K - 1u; slot > position; slot--) {
            bestEdge[slot] = bestEdge[slot - 1u];
            bestFraction[slot] = bestFraction[slot - 1u];
            bestDistance[slot] = bestDistance[slot - 1u];
          }
          bestEdge[position] = edge;
          bestFraction[position] = fraction;
          bestDistance[position] = distance;
        }
      }
    }
  }
  }
  for (var slot = 0u; slot < K; slot++) {
    let base = candidatesOffset + (index * K + slot) * 3u;
    candidates[base] = bestEdge[slot];
    candidates[base + 1u] = bitcast<u32>(bestFraction[slot]);
    candidates[base + 2u] = bitcast<u32>(bestDistance[slot]);
  }`,
        `${edgeTableAccess}
${getLongitudeRangeWGSL(geodesicRadius !== undefined, maxX)}`
      ),
      ...(hasFixFilter
        ? [
            createFillNode<Parameters>(graph, {
              id: `${id}-clear-predecessors`,
              operation: OPERATION,
              view: predecessors,
              type: 'u32',
              value: '0xffffffffu'
            }),
            // Newson and Krumm preprocessing: a fix within the minimum spacing of the previous
            // retained fix is dropped; the next fix is measured from the retained one.
            kernel(
              'fix-filter',
              [
                read('trackOffsets', props.trackOffsets),
                read('points', props.positions, 'f32'),
                read('parameters', props.parameters, 'f32'),
                write('predecessors', predecessors)
              ],
              trackCount,
              `let first = trackOffsets[trackOffsetsOffset + index];
  let last = min(trackOffsets[trackOffsetsOffset + index + 1u], POINT_COUNT);
  let minimumSpacing = max(parameters[parametersOffset], 1.0e-6) * MINIMUM_FIX_SPACING_SIGMAS;
  var kept = NONE;
  for (var point = first; point < last; point++) {
    var predecessor = kept;
    if (kept != NONE && networkDistance(readFixPoint(point), readFixPoint(kept)) < minimumSpacing) {
      predecessor = DROPPED_FIX;
    } else {
      kept = point;
    }
    predecessors[predecessorsOffset + point] = predecessor;
  }`,
              `const MINIMUM_FIX_SPACING_SIGMAS: f32 = ${toLiteral(props.minimumFixSpacingSigmas!)};
fn readFixPoint(point: u32) -> vec2<f32> {
  return vec2<f32>(points[pointsOffset + point * 2u], points[pointsOffset + point * 2u + 1u]);
}`
            )
          ]
        : [
            createFillNode<Parameters>(graph, {
              id: `${id}-clear-starts`,
              operation: OPERATION,
              view: pointStarts,
              type: 'u32',
              value: '0u'
            }),
            kernel(
              'starts',
              [read('trackOffsets', props.trackOffsets), write('pointStarts', pointStarts)],
              trackCount,
              `let first = trackOffsets[trackOffsetsOffset + index];
  if (first < POINT_COUNT) {
    pointStarts[pointStartsOffset + first] = 1u;
  }`
            )
          ]),
      ...getRouteTableNodes(),
      kernel(
        'routes',
        [
          hasFixFilter ? read('predecessors', predecessors) : read('pointStarts', pointStarts),
          read('points', props.positions, 'f32'),
          read('nodePositions', props.topology.nodePositions, 'f32'),
          read('adjacency', props.topology.offsets),
          read('edgeTable', edgeTable),
          read('candidates', candidates),
          read('parameters', props.parameters, 'f32'),
          write('routes', routes, 'f32')
        ],
        pointCount * candidateCount,
        getRoutesBody(Boolean(props.routeTable?.status?.complete), hasFixFilter),
        getForwardDeclarations(budget, getHashSize(budget), candidateCount, edgeTableAccess)
      ),
      kernel(
        'forward',
        [
          read('trackOffsets', props.trackOffsets),
          read('points', props.positions, 'f32'),
          read('candidates', candidates),
          read('parameters', props.parameters, 'f32'),
          read('routes', routes, 'f32'),
          write('scores', scores, 'f32'),
          write('backPointers', backPointers),
          ...(hasFixFilter ? [read('predecessors', predecessors)] : [])
        ],
        trackCount,
        getForwardBody(hasFixFilter),
        getViterbiDeclarations(candidateCount)
      ),
      kernel(
        'backward',
        [
          read('trackOffsets', props.trackOffsets),
          read('scores', scores, 'f32'),
          read('backPointers', backPointers),
          write('chosen', chosen),
          write('breaks', breaks),
          ...(output.trackLogLikelihoods
            ? [write('logLikelihoods', output.trackLogLikelihoods, 'f32')]
            : []),
          ...(hasFixFilter ? [read('predecessors', predecessors)] : [])
        ],
        trackCount,
        getBackwardBody(Boolean(output.trackLogLikelihoods), hasFixFilter),
        getSumDeclarations()
      )
    );

    // Route-search drops raise candidateOverflow and accumulate routeOverflow.
    if (output.status?.routeOverflow) {
      nodes.push(
        createFillNode<Parameters>(graph, {
          id: `${id}-clear-route-overflow`,
          operation: OPERATION,
          view: output.status.routeOverflow,
          type: 'u32',
          value: '0u'
        })
      );
    }
    nodes.push(
      kernel(
        'route-status',
        [
          read('routes', routes, 'f32'),
          write('overflow', overflow, 'atomic<u32>'),
          ...(output.status?.routeOverflow
            ? [write('routeOverflow', output.status.routeOverflow, 'atomic<u32>')]
            : [])
        ],
        pointCount * candidateCount,
        `let dropped = u32(routes[routesOffset + index * ROUTE_STRIDE + K]);
  if (dropped == 0u) {
    return;
  }
  atomicStore(&overflow[overflowOffset], 1u);
  ${
    output.status?.routeOverflow
      ? `// Saturating sum: an add that wraps pins the total at the maximum, and every later add
  // that wraps from there pins it again.
  let previous = atomicAdd(&routeOverflow[routeOverflowOffset], dropped);
  if (previous + dropped < previous) {
    atomicMax(&routeOverflow[routeOverflowOffset], 0xffffffffu);
  }`
      : ''
  }`
      )
    );

    nodes.push(
      kernel(
        'matched',
        [
          read('candidates', candidates),
          read('chosen', chosen),
          read('edgeTable', edgeTable),
          write('matchedEdges', output.matchedEdges),
          ...(output.matchedFractions ? [write('fractions', output.matchedFractions, 'f32')] : []),
          ...(output.matchedOffsets ? [write('offsets', output.matchedOffsets, 'f32')] : []),
          ...(output.snapDistances ? [write('distances', output.snapDistances, 'f32')] : [])
        ],
        pointCount,
        `let slot = chosen[chosenOffset + index];
  let isMatched = slot != NONE;
  let base = candidatesOffset + (index * K + min(slot, K - 1u)) * 3u;
  let edge = candidates[base];
  matchedEdges[matchedEdgesOffset + index] = select(NONE, edge, isMatched);
  ${
    output.matchedFractions
      ? 'fractions[fractionsOffset + index] = select(-1.0, bitcast<f32>(candidates[base + 1u]), isMatched);'
      : ''
  }
  ${
    output.matchedOffsets
      ? `offsets[offsetsOffset + index] = select(-1.0, bitcast<f32>(candidates[base + 1u]) * edgeLength(select(0u, edge, isMatched)), isMatched);`
      : ''
  }
  ${
    output.snapDistances
      ? 'distances[distancesOffset + index] = select(-1.0, bitcast<f32>(candidates[base + 2u]), isMatched);'
      : ''
  }`,
        edgeTableAccess
      )
    );
    if (traversedEdges) {
      const hops = transient('hops', 'uint32', pointCount * HOP_WORD_COUNT);
      const edgeCapacity = props.traversedEdgeCapacity!;
      // Pack final output overflow and intermediate search loss into one binding (the search
      // already uses all eight storage bindings). Publish the caller-owned scalar ports later.
      const traversedStatus = transient('traversed-status', 'uint32', 2);
      nodes.push(
        createFillNode<Parameters>(graph, {
          id: `${id}-clear-traversed-overflow`,
          operation: OPERATION,
          view: traversedStatus,
          type: 'u32',
          value: '0u'
        }),
        // Per point, in parallel: the hop that leads to the point's matched edge from the previous
        // retained fix, as start node, target node, search allowance and end edge.
        kernel(
          'hops',
          [
            read('chosen', chosen),
            read('breaks', breaks),
            read('candidates', candidates),
            read('points', props.positions, 'f32'),
            read('parameters', props.parameters, 'f32'),
            read('edgeTable', edgeTable),
            hasFixFilter ? read('predecessors', predecessors) : read('pointStarts', pointStarts),
            write('hops', hops)
          ],
          pointCount,
          getHopsBody(hasFixFilter),
          `${edgeTableAccess}
${getAccessDeclarations()}`
        ),
        // Per track, serially: one bounded search per hop and the edges of its route, in order.
        kernel(
          'route-edges',
          [
            read('trackOffsets', props.trackOffsets),
            read('hops', hops),
            read('nodePositions', props.topology.nodePositions, 'f32'),
            read('adjacency', props.topology.offsets),
            read('edgeTable', edgeTable),
            write('traversedEdges', traversedEdges.edges),
            write('traversedCounts', traversedEdges.counts),
            write('traversedStatus', traversedStatus, 'atomic<u32>')
          ],
          trackCount,
          getRouteEdgesBody(),
          `const EDGE_CAPACITY: u32 = ${edgeCapacity}u;
${getRouteSearchDeclarations(budget, getHashSize(budget), candidateCount, edgeTableAccess)}
${getRouteEdgesDeclarations(budget)}`
        )
      );
      nodes.push(
        kernel(
          'traversed-status',
          [
            read('traversedStatus', traversedStatus),
            write('candidateOverflow', overflow, 'atomic<u32>'),
            ...(traversedEdges.status?.overflow
              ? [write('outputOverflow', traversedEdges.status.overflow)]
              : []),
            ...(traversedEdges.status?.candidateOverflow
              ? [write('outputCandidateOverflow', traversedEdges.status.candidateOverflow)]
              : [])
          ],
          1,
          `let lost = traversedStatus[traversedStatusOffset + 1u];
  if (lost != 0u) { atomicStore(&candidateOverflow[candidateOverflowOffset], 1u); }
  ${traversedEdges.status?.overflow ? 'outputOverflow[outputOverflowOffset] = traversedStatus[traversedStatusOffset];' : ''}
  ${traversedEdges.status?.candidateOverflow ? 'outputCandidateOverflow[outputCandidateOverflowOffset] = atomicLoad(&candidateOverflow[candidateOverflowOffset]);' : ''}`
        )
      );
    }
    if (output.snappedPositions) {
      nodes.push(
        kernel(
          'positions',
          [
            read('points', props.positions, 'f32'),
            read('candidates', candidates),
            read('chosen', chosen),
            read('edgeTable', edgeTable),
            read('nodePositions', props.topology.nodePositions, 'f32'),
            write('snappedPositions', output.snappedPositions, 'f32')
          ],
          pointCount,
          `var position = vec2<f32>(points[pointsOffset + index * 2u], points[pointsOffset + index * 2u + 1u]);
  let slot = chosen[chosenOffset + index];
  if (slot != NONE) {
    let base = candidatesOffset + (index * K + slot) * 3u;
    let edge = candidates[base];
    let origin = readNode(edgeSource(edge));
    position = networkInterpolate(origin, readNode(edgeTarget(edge)), bitcast<f32>(candidates[base + 1u]));
  }
  snappedPositions[snappedPositionsOffset + index * 2u] = position.x;
  snappedPositions[snappedPositionsOffset + index * 2u + 1u] = position.y;`,
          edgeTableAccess
        )
      );
    }
    if (output.matchedCount || output.breakCount) {
      const countViews = [output.matchedCount, output.breakCount].filter(view => view);
      for (const view of countViews) {
        nodes.push(
          createFillNode<Parameters>(graph, {
            id: `${id}-clear-${view === output.matchedCount ? 'matched' : 'break'}-count`,
            operation: OPERATION,
            view: view!,
            type: 'u32',
            value: '0u'
          })
        );
      }
      // Summary counts fold per workgroup, one global atomic per count per workgroup (rule 14).
      const counters: WorkgroupCounter[] = [];
      const updates: string[] = [];
      if (output.matchedCount) {
        updates.push(
          `if (chosen[chosenOffset + index] != NONE) { addCounter(${counters.length}u, 1u); }`
        );
        counters.push({target: 'matchedCount', word: 0});
      }
      if (output.breakCount) {
        updates.push(
          `if (breaks[breaksOffset + index] != 0u) { addCounter(${counters.length}u, 1u); }`
        );
        counters.push({target: 'breakCount', word: 0});
      }
      const summarySource = getWorkgroupCounterKernelSource({
        counters,
        rowCount: 'POINT_COUNT',
        rowSource: updates.join('\n  ')
      });
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-summary`,
          operation: OPERATION,
          variant: 'summary',
          bindings: [
            read('chosen', chosen),
            read('breaks', breaks),
            ...(output.matchedCount
              ? [write('matchedCount', output.matchedCount, 'atomic<u32>')]
              : []),
            ...(output.breakCount ? [write('breakCount', output.breakCount, 'atomic<u32>')] : [])
          ],
          invocationCount: pointCount,
          workgroupSize: WORKGROUP_COUNTER_WORKGROUP_SIZE,
          guardIndex: false,
          declarations: `${common}
${summarySource.declarations}`,
          body: summarySource.body
        })
      );
    }
    return nodes;
  }
}

/**
 * Declares the candidate `radius` (distance units) and the `latitudeExtent` and `longitudeExtent`
 * (coordinate units) of the window around `point`. A great-circle window widens in longitude by
 * `1 / cos` of the highest latitude of the band, up to every longitude near a pole.
 */
function getCandidateRadiusWGSL(isGeodesic: boolean): string {
  if (!isGeodesic) {
    return `let radius = min(parameters[parametersOffset + 2u], 2.0 * CELL_SIZE);
  let latitudeExtent = radius;
  let longitudeExtent = radius;`;
  }
  return `let radius = min(
    parameters[parametersOffset + 2u],
    2.0 * CELL_SIZE * GEODESIC_DEGREES_TO_RADIANS * NETWORK_EARTH_RADIUS
  );
  let latitudeExtent = getScaledQuotient(radius, NETWORK_EARTH_RADIUS, 1.0, 0) * GEODESIC_RADIANS_TO_DEGREES;
  let bandLatitude = min(abs(point.y) + latitudeExtent, 90.0);
  let longitudeExtent = min(latitudeExtent / max(geodesicCosLatitude(bandLatitude), 1.0e-6), 180.0);`;
}

/**
 * Declares `getGridX(x)`, the x of a coordinate in the edge grid, and
 * `getGridXNear(x, anchorX, anchorGridX)`, the grid x of `x` continuous with an anchor whose grid
 * x is known. Planar coordinates are used as they are. Longitudes are unwrapped to be continuous
 * around the grid center, so a grid whose bounds cross the antimeridian (for example 179.9 to
 * 180.1) buckets nodes on both sides, and longitude differences take the shorter way around.
 */
function getGridXWGSL(isGeodesic: boolean, minX: number, maxX: number): string {
  if (!isGeodesic) {
    return `fn getGridX(x: f32) -> f32 {
  return x;
}
fn getGridXNear(x: f32, anchorX: f32, anchorGridX: f32) -> f32 {
  return x;
}`;
  }
  return `const GRID_CENTER_X: f32 = ${toLiteral((minX + maxX) / 2)};
fn getGridX(x: f32) -> f32 {
  return GRID_CENTER_X + geodesicWrapLongitudeDelta(x - GRID_CENTER_X);
}
fn getGridXNear(x: f32, anchorX: f32, anchorGridX: f32) -> f32 {
  return anchorGridX + geodesicWrapLongitudeDelta(x - anchorX);
}`;
}

/**
 * Declares `WINDOW_COUNT` and `getLongitudeRange(x, extent, windowIndex)`, the grid x interval of
 * candidate window `windowIndex`, empty when `x > y`. Planar searches use one window. Great-circle
 * searches add the parts of the window beyond the grid's west and east edges, wrapped by 360
 * degrees, when they land inside a grid that spans the globe; the candidate list skips edges it
 * already holds, so overlapping windows are harmless.
 */
function getLongitudeRangeWGSL(isGeodesic: boolean, maxX: number): string {
  if (!isGeodesic) {
    return `const WINDOW_COUNT: u32 = 1u;
fn getLongitudeRange(x: f32, extent: f32, windowIndex: u32) -> vec2<f32> {
  return vec2<f32>(x - extent, x + extent);
}`;
  }
  return `const WINDOW_COUNT: u32 = 3u;
const GRID_MAXIMUM_X: f32 = ${toLiteral(maxX)};
fn getLongitudeRange(x: f32, extent: f32, windowIndex: u32) -> vec2<f32> {
  let low = x - extent;
  let high = x + extent;
  let empty = vec2<f32>(1.0, 0.0);
  if (windowIndex == 1u) {
    let isWrapped = low < GRID_MINIMUM.x && low + 360.0 < GRID_MAXIMUM_X;
    return select(empty, vec2<f32>(low + 360.0, GRID_MAXIMUM_X), isWrapped);
  }
  if (windowIndex == 2u) {
    let isWrapped = high > GRID_MAXIMUM_X && high - 360.0 > GRID_MINIMUM.x;
    return select(empty, vec2<f32>(GRID_MINIMUM.x, high - 360.0), isWrapped);
  }
  return vec2<f32>(low, high);
}`;
}

function toLiteral(value: number): string {
  const text = String(Math.fround(value));
  return /[.eE]/.test(text) ? text : `${text}.0`;
}

function getInputs(props: GPUMapMatchingProps): (GraphDataView | undefined)[] {
  return [
    props.positions,
    props.trackOffsets,
    ...getGPUNetworkTopologyViews(props.topology),
    props.parameters,
    props.routeTable?.offsets,
    props.routeTable?.targets,
    props.routeTable?.costs,
    props.routeTable?.costLimit,
    props.routeTable?.status?.complete
  ];
}

/** Returns every writable view of a map-matching contributor. */
function getOutputs(props: GPUMapMatchingProps): (GraphDataView | undefined)[] {
  const {output} = props;
  return [
    output.matchedEdges,
    output.matchedFractions,
    output.matchedOffsets,
    output.snapDistances,
    output.snappedPositions,
    output.breaks,
    output.trackLogLikelihoods,
    output.matchedCount,
    output.breakCount,
    output.status?.overflow,
    output.status?.candidateOverflow,
    output.status?.routeOverflow,
    output.traversedEdges?.edges,
    output.traversedEdges?.counts,
    output.traversedEdges?.status?.overflow,
    output.traversedEdges?.status?.candidateOverflow
  ];
}
