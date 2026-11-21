// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  snapshotConstructionFields,
  snapshotConstructionSpatialContext
} from '../../analysis-core/kernels/construction-snapshot';
import {snapshotJoinGeometry, snapshotJoinPairs} from './spatial-join-construction-snapshot';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  getStorageBindingElementLimit,
  validatePowerOfTwo,
  validateCapacity,
  validateRequirement,
  validateFiniteNumber,
  validateSinglePackedView,
  validateGraphViewsBelongToGraph
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  createBoundsNode,
  createFeatureRingsNode,
  getSpatialJoinFeatureCount,
  getSpatialJoinGeometryViews,
  isSameSpatialJoinGeometry,
  validateSpatialJoinGeometry
} from './spatial-join-geometry';
import type {GPUSpatialJoinPrepared} from './gpu-spatial-join-prepared';
import {
  getFeatureBVHNodes,
  getNextPowerOfTwo,
  SPATIAL_JOIN_WGSL_HELPERS,
  validateDisjointOutputs
} from './spatial-join-passes';
import {type GPUSpatialJoinGeometry} from './spatial-join-types';

import {
  resolveGPUSpatialContext,
  resolveGPUSecondarySpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import type {GPUCompactPairPort} from '../../analysis-core/contracts/index';

import {
  getSpatialJoinCandidateNodes,
  type SpatialJoinCandidateTree
} from './spatial-join-candidates';
import {validateGPUCompactPairPort} from '../../analysis-core/contracts/index';
import {
  validateSpatialJoinBindings,
  validateSpatialJoinReadWriteBindings
} from './spatial-join-binding-preflight';
import {validateGraphChunkGroupKernelSupport} from '../../analysis-core/kernels/graph-chunk-groups';

const OPERATION = 'GPUSpatialJoinCandidates';
/** Compile-time distance is emitted as an f32 literal; larger JS values round to infinity. */
const GPU_SPATIAL_JOIN_CANDIDATES_MAXIMUM_DISTANCE = 3.4028234663852886e38;

/**
 * Largest candidate pair capacity (`output.leftRows.length`) of {@link GPUSpatialJoinCandidates} on
 * `device`: the scratch table of pairs holds one `uint32x2` (8 bytes) per pair in one storage
 * binding, so 2^24 on a default device (128 MiB bindings).
 */
export function getGPUSpatialJoinCandidatesMaximumPairCapacity(device: Device): number {
  return getStorageBindingElementLimit(device, 8);
}

/**
 * Properties for {@link GPUSpatialJoinCandidates}.
 *
 * Per-frame: the contents of every input buffer, including a view-backed `distance`. Topology:
 * view lengths, whether `distance` is a number or view, `leafCapacity`, the pair capacity and the
 * presence of `prepared`.
 */
export type GPUSpatialJoinCandidatesProps<Parameters = unknown> = {
  /** Prefix for generated node and transient IDs. Defaults to `'spatial-join-candidates'`. */
  id?: string;
  /**
   * Coordinate space and metric of both sides. Omitted means native planar coordinates, with
   * `distance` in coordinate units. `longitude-latitude` throws at construction: the box expansion
   * here is planar. For great-circle candidates use `GPUSpatialPredicateJoin` with `dwithin`.
   */
  spatialContext?: GPUSpatialContext;
  /**
   * Context of `right` when declared separately from `spatialContext`. Must be compatible
   * with `spatialContext` (datum, coordinate space, metric, units, planar frame).
   * Omitted means `spatialContext` describes both inputs.
   */
  rightSpatialContext?: GPUSpatialContext;
  /** Left features. Their bounding boxes probe the right-hand tree. */
  left: GPUSpatialJoinGeometry;
  /** Right features, indexed by a BVH unless `prepared` supplies one. */
  right: GPUSpatialJoinGeometry;
  /**
   * Optional prepared right-hand side over the same views as `right`. It must be added to the
   * graph before this contributor. Spatially sorted candidates are restored to source row order.
   */
  prepared?: GPUSpatialJoinPrepared<Parameters>;
  /**
   * Expand each left box by this planar distance before probing, which makes the candidates
   * of a `dwithin` join. A one-row float32 view is read every encoding; negative, NaN and infinite
   * values act as `0`. Defaults to `0` (boxes that overlap or touch).
   */
  distance?: number | GraphDataView<'float32'>;
  /** Power-of-two BVH leaf slots when no `prepared` handle is given. Defaults to the next power of two. */
  leafCapacity?: number;
  /**
   * Caller-owned candidate pairs sorted by `(left, right)`. `output.leftRows.length` is the
   * candidate capacity; `count` is clamped to it, `requiredCount` is the unclamped number and
   * `overflow` reports only pair-capacity loss. `candidateOverflow` reports BVH leaf loss and is
   * required when leafCapacity is below the right feature count. Both flags are rewritten on reuse.
   * The capacity is at most
   * {@link getGPUSpatialJoinCandidatesMaximumPairCapacity} (2^24 on a default device).
   */
  output: GPUCompactPairPort;
};

/**
 * The bounding-box candidate stage of a spatial join on its own: every `(left, right)` feature pair
 * whose bounding boxes overlap (after expanding the left box by `distance`), sorted by left row and
 * then right row.
 *
 * This is the pair table that `GPUSpatialPredicateJoin` refines with an exact predicate. Expose it
 * to share one BVH probe between relate, distance, intersection and clip kernels: the output has the
 * layout of {@link GPUCompactPairPort}, so downstream contributors read it like any join result.
 * It is conservative: a candidate pair may not intersect, and no intersecting pair is missing.
 * Features that are empty, or have non-finite coordinates, have inverted bounds and never match.
 */
export class GPUSpatialJoinCandidates<Parameters = unknown> implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSpatialJoinCandidatesProps<Parameters>;
  /** Number of left features. */
  readonly leftCount: number;
  /** Number of right features. */
  readonly rightCount: number;
  /** Candidate pair capacity, `output.leftRows.length`. */
  readonly candidateCapacity: number;
  /** Resolved BVH leaf capacity. */
  readonly leafCapacity: number;
  /** Resolved spatial context: native planar coordinates. */
  readonly spatialContext: GPUSpatialContext;

  constructor(props: GPUSpatialJoinCandidatesProps<Parameters>) {
    const snapshot = snapshotConstructionFields(props, [
      'id',
      'spatialContext',
      'rightSpatialContext',
      'left',
      'right',
      'prepared',
      'distance',
      'leafCapacity',
      'output'
    ]);
    props = Object.freeze({
      ...snapshot,
      left: snapshotJoinGeometry(snapshot.left),
      right: snapshotJoinGeometry(snapshot.right),
      output: snapshotJoinPairs(snapshot.output),
      ...(snapshot.spatialContext
        ? {spatialContext: snapshotConstructionSpatialContext(snapshot.spatialContext)}
        : {}),
      ...(snapshot.rightSpatialContext
        ? {rightSpatialContext: snapshotConstructionSpatialContext(snapshot.rightSpatialContext)}
        : {})
    });
    this.id = props.id ?? 'spatial-join-candidates';
    this.props = props;
    const {id} = this;
    this.spatialContext = resolveGPUSpatialContext(id, props.spatialContext);
    resolveGPUSecondarySpatialContext(id, 'right', props.rightSpatialContext, this.spatialContext);
    validateSpatialJoinGeometry(id, 'left', props.left, {allowMultipart: true});
    validateSpatialJoinGeometry(id, 'right', props.right, {allowMultipart: true});
    this.leftCount = getSpatialJoinFeatureCount(props.left);
    this.rightCount = getSpatialJoinFeatureCount(props.right);
    if (
      typeof props.distance === 'number' &&
      (!Number.isFinite(props.distance) || props.distance < 0)
    ) {
      throw new Error(`${id} distance must be finite and non-negative`);
    }
    if (typeof props.distance === 'number') {
      validateFiniteNumber(
        id,
        'distance',
        props.distance,
        0,
        GPU_SPATIAL_JOIN_CANDIDATES_MAXIMUM_DISTANCE
      );
    }
    if (props.distance !== undefined && typeof props.distance !== 'number') {
      validateSinglePackedView(props.distance, ['float32'], `${id} distance`);
      if (props.distance.length < 1) {
        throw new Error(`${id} distance must contain one float32 row`);
      }
    }
    const {output: pairs, prepared} = props;
    validateGPUCompactPairPort(id, pairs);
    for (const view of [
      pairs.leftRows,
      pairs.rightRows,
      pairs.status.count,
      pairs.status.overflow,
      pairs.status.candidateOverflow,
      pairs.status.requiredCount
    ]) {
      if (view) {
        validateSinglePackedView(view, ['uint32'], `${id} output`);
      }
    }
    if (pairs.leftRows.length < 1) {
      throw new Error(`${id} output.leftRows needs at least one row`);
    }
    this.candidateCapacity = pairs.leftRows.length;
    this.leafCapacity = prepared
      ? prepared.leafCapacity
      : (props.leafCapacity ?? getNextPowerOfTwo(Math.max(this.rightCount, 1)));
    validatePowerOfTwo(id, 'leafCapacity', this.leafCapacity);
    validateRequirement(
      id,
      this.leafCapacity >= this.rightCount || pairs.status.candidateOverflow !== undefined,
      'output.status.candidateOverflow is required when leafCapacity is below the right feature count'
    );
    if (prepared) {
      if (!isSameSpatialJoinGeometry(prepared.geometry, props.right)) {
        throw new Error(`${id} prepared must index the same geometry views as right`);
      }
    }
    validateDisjointOutputs(
      id,
      [
        ...getSpatialJoinGeometryViews(props.left),
        ...getSpatialJoinGeometryViews(props.right),
        typeof props.distance === 'number' ? undefined : props.distance,
        ...Object.values(
          prepared?.isDeclaredIn(props.left.positions.buffer.graph)
            ? prepared.storage
            : (prepared?.props.storage ?? {})
        )
      ],
      [
        pairs.leftRows,
        pairs.rightRows,
        pairs.status.count,
        pairs.status.overflow,
        pairs.status.candidateOverflow,
        pairs.status.requiredCount
      ],
      [
        {
          leftRows: pairs.leftRows,
          rightRows: pairs.rightRows,
          count: pairs.status.count,
          overflow: pairs.status.overflow,
          requiredCount: pairs.status.requiredCount
        },
        {candidateOverflow: pairs.status.candidateOverflow}
      ]
    );
    if (prepared)
      validateSpatialJoinReadWriteBindings(
        id,
        [
          (prepared.isDeclaredIn(props.left.positions.buffer.graph)
            ? prepared.storage
            : prepared.props.storage
          )?.overflow
        ],
        [
          pairs.leftRows,
          pairs.rightRows,
          pairs.status.count,
          pairs.status.overflow,
          pairs.status.candidateOverflow,
          pairs.status.requiredCount
        ]
      );
  }

  /** Returns bounds, optional BVH build, candidate probe and finalize nodes in order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, leftCount, rightCount, candidateCapacity} = this;
    const {left, prepared, output: pairs} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      ...getSpatialJoinGeometryViews(left),
      ...getSpatialJoinGeometryViews(props.right),
      left.sourceIds,
      props.right.sourceIds
    ]);
    if (prepared && !prepared.isDeclaredIn(graph))
      throw new Error(`${id} requires prepared to be added to the graph first`);
    const right = prepared
      ? {...prepared.getGeometry(graph), sourceIds: props.right.sourceIds}
      : props.right;
    // Keep the public pair-capacity diagnostic before checking the allocation envelope.
    validateCapacity(
      id,
      'output.leftRows.length',
      candidateCapacity,
      1,
      getGPUSpatialJoinCandidatesMaximumPairCapacity(graph.device)
    );
    const boundViews = [
      ...getSpatialJoinGeometryViews(left),
      ...getSpatialJoinGeometryViews(right),
      typeof props.distance === 'number' ? undefined : props.distance,
      pairs.leftRows,
      pairs.rightRows,
      pairs.status.count,
      pairs.status.overflow,
      pairs.status.candidateOverflow,
      pairs.status.requiredCount,
      ...Object.values(prepared?.storage ?? {})
    ];
    validateSpatialJoinBindings(
      graph,
      id,
      Object.fromEntries(boundViews.map((view, index) => [`binding[${index}]`, view])),
      {
        state: 16,
        'left bounds and feature rings': leftCount * 8,
        'right bounds and feature rings': prepared ? 0 : rightCount * 8,
        'BVH node bounds/children': prepared ? 0 : (2 * this.leafCapacity - 1) * 8,
        'BVH leaf IDs': prepared ? 0 : this.leafCapacity * 4,
        'candidate pairs': candidateCapacity * 8,
        'candidate probe counts/offsets/scan': leftCount * 4
      }
    );
    validateGraphChunkGroupKernelSupport(graph, id, 8);
    const nodes: GPUCommandNode<Parameters>[] = [];

    const leftMinima = createTransientView(graph, `${id}-left-minima`, 'float32x2', leftCount);
    const leftMaxima = createTransientView(graph, `${id}-left-maxima`, 'float32x2', leftCount);
    let leftFeatureRings: GraphDataView<'uint32x2'> | undefined;
    if (left.kind === 'polygons') {
      leftFeatureRings = createTransientView(
        graph,
        `${id}-left-feature-rings`,
        'uint32x2',
        leftCount
      );
      nodes.push(
        createFeatureRingsNode<Parameters>(graph, `${id}-left-rings`, {
          featureCount: leftCount,
          geometry: left,
          featureRings: leftFeatureRings
        })
      );
    }
    nodes.push(
      createBoundsNode<Parameters>(graph, `${id}-left-bounds`, {
        featureCount: leftCount,
        geometry: left,
        featureRings: leftFeatureRings,
        minima: leftMinima,
        maxima: leftMaxima
      })
    );

    let tree: SpatialJoinCandidateTree;
    let bvhOverflow: GraphDataView<'uint32'>;
    if (prepared) {
      tree = {
        nodeMinima: prepared.storage.nodeMinima,
        nodeMaxima: prepared.storage.nodeMaxima,
        leafIds: prepared.storage.leafIds,
        internalNodeCount: prepared.bvhInternalNodeCount,
        spatialSort: prepared.spatialSort
      };
      bvhOverflow = prepared.storage.overflow;
    } else {
      const rightMinima = createTransientView(graph, `${id}-right-minima`, 'float32x2', rightCount);
      const rightMaxima = createTransientView(graph, `${id}-right-maxima`, 'float32x2', rightCount);
      let rightFeatureRings: GraphDataView<'uint32x2'> | undefined;
      if (right.kind === 'polygons') {
        rightFeatureRings = createTransientView(
          graph,
          `${id}-right-feature-rings`,
          'uint32x2',
          rightCount
        );
        nodes.push(
          createFeatureRingsNode<Parameters>(graph, `${id}-right-rings`, {
            featureCount: rightCount,
            geometry: right,
            featureRings: rightFeatureRings
          })
        );
      }
      nodes.push(
        createBoundsNode<Parameters>(graph, `${id}-right-bounds`, {
          featureCount: rightCount,
          geometry: right,
          featureRings: rightFeatureRings,
          minima: rightMinima,
          maxima: rightMaxima
        })
      );
      const built = getFeatureBVHNodes(graph, id, rightMinima, rightMaxima, this.leafCapacity);
      nodes.push(...built.nodes);
      tree = {
        nodeMinima: built.bvh.nodeMinima as GraphDataView<'float32x2'>,
        nodeMaxima: built.bvh.nodeMaxima as GraphDataView<'float32x2'>,
        leafIds: built.bvh.leafIds,
        internalNodeCount: built.bvh.internalNodeCount
      };
      bvhOverflow = built.bvh.overflow;
    }

    const state = createTransientView(graph, `${id}-state`, 'uint32', 4);
    const candidatePairs = createTransientView(
      graph,
      `${id}-candidate-pairs`,
      'uint32x2',
      candidateCapacity
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-clear`,
        operation: OPERATION,
        variant: 'clear',
        bindings: [
          {name: 'state', view: state, type: 'u32', access: 'read_write'},
          {name: 'candidatePairs', view: candidatePairs, type: 'u32', access: 'read_write'}
        ],
        invocationCount: Math.max(4, candidateCapacity),
        declarations: SPATIAL_JOIN_WGSL_HELPERS,
        body: `if (index < 4u) { state[stateOffset + index] = 0u; }
  if (index < ${candidateCapacity}u) {
    candidatePairs[candidatePairsOffset + index * 2u] = NO_FEATURE;
    candidatePairs[candidatePairsOffset + index * 2u + 1u] = NO_FEATURE;
  }`
      })
    );
    nodes.push(
      ...getSpatialJoinCandidateNodes<Parameters>(graph, {
        id,
        leftCount,
        rightCount,
        leftMinima,
        leftMaxima,
        tree,
        candidateCapacity,
        margin: props.distance ?? 0,
        state,
        candidatePairs
      })
    );
    const scalars: [string, GraphDataView<'uint32'> | undefined][] = [
      ['pairsCount', pairs.status.count],
      ['pairsOverflow', pairs.status.overflow],
      ['pairsTotal', pairs.status.requiredCount]
    ];
    const bindings: WGSLKernelBinding[] = [
      {name: 'candidatePairs', view: candidatePairs, type: 'u32', access: 'read'},
      {name: 'leftRows', view: pairs.leftRows, type: 'u32', access: 'read_write'},
      {name: 'rightRows', view: pairs.rightRows, type: 'u32', access: 'read_write'},
      {name: 'state', view: state, type: 'u32', access: 'read'}
    ];
    for (const [name, view] of scalars) {
      if (view) {
        bindings.push({name, view, type: 'u32', access: 'read_write'});
      }
    }
    const has = (name: string) => scalars.some(([key, view]) => key === name && view);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-write-pairs`,
        operation: OPERATION,
        variant: 'write-pairs',
        bindings,
        invocationCount: candidateCapacity,
        declarations: `const CANDIDATE_CAPACITY: u32 = ${candidateCapacity}u;`,
        body: `leftRows[leftRowsOffset + index] = candidatePairs[candidatePairsOffset + index * 2u];
  rightRows[rightRowsOffset + index] = candidatePairs[candidatePairsOffset + index * 2u + 1u];
  if (index == 0u) {
    let total = state[stateOffset];
    let overflowValue = select(0u, 1u, total > CANDIDATE_CAPACITY);
    ${has('pairsCount') ? 'pairsCount[pairsCountOffset] = min(total, CANDIDATE_CAPACITY);' : ''}
    ${has('pairsOverflow') ? 'pairsOverflow[pairsOverflowOffset] = overflowValue;' : ''}
    ${has('pairsTotal') ? 'pairsTotal[pairsTotalOffset] = total;' : ''}
  }`
      })
    );
    if (pairs.status.candidateOverflow) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-candidate-status`,
          operation: OPERATION,
          variant: 'candidate-status',
          bindings: [
            {name: 'bvhOverflow', view: bvhOverflow, type: 'u32', access: 'read'},
            {
              name: 'candidateOverflow',
              view: pairs.status.candidateOverflow,
              type: 'u32',
              access: 'read_write'
            }
          ],
          invocationCount: 1,
          body: 'candidateOverflow[candidateOverflowOffset] = select(0u, 1u, bvhOverflow[bvhOverflowOffset] != 0u);'
        })
      );
    }
    return nodes;
  }
}
