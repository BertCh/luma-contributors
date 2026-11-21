// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {validateSpatialOutputViews} from './spatial-output-validation';
import {
  createTransientView,
  doGraphDataViewsOverlap,
  GPUFlagOffsets,
  GPUScan,
  GPUSort,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createFillNode,
  createWGSLDispatchGate,
  createWGSLKernelNode
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {getBoundedMergeTreeSpliceRoundCount} from '../../analysis-core/kernels/parallel-merge-tree';
import {getSortedSegmentSumNodes} from '../../analysis-core/kernels/sorted-segment-sums';
import {
  getNextPowerOfTwo,
  validateCapacity,
  validateCommandNodeBudget,
  validateRequirement,
  validateGPUStorageBindingByteLength,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  resolveGPUTemporalContext,
  type GPUTemporalContext
} from '../../analysis-core/contracts/temporal-context';
import {GPU_HDBSCAN_PARAMETER_LENGTH} from './hierarchical-density-clustering-parameters';
import type {GPUCardinalityPort} from '../../analysis-core/contracts/index';
import {GPUNeighborSearch} from '../neighbor-search/gpu-neighbor-search';
import {GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH} from '../neighbor-search/neighbor-search-parameters';
import {createNeighborSearchDerivedParameterNodes} from '../neighbor-search/neighbor-search-derived-parameters';
import {getSortedFeatureBVHNodes} from '../spatial-join/spatial-join-passes';
import {
  createHierarchicalDensityAddNode,
  createHierarchicalDensityEdgeKeysNode,
  createHierarchicalDensityGateNode,
  createHierarchicalDensityInitNode,
  createHierarchicalDensityLabelsNode,
  createHierarchicalDensityResolveNode,
  createHierarchicalDensityTieNode,
  HIERARCHICAL_DENSITY_CLUSTER_WORDS,
  HIERARCHICAL_DENSITY_EDGE_WORDS,
  HIERARCHICAL_DENSITY_SELECTION_WORDS,
  HIERARCHICAL_DENSITY_STATE_WORDS,
  type HierarchicalDensityShape
} from './hierarchical-density-clustering-kernels';

import {
  getHierarchicalDensityCondensingNodes,
  getHierarchicalDensitySelectionNodes
} from './hierarchical-density-tree';

import {
  createHierarchicalDensityCoreNode,
  createHierarchicalDensityClusterInfoNode,
  createHierarchicalDensityContributionsNode,
  createHierarchicalDensityLeafFlagsNode,
  createHierarchicalDensityNodeDataNode,
  createHierarchicalDensityPointerInitNode,
  createHierarchicalDensitySearchNode,
  HDBSCAN_NODE_WORDS,
  HDBSCAN_DENDROGRAM_WORDS
} from './hdbscan-kernels';
import {getHDBSCANDendrogramNodes} from './hdbscan-tree';

/** Planar contexts suitable for distance measurements. */
const HIERARCHICAL_DENSITY_SPATIAL_CONTEXT_SUPPORT = Object.freeze({
  ...GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  measure: 'distance'
} as const);

const OPERATION = 'GPUHDBSCAN';

/** Upper bound for published BVH, scans, sorts and segment sums, which do not expose node counts. */
const GPU_HDBSCAN_COMMAND_NODE_ALLOWANCE = 2048;

/** Largest `minimumSamples` of {@link GPUHDBSCAN}, bounding the per-query private top-k distance list. */
export const GPU_HDBSCAN_MAXIMUM_SAMPLES = 32;

/**
 * Largest `positions` row count of {@link GPUHDBSCAN}, bounding the
 * compact BVH binding to 128 MiB. Tree planes and MST records are smaller than that binding;
 * all integer counts (including the four-tour-events-per-row scan) stay below uint32.
 */
export const GPU_HDBSCAN_MAXIMUM_ROW_COUNT = 2 ** 22;

/** Packed neighbour-search reference route retains the old cap because its CSR scratch scales as rows × samples. */
export const GPU_HDBSCAN_REFERENCE_MAXIMUM_ROW_COUNT = 2 ** 20;

/** Caller-owned outputs of {@link GPUHDBSCAN}. */
export type GPUHDBSCANOutput = {
  /** One label per point: compact cluster ID `0..k-1`, or `0xffffffff` for noise. */
  labels: GraphDataView<'uint32'>;
  /**
   * Optional core distance per point: the distance to its `minimumSamples`-th nearest neighbor,
   * the point itself included (+infinity for non-finite points or fewer neighbors).
   */
  coreDistances?: GraphDataView<'float32'>;
  /** Optional selected-cluster count. The selected clusters always fit the label output. */
  status?: Pick<Partial<GPUCardinalityPort>, 'count'>;
};

/**
 * Properties for {@link GPUHDBSCAN}.
 *
 * Per-frame (no recompile): the contents of `positions`. Compile-time: `positions.length`,
 * `minimumSamples`, `minimumClusterSize`, `coreDistanceSearch`, temporal/spatial contexts and which optional views are present.
 * With `times`, their contents and `parameters` are also per-frame.
 */
export type GPUHDBSCANProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'hierarchical-density-clustering'`. */
  id?: string;
  /** Planar only (the default); project longitude/latitude first. Other contexts throw. */
  spatialContext?: GPUSpatialContext;
  /**
   * Points, one row per point, at most {@link GPU_HDBSCAN_MAXIMUM_ROW_COUNT}.
   * Non-finite points are noise and take no part in the hierarchy.
   */
  positions: GraphDataView<'float32x2'>;
  /** Relative float32 time per point. Non-finite times are noise. Requires temporalContext and parameters. */
  times?: GraphDataView<'float32'>;
  /** Describes times; only float32-relative encoding is supported. No unit conversion. */
  temporalContext?: GPUTemporalContext;
  /** Per-frame [timeScale, 0, 0, 0], packed by getGPUHDBSCANParameterValues. */
  parameters?: GraphDataView<'float32'>;
  /**
   * Smallest condensed cluster, scikit-learn `min_cluster_size`, at least 2. Compile-time.
   */
  minimumClusterSize: number;
  /**
   * Neighbors of the core distance, scikit-learn `min_samples` (the point itself counts), from 1
   * to {@link GPU_HDBSCAN_MAXIMUM_SAMPLES}. Defaults to
   * `min(minimumClusterSize, 32)`, scikit-learn's default when that is at most 32. Compile-time.
   */
  minimumSamples?: number;
  /**
   * Compile-time exact core-distance route. Default 'bvh' keeps only a private top-k list per
   * point and reuses the MST's spatial BVH. 'neighbor-search' retains the packed CSR reference
   * route, limited to REFERENCE_MAXIMUM_ROW_COUNT and planar points without times.
   */
  coreDistanceSearch?: 'bvh' | 'neighbor-search';
  /** Caller-owned outputs. */
  output: GPUHDBSCANOutput;
};

/**
 * HDBSCAN density clustering of planar points, fully on the GPU, matching scikit-learn's
 * `HDBSCAN` with its defaults (`cluster_selection_method='eom'`, `allow_single_cluster=False`,
 * `cluster_selection_epsilon=0`, Euclidean metric) up to a relabeling of the clusters.
 *
 * Composition:
 * 1. Exact core distances from a bounded-private-memory top-k walk of the spatial BVH, with
 *    the point itself counted as the first sample. No rows × neighbours table is allocated.
 *    `coreDistanceSearch: 'neighbor-search'` retains the former packed CSR reference route.
 * 2. The exact minimum spanning tree over the mutual reachability distance
 *    `max(core a, core b, |a - b|)` by Boruvka: at most `ceil(log2 rows) + 1` unrolled rounds of
 *    the same kernels (rule 12), gated on the GPU component count (rule 13). Each point walks a
 *    Hilbert-sorted point BVH for its nearest point of another component, pruning nodes by
 *    `max(core, box distance)` and nodes whose leaves all lie in its component (one inclusive scan
 *    of leaf component changes per round). Edges are totally ordered by `(weight, low row, high
 *    row)`, three integer atomic-min passes pick each component's minimum, and lock-free
 *    union-find hooks the components.
 * 3. Compact four-word BVH records omit duplicate leaf maxima. A stable sort of the spanning tree by `(weight, low, high)` and the single-linkage dendrogram
 *    by bounded, GPU-gated atomic parent splicing, one invocation per edge. Euler-tour pointer doubling
 *    computes subtree sizes in logarithmic dispatch depth, including chain-shaped dendrograms.
 *    Parent/size/weight, children, tour successors and tour totals use separate storage planes;
 *    no tree binding exceeds the compact BVH binding at the maximum row count.
 * 4. Condensing by pointer jumping (`ceil(log2(2 rows)) + 1` rounds of one kernel): each big node's
 *    cluster start and each point's lowest node of at least `minimumClusterSize` points, where it
 *    falls out of its cluster; compact cluster IDs with `GPUFlagOffsets`.
 * 5. Cluster stability as a fixed-order segmented sum of the condensed-tree rows, then the
 *    excess-of-mass selection by heavy-path tree contraction in logarithmic-squared dispatch
 *    depth. GPU cursors and indirect gates control the levels and convergence. Ancestor pointer
 *    doubling suppresses selected descendants; a flag scan assigns the compact labels.
 *
 * Ties: with equal mutual reachability distances several spanning trees exist; this one breaks
 * ties by point rows, which can differ from scikit-learn's order and, rarely, the selected
 * clusters. Duplicate points (zero distance) get the largest float32 density instead of
 * scikit-learn's infinity.
 *
 * With `times`, ST-HDBSCAN uses `sqrt(dx² + dy² + (timeScale * dt)²)`, equivalent to
 * scikit-learn on `[x, y, timeScale * t]`. An indexed top-k walk of the same spatial BVH
 * computes core distances. Spatial box distance is a conservative lower bound for the scaled
 * metric, so the exact Boruvka search still prunes safely. Time does not prune boxes, which can
 * make spatially coincident events expensive. The hierarchy and EOM selection are unchanged.
 * This is a continuous scaled metric, with no hard temporal cutoff.
 *
 * Non-goals: `allow_single_cluster`, `cluster_selection_epsilon`, `'leaf'` selection, membership
 * probabilities, outlier scores, other metrics and 3D points.
 */
export class GPUHDBSCAN implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUHDBSCANProps;
  /** Resolved spatial context (planar). */
  readonly spatialContext: GPUSpatialContext;
  /** Resolved `minimumSamples`. */
  readonly minimumSamples: number;
  /** Number of unrolled Boruvka rounds. */
  readonly roundCount: number;
  readonly temporalContext?: GPUTemporalContext;
  /** Conservative bound including logarithmic-squared tree contraction. */
  readonly estimatedCommandNodeCount: number;

  constructor(props: GPUHDBSCANProps) {
    this.id = props.id ?? 'hierarchical-density-clustering';
    this.props = props;
    const {id} = this;
    this.spatialContext = resolveGPUSpatialContext(
      id,
      props.spatialContext,
      HIERARCHICAL_DENSITY_SPATIAL_CONTEXT_SUPPORT
    );
    validateSinglePackedView(props.positions, ['float32x2'], `${id} positions`);
    const rows = props.positions.length;
    if (props.times) {
      validateRequirement(id, props.temporalContext !== undefined, 'times require temporalContext');
      validateRequirement(id, props.parameters !== undefined, 'times require parameters');
      this.temporalContext = resolveGPUTemporalContext(id, props.temporalContext, {
        encodings: ['float32-relative']
      });
      validateSinglePackedView(props.times, ['float32'], `${id} times`);
      validateRequirement(
        id,
        props.times.length === rows,
        'times length must equal positions length'
      );
      validateSinglePackedView(props.parameters!, ['float32'], `${id} parameters`);
      validateRequirement(
        id,
        props.parameters!.length === GPU_HDBSCAN_PARAMETER_LENGTH,
        `parameters length must equal ${GPU_HDBSCAN_PARAMETER_LENGTH}`
      );
    } else {
      validateRequirement(
        id,
        props.temporalContext === undefined && props.parameters === undefined,
        'temporalContext and parameters require times'
      );
    }
    validateCapacity(id, 'positions rows', rows, 1, GPU_HDBSCAN_MAXIMUM_ROW_COUNT);
    validateRequirement(
      id,
      props.coreDistanceSearch === undefined ||
        props.coreDistanceSearch === 'bvh' ||
        props.coreDistanceSearch === 'neighbor-search',
      "coreDistanceSearch must be 'bvh' or 'neighbor-search'"
    );
    if (props.coreDistanceSearch === 'neighbor-search') {
      validateCapacity(
        id,
        'positions rows for coreDistanceSearch neighbor-search',
        rows,
        1,
        GPU_HDBSCAN_REFERENCE_MAXIMUM_ROW_COUNT
      );
      validateRequirement(
        id,
        !props.times,
        'coreDistanceSearch neighbor-search requires positions without times'
      );
    }
    validateCapacity(id, 'minimumClusterSize', props.minimumClusterSize, 2);
    this.minimumSamples =
      props.minimumSamples ?? Math.min(props.minimumClusterSize, GPU_HDBSCAN_MAXIMUM_SAMPLES);
    validateCapacity(id, 'minimumSamples', this.minimumSamples, 1, GPU_HDBSCAN_MAXIMUM_SAMPLES);
    const {output} = props;
    validateRequirement(
      id,
      !output.status || !('requiredCount' in output.status),
      'output.status.requiredCount was removed; use output.status.count for the selected-cluster count'
    );
    validateSinglePackedView(output.labels, ['uint32'], `${id} output.labels`);
    validateRequirement(
      id,
      !(output.labels.length !== rows),
      `output.labels length must equal positions length`
    );
    if (output.coreDistances) {
      validateSinglePackedView(output.coreDistances, ['float32'], `${id} output.coreDistances`);
      validateRequirement(
        id,
        !(output.coreDistances.length !== rows),
        `output.coreDistances length must equal positions length`
      );
    }
    if (output.status?.count) {
      validateSinglePackedView(output.status.count, ['uint32'], `${id} output.status.count`);
      validateCapacity(id, 'output.status.count length', output.status.count.length, 1);
    }
    validateSpatialOutputViews(
      id,
      {
        labels: output.labels,
        coreDistances: output.coreDistances,
        count: output.status?.count
      },
      []
    );
    const writableViews = [output.labels, output.coreDistances, output.status?.count].filter(
      (view): view is GraphDataView<'uint32'> | GraphDataView<'float32'> => Boolean(view)
    );
    for (let first = 0; first < writableViews.length; first++) {
      for (let second = first + 1; second < writableViews.length; second++) {
        validateRequirement(
          id,
          !doGraphDataViewsOverlap(writableViews[first], writableViews[second]),
          'output fields must have disjoint writable byte ranges'
        );
      }
    }
    this.roundCount = Math.ceil(Math.log2(Math.max(rows, 2))) + 1;
    // Each light level has at most ceil(log2(2 rows)) + 1 three-node chain rounds.
    // The linear allowance covers Boruvka and the other pointer-jump passes. Each bounded
    // splice round adds a gate and a work node.
    const treeRounds = Math.ceil(Math.log2(2 * Math.max(rows, 2))) + 1;
    this.estimatedCommandNodeCount =
      GPU_HDBSCAN_COMMAND_NODE_ALLOWANCE +
      64 * treeRounds +
      4 * treeRounds * treeRounds +
      2 * getBoundedMergeTreeSpliceRoundCount(2 * rows) +
      1;
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
    validateGraphOutputsDisjointFromInputs(id, writableViews, [
      props.positions,
      props.times,
      props.parameters
    ]);
  }

  /** Returns the k-nearest, Boruvka, dendrogram, condensing, stability, selection and label nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, minimumSamples} = this;
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.times,
      props.parameters,
      output.labels,
      output.coreDistances,
      output.status?.count
    ]);
    const rows = props.positions.length;
    // Complete preflight before declaring any resource or node. Caller views can have a
    // nonzero aligned-binding prefix even when their logical lengths fit the device limit.
    validateGraphChunkBindingRanges(graph, id, {
      positions: props.positions,
      times: props.times,
      parameters: props.parameters,
      labels: output.labels,
      coreDistances: output.coreDistances,
      count: output.status?.count
    });
    const leafCapacity = getNextPowerOfTwo(rows);
    const clusterCapacity = Math.floor((2 * rows) / props.minimumClusterSize) + 1;
    // These envelopes include every derived binding of the BVH, MST, merge-tree builder,
    // pointer doubling, EOM selection, scans, sorts and fixed-order segmented reduction.
    // Each buffer starts at byte zero, so its aligned binding range equals its byte length.
    for (const [name, byteLength] of [
      ['BVH node data', (2 * leafCapacity - 1) * HDBSCAN_NODE_WORDS * 4],
      ['BVH boxes and children', (2 * leafCapacity - 1) * 8],
      ['bounded reconstruction endpoint task planes', rows * 2 * 4],
      ['MST edges and candidates', rows * HIERARCHICAL_DENSITY_EDGE_WORDS * 4],
      ['dendrogram parent/size/weight', 2 * rows * HDBSCAN_DENDROGRAM_WORDS * 4],
      ['tree children, pointers and Euler-tour planes', rows * 4 * 4],
      ['condensed-tree selection and pointer scratch', clusterCapacity * 4 * 4],
      ['row scans, sorts and segment sums', Math.max(2 * rows + 1, leafCapacity) * 4],
      ['time data', (rows + 1) * 4],
      ['dispatch and state', 24]
    ] as const) {
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        name,
        byteLength,
        'lower positions rows'
      );
    }
    if (props.coreDistanceSearch === 'neighbor-search' && minimumSamples > 1) {
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        'neighbor parameter data',
        GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH * 4,
        'use a device with a larger storage binding limit'
      );
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        'neighbor distances, IDs and weights',
        rows * (minimumSamples - 1) * 4,
        'lower minimumSamples or positions rows'
      );
    }
    const u32 = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', Math.max(length, 1));
    const f32 = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'float32', Math.max(length, 1));
    const nodes: GPUCommandNode<Parameters>[] = [];

    // 1. Core distances: the self join lists the other points, so `minimumSamples - 1` of them
    // reach the self-inclusive `minimumSamples`-th distance. One sample needs no search.
    const otherSamples = minimumSamples - 1;
    let neighborOffsets: GraphDataView<'uint32'> | undefined;
    let neighborDistances: GraphDataView<'float32'> | undefined;
    if (otherSamples > 0 && props.coreDistanceSearch === 'neighbor-search') {
      const search = createNeighborSearchDerivedParameterNodes<Parameters>(graph, {
        id: `${id}-neighbors-setup`,
        operation: OPERATION,
        positions: props.positions,
        radius: Infinity
      });
      const slotCount = rows * otherSamples;
      neighborOffsets = u32('neighbor-offsets', rows + 1);
      neighborDistances = f32('neighbor-distances', slotCount);
      nodes.push(
        ...search.nodes,
        ...new GPUNeighborSearch({
          id: `${id}-neighbors`,
          mode: 'knn',
          neighborCount: otherSamples,
          positions: props.positions,
          parameters: search.parameters,
          gridSize: search.gridSize,
          output: {
            offsets: neighborOffsets,
            neighbors: u32('neighbor-ids', slotCount),
            weights: f32('neighbor-weights', slotCount),
            distances: neighborDistances,
            status: {overflow: u32('neighbor-overflow', 1)}
          }
        }).getCommandNodes(graph)
      );
    }
    // Pack the time scale beside the times to keep the MST search within eight bindings.
    // The subtraction precedes scaling, preserving precision near the temporal origin.
    const timeData = props.times ? f32('time-data', rows + 1) : undefined;
    if (props.times && props.parameters && timeData) {
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-time-data`,
          operation: OPERATION,
          variant: 'time-data',
          invocationCount: rows,
          bindings: [
            {name: 'times', view: props.times, type: 'f32', access: 'read'},
            {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
            {name: 'timeData', view: timeData, type: 'f32', access: 'read_write'}
          ],
          body: `let scale = parameters[parametersOffset];
  let validScale = scale >= 0.0 && (bitcast<u32>(scale) & 0x7f800000u) != 0x7f800000u;
  timeData[timeDataOffset + index] = select(bitcast<f32>(0x7fc00000u | (index & 0u)), times[timesOffset + index], validScale);
  if (index == 0u) { timeData[timeDataOffset + ${rows}u] = scale; }`
        })
      );
    }
    const state = u32('state', HIERARCHICAL_DENSITY_STATE_WORDS);
    const boxMinima = createTransientView(graph, `${id}-box-minima`, 'float32x2', rows);
    const boxMaxima = createTransientView(graph, `${id}-box-maxima`, 'float32x2', rows);
    const parents = u32('parents', rows);
    const coreDistances = output.coreDistances ?? f32('core-distances', rows);
    nodes.push(
      createFillStateNode(graph, `${id}-state-clear`, state),
      createHierarchicalDensityInitNode<Parameters>(graph, {
        id: `${id}-init`,
        shape: this.getShape(0, 0, 1),
        positions: props.positions,
        timeData,
        neighborOffsets,
        neighborDistances,
        boxMinima,
        boxMaxima,
        parents,
        coreDistances,
        state
      })
    );
    const {bvh, nodes: bvhNodes} = getSortedFeatureBVHNodes<Parameters>(
      graph,
      `${id}-points`,
      OPERATION,
      boxMinima,
      boxMaxima,
      leafCapacity,
      rows >= 2
    );
    const shape = this.getShape(leafCapacity, bvh.internalNodeCount, bvh.levelCount);
    const nodeData = u32('node-data', bvh.nodeCount * HDBSCAN_NODE_WORDS);
    nodes.push(
      ...bvhNodes,
      createHierarchicalDensityNodeDataNode<Parameters>(graph, {
        id: `${id}-node-data`,
        shape,
        nodeMinima: bvh.nodeMinima,
        nodeMaxima: bvh.nodeMaxima,
        leafIds: bvh.leafIds,
        nodeData
      })
    );

    if (otherSamples > 0 && props.coreDistanceSearch !== 'neighbor-search') {
      nodes.push(
        createHierarchicalDensityCoreNode(graph, {
          id: `${id}-${timeData ? 'space-time' : 'bvh'}-core`,
          shape,
          nodeData,
          positions: props.positions,
          timeData,
          coreDistances
        })
      );
    }

    // 2. Boruvka rounds, gated on the GPU component count.
    const components = u32('components', rows);
    const componentBest = u32('component-best', rows * HIERARCHICAL_DENSITY_EDGE_WORDS);
    const candidates = u32('candidates', rows * HIERARCHICAL_DENSITY_EDGE_WORDS);
    const changeFlags = u32('change-flags', leafCapacity);
    const changeScan = u32('change-scan', leafCapacity);
    const edges = u32('edges', rows * HIERARCHICAL_DENSITY_EDGE_WORDS);
    const dispatchGate = createWGSLDispatchGate<Parameters>(graph, {
      id: `${id}-round-gate`,
      operation: OPERATION,
      shapes: [[rows], [leafCapacity]]
    });
    const rowGate = dispatchGate.getGate(0);
    const leafGate = dispatchGate.getGate(1);
    for (let round = 0; round < this.roundCount; round++) {
      const suffix = `-${round}`;
      nodes.push(
        createHierarchicalDensityGateNode<Parameters>(graph, {
          id: `${id}-gate${suffix}`,
          state,
          dispatchBinding: dispatchGate.binding,
          armWGSL: dispatchGate.armWGSL,
          disarmWGSL: dispatchGate.disarmWGSL
        }),
        createHierarchicalDensityResolveNode<Parameters>(graph, {
          id: `${id}-resolve${suffix}`,
          shape,
          positions: props.positions,
          timeData,
          parents,
          components,
          componentBest,
          gate: rowGate
        }),
        createHierarchicalDensityLeafFlagsNode<Parameters>(graph, {
          id: `${id}-leaf-flags${suffix}`,
          shape,
          nodeData,
          components,
          changeFlags,
          gate: leafGate
        }),
        ...new GPUScan({
          id: `${id}-leaf-scan${suffix}`,
          input: changeFlags,
          output: changeScan,
          mode: 'inclusive'
        }).getCommandNodes(graph),
        createHierarchicalDensitySearchNode<Parameters>(graph, {
          id: `${id}-search${suffix}`,
          shape,
          nodeData,
          changeScan,
          positions: props.positions,
          timeData,
          coreDistances,
          components,
          candidates,
          componentBest,
          gate: rowGate
        }),
        createHierarchicalDensityTieNode<Parameters>(graph, {
          id: `${id}-tie-low${suffix}`,
          shape,
          word: 1,
          components,
          candidates,
          componentBest,
          gate: rowGate
        }),
        createHierarchicalDensityTieNode<Parameters>(graph, {
          id: `${id}-tie-high${suffix}`,
          shape,
          word: 2,
          components,
          candidates,
          componentBest,
          gate: rowGate
        }),
        createHierarchicalDensityAddNode<Parameters>(graph, {
          id: `${id}-add${suffix}`,
          shape,
          components,
          componentBest,
          parents,
          edges,
          state,
          gate: rowGate
        })
      );
    }

    // 3. Spanning tree in (weight, low, high) order and the single-linkage dendrogram.
    const edgeIdentity = u32('edge-identity', rows);
    let edgeOrder: GraphDataView<'uint32'> | undefined;
    for (const word of [1, 0, 2] as const) {
      const keys = u32(`edge-keys-${word}`, rows);
      const sortedKeys = u32(`edge-sorted-keys-${word}`, rows);
      const nextOrder = u32(`edge-order-${word}`, rows);
      nodes.push(
        createHierarchicalDensityEdgeKeysNode<Parameters>(graph, {
          id: `${id}-edge-keys-${word}`,
          shape,
          word,
          state,
          edges,
          order: edgeOrder,
          identity: edgeOrder ? undefined : edgeIdentity,
          keys
        }),
        ...new GPUSort({
          id: `${id}-edge-sort-${word}`,
          keys,
          values: edgeOrder ?? edgeIdentity,
          outputKeys: sortedKeys,
          outputValues: nextOrder
        }).getCommandNodes(graph)
      );
      edgeOrder = nextOrder;
    }
    const sortedEdgeOrder = edgeOrder as GraphDataView<'uint32'>;
    const nodeCount = 2 * rows;
    const dendrogram = u32('dendrogram', nodeCount * HDBSCAN_DENDROGRAM_WORDS);
    const dendrogramTree = getHDBSCANDendrogramNodes(graph, {
      id: `${id}-dendrogram`,
      shape,
      state,
      edges,
      edgeOrder: sortedEdgeOrder,
      dendrogram
    });
    nodes.push(...dendrogramTree.nodes);

    // 4. Condensing by pointer jumping and compact cluster IDs.
    const pointers = u32('pointers', nodeCount * 2);
    const startFlags = u32('start-flags', nodeCount);
    const clusterOffsets = u32('cluster-offsets', nodeCount);
    const clusterCount = u32('cluster-count', 1);
    nodes.push(
      createHierarchicalDensityPointerInitNode<Parameters>(graph, {
        id: `${id}-pointer-init`,
        shape,
        dendrogram,
        dendrogramChildren: dendrogramTree.children,
        pointers,
        startFlags
      })
    );
    nodes.push(
      ...getHierarchicalDensityCondensingNodes(graph, {
        id: `${id}-condensing`,
        rowCount: rows,
        pointers
      })
    );
    nodes.push(
      ...new GPUFlagOffsets({
        id: `${id}-cluster-offsets`,
        flags: startFlags,
        offsets: clusterOffsets,
        count: clusterCount
      }).getCommandNodes(graph)
    );

    // 5. Stability, excess-of-mass selection and labels.
    const clusterInfo = u32('cluster-info', clusterCapacity * HIERARCHICAL_DENSITY_CLUSTER_WORDS);
    const contributionKeys = u32('contribution-keys', nodeCount);
    const contributions = f32('contributions', nodeCount);
    const lowClusters = u32('low-clusters', rows);
    const stability = f32('stability', clusterCapacity);
    const selection = u32('selection', clusterCapacity * HIERARCHICAL_DENSITY_SELECTION_WORDS);
    const selectedCount = output.status?.count ?? u32('selected-count', 1);
    nodes.push(
      createHierarchicalDensityClusterInfoNode<Parameters>(graph, {
        id: `${id}-cluster-info`,
        shape,
        dendrogram,
        startFlags,
        clusterOffsets,
        pointers,
        clusterInfo
      }),
      createHierarchicalDensityContributionsNode<Parameters>(graph, {
        id: `${id}-contributions`,
        shape,
        dendrogram,
        startFlags,
        clusterOffsets,
        pointers,
        clusterInfo,
        contributionKeys,
        contributions,
        lowClusters
      }),
      ...getSortedSegmentSumNodes<Parameters>(graph, {
        id: `${id}-stability`,
        operation: OPERATION,
        segmentCount: clusterCapacity,
        segmentKeys: contributionKeys,
        reductions: [{name: 'stability', contributions, output: stability}]
      }),
      ...getHierarchicalDensitySelectionNodes<Parameters>(graph, {
        id: `${id}-selection`,
        shape,
        clusterCount,
        stability,
        clusterInfo,
        selection,
        selectedCount
      }),
      createHierarchicalDensityLabelsNode<Parameters>(graph, {
        id: `${id}-labels`,
        shape,
        lowClusters,
        selection,
        labels: output.labels
      })
    );
    return nodes;
  }

  private getShape(
    leafCapacity: number,
    internalNodeCount: number,
    levelCount: number
  ): HierarchicalDensityShape {
    const rows = this.props.positions.length;
    return {
      rowCount: rows,
      minimumSamples: this.minimumSamples,
      minimumClusterSize: this.props.minimumClusterSize,
      clusterCapacity: Math.floor((2 * rows) / this.props.minimumClusterSize) + 1,
      leafCapacity,
      internalNodeCount,
      levelCount
    };
  }
}

function createFillStateNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  state: GraphDataView<'uint32'>
): GPUCommandNode<Parameters> {
  return createFillNode<Parameters>(graph, {
    id,
    operation: OPERATION,
    view: state,
    type: 'u32',
    value: '0u'
  });
}
