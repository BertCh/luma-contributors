// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  GPUScan,
  GPUSort,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPUCompactOutput} from '../../analysis-core/kernels/gpu-contributor-types';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {
  createFillNode,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {getSortKeyBits} from '../../analysis-core/kernels/sorted-segment-sums';
import {
  FIXED_POINT_WGSL,
  getFixedPointHeadroomBits
} from '../../analysis-core/kernels/fixed-point-accumulator';
import {
  getStorageBindingElementLimit,
  validateCapacity,
  validateMaximumCount,
  validateSinglePackedView,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateCompactOutput,
  validatePowerOfTwo
} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {GPUNetworkAdjacency} from '../../analysis-core/network-topology';
import {
  getGPUNetworkTopologyViews,
  validateGPUNetworkTopology
} from '../../analysis-core/network-topology';
import {createPublishNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  DECODE_ORDERED_KEY_WGSL,
  GET_ORDERED_KEY_WGSL
} from '../../analysis-core/kernels/wgsl-common';
import {
  getWorkgroupCounterKernelSource,
  WORKGROUP_COUNTER_WORKGROUP_SIZE
} from '../network-workgroup-counters';

/**
 * Largest supported `groupCapacity`. A superedge key `source * groupCapacity + target` must stay
 * below `2^32`, and `65535^2 + 1` does.
 */
export const GPU_NETWORK_COARSENING_MAXIMUM_GROUP_CAPACITY = 65535;

/**
 * Largest superedge capacity (`output.edges.ids` rows) of {@link GPUNetworkCoarsening} with
 * `output.edgeWeights` on `device`: the 64-bit fixed-point weight accumulators hold 8 bytes per
 * superedge slot in one storage binding, so 16777216 on a default device (128 MiB bindings).
 */
export function getGPUNetworkCoarseningMaximumWeightedEdgeCapacity(device: Device): number {
  return getStorageBindingElementLimit(device, 8);
}

/** Number of words of the optional `summary` view. */
export const GPU_NETWORK_COARSENING_SUMMARY_LENGTH = 8;

/** Word indices of the `summary` view. */
export const GPU_NETWORK_COARSENING_SUMMARY_WORD = {
  /** Live nodes, including nodes whose label is `>= groupCapacity`. */
  liveNodeCount: 0,
  /** Live nodes whose label is `>= groupCapacity`; they are excluded from every group. */
  overflowedNodeCount: 1,
  /** Counted live edges (directed: live slots; undirected: live slots `u -> v` with `u <= v`). */
  countedEdgeCount: 2,
  /** Counted edges dropped because an endpoint label is `>= groupCapacity`. */
  droppedEdgeCount: 3,
  /** Counted edges whose endpoints share a valid group (self-loops included). */
  intraEdgeCount: 4,
  /** Counted edges between two distinct valid groups; the sum of every superedge count. */
  interEdgeCount: 5,
  /** Groups with at least one live node. */
  groupCount: 6,
  /** Distinct superedges, before clamping to the superedge capacity. */
  superedgeCount: 7
} as const;

/** Decoded `summary` words. */
export type GPUNetworkCoarseningSummary = {
  liveNodeCount: number;
  overflowedNodeCount: number;
  countedEdgeCount: number;
  droppedEdgeCount: number;
  intraEdgeCount: number;
  interEdgeCount: number;
  groupCount: number;
  superedgeCount: number;
};

/** Decodes the `summary` words read back from the view. */
export function decodeGPUNetworkCoarseningSummary(
  words: Uint32Array | ArrayLike<number>
): GPUNetworkCoarseningSummary {
  if (words.length < GPU_NETWORK_COARSENING_SUMMARY_LENGTH) {
    throw new Error('decodeGPUNetworkCoarseningSummary needs eight words');
  }
  const word = GPU_NETWORK_COARSENING_SUMMARY_WORD;
  return {
    liveNodeCount: words[word.liveNodeCount],
    overflowedNodeCount: words[word.overflowedNodeCount],
    countedEdgeCount: words[word.countedEdgeCount],
    droppedEdgeCount: words[word.droppedEdgeCount],
    intraEdgeCount: words[word.intraEdgeCount],
    interEdgeCount: words[word.interEdgeCount],
    groupCount: words[word.groupCount],
    superedgeCount: words[word.superedgeCount]
  };
}

/**
 * Properties for {@link GPUNetworkCoarsening}.
 *
 * Compile-time: node and slot counts, `directed`, `groupCapacity`, `fixedPointScale`, which
 * optional views exist and every output capacity. Per-frame: the contents of every input view, so
 * re-labelling or filtering a live network changes the summary on the next encoding.
 */
export type GPUNetworkCoarseningProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'network-coarsening'`. */
  id?: string;
  /**
   * Forward CSR network with at least one slot. Neighbor indices `>= nodeCount` are dead slots.
   * Optional `weights` are summed per superedge (missing weights are 1); optional `nodePositions`
   * (finite values) feed centroids and bounds.
   */
  topology: GPUNetworkAdjacency;
  /**
   * Treat slots as directed edges. Default false: the CSR lists each undirected edge in both
   * directions (a self-loop once), like `GPUNetworkStatistics`, and only the slot `u -> v` with
   * `u <= v` is counted, so each edge counts once. An undirected `edgeMask` should be symmetric.
   */
  directed?: boolean;
  /** Optional per-node mask, nonzero = live. */
  nodeMask?: GraphDataView<'uint32'>;
  /**
   * Optional per-slot mask aligned with `topology.neighbors`, nonzero = live. A slot `u -> v` is
   * live iff this mask (if any) is nonzero, both endpoints are live and `v < nodeCount`.
   */
  edgeMask?: GraphDataView<'uint32'>;
  /**
   * Group label per node (one `uint32` row per node), for example a community from label
   * propagation, a component or a spatial cell. Dense labels `< groupCapacity` are the group IDs.
   */
  labels: GraphDataView<'uint32'>;
  /** Optional per-node finite `float32` value column summed per group. */
  nodeValues?: GraphDataView<'float32'>;
  /**
   * Compile-time number of dense groups, `1..65535`. Every `groupXxx` output has exactly this many
   * rows. A live node with a label at or above it is excluded and raises `output.edges.overflow`.
   */
  groupCapacity: number;
  /**
   * Optional fixed unit of the 64-bit fixed-point accumulators used for weight, position and value
   * sums: a power of two, and each addend is rounded to a multiple of `1 / fixedPointScale`.
   * Omitted (the default), the unit of each family of sums follows the data, as in the shared
   * fixed-point accumulator: it is a power of two derived on the GPU from the largest term that
   * is summed, so small weights (kilometres, hours, probabilities) are not rounded away. Set it
   * only to pin the rounding step, for example 65536 for the step this contributor used before.
   */
  fixedPointScale?: number;
  /** Caller-owned outputs. */
  output: GPUNetworkCoarseningOutput;
};

/** GPU-written outputs of {@link GPUNetworkCoarsening}. */
export type GPUNetworkCoarseningOutput = {
  /** Exact-size (`groupCapacity` rows) per-group live node count. */
  groupNodeCount: GraphDataView<'uint32'>;
  /** Optional per-group count of intra-group counted edges (self-loops included). */
  groupIntraEdgeCount?: GraphDataView<'uint32'>;
  /** Optional per-group sum of intra-group edge weights. */
  groupIntraWeight?: GraphDataView<'float32'>;
  /** Optional per-group centroid `sum / count`; requires `topology.nodePositions`. Empty groups are 0. */
  groupCentroid?: GraphDataView<'float32x2'>;
  /** Optional per-group `[minX, minY, maxX, maxY]`; requires `topology.nodePositions`. Empty groups are 0. */
  groupBounds?: GraphDataView<'float32x4'>;
  /** Optional per-group sum of `nodeValues`; requires `nodeValues`. */
  groupValueSum?: GraphDataView<'float32'>;
  /**
   * Bounded superedge list: `ids` is the source group. The capacity is `ids.length`. Rows are
   * sorted ascending by `(source, target)`; undirected superedges have `source < target`. When the
   * list overflows the first rows in that order are kept.
   * `overflow` is also set by label overflow.
   */
  edges: GPUCompactOutput;
  /** Target group per superedge row, `edges.ids.length` rows. */
  edgeTargetGroups: GraphDataView<'uint32'>;
  /** Edge count per superedge row, `edges.ids.length` rows. */
  edgeCounts: GraphDataView<'uint32'>;
  /** Optional summed weight per superedge row, `edges.ids.length` rows. */
  edgeWeights?: GraphDataView<'float32'>;
  /** Optional {@link GPU_NETWORK_COARSENING_SUMMARY_LENGTH}-word diagnostic view. */
  summary?: GraphDataView<'uint32'>;
};

const OPERATION = 'GPUNetworkCoarsening';
const GROUP_ACC_STRIDE = 10;
/** Sorted slots folded per invocation when accumulating superedge counts and weights. */
const SEGMENT_RUN_LENGTH = 16;
const WORD = GPU_NETWORK_COARSENING_SUMMARY_WORD;
/**
 * Words of the internal statistics view after the summary words: the largest biased float32
 * exponent of the terms of each family of sums, from which the unit of that family follows.
 */
const SCALE_WORD = {
  positions: GPU_NETWORK_COARSENING_SUMMARY_LENGTH,
  values: GPU_NETWORK_COARSENING_SUMMARY_LENGTH + 1,
  weights: GPU_NETWORK_COARSENING_SUMMARY_LENGTH + 2
} as const;
const STATISTICS_LENGTH = GPU_NETWORK_COARSENING_SUMMARY_LENGTH + 3;
/** Biased float32 exponent of 1.0, the only term of an unweighted sum. */
const UNIT_WEIGHT_EXPONENT_BITS = 127;

/**
 * Shared WGSL for the 64-bit signed fixed-point accumulators. Quantizing, the unit and decoding
 * are those of the shared fixed-point accumulator (`FIXED_POINT_WGSL`: integer bit manipulation,
 * round to nearest with ties to even, correctly rounded decode). A sum is stored in two words
 * here instead of three, so the unit is 2^32 times the shared one: every term is below
 * 2^(63 - headroomBits) units and no sum of at most 2^headroomBits - 1 terms can overflow.
 */
function getFixedPointSource(saturates: boolean): string {
  // With a data-derived unit no term can leave the two stored words. With a fixed unit
  // (`fixedPointScale`) a large term can: it saturates at the float32 nearest 4e18 units, with
  // its sign, as this contributor's fixed scale always did, instead of wrapping into the sign
  // bit or vanishing. An infinite term saturates too; NaN adds nothing.
  const saturationUnits = BigInt(Math.fround(4.0e18));
  const quantizeSource = saturates
    ? `const SATURATION_LOW: u32 = ${saturationUnits & 0xffffffffn}u;
const SATURATION_HIGH: u32 = ${saturationUnits >> 32n}u;
fn quantize(value: f32, unitExponent: i32) -> vec2<u32> {
  let wide = quantizeFixedPoint(abs(value), unitExponent);
  var magnitude = wide.xy;
  let isInfinite = (bitcast<u32>(value) & 0x7fffffffu) == 0x7f800000u;
  if (isInfinite || wide.z != 0u || wide.y > SATURATION_HIGH ||
      (wide.y == SATURATION_HIGH && wide.x > SATURATION_LOW)) {
    magnitude = vec2<u32>(SATURATION_LOW, SATURATION_HIGH);
  }
  if (value < 0.0) {
    let low = ~magnitude.x + 1u;
    magnitude = vec2<u32>(low, ~magnitude.y + select(0u, 1u, low == 0u));
  }
  return magnitude;
}`
    : `fn quantize(value: f32, unitExponent: i32) -> vec2<u32> {
  return quantizeFixedPoint(value, unitExponent).xy;
}`;
  return `${FIXED_POINT_WGSL}
fn getSumUnitExponent(maximumExponentBits: u32, headroomBits: u32) -> i32 {
  return getFixedPointUnitExponent(maximumExponentBits, headroomBits) + 32;
}
${quantizeSource}
fn dequantize(low: u32, high: u32, unitExponent: i32) -> f32 {
  let signWord = select(0u, 0xffffffffu, (high & 0x80000000u) != 0u);
  return getFixedPointFloat(vec3<u32>(low, high, signWord), unitExponent);
}
${GET_ORDERED_KEY_WGSL}${DECODE_ORDERED_KEY_WGSL}`;
}

/** WGSL statements that add `vec2<u32>` `quantized` into the atomic array words `index`, `index + 1`. */
function getFixedPointAddSource(array: string, index: string, quantized: string): string {
  return `{
    let addendOld = atomicAdd(&${array}[${index}], ${quantized}.x);
    let addendCarry = select(0u, 1u, addendOld + ${quantized}.x < addendOld);
    atomicAdd(&${array}[${index} + 1u], ${quantized}.y + addendCarry);
  }`;
}

/**
 * Coarsens a CSR network by a node group label into a summary graph for zoomed-out compound
 * nodes: per-group node counts, centroids, bounds, value sums and intra-group edge statistics,
 * plus a bounded list of superedges (source group, target group, edge count, summed weight).
 *
 * Labels. Labels are arbitrary `uint32`, but the group ID is the label itself and must be below
 * the compile-time `groupCapacity` (at most 65535). A live node with a larger label is excluded
 * from every group and superedge, edges touching it are counted in `droppedEdgeCount`, and
 * `edges.overflow` is set. A GPU-side dense relabel was rejected: first-come hash slot assignment
 * is nondeterministic across runs, and a deterministic rank needs a sort of the vertices plus a
 * hash probe per edge endpoint, a large cost for the labels that matter (communities and
 * components below `nodeCount`, spatial cells indexed by the caller). Remap sparse labels first,
 * for example with a caller-owned lookup table.
 *
 * Exactness. Node, intra and edge counts are `u32` atomics, so they are integer exact. Every sum
 * (weights, positions, values) is accumulated as a two-word signed 64-bit fixed-point integer
 * (carry propagated between two `atomicAdd`s). Integer addition is order independent, so sums are
 * bitwise reproducible. The unit, the rounding of each addend (nearest, ties to even, by integer
 * bit manipulation) and the correctly rounded conversion back to `float32` are those of the
 * shared fixed-point accumulator (`analysis-core/kernels/fixed-point-accumulator.ts`); only the
 * storage is two words instead of three. By default the unit of each family of sums is
 * `2^(E + 1 + h - 63)`, where `2^(E + 1)` bounds the largest term that is summed (reduced on the
 * GPU over the live nodes, or over the counted slots between valid groups) and `h` is the bit
 * length of the node or slot count, so no sum can overflow. A term keeps all 24 significand bits
 * while it is at least `2^(h - 39)` of the largest; a smaller one is rounded to the unit, an
 * error of at most `2^(h - 63)` of the largest term per addend. With `fixedPointScale` the unit
 * is the fixed `1 / fixedPointScale` instead: each addend is rounded to that step and saturates,
 * with its sign, at the float32 nearest `4e18` steps (about `6.1e13` at 65536; an infinite addend
 * saturates too), and sums wrap at `2^63 / fixedPointScale` in total. Centroids divide the
 * decoded sum by the count. Bounds use ordered-integer `atomicMin`/`atomicMax`, so they are
 * exact. NaN inputs add nothing; with a data-derived unit infinite inputs add nothing either.
 *
 * Superedges use sort-and-segment: each counted inter-group slot gets the key `a * groupCapacity +
 * b` (`a < b` for undirected edges), keys sort with the stable radix `GPUSort`, a scan numbers the
 * segments, and counts and fixed-point weight sums accumulate per segment. The output order, the
 * kept prefix on overflow, the counts and the weights are all deterministic. A bounded hash table
 * was rejected because a full table drops keys depending on atomic race order.
 *
 * The contributor owns no GPU resources; every transient belongs to the graph. Every output word is
 * rewritten on every encoding.
 */
export class GPUNetworkCoarsening implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUNetworkCoarseningProps;
  /** Number of nodes, `topology.offsets.length - 1`. */
  readonly nodeCount: number;
  /** Number of dense groups. */
  readonly groupCapacity: number;
  /** Maximum number of superedge rows, `edges.ids.length`. */
  readonly edgeCapacity: number;

  constructor(props: GPUNetworkCoarseningProps) {
    this.id = props.id ?? 'network-coarsening';
    this.props = props;
    const {id} = this;
    const {output} = props;
    const topologyShape = validateGPUNetworkTopology(id, props.topology);
    validateSinglePackedView(props.labels, ['uint32'], `${id} labels`);
    this.nodeCount = topologyShape.nodeCount;
    const {nodeCount} = this;
    if (topologyShape.edgeCount < 1) {
      throw new Error(`${id} topology.neighbors must contain at least one slot`);
    }
    this.groupCapacity = props.groupCapacity;
    const groupCapacity = this.groupCapacity;
    validateCapacity(
      id,
      'groupCapacity',
      groupCapacity,
      1,
      GPU_NETWORK_COARSENING_MAXIMUM_GROUP_CAPACITY
    );
    if (props.fixedPointScale !== undefined) {
      validatePowerOfTwo(id, 'fixedPointScale', props.fixedPointScale, 1, 2 ** 24);
    }
    const rowChecks: [string, GraphDataView | undefined, number, readonly string[]][] = [
      ['nodeMask', props.nodeMask, nodeCount, ['uint32']],
      ['labels', props.labels, nodeCount, ['uint32']],
      ['nodeValues', props.nodeValues, nodeCount, ['float32']],
      ['edgeMask', props.edgeMask, props.topology.neighbors.length, ['uint32']],
      ['output.groupNodeCount', output.groupNodeCount, groupCapacity, ['uint32']],
      ['output.groupIntraEdgeCount', output.groupIntraEdgeCount, groupCapacity, ['uint32']],
      ['output.groupIntraWeight', output.groupIntraWeight, groupCapacity, ['float32']],
      ['output.groupCentroid', output.groupCentroid, groupCapacity, ['float32x2']],
      ['output.groupBounds', output.groupBounds, groupCapacity, ['float32x4']],
      ['output.groupValueSum', output.groupValueSum, groupCapacity, ['float32']]
    ];
    for (const [name, view, length, formats] of rowChecks) {
      if (!view) {
        continue;
      }
      validateSinglePackedView(view, formats as any, `${id} ${name}`);
      if (view.length !== length) {
        throw new Error(`${id} ${name} must contain exactly ${length} rows`);
      }
    }
    if ((output.groupCentroid || output.groupBounds) && !props.topology.nodePositions) {
      throw new Error(
        `${id} output.groupCentroid and output.groupBounds require topology.nodePositions`
      );
    }
    if (output.groupValueSum && !props.nodeValues) {
      throw new Error(`${id} output.groupValueSum requires nodeValues`);
    }
    validateCompactOutput(id, output.edges);
    this.edgeCapacity = output.edges.ids.length;
    if (this.edgeCapacity < 1) {
      throw new Error(`${id} output.edges.ids must contain at least one row`);
    }
    for (const [name, view, formats] of [
      ['output.edgeTargetGroups', output.edgeTargetGroups, ['uint32']],
      ['output.edgeCounts', output.edgeCounts, ['uint32']],
      ['output.edgeWeights', output.edgeWeights, ['float32']]
    ] as const) {
      if (!view) {
        continue;
      }
      validateSinglePackedView(view, formats as any, `${id} ${name}`);
      if (view.length !== this.edgeCapacity) {
        throw new Error(`${id} ${name} must match output.edges.ids length`);
      }
    }
    if (props.output.summary) {
      validateSinglePackedView(props.output.summary, ['uint32'], `${id} output.summary`);
      if (props.output.summary.length !== GPU_NETWORK_COARSENING_SUMMARY_LENGTH) {
        throw new Error(
          `${id} output.summary must contain exactly ${GPU_NETWORK_COARSENING_SUMMARY_LENGTH} rows`
        );
      }
    }
    validateGraphOutputsDisjointFromInputs(id, this.getOutputs(), this.getInputs());
  }

  private getInputs(): (GraphDataView | undefined)[] {
    const {props} = this;
    return [
      ...getGPUNetworkTopologyViews(props.topology),
      props.nodeMask,
      props.edgeMask,
      props.labels,
      props.nodeValues
    ];
  }

  private getOutputs(): (GraphDataView | undefined)[] {
    const {output} = this.props;
    return [
      output.groupNodeCount,
      output.groupIntraEdgeCount,
      output.groupIntraWeight,
      output.groupCentroid,
      output.groupBounds,
      output.groupValueSum,
      output.edges.ids,
      output.edges.status.count,
      output.edges.status.overflow,
      output.edges.status.requiredCount,
      output.edgeTargetGroups,
      output.edgeCounts,
      output.edgeWeights,
      output.summary
    ];
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, nodeCount, groupCapacity, edgeCapacity} = this;
    validateGraphViewsBelongToGraph(id, graph, [...this.getInputs(), ...this.getOutputs()]);
    if (props.output.edgeWeights) {
      validateMaximumCount(
        id,
        'output.edges.ids rows with output.edgeWeights',
        edgeCapacity,
        getGPUNetworkCoarseningMaximumWeightedEdgeCapacity(graph.device)
      );
    }
    const slotCount = props.topology.neighbors.length;
    const directed = Boolean(props.directed);
    // Unit of the fixed-point sums. Without `fixedPointScale` it follows the data: the largest
    // exponent of the terms that are summed is reduced on the GPU into a scale word. With
    // `fixedPointScale` it is the fixed `1 / fixedPointScale`.
    const usesDataUnit = props.fixedPointScale === undefined;
    const fixedUnitExponent = usesDataUnit ? 0 : -Math.log2(props.fixedPointScale as number);
    const needsWeightScale = usesDataUnit && Boolean(props.topology.weights);
    /** WGSL `i32` expression of the unit exponent of one family of sums. */
    const getUnitExponentSource = (
      scaleWord: number | undefined,
      maximumTermCount: number,
      statsAccess: 'atomic' | 'plain'
    ): string => {
      if (!usesDataUnit) {
        return `${fixedUnitExponent}`;
      }
      const exponentBits =
        scaleWord === undefined
          ? `${UNIT_WEIGHT_EXPONENT_BITS}u`
          : statsAccess === 'atomic'
            ? `atomicLoad(&stats[statsOffset + ${scaleWord}u])`
            : `stats[statsOffset + ${scaleWord}u]`;
      return `getSumUnitExponent(${exponentBits}, ${getFixedPointHeadroomBits(maximumTermCount)}u)`;
    };
    const getWeightUnitExponentSource = (statsAccess: 'atomic' | 'plain') =>
      getUnitExponentSource(
        props.topology.weights ? SCALE_WORD.weights : undefined,
        props.topology.neighbors.length,
        statsAccess
      );
    const sentinelKey = groupCapacity * groupCapacity;
    const hasPositions = Boolean(props.topology.nodePositions);
    const hasValues = Boolean(props.nodeValues);
    const hasIntra = Boolean(props.output.groupIntraEdgeCount || props.output.groupIntraWeight);
    const hasIntraWeight = Boolean(props.output.groupIntraWeight);
    const nodes: GPUCommandNode<Parameters>[] = [];
    const u32 = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', length);

    const effectiveLabels = u32('effective-labels', nodeCount);
    const groupAcc = u32('group-acc', groupCapacity * GROUP_ACC_STRIDE);
    const intraAcc = hasIntra ? u32('intra-acc', groupCapacity * 3) : undefined;
    // The summary words, then the scale words of the sums. The kernels that sum already bind this
    // view (and the device limit of storage buffers), so the scale words ride in it; the summary
    // words are copied to `output.summary` at the end.
    const stats = u32('stats', STATISTICS_LENGTH);
    const sortKeys = u32('sort-keys', slotCount);
    const sortIndices = u32('sort-indices', slotCount);
    const sortedKeys = u32('sorted-keys', slotCount);
    const sortedIndices = u32('sorted-indices', slotCount);
    const flags = u32('segment-flags', slotCount);
    const segmentScan = u32('segment-scan', slotCount);
    const segmentTotal = u32('segment-total', 1);
    const overflowFlag = u32('overflow-flag', 1);
    const segmentWeightAcc = props.output.edgeWeights
      ? u32('segment-weight-acc', edgeCapacity * 2)
      : undefined;

    const fill = (
      step: string,
      view: GraphDataView | undefined,
      type: 'u32' | 'f32' = 'u32',
      value = '0u'
    ) => {
      if (view) {
        nodes.push(
          createFillNode<Parameters>(graph, {
            id: `${id}-${step}`,
            operation: OPERATION,
            view,
            type,
            value
          })
        );
      }
    };
    fill('zero-stats', stats);
    fill('zero-group-counts', props.output.groupNodeCount);
    fill('zero-intra', intraAcc);
    fill('zero-edge-ids', props.output.edges.ids);
    fill('zero-edge-targets', props.output.edgeTargetGroups);
    fill('zero-edge-counts', props.output.edgeCounts);
    fill('zero-edge-weight-acc', segmentWeightAcc);
    // Slots past offsets[NODE_COUNT] belong to no row, so the per-row slots pass never writes them.
    // They take the sentinel key and sort last; otherwise stale words (transients alias) would
    // enter the sort as superedges (review B-09, library rule 11).
    fill('clear-sort-keys', sortKeys, 'u32', `${sentinelKey}u`);

    // Initialise the per-group accumulator: sums are 0, bound minima are the largest ordered key.
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-init-groups`,
        operation: OPERATION,
        variant: 'init-groups',
        bindings: [
          {
            name: 'groupAcc',
            view: groupAcc,
            type: 'u32',
            access: 'read_write'
          }
        ],
        invocationCount: groupCapacity,
        body: `let base = index * ${GROUP_ACC_STRIDE}u;
  for (var word = 0u; word < 6u; word++) {
    groupAcc[groupAccOffset + base + word] = 0u;
  }
  groupAcc[groupAccOffset + base + 6u] = 0xffffffffu;
  groupAcc[groupAccOffset + base + 7u] = 0xffffffffu;
  groupAcc[groupAccOffset + base + 8u] = 0u;
  groupAcc[groupAccOffset + base + 9u] = 0u;`
      })
    );

    // Scale words of the position and value sums: the largest exponent over the nodes that are
    // summed (live, with a label below the group capacity). A masked or overflowed node, however
    // large its coordinates, does not coarsen the unit.
    if (usesDataUnit && (hasPositions || hasValues)) {
      const scaleBindings: WGSLKernelBinding[] = [
        {name: 'labels', view: props.labels, type: 'u32', access: 'read'}
      ];
      if (props.nodeMask) {
        scaleBindings.push({name: 'vertexMask', view: props.nodeMask, type: 'u32', access: 'read'});
      }
      if (props.topology.nodePositions) {
        scaleBindings.push({
          name: 'positions',
          view: props.topology.nodePositions,
          type: 'f32',
          access: 'read'
        });
      }
      if (props.nodeValues) {
        scaleBindings.push({name: 'values', view: props.nodeValues, type: 'f32', access: 'read'});
      }
      scaleBindings.push({name: 'stats', view: stats, type: 'atomic<u32>', access: 'read_write'});
      const nodeScaleCounters = getWorkgroupCounterKernelSource({
        counters: [
          {target: 'stats', word: SCALE_WORD.positions, operation: 'max'},
          {target: 'stats', word: SCALE_WORD.values, operation: 'max'}
        ],
        rowCount: 'NODE_COUNT',
        rowSource: `${props.nodeMask ? 'if (vertexMask[vertexMaskOffset + index] == 0u) { return; }' : ''}
  if (labels[labelsOffset + index] >= GROUP_CAPACITY) {
    return;
  }
  ${
    hasPositions
      ? `maxCounter(0u, max(
    getFixedPointExponentBits(positions[positionsOffset + 2u * index]),
    getFixedPointExponentBits(positions[positionsOffset + 2u * index + 1u])));`
      : ''
  }
  ${hasValues ? 'maxCounter(1u, getFixedPointExponentBits(values[valuesOffset + index]));' : ''}`
      });
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-node-sum-scale`,
          operation: OPERATION,
          variant: 'node-sum-scale',
          bindings: scaleBindings,
          invocationCount: nodeCount,
          workgroupSize: WORKGROUP_COUNTER_WORKGROUP_SIZE,
          guardIndex: false,
          declarations: `const NODE_COUNT: u32 = ${nodeCount}u;
const GROUP_CAPACITY: u32 = ${groupCapacity}u;
${FIXED_POINT_WGSL}
${nodeScaleCounters.declarations}`,
          body: nodeScaleCounters.body
        })
      );
    }

    // Vertices: liveness, label validation, counts, position/value sums and bounds.
    const vertexBindings: WGSLKernelBinding[] = [
      {name: 'labels', view: props.labels, type: 'u32', access: 'read'}
    ];
    if (props.nodeMask) {
      vertexBindings.push({
        name: 'vertexMask',
        view: props.nodeMask,
        type: 'u32',
        access: 'read'
      });
    }
    if (props.topology.nodePositions) {
      vertexBindings.push({
        name: 'positions',
        view: props.topology.nodePositions,
        type: 'f32',
        access: 'read'
      });
    }
    if (props.nodeValues) {
      vertexBindings.push({
        name: 'values',
        view: props.nodeValues,
        type: 'f32',
        access: 'read'
      });
    }
    vertexBindings.push(
      {
        name: 'effectiveLabels',
        view: effectiveLabels,
        type: 'u32',
        access: 'read_write'
      },
      {
        name: 'groupCounts',
        view: props.output.groupNodeCount,
        type: 'atomic<u32>',
        access: 'read_write'
      },
      {
        name: 'groupAcc',
        view: groupAcc,
        type: 'atomic<u32>',
        access: 'read_write'
      },
      {name: 'stats', view: stats, type: 'atomic<u32>', access: 'read_write'}
    );
    // Summary counters fold per workgroup, one global atomic per word per workgroup (rule 14).
    const vertexCounters = getWorkgroupCounterKernelSource({
      counters: [
        {target: 'stats', word: WORD.liveNodeCount},
        {target: 'stats', word: WORD.overflowedNodeCount},
        {target: 'stats', word: WORD.groupCount}
      ],
      rowCount: 'NODE_COUNT',
      rowSource: `${props.nodeMask ? 'if (vertexMask[vertexMaskOffset + index] == 0u) {' : 'if (false) {'}
    effectiveLabels[effectiveLabelsOffset + index] = DEAD;
    return;
  }
  addCounter(0u, 1u);
  let label = labels[labelsOffset + index];
  if (label >= GROUP_CAPACITY) {
    effectiveLabels[effectiveLabelsOffset + index] = OVERFLOWED;
    addCounter(1u, 1u);
    return;
  }
  effectiveLabels[effectiveLabelsOffset + index] = label;
  let previous = atomicAdd(&groupCounts[groupCountsOffset + label], 1u);
  if (previous == 0u) {
    addCounter(2u, 1u);
  }
  let base = label * ${GROUP_ACC_STRIDE}u;
  ${
    hasPositions
      ? `let position = vec2<f32>(positions[positionsOffset + 2u * index], positions[positionsOffset + 2u * index + 1u]);
  let positionUnitExponent = ${getUnitExponentSource(SCALE_WORD.positions, nodeCount, 'atomic')};
  let quantizedX = quantize(position.x, positionUnitExponent);
  let quantizedY = quantize(position.y, positionUnitExponent);
  ${getFixedPointAddSource('groupAcc', 'groupAccOffset + base', 'quantizedX')}
  ${getFixedPointAddSource('groupAcc', 'groupAccOffset + base + 2u', 'quantizedY')}
  atomicMin(&groupAcc[groupAccOffset + base + 6u], getOrderedKey(position.x));
  atomicMin(&groupAcc[groupAccOffset + base + 7u], getOrderedKey(position.y));
  atomicMax(&groupAcc[groupAccOffset + base + 8u], getOrderedKey(position.x));
  atomicMax(&groupAcc[groupAccOffset + base + 9u], getOrderedKey(position.y));`
      : ''
  }
  ${
    hasValues
      ? `let quantizedValue = quantize(values[valuesOffset + index], ${getUnitExponentSource(SCALE_WORD.values, nodeCount, 'atomic')});
  ${getFixedPointAddSource('groupAcc', 'groupAccOffset + base + 4u', 'quantizedValue')}`
      : ''
  }`
    });
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-vertices`,
        operation: OPERATION,
        variant: 'vertices',
        bindings: vertexBindings,
        invocationCount: nodeCount,
        workgroupSize: WORKGROUP_COUNTER_WORKGROUP_SIZE,
        guardIndex: false,
        declarations: `const NODE_COUNT: u32 = ${nodeCount}u;
const GROUP_CAPACITY: u32 = ${groupCapacity}u;
const DEAD: u32 = 0xffffffffu;
const OVERFLOWED: u32 = 0xfffffffeu;
${getFixedPointSource(!usesDataUnit)}
${vertexCounters.declarations}`,
        body: vertexCounters.body
      })
    );
    // Scale word of the weight sums: the largest exponent over the slots whose weight is summed
    // (the counted slots between two valid groups, by the rules of the slots kernel below). Slots
    // past the last row, masked slots and sentinel weights do not coarsen the unit.
    const sumsWeights = hasIntraWeight || Boolean(props.output.edgeWeights);
    if (needsWeightScale && sumsWeights && props.topology.weights) {
      const weightScaleBindings: WGSLKernelBinding[] = [
        {name: 'offsets', view: props.topology.offsets, type: 'u32', access: 'read'},
        {name: 'neighbors', view: props.topology.neighbors, type: 'u32', access: 'read'},
        {name: 'effectiveLabels', view: effectiveLabels, type: 'u32', access: 'read'},
        {name: 'weights', view: props.topology.weights, type: 'f32', access: 'read'}
      ];
      if (props.edgeMask) {
        weightScaleBindings.push({
          name: 'edgeMask',
          view: props.edgeMask,
          type: 'u32',
          access: 'read'
        });
      }
      weightScaleBindings.push({
        name: 'stats',
        view: stats,
        type: 'atomic<u32>',
        access: 'read_write'
      });
      const weightScaleCounters = getWorkgroupCounterKernelSource({
        counters: [{target: 'stats', word: SCALE_WORD.weights, operation: 'max'}],
        rowCount: 'NODE_COUNT',
        rowSource: `let sourceLabel = effectiveLabels[effectiveLabelsOffset + index];
  if (sourceLabel >= GROUP_CAPACITY) {
    return;
  }
  let rowEnd = min(offsets[offsetsOffset + index + 1u], SLOT_COUNT);
  let rowBegin = min(offsets[offsetsOffset + index], rowEnd);
  var exponentBits = 0u;
  for (var slot = rowBegin; slot < rowEnd; slot++) {
    let targetVertex = neighbors[neighborsOffset + slot];
    if (targetVertex >= NODE_COUNT) {
      continue;
    }
    var live = effectiveLabels[effectiveLabelsOffset + targetVertex] < GROUP_CAPACITY;
    ${props.edgeMask ? 'if (live) { live = edgeMask[edgeMaskOffset + slot] != 0u; }' : ''}
    ${directed ? '' : 'live = live && index <= targetVertex;'}
    if (live) {
      exponentBits = max(exponentBits, getFixedPointExponentBits(weights[weightsOffset + slot]));
    }
  }
  maxCounter(0u, exponentBits);`
      });
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-weight-sum-scale`,
          operation: OPERATION,
          variant: 'weight-sum-scale',
          bindings: weightScaleBindings,
          invocationCount: nodeCount,
          workgroupSize: WORKGROUP_COUNTER_WORKGROUP_SIZE,
          guardIndex: false,
          declarations: `const NODE_COUNT: u32 = ${nodeCount}u;
const SLOT_COUNT: u32 = ${slotCount}u;
const GROUP_CAPACITY: u32 = ${groupCapacity}u;
${FIXED_POINT_WGSL}
${weightScaleCounters.declarations}`,
          body: weightScaleCounters.body
        })
      );
    }

    // Slots: classify each counted slot, count intra-group edges and emit inter-group keys.
    const slotBindings: WGSLKernelBinding[] = [
      {name: 'offsets', view: props.topology.offsets, type: 'u32', access: 'read'},
      {name: 'neighbors', view: props.topology.neighbors, type: 'u32', access: 'read'},
      {
        name: 'effectiveLabels',
        view: effectiveLabels,
        type: 'u32',
        access: 'read'
      }
    ];
    if (props.edgeMask) {
      slotBindings.push({
        name: 'edgeMask',
        view: props.edgeMask,
        type: 'u32',
        access: 'read'
      });
    }
    if (props.topology.weights && hasIntraWeight) {
      slotBindings.push({
        name: 'weights',
        view: props.topology.weights,
        type: 'f32',
        access: 'read'
      });
    }
    slotBindings.push({
      name: 'sortKeys',
      view: sortKeys,
      type: 'u32',
      access: 'read_write'
    });
    if (intraAcc) {
      slotBindings.push({
        name: 'intraAcc',
        view: intraAcc,
        type: 'atomic<u32>',
        access: 'read_write'
      });
    }
    slotBindings.push({
      name: 'stats',
      view: stats,
      type: 'atomic<u32>',
      access: 'read_write'
    });
    const pairKey = directed
      ? 'sourceLabel * GROUP_CAPACITY + targetLabel'
      : 'min(sourceLabel, targetLabel) * GROUP_CAPACITY + max(sourceLabel, targetLabel)';
    const slotCounters = getWorkgroupCounterKernelSource({
      counters: [
        {target: 'stats', word: WORD.countedEdgeCount},
        {target: 'stats', word: WORD.droppedEdgeCount},
        {target: 'stats', word: WORD.intraEdgeCount},
        {target: 'stats', word: WORD.interEdgeCount}
      ],
      rowCount: 'NODE_COUNT',
      rowSource: `let sourceLabel = effectiveLabels[effectiveLabelsOffset + index];
  let rowEnd = min(offsets[offsetsOffset + index + 1u], SLOT_COUNT);
  let rowBegin = min(offsets[offsetsOffset + index], rowEnd);
  var counted = 0u;
  var dropped = 0u;
  var intra = 0u;
  var inter = 0u;
  var intraWeightLow = 0u;
  var intraWeightHigh = 0u;
  ${hasIntraWeight ? `let weightUnitExponent = ${getWeightUnitExponentSource('atomic')};` : ''}
  for (var slot = rowBegin; slot < rowEnd; slot++) {
    var key = SENTINEL_KEY;
    let targetVertex = neighbors[neighborsOffset + slot];
    if (sourceLabel != DEAD && targetVertex < NODE_COUNT) {
      let targetLabel = effectiveLabels[effectiveLabelsOffset + targetVertex];
      var live = targetLabel != DEAD;
      ${props.edgeMask ? 'if (live) { live = edgeMask[edgeMaskOffset + slot] != 0u; }' : ''}
      ${directed ? '' : 'live = live && index <= targetVertex;'}
      if (live) {
        counted++;
        if (sourceLabel == OVERFLOWED || targetLabel == OVERFLOWED) {
          dropped++;
        } else if (sourceLabel == targetLabel) {
          intra++;
          ${
            hasIntraWeight
              ? `let quantizedWeight = quantize(${props.topology.weights ? 'weights[weightsOffset + slot]' : '1.0'}, weightUnitExponent);
          let previousLow = intraWeightLow;
          intraWeightLow += quantizedWeight.x;
          intraWeightHigh += quantizedWeight.y + select(0u, 1u, intraWeightLow < previousLow);`
              : ''
          }
        } else {
          inter++;
          key = ${pairKey};
        }
      }
    }
    sortKeys[sortKeysOffset + slot] = key;
  }
  ${
    intraAcc
      ? `// Every intra-group slot of a row lands in the row's own group, so the row folds its intra
  // count and 64-bit fixed-point weight locally and issues one atomic set instead of one per slot
  // (integer addition is associative, so the sums are unchanged).
  if (intra > 0u) {
    let intraBase = sourceLabel * 3u;
    atomicAdd(&intraAcc[intraAccOffset + intraBase], intra);
    ${
      hasIntraWeight
        ? `let rowIntraWeight = vec2<u32>(intraWeightLow, intraWeightHigh);
    ${getFixedPointAddSource('intraAcc', 'intraAccOffset + intraBase + 1u', 'rowIntraWeight')}`
        : ''
    }
  }`
      : ''
  }
  addCounter(0u, counted);
  addCounter(1u, dropped);
  addCounter(2u, intra);
  addCounter(3u, inter);`
    });
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-slots`,
        operation: OPERATION,
        variant: 'slots',
        bindings: slotBindings,
        invocationCount: nodeCount,
        workgroupSize: WORKGROUP_COUNTER_WORKGROUP_SIZE,
        guardIndex: false,
        declarations: `const NODE_COUNT: u32 = ${nodeCount}u;
const SLOT_COUNT: u32 = ${slotCount}u;
const GROUP_CAPACITY: u32 = ${groupCapacity}u;
const SENTINEL_KEY: u32 = ${sentinelKey}u;
const DEAD: u32 = 0xffffffffu;
const OVERFLOWED: u32 = 0xfffffffeu;
${getFixedPointSource(!usesDataUnit)}
${slotCounters.declarations}`,
        body: slotCounters.body
      })
    );
    // Superedges: stable sort of (key, slot), segment numbering, then per-segment accumulation.
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-iota`,
        operation: OPERATION,
        variant: 'iota',
        bindings: [
          {
            name: 'sortIndices',
            view: sortIndices,
            type: 'u32',
            access: 'read_write'
          }
        ],
        invocationCount: slotCount,
        body: 'sortIndices[sortIndicesOffset + index] = index;'
      })
    );
    nodes.push(
      ...new GPUSort({
        id: `${id}-sort`,
        keys: sortKeys,
        values: sortIndices,
        outputKeys: sortedKeys,
        outputValues: sortedIndices,
        keyBits: getSortKeyBits(sentinelKey)
      }).getCommandNodes(graph)
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-flags`,
        operation: OPERATION,
        variant: 'flags',
        bindings: [
          {name: 'sortedKeys', view: sortedKeys, type: 'u32', access: 'read'},
          {name: 'flags', view: flags, type: 'u32', access: 'read_write'}
        ],
        invocationCount: slotCount,
        declarations: `const SENTINEL_KEY: u32 = ${sentinelKey}u;`,
        body: `let key = sortedKeys[sortedKeysOffset + index];
  var isStart = key != SENTINEL_KEY;
  if (isStart && index > 0u) {
    isStart = sortedKeys[sortedKeysOffset + index - 1u] != key;
  }
  flags[flagsOffset + index] = select(0u, 1u, isStart);`
      })
    );
    nodes.push(
      ...new GPUScan({
        id: `${id}-segment-scan`,
        input: flags,
        output: segmentScan,
        mode: 'exclusive'
      }).getCommandNodes(graph)
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-segment-total`,
        operation: OPERATION,
        variant: 'segment-total',
        bindings: [
          {
            name: 'segmentScan',
            view: segmentScan,
            type: 'u32',
            access: 'read'
          },
          {name: 'flags', view: flags, type: 'u32', access: 'read'},
          {name: 'stats', view: stats, type: 'u32', access: 'read_write'},
          {
            name: 'totalOut',
            view: segmentTotal,
            type: 'u32',
            access: 'read_write'
          },
          {
            name: 'overflowOut',
            view: overflowFlag,
            type: 'u32',
            access: 'read_write'
          }
        ],
        invocationCount: 1,
        body: `let total = segmentScan[segmentScanOffset + ${slotCount - 1}u] + flags[flagsOffset + ${slotCount - 1}u];
  totalOut[totalOutOffset] = total;
  stats[statsOffset + ${WORD.superedgeCount}u] = total;
  let labelOverflow = stats[statsOffset + ${WORD.overflowedNodeCount}u] > 0u ||
    stats[statsOffset + ${WORD.droppedEdgeCount}u] > 0u;
  overflowOut[overflowOutOffset] = select(0u, 1u, labelOverflow);`
      })
    );
    const segmentBindings: WGSLKernelBinding[] = [
      {name: 'sortedKeys', view: sortedKeys, type: 'u32', access: 'read'},
      {name: 'segmentScan', view: segmentScan, type: 'u32', access: 'read'},
      {
        name: 'edgeIds',
        view: props.output.edges.ids,
        type: 'u32',
        access: 'read_write'
      },
      {
        name: 'edgeTargets',
        view: props.output.edgeTargetGroups,
        type: 'u32',
        access: 'read_write'
      },
      {
        name: 'edgeCounts',
        view: props.output.edgeCounts,
        type: 'atomic<u32>',
        access: 'read_write'
      }
    ];
    // Each invocation owns SEGMENT_RUN_LENGTH consecutive sorted slots and folds every run of one
    // key locally, flushing one atomic per run. A heavy superedge then costs one atomic per run
    // instead of one per slot on a single address; integer addition is associative, so counts and
    // weights are unchanged. Counts and weights are two kernels over the same runs: together with
    // the scale word of the weights they would bind more storage buffers than a device allows.
    const getRunSegmentSource = (
      declareRun: string,
      resetRun: string,
      flushRun: string,
      addSlot: string
    ) =>
      `let first = index * ${SEGMENT_RUN_LENGTH}u;
  let last = min(first + ${SEGMENT_RUN_LENGTH}u, SLOT_COUNT);
  var runSegment = 0xffffffffu;
  var runCount = 0u;
  ${declareRun}
  for (var position = first; position < last; position++) {
    let key = sortedKeys[sortedKeysOffset + position];
    if (key == SENTINEL_KEY) {
      break;
    }
    var isStart = true;
    if (position > 0u) {
      isStart = sortedKeys[sortedKeysOffset + position - 1u] != key;
    }
    let segment = segmentScan[segmentScanOffset + position] + select(0u, 1u, isStart) - 1u;
    if (segment != runSegment) {
      if (runCount > 0u) {
        ${flushRun}
      }
      runSegment = segment;
      runCount = 0u;
      ${resetRun}
    }
    if (segment >= EDGE_CAPACITY) {
      continue;
    }
    runCount++;
    ${addSlot}
  }
  if (runCount > 0u) {
    ${flushRun}
  }`;
    const segmentDeclarations = `const SENTINEL_KEY: u32 = ${sentinelKey}u;
const GROUP_CAPACITY: u32 = ${groupCapacity}u;
const EDGE_CAPACITY: u32 = ${edgeCapacity}u;
const SLOT_COUNT: u32 = ${slotCount}u;`;
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-segments`,
        operation: OPERATION,
        variant: 'segments',
        bindings: segmentBindings,
        invocationCount: Math.ceil(slotCount / SEGMENT_RUN_LENGTH),
        declarations: segmentDeclarations,
        body: getRunSegmentSource(
          '',
          '',
          'atomicAdd(&edgeCounts[edgeCountsOffset + runSegment], runCount);',
          `if (isStart) {
      edgeIds[edgeIdsOffset + segment] = key / GROUP_CAPACITY;
      edgeTargets[edgeTargetsOffset + segment] = key % GROUP_CAPACITY;
    }`
        )
      })
    );
    if (segmentWeightAcc && props.output.edgeWeights) {
      const segmentWeightBindings: WGSLKernelBinding[] = [
        {name: 'sortedKeys', view: sortedKeys, type: 'u32', access: 'read'},
        {name: 'segmentScan', view: segmentScan, type: 'u32', access: 'read'},
        {
          name: 'segmentWeightAcc',
          view: segmentWeightAcc,
          type: 'atomic<u32>',
          access: 'read_write'
        }
      ];
      if (props.topology.weights) {
        segmentWeightBindings.push(
          {name: 'sortedIndices', view: sortedIndices, type: 'u32', access: 'read'},
          {name: 'weights', view: props.topology.weights, type: 'f32', access: 'read'}
        );
      }
      const weightStatsBinding: WGSLKernelBinding[] = needsWeightScale
        ? [{name: 'stats', view: stats, type: 'u32', access: 'read'}]
        : [];
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-segment-weights`,
          operation: OPERATION,
          variant: 'segment-weights',
          bindings: [...segmentWeightBindings, ...weightStatsBinding],
          invocationCount: Math.ceil(slotCount / SEGMENT_RUN_LENGTH),
          declarations: `${segmentDeclarations}
${getFixedPointSource(!usesDataUnit)}`,
          body: `let weightUnitExponent = ${getWeightUnitExponentSource('plain')};
  ${getRunSegmentSource(
    `var runWeightLow = 0u;
  var runWeightHigh = 0u;`,
    `runWeightLow = 0u;
      runWeightHigh = 0u;`,
    `let runWeight = vec2<u32>(runWeightLow, runWeightHigh);
        ${getFixedPointAddSource('segmentWeightAcc', 'segmentWeightAccOffset + 2u * runSegment', 'runWeight')}`,
    `let quantizedWeight = quantize(${props.topology.weights ? 'weights[weightsOffset + sortedIndices[sortedIndicesOffset + position]]' : '1.0'}, weightUnitExponent);
    let previousLow = runWeightLow;
    runWeightLow += quantizedWeight.x;
    runWeightHigh += quantizedWeight.y + select(0u, 1u, runWeightLow < previousLow);`
  )}`
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-edge-weights`,
          operation: OPERATION,
          variant: 'edge-weights',
          bindings: [
            {
              name: 'segmentWeightAcc',
              view: segmentWeightAcc,
              type: 'u32',
              access: 'read'
            },
            {
              name: 'edgeWeights',
              view: props.output.edgeWeights,
              type: 'f32',
              access: 'read_write'
            },
            ...weightStatsBinding
          ],
          invocationCount: edgeCapacity,
          declarations: getFixedPointSource(!usesDataUnit),
          body: `edgeWeights[edgeWeightsOffset + index] = dequantize(
    segmentWeightAcc[segmentWeightAccOffset + 2u * index],
    segmentWeightAcc[segmentWeightAccOffset + 2u * index + 1u],
    ${getWeightUnitExponentSource('plain')});`
        })
      );
    }

    // Per-group finish: centroids, bounds, value sums and intra-group outputs.
    const finishBindings: WGSLKernelBinding[] = [
      {
        name: 'groupCounts',
        view: props.output.groupNodeCount,
        type: 'u32',
        access: 'read'
      },
      {name: 'groupAcc', view: groupAcc, type: 'u32', access: 'read'}
    ];
    const finishStatements: string[] = [];
    if (props.output.groupCentroid) {
      finishBindings.push({
        name: 'centroidOut',
        view: props.output.groupCentroid,
        type: 'f32',
        access: 'read_write'
      });
      finishStatements.push(`let positionUnitExponent = ${getUnitExponentSource(SCALE_WORD.positions, nodeCount, 'plain')};
  centroidOut[centroidOutOffset + 2u * index] = select(0.0, dequantize(groupAcc[groupAccOffset + base], groupAcc[groupAccOffset + base + 1u], positionUnitExponent) / f32(count), count > 0u);
  centroidOut[centroidOutOffset + 2u * index + 1u] = select(0.0, dequantize(groupAcc[groupAccOffset + base + 2u], groupAcc[groupAccOffset + base + 3u], positionUnitExponent) / f32(count), count > 0u);`);
    }
    if (props.output.groupBounds) {
      finishBindings.push({
        name: 'boundsOut',
        view: props.output.groupBounds,
        type: 'f32',
        access: 'read_write'
      });
      finishStatements.push(`boundsOut[boundsOutOffset + 4u * index] = select(0.0, decodeOrderedKey(groupAcc[groupAccOffset + base + 6u]), count > 0u);
  boundsOut[boundsOutOffset + 4u * index + 1u] = select(0.0, decodeOrderedKey(groupAcc[groupAccOffset + base + 7u]), count > 0u);
  boundsOut[boundsOutOffset + 4u * index + 2u] = select(0.0, decodeOrderedKey(groupAcc[groupAccOffset + base + 8u]), count > 0u);
  boundsOut[boundsOutOffset + 4u * index + 3u] = select(0.0, decodeOrderedKey(groupAcc[groupAccOffset + base + 9u]), count > 0u);`);
    }
    if (props.output.groupValueSum) {
      finishBindings.push({
        name: 'valueOut',
        view: props.output.groupValueSum,
        type: 'f32',
        access: 'read_write'
      });
      finishStatements.push(
        `valueOut[valueOutOffset + index] = dequantize(groupAcc[groupAccOffset + base + 4u], groupAcc[groupAccOffset + base + 5u], ${getUnitExponentSource(SCALE_WORD.values, nodeCount, 'plain')});`
      );
    }
    if (usesDataUnit && (props.output.groupCentroid || props.output.groupValueSum)) {
      finishBindings.push({name: 'stats', view: stats, type: 'u32', access: 'read'});
    }
    // The intra-group outputs are their own kernel: with the scale word of the weights one
    // kernel for every group output would bind more storage buffers than a device allows.
    if (intraAcc) {
      const intraBindings: WGSLKernelBinding[] = [
        {name: 'intraAcc', view: intraAcc, type: 'u32', access: 'read'}
      ];
      const intraStatements: string[] = [];
      if (props.output.groupIntraEdgeCount) {
        intraBindings.push({
          name: 'intraCountOut',
          view: props.output.groupIntraEdgeCount,
          type: 'u32',
          access: 'read_write'
        });
        intraStatements.push(
          'intraCountOut[intraCountOutOffset + index] = intraAcc[intraAccOffset + 3u * index];'
        );
      }
      if (props.output.groupIntraWeight) {
        intraBindings.push({
          name: 'intraWeightOut',
          view: props.output.groupIntraWeight,
          type: 'f32',
          access: 'read_write'
        });
        if (needsWeightScale) {
          intraBindings.push({name: 'stats', view: stats, type: 'u32', access: 'read'});
        }
        intraStatements.push(
          `intraWeightOut[intraWeightOutOffset + index] = dequantize(intraAcc[intraAccOffset + 3u * index + 1u], intraAcc[intraAccOffset + 3u * index + 2u], ${getWeightUnitExponentSource('plain')});`
        );
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-intra-groups`,
          operation: OPERATION,
          variant: 'intra-groups',
          bindings: intraBindings,
          invocationCount: groupCapacity,
          declarations: getFixedPointSource(!usesDataUnit),
          body: intraStatements.join('\n  ')
        })
      );
    }
    if (finishStatements.length > 0) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-groups`,
          operation: OPERATION,
          variant: 'groups',
          bindings: finishBindings,
          invocationCount: groupCapacity,
          declarations: getFixedPointSource(!usesDataUnit),
          body: `let base = index * ${GROUP_ACC_STRIDE}u;
  let count = groupCounts[groupCountsOffset + index];
  ${finishStatements.join('\n  ')}`
        })
      );
    }

    nodes.push(
      createPublishNode<Parameters>(graph, {
        id: `${id}-publish`,
        operation: OPERATION,
        requiredCount: segmentTotal,
        output: props.output.edges,
        overflowSources: [overflowFlag]
      })
    );
    if (props.output.summary) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-summary`,
          operation: OPERATION,
          variant: 'summary',
          bindings: [
            {name: 'stats', view: stats, type: 'u32', access: 'read'},
            {name: 'summary', view: props.output.summary, type: 'u32', access: 'read_write'}
          ],
          invocationCount: GPU_NETWORK_COARSENING_SUMMARY_LENGTH,
          body: 'summary[summaryOffset + index] = stats[statsOffset + index];'
        })
      );
    }
    return nodes;
  }
}
