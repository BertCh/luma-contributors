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
import {createWGSLKernelNode} from '../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateMaximumCount,
  validatePackedView,
  validateRequirement
} from '../analysis-core/kernels/gpu-contributor-utils';
import {IS_FINITE_FLOAT_WGSL} from '../analysis-core/kernels/wgsl-common';
import {validatePointProcessView} from '../gpu-spatial-analysis/point-process/point-process-utils';

/** Largest event count: scratch and work are quadratic in the events (1,048,576 pairs). */
const MAXIMUM_EVENT_COUNT = 1024;
/** Largest `maximumPairCount`, the all-pairs scratch budget (events squared). */
const MAXIMUM_PAIR_COUNT = 1048576;
/** Largest node count: the Ang shell count scans every node once per pair. */
const MAXIMUM_NODE_COUNT = 65536;
/** Largest event-by-node distance table, 2^24 entries (64 MiB). */
const MAXIMUM_DISTANCE_COUNT = 16777216;
/** Largest edge count: the Ang shell count scans every edge once per pair. */
const MAXIMUM_EDGE_COUNT = 65536;
/** Largest radius count: bounds the per-radius status bitmask and finishing work. */
const MAXIMUM_RADIUS_COUNT = 256;
/** Threads of the workgroup that reduces the pairs of one radius. A power of two for the tree. */
const FINISH_WORKGROUP_SIZE = 256;
const CORRECTIONS = ['none', 'ang'];
const NORMALIZATIONS = ['network-length', 'inverse-intensity', 'inverse-intensity-squared'];

/** Caller-owned results of the inhomogeneous network K estimator. */
export type GPUNetworkInhomogeneousKOutput = {
  k: GraphDataView<'float32'>;
  /** Weighted ordered-pair sum before normalization, one value per radius. */
  numerator: GraphDataView<'float32'>;
  /** [networkLength, inverseIntensitySum, effectiveDenominator]. */
  normalization: GraphDataView<'float32'>;
  /** Per-radius bitmask: 1 event/intensity, 2 edge topology, 4 distance table,
   * 8 undefined Ang shell, 16 radius, 32 nonfinite arithmetic. Zero means valid. */
  status: GraphDataView<'uint32'>;
};

/** Node-event inhomogeneous linear K, optionally corrected for network geometry. */
export type GPUNetworkInhomogeneousKProps = {
  id?: string;
  /** Network node of each event. Interior events must first split their containing edge. */
  eventNodeIds: GraphDataView<'uint32'>;
  /** Positive finite intensity per event, supplied directly or estimated by a preceding graph. */
  intensity: GraphDataView<'float32'>;
  /** Event-major shortest path distances to every network node (for example a cost-matrix output).
   * Must cover the complete undirected network, not a radius-truncated routing result. */
  nodeDistances: GraphDataView<'float32'>;
  nodeCount: number;
  /** One row [source,target] per undirected edge; caller guarantees no duplicate geometric edges. */
  edgeNodes: GraphDataView<'uint32x2'>;
  /** Positive finite arc lengths. Zero-length edges must be contracted upstream. */
  edgeLengths: GraphDataView<'float32'>;
  /** Increasing nonnegative thresholds; pairs use <=, including distinct coincident events at 0. */
  radii: GraphDataView<'float32'>;
  correction?: 'none' | 'ang';
  /** Length normalization by default. Alternatives reproduce linearKinhom normalise powers 1/2. */
  normalization?: 'network-length' | 'inverse-intensity' | 'inverse-intensity-squared';
  /** Explicit quadratic pair budget, default 1048576; insufficient capacity throws, never truncates. */
  maximumPairCount?: number;
  output: GPUNetworkInhomogeneousKOutput;
};

/**
 * Inhomogeneous network K = sum_{i != j,dij <= r} e(i,j)/(lambda_i lambda_j) / denominator.
 * Ang correction e(i,j)=1/m(i,dij) counts network locations at exactly that shortest-path distance.
 * Vertices count once and each unique undirected edge contributes its interior level-set crossings.
 * This handles junctions, cycles and disconnected support in the supplied f32 shortest-path metric.
 *
 * Events are network nodes, including nodes inserted by upstream noding/snapping preparation.
 * No Euclidean distance substitute or CPU numerical fallback is used. O(events²*(nodes+edges))
 * Ang preparation, O(events²*radii) finishing, O(events²) scratch. At most 1024 events and 256 radii.
 * This is distinct from the existing homogeneous spaghetti-normalized GPUNetworkKFunction.
 *
 * Parallelism: the pair kernel is one thread per ordered pair. The radius finish is one workgroup
 * per radius whose 256 threads stride over the pairs and combine with a fixed-order tree, so the
 * sums are deterministic. Two stages stay sequential: the `normalize` kernel is one invocation
 * over all edges and events, and the Ang shell count scans every node and edge for each pair
 * (about 1.3e11 operations at the limits). Sorting each source's level-set crossings once and
 * binary-searching per pair would replace the Ang scan.
 */
export class GPUNetworkInhomogeneousK implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUNetworkInhomogeneousKProps;

  constructor(props: GPUNetworkInhomogeneousKProps) {
    this.id = props.id ?? 'network-inhomogeneous-k';
    this.props = props;
    const eventCount = props.eventNodeIds.length;
    const pairCount = eventCount * eventCount;
    const budget = props.maximumPairCount ?? 1048576;
    const {id} = this;
    // The explicit all-pairs path has a finite storage/work envelope and never truncates pairs.
    validateCapacity(id, 'maximumPairCount', budget, 1, MAXIMUM_PAIR_COUNT);
    validateMaximumCount(id, 'event pairs (eventNodeIds rows squared)', pairCount, budget);
    validateCapacity(id, 'eventNodeIds rows', eventCount, 1, MAXIMUM_EVENT_COUNT);
    validateCapacity(id, 'nodeCount', props.nodeCount, 1, MAXIMUM_NODE_COUNT);
    validateMaximumCount(
      id,
      'nodeDistances rows (eventNodeIds rows times nodeCount)',
      eventCount * props.nodeCount,
      MAXIMUM_DISTANCE_COUNT
    );
    validateCapacity(id, 'edgeLengths rows', props.edgeLengths.length, 1, MAXIMUM_EDGE_COUNT);
    validateCapacity(id, 'radii rows', props.radii.length, 1, MAXIMUM_RADIUS_COUNT);
    validateRequirement(
      id,
      props.correction === undefined || CORRECTIONS.includes(props.correction),
      `correction must be one of ${CORRECTIONS.join(', ')}`
    );
    validateRequirement(
      id,
      props.normalization === undefined || NORMALIZATIONS.includes(props.normalization),
      `normalization must be one of ${NORMALIZATIONS.join(', ')}`
    );
    validatePointProcessView(props.eventNodeIds, 'uint32', eventCount);
    validatePointProcessView(props.intensity, 'float32', eventCount);
    validatePointProcessView(props.nodeDistances, 'float32', eventCount * props.nodeCount);
    validatePointProcessView(props.edgeLengths, 'float32', props.edgeLengths.length);
    validatePackedView(props.edgeNodes, ['uint32x2'], 'network-inhomogeneous-k edgeNodes');
    validateRequirement(
      id,
      props.edgeNodes.length === props.edgeLengths.length,
      'edgeNodes must hold one row per edge length'
    );
    validatePointProcessView(props.radii, 'float32', props.radii.length);
    validatePointProcessView(props.output.k, 'float32', props.radii.length);
    validatePointProcessView(props.output.numerator, 'float32', props.radii.length);
    validatePointProcessView(props.output.normalization, 'float32', 3);
    validatePointProcessView(props.output.status, 'uint32', props.radii.length);
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), getInputs(props));
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const eventCount = props.eventNodeIds.length;
    const pairCount = eventCount * eventCount;
    const correction = props.correction ?? 'ang';
    const normalization = props.normalization ?? 'network-length';
    validateGraphViewsBelongToGraph(id, graph, [
      ...getInputs(props),
      ...Object.values(props.output)
    ]);
    const pairs = createTransientView(graph, `${id}-pairs`, 'float32x2', pairCount);
    const pairStatus = createTransientView(graph, `${id}-pair-status`, 'uint32', pairCount);
    const inputStatus = createTransientView(graph, `${id}-input-status`, 'uint32', 1);
    const distanceDeclaration = `${IS_FINITE_FLOAT_WGSL}
fn validDistance(value: f32) -> bool {
  return value >= 0.0 && (isFiniteFloat(value) || bitcast<u32>(value) == 0x7f800000u);
}
fn nodeDistance(event: u32, node: u32) -> f32 {
  return distances[distancesOffset + event * ${props.nodeCount}u + node];
}`;
    return [
      createWGSLKernelNode(graph, {
        id: `${id}-normalize`,
        operation: 'GPUNetworkInhomogeneousK',
        variant: normalization,
        invocationCount: 1,
        bindings: [
          {name: 'eventNodes', view: props.eventNodeIds, type: 'u32', access: 'read'},
          {name: 'intensity', view: props.intensity, type: 'f32', access: 'read'},
          {name: 'edges', view: props.edgeNodes, type: 'u32', access: 'read'},
          {name: 'lengths', view: props.edgeLengths, type: 'f32', access: 'read'},
          {
            name: 'normalization',
            view: props.output.normalization,
            type: 'f32',
            access: 'read_write'
          },
          {name: 'status', view: inputStatus, type: 'u32', access: 'read_write'}
        ],
        declarations: IS_FINITE_FLOAT_WGSL,
        body: `var flags = 0u;
  var length = 0.0;
  var inverseSum = 0.0;
  for (var edge = 0u; edge < ${props.edgeLengths.length}u; edge++) {
    let value = lengths[lengthsOffset + edge];
    let first = edges[edgesOffset + 2u * edge];
    let second = edges[edgesOffset + 2u * edge + 1u];
    if (!isFiniteFloat(value) || value <= 0.0 || first >= ${props.nodeCount}u || second >= ${props.nodeCount}u || first == second) { flags |= 2u; }
    length += value;
  }
  for (var event = 0u; event < ${eventCount}u; event++) {
    let value = intensity[intensityOffset + event];
    if (!isFiniteFloat(value) || value <= 0.0 || eventNodes[eventNodesOffset + event] >= ${props.nodeCount}u) { flags |= 1u; }
    inverseSum += 1.0 / value;
  }
  var denominator = ${normalization === 'network-length' ? 'length' : normalization === 'inverse-intensity' ? 'inverseSum' : 'inverseSum * (inverseSum / length)'};
  if (!isFiniteFloat(length) || !isFiniteFloat(inverseSum) || !isFiniteFloat(denominator) || denominator <= 0.0) { flags |= 32u; }
  normalization[normalizationOffset] = length;
  normalization[normalizationOffset + 1u] = inverseSum;
  normalization[normalizationOffset + 2u] = denominator;
  status[statusOffset] = flags;`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-pairs`,
        operation: 'GPUNetworkInhomogeneousK',
        variant: correction,
        invocationCount: pairCount,
        bindings: [
          {name: 'distances', view: props.nodeDistances, type: 'f32', access: 'read'},
          {name: 'eventNodes', view: props.eventNodeIds, type: 'u32', access: 'read'},
          {name: 'intensity', view: props.intensity, type: 'f32', access: 'read'},
          {name: 'edges', view: props.edgeNodes, type: 'u32', access: 'read'},
          {name: 'lengths', view: props.edgeLengths, type: 'f32', access: 'read'},
          {name: 'pairs', view: pairs, type: 'f32', access: 'read_write'},
          {name: 'status', view: pairStatus, type: 'u32', access: 'read_write'}
        ],
        declarations: distanceDeclaration,
        body: `let source = index / ${eventCount}u;
  let destination = index % ${eventCount}u;
  let sourceNode = eventNodes[eventNodesOffset + source];
  let destinationNode = eventNodes[eventNodesOffset + destination];
  var flags = 0u;
  var pairDistance = bitcast<f32>(0x7f800000u | (index & 0u));
  var weight = 0.0;
  if (sourceNode >= ${props.nodeCount}u || destinationNode >= ${props.nodeCount}u) { flags |= 1u; }
  else {
    pairDistance = nodeDistance(source, destinationNode);
    if (!validDistance(pairDistance) || nodeDistance(source, sourceNode) != 0.0) { flags |= 4u; }
    if (source != destination && isFiniteFloat(pairDistance) && pairDistance >= 0.0) {
      var shellCount = 1u;
      ${
        correction === 'ang'
          ? `
      shellCount = 0u;
      for (var node = 0u; node < ${props.nodeCount}u; node++) {
        let value = nodeDistance(source, node);
        if (!validDistance(value)) { flags |= 4u; }
        if (value == pairDistance) { shellCount++; }
      }
      for (var edge = 0u; edge < ${props.edgeLengths.length}u; edge++) {
        let first = edges[edgesOffset + 2u * edge];
        let second = edges[edgesOffset + 2u * edge + 1u];
        if (first >= ${props.nodeCount}u || second >= ${props.nodeCount}u) { continue; }
        let length = lengths[lengthsOffset + edge];
        let firstDistance = nodeDistance(source, first);
        let secondDistance = nodeDistance(source, second);
        // Interior roots of min(firstDistance+s, secondDistance+length-s)=pairDistance.
        let fromFirst = pairDistance - firstDistance;
        let fromSecond = length - (pairDistance - secondDistance);
        let firstRoot = isFiniteFloat(firstDistance) && fromFirst > 0.0 && fromFirst < length && pairDistance <= secondDistance + (length - fromFirst);
        let secondRoot = isFiniteFloat(secondDistance) && fromSecond > 0.0 && fromSecond < length && pairDistance <= firstDistance + fromSecond;
        if (firstRoot) { shellCount++; }
        if (secondRoot && (!firstRoot || fromSecond != fromFirst)) { shellCount++; }
      }
      if (shellCount == 0u) { flags |= 8u; }`
          : ''
      }
      weight = (1.0 / intensity[intensityOffset + source]) / intensity[intensityOffset + destination] / f32(shellCount);
      if (!isFiniteFloat(weight)) { flags |= 32u; }
    }
  }
  pairs[pairsOffset + 2u * index] = pairDistance;
  pairs[pairsOffset + 2u * index + 1u] = weight;
  status[statusOffset + index] = flags;`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-finish`,
        operation: 'GPUNetworkInhomogeneousK',
        variant: 'radius-reduction',
        // One workgroup per radius: its threads stride over the pairs and a fixed-order tree sums
        // their partial sums and ORs their status flags, so the result does not depend on scheduling.
        workgroupSize: FINISH_WORKGROUP_SIZE,
        invocationCount: props.radii.length * FINISH_WORKGROUP_SIZE,
        guardIndex: false,
        bindings: [
          {name: 'pairs', view: pairs, type: 'f32', access: 'read'},
          {name: 'pairStatus', view: pairStatus, type: 'u32', access: 'read'},
          {name: 'inputStatus', view: inputStatus, type: 'u32', access: 'read'},
          {name: 'normalization', view: props.output.normalization, type: 'f32', access: 'read'},
          {name: 'radii', view: props.radii, type: 'f32', access: 'read'},
          {name: 'numerator', view: props.output.numerator, type: 'f32', access: 'read_write'},
          {name: 'result', view: props.output.k, type: 'f32', access: 'read_write'},
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
        ],
        declarations: `${IS_FINITE_FLOAT_WGSL}
var<workgroup> partialSums: array<f32, ${FINISH_WORKGROUP_SIZE}>;
var<workgroup> partialFlags: array<u32, ${FINISH_WORKGROUP_SIZE}>;`,
        // No early return: every invocation of a workgroup must reach the barriers.
        body: `let radiusIndex = index / ${FINISH_WORKGROUP_SIZE}u;
  let radius = radii[radiiOffset + radiusIndex];
  var partialFlag = 0u;
  var partialSum = 0.0;
  for (var pair = localInvocationIndex; pair < ${pairCount}u; pair += ${FINISH_WORKGROUP_SIZE}u) {
    partialFlag |= pairStatus[pairStatusOffset + pair];
    if (pairs[pairsOffset + 2u * pair] <= radius) { partialSum += pairs[pairsOffset + 2u * pair + 1u]; }
  }
  partialSums[localInvocationIndex] = partialSum;
  partialFlags[localInvocationIndex] = partialFlag;
  workgroupBarrier();
  for (var stride = ${FINISH_WORKGROUP_SIZE / 2}u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partialSums[localInvocationIndex] += partialSums[localInvocationIndex + stride];
      partialFlags[localInvocationIndex] |= partialFlags[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u) {
    var flags = inputStatus[inputStatusOffset] | partialFlags[0];
    let sum = partialSums[0];
    if (!isFiniteFloat(radius) || radius < 0.0) { flags |= 16u; }
    if (radiusIndex > 0u && radius <= radii[radiiOffset + radiusIndex - 1u]) { flags |= 16u; }
    let value = sum / normalization[normalizationOffset + 2u];
    if (!isFiniteFloat(value) || !isFiniteFloat(sum)) { flags |= 32u; }
    let invalid = bitcast<f32>(0x7fc00000u | (radiusIndex & 0u));
    result[resultOffset + radiusIndex] = select(invalid, value, flags == 0u);
    numerator[numeratorOffset + radiusIndex] = select(invalid, sum, flags == 0u);
    status[statusOffset + radiusIndex] = flags;
  }`
      })
    ];
  }
}

function getInputs(props: GPUNetworkInhomogeneousKProps): GraphDataView[] {
  return [
    props.eventNodeIds,
    props.intensity,
    props.nodeDistances,
    props.edgeNodes,
    props.edgeLengths,
    props.radii
  ];
}
