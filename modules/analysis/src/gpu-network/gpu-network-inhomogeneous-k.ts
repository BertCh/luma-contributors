// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {SCALED_QUOTIENT_WGSL} from '../analysis-core/kernels/scaled-quotient-wgsl';

import {validateNetworkViewBindings, validateNetworkWritableViews} from './network-view-utils';

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
  validateMaximumCount,
  validateSinglePackedView,
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
/**
 * Ang shell count: level-set locations closer than this fraction of the pair distance are one
 * location. About 100 f32 rounding steps of the pair distance, which covers a distance table
 * accumulated over long paths; real gaps of whole edge-length units stay far above it.
 */
const ANG_SHELL_RELATIVE_TOLERANCE = '1.0e-5';
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
 * The count uses a tolerance: locations within 1e-5 of the pair distance of
 * each other along the network are one location, so a crossing that close to an edge end is that
 * node and a node that close to the pair distance is on the shell. The distance table holds
 * rounded sums, and without the tolerance the edge into the destination is counted as a second
 * crossing whenever its sum rounded down. Counts are exact while real gaps exceed the tolerance.
 * The tolerance is relative to the pair distance, not to the edge lengths. Where an edge is
 * shorter than about two tolerances, both of its ends can be on the shell with one crossing
 * between them: nodes on the shell that are joined by such a shortest-path edge are counted
 * once, as the earliest of them (smallest distance, then smallest node ID). A junction just
 * inside the tolerance with several edges leaving it is likewise one location.
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
    validateNetworkWritableViews(this.id, props.output, [
      {
        outputs: {
          k: props.output.k,
          numerator: props.output.numerator,
          status: props.output.status
        },
        inputs: {normalization: props.output.normalization}
      }
    ]);
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
    validateSinglePackedView(props.edgeNodes, ['uint32x2'], 'network-inhomogeneous-k edgeNodes');
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
    validateNetworkViewBindings(id, graph, [...getInputs(props), ...Object.values(props.output)]);
    validateNetworkViewBindings(id, graph, [], {pairs: pairCount * 8, pairStatus: pairCount * 4});
    const pairs = createTransientView(graph, `${id}-pairs`, 'float32x2', pairCount);
    const pairStatus = createTransientView(graph, `${id}-pair-status`, 'uint32', pairCount);
    const inputStatus = createTransientView(graph, `${id}-input-status`, 'uint32', 1);
    const distanceDeclaration = `${IS_FINITE_FLOAT_WGSL}
${SCALED_QUOTIENT_WGSL}
fn validDistance(value: f32) -> bool {
  return value >= 0.0 && (isFiniteFloat(value) || bitcast<u32>(value) == 0x7f800000u);
}
fn nodeDistance(event: u32, node: u32) -> f32 {
  return distances[distancesOffset + event * ${props.nodeCount}u + node];
}
const NO_MERGED_NODE: u32 = 0xffffffffu;
// Two nodes that are both within the tolerance of the shell and joined by an edge that is tight
// within the tolerance (a shortest-path edge, so at most about two tolerances long) hold one
// level-set location between them, not two. The later of the two (larger distance, then larger
// node ID) is merged into the earlier; this returns the merged node of an edge, or NO_MERGED_NODE.
// The earliest node of such a run is never merged, so every run keeps exactly its entries.
fn getShellMergedNode(event: u32, edge: u32, pairDistance: f32, shellTolerance: f32) -> u32 {
  let first = edges[edgesOffset + 2u * edge];
  let second = edges[edgesOffset + 2u * edge + 1u];
  if (first >= ${props.nodeCount}u || second >= ${props.nodeCount}u || first == second) { return NO_MERGED_NODE; }
  let firstDistance = nodeDistance(event, first);
  let secondDistance = nodeDistance(event, second);
  if (!isFiniteFloat(firstDistance) || !isFiniteFloat(secondDistance)) { return NO_MERGED_NODE; }
  if (abs(firstDistance - pairDistance) > shellTolerance || abs(secondDistance - pairDistance) > shellTolerance) { return NO_MERGED_NODE; }
  let length = lengths[lengthsOffset + edge];
  let isTight = abs(firstDistance + length - secondDistance) <= shellTolerance ||
    abs(secondDistance + length - firstDistance) <= shellTolerance;
  if (!isTight) { return NO_MERGED_NODE; }
  let firstIsEarlier = firstDistance < secondDistance || (firstDistance == secondDistance && first < second);
  return select(first, second, firstIsEarlier);
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
        declarations: `${IS_FINITE_FLOAT_WGSL}
${SCALED_QUOTIENT_WGSL}`,
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
    if (isFiniteFloat(value) && value > 0.0) { inverseSum += getScaledQuotient(1.0, value, 1.0, 0); }
  }
  var denominator = 0.0;
  if (isFiniteFloat(length) && length > 0.0 && isFiniteFloat(inverseSum)) {
    denominator = ${normalization === 'network-length' ? 'length' : normalization === 'inverse-intensity' ? 'inverseSum' : 'inverseSum * getScaledQuotient(inverseSum, length, 1.0, 0)'};
  }
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
      // Two level-set locations closer than this along the network are one location. The table
      // entries are rounded sums, so a root that is really an edge end lands a rounding step
      // inside the edge about half the time; an exact comparison would count it twice.
      let shellTolerance = pairDistance * ${ANG_SHELL_RELATIVE_TOLERANCE};
      for (var node = 0u; node < ${props.nodeCount}u; node++) {
        let value = nodeDistance(source, node);
        if (!validDistance(value)) { flags |= 4u; }
        if (node == destinationNode || (isFiniteFloat(value) && abs(value - pairDistance) <= shellTolerance)) { shellCount++; }
      }
      for (var edge = 0u; edge < ${props.edgeLengths.length}u; edge++) {
        let first = edges[edgesOffset + 2u * edge];
        let second = edges[edgesOffset + 2u * edge + 1u];
        if (first >= ${props.nodeCount}u || second >= ${props.nodeCount}u) { continue; }
        // Both ends on the shell and the edge tight: the node loop counted two nodes for one
        // location. Take the later one off again, once per node however many edges merge it.
        // Such an edge has no interior root: both candidate roots are within the tolerance of
        // an end.
        let mergedNode = getShellMergedNode(source, edge, pairDistance, shellTolerance);
        if (mergedNode != NO_MERGED_NODE) {
          var isFirstMerge = true;
          for (var earlier = 0u; earlier < edge; earlier++) {
            if (getShellMergedNode(source, earlier, pairDistance, shellTolerance) == mergedNode) {
              isFirstMerge = false;
              break;
            }
          }
          if (isFirstMerge) { shellCount--; }
          continue;
        }
        let length = lengths[lengthsOffset + edge];
        let firstDistance = nodeDistance(source, first);
        let secondDistance = nodeDistance(source, second);
        // Roots of min(firstDistance+s, secondDistance+length-s)=pairDistance, both measured from
        // the first end. A root within the tolerance of an edge end is that node, counted above.
        let fromFirst = pairDistance - firstDistance;
        let fromSecond = length - (pairDistance - secondDistance);
        let interiorEnd = length - shellTolerance;
        let firstInterior = isFiniteFloat(firstDistance) && fromFirst > shellTolerance && fromFirst < interiorEnd;
        let secondInterior = isFiniteFloat(secondDistance) && fromSecond > shellTolerance && fromSecond < interiorEnd;
        if (!isFiniteFloat(firstDistance) || !isFiniteFloat(secondDistance)) {
          // Only one end is reachable in the supplied table: its branch is the whole profile.
          if (firstInterior || secondInterior) { shellCount++; }
          continue;
        }
        // fromSecond - fromFirst is twice the height of the edge profile's peak above pairDistance:
        // positive means two separate roots, about zero one root at the peak, negative none.
        let rootSeparation = fromSecond - fromFirst;
        if (rootSeparation > shellTolerance) {
          shellCount += u32(firstInterior) + u32(secondInterior);
        } else if (rootSeparation >= -shellTolerance && (firstInterior || secondInterior)) {
          shellCount++;
        }
      }
      if (shellCount == 0u) { flags |= 8u; }`
          : ''
      }
      let sourceIntensity = intensity[intensityOffset + source];
      let destinationIntensity = intensity[intensityOffset + destination];
      if (isFiniteFloat(sourceIntensity) && sourceIntensity > 0.0 &&
          isFiniteFloat(destinationIntensity) && destinationIntensity > 0.0 && shellCount > 0u) {
        weight = getScaledQuotient(1.0, sourceIntensity, destinationIntensity, 0) / f32(shellCount);
      } else { flags |= 32u; }
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
${SCALED_QUOTIENT_WGSL}
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
    let denominator = normalization[normalizationOffset + 2u];
    var value = 0.0;
    if (isFiniteFloat(sum) && isFiniteFloat(denominator) && denominator > 0.0) {
      value = getScaledQuotient(sum, denominator, 1.0, 0);
    }
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
