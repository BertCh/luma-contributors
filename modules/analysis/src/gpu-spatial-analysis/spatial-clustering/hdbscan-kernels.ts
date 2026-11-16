// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/graph/gpu-core';
import type {GPUCommandGraphNodeCondition, GraphBufferUse} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';

import {
  HIERARCHICAL_DENSITY_EDGE_WORDS,
  HIERARCHICAL_DENSITY_CLUSTER_WORDS,
  HIERARCHICAL_DENSITY_SELECTION_WORDS,
  type HierarchicalDensityShape,
  type HierarchicalDensityGate
} from './hierarchical-density-clustering-kernels';

/** Four words per BVH node; a leaf stores xy and its row, with no duplicate maximum xy. @internal */
export const HDBSCAN_NODE_WORDS = 4;
/** Parent, subtree size and merge weight; the two children live in a separate plane. @internal */
export const HDBSCAN_DENDROGRAM_WORDS = 3;

const OPERATION = 'GPUHierarchicalDensityClustering';

function getConstants(shape: HierarchicalDensityShape): string {
  return `const ROW_COUNT: u32 = ${shape.rowCount}u;
const NODE_COUNT: u32 = ${2 * shape.rowCount}u;
const MINIMUM_SAMPLES: u32 = ${shape.minimumSamples}u;
const MINIMUM_CLUSTER_SIZE: u32 = ${shape.minimumClusterSize}u;
const CLUSTER_CAPACITY: u32 = ${shape.clusterCapacity}u;
const LEAF_CAPACITY: u32 = ${shape.leafCapacity}u;
const INTERNAL_NODE_COUNT: u32 = ${shape.internalNodeCount}u;
const LEVEL_COUNT: u32 = ${shape.levelCount}u;
const EDGE_WORDS: u32 = ${HIERARCHICAL_DENSITY_EDGE_WORDS}u;
const NODE_WORDS: u32 = ${HDBSCAN_NODE_WORDS}u;
const DENDROGRAM_WORDS: u32 = ${HDBSCAN_DENDROGRAM_WORDS}u;
const CLUSTER_WORDS: u32 = ${HIERARCHICAL_DENSITY_CLUSTER_WORDS}u;
const SELECTION_WORDS: u32 = ${HIERARCHICAL_DENSITY_SELECTION_WORDS}u;
const NONE: u32 = 0xffffffffu;
const FLOAT32_MAXIMUM: f32 = 3.402823e38;
fn isFiniteNumber(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}`;
}

function getGate<Parameters>(gate?: HierarchicalDensityGate<Parameters>): {
  condition?: GPUCommandGraphNodeCondition<Parameters>;
  extraResources?: readonly GraphBufferUse[];
} {
  return gate ? {condition: gate.condition, extraResources: gate.extraResources} : {};
}

/** Packs internal boxes and leaf xy/row records without duplicating each leaf coordinate. @internal */
export function createHierarchicalDensityNodeDataNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: HierarchicalDensityShape;
    nodeMinima: GraphDataView;
    nodeMaxima: GraphDataView;
    leafIds: GraphDataView<'uint32'>;
    nodeData: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'node-data',
    bindings: [
      {name: 'nodeMinima', view: props.nodeMinima, type: 'f32', access: 'read'},
      {name: 'nodeMaxima', view: props.nodeMaxima, type: 'f32', access: 'read'},
      {name: 'leafIds', view: props.leafIds, type: 'u32', access: 'read'},
      {name: 'nodeData', view: props.nodeData, type: 'u32', access: 'read_write'}
    ],
    invocationCount: 2 * props.shape.leafCapacity - 1,
    declarations: getConstants(props.shape),
    body: `let base = nodeDataOffset + index * NODE_WORDS;
  nodeData[base] = bitcast<u32>(nodeMinima[nodeMinimaOffset + index * 2u]);
  nodeData[base + 1u] = bitcast<u32>(nodeMinima[nodeMinimaOffset + index * 2u + 1u]);
  nodeData[base + 2u] = bitcast<u32>(nodeMaxima[nodeMaximaOffset + index * 2u]);
  nodeData[base + 3u] = bitcast<u32>(nodeMaxima[nodeMaximaOffset + index * 2u + 1u]);
  var row = NONE;
  if (index >= INTERNAL_NODE_COUNT) {
    row = leafIds[leafIdsOffset + index - INTERNAL_NODE_COUNT];
    if (row >= ROW_COUNT ||
        nodeMinima[nodeMinimaOffset + index * 2u] > nodeMaxima[nodeMaximaOffset + index * 2u] ||
        nodeMinima[nodeMinimaOffset + index * 2u + 1u] > nodeMaxima[nodeMaximaOffset + index * 2u + 1u]) { row = NONE; }
  }
  if (index >= INTERNAL_NODE_COUNT) { nodeData[base + 2u] = row; }`
  });
}

/**
 * Flags the BVH leaf slots whose component differs from the previous slot's, for the inclusive
 * scan that answers "does this node hold a single component" in O(1). @internal
 */
export function createHierarchicalDensityLeafFlagsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: HierarchicalDensityShape;
    nodeData: GraphDataView<'uint32'>;
    components: GraphDataView<'uint32'>;
    changeFlags: GraphDataView<'uint32'>;
    gate?: HierarchicalDensityGate<Parameters>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'leaf-flags',
    bindings: [
      {name: 'nodeData', view: props.nodeData, type: 'u32', access: 'read'},
      {name: 'components', view: props.components, type: 'u32', access: 'read'},
      {name: 'changeFlags', view: props.changeFlags, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.shape.leafCapacity,
    declarations: `${getConstants(props.shape)}
fn getLeafComponent(slot: u32) -> u32 {
  let row = nodeData[nodeDataOffset + (INTERNAL_NODE_COUNT + slot) * NODE_WORDS + 2u];
  return select(components[componentsOffset + min(row, ROW_COUNT - 1u)], NONE, row == NONE);
}`,
    body: `var isChange = 0u;
  if (index > 0u && getLeafComponent(index) != getLeafComponent(index - 1u)) { isChange = 1u; }
  changeFlags[changeFlagsOffset + index] = isChange;`,
    ...getGate(props.gate)
  });
}

function getDensitySearchWGSL(shape: HierarchicalDensityShape, hasTime: boolean): string {
  return `const STACK_SIZE: u32 = ${2 * shape.levelCount}u;
const BOUND_SLACK: f32 = 1.000001;
fn getPoint(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + row * 2u], positions[positionsOffset + row * 2u + 1u]);
}
fn getBoxDistance(node: u32, point: vec2<f32>) -> f32 {
  let base = nodeDataOffset + node * NODE_WORDS;
  if (node >= INTERNAL_NODE_COUNT && nodeData[base + 2u] == NONE) { return FLOAT32_MAXIMUM; }
  let minimum = vec2<f32>(bitcast<f32>(nodeData[base]), bitcast<f32>(nodeData[base + 1u]));
  var maximum = minimum;
  if (node < INTERNAL_NODE_COUNT) { maximum = vec2<f32>(bitcast<f32>(nodeData[base + 2u]), bitcast<f32>(nodeData[base + 3u])); }
  if (minimum.x > maximum.x || minimum.y > maximum.y) { return FLOAT32_MAXIMUM; }
  let gap = max(max(minimum - point, point - maximum), vec2<f32>(0.0));
  return sqrt(dot(gap, gap));
}

fn getPairDistance(first: u32, second: u32) -> f32 {
  let spatialDifference = getPoint(first) - getPoint(second);
  ${
    hasTime
      ? `let scale = timeData[timeDataOffset + ROW_COUNT];
  var timeDifference = 0.0;
  if (scale > 0.0) { timeDifference = (timeData[timeDataOffset + first] - timeData[timeDataOffset + second]) * scale; }
  return length(vec3<f32>(spatialDifference, timeDifference));`
      : 'return length(spatialDifference);'
  }
}`;
}

/**
 * Boruvka search: every valid point walks the BVH (nearer child first) for the nearest point of
 * another component under the mutual reachability distance `max(core a, core b, |a - b|)`, with
 * the strict total edge order `(weight, low row, high row)` so that Boruvka never closes a cycle.
 * A node is pruned when `max(core a, box distance)` exceeds the best weight (with a 1e-6 relative
 * slack, so rounding never drops a tie), or when all its leaves lie in the point's component: a
 * node covers a contiguous leaf range, and the inclusive scan of component changes is equal at both
 * ends exactly when the range holds one component. Writes the candidate and the component minimum
 * weight. @internal
 */
export function createHierarchicalDensitySearchNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: HierarchicalDensityShape;
    nodeData: GraphDataView<'uint32'>;
    changeScan: GraphDataView<'uint32'>;
    positions: GraphDataView<'float32x2'>;
    timeData?: GraphDataView<'float32'>;
    coreDistances: GraphDataView<'float32'>;
    components: GraphDataView<'uint32'>;
    candidates: GraphDataView<'uint32'>;
    componentBest: GraphDataView<'uint32'>;
    gate?: HierarchicalDensityGate<Parameters>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'search',
    bindings: [
      {name: 'nodeData', view: props.nodeData, type: 'u32', access: 'read'},
      {name: 'changeScan', view: props.changeScan, type: 'u32', access: 'read'},
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      ...(props.timeData
        ? [{name: 'timeData', view: props.timeData, type: 'f32' as const, access: 'read' as const}]
        : []),
      {name: 'coreDistances', view: props.coreDistances, type: 'f32', access: 'read'},
      {name: 'components', view: props.components, type: 'u32', access: 'read'},
      {name: 'candidates', view: props.candidates, type: 'u32', access: 'read_write'},
      {name: 'componentBest', view: props.componentBest, type: 'atomic<u32>', access: 'read_write'}
    ],
    invocationCount: props.shape.rowCount,
    declarations: `${getConstants(props.shape)}
${getDensitySearchWGSL(props.shape, Boolean(props.timeData))}
// True when every leaf under the node belongs to 'component' (the node covers a contiguous range
// of leaf slots in the flat complete tree).
fn isNodeInComponent(node: u32, component: u32) -> bool {
  let depth = firstLeadingBit(node + 1u);
  let span = 1u << (LEVEL_COUNT - 1u - depth);
  let first = (node + 1u - (1u << depth)) * span;
  let last = first + span - 1u;
  if (changeScan[changeScanOffset + last] != changeScan[changeScanOffset + first]) { return false; }
  let row = nodeData[nodeDataOffset + (INTERNAL_NODE_COUNT + first) * NODE_WORDS + 2u];
  return row != NONE && components[componentsOffset + row] == component;
}
// Lexicographic (weight bits, low row, high row) order of candidate edges; weights are
// non-negative, so their f32 bits order like the values.
fn isBetter(weight: u32, low: u32, high: u32, bestWeight: u32, bestLow: u32, bestHigh: u32) -> bool {
  if (weight != bestWeight) { return weight < bestWeight; }
  if (low != bestLow) { return low < bestLow; }
  return high < bestHigh;
}
// A degenerate node holds identical xy values. The stable Hilbert sort preserves ascending
// source rows inside its single key, so its first leaf is a conservative row tie bound.
// This avoids a quadratic equal-distance MST search for large duplicate populations.
fn cannotImproveCoincidentNode(node: u32, queryRow: u32, core: f32,
    bestWeight: u32, bestLow: u32, bestHigh: u32) -> bool {
  if (bestWeight == NONE || node >= INTERNAL_NODE_COUNT) { return false; }
  let base = nodeDataOffset + node * NODE_WORDS;
  if (nodeData[base] != nodeData[base + 2u] || nodeData[base + 1u] != nodeData[base + 3u]) { return false; }
  let depth = firstLeadingBit(node + 1u);
  let span = 1u << (LEVEL_COUNT - 1u - depth);
  let first = (node + 1u - (1u << depth)) * span;
  let lowestRow = nodeData[nodeDataOffset + (INTERNAL_NODE_COUNT + first) * NODE_WORDS + 2u];
  if (lowestRow == NONE) { return false; }
  if (bestWeight != 0u) {
    ${props.timeData ? 'if (timeData[timeDataOffset + ROW_COUNT] != 0.0) { return false; }' : ''}
    // With no time contribution every point in this node has the same pair distance.
    if (bitcast<u32>(max(core, getPairDistance(queryRow, lowestRow))) != bestWeight) { return false; }
  }
  return !isBetter(bestWeight, min(queryRow, lowestRow), max(queryRow, lowestRow), bestWeight, bestLow, bestHigh);
}`,
    body: `let component = components[componentsOffset + index];
  var bestWeight = NONE;
  var bestLow = NONE;
  var bestHigh = NONE;
  if (component != NONE) {
    let point = getPoint(index);
    let core = coreDistances[coreDistancesOffset + index];
    var stackNode: array<u32, STACK_SIZE>;
    var stackLower: array<f32, STACK_SIZE>;
    var top = 0u;
    if (!isNodeInComponent(0u, component)) {
      stackNode[0] = 0u;
      stackLower[0] = max(core, getBoxDistance(0u, point));
      top = 1u;
    }
    loop {
      if (top == 0u) { break; }
      top = top - 1u;
      let node = stackNode[top];
      let bound = select(FLOAT32_MAXIMUM, bitcast<f32>(bestWeight) * BOUND_SLACK, bestWeight != NONE);
      if (stackLower[top] > bound ||
          cannotImproveCoincidentNode(node, index, core, bestWeight, bestLow, bestHigh)) { continue; }
      if (node < INTERNAL_NODE_COUNT) {
        let first = node * 2u + 1u;
        let second = first + 1u;
        let firstLower = max(core, getBoxDistance(first, point));
        let secondLower = max(core, getBoxDistance(second, point));
        let firstUsable = firstLower <= bound && getBoxDistance(first, point) < FLOAT32_MAXIMUM &&
          !isNodeInComponent(first, component);
        let secondUsable = secondLower <= bound && getBoxDistance(second, point) < FLOAT32_MAXIMUM &&
          !isNodeInComponent(second, component);
        if (firstUsable && secondUsable && top + 2u <= STACK_SIZE) {
          if (firstLower <= secondLower) {
            stackNode[top] = second; stackLower[top] = secondLower;
            stackNode[top + 1u] = first; stackLower[top + 1u] = firstLower;
          } else {
            stackNode[top] = first; stackLower[top] = firstLower;
            stackNode[top + 1u] = second; stackLower[top + 1u] = secondLower;
          }
          top = top + 2u;
        } else if (firstUsable && top + 1u <= STACK_SIZE) {
          stackNode[top] = first; stackLower[top] = firstLower;
          top = top + 1u;
        } else if (secondUsable && top + 1u <= STACK_SIZE) {
          stackNode[top] = second; stackLower[top] = secondLower;
          top = top + 1u;
        }
        continue;
      }
      let other = nodeData[nodeDataOffset + node * NODE_WORDS + 2u];
      if (other == NONE || other == index || components[componentsOffset + other] == component) { continue; }
      let otherCore = coreDistances[coreDistancesOffset + other];
      // The same expression from both ends gives the same f32 weight (a - b is -(b - a)).
      let weight = bitcast<u32>(max(max(core, otherCore), getPairDistance(index, other)));
      let low = min(index, other);
      let high = max(index, other);
      if (isBetter(weight, low, high, bestWeight, bestLow, bestHigh)) {
        bestWeight = weight;
        bestLow = low;
        bestHigh = high;
      }
    }
    if (bestWeight != NONE) {
      atomicMin(&componentBest[componentBestOffset + component * EDGE_WORDS], bestWeight);
    }
  }
  candidates[candidatesOffset + index * EDGE_WORDS] = bestWeight;
  candidates[candidatesOffset + index * EDGE_WORDS + 1u] = bestLow;
  candidates[candidatesOffset + index * EDGE_WORDS + 2u] = bestHigh;`,
    ...getGate(props.gate)
  });
}

/** WGSL helpers over a `dendrogram` binding. */
const DENDROGRAM_WGSL = /* wgsl */ `
fn getNodeParent(node: u32) -> u32 {
  return dendrogram[dendrogramOffset + node * DENDROGRAM_WORDS];
}
fn getNodeSize(node: u32) -> u32 {
  return dendrogram[dendrogramOffset + node * DENDROGRAM_WORDS + 1u];
}
// Merge density 1 / weight; a zero weight (duplicate points) gives the largest f32 instead of
// scikit-learn's infinity, so differences stay finite.
fn getNodeLambda(node: u32) -> f32 {
  let weight = bitcast<f32>(dendrogram[dendrogramOffset + node * DENDROGRAM_WORDS + 2u]);
  return select(FLOAT32_MAXIMUM, 1.0 / weight, weight > 0.0);
}
fn isBigNode(node: u32) -> bool {
  return node != NONE && getNodeSize(node) >= MINIMUM_CLUSTER_SIZE;
}
// A node of at least minimumClusterSize points whose parent is a true split (both children big),
// or the root, starts a condensed cluster.
`;

const CLUSTER_START_WGSL = /* wgsl */ `
fn isClusterStart(node: u32) -> bool {
  if (!isBigNode(node)) { return false; }
  let parent = getNodeParent(node);
  if (parent == NONE) { return true; }
  let base = dendrogramChildrenOffset + parent * 2u;
  return isBigNode(dendrogramChildren[base]) && isBigNode(dendrogramChildren[base + 1u]);
}`;

/**
 * Initial condensing pointers per dendrogram node: word 0 points to the node itself when it starts
 * a cluster, else to its parent (pointer jumping then finds each big node's cluster start); word 1
 * points to the node itself when it is big, else to its parent (pointer jumping then finds each
 * point's lowest big ancestor, where it falls out of its cluster). Also writes the cluster-start
 * flags for the compact cluster IDs. @internal
 */
export function createHierarchicalDensityPointerInitNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: HierarchicalDensityShape;
    dendrogram: GraphDataView<'uint32'>;
    dendrogramChildren: GraphDataView<'uint32'>;
    pointers: GraphDataView<'uint32'>;
    startFlags: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'pointer-init',
    bindings: [
      {name: 'dendrogram', view: props.dendrogram, type: 'u32', access: 'read'},
      {name: 'dendrogramChildren', view: props.dendrogramChildren, type: 'u32', access: 'read'},
      {name: 'pointers', view: props.pointers, type: 'u32', access: 'read_write'},
      {name: 'startFlags', view: props.startFlags, type: 'u32', access: 'read_write'}
    ],
    invocationCount: 2 * props.shape.rowCount,
    declarations: `${getConstants(props.shape)}
${DENDROGRAM_WGSL}
${CLUSTER_START_WGSL}`,
    body: `let parent = getNodeParent(index);
  let isStart = isClusterStart(index);
  pointers[pointersOffset + index * 2u] = select(parent, index, isStart);
  pointers[pointersOffset + index * 2u + 1u] = select(parent, index, isBigNode(index));
  startFlags[startFlagsOffset + index] = select(0u, 1u, isStart);`
  });
}

/**
 * Per condensed cluster (from its start node): birth lambda (the parent split's lambda, 0 for the
 * root), parent cluster ID, and point count for heavy-path contraction. @internal
 */
export function createHierarchicalDensityClusterInfoNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: HierarchicalDensityShape;
    dendrogram: GraphDataView<'uint32'>;
    startFlags: GraphDataView<'uint32'>;
    clusterOffsets: GraphDataView<'uint32'>;
    pointers: GraphDataView<'uint32'>;
    clusterInfo: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'cluster-info',
    bindings: [
      {name: 'dendrogram', view: props.dendrogram, type: 'u32', access: 'read'},
      {name: 'startFlags', view: props.startFlags, type: 'u32', access: 'read'},
      {name: 'clusterOffsets', view: props.clusterOffsets, type: 'u32', access: 'read'},
      {name: 'pointers', view: props.pointers, type: 'u32', access: 'read'},
      {name: 'clusterInfo', view: props.clusterInfo, type: 'u32', access: 'read_write'}
    ],
    invocationCount: 2 * props.shape.rowCount,
    declarations: `${getConstants(props.shape)}
${DENDROGRAM_WGSL}`,
    body: `if (startFlags[startFlagsOffset + index] == 0u) { return; }
  let cluster = clusterOffsets[clusterOffsetsOffset + index];
  if (cluster >= CLUSTER_CAPACITY) { return; }
  let parent = getNodeParent(index);
  var birth = 0.0;
  var parentCluster = NONE;
  if (parent != NONE) {
    birth = getNodeLambda(parent);
    let parentStart = pointers[pointersOffset + parent * 2u];
    parentCluster = clusterOffsets[clusterOffsetsOffset + parentStart];
  }
  clusterInfo[clusterInfoOffset + cluster * CLUSTER_WORDS] = bitcast<u32>(birth);
  clusterInfo[clusterInfoOffset + cluster * CLUSTER_WORDS + 1u] = parentCluster;
  clusterInfo[clusterInfoOffset + cluster * CLUSTER_WORDS + 2u] = getNodeSize(index);`
  });
}

/**
 * Stability contributions, one row per dendrogram node (scikit-learn's condensed tree rows): a
 * point contributes `lambda_p - birth(C)` to the cluster `C` it falls out of at `lambda_p` (its
 * lowest big ancestor), and a child cluster `D` contributes `size(D) (birth(D) - birth(C))` to its
 * parent `C`. Other rows get the key `CLUSTER_CAPACITY`, which the segment sum skips. Also writes
 * each point's lowest cluster for the labels. @internal
 */
export function createHierarchicalDensityContributionsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: HierarchicalDensityShape;
    dendrogram: GraphDataView<'uint32'>;
    startFlags: GraphDataView<'uint32'>;
    clusterOffsets: GraphDataView<'uint32'>;
    pointers: GraphDataView<'uint32'>;
    clusterInfo: GraphDataView<'uint32'>;
    contributionKeys: GraphDataView<'uint32'>;
    contributions: GraphDataView<'float32'>;
    lowClusters: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'contributions',
    bindings: [
      {name: 'dendrogram', view: props.dendrogram, type: 'u32', access: 'read'},
      {name: 'startFlags', view: props.startFlags, type: 'u32', access: 'read'},
      {name: 'clusterOffsets', view: props.clusterOffsets, type: 'u32', access: 'read'},
      {name: 'pointers', view: props.pointers, type: 'u32', access: 'read'},
      {name: 'clusterInfo', view: props.clusterInfo, type: 'u32', access: 'read'},
      {name: 'contributionKeys', view: props.contributionKeys, type: 'u32', access: 'read_write'},
      {name: 'contributions', view: props.contributions, type: 'f32', access: 'read_write'},
      {name: 'lowClusters', view: props.lowClusters, type: 'u32', access: 'read_write'}
    ],
    invocationCount: 2 * props.shape.rowCount,
    declarations: `${getConstants(props.shape)}
${DENDROGRAM_WGSL}
fn getBirth(cluster: u32) -> f32 {
  return bitcast<f32>(clusterInfo[clusterInfoOffset + cluster * CLUSTER_WORDS]);
}`,
    body: `var key = CLUSTER_CAPACITY;
  var value = 0.0;
  if (index < ROW_COUNT) {
    var lowCluster = NONE;
    let fallNode = pointers[pointersOffset + index * 2u + 1u];
    if (fallNode != NONE) {
      let start = pointers[pointersOffset + fallNode * 2u];
      if (start != NONE) {
        let cluster = clusterOffsets[clusterOffsetsOffset + start];
        if (cluster < CLUSTER_CAPACITY) {
          lowCluster = cluster;
          key = cluster;
          value = getNodeLambda(fallNode) - getBirth(cluster);
        }
      }
    }
    lowClusters[lowClustersOffset + index] = lowCluster;
  } else if (startFlags[startFlagsOffset + index] != 0u) {
    let cluster = clusterOffsets[clusterOffsetsOffset + index];
    if (cluster < CLUSTER_CAPACITY) {
      let parentCluster = clusterInfo[clusterInfoOffset + cluster * CLUSTER_WORDS + 1u];
      if (parentCluster != NONE) {
        key = parentCluster;
        value = f32(getNodeSize(index)) * (getBirth(cluster) - getBirth(parentCluster));
      }
    }
  }
  contributionKeys[contributionKeysOffset + index] = key;
  contributions[contributionsOffset + index] = value;`
  });
}

/** Exact planar or scaled space-time top-k; one bounded private list per query, no rows × samples table. @internal */
export function createHierarchicalDensityCoreNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: HierarchicalDensityShape;
    nodeData: GraphDataView<'uint32'>;
    positions: GraphDataView<'float32x2'>;
    timeData?: GraphDataView<'float32'>;
    coreDistances: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode(graph, {
    id: props.id,
    operation: OPERATION,
    variant: props.timeData ? 'space-time-core' : 'bvh-core',
    invocationCount: props.shape.rowCount,
    bindings: [
      {name: 'nodeData', view: props.nodeData, type: 'u32', access: 'read'},
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      ...(props.timeData
        ? [{name: 'timeData', view: props.timeData, type: 'f32' as const, access: 'read' as const}]
        : []),
      {name: 'coreDistances', view: props.coreDistances, type: 'f32', access: 'read_write'}
    ],
    declarations: `${getConstants(props.shape)}
${getDensitySearchWGSL(props.shape, Boolean(props.timeData))}`,
    body: `let infinity = bitcast<f32>(0x7f800000u | (index & 0u));
  coreDistances[coreDistancesOffset + index] = infinity;
  let point = getPoint(index);
  if (!isFiniteNumber(point.x) || !isFiniteNumber(point.y)${props.timeData ? ' || !isFiniteNumber(timeData[timeDataOffset + index])' : ''}) { return; }
  // Only initialized slots below count are ever read; insertion writes each slot first.
  var nearest: array<f32, MINIMUM_SAMPLES>;
  var count = 0u;
  var stack: array<u32, STACK_SIZE>;
  stack[0] = 0u;
  var top = 1u;
  loop {
    if (top == 0u) { break; }
    top--;
    let node = stack[top];
    var bound = infinity;
    if (count == MINIMUM_SAMPLES - 1u) { bound = ${props.timeData ? 'nearest[count - 1u]' : 'sqrt(nearest[count - 1u])'} * BOUND_SLACK; }
    let lower = getBoxDistance(node, point);
    if (lower >= FLOAT32_MAXIMUM || lower > bound) { continue; }
    if (node < INTERNAL_NODE_COUNT) {
      let first = node * 2u + 1u;
      let second = first + 1u;
      // Complete binary tree: DFS needs at most LEVEL_COUNT pending nodes.
      if (getBoxDistance(first, point) <= getBoxDistance(second, point)) {
        stack[top] = second; stack[top + 1u] = first;
      } else {
        stack[top] = first; stack[top + 1u] = second;
      }
      top += 2u;
      continue;
    }
    let other = nodeData[nodeDataOffset + node * NODE_WORDS + 2u];
    if (other == NONE || other == index${props.timeData ? ' || !isFiniteNumber(timeData[timeDataOffset + other])' : ''}) { continue; }
    ${
      props.timeData
        ? 'let candidate = getPairDistance(index, other);'
        : `// Match the packed reference's scalar squared-distance expression and round sqrt once.
    let deltaX = positions[positionsOffset + other * 2u] - point.x;
    let deltaY = positions[positionsOffset + other * 2u + 1u] - point.y;
    let candidate = deltaX * deltaX + deltaY * deltaY;`
    }
    var slot = min(count, MINIMUM_SAMPLES - 2u);
    if (count == MINIMUM_SAMPLES - 1u && candidate >= nearest[slot]) { continue; }
    loop {
      if (slot == 0u) { break; }
      if (nearest[slot - 1u] <= candidate) { break; }
      nearest[slot] = nearest[slot - 1u];
      slot--;
    }
    nearest[slot] = candidate;
    count = min(count + 1u, MINIMUM_SAMPLES - 1u);
    // Distances are nonnegative: a full list ending at zero is already the exact answer.
    if (count == MINIMUM_SAMPLES - 1u && nearest[count - 1u] == 0.0) { break; }
  }
  if (count == MINIMUM_SAMPLES - 1u) {
    coreDistances[coreDistancesOffset + index] = ${props.timeData ? 'nearest[count - 1u]' : 'sqrt(nearest[count - 1u])'};
  }`
  });
}
