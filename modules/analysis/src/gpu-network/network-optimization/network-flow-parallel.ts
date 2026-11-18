// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateCommandNodeBudget,
  validateGPUStorageBindingByteLength,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateGPUNetworkTopology} from '../../analysis-core/contracts/network-topology';
import {
  createWGSLDispatchGate,
  createWGSLKernelNode,
  getWGSLKernelBinding,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {UINT64_WGSL} from '../../analysis-core/kernels/uint64-wgsl';
import {
  getNetworkFlowPushRelabelNodes,
  DEFAULT_NETWORK_FLOW_RELABELS,
  MAXIMUM_NETWORK_FLOW_RELABELS,
  DEFAULT_NETWORK_FLOW_COST_PHASES,
  MAXIMUM_NETWORK_FLOW_COST_PHASES
} from './network-flow-push-relabel';
import type {GPUNetworkFlowProps} from './gpu-network-flow';

import {
  NETWORK_FLOW_PARALLEL_MAXIMUM_NODE_COUNT as MAXIMUM_PARALLEL_NODE_COUNT,
  NETWORK_FLOW_PARALLEL_MAXIMUM_EDGE_COUNT as MAXIMUM_PARALLEL_EDGE_COUNT,
  NETWORK_FLOW_MAXIMUM_DISTANCE as MAXIMUM_DISTANCE,
  NETWORK_FLOW_MAXIMUM_TOTAL_CAPACITY as MAXIMUM_TOTAL_CAPACITY,
  NETWORK_FLOW_MAXIMUM_UNIT_COST as MAXIMUM_UNIT_COST
} from './network-flow-limits';
/** Bounds unrolled search commands; the shared command-node budget also bounds their product. */
const MAXIMUM_RELAXATION_ROUNDS = 8192;
/** Keeps the default large-graph schedule below the command budget, including path doubling. */
const DEFAULT_PARALLEL_AUGMENTATIONS = 8;
/** Covers moderate graph diameters; longer paths require an explicit budget. */
const DEFAULT_RELAXATION_ROUNDS = 1024;
/** A label contains distance bits, hop count and incoming residual arc ID. */
const LABEL_WORD_COUNT = 3;
/** A path record contains doubling ancestor, bottleneck, predecessor arc and membership flag. */
const PATH_WORD_COUNT = 4;
/** GPUScan has at most three commands per hierarchy level, far below this at 30M rows.
 * Node specs check the estimate against actual nodes at the maximum accepted shape. */
const INITIAL_COMMAND_ALLOWANCE = 64;
/** Fixed scalar state: active, valid, complete, value, augmentations, changed, searchActive,
 * labelParity, invalid, totalCapacity, increment, cutCapacity, costLow, costHigh, source, sink,
 * required, searchConverged, augmentActive, relabelCount, strandedExcess. All words are rewritten on every encoding. */
const STATE_WORD_COUNT = 21;

/** New props explicitly opt into batching; omitted props retain legacy augmentation semantics. */
function usesPrimalDualFlow(props: GPUNetworkFlowProps): boolean {
  return (
    props.mode === 'primal-dual' ||
    (props.mode === undefined && props.maximumCostPhases !== undefined)
  );
}

function getRelaxationRounds(props: GPUNetworkFlowProps): number {
  return props.maximumRelaxationRounds ?? Math.min(props.nodeCount, DEFAULT_RELAXATION_ROUNDS);
}

function getAugmentations(props: GPUNetworkFlowProps): number {
  return props.maximumAugmentations ?? DEFAULT_PARALLEL_AUGMENTATIONS;
}

function getDoublingRounds(props: GPUNetworkFlowProps): number {
  return Math.ceil(Math.log2(props.nodeCount));
}

/** Upper bound includes the final search that certifies exhaustion after the last augmentation.
 * @internal */
export function estimateParallelNetworkFlowNodes(props: GPUNetworkFlowProps): number {
  if (!props.costs || usesPrimalDualFlow(props)) {
    const rounds = getRelaxationRounds(props);
    return (
      INITIAL_COMMAND_ALLOWANCE +
      (1 +
        (props.costs
          ? (props.maximumCostPhases ?? DEFAULT_NETWORK_FLOW_COST_PHASES)
          : (props.maximumRelabels ?? DEFAULT_NETWORK_FLOW_RELABELS))) *
        (6 * rounds + 16) +
      2 * rounds +
      4
    );
  }
  return (
    INITIAL_COMMAND_ALLOWANCE +
    (getAugmentations(props) + 1) *
      (2 * getRelaxationRounds(props) + 3 * getDoublingRounds(props) + 7)
  );
}

/** Shape and schedule validation only; per-frame CSR/edge contents are validated on the GPU.
 * @internal */
export function validateParallelNetworkFlow(
  id: string,
  props: GPUNetworkFlowProps,
  estimate: number
): void {
  validateCapacity(id, 'nodeCount', props.nodeCount, 2, MAXIMUM_PARALLEL_NODE_COUNT);
  validateCapacity(id, 'edges rows', props.edges.length, 0, MAXIMUM_PARALLEL_EDGE_COUNT);
  validateCapacity(
    id,
    'maximumRelaxationRounds',
    getRelaxationRounds(props),
    1,
    MAXIMUM_RELAXATION_ROUNDS
  );
  validateCapacity(
    id,
    'maximumRelabels',
    props.maximumRelabels ?? DEFAULT_NETWORK_FLOW_RELABELS,
    0,
    MAXIMUM_NETWORK_FLOW_RELABELS
  );
  validateCapacity(
    id,
    'maximumCostPhases',
    props.maximumCostPhases ?? DEFAULT_NETWORK_FLOW_COST_PHASES,
    0,
    MAXIMUM_NETWORK_FLOW_COST_PHASES
  );
  if (props.topology) {
    const shape = validateGPUNetworkTopology(id, props.topology);
    validateRequirement(
      id,
      shape.nodeCount === props.nodeCount && shape.edgeCount === props.edges.length,
      'topology must hold nodeCount vertices and one neighbor per original edge'
    );
  }
  validateCommandNodeBudget(id, estimate);
}

/** Parallel successive shortest augmenting paths. No invocation scans a graph-sized range:
 * relaxation visits a CSR neighborhood, and ancestor/bottleneck/membership propagation doubles
 * its reach in each parallel pass. The committed flow is always feasible, even if a search stops
 * before convergence. Uncosted problems use blocking preflow
 * pushes with global BFS relabelling; costed problems retain successive shortest paths unless mode or maximumCostPhases opts into primal-dual batches. @internal */
export function getParallelNetworkFlowNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: GPUNetworkFlowProps
): GPUCommandNode<Parameters>[] {
  const nodeCount = props.nodeCount;
  const edgeCount = props.edges.length;
  // All row-proportional allocations, including the caller's packed edge binding, are checked
  // before creating a gate, primitive or kernel node.
  for (const [name, byteLength] of [
    ['flow control state', STATE_WORD_COUNT * 4],
    ['dispatch gates', 24],
    ['edges', edgeCount * 8],
    ['residual CSR arcs', Math.max(2 * edgeCount, 1) * 4],
    ['edge records', Math.max(edgeCount, 1) * 16],
    ['residual CSR offsets', (nodeCount + 1) * 4],
    ...(props.costs && !usesPrimalDualFlow(props)
      ? ([
          ['shortest-path labels', nodeCount * LABEL_WORD_COUNT * 4],
          ['path doubling scratch', nodeCount * PATH_WORD_COUNT * 4]
        ] as const)
      : ([
          ['push-relabel heights/excess', nodeCount * (props.costs ? 8 : 4)],
          ['push-relabel edge increments/snapshot', Math.max(edgeCount, 1) * 4]
        ] as const))
  ] as const) {
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      name,
      byteLength,
      'reduce nodeCount or edges rows, or request a larger device binding limit'
    );
  }
  const word = (name: string, count: number) =>
    createTransientView(graph, `${id}-${name}`, 'uint32', count);
  const read = (name: string, view: GraphDataView) => getWGSLKernelBinding(name, view);
  const write = (name: string, view: GraphDataView) =>
    getWGSLKernelBinding(name, view, 'read_write');
  const atomic = (name: string, view: GraphDataView) => getWGSLKernelBinding(name, view, 'atomic');
  const kernel = (
    name: string,
    count: number,
    bindings: WGSLKernelBinding[],
    body: string,
    gate?: ReturnType<typeof createWGSLDispatchGate<Parameters>>,
    declarations = ''
  ) =>
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-${name}`,
      operation: 'GPUNetworkFlow',
      variant: 'parallel-residual-csr',
      invocationCount: count,
      bindings,
      declarations: `const NODE_COUNT: u32 = ${nodeCount}u;
const EDGE_COUNT: u32 = ${edgeCount}u;
const INFINITY: i32 = 0x7fffffff;
const NONE: u32 = 0xffffffffu;
${declarations}`,
      body,
      ...(gate ? gate.getGate(0) : {})
    });
  const state = word('state', STATE_WORD_COUNT);
  const records = word('edge-records', Math.max(edgeCount, 1) * 4);
  const counts = word('incident-counts', nodeCount + 1);
  const offsets = word('residual-offsets', nodeCount + 1);
  const arcs = word('residual-arcs', Math.max(2 * edgeCount, 1));
  const nodes: GPUCommandNode<Parameters>[] = [
    kernel(
      'initialize-state',
      STATE_WORD_COUNT,
      [write('state', state)],
      'state[stateOffset + index] = 0u;'
    ),
    kernel(
      'initialize-counts',
      nodeCount + 1,
      [write('counts', counts), write('cut', props.output.cut)],
      `counts[countsOffset + index] = 0u;
if (index < NODE_COUNT) { cut[cutOffset + index] = 0u; }`
    ),
    kernel(
      'validate-edges',
      edgeCount,
      [
        read('edges', props.edges),
        read('capacities', props.capacities),
        ...(props.costs ? [read('costs', props.costs)] : []),
        write('flow', props.output.flow),
        atomic('counts', counts),
        atomic('state', state),
        ...(props.topology
          ? [
              read('inputOffsets', props.topology.offsets),
              read('neighbors', props.topology.neighbors)
            ]
          : [])
      ],
      `flow[flowOffset + index] = 0u;
let source = edges[edgesOffset + 2u * index];
let destination = edges[edgesOffset + 2u * index + 1u];
let capacity = capacities[capacitiesOffset + index];
if (source >= NODE_COUNT || destination >= NODE_COUNT || capacity > ${MAXIMUM_TOTAL_CAPACITY}u ${
        props.costs
          ? `|| costs[costsOffset + index] > ${Math.min(MAXIMUM_UNIT_COST, Math.floor(MAXIMUM_DISTANCE / nodeCount))}u`
          : ''
      }) { atomicOr(&state[stateOffset + 8u], 1u); return; }
${
  props.topology
    ? `if (index < inputOffsets[inputOffsetsOffset + source] || index >= inputOffsets[inputOffsetsOffset + source + 1u] || neighbors[neighborsOffset + index] != destination) { atomicOr(&state[stateOffset + 8u], 1u); }`
    : ''
}
let previous = atomicAdd(&state[stateOffset + 9u], capacity);
if (previous > ${MAXIMUM_TOTAL_CAPACITY}u - capacity) { atomicOr(&state[stateOffset + 8u], 1u); }
atomicAdd(&counts[countsOffset + source], 1u);
atomicAdd(&counts[countsOffset + destination], 1u);`
    )
  ];
  if (props.topology) {
    nodes.push(
      kernel(
        'validate-topology',
        nodeCount + 1,
        [read('inputOffsets', props.topology.offsets), atomic('state', state)],
        `let position = inputOffsets[inputOffsetsOffset + index];
var invalid = position > EDGE_COUNT;
if (index == 0u) { invalid = invalid || position != 0u; }
if (index == NODE_COUNT) { invalid = invalid || position != EDGE_COUNT; }
else { invalid = invalid || position > inputOffsets[inputOffsetsOffset + index + 1u]; }
if (invalid) { atomicOr(&state[stateOffset + 8u], 1u); }`
      )
    );
  }
  nodes.push(
    kernel(
      'prepare-edge-records',
      edgeCount,
      [
        read('edges', props.edges),
        read('capacities', props.capacities),
        ...(props.costs ? [read('costs', props.costs)] : []),
        write('records', records)
      ],
      `records[recordsOffset + 4u * index] = edges[edgesOffset + 2u * index];
records[recordsOffset + 4u * index + 1u] = edges[edgesOffset + 2u * index + 1u];
records[recordsOffset + 4u * index + 2u] = capacities[capacitiesOffset + index];
records[recordsOffset + 4u * index + 3u] = ${props.costs ? 'costs[costsOffset + index]' : '0u'};`
    ),
    ...new GPUScan({
      id: `${id}-residual-scan`,
      input: counts,
      output: offsets,
      mode: 'exclusive'
    }).getCommandNodes(graph),
    kernel(
      'initialize-cursors',
      nodeCount,
      [read('offsets', offsets), write('counts', counts)],
      'counts[countsOffset + index] = offsets[offsetsOffset + index];'
    ),
    kernel(
      'scatter-residual-arcs',
      edgeCount,
      [
        read('edges', props.edges),
        read('state', state),
        atomic('counts', counts),
        write('arcs', arcs)
      ],
      `if (state[stateOffset + 8u] != 0u) { return; }
let source = edges[edgesOffset + 2u * index]; let destination = edges[edgesOffset + 2u * index + 1u];
let forward = atomicAdd(&counts[countsOffset + source], 1u);
let reverse = atomicAdd(&counts[countsOffset + destination], 1u);
arcs[arcsOffset + forward] = 2u * index;
arcs[arcsOffset + reverse] = 2u * index + 1u;`
    ),
    kernel(
      'initialize-problem',
      1,
      [read('parameters', props.parameters), write('state', state)],
      `let source = parameters[parametersOffset]; let destination = parameters[parametersOffset + 1u];
let valid = state[stateOffset + 8u] == 0u && source < NODE_COUNT && destination < NODE_COUNT && source != destination;
state[stateOffset] = select(0u, 1u, valid);
state[stateOffset + 1u] = select(0u, 1u, valid);
state[stateOffset + 14u] = source; state[stateOffset + 15u] = destination;
state[stateOffset + 16u] = parameters[parametersOffset + 2u];`
    )
  );

  if (!props.costs || usesPrimalDualFlow(props)) {
    nodes.push(
      ...getNetworkFlowPushRelabelNodes(graph, id, props, {
        state,
        offsets,
        arcs,
        records,
        maximumRounds: getRelaxationRounds(props)
      })
    );
  } else {
    const labels = [word('labels-first', nodeCount * 3), word('labels-second', nodeCount * 3)];
    const paths = [word('paths-first', nodeCount * 4), word('paths-second', nodeCount * 4)];
    const searchGate = createWGSLDispatchGate(graph, {
      id: `${id}-search-dispatch`,
      operation: 'GPUNetworkFlow',
      shapes: [[nodeCount]]
    });
    const augmentGate = createWGSLDispatchGate(graph, {
      id: `${id}-augment-dispatch`,
      operation: 'GPUNetworkFlow',
      shapes: [[nodeCount]]
    });
    const maximumAugmentations = getAugmentations(props);
    const doublingRounds = getDoublingRounds(props);
    for (let augmentation = 0; augmentation <= maximumAugmentations; augmentation++) {
      const prefix = `search-${augmentation}`;
      nodes.push(
        kernel(
          `${prefix}-start`,
          1,
          [write('state', state), searchGate.binding],
          `state[stateOffset + 18u] = 0u; state[stateOffset + 6u] = state[stateOffset];
if (state[stateOffset] != 0u) {
state[stateOffset + 5u] = 0u; state[stateOffset + 7u] = 0u; state[stateOffset + 17u] = 0u;
${searchGate.armWGSL}
} else { ${searchGate.disarmWGSL} }`
        ),
        kernel(
          `${prefix}-labels`,
          nodeCount,
          [read('state', state), write('labels', labels[0])],
          `let source = state[stateOffset + 14u];
labels[labelsOffset + 3u * index] = bitcast<u32>(select(INFINITY, 0, index == source));
labels[labelsOffset + 3u * index + 1u] = select(NONE, 0u, index == source);
labels[labelsOffset + 3u * index + 2u] = NONE;`,
          searchGate
        )
      );
      for (let round = 0; round < getRelaxationRounds(props); round++) {
        const input = labels[round % 2];
        const output = labels[1 - (round % 2)];
        nodes.push(
          kernel(
            `${prefix}-relax-${round}`,
            nodeCount,
            [
              read('offsets', offsets),
              read('arcs', arcs),
              read('records', records),
              read('flow', props.output.flow),
              read('inputLabels', input),
              write('outputLabels', output),
              atomic('state', state)
            ],
            `let base = 3u * index;
var distance = bitcast<i32>(inputLabels[inputLabelsOffset + base]);
var hops = inputLabels[inputLabelsOffset + base + 1u];
var predecessor = inputLabels[inputLabelsOffset + base + 2u];
let previousDistance = distance; let previousHops = hops;
for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
  // CSR lists outgoing arcs. Its opposite is an incoming arc for this vertex.
  let incoming = arcs[arcsOffset + slot] ^ 1u; let edge = incoming / 2u;
  let reverse = (incoming & 1u) != 0u;
  let amount = flow[flowOffset + edge]; let capacity = records[recordsOffset + 4u * edge + 2u];
  if ((!reverse && amount == capacity) || (reverse && amount == 0u)) { continue; }
  let neighbor = records[recordsOffset + 4u * edge + select(0u, 1u, reverse)];
  let neighborDistance = bitcast<i32>(inputLabels[inputLabelsOffset + 3u * neighbor]);
  if (neighborDistance == INFINITY) { continue; }
  let unitCost = i32(records[recordsOffset + 4u * edge + 3u]);
  let candidate = neighborDistance + select(unitCost, -unitCost, reverse);
  let candidateHops = inputLabels[inputLabelsOffset + 3u * neighbor + 1u] + 1u;
  if (candidate < distance || (candidate == distance && (candidateHops < hops || (candidateHops == hops && incoming < predecessor)))) {
    distance = candidate; hops = candidateHops; predecessor = incoming;
  }
}
outputLabels[outputLabelsOffset + base] = bitcast<u32>(distance);
outputLabels[outputLabelsOffset + base + 1u] = hops;
outputLabels[outputLabelsOffset + base + 2u] = predecessor;
if (distance != previousDistance || hops != previousHops) { atomicOr(&state[stateOffset + 5u], 1u); }`,
            searchGate
          ),
          kernel(
            `${prefix}-relax-control-${round}`,
            1,
            [write('state', state), searchGate.binding],
            `if (state[stateOffset + 6u] == 0u) { return; }
state[stateOffset + 7u] = 1u - state[stateOffset + 7u];
if (state[stateOffset + 5u] == 0u) {
  state[stateOffset + 6u] = 0u; state[stateOffset + 17u] = 1u; ${searchGate.disarmWGSL}
}
state[stateOffset + 5u] = 0u;`
          )
        );
      }
      // Normalize the last actually dispatched ping-pong buffer. Later gated rounds did not write
      // their outputs, so static round parity alone cannot identify the final labels.
      nodes.push(
        kernel(
          `${prefix}-normalize`,
          nodeCount,
          [
            read('state', state),
            read('first', labels[0]),
            read('second', labels[1]),
            write('path', paths[0]),
            write('cut', props.output.cut)
          ],
          `if (state[stateOffset] == 0u) { return; }
let base = 3u * index; let useSecond = state[stateOffset + 7u] != 0u;
let distance = select(first[firstOffset + base], second[secondOffset + base], useSecond);
let predecessor = select(first[firstOffset + base + 2u], second[secondOffset + base + 2u], useSecond);
cut[cutOffset + index] = select(0u, 1u, bitcast<i32>(distance) != INFINITY);
path[pathOffset + 4u * index + 2u] = predecessor;`
        ),
        kernel(
          `${prefix}-finish`,
          1,
          [write('state', state), read('cut', props.output.cut), augmentGate.binding],
          `${augmentGate.disarmWGSL}
if (state[stateOffset] == 0u) { return; }
// The requested amount is already optimal after committed shortest-path augmentations.
// A later cut search need not finish to certify that finite request.
if (state[stateOffset + 3u] == state[stateOffset + 16u]) {
  state[stateOffset + 2u] = 1u; state[stateOffset] = 0u; return;
}
if (state[stateOffset + 17u] == 0u) { state[stateOffset] = 0u; return; }
let reachable = cut[cutOffset + state[stateOffset + 15u]] != 0u;
if (!reachable) {
  state[stateOffset + 2u] = 1u; state[stateOffset] = 0u; return;
}
if (state[stateOffset + 4u] == ${maximumAugmentations}u) { state[stateOffset] = 0u; return; }
state[stateOffset + 18u] = 1u;
${augmentGate.armWGSL}`
        ),
        kernel(
          `${prefix}-path-initialize`,
          nodeCount,
          [
            read('edges', props.edges),
            read('capacities', props.capacities),
            read('flow', props.output.flow),
            read('state', state),
            write('path', paths[0])
          ],
          `let base = 4u * index; let predecessor = path[pathOffset + base + 2u];
var ancestor = index; var residual = ${MAXIMUM_TOTAL_CAPACITY}u;
if (predecessor != NONE) {
  let edge = predecessor / 2u; let reverse = (predecessor & 1u) != 0u;
  ancestor = edges[edgesOffset + 2u * edge + select(0u, 1u, reverse)];
  residual = select(capacities[capacitiesOffset + edge] - flow[flowOffset + edge], flow[flowOffset + edge], reverse);
}
path[pathOffset + base] = ancestor; path[pathOffset + base + 1u] = residual;
path[pathOffset + base + 3u] = select(0u, 1u, index == state[stateOffset + 15u]);`,
          augmentGate
        )
      );
      for (let round = 0; round < doublingRounds; round++) {
        const input = paths[round % 2];
        const output = paths[1 - (round % 2)];
        nodes.push(
          kernel(
            `${prefix}-double-${round}`,
            nodeCount,
            [read('inputPath', input), write('outputPath', output)],
            `let base = 4u * index; let ancestor = inputPath[inputPathOffset + base];
outputPath[outputPathOffset + base] = inputPath[inputPathOffset + 4u * ancestor];
outputPath[outputPathOffset + base + 1u] = min(inputPath[inputPathOffset + base + 1u], inputPath[inputPathOffset + 4u * ancestor + 1u]);
outputPath[outputPathOffset + base + 2u] = inputPath[inputPathOffset + base + 2u];
outputPath[outputPathOffset + base + 3u] = inputPath[inputPathOffset + base + 3u];`,
            augmentGate
          ),
          kernel(
            `${prefix}-mark-${round}`,
            nodeCount,
            [read('inputPath', input), atomic('outputPath', output)],
            `let base = 4u * index;
if (inputPath[inputPathOffset + base + 3u] != 0u) {
  let ancestor = inputPath[inputPathOffset + base];
  atomicOr(&outputPath[outputPathOffset + 4u * ancestor + 3u], 1u);
}`,
            augmentGate
          )
        );
      }
      const path = paths[doublingRounds % 2];
      nodes.push(
        kernel(
          `${prefix}-increment`,
          1,
          [read('path', path), write('state', state)],
          `state[stateOffset + 10u] = 0u;
if (state[stateOffset + 18u] == 0u) { return; }
let destination = state[stateOffset + 15u];
let increment = min(path[pathOffset + 4u * destination + 1u], state[stateOffset + 16u] - state[stateOffset + 3u]);
state[stateOffset + 10u] = increment;
state[stateOffset + 3u] += increment; state[stateOffset + 4u] += 1u;`
        ),
        kernel(
          `${prefix}-augment`,
          nodeCount,
          [read('path', path), read('state', state), write('flow', props.output.flow)],
          `let base = 4u * index; let predecessor = path[pathOffset + base + 2u];
if (path[pathOffset + base + 3u] == 0u || predecessor == NONE) { return; }
let edge = predecessor / 2u; let increment = state[stateOffset + 10u];
if ((predecessor & 1u) != 0u) { flow[flowOffset + edge] -= increment; }
else { flow[flowOffset + edge] += increment; }`,
          augmentGate
        )
      );
    }
  }
  nodes.push(
    kernel(
      'collect-objective',
      edgeCount,
      [
        read('edges', props.edges),
        read('capacities', props.capacities),
        read('flow', props.output.flow),
        read('cut', props.output.cut),
        ...(props.costs ? [read('costs', props.costs)] : []),
        atomic('state', state)
      ],
      `if (atomicLoad(&state[stateOffset + 1u]) == 0u) { return; }
let source = edges[edgesOffset + 2u * index]; let destination = edges[edgesOffset + 2u * index + 1u];
if (cut[cutOffset + source] != 0u && cut[cutOffset + destination] == 0u) {
  atomicAdd(&state[stateOffset + 11u], capacities[capacitiesOffset + index]);
}
let amount = flow[flowOffset + index]; let unitCost = ${props.costs ? 'costs[costsOffset + index]' : '0u'};
let product = multiplyUint32Wide(amount, unitCost);
let previous = atomicAdd(&state[stateOffset + 12u], product.y);
atomicAdd(&state[stateOffset + 13u], product.x + select(0u, 1u, previous > 0xffffffffu - product.y));`,
      undefined,
      UINT64_WGSL
    ),
    // contract-allow: single-invocation -- clears exactly eight fixed summary words, not input rows.
    kernel(
      'publish',
      1,
      [
        read('state', state),
        read('cut', props.output.cut),
        write('summary', props.output.summary),
        ...(props.output.exactCost ? [write('exactCost', props.output.exactCost)] : [])
      ],
      `let valid = state[stateOffset + 1u];
for (var word = 0u; word < 8u; word++) { summary[summaryOffset + word] = 0u; }
${props.output.exactCost ? 'exactCost[exactCostOffset] = 0u; exactCost[exactCostOffset + 1u] = 0u;' : ''}
if (valid == 0u) { return; }
let value = state[stateOffset + 3u]; let required = state[stateOffset + 16u];
summary[summaryOffset] = 1u; summary[summaryOffset + 1u] = state[stateOffset + 2u];
summary[summaryOffset + 2u] = value;
summary[summaryOffset + 3u] = select(required - value, 0u, required == NONE);
summary[summaryOffset + 4u] = state[stateOffset + 4u];
summary[summaryOffset + 5u] = bitcast<u32>(uint64ToFloat(vec2<u32>(state[stateOffset + 13u], state[stateOffset + 12u])));
summary[summaryOffset + 6u] = state[stateOffset + 11u];
summary[summaryOffset + 7u] = select(0u, 1u, state[stateOffset + 17u] != 0u && cut[cutOffset + state[stateOffset + 15u]] == 0u && state[stateOffset + 11u] == value);
${props.output.exactCost ? 'exactCost[exactCostOffset] = state[stateOffset + 12u]; exactCost[exactCostOffset + 1u] = state[stateOffset + 13u];' : ''}`,
      undefined,
      UINT64_WGSL
    )
  );
  return nodes;
}
