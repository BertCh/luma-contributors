// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {gateNetworkLouvainPrimitive} from './network-louvain-dispatch';
import {
  createTransientView,
  GPUSort,
  GPUScan,
  GPUGather,
  GPUReduction,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateRequirement,
  validateSinglePackedView,
  validateCommandNodeBudget,
  validateGraphViewsBelongToGraph,
  validateGraphOutputsDisjointFromInputs,
  validateGPUStorageBindingByteLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  validateGPUNetworkTopology,
  getGPUNetworkTopologyViews
} from '../../analysis-core/network-topology';
import {
  createWGSLKernelNode,
  createWGSLDispatchGate,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  getSortedSegmentSumNodes,
  createSegmentSumNode
} from '../../analysis-core/kernels/sorted-segment-sums';
import {getSortedSegmentOffsetNodes} from '../../analysis-core/kernels/sorted-segment-offsets';
import {getBinarySearchWGSL} from '../../analysis-core/kernels/binary-search-wgsl';
import {GET_NAN_WGSL, IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {getNetworkLouvainOutputViews, type GPUNetworkLouvainProps} from './gpu-network-louvain';

/** Pair keys source * nodeCount + destination must leave 0xffffffff available for padding. */
const MAXIMUM_SPARSE_NODE_COUNT = 65535;
/** Two weak-neighbor records per CSR slot bound sort/scan scratch and dispatch size. */
const MAXIMUM_SPARSE_EDGE_COUNT = 1048576;
/** Limits compilation of unrolled multilevel schedules, independently of graph size. */
const MAXIMUM_LEVEL_COUNT = 32;
/** Limits the number of unrolled parallel local-moving rounds at each level. */
const MAXIMUM_ROUND_COUNT = 64;
/** Radix sort (8 digits, <=9 nodes/digit), scans, gathers and reductions, including gates.
 * The node spec checks this allowance at the maximum shape as well as small/empty shapes. */
const INITIAL_NODE_ALLOWANCE = 256;
const ROUND_NODE_ALLOWANCE = 224;
const AGGREGATION_NODE_ALLOWANCE = 112;

type SparseProps = GPUNetworkLouvainProps & {
  topology: NonNullable<GPUNetworkLouvainProps['topology']>;
};
type WordView = GraphDataView<'uint32'>;
type FloatView = GraphDataView<'float32'>;
const OPERATION = 'GPUNetworkLouvain';

/** @internal */
export function estimateSparseNetworkLouvainNodes(props: GPUNetworkLouvainProps): number {
  return (
    INITIAL_NODE_ALLOWANCE +
    (props.maximumLevels ?? 8) *
      ((props.maximumIterations ?? 16) * ROUND_NODE_ALLOWANCE + AGGREGATION_NODE_ALLOWANCE) +
    1
  );
}

/** @internal */
export function validateSparseNetworkLouvain(
  id: string,
  props: SparseProps,
  estimate: number
): void {
  validateCapacity(id, 'nodeCount', props.nodeCount, 1, MAXIMUM_SPARSE_NODE_COUNT);
  const shape = validateGPUNetworkTopology(id, props.topology, {weights: true});
  validateCapacity(id, 'topology.neighbors.length', shape.edgeCount, 0, MAXIMUM_SPARSE_EDGE_COUNT);
  validateRequirement(
    id,
    shape.nodeCount === props.nodeCount,
    'topology.offsets must hold nodeCount + 1 rows'
  );
  validateCapacity(id, 'maximumLevels', props.maximumLevels ?? 8, 1, MAXIMUM_LEVEL_COUNT);
  validateCapacity(id, 'maximumIterations', props.maximumIterations ?? 16, 1, MAXIMUM_ROUND_COUNT);
  validateCapacity(id, 'seed', props.seed ?? 0, 0, 0xffffffff);
  validateCommandNodeBudget(id, estimate);
  for (const [name, view, format, length] of [
    ['parameters', props.parameters, 'float32', 2],
    ['output.communities', props.output.communities, 'uint32', props.nodeCount],
    [
      'output.objectiveTrace',
      props.output.objectiveTrace,
      'float32',
      1 + (props.maximumLevels ?? 8) * (props.maximumIterations ?? 16)
    ],
    ['output.summary', props.output.summary, 'uint32', 4]
  ] as const) {
    validateSinglePackedView(view, [format], id);
    validateRequirement(id, view.length === length, `${name} must hold ${length} rows`);
  }
  validateGraphOutputsDisjointFromInputs(id, getNetworkLouvainOutputViews(props.output), [
    ...getGPUNetworkTopologyViews(props.topology),
    props.parameters
  ]);
}

/** Sparse multilevel moving with fixed-order reductions and community-disjoint commits.
 * No per-invocation graph walk or per-thread neighborhood table. Quotient edges are sorted
 * and reduced by (source,destination); their storage never exceeds twice the original CSR capacity.
 * @internal
 */
export function getSparseNetworkLouvainNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: SparseProps
): GPUCommandNode<Parameters>[] {
  const nodeCount = props.nodeCount;
  const edgeCount = props.topology.neighbors.length;
  const recordCount = Math.max(2 * edgeCount, 1);
  // Radix histograms use 16 words per 256-row block; the largest allocation is the payload.
  validateGPUStorageBindingByteLength(
    graph.device,
    id,
    'sparse edge scratch',
    (recordCount + 1) * 4,
    'reduce topology edge capacity'
  );
  validateGPUStorageBindingByteLength(
    graph.device,
    id,
    'sparse node scratch',
    Math.max(nodeCount * 4, nodeCount + 1) * 4,
    'reduce nodeCount'
  );
  validateGraphViewsBelongToGraph(id, graph, [
    ...getGPUNetworkTopologyViews(props.topology),
    props.parameters,
    ...getNetworkLouvainOutputViews(props.output)
  ]);
  const nodes: GPUCommandNode<Parameters>[] = [];
  const word = (name: string, length: number) =>
    createTransientView(graph, `${id}-${name}`, 'uint32', length);
  const float = (name: string, length: number) =>
    createTransientView(graph, `${id}-${name}`, 'float32', length);
  const read = (name: string, view: GraphDataView): WGSLKernelBinding => ({
    name,
    view,
    type: view.format === 'float32' ? 'f32' : 'u32',
    access: 'read'
  });
  const write = (name: string, view: GraphDataView, atomic = false): WGSLKernelBinding => ({
    name,
    view,
    type: atomic ? 'atomic<u32>' : view.format === 'float32' ? 'f32' : 'u32',
    access: 'read_write'
  });
  const declarations = `const NODE_COUNT: u32 = ${nodeCount}u;\nconst RECORD_COUNT: u32 = ${recordCount}u;\nconst NONE: u32 = 0xffffffffu;\n${IS_FINITE_VALUE_WGSL}`;
  const kernel = (
    name: string,
    count: number,
    bindings: WGSLKernelBinding[],
    body: string,
    extra = ''
  ) =>
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-${name}`,
      operation: OPERATION,
      invocationCount: count,
      bindings,
      declarations: `${declarations}\n${extra}`,
      body
    });
  const state = word('state', 8);
  // [running, roundActive, levelChanged, unused, valid, converged, levels, rounds]
  const stage = (
    name: string,
    commands: readonly GPUCommandNode<Parameters>[],
    predicate: string
  ) => {
    const gate = createWGSLDispatchGate(graph, {
      id: `${id}-${name}-dispatch`,
      operation: OPERATION,
      shapes: commands.map(
        command => [(command.workload?.maximumWorkgroupCount ?? 1) * 2, 2] as const
      )
    });
    nodes.push(
      kernel(
        `${name}-gate`,
        1,
        [read('state', state), gate.binding],
        `if (${predicate}) { ${gate.armWGSL} } else { ${gate.disarmWGSL} }`
      )
    );
    commands.forEach((command, slot) => {
      if (command.type !== 'compute') throw new Error(`${id} sparse stage requires compute nodes`);
      if (!command.workload?.maximumWorkgroupCount) {
        nodes.push(gateNetworkLouvainPrimitive(command, state, predicate));
        return;
      }
      const gated = gate.getGate(slot);
      nodes.push({
        ...command,
        condition: gated.condition,
        resources: [...(command.resources ?? []), ...gated.extraResources]
      });
    });
  };
  const reduction = (name: string, input: FloatView, output: FloatView) =>
    new GPUReduction({id: `${id}-${name}`, input, output, operation: 'sum'}).getCommandNodes(graph);
  const labels = word('labels', nodeCount);
  const groups = word('groups', nodeCount);
  const degreeOut = float('degree-out', nodeCount);
  const degreeIn = float('degree-in', nodeCount);
  const totalsOut = float('totals-out', nodeCount);
  const totalsIn = float('totals-in', nodeCount);
  const edgeKeys = word('edge-keys', recordCount);
  const edgeWeights = float('edge-weights', recordCount);
  const sourceKeys = word('source-keys', Math.max(edgeCount, 1));
  const targetKeys = word('destination-keys', Math.max(edgeCount, 1));
  const rawWeights = float('raw-weights', Math.max(edgeCount, 1));
  const invalid = word('invalid', Math.max(nodeCount + 1, edgeCount, 1));
  const invalidCount = word('invalid-count', 1);
  const total = float('total', 1);
  const changed = word('changed', nodeCount);
  const anyChanged = word('any-changed', 1);
  const proposals = word('proposals', nodeCount);
  const reservations = word('reservations', nodeCount);
  const accepted = word('accepted', nodeCount);
  const proposalKeys = word('proposal-keys', recordCount);
  const proposalWeights = float('proposal-weights', recordCount);

  const collectTotals = (prefix: string) =>
    getSortedSegmentSumNodes(graph, {
      id: `${id}-${prefix}`,
      operation: OPERATION,
      segmentCount: nodeCount,
      segmentKeys: labels,
      reductions: [
        {name: 'out', contributions: degreeOut, output: totalsOut},
        {name: 'in', contributions: degreeIn, output: totalsIn}
      ]
    });
  const objective = (prefix: string, initial: boolean): GPUCommandNode<Parameters>[] => {
    const internal = float(`${prefix}-internal`, recordCount);
    const penalty = float(`${prefix}-penalty`, nodeCount);
    const internalTotal = float(`${prefix}-internal-total`, 1);
    const penaltyTotal = float(`${prefix}-penalty-total`, 1);
    return [
      kernel(
        `${prefix}-internal`,
        recordCount,
        [
          read('edgeKeys', edgeKeys),
          read('edgeWeights', edgeWeights),
          read('labels', labels),
          write('internal', internal)
        ],
        `let key = edgeKeys[edgeKeysOffset + index];
var value = 0.0;
if (key != NONE && labels[labelsOffset + key / NODE_COUNT] == labels[labelsOffset + key % NODE_COUNT]) { value = 0.5 * edgeWeights[edgeWeightsOffset + index]; }
internal[internalOffset + index] = value;`
      ),
      kernel(
        `${prefix}-penalty`,
        nodeCount,
        [
          read('totalsOut', totalsOut),
          read('totalsIn', totalsIn),
          read('parameters', props.parameters),
          write('penalty', penalty)
        ],
        'penalty[penaltyOffset + index] = parameters[parametersOffset] * totalsOut[totalsOutOffset + index] * totalsIn[totalsInOffset + index];'
      ),
      ...reduction(`${prefix}-internal-total`, internal, internalTotal),
      ...reduction(`${prefix}-penalty-total`, penalty, penaltyTotal),
      kernel(
        `${prefix}-trace`,
        1,
        [
          read('internalTotal', internalTotal),
          read('penaltyTotal', penaltyTotal),
          write('state', state),
          write('trace', props.output.objectiveTrace)
        ],
        `let value = internalTotal[internalTotalOffset] - penaltyTotal[penaltyTotalOffset];
if (!isFiniteValue(value)) { state[stateOffset] = 0u; state[stateOffset + 4u] = 0u; return; }
trace[traceOffset + ${initial ? '0u' : 'state[stateOffset + 7u] + 1u'}] = value;`
      )
    ];
  };

  // Arbitrary pair keys become consecutive segment numbers through the published scan.
  // A fixed tree then reduces each segment cooperatively, including arbitrarily high degree.
  const aggregatePairs = (prefix: string, keys: WordView, weights: FloatView) => {
    const order = word(`${prefix}-order`, recordCount);
    const sortedKeys = word(`${prefix}-sorted-keys`, recordCount);
    const sortedOrder = word(`${prefix}-sorted-order`, recordCount);
    const sortedWeights = float(`${prefix}-sorted-weights`, recordCount);
    const flags = word(`${prefix}-flags`, recordCount);
    const ranks = word(`${prefix}-ranks`, recordCount);
    const offsets = word(`${prefix}-offsets`, recordCount + 1);
    const pairKeys = word(`${prefix}-pair-keys`, recordCount);
    const pairWeights = float(`${prefix}-pair-weights`, recordCount);
    const commands: GPUCommandNode<Parameters>[] = [
      kernel(
        `${prefix}-order`,
        recordCount,
        [write('order', order)],
        'order[orderOffset + index] = index;'
      ),
      ...new GPUSort({
        id: `${id}-${prefix}-sort`,
        keys,
        values: order,
        outputKeys: sortedKeys,
        outputValues: sortedOrder,
        algorithm: 'radix'
      }).getCommandNodes(graph),
      ...new GPUGather({
        id: `${id}-${prefix}-gather`,
        source: weights,
        indices: sortedOrder,
        output: sortedWeights
      }).getCommandNodes(graph),
      kernel(
        `${prefix}-flags`,
        recordCount,
        [
          read('sortedKeys', sortedKeys),
          write('flags', flags),
          write('offsets', offsets),
          write('pairKeys', pairKeys)
        ],
        `var starts = index == 0u;
if (index > 0u) { starts = sortedKeys[sortedKeysOffset + index] != sortedKeys[sortedKeysOffset + index - 1u]; }
flags[flagsOffset + index] = select(0u, 1u, starts);
offsets[offsetsOffset + index] = RECORD_COUNT;
if (index == 0u) { offsets[offsetsOffset + RECORD_COUNT] = RECORD_COUNT; }
pairKeys[pairKeysOffset + index] = NONE;`
      ),
      ...new GPUScan({
        id: `${id}-${prefix}-scan`,
        input: flags,
        output: ranks,
        mode: 'inclusive'
      }).getCommandNodes(graph),
      kernel(
        `${prefix}-segments`,
        recordCount,
        [
          read('sortedKeys', sortedKeys),
          read('flags', flags),
          read('ranks', ranks),
          write('offsets', offsets),
          write('pairKeys', pairKeys)
        ],
        `if (flags[flagsOffset + index] != 0u) {
let segment = ranks[ranksOffset + index] - 1u;
offsets[offsetsOffset + segment] = index;
pairKeys[pairKeysOffset + segment] = sortedKeys[sortedKeysOffset + index];
}`
      ),
      createSegmentSumNode(graph, {
        id: `${id}-${prefix}-sum`,
        operation: OPERATION,
        segmentCount: recordCount,
        input: sortedWeights,
        segmentOffsets: offsets,
        output: pairWeights
      })
    ];
    return {commands, pairKeys, pairWeights};
  };

  nodes.push(
    kernel(
      'initialize-state',
      Math.max(props.output.objectiveTrace.length, 8),
      [write('state', state), write('trace', props.output.objectiveTrace)],
      `if (index < 8u) { state[stateOffset + index] = 0u; }
if (index < ${props.output.objectiveTrace.length}u) { trace[traceOffset + index] = getNaN(); }`,
      GET_NAN_WGSL
    )
  );
  nodes.push(
    kernel(
      'validate-offsets',
      invalid.length,
      [read('offsets', props.topology.offsets), write('invalid', invalid)],
      `var failed = false;
if (index <= NODE_COUNT) {
let offset = offsets[offsetsOffset + index];
failed = offset > ${edgeCount}u || (index == 0u && offset != 0u);
if (index < NODE_COUNT) { failed = failed || offset > offsets[offsetsOffset + index + 1u]; }
}
invalid[invalidOffset + index] = select(0u, 1u, failed);`
    )
  );
  nodes.push(
    ...new GPUReduction({
      id: `${id}-invalid-offset-count`,
      input: invalid,
      output: invalidCount,
      operation: 'sum'
    }).getCommandNodes(graph)
  );
  nodes.push(
    kernel(
      'prepare-arcs',
      rawWeights.length,
      [
        read('offsets', props.topology.offsets),
        read('neighbors', props.topology.neighbors),
        read('weights', props.topology.weights),
        read('invalidCount', invalidCount),
        write('sourceKeys', sourceKeys),
        write('targetKeys', targetKeys),
        write('rawWeights', rawWeights),
        write('invalid', invalid)
      ],
      `sourceKeys[sourceKeysOffset + index] = NODE_COUNT;
targetKeys[targetKeysOffset + index] = NODE_COUNT;
rawWeights[rawWeightsOffset + index] = 0.0;
invalid[invalidOffset + index] = 0u;
if (invalidCount[invalidCountOffset] != 0u || index >= ${edgeCount}u || index >= offsets[offsetsOffset + NODE_COUNT]) { return; }
let destination = neighbors[neighborsOffset + index];
if (destination >= NODE_COUNT) { return; }
let weight = weights[weightsOffset + index];
if (!isFiniteValue(weight) || weight < 0.0) { invalid[invalidOffset + index] = 1u; return; }
sourceKeys[sourceKeysOffset + index] = upperBoundOffsets(0u, NODE_COUNT + 1u, index) - 1u;
targetKeys[targetKeysOffset + index] = destination;
rawWeights[rawWeightsOffset + index] = weight;`,
      getBinarySearchWGSL('offsets')
    )
  );
  const invalidWeights = word('invalid-weight-count', 1);
  nodes.push(
    ...new GPUReduction({
      id: `${id}-invalid-weight-count`,
      input: invalid,
      output: invalidWeights,
      operation: 'sum'
    }).getCommandNodes(graph)
  );
  nodes.push(...reduction('total-weight', rawWeights, total));
  for (const [name, keys, output] of [
    ['out', sourceKeys, degreeOut],
    ['in', targetKeys, degreeIn]
  ] as const) {
    nodes.push(
      ...getSortedSegmentSumNodes(graph, {
        id: `${id}-initial-${name}`,
        operation: OPERATION,
        segmentCount: nodeCount,
        segmentKeys: keys,
        sumContributions: rawWeights,
        sums: output
      })
    );
  }
  nodes.push(
    kernel(
      'initialize-vertices',
      nodeCount,
      [
        write('degreeOut', degreeOut),
        write('degreeIn', degreeIn),
        read('total', total),
        write('labels', labels),
        write('groups', groups),
        write('communities', props.output.communities),
        write('totalsOut', totalsOut),
        write('totalsIn', totalsIn)
      ],
      `let mass = total[totalOffset];
var outgoing = 0.0; var incoming = 0.0;
if (mass > 0.0 && isFiniteValue(mass)) { outgoing = degreeOut[degreeOutOffset + index] / mass; incoming = degreeIn[degreeInOffset + index] / mass; }
degreeOut[degreeOutOffset + index] = outgoing; degreeIn[degreeInOffset + index] = incoming;
totalsOut[totalsOutOffset + index] = outgoing; totalsIn[totalsInOffset + index] = incoming;
labels[labelsOffset + index] = index; groups[groupsOffset + index] = index; communities[communitiesOffset + index] = index;`
    )
  );
  nodes.push(
    kernel(
      'initialize-edges',
      recordCount,
      [
        read('sourceKeys', sourceKeys),
        read('targetKeys', targetKeys),
        read('rawWeights', rawWeights),
        read('total', total),
        write('edgeKeys', edgeKeys),
        write('edgeWeights', edgeWeights)
      ],
      `edgeKeys[edgeKeysOffset + index] = NONE; edgeWeights[edgeWeightsOffset + index] = 0.0;
let slot = index / 2u;
if (slot >= ${edgeCount}u) { return; }
let source = sourceKeys[sourceKeysOffset + slot]; let destination = targetKeys[targetKeysOffset + slot];
let mass = total[totalOffset];
if (rawWeights[rawWeightsOffset + slot] == 0.0 || source >= NODE_COUNT || destination >= NODE_COUNT || !(mass > 0.0) || !isFiniteValue(mass)) { return; }
var first = source; var second = destination;
if (index % 2u == 1u) { first = destination; second = source; }
edgeKeys[edgeKeysOffset + index] = first * NODE_COUNT + second;
edgeWeights[edgeWeightsOffset + index] = rawWeights[rawWeightsOffset + slot] / mass;`
    )
  );
  const symmetryInvalidCount = word('symmetry-invalid-count', 1);
  if (!props.directed) {
    nodes.push(
      kernel(
        'symmetry-keys',
        recordCount,
        [
          read('sourceKeys', sourceKeys),
          read('targetKeys', targetKeys),
          read('rawWeights', rawWeights),
          write('proposalKeys', proposalKeys),
          write('proposalWeights', proposalWeights)
        ],
        `proposalKeys[proposalKeysOffset + index] = NONE; proposalWeights[proposalWeightsOffset + index] = 0.0;
if (index >= ${edgeCount}u) { return; }
let source = sourceKeys[sourceKeysOffset + index]; let destination = targetKeys[targetKeysOffset + index];
if (source >= NODE_COUNT || destination >= NODE_COUNT) { return; }
proposalKeys[proposalKeysOffset + index] = source * NODE_COUNT + destination;
proposalWeights[proposalWeightsOffset + index] = rawWeights[rawWeightsOffset + index];`
      )
    );
    const pairs = aggregatePairs('symmetry', proposalKeys, proposalWeights);
    const failures = word('symmetry-invalid', recordCount);
    nodes.push(
      ...pairs.commands,
      kernel(
        'symmetry-validate',
        recordCount,
        [
          read('pairKeys', pairs.pairKeys),
          read('pairWeights', pairs.pairWeights),
          write('failures', failures)
        ],
        `failures[failuresOffset + index] = 0u;
let key = pairKeys[pairKeysOffset + index]; if (key == NONE) { return; }
let reverseKey = (key % NODE_COUNT) * NODE_COUNT + key / NODE_COUNT;
let position = lowerBoundPairKeys(0u, RECORD_COUNT, reverseKey);
var reverseWeight = 0.0;
if (position < RECORD_COUNT && pairKeys[pairKeysOffset + position] == reverseKey) { reverseWeight = pairWeights[pairWeightsOffset + position]; }
let weight = pairWeights[pairWeightsOffset + index];
if (abs(weight - reverseWeight) > 0.000001 * max(abs(weight), abs(reverseWeight))) { failures[failuresOffset + index] = 1u; }`,
        getBinarySearchWGSL('pairKeys')
      ),
      ...new GPUReduction({
        id: `${id}-symmetry-invalid-count`,
        input: failures,
        output: symmetryInvalidCount,
        operation: 'sum'
      }).getCommandNodes(graph)
    );
  } else {
    nodes.push(
      kernel(
        'symmetry-skip',
        1,
        [write('symmetryInvalidCount', symmetryInvalidCount)],
        'symmetryInvalidCount[symmetryInvalidCountOffset] = 0u;'
      )
    );
  }
  nodes.push(
    kernel(
      'initialize-control',
      1,
      [
        write('state', state),
        read('parameters', props.parameters),
        read('invalidCount', invalidCount),
        read('invalidWeights', invalidWeights),
        read('symmetryInvalidCount', symmetryInvalidCount),
        read('total', total)
      ],
      `let resolution = parameters[parametersOffset]; let tolerance = parameters[parametersOffset + 1u];
let valid = symmetryInvalidCount[symmetryInvalidCountOffset] == 0u && invalidCount[invalidCountOffset] == 0u && invalidWeights[invalidWeightsOffset] == 0u && isFiniteValue(total[totalOffset]) && isFiniteValue(resolution) && resolution >= 0.0 && isFiniteValue(tolerance) && tolerance >= 0.0;
state[stateOffset + 4u] = select(0u, 1u, valid);
state[stateOffset] = select(0u, 1u, valid && total[totalOffset] > 0.0);
state[stateOffset + 1u] = 1u;
state[stateOffset + 5u] = select(0u, 1u, valid && total[totalOffset] == 0.0);`
    )
  );
  stage('initial-objective', objective('initial', true), 'state[stateOffset + 4u] != 0u');

  const maximumLevels = props.maximumLevels ?? 8;
  const maximumIterations = props.maximumIterations ?? 16;
  for (let level = 0; level < maximumLevels; level++) {
    for (let round = 0; round < maximumIterations; round++) {
      const prefix = `level-${level}-round-${round}`;
      const commands: GPUCommandNode<Parameters>[] = [
        kernel(
          `${prefix}-reset`,
          nodeCount,
          [write('proposals', proposals), write('reservations', reservations)],
          'proposals[proposalsOffset + index] = NONE; reservations[reservationsOffset + index] = NONE;'
        ),
        kernel(
          `${prefix}-neighbor-communities`,
          recordCount,
          [
            read('edgeKeys', edgeKeys),
            read('edgeWeights', edgeWeights),
            read('labels', labels),
            write('proposalKeys', proposalKeys),
            write('proposalWeights', proposalWeights)
          ],
          `let key = edgeKeys[edgeKeysOffset + index];
proposalKeys[proposalKeysOffset + index] = NONE; proposalWeights[proposalWeightsOffset + index] = 0.0;
if (key == NONE) { return; }
let source = key / NODE_COUNT; let destination = key % NODE_COUNT;
proposalKeys[proposalKeysOffset + index] = source * NODE_COUNT + labels[labelsOffset + destination];
// A self-loop is invariant under a move and must not enter its connection gain.
if (source != destination) { proposalWeights[proposalWeightsOffset + index] = edgeWeights[edgeWeightsOffset + index]; }`
        )
      ];
      const pairs = aggregatePairs(`${prefix}-candidates`, proposalKeys, proposalWeights);
      commands.push(...pairs.commands);
      const gains = float(`${prefix}-gains`, recordCount);
      const nodeKeys = word(`${prefix}-node-keys`, recordCount);
      const bestGain = float(`${prefix}-best-gain`, nodeCount);
      const nodeOffsets = word(`${prefix}-node-offsets`, nodeCount + 1);
      // Bind the two strengths and two community totals as one scalar view each to stay at eight bindings.
      const strengths = float(`${prefix}-strengths`, nodeCount * 4);
      commands.push(
        kernel(
          `${prefix}-strengths`,
          nodeCount,
          [
            read('degreeOut', degreeOut),
            read('degreeIn', degreeIn),
            read('totalsOut', totalsOut),
            read('totalsIn', totalsIn),
            write('strengths', strengths)
          ],
          `strengths[strengthsOffset + index * 4u] = degreeOut[degreeOutOffset + index];
strengths[strengthsOffset + index * 4u + 1u] = degreeIn[degreeInOffset + index];
strengths[strengthsOffset + index * 4u + 2u] = totalsOut[totalsOutOffset + index];
strengths[strengthsOffset + index * 4u + 3u] = totalsIn[totalsInOffset + index];`
        )
      );
      commands.push(
        kernel(
          `${prefix}-gain`,
          recordCount,
          [
            read('pairKeys', pairs.pairKeys),
            read('pairWeights', pairs.pairWeights),
            read('labels', labels),
            read('strengths', strengths),
            read('parameters', props.parameters),
            write('gains', gains),
            write('nodeKeys', nodeKeys)
          ],
          `let key = pairKeys[pairKeysOffset + index];
gains[gainsOffset + index] = 0.0; nodeKeys[nodeKeysOffset + index] = NODE_COUNT;
if (key == NONE) { return; }
let source = key / NODE_COUNT; let destination = key % NODE_COUNT; let original = labels[labelsOffset + source];
nodeKeys[nodeKeysOffset + index] = source;
if (destination == original) { return; }
let originalKey = source * NODE_COUNT + original;
let position = lowerBoundPairKeys(0u, RECORD_COUNT, originalKey);
var connection = 0.0;
if (position < RECORD_COUNT && pairKeys[pairKeysOffset + position] == originalKey) { connection = pairWeights[pairWeightsOffset + position]; }
let outgoing = strengths[strengthsOffset + source * 4u]; let incoming = strengths[strengthsOffset + source * 4u + 1u];
let penalty = outgoing * (strengths[strengthsOffset + destination * 4u + 3u] - strengths[strengthsOffset + original * 4u + 3u] + incoming) + incoming * (strengths[strengthsOffset + destination * 4u + 2u] - strengths[strengthsOffset + original * 4u + 2u] + outgoing);
let gain = pairWeights[pairWeightsOffset + index] - connection - parameters[parametersOffset] * penalty;
if (isFiniteValue(gain) && gain > parameters[parametersOffset + 1u]) { gains[gainsOffset + index] = gain; }`,
          getBinarySearchWGSL('pairKeys')
        )
      );
      commands.push(
        ...getSortedSegmentOffsetNodes(graph, {
          id: `${id}-${prefix}-nodes`,
          operation: OPERATION,
          groupCount: nodeCount,
          sortedGroupKeys: nodeKeys,
          segmentOffsets: nodeOffsets
        })
      );
      commands.push(
        createSegmentSumNode(graph, {
          id: `${id}-${prefix}-best-gain`,
          operation: OPERATION,
          segmentCount: nodeCount,
          input: gains,
          segmentOffsets: nodeOffsets,
          output: bestGain,
          reduction: 'max'
        })
      );
      commands.push(
        kernel(
          `${prefix}-choose`,
          recordCount,
          [
            read('pairKeys', pairs.pairKeys),
            read('gains', gains),
            read('bestGain', bestGain),
            write('proposals', proposals, true)
          ],
          `let key = pairKeys[pairKeysOffset + index]; if (key == NONE) { return; }
let source = key / NODE_COUNT; let gain = gains[gainsOffset + index];
if (gain > 0.0 && gain == bestGain[bestGainOffset + source]) { atomicMin(&proposals[proposalsOffset + source], key % NODE_COUNT); }`
        )
      );
      commands.push(
        kernel(
          `${prefix}-reserve`,
          nodeCount,
          [
            read('proposals', proposals),
            read('labels', labels),
            write('reservations', reservations, true)
          ],
          `let destination = proposals[proposalsOffset + index]; if (destination == NONE) { return; }
let priority = index ^ ${props.seed ?? 0}u;
atomicMin(&reservations[reservationsOffset + labels[labelsOffset + index]], priority);
atomicMin(&reservations[reservationsOffset + destination], priority);`
        )
      );
      commands.push(
        kernel(
          `${prefix}-accept`,
          nodeCount,
          [
            read('proposals', proposals),
            read('labels', labels),
            read('reservations', reservations),
            write('accepted', accepted),
            write('changed', changed)
          ],
          `let proposedCommunity = proposals[proposalsOffset + index]; let originalCommunity = labels[labelsOffset + index]; let priority = index ^ ${props.seed ?? 0}u;
var acceptedCommunity = originalCommunity;
if (proposedCommunity != NONE) {
  if (reservations[reservationsOffset + originalCommunity] == priority && reservations[reservationsOffset + proposedCommunity] == priority) { acceptedCommunity = proposedCommunity; }
}
accepted[acceptedOffset + index] = acceptedCommunity; changed[changedOffset + index] = select(0u, 1u, acceptedCommunity != originalCommunity);`
        )
      );
      commands.push(
        kernel(
          `${prefix}-apply`,
          nodeCount,
          [
            read('accepted', accepted),
            read('groups', groups),
            write('labels', labels),
            write('communities', props.output.communities)
          ],
          'labels[labelsOffset + index] = accepted[acceptedOffset + index]; communities[communitiesOffset + index] = accepted[acceptedOffset + groups[groupsOffset + index]];'
        )
      );
      commands.push(
        ...new GPUReduction({
          id: `${id}-${prefix}-changed`,
          input: changed,
          output: anyChanged,
          operation: 'max'
        }).getCommandNodes(graph)
      );
      commands.push(...collectTotals(`${prefix}-totals`), ...objective(prefix, false));
      stage(prefix, commands, 'state[stateOffset] != 0u && state[stateOffset + 1u] != 0u');
      nodes.push(
        kernel(
          `${prefix}-control`,
          1,
          [write('state', state), read('anyChanged', anyChanged)],
          `if (state[stateOffset] == 0u || state[stateOffset + 1u] == 0u) { return; }
state[stateOffset + 6u] = ${level + 1}u; state[stateOffset + 7u]++;
if (anyChanged[anyChangedOffset] == 0u) {
  state[stateOffset + 1u] = 0u;
  if (state[stateOffset + 2u] == 0u) { state[stateOffset + 5u] = 1u; state[stateOffset] = 0u; }
  ${level + 1 === maximumLevels ? 'state[stateOffset] = 0u;' : ''}
} else {
  state[stateOffset + 2u] = 1u;
  ${round + 1 === maximumIterations ? 'state[stateOffset] = 0u;' : ''}
}`
        )
      );
    }
    if (level + 1 < maximumLevels) {
      const prefix = `aggregate-${level}`;
      const commands = [
        kernel(
          `${prefix}-keys`,
          recordCount,
          [read('edgeKeys', edgeKeys), read('labels', labels), write('proposalKeys', proposalKeys)],
          `let key = edgeKeys[edgeKeysOffset + index]; var next = NONE;
if (key != NONE) { next = labels[labelsOffset + key / NODE_COUNT] * NODE_COUNT + labels[labelsOffset + key % NODE_COUNT]; }
proposalKeys[proposalKeysOffset + index] = next;`
        )
      ];
      const pairs = aggregatePairs(prefix, proposalKeys, edgeWeights);
      commands.push(...pairs.commands);
      commands.push(
        kernel(
          `${prefix}-edges`,
          recordCount,
          [
            read('pairKeys', pairs.pairKeys),
            read('pairWeights', pairs.pairWeights),
            write('edgeKeys', edgeKeys),
            write('edgeWeights', edgeWeights)
          ],
          'edgeKeys[edgeKeysOffset + index] = pairKeys[pairKeysOffset + index]; edgeWeights[edgeWeightsOffset + index] = pairWeights[pairWeightsOffset + index];'
        )
      );
      commands.push(
        kernel(
          `${prefix}-vertices`,
          nodeCount,
          [
            read('totalsOut', totalsOut),
            read('totalsIn', totalsIn),
            read('communities', props.output.communities),
            write('degreeOut', degreeOut),
            write('degreeIn', degreeIn),
            write('groups', groups),
            write('labels', labels)
          ],
          'degreeOut[degreeOutOffset + index] = totalsOut[totalsOutOffset + index]; degreeIn[degreeInOffset + index] = totalsIn[totalsInOffset + index]; groups[groupsOffset + index] = communities[communitiesOffset + index]; labels[labelsOffset + index] = index;'
        )
      );
      stage(prefix, commands, 'state[stateOffset] != 0u && state[stateOffset + 1u] == 0u');
      nodes.push(
        kernel(
          `${prefix}-control`,
          1,
          [write('state', state)],
          'if (state[stateOffset] != 0u && state[stateOffset + 1u] == 0u) { state[stateOffset + 1u] = 1u; state[stateOffset + 2u] = 0u; }'
        )
      );
    }
  }
  nodes.push(
    kernel(
      'publish-summary',
      1,
      [read('state', state), write('summary', props.output.summary)],
      'summary[summaryOffset] = state[stateOffset + 4u]; summary[summaryOffset + 1u] = state[stateOffset + 5u]; summary[summaryOffset + 2u] = state[stateOffset + 6u]; summary[summaryOffset + 3u] = state[stateOffset + 7u];'
    )
  );
  return nodes;
}
