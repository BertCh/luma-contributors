// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {GPUCommandGraph, type GPUCommandNode, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  GPUParameterBuffer,
  importGraphBuffer,
  validateCapacity,
  validateRequirement,
  validateGPUStorageBindingByteLength,
  validateCommandNodeBudget
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getCoreDefaultBuffer} from '../../analysis-core/contracts/network-topology';
import {createFillNode, createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  createReachabilityInitializeNode,
  createReachabilitySeedNode,
  createReachabilityRelaxNode
} from '../network-reachability/network-reachability-passes';
import {
  getFrontierPhaseLayout,
  FRONTIER_DISPATCH_SLOT_WORDS,
  getFrontierDispatchSlotByteOffset,
  type FrontierState,
  type FrontierPhase
} from '../network-reachability/network-frontier';
import {
  createCentralityPackNode,
  createCentralityExpandSeedsNode,
  createCentralityResetNode,
  createCentralitySigmaInitializeNode,
  createCentralitySigmaRoundNode,
  createCentralityDeltaInitializeNode,
  createCentralityDeltaRoundNode,
  createCentralityAccumulateNode,
  createCentralityEdgeAccumulateNode,
  createCentralityClosenessNode,
  getCentralityAdjacencyLayout
} from './network-centrality-passes';
import {
  createNetworkCentralityForComposition,
  validateNetworkCentralityBindings,
  type GPUNetworkCentralityProps
} from './gpu-network-centrality';

/** Reserve a boundary round while retaining the existing 1024-round per-window validation envelope. */
const CENTRALITY_CONTINUATION_MAXIMUM_ROUNDS_PER_STEP = 1023;
/** Total phase rounds use uint32 host metadata; window-local stamps are cleared on every carry. */
const CENTRALITY_CONTINUATION_MAXIMUM_ITERATIONS = 0xfffffffe;
/** Small reusable graphs amortize readbacks without a graph proportional to the hop diameter. */
const CENTRALITY_CONTINUATION_DEFAULT_ROUNDS_PER_STEP = 64;

export type NetworkCentralityContinuationProps = GPUNetworkCentralityProps & {
  /** Rounds per submitted window, 1..1023 (default 64). Independent of total maximumIterations. */
  roundsPerStep?: number;
};

type CentralityContinuationPhase = 'search' | 'sigma' | 'delta';

/** Host round loop shared by the real runner and the long-path CPU oracle spec. @internal */
export async function runCentralityContinuationPhases(props: {
  maximumIterations: number;
  roundsPerStep: number;
  hasBetweenness: boolean;
  initialize: (phase: CentralityContinuationPhase) => void;
  execute: (
    phase: CentralityContinuationPhase,
    roundCount: number,
    carry: boolean
  ) => Promise<number>;
}): Promise<readonly boolean[]> {
  const completed: boolean[] = [];
  for (const phase of ['search', 'sigma', 'delta'] as const) {
    if (phase !== 'search' && !props.hasBetweenness) {
      completed.push(true);
      continue;
    }
    props.initialize(phase);
    let done = false;
    for (let round = 0; round < props.maximumIterations; round += props.roundsPerStep) {
      const count = await props.execute(
        phase,
        Math.min(props.roundsPerStep, props.maximumIterations - round),
        round !== 0
      );
      if (count === 0) {
        done = true;
        break;
      }
    }
    completed.push(done);
  }
  return completed;
}

/**
 * Builds bounded graphs and persistent frontier storage. Kept separate for recording-device
 * tests; hosts use createNetworkCentralityContinuation. No graph or buffer is declared until
 * the complete external-view and derived-binding preflight succeeds. @internal
 */
export function createCentralityContinuationGraphs(
  originalGraph: GPUCommandGraph,
  props: NetworkCentralityContinuationProps
) {
  const id = props.id ?? 'network-centrality-continuation';
  const roundsPerStep = props.roundsPerStep ?? CENTRALITY_CONTINUATION_DEFAULT_ROUNDS_PER_STEP;
  validateCapacity(
    id,
    'roundsPerStep',
    roundsPerStep,
    1,
    CENTRALITY_CONTINUATION_MAXIMUM_ROUNDS_PER_STEP
  );
  const nodeCount = props.topology.offsets.length - 1;
  const maximumIterations = props.maximumIterations ?? nodeCount;
  validateCapacity(
    id,
    'maximumIterations',
    maximumIterations,
    1,
    CENTRALITY_CONTINUATION_MAXIMUM_ITERATIONS
  );
  validateRequirement(
    id,
    props.laneCount === undefined || props.laneCount === props.pivotNodes.length,
    'laneCount must equal pivotNodes.length (one persistent batch)'
  );
  const laneCount = props.pivotNodes.length;
  const expandedNodeCount = nodeCount * laneCount;
  const phase: FrontierPhase = {
    maxRounds: roundsPerStep + 1,
    stampBase: 0,
    controlWordOffset: 0,
    dispatchSlotOffset: 0
  };
  const phaseLayout = getFrontierPhaseLayout(phase.maxRounds);
  const device = originalGraph.device;
  const hasBetweenness = Boolean(props.output.betweenness || props.output.edgeBetweenness);
  const validator = createNetworkCentralityForComposition({
    ...props,
    id,
    laneCount,
    maximumIterations: phase.maxRounds
  });
  validateNetworkCentralityBindings(originalGraph, validator);
  props = {...props, reverseTopology: validator.props.reverseTopology};
  const externalViews = [
    props.pivotNodes,
    props.costLimit,
    props.topology.offsets,
    props.topology.neighbors,
    props.topology.weights,
    props.reverseTopology!.offsets,
    props.reverseTopology!.neighbors,
    props.reverseTopology!.weights,
    props.output.betweenness,
    props.output.edgeBetweenness,
    props.output.closeness,
    props.output.harmonicCloseness,
    props.output.reachedCounts,
    ...Object.values(props.output.status ?? {})
  ];
  for (const view of externalViews)
    if (view)
      validateRequirement(
        id,
        Boolean(view.buffer.defaultBuffer),
        'continuation inputs and outputs must use caller-owned imported buffers'
      );
  const externalPortNames = [
    'pivotNodes',
    'costLimit',
    'topology.offsets',
    'topology.neighbors',
    'topology.weights',
    'reverseTopology.offsets',
    'reverseTopology.neighbors',
    'reverseTopology.weights',
    'output.betweenness',
    'output.edgeBetweenness',
    'output.closeness',
    'output.harmonicCloseness',
    'output.reachedCounts',
    ...Object.keys(props.output.status ?? {}).map(name => `output.status.${name}`)
  ];
  const logicalCapacities = new Map<
    NonNullable<GraphDataView['buffer']['defaultBuffer']>,
    number
  >();
  for (const view of externalViews)
    if (view?.buffer.defaultBuffer) {
      const identity = view.buffer.defaultBuffer;
      logicalCapacities.set(
        identity,
        Math.max(logicalCapacities.get(identity) ?? 0, view.buffer.byteLength)
      );
    }
  const physicalImports = new Map<Buffer, NonNullable<GraphDataView['buffer']['defaultBuffer']>>();
  for (const [port, view] of externalViews.entries())
    if (view?.buffer.defaultBuffer) {
      const imported = view.buffer.defaultBuffer;
      const physical = getCoreDefaultBuffer(imported);
      const prior = physicalImports.get(physical);
      validateRequirement(
        id,
        !prior || prior === imported,
        `external port ${externalPortNames[port]} aliases a different imported wrapper; rebuild using the same buffer identity for all aliased ports`
      );
      physicalImports.set(physical, imported);
    }
  const adjacencyLayout = getCentralityAdjacencyLayout(
    nodeCount,
    props.topology.neighbors.length,
    props.reverseTopology!.neighbors.length
  );
  const lengths = {
    queues: 2 * expandedNodeCount,
    queuedRound: expandedNodeCount,
    control: phaseLayout.controlWordCount,
    dispatchEven: phaseLayout.dispatchSlotCount * FRONTIER_DISPATCH_SLOT_WORDS,
    dispatchOdd: phaseLayout.dispatchSlotCount * FRONTIER_DISPATCH_SLOT_WORDS,
    costs: expandedNodeCount,
    seeds: laneCount,
    adjacency: hasBetweenness ? adjacencyLayout.wordCount : 1,
    scratch: hasBetweenness ? 3 * expandedNodeCount : 1,
    exponents: hasBetweenness ? expandedNodeCount : 1,
    signal: 3,
    batchInfo: 1,
    partialNodes: props.output.betweenness ? nodeCount : 1,
    partialEdges: props.output.edgeBetweenness ? props.topology.neighbors.length : 1
  };
  for (const [name, count] of Object.entries(lengths))
    validateGPUStorageBindingByteLength(
      device,
      id,
      name,
      Math.max(count, 1) * 4,
      'reduce laneCount, edges or roundsPerStep'
    );
  // Each phase has a guard and round per slot plus a boundary snapshot. Count every fixed
  // initialization, carry, closeness, accumulation, merge and optional status node exactly.
  const phaseCount = hasBetweenness ? 3 : 1;
  const betweennessOutputCount =
    Number(Boolean(props.output.betweenness)) + Number(Boolean(props.output.edgeBetweenness));
  const hasCloseness = Boolean(
    props.output.closeness || props.output.harmonicCloseness || props.output.reachedCounts
  );
  const estimatedCommandNodeCount =
    phaseCount * (2 * roundsPerStep + 1) +
    3 +
    Number(hasBetweenness) +
    4 * Number(hasBetweenness) +
    2 +
    2 * Number(hasCloseness) +
    3 * betweennessOutputCount +
    Object.values(props.output.status ?? {}).filter(view => view !== undefined).length;
  validateCommandNodeBudget(id, estimatedCommandNodeCount);
  validateGPUStorageBindingByteLength(
    device,
    id,
    'parameters',
    5 * Uint32Array.BYTES_PER_ELEMENT,
    'use a device supporting five parameter words'
  );
  const resources: {destroy(): void}[] = [];
  const destroyResources = () => {
    for (const resource of resources.splice(0).reverse()) resource.destroy();
  };
  try {
    const buffers = {} as Record<keyof typeof lengths, Buffer>;
    for (const [name, count] of Object.entries(lengths)) {
      const buffer = device.createBuffer({
        id: `${id}-${name}`,
        byteLength: Math.max(count, 1) * 4,
        usage:
          Buffer.STORAGE |
          Buffer.COPY_SRC |
          Buffer.COPY_DST |
          (name === 'dispatchEven' || name === 'dispatchOdd' ? Buffer.INDIRECT : 0)
      });
      resources.push(buffer);
      buffers[name as keyof typeof lengths] = buffer;
    }
    // [window rounds, search complete, sigma complete, delta complete, reset totals]
    const parameters = new GPUParameterBuffer(device, {
      id: `${id}-parameters`,
      format: 'uint32',
      length: 5
    });
    resources.push(parameters);
    const graphs: Record<string, GPUCommandGraph> = {};
    const buildGraph = (
      name: string,
      build: (
        graph: GPUCommandGraph,
        views: ReturnType<typeof createViews>
      ) => readonly GPUCommandNode[]
    ) => {
      const graph = new GPUCommandGraph(device, {id: `${id}-${name}`});
      const views = createViews(graph);
      for (const node of build(graph, views)) graph.add(node);
      graphs[name] = graph;
    };
    function createViews(graph: GPUCommandGraph) {
      // Distinct caller handles may describe disjoint fields of one packed physical buffer.
      // Import that buffer once so luma.gl can track hazards across every rebound field.
      const imported = new Map<
        NonNullable<GraphDataView['buffer']['defaultBuffer']>,
        GraphDataView['buffer']
      >();
      const rebind = <Format extends 'uint32' | 'float32'>(
        view: GraphDataView<Format>
      ): GraphDataView<Format> => {
        const buffer = view.buffer.defaultBuffer!;
        let handle = imported.get(buffer);
        if (!handle) {
          handle = graph.importBuffer(
            {
              id: `${id}-external-${imported.size}`,
              byteLength: logicalCapacities.get(buffer)!,
              usage: getCoreDefaultBuffer(buffer).usage
            },
            buffer
          );
          imported.set(buffer, handle);
        }
        return graph.createDataView(handle, {
          format: view.format,
          length: view.length,
          byteOffset: view.byteOffset
        });
      };
      const word = (name: keyof typeof lengths) =>
        importGraphBuffer(graph, `${id}-${name}`, buffers[name], 'uint32', lengths[name]);
      const float = (name: keyof typeof lengths) =>
        importGraphBuffer(graph, `${id}-${name}`, buffers[name], 'float32', lengths[name]);
      const state: FrontierState = {
        nodeCount: expandedNodeCount,
        queues: word('queues'),
        queuedRound: word('queuedRound'),
        control: word('control'),
        dispatchEven: word('dispatchEven'),
        dispatchOdd: word('dispatchOdd')
      };
      const costs = float('costs');
      return {
        state,
        costs,
        parameters: parameters.importToGraph(graph),
        signal: word('signal'),
        seeds: word('seeds'),
        adjacency: word('adjacency'),
        scratch: word('scratch'),
        exponents: word('exponents'),
        batchInfo: word('batchInfo'),
        partialNodes: float('partialNodes'),
        partialEdges: float('partialEdges'),
        pivots: rebind(props.pivotNodes),
        costLimit: props.costLimit ? rebind(props.costLimit) : undefined,
        topology: {
          offsets: rebind(props.topology.offsets),
          neighbors: rebind(props.topology.neighbors),
          weights: rebind(props.topology.weights!)
        },
        reverseTopology: {
          offsets: rebind(props.reverseTopology!.offsets),
          neighbors: rebind(props.reverseTopology!.neighbors),
          weights: rebind(props.reverseTopology!.weights!)
        },
        output: {
          betweenness: props.output.betweenness ? rebind(props.output.betweenness) : undefined,
          edgeBetweenness: props.output.edgeBetweenness
            ? rebind(props.output.edgeBetweenness)
            : undefined,
          closeness: props.output.closeness ? rebind(props.output.closeness) : undefined,
          harmonicCloseness: props.output.harmonicCloseness
            ? rebind(props.output.harmonicCloseness)
            : undefined,
          reachedCounts: props.output.reachedCounts
            ? rebind(props.output.reachedCounts)
            : undefined,
          status: Object.fromEntries(
            Object.entries(props.output.status ?? {})
              .filter((entry): entry is [string, GraphDataView<'uint32'>] => entry[1] !== undefined)
              .map(([name, view]) => [name, rebind(view)])
          )
        }
      };
    }
    const phaseProps = (views: ReturnType<typeof createViews>) => ({
      layout: adjacencyLayout,
      expandedNodeCount,
      state: views.state,
      phase,
      adjacency: views.adjacency,
      costs: views.costs,
      scratch: views.scratch,
      exponents: views.exponents
    });
    const reset = (graph: GPUCommandGraph, views: ReturnType<typeof createViews>) =>
      createCentralityResetNode(graph, {
        id: `${id}-reset-frontier`,
        state: views.state,
        sigmaPhase: phase,
        deltaPhase: phase
      });
    buildGraph('initialize-search', (graph, views) => [
      ...(hasBetweenness
        ? [
            createCentralityPackNode(graph, {
              id: `${id}-pack`,
              layout: adjacencyLayout,
              forward: views.topology,
              reverse: views.reverseTopology,
              adjacency: views.adjacency
            })
          ]
        : []),
      createCentralityExpandSeedsNode(graph, {
        id: `${id}-seeds`,
        nodeCount,
        laneCount,
        batchCount: 1,
        seedStride: laneCount,
        pivotNodes: views.pivots,
        expandedSeeds: views.seeds
      }),
      createReachabilityInitializeNode(graph, {
        id: `${id}-initialize`,
        nodeCount: expandedNodeCount,
        state: views.state,
        phase,
        costs: views.costs,
        costLimit: views.costLimit
      }),
      createReachabilitySeedNode(graph, {
        id: `${id}-seed-frontier`,
        nodeCount: expandedNodeCount,
        state: views.state,
        phase,
        costs: views.costs,
        sources: views.seeds,
        hasCostLimit: Boolean(props.costLimit)
      })
    ]);
    if (hasBetweenness) {
      buildGraph('initialize-sigma', (graph, views) => [
        reset(graph, views),
        createCentralitySigmaInitializeNode(graph, {
          id: `${id}-sigma-initialize`,
          ...phaseProps(views)
        })
      ]);
      buildGraph('initialize-delta', (graph, views) => [
        reset(graph, views),
        createCentralityDeltaInitializeNode(graph, {
          id: `${id}-delta-initialize`,
          ...phaseProps(views)
        })
      ]);
    }
    for (const name of [
      'search',
      ...(hasBetweenness ? (['sigma', 'delta'] as const) : [])
    ] as CentralityContinuationPhase[]) {
      buildGraph(name, (graph, views) => {
        const nodes: GPUCommandNode[] = [];
        for (let round = 0; round < roundsPerStep; round++) {
          const dispatch = round % 2 === 0 ? views.state.dispatchEven : views.state.dispatchOdd;
          const slot = graph.createDataView(dispatch.buffer, {
            format: 'uint32',
            length: FRONTIER_DISPATCH_SLOT_WORDS,
            byteOffset: getFrontierDispatchSlotByteOffset(phase, round)
          });
          nodes.push(
            createWGSLKernelNode(graph, {
              id: `${id}-guard-${round}`,
              operation: 'NetworkCentralityContinuation',
              variant: 'guard',
              invocationCount: 1,
              bindings: [
                {name: 'dispatchSlot', view: slot, type: 'u32', access: 'read_write'},
                {name: 'parameters', view: views.parameters, type: 'u32', access: 'read'}
              ],
              body: 'if (dispatchSlot[dispatchSlotOffset + 3u] >= parameters[parametersOffset]) { dispatchSlot[dispatchSlotOffset] = 0u; }'
            })
          );
          nodes.push(
            name === 'search'
              ? createReachabilityRelaxNode(graph, {
                  id: `${id}-search-${round}`,
                  nodeCount: expandedNodeCount,
                  state: views.state,
                  phase,
                  round,
                  localIterations: validator.localIterations,
                  csr: views.topology,
                  hasCostLimit: Boolean(props.costLimit),
                  costs: views.costs,
                  laneNodeCount: nodeCount
                })
              : (name === 'sigma'
                  ? createCentralitySigmaRoundNode
                  : createCentralityDeltaRoundNode)(graph, {
                  id: `${id}-${name}-${round}`,
                  ...phaseProps(views),
                  round,
                  localIterations: 1
                })
          );
        }
        nodes.push(
          createWGSLKernelNode(graph, {
            id: `${id}-snapshot`,
            operation: 'NetworkCentralityContinuation',
            variant: 'snapshot',
            invocationCount: 1,
            bindings: [
              {name: 'control', view: views.state.control, type: 'u32', access: 'read'},
              {name: 'parameters', view: views.parameters, type: 'u32', access: 'read'},
              {name: 'signal', view: views.signal, type: 'u32', access: 'read_write'}
            ],
            body: `let boundary = parameters[parametersOffset];
  signal[signalOffset] = control[controlOffset + 4u + boundary];
  signal[signalOffset + 1u] = control[controlOffset + 2u];
  signal[signalOffset + 2u] = boundary & 1u;`
          })
        );
        return nodes;
      });
    }
    buildGraph('carry', (graph, views) => [
      reset(graph, views),
      createWGSLKernelNode(graph, {
        id: `${id}-carry`,
        operation: 'NetworkCentralityContinuation',
        variant: 'carry',
        invocationCount: expandedNodeCount,
        bindings: [
          {name: 'queues', view: views.state.queues, type: 'u32', access: 'read_write'},
          {name: 'control', view: views.state.control, type: 'u32', access: 'read_write'},
          {name: 'dispatchEven', view: views.state.dispatchEven, type: 'u32', access: 'read_write'},
          {name: 'signal', view: views.signal, type: 'u32', access: 'read'}
        ],
        declarations: `const NODE_COUNT: u32 = ${expandedNodeCount}u;`,
        body: `let count = signal[signalOffset];
  if (index < count && signal[signalOffset + 2u] != 0u) { queues[queuesOffset + index] = queues[queuesOffset + NODE_COUNT + index]; }
  if (index == 0u) {
    control[controlOffset + 4u] = count;
    control[controlOffset + 2u] = signal[signalOffset + 1u];
    dispatchEven[dispatchEvenOffset] = min((count + 255u) / 256u, ${device.limits.maxComputeWorkgroupsPerDimension}u);
  }`
      })
    ]);
    buildGraph('publish', (graph, views) => {
      const nodes: GPUCommandNode[] = [];
      if (props.output.closeness || props.output.harmonicCloseness || props.output.reachedCounts)
        nodes.push(
          createFillNode(graph, {
            id: `${id}-batch-info`,
            operation: 'NetworkCentralityContinuation',
            view: views.batchInfo,
            type: 'u32',
            value: '0u'
          }),
          createCentralityClosenessNode(graph, {
            id: `${id}-closeness`,
            nodeCount,
            laneCount,
            rowCount: laneCount,
            costs: views.costs,
            batchInfo: views.batchInfo,
            closeness: views.output.closeness,
            harmonicCloseness: views.output.harmonicCloseness,
            reachedCounts: views.output.reachedCounts
          })
        );
      for (const [name, destination, partial] of [
        ['nodes', views.output.betweenness, views.partialNodes],
        ['edges', views.output.edgeBetweenness, views.partialEdges]
      ] as const)
        if (destination) {
          nodes.push(
            createFillNode(graph, {
              id: `${id}-zero-${name}`,
              operation: 'NetworkCentralityContinuation',
              view: partial,
              type: 'f32',
              value: '0.0'
            })
          );
          nodes.push(
            name === 'nodes'
              ? createCentralityAccumulateNode(graph, {
                  id: `${id}-accumulate-nodes`,
                  nodeCount,
                  laneCount,
                  expandedNodeCount,
                  betweennessScale: validator.betweennessScale,
                  scratch: views.scratch,
                  seeds: views.seeds,
                  betweenness: partial
                })
              : createCentralityEdgeAccumulateNode(graph, {
                  id: `${id}-accumulate-edges`,
                  layout: adjacencyLayout,
                  expandedNodeCount,
                  laneCount,
                  betweennessScale: validator.betweennessScale,
                  adjacency: views.adjacency,
                  scratch: views.scratch,
                  exponents: views.exponents,
                  costs: views.costs,
                  edgeBetweenness: partial
                })
          );
          nodes.push(
            createWGSLKernelNode(graph, {
              id: `${id}-merge-${name}`,
              operation: 'NetworkCentralityContinuation',
              variant: 'sum',
              invocationCount: destination.length,
              bindings: [
                {name: 'partialValues', view: partial, type: 'f32', access: 'read'},
                {name: 'totals', view: destination, type: 'f32', access: 'read_write'},
                {name: 'parameters', view: views.parameters, type: 'u32', access: 'read'}
              ],
              body: `var previous = 0.0;
  if (parameters[parametersOffset + 4u] == 0u) { previous = totals[totalsOffset + index]; }
  totals[totalsOffset + index] = previous + partialValues[partialValuesOffset + index];`
            })
          );
        }
      for (const [name, destination] of Object.entries(views.output.status)) {
        const value =
          name === 'approximation'
            ? props.sampling === 'sampled'
              ? '1u'
              : '0u'
            : name === 'zeroWeightCycle'
              ? 'cycle'
              : name === 'converged'
                ? 'complete * (1u - cycle)'
                : '1u - complete';
        const operation = name === 'converged' ? 'min' : 'max';
        nodes.push(
          createWGSLKernelNode(graph, {
            id: `${id}-${name}`,
            operation: 'NetworkCentralityContinuation',
            variant: `status-${name}`,
            invocationCount: 1,
            bindings: [
              {name: 'control', view: views.state.control, type: 'u32', access: 'read'},
              {name: 'parameters', view: views.parameters, type: 'u32', access: 'read'},
              {name: 'result', view: destination, type: 'u32', access: 'read_write'}
            ],
            body: `let complete = parameters[parametersOffset + 1u] * parameters[parametersOffset + 2u] * parameters[parametersOffset + 3u];
  let cycle = ${hasBetweenness ? 'select(0u, 1u, control[controlOffset + 2u] != 0u && parameters[parametersOffset + 2u] != 0u)' : '0u'};
  var previous = ${name === 'converged' ? '1u' : '0u'};
  if (parameters[parametersOffset + 4u] == 0u) { previous = result[resultOffset]; }
  result[resultOffset] = ${operation}(previous, ${value});`
          })
        );
      }
      return nodes;
    });
    return {
      graphs,
      parameters,
      signal: buffers.signal,
      maximumIterations,
      roundsPerStep,
      hasBetweenness,
      estimatedCommandNodeCount,
      destroy: destroyResources
    };
  } catch (error) {
    destroyResources();
    throw error;
  }
}

/**
 * Creates a host-driven runner for one fixed-capacity pivot batch. All views must be backed by
 * caller-owned imported buffers. Compile once, update pivot contents between completed runBatch
 * calls, and combine with planNetworkCentralitySchedule for arbitrarily many batches. Pad unused
 * lanes with 0xffffffff. Outputs and CSR slots retain GPUNetworkCentrality's layouts/tolerances.
 *
 * maximumIterations bounds each phase per pivot batch across ALL windows (1..4294967294,
 * default nodeCount). roundsPerStep bounds rounds per submitted window (1..1023, default 64),
 * independently of that total. Distances, sigma/exponents, counters, dependencies and frontier queues persist
 * on the GPU. Each window reads only one uint32 boundary-queue count. Completed phases stop;
 * exhausted phases publish incumbent values with iterationLimitReached=1. Zero cycles remain a
 * separate failure. reset=true or 1 starts a new run (the default); reset=false or 0 adds totals
 * and merges status: iterationLimitReached stays set through later successful batches until
 * reset. A planner descriptor can be passed directly to runBatch. Per-pivot
 * outputs are overwritten by each batch. Changing topology/pivots mid-batch is unsupported.
 * The runner explicitly submits to the device queue; it is a host scheduler, not a contributor.
 */
export function createNetworkCentralityContinuation(
  graph: GPUCommandGraph,
  props: NetworkCentralityContinuationProps
) {
  const schedule = createCentralityContinuationGraphs(graph, props);
  const device = graph.device;
  const compiled: Record<string, ReturnType<GPUCommandGraph['compile']>> = {};
  try {
    for (const [name, commandGraph] of Object.entries(schedule.graphs))
      compiled[name] = commandGraph.compile();
  } catch (error) {
    for (const commandGraph of Object.values(compiled)) commandGraph.destroy();
    schedule.destroy();
    throw error;
  }
  let busy = false;
  let destroyed = false;
  const submit = (name: string) => {
    const encoder = device.createCommandEncoder();
    compiled[name].encode(encoder, {parameters: undefined});
    device.submit(encoder.finish());
  };
  return {
    estimatedCommandNodeCount: schedule.estimatedCommandNodeCount,
    async runBatch(options: {reset?: boolean | 0 | 1} = {}) {
      validateRequirement(
        props.id ?? 'network-centrality-continuation',
        !busy && !destroyed,
        'runner must be idle and not destroyed'
      );
      const reset = Number(Boolean(options.reset ?? true));
      busy = true;
      try {
        const completed = await runCentralityContinuationPhases({
          ...schedule,
          initialize: phase => submit(`initialize-${phase}`),
          execute: async (phase, roundCount, carry) => {
            schedule.parameters.write(Uint32Array.of(roundCount), 0);
            if (carry) submit('carry');
            submit(phase);
            const bytes = await schedule.signal.readAsync(0, 4);
            return new Uint32Array(bytes.buffer, bytes.byteOffset, 1)[0];
          }
        });
        schedule.parameters.write(Uint32Array.of(...completed.map(Number), reset), 1);
        submit('publish');
      } finally {
        busy = false;
      }
    },
    destroy() {
      validateRequirement(
        props.id ?? 'network-centrality-continuation',
        !busy,
        'runner must be idle before destroy'
      );
      if (destroyed) return;
      destroyed = true;
      for (const graph of Object.values(compiled)) graph.destroy();
      schedule.destroy();
    }
  };
}
