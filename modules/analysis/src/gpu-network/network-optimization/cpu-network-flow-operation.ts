// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {validateCapacity} from '../../analysis-core/kernels/gpu-contributor-utils';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {getCPUOperationUint32Rows} from '../../analysis-core/operations/cpu-operation-rows';
import {
  getNetworkOptimizationCPUStatus,
  type NetworkOptimizationCPUStatus
} from './network-optimization-status';
import {
  GPU_NETWORK_FLOW_WORD,
  validateNetworkFlowOptions,
  type GPUNetworkFlowProps
} from './gpu-network-flow';
import {NETWORK_FLOW_MAXIMUM_TOTAL_CAPACITY as MAXIMUM_TOTAL_CAPACITY} from './network-flow-limits';
import {
  GPUNetworkFlowOperation,
  snapshotNetworkFlowOperationProps,
  type NetworkFlowOperationProps
} from './gpu-network-flow-operation';
import {
  computeNetworkFlowPaths,
  validateNetworkFlowPathShape,
  type CPUNetworkFlowPaths
} from './network-flow-paths';
import {computeNetworkFlowPhasesOnCPU} from './cpu-network-flow-phases';

const MAXIMIZE_FLOW = 0xffffffff;

export type CPUNetworkFlowInput = {
  edges: Uint32Array;
  capacities: Uint32Array;
  costs?: Uint32Array;
  parameters: Uint32Array;
  /** Optional CSR slots aligned with edges; inconsistent connectivity rejects the problem. */
  topology?: {offsets: Uint32Array; neighbors: Uint32Array};
};
export type CPUNetworkFlowOptions = Pick<
  GPUNetworkFlowProps,
  | 'nodeCount'
  | 'solver'
  | 'mode'
  | 'maximumAugmentations'
  | 'maximumRelabels'
  | 'maximumCostPhases'
  | 'maximumRelaxationRounds'
> & {
  /** Optional fixed-width path output capacity (1..1024); enables bounded decomposition. */
  pathCapacity?: number;
};
export type CPUNetworkFlowResult = {
  flow: Uint32Array;
  cut: Uint32Array;
  summary: Uint32Array;
  exactCost: Uint32Array;
  paths?: CPUNetworkFlowPaths;
  status: NetworkOptimizationCPUStatus;
};

/**
 * Integral flow CPU twin. Reference and successive-shortest-path solves scan residual arcs in
 * input order (forward then reverse per edge), with strict-improvement predecessor updates.
 * Parallel phase solves use bounded preflow discharges and excess returns. Parallel, antiparallel
 * and self-loop edges retain their slots. complete=1 means the solve finished, including an
 * infeasible finite request (unserved > 0). cutCertified is set only on residual exhaustion.
 * Invalid endpoints/source/sink, capacity totals or supplied CSR connectivity return zero outputs
 * and valid=0. CSR slots must match the edge-list source row and destination at every slot.
 * Optional pathCapacity enables final-flow decomposition: FIFO BFS, ascending outgoing edge rows,
 * first predecessor wins, one bottleneck volume per path. Paths retain expanded super arcs.
 *
 * What differs from the GPU contributor:
 * - Solver selection matches the contributor. Reference and costed successive-shortest-path use
 *   maximumAugmentations; uncosted parallel uses maximumRelabels; primal-dual uses maximumCostPhases.
 *   Parallel phases use sink-rooted labels, bounded preflow discharges and excess returns, then a
 *   separate source-rooted cut search. CPU discharges scan input edge rows; GPU atomic ordering may
 *   choose another equal-cost branch. Explicit maximumRelaxationRounds bounds synchronous searches,
 *   including the final unchanged sweep. Omitted CPU budgets are uncapped. GPU private-array caps
 *   are dropped for the solve; optional decomposition keeps the 128-node/1024-edge layout limits.
 *   Decomposition and padding agree exactly for the same final edge flow.
 * - Total capacity may reach 0xffffffff (the uint32 output limit), rather than 0x00ffffff;
 *   original costs may use the entire uint32 range rather than 0..32767.
 * - Integral distances and the objective use exact integers (BigInt), avoiding signed-distance
 *   overflow. The summary cost is rounded once directly to float32; exactCost is [low32, high32].
 *   The GPU accumulates a float32 summary cost, compared with relative tolerance 2e-5.
 *   Aggregate value and objective agree when both complete within their numerical limits; parallel
 *   path allocation, phase progress and diagnostics can differ with GPU scheduling.
 * - Both operation forms and the pure function support zero edges and empty edge-flow outputs.
 *
 * @throws If edges/capacities/costs/parameters lengths are inconsistent or nodeCount is invalid.
 */
export function computeNetworkFlowOnCPU(
  input: CPUNetworkFlowInput,
  options: CPUNetworkFlowOptions
): CPUNetworkFlowResult {
  options = snapshotConstructionFields(options, [
    'nodeCount',
    'solver',
    'mode',
    'maximumAugmentations',
    'maximumRelabels',
    'maximumCostPhases',
    'maximumRelaxationRounds',
    'pathCapacity'
  ]);
  const {edges, capacities, costs, parameters, topology} = input;
  const {nodeCount} = options;
  const solver = validateNetworkFlowOptions(
    'computeNetworkFlowOnCPU',
    {...options, topology},
    capacities.length,
    Boolean(costs)
  );
  const parallelPhases =
    solver === 'parallel' &&
    (!costs ||
      options.mode === 'primal-dual' ||
      (options.mode === undefined && options.maximumCostPhases !== undefined));
  const maximumAugmentations = parallelPhases
    ? Infinity
    : (options.maximumAugmentations ?? Infinity);
  const maximumPhases = parallelPhases
    ? ((!costs ? options.maximumRelabels : options.maximumCostPhases) ?? Infinity)
    : Infinity;
  const maximumRelaxationRounds =
    solver === 'parallel' ? (options.maximumRelaxationRounds ?? nodeCount) : nodeCount;
  for (const name of [
    'maximumAugmentations',
    'maximumRelabels',
    'maximumCostPhases',
    'maximumRelaxationRounds'
  ] as const)
    if (options[name] !== undefined)
      validateCapacity(
        'computeNetworkFlowOnCPU',
        name,
        options[name],
        name === 'maximumRelaxationRounds' ? 1 : 0
      );
  validateCapacity('computeNetworkFlowOnCPU', 'nodeCount', nodeCount, 2);
  if (edges.length !== capacities.length * 2)
    throw new Error('edges length must equal capacities.length * 2');
  if (costs && costs.length !== capacities.length)
    throw new Error('costs length must equal capacities.length');
  if (parameters.length !== 3)
    throw new Error('parameters must contain source, sink and required flow');
  const edgeCount = capacities.length;
  if (options.pathCapacity !== undefined)
    validateNetworkFlowPathShape(
      'computeNetworkFlowOnCPU',
      nodeCount,
      edgeCount,
      options.pathCapacity
    );
  if (
    topology &&
    (topology.offsets.length !== nodeCount + 1 || topology.neighbors.length !== edgeCount)
  )
    throw new Error(
      'topology offsets must hold nodeCount + 1 rows and neighbors must hold edgeCount rows'
    );
  const flow = new Uint32Array(edgeCount);
  const cut = new Uint32Array(nodeCount);
  const summary = new Uint32Array(8);
  const exactCost = new Uint32Array(2);
  const result = {flow, cut, summary, exactCost};
  const finishResult = (): CPUNetworkFlowResult => ({
    ...result,
    ...(options.pathCapacity === undefined
      ? {}
      : {
          paths: computeNetworkFlowPaths(
            nodeCount,
            edges,
            flow,
            parameters,
            summary,
            options.pathCapacity
          )
        }),
    status: getNetworkOptimizationCPUStatus(summary, summary[GPU_NETWORK_FLOW_WORD.augmentations])
  });
  // The contributor exposes summary offsets, but no parameter packer/layout.
  const [source, destination, required] = parameters;
  if (source >= nodeCount || destination >= nodeCount || source === destination)
    return finishResult();
  if (topology) {
    const {offsets} = topology;
    if (offsets[0] !== 0 || offsets[nodeCount] !== edgeCount) return finishResult();
    for (let vertex = 0; vertex < nodeCount; vertex++) {
      if (offsets[vertex] > offsets[vertex + 1] || offsets[vertex + 1] > edgeCount)
        return finishResult();
    }
  }
  let totalCapacity = 0;
  for (let edge = 0; edge < edgeCount; edge++) {
    if (
      edges[edge * 2] >= nodeCount ||
      edges[edge * 2 + 1] >= nodeCount ||
      capacities[edge] > MAXIMUM_TOTAL_CAPACITY - totalCapacity
    )
      return finishResult();
    if (topology) {
      const from = edges[edge * 2];
      if (
        edge < topology.offsets[from] ||
        edge >= topology.offsets[from + 1] ||
        topology.neighbors[edge] !== edges[edge * 2 + 1]
      )
        return finishResult();
    }
    totalCapacity += capacities[edge];
  }
  summary[GPU_NETWORK_FLOW_WORD.valid] = 1;
  let value = 0;
  let augmentations = 0;
  let phaseCount = 0;
  let phaseComplete = false;
  let distances = new Array<bigint | undefined>(nodeCount);
  let exhausted = false;
  let searchConverged = false;
  if (parallelPhases) {
    const phaseResult = computeNetworkFlowPhasesOnCPU({
      nodeCount,
      edges,
      capacities,
      costs,
      source,
      sink: destination,
      required,
      maximumPhases,
      maximumRounds: maximumRelaxationRounds,
      flow,
      cut
    });
    value = phaseResult.value;
    phaseCount = phaseResult.phaseCount;
    phaseComplete = phaseResult.complete;
    exhausted = phaseResult.cutSearchConverged && !cut[destination];
  }
  for (; !parallelPhases; ) {
    distances = new Array<bigint | undefined>(nodeCount);
    const predecessors = new Float64Array(nodeCount).fill(-1);
    const reversed = new Uint8Array(nodeCount);
    const hopCounts = new Float64Array(nodeCount).fill(Infinity);
    distances[source] = 0n;
    hopCounts[source] = 0;
    searchConverged = false;
    for (let round = 0; round < Math.min(nodeCount, maximumRelaxationRounds); round++) {
      let changed = false;
      const updates = new Map<
        number,
        {distance: bigint; hops: number; edge: number; reversed: number}
      >();
      const improveLabel = (
        vertex: number,
        distance: bigint,
        hopCount: number,
        edge: number,
        reverse: number
      ) => {
        const currentDistance = updates.get(vertex)?.distance ?? distances[vertex];
        const currentHops = updates.get(vertex)?.hops ?? hopCounts[vertex];
        if (
          currentDistance !== undefined &&
          (distance > currentDistance ||
            (distance === currentDistance && (!parallelPhases || hopCount >= currentHops)))
        )
          return;
        changed = true;
        if (solver === 'parallel')
          updates.set(vertex, {distance, hops: hopCount, edge, reversed: reverse});
        else {
          distances[vertex] = distance;
          hopCounts[vertex] = hopCount;
          predecessors[vertex] = edge;
          reversed[vertex] = reverse;
        }
      };
      for (let edge = 0; edge < edgeCount; edge++) {
        const from = edges[edge * 2];
        const to = edges[edge * 2 + 1];
        const cost = parallelPhases && !costs ? 1n : BigInt(costs?.[edge] ?? 0);
        const reverseCost = parallelPhases && !costs ? 1n : -cost;
        const fromDistance = distances[from];
        if (flow[edge] < capacities[edge] && fromDistance !== undefined) {
          improveLabel(to, fromDistance + cost, hopCounts[from] + 1, edge, 0);
        }
        // Load after the forward relaxation, as in the GPU's in-place Bellman-Ford scan.
        const toDistance = distances[to];
        if (flow[edge] > 0 && toDistance !== undefined) {
          improveLabel(from, toDistance + reverseCost, hopCounts[to] + 1, edge, 1);
        }
      }
      for (const [vertex, update] of updates) {
        distances[vertex] = update.distance;
        hopCounts[vertex] = update.hops;
        predecessors[vertex] = update.edge;
        reversed[vertex] = update.reversed;
      }
      if (!changed) {
        searchConverged = true;
        break;
      }
    }
    exhausted = searchConverged && distances[destination] === undefined;
    if (exhausted || value === required) break;
    if (!searchConverged) break;
    if (augmentations >= maximumAugmentations) break;
    let increment = required - value;
    let cursor = destination;
    let hops = 0;
    while (cursor !== source && hops < nodeCount) {
      const edge = predecessors[cursor];
      if (edge < 0) {
        summary[GPU_NETWORK_FLOW_WORD.valid] = 0;
        return finishResult();
      }
      increment = Math.min(
        increment,
        reversed[cursor] ? flow[edge] : capacities[edge] - flow[edge]
      );
      cursor = edges[edge * 2 + (reversed[cursor] ? 1 : 0)];
      hops++;
    }
    if (cursor !== source || increment === 0) {
      summary[GPU_NETWORK_FLOW_WORD.valid] = 0;
      return finishResult();
    }
    cursor = destination;
    while (cursor !== source) {
      const edge = predecessors[cursor];
      if (reversed[cursor]) flow[edge] -= increment;
      else flow[edge] += increment;
      cursor = edges[edge * 2 + (reversed[cursor] ? 1 : 0)];
    }
    value += increment;
    augmentations++;
  }
  if (!parallelPhases)
    for (let vertex = 0; vertex < nodeCount; vertex++)
      cut[vertex] = Number(distances[vertex] !== undefined);
  let objective = 0n;
  let cutCapacity = 0;
  for (let edge = 0; edge < edgeCount; edge++) {
    objective += BigInt(flow[edge]) * BigInt(costs?.[edge] ?? 0);
    if (cut[edges[edge * 2]] && !cut[edges[edge * 2 + 1]]) cutCapacity += capacities[edge];
  }
  exactCost[0] = Number(objective & 0xffffffffn);
  exactCost[1] = Number(objective >> 32n);
  summary[GPU_NETWORK_FLOW_WORD.complete] = Number(
    parallelPhases ? phaseComplete : exhausted || value === required
  );
  summary[GPU_NETWORK_FLOW_WORD.value] = value;
  summary[GPU_NETWORK_FLOW_WORD.unserved] = required === MAXIMIZE_FLOW ? 0 : required - value;
  summary[GPU_NETWORK_FLOW_WORD.augmentations] = parallelPhases ? phaseCount : augmentations;
  summary[GPU_NETWORK_FLOW_WORD.cost] = getObjectiveBits(objective);
  summary[GPU_NETWORK_FLOW_WORD.cutCapacity] = cutCapacity;
  summary[GPU_NETWORK_FLOW_WORD.cutCertified] = Number(exhausted && cutCapacity === value);
  return finishResult();
}

/** Direct round-to-nearest, ties-to-even, without a BigInt -> float64 -> float32 double rounding. */
function getObjectiveBits(objective: bigint): number {
  if (objective === 0n) return 0;
  let exponent = objective.toString(2).length - 1;
  const shift = exponent - 23;
  let significand: bigint;
  if (shift <= 0) significand = objective << BigInt(-shift);
  else {
    significand = objective >> BigInt(shift);
    const remainder = objective - (significand << BigInt(shift));
    const half = 1n << BigInt(shift - 1);
    if (remainder > half || (remainder === half && (significand & 1n) !== 0n)) significand++;
    if (significand === 0x1000000n) {
      significand >>= 1n;
      exponent++;
    }
  }
  return (((exponent + 127) << 23) | Number(significand & 0x7fffffn)) >>> 0;
}

/** Exact CPU flow twin with explicit augmentation budgets and no GPU signed-cost cap. */
export class CPUNetworkFlowOperation extends GPUCPUOperation {
  constructor(props: NetworkFlowOperationProps) {
    props = snapshotNetworkFlowOperationProps(props);
    const id = props.id ?? 'network-flow';
    validateCapacity(id, 'nodeCount', props.nodeCount, 2);
    validateCapacity(id, 'edgeCount', props.edgeCount, 0);
    if (props.maximumAugmentations !== undefined)
      validateCapacity(id, 'maximumAugmentations', props.maximumAugmentations, 0);
    for (const name of ['maximumCostPhases', 'maximumRelabels', 'maximumRelaxationRounds'] as const)
      if (props[name] !== undefined)
        validateCapacity(id, name, props[name], name === 'maximumRelaxationRounds' ? 1 : 0);
    const operation = new GPUNetworkFlowOperation(props);
    super({
      id,
      inputs: operation.inputs,
      outputs: Object.fromEntries(
        Object.values(operation.outputs).map(output => [
          output.value,
          gpuGraph.vector(output.format, {length: output.length})
        ])
      ),
      execute({inputs}) {
        const capacities = getCPUOperationUint32Rows(id, inputs, 'capacities');
        if (capacities.length !== props.edgeCount)
          throw new Error(`${id} capacities length must equal edgeCount`);
        const result = computeNetworkFlowOnCPU(
          {
            edges: getCPUOperationUint32Rows(id, inputs, 'edges', 'uint32x2'),
            capacities,
            costs:
              props.costs === undefined
                ? undefined
                : getCPUOperationUint32Rows(id, inputs, 'costs'),
            parameters: getCPUOperationUint32Rows(id, inputs, 'parameters'),
            topology: props.topology
              ? {
                  offsets: getCPUOperationUint32Rows(id, inputs, 'topologyOffsets'),
                  neighbors: getCPUOperationUint32Rows(id, inputs, 'topologyNeighbors')
                }
              : undefined
          },
          props
        );
        const {status, paths, ...primaryRows} = result;
        const outputRows: Record<string, Uint32Array> = {
          ...primaryRows,
          ...status,
          ...(paths
            ? {
                pathEdgeRows: paths.edgeRows,
                pathLengths: paths.lengths,
                pathVolumes: paths.volumes,
                pathCount: paths.status.count,
                pathRequiredCount: paths.status.requiredCount,
                pathOverflow: paths.status.overflow,
                pathInvalidCount: paths.status.invalidCount,
                pathConverged: paths.status.converged,
                pathIterationLimitReached: paths.status.iterationLimitReached
              }
            : {})
        };
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            outputRows[port as keyof typeof outputRows]
          ])
        );
      }
    });
  }
}
