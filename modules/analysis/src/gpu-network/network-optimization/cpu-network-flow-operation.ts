// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {validateCapacity} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getCPUOperationUint32Rows} from '../../analysis-core/operations/cpu-operation-rows';
import {
  getNetworkOptimizationCPUStatus,
  type NetworkOptimizationCPUStatus
} from './network-optimization-status';
import {GPU_NETWORK_FLOW_WORD} from './gpu-network-flow';
import {
  GPUNetworkFlowOperation,
  type NetworkFlowOperationProps
} from './gpu-network-flow-operation';

/** Output values and cut capacities occupy one uint32 word. */
const MAXIMUM_TOTAL_CAPACITY = 0xffffffff;
const MAXIMIZE_FLOW = 0xffffffff;

export type CPUNetworkFlowInput = {
  edges: Uint32Array;
  capacities: Uint32Array;
  costs?: Uint32Array;
  parameters: Uint32Array;
};
export type CPUNetworkFlowOptions = {nodeCount: number};
export type CPUNetworkFlowResult = {
  flow: Uint32Array;
  cut: Uint32Array;
  summary: Uint32Array;
  exactCost: Uint32Array;
  status: NetworkOptimizationCPUStatus;
};

/**
 * Integral successive shortest augmenting paths, scanning residual arcs in input order (forward
 * then reverse per edge), with strict-improvement predecessor updates. Parallel, antiparallel
 * and self-loop edges retain their slots. complete=1 means the solve finished, including an
 * infeasible finite request (unserved > 0). cutCertified is set only on residual exhaustion.
 * Invalid endpoints/source/sink or capacity totals return zero outputs and valid=0.
 *
 * What differs from the GPU contributor:
 * - The CPU always finishes the solve; maximumAugmentations is a GPU budget and is ignored.
 *   Node/edge/augmentation private-array caps are dropped, and complete is always 1 for valid data.
 * - Total capacity may reach 0xffffffff (the uint32 output limit), rather than 0x00ffffff;
 *   original costs may use the entire uint32 range rather than 0..32767.
 * - Integral distances and the objective use exact integers (BigInt), avoiding signed-distance
 *   overflow. The summary cost is rounded once directly to float32; exactCost is [low32, high32].
 *   The GPU accumulates a float32 summary cost, compared with relative tolerance 2e-5; all other
 *   outputs compare exactly when the GPU budget suffices.
 * - Graph operation outputs require at least one edge slot because GPUContributorOperation
 *   requires nonempty output vectors. The pure function also supports zero edges.
 *
 * @throws If edges/capacities/costs/parameters lengths are inconsistent or nodeCount is invalid.
 */
export function computeNetworkFlowOnCPU(
  input: CPUNetworkFlowInput,
  options: CPUNetworkFlowOptions
): CPUNetworkFlowResult {
  const {edges, capacities, costs, parameters} = input;
  const {nodeCount} = options;
  validateCapacity('computeNetworkFlowOnCPU', 'nodeCount', nodeCount, 2);
  if (edges.length !== capacities.length * 2)
    throw new Error('edges length must equal capacities.length * 2');
  if (costs && costs.length !== capacities.length)
    throw new Error('costs length must equal capacities.length');
  if (parameters.length !== 3)
    throw new Error('parameters must contain source, sink and required flow');
  const edgeCount = capacities.length;
  const flow = new Uint32Array(edgeCount);
  const cut = new Uint32Array(nodeCount);
  const summary = new Uint32Array(8);
  const exactCost = new Uint32Array(2);
  const result = {flow, cut, summary, exactCost};
  const finishResult = (): CPUNetworkFlowResult => ({
    ...result,
    status: getNetworkOptimizationCPUStatus(summary, summary[GPU_NETWORK_FLOW_WORD.augmentations])
  });
  // The contributor exposes summary offsets, but no parameter packer/layout.
  const [source, destination, required] = parameters;
  if (source >= nodeCount || destination >= nodeCount || source === destination)
    return finishResult();
  let totalCapacity = 0;
  for (let edge = 0; edge < edgeCount; edge++) {
    if (
      edges[edge * 2] >= nodeCount ||
      edges[edge * 2 + 1] >= nodeCount ||
      capacities[edge] > MAXIMUM_TOTAL_CAPACITY - totalCapacity
    )
      return finishResult();
    totalCapacity += capacities[edge];
  }
  summary[GPU_NETWORK_FLOW_WORD.valid] = 1;
  let value = 0;
  let augmentations = 0;
  let distances: (bigint | undefined)[];
  let exhausted = false;
  for (;;) {
    distances = new Array<bigint | undefined>(nodeCount);
    const predecessors = new Float64Array(nodeCount).fill(-1);
    const reversed = new Uint8Array(nodeCount);
    distances[source] = 0n;
    for (let round = 1; round < nodeCount; round++) {
      let changed = false;
      for (let edge = 0; edge < edgeCount; edge++) {
        const from = edges[edge * 2];
        const to = edges[edge * 2 + 1];
        const cost = BigInt(costs?.[edge] ?? 0);
        const fromDistance = distances[from];
        if (
          flow[edge] < capacities[edge] &&
          fromDistance !== undefined &&
          (distances[to] === undefined || fromDistance + cost < distances[to]!)
        ) {
          distances[to] = fromDistance + cost;
          predecessors[to] = edge;
          reversed[to] = 0;
          changed = true;
        }
        // Load after the forward relaxation, as in the GPU's in-place Bellman-Ford scan.
        const toDistance = distances[to];
        if (
          flow[edge] > 0 &&
          toDistance !== undefined &&
          (distances[from] === undefined || toDistance - cost < distances[from]!)
        ) {
          distances[from] = toDistance - cost;
          predecessors[from] = edge;
          reversed[from] = 1;
          changed = true;
        }
      }
      if (!changed) break;
    }
    exhausted = distances[destination] === undefined;
    if (exhausted || value === required) break;
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
  summary[GPU_NETWORK_FLOW_WORD.complete] = 1;
  summary[GPU_NETWORK_FLOW_WORD.value] = value;
  summary[GPU_NETWORK_FLOW_WORD.unserved] = required === MAXIMIZE_FLOW ? 0 : required - value;
  summary[GPU_NETWORK_FLOW_WORD.augmentations] = augmentations;
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

/** Exact CPU flow twin without the GPU's signed-cost or augmentation limits. */
export class CPUNetworkFlowOperation extends GPUCPUOperation {
  constructor(props: NetworkFlowOperationProps) {
    const id = props.id ?? 'network-flow';
    validateCapacity(id, 'nodeCount', props.nodeCount, 2);
    validateCapacity(id, 'edgeCount', props.edgeCount);
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
            parameters: getCPUOperationUint32Rows(id, inputs, 'parameters')
          },
          props
        );
        const {status, ...primaryRows} = result;
        const outputRows = {...primaryRows, ...status};
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
