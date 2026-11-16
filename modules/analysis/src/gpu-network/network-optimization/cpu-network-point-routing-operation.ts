// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {validateCapacity} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {
  GPUNetworkPointRoutingOperation,
  type NetworkPointRoutingOperationProps
} from './gpu-network-point-routing-operation';

/** Packed directed-edge turns, queries and fractional endpoints. */
export type CPUNetworkPointRoutingInput = {
  turns: Uint32Array;
  costs: Float32Array;
  queries: Uint32Array;
  fractions: Float32Array;
};
/** Per-query costs/status and row-major query by edge parents, including same-edge return parents. */
export type CPUNetworkPointRoutingResult = {
  costs: Float32Array;
  status: Uint32Array;
  predecessorEdges: Uint32Array;
};

/**
 * Routes between positions on directed edges. Turn order resolves equal-cost discoveries, with
 * strict improvements only, as in the contributor's explicit `mode: 'reference'` route. Same-edge forward travel competes
 * with legal returns, including when the target fraction precedes the source fraction.
 *
 * What differs from the GPU contributor:
 * - The default GPU mode is scalable CSR: full-edge predecessors use reachability's smallest-parent,
 *   acyclic plateau rule and terminal partial-edge ties use the smallest cycle-safe parent.
 *   The CPU retains turn-order ties for either mode; pass `mode: 'reference'` to the GPU operation
 *   when comparing those ties. The CPU accepts mode for interchangeability without changing its route.
 * - Float64 turn relaxation and partial-edge arithmetic precede one float32 output rounding;
 *   near ties, subnormals and paths overflowing float32 may produce different costs or parents.
 * - A full-edge sum overflowing float32 does not trigger the GPU's status-1 early return. The
 *   CPU continues in float64 and can publish a finite partial-target cost with status 0 and
 *   predecessors where the GPU leaves Infinity and status 1. Scalable mode clears invalid-query
 *   predecessors; reference mode can retain partially written predecessors.
 * - Relaxation continues to a fixpoint. GPU edge/turn/private-array and maximumRelaxations caps
 *   are dropped; status 2 (iteration exhaustion) is never set. Invalid data still sets status 1.
 * - Invalid edge IDs, fractions, costs and turn endpoints retain the GPU's per-query status result;
 *   inconsistent packed array lengths throw instead of being interpreted as partial rows.
 *
 * @throws Error naming turns, costs, queries or fractions for an inconsistent packed array.
 */
export function computeNetworkPointRoutingOnCPU(
  input: CPUNetworkPointRoutingInput
): CPUNetworkPointRoutingResult {
  if (!input.costs.length) throw new Error('costs must contain at least one edge');
  if (input.turns.length % 2) throw new Error('turns must contain uint32x2 rows');
  if (input.queries.length % 2) throw new Error('queries must contain uint32x2 rows');
  if (input.fractions.length !== input.queries.length)
    throw new Error('fractions must contain one float32x2 row per query');
  const edgeCount = input.costs.length;
  const queryCount = input.queries.length / 2;
  const costs = new Float32Array(queryCount).fill(Infinity);
  const status = new Uint32Array(queryCount).fill(1);
  const predecessorEdges = new Uint32Array(queryCount * edgeCount).fill(0xffffffff);
  const invalidNetwork =
    input.costs.some(cost => !Number.isFinite(cost) || cost < 0) ||
    input.turns.some(edge => edge >= edgeCount);
  for (let query = 0; query < queryCount; query++) {
    const source = input.queries[query * 2];
    const destination = input.queries[query * 2 + 1];
    const sourceFraction = input.fractions[query * 2];
    const targetFraction = input.fractions[query * 2 + 1];
    if (
      invalidNetwork ||
      source >= edgeCount ||
      destination >= edgeCount ||
      !Number.isFinite(sourceFraction) ||
      !Number.isFinite(targetFraction) ||
      sourceFraction < 0 ||
      sourceFraction > 1 ||
      targetFraction < 0 ||
      targetFraction > 1
    )
      continue;
    let bestCost =
      source === destination && targetFraction >= sourceFraction
        ? (targetFraction - sourceFraction) * input.costs[source]
        : Infinity;
    let targetPredecessor = 0xffffffff;
    const distances = new Float64Array(edgeCount).fill(Infinity);
    distances[source] = (1 - sourceFraction) * input.costs[source];
    let changed: boolean;
    do {
      changed = false;
      for (let turn = 0; turn < input.turns.length; turn += 2) {
        const incoming = input.turns[turn];
        const outgoing = input.turns[turn + 1];
        if (!Number.isFinite(distances[incoming])) continue;
        if (outgoing === destination) {
          const targetCost = distances[incoming] + targetFraction * input.costs[destination];
          if (targetCost < bestCost) {
            bestCost = targetCost;
            targetPredecessor = incoming;
          }
        }
        const candidate = distances[incoming] + input.costs[outgoing];
        if (candidate < distances[outgoing]) {
          distances[outgoing] = candidate;
          predecessorEdges[query * edgeCount + outgoing] = incoming;
          changed = true;
        }
      }
    } while (changed);
    status[query] = 0;
    costs[query] = bestCost;
    if (source === destination) predecessorEdges[query * edgeCount + source] = targetPredecessor;
  }
  return {costs, status, predecessorEdges};
}

/** CPU twin; both modes retain turn-order ties and ignore GPU storage/iteration budgets. */
export class CPUNetworkPointRoutingOperation extends GPUCPUOperation {
  constructor(props: NetworkPointRoutingOperationProps) {
    const id = props.id ?? 'network-point-routing';
    validateCapacity(id, 'edgeCount', props.edgeCount);
    validateCapacity(id, 'queryCount', props.queryCount);
    validateCapacity(id, 'queryCount * edgeCount', props.queryCount * props.edgeCount);
    const operation = new GPUNetworkPointRoutingOperation(props);
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
        const edgeCosts = getCPUOperationFloat32Rows(id, inputs, 'edgeCosts');
        const queries = getCPUOperationUint32Rows(id, inputs, 'queries', 'uint32x2');
        if (edgeCosts.length !== props.edgeCount)
          throw new Error(`${id} costs length must equal edgeCount`);
        if (queries.length !== props.queryCount * 2)
          throw new Error(`${id} queries length must equal queryCount * 2`);
        const result = computeNetworkPointRoutingOnCPU({
          costs: edgeCosts,
          queries,
          turns: getCPUOperationUint32Rows(id, inputs, 'turns', 'uint32x2'),
          fractions: getCPUOperationFloat32Rows(id, inputs, 'fractions', 'float32x2')
        });
        return {
          [props.output.costs]: result.costs,
          [props.output.status]: result.status,
          [props.output.predecessorEdges]: result.predecessorEdges
        };
      }
    });
  }
}
