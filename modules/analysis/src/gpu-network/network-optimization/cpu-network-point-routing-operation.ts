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

/** Matches the bounded reference turn-scan schedule. */
const MAXIMUM_REFERENCE_RELAXATIONS = 65536;
/** Matches the composed reachability frontier schedule. */
const MAXIMUM_SCALABLE_RELAXATIONS = 1024;
/** Each scalable frontier round permits up to 64 local hops. */
const SCALABLE_LOCAL_ITERATIONS = 64;

/** Packed directed-edge turns, queries and fractional endpoints. */
export type CPUNetworkPointRoutingInput = {
  turns: Uint32Array;
  turnPenalties?: Float32Array;
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
 *   when comparing those ties. With an explicit budget, mode selects turn scans or synchronous hops.
 * - Float64 turn relaxation and partial-edge arithmetic precede one float32 output rounding;
 *   near ties, subnormals and paths overflowing float32 may produce different costs or parents.
 * - A full-edge sum overflowing float32 does not trigger the GPU's status-1 early return. The
 *   CPU continues in float64 and can publish a finite partial-target cost with status 0 and
 *   predecessors where the GPU leaves Infinity and status 1. Scalable mode clears invalid-query
 *   predecessors; reference mode can retain partially written predecessors.
 * - Without an explicit budget, relaxation continues to a fixpoint. An explicit maximumRelaxations
 *   binds reference turn scans or scalable rounds of up to 64 synchronous hops. Exhaustion sets
 *   status 2 and publishes the incumbent; zero permits only direct same-edge travel.
 * - Negative or non-finite base costs close that edge: turns touching it are skipped and queries
 *   with a closed endpoint get status 1. Other queries remain usable.
 * - Invalid edge IDs, fractions and turn endpoints retain the GPU's per-query status result;
 *   inconsistent packed array lengths throw instead of being interpreted as partial rows.
 *
 * @throws Error naming turns, costs, queries or fractions for an inconsistent packed array.
 */
export function computeNetworkPointRoutingOnCPU(
  input: CPUNetworkPointRoutingInput,
  options: {mode?: 'reference' | 'scalable'; maximumRelaxations?: number} = {}
): CPUNetworkPointRoutingResult {
  if (options.maximumRelaxations !== undefined)
    validateCapacity(
      'computeNetworkPointRoutingOnCPU',
      'maximumRelaxations',
      options.maximumRelaxations,
      0,
      options.mode === 'reference' ? MAXIMUM_REFERENCE_RELAXATIONS : MAXIMUM_SCALABLE_RELAXATIONS
    );
  if (!input.costs.length) throw new Error('costs must contain at least one edge');
  if (input.turns.length % 2) throw new Error('turns must contain uint32x2 rows');
  if (input.turnPenalties && input.turnPenalties.length * 2 !== input.turns.length)
    throw new Error('turnPenalties must contain one row per turn');
  if (input.queries.length % 2) throw new Error('queries must contain uint32x2 rows');
  if (input.fractions.length !== input.queries.length)
    throw new Error('fractions must contain one float32x2 row per query');
  const edgeCount = input.costs.length;
  const queryCount = input.queries.length / 2;
  const costs = new Float32Array(queryCount).fill(Infinity);
  const status = new Uint32Array(queryCount).fill(1);
  const predecessorEdges = new Uint32Array(queryCount * edgeCount).fill(0xffffffff);
  const invalidNetwork =
    input.turns.some(edge => edge >= edgeCount) ||
    Boolean(input.turnPenalties?.some(penalty => !Number.isFinite(penalty) || penalty < 0));
  for (let query = 0; query < queryCount; query++) {
    const source = input.queries[query * 2];
    const destination = input.queries[query * 2 + 1];
    const sourceFraction = input.fractions[query * 2];
    const targetFraction = input.fractions[query * 2 + 1];
    if (
      invalidNetwork ||
      source >= edgeCount ||
      destination >= edgeCount ||
      !Number.isFinite(input.costs[source]) ||
      input.costs[source] < 0 ||
      !Number.isFinite(input.costs[destination]) ||
      input.costs[destination] < 0 ||
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
    const synchronous = options.mode !== 'reference' && options.maximumRelaxations !== undefined;
    const maximumScans =
      options.maximumRelaxations === undefined
        ? Infinity
        : options.maximumRelaxations * (synchronous ? SCALABLE_LOCAL_ITERATIONS : 1);
    let rounds = 0;
    let converged = false;
    while (rounds < maximumScans) {
      let changed = false;
      const previousDistances = synchronous ? distances.slice() : distances;
      for (let turn = 0; turn < input.turns.length; turn += 2) {
        const incoming = input.turns[turn];
        const outgoing = input.turns[turn + 1];
        const penalty = input.turnPenalties?.[turn / 2] ?? 0;
        if (
          !Number.isFinite(previousDistances[incoming]) ||
          !Number.isFinite(input.costs[outgoing]) ||
          input.costs[outgoing] < 0
        )
          continue;
        const candidate = previousDistances[incoming] + penalty + input.costs[outgoing];
        if (candidate < distances[outgoing]) {
          distances[outgoing] = candidate;
          predecessorEdges[query * edgeCount + outgoing] = incoming;
          changed = true;
        }
      }
      rounds++;
      if (!changed) {
        converged = true;
        break;
      }
    }
    if (rounds > 0) {
      for (let turn = 0; turn < input.turns.length; turn += 2) {
        const incoming = input.turns[turn];
        if (input.turns[turn + 1] !== destination || !Number.isFinite(distances[incoming]))
          continue;
        let cursor = incoming;
        for (let hop = 0; hop < edgeCount && cursor !== source; hop++) {
          if (cursor >= edgeCount || cursor === destination) break;
          cursor = predecessorEdges[query * edgeCount + cursor];
        }
        if (cursor !== source) continue;
        const targetCost =
          distances[incoming] +
          (input.turnPenalties?.[turn / 2] ?? 0) +
          targetFraction * input.costs[destination];
        if (targetCost < bestCost) {
          bestCost = targetCost;
          targetPredecessor = incoming;
        }
      }
    }
    status[query] = converged ? 0 : 2;
    costs[query] = bestCost;
    predecessorEdges[query * edgeCount + destination] = targetPredecessor;
  }
  return {costs, status, predecessorEdges};
}

/** CPU twin; retains turn-order ties, drops GPU storage caps and honors explicit iteration budgets. */
export class CPUNetworkPointRoutingOperation extends GPUCPUOperation {
  constructor(props: NetworkPointRoutingOperationProps) {
    const id = props.id ?? 'network-point-routing';
    validateCapacity(id, 'edgeCount', props.edgeCount);
    validateCapacity(id, 'queryCount', props.queryCount);
    validateCapacity(id, 'queryCount * edgeCount', props.queryCount * props.edgeCount);
    if (props.maximumRelaxations !== undefined)
      validateCapacity(
        id,
        'maximumRelaxations',
        props.maximumRelaxations,
        0,
        props.mode === 'reference' ? MAXIMUM_REFERENCE_RELAXATIONS : MAXIMUM_SCALABLE_RELAXATIONS
      );
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
        const result = computeNetworkPointRoutingOnCPU(
          {
            costs: edgeCosts,
            queries,
            turns: getCPUOperationUint32Rows(id, inputs, 'turns', 'uint32x2'),
            turnPenalties:
              props.turnPenalties === undefined
                ? undefined
                : getCPUOperationFloat32Rows(id, inputs, 'turnPenalties'),
            fractions: getCPUOperationFloat32Rows(id, inputs, 'fractions', 'float32x2')
          },
          {mode: props.mode, maximumRelaxations: props.maximumRelaxations}
        );
        return {
          [props.output.costs]: result.costs,
          [props.output.status]: result.status,
          [props.output.predecessorEdges]: result.predecessorEdges
        };
      }
    });
  }
}
