// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateMaximumCount
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {PhiloxStream} from '../../analysis-core/kernels/permutation-random';
import {PHILOX_PURPOSES} from '../../analysis-core/kernels/philox-random';
import {getCPUOperationFloat32Rows} from '../../analysis-core/operations/cpu-operation-rows';
import {
  GPUNetworkLouvainOperation,
  type NetworkLouvainOperationProps
} from './gpu-network-louvain-operation';

export type CPUNetworkLouvainInput = {weights: Float32Array; parameters: Float32Array};
export type CPUNetworkLouvainOptions = Pick<
  NetworkLouvainOperationProps,
  'nodeCount' | 'directed' | 'seed' | 'maximumLevels' | 'maximumIterations'
>;
export type CPUNetworkLouvainResult = {
  communities: Uint32Array;
  objectiveTrace: Float32Array;
  summary: Uint32Array;
};

/**
 * Sequential Louvain local moves and implicit coarsening, with full float64 objective evaluation.
 * Only existing neighboring communities are candidates, visited by ascending community ID. A
 * move must improve the best objective by more than the supplied tolerance. A seeded visit order
 * draws the same Fisher-Yates permutation once from the contributor's registered Philox stream.
 * Invalid parameters/weights return identity labels, NaN trace and valid=0.
 *
 * What differs from the GPU contributor:
 * - Float64 objective evaluation can choose different moves at near ties or tolerance boundaries.
 *   Well-separated partitions compare exactly; their objective traces use relative tolerance 2e-5.
 *   The GPU can leave a finite cancellation residual where the float64 objective is zero. For
 *   gamma=1, the coarsening comparison allows absolute error 8 * 2^-23 at zero: eight float32
 *   spacings at unit magnitude, the scale of the normalized internal-weight and penalty terms.
 *   Nonzero trace entries retain the relative tolerance; the CPU keeps its exact zero.
 * - GPU-only 32-node, 32-level and 64-sweep caps are dropped. The requested maximumLevels and
 *   maximumIterations (defaults 8 and 16) remain because they define the trace length and the
 *   documented nonconvergence status. This remains heuristic Louvain, not global optimization.
 * - Float32 intermediate overflow does not invalidate finite float64 work; output overflow becomes
 *   Infinity. Subnormal weights are retained.
 *
 * @throws If weights/parameters lengths are inconsistent or shape/seed/budgets are invalid.
 */
export function computeNetworkLouvainOnCPU(
  input: CPUNetworkLouvainInput,
  options: CPUNetworkLouvainOptions
): CPUNetworkLouvainResult {
  validateLouvainOptions('computeNetworkLouvainOnCPU', options);
  const {nodeCount} = options;
  const {weights, parameters} = input;
  const maximumLevels = options.maximumLevels ?? 8;
  const maximumIterations = options.maximumIterations ?? 16;
  if (weights.length !== nodeCount ** 2)
    throw new Error('weights length must equal nodeCount squared');
  if (parameters.length !== 2)
    throw new Error('parameters must contain resolution and minimum improvement');
  const communities = Uint32Array.from({length: nodeCount}, (_, node) => node);
  const groups = Uint32Array.from(communities);
  const objectiveTrace = new Float32Array(1 + maximumLevels * maximumIterations).fill(NaN);
  const summary = new Uint32Array(4);
  const result = {communities, objectiveTrace, summary};
  // No exported packer/layout exists in the contributor; this is its documented row order.
  const [resolution, tolerance] = parameters;
  if (
    !Number.isFinite(resolution) ||
    resolution < 0 ||
    !Number.isFinite(tolerance) ||
    tolerance < 0
  )
    return result;
  let total = 0;
  for (let source = 0; source < nodeCount; source++)
    for (let destination = 0; destination < nodeCount; destination++) {
      const weight = weights[source * nodeCount + destination];
      if (
        !Number.isFinite(weight) ||
        weight < 0 ||
        (!options.directed && weight !== weights[destination * nodeCount + source])
      )
        return result;
      total += weight;
    }
  summary[0] = 1;
  if (total === 0) {
    objectiveTrace[0] = 0;
    summary[1] = 1;
    return result;
  }
  const evaluateObjective = (): number => {
    let internal = 0;
    const strengths = new Float64Array(nodeCount);
    const incoming = new Float64Array(nodeCount);
    for (let source = 0; source < nodeCount; source++)
      for (let destination = 0; destination < nodeCount; destination++) {
        const weight = weights[source * nodeCount + destination];
        strengths[communities[source]] += weight;
        incoming[communities[destination]] += weight;
        if (communities[source] === communities[destination]) internal += weight;
      }
    let penalty = 0;
    for (let community = 0; community < nodeCount; community++)
      penalty += (strengths[community] / total) * (incoming[community] / total);
    return internal / total - resolution * penalty;
  };
  const order = Array.from({length: nodeCount}, (_, node) => node);
  if (options.seed !== undefined) {
    const stream = new PhiloxStream(
      [options.seed, PHILOX_PURPOSES.networkLouvainVisitOrder.id],
      0,
      0,
      0
    );
    for (let remaining = nodeCount; remaining > 1; remaining--) {
      const selected = stream.nextBelow(remaining);
      [order[remaining - 1], order[selected]] = [order[selected], order[remaining - 1]];
    }
  }
  let current = evaluateObjective();
  objectiveTrace[0] = current;
  let sweeps = 0;
  for (let level = 0; level < maximumLevels; level++) {
    let levelChanged = false;
    let stable = false;
    for (let round = 0; round < maximumIterations; round++) {
      let changed = false;
      for (const group of order) {
        const members: number[] = [];
        let original = 0;
        for (let node = 0; node < nodeCount; node++)
          if (groups[node] === group) {
            members.push(node);
            original = communities[node];
          }
        if (!members.length) continue;
        let best = original;
        let bestObjective = current;
        for (let candidate = 0; candidate < nodeCount; candidate++) {
          if (candidate === original) continue;
          let adjacent = false;
          for (const source of members)
            for (let destination = 0; destination < nodeCount; destination++) {
              if (
                groups[destination] !== group &&
                communities[destination] === candidate &&
                (weights[source * nodeCount + destination] > 0 ||
                  weights[destination * nodeCount + source] > 0)
              )
                adjacent = true;
            }
          if (!adjacent) continue;
          for (const node of members) communities[node] = candidate;
          const candidateObjective = evaluateObjective();
          if (candidateObjective > bestObjective + tolerance) {
            best = candidate;
            bestObjective = candidateObjective;
          }
          for (const node of members) communities[node] = original;
        }
        for (const node of members) communities[node] = best;
        if (best !== original) {
          changed = true;
          current = bestObjective;
        }
      }
      objectiveTrace[++sweeps] = current;
      levelChanged ||= changed;
      if (!changed) {
        stable = true;
        break;
      }
    }
    summary[2] = level + 1;
    summary[3] = sweeps;
    if (!stable) return result;
    if (!levelChanged) {
      summary[1] = 1;
      return result;
    }
    groups.set(communities);
  }
  return result;
}

function validateLouvainOptions(id: string, options: CPUNetworkLouvainOptions): void {
  validateCapacity(id, 'nodeCount', options.nodeCount);
  validateMaximumCount(id, 'nodeCount squared', options.nodeCount ** 2, 0xffffffff);
  validateCapacity(id, 'maximumLevels', options.maximumLevels ?? 8);
  validateCapacity(id, 'maximumIterations', options.maximumIterations ?? 16);
  validateMaximumCount(
    id,
    '1 + maximumLevels * maximumIterations',
    1 + (options.maximumLevels ?? 8) * (options.maximumIterations ?? 16),
    0xffffffff
  );
  if (options.seed !== undefined) validateCapacity(id, 'seed', options.seed, 0);
}

/** CPU Louvain twin; requested trace budgets remain, private GPU array caps do not. */
export class CPUNetworkLouvainOperation extends GPUCPUOperation {
  constructor(props: NetworkLouvainOperationProps) {
    const id = props.id ?? 'network-louvain';
    validateLouvainOptions(id, props);
    const operation = new GPUNetworkLouvainOperation(props);
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
        const result = computeNetworkLouvainOnCPU(
          {
            weights: getCPUOperationFloat32Rows(id, inputs, 'weights'),
            parameters: getCPUOperationFloat32Rows(id, inputs, 'parameters')
          },
          props
        );
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPUNetworkLouvainResult]
          ])
        );
      }
    });
  }
}
