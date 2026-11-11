// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {
  GPUNetworkReachabilityOperation,
  type NetworkReachabilityOperationProps
} from './gpu-network-reachability-operation';

import {
  computeNetworkRoutingDistances,
  computeNetworkRoutingPredecessors,
  type CPUNetworkRoutingInput
} from '../../analysis-core/operations/cpu-network-routing';

const NONE = 0xffffffff;

/** Packed routing inputs. Offsets are clamped per row exactly as in the GPU kernels. */
export type CPUNetworkReachabilityInput = CPUNetworkRoutingInput;
/** Layout options. GPU round and workgroup budgets are deliberately absent. */
export type CPUNetworkReachabilityOptions = {laneCount?: number};
/** Every output port, including optional diagnostics, in the contributor's packed layout. */
export type CPUNetworkReachabilityResult = {
  costs: Float32Array;
  predecessors: Uint32Array;
  bands: Uint32Array;
  bandCounts: Uint32Array;
  unresolvedCount: Uint32Array;
  converged: Uint32Array;
  iterationCount: Uint32Array;
  iterationLimitReached: Uint32Array;
  relaxationCount: Uint32Array;
  reachedEdgeCount: Uint32Array;
};

/**
 * Computes uncapped shortest paths and the contributor's cycle-safe predecessor and band rules.
 * Overlapping CSR rows share slot-indexed tie flags exactly as on the GPU; predecessors can then
 * follow an edge that is not tight for that source, outside the valid-CSR cycle-safety guarantee.
 *
 * What differs from the GPU contributor:
 * - Float64 costs and tight-edge tests precede one float32 output rounding. Near ties, cutoff
 *   decisions, subnormal weights and sums overflowing float32 can differ from GPU arithmetic.
 * - All searches and tie levels finish exactly; GPU round, local-hop, device and activeIterations
 *   limits are ignored. converged is 1, iterationLimitReached and unresolvedCount are 0.
 * - iterationCount is 1 for the single completed CPU search. relaxationCount counts CSR slots
 *   scanned once per settled node, rather than GPU visits. reachedEdgeCount counts in-range edges
 *   from finite float64 labels; labels that round to float32 infinity can therefore contribute.
 * - Negative/NaN weights and starts are ignored; NaN costLimit disables the cutoff, like the GPU.
 *   Out-of-range or decreasing offsets retain the GPU's per-row clamping interpretation.
 *
 * @throws Error naming an argument whose array length is inconsistent.
 */
export function computeNetworkReachabilityOnCPU(
  input: CPUNetworkReachabilityInput,
  options: CPUNetworkReachabilityOptions = {}
): CPUNetworkReachabilityResult {
  const laneCount = options.laneCount ?? 1;
  const {distances, scannedEdges, reachedEdges} = computeNetworkRoutingDistances(
    'computeNetworkReachabilityOnCPU',
    input,
    laneCount
  );
  const predecessors =
    laneCount === 1
      ? computeNetworkRoutingPredecessors(input, distances)
      : new Uint32Array(distances.length).fill(NONE);
  const bands = new Uint32Array(distances.length).fill(NONE);
  const bandCounts = new Uint32Array(input.costBreaks?.length ?? 0);
  if (input.costBreaks) {
    if (!input.costBreaks.length) throw new Error('costBreaks must contain at least one row');
    for (let node = 0; node < distances.length; node++) {
      if (!Number.isFinite(distances[node])) continue;
      let band = 0;
      for (const threshold of input.costBreaks) if (threshold < distances[node]) band++;
      if (band < bandCounts.length) {
        bands[node] = band;
        bandCounts[band]++;
      }
    }
  }
  const getWords = (value: number) => Uint32Array.of(value % 2 ** 32, Math.floor(value / 2 ** 32));
  return {
    costs: Float32Array.from(distances),
    predecessors,
    bands,
    bandCounts,
    unresolvedCount: Uint32Array.of(0),
    converged: Uint32Array.of(1),
    iterationCount: Uint32Array.of(1),
    iterationLimitReached: Uint32Array.of(0),
    relaxationCount: getWords(scannedEdges),
    reachedEdgeCount: getWords(reachedEdges)
  };
}

/** CPU twin; GPU-only iteration and storage limits are dropped. */
export class CPUNetworkReachabilityOperation extends GPUCPUOperation {
  constructor(props: NetworkReachabilityOperationProps) {
    const id = props.id ?? 'network-reachability';
    validateCapacity(id, 'nodeCount', props.nodeCount);
    validateCapacity(id, 'laneCount', props.laneCount ?? 1);
    validateCapacity(id, 'nodeCount * laneCount', props.nodeCount * (props.laneCount ?? 1));
    validateRequirement(
      id,
      (props.laneCount ?? 1) === 1 || (!props.output.predecessors && !props.output.unresolvedCount),
      'laneCount > 1 cannot request predecessors or unresolvedCount'
    );
    validateRequirement(
      id,
      !props.output.unresolvedCount || Boolean(props.output.predecessors),
      'output.unresolvedCount requires output.predecessors'
    );
    validateRequirement(
      id,
      Boolean(props.costBreaks) === Boolean(props.output.bands || props.output.bandCounts),
      'costBreaks is required exactly when bands or bandCounts is requested'
    );
    if (props.costBreakCount !== undefined || props.output.bandCounts !== undefined)
      validateCapacity(id, 'costBreakCount', props.costBreakCount ?? 0);
    if (typeof props.startCount === 'number')
      validateCapacity(id, 'startCount', props.startCount, 0);
    const operation = new GPUNetworkReachabilityOperation(props);
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
        const topology = {
          offsets: getCPUOperationUint32Rows(id, inputs, 'offsets'),
          neighbors: getCPUOperationUint32Rows(id, inputs, 'neighbors'),
          weights: getCPUOperationFloat32Rows(id, inputs, 'weights')
        };
        if (props.topology.nodePositions !== undefined) {
          const nodePositions = getCPUOperationFloat32Rows(
            id,
            inputs,
            'nodePositions',
            'float32x2'
          );
          if (nodePositions.length !== props.nodeCount * 2)
            throw new Error(`${id} topology.nodePositions must contain one float32x2 row per node`);
        }
        if (topology.offsets.length !== props.nodeCount + 1)
          throw new Error(`${id} topology.offsets length must equal nodeCount + 1`);
        const result = computeNetworkReachabilityOnCPU(
          {
            topology,
            startNodes: getCPUOperationUint32Rows(id, inputs, 'startNodes'),
            startCosts:
              props.startCosts === undefined
                ? undefined
                : getCPUOperationFloat32Rows(id, inputs, 'startCosts'),
            startCount:
              typeof props.startCount === 'string'
                ? getCPUOperationUint32Rows(id, inputs, 'startCount')
                : props.startCount,
            costLimit:
              props.costLimit === undefined
                ? undefined
                : getCPUOperationFloat32Rows(id, inputs, 'costLimit'),
            activeIterations:
              props.activeIterations === undefined
                ? undefined
                : getCPUOperationUint32Rows(id, inputs, 'activeIterations'),
            costBreaks:
              props.costBreaks === undefined
                ? undefined
                : getCPUOperationFloat32Rows(id, inputs, 'costBreaks')
          },
          {laneCount: props.laneCount}
        );
        if (props.costBreakCount !== undefined && result.bandCounts.length !== props.costBreakCount)
          throw new Error(`${id} costBreaks length must equal costBreakCount`);
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPUNetworkReachabilityResult]
          ])
        );
      }
    });
  }
}
