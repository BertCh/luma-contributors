// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {snapshotNetworkReachabilityConstructionProps} from './network-reachability-construction';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
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
  resolveNetworkRoutingCostPrecision,
  computeNetworkRoutingPredecessorsWithStatus,
  type CPUNetworkRoutingInput
} from '../../analysis-core/operations/cpu-network-routing';

const NONE = 0xffffffff;

/** Packed routing inputs. Offsets are clamped per row exactly as in the GPU kernels. */
export type CPUNetworkReachabilityInput = CPUNetworkRoutingInput;
/** Explicit round budgets bind the deterministic CPU frontier too; omitted budgets are uncapped. */
export type CPUNetworkReachabilityOptions = {
  laneCount?: number;
  maximumIterations?: number;
  localIterations?: number;
  maximumTieIterations?: number;
};
/** Every output port, including optional diagnostics, in the contributor's packed layout. */
export type CPUNetworkReachabilityResult = {
  costs: Float32Array;
  predecessors: Uint32Array;
  /** Authoritative CSR slots; minimum passable parallel weight, then smallest slot. */
  predecessorEdges: Uint32Array;
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
 * Computes shortest paths and the contributor's cycle-safe predecessor and band rules.
 * The CPU predecessor reader keeps tie witnesses specific to their source, including when
 * clamped CSR rows overlap. An edge tight for one source cannot supply another source's parent.
 *
 * What differs from the GPU contributor:
 * - costPrecision defaults to float64 costs/tight-edge tests and one output rounding.
 *   float32-per-edge rounds every addition before cutoff and predecessor tests. Subnormal
 *   weights and overflowing sums can still differ from GPU arithmetic.
 * - With explicit round budgets or activeIterations, a synchronous CPU frontier consumes at most
 *   localIterations hops per round (default 32). GPU workgroup scheduling can finish fewer hops.
 *   Without explicit budgets, searches and tie levels run to completion. Device limits are dropped.
 * - iterationCount counts bounded frontier rounds, or 1 for an uncapped CPU search.
 *   relaxationCount counts CPU edge visits (bounded frontiers can revisit nodes).
 *   reachedEdgeCount counts in-range edges
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
  input = snapshotConstructionFields(input, [
    'topology',
    'startNodes',
    'startCosts',
    'startCount',
    'costLimit',
    'costBreaks',
    'activeIterations',
    'costPrecision'
  ]);
  input = {
    ...input,
    costPrecision: resolveNetworkRoutingCostPrecision(
      'computeNetworkReachabilityOnCPU',
      input.costPrecision
    )
  };
  options = snapshotConstructionFields(options, [
    'laneCount',
    'maximumIterations',
    'localIterations',
    'maximumTieIterations'
  ]);
  const laneCount = options.laneCount ?? 1;
  const localIterations = options.localIterations ?? 32;
  for (const [name, value] of Object.entries(options))
    if (value !== undefined) validateCapacity('computeNetworkReachabilityOnCPU', name, value);
  const bounded = options.maximumIterations !== undefined || input.activeIterations !== undefined;
  const roundLimit = Math.min(
    options.maximumIterations ?? 64,
    input.activeIterations?.[0] ?? Infinity
  );
  const {
    distances,
    scannedEdges,
    reachedEdges,
    converged: costsConverged,
    hopCount
  } = computeNetworkRoutingDistances(
    'computeNetworkReachabilityOnCPU',
    input,
    laneCount,
    bounded ? roundLimit * localIterations : Infinity
  );
  const predecessorResult =
    laneCount === 1
      ? computeNetworkRoutingPredecessorsWithStatus(
          input,
          distances,
          undefined,
          options.maximumTieIterations === undefined
            ? Infinity
            : options.maximumTieIterations * localIterations
        )
      : undefined;
  const predecessors =
    predecessorResult?.predecessors ?? new Uint32Array(distances.length).fill(NONE);
  const predecessorEdges =
    predecessorResult?.predecessorEdges ?? new Uint32Array(predecessors.length).fill(NONE);
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
  const roots = new Set<number>();
  const startCount = Math.min(
    typeof input.startCount === 'number'
      ? input.startCount
      : (input.startCount?.[0] ?? input.startNodes.length),
    input.startNodes.length
  );
  for (let start = 0; start < startCount; start++) {
    const node = input.startNodes[start];
    if (node < distances.length && distances[node] === (input.startCosts?.[start] ?? 0))
      roots.add(node);
  }
  let unresolvedCount = 0;
  if (laneCount === 1)
    for (let node = 0; node < distances.length; node++)
      if (Number.isFinite(distances[node]) && predecessors[node] === NONE && !roots.has(node))
        unresolvedCount++;
  const converged =
    costsConverged &&
    (predecessorResult?.converged ?? true) &&
    unresolvedCount === 0 &&
    (!bounded || roundLimit > 0);
  return {
    costs: Float32Array.from(distances),
    predecessors,
    predecessorEdges,
    bands,
    bandCounts,
    unresolvedCount: Uint32Array.of(unresolvedCount),
    converged: Uint32Array.of(Number(converged)),
    iterationCount: Uint32Array.of(
      bounded ? (roundLimit === 0 ? 0 : Math.max(1, Math.ceil(hopCount / localIterations))) : 1
    ),
    iterationLimitReached: Uint32Array.of(Number(!converged)),
    relaxationCount: getWords(scannedEdges),
    reachedEdgeCount: getWords(reachedEdges)
  };
}

/** CPU twin with explicit round limits; GPU storage limits are dropped. */
export class CPUNetworkReachabilityOperation extends GPUCPUOperation {
  constructor(props: NetworkReachabilityOperationProps) {
    props = snapshotNetworkReachabilityConstructionProps(props);
    const id = props.id ?? 'network-reachability';
    const costPrecision = resolveNetworkRoutingCostPrecision(id, props.costPrecision);
    validateCapacity(id, 'nodeCount', props.nodeCount);
    validateCapacity(id, 'laneCount', props.laneCount ?? 1);
    validateCapacity(id, 'nodeCount * laneCount', props.nodeCount * (props.laneCount ?? 1));
    for (const name of ['maximumIterations', 'localIterations', 'maximumTieIterations'] as const)
      if (props[name] !== undefined) validateCapacity(id, name, props[name]);
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
    const operation = new GPUNetworkReachabilityOperation({...props, costPrecision: undefined});
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
            costPrecision,
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
          {
            laneCount: props.laneCount,
            maximumIterations: props.maximumIterations,
            localIterations: props.localIterations,
            maximumTieIterations:
              props.output.predecessors === undefined ? undefined : props.maximumTieIterations
          }
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
