// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/graph/gpu-core';
import type {
  GPUInvalidStatusPort,
  GPUIterationStatusPort
} from '../../analysis-core/contracts/status';
import {getCoreDefaultBuffer} from '../../analysis-core/network-topology';
import {
  validateRequirement,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';

/** Named diagnostics shared by bounded network optimizers. @internal */
export type NetworkOptimizationStatusPort = GPUInvalidStatusPort & GPUIterationStatusPort;

/** Only these scalar ports are implemented by the bounded network optimizers. */
const NETWORK_OPTIMIZATION_STATUS_FIELDS = {
  invalidCount: true,
  iterationCount: true,
  converged: true,
  iterationLimitReached: true
} satisfies Record<keyof NetworkOptimizationStatusPort, true>;

/** Flattens the optional ports for ownership and alias validation. @internal */
export function getNetworkOptimizationStatusViews(
  status?: NetworkOptimizationStatusPort
): GraphDataView<'uint32'>[] {
  return Object.values(status ?? {}).filter((view): view is GraphDataView<'uint32'> =>
    Boolean(view)
  );
}

/** Validate scalars and independently returned output allocations once, at construction. @internal */
export function validateNetworkOptimizationStatus(
  id: string,
  outputs: readonly GraphDataView[],
  status?: NetworkOptimizationStatusPort
): void {
  validateNetworkOptimizationStatusFields(id, status);
  for (const [name, view] of Object.entries(status ?? {})) {
    if (!view) continue;
    validateSinglePackedView(view, ['uint32'], `${id} output.status.${name}`);
    validateRequirement(id, view.length === 1, `output.status.${name} must hold one row`);
  }
  const outputBuffers = [...outputs, ...getNetworkOptimizationStatusViews(status)].map(view =>
    view.buffer.defaultBuffer ? getCoreDefaultBuffer(view.buffer.defaultBuffer) : view.buffer
  );
  validateRequirement(
    id,
    new Set(outputBuffers).size === outputBuffers.length,
    'output views must use distinct buffers, including summary and status ports'
  );
}

function validateNetworkOptimizationStatusFields(id: string, status?: object): void {
  for (const name of Object.keys(status ?? {})) {
    validateRequirement(
      id,
      Object.hasOwn(NETWORK_OPTIMIZATION_STATUS_FIELDS, name),
      `output.status.${name} must be a supported field (${Object.keys(NETWORK_OPTIMIZATION_STATUS_FIELDS).join(', ')})`
    );
  }
}

/** Publish named scalars in a separate pass to stay within eight storage bindings. @internal */
export function createNetworkOptimizationStatusNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  operation: string,
  summary: GraphDataView<'uint32'>,
  status: NetworkOptimizationStatusPort | undefined,
  expressions: Record<keyof NetworkOptimizationStatusPort, string>,
  additionalBindings: readonly WGSLKernelBinding[] = []
): GPUCommandNode<Parameters>[] {
  const bindings = Object.entries(status ?? {}).flatMap(([name, view]) =>
    view && Object.hasOwn(expressions, name)
      ? [{name, view, type: 'u32' as const, access: 'read_write' as const}]
      : []
  );
  if (!bindings.length) return [];
  return [
    createWGSLKernelNode(graph, {
      id: `${id}-status`,
      operation,
      variant: 'named-status',
      invocationCount: 1,
      workgroupSize: 2,
      bindings: [
        {name: 'summary', view: summary, type: 'u32', access: 'read'},
        ...additionalBindings,
        ...bindings
      ],
      body: bindings
        .map(
          ({name}) =>
            `${name}[${name}Offset] = ${expressions[name as keyof NetworkOptimizationStatusPort]};`
        )
        .join('\n')
    })
  ];
}

/** Graph-value names for optional named optimizer diagnostics. @internal */
export type NetworkOptimizationStatusValues = Partial<
  Record<keyof NetworkOptimizationStatusPort, string>
>;

/** Packed one-row CPU outputs with the same names and uint32 formats as the GPU ports. @internal */
export type NetworkOptimizationCPUStatus = Record<keyof NetworkOptimizationStatusPort, Uint32Array>;

/** Applies the contributor's field validation before declaring scalar operation outputs. @internal */
export function getNetworkOptimizationStatusOutputs(
  id: string,
  status?: NetworkOptimizationStatusValues
) {
  validateNetworkOptimizationStatusFields(id, status);
  return Object.fromEntries(
    Object.entries(status ?? {}).flatMap(([name, value]) =>
      value === undefined ? [] : [[name, {value, format: 'uint32' as const, length: 1}]]
    )
  );
}

/** Maps the common validity/completion summary header and an executed-work count. @internal */
export function getNetworkOptimizationCPUStatus(
  summary: Uint32Array,
  iterationCount: number
): NetworkOptimizationCPUStatus {
  const valid = summary[0];
  return {
    invalidCount: Uint32Array.of(1 - valid),
    iterationCount: Uint32Array.of(iterationCount),
    converged: Uint32Array.of(valid * summary[1]),
    iterationLimitReached: Uint32Array.of(valid * (1 - summary[1]))
  };
}
