// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  getViewBindingRange,
  GPUReduction,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  validateGPUStatusPort,
  type GPUInvalidStatusPort,
  type GPUOverflowStatusPort
} from '../../analysis-core/contracts/status';
import {
  validateGPUStorageBindingByteLength,
  validateGraphOutputsDisjointFromInputs,
  validateSinglePackedView,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {IS_FINITE_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';
import type {GPUFloat32OrUint32Values} from '../spatial-weights/float32-values';

/** Input diagnostics shared by the analytic local statistics. */
export type GPUSpatialAutocorrelationStatusPort = GPUInvalidStatusPort &
  Partial<GPUOverflowStatusPort>;

/** Validates optional scalar status outputs and rejects aliases with other result buffers. @internal */
export function validateSpatialAutocorrelationStatus(
  id: string,
  status: GPUSpatialAutocorrelationStatusPort | undefined,
  outputs: readonly (GraphDataView | undefined)[]
): void {
  validateGPUStatusPort(id, status ?? {});
  const views = Object.values(status ?? {});
  for (const [name, view] of Object.entries(status ?? {})) {
    if (view) validateSinglePackedView(view, ['uint32'], `${id} output.status.${name}`);
  }
  if (status?.invalidCount) {
    validateViewLength(id, 'output.status.invalidCount', status.invalidCount, 1);
  }
  validateGraphOutputsDisjointFromInputs(id, views, outputs);
  if (status?.invalidCount && status.overflow) {
    validateGraphOutputsDisjointFromInputs(id, [status.invalidCount], [status.overflow]);
  }
}

/** Reduces rejected input rows and forwards weight-capacity failure, without readback. @internal */
export function getSpatialAutocorrelationStatusNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  operation: string,
  props: {
    values: GPUFloat32OrUint32Values;
    mask?: GraphDataView<'uint32'>;
    weights: {status?: Partial<GPUOverflowStatusPort>};
    output: {status?: GPUSpatialAutocorrelationStatusPort};
  }
): GPUCommandNode<Parameters>[] {
  const status = props.output.status;
  for (const [name, view] of [
    ['output.status.invalidCount', status?.invalidCount],
    ['output.status.overflow', status?.overflow],
    ['weights.status.overflow', status?.overflow ? props.weights.status?.overflow : undefined]
  ] as const) {
    if (view) {
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        name,
        getViewBindingRange(view).size,
        `use a smaller packed ${name} view`
      );
    }
  }
  const nodes: GPUCommandNode<Parameters>[] = [];
  if (status?.invalidCount) {
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'invalid-row flags',
      props.values.length * 4,
      'use fewer values rows'
    );
    const invalidRows = createTransientView(
      graph,
      `${id}-invalid-rows`,
      'uint32',
      props.values.length
    );
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-invalid-rows`,
        operation,
        variant: 'invalid-rows',
        invocationCount: props.values.length,
        bindings: [
          {
            name: 'values',
            view: props.values,
            type: props.values.format === 'uint32' ? 'u32' : 'f32',
            access: 'read'
          },
          ...(props.mask
            ? [{name: 'mask', view: props.mask, type: 'u32' as const, access: 'read' as const}]
            : []),
          {name: 'invalidRows', view: invalidRows, type: 'u32', access: 'read_write'}
        ],
        declarations: IS_FINITE_BITS_WGSL,
        body: `let selected = ${props.mask ? 'mask[maskOffset + index] != 0u' : 'true'};
  invalidRows[invalidRowsOffset + index] = u32(selected && !isFiniteBits(f32(values[valuesOffset + index])));`
      }),
      ...new GPUReduction({
        id: `${id}-invalid-count`,
        input: invalidRows,
        output: status.invalidCount,
        operation: 'sum'
      }).getCommandNodes(graph)
    );
  }
  if (status?.overflow) {
    const overflow = props.weights.status?.overflow;
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-overflow-status`,
        operation,
        variant: 'overflow-status',
        invocationCount: 1,
        bindings: [
          ...(overflow
            ? [
                {
                  name: 'inputOverflow',
                  view: overflow,
                  type: 'u32' as const,
                  access: 'read' as const
                }
              ]
            : []),
          {name: 'overflow', view: status.overflow, type: 'u32', access: 'read_write'}
        ],
        body: `overflow[overflowOffset] = ${overflow ? 'u32(inputOverflow[inputOverflowOffset] != 0u)' : '0u'};`
      })
    );
  }
  return nodes;
}
