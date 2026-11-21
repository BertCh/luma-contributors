// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  type GraphDataView,
  type GPUCommandGraph,
  type GPUCommandNode
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  isGraphDataView,
  validateSinglePackedView,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateRasterWritableDispatches} from '../raster-grid/raster-view-validation';

/** Optional guarded composition; existing unguarded callers keep their output contract. @internal */
export function validateRasterProducerCompleteness(
  id: string,
  producerOverflow: GraphDataView<'uint32'> | undefined,
  incomplete: GraphDataView<'uint32'> | undefined,
  outputs: readonly GraphDataView[],
  incompleteSources: readonly GraphDataView<'uint32'>[] = []
): void {
  for (const [name, view] of [
    ['producerOverflow', producerOverflow],
    ...incompleteSources.map((view, index) => [`incompleteSources[${index}]`, view] as const),
    ['incomplete', incomplete]
  ] as const) {
    if (!view) continue;
    validateSinglePackedView(view, ['uint32'], `${id} ${name}`);
    validateRequirement(id, view.length === 1, `${name} must contain one uint32 row`);
  }
  validateRequirement(
    id,
    (!producerOverflow && incompleteSources.length === 0) || Boolean(incomplete),
    'producerOverflow or incompleteSources requires the incomplete output to distinguish suppressed results from empty input'
  );
  for (const source of [...(producerOverflow ? [producerOverflow] : []), ...incompleteSources]) {
    validateRasterWritableDispatches(
      id,
      [source],
      outputs,
      outputs.map(view => ({inputs: [source], outputs: [view]}))
    );
  }
}

/** Final parallel publication guard, after every reduction (including counts-only). @internal */
export function getRasterProducerCompletenessNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  producerOverflow: GraphDataView<'uint32'> | undefined,
  incomplete: GraphDataView<'uint32'> | undefined,
  columns: object,
  incompleteSources: readonly GraphDataView<'uint32'>[] = []
): GPUCommandNode<Parameters>[] {
  const nodes: GPUCommandNode<Parameters>[] = [];
  const sources = [...(producerOverflow ? [producerOverflow] : []), ...incompleteSources];
  // One source per dispatch keeps the composition independent of device binding-count limits.
  for (const [sourceIndex, source] of sources.entries()) {
    const suffix = sourceIndex === 0 ? '' : `-${sourceIndex}`;
    for (const [name, view] of Object.entries(columns)) {
      if (!isGraphDataView(view) || name === 'incomplete') continue;
      const isFloatColumn = view.format === 'float32';
      // Scalar results occupy four bytes. Write NaN bits through u32 storage: a constant
      // bitcast to non-finite f32 is invalid at WGSL shader creation, even in a guarded branch.
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-producer-${name}${suffix}`,
          operation: 'RasterProducerCompleteness',
          variant: 'suppress',
          bindings: [
            {name: 'producerOverflow', view: source, type: 'u32', access: 'read'},
            {name: 'result', view, type: 'u32', access: 'read_write'}
          ],
          invocationCount: view.length,
          body: `if (producerOverflow[producerOverflowOffset] != 0u) {
  result[resultOffset + index] = ${isFloatColumn ? '0x7fc00000u' : '0xffffffffu'};
}`
        })
      );
    }
    if (incomplete) {
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-producer-incomplete${suffix}`,
          operation: 'RasterProducerCompleteness',
          variant: sourceIndex === 0 ? 'status' : 'status-or',
          bindings: [
            {name: 'producerOverflow', view: source, type: 'u32', access: 'read'},
            {name: 'incomplete', view: incomplete, type: 'u32', access: 'read_write'}
          ],
          invocationCount: 1,
          body: `incomplete[incompleteOffset] ${sourceIndex === 0 ? '=' : '|='} select(0u, 1u, producerOverflow[producerOverflowOffset] != 0u);`
        })
      );
    }
  }
  if (incomplete && sources.length === 0) {
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-producer-incomplete`,
        operation: 'RasterProducerCompleteness',
        variant: 'status',
        bindings: [{name: 'incomplete', view: incomplete, type: 'u32', access: 'read_write'}],
        invocationCount: 1,
        body: 'incomplete[incompleteOffset] = 0u;'
      })
    );
  }
  return nodes;
}
