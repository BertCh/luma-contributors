// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {VertexFormat} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  type WGSLKernelProps
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateGraphViewsBelongToGraph,
  validateGraphOutputsDisjointFromInputs,
  validateSinglePackedView,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getOrCreateView} from './recipe-utils';

/** Exact output sizes keep recipe indexing independent of caller buffer capacity. @internal */
export function getRecipeView<Format extends VertexFormat, Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  format: Format,
  length: number,
  provided?: GraphDataView<Format>
): GraphDataView<Format> {
  if (provided) {
    validateSinglePackedView(provided, [format], id);
    // The provided view must have the exact row count promised by the recipe output.
    validateViewLength(id, 'provided output', provided, length);
    validateGraphViewsBelongToGraph(id, graph, [provided]);
  }
  return getOrCreateView(graph, id, format, length, provided);
}
/** Recipe arithmetic follows the same ownership and packed-layout contract as contributors. @internal */
export function createRecipeKernelNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: WGSLKernelProps<Parameters>
) {
  validateGraphViewsBelongToGraph(
    props.id,
    graph,
    props.bindings.map(binding => binding.view)
  );
  for (const binding of props.bindings)
    validateSinglePackedView(binding.view, [binding.view.format], props.id);
  validateGraphOutputsDisjointFromInputs(
    props.id,
    props.bindings.filter(binding => binding.access !== 'read').map(binding => binding.view),
    props.bindings.filter(binding => binding.access === 'read').map(binding => binding.view)
  );
  return createWGSLKernelNode(graph, props);
}

/** Validate the whole recipe boundary, including buffers used by nonadjacent stages. @internal */
export function validateRecipeBoundary<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  inputs: readonly (GraphDataView | undefined)[],
  outputs: readonly (GraphDataView | undefined)[]
): void {
  validateGraphViewsBelongToGraph(id, graph, [...inputs, ...outputs]);
  validateGraphOutputsDisjointFromInputs(id, outputs, inputs);
  const buffers = outputs.filter(view => view !== undefined).map(view => view.buffer);
  // Independent named outputs must not share a buffer, even through disjoint views.
  validateRequirement(
    id,
    new Set(buffers).size === buffers.length,
    'outputs must not share a buffer'
  );
}
