// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {VertexFormat} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {getCoreDefaultBuffer} from '../../analysis-core/contracts/network-topology';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateRequirement,
  validateSinglePackedView,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {validateLandscapeViews} from './landscape-validation';

type Views = Readonly<Record<string, GraphDataView | undefined>>;
type OutputLayouts = Readonly<
  Record<
    string,
    {
      view: GraphDataView | undefined;
      format: GraphDataView['format'];
      length: number;
    }
  >
>;

/**
 * Check caller-owned output views exactly as `getRecipeView` will, with the same view IDs and
 * messages, so a recipe can reject them before it adds its first node. Entries are
 * `[viewId, format, length, provided]`; absent views are skipped. @internal
 */
export function validateProvidedRecipeViews<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  entries: readonly (readonly [string, VertexFormat, number, GraphDataView | undefined])[]
): void {
  for (const [viewId, format, length, view] of entries) {
    if (!view) continue;
    validateSinglePackedView(view, [format], viewId);
    validateViewLength(viewId, 'provided output', view, length);
    validateGraphViewsBelongToGraph(viewId, graph, [view]);
  }
}

/** Preflight the isovist/counterfactual boundary without allocating resources or adding nodes. @internal */
export function validateUrbanLandscapeRecipeViews<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  inputs: Views,
  layouts: OutputLayouts,
  dispatches: readonly {reads: Views; writes: Views}[]
): void {
  for (const [name, layout] of Object.entries(layouts)) {
    if (!layout.view) continue;
    validateSinglePackedView(layout.view, [layout.format], `${id} output.${name}`);
    validateViewLength(id, `output.${name}`, layout.view, layout.length);
  }
  const outputs: Views = Object.fromEntries(
    Object.entries(layouts).map(([name, layout]) => [name, layout.view])
  );
  validateGraphViewsBelongToGraph(id, graph, [...Object.values(inputs), ...Object.values(outputs)]);
  validateGraphChunkBindingRanges(
    graph,
    id,
    Object.fromEntries([
      ...Object.entries(inputs).map(([name, view]) => [`input.${name}`, view]),
      ...Object.entries(outputs).map(([name, view]) => [`output.${name}`, view])
    ])
  );
  validateLandscapeViews(id, inputs, outputs, dispatches);

  const physical = (view: GraphDataView) =>
    view.buffer.defaultBuffer ? getCoreDefaultBuffer(view.buffer.defaultBuffer) : view.buffer;
  // luma.gl requires one graph handle for each physical buffer if any handle is writable,
  // even when writes occur in different nodes and their logical ranges do not overlap.
  for (const [outputName, output] of Object.entries(outputs)) {
    if (!output) continue;
    for (const [name, other] of [...Object.entries(inputs), ...Object.entries(outputs)]) {
      if (!other || physical(output) !== physical(other)) continue;
      validateRequirement(
        id,
        output.buffer === other.buffer,
        `${outputName} and ${name} must use one graph buffer handle when sharing a physical buffer`
      );
    }
  }
  // Recipe kernels forbid an entire physical buffer from being both read-only and writable
  // within one dispatch, including disjoint aligned ranges of the same handle.
  for (const dispatch of dispatches) {
    for (const [outputName, output] of Object.entries(dispatch.writes)) {
      for (const [inputName, input] of Object.entries(dispatch.reads)) {
        validateGraphOutputsDisjointFromInputs(
          `${id} ${outputName} and ${inputName}`,
          [output],
          [input]
        );
      }
    }
  }
}
