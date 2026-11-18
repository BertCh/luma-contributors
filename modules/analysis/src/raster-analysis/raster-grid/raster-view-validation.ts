// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GraphVectorView, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  getGraphViewChunks,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateDisjointWritableViews} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {getCoreDefaultBuffer} from '../../analysis-core/contracts/network-topology';

type RasterView = GraphDataView | GraphVectorView;

/** Caller-owned bindings that coexist in one dispatch per cell chunk. @internal */
export type RasterWritableDispatch = {
  inputs: readonly RasterView[];
  outputs: readonly RasterView[];
};

function getPhysicalBuffer(view: GraphDataView) {
  return view.buffer.defaultBuffer ? getCoreDefaultBuffer(view.buffer.defaultBuffer) : view.buffer;
}

/** Logical separation globally; storage binding separation only where bindings coexist. @internal */
export function validateRasterWritableDispatches(
  id: string,
  inputs: readonly RasterView[],
  outputs: readonly RasterView[],
  dispatches?: readonly RasterWritableDispatch[]
): void {
  const ranges = new Map<
    object,
    {offset: number; end: number; writable: boolean; name: string}[]
  >();
  const importedHandles = new Map<
    object,
    {handles: Set<object>; writable: boolean; name: string}
  >();
  for (const [kind, views] of [
    ['input', inputs],
    ['output', outputs]
  ] as const) {
    for (const [fieldIndex, view] of views.entries()) {
      for (const [chunkIndex, chunk] of getGraphViewChunks(view).entries()) {
        if (chunk.length === 0) continue;
        const buffer = getPhysicalBuffer(chunk);
        const imports = importedHandles.get(buffer) ?? {
          handles: new Set<object>(),
          writable: false,
          name: `${kind}[${fieldIndex}] chunk ${chunkIndex}`
        };
        imports.handles.add(chunk.buffer);
        imports.writable ||= kind === 'output';
        importedHandles.set(buffer, imports);
        const entries = ranges.get(buffer) ?? [];
        entries.push({
          offset: chunk.byteOffset,
          end: chunk.byteOffset + (chunk.length - 1) * chunk.byteStride + chunk.rowByteLength,
          writable: kind === 'output',
          name: `${kind}[${fieldIndex}] chunk ${chunkIndex}`
        });
        ranges.set(buffer, entries);
      }
    }
  }
  for (const entries of ranges.values()) {
    let readEnd = 0;
    let writeEnd = 0;
    for (const range of entries.sort((first, second) => first.offset - second.offset)) {
      validateRequirement(
        id,
        (range.writable ? Math.max(readEnd, writeEnd) : writeEnd) <= range.offset,
        `${range.name} logical byte ranges must not overlap (writable storage)`
      );
      if (range.writable) writeEnd = Math.max(writeEnd, range.end);
      else readEnd = Math.max(readEnd, range.end);
    }
  }
  for (const imports of importedHandles.values()) {
    validateRequirement(
      id,
      !imports.writable || imports.handles.size === 1,
      `${imports.name}: views for one physical buffer that participates in a write must use one imported handle`
    );
  }
  if (!dispatches) {
    validateDisjointWritableViews(
      id,
      Object.fromEntries(outputs.map((view, index) => [`output[${index}]`, view]))
    );
    // An output may become read-only in a later stage. Without actual dispatch accesses, two
    // output fields in one buffer cannot be proved safe even when their ranges are disjoint.
    const outputOwners = new Map<object, number>();
    for (const [fieldIndex, view] of outputs.entries()) {
      for (const chunk of getGraphViewChunks(view)) {
        if (chunk.length === 0) continue;
        const buffer = getPhysicalBuffer(chunk);
        const previous = outputOwners.get(buffer);
        validateRequirement(
          id,
          previous === undefined || previous === fieldIndex,
          `output[${previous}] and output[${fieldIndex}] must not share a physical buffer without explicit per-dispatch accesses`
        );
        outputOwners.set(buffer, fieldIndex);
      }
    }
  }
  for (const dispatch of dispatches ?? [{inputs, outputs}]) {
    validateDisjointWritableViews(
      id,
      Object.fromEntries(dispatch.outputs.map((view, index) => [`output[${index}]`, view]))
    );
    // A packed field is compatible with a one-chunk vector and is bound on every dispatch.
    // Different vector chunk indices execute separately, so compare only matching indices.
    for (const [outputIndex, output] of dispatch.outputs.entries()) {
      for (const [chunkIndex, outputChunk] of getGraphViewChunks(output).entries()) {
        if (output instanceof GraphVectorView && outputChunk.length === 0) continue;
        for (const [inputIndex, input] of dispatch.inputs.entries()) {
          const inputChunks = getGraphViewChunks(input);
          const boundInputs =
            input instanceof GraphVectorView && output instanceof GraphVectorView
              ? [inputChunks[chunkIndex]]
              : inputChunks;
          for (const inputChunk of boundInputs) {
            validateRequirement(
              id,
              getPhysicalBuffer(outputChunk) !== getPhysicalBuffer(inputChunk),
              `input[${inputIndex}] and output[${outputIndex}] must not share buffers in one dispatch: read-only and writable storage must not use the same physical buffer`
            );
          }
        }
      }
    }
  }
}
