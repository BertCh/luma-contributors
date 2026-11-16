// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView, GraphVectorView} from '@luma.gl/graph/gpu-core';
import {getCoreDefaultBuffer} from '../../analysis-core/network-topology';
import {
  getGraphViewChunks,
  validateRequirement,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  validateDisjointWritableViews,
  type GraphChunkDispatchViews
} from '../../analysis-core/kernels/graph-chunk-dispatch';

function getInterpolationBuffer(view: GraphDataView) {
  return view.buffer.defaultBuffer ? getCoreDefaultBuffer(view.buffer.defaultBuffer) : view.buffer;
}

/** Checks packed ports without imposing whole-buffer separation on otherwise valid output subviews. */
export function validateInterpolationPredictionViews(
  id: string,
  inputs: readonly GraphDataView[],
  outputs: readonly GraphDataView[]
): void {
  for (const view of [...inputs, ...outputs]) validateSinglePackedView(view, [view.format], id);
  validateInterpolationOutputsDisjointFromInputs(id, outputs, inputs);
}

/** The interpolation input/output separation contract applies to physical imports as well as handles. */
export function validateInterpolationOutputsDisjointFromInputs(
  id: string,
  outputs: readonly (GraphDataView | GraphVectorView | undefined)[],
  inputs: readonly (GraphDataView | GraphVectorView | undefined)[]
): void {
  const inputBuffers = new Set(
    inputs.flatMap(view => (view ? getGraphViewChunks(view).map(getInterpolationBuffer) : []))
  );
  for (const output of outputs) {
    validateRequirement(
      id,
      !output ||
        getGraphViewChunks(output).every(view => !inputBuffers.has(getInterpolationBuffer(view))),
      'outputs must not share buffers with inputs'
    );
  }
}

/** Retained outputs cannot overlap; one dispatch cannot read and write one physical storage buffer. */
export function validateInterpolationOutputPasses(
  id: string,
  outputs: Record<string, GraphDataView | undefined>,
  passes: readonly {read?: GraphChunkDispatchViews; write: GraphChunkDispatchViews}[]
): void {
  const rangesByBuffer = new Map<
    ReturnType<typeof getInterpolationBuffer>,
    {name: string; start: number; end: number}[]
  >();
  for (const [name, view] of Object.entries(outputs)) {
    if (!view || view.length === 0) continue;
    const buffer = getInterpolationBuffer(view);
    const ranges = rangesByBuffer.get(buffer) ?? [];
    ranges.push({
      name,
      start: view.byteOffset,
      end: view.byteOffset + (view.length - 1) * view.byteStride + view.rowByteLength
    });
    rangesByBuffer.set(buffer, ranges);
  }
  for (const ranges of rangesByBuffer.values()) {
    ranges.sort((first, second) => first.start - second.start);
    for (let index = 1; index < ranges.length; index++) {
      validateRequirement(
        id,
        ranges[index - 1].end <= ranges[index].start,
        `${ranges[index - 1].name} and ${ranges[index].name} logical byte ranges must not overlap`
      );
    }
  }
  for (const pass of passes) {
    validateDisjointWritableViews(id, pass.write);
    for (const [writeName, writeView] of Object.entries(pass.write)) {
      for (const [readName, readView] of Object.entries(pass.read ?? {})) {
        if (!writeView || !readView) continue;
        const readBuffers = new Set(getGraphViewChunks(readView).map(getInterpolationBuffer));
        validateRequirement(
          id,
          getGraphViewChunks(writeView).every(
            view => !readBuffers.has(getInterpolationBuffer(view))
          ),
          `${writeName} storage-write and ${readName} storage-read must not share a physical buffer in one dispatch`
        );
      }
    }
  }
}
