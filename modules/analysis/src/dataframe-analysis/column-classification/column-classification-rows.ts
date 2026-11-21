// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GraphVectorView, type GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  getGraphViewChunks,
  isGraphVectorView,
  validateRequirement,
  validateCapacity,
  validateGPUStorageBindingByteLength,
  validateGraphViewTopology,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';

import {
  validateDisjointWritableViews,
  validateGraphChunkBindingRanges
} from '../../analysis-core/kernels/graph-chunk-dispatch';

import {validateGraphChunkGroupKernelSupport} from '../../analysis-core/kernels/graph-chunk-groups';

/** Every compiled classification kernel fits WebGPU's eight storage bindings. */
const CLASSIFICATION_MAXIMUM_STORAGE_BINDING_COUNT = 8;

type ClassificationFormat = 'float32' | 'uint32' | 'sint32';

/** Source-aligned packed rows or ordered packed chunks. */
export type ClassificationRows<Format extends ClassificationFormat> =
  | GraphDataView<Format>
  | GraphVectorView<Format>;

/**
 * Keeps packed public output types while inferring vector outputs from vector inputs.
 * A never-inferred source retains the packed default instead of erasing optional output types.
 */
export type ClassificationRowView<
  Format extends ClassificationFormat,
  Rows extends ClassificationRows<ClassificationFormat>
> = [Rows] extends [never]
  ? GraphDataView<Format>
  : Rows extends GraphDataView
    ? GraphDataView<Format>
    : GraphVectorView<Format>;

/** Validates each binding and, for chunked rows, the exact source chunk boundaries. */
export function validateClassificationRows(
  id: string,
  name: string,
  view: ClassificationRows<ClassificationFormat>,
  formats: readonly ClassificationFormat[],
  source?: ClassificationRows<ClassificationFormat>
): void {
  const chunks = getGraphViewChunks(view);
  for (const chunk of chunks) {
    validateSinglePackedView(chunk, formats, `${id} ${name}`);
    validateRequirement(
      id,
      chunk.format === view.format,
      `${name} chunk format ${chunk.format} must equal declared format ${view.format}`
    );
  }
  validateRequirement(
    id,
    chunks.reduce((total, chunk) => total + chunk.length, 0) === view.length,
    `${name} chunk row total must equal declared length ${view.length}`
  );
  if (source && (isGraphVectorView(source) || isGraphVectorView(view)))
    validateGraphViewTopology(id, name, source, view);
}

/** Keeps the existing packed-output capacity contract; vectors must match every source chunk. */
export function validateClassificationOutput(
  id: string,
  name: string,
  view: ClassificationRows<ClassificationFormat>,
  format: ClassificationFormat,
  source: ClassificationRows<ClassificationFormat>
): void {
  validateClassificationRows(id, name, view, [format]);
  if (isGraphVectorView(source) || isGraphVectorView(view)) {
    validateGraphViewTopology(id, name, source, view);
  } else {
    validateRequirement(
      id,
      view.length >= source.length,
      `${name} must hold at least one value per source row`
    );
  }
}

/** Co-bound outputs have disjoint aligned ranges; sequential chunks have disjoint logical ranges. */
export function validateClassificationOutputs(
  id: string,
  outputs: readonly (ClassificationRows<ClassificationFormat> | undefined)[]
): void {
  validateDisjointWritableViews(
    id,
    Object.fromEntries(outputs.map((view, index) => [`output${index}`, view]))
  );
}

/** Preflights every supplied binding, including its alignment prefix, before graph mutation. */
export function validateClassificationBindings<Parameters>(
  id: string,
  graph: GPUCommandGraph<Parameters>,
  views: Readonly<Record<string, ClassificationRows<ClassificationFormat> | undefined>>
): void {
  validateGraphChunkBindingRanges(graph, id, views);
  for (const [name, view] of Object.entries(views)) {
    for (const chunk of view ? getGraphViewChunks(view) : []) {
      validateCapacity(
        id,
        `${name}.buffer.byteLength (device.limits.maxBufferSize)`,
        chunk.buffer.byteLength,
        0,
        graph.device.limits.maxBufferSize
      );
    }
  }
  validateGraphChunkGroupKernelSupport(graph, id, CLASSIFICATION_MAXIMUM_STORAGE_BINDING_COUNT);
}

/** Checks all derived scalar storage allocations before the contributor declares scratch. */
export function validateClassificationScratch<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  wordCounts: Readonly<Record<string, number>>
): void {
  for (const [name, words] of Object.entries(wordCounts)) {
    validateCapacity(
      id,
      `${name} byteLength (device.limits.maxBufferSize)`,
      Math.max(words, 1) * 4,
      0,
      graph.device.limits.maxBufferSize
    );
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      name,
      Math.max(words, 1) * 4,
      'reduce the row count, capacity or chunk count'
    );
  }
}

/** Quantile reductions coexist in finish; mask chunks write in later, separate dispatches. */
export function validateQuantileWritableViews(
  id: string,
  output: {
    quantiles: GraphDataView<'float32'>;
    validCount: GraphDataView<'uint32'>;
    filterBounds?: GraphDataView<'float32'>;
    outputMask?: ClassificationRows<'uint32'>;
  }
): void {
  const atDispatch = (name: string, chunks: readonly GraphDataView[], firstDispatch: number) => {
    if (chunks.length === 0) return undefined;
    const first = chunks[0];
    const empty = new GraphDataView(first.buffer, {
      format: first.format,
      length: 0,
      byteOffset: first.byteOffset,
      byteStride: first.byteStride,
      rowByteLength: first.rowByteLength
    });
    const data = [...Array.from({length: firstDispatch}, () => empty), ...chunks];
    const length = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
    // A vector groups these physical views by dispatch index for the shared byte-range validator.
    return new GraphVectorView({
      id: `${id}-${name}-dispatches`,
      name,
      format: first.format,
      length,
      valueLength: length,
      stride: 1,
      byteStride: 4,
      rowByteLength: 4,
      data
    });
  };
  validateDisjointWritableViews(id, {
    quantiles: atDispatch('quantiles', [output.quantiles], 0),
    validCount: atDispatch('validCount', [output.validCount], 0),
    filterBounds: output.filterBounds
      ? atDispatch('filterBounds', [output.filterBounds], 0)
      : undefined,
    outputMask: output.outputMask
      ? atDispatch('outputMask', getGraphViewChunks(output.outputMask), 1)
      : undefined
  });
}
