// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GraphVectorView,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  getGraphViewChunks,
  validateCapacity,
  validateCommandNodeBudget,
  validateGraphViewsBelongToGraph,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getCoreDefaultBuffer} from '../../analysis-core/network-topology';
import {
  validateDisjointWritableViews,
  validateGraphChunkBindingRanges
} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {
  createWGSLKernelNode,
  type WGSLKernelProps,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';

import {
  createRasterTimeSeriesKernelNode,
  RASTER_TIME_SERIES_SPLIT_CELL_RECORD
} from './raster-time-series-chunks';

/** The chunk kernel binds one count and fifteen offsets in four uniform vec4s. */
const RASTER_TIME_SERIES_CHUNK_UNIFORM_BYTE_LENGTH = 64;

export type RasterTimeSeriesScalarView = GraphDataView | GraphVectorView;
export type RasterTimeSeriesView<
  Format extends 'float32' | 'uint32' | 'sint32',
  Rows extends RasterTimeSeriesScalarView = RasterTimeSeriesScalarView
> = Rows extends GraphVectorView ? GraphVectorView<Format> : GraphDataView<Format>;
export type RasterTimeSeriesBinding = Omit<WGSLKernelBinding, 'view'> & {
  view: RasterTimeSeriesScalarView;
};
export type RasterTimeSeriesField = {
  view: RasterTimeSeriesScalarView | undefined;
  entriesPerCell: number;
};
export type RasterTimeSeriesChunkPlan = {
  readonly cellCount: number;
  readonly chunked: boolean;
  readonly chunks: readonly {
    cellCount: number;
    views: ReadonlyMap<RasterTimeSeriesScalarView, GraphDataView>;
  }[];
  readonly estimatedCommandNodeCount: number;
};

/** Plans complete cell records, preserving legacy packed over-allocation. No resources are allocated. */
export function planRasterTimeSeriesChunks(
  id: string,
  cellCount: number,
  fields: Readonly<Record<string, RasterTimeSeriesField>>
): RasterTimeSeriesChunkPlan {
  const present = Object.entries(fields).filter(
    (entry): entry is [string, RasterTimeSeriesField & {view: RasterTimeSeriesScalarView}] =>
      entry[1].view !== undefined
  );
  const template = present[0][1];
  const templateChunks = getGraphViewChunks(template.view);
  const vectorMode = present.some(([, field]) => field.view instanceof GraphVectorView);
  const cells = templateChunks.map(chunk => chunk.length / template.entriesPerCell);
  if (!vectorMode) cells[0] = cellCount;
  for (const [name, field] of present) {
    validateCapacity(id, `${name} entriesPerCell`, field.entriesPerCell);
    const chunks = getGraphViewChunks(field.view);
    validateRequirement(
      id,
      chunks.length === cells.length,
      `${name} must match the cell chunk topology`
    );
    for (const [chunkIndex, chunk] of chunks.entries()) {
      validatePackedView(chunk, [field.view.format], `${id} ${name} chunk ${chunkIndex}`);
      validateRequirement(
        id,
        !vectorMode ||
          (chunk.length % field.entriesPerCell === 0 &&
            chunk.length / field.entriesPerCell === cells[chunkIndex]),
        `${name} chunk ${chunkIndex} ${RASTER_TIME_SERIES_SPLIT_CELL_RECORD}: must contain complete cell records and match the cell chunk topology`
      );
    }
    if (vectorMode) {
      validateRequirement(
        id,
        chunks.reduce((total, chunk) => total + chunk.length, 0) === field.view.length &&
          field.view.length === cellCount * field.entriesPerCell,
        `${name} chunk total must equal cellCount times entriesPerCell`
      );
    }
  }
  const chunks = cells.map((count, chunkIndex) => ({
    cellCount: count,
    views: new Map(
      present.map(([, field]) => [field.view, getGraphViewChunks(field.view)[chunkIndex]])
    )
  }));
  const estimatedCommandNodeCount = chunks.filter(chunk => chunk.cellCount > 0).length;
  validateCommandNodeBudget(id, estimatedCommandNodeCount);
  return {cellCount, chunked: vectorMode, chunks, estimatedCommandNodeCount};
}

/** Construction checks shared tables, every chunk, and writable aligned/logical separation. */
export function validateRasterTimeSeriesViews(
  id: string,
  inputs: readonly RasterTimeSeriesScalarView[],
  outputs: readonly RasterTimeSeriesScalarView[],
  plan: RasterTimeSeriesChunkPlan,
  dispatches?: readonly RasterWritableDispatch[]
): void {
  const cellViews = plan.chunks[0]?.views;
  for (const view of [...inputs, ...outputs]) {
    if (!cellViews?.has(view)) {
      validateRequirement(
        id,
        !(view instanceof GraphVectorView),
        'shared tables must be one packed view'
      );
    }
    for (const chunk of getGraphViewChunks(view)) validatePackedView(chunk, [view.format], id);
  }
  validateRasterWritableDispatches(id, inputs, outputs, dispatches);
}

/** Caller-owned bindings that coexist in one statistic's dispatch (one dispatch per cell chunk). */
export type RasterWritableDispatch = {
  inputs: readonly RasterTimeSeriesScalarView[];
  outputs: readonly RasterTimeSeriesScalarView[];
};

/** Validate logical writes globally, then aligned binding ranges only where bindings coexist. */
export function validateRasterWritableDispatches(
  id: string,
  inputs: readonly RasterTimeSeriesScalarView[],
  outputs: readonly RasterTimeSeriesScalarView[],
  dispatches: readonly RasterWritableDispatch[] = [{inputs, outputs}]
): void {
  // Separate statistics and separate cell chunks may share an alignment prefix, but never a
  // logical writable byte. Read-only inputs may alias each other. Sort once per physical buffer.
  const ranges = new Map<
    object,
    {offset: number; end: number; writable: boolean; name: string}[]
  >();
  const importedHandles = new Map<object, object>();
  for (const [kind, views] of [
    ['input', inputs],
    ['output', outputs]
  ] as const) {
    for (const [fieldIndex, view] of views.entries()) {
      for (const [chunkIndex, chunk] of getGraphViewChunks(view).entries()) {
        if (chunk.length === 0) continue;
        const buffer = chunk.buffer.defaultBuffer
          ? getCoreDefaultBuffer(chunk.buffer.defaultBuffer)
          : chunk.buffer;
        const importedHandle = importedHandles.get(buffer);
        validateRequirement(
          id,
          !importedHandle || importedHandle === chunk.buffer,
          'each physical buffer must be imported once and all views must share that graph buffer handle'
        );
        importedHandles.set(buffer, chunk.buffer);
        const entries = ranges.get(buffer) ?? [];
        entries.push({
          offset: chunk.byteOffset,
          end: chunk.byteOffset + (chunk.length - 1) * chunk.byteStride + chunk.rowByteLength,
          writable: kind === 'output',
          name: `${kind}${fieldIndex} chunk ${chunkIndex}`
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
        `${range.name} logical writable byte ranges must not overlap`
      );
      if (range.writable) writeEnd = Math.max(writeEnd, range.end);
      else readEnd = Math.max(readEnd, range.end);
    }
  }
  for (const dispatch of dispatches) {
    const writable = Object.fromEntries(
      dispatch.outputs.map((view, index) => [`output${index}`, view])
    );
    validateDisjointWritableViews(id, writable);
    // Compare only co-bound input/output chunks. Read-only chunks may reuse the same bytes
    // in different dispatches; treating a whole input vector as writable would reject that.
    for (const output of dispatch.outputs) {
      for (const [chunkIndex, outputChunk] of getGraphViewChunks(output).entries()) {
        if (output instanceof GraphVectorView && outputChunk.length === 0) continue;
        for (const input of dispatch.inputs) {
          const inputChunks = getGraphViewChunks(input);
          const boundInputs =
            input instanceof GraphVectorView && output instanceof GraphVectorView
              ? [inputChunks[chunkIndex]]
              : inputChunks;
          for (const inputChunk of boundInputs) {
            const physicalBuffer = (view: GraphDataView) =>
              view.buffer.defaultBuffer
                ? getCoreDefaultBuffer(view.buffer.defaultBuffer)
                : view.buffer;
            validateRequirement(
              id,
              physicalBuffer(outputChunk) !== physicalBuffer(inputChunk),
              'input and output in one dispatch must not use the same physical buffer (read-only and writable storage)'
            );
          }
        }
      }
    }
  }
}

/** Checks every actual aligned binding before parameters, scratch or nodes are declared. */
export function validateRasterTimeSeriesGraph<Parameters>(
  id: string,
  graph: GPUCommandGraph<Parameters>,
  inputs: readonly RasterTimeSeriesScalarView[],
  outputs: readonly RasterTimeSeriesScalarView[]
): void {
  validateGraphViewsBelongToGraph(id, graph, [...inputs, ...outputs]);
  const views = [...inputs, ...outputs];
  validateGraphChunkBindingRanges(
    graph,
    id,
    Object.fromEntries(
      views.flatMap((view, fieldIndex) =>
        getGraphViewChunks(view).flatMap((chunk, chunkIndex) =>
          chunk.length > 0 ? [[`view ${fieldIndex} chunk ${chunkIndex}`, chunk]] : []
        )
      )
    )
  );
  if (views.some(view => view instanceof GraphVectorView)) {
    validateCapacity(
      id,
      'chunk dispatch uniform byteLength (maxUniformBufferBindingSize)',
      RASTER_TIME_SERIES_CHUNK_UNIFORM_BYTE_LENGTH,
      1,
      graph.device.limits.maxUniformBufferBindingSize
    );
    validateCapacity(
      id,
      'chunk dispatch uniform byteLength (maxBufferSize)',
      RASTER_TIME_SERIES_CHUNK_UNIFORM_BYTE_LENGTH,
      1,
      graph.device.limits.maxBufferSize
    );
  }
}

/** Builds a scalar storage binding; cell-record fields are resolved by the dispatch plan. */
export function rasterTimeSeriesBinding(
  name: string,
  view: RasterTimeSeriesScalarView,
  write = false
): RasterTimeSeriesBinding {
  return {
    name,
    view,
    type: view.format.startsWith('uint32')
      ? 'u32'
      : view.format.startsWith('sint32')
        ? 'i32'
        : 'f32',
    access: write ? 'read_write' : 'read'
  };
}

/** Uses the shared cell kernel for chunked calls and preserves the packed contributor route. */
export function createRasterTimeSeriesNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  plan: RasterTimeSeriesChunkPlan,
  props: Pick<
    WGSLKernelProps<Parameters>,
    'id' | 'operation' | 'variant' | 'invocationCount' | 'workgroupSize' | 'declarations' | 'body'
  > & {
    bindings: readonly RasterTimeSeriesBinding[];
  }
): GPUCommandNode<Parameters>[] {
  if (!plan.chunked) return [createWGSLKernelNode(graph, props as WGSLKernelProps<Parameters>)];
  const invocationsPerCell = props.invocationCount / plan.cellCount;
  return plan.chunks.flatMap((chunk, chunkIndex) =>
    chunk.cellCount === 0
      ? []
      : [
          createRasterTimeSeriesKernelNode(graph, {
            ...props,
            id: `${props.id}-chunk-${chunkIndex}`,
            workgroupSize: props.workgroupSize ?? 64,
            invocationCount: chunk.cellCount * invocationsPerCell,
            bindings: props.bindings.map(binding => ({
              ...binding,
              view: chunk.views.get(binding.view) ?? (binding.view as GraphDataView)
            }))
          })
        ]
  );
}
