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
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {
  createWGSLKernelNode,
  type WGSLKernelProps,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';

import {createRasterTimeSeriesKernelNode} from './raster-time-series-chunks';
import {planRasterTimeSeriesCellTopology} from './raster-time-series-cell-topology';
import {
  validateRasterWritableDispatches,
  type RasterWritableDispatch
} from '../raster-grid/raster-view-validation';
export {
  validateRasterWritableDispatches,
  type RasterWritableDispatch
} from '../raster-grid/raster-view-validation';

/** The chunk kernel binds one count and fifteen offsets in four uniform vec4s. */
const RASTER_TIME_SERIES_CHUNK_UNIFORM_BYTE_LENGTH = 64;

export type RasterTimeSeriesScalarView = GraphDataView | GraphVectorView;
export type RasterTimeSeriesView<
  Format extends 'float32' | 'uint32' | 'sint32',
  Rows extends RasterTimeSeriesScalarView = RasterTimeSeriesScalarView
> = GraphDataView<Format> | (Rows extends GraphVectorView ? GraphVectorView<Format> : never);
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
  const {cellCounts: cells, chunked: vectorMode} = planRasterTimeSeriesCellTopology(
    id,
    cellCount,
    fields
  );
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
  // This lowering binds all listed fields in one statistic's dispatch per cell chunk.
  // Multi-statistic routes supply their narrower binding lists explicitly.
  validateRasterWritableDispatches(id, inputs, outputs, dispatches ?? [{inputs, outputs}]);
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
