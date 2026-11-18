// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type BindingDeclaration} from '@luma.gl/core';
import {Kernel} from '@luma.gl/graph';
import {
  createGPUComputeCommandNode,
  getBoundedDispatchLayout,
  getViewBinding,
  getViewElementOffset,
  GraphVectorView,
  validatePackedView,
  type GraphDataView,
  type GPUCommandGraph,
  type GPUCommandNode
} from '@luma.gl/graph/gpu-core';
import {
  getGraphViewChunks,
  validateCapacity,
  validateRequirement,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import type {WGSLKernelProps} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateRasterWritableDispatches,
  type RasterWritableDispatch
} from '../raster-grid/raster-view-validation';
import {planRasterTimeSeriesCellTopology} from './raster-time-series-cell-topology';
export {RASTER_TIME_SERIES_SPLIT_CELL_RECORD} from './raster-time-series-cell-topology';

/** Scalar time-series fields, either packed or split between complete cell records. */
export type RasterTimeSeriesScalarView = GraphDataView | GraphVectorView;
/** Packed fields may accompany a matching one-chunk vector; multiple chunks must align. */
export type RasterTimeSeriesField<
  Format extends 'float32' | 'uint32',
  Rows extends RasterTimeSeriesScalarView
> = GraphDataView<Format> | (Rows extends GraphVectorView ? GraphVectorView<Format> : never);

/** One count and up to fifteen binding offsets fit four uniform vec4s without storage slots. */
const RASTER_TIME_SERIES_DISPATCH_PARAMETER_WORD_COUNT = 16;

/** Plans complete records at construction; dates, parameters and other unlisted tables stay packed. */
export function planRasterTimeSeriesChunks<Props extends {cellCount: number}>(
  id: string,
  props: Props,
  recordWidths: Readonly<Record<string, number>>
): readonly Props[] {
  for (const path of [
    'parameters',
    'accumulatorParameters',
    'slotIndices',
    'temporalModes',
    'singularValues',
    'sliceAttributes',
    'dateAxis.acquisitionTimes'
  ]) {
    const view = getField(props, path) as RasterTimeSeriesScalarView | undefined;
    if (view) validateSinglePackedView(view, [view.format], `${id} ${path}`);
  }
  const fields = Object.entries(recordWidths).flatMap(([path, width]) => {
    const view = getField(props, path) as RasterTimeSeriesScalarView | undefined;
    if (!view) return [];
    validateCapacity(id, `${path} cell record length`, width);
    validateCapacity(id, `${path} required entries`, props.cellCount * width);
    return [{path, width, view}];
  });
  const {cellCounts, chunked} = planRasterTimeSeriesCellTopology(
    id,
    props.cellCount,
    Object.fromEntries(fields.map(({path, width, view}) => [path, {view, entriesPerCell: width}]))
  );
  if (!chunked) return [props];
  return cellCounts.flatMap((cellCount, chunkIndex) => {
    if (!cellCount) return [];
    const chunkProps = {...props, cellCount};
    for (const {path, view} of fields)
      setField(chunkProps, path, getGraphViewChunks(view)[chunkIndex]);
    return [chunkProps];
  });
}

/** Packed-layout validation and writable-range separation across every field and chunk. */
export function validateRasterTimeSeriesViews(
  id: string,
  inputs: readonly RasterTimeSeriesScalarView[],
  outputs: readonly RasterTimeSeriesScalarView[],
  dispatches?: readonly RasterWritableDispatch[]
): void {
  for (const view of [...inputs, ...outputs]) {
    for (const chunk of getGraphViewChunks(view)) {
      validatePackedView(chunk, [view.format], `${id} views`);
    }
  }
  validateRasterWritableDispatches(id, inputs, outputs, dispatches);
}

/** Preflight every actual binding, including shared tables, before any node is declared. */
export function validateRasterTimeSeriesBindings<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  views: readonly RasterTimeSeriesScalarView[]
): void {
  validateGraphChunkBindingRanges(
    graph,
    id,
    Object.fromEntries(views.map((view, index) => [`view ${index}`, view]))
  );
  validateRequirement(
    id,
    RASTER_TIME_SERIES_DISPATCH_PARAMETER_WORD_COUNT * Uint32Array.BYTES_PER_ELEMENT <=
      graph.device.limits.maxUniformBufferBindingSize,
    'dispatch parameters must fit maxUniformBufferBindingSize'
  );
}

/**
 * Cell-parallel raster kernel. Counts and aligned binding prefixes are runtime uniforms, so chunk
 * size and byte offset never specialize its shader or pipeline. Each node dispatches its own
 * bounded grid. The small immutable uniform is owned by the compiled node and destroyed with it.
 * This avoids spending a ninth storage binding on parameters for the eight-binding raster fits.
 */
export function createRasterTimeSeriesKernelNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: Pick<
    WGSLKernelProps<Parameters>,
    'id' | 'operation' | 'variant' | 'bindings' | 'invocationCount' | 'declarations' | 'body'
  > & {workgroupSize: number}
): GPUCommandNode<Parameters> {
  const workgroupSize = props.workgroupSize;
  validateCapacity(
    props.id,
    'storage binding count',
    props.bindings.length,
    1,
    Math.min(
      graph.device.limits.maxStorageBuffersPerShaderStage,
      RASTER_TIME_SERIES_DISPATCH_PARAMETER_WORD_COUNT - 1
    )
  );
  const layout = getBoundedDispatchLayout(
    props.operation,
    props.invocationCount,
    workgroupSize,
    graph.device.limits.maxComputeWorkgroupsPerDimension
  );
  const word = (index: number) => `rasterDispatch.words[${Math.floor(index / 4)}u][${index % 4}u]`;
  const source = `${props.bindings
    .map(
      (binding, location) => `var<private> ${binding.name}Offset: u32;
@group(0) @binding(${location}) var<storage, ${binding.access}> ${binding.name}: array<${binding.type}>;`
    )
    .join('\n')}
struct RasterDispatchParameters { words: array<vec4<u32>, 4> }
@group(0) @binding(${props.bindings.length}) var<uniform> rasterDispatch: RasterDispatchParameters;
${props.declarations ?? ''}
@compute @workgroup_size(${workgroupSize})
fn main(@builtin(workgroup_id) workgroupId: vec3<u32>, @builtin(local_invocation_index) localInvocationIndex: u32, @builtin(num_workgroups) workgroupCount: vec3<u32>) {
  let workgroupIndex = (workgroupId.z * workgroupCount.y + workgroupId.y) * workgroupCount.x + workgroupId.x;
  if (workgroupIndex >= ${Math.floor(0xffffffff / workgroupSize) + 1}u) { return; }
  let index = workgroupIndex * ${workgroupSize}u + localInvocationIndex;
  if (index >= ${word(0)}) { return; }
  ${props.bindings.map((binding, index) => `${binding.name}Offset = ${word(index + 1)}; _ = &${binding.name};`).join('\n  ')}
  ${props.body}
}`;
  const parameterValues = new Uint32Array(RASTER_TIME_SERIES_DISPATCH_PARAMETER_WORD_COUNT);
  parameterValues[0] = props.invocationCount;
  props.bindings.forEach((binding, index) => {
    parameterValues[index + 1] = getViewElementOffset(binding.view);
  });
  return createGPUComputeCommandNode<Parameters>({
    id: props.id,
    resources: props.bindings.map(binding => ({
      buffer: binding.view,
      usage: binding.access === 'read' ? 'storage-read' : 'storage-read-write'
    })),
    workload: {
      operation: props.operation,
      ...(props.variant ? {variant: props.variant} : {}),
      commandCount: 1,
      maximumWorkgroupCount: layout.x * layout.y * layout.z,
      maximumInvocationCount: props.invocationCount
    },
    compile: ({device}) => {
      const parameters = device.createBuffer({
        id: `${props.id}-dispatch`,
        data: parameterValues,
        usage: Buffer.UNIFORM | Buffer.COPY_DST
      });
      const kernel = new Kernel(device, {
        id: props.id,
        source,
        shaderLayout: {
          bindings: [
            ...props.bindings.map(
              (binding, location): BindingDeclaration => ({
                name: binding.name,
                type: binding.access === 'read' ? 'read-only-storage' : 'storage',
                group: 0,
                location
              })
            ),
            {name: 'rasterDispatch', type: 'uniform', group: 0, location: props.bindings.length}
          ]
        }
      });
      return {
        encode: ({computePass, getBuffer}) => {
          const bindings = Object.fromEntries(
            props.bindings.map(binding => [binding.name, getViewBinding(binding.view, getBuffer)])
          );
          kernel.dispatch(computePass, {
            bindings: {...bindings, rasterDispatch: parameters},
            x: layout.x,
            y: layout.y,
            z: layout.z
          });
        },
        destroy: () => {
          kernel.destroy();
          parameters.destroy();
        }
      };
    }
  });
}

function getField(props: object, path: string): unknown {
  return path
    .split('.')
    .reduce<unknown>(
      (value, name) => (value as Record<string, unknown> | undefined)?.[name],
      props
    );
}
function setField(props: object, path: string, value: unknown): void {
  const [name, nested] = path.split('.');
  const fields = props as Record<string, unknown>;
  if (nested) fields[name] = {...(fields[name] as object), [nested]: value};
  else fields[name] = value;
}

/**
 * Splits already validated cell chunks into bounded scratch batches without copying. Shared tables
 * remain unchanged. Subviews are metadata only and their aligned ranges are contained in those of
 * the original packed chunks; the caller preflights originals and derived scratch before nodes.
 */
export function splitRasterTimeSeriesChunks<Props extends {cellCount: number}>(
  chunks: readonly Props[],
  recordWidths: Readonly<Record<string, number>>,
  maximumCellCount: number
): readonly Props[] {
  return chunks.flatMap(props => {
    if (props.cellCount <= maximumCellCount) return [props];
    const batches: Props[] = [];
    for (let firstCell = 0; firstCell < props.cellCount; firstCell += maximumCellCount) {
      const cellCount = Math.min(maximumCellCount, props.cellCount - firstCell);
      const batch = {...props, cellCount};
      for (const [path, width] of Object.entries(recordWidths)) {
        const view = getField(props, path) as GraphDataView | undefined;
        if (!view) continue;
        setField(
          batch,
          path,
          (view.buffer.graph as GPUCommandGraph<unknown>).createDataView(view.buffer, {
            format: view.format,
            length: cellCount * width,
            byteOffset: view.byteOffset + firstCell * width * view.byteStride
          })
        );
      }
      batches.push(batch);
    }
    return batches;
  });
}
