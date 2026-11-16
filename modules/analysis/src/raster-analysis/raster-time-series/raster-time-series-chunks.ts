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
  validateGraphOutputsDisjointFromInputs,
  validateRequirement,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  validateDisjointWritableViews,
  validateGraphChunkBindingRanges
} from '../../analysis-core/kernels/graph-chunk-dispatch';
import type {WGSLKernelProps} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {getCoreDefaultBuffer} from '../../analysis-core/network-topology';

/** Scalar time-series fields, either packed or split between complete cell records. */
export type RasterTimeSeriesScalarView = GraphDataView | GraphVectorView;
/** Retains packed field types for packed callers. All cell fields share the same view kind. */
export type RasterTimeSeriesField<
  Format extends 'float32' | 'uint32',
  Rows extends RasterTimeSeriesScalarView
> = Rows extends GraphVectorView ? GraphVectorView<Format> : GraphDataView<Format>;

/** One count and up to fifteen binding offsets fit four uniform vec4s without storage slots. */
const RASTER_TIME_SERIES_DISPATCH_PARAMETER_WORD_COUNT = 16;
/** A stable name in errors identifies a boundary through the middle of one cell's record. */
export const RASTER_TIME_SERIES_SPLIT_CELL_RECORD = 'split cell record';

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
  const template = fields.find(field => field.view instanceof GraphVectorView);
  if (!template) return [props];
  const cellCounts = getGraphViewChunks(template.view).map((chunk, chunkIndex) => {
    validateRequirement(
      id,
      chunk.length % template.width === 0,
      `${template.path} chunk ${chunkIndex} ${RASTER_TIME_SERIES_SPLIT_CELL_RECORD}: length must be a multiple of ${template.width}`
    );
    return chunk.length / template.width;
  });
  validateRequirement(
    id,
    cellCounts.reduce((total, count) => total + count, 0) === props.cellCount,
    `${template.path} chunks must hold exactly cellCount complete cell records`
  );
  for (const {path, width, view} of fields) {
    validateRequirement(
      id,
      view instanceof GraphVectorView,
      `${path} must match the cell chunk topology`
    );
    const chunks = getGraphViewChunks(view);
    validateRequirement(
      id,
      chunks.length === cellCounts.length,
      `${path} must match the cell chunk topology`
    );
    for (const [chunkIndex, chunk] of chunks.entries()) {
      validateRequirement(
        id,
        chunk.length % width === 0,
        `${path} chunk ${chunkIndex} ${RASTER_TIME_SERIES_SPLIT_CELL_RECORD}: length must be a multiple of ${width}`
      );
      validateRequirement(
        id,
        chunk.length / width === cellCounts[chunkIndex],
        `${path} chunk ${chunkIndex} must match the cell chunk topology`
      );
    }
  }
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
  outputs: readonly RasterTimeSeriesScalarView[]
): void {
  // Hazards follow graph handles: multiple imports of one physical buffer hide cross-field writes.
  // DynamicBuffer defaults are resolved by the shared analysis-core helper, as in writable ranges.
  const importedHandles = new Map<Buffer, GraphDataView['buffer']>();
  for (const view of [...inputs, ...outputs]) {
    for (const chunk of getGraphViewChunks(view)) {
      validatePackedView(chunk, [view.format], `${id} views`);
      if (chunk.buffer.defaultBuffer) {
        const physicalBuffer = getCoreDefaultBuffer(chunk.buffer.defaultBuffer);
        const previousHandle = importedHandles.get(physicalBuffer);
        validateRequirement(
          id,
          !previousHandle || previousHandle === chunk.buffer,
          'views for one physical buffer must use one imported handle'
        );
        importedHandles.set(physicalBuffer, chunk.buffer);
      }
    }
  }
  validateGraphOutputsDisjointFromInputs(id, outputs, inputs);
  validateDisjointWritableViews(
    id,
    Object.fromEntries(outputs.map((view, index) => [`output ${index}`, view]))
  );
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
