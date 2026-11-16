// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Binding} from '@luma.gl/core';
import {Kernel} from '@luma.gl/graph';
import {
  createGPUComputeCommandNode,
  createTransientView,
  getBoundedDispatchLayout,
  getViewBinding,
  getViewElementOffset,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView,
  GraphVectorView
} from '@luma.gl/graph/gpu-core';
import {
  getGraphViewChunks,
  validateCapacity,
  validateGPUStorageBindingByteLength,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  validateDisjointWritableViews,
  validateGraphChunkBindingRanges
} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {getCoreDefaultBuffer} from '../../analysis-core/network-topology';
import type {WGSLKernelProps} from '../../analysis-core/kernels/wgsl-kernel-nodes';

/** Sixteen uniform words hold the point count, at most eight binding offsets and row bases. */
const DENSITY_DISPATCH_WORD_COUNT = 16;
/** Discard padded workgroups before multiplication can wrap a uint32 invocation index. */
const DENSITY_MAXIMUM_INVOCATION_INDEX = 0xffffffff;

/** Published GPUSort uses a signed 31-bit logical index plus the final endpoint. @internal */
export const DENSITY_MAXIMUM_SORT_ROW_COUNT = 0x80000000;
/** Grid dimensions and slice indices are emitted as uint32 WGSL literals. @internal */
export const DENSITY_MAXIMUM_INDEX_DIMENSION = 0xffffffff;

type DensityOutputView = GraphDataView<'float32'> | GraphDataView<'uint32'>;

/** Shared caller allocations are staged and published one field per dispatch. @internal */
export function validateDensityWritableViews(
  id: string,
  outputs: Readonly<Record<string, GraphDataView | undefined>>
): void {
  const chunks = Object.values(outputs).flatMap(view =>
    view
      ? [
          (view.buffer.graph as GPUCommandGraph).createDataView(view.buffer, {
            format: 'uint32',
            length: (view.length * view.rowByteLength) / 4,
            byteOffset: view.byteOffset
          })
        ]
      : []
  );
  const length = chunks.reduce((total, chunk) => total + chunk.length, 0);
  validateDisjointWritableViews(id, {
    'output fields': new GraphVectorView({
      id: `${id}-output-fields`,
      name: 'output fields',
      format: 'uint32',
      length,
      valueLength: length,
      stride: 1,
      byteStride: 4,
      rowByteLength: 4,
      data: chunks
    })
  });
}

/**
 * WebGPU forbids mixing read-only and writable storage usage of one physical buffer in a
 * dispatch, even through disjoint bindings. Give fields sharing an allocation separate working
 * buffers, then publish their bits in dispatches that only write the caller's allocation.
 * The caller's packed views and formats stay unchanged.
 * All caller ranges and other derived scratch must be preflighted before calling this. @internal
 */
export function stageDensityOutputViews<
  Parameters,
  Views extends Record<string, DensityOutputView | undefined>
>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  operation: string,
  outputs: Views,
  writtenLengths: Readonly<Record<string, number>> = {}
): {views: Views; publishNodes: GPUCommandNode<Parameters>[]} {
  const getBuffer = (view: DensityOutputView) =>
    view.buffer.defaultBuffer ? getCoreDefaultBuffer(view.buffer.defaultBuffer) : view.buffer;
  const entries = Object.entries(outputs);
  const shared = entries.flatMap(([name, view]) =>
    view &&
    entries.some(
      ([otherName, other]) => name !== otherName && other && getBuffer(view) === getBuffer(other)
    )
      ? [{name, view, length: writtenLengths[name] ?? view.length}]
      : []
  );
  if (shared.length > 0) {
    // Packed contributors do not otherwise use dispatch uniforms. Check the publication metadata
    // here, before even the first working buffer is declared.
    validateDensityViewBindings(graph, id, []);
  }
  validateDensityScratchBindings(
    graph,
    id,
    shared.map(({name, length}) => [`output.${name} staging scratch`, length * 4] as const)
  );
  const views: Record<string, DensityOutputView | undefined> = {...outputs};
  const publishNodes: GPUCommandNode<Parameters>[] = [];
  for (const {name, view: output, length} of shared) {
    const source = createTransientView(graph, `${id}-${name}-scratch`, output.format, length);
    views[name] = source;
    publishNodes.push(
      createDensityPointPass(graph, {
        id: `${id}-publish-${name}`,
        operation,
        variant: 'publish-output',
        invocationCount: length,
        bindings: [
          {name: 'sourceWords', view: source, type: 'u32', access: 'read'},
          {name: 'outputWords', view: output, type: 'u32', access: 'read_write'}
        ],
        body: 'outputWords[outputWordsOffset + index] = sourceWords[sourceWordsOffset + index];'
      })
    );
  }
  return {views: views as Views, publishNodes};
}

/** Checks packed, offset-zero transient bindings before any allocation. @internal */
export function validateDensityScratchBindings<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  scratch: readonly (readonly [string, number])[]
): void {
  for (const [name, byteLength] of scratch) {
    validateGPUStorageBindingByteLength(graph.device, id, name, byteLength, `reduce ${name}`);
    validateCapacity(id, `${name} byteLength`, byteLength, 0, graph.device.limits.maxBufferSize);
  }
}

/**
 * The published grid index and stable cell-row sort use uint32 row arrays and (cells + 1)
 * offset/count/cursor arrays. Radix histograms and scan/reduction summaries are no larger than
 * these: radix runs only above 256 rows and has at most 16 bins per 256-row workgroup.
 * Extra wider domain scratch must be listed separately by the contributor. @internal
 */
export function validateDensityIndexedScratch<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  rowCount: number,
  cellCount: number
): void {
  validateDensityScratchBindings(graph, id, [
    ['index row and sort scratch', Math.max(rowCount, 1) * 4],
    ['index offset and scan scratch', (cellCount + 1) * 4]
  ]);
}

/** Checks the actual aligned ranges before density allocates any graph resources. @internal */
export function validateDensityViewBindings<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  views: readonly (readonly [string, GraphDataView | GraphVectorView | undefined])[],
  options: {skipEmptyChunks?: boolean; dispatchMetadata?: boolean} = {}
): void {
  if (options.dispatchMetadata !== false) {
    validateRequirement(
      id,
      DENSITY_DISPATCH_WORD_COUNT * Uint32Array.BYTES_PER_ELEMENT <=
        graph.device.limits.maxUniformBufferBindingSize,
      'density dispatch metadata must fit maxUniformBufferBindingSize'
    );
    validateCapacity(
      id,
      'density dispatch metadata byteLength',
      DENSITY_DISPATCH_WORD_COUNT * Uint32Array.BYTES_PER_ELEMENT,
      0,
      graph.device.limits.maxBufferSize
    );
  }
  for (const [name, view] of views) {
    // Row dispatches skip empty chunks, including zero-length subviews at buffer end.
    // Actual kernel bindings retain the default check, including their empty backing row.
    const chunks = (view ? getGraphViewChunks(view) : []).filter(
      chunk => !options.skipEmptyChunks || chunk.length > 0
    );
    validateGraphChunkBindingRanges(
      graph,
      id,
      Object.fromEntries(chunks.map((chunk, index) => [`${name} chunk ${index}`, chunk]))
    );
    for (const chunk of chunks) {
      validateCapacity(
        id,
        `${name} buffer byteLength`,
        chunk.buffer.byteLength,
        0,
        graph.device.limits.maxBufferSize
      );
    }
  }
}

/**
 * Density point passes use runtime uniforms for row bases, binding prefixes and row counts.
 * Public Kernel and command nodes supply dispatch/encoding; no per-chunk shader specialization.
 * The uniform is owned by the compiled node and consumes no storage binding. @internal
 */
export function createDensityPointPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: WGSLKernelProps<Parameters> & {runtimeConstants?: Readonly<Record<string, number>>}
): GPUCommandNode<Parameters> {
  const workgroupSize = props.workgroupSize ?? 64;
  const runtimeConstants = Object.entries(props.runtimeConstants ?? {});
  validateRequirement(
    props.id,
    props.bindings.length + runtimeConstants.length + 1 <= DENSITY_DISPATCH_WORD_COUNT,
    'density dispatch metadata must fit DENSITY_DISPATCH_WORD_COUNT'
  );
  validateRequirement(
    props.id,
    props.bindings.length <= graph.device.limits.maxStorageBuffersPerShaderStage,
    'density storage bindings must fit maxStorageBuffersPerShaderStage'
  );
  validateDensityViewBindings(
    graph,
    props.id,
    props.bindings.map(binding => [binding.name, binding.view])
  );
  const values = new Uint32Array(DENSITY_DISPATCH_WORD_COUNT);
  values[0] = props.invocationCount;
  const expressions = new Map<string, string>([['INVOCATION_COUNT', 'densityDispatch.words[0].x']]);
  const expression = (slot: number) =>
    `densityDispatch.words[${Math.floor(slot / 4)}][${slot % 4}]`;
  for (const [index, binding] of props.bindings.entries()) {
    values[index + 1] = getViewElementOffset(binding.view);
    expressions.set(`${binding.name}Offset`, expression(index + 1));
  }
  for (const [index, [name, value]] of runtimeConstants.entries()) {
    const slot = props.bindings.length + 1 + index;
    values[slot] = value;
    expressions.set(name, expression(slot));
  }
  const substitute = (source: string) =>
    source.replace(/\b[A-Za-z_][A-Za-z_0-9]*\b/g, name => expressions.get(name) ?? name);
  const source = `struct DensityDispatch { words: array<vec4u, 4> };
@group(0) @binding(${props.bindings.length}) var<uniform> densityDispatch: DensityDispatch;
${props.bindings.map((binding, index) => `@group(0) @binding(${index}) var<storage, ${binding.access}> ${binding.name}: array<${binding.type}>;`).join('\n')}
${substitute(props.declarations ?? '')}
@compute @workgroup_size(${workgroupSize})
fn main(@builtin(workgroup_id) workgroupId: vec3u,
        @builtin(num_workgroups) workgroupCount: vec3u,
        @builtin(local_invocation_index) localInvocationIndex: u32) {
  ${props.bindings.map(binding => `_ = &${binding.name};`).join('\n  ')}
  let workgroupIndex = (workgroupId.z * workgroupCount.y + workgroupId.y) * workgroupCount.x + workgroupId.x;
  if (workgroupIndex > ${Math.floor(DENSITY_MAXIMUM_INVOCATION_INDEX / workgroupSize)}u) { return; }
  let index = workgroupIndex * ${workgroupSize}u + localInvocationIndex;
  ${props.guardIndex === false ? '' : 'if (index >= densityDispatch.words[0].x) { return; }'}
  ${substitute(props.body)}
}`;
  const dispatch = getBoundedDispatchLayout(
    props.id,
    Math.max(1, props.invocationCount),
    workgroupSize,
    graph.device.limits.maxComputeWorkgroupsPerDimension
  );
  return createGPUComputeCommandNode<Parameters>({
    id: props.id,
    condition: props.condition,
    resources: [
      ...props.bindings.map(binding => ({
        buffer: binding.view,
        usage:
          binding.access === 'read' ? ('storage-read' as const) : ('storage-read-write' as const)
      })),
      ...(props.extraResources ?? [])
    ],
    workload: {
      operation: props.operation,
      ...(props.variant ? {variant: props.variant} : {}),
      commandCount: 1,
      maximumInvocationCount: props.invocationCount,
      maximumWorkgroupCount: dispatch.x * dispatch.y * dispatch.z
    },
    compile: ({device}) => {
      const metadata = device.createBuffer({
        id: `${props.id}-dispatch`,
        data: values,
        usage: Buffer.UNIFORM | Buffer.COPY_DST
      });
      const kernel = new Kernel(device, {
        id: props.id,
        source,
        constants: props.constants,
        shaderLayout: {
          bindings: [
            ...props.bindings.map((binding, location) => ({
              name: binding.name,
              group: 0,
              location,
              type:
                binding.access === 'read' ? ('read-only-storage' as const) : ('storage' as const)
            })),
            {name: 'densityDispatch', group: 0, location: props.bindings.length, type: 'uniform'}
          ]
        }
      });
      return {
        encode: ({computePass, getBuffer}) => {
          const bindings: Record<string, Binding> = {
            densityDispatch: {buffer: metadata, offset: 0, size: values.byteLength}
          };
          for (const binding of props.bindings) {
            bindings[binding.name] = getViewBinding(binding.view, getBuffer);
          }
          kernel.dispatch(computePass, {bindings, ...dispatch});
        },
        destroy: () => {
          kernel.destroy();
          metadata.destroy();
        }
      };
    }
  });
}
