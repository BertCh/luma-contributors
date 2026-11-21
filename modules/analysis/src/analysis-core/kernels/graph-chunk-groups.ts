// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {Kernel} from '@luma.gl/graph';
import {
  registerWGSLKernelNodeRecord,
  registerCompiledWGSLKernelNodeRecord
} from './wgsl-kernel-nodes';
import {getGPUVectorFormatInfo} from '@luma.gl/graph/gpu-data';
import {
  createGPUComputeCommandNode,
  getBoundedDispatchLayout,
  getViewBinding,
  getViewElementOffset,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView,
  type GraphVectorView
} from '@luma.gl/graph/gpu-core';
import {
  GRAPH_CHUNK_MAXIMUM_ROW_COUNT,
  validateGraphChunkBindingRanges,
  type GraphChunkDispatchPlan
} from './graph-chunk-dispatch';
import {
  getGraphViewChunks,
  validateCapacity,
  validateMaximumCount,
  validatePowerOfTwo,
  validateRequirement,
  validateSinglePackedView
} from './gpu-contributor-utils';
import {validateKernelNamespace} from './kernel-namespace';
import type {AnalysisValidationIdentity} from './analysis-validation-error';
import type {WGSLKernelProps, WGSLKernelNodeRecord} from './wgsl-kernel-nodes';

/** A fixed-size group (quadrature, drift columns, weights) that must stay with its query. */
export type GraphChunkGroupField = {
  view: GraphDataView | GraphVectorView;
  rowsPerQuery?: number;
  /** Keeps the historical packed output-capacity contract; chunked ports remain exact. */
  allowPackedCapacity?: boolean;
};

/**
 * Plans query-aligned groups without allocating buffers, nodes, or views. The template may hold
 * several physical rows per query. Fields mirror its ordered chunk boundaries in query units.
 * No group may straddle a boundary. Empty groups/fields and zero-chunk vectors are accepted.
 * Pass an explicit validation identity to retain the caller ID and contributor separately.
 */
export function planGraphChunkGroups(
  id: AnalysisValidationIdentity,
  template: GraphChunkGroupField,
  fields: Readonly<Record<string, GraphChunkGroupField | undefined>>
): GraphChunkDispatchPlan {
  const identity = typeof id === 'string' ? {id, contributor: id} : id;
  const templateWidth = template.rowsPerQuery ?? 1;
  validateCapacity(id, 'template rowsPerQuery', templateWidth);
  const sourceChunks = getGraphViewChunks(template.view);
  const totalLength = template.view.length / templateWidth;
  validateCapacity(id, 'query total rows', totalLength, 0, GRAPH_CHUNK_MAXIMUM_ROW_COUNT);
  const definedFields = Object.entries(fields).filter(
    (entry): entry is [string, GraphChunkGroupField] => entry[1] !== undefined
  );
  const columns = definedFields.map(([name, field]) => {
    const width = field.rowsPerQuery ?? 1;
    validateCapacity(id, `${name} rowsPerQuery`, width, 0);
    const chunks = getGraphViewChunks(field.view);
    validateRequirement(
      id,
      chunks.length === sourceChunks.length,
      `${name} must match the query chunk topology`
    );
    const capacity =
      field.allowPackedCapacity &&
      sourceChunks.length === 1 &&
      !('data' in field.view) &&
      !('data' in template.view);
    validateRequirement(
      id,
      capacity
        ? field.view.length >= totalLength * width
        : field.view.length === totalLength * width,
      `${name} total rows must match the query groups`
    );
    validateRequirement(
      id,
      chunks.reduce((total, chunk) => total + chunk.length, 0) === field.view.length,
      `${name} length must equal its chunk total`
    );
    return {name, field, width, chunks};
  });
  let rowBase = 0;
  const chunks = sourceChunks.map((source, chunkIndex) => {
    validateSinglePackedView(
      source,
      [template.view.format],
      identity,
      `template chunk ${chunkIndex}`
    );
    const length = source.length / templateWidth;
    validateCapacity(
      id,
      `query chunk ${chunkIndex} rows`,
      length,
      0,
      GRAPH_CHUNK_MAXIMUM_ROW_COUNT
    );
    const views: Record<string, GraphDataView> = {};
    for (const column of columns) {
      const view = column.chunks[chunkIndex];
      validateSinglePackedView(
        view,
        [column.field.view.format],
        identity,
        `${column.name} chunk ${chunkIndex}`
      );
      const requiredLength = length * column.width;
      const capacity =
        column.field.allowPackedCapacity &&
        sourceChunks.length === 1 &&
        !('data' in column.field.view) &&
        !('data' in template.view);
      validateRequirement(
        id,
        capacity ? view.length >= requiredLength : view.length === requiredLength,
        `${column.name} chunk ${chunkIndex} must ${capacity ? 'hold at least' : 'hold exactly'} ${requiredLength} rows for the query chunk topology`,
        {
          code: 'VIEW_LENGTH',
          path: `${column.name} chunk ${chunkIndex}`,
          received: view.length,
          expected: capacity ? {minimumRows: requiredLength} : {rows: requiredLength}
        }
      );
      validateCapacity(id, `${column.name} chunk ${chunkIndex} rows`, view.length, 0);
      views[column.name] = view;
    }
    const chunk = {chunkIndex, rowBase, length, views};
    rowBase += length;
    validateCapacity(id, 'query prefix rows', rowBase, 0, GRAPH_CHUNK_MAXIMUM_ROW_COUNT);
    return chunk;
  });
  validateRequirement(id, rowBase === totalLength, 'query length must equal its chunk total');
  return {chunks, totalLength, nonemptyChunkCount: chunks.filter(chunk => chunk.length > 0).length};
}

/**
 * Borrows a packed subview within the parent's rows, including an empty slice at its end.
 * Row bounds must be nonnegative integers. Strided parents are rejected. Byte offsets need not
 * be storage-word aligned; binding helpers enforce their own alignment requirements. The graph
 * owns the subview's creation and continues tracking the original handle.
 */
export function sliceGraphChunkGroupView<Format extends GraphDataView['format'], Parameters>(
  graph: GPUCommandGraph<Parameters>,
  view: GraphDataView<Format>,
  rowStart: number,
  rowCount: number
): GraphDataView<Format> {
  const id = 'sliceGraphChunkGroupView';
  const rowByteLength = getGPUVectorFormatInfo(view.format).byteLength;
  validateRequirement(
    id,
    view.rowByteLength === rowByteLength && view.byteStride === rowByteLength,
    'view must have packed rows matching its format'
  );
  validateCapacity(id, 'rowStart', rowStart, 0, view.length);
  validateCapacity(id, 'rowCount', rowCount, 0, view.length - rowStart);
  return graph.createDataView(view.buffer, {
    format: view.format,
    length: rowCount,
    byteOffset: view.byteOffset + rowStart * view.rowByteLength
  });
}

/** One aligned uniform slot per node; its fixed layout accommodates offsets and runtime indices. */
export const GRAPH_CHUNK_GROUP_PARAMETER_WORD_COUNT = 64;
/** The four header words precede offsets for up to 32 storage bindings. */
export const GRAPH_CHUNK_GROUP_MAXIMUM_STORAGE_BINDING_COUNT = 32;
/** Runtime auxiliary values follow the binding-offset slots, leaving 28 words in the uniform. */
export const GRAPH_CHUNK_GROUP_EXTRA_PARAMETER_OFFSET = 36;
/** Default independent-row kernels use the standard WebGPU 256-thread workgroup. */
const GRAPH_CHUNK_GROUP_WORKGROUP_SIZE = 256;
/** Published getBoundedDispatchLayout requires a power-of-two workgroup greater than one. */
const GRAPH_CHUNK_GROUP_MINIMUM_WORKGROUP_SIZE = 2;

/** Immutable node metadata uploaded to a node-owned uniform, never shader/pipeline constants. */
export type GraphChunkGroupKernelProps<Parameters> = Omit<
  WGSLKernelProps<Parameters>,
  'constants' | 'guardIndex' | 'modules' | 'defines' | 'textures'
> & {
  /** Stable debug label for a kernel family; it is not a pipeline specialization. */
  pipelineId?: string;
  rowBase?: number;
  totalLength?: number;
  /** Read auxiliary word i with getChunkParameter(GRAPH_CHUNK_GROUP_EXTRA_PARAMETER_OFFSET + i). */
  parameterValues?: readonly number[];
};

/** Device-only checks; call before declaring any contributor resources, including derived scratch. */
export function validateGraphChunkGroupKernelSupport<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  storageBindingCount: number,
  workgroupSize: number = GRAPH_CHUNK_GROUP_WORKGROUP_SIZE
): void {
  validateMaximumCount(
    id,
    'storage bindings',
    storageBindingCount,
    Math.min(
      GRAPH_CHUNK_GROUP_MAXIMUM_STORAGE_BINDING_COUNT,
      graph.device.limits.maxStorageBuffersPerShaderStage
    )
  );
  validateMaximumCount(
    id,
    'chunk uniform byteLength',
    GRAPH_CHUNK_GROUP_PARAMETER_WORD_COUNT * 4,
    graph.device.limits.maxUniformBufferBindingSize
  );
  validateMaximumCount(
    id,
    'chunk uniform buffer byteLength',
    GRAPH_CHUNK_GROUP_PARAMETER_WORD_COUNT * 4,
    graph.device.limits.maxBufferSize
  );
  validateMaximumCount(
    id,
    'chunk uniform bindings',
    1,
    graph.device.limits.maxUniformBuffersPerShaderStage
  );
  validatePowerOfTwo(
    id,
    'workgroupSize',
    workgroupSize,
    GRAPH_CHUNK_GROUP_MINIMUM_WORKGROUP_SIZE,
    Math.min(
      graph.device.limits.maxComputeInvocationsPerWorkgroup,
      graph.device.limits.maxComputeWorkgroupSizeX
    )
  );
}

/**
 * Dispatches exactly this node's bounded workgroup count, with fixed source and no pipeline
 * overrides. Counts, binding prefixes and auxiliary indices live in a 256-byte uniform owned by
 * the compiled node. It is immutable, survives repeated encodings/transient aliasing, and is freed
 * with the kernel. No graph resources are allocated by this function. Storage uses remain declared
 * on their original views. The uniform needs no storage binding or graph upload/copy node.
 *
 * Precheck every caller-owned view and derived scratch BEFORE allocating contributor resources;
 * this helper repeats its own checks. Kernel declarations may use the generated private offsets.
 */
export function createGraphChunkGroupKernelNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GraphChunkGroupKernelProps<Parameters>
): GPUCommandNode<Parameters> {
  validateKernelNamespace(
    props.id,
    props.bindings,
    [],
    [
      'main',
      'GraphChunkGroupParameters',
      'graphChunkGroupParameters',
      'getChunkParameter',
      'workgroupCount'
    ]
  );
  const workgroupSize = props.workgroupSize ?? GRAPH_CHUNK_GROUP_WORKGROUP_SIZE;
  validateGraphChunkGroupKernelSupport(graph, props.id, props.bindings.length, workgroupSize);
  validateGraphChunkBindingRanges(
    graph,
    props.id,
    Object.fromEntries(props.bindings.map(binding => [binding.name, binding.view]))
  );
  validateCapacity(props.id, 'invocationCount', props.invocationCount, 0);
  validateCapacity(props.id, 'rowBase', props.rowBase ?? 0, 0);
  validateCapacity(
    props.id,
    'rowBase plus invocationCount',
    (props.rowBase ?? 0) + props.invocationCount,
    0
  );
  validateCapacity(props.id, 'totalLength', props.totalLength ?? props.invocationCount, 0);
  validateMaximumCount(
    props.id,
    'extra parameter count',
    props.parameterValues?.length ?? 0,
    GRAPH_CHUNK_GROUP_PARAMETER_WORD_COUNT - GRAPH_CHUNK_GROUP_EXTRA_PARAMETER_OFFSET
  );
  const values = new Uint32Array(GRAPH_CHUNK_GROUP_PARAMETER_WORD_COUNT);
  values.set([
    props.rowBase ?? 0,
    props.invocationCount,
    props.totalLength ?? props.invocationCount,
    0
  ]);
  for (const [bindingIndex, binding] of props.bindings.entries())
    values[4 + bindingIndex] = getViewElementOffset(binding.view);
  for (const [parameterIndex, value] of (props.parameterValues ?? []).entries()) {
    validateCapacity(props.id, `parameterValues[${parameterIndex}]`, value, 0);
    values[GRAPH_CHUNK_GROUP_EXTRA_PARAMETER_OFFSET + parameterIndex] = value;
  }
  const layout = getBoundedDispatchLayout(
    props.id,
    props.invocationCount,
    workgroupSize,
    graph.device.limits.maxComputeWorkgroupsPerDimension
  );
  // Guard before multiplying: the final partial workgroup may reach uint32's last value.
  const MAXIMUM_LINEAR_WORKGROUP_COUNT = (GRAPH_CHUNK_MAXIMUM_ROW_COUNT + 1) / workgroupSize;
  const source = `${props.bindings
    .map(
      (binding, bindingIndex) =>
        `@group(0) @binding(${bindingIndex}) var<storage, ${binding.access}> ${binding.name}: array<${binding.type}>;\nvar<private> ${binding.name}Offset: u32;`
    )
    .join('\n')}
struct GraphChunkGroupParameters { words: array<vec4<u32>, ${GRAPH_CHUNK_GROUP_PARAMETER_WORD_COUNT / 4}> }
@group(0) @binding(${props.bindings.length}) var<uniform> graphChunkGroupParameters: GraphChunkGroupParameters;
fn getChunkParameter(word: u32) -> u32 { return graphChunkGroupParameters.words[word / 4u][word % 4u]; }
${props.declarations ?? ''}
@compute @workgroup_size(${workgroupSize})
fn main(@builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(num_workgroups) workgroupCount: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32) {
  ${props.bindings.map(binding => `_ = &${binding.name};`).join('\n  ')}
  let workgroupIndex = (workgroupId.z * workgroupCount.y + workgroupId.y) * workgroupCount.x + workgroupId.x;
  if (workgroupIndex >= ${MAXIMUM_LINEAR_WORKGROUP_COUNT}u) { return; }
  let index = workgroupIndex * ${workgroupSize}u + localInvocationIndex;
  if (index >= getChunkParameter(1u)) { return; }
  ${props.bindings.map((binding, bindingIndex) => `${binding.name}Offset = getChunkParameter(${4 + bindingIndex}u);`).join('\n  ')}
  ${props.body}
}`;
  const record: WGSLKernelNodeRecord = {
    id: props.id,
    operation: props.operation,
    variant: props.variant,
    source,
    body: props.body,
    declarations: props.declarations,
    entryPoint: 'main',
    workgroupSize: [workgroupSize, 1, 1],
    workgroups: [layout.x, layout.y, layout.z],
    invocationCount: props.invocationCount,
    constants: {},
    bindings: props.bindings,
    textures: [],
    indirect: props.condition?.source === 'gpu' && props.condition.mode === 'indirect'
  };
  registerWGSLKernelNodeRecord(graph, record);
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
      maximumWorkgroupCount: layout.x * layout.y * layout.z,
      readByteLength: props.bindings
        .filter(binding => binding.access === 'read')
        .reduce((sum, binding) => sum + binding.view.length * binding.view.rowByteLength, 0),
      writeByteLength: props.bindings
        .filter(binding => binding.access !== 'read')
        .reduce((sum, binding) => sum + binding.view.length * binding.view.rowByteLength, 0)
    },
    compile: function compileRecordedKernel({device}) {
      const parameters = device.createBuffer({
        id: `${props.id}-row-parameters`,
        data: values,
        usage: Buffer.UNIFORM | Buffer.COPY_DST
      });
      let kernel: Kernel;
      try {
        kernel = new Kernel(device, {
          id: props.pipelineId ?? props.id,
          source,
          shaderLayout: {
            bindings: [
              ...props.bindings.map((binding, location) => ({
                name: binding.name,
                group: 0,
                location,
                type:
                  binding.access === 'read' ? ('read-only-storage' as const) : ('storage' as const)
              })),
              {
                name: 'graphChunkGroupParameters',
                group: 0,
                location: props.bindings.length,
                type: 'uniform'
              }
            ]
          }
        });
        registerCompiledWGSLKernelNodeRecord(graph, record, compileRecordedKernel);
      } catch (error) {
        parameters.destroy();
        throw error;
      }
      return {
        encode: ({computePass, getBuffer}) =>
          kernel.dispatch(computePass, {
            bindings: Object.fromEntries([
              ...props.bindings.map(binding => [
                binding.name,
                getViewBinding(binding.view, getBuffer)
              ]),
              [
                'graphChunkGroupParameters',
                {buffer: parameters, offset: 0, size: values.byteLength}
              ]
            ]),
            ...layout
          }),
        destroy: () => {
          kernel.destroy();
          parameters.destroy();
        }
      };
    }
  });
}
