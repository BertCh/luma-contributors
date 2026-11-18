// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Kernel} from '@luma.gl/graph';
import {registerWGSLKernelNodeRecord} from '../../analysis-core/kernels/wgsl-kernel-nodes';
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
  createGraphChunkDispatchParameters,
  GRAPH_CHUNK_PARAMETER_SLOT_WORD_COUNT,
  GRAPH_CHUNK_PARAMETER_WORD_COUNT,
  validateGraphChunkBindingRanges,
  validateGraphChunkDispatchBindings,
  type GraphChunkDispatchPlan
} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {createSpatialStatisticsParameterNodes} from '../../analysis-core/kernels/spatial-statistics-parameter-nodes';
import {
  getGraphViewChunks,
  validateRequirement,
  validateSinglePackedView,
  validateCapacity,
  validateGPUStorageBindingByteLength,
  validateMaximumCount
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  validatePointCloudLogicalViews,
  validatePointCloudPackedViews,
  getPointCloudKernelDeclarations,
  type PointCloudKernelProps
} from './pointcloud-kernel-nodes';

// A fixed workgroup shape shares one pipeline across every chunk, including short tails.
const POINT_CLOUD_CHUNK_WORKGROUP_SIZE = 256;
// A flattened workgroup index above this value would overflow its uint32 invocation index.
const POINT_CLOUD_CHUNK_MAXIMUM_WORKGROUP_COUNT = 0x1000000;

/**
 * Row-independent point-cloud dispatches. Plan field names are shader binding names. Resident
 * inputs remain packed. Offsets and lengths come from the binding range and runtime slots;
 * neither shader source nor pipeline constants depend on chunk index, length or padding.
 */
export function getPointCloudChunkKernelNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: PointCloudKernelProps & {
    plan: GraphChunkDispatchPlan;
    /** Fixed-size runtime data, such as an iteration or source-row offset. */
    runtimeWords?: readonly number[];
    /** Encode every binding offset in the slot, including differently padded packed views. */
    runtimeOffsets?: boolean;
    /** Flattened rows per logical row for fixed-stride fields such as neighbor IDs. */
    rowStrides?: Readonly<Record<string, number>>;
  }
): GPUCommandNode<Parameters>[] {
  const {id, plan} = props;
  validateGraphChunkDispatchBindings(graph, id, plan);
  validateGraphChunkBindingRanges(graph, id, props.inputs);
  if (!plan.nonemptyChunkCount) return [];
  const inputs = Object.entries(props.inputs);
  const entries = [...inputs, ...Object.entries(props.outputs)];
  validateMaximumCount(
    id,
    'storage binding count including chunk parameters',
    entries.length + 1,
    graph.device.limits.maxStorageBuffersPerShaderStage
  );
  const runtimeWords = props.runtimeWords ?? [];
  const wordCount =
    GRAPH_CHUNK_PARAMETER_WORD_COUNT +
    runtimeWords.length +
    (props.runtimeOffsets ? entries.length : 0);
  validateCapacity(
    id,
    'dispatch parameter words',
    wordCount,
    GRAPH_CHUNK_PARAMETER_WORD_COUNT,
    GRAPH_CHUNK_PARAMETER_SLOT_WORD_COUNT
  );
  for (const [index, word] of runtimeWords.entries())
    validateCapacity(id, `runtimeWords[${index}]`, word, 0, 0xffffffff);
  validatePointCloudDispatchParameterBindings(graph, id, wordCount, plan.chunks.length);
  const rowFields = new Set(Object.keys(plan.chunks[0].views));
  const offsets = Object.fromEntries(
    entries.map(([name, view], bindingIndex) => [
      name,
      props.runtimeOffsets
        ? `chunkParameters[${GRAPH_CHUNK_PARAMETER_WORD_COUNT + runtimeWords.length + bindingIndex}u]`
        : rowFields.has(name)
          ? `(arrayLength(&${name}) - chunkParameters[1] * ${((props.rowStrides?.[name] ?? 1) * view.rowByteLength) / 4}u)`
          : `${getViewElementOffset(view)}u`
    ])
  );
  const declarations = entries.map(
    ([name, view], bindingIndex) =>
      `@group(0) @binding(${bindingIndex + 1}) var<storage, ${bindingIndex < inputs.length ? 'read' : 'read_write'}> ${name}: array<${view.format === 'uint32' ? 'u32' : 'f32'}>;`
  );
  const source = `@group(0) @binding(0) var<storage, read> chunkParameters: array<u32>;
${declarations.join('\n')}
${getPointCloudKernelDeclarations(props, offsets)}
fn readDispatchParameter(row: u32) -> u32 { return chunkParameters[3u + row]; }
${props.declarations ?? ''}
@compute @workgroup_size(${POINT_CLOUD_CHUNK_WORKGROUP_SIZE})
fn main(@builtin(workgroup_id) workgroupId: vec3u,
        @builtin(num_workgroups) workgroupCount: vec3u,
        @builtin(local_invocation_index) localInvocationIndex: u32) {
  ${entries.map(([name]) => `_ = &${name};`).join('\n  ')}
  let workgroupIndex = (workgroupId.z * workgroupCount.y + workgroupId.y) * workgroupCount.x + workgroupId.x;
  if (workgroupIndex >= ${POINT_CLOUD_CHUNK_MAXIMUM_WORKGROUP_COUNT}u) { return; }
  let index = workgroupIndex * ${POINT_CLOUD_CHUNK_WORKGROUP_SIZE}u + localInvocationIndex;
  if (index >= chunkParameters[1]) { return; }
  ${props.body}
}`;
  const layouts = plan.chunks.map(chunk =>
    getBoundedDispatchLayout(
      id,
      chunk.length,
      POINT_CLOUD_CHUNK_WORKGROUP_SIZE,
      graph.device.limits.maxComputeWorkgroupsPerDimension
    )
  );
  const {slots, nodes} =
    wordCount === GRAPH_CHUNK_PARAMETER_WORD_COUNT
      ? createGraphChunkDispatchParameters(graph, {id: `${id}-chunks`, plan})
      : createPointCloudDispatchParameters(
          graph,
          `${id}-chunks`,
          plan,
          runtimeWords,
          props.runtimeOffsets ? entries.map(([name, view]) => ({name, view})) : []
        );
  return [
    ...nodes,
    ...plan.chunks
      .filter(chunk => chunk.length > 0)
      .map(chunk => {
        const parameters = slots[chunk.chunkIndex];
        const bindings = entries.map(([name, view], bindingIndex) => ({
          name,
          view: chunk.views[name] ?? view,
          writable: bindingIndex >= inputs.length
        }));
        const layout = layouts[chunk.chunkIndex];
        registerWGSLKernelNodeRecord(graph, {
          id: `${id}-chunk-${chunk.chunkIndex}`,
          operation: 'GPUPointCloud',
          source,
          body: props.body,
          declarations: props.declarations,
          entryPoint: 'main',
          workgroupSize: [POINT_CLOUD_CHUNK_WORKGROUP_SIZE, 1, 1],
          workgroups: [layout.x, layout.y, layout.z],
          invocationCount: chunk.length,
          constants: {},
          textures: [],
          indirect: false,
          bindings: [
            {name: 'chunkParameters', view: parameters, type: 'u32', access: 'read'},
            ...bindings.map(binding => ({
              name: binding.name,
              view: binding.view,
              type: binding.view.format === 'uint32' ? ('u32' as const) : ('f32' as const),
              access: binding.writable ? ('read_write' as const) : ('read' as const)
            }))
          ]
        });
        return createGPUComputeCommandNode<Parameters>({
          id: `${id}-chunk-${chunk.chunkIndex}`,
          workload: {
            operation: 'GPUPointCloud',
            commandCount: 1,
            maximumInvocationCount: chunk.length,
            maximumWorkgroupCount: layout.x * layout.y * layout.z
          },
          resources: [
            {buffer: parameters, usage: 'storage-read'},
            ...bindings.map(binding => ({
              buffer: binding.view,
              usage: binding.writable ? ('storage-read-write' as const) : ('storage-read' as const)
            }))
          ],
          compile: ({device}) => {
            const kernel = new Kernel(device, {
              id: 'pointcloud-chunk',
              source,
              shaderLayout: {
                bindings: [
                  {name: 'chunkParameters', group: 0, location: 0, type: 'read-only-storage'},
                  ...bindings.map((binding, bindingIndex) => ({
                    name: binding.name,
                    group: 0,
                    location: bindingIndex + 1,
                    type: binding.writable ? ('storage' as const) : ('read-only-storage' as const)
                  }))
                ]
              }
            });
            return {
              encode: ({computePass, getBuffer}) =>
                kernel.dispatch(computePass, {
                  bindings: {
                    chunkParameters: getViewBinding(parameters, getBuffer),
                    ...Object.fromEntries(
                      bindings.map(binding => [
                        binding.name,
                        getViewBinding(binding.view, getBuffer)
                      ])
                    )
                  },
                  ...layout
                }),
              destroy: () => kernel.destroy()
            };
          }
        });
      })
  ];
}

/** Preflight the fixed slot binding and complete aligned allocation before any resources exist. */
export function validatePointCloudDispatchParameterBindings<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  wordCount: number,
  dispatchCount: number
): void {
  validateGPUStorageBindingByteLength(
    graph.device,
    id,
    'point-cloud dispatch parameter slot',
    wordCount * 4,
    'use a device with a larger storage binding limit'
  );
  validateCapacity(
    id,
    'point-cloud dispatch parameter allocation',
    dispatchCount * GRAPH_CHUNK_PARAMETER_SLOT_WORD_COUNT * 4,
    0,
    graph.device.limits.maxBufferSize
  );
}

function createPointCloudDispatchParameters<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  plan: GraphChunkDispatchPlan,
  runtimeWords: readonly number[],
  bindings: readonly {name: string; view: GraphDataView}[]
) {
  const values = new Uint32Array(plan.chunks.length * GRAPH_CHUNK_PARAMETER_SLOT_WORD_COUNT);
  for (const chunk of plan.chunks)
    values.set(
      [
        chunk.rowBase,
        chunk.length,
        plan.totalLength,
        ...runtimeWords,
        ...bindings.map(binding => getViewElementOffset(chunk.views[binding.name] ?? binding.view))
      ],
      chunk.chunkIndex * GRAPH_CHUNK_PARAMETER_SLOT_WORD_COUNT
    );
  const {parameters, nodes} = createSpatialStatisticsParameterNodes(graph, id, values);
  const slots = plan.chunks.map(chunk =>
    graph.createDataView(parameters.buffer, {
      format: 'uint32',
      byteOffset: chunk.chunkIndex * GRAPH_CHUNK_PARAMETER_SLOT_WORD_COUNT * 4,
      length: GRAPH_CHUNK_PARAMETER_WORD_COUNT + runtimeWords.length + bindings.length
    })
  );
  return {slots, nodes};
}

/** Packed dispatches whose row count, offsets or ordinal varies share one pipeline. */
export function getPointCloudRuntimeKernelNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: PointCloudKernelProps & {runtimeWords?: readonly number[]}
): GPUCommandNode<Parameters>[] {
  // These inputs need not be row-aligned; every offset is encoded separately in the slot.
  validateGraphChunkBindingRanges(graph, props.id, {...props.inputs, ...props.outputs});
  return getPointCloudChunkKernelNodes(graph, {
    ...props,
    runtimeOffsets: true,
    plan: {
      totalLength: props.length,
      nonemptyChunkCount: props.length ? 1 : 0,
      chunks: [{chunkIndex: 0, rowBase: 0, length: props.length, views: {}}]
    }
  });
}

/** Adds fixed-stride fields to an already validated row plan without gathering or repacking. */
export function addPointCloudStridedFields(
  id: string,
  plan: GraphChunkDispatchPlan,
  fields: Readonly<Record<string, {view?: GraphDataView | GraphVectorView; stride: number}>>
): GraphChunkDispatchPlan {
  const additions = Object.entries(fields).flatMap(([name, field]) => {
    if (!field.view) return [];
    validateCapacity(id, `${name} stride`, field.stride, 1, 0xffffffff);
    const chunks = getGraphViewChunks(field.view);
    validateRequirement(
      id,
      chunks.length === plan.chunks.length && field.view.length === plan.totalLength * field.stride,
      `${name} must match row count and chunk topology with stride ${field.stride}`
    );
    for (const [index, chunk] of chunks.entries()) {
      validateSinglePackedView(chunk, [field.view.format], `${id} ${name} chunk ${index}`);
      validateCapacity(id, `${name} chunk ${index} rows`, chunk.length, 0, 0xffffffff);
      validateRequirement(
        id,
        chunk.length === plan.chunks[index].length * field.stride,
        `${name} chunk ${index} length must equal logical rows times stride ${field.stride}`
      );
    }
    return [{name, chunks}];
  });
  return {
    ...plan,
    chunks: plan.chunks.map(chunk => ({
      ...chunk,
      views: {
        ...chunk.views,
        ...Object.fromEntries(additions.map(field => [field.name, field.chunks[chunk.chunkIndex]]))
      }
    }))
  };
}

/** Construction-time logical separation across rows and aligned separation in actual dispatches. */
export function validatePointCloudChunkViews(
  id: string,
  plan: GraphChunkDispatchPlan,
  readOnlyNames: readonly string[],
  outputNames: readonly string[],
  residentInputs: Readonly<Record<string, GraphDataView | undefined>>,
  dispatches: readonly {
    inputNames: readonly string[];
    outputNames: readonly string[];
    inputs?: Readonly<Record<string, GraphDataView | undefined>>;
  }[]
): void {
  const outputs = Object.fromEntries(
    plan.chunks
      .filter(chunk => chunk.length)
      .flatMap(chunk =>
        outputNames
          .filter(name => chunk.views[name])
          .map(name => [`${name} chunk ${chunk.chunkIndex}`, chunk.views[name]])
      )
  );
  validatePointCloudLogicalViews(id, outputs);
  const inputs = [
    ...Object.entries(residentInputs),
    ...plan.chunks
      .filter(chunk => chunk.length)
      .flatMap(chunk =>
        readOnlyNames
          .filter(name => chunk.views[name])
          .map(name => [`${name} chunk ${chunk.chunkIndex}`, chunk.views[name]] as const)
      )
  ];
  for (const [name, view] of inputs) {
    if (view) validatePointCloudLogicalViews(id, {...outputs, [`input ${name}`]: view});
  }
  for (const chunk of plan.chunks.filter(chunk => chunk.length)) {
    for (const dispatch of dispatches)
      validatePointCloudPackedViews(
        id,
        {
          ...dispatch.inputs,
          ...Object.fromEntries(
            dispatch.inputNames
              .filter(name => chunk.views[name])
              .map(name => [name, chunk.views[name]])
          )
        },
        Object.fromEntries(
          dispatch.outputNames
            .filter(name => chunk.views[name])
            .map(name => [name, chunk.views[name]])
        )
      );
  }
}
