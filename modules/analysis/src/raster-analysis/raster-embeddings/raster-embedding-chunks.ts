// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, PipelineFactory, type Device} from '@luma.gl/core';
import {Kernel} from '@luma.gl/graph';
import {
  createGPUComputeCommandNode,
  createGPUCopyCommandNode,
  createTransientView,
  getBoundedDispatchLayout,
  getViewBinding,
  getViewElementOffset,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView,
  type GraphVectorView
} from '@luma.gl/graph/gpu-core';
import {
  getGraphViewChunks,
  GPUParameterBuffer,
  validateCapacity,
  validateCommandNodeBudget,
  validateGraphPorts,
  validateRequirement,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {validateRasterWritableDispatches} from '../raster-time-series/raster-time-series-chunks-b';
import {MAXIMUM_RASTER_EMBEDDING_ELEMENT_COUNT} from './raster-embedding-common';
import type {WGSLKernelBinding} from '../../analysis-core/kernels/wgsl-kernel-nodes';

/** Complete pixel records; vector chunk boundaries are shared by all raster fields. */
export type RasterEmbeddingView<Format extends 'float32' | 'uint32'> =
  | GraphDataView<Format>
  | GraphVectorView<Format>;
export type RasterEmbeddingScalarView =
  | RasterEmbeddingView<'float32'>
  | RasterEmbeddingView<'uint32'>;
type EmbeddingField = {
  format: 'float32' | 'uint32';
  view: RasterEmbeddingScalarView | undefined;
  /** Components per pixel; packed words use componentsPerWord as well. */
  components: number;
  componentsPerWord?: number;
};
export type RasterEmbeddingChunk = {
  pixelCount: number;
  views: Record<string, GraphDataView>;
};
export type PackedEmbeddingProps<Props> = Props extends
  | GraphDataView<infer Format>
  | GraphVectorView<infer Format>
  ? GraphDataView<Format>
  : Props extends object
    ? {[Key in keyof Props]: PackedEmbeddingProps<Props[Key]>}
    : Props;

/** Construction-time planning only. Pixels and native packed words never cross a boundary. */
export function planRasterEmbeddingChunks(
  id: string,
  pixelCount: number,
  fields: Record<string, EmbeddingField>,
  templateName: string,
  stages: number
): readonly RasterEmbeddingChunk[] {
  validateGraphPorts(
    id,
    Object.fromEntries(Object.entries(fields).map(([name, field]) => [name, field.view])),
    Object.fromEntries(
      Object.entries(fields).map(([name, field]) => [
        name,
        {required: name === templateName, viewType: 'either' as const, formats: [field.format]}
      ])
    )
  );
  const template = fields[templateName];
  const templateChunks = getGraphViewChunks(template.view!);
  const lastNonemptyChunkIndex = templateChunks.reduce(
    (last, chunk, index) => (chunk.length > 0 ? index : last),
    -1
  );
  const chunks = templateChunks.map(chunk => ({
    pixelCount: chunk.length / template.components,
    views: {} as Record<string, GraphDataView>
  }));
  validateRequirement(
    id,
    chunks.reduce((total, chunk) => total + chunk.pixelCount, 0) === pixelCount,
    `${templateName} must hold pixelCount times dimensions entries`
  );
  for (const [name, field] of Object.entries(fields)) {
    if (!field.view) continue;
    const views = getGraphViewChunks(field.view);
    validateRequirement(
      id,
      views.length === chunks.length,
      `${name} must match ${templateName} chunk topology`
    );
    for (const [chunkIndex, view] of views.entries()) {
      const rows = chunks[chunkIndex].pixelCount;
      validateCapacity(id, `${templateName} chunk ${chunkIndex} pixelCount`, rows, 0);
      validateSinglePackedView(view, [field.format], `${id} ${name} chunk ${chunkIndex}`);
      validateCapacity(
        id,
        `${name} chunk ${chunkIndex} component count`,
        rows * field.components,
        0,
        MAXIMUM_RASTER_EMBEDDING_ELEMENT_COUNT
      );
      const lanes = field.componentsPerWord ?? 1;
      validateRequirement(
        id,
        view.length === Math.ceil((rows * field.components) / lanes),
        `${name} chunk ${chunkIndex} must hold complete pixel records matching ${templateName}`
      );
      // Only the final nonempty chunk may have native word padding.
      validateRequirement(
        id,
        (rows * field.components) % lanes === 0 || chunkIndex >= lastNonemptyChunkIndex,
        `${name} chunk ${chunkIndex} must preserve native packed-word alignment`
      );
      chunks[chunkIndex].views[name] = view;
    }
  }
  validateCommandNodeBudget(id, 1 + stages * chunks.filter(chunk => chunk.pixelCount > 0).length);
  return chunks;
}

/** Replaces row fields by their chunk views, leaving shared packed tables intact. */
export function getRasterEmbeddingChunkProps<Props extends {pixelCount: number; output: object}>(
  props: Props,
  chunk: RasterEmbeddingChunk
): PackedEmbeddingProps<Props> {
  const result = {...props, pixelCount: chunk.pixelCount, output: {...props.output}};
  for (const [name, view] of Object.entries(chunk.views)) {
    if (name.startsWith('output.')) Object.assign(result.output, {[name.slice(7)]: view});
    else Object.assign(result, {[name]: view});
  }
  return result as PackedEmbeddingProps<Props>;
}

/** Validates writable separation, including alignment within each dispatch.
 * Row formats come from planRasterEmbeddingChunks fields; shared tables use their declared formats.
 */
export function validateRasterEmbeddingViews(
  id: string,
  inputs: readonly RasterEmbeddingScalarView[],
  outputs: readonly RasterEmbeddingScalarView[],
  dispatches?: readonly {
    inputs: readonly RasterEmbeddingScalarView[];
    outputs: readonly RasterEmbeddingScalarView[];
  }[]
): void {
  validateRasterWritableDispatches(id, inputs, outputs, dispatches);
}

/** Each metadata slot starts at the core 256-byte uniform alignment (64 uint32 words). */
const EMBEDDING_DISPATCH_SLOT_WORD_COUNT = 64;
/** Four header words and up to twelve binding offsets fit four uniform vec4 words. */
const EMBEDDING_DISPATCH_WORD_COUNT = 16;
/** A fixed workgroup size keeps uneven chunks on the same pipeline. */
const EMBEDDING_WORKGROUP_SIZE = 64;
/** Guard padded 3D workgroups before multiplication so the flattened invocation index cannot wrap. */
const MAXIMUM_EMBEDDING_LINEAR_WORKGROUP_COUNT =
  Math.floor(MAXIMUM_RASTER_EMBEDDING_ELEMENT_COUNT / EMBEDDING_WORKGROUP_SIZE) + 1;

export type RasterEmbeddingKernel = {
  id: string;
  operation: string;
  variant?: string;
  invocationCount: number;
  pixelCount: number;
  bindings: readonly WGSLKernelBinding[];
  declarations?: string;
  body: string;
  /** Runtime tile base and length, never pipeline constants. */
  tileBase?: number;
  tileCount?: number;
};

/** Checks all caller bindings and metadata allocation before declaring any scratch or nodes. */
export function validateRasterEmbeddingBindings<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  inputs: readonly RasterEmbeddingScalarView[],
  outputs: readonly RasterEmbeddingScalarView[],
  kernelCount: number,
  props: {output: object},
  maximumStorageBindingCount: number,
  chunks: readonly RasterEmbeddingChunk[]
): void {
  for (const chunk of chunks) {
    if (chunk.pixelCount === 0) continue;
    const invocationCounts = [chunk.pixelCount];
    if (chunk.views['output.quantized'])
      invocationCounts.push(chunk.views['output.quantized'].length);
    for (const invocationCount of invocationCounts)
      getBoundedDispatchLayout(
        id,
        invocationCount,
        EMBEDDING_WORKGROUP_SIZE,
        graph.device.limits.maxComputeWorkgroupsPerDimension
      );
  }
  const fields = Object.entries(props).filter(([, value]) =>
    inputs.includes(value as RasterEmbeddingScalarView)
  );
  const outputFields = Object.entries(props.output).map(
    ([name, view]) => [`output.${name}`, view] as const
  );
  validateGraphChunkBindingRanges(graph, id, Object.fromEntries([...fields, ...outputFields]));
  validateCapacity(
    id,
    'kernel storage binding count',
    maximumStorageBindingCount,
    1,
    graph.device.limits.maxStorageBuffersPerShaderStage
  );
  validateCapacity(
    id,
    'dispatch parameter buffer byteLength',
    Math.max(kernelCount, 1) * EMBEDDING_DISPATCH_SLOT_WORD_COUNT * 4,
    1,
    graph.device.limits.maxBufferSize
  );
  validateCapacity(
    id,
    'dispatch parameter binding byteLength',
    EMBEDDING_DISPATCH_WORD_COUNT * 4,
    1,
    graph.device.limits.maxUniformBufferBindingSize
  );
}

/**
 * One published Kernel pipeline per stage, regardless of chunk or tile count. Offsets, lengths
 * and tile bounds are uniforms; the bounded workgroup layout belongs exclusively to each node.
 * The upload repeats on every encoding, so graph transient aliasing cannot corrupt metadata.
 */
export function createRasterEmbeddingKernelNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  kernels: readonly RasterEmbeddingKernel[]
): GPUCommandNode<Parameters>[] {
  if (kernels.length === 0) return [];
  const values = new Uint32Array(kernels.length * EMBEDDING_DISPATCH_SLOT_WORD_COUNT);
  const layouts = kernels.map((kernel, kernelIndex) => {
    validateCapacity(
      id,
      'kernel storage binding count',
      kernel.bindings.length,
      1,
      Math.min(
        EMBEDDING_DISPATCH_WORD_COUNT - 4,
        graph.device.limits.maxStorageBuffersPerShaderStage
      )
    );
    values.set(
      [
        kernel.invocationCount,
        kernel.pixelCount,
        kernel.tileBase ?? 0,
        kernel.tileCount ?? 0,
        ...kernel.bindings.map(binding => getViewElementOffset(binding.view))
      ],
      kernelIndex * EMBEDDING_DISPATCH_SLOT_WORD_COUNT
    );
    return getBoundedDispatchLayout(
      id,
      kernel.invocationCount,
      EMBEDDING_WORKGROUP_SIZE,
      graph.device.limits.maxComputeWorkgroupsPerDimension
    );
  });
  const parameters = createTransientView(
    graph,
    `${id}-dispatch-parameters`,
    'uint32',
    values.length,
    Buffer.STORAGE | Buffer.UNIFORM | Buffer.COPY_DST
  );
  const pipelineFactories = new WeakMap<Device, PipelineFactory>();
  const nodes: GPUCommandNode<Parameters>[] = [
    createGPUCopyCommandNode({
      id: `${id}-dispatch-upload`,
      resources: [{buffer: parameters, usage: 'copy-destination'}],
      compile: ({device}) => {
        const staging = new GPUParameterBuffer(device, {
          id: `${id}-dispatch-defaults`,
          format: 'uint32',
          length: values.length,
          values
        });
        return {
          encode: ({commandEncoder, getBuffer}) =>
            commandEncoder.copyBufferToBuffer({
              sourceBuffer: staging.buffer,
              sourceOffset: 0,
              destinationBuffer: getBuffer(parameters),
              destinationOffset: parameters.byteOffset,
              size: values.byteLength
            }),
          destroy: () => staging.destroy()
        };
      }
    })
  ];
  for (const [kernelIndex, kernel] of kernels.entries()) {
    const slot = graph.createDataView(parameters.buffer, {
      format: 'uint32',
      length: EMBEDDING_DISPATCH_WORD_COUNT,
      byteOffset: kernelIndex * EMBEDDING_DISPATCH_SLOT_WORD_COUNT * 4
    });
    const uniformWord = (word: number) =>
      `embeddingDispatch.words[${Math.floor(word / 4)}u][${word % 4}u]`;
    const replaceOffsets = (source: string) =>
      kernel.bindings.reduce(
        (result, binding, bindingIndex) =>
          result.replaceAll(
            new RegExp(`\\b${binding.name}Offset\\b`, 'g'),
            uniformWord(bindingIndex + 4)
          ),
        source
      );
    const source = `struct EmbeddingDispatch { words: array<vec4<u32>, 4> }
@group(0) @binding(${kernel.bindings.length}) var<uniform> embeddingDispatch: EmbeddingDispatch;
${kernel.bindings.map((binding, bindingIndex) => `@group(0) @binding(${bindingIndex}) var<storage, ${binding.access}> ${binding.name}: array<${binding.type}>;`).join('\n')}
${replaceOffsets(kernel.declarations ?? '')}
@compute @workgroup_size(${EMBEDDING_WORKGROUP_SIZE})
fn main(@builtin(workgroup_id) workgroupId: vec3<u32>, @builtin(num_workgroups) workgroupCount: vec3<u32>, @builtin(local_invocation_index) localInvocationIndex: u32) {
  ${kernel.bindings.map(binding => `_ = &${binding.name};`).join('\n  ')}
  let workgroupIndex = (workgroupId.z * workgroupCount.y + workgroupId.y) * workgroupCount.x + workgroupId.x;
  if (workgroupIndex >= ${MAXIMUM_EMBEDDING_LINEAR_WORKGROUP_COUNT}u) { return; }
  let index = workgroupIndex * ${EMBEDDING_WORKGROUP_SIZE}u + localInvocationIndex;
  if (index >= ${uniformWord(0)}) { return; }
  ${replaceOffsets(kernel.body)}
}`;
    nodes.push(
      createGPUComputeCommandNode({
        id: kernel.id,
        resources: [
          ...kernel.bindings.map(binding => ({
            buffer: binding.view,
            usage:
              binding.access === 'read'
                ? ('storage-read' as const)
                : ('storage-read-write' as const)
          })),
          {buffer: slot, usage: 'uniform'}
        ],
        workload: {
          operation: kernel.operation,
          ...(kernel.variant ? {variant: kernel.variant} : {}),
          commandCount: 1,
          maximumInvocationCount: kernel.invocationCount,
          maximumWorkgroupCount:
            layouts[kernelIndex].x * layouts[kernelIndex].y * layouts[kernelIndex].z
        },
        compile: ({device}) => {
          let pipelineFactory = pipelineFactories.get(device);
          if (!pipelineFactory) {
            pipelineFactory = new PipelineFactory(device);
            pipelineFactories.set(device, pipelineFactory);
          }
          const compiledKernel = new Kernel(device, {
            id: kernel.operation,
            source,
            pipelineFactory,
            shaderLayout: {
              bindings: [
                ...kernel.bindings.map((binding, location) => ({
                  name: binding.name,
                  group: 0,
                  location,
                  type:
                    binding.access === 'read'
                      ? ('read-only-storage' as const)
                      : ('storage' as const)
                })),
                {
                  name: 'embeddingDispatch',
                  group: 0,
                  location: kernel.bindings.length,
                  type: 'uniform'
                }
              ]
            }
          });
          return {
            encode: ({computePass, getBuffer}) =>
              compiledKernel.dispatch(computePass, {
                bindings: Object.fromEntries([
                  ...kernel.bindings.map(binding => [
                    binding.name,
                    getViewBinding(binding.view, getBuffer)
                  ]),
                  ['embeddingDispatch', getViewBinding(slot, getBuffer)]
                ]),
                ...layouts[kernelIndex]
              }),
            destroy: () => compiledKernel.destroy()
          };
        }
      })
    );
  }
  return nodes;
}
