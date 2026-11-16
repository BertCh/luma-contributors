// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Kernel} from '@luma.gl/graph';
import {
  createGPUComputeCommandNode,
  getBoundedDispatchLayout,
  getViewBinding,
  getViewElementOffset,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createGraphChunkDispatchParameters,
  type GraphChunkDispatchPlan
} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {getBinarySearchWGSL} from '../../analysis-core/kernels/binary-search-wgsl';

// A fixed workgroup size keeps every chunk on one pipeline; the guard prevents uint32 wrap.
const AFFINE_CHUNK_WORKGROUP_SIZE = 256;
// A 256-thread workgroup must be guarded before its index multiplication can wrap uint32.
const AFFINE_MAXIMUM_LINEAR_WORKGROUP_COUNT = 0x1000000;

/** Transforms independent vertices while retaining global feature lookup across chunks. @internal */
export function createAffineChunkNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    plan: GraphChunkDispatchPlan;
    ringOffsets?: GraphDataView<'uint32'>;
    featureOffsets?: GraphDataView<'uint32'>;
    polygonOffsets?: GraphDataView<'uint32'>;
    parameters?: GraphDataView<'float32'>;
    featureTransforms?: GraphDataView<'float32'>;
    featureCount: number;
  }
): readonly GPUCommandNode<Parameters>[] {
  const sideBindings = Object.entries({
    ringOffsets: props.ringOffsets,
    featureOffsets: props.featureOffsets,
    polygonOffsets: props.polygonOffsets,
    parameters: props.parameters,
    featureTransforms: props.featureTransforms
  }).flatMap(([name, view]) => (view ? [{name, view}] : []));
  // Resolve dispatch bounds before declaring the parameter allocation.
  const layouts = props.plan.chunks.map(chunk =>
    getBoundedDispatchLayout(
      props.id,
      chunk.length,
      AFFINE_CHUNK_WORKGROUP_SIZE,
      graph.device.limits.maxComputeWorkgroupsPerDimension
    )
  );
  const source = `
@group(0) @binding(0) var<storage, read> chunkParameters: array<u32>;
@group(0) @binding(1) var<storage, read> positions: array<f32>;
@group(0) @binding(2) var<storage, read_write> outputPositions: array<f32>;
${sideBindings
  .map(
    ({name, view}, index) => `
const ${name}Offset: u32 = ${getViewElementOffset(view)}u;
@group(0) @binding(${index + 3}) var<storage, read> ${name}: array<${view.format === 'uint32' ? 'u32' : 'f32'}>;
${view.format === 'uint32' ? getBinarySearchWGSL(name) : ''}`
  )
  .join('\n')}
@compute @workgroup_size(${AFFINE_CHUNK_WORKGROUP_SIZE})
fn main(@builtin(workgroup_id) workgroupId: vec3u,
        @builtin(num_workgroups) workgroupCount: vec3u,
        @builtin(local_invocation_index) localInvocationIndex: u32) {
  ${sideBindings.map(({name}) => `_ = &${name};`).join('\n  ')}
  let workgroupIndex = (workgroupId.z * workgroupCount.y + workgroupId.y) * workgroupCount.x + workgroupId.x;
  if (workgroupIndex >= ${AFFINE_MAXIMUM_LINEAR_WORKGROUP_COUNT}u) { return; }
  let index = workgroupIndex * ${AFFINE_CHUNK_WORKGROUP_SIZE}u + localInvocationIndex;
  let rowCount = chunkParameters[1];
  if (index >= rowCount) { return; }
  // arrayLength includes only the aligned prefix plus the logical rows, never buffer suffixes.
  let positionsOffset = arrayLength(&positions) - 2u * rowCount;
  let outputPositionsOffset = arrayLength(&outputPositions) - 2u * rowCount;
  let point = vec2f(positions[positionsOffset + 2u * index], positions[positionsOffset + 2u * index + 1u]);
  ${
    props.featureTransforms
      ? `
  let globalIndex = chunkParameters[0] + index;
  let ring = max(upperBoundRingOffsets(0u, ${props.ringOffsets!.length - 1}u, globalIndex), 1u) - 1u;
  ${props.polygonOffsets ? `let polygon = max(upperBoundPolygonOffsets(0u, ${props.polygonOffsets.length - 1}u, ring), 1u) - 1u;` : ''}
  let featureIndex = min(${props.featureOffsets ? `max(upperBoundFeatureOffsets(0u, ${props.featureOffsets.length - 1}u, ${props.polygonOffsets ? 'polygon' : 'ring'}), 1u) - 1u` : 'ring'}, ${props.featureCount - 1}u);
  let base = 6u * featureIndex + featureTransformsOffset;
  let result = vec2f(
    featureTransforms[base] * point.x + featureTransforms[base + 1u] * point.y + featureTransforms[base + 4u],
    featureTransforms[base + 2u] * point.x + featureTransforms[base + 3u] * point.y + featureTransforms[base + 5u]);`
      : `
  let a = parameters[parametersOffset];
  let b = parameters[parametersOffset + 1u];
  let d = parameters[parametersOffset + 2u];
  let e = parameters[parametersOffset + 3u];
  let translation = vec2f(parameters[parametersOffset + 4u], parameters[parametersOffset + 5u]);
  let origin = vec2f(parameters[parametersOffset + 6u], parameters[parametersOffset + 7u]);
  let relative = point - origin;
  let result = vec2f(a * relative.x + b * relative.y, d * relative.x + e * relative.y) + origin + translation;`
  }
  outputPositions[outputPositionsOffset + 2u * index] = result.x;
  outputPositions[outputPositionsOffset + 2u * index + 1u] = result.y;
}`;
  const {slots, nodes} = createGraphChunkDispatchParameters(graph, {
    id: `${props.id}-chunks`,
    plan: props.plan
  });
  return [
    ...nodes,
    ...props.plan.chunks
      .filter(chunk => chunk.length > 0)
      .map(chunk => {
        const parameters = slots[chunk.chunkIndex];
        const positions = chunk.views['positions'];
        const outputPositions = chunk.views['output.positions'];
        return createGPUComputeCommandNode<Parameters>({
          id: `${props.id}-transform-${chunk.chunkIndex}`,
          workload: {
            operation: 'GPUAffineTransform',
            variant: 'affine',
            commandCount: 1,
            maximumInvocationCount: chunk.length,
            maximumWorkgroupCount:
              layouts[chunk.chunkIndex].x *
              layouts[chunk.chunkIndex].y *
              layouts[chunk.chunkIndex].z
          },
          resources: [
            {buffer: parameters, usage: 'storage-read'},
            {buffer: positions, usage: 'storage-read'},
            {buffer: outputPositions, usage: 'storage-read-write'},
            ...sideBindings.map(({view}) => ({buffer: view, usage: 'storage-read' as const}))
          ],
          compile: ({device}) => {
            const kernel = new Kernel(device, {
              id: 'affine-transform-chunk',
              source,
              shaderLayout: {
                bindings: [
                  {name: 'chunkParameters', group: 0, location: 0, type: 'read-only-storage'},
                  {name: 'positions', group: 0, location: 1, type: 'read-only-storage'},
                  {name: 'outputPositions', group: 0, location: 2, type: 'storage'},
                  ...sideBindings.map(({name}, index) => ({
                    name,
                    group: 0,
                    location: index + 3,
                    type: 'read-only-storage' as const
                  }))
                ]
              }
            });
            return {
              encode: ({computePass, getBuffer}) =>
                kernel.dispatch(computePass, {
                  bindings: {
                    chunkParameters: getViewBinding(parameters, getBuffer),
                    positions: getViewBinding(positions, getBuffer),
                    outputPositions: getViewBinding(outputPositions, getBuffer),
                    ...Object.fromEntries(
                      sideBindings.map(({name, view}) => [name, getViewBinding(view, getBuffer)])
                    )
                  },
                  ...layouts[chunk.chunkIndex]
                }),
              destroy: () => kernel.destroy()
            };
          }
        });
      })
  ];
}
