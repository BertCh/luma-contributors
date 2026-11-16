// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Kernel} from '@luma.gl/graph';
import {
  createGPUComputeCommandNode,
  getViewBinding,
  getViewElementOffset,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {getUrbanPointGridRadiusWalk, type UrbanPointGrid} from './visibility-graph-point-grid';
import type {WGSLKernelBinding} from '../../analysis-core/kernels/wgsl-kernel-nodes';

/** One shared visibility pipeline for unequal query chunks; dispatch sizes belong to nodes. @internal */
export function getVisibilityCSRQueryNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    count: number;
    chunkSize: number;
    chunkStride: number;
    queryState: GraphDataView<'uint32'>;
    declarations: string;
    bindings: WGSLKernelBinding[];
    body: string;
  }
) {
  const chunkCount = Math.ceil(props.count / props.chunkSize);
  return Array.from({length: chunkCount}, (_, chunk) => {
    const rowCount = Math.min(props.chunkSize, props.count - chunk * props.chunkSize);
    const state = graph.createDataView(props.queryState.buffer, {
      format: 'uint32',
      byteOffset: chunk * props.chunkStride * 4,
      length: rowCount + 4
    });
    const bindings = [
      ...props.bindings,
      {name: 'queryState', view: state, type: 'u32', access: 'read'} as const
    ];
    // Every chunk begins on a storage-aligned boundary; its parameter element offset is zero.
    const source = `${bindings.map((binding, location) => `const ${binding.name}Offset:u32=${getViewElementOffset(binding.view)}u;\n@group(0) @binding(${location}) var<storage,${binding.access}> ${binding.name}:array<${binding.type}>;`).join('\n')}
${props.declarations}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) invocation:vec3<u32>){
 ${bindings.map(binding => `_ = &${binding.name};`).join('\n')}
 let localIndex=invocation.x;if(localIndex>=queryState[queryStateOffset+3u]){return;}
 let index=queryState[queryStateOffset+2u]+localIndex;
 ${props.body}
}`;
    return createGPUComputeCommandNode<Parameters>({
      id: `${props.id}-chunk-${chunk}`,
      resources: bindings.map(binding => ({
        buffer: binding.view,
        usage:
          binding.access === 'read' ? ('storage-read' as const) : ('storage-read-write' as const)
      })),
      workload: {
        operation: 'GPUVisibilityGraph',
        variant: 'csr-radius-query',
        commandCount: 1,
        maximumInvocationCount: rowCount,
        maximumWorkgroupCount: Math.ceil(rowCount / 64)
      },
      compile: ({device}) => {
        const kernel = new Kernel(device, {
          id: props.id,
          source,
          shaderLayout: {
            bindings: bindings.map((binding, location) => ({
              name: binding.name,
              type:
                binding.access === 'read' ? ('read-only-storage' as const) : ('storage' as const),
              group: 0,
              location
            }))
          }
        });
        return {
          encode: ({computePass, getBuffer}) => {
            kernel.dispatch(computePass, {
              bindings: Object.fromEntries(
                bindings.map(binding => [binding.name, getViewBinding(binding.view, getBuffer)])
              ),
              x: Math.ceil(rowCount / 64),
              y: 1,
              z: 1
            });
          },
          destroy: () => kernel.destroy()
        };
      }
    });
  });
}

/** Count-stage bounded walk, shared with the traversal instrumentation spec. @internal */
export function getVisibilityCSRCountDeclarations(
  grid: UrbanPointGrid,
  maximumGridVisits: number
): string {
  return `
fn countVisibleNeighbors(sourceRow:u32)->vec3<u32>{
 let center=readPoint(sourceRow);var found=0u;var visited=0u;
 if(!isFiniteValue(center.x)||!isFiniteValue(center.y)){return vec3<u32>(0u);}
 ${getUrbanPointGridRadiusWalk(
   grid,
   'center',
   'radiusValue',
   'other',
   'visited++;if(isVisible(sourceRow,other)){found++;}',
   `if(visited>=${maximumGridVisits}u){return vec3<u32>(found,1u,visited);}`
 )}
 return vec3<u32>(found,0u,visited);
}
`;
}
