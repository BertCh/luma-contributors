// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// A one-dimensional compute pass for the small helper kernels of these runs (turning packed line
// output into drawable segments, evaluating a picture from the buffers an operation wrote). The
// kernel reads the buffers the real operation wrote and writes buffers the layers draw.

import type {Binding} from '@luma.gl/core';
import {Computation} from '@luma.gl/engine';
import {getViewBinding, getViewElementOffset, type GraphDataView} from '@luma.gl/graph/gpu-core';
import type {RunBuffer, RunBuilder} from '../../../define-run';

const WORKGROUP_SIZE = 256;

const WGSL_TYPES = {
  float32: 'f32',
  uint32: 'u32',
  sint32: 'i32',
  float32x2: 'vec2f',
  float32x4: 'vec4f',
  uint32x2: 'vec2u',
  uint32x4: 'vec4u'
} as const;

/** One storage binding of a kernel pass. */
export type KernelBinding = {
  /** WGSL variable name; `${name}Offset` holds the view's element offset. */
  name: string;
  buffer: RunBuffer;
  access: 'read' | 'read_write';
};

/** Adds one compute pass running `body` once per `index` below `invocationCount`. */
export function addKernelPass(
  builder: RunBuilder,
  props: {
    id: string;
    bindings: readonly KernelBinding[];
    invocationCount: number;
    declarations?: string;
    body: string;
  }
): void {
  const workgroupCount = Math.max(Math.ceil(props.invocationCount / WORKGROUP_SIZE), 1);
  const views: GraphDataView[] = props.bindings.map(binding => binding.buffer.view);
  const declarations = props.bindings
    .map((binding, location) => {
      const wgslType = WGSL_TYPES[binding.buffer.format as keyof typeof WGSL_TYPES];
      return `const ${binding.name}Offset: u32 = ${getViewElementOffset(views[location])}u;
@group(0) @binding(${location}) var<storage, ${binding.access}> ${binding.name}: array<${wgslType}>;`;
    })
    .join('\n');
  const source = /* wgsl */ `
const INVOCATION_COUNT: u32 = ${props.invocationCount}u;
${declarations}
${props.declarations ?? ''}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let index = invocation.x;
  ${props.bindings.map(binding => `_ = &${binding.name};`).join('\n  ')}
  if (index >= INVOCATION_COUNT) {
    return;
  }
  ${props.body}
}`;
  builder.graph.addComputePass({
    id: props.id,
    workload: {
      operation: 'RunKernel',
      variant: props.id,
      commandCount: 1,
      maximumWorkgroupCount: workgroupCount,
      maximumInvocationCount: workgroupCount * WORKGROUP_SIZE,
      readByteLength: 0,
      writeByteLength: 0
    },
    resources: props.bindings.map((binding, location) => ({
      buffer: views[location],
      usage: binding.access === 'read' ? ('storage-read' as const) : ('storage-read-write' as const)
    })),
    compile: ({device}) => {
      const computation = new Computation(device, {
        id: props.id,
        source,
        shaderLayout: {
          bindings: props.bindings.map((binding, location) => ({
            name: binding.name,
            type: binding.access === 'read' ? ('read-only-storage' as const) : ('storage' as const),
            group: 0,
            location
          }))
        }
      });
      return {
        encode: ({computePass, getBuffer}) => {
          const resolved: Record<string, Binding> = {};
          props.bindings.forEach((binding, location) => {
            resolved[binding.name] = getViewBinding(views[location], getBuffer);
          });
          computation.setBindings(resolved);
          computation.dispatch(computePass, workgroupCount);
        },
        destroy: () => computation.destroy()
      };
    }
  });
}

/**
 * Wraps a graph view that has no buffer of ours (an output of a recipe, which the graph owns) so a
 * helper kernel can read it. The wrapper only carries what `addKernelPass` needs.
 */
export function getViewBuffer<Format extends RunBuffer['format']>(
  view: GraphDataView<Format>
): RunBuffer<Format> {
  return {
    id: view.format,
    format: view.format,
    length: view.length,
    view,
    // Never read: kernels bind the view, and layers only draw buffers a run allocated itself.
    buffer: undefined as never,
    write: () => undefined
  };
}
