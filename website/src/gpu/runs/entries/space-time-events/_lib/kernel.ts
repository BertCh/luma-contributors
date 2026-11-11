// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Local copy of regression-weights/shared/kernel.ts (runs own their folders). A one-dimensional compute pass for the small helper kernels of the runs (turning a CSR into
// drawable segments, evaluating fitted values from a model's coefficients). The kernel reads the
// buffers the real operation wrote and writes buffers the layers draw.

import type {Binding} from '@luma.gl/core';
import {Computation} from '@luma.gl/engine';
import {
  getViewBinding,
  getViewElementOffset,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {RunBuffer, RunBuilder} from '../../../define-run';

const WORKGROUP_SIZE = 256;

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
  const graph: GPUCommandGraph<void> = builder.graph;
  const workgroupCount = Math.ceil(props.invocationCount / WORKGROUP_SIZE);
  const views: GraphDataView[] = props.bindings.map(binding => binding.buffer.view);
  const typeOf = (binding: KernelBinding) =>
    binding.buffer.format.startsWith('uint32') ? 'u32' : 'f32';
  const declarations = props.bindings
    .map(
      (binding, location) => `const ${binding.name}Offset: u32 = ${getViewElementOffset(
        views[location]
      )}u;
@group(0) @binding(${location}) var<storage, ${binding.access}> ${binding.name}: array<${typeOf(binding)}>;`
    )
    .join('\n');
  const source = /* wgsl */ `
const INVOCATION_COUNT: u32 = ${props.invocationCount}u;
${declarations}
${props.declarations ?? ''}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let index = invocation.x;
  if (index >= INVOCATION_COUNT) {
    return;
  }
  ${props.body}
}`;
  graph.addComputePass({
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
