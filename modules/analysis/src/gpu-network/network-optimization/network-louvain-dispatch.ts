// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Bindings, type BindingsByGroup, type ComputePipeline} from '@luma.gl/core';
import {Kernel} from '@luma.gl/graph';
import {
  getViewBinding,
  type GPUComputeCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';

/**
 * GPU-gates published Louvain sort/reduction nodes through their public encode interface.
 * alpha.14's GPUSort and GPUReduction nodes omit public dispatch workload metadata. Inferring
 * their geometry from private fields or node IDs would couple analysis to upstream internals.
 * Instead this adapter observes their actual public ComputePass.dispatch call, writes its
 * constant geometry once, and precedes dispatchIndirect with a one-thread GPU predicate.
 * No submission/readback, copied kernels, private metadata or replacement primitive is involved.
 * Each node still owns its original executable; its three small gate resources die with it.
 * @internal
 */
export function gateNetworkLouvainPrimitive<Parameters>(
  command: GPUComputeCommandNode<Parameters>,
  state: GraphDataView<'uint32'>,
  predicate: string
): GPUComputeCommandNode<Parameters> {
  return {
    ...command,
    description: 'GPU-gated Louvain primitive (public ComputePass dispatch adapter)',
    resources: [...(command.resources ?? []), {buffer: state, usage: 'storage-read'}],
    workload: {...command.workload, commandCount: 2},
    compile: context => {
      const executable = command.compile(context);
      const cleanup: (() => void)[] = [() => executable.destroy?.()];
      try {
        const geometry = context.device.createBuffer({
          byteLength: 16,
          usage: Buffer.UNIFORM | Buffer.COPY_DST
        });
        cleanup.push(() => geometry.destroy());
        const dispatch = context.device.createBuffer({
          byteLength: 12,
          usage: Buffer.STORAGE | Buffer.INDIRECT
        });
        cleanup.push(() => dispatch.destroy());
        const gate = new Kernel(context.device, {
          id: `${command.id}-louvain-gate`,
          source: `const stateOffset: u32 = ${state.byteOffset / 4}u;
@group(0) @binding(0) var<storage, read> state: array<u32>;
@group(0) @binding(1) var<uniform> geometry: vec4<u32>;
@group(0) @binding(2) var<storage, read_write> dispatch: array<u32>;
@compute @workgroup_size(1) fn main() {
  dispatch[0] = select(0u, geometry.x, ${predicate});
  dispatch[1] = geometry.y; dispatch[2] = geometry.z;
}`,
          shaderLayout: {
            bindings: [
              {name: 'state', group: 0, location: 0, type: 'read-only-storage'},
              {name: 'geometry', group: 0, location: 1, type: 'uniform'},
              {name: 'dispatch', group: 0, location: 2, type: 'storage'}
            ]
          }
        });
        cleanup.push(() => gate.destroy());
        let previousGeometry = '';
        return {
          encode: encodeContext => {
            const pass = encodeContext.computePass;
            let pipeline: ComputePipeline;
            let bindings: Bindings | BindingsByGroup;
            const adapter = new Proxy(pass, {
              get(target, property) {
                if (property === 'setPipeline')
                  return (value: ComputePipeline) => {
                    pipeline = value;
                  };
                if (property === 'setBindings')
                  return (value: Bindings | BindingsByGroup) => {
                    bindings = value;
                  };
                if (property === 'dispatch')
                  return (x: number, y = 1, z = 1) => {
                    const currentGeometry = `${x},${y},${z}`;
                    if (currentGeometry !== previousGeometry) {
                      geometry.write(Uint32Array.of(x, y, z, 0));
                      previousGeometry = currentGeometry;
                    }
                    gate.dispatch(pass, {
                      bindings: {
                        state: getViewBinding(state, encodeContext.getBuffer),
                        geometry,
                        dispatch
                      },
                      x: 1
                    });
                    pass.setPipeline(pipeline);
                    pass.setBindings(bindings);
                    pass.dispatchIndirect(dispatch);
                  };
                const value = Reflect.get(target, property, target);
                return typeof value === 'function' ? value.bind(target) : value;
              }
            });
            executable.encode({...encodeContext, computePass: adapter});
          },
          destroy: () => {
            executable.destroy?.();
            gate.destroy();
            geometry.destroy();
            dispatch.destroy();
          }
        };
      } catch (error) {
        // The graph cannot own an executable that compile never returned. Release every
        // resource acquired so far, in reverse order, and preserve the compilation error.
        for (const destroy of cleanup.reverse()) {
          try {
            destroy();
          } catch {
            /* Continue releasing the other resources. */
          }
        }
        throw error;
      }
    }
  };
}
