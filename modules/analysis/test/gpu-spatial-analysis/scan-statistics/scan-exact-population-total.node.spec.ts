// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, GraphDataView, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslExec, WgslParser, WgslReflect} from 'wgsl_reflect';
import {createScanExactPopulationTotalNodes} from '../../../src/gpu-spatial-analysis/scan-statistics/scan-exact-population-total';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

it('W5-01 exact population tree retains carries, odd block tails and replayed totals', () => {
  // Python struct f32 rounding: f32(171*(1+2*f32(.3)))=273.6000061035156;
  // f32(257*(1+2^-24+2^-25))=257.0000305175781. Binary64 represents these sums exactly.
  for (const [values, expected] of [
    [Float32Array.from({length: 513}, (_, index) => [1, 0.3, 0.3][index % 3]), 273.6000061035156],
    [
      Float32Array.from({length: 771}, (_, index) => [1, 2 ** -24, 2 ** -25][index % 3]),
      257.0000305175781
    ],
    [Float32Array.of(1), 1]
  ] as const) {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const baseline = createTransientView(graph, 'exact-input', 'float32', values.length);
      const totals = createTransientView(graph, 'exact-totals', 'uint32', 17);
      const nodes = createScanExactPopulationTotalNodes(graph, {
        id: 'exact',
        operation: 'scan-test',
        baseline,
        totals,
        cells: values.length,
        blocks: Math.min(256, values.length)
      });
      const kernels = getGPUCommandNodeKernelSources(nodes);
      const memory = new Map<object, ArrayBuffer>();
      const input = Float32Array.from(values);
      const output = new Uint32Array(17);
      output.set([17, 0, 19, 23, 29, 31, 37]);
      memory.set(baseline.buffer, input.buffer);
      memory.set(totals.buffer, output.buffer);
      for (const zero of [false, true, false]) {
        input.set(zero ? new Float32Array(values.length) : values);
        for (let index = 0; index < nodes.length; index++) {
          const kernel = kernels[index];
          const resources = nodes[index].resources!;
          const bindings: Record<number, Record<number, ArrayBuffer>> = {0: {}};
          const reflection = new WgslReflect(kernel.source);
          for (const binding of reflection.storage) {
            const resource = resources[binding.binding];
            if (!('buffer' in resource) || !(resource.buffer instanceof GraphDataView))
              throw Error('expected packed view');
            const buffer = resource.buffer.buffer;
            if (!memory.has(buffer)) memory.set(buffer, new ArrayBuffer(buffer.byteLength));
            bindings[binding.group][binding.binding] = memory.get(buffer)!;
          }
          // Execute the pointwise kernel with one thread per workgroup, avoiding the
          // interpreter's cooperative workgroup scheduling; retain the actual index body.
          const count = Number(kernel.constants['INVOCATION_COUNT']);
          const source = kernel.source
            .replace(/^\s*_ = [^;]+;/gm, '')
            .replace(/@workgroup_size\(\d+\)/, '@workgroup_size(1)')
            .replace(
              /let index = workgroupIndex \* \d+u \+ localInvocationIndex;/,
              'let index = workgroupIndex + localInvocationIndex;'
            );
          new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups(
            'main',
            [count, 1, 1],
            bindings,
            {
              constants: {
                ...kernel.constants,
                WGSL_KERNEL_DISPATCH_X: count,
                WGSL_KERNEL_DISPATCH_Y: 1
              }
            }
          );
        }
        const result = new Float32Array(output.buffer)[1];
        expect(Number.isFinite(result)).toBe(true);
        expect(result).toBe(zero ? 0 : expected);
        // Independent BigInt sum of the stored IEEE significands, in 2^-149 units.
        const sum = zero
          ? 0n
          : Array.from(new Uint32Array(values.buffer)).reduce((total, bits) => {
              const exponent = (bits >>> 23) & 255;
              return (
                total +
                (BigInt((bits & 0x7fffff) | (exponent ? 0x800000 : 0)) <<
                  BigInt(exponent ? exponent - 1 : 0))
              );
            }, 0n);
        expect(Array.from(output.slice(7))).toEqual(
          Array.from({length: 10}, (_, word) => Number((sum >> BigInt(word * 32)) & 0xffffffffn))
        );
        expect(Array.from(output.slice(0, 7)).filter((_, index) => index !== 1)).toEqual([
          17, 19, 23, 29, 31, 37
        ]);
      }
    } finally {
      device.destroy();
    }
  }
});
