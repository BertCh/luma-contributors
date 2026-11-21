// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUCommandGraph,
  GraphDataView,
  createTransientView,
  type GraphBufferHandle
} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {createScanOrderedPrefixNodes} from '../../../src/gpu-spatial-analysis/scan-statistics/scan-ordered-prefix';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

it('parallel prefixes preserve actual resource wiring, lane order and retained frame memory (S2)', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    for (const rows of [1, 2, 3, 17, 257]) {
      const lanes = 2;
      const length = rows * lanes;
      const inputAllocation = createTransientView(
        graph,
        `prefix-input-${rows}`,
        'uint32',
        length + 2
      );
      const outputAllocation = createTransientView(
        graph,
        `prefix-output-${rows}`,
        'uint32',
        length + 2
      );
      const input = graph.createDataView(inputAllocation.buffer, {
        format: 'uint32',
        byteOffset: 8,
        length
      });
      const output = graph.createDataView(outputAllocation.buffer, {
        format: 'uint32',
        byteOffset: 4,
        length
      });
      const nodes = createScanOrderedPrefixNodes(graph, {
        id: `prefix-${rows}`,
        operation: 'prefix-test',
        input,
        output,
        rowCount: rows,
        laneCount: lanes
      });
      const kernels = getGPUCommandNodeKernelSources(nodes);
      const memory = new Map<GraphBufferHandle, Uint32Array>();
      const backing = (handle: GraphBufferHandle) => {
        if (!memory.has(handle))
          memory.set(handle, new Uint32Array(handle.byteLength / 4).fill(77));
        return memory.get(handle)!;
      };
      const execute = (frame: number, breakFinalDestination = false) => {
        backing(input.buffer).set(
          Uint32Array.from({length}, (_, index) =>
            frame === 0 ? 0 : index % lanes === 0 ? 16777217 : frame
          ),
          input.byteOffset / 4
        );
        for (const [stage, kernel] of kernels.entries()) {
          const bindings: Record<number, Uint32Array> = {};
          const resources = nodes[stage].resources!.filter(resource => 'buffer' in resource);
          for (const [slot, resource] of resources.entries()) {
            const bound = resource.buffer;
            bindings[slot] = backing(bound instanceof GraphDataView ? bound.buffer : bound);
          }
          // Counterfactual final-store misbinding: the caller's output must expose the loss.
          if (breakFinalDestination && stage === kernels.length - 1)
            bindings[1] = new Uint32Array(bindings[1].length).fill(77);
          for (let index = length - 1; index >= 0; index--) {
            const source = kernel.source
              .replace(/^\s*_ = [^;]+;/gm, '')
              .replace(/@workgroup_size\(\d+\)/, '@workgroup_size(1)')
              .replace(
                /let index = workgroupIndex \* \d+u \+ localInvocationIndex;/,
                `let index = ${index}u;`
              );
            new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups(
              'main',
              1,
              {0: bindings},
              {constants: kernel.constants}
            );
          }
        }
        return Array.from(
          backing(output.buffer).slice(output.byteOffset / 4, output.byteOffset / 4 + length)
        );
      };
      for (const frame of [1, 0, 2]) {
        const expected = new Uint32Array(length);
        // Exact integer closed form: lane weights are constant; saturation begins at 2^32 - 1.
        for (let row = 0; row < rows; row++) {
          expected[row * 2] = Math.min(0xffffffff, (row + 1) * (frame === 0 ? 0 : 16777217));
          expected[row * 2 + 1] = (row + 1) * frame;
        }
        if (frame === 2) expect(execute(frame, true)).not.toEqual(Array.from(expected));
        expect(execute(frame)).toEqual(Array.from(expected));
        expect(backing(output.buffer)[0]).toBe(77); // Caller prefix guard is untouched.
      }
    }
  } finally {
    device.destroy();
  }
});
