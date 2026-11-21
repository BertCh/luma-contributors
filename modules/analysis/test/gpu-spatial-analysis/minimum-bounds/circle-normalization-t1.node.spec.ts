// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUCommandGraph,
  createTransientView,
  type GraphDataView,
  type GPUGraphVectorFormat
} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUMinimumBounds} from '../../../src/gpu-spatial-analysis/minimum-bounds/gpu-minimum-bounds';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

for (const exponent of [-80, 0, 66]) {
  for (const translated of [false, true]) {
    it(`t1 evaluates the actual minimum-circle kernel in a safe squared-distance frame at 2^${exponent} (translated ${translated})`, () => {
      const device = createNullWebGPUDevice();
      const graph = new GPUCommandGraph(device);
      let sequence = 0;
      const view = <Format extends GPUGraphVectorFormat>(format: Format, length: number) =>
        createTransientView(graph, `circle-range-${sequence++}`, format, length);
      const circles = view('float32x3', 1);
      try {
        const nodes = new GPUMinimumBounds({
          positions: view('float32x2', 3),
          labels: view('uint32', 3),
          groupCount: 1,
          output: {circles, status: {overflow: view('uint32', 1)}}
        }).getCommandNodes(graph);
        const circle = nodes.find(node => node.id === 'minimum-bounds-circle');
        if (!circle) throw new Error('Missing minimum-bounds circle kernel');
        const input = (slot: number): GraphDataView => {
          const resource = circle.resources?.[slot];
          if (!resource || !('buffer' in resource)) throw new Error(`Missing circle input ${slot}`);
          return resource.buffer as GraphDataView;
        };
        const scale = 2 ** exponent;
        const originX = translated ? 1000 : 0;
        const originY = translated ? 2000 : 0;
        const memory = createCPUGraphMemory();
        // These are already the strict convex hull of the independent 3-4-5 right triangle.
        // Its hypotenuse midpoint is (1.5,2), and its radius is half the hypotenuse, 2.5.
        memory.write(input(0), new Uint32Array([0, 3]));
        memory.write(input(1), new Uint32Array([3]));
        memory.write(
          input(2),
          new Float32Array([
            originX * scale,
            originY * scale,
            (originX + 3) * scale,
            originY * scale,
            originX * scale,
            (originY + 4) * scale
          ])
        );
        executeCommandNodesOnCPU([circle], memory);
        const result = memory.readFloat32(circles);
        expect(result.every(Number.isFinite)).toBe(true);
        expect(Array.from(result, value => value / scale)).toEqual([
          originX + 1.5,
          originY + 2,
          2.5
        ]);
        for (const source of getGPUCommandNodeKernelSources([circle]))
          expect(auditWGSLModule(source.source, source.constants)).toEqual([]);
      } finally {
        device.destroy();
      }
    });
  }
}
