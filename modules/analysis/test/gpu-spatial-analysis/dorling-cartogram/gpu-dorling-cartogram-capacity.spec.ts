// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUDorlingCartogram} from '../../../src/gpu-spatial-analysis/dorling-cartogram/gpu-dorling-cartogram';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

// Fixed regressions just above the lifted all-pairs cap. The 8.4M dispatch boundary and 30M
// device-limit cases remain declaration/encode checks on null devices in the node spec.
const HEADLESS_DORLING_CAPACITY_ROWS = [65537, 65536 + 256] as const;

for (const rows of HEADLESS_DORLING_CAPACITY_ROWS) {
  it(`Dorling executes ${rows} rows and checks every output, including the tail`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const graph = new GPUCommandGraph(device);
    const buffers: Buffer[] = [];
    const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
      name: string,
      format: Format,
      length: number,
      words = 1
    ) => {
      const buffer = createOutputBuffer(device, length * words);
      buffers.push(buffer);
      return {buffer, view: importGraphBuffer(graph, name, buffer, format, length)};
    };
    const positions = view('input-positions', 'float32x2', rows, 2);
    const values = view('input-values', 'float32', rows);
    const centers = view('centers', 'float32x2', rows, 2);
    const radii = view('radii', 'float32', rows);
    const sourceIds = view('source-ids', 'uint32', rows);
    const metrics = view('metrics', 'float32x2', 1, 2);
    const status = view('status', 'uint32', 4);
    const failures = view('failures', 'uint32', 1);
    // Generate and verify on the GPU; caller buffers plus graph scratch stay well below 512 MiB.
    // Independent horizontal pairs have radius .5 and separation .5, so one pass moves each
    // endpoint by .25. An odd final row is an isolated singleton and must remain unchanged.
    const coordinates = `let pair = index / 2u;
let original = vec2f(f32(pair % 2048u) * 4.0 + f32(index % 2u) * 0.5, f32(pair / 2048u) * 4.0);`;
    graph.add(
      createWGSLKernelNode(graph, {
        id: 'generate',
        operation: 'DorlingCapacityFixture',
        invocationCount: rows,
        bindings: [
          {name: 'positions', view: positions.view, type: 'f32', access: 'read_write'},
          {name: 'values', view: values.view, type: 'f32', access: 'read_write'}
        ],
        body: `${coordinates}
positions[positionsOffset + index * 2u] = original.x;
positions[positionsOffset + index * 2u + 1u] = original.y;
values[valuesOffset + index] = ${Math.fround(Math.PI * 0.25)};`
      })
    );
    graph.add(
      new GPUDorlingCartogram({
        positions: positions.view,
        values: values.view,
        maximumIterations: 1,
        attraction: 0,
        output: {
          positions: centers.view,
          radii: radii.view,
          sourceIds: sourceIds.view,
          metrics: metrics.view,
          status: status.view
        }
      })
    );
    graph.add(
      createWGSLKernelNode(graph, {
        id: 'clear-failures',
        operation: 'DorlingCapacityFixture',
        invocationCount: 1,
        bindings: [{name: 'failures', view: failures.view, type: 'u32', access: 'read_write'}],
        body: 'failures[failuresOffset] = 0u;'
      })
    );
    graph.add(
      createWGSLKernelNode(graph, {
        id: 'verify',
        operation: 'DorlingCapacityFixture',
        invocationCount: rows,
        bindings: [
          {name: 'centers', view: centers.view, type: 'f32', access: 'read'},
          {name: 'radii', view: radii.view, type: 'f32', access: 'read'},
          {name: 'sourceIds', view: sourceIds.view, type: 'u32', access: 'read'},
          {name: 'failures', view: failures.view, type: 'atomic<u32>', access: 'read_write'}
        ],
        body: `${coordinates}
let paired = (index ^ 1u) < ${rows}u;
let expected = original + vec2f(select(0.0, select(-0.25, 0.25, index % 2u != 0u), paired), 0.0);
let actual = vec2f(centers[centersOffset + index * 2u], centers[centersOffset + index * 2u + 1u]);
if (!all(abs(actual - expected) <= vec2f(0.001)) || !(abs(radii[radiiOffset + index] - 0.5) <= 0.000001) || sourceIds[sourceIdsOffset + index] != index) {
  atomicAdd(&failures[failuresOffset], 1u);
}`
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      expect(await readUint32(failures.buffer, 1)).toEqual([0]);
      expect(await readUint32(status.buffer, 4)).toEqual([1, 1, 0, 0]);
      const metricValues = await readFloat32(metrics.buffer, 2);
      expect(metricValues[0]).toBeLessThan(0.001);
      expect(metricValues[1]).toBeCloseTo(0.25, 4);
    } finally {
      compiled.destroy();
      buffers.forEach(buffer => buffer.destroy());
    }
  }, 300000);
}
