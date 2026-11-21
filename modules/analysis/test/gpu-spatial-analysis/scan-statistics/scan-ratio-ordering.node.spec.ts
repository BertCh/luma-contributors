// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslExec, WgslParser, WgslReflect} from 'wgsl_reflect';
import {GPUExpectationBasedSpaceTimeScan} from '../../../src/gpu-spatial-analysis/scan-statistics/gpu-expectation-based-space-time-scan';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

it('orders Poisson ratios without overflow while retaining exact ties and zero-baseline exclusion (CORE-104)', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `ratio-${serial++}`, format, length);
  try {
    const scan = new GPUExpectationBasedSpaceTimeScan({
      positions: view('float32x2', 2),
      cases: view('uint32', 2),
      baseline: view('float32', 2),
      timeBuckets: 1,
      maximumNeighbors: 2,
      maximumWindowLength: 1,
      maximumReplicates: 1,
      parameters: view('uint32', 4),
      output: {
        clusterZones: view('uint32', 2),
        clusterStatistics: view('float32', 4),
        statistics: view('float32', 2),
        summary: view('uint32', 5)
      }
    });
    const kernel = getGPUCommandNodeKernelSources(scan.getCommandNodes(graph)).find(kernel =>
      kernel.id.endsWith('-cluster')
    )!;
    const source = `${kernel.source.slice(0, kernel.source.indexOf('@compute'))}
@group(1) @binding(0) var<storage, read_write> probeOrder: array<u32>;
@group(1) @binding(1) var<storage, read_write> probeScore: array<f32>;
@compute @workgroup_size(1) fn main() {
 var sums: NeighborhoodSums;
 sums.caseSums[0] = f32(counts[0]); sums.caseSums[1] = f32(counts[1]);
 sums.baselineSums[0] = baseline[0]; sums.baselineSums[1] = baseline[1];
 let order = getOrder(&sums);
 probeOrder[0] = order[0]; probeOrder[1] = order[1];
 let result = searchCenter(0u, 0u);
 probeOrder[2] = result.memberCount;
 probeScore[0] = result.score;
}`;
    for (const [cases, baseline, expected, expectedScore] of [
      [[1e6, 1e9], [1e-33, 1e-36], [1, 0], 102616329148.79292],
      [[3e6, 3e8], [3e-33, 1e-36], [1, 0], 30423706903.340096],
      [[1, 2], [1, 2], [0, 1], 0],
      [[5, 0], [0, 1], [1, 0], 0],
      [[0, 0], [1, 2], [0, 1], 0]
    ] as const) {
      // Expected order follows exact ratios of the float32 inputs; equal fractions retain slot order.
      const memory: Record<string, Uint32Array | Float32Array> = {
        counts: Uint32Array.from(cases),
        baseline: Float32Array.from(baseline),
        neighbors: Uint32Array.of(0, 1, 1, 0),
        parameters: Uint32Array.of(0, 0, 1, 1),
        statistics: new Float32Array(2),
        centerBest: new Uint32Array(6),
        state: new Uint32Array(4),
        clusterStatistics: new Float32Array(4),
        probeOrder: new Uint32Array(3),
        probeScore: new Float32Array(1)
      };
      const bindings: Record<number, Record<number, Uint32Array | Float32Array>> = {0: {}, 1: {}};
      for (const binding of new WgslReflect(source).storage) {
        expect(memory[binding.name], binding.name).toBeDefined();
        bindings[binding.group][binding.binding] = memory[binding.name];
      }
      new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups('main', 1, bindings, {
        constants: kernel.constants
      });
      expect(Array.from(memory['probeOrder'].slice(0, 2))).toEqual(expected);
      expect(Number.isFinite(memory['probeScore'][0])).toBe(true);
      // /usr/bin/python3: C*(math.log(C)-math.log(struct.unpack('f',struct.pack('f',B))[0]))-C+B.
      expect(Math.abs(memory['probeScore'][0] - expectedScore)).toBeLessThanOrEqual(
        2e-6 * expectedScore
      );
      if (expectedScore > 0) expect(memory['probeOrder'][2]).toBe(1);
    }
  } finally {
    device.destroy();
  }
});
