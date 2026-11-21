// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {IS_FINITE_VALUE_WGSL} from '../../../src/analysis-core/kernels/wgsl-common';
import {GPUNetworkTravelTimePercentiles} from '../../../src/gpu-network/network-time-geography/gpu-network-travel-time-percentiles';
import {
  createNullWebGPUDevice,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {createOracleContext} from '../adjacency-matrix/network-output-oracle-test-utils';

it.each([
  0,
  2 ** 100
])('counts finite elapsed %s and excludes a disconnected vertex with exponent-bit classification', duration => {
  const device = createNullWebGPUDevice();
  const context = createOracleContext(device);
  const {graph, input, output} = context;
  try {
    const nodes = new GPUNetworkTravelTimePercentiles({
      stateVertices: input('vertices', 'uint32', [0, 1]),
      transitions: input('edges', 'uint32x2', [0, 0]),
      costs: input('costs', 'float32x2', [0, 0]),
      initialState: input('initial', 'uint32', [0]),
      window: input('window', 'float32', [1, 1]),
      vertexCount: 2,
      departureCount: 1,
      percentiles: [50],
      output: {
        travelTimes: output('times', 'float32', 2),
        reachedCounts: output('reached', 'uint32', 2),
        status: output('status', 'uint32', 2)
      }
    }).getCommandNodes(graph);
    let source = getParsedKernelSources(nodes).get('network-travel-time-percentiles-percentiles')!;
    // Host interpretation cannot reproduce Metal's infinity optimizations. Require the shared
    // bit classifier in the emitted kernel, then independently check its finite/disconnected rows.
    expect.soft(source).toContain(IS_FINITE_VALUE_WGSL);
    expect.soft(source).toContain('if (isFiniteValue(best)) { reached++; }');
    source = source
      .replace('override INVOCATION_COUNT: u32;', 'const INVOCATION_COUNT: u32 = 2u;')
      .replace('override WGSL_KERNEL_DISPATCH_X: u32;', 'const WGSL_KERNEL_DISPATCH_X: u32 = 1u;')
      .replace('override WGSL_KERNEL_DISPATCH_Y: u32;', 'const WGSL_KERNEL_DISPATCH_Y: u32 = 1u;')
      .replace(/^\s*_ = .*;$/gm, '')
      // wgsl_reflect eagerly evaluates && and would index values[-1] at slot zero.
      // Preserve the shader's short-circuit semantics for this interpreter only.
      .replace(
        'while (slot > 0u && values[slot - 1u] > best) {',
        'while (slot > 0u) { if (values[slot - 1u] <= best) { break; }'
      );
    const values: Record<string, Float32Array | Uint32Array> = {
      stateVertices: Uint32Array.of(0, 1),
      arrivals: Float32Array.of(duration, Infinity),
      departures: Float32Array.of(1),
      summaries: Uint32Array.of(1, 1, 0, 0),
      travelTimes: Float32Array.of(913, 913),
      reachedCounts: Uint32Array.of(913, 913),
      status: Uint32Array.of(913, 913)
    };
    const bindings: Record<number, Float32Array | Uint32Array> = {};
    for (const match of source.matchAll(/@binding\((\d+)\)\s+var<storage,\s*\w+>\s+(\w+)/g)) {
      expect(values[match[2]], match[2]).toBeDefined();
      bindings[Number(match[1])] = values[match[2]];
    }
    new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups('main', 1, {0: bindings});
    // One valid, converged departure reaches only vertex 0: counts 1 and 0, independent of duration.
    expect(values['travelTimes']).toEqual(Float32Array.of(duration, Infinity));
    expect(values['reachedCounts']).toEqual(Uint32Array.of(1, 0));
    expect(values['status']).toEqual(Uint32Array.of(1, 1));
  } finally {
    context.destroy();
    device.destroy();
  }
});
