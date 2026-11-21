// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import type {GPUCPUVector} from '@luma.gl/graph/gpu-core';
import {CPUNetworkTimeRoutingOperation} from '../../../src/gpu-network/network-optimization/cpu-network-time-routing-operation';
import {CPUNetworkTravelTimePercentilesOperation} from '../../../src/gpu-network/network-time-geography/cpu-network-travel-time-percentiles-operation';
const vector = (
  format: GPUCPUVector['format'],
  values: Uint32Array | Float32Array
): GPUCPUVector => {
  const length = values.length / (format.endsWith('x2') ? 2 : 1);
  return {format, length, chunks: [{values, length}], pack: () => values};
};
it('S2 publishes CPU graph masks that distinguish reached overflow and a disconnected state', async () => {
  const props = {
    stateCount: 4,
    transitionCount: 2,
    stateVertices: 'vertices',
    transitions: 'edges',
    costs: 'costs',
    endpoints: 'endpoints',
    departure: 'departure',
    output: {
      arrivals: 'arrivals',
      predecessorTransitions: 'parents',
      summary: 'summary',
      unrepresentable: 'mask'
    }
  };
  const inputs = {
    stateVertices: vector('uint32', Uint32Array.of(0, 1, 2, 3)),
    transitions: vector('uint32x2', Uint32Array.of(0, 1, 1, 2)),
    costs: vector('float32x2', Float32Array.of(3e38, 0, 3e38, 0)),
    endpoints: vector('uint32', Uint32Array.of(0, 2)),
    departure: vector('float32', Float32Array.of(0))
  };
  const route = await new CPUNetworkTimeRoutingOperation(props).execute({inputs});
  expect(Array.from(route['mask'] as Uint32Array)).toEqual([0, 0, 1, 0]);
  const percentiles = await new CPUNetworkTravelTimePercentilesOperation({
    ...props,
    vertexCount: 4,
    departureCount: 1,
    percentiles: [50],
    initialState: 'initial',
    window: 'window',
    output: {
      travelTimes: 'times',
      reachedCounts: 'reached',
      status: 'status',
      unrepresentable: 'mask'
    }
  }).execute({
    inputs: {
      ...inputs,
      initialState: vector('uint32', Uint32Array.of(0)),
      window: vector('float32', Float32Array.of(0, 0))
    }
  });
  expect(Array.from(percentiles['mask'] as Uint32Array)).toEqual([0, 0, 1, 0]);
  expect(Array.from(percentiles['reached'] as Uint32Array)).toEqual([1, 1, 1, 0]);
});

it('B1 lowers both clock models to a bounded cooperative shader with separate elapsed publication', async () => {
  const {GPUNetworkTimeRouting} = await import(
    '../../../src/gpu-network/network-optimization/gpu-network-time-routing'
  );
  const {createNullWebGPUDevice, getParsedKernelSources} = await import(
    '../../utils/gpu-contributor-test-utils'
  );
  const {createOracleContext} = await import(
    '../adjacency-matrix/network-output-oracle-test-utils'
  );
  const {WgslParser} = await import('wgsl_reflect');
  for (const piecewise of [false, true]) {
    const device = createNullWebGPUDevice();
    const context = createOracleContext(device);
    const {graph, input, output} = context;
    try {
      const nodes = new GPUNetworkTimeRouting({
        stateVertices: input('vertices', 'uint32', [0, 1, 2]),
        transitions: input('transitions', 'uint32x2', [0, 1, 1, 2, 0, 2]),
        costs: input(
          'costs',
          'float32x2',
          piecewise
            ? [0, 1.5, 10, 11.5, 0, 1.5, 10, 11.5, 0, 2.5, 10, 12.5]
            : [1.5, 0, 1.5, 0, 2.5, 0]
        ),
        ...(piecewise ? {costModel: 'piecewise-arrival' as const, samplesPerTransition: 2} : {}),
        endpoints: input('endpoints', 'uint32', [0, 2]),
        departure: input('departure', 'float32', [2 ** 25]),
        output: {
          arrivals: output('arrivals', 'float32', 3),
          elapsedTimes: output('elapsed', 'float32', 3),
          predecessorTransitions: output('predecessors', 'uint32', 3),
          summary: output('summary', 'uint32', 4)
        }
      }).getCommandNodes(graph);
      const sources = getParsedKernelSources(nodes);
      for (const {source: elapsed} of sources.getConfigurations('network-time-routing-elapsed')) {
        expect(() => new WgslParser().parse(elapsed)).not.toThrow();
        expect(elapsed).toContain('workgroupUniformLoad(&running)');
        expect(elapsed.match(/@binding\(/g)).toHaveLength(8);
      }
      expect(sources.has('network-time-routing-publish-arrivals')).toBe(true);
      expect(nodes.some(node => node.id === 'network-time-routing')).toBe(false);
    } finally {
      context.destroy();
      device.destroy();
    }
  }
});

it('B1 generated elapsed arithmetic preserves durations, original-clock schedules and predecessors', async () => {
  const {GPUNetworkTimeRouting} = await import(
    '../../../src/gpu-network/network-optimization/gpu-network-time-routing'
  );
  const {createNullWebGPUDevice, getParsedKernelSources} = await import(
    '../../utils/gpu-contributor-test-utils'
  );
  const {createOracleContext} = await import(
    '../adjacency-matrix/network-output-oracle-test-utils'
  );
  const {WgslExec, WgslParser} = await import('wgsl_reflect');
  for (const piecewise of [false, true]) {
    const device = createNullWebGPUDevice();
    const context = createOracleContext(device);
    const {graph, input, output} = context;
    const costs = piecewise
      ? Float32Array.of(0, 1.5, 10, 11.5, 0, 1.5, 10, 11.5, 0, 2.5, 10, 12.5)
      : Float32Array.of(1.5, 0, 1.5, 0, 2.5, 0);
    try {
      const nodes = new GPUNetworkTimeRouting({
        stateVertices: input('vertices', 'uint32', [0, 1, 2]),
        transitions: input('transitions', 'uint32x2', [0, 1, 1, 2, 0, 2]),
        costs: input('costs', 'float32x2', Array.from(costs)),
        ...(piecewise ? {costModel: 'piecewise-arrival' as const, samplesPerTransition: 2} : {}),
        endpoints: input('endpoints', 'uint32', [0, 2]),
        departure: input('departure', 'float32', [2 ** 25]),
        output: {
          arrivals: output('arrivals', 'float32', 3),
          elapsedTimes: output('elapsed', 'float32', 3),
          predecessorTransitions: output('predecessors', 'uint32', 3),
          summary: output('summary', 'uint32', 4)
        }
      }).getCommandNodes(graph);
      let source = getParsedKernelSources(nodes).get('network-time-routing-elapsed')!;
      // wgsl_reflect has no barrier scheduler. Project the ordered cooperative transitions
      // onto one lane to execute the actual generated arithmetic; this does not test GPU races.
      source = source
        .replace(/override INVOCATION_COUNT: u32;/, 'const INVOCATION_COUNT: u32 = 1u;')
        .replace(/override WGSL_KERNEL_DISPATCH_X: u32;/, 'const WGSL_KERNEL_DISPATCH_X: u32 = 1u;')
        .replace(/override WGSL_KERNEL_DISPATCH_Y: u32;/, 'const WGSL_KERNEL_DISPATCH_Y: u32 = 1u;')
        .replace(/@workgroup_size\(64\)/, '@workgroup_size(1)')
        .replaceAll('workgroupBarrier();', '')
        .replaceAll('workgroupUniformLoad(&running)', 'running')
        .replaceAll('+= 64u', '+= 1u')
        .replace('index == targetState % 64u && ', '')
        .replace(/^\s*_ = .*;$/gm, '');
      source = source
        .replaceAll('atomic<u32>', 'u32')
        .replace(/atomicLoad\(&([a-zA-Z]+)\)/g, '$1')
        .replace(/atomicStore\(&([a-zA-Z]+), ([^;\n]+)\);/g, '$1 = $2;');
      const elapsed = new Float32Array(3);
      const predecessors = new Uint32Array(3);
      const summary = new Uint32Array(4);
      const values: Record<string, Float32Array | Uint32Array> = {
        vertices: Uint32Array.of(0, 1, 2),
        transitions: Uint32Array.of(0, 1, 1, 2, 0, 2),
        costs,
        endpoints: Uint32Array.of(0, 2),
        departure: Float32Array.of(2 ** 25),
        elapsed,
        predecessors,
        summary
      };
      const bindings: Record<number, Float32Array | Uint32Array> = {};
      for (const match of source.matchAll(/@binding\((\d+)\)\s+var<storage,\s*\w+>\s+(\w+)/g))
        bindings[Number(match[1])] = values[match[2]];
      new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups('main', 1, {0: bindings});
      expect(Array.from(elapsed)).toEqual([0, 1.5, 2.5]);
      expect(Array.from(predecessors)).toEqual([0xffffffff, 0, 2]);
      expect(Array.from(summary.slice(0, 3))).toEqual([1, 1, 2]);
    } finally {
      context.destroy();
      device.destroy();
    }
  }
});
