// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {expect, it, vi} from 'vitest';
import {ShaderFactory} from '@luma.gl/core';
import {createGPUCopyCommandNode} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createTransientView, GPUCommandGraph, type GPUCommandNode} from '@luma.gl/graph/gpu-core';
import {createTerrainFlowAccumulationNodes} from '../../../src/gpu-terrain/hydrology/terrain-flow-passes';
import {createTerrainFlowRoutingAccumulationNodes} from '../../../src/gpu-terrain/hydrology/terrain-flow-routing';
import {GPUTerrainStreamOrder} from '../../../src/gpu-terrain/hydrology/gpu-terrain-stream-order';
import {makeGPURasterMetadata} from '../../../src/raster-analysis';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {WgslReflect} from 'wgsl_reflect';
import {getRasterPipelines} from '../../gpu-raster/raster-pipeline-count';
import {injectSpuriousFailure} from './terrain-finalizer-test-utils';

for (const [file, storage] of [
  ['terrain-flow-passes.ts', 'acc'],
  ['terrain-flow-routing.ts', 'acc'],
  ['gpu-terrain-stream-order.ts', 'order']
] as const) {
  it(`WGS-12 ${file} retries a spurious failure and stops after ownership changes`, () => {
    const source = readFileSync(
      new URL(`../../../src/gpu-terrain/hydrology/${file}`, import.meta.url),
      'utf8'
    );
    const finalizer = source.slice(source.indexOf('fn tryFinalize('));
    const body = finalizer.slice(0, finalizer.indexOf('}`;'));
    // A failed weak exchange with the sentinel still present has no competing owner.
    expect(body).toContain('loop {');
    expect(body).toContain(
      `let exchange = atomicCompareExchangeWeak(&${storage}[${storage}Offset + cell]`
    );
    expect(body).toContain('if (exchange.exchanged) { return true; }');
    expect(body).toContain('if (exchange.old_value != SENTINEL) { return false; }');
  });
}

for (const route of ['d8', 'weighted', 'stream-order'] as const) {
  it(`WGS-12 compiles ${route} with one injected failure in a one-round budget`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const input = <Format extends 'uint32' | 'float32'>(
      name: string,
      format: Format,
      rows: Uint32Array | Float32Array
    ) => {
      const view = createTransientView(graph, name, format, rows.length);
      return view;
    };
    const accumulation = createTransientView(graph, 'accumulation', 'float32', 1);
    const order = createTransientView(graph, 'order', 'uint32', 1);
    let nodes: readonly GPUCommandNode[];
    if (route === 'stream-order') {
      const contributor = new GPUTerrainStreamOrder({
        grid: makeGPURasterMetadata({width: 1, height: 1}),
        maximumIterations: 1,
        flowDirections: input('directions', 'uint32', Uint32Array.of(0)),
        streams: input('streams', 'uint32', Uint32Array.of(1)),
        output: {streamOrder: order}
      });
      nodes = contributor['getPullCommandNodes'](graph);
    } else {
      const props = {
        id: 'headwater',
        width: 1,
        height: 1,
        cellSizeMode: 'uniform' as const,
        maxIterations: 1,
        area: false,
        accumulation,
        elevation: input('elevation', 'float32', Float32Array.of(5)),
        receivers: input('receivers', 'uint32', Uint32Array.of(0xffffffff)),
        settings: input('settings', 'float32', Float32Array.of(1, 1, 0, 0, 0, Infinity, 0, 0)),
        runoff: input('runoff', 'float32', Float32Array.of(1))
      };
      nodes =
        route === 'd8'
          ? createTerrainFlowAccumulationNodes(graph, props)
          : createTerrainFlowRoutingAccumulationNodes(graph, {
              ...props,
              surface: props.elevation,
              routing: 'mfd-freeman'
            });
    }
    try {
      const pipelines = getRasterPipelines(nodes.map(node => injectSpuriousFailure(node)));
      const finalizer = pipelines.find(pipeline => pipeline.source.includes('fn tryFinalize('));
      expect(finalizer).toBeDefined();
      expect(finalizer!.source).toContain('return TestExchange(SENTINEL, false);');
      expect(finalizer!.source).toContain('let exchange = testExchange(cell,');
      expect(() => new WgslReflect(finalizer!.source)).not.toThrow();
    } finally {
      device.destroy();
    }
  });
}

for (const injectedFirst of [false, true]) {
  it(`B3 isolates injected and ordinary shader caches when injectedFirst=${injectedFirst}`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    Object.defineProperty(device, 'createComputePipeline', {
      value: (props: {id?: string}) => ({id: props.id ?? '', hash: '', destroy: () => {}})
    });
    Object.defineProperty(device, 'props', {value: {...device.props, _cachePipelines: false}});
    const factory = ShaderFactory.getDefaultShaderFactory(device);
    const shaders = vi.spyOn(factory, 'createShader');
    const node = createWGSLKernelNode(graph, {
      id: 'cache-probe',
      operation: 'test',
      bindings: [
        {
          name: 'acc',
          view: createTransientView(graph, 'cache-values', 'uint32', 1),
          type: 'atomic<u32>',
          access: 'read_write'
        }
      ],
      invocationCount: 1,
      declarations: `const SENTINEL: u32 = 0xffffffffu;
fn tryFinalize(cell: u32) -> bool {
  let sum = 1.0;
  loop {
    let exchange = atomicCompareExchangeWeak(&acc[accOffset + cell], SENTINEL, bitcast<u32>(sum));
    if (exchange.exchanged) { return true; }
    if (exchange.old_value != SENTINEL) { return false; }
  }
}`,
      body: '_ = tryFinalize(index);'
    });
    const compiled: ReturnType<typeof node.compile>[] = [];
    try {
      expect(device.props._cacheShaders).toBe(true);
      for (const injected of [injectedFirst, !injectedFirst, injectedFirst]) {
        compiled.push((injected ? injectSpuriousFailure(node) : node).compile({device}));
        const shader = shaders.mock.results.at(-1)!.value;
        expect(shader.source.includes('fn testExchange(')).toBe(injected);
      }
    } finally {
      compiled.forEach(executable => executable.destroy?.());
      shaders.mockRestore();
      device.destroy();
    }
  });
}

it('B4 leaves non-compute command nodes unchanged', () => {
  const node = createGPUCopyCommandNode({
    id: 'copy-control',
    resources: [],
    compile: () => ({encode: () => {}})
  });
  expect(injectSpuriousFailure(node)).toBe(node);
});
