// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph, type GPUCommandNode} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createTerrainFlowAccumulationNodes} from '../../../src/gpu-terrain/hydrology/terrain-flow-passes';
import {createTerrainFlowRoutingAccumulationNodes} from '../../../src/gpu-terrain/hydrology/terrain-flow-routing';
import {GPUTerrainStreamOrder} from '../../../src/gpu-terrain/hydrology/gpu-terrain-stream-order';
import {makeGPURasterMetadata} from '../../../src/raster-analysis';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

import {injectSpuriousFailure} from './terrain-finalizer-test-utils';

for (const route of ['d8', 'weighted', 'stream-order'] as const) {
  it(`WGS-12 ${route} finalizes a one-cell headwater in one round after a spurious failure`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const graph = new GPUCommandGraph(device);
    const buffers: Buffer[] = [];
    const input = <Format extends 'uint32' | 'float32'>(
      name: string,
      format: Format,
      values: Uint32Array | Float32Array
    ) => {
      const buffer = createInputBuffer(device, values);
      buffers.push(buffer);
      return importGraphBuffer(graph, name, buffer, format, values.length);
    };
    const result = createOutputBuffer(device, 1);
    buffers.push(result);
    const output = importGraphBuffer(
      graph,
      'output',
      result,
      route === 'stream-order' ? 'uint32' : 'float32',
      1
    );
    let nodes: readonly GPUCommandNode[];
    if (route === 'stream-order') {
      const contributor = new GPUTerrainStreamOrder({
        grid: makeGPURasterMetadata({width: 1, height: 1}),
        maximumIterations: 1,
        flowDirections: input('directions', 'uint32', Uint32Array.of(0)),
        streams: input('streams', 'uint32', Uint32Array.of(1)),
        output: {streamOrder: graph.createDataView(output.buffer, {format: 'uint32', length: 1})}
      });
      // Exercise the pull fallback on a tiny fixture rather than allocating >4096² cells.
      nodes = contributor['getPullCommandNodes'](graph);
    } else {
      const props = {
        id: 'headwater',
        width: 1,
        height: 1,
        cellSizeMode: 'uniform' as const,
        maxIterations: 1,
        area: false,
        elevation: input('elevation', 'float32', Float32Array.of(5)),
        receivers: input('receivers', 'uint32', Uint32Array.of(0xffffffff)),
        settings: input('settings', 'float32', Float32Array.of(1, 1, 0, 0, 0, Infinity, 0, 0)),
        runoff: input('runoff', 'float32', Float32Array.of(1)),
        accumulation: graph.createDataView(output.buffer, {format: 'float32', length: 1})
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
    const injectedModules: string[] = [];
    graph.add(
      nodes.map(node => injectSpuriousFailure(node, source => injectedModules.push(source)))
    );
    const compiled = graph.compile();
    try {
      expect(injectedModules.length).toBeGreaterThan(0);
      expect(injectedModules.every(source => source.includes('fn testExchange('))).toBe(true);
      submitGraph(device, compiled, undefined);
      // No donors: accumulation is the unit self-weight; a stream headwater has Strahler order 1.
      const values =
        route === 'stream-order' ? await readUint32(result, 1) : await readFloat32(result, 1);
      expect(Number.isFinite(values[0])).toBe(true);
      expect(values).toEqual([1]);
    } finally {
      compiled.destroy();
      buffers.forEach(buffer => buffer.destroy());
    }
  });
}
