// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  makeGPURasterBand,
  makeGPURasterMetadata
} from '@luma-contributors/analysis/raster-analysis';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPUTerrainFlowDistance} from '../../../src/gpu-terrain/hydrology/gpu-terrain-flow-distance';
import {GPUTerrainStreamNetwork} from '../../../src/gpu-terrain/hydrology/gpu-terrain-stream-network';
import {GPUTerrainStreamOrder} from '../../../src/gpu-terrain/hydrology/gpu-terrain-stream-order';
import {GPUTerrainWatersheds} from '../../../src/gpu-terrain/hydrology/gpu-terrain-watersheds';
import {GPUTerrainHeightAboveDrainage} from '../../../src/gpu-terrain/hydrology/gpu-terrain-height-above-drainage';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

function createFixture(device: Device, cellCount: number, width = cellCount) {
  const grid = makeGPURasterMetadata({width, height: cellCount / width});
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const outputs: Record<string, Buffer> = {};
  return {
    grid,
    graph,
    input<Format extends 'float32' | 'uint32'>(
      name: string,
      format: Format,
      values: Float32Array | Uint32Array
    ) {
      const buffer = createInputBuffer(device, values);
      buffers.push(buffer);
      return importGraphBuffer(graph, name, buffer, format, values.length);
    },
    output<Format extends 'float32' | 'uint32' | 'float32x2'>(
      name: string,
      format: Format,
      length = cellCount
    ) {
      const buffer = createOutputBuffer(device, length * (format === 'float32x2' ? 2 : 1));
      buffers.push(buffer);
      outputs[name] = buffer;
      return importGraphBuffer(graph, name, buffer, format, length);
    },
    words: (name: string, length = cellCount) => readUint32(outputs[name], length),
    floats: (name: string, length = cellCount) => readFloat32(outputs[name], length),
    destroy: () => buffers.forEach(buffer => buffer.destroy())
  };
}

it('FlowDistance rejects a cycle and its incoming tributary in its own state planes (RT-123)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createFixture(device, 6);
  const {graph, grid, input, output} = fixture;
  graph.add(
    new GPUTerrainFlowDistance({
      grid,
      // 0 <-> 1 <- 2 is rootless; 3 -> 4 -> 5 has two unit edges and an original terminal.
      flowDirections: input('directions', 'uint32', Uint32Array.of(1, 16, 16, 1, 1, 0)),
      output: {
        distance: output('distance', 'float32'),
        validity: output('validity', 'uint32'),
        terminalCells: output('terminal', 'uint32')
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await fixture.words('terminal')).toEqual([0xfffffffe, 0xfffffffe, 0xfffffffe, 5, 5, 5]);
    expect(await fixture.words('validity')).toEqual([0, 0, 0, 1, 1, 1]);
    const distance = await fixture.floats('distance');
    expect(distance).toEqual([NaN, NaN, NaN, 2, 1, 0]);
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});

it('StreamNetwork emits no cycle reaches and retains a separate acyclic reach (RT-123)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // First isolate the cycle, then add a separate 3 -> 4 -> 5 reach with length two.
  for (const cellCount of [2, 6]) {
    const fixture = createFixture(device, cellCount);
    const {grid, graph, input, output} = fixture;
    graph.add(
      new GPUTerrainStreamNetwork({
        grid,
        flowDirections: input(
          'directions',
          'uint32',
          cellCount === 2 ? Uint32Array.of(1, 16) : Uint32Array.of(1, 16, 0, 1, 1, 0)
        ),
        streams: input(
          'streams',
          'uint32',
          cellCount === 2 ? Uint32Array.of(1, 1) : Uint32Array.of(1, 1, 0, 1, 1, 1)
        ),
        output: {
          sourceCells: output('source', 'uint32'),
          targetCells: output('target', 'uint32'),
          vertexOffsets: output('offsets', 'uint32', cellCount + 1),
          positions: output('positions', 'float32x2'),
          lengths: output('lengths', 'float32'),
          reachCount: output('reaches', 'uint32', 1),
          vertexCount: output('vertices', 'uint32', 1),
          unresolvedCount: output('unresolved', 'uint32', 1),
          status: {overflow: output('overflow', 'uint32', 1)}
        }
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      expect(await fixture.words('unresolved', 1)).toEqual([2]);
      expect(await fixture.words('overflow', 1)).toEqual([0]);
      expect(await fixture.words('reaches', 1)).toEqual([cellCount === 2 ? 0 : 1]);
      expect(await fixture.words('vertices', 1)).toEqual([cellCount === 2 ? 0 : 3]);
      if (cellCount === 6) {
        expect(await fixture.words('source', 1)).toEqual([3]);
        expect(await fixture.words('target', 1)).toEqual([5]);
        expect(await fixture.words('offsets', 2)).toEqual([0, 3]);
        expect(await fixture.floats('lengths', 1)).toEqual([2]);
      }
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  }
});

it('StreamOrder leaves the tributary and cycle unknown beside a hand-worked outlet (RT-123)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createFixture(device, 6);
  const {grid, graph, input, output} = fixture;
  graph.add(
    new GPUTerrainStreamOrder({
      grid,
      flowDirections: input('directions', 'uint32', Uint32Array.of(1, 16, 16, 0, 1, 0)),
      streams: input('streams', 'uint32', Uint32Array.of(1, 1, 1, 0, 1, 1)),
      output: {
        streamOrder: output('order', 'uint32'),
        status: {
          converged: output('converged', 'uint32', 1),
          iterationLimitReached: output('limit', 'uint32', 1)
        }
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    // The separate leaf and outlet have only one donor chain, hence order 1 throughout.
    expect(await fixture.words('order')).toEqual([0xffffffff, 0xffffffff, 0xffffffff, 0, 1, 1]);
    expect(await fixture.words('converged', 1)).toEqual([0]);
    expect(await fixture.words('limit', 1)).toEqual([0]);
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});

it('Watersheds and HAND reject a cycle with an incoming path beside a genuine no-drainage terminal (RT-123)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createFixture(device, 4);
  const {grid, graph, input, output} = fixture;
  const directions = input('directions', 'uint32', Uint32Array.of(1, 16, 16, 0));
  graph.add(
    new GPUTerrainWatersheds({
      grid,
      flowDirections: directions,
      maximumIterations: 16,
      output: {
        labels: output('labels', 'uint32'),
        status: {
          converged: output('basinConverged', 'uint32', 1),
          iterationLimitReached: output('basinLimit', 'uint32', 1)
        }
      }
    })
  );
  graph.add(
    new GPUTerrainHeightAboveDrainage({
      grid,
      flowDirections: directions,
      maximumIterations: 16,
      elevation: makeGPURasterBand(
        input('elevation', 'float32', Float32Array.of(10, 11, 12, 13)),
        grid
      ),
      streams: input('streams', 'uint32', new Uint32Array(4)),
      output: {
        heightAboveDrainage: output('hand', 'float32'),
        drainageCells: output('drainage', 'uint32'),
        status: {
          converged: output('handConverged', 'uint32', 1),
          iterationLimitReached: output('handLimit', 'uint32', 1)
        }
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    // 0 <-> 1 <- 2 has no original stop; only cell 3 has no receiver. With no streams,
    // its HAND is the documented no-drainage NaN/0xffffffff, while its basin is itself.
    expect(await fixture.words('labels')).toEqual([0xfffffffe, 0xfffffffe, 0xfffffffe, 3]);
    expect(await fixture.words('drainage')).toEqual([
      0xfffffffe, 0xfffffffe, 0xfffffffe, 0xffffffff
    ]);
    expect(await fixture.floats('hand')).toEqual([NaN, NaN, NaN, NaN]);
    for (const name of ['basinConverged', 'basinLimit', 'handConverged', 'handLimit']) {
      expect(await fixture.words(name, 1)).toEqual([0]);
    }
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});

it('StreamOrder reports an odd cycle as structural failure with no iteration limit (B1, RT-123)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createFixture(device, 4, 2);
  const {grid, graph, input, output} = fixture;
  graph.add(
    new GPUTerrainStreamOrder({
      grid,
      flowDirections: input('directions', 'uint32', Uint32Array.of(1, 8, 64, 0)),
      streams: input('streams', 'uint32', new Uint32Array(4).fill(1)),
      output: {
        streamOrder: output('order', 'uint32'),
        status: {
          converged: output('converged', 'uint32', 1),
          iterationLimitReached: output('limit', 'uint32', 1)
        }
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    // 0 -> 1 -> 2 -> 0 is rootless. Cell 3 has no donors or receiver, so its order is 1.
    expect(await fixture.words('order')).toEqual([0xffffffff, 0xffffffff, 0xffffffff, 1]);
    expect(await fixture.words('converged', 1)).toEqual([0]);
    expect(await fixture.words('limit', 1)).toEqual([0]);
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});

it('StreamNetwork suppresses uncertain junction reaches and retains a separate definite reach (B3)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const unknownWord of [0, 0xfffffffe, 0xffffffff]) {
    const fixture = createFixture(device, 12, 6);
    const {grid, graph, input, output} = fixture;
    graph.add(
      new GPUTerrainStreamNetwork({
        grid,
        flowDirections: input(
          'directions',
          'uint32',
          Uint32Array.of(1, 1, 0, 0, 1, 0, 128, 0, 0, 0, 0, 0)
        ),
        streams: input(
          'streams',
          'uint32',
          Uint32Array.of(1, 1, 1, 0, 1, 1, unknownWord, 0, 0, 0, 0, 0)
        ),
        output: {
          sourceCells: output('source', 'uint32'),
          targetCells: output('target', 'uint32'),
          vertexOffsets: output('offsets', 'uint32', 13),
          positions: output('positions', 'float32x2'),
          lengths: output('lengths', 'float32'),
          reachCount: output('reaches', 'uint32', 1),
          vertexCount: output('vertices', 'uint32', 1),
          unresolvedCount: output('unresolved', 'uint32', 1),
          status: {overflow: output('overflow', 'uint32', 1)}
        }
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const unknown = unknownWord !== 0;
      const reachCount = unknown ? 1 : 2;
      // Unknown cell 6 may donate NE to 1. Without that membership, 0->2 is one reach;
      // with it, both endpoints at 1 would be speculative. Suppress those paths, retaining
      // definite reach 4->5 (one unit edge, two vertices). Count 0,1,6 unresolved.
      expect(await fixture.words('unresolved', 1)).toEqual([unknown ? 3 : 0]);
      expect(await fixture.words('reaches', 1)).toEqual([reachCount]);
      expect(await fixture.words('vertices', 1)).toEqual([unknown ? 2 : 5]);
      expect(await fixture.words('overflow', 1)).toEqual([0]);
      expect(await fixture.words('source', reachCount)).toEqual(unknown ? [4] : [0, 4]);
      expect(await fixture.words('target', reachCount)).toEqual(unknown ? [5] : [2, 5]);
      expect(await fixture.words('offsets', reachCount + 1)).toEqual(unknown ? [0, 2] : [0, 3, 5]);
      expect(await fixture.floats('lengths', reachCount)).toEqual(unknown ? [1] : [2, 1]);
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  }
});
