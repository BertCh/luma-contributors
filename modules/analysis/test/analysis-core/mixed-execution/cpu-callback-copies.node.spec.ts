// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUGraph, GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {CPUData, CPUVector} from '@luma.gl/graph/gpu-data';
import {expect, it, vi} from 'vitest';
import {getCPUOperationFloat32Rows} from '../../../src/analysis-core/operations/cpu-operation-rows';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it.each([
  1, 2
])('distinguishes executor snapshots from helper packing for %i CPU chunks', async chunkCount => {
  const device = createNullWebGPUDevice();
  const graph = new GPUGraph();
  const arrays = Array.from({length: chunkCount}, (_, index) =>
    Float32Array.of(index + 1, index + 2)
  );
  const values = new CPUVector({
    format: 'float32',
    data: arrays.map(buffer => new CPUData({buffer, format: 'float32', length: 2}))
  });
  for (const [index, chunk] of values.data.entries()) {
    expect(chunk.buffer).toBe(arrays[index].buffer);
  }
  graph.addInput(
    'values',
    gpuGraph.vector('float32', {
      length: chunkCount * 2,
      chunkLengths: arrays.map(array => array.length)
    })
  );
  let called = false;
  graph.add(
    new GPUCPUOperation({
      id: 'cpu-copy-probe',
      inputs: {values: 'values'},
      outputs: {result: gpuGraph.vector('float32', {length: chunkCount * 2})},
      execute({inputs}) {
        called = true;
        const input = inputs['values'];
        if (typeof input === 'number') throw new Error('expected a vector');
        for (const [index, chunk] of input.chunks.entries()) {
          expect(chunk.values.buffer).not.toBe(arrays[index].buffer);
          expect(Array.from(chunk.values)).toEqual(Array.from(arrays[index]));
        }
        const pack = vi.spyOn(input, 'pack');
        const rows = getCPUOperationFloat32Rows('cpu-copy-probe', inputs, 'values');
        expect(pack).toHaveBeenCalledTimes(chunkCount === 1 ? 0 : 1);
        if (chunkCount === 1) expect(rows).toBe(input.chunks[0].values);
        else expect(input.chunks.every(chunk => rows.buffer !== chunk.values.buffer)).toBe(true);
        expect(rows.buffer).not.toBe(arrays[0].buffer);
        return {result: rows};
      }
    })
  );
  const compiled = graph.compile(device, {outputs: ['result'], outputLocations: {result: 'cpu'}});
  try {
    await compiled.run({inputs: {values}});
    expect(called).toBe(true);
  } finally {
    compiled.destroy();
    device.destroy();
  }
});
