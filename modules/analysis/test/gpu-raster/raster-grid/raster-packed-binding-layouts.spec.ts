// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPURasterReconstruction} from '../../../src/raster-analysis/gpu-raster-reconstruction';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createInputBuffer, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it('uses one physical import for disjoint writable outputs and reads a native band in an earlier dispatch', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const sentinel = 0xdeadbeef;
  const initial = new Uint32Array(192).fill(sentinel);
  initial.set([2, 0, 0, 0, 0]);
  const packed = device.createBuffer({
    data: initial,
    usage: Buffer.STORAGE | Buffer.COPY_DST | Buffer.COPY_SRC
  });
  const constraint = createInputBuffer(device, Uint32Array.of(3, 3, 3, 3, 3));
  const graph = new GPUCommandGraph(device);
  // Import once: graph.compile rejects two IDs resolving to one physical buffer.
  const imported = importGraphBuffer(graph, 'packed', packed, 'uint32', initial.length);
  const marker = graph.createDataView(imported.buffer, {format: 'uint32', length: 5});
  const values = graph.createDataView(imported.buffer, {
    format: 'float32',
    byteOffset: 256,
    length: 5
  });
  const validity = graph.createDataView(imported.buffer, {
    format: 'uint32',
    byteOffset: 512,
    length: 5
  });
  const converged = graph.createDataView(imported.buffer, {
    format: 'uint32',
    byteOffset: 20,
    length: 1
  });
  graph.add(
    new GPURasterReconstruction({
      grid: {width: 5, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
      marker: {id: 'marker', format: 'uint32', storage: {kind: 'buffer', values: marker}, scale: 2},
      mask: {
        id: 'constraint',
        format: 'uint32',
        storage: {
          kind: 'buffer',
          values: importGraphBuffer(graph, 'constraint', constraint, 'uint32', 5)
        },
        scale: 2
      },
      maximumIterations: 16,
      output: {values, validity, status: {converged}}
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const words = await readUint32(packed, initial.length);
    const floats = new Float32Array(Uint32Array.from(words).buffer);
    expect(Array.from(floats.slice(64, 69))).toEqual([4, 4, 4, 4, 4]);
    expect(words.slice(128, 133)).toEqual([1, 1, 1, 1, 1]);
    expect(words.slice(0, 5)).toEqual([2, 0, 0, 0, 0]);
    expect(words[5]).toBe(1);
    for (let index = 6; index < words.length; index++) {
      if ((index >= 64 && index < 69) || (index >= 128 && index < 133)) continue;
      expect(words[index], `sentinel word ${index}`).toBe(sentinel);
    }
  } finally {
    compiled.destroy();
    packed.destroy();
    constraint.destroy();
  }
});
