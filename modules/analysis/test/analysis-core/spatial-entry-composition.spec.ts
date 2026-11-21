// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph, GPUGather} from '@luma.gl/graph/gpu-core';
import {
  createOrdinaryLeastSquaresDesign,
  createOrdinaryLeastSquaresJoinedTableMask
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {requireTestDevice} from '../../src/testing/require-test-device';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../utils/gpu-contributor-test-utils';

function createFixture(device: Device) {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  let serial = 0;
  return {
    graph,
    input<Format extends 'uint32' | 'float32'>(format: Format, values: Uint32Array | Float32Array) {
      const buffer = createInputBuffer(device, values);
      buffers.push(buffer);
      return importGraphBuffer(graph, `input-${serial++}`, buffer, format, values.length);
    },
    output<Format extends 'uint32' | 'float32'>(format: Format, length: number) {
      const buffer = createOutputBuffer(device, length);
      buffers.push(buffer);
      return {buffer, view: importGraphBuffer(graph, `output-${serial++}`, buffer, format, length)};
    },
    destroy() {
      for (const buffer of buffers) buffer.destroy();
    }
  };
}

for (const [complete, candidateOverflow, expected] of [
  [1, 0, [1, 0, 1]],
  [0, 1, [0, 0, 0]],
  [0, 0, [0, 0, 0]],
  [1, 1, [0, 0, 0]]
] as const) {
  it(`CMP-01 public joined-table adapter retains complete=${complete}, candidateOverflow=${candidateOverflow}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createFixture(device);
    const {graph, input, output} = fixture;
    const joined = {
      matched: input('uint32', new Uint32Array([1, 0, 1])),
      status: {
        complete: input('uint32', new Uint32Array([complete])),
        candidateOverflow: input('uint32', new Uint32Array([candidateOverflow]))
      }
    };
    const result = createOrdinaryLeastSquaresJoinedTableMask(graph, joined);
    expect(result.status.complete).toBe(joined.status.complete);
    expect(result.status.candidateOverflow).toBe(joined.status.candidateOverflow);
    const mask = output('uint32', 3);
    graph.add(
      new GPUGather({
        source: result.mask,
        indices: input('uint32', new Uint32Array([0, 1, 2])),
        output: mask.view
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      // Hand Boolean conjunction: matched AND complete AND NOT candidateOverflow.
      expect(await readUint32(mask.buffer)).toEqual(expected);
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  });
}

it('CMP-11 public design adapter emits predictor rows in coefficient order', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createFixture(device);
  const {graph, input, output} = fixture;
  const predictors = createOrdinaryLeastSquaresDesign(graph, [
    input('float32', new Float32Array([2, 3, 5])),
    input('float32', new Float32Array([7, 11, 13]))
  ]);
  const rows = output('float32', 6);
  graph.add(
    new GPUGather({
      source: predictors,
      indices: input('uint32', new Uint32Array([0, 1, 2, 3, 4, 5])),
      output: rows.view
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    // Hand interleaving of [x,z] columns gives [x0,z0,x1,z1,x2,z2].
    expect(await readFloat32(rows.buffer)).toEqual([2, 7, 3, 11, 5, 13]);
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});
