// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {createMomentsNode} from '../../../src/gpu-spatial-analysis/shape-descriptors/shape-descriptors-kernels';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

// Math correctness review 2026-11-10, follow-up on compensated sums. The moments kernel keeps
// (sum, compensation) pairs. Written as the float Neumaier term `(sum - total) + value`, the
// compensation is algebraically zero and the test GPU's shader compiler (Metal, fast math) folds
// it away, leaving the plain f32 sum.
//
// The ring walks a triangle 100,000 times. Relative to the given centroid its vertices are
// (0, 0), (4001, 0), (0, 3989), so every loop adds one cross product c = 4001 * 3989 = 15,959,989
// (exact in f32) and, for the second moments, c * 4001^2, c * 3989^2 and c * c, each rounded once
// to f32. The binary64 totals are loops times those constants. A plain f32 running sum of
// 100,000 equal terms of this size is 1e-4 to 1e-3 off (1.2e-3 for the area).
const ANCHOR = [100000, 200000];
const FIRST_LEG = 4001;
const SECOND_LEG = 3989;
const LOOPS = 100000;

it('shape descriptor moments carry the rounding error of every addition', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const positions = new Float32Array(6 * LOOPS);
  for (let loop = 0; loop < LOOPS; loop++) {
    positions.set(
      [ANCHOR[0], ANCHOR[1], ANCHOR[0] + FIRST_LEG, ANCHOR[1], ANCHOR[0], ANCHOR[1] + SECOND_LEG],
      6 * loop
    );
  }
  const rowCount = positions.length / 2;
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const graph = new GPUCommandGraph(device, {id: 'moments-summation'});
  const importOffsets = (name: string, values: number[]) =>
    importGraphBuffer(
      graph,
      name,
      track(createInputBuffer(device, new Uint32Array(values))),
      'uint32',
      values.length
    );
  const moments = track(createOutputBuffer(device, 4));
  graph.add(
    createMomentsNode(graph, {
      id: 'moments',
      operation: 'GPUShapeDescriptors',
      positions: importGraphBuffer(
        graph,
        'positions',
        track(createInputBuffer(device, positions)),
        'float32x2',
        rowCount
      ),
      ringOffsets: importOffsets('ring-offsets', [0, rowCount]),
      polygonOffsets: importOffsets('polygon-offsets', [0, 1]),
      featureOffsets: importOffsets('feature-offsets', [0, 1]),
      featureCount: 1,
      centroids: importGraphBuffer(
        graph,
        'centroids',
        track(createInputBuffer(device, new Float32Array(ANCHOR))),
        'float32x2',
        1
      ),
      moments: importGraphBuffer(graph, 'moments', moments, 'float32x4', 1)
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const [momentXX, momentYY, momentXY, firstArea] = await readFloat32(moments, 4);
    const cross = FIRST_LEG * SECOND_LEG;
    expect(Math.abs(firstArea / ((LOOPS * cross) / 2) - 1), 'area').toBeLessThan(2e-6);
    expect(
      Math.abs(momentXX / ((LOOPS * Math.fround(cross * FIRST_LEG ** 2)) / 12) - 1),
      'xx'
    ).toBeLessThan(2e-6);
    expect(
      Math.abs(momentYY / ((LOOPS * Math.fround(cross * SECOND_LEG ** 2)) / 12) - 1),
      'yy'
    ).toBeLessThan(2e-6);
    expect(Math.abs(momentXY / ((LOOPS * Math.fround(cross * cross)) / 24) - 1), 'xy').toBeLessThan(
      2e-6
    );
  } finally {
    compiled.destroy();
    for (const buffer of buffers) {
      buffer.destroy();
    }
  }
}, 120000);
