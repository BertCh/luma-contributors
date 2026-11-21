// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {expectUrbanFloat32Output, expectUrbanMissingValues} from './urban-ecology-assertions';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('S2 extra logical output rows cannot hide behind a shortened readback', async () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const buffer = device.createBuffer({byteLength: 32});
  try {
    const output = {buffer, view: createTransientView(graph, 'two-rows', 'float32x4', 2)};
    const fixture = {
      readFloat32: async (_output: {buffer: typeof buffer}, count: number) =>
        new Array<number>(8).fill(1).slice(0, count)
    };
    await expect(expectUrbanFloat32Output(fixture, output, [1, 1, 1, 1])).rejects.toThrow();
    await expectUrbanFloat32Output(fixture, output, new Array<number>(8).fill(1));
  } finally {
    buffer.destroy();
    device.destroy();
  }
});

it.each([
  {actual: []},
  {actual: [NaN]},
  {actual: [NaN, NaN, NaN]}
])('S2 missing rows must include every expected NaN ($actual)', ({actual}) => {
  expect(() => expectUrbanMissingValues(actual, 4)).toThrow();
  expectUrbanMissingValues([NaN, NaN, NaN, NaN], 4);
});
