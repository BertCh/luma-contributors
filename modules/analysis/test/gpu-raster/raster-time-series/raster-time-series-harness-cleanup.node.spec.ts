// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it, vi} from 'vitest';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createRasterTimeSeriesHarness} from './raster-time-series-test-utils';

it('destroys all harness buffers once when setup ends before run', () => {
  const device = createNullWebGPUDevice();
  const createBuffer = vi.spyOn(device, 'createBuffer');
  const harness = createRasterTimeSeriesHarness(device);
  harness.floats([1]);
  harness.output('result', 'float32', 1);
  expect(createBuffer).toHaveBeenCalledTimes(2);
  const destroyCalls = createBuffer.mock.results.map(({value}) => vi.spyOn(value, 'destroy'));

  harness.destroy();
  harness.destroy();

  for (const destroy of destroyCalls) expect(destroy).toHaveBeenCalledTimes(1);
});

it('releases harness buffers when graph compilation fails', async () => {
  const device = createNullWebGPUDevice();
  const createBuffer = vi.spyOn(device, 'createBuffer');
  const harness = createRasterTimeSeriesHarness(device);
  harness.integers([1]);
  const failure = new Error('test compilation failure');
  vi.spyOn(harness.graph, 'compile').mockImplementationOnce(() => {
    throw failure;
  });

  await expect(harness.run([])).rejects.toBe(failure);

  expect(createBuffer).toHaveBeenCalledTimes(1);
  expect(createBuffer.mock.results[0].value.destroyed).toBe(true);
});
