// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../utils/require-test-device';
import {createMedialAxisFixture} from './medial-axis-fixture';
import {computeFilledRectangleReference} from './medial-axis-oracle';

// Run with yarn test-benchmarks modules/analysis/test/gpu-raster/medial-axis/gpu-raster-medial-axis-benchmark.spec.ts
const RUN_BENCHMARK =
  (import.meta as unknown as {env?: Record<string, string>}).env?.[
    'VITE_RASTER_MEDIAL_AXIS_BENCHMARK'
  ] === 'true';

/** Background walls isolate 31x31 filled squares, including the final tile's exterior boundary. */
function createInteriorBackgroundCase(size: number) {
  const count = size * size;
  const values = new Uint32Array(count);
  const radius = Array<number>(count).fill(0);
  const medialAxis = Array<number>(count).fill(0);
  const tile = computeFilledRectangleReference(31, 31);
  for (let cell = 0; cell < count; cell++) {
    const column = (cell % size) % 32;
    const row = Math.floor(cell / size) % 32;
    if (column === 0 || row === 0) continue;
    const tileCell = (row - 1) * 31 + column - 1;
    values[cell] = 1;
    radius[cell] = tile.radius[tileCell];
    medialAxis[cell] = tile.medialAxis[tileCell];
  }
  return {values, reference: {radius, medialAxis}};
}

it.skipIf(!RUN_BENCHMARK)(
  'medial axis records completed resident workloads with exterior and interior background',
  {timeout: 120000},
  async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const queue = (device as unknown as {handle: GPUDevice}).handle.queue;
    const cases = [
      {size: 128, label: 'filled-rectangle'},
      {size: 512, label: 'filled-rectangle'},
      {size: 1024, label: 'filled-rectangle'},
      {size: 512, label: '31x31-squares-with-background-walls'},
      {size: 1024, label: '31x31-squares-with-background-walls'}
    ];
    for (const {size, label} of cases) {
      const {values, reference} =
        label === 'filled-rectangle'
          ? {
              values: new Uint32Array(size * size).fill(1),
              reference: computeFilledRectangleReference(size, size)
            }
          : createInteriorBackgroundCase(size);
      const started = performance.now();
      const fixture = createMedialAxisFixture(device, size, size, values);
      try {
        fixture.submit();
        await queue.onSubmittedWorkDone();
        const firstMilliseconds = performance.now() - started;
        const samples: number[] = [];
        for (let iteration = 0; iteration < 5; iteration++) {
          const before = performance.now();
          fixture.submit();
          await queue.onSubmittedWorkDone();
          samples.push(performance.now() - before);
        }
        const actual = await fixture.read();
        expect(actual.medialAxis).toEqual(reference.medialAxis);
        expect(actual.radius).toEqual(reference.radius);
        samples.sort((first, second) => first - second);
        console.log(
          JSON.stringify({
            operation: 'GPURasterMedialAxis',
            device: device.info,
            case: label,
            shape: [size, size],
            size,
            firstMilliseconds,
            residentMedianMilliseconds: samples[2]
          })
        );
      } finally {
        fixture.destroy();
      }
    }
  }
);
