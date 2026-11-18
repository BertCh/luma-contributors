// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUSpaceTimeCrossCorrelation} from '../../../src/gpu-spatial-analysis/space-time-statistics/gpu-space-time-cross-correlation';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {RecipeTestFixture} from '../recipes/recipe-harness';

// All combinations of the three independent optional outputs, on reused, prefilled buffers.
it.each(
  Array.from({length: 8}, (_, combination) => combination)
)('writes cross-correlation optional outputs (combination=%s)', async combination => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'cross-correlation-optional');
  let compiled: ReturnType<typeof fixture.graph.compile> | undefined;
  try {
    const first = fixture.output('first', 'float32', 8);
    const second = fixture.input('second', Float32Array.of(1, 2, 3, 4, 1, 2, 3, 4), 'float32', 8);
    const correlations = fixture.output('correlations', 'float32', 2);
    const storage = fixture.output('p-values-and-count', 'float32', 65);
    const pValues = {
      buffer: storage.buffer,
      view: fixture.graph.createDataView(storage.view.buffer, {format: 'float32', length: 2})
    };
    const adjustedPValues = combination & 1 ? fixture.output('adjusted', 'float32', 2) : undefined;
    const testedCount =
      combination & 2
        ? fixture.graph.createDataView(storage.view.buffer, {
            format: 'uint32',
            byteOffset: 256,
            length: 1
          })
        : undefined;
    const pairCounts = combination & 4 ? fixture.output('pair-counts', 'uint32', 2) : undefined;
    adjustedPValues?.buffer.write(Float32Array.of(99, 99));
    if (testedCount) storage.buffer.write(Uint32Array.of(99), 256);
    pairCounts?.buffer.write(Uint32Array.of(99, 99));
    fixture.graph.add(
      new GPUSpaceTimeCrossCorrelation({
        first: first.view,
        second,
        layout: 'cell-major',
        cellCount: 2,
        periodCount: 4,
        lags: [0],
        output: {
          correlations: correlations.view,
          pValues: pValues.view,
          adjustedPValues: adjustedPValues?.view,
          testedCount,
          pairCounts: pairCounts?.view
        }
      })
    );
    compiled = fixture.graph.compile();
    for (const count of [1, 0]) {
      first.buffer.write(
        count ? Float32Array.of(1, 2, 3, 4, 2, 2, 2, 2) : new Float32Array(8).fill(2)
      );
      submitGraph(device, compiled, undefined);
      if (testedCount) expect((await fixture.readUint32(storage, 65))[64]).toBe(count);
      if (pairCounts) expect(await fixture.readUint32(pairCounts, 2)).toEqual([4, 4]);
      const raw = await fixture.readFloat32(pValues, 2);
      const adjusted = adjustedPValues ? await fixture.readFloat32(adjustedPValues, 2) : undefined;
      expect(raw[1]).toBeNaN();
      if (adjusted) expect(adjusted[1]).toBeNaN();
      if (count) {
        expect(raw[0]).toBeCloseTo(0, 5);
        if (adjusted) expect(adjusted[0]).toBeCloseTo(0, 5);
      } else {
        expect(raw[0]).toBeNaN();
        if (adjusted) expect(adjusted[0]).toBeNaN();
      }
    }
  } finally {
    compiled?.destroy();
    fixture.destroy();
  }
});
