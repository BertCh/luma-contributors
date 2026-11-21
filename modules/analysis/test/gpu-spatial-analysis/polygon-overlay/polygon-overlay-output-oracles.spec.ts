// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../utils/require-test-device';
import {createPolygonOverlayOutputOracle} from './polygon-overlay-output-oracles';
import {
  expectPolygonOverlayBoundaryOracle,
  expectPolygonOverlayNodingDiagnostics
} from './polygon-overlay-output-oracles.expect';

it('preserves nonzero fill from an inherited getter on a twice-traced square', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Winding is two throughout [0,4]^2: nonzero retains one square; even-odd would drop it.
  for (const origin of ['own', 'inherited'] as const) {
    const fixture = createPolygonOverlayOutputOracle(device, false, origin);
    try {
      const result = await fixture.runGPU();
      expect(result['boundaryCount']).toEqual([4]);
      expect(result['count']).toEqual([1]);
      expect(result['candidateOverflow']).toEqual([0]);
      expect(result['overflow']).toEqual([0]);
      expect(
        Array.from({length: 4}, (_, row) => result['positions'].slice(row * 2, row * 2 + 2)).sort()
      ).toEqual(
        [
          [0, 0],
          [4, 0],
          [4, 4],
          [0, 4]
        ].sort()
      );
    } finally {
      fixture.destroy();
    }
  }
});
it('publishes hand-derived boundary owners with uncertainty alone and absent', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const uncertainty of [true, false]) {
    const fixture = createPolygonOverlayOutputOracle(device, uncertainty);
    try {
      for (const usable of [true, false, true]) {
        for (const entry of fixture.columns.filter(
          entry => entry.name === 'left' || entry.name === 'right'
        ))
          entry.buffer.write(
            usable ? entry.values : new Float32Array(entry.values.length).fill(Number.NaN)
          );
        for (const entry of fixture.columns.filter(
          entry =>
            entry.output &&
            (entry.name.startsWith('boundary') ||
              entry.name === 'uncertainCount' ||
              entry.name === 'candidateOverflow')
        ))
          entry.buffer.write(entry.values);
        const result = await fixture.runGPU();
        expectPolygonOverlayNodingDiagnostics(result, usable);
        expect(result['candidateOverflow']).toEqual([usable ? 0 : 1]);
        expect(result['overflow']).toEqual([0]);
        if (!usable) {
          expect(result['count']).toEqual([0]);
          continue;
        }
        expectPolygonOverlayBoundaryOracle(result);
        expect(result['count']).toEqual([1]);
        expect(result['overflow']).toEqual([0]);
        expect(result['candidateOverflow']).toEqual([0]);
        expect(result['boundaryCount']).toEqual([4]);
        expect(result['boundaryRequiredCount']).toEqual([4]);
        expect(result['boundaryOverflow']).toEqual([0]);
        expect(result['boundaryCandidateOverflow']).toEqual([0]);
        expect(result['features']).toEqual([0, 1]);
        expect(result['polygons']).toEqual([0, 1]);
        expect(result['rings']).toEqual([0, 4]);
        expect(
          Array.from({length: 4}, (_, row) =>
            result['positions'].slice(row * 2, row * 2 + 2)
          ).sort()
        ).toEqual(
          [
            [2, 2],
            [4, 2],
            [4, 4],
            [2, 4]
          ].sort()
        );
        if (uncertainty) expect(result['uncertainCount']).toEqual([0]);
      }
    } finally {
      fixture.destroy();
    }
  }
});
