// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUPointToCellOperation} from '../../../src/gpu-spatial-analysis/cell-indexing/cpu-point-to-cell-operation';
import {
  GPUPointToCellOperation,
  type PointToCellOperationProps
} from '../../../src/gpu-spatial-analysis/cell-indexing/gpu-point-to-cell-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const family of ['quadbin', 'quadkey', 'geohash', 's2', 'h3'] as const) {
  const resolutions =
    family === 'quadbin' ? [3, 26] : family === 'quadkey' ? [3, 26, 29] : [family === 'h3' ? 2 : 3];
  for (const resolution of resolutions) {
    for (const edgeCase of [false, true]) {
      it(`${family} operations agree exactly at resolution ${resolution} on ${edgeCase ? 'masked, clipped and non-finite' : 'ordinary multi-face'} rows`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        // H3 and S2 use coarse levels to avoid their documented precision disagreements.
        // The integer-defined families also exercise their finest supported tile resolutions.
        // Comparisons are exact integer comparisons: neither a key word nor validity has a tolerance.
        const positions = edgeCase
          ? Float32Array.of(-180, 12, 180, 12, 0, 90, 90, -90, 15, 32, NaN, 20, 20, Infinity)
          : Float32Array.of(
              -122.4194,
              37.7749,
              10.40744,
              57.64911,
              151.2093,
              -33.8688,
              -70.6693,
              -33.4489,
              77.209,
              28.6139,
              132.25,
              72.5
            );
        const mask = new Uint32Array(positions.length / 2).fill(3);
        if (edgeCase) mask[4] = 0;
        const props: PointToCellOperationProps = {
          family,
          resolution,
          rowCount: mask.length,
          positions: 'positions',
          mask: 'mask',
          output: {cells: 'cells', validity: 'validity'}
        };
        const results = [];
        for (const Operation of [GPUPointToCellOperation, CPUPointToCellOperation]) {
          results.push(
            await runOperationGraph(device, {
              inputs: {
                positions: {format: 'float32x2', values: positions},
                mask: {format: 'uint32', values: mask}
              },
              operations: [new Operation(props)],
              outputs: ['cells', 'validity']
            })
          );
        }
        for (const output of ['cells', 'validity'])
          expect(Array.from(results[0][output])).toEqual(Array.from(results[1][output]));
        if (edgeCase) expect(Array.from(results[0]['validity'])).toEqual([1, 1, 1, 1, 0, 0, 0]);
        else {
          expect(Array.from(results[0]['validity'])).toEqual(new Array(mask.length).fill(1));
          expect(new Set(Array.from(results[0]['cells'])).size).toBeGreaterThan(3);
        }
      });
    }
  }
}

it('both operations support omitted mask and validity ports', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props: PointToCellOperationProps = {
    family: 'h3',
    precision: 'double-single',
    resolution: 12,
    rowCount: 3,
    positions: 'positions',
    output: {cells: 'cells'}
  };
  const results = [];
  for (const Operation of [GPUPointToCellOperation, CPUPointToCellOperation])
    results.push(
      await runOperationGraph(device, {
        inputs: {
          positions: {
            format: 'float32x2',
            values: Float32Array.of(-122.4194, 37.7749, 10.40744, 57.64911, 151.2093, -33.8688)
          }
        },
        operations: [new Operation(props)],
        outputs: ['cells']
      })
    );
  expect(Array.from(results[0]['cells'])).toEqual(Array.from(results[1]['cells']));
});

for (const [family, expected] of [
  ['quadbin', 0x49a374964e3145ecn],
  ['quadkey', 0x680374964e3145ecn]
] as const) {
  it(`${family} operations agree on the integer-defined resolution-26 counterexample`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const props: PointToCellOperationProps = {
      family,
      resolution: 26,
      rowCount: 1,
      positions: 'positions',
      output: {cells: 'cells', validity: 'validity'}
    };
    const results = [];
    for (const Operation of [GPUPointToCellOperation, CPUPointToCellOperation]) {
      const result = await runOperationGraph(device, {
        inputs: {
          positions: {
            format: 'float32x2',
            values: Float32Array.of(-8.913032531738281, 53.91400146484375)
          }
        },
        operations: [new Operation(props)],
        outputs: ['cells', 'validity']
      });
      // The expected key follows integer arithmetic; this case observes that prediction on-device.
      expect(Array.from(result['cells'])).toEqual([
        Number(expected & 0xffffffffn),
        Number(expected >> 32n)
      ]);
      expect(Array.from(result['validity'])).toEqual([1]);
      results.push(result);
    }
    for (const output of ['cells', 'validity'])
      expect(Array.from(results[0][output])).toEqual(Array.from(results[1][output]));
  });
}

it('the H3 operations check the formerly omitted resolution-zero boundary row', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props: PointToCellOperationProps = {
    family: 'h3',
    precision: 'double-single',
    resolution: 0,
    rowCount: 1,
    positions: 'positions',
    output: {cells: 'cells', validity: 'validity'}
  };
  const expected = (1n << 59n) | (40n << 45n) | ((1n << 45n) - 1n);
  const results = [];
  for (const Operation of [GPUPointToCellOperation, CPUPointToCellOperation]) {
    const result = await runOperationGraph(device, {
      inputs: {
        positions: {
          format: 'float32x2',
          values: Float32Array.of(-121.33663177490234, 28.653018951416016)
        }
      },
      operations: [new Operation(props)],
      outputs: ['cells', 'validity']
    });
    expect(Array.from(result['cells'])).toEqual([
      Number(expected & 0xffffffffn),
      Number(expected >> 32n)
    ]);
    expect(Array.from(result['validity'])).toEqual([1]);
    results.push(result);
  }
  for (const output of ['cells', 'validity'])
    expect(Array.from(results[0][output])).toEqual(Array.from(results[1][output]));
});
