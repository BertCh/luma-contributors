// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPUBrownianBridgeUtilizationOperation} from '../../../src/gpu-spatial-analysis/trajectory-home-range/gpu-brownian-bridge-utilization-operation';
import {CPUBrownianBridgeUtilizationOperation} from '../../../src/gpu-spatial-analysis/trajectory-home-range/cpu-brownian-bridge-utilization-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {GPUBrownianBridgeUtilization} from '../../../src/gpu-spatial-analysis/trajectory-home-range/gpu-brownian-bridge-utilization';
import {RecipeTestFixture} from '../recipes/recipe-harness';
import {createEmptyBridgeInputs} from './empty-bridge-inputs.test-utils';

it('B2 bridge preserves a small displacement from the centre of a distant-minimum grid', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const fixture of [
    {positions: Float32Array.of(0.5, 0, 0.5, 0), columns: 1, expected: [0.46839866]},
    {positions: Float32Array.of(-1e7, 0, 1e7 + 1, 0), columns: 1, expected: [0.46839866]},
    // First centre is -20,000,000/3: offset=1/6, density=(4/pi)*exp(-1/9).
    {
      positions: Float32Array.of(-6_666_666.5, 0, -6_666_666.5, 0),
      columns: 3,
      expected: [1.1393448043518524, 0, 0]
    }
  ])
    for (const Operation of [
      CPUBrownianBridgeUtilizationOperation,
      GPUBrownianBridgeUtilizationOperation
    ]) {
      const result = await runOperationGraph(device, {
        inputs: {
          positions: {format: 'float32x2', values: fixture.positions},
          timestamps: {format: 'float32', values: Float32Array.of(0, 1)},
          trackOffsets: {format: 'uint32', values: Uint32Array.of(0, 2)},
          motionVariances: {format: 'float32', values: new Float32Array(2)},
          locationErrors: {format: 'float32', values: Float32Array.of(0.5, 0.5)},
          parameters: {format: 'float32', values: Float32Array.of(-1e7, -1, 1e7, 1)}
        },
        operations: [
          new Operation({
            positions: 'positions',
            timestamps: 'timestamps',
            trackOffsets: 'trackOffsets',
            motionVariances: 'motionVariances',
            locationErrors: 'locationErrors',
            parameters: 'parameters',
            rowCount: 2,
            gridSize: [fixture.columns, 1],
            subStepCount: 1,
            output: {density: 'density'}
          })
        ],
        outputs: ['density']
      });
      expect(result['density'].length).toBe(fixture.columns);
      for (const [cell, value] of fixture.expected.entries()) {
        expect(Number.isFinite(result['density'][cell])).toBe(true);
        expect(result['density'][cell]).toBeCloseTo(value, 6);
      }
    }
});

it('B9 empty bridge overwrites caller-owned nonzero density on both offset shapes', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const trackOffsets of [Uint32Array.of(0), Uint32Array.of(0, 0)]) {
    const fixture = new RecipeTestFixture(device, 'empty-overwrite');
    try {
      const density = fixture.output('density', 'float32', 6);
      density.buffer.write(Float32Array.of(1, 2, 3, 4, 5, 6));
      fixture.graph.add(
        new GPUBrownianBridgeUtilization({
          ...createEmptyBridgeInputs(fixture, trackOffsets),
          gridSize: [3, 2],
          output: {density: density.view}
        })
      );
      fixture.run();
      expect(await fixture.readFloat32(density, 6)).toEqual([0, 0, 0, 0, 0, 0]);
    } finally {
      fixture.destroy();
    }
  }
});

it('B4 seven-sigma squared cutoff agrees at the boundary and adjacent cells', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const variance = 2 * 2 ** -32;
  const radius = Math.fround(7 * Math.fround(Math.sqrt(variance)));
  const halfWidth = 2 ** -36;
  for (const Operation of [
    CPUBrownianBridgeUtilizationOperation,
    GPUBrownianBridgeUtilizationOperation
  ]) {
    const result = await runOperationGraph(device, {
      inputs: {
        positions: {format: 'float32x2', values: new Float32Array(4)},
        timestamps: {format: 'float32', values: Float32Array.of(0, 1)},
        trackOffsets: {format: 'uint32', values: Uint32Array.of(0, 2)},
        motionVariances: {format: 'float32', values: Float32Array.of(4 * variance, 0)},
        locationErrors: {format: 'float32', values: new Float32Array(2)},
        parameters: {
          format: 'float32',
          values: Float32Array.of(
            radius - 3 * halfWidth,
            -(2 ** -16),
            radius + 3 * halfWidth,
            2 ** -16
          )
        }
      },
      operations: [
        new Operation({
          positions: 'positions',
          timestamps: 'timestamps',
          trackOffsets: 'trackOffsets',
          motionVariances: 'motionVariances',
          locationErrors: 'locationErrors',
          parameters: 'parameters',
          rowCount: 2,
          gridSize: [3, 1],
          subStepCount: 1,
          tailPolicy: 'truncate-at-seven-sigma',
          output: {density: 'density'}
        })
      ],
      outputs: ['density']
    });
    const expected = [0.007825982747350383, 0.007825908863430778, 0];
    expect(result['density'].length).toBe(3);
    for (const [cell, value] of expected.entries()) {
      expect(Number.isFinite(result['density'][cell])).toBe(true);
      expect(Math.abs(result['density'][cell] - value)).toBeLessThan(2e-7);
    }
  }
});

it('FLT-08 bridge density is invariant under a representable translation', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const origin of [0, 10_000_000])
    for (const Operation of [
      CPUBrownianBridgeUtilizationOperation,
      GPUBrownianBridgeUtilizationOperation
    ]) {
      const result = await runOperationGraph(device, {
        inputs: {
          positions: {format: 'float32x2', values: Float32Array.of(origin, 0.5, origin, 0.5)},
          timestamps: {format: 'float32', values: Float32Array.of(0, 1)},
          trackOffsets: {format: 'uint32', values: Uint32Array.of(0, 2)},
          motionVariances: {format: 'float32', values: Float32Array.of(0, 0)},
          locationErrors: {format: 'float32', values: Float32Array.of(0.5, 0.5)},
          parameters: {format: 'float32', values: Float32Array.of(origin, 0, origin + 1, 1)}
        },
        operations: [
          new Operation({
            positions: 'positions',
            timestamps: 'timestamps',
            trackOffsets: 'trackOffsets',
            motionVariances: 'motionVariances',
            locationErrors: 'locationErrors',
            parameters: 'parameters',
            rowCount: 2,
            gridSize: [1, 1],
            subStepCount: 1,
            output: {density: 'density'}
          })
        ],
        outputs: ['density']
      });
      // Midpoint variance = 1/8, squared distance = 1/4: exp(-1)/(pi/4).
      expect(result['density'].length).toBe(1);
      expect(Number.isFinite(result['density'][0])).toBe(true);
      expect(result['density'][0]).toBeCloseTo(0.46839866, 6);
    }
});

it('W3-03 bridge named forms keep a 3 by 2 raster for zero tracks and one empty track', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const trackOffsets of [Uint32Array.of(0), Uint32Array.of(0, 0)]) {
    for (const Operation of [
      CPUBrownianBridgeUtilizationOperation,
      GPUBrownianBridgeUtilizationOperation
    ]) {
      const result = await runOperationGraph(device, {
        inputs: {
          positions: {format: 'float32x2', values: new Float32Array()},
          timestamps: {format: 'float32', values: new Float32Array()},
          trackOffsets: {format: 'uint32', values: trackOffsets},
          motionVariances: {format: 'float32', values: new Float32Array()},
          locationErrors: {format: 'float32', values: new Float32Array()},
          parameters: {format: 'float32', values: Float32Array.of(0, 0, 1, 1)}
        },
        operations: [
          new Operation({
            positions: 'positions',
            timestamps: 'timestamps',
            trackOffsets: 'trackOffsets',
            motionVariances: 'motionVariances',
            locationErrors: 'locationErrors',
            parameters: 'parameters',
            rowCount: 0,
            gridSize: [3, 2],
            output: {density: 'density'}
          })
        ],
        outputs: ['density']
      });
      expect(Array.from(result['density'])).toEqual([0, 0, 0, 0, 0, 0]);
    }
  }
});

it('TWN-13 explicit tail policies agree on the 7.5-sigma cell across backends', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const scale of [1, 2 ** -16]) {
    for (const tailPolicy of [undefined, 'full', 'truncate-at-seven-sigma'] as const) {
      for (const Operation of [
        CPUBrownianBridgeUtilizationOperation,
        GPUBrownianBridgeUtilizationOperation
      ]) {
        const result = await runOperationGraph(device, {
          inputs: {
            positions: {format: 'float32x2', values: new Float32Array(4)},
            timestamps: {format: 'float32', values: Float32Array.of(0, 1)},
            trackOffsets: {format: 'uint32', values: Uint32Array.of(0, 2)},
            motionVariances: {format: 'float32', values: Float32Array.of(4 * scale * scale, 0)},
            locationErrors: {format: 'float32', values: new Float32Array(2)},
            parameters: {
              format: 'float32',
              values: Float32Array.of(7 * scale, -scale, 8 * scale, scale)
            }
          },
          operations: [
            new Operation({
              positions: 'positions',
              timestamps: 'timestamps',
              trackOffsets: 'trackOffsets',
              motionVariances: 'motionVariances',
              locationErrors: 'locationErrors',
              parameters: 'parameters',
              rowCount: 2,
              gridSize: [1, 1],
              subStepCount: 1,
              tailPolicy,
              output: {density: 'density'}
            })
          ],
          outputs: ['density']
        });
        const value = result['density'][0];
        expect(result['density'].length).toBe(1);
        expect(Number.isFinite(value)).toBe(true);
        if (
          tailPolicy === 'truncate-at-seven-sigma' ||
          (tailPolicy === undefined && Operation === GPUBrownianBridgeUtilizationOperation)
        )
          expect(value).toBe(0);
        else {
          // Python math: exp(-7.5**2/2)/(2*pi*s*s).
          const expected = scale === 1 ? 9.711533846746243e-14 : 0.0004171072026577219;
          expect(Math.abs(value / expected - 1)).toBeLessThan(2e-5);
        }
      }
    }
  }
});
