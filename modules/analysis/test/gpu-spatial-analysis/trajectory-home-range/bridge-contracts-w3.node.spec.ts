// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getGPUBrownianBridgeUtilizationParameterValues} from '../../../src/gpu-spatial-analysis/trajectory-home-range/gpu-brownian-bridge-utilization';
import {
  computeBrownianBridgeUtilizationOnCPU,
  type CPUBrownianBridgeUtilizationResult,
  type BrownianBridgeTailPolicy
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {CPUBrownianBridgeUtilizationOperation} from '../../../src/gpu-spatial-analysis/trajectory-home-range/cpu-brownian-bridge-utilization-operation';
import {GPUBrownianBridgeUtilizationOperation} from '../../../src/gpu-spatial-analysis/trajectory-home-range/gpu-brownian-bridge-utilization-operation';
import {GPUCommandGraph, createTransientView, type GPUCPUVector} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {getBrownianBridgeCellFixOffset} from '../../../src/gpu-spatial-analysis/trajectory-home-range/brownian-bridge-coordinates';

it('UCT-07 validates encoded extents before modifying the target', () => {
  const longTarget = Float32Array.of(9, 8, 7, 6, 55);
  getGPUBrownianBridgeUtilizationParameterValues(
    {extent: [0, 0, 1, 1, 33] as unknown as [number, number, number, number]},
    longTarget
  );
  expect(Array.from(longTarget)).toEqual([0, 0, 1, 1, 55]);
  expect(() =>
    getGPUBrownianBridgeUtilizationParameterValues(
      {extent: ['0', 0, 1, 1] as unknown as [number, number, number, number]},
      longTarget
    )
  ).toThrow(/extent/);
  expect(Array.from(longTarget)).toEqual([0, 0, 1, 1, 55]);
  for (const extent of [
    [5_000_000, 0, 5_000_000.1, 1],
    [0, 5_000_000, 1, 5_000_000.1],
    [0, 0, 1e40, 1]
  ] as const) {
    const target = Float32Array.of(9, 8, 7, 6);
    expect(() => getGPUBrownianBridgeUtilizationParameterValues({extent}, target)).toThrow(
      /extent.*float32.*subtract.*origin/
    );
    expect(Array.from(target)).toEqual([9, 8, 7, 6]);
  }
  const local = getGPUBrownianBridgeUtilizationParameterValues({extent: [0, 0, 0.1, 1]});
  expect(Array.from(local)).toEqual([0, 0, Math.fround(0.1), 1]);
});

it('B2 compensated cell offsets preserve both narrow translations and distant grid minima', () => {
  expect(getBrownianBridgeCellFixOffset(10_000_000, 10_000_000, 10_000_001, 0, 1)).toBe(-0.5);
  expect(getBrownianBridgeCellFixOffset(0.5, -10_000_000, 10_000_000, 0, 1)).toBe(0.5);
  // The first of three cells is centred at -20,000,000/3; its fix is exactly 1/6 above it.
  expect(getBrownianBridgeCellFixOffset(-6_666_666.5, -10_000_000, 10_000_000, 0, 3)).toBeCloseTo(
    1 / 6,
    6
  );
});

it('UCT-15 coarse cell-centre quadrature can exceed unit captured mass', () => {
  const result = computeBrownianBridgeUtilizationOnCPU(
    {
      positions: new Float32Array(4),
      timestamps: Float32Array.of(0, 1),
      trackOffsets: Uint32Array.of(0, 2),
      motionVariances: new Float32Array(2),
      locationErrors: Float32Array.of(1, 1),
      parameters: Float32Array.of(-5, -5, 5, 5)
    },
    {gridSize: [1, 1], subStepCount: 1}
  );
  // Midpoint variance=1/2, centre density=1/pi, cell area=100.
  expect(Number.isFinite(result.density[0])).toBe(true);
  expect(result.density[0]).toBeCloseTo(0.3183098861837907, 7);
  expect(result.density[0] * 100).toBeGreaterThan(31.8);
});

it('W3-03 preserves the requested raster for zero tracks and one empty track in named forms', async () => {
  const device = createNullWebGPUDevice();
  try {
    for (const trackOffsets of [Uint32Array.of(0), Uint32Array.of(0, 0)]) {
      const tailPolicy: BrownianBridgeTailPolicy = 'full';
      const props = {
        tailPolicy,
        positions: 'positions',
        timestamps: 'timestamps',
        trackOffsets: 'trackOffsets',
        motionVariances: 'motionVariances',
        locationErrors: 'locationErrors',
        parameters: 'parameters',
        rowCount: 0,
        gridSize: [3, 2] as const,
        output: {density: 'density'}
      };
      const operation = new CPUBrownianBridgeUtilizationOperation(props);
      const pure: CPUBrownianBridgeUtilizationResult = computeBrownianBridgeUtilizationOnCPU(
        {
          positions: new Float32Array(),
          timestamps: new Float32Array(),
          trackOffsets,
          motionVariances: new Float32Array(),
          locationErrors: new Float32Array(),
          parameters: Float32Array.of(0, 0, 1, 1)
        },
        props
      );
      expect(pure.density).toEqual(new Float32Array(6));
      const vector = (
        format: GPUCPUVector['format'],
        values: Float32Array | Uint32Array
      ): GPUCPUVector => ({
        format,
        length: format === 'float32x2' ? values.length / 2 : values.length,
        chunks: [{values, length: format === 'float32x2' ? values.length / 2 : values.length}],
        pack: () => values
      });
      const result = await operation.execute({
        inputs: {
          positions: vector('float32x2', new Float32Array()),
          timestamps: vector('float32', new Float32Array()),
          trackOffsets: vector('uint32', trackOffsets),
          motionVariances: vector('float32', new Float32Array()),
          locationErrors: vector('float32', new Float32Array()),
          parameters: vector('float32', Float32Array.of(0, 0, 1, 1))
        }
      });
      expect(result['density']).toEqual(new Float32Array(6));
      const graph = new GPUCommandGraph(device);
      const views = {
        positions: createTransientView(graph, 'positions', 'float32x2', 0),
        timestamps: createTransientView(graph, 'timestamps', 'float32', 0),
        trackOffsets: createTransientView(graph, 'trackOffsets', 'uint32', trackOffsets.length),
        motionVariances: createTransientView(graph, 'motionVariances', 'float32', 0),
        locationErrors: createTransientView(graph, 'locationErrors', 'float32', 0),
        parameters: createTransientView(graph, 'parameters', 'float32', 4),
        density: createTransientView(graph, 'density', 'float32', 6)
      };
      const gpu = new GPUBrownianBridgeUtilizationOperation(props);
      const contributor = gpu.createContributor(port => views[port as keyof typeof views] as never);
      expect(contributor.getCommandNodes(graph).length).toBeGreaterThan(0);
      expect(gpu.outputs['density'].length).toBe(6);
      expect(() =>
        gpu.createContributor(port => {
          if (port === 'positions')
            return createTransientView(graph, 'wrong-positions', 'float32x2', 2) as never;
          if (port === 'timestamps' || port === 'motionVariances' || port === 'locationErrors')
            return createTransientView(graph, `wrong-${port}`, 'float32', 2) as never;
          return views[port as keyof typeof views] as never;
        })
      ).toThrow(/rowCount/);
    }
  } finally {
    device.destroy();
  }
});

it('S3 GPU bridge rejects an unknown timestamp format at declaration', () => {
  expect(
    () =>
      new GPUBrownianBridgeUtilizationOperation({
        positions: 'positions',
        timestamps: 'timestamps',
        trackOffsets: 'trackOffsets',
        motionVariances: 'motionVariances',
        locationErrors: 'locationErrors',
        parameters: 'parameters',
        rowCount: 2,
        gridSize: [1, 1],
        timestampFormat: 'nonsense' as 'float32',
        output: {density: 'density'}
      })
  ).toThrow(/timestampFormat.*float32.*uint32x2/);
});

it('B4 truncation includes the rounded seven-sigma boundary and excludes its next cell', () => {
  const variance = 2 * 2 ** -32;
  const radius = Math.fround(7 * Math.fround(Math.sqrt(variance)));
  const halfWidth = 2 ** -36;
  const {density} = computeBrownianBridgeUtilizationOnCPU(
    {
      positions: new Float32Array(4),
      timestamps: Float32Array.of(0, 1),
      trackOffsets: Uint32Array.of(0, 2),
      motionVariances: Float32Array.of(4 * variance, 0),
      locationErrors: new Float32Array(2),
      parameters: Float32Array.of(
        radius - 3 * halfWidth,
        -(2 ** -16),
        radius + 3 * halfWidth,
        2 ** -16
      )
    },
    {gridSize: [3, 1], subStepCount: 1, tailPolicy: 'truncate-at-seven-sigma'}
  );
  // Python math on the encoded centres: exp(-x*x/(2*v))/(2*pi*v).
  const expected = [0.007825982747350383, 0.007825908863430778, 0];
  expect(density.length).toBe(3);
  for (const [cell, value] of expected.entries()) {
    expect(Number.isFinite(density[cell])).toBe(true);
    expect(density[cell]).toBeCloseTo(value, 8);
  }
});

it('TWN-13 explicit policies preserve or truncate the 7.5-sigma cell at both scales', () => {
  for (const scale of [1, 2 ** -16]) {
    const input = {
      positions: new Float32Array(4),
      timestamps: Float32Array.of(0, 1),
      trackOffsets: Uint32Array.of(0, 2),
      motionVariances: Float32Array.of(4 * scale * scale, 0),
      locationErrors: new Float32Array(2),
      parameters: Float32Array.of(7 * scale, -scale, 8 * scale, scale)
    };
    // Closed form exp(-7.5²/2)/(2*pi*s²), computed independently with Python math.
    const expected = scale === 1 ? 9.711533846746243e-14 : 0.0004171072026577219;
    for (const tailPolicy of [undefined, 'full', 'truncate-at-seven-sigma'] as const) {
      const {density} = computeBrownianBridgeUtilizationOnCPU(input, {
        gridSize: [1, 1],
        subStepCount: 1,
        tailPolicy
      });
      expect(Number.isFinite(density[0])).toBe(true);
      if (tailPolicy === 'truncate-at-seven-sigma') expect(density[0]).toBe(0);
      else expect(Math.abs(density[0] / expected - 1)).toBeLessThan(1e-6);
    }
    expect(() =>
      computeBrownianBridgeUtilizationOnCPU(input, {
        gridSize: [1, 1],
        tailPolicy: 'other' as 'full'
      })
    ).toThrow(/tailPolicy.*full.*truncate-at-seven-sigma/);
  }
  for (const Operation of [
    CPUBrownianBridgeUtilizationOperation,
    GPUBrownianBridgeUtilizationOperation
  ]) {
    expect(
      () =>
        new Operation({
          positions: 'positions',
          timestamps: 'timestamps',
          trackOffsets: 'trackOffsets',
          motionVariances: 'motionVariances',
          locationErrors: 'locationErrors',
          parameters: 'parameters',
          rowCount: 2,
          gridSize: [1, 1],
          tailPolicy: 'other' as 'full',
          output: {density: 'density'}
        })
    ).toThrow(/tailPolicy.*full.*truncate-at-seven-sigma/);
  }
});
