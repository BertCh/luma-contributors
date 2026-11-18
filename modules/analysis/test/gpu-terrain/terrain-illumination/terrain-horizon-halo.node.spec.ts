// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, expectTypeOf, it} from 'vitest';
import type {GPURasterHaloStage} from '@luma.gl/graph/gpu-raster';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import {
  GPUTerrainHorizon,
  getGPUTerrainHorizonParameterValues,
  getGPUTerrainHorizonStepDistances
} from '../../../src/gpu-terrain/terrain-illumination/gpu-terrain-horizon';
import {GPUGeomorphons} from '../../../src/gpu-terrain/geomorphons/gpu-geomorphons';
import {GPUTerrainWeissLandforms} from '../../../src/gpu-terrain/topographic-position/gpu-terrain-weiss-landforms';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createBand} from '../terrain-test-utils';
import {
  computeTerrainHorizonSweepBruteForce,
  computeTerrainHorizonSweepHull
} from './terrain-horizon-sweep-oracle';
import {computeTerrainHorizon} from './terrain-horizon-oracle';
import {getTerrainSweepMinorOffset} from '../../../src/gpu-terrain/terrain-illumination/terrain-horizon-sweep';
import {WgslReflect} from 'wgsl_reflect';
import {getRasterPipelines} from '../../gpu-raster/raster-pipeline-count';

it('halo assembly requires an explicit supported stage in the public types', () => {
  expectTypeOf<GPUTerrainHorizon>().not.toExtend<GPURasterHaloStage>();
  expectTypeOf<GPUGeomorphons>().not.toExtend<GPURasterHaloStage>();
  expectTypeOf<GPUTerrainWeissLandforms>().not.toExtend<GPURasterHaloStage>();
  expectTypeOf<ReturnType<GPUTerrainHorizon['getHaloStage']>>().toExtend<GPURasterHaloStage>();
  expectTypeOf<
    ReturnType<GPUTerrainWeissLandforms['getHaloStage']>
  >().toExtend<GPURasterHaloStage>();
});

it('RT-03 refuses a seamless sweep halo for the 84-degree versus zero blocker', () => {
  const elevation = new Float32Array(81);
  elevation[5 * 9 + 4] = 10;
  const window = new Float32Array(49);
  window[4 * 7 + 3] = 10;
  for (const compute of [computeTerrainHorizonSweepBruteForce, computeTerrainHorizonSweepHull]) {
    const options = {
      directionCount: 16,
      rowDirection: 'north' as const,
      maximumRadius: 2,
      cellSize: [1, 1] as const
    };
    const full = compute({...options, width: 9, height: 9, elevation});
    const cropped = compute({...options, width: 7, height: 7, elevation: window});
    expect(full.horizon[(4 * 9 + 4) * 16 + 1]).toBeCloseTo(83.822396, 5);
    expect(cropped.horizon[(3 * 7 + 3) * 16 + 1]).toBe(0);
    expect(full.validity[4 * 9 + 4]).toBe(1);
    expect(cropped.validity[3 * 7 + 3]).toBe(1);
  }
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = {
      grid: makeGPURasterMetadata({width: 9, height: 9}),
      maximumRadius: 2,
      elevation: createBand(graph, 'elevation', 81),
      parameters: createTransientView(graph, 'parameters', 'float32', 8),
      output: {skyViewFactor: createTransientView(graph, 'sky-view', 'float32', 81)}
    };
    const sweep = new GPUTerrainHorizon({...props, algorithm: 'sweep'});
    expect('requiredHalo' in sweep).toBe(false);
    expect(() => sweep.getHaloStage()).toThrow(/terrain-horizon.*sweep.*halo/);
    expect(() =>
      new GPUTerrainHorizon({
        ...props,
        algorithm: 'sweep',
        grid: {...props.grid, level: 1, levelZeroOrigin: [2, 2]}
      }).getHaloStage()
    ).toThrow(/parameterLayout/);
    expect(() =>
      getGPUTerrainHorizonParameterValues(
        {sweepOrigin: [0.5, 0], sweepReferenceCellSize: [1, 1], sweepSourceGrid: props.grid},
        props.grid
      )
    ).toThrow(/sweepOrigin.*safe integer/);
    const geographic = {
      ...props,
      algorithm: 'sweep' as const,
      cellSizeMode: 'geographic' as const,
      parameterLayout: 'sweep' as const
    };
    expect(() => new GPUTerrainHorizon(geographic)).toThrow(/24/);
    const sharedReference = new GPUTerrainHorizon({
      ...geographic,
      parameters: createTransientView(graph, 'reference-parameters', 'float32', 24)
    });
    expect(sharedReference.getHaloStage()).toEqual({requiredHalo: 2});
    for (const pipeline of getRasterPipelines(sharedReference.getCommandNodes(graph))) {
      expect(() => new WgslReflect(pipeline.source), pipeline.id).not.toThrow();
    }
    // Bilinear validity reads floor + 1 even when the sample lies on an integer pixel.
    expect(new GPUTerrainHorizon(props).getHaloStage()).toEqual({requiredHalo: 3});
  } finally {
    device.destroy();
  }
});

it('RT-03 march agrees on the cropped blocker with its complete advertised halo', () => {
  const elevation = new Float32Array(81);
  elevation[5 * 9 + 4] = 10;
  const window = new Float32Array(49);
  window[4 * 7 + 3] = 10;
  const options = {
    directionCount: 16,
    rowDirection: 'north' as const,
    stepDistances: getGPUTerrainHorizonStepDistances(2),
    cellSize: [1, 1] as const
  };
  const full = computeTerrainHorizon({...options, width: 9, height: 9, elevation});
  const cropped = computeTerrainHorizon({...options, width: 7, height: 7, elevation: window});
  for (let sector = 0; sector < 16; sector++) {
    const angle = full.horizon[(4 * 9 + 4) * 16 + sector];
    expect(Number.isFinite(angle)).toBe(true);
    expect(cropped.horizon[(3 * 7 + 3) * 16 + sector]).toBeCloseTo(angle, 10);
  }
});

it('RT-03 anchored sweep gives the same 84-degree blocker horizon in an offset tile', () => {
  const elevation = new Float32Array(81);
  elevation[5 * 9 + 4] = 10;
  const window = new Float32Array(49);
  window[4 * 7 + 3] = 10;
  for (const compute of [computeTerrainHorizonSweepBruteForce, computeTerrainHorizonSweepHull]) {
    const options = {
      directionCount: 16,
      rowDirection: 'north' as const,
      maximumRadius: 2,
      cellSize: [1, 1] as const
    };
    const full = compute({...options, width: 9, height: 9, elevation, origin: [0, 0]});
    const cropped = compute({...options, width: 7, height: 7, elevation: window, origin: [1, 1]});
    expect(full.horizon[(4 * 9 + 4) * 16 + 1]).toBeCloseTo(83.822396, 5);
    expect(cropped.horizon[(3 * 7 + 3) * 16 + 1]).toBeCloseTo(83.822396, 5);
    for (let sector = 0; sector < 16; sector++) {
      expect(cropped.horizon[(3 * 7 + 3) * 16 + sector]).toBeCloseTo(
        full.horizon[(4 * 9 + 4) * 16 + sector],
        10
      );
    }
  }
});

it('RT-03 packs the shared reference cell independently of the cropped geographic window', () => {
  const referenceCellSize = [55.8, 111.4] as const;
  const grid = makeGPURasterMetadata({width: 7, height: 7, cellSize: 0.001, origin: [0, 60]});
  const values = getGPUTerrainHorizonParameterValues(
    {sweepReferenceCellSize: referenceCellSize, sweepSourceGrid: grid},
    grid
  );
  expect(values).toHaveLength(24);
  expect(Array.from(values.slice(12, 14))).toEqual(Array.from(referenceCellSize, Math.fround));
  for (const invalid of [0, -1, Infinity, NaN, 1e-50, 1e40]) {
    expect(() =>
      getGPUTerrainHorizonParameterValues(
        {sweepReferenceCellSize: [invalid, 1], sweepSourceGrid: grid},
        grid
      )
    ).toThrow(/sweepReferenceCellSize/);
  }
  expect(() =>
    getGPUTerrainHorizonParameterValues(
      {sweepReferenceCellSize: referenceCellSize, sweepSourceGrid: grid},
      grid,
      new Float32Array(12)
    )
  ).toThrow(/24/);
});

it('RT-03 anchored phases match an independent BigInt global-coordinate oracle', () => {
  const divideFloor = (value: bigint) => (value >= 0n ? value / 65536n : (value - 65535n) / 65536n);
  for (const origin of [-(2 ** 40) + 13, -1, 0, 1, 2 ** 40 + 17, Number.MAX_SAFE_INTEGER]) {
    for (const slope of [-65536, -27146, 0, 27146, 65536]) {
      const base = divideFloor(BigInt(slope) * BigInt(origin) + 32768n);
      for (let major = 0; major < 20; major++) {
        const globalOffset = divideFloor(BigInt(slope) * (BigInt(origin) + BigInt(major)) + 32768n);
        expect(getTerrainSweepMinorOffset(slope, major, origin)).toBe(Number(globalOffset - base));
      }
    }
  }
});

it('RT-03 geographic sweep tiles share reference geometry and curvature for every output', () => {
  const width = 13;
  const height = 11;
  const elevation = Float32Array.from(
    {length: width * height},
    (_, pixel) => ((pixel * 17) % 23) * 10
  );
  const window = Float32Array.from(
    {length: 49},
    (_, pixel) => elevation[(Math.floor(pixel / 7) + 1) * width + (pixel % 7) + 2]
  );
  for (const compute of [computeTerrainHorizonSweepBruteForce, computeTerrainHorizonSweepHull]) {
    const options = {
      directionCount: 64,
      maximumRadius: 2,
      cellSize: [0.01, 0.01] as const,
      cellSizeMode: 'geographic' as const,
      referenceCellSize: [500, 1100] as const,
      curvatureCoefficient: 1e-5,
      nadir: true
    };
    const full = compute({
      ...options,
      width,
      height,
      elevation,
      origin: [0, 0],
      northEdge: 70,
      southEdge: 20
    });
    const cropped = compute({
      ...options,
      width: 7,
      height: 7,
      elevation: window,
      origin: [2, 1],
      northEdge: 70 - 50 / 11,
      southEdge: 70 - (50 * 8) / 11
    });
    const fullPixel = 4 * width + 5;
    const windowPixel = 3 * 7 + 3;
    for (const key of ['horizon', 'nadirHorizon'] as const) {
      for (let sector = 0; sector < 64; sector++) {
        const expected = full[key][fullPixel * 64 + sector];
        expect(Number.isFinite(expected)).toBe(true);
        expect(cropped[key][windowPixel * 64 + sector]).toBeCloseTo(expected, 9);
      }
    }
    for (const key of [
      'skyViewFactor',
      'positiveOpenness',
      'negativeOpenness',
      'validity'
    ] as const) {
      expect(Number.isFinite(full[key][fullPixel])).toBe(true);
      expect(cropped[key][windowPixel]).toBeCloseTo(full[key][fullPixel], 9);
    }
  }
});

it('RT-03 bilinear validity needs the extra corner beyond maximumRadius', () => {
  const elevation = new Float32Array(81);
  elevation[4 * 9 + 6] = 10;
  elevation[4 * 9 + 7] = NaN;
  const compute = (values: Float32Array, width: number) =>
    computeTerrainHorizon({
      width,
      height: width,
      elevation: values,
      directionCount: 4,
      cellSize: [1, 1],
      stepDistances: [1, 2]
    });
  const full = compute(elevation, 9);
  const insufficientWindow = new Float32Array(25);
  insufficientWindow[2 * 5 + 4] = 10;
  const completeWindow = new Float32Array(49);
  completeWindow[3 * 7 + 5] = 10;
  completeWindow[3 * 7 + 6] = NaN;
  // The east sample at distance two is rejected only if its zero-weight next corner is visible.
  expect(full.horizon[(4 * 9 + 4) * 4 + 1]).toBe(0);
  expect(compute(insufficientWindow, 5).horizon[(2 * 5 + 2) * 4 + 1]).toBeCloseTo(78.690068, 5);
  expect(compute(completeWindow, 7).horizon[(3 * 7 + 3) * 4 + 1]).toBe(0);
});

it('RT-06 exposes advisory radii without structurally advertising a complete halo', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = {
      grid: makeGPURasterMetadata({width: 9, height: 9}),
      elevation: createBand(graph, 'elevation', 81),
      parameters: createTransientView(graph, 'parameters', 'float32', 8)
    };
    const geomorphons = new GPUGeomorphons({
      ...props,
      searchRadius: 4,
      output: {forms: createTransientView(graph, 'forms', 'uint32', 81)}
    });
    const global = new GPUTerrainWeissLandforms({
      ...props,
      output: {landforms: createTransientView(graph, 'global', 'uint32', 81)}
    });
    const local = new GPUTerrainWeissLandforms({
      ...props,
      standardization: 'local',
      largeScale: {radius: 6},
      output: {landforms: createTransientView(graph, 'local', 'uint32', 81)}
    });
    for (const contributor of [geomorphons, global, local]) {
      expect('requiredHalo' in contributor).toBe(false);
    }
    expect(geomorphons.searchRadius).toBe(4);
    expect(global.localRadius).toBe(15);
    expect(() => global.getHaloStage()).toThrow(/terrain-weiss-landforms.*global.*halo/);
    expect(local.getHaloStage()).toEqual({requiredHalo: 6});
  } finally {
    device.destroy();
  }
});

it('RT-13 bounds march samples by the grid without appending an unscheduled endpoint', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = {
      grid: makeGPURasterMetadata({width: 6, height: 5}),
      elevation: createBand(graph, 'elevation', 30),
      parameters: createTransientView(graph, 'parameters', 'float32', 8),
      output: {skyViewFactor: createTransientView(graph, 'sky-view', 'float32', 30)},
      maximumRadius: 100
    };
    const unitMarch = new GPUTerrainHorizon(props);
    expect(Array.from(unitMarch.stepDistances)).toEqual([1, 2, 3, 4, 5, 6, 7]);
    // The conservative diagonal ceiling is 7. Clamping the radius first would append 7.
    const geometricMarch = new GPUTerrainHorizon({...props, stepGrowth: 2});
    expect(Array.from(geometricMarch.stepDistances)).toEqual([1, 2, 4]);
    const sweep = new GPUTerrainHorizon({...props, algorithm: 'sweep'});
    expect(sweep.stepDistances).toHaveLength(0);
    // A huge configured radius must remain cheap on a small grid, including sweep encoding.
    const hugeSweep = new GPUTerrainHorizon({
      ...props,
      algorithm: 'sweep',
      maximumRadius: 10_000_000
    });
    expect(hugeSweep.stepDistances).toHaveLength(0);
    expect(hugeSweep.getCommandNodes(graph).length).toBeGreaterThan(0);
    expect(new GPUTerrainHorizon({...props, maximumRadius: 10_000_000}).stepDistances).toHaveLength(
      7
    );
    expect(Array.from(getGPUTerrainHorizonStepDistances(5, 2, 7))).toEqual([1, 2, 4, 5]);
    expect(Array.from(getGPUTerrainHorizonStepDistances(100, 2, 7))).toEqual([1, 2, 4]);
  } finally {
    device.destroy();
  }
});
