// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, expectTypeOf, it} from 'vitest';
import type {GPURasterHaloStage} from '@luma.gl/graph/gpu-raster';
import {WgslReflect} from 'wgsl_reflect';
import {
  getGPUTerrainCastShadowParameterValues,
  GPU_TERRAIN_CAST_SHADOW_PARAMETER_LENGTH,
  GPU_TERRAIN_CAST_SHADOW_SWEEP_PARAMETER_LENGTH,
  GPUTerrainCastShadow,
  type GPUTerrainCastShadowOutput,
  type GPUTerrainCastShadowProps
} from '../../../src/gpu-terrain/terrain-illumination/gpu-terrain-cast-shadow';
import {
  getTerrainSweepGroundLineGeometry,
  getTerrainSweepLineGeometry,
  getTerrainSweepMirroredLineGeometry
} from '../../../src/gpu-terrain/terrain-illumination/terrain-horizon-sweep';
import {
  getRasterGridGeoreference,
  getRasterGroundCellSize
} from '../../../src/analysis-core/contracts/raster-grid';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  computeTerrainHorizonSweepBruteForce,
  computeTerrainHorizonSweepHull
} from './terrain-horizon-sweep-oracle';
import {getVisibleDiskFraction} from './terrain-horizon-oracle';
import {computeTerrainCastShadowOracle} from './terrain-cast-shadow-oracle';
import {getRasterPipelines} from '../../gpu-raster/raster-pipeline-count';

it('RT-03 cast shadow rejects a local halo that changes a blocker from shadow to sun', () => {
  const elevation = new Float32Array(81);
  elevation[5 * 9 + 4] = 10;
  const window = new Float32Array(49);
  window[4 * 7 + 3] = 10;
  const azimuth = Math.PI / 8;
  for (const compute of [computeTerrainHorizonSweepBruteForce, computeTerrainHorizonSweepHull]) {
    const options = {
      directionCount: 1,
      directions: [[Math.sin(azimuth), Math.cos(azimuth)]] as const,
      maximumRadius: 2,
      cellSize: [1, 1] as const
    };
    const full = compute({...options, width: 9, height: 9, elevation});
    const cropped = compute({...options, width: 7, height: 7, elevation: window});
    expect(full.horizon[4 * 9 + 4]).toBeCloseTo(83.822396, 5);
    expect(cropped.horizon[3 * 7 + 3]).toBe(0);
    expect(getVisibleDiskFraction(20, full.horizon[4 * 9 + 4], 0)).toBe(0);
    expect(getVisibleDiskFraction(20, cropped.horizon[3 * 7 + 3], 0)).toBe(1);
  }
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const contributor = new GPUTerrainCastShadow({
      grid: makeGPURasterMetadata({width: 9, height: 9}),
      maximumRadius: 2,
      parameters: createTransientView(graph, 'cast-parameters', 'float32', 32),
      elevation: {
        id: 'elevation',
        format: 'float32',
        storage: {kind: 'buffer', values: createTransientView(graph, 'elevation', 'float32', 81)}
      },
      output: {horizonAngle: createTransientView(graph, 'horizon', 'float32', 81)}
    });
    expect('requiredHalo' in contributor).toBe(false);
    expect(() => contributor.getHaloStage()).toThrow(/terrain-cast-shadow.*parameterLayout.*sweep/);
  } finally {
    device.destroy();
  }
});

it('RT-03 cast shadow explicitly packs a shared source reference and per-frame placement', () => {
  expect(GPU_TERRAIN_CAST_SHADOW_SWEEP_PARAMETER_LENGTH).toBe(32);
  const sourceGrid = makeGPURasterMetadata({
    width: 9,
    height: 9,
    cellSize: 1,
    rowDirection: 'north'
  });
  const windowGrid = makeGPURasterMetadata({
    width: 7,
    height: 7,
    cellSize: 1,
    rowDirection: 'north',
    origin: [1, 1]
  });
  const values = getGPUTerrainCastShadowParameterValues(
    {
      azimuthDegrees: 22.5,
      altitudeDegrees: 20,
      angularRadiusDegrees: 0,
      sweepReferenceCellSize: [1, 1],
      sweepSourceGrid: sourceGrid,
      sweepOrigin: [1, 1]
    },
    windowGrid
  );
  expect(values).toHaveLength(32);
  expect(Array.from(values.slice(16, 26))).toEqual([1, 1, 1, 1, 1, 1, 0, 9, 9, 1]);
});

it('RT-03 cast-shadow CPU oracle preserves blocker shadow with shared source geometry', () => {
  const sourceGrid = makeGPURasterMetadata({
    width: 9,
    height: 9,
    cellSize: 1,
    rowDirection: 'north'
  });
  const windowGrid = makeGPURasterMetadata({
    width: 7,
    height: 7,
    cellSize: 1,
    rowDirection: 'north',
    origin: [1, 1]
  });
  const elevation = new Float32Array(81);
  elevation[5 * 9 + 4] = 10;
  const window = new Float32Array(49);
  window[4 * 7 + 3] = 10;
  for (const compute of [computeTerrainHorizonSweepBruteForce, computeTerrainHorizonSweepHull]) {
    for (const [grid, values, origin, pixel] of [
      [sourceGrid, elevation, [0, 0], 4 * 9 + 4],
      [windowGrid, window, [1, 1], 3 * 7 + 3]
    ] as const) {
      const parameters = getGPUTerrainCastShadowParameterValues(
        {
          azimuthDegrees: 22.5,
          altitudeDegrees: 20,
          angularRadiusDegrees: 0,
          sweepReferenceCellSize: [1, 1],
          sweepSourceGrid: sourceGrid,
          sweepOrigin: origin
        },
        grid
      );
      const result = computeTerrainCastShadowOracle(
        {
          width: grid.width,
          height: grid.height,
          elevation: values,
          parameters,
          parameterLayout: 'sweep',
          maximumRadius: 2,
          rowDirection: 'north'
        },
        compute
      );
      expect(Number.isFinite(result.horizonAngle[pixel])).toBe(true);
      expect(result.horizonAngle[pixel]).toBeCloseTo(83.822396, 5);
      expect(result.validity[pixel]).toBe(1);
      expect(result.sunVisibility[pixel]).toBe(0);
    }
  }
});

it('RT-03 cast shadow validates explicit halo layout independently of padding and parses shared-source WGSL', () => {
  expectTypeOf<GPUTerrainCastShadow>().not.toExtend<GPURasterHaloStage>();
  expectTypeOf<ReturnType<GPUTerrainCastShadow['getHaloStage']>>().toExtend<GPURasterHaloStage>();
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = {
      grid: makeGPURasterMetadata({width: 9, height: 9}),
      maximumRadius: 2,
      elevation: {
        id: 'elevation',
        format: 'float32' as const,
        storage: {
          kind: 'buffer' as const,
          values: createTransientView(graph, 'elevation', 'float32', 81)
        }
      },
      output: {horizonAngle: createTransientView(graph, 'horizon', 'float32', 81)}
    };
    for (const length of [16, 32, 48]) {
      const standard = new GPUTerrainCastShadow({
        ...props,
        parameters: createTransientView(graph, `standard-${length}`, 'float32', length)
      });
      expect(() => standard.getHaloStage()).toThrow(/parameterLayout/);
    }
    expect(
      () =>
        new GPUTerrainCastShadow({
          ...props,
          parameterLayout: 'sweep',
          parameters: createTransientView(graph, 'short-shared', 'float32', 16)
        })
    ).toThrow(/32/);
    const sharedProps = {
      ...props,
      parameterLayout: 'sweep' as const,
      cellSizeMode: 'geographic' as const,
      parameters: createTransientView(graph, 'shared', 'float32', 32)
    };
    const contributor = new GPUTerrainCastShadow(sharedProps);
    sharedProps.maximumRadius = 1;
    expect(contributor.getHaloStage()).toEqual({requiredHalo: 2});
    for (const pipeline of getRasterPipelines(contributor.getCommandNodes(graph))) {
      expect(() => new WgslReflect(pipeline.source), pipeline.id).not.toThrow();
    }
  } finally {
    device.destroy();
  }
});

it('RT-03 cast-shadow shared source metric retains a geographic cutoff blocker in both CPU oracles', () => {
  const sourceGrid = makeGPURasterMetadata({
    width: 9,
    height: 11,
    origin: [0, 89],
    cellSize: [0.0001, 0.00001]
  });
  const windowGrid = makeGPURasterMetadata({
    width: 5,
    height: 5,
    origin: [0.0002, 88.99998],
    cellSize: [0.0001, 0.00001]
  });
  const elevation = new Float32Array(99);
  elevation[3 * 9 + 5] = 10;
  const window = new Float32Array(25);
  window[1 * 5 + 3] = 10;
  for (const compute of [computeTerrainHorizonSweepBruteForce, computeTerrainHorizonSweepHull]) {
    for (const [grid, values, origin, pixel] of [
      [sourceGrid, elevation, [0, 0], 3 * 9 + 4],
      [windowGrid, window, [2, 2], 1 * 5 + 2]
    ] as const) {
      const parameters = getGPUTerrainCastShadowParameterValues(
        {
          azimuthDegrees: 90,
          altitudeDegrees: 20,
          angularRadiusDegrees: 0,
          sweepReferenceCellSize: [1, 1],
          sweepSourceGrid: sourceGrid,
          sweepOrigin: origin,
          maximumDistance: 0.19493955373764038
        },
        grid
      );
      const result = computeTerrainCastShadowOracle(
        {
          width: grid.width,
          height: grid.height,
          elevation: values,
          parameters,
          parameterLayout: 'sweep',
          maximumRadius: 1,
          cellSizeMode: 'geographic'
        },
        compute
      );
      expect(result.horizonAngle[pixel]).toBeCloseTo(88.88322, 3);
      expect(result.sunVisibility[pixel]).toBe(0);
      expect(result.validity[pixel]).toBe(1);
    }
  }
});

it('RT-03 cast-shadow ordinary padded targets stay ordinary and shared settings validate their source', () => {
  const grid = makeGPURasterMetadata({width: 7, height: 7, cellSize: 1});
  const sun = {azimuthDegrees: 22.5, altitudeDegrees: 20};
  const padded = new Float32Array(48).fill(123);
  getGPUTerrainCastShadowParameterValues(sun, grid, padded);
  expect(Array.from(padded.slice(0, 16))).toEqual(
    Array.from(getGPUTerrainCastShadowParameterValues(sun, grid))
  );
  expect(Array.from(padded.slice(16))).toEqual(new Array(32).fill(123));
  const shared = {...sun, sweepReferenceCellSize: [1, 1] as const, sweepSourceGrid: grid};
  expect(() => getGPUTerrainCastShadowParameterValues(shared, grid, new Float32Array(16))).toThrow(
    /32 values/
  );
  expect(() => getGPUTerrainCastShadowParameterValues({...sun, sweepOrigin: [0, 0]}, grid)).toThrow(
    /getGPUTerrainCastShadowParameterValues.*sweepReferenceCellSize.*sweepSourceGrid/
  );
  expect(() =>
    getGPUTerrainCastShadowParameterValues({...shared, sweepReferenceCellSize: [0, 1]}, grid)
  ).toThrow(/getGPUTerrainCastShadowParameterValues.*sweepReferenceCellSize/);
  expect(() =>
    getGPUTerrainCastShadowParameterValues({...shared, sweepOrigin: [1, 0]}, grid)
  ).toThrow(/sweepOrigin.*inside/);
});

it('getGPUTerrainCastShadowParameterValues packs the sun line geometry', () => {
  expect(GPU_TERRAIN_CAST_SHADOW_PARAMETER_LENGTH).toBe(16);
  // Cell size 10 x 12 on a 48 x 40 grid: the row edges are y = 0 and y = -480.
  const grid = makeGPURasterMetadata({width: 48, height: 40, cellSize: [10, 12]});
  const pack = (azimuthDegrees: number) =>
    Array.from(
      getGPUTerrainCastShadowParameterValues(
        {azimuthDegrees, altitudeDegrees: 20, zFactor: 2},
        grid
      )
    );
  // East: x-major, travel +1, no slope; the last two slots hold the ground direction (1, 0).
  expect(pack(90)).toEqual([
    10,
    12,
    2,
    0,
    0,
    -480,
    0,
    90,
    20,
    expect.any(Number),
    1,
    0,
    1,
    1,
    1,
    0
  ]);
  // The packer writes the grid georeference into the cell size and row edge slots.
  const georeference = getRasterGridGeoreference(grid);
  expect(pack(90).slice(0, 2)).toEqual([...georeference.cellSize]);
  expect(pack(90).slice(4, 6)).toEqual([georeference.northEdge, georeference.southEdge]);
  // North on a south-increasing grid goes toward row 0: y-major, travel -1.
  expect(pack(0).slice(10, 14)).toEqual([0, 0, -1, 1]);
  // The packer always uses south-advancing rows (the contributor mirrors the line for
  // north-advancing rows).
  // North-east: the packer stores the square-cell line (slope 1, step sqrt(2)) and the ground
  // direction (sin 45, -cos 45). On 10 x 12 m cells the kernel converts that direction (CPU twin
  // getTerrainSweepGroundLineGeometry) to the ground diagonal, not the pixel diagonal: the pixel
  // step is normalize((sin 45 / 10, cos 45 / 12)), slope 10 / 12, slopeFixed
  // round(65536 * 10 / 12) = 54613 and step hypot(1, 54613 / 65536) = 1.3017050231774017
  // (python3 -I).
  const diagonal = pack(45);
  expect(Math.abs(diagonal[11])).toBe(65536);
  expect(diagonal.slice(14)).toEqual([Math.fround(Math.SQRT1_2), Math.fround(-Math.SQRT1_2)]);
  const converted = getTerrainSweepGroundLineGeometry(
    [diagonal[14], diagonal[15]],
    [10, 12],
    48,
    40
  );
  expect(Math.abs(converted.slopeFixed)).toBe(54613);
  expect(converted.stepPixelLength).toBe(Math.fround(1.3017050231774017));
  // On square cells north-east is the exact pixel diagonal: slope 1, step sqrt(2).
  const square = Array.from(
    getGPUTerrainCastShadowParameterValues(
      {azimuthDegrees: 45, altitudeDegrees: 20},
      makeGPURasterMetadata({width: 48, height: 40, cellSize: 10})
    )
  );
  expect(Math.abs(square[11])).toBe(65536);
  expect(square[13]).toBe(Math.fround(Math.SQRT2));
  // A 22.5 degree sun is steep in one axis only.
  const steep = pack(22.5);
  expect(steep[10]).toBe(0); // y-major
  expect(Math.abs(steep[11])).toBeLessThan(65536);
  expect(Math.abs(steep[11])).toBeGreaterThan(0);
  expect(
    Array.from(
      getGPUTerrainCastShadowParameterValues(
        {azimuthDegrees: 5, altitudeDegrees: 5},
        makeGPURasterMetadata({width: 48, height: 40})
      )
    )[9]
  ).toBeCloseTo(0.2666, 6);
});

it('cast shadow takes the cell size mode from the contributor only', () => {
  // The packer has no cell size mode, so the settings cannot disagree with the contributor: the
  // same packed values serve a uniform and a geographic contributor, and the kernel converts the
  // packed ground direction with the contributor's mode. At 60 N (0.001 degree cells, python3 -I
  // WGS84: [55.80000157243614, 111.41228745777943] m) a 45 degree sun converts to slopeFixed
  // round(-32823.209957312734) = -32823, the true 45 degree ground line; the packed square-cell
  // line (slopeFixed -65536) is a 26.60365263545291 degree ground ray.
  const values = getGPUTerrainCastShadowParameterValues(
    {azimuthDegrees: 45, altitudeDegrees: 20},
    makeGPURasterMetadata({width: 48, height: 24, cellSize: 0.001, origin: [0, 60]})
  );
  expect(values[11]).toBe(-65536);
  const groundCell = getRasterGroundCellSize(
    'geographic',
    {cellSize: [0.001, 0.001], northEdge: 60, southEdge: 60},
    0.5,
    1
  );
  const converted = getTerrainSweepGroundLineGeometry([values[14], values[15]], groundCell, 48, 24);
  expect(converted).toMatchObject({xMajor: true, slopeFixed: -32823, travelSign: 1});
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  for (const cellSizeMode of ['uniform', 'web-mercator', 'geographic'] as const) {
    const contributor = new GPUTerrainCastShadow({
      id: `cast-${cellSizeMode}`,
      grid: makeGPURasterMetadata({width: 6, height: 5}),
      cellSizeMode,
      elevation: {
        id: `elevation-${cellSizeMode}`,
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: createTransientView(graph, `values-${cellSizeMode}`, 'float32', 30)
        }
      },
      parameters: createTransientView(graph, `parameters-${cellSizeMode}`, 'float32', 16),
      output: {
        horizonAngle: createTransientView(graph, `horizon-${cellSizeMode}`, 'float32', 30)
      }
    });
    expect(contributor.getCommandNodes(graph).some(node => node.id.endsWith('-sweep'))).toBe(true);
  }
  device.destroy();
});

it('cast shadow mirrors the packed line for north rows exactly like the old north packing', () => {
  // The kernel negates slopeFixed (and the travel sign of y-major lines) of the south-packed line
  // for rowDirection 'north'. That equals packing (sin az, +cos az) directly except at exact
  // half-unit rounding ties of slopeFixed; none occurs on a 0.1 degree azimuth grid.
  const snap = (value: number) => (Math.abs(value) < 1e-9 ? 0 : value);
  for (let tenth = 0; tenth < 3600; tenth++) {
    const azimuthDegrees = tenth / 10;
    const values = getGPUTerrainCastShadowParameterValues(
      {azimuthDegrees, altitudeDegrees: 10},
      makeGPURasterMetadata({width: 48, height: 40, cellSize: 10})
    );
    const packed = {
      xMajor: values[10] !== 0,
      slopeFixed: values[11],
      travelSign: values[12] > 0 ? (1 as const) : (-1 as const),
      firstLine: 0,
      lineCount: 0,
      stepPixelLength: values[13]
    };
    const mirrored = getTerrainSweepMirroredLineGeometry(packed, 1, 1);
    const azimuth = (azimuthDegrees * Math.PI) / 180;
    const north = getTerrainSweepLineGeometry(
      [snap(Math.sin(azimuth)), snap(Math.cos(azimuth))],
      1,
      1
    );
    expect(
      [mirrored.xMajor, mirrored.slopeFixed, mirrored.travelSign, mirrored.stepPixelLength],
      `${azimuthDegrees}`
    ).toEqual([north.xMajor, north.slopeFixed, north.travelSign, north.stepPixelLength]);
  }
});

it('getGPUTerrainCastShadowParameterValues validates its input', () => {
  const valid = {azimuthDegrees: 1, altitudeDegrees: 1};
  const grid = makeGPURasterMetadata({width: 48, height: 40});
  expect(() =>
    getGPUTerrainCastShadowParameterValues(
      valid,
      makeGPURasterMetadata({width: 48, height: 40, cellSize: [0, 1]})
    )
  ).toThrow(/cell size/);
  expect(() =>
    getGPUTerrainCastShadowParameterValues({...valid, angularRadiusDegrees: -1}, grid)
  ).toThrow(/angular radius/);
  expect(() =>
    getGPUTerrainCastShadowParameterValues({...valid, azimuthDegrees: Number.NaN}, grid)
  ).toThrow(/finite/);
  expect(() => getGPUTerrainCastShadowParameterValues(valid, grid, new Float32Array(15))).toThrow(
    /16 values/
  );
});

it('GPUTerrainCastShadow validates props and schedules one sweep node', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let instance = 0;
  const create = (
    overrides: Partial<Omit<GPUTerrainCastShadowProps, 'output'>> = {},
    outputOverrides?: GPUTerrainCastShadowOutput
  ) => {
    instance++;
    const length = 30;
    return new GPUTerrainCastShadow({
      grid: makeGPURasterMetadata({width: 6, height: 5}),
      elevation: {
        id: `elevation-${instance}`,
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: createTransientView(graph, `elevation-${instance}`, 'float32', length)
        }
      },
      parameters: createTransientView(graph, `parameters-${instance}`, 'float32', 16),
      output: outputOverrides ?? {
        sunVisibility: createTransientView(graph, `visibility-${instance}`, 'float32', length)
      },
      ...overrides
    });
  };
  const contributor = create();
  expect(contributor.id).toBe('terrain-cast-shadow');
  // Whole tile by default: the diagonal covers every azimuth.
  expect(contributor.maximumRadius).toBe(Math.ceil(Math.hypot(6, 5)));
  expect('requiredHalo' in contributor).toBe(false);
  expect(() => contributor.getHaloStage()).toThrow(/parameterLayout.*sweep/);
  expect(
    create({
      maximumRadius: 4,
      parameterLayout: 'sweep',
      parameters: createTransientView(graph, 'shared-parameters', 'float32', 32)
    }).getHaloStage()
  ).toEqual({requiredHalo: 4});
  const nodes = contributor.getCommandNodes(graph);
  expect(nodes.some(node => node.id === 'terrain-cast-shadow-sweep')).toBe(true);
  expect(nodes.filter(node => node.id.includes('sweep'))).toHaveLength(1);

  expect(() => create({}, {})).toThrow(/at least one output/);
  expect(() => create({maximumRadius: 0})).toThrow(/maximumRadius/);
  expect(() => create({maximumRadius: 2.5})).toThrow(/maximumRadius/);
  expect(() => create({grid: makeGPURasterMetadata({width: 40000, height: 5})})).toThrow(/32767/);
  expect(() =>
    create({parameters: createTransientView(graph, 'short-parameters', 'float32', 8)})
  ).toThrow(/at least 16/);
  expect(() =>
    create({}, {sunVisibility: createTransientView(graph, 'short-visibility', 'float32', 3)})
  ).toThrow(/30 float32/);
  expect(() => create({cellSizeMode: 'bogus' as 'uniform'})).toThrow(/cellSizeMode/);
  expect(() => create({rowDirection: 'up' as 'south'})).toThrow(/rowDirection/);
  const other = new GPUCommandGraph(device);
  expect(() => create().getCommandNodes(other)).toThrow(/belong/);
});
