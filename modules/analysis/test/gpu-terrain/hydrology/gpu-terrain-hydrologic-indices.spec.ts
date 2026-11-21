// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  makeGPURasterBand,
  makeGPURasterMetadata
} from '@luma-contributors/analysis/raster-analysis';
import {type GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPU_TERRAIN_HYDROLOGIC_INDICES_PARAMETER_LENGTH,
  GPUTerrainHydrologicIndices,
  getGPUTerrainHydrologicIndicesParameterValues,
  type GPUTerrainHydrologicIndicesSettings
} from '../../../src/gpu-terrain/hydrology/gpu-terrain-hydrologic-indices';
import {
  GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH,
  GPUTerrainDerivatives,
  getGPUTerrainDerivativesParameterValues
} from '../../../src/gpu-terrain/terrain-analysis/gpu-terrain-derivatives';
import type {GPURasterCellSizeMode} from '../../../src/analysis-core/contracts/raster-grid';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {computeHydrologicIndices} from './terrain-hydrologic-indices-oracle';
import {getOracleWebMercatorY} from '../terrain-oracle-utils';
import {TERRAIN_FLOW_DEMS as DEMS, computeTerrainFlow} from './terrain-flow-oracle';
import {requireTestDevice} from '../../utils/require-test-device';

import {
  expectIndicesClose,
  HYDROLOGIC_INDEX_OUTPUTS as OUTPUTS
} from './terrain-indices-test-assertions';

type IndicesOptions = {
  settings: GPUTerrainHydrologicIndicesSettings;
  cellSizeMode?: GPURasterCellSizeMode;
  /** Ground cell size of the grid. */
  cellSize: number | [number, number];
  /** World position of the top-left corner of the grid. Defaults to `[0, 0]`. */
  origin?: [number, number];
};

/** Returns the raster grid that a spec places its DEM on. */
function getIndicesGrid(width: number, height: number, options: IndicesOptions): GPURasterMetadata {
  return makeGPURasterMetadata({width, height, cellSize: options.cellSize, origin: options.origin});
}

async function runIndices(
  device: Device,
  elevation: Float32Array,
  accumulation: Float32Array,
  width: number,
  height: number,
  options: IndicesOptions
): Promise<Record<(typeof OUTPUTS)[number], Float32Array>> {
  const cellCount = width * height;
  const grid = getIndicesGrid(width, height, options);
  const elevationBuffer = createInputBuffer(device, elevation);
  const accumulationBuffer = createInputBuffer(device, accumulation);
  const outputs = Object.fromEntries(
    OUTPUTS.map(name => [name, createOutputBuffer(device, cellCount)])
  ) as Record<(typeof OUTPUTS)[number], ReturnType<typeof createOutputBuffer>>;
  const settingsBuffer = new GPUParameterBuffer(device, {
    id: 'indices-settings',
    format: 'float32',
    length: 8,
    values: getGPUTerrainHydrologicIndicesParameterValues(options.settings, grid)
  });
  const graph = new GPUCommandGraph(device, {id: 'hydrologic-indices-test'});
  graph.add(
    new GPUTerrainHydrologicIndices({
      grid,
      cellSizeMode: options.cellSizeMode,
      elevation: makeGPURasterBand(
        importGraphBuffer(graph, 'elevation', elevationBuffer, 'float32', cellCount),
        grid,
        {id: 'elevation'}
      ),
      accumulation: importGraphBuffer(
        graph,
        'accumulation',
        accumulationBuffer,
        'float32',
        cellCount
      ),
      parameters: settingsBuffer.importToGraph(graph),
      output: {
        // Anchor one terminal for TypeScript; the loop supplies every named sibling.
        descentTangent: importGraphBuffer(
          graph,
          'descentTangent',
          outputs.descentTangent,
          'float32',
          cellCount
        ),
        ...Object.fromEntries(
          OUTPUTS.filter(name => name !== 'descentTangent').map(name => [
            name,
            importGraphBuffer(graph, name, outputs[name], 'float32', cellCount)
          ])
        )
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {} as Record<(typeof OUTPUTS)[number], Float32Array>;
  for (const name of OUTPUTS) {
    result[name] = Float32Array.from(await readFloat32(outputs[name], cellCount));
    outputs[name].destroy();
  }
  compiled.destroy();
  settingsBuffer.destroy();
  elevationBuffer.destroy();
  accumulationBuffer.destroy();
  return result;
}

it('GPUTerrainHydrologicIndices matches the float64 oracle', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const cases: [string, IndicesOptions, GPURasterCellSizeMode | undefined][] = [
    ['uniform', {settings: {}, cellSize: 30}, undefined],
    ['rectangular', {settings: {minimumSlope: 0.01}, cellSize: [25, 40]}, undefined],
    [
      'geographic',
      {
        cellSizeMode: 'geographic',
        settings: {},
        cellSize: [0.0003, 0.0002],
        origin: [0, 46.8]
      },
      'geographic'
    ],
    [
      'web-mercator',
      {
        cellSizeMode: 'web-mercator',
        settings: {},
        cellSize: 40,
        origin: [0, getOracleWebMercatorY(0.35)]
      },
      'web-mercator'
    ]
  ];
  let largestWetness = 0;
  for (const [width, height] of [
    [37, 29],
    [70, 45]
  ]) {
    for (const [demName, createDem] of Object.entries(DEMS)) {
      const elevation = createDem(width, height);
      for (const [caseName, options] of cases) {
        // Contributing area from the CPU flow oracle (the GPU flow contributor has its own tests).
        const flow = computeTerrainFlow({
          elevation,
          grid: getIndicesGrid(width, height, options),
          settings: {fillEpsilon: 0.25},
          cellSizeMode: options.cellSizeMode,
          fillDepressions: true,
          area: true
        });
        const name = `${demName} ${caseName} ${width}x${height}`;
        const actual = await runIndices(
          device,
          flow.filled,
          flow.accumulation,
          width,
          height,
          options
        );
        const expected = computeHydrologicIndices({
          elevation: flow.filled,
          accumulation: flow.accumulation,
          grid: getIndicesGrid(width, height, options),
          settings: options.settings,
          cellSizeMode: options.cellSizeMode
        });
        expectIndicesClose(name, actual, expected);
        for (const value of actual.wetnessIndex) {
          if (Number.isFinite(value)) {
            largestWetness = Math.max(largestWetness, value);
          }
        }
      }
    }
  }
  // Wetness grows along drainage lines; a silently failed shader would leave zeros.
  expect(largestWetness).toBeGreaterThan(5);
});

it('GPUTerrainHydrologicIndices handles flats, zero area, nodata, and the analytic plane', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const width = 6;
  const height = 4;
  const cellCount = width * height;
  // A plane falling 3 m per 10 m column: tan(beta) = 0.3 everywhere except the lowest column,
  // which has no lower neighbor and takes the minimum slope.
  const plane = Float32Array.from({length: cellCount}, (_, cell) => 30 - 3 * (cell % width));
  const area = Float32Array.from({length: cellCount}, (_, cell) => 100 * (1 + (cell % width)));
  area[7] = 0;
  area[8] = NaN;
  plane[15] = NaN;
  const settings: GPUTerrainHydrologicIndicesSettings = {minimumSlope: 0};
  const actual = await runIndices(device, plane, area, width, height, {settings, cellSize: 10});
  expectIndicesClose(
    'plane',
    actual,
    computeHydrologicIndices({
      elevation: plane,
      accumulation: area,
      grid: makeGPURasterMetadata({width, height, cellSize: 10}),
      settings
    })
  );
  expect(actual.descentTangent[0]).toBeCloseTo(0.3, 6);
  expect(actual.specificCatchmentArea[0]).toBeCloseTo(10, 5);
  expect(actual.wetnessIndex[0]).toBeCloseTo(Math.log(10 / 0.3), 5);
  expect(actual.streamPowerIndex[0]).toBeCloseTo(3, 5);
  expect(actual.wetnessIndex[7]).toBe(-Infinity);
  expect(actual.wetnessIndex[8]).toBeNaN();
  expect(actual.descentTangent[15]).toBeNaN();
  // The lowest column (index 5) has no lower neighbor: zero slope, infinite wetness, no power.
  expect(actual.descentTangent[5]).toBe(0);
  expect(actual.wetnessIndex[5]).toBe(Infinity);
  expect(actual.streamPowerIndex[5]).toBe(0);

  const clamped = await runIndices(device, plane, area, width, height, {
    settings: {},
    cellSize: 10
  });
  expect(clamped.descentTangent[5]).toBeCloseTo(0.001, 8);
  expect(clamped.wetnessIndex[5]).toBeCloseTo(Math.log(60 / 0.001), 4);
});

it.each([
  {
    name: 'a unit-grade plane',
    // At the center, the Horn gradient has magnitude 1 and the westward drop is 1 over 1.
    elevation: [0, 1, 2, 0, 1, 2, 0, 1, 2],
    expectedSlopeDegrees: 45,
    expectedDescentTangent: 1
  },
  {
    name: 'a northeast-corner spike',
    // The two Horn gradient components have magnitude 8 / 8. The center has no lower
    // D8 neighbor, so the descent tangent equals the default minimumSlope of 0.001.
    elevation: [0, 0, 8, 0, 0, 0, 0, 0, 0],
    expectedSlopeDegrees: 54.7356,
    expectedDescentTangent: 0.001
  }
])('URT-10 reads paired Horn slope and hydrologic descent tangent on $name', async fixture => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const grid = makeGPURasterMetadata({width: 3, height: 3, cellSize: 1});
  const cellCount = grid.width * grid.height;
  const center = 4;
  const graph = new GPUCommandGraph(device, {id: 'terrain-slope-comparison'});
  const elevationBuffer = createInputBuffer(device, Float32Array.from(fixture.elevation));
  const accumulationBuffer = createInputBuffer(device, new Float32Array(cellCount).fill(1));
  const slopeBuffer = createOutputBuffer(device, cellCount);
  const descentTangentBuffer = createOutputBuffer(device, cellCount);
  const derivativeParameters = new GPUParameterBuffer(device, {
    id: 'derivative-settings',
    format: 'float32',
    length: GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH,
    values: getGPUTerrainDerivativesParameterValues({}, grid)
  });
  const hydrologicParameters = new GPUParameterBuffer(device, {
    id: 'hydrologic-settings',
    format: 'float32',
    length: GPU_TERRAIN_HYDROLOGIC_INDICES_PARAMETER_LENGTH,
    values: getGPUTerrainHydrologicIndicesParameterValues({}, grid)
  });
  let compiled: ReturnType<typeof graph.compile> | undefined;
  try {
    const elevation = makeGPURasterBand(
      importGraphBuffer(graph, 'elevation', elevationBuffer, 'float32', cellCount),
      grid,
      {id: 'elevation'}
    );
    graph.add(
      new GPUTerrainDerivatives({
        grid,
        elevation,
        slopeUnits: 'degrees',
        parameters: derivativeParameters.importToGraph(graph),
        output: {slope: importGraphBuffer(graph, 'slope', slopeBuffer, 'float32', cellCount)}
      })
    );
    graph.add(
      new GPUTerrainHydrologicIndices({
        grid,
        elevation,
        accumulation: importGraphBuffer(
          graph,
          'accumulation',
          accumulationBuffer,
          'float32',
          cellCount
        ),
        parameters: hydrologicParameters.importToGraph(graph),
        output: {
          descentTangent: importGraphBuffer(
            graph,
            'descentTangent',
            descentTangentBuffer,
            'float32',
            cellCount
          )
        }
      })
    );
    compiled = graph.compile();
    submitGraph(device, compiled, undefined);
    const slopeDegrees = (await readFloat32(slopeBuffer, cellCount))[center];
    const descentTangent = (await readFloat32(descentTangentBuffer, cellCount))[center];
    expect(Number.isFinite(slopeDegrees)).toBe(true);
    expect(Number.isFinite(descentTangent)).toBe(true);
    expect(slopeDegrees).toBeCloseTo(fixture.expectedSlopeDegrees, 4);
    expect(descentTangent).toBeCloseTo(fixture.expectedDescentTangent, 8);
  } finally {
    compiled?.destroy();
    hydrologicParameters.destroy();
    derivativeParameters.destroy();
    descentTangentBuffer.destroy();
    slopeBuffer.destroy();
    accumulationBuffer.destroy();
    elevationBuffer.destroy();
  }
});
