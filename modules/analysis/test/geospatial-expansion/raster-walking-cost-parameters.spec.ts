// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  getRasterGridGeoreference,
  getRasterGroundScale,
  type GPURasterCellSizeMode
} from '../../src/analysis-core/raster-grid';
import {makeGPURasterMetadata} from '../../src/raster-analysis/raster-grid/gpu-raster-grid';
import {
  GPU_RASTER_WALKING_COST_PARAMETER_LENGTH,
  GPURasterWalkingCost,
  getGPURasterWalkingCostParameterValues,
  type GPURasterWalkingCostModel,
  type GPURasterWalkingCostProps
} from '../../src/raster-analysis/gpu-raster-walking-cost';
import {
  createInputBuffer,
  createNullWebGPUDevice,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../utils/require-test-device';
import {WALKING_COST_74DDDCF5_COST_BITS} from './raster-walking-cost-74dddcf5-fixture';

const WIDTH = 7;
const HEIGHT = 5;
const COUNT = WIDTH * HEIGHT;
const SOURCE = 2 * WIDTH + 1;

const ELEVATION = Float32Array.from({length: COUNT}, (_, index) => {
  const column = index % WIDTH;
  const row = Math.floor(index / WIDTH);
  return 3 * column + 2 * Math.sin(row * 1.3 + column * 0.7) - row;
});
const FRICTION = Float32Array.from({length: COUNT}, (_, index) => 0.01 * ((index * 7) % 5));

type Settings = {frictionWeight: number; model: GPURasterWalkingCostModel};

/** Float64 Bellman-Ford twin of the walking-cost kernel. */
function computeWalkingCosts(
  grid: GPURasterMetadata,
  settings: Settings,
  cellSizeMode: GPURasterCellSizeMode
): Float64Array {
  const [a, b, , d, e] = grid.affine;
  const georeference = getRasterGridGeoreference(grid);
  const getStepDistance = (row: number, columnStep: number, rowStep: number): number => {
    let x = a * columnStep + b * rowStep;
    let y = d * columnStep + e * rowStep;
    if (cellSizeMode !== 'uniform') {
      const rowFraction = (row + 0.5 + 0.5 * rowStep) / HEIGHT;
      const edge =
        georeference.northEdge + (georeference.southEdge - georeference.northEdge) * rowFraction;
      const [scaleX, scaleY] = getRasterGroundScale(cellSizeMode, edge);
      x *= scaleX;
      y *= scaleY;
    }
    return Math.hypot(x, y);
  };
  const getTransitionCost = (cell: number, neighbor: number, distance: number): number => {
    const rise = ELEVATION[cell] - ELEVATION[neighbor];
    const slope = rise / distance;
    const {model} = settings;
    let energy: number;
    if (model.kind === 'grass') {
      const [c0, c1, c2, c3] = model.coefficients;
      const downhill = slope < model.downhillSlopeThreshold ? c3 * rise : c2 * rise;
      energy = c0 * distance + (rise > 0 ? c1 * rise : downhill);
    } else {
      energy = distance * 0.6 * Math.exp(3.5 * Math.abs(slope + 0.05));
    }
    return (
      energy + settings.frictionWeight * 0.5 * (FRICTION[cell] + FRICTION[neighbor]) * distance
    );
  };
  const costs = new Float64Array(COUNT).fill(Infinity);
  costs[SOURCE] = 0;
  for (let changed = true; changed; ) {
    changed = false;
    for (let cell = 0; cell < COUNT; cell++) {
      const column = cell % WIDTH;
      const row = Math.floor(cell / WIDTH);
      for (let rowStep = -1; rowStep <= 1; rowStep++) {
        for (let columnStep = -1; columnStep <= 1; columnStep++) {
          const neighborColumn = column + columnStep;
          const neighborRow = row + rowStep;
          if ((columnStep === 0 && rowStep === 0) || neighborColumn < 0 || neighborRow < 0) {
            continue;
          }
          if (neighborColumn >= WIDTH || neighborRow >= HEIGHT) {
            continue;
          }
          const neighbor = neighborRow * WIDTH + neighborColumn;
          const cost = getTransitionCost(cell, neighbor, getStepDistance(row, columnStep, rowStep));
          if (cost >= 0 && costs[neighbor] + cost < costs[cell] - 1e-12) {
            costs[cell] = costs[neighbor] + cost;
            changed = true;
          }
        }
      }
    }
  }
  return costs;
}

function expectClose(actual: ArrayLike<number>, expected: Float64Array): void {
  for (let index = 0; index < COUNT; index++) {
    expect(Math.abs(actual[index] - expected[index])).toBeLessThanOrEqual(
      1e-4 * Math.max(1, Math.abs(expected[index]))
    );
  }
}

type Fixture = {
  graph: GPUCommandGraph;
  costs: Buffer;
  converged: Buffer;
  invalidCount: Buffer;
  settings: GPUParameterBuffer<'float32'>;
  owned: Buffer[];
};

function createFixture(
  device: Device,
  grid: GPURasterMetadata,
  settings: Settings,
  options: {cellSizeMode?: GPURasterCellSizeMode; packedSettings?: Settings} = {}
): Fixture {
  const graph = new GPUCommandGraph(device, {id: 'walking-cost-parameters-test'});
  const sources = new Uint32Array(COUNT);
  sources[SOURCE] = 1;
  const elevationBuffer = createInputBuffer(device, ELEVATION);
  const frictionBuffer = createInputBuffer(device, FRICTION);
  const sourcesBuffer = createInputBuffer(device, sources);
  const costs = createOutputBuffer(device, COUNT);
  const validity = createOutputBuffer(device, COUNT);
  const converged = createOutputBuffer(device, 1);
  const invalidCount = createOutputBuffer(device, 1);
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'walking-cost-settings',
    format: 'float32',
    length: GPU_RASTER_WALKING_COST_PARAMETER_LENGTH,
    values: getGPURasterWalkingCostParameterValues(options.packedSettings ?? settings, grid)
  });
  graph.add(
    new GPURasterWalkingCost({
      grid,
      cellSizeMode: options.cellSizeMode,
      elevation: {
        id: 'elevation',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: importGraphBuffer(graph, 'elevation', elevationBuffer, 'float32', COUNT)
        }
      },
      friction: {
        id: 'friction',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: importGraphBuffer(graph, 'friction', frictionBuffer, 'float32', COUNT)
        }
      },
      sources: importGraphBuffer(graph, 'sources', sourcesBuffer, 'uint32', COUNT),
      parameters: parameterBuffer.importToGraph(graph),
      model: settings.model,
      maximumIterations: 64,
      output: {
        costs: importGraphBuffer(graph, 'costs', costs, 'float32', COUNT),
        validity: importGraphBuffer(graph, 'validity', validity, 'uint32', COUNT),
        status: {
          converged: importGraphBuffer(graph, 'converged', converged, 'uint32', 1),
          invalidCount: importGraphBuffer(graph, 'invalid-count', invalidCount, 'uint32', 1)
        }
      }
    })
  );
  return {
    graph,
    costs,
    converged,
    invalidCount,
    settings: parameterBuffer,
    owned: [
      elevationBuffer,
      frictionBuffer,
      sourcesBuffer,
      costs,
      validity,
      converged,
      invalidCount
    ]
  };
}

/** Units in the last place between two float32 values, through their ordered bit patterns. */
function getUlpDistance(first: number, second: number): number {
  const ordered = (value: number) => {
    const bits = new Uint32Array(Float32Array.of(value).buffer)[0]!;
    return bits & 0x80000000 ? -(bits & 0x7fffffff) : bits;
  };
  return Math.abs(ordered(first) - ordered(second));
}

/**
 * The fixture was captured on the macOS (Metal) test adapter. Measured there on 2026-11-10 against
 * the `parameters` path, now the only one: Tobler reproduces the old kernel bit for bit (0 ulp in
 * all 35 cells), GRASS differs by 1 ulp in 8 of 35 cells (1, 2, 8, 23, 24, 25, 33, 34). An earlier
 * wave 1 measurement of the removed compile-time path recorded the same 8 cells under Tobler
 * instead; the labels of that note were swapped or the paths differ, and only this measurement
 * is of the path that remains. The rewrite is therefore not bit-identical for GRASS; the fixture
 * is the old kernel's output, not adjusted. `exp`
 * and `sqrt` are not correctly rounded in WGSL, so other adapters may differ in the last bits
 * too: the bound is 4 ulp, far below any change of the cost formula.
 */
function expectMatchesOldKernel(actual: ArrayLike<number>, expectedBits: readonly number[]): void {
  const expected = Array.from(new Float32Array(Uint32Array.from(expectedBits).buffer));
  const distances = expected.map((value, index) => getUlpDistance(actual[index]!, value));
  expect(Math.max(...distances), `ulp distances ${distances.join(',')}`).toBeLessThanOrEqual(4);
}

const TOBLER: Settings = {frictionWeight: 1, model: {kind: 'tobler'}};
const GRASS: Settings = {
  frictionWeight: 0.5,
  model: {kind: 'grass', coefficients: [0.72, 6, 1.9998, -1.9998], downhillSlopeThreshold: -0.2}
};

it('GPURasterWalkingCost parameters reproduce the 74dddcf5 kernel and rewrite the affine without recompiling', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const grid = makeGPURasterMetadata({
    width: WIDTH,
    height: HEIGHT,
    cellSize: 3,
    origin: [500000, 4600000]
  });
  for (const settings of [TOBLER, GRASS]) {
    const fixture = createFixture(device, grid, settings);
    const compiled = fixture.graph.compile();
    submitGraph(device, compiled, undefined);
    const costs = await readFloat32(fixture.costs, COUNT);
    // Regression against the kernel as it was at 74dddcf5, before the `parameters` rewrite.
    expectMatchesOldKernel(
      costs,
      WALKING_COST_74DDDCF5_COST_BITS[settings === TOBLER ? 'tobler' : 'grass']
    );
    expect(Array.from(await readUint32(fixture.converged, 1))).toEqual([1]);
    expect(Array.from(await readUint32(fixture.invalidCount, 1))).toEqual([0]);
    expectClose(costs, computeWalkingCosts(grid, settings, 'uniform'));

    // A different georeference (anisotropic cells) and friction weight, same compiled graph.
    const movedGrid = makeGPURasterMetadata({
      width: WIDTH,
      height: HEIGHT,
      cellSize: [5, 2],
      origin: [-120, 75]
    });
    const movedSettings = {...settings, frictionWeight: 2};
    fixture.settings.write(getGPURasterWalkingCostParameterValues(movedSettings, movedGrid));
    submitGraph(device, compiled, undefined);
    const movedCosts = await readFloat32(fixture.costs, COUNT);
    expectClose(movedCosts, computeWalkingCosts(movedGrid, movedSettings, 'uniform'));
    expect(Array.from(movedCosts)).not.toEqual(Array.from(costs));

    fixture.owned.forEach(buffer => buffer.destroy());
    fixture.settings.destroy();
    compiled.destroy();
  }
});

// The kernel cannot be told at construction what the parameter buffer holds. Before 2026-11-11,
// parameters packed for the other model were used as they were: GRASS read Tobler's zero
// coefficients (movement free, only friction cost), and Tobler ignored GRASS's. The packed
// threshold slot tells the models apart (zero for Tobler, negative for GRASS), so the kernel now
// reports the mismatch in `status.invalidCount` and moves nothing: every cell but the source stays
// unreached.
it('GPURasterWalkingCost reports parameters packed for the other model', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const grid = makeGPURasterMetadata({width: WIDTH, height: HEIGHT, cellSize: 3});
  for (const [settings, packedSettings] of [
    [GRASS, TOBLER],
    [TOBLER, GRASS]
  ] as const) {
    const fixture = createFixture(device, grid, settings, {packedSettings});
    const compiled = fixture.graph.compile();
    submitGraph(device, compiled, undefined);
    const label = `${settings.model.kind} kernel, ${packedSettings.model.kind} parameters`;
    expect(Array.from(await readUint32(fixture.invalidCount, 1)), label).toEqual([1]);
    const costs = await readFloat32(fixture.costs, COUNT);
    expect(Array.from(costs), label).toEqual(
      Array.from({length: COUNT}, (_, index) => (index === SOURCE ? 0 : Infinity))
    );

    // Repacking for the compiled model clears the report without recompiling.
    fixture.settings.write(getGPURasterWalkingCostParameterValues(settings, grid));
    submitGraph(device, compiled, undefined);
    expect(Array.from(await readUint32(fixture.invalidCount, 1)), label).toEqual([0]);
    expectClose(
      await readFloat32(fixture.costs, COUNT),
      computeWalkingCosts(grid, settings, 'uniform')
    );
    fixture.owned.forEach(buffer => buffer.destroy());
    fixture.settings.destroy();
    compiled.destroy();
  }
});

it('GPURasterWalkingCost converts geographic cells to ground meters with cellSizeMode', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // About 33 m north-south and 17 m east-west cells near 60 degrees north.
  const grid = makeGPURasterMetadata({
    width: WIDTH,
    height: HEIGHT,
    cellSize: 0.0003,
    origin: [10, 60]
  });
  const fixture = createFixture(device, grid, TOBLER, {
    cellSizeMode: 'geographic'
  });
  const compiled = fixture.graph.compile();
  submitGraph(device, compiled, undefined);
  const costs = await readFloat32(fixture.costs, COUNT);
  expectClose(costs, computeWalkingCosts(grid, TOBLER, 'geographic'));
  // Ground meters, not degrees: the cheapest neighbour step costs at least seconds of walking.
  expect(costs[SOURCE + 1]).toBeGreaterThan(1);
  fixture.owned.forEach(buffer => buffer.destroy());
  fixture.settings.destroy();
  compiled.destroy();
});

it('GPURasterWalkingCost rejects short parameters, rotated geographic grids and bad settings', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let identifier = 0;
  const view = <Format extends 'uint32' | 'float32'>(id: string, format: Format, length = 4) =>
    createTransientView(graph, `${id}-${identifier++}`, format, length);
  const band = (id: string) => ({
    id,
    format: 'float32' as const,
    storage: {kind: 'buffer' as const, values: view(id, 'float32')}
  });
  const create = (props: Partial<GPURasterWalkingCostProps>) =>
    new GPURasterWalkingCost({
      id: 'walking',
      grid: makeGPURasterMetadata({width: 2, height: 2}),
      elevation: band('elevation'),
      friction: band('friction'),
      sources: view('sources', 'uint32'),
      parameters: view('parameters', 'float32', GPU_RASTER_WALKING_COST_PARAMETER_LENGTH),
      output: {
        costs: view(`costs-${identifier++}`, 'float32'),
        validity: view(`validity-${identifier++}`, 'uint32'),
        status: {converged: view(`converged-${identifier++}`, 'uint32', 1)}
      },
      ...props
    });
  expect(() => create({parameters: view('short', 'float32', 8)})).toThrow(
    /^walking parameters must contain at least 16 values/
  );
  expect(() =>
    create({
      cellSizeMode: 'geographic',
      grid: {width: 2, height: 2, affine: [1, 0.1, 0, 0, -1, 0], pixelInterpretation: 'area'}
    })
  ).toThrow(/^walking geographic cell sizes need an axis-aligned grid/);
  expect(() => create({cellSizeMode: 'spherical' as GPURasterCellSizeMode})).toThrow(
    /^walking cellSizeMode must be uniform, web-mercator, or geographic/
  );
  const grid = makeGPURasterMetadata({width: 2, height: 2});
  expect(() => getGPURasterWalkingCostParameterValues({}, grid, new Float32Array(4))).toThrow(
    /^getGPURasterWalkingCostParameterValues target must hold 16 values/
  );
  expect(() =>
    getGPURasterWalkingCostParameterValues(
      {model: {kind: 'grass', coefficients: [1, 1, 1, 1], downhillSlopeThreshold: 0.1}},
      grid
    )
  ).toThrow(/^getGPURasterWalkingCostParameterValues.*downhillSlopeThreshold/);
});
