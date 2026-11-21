// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUPointCloudCoverageValidity} from '../../../src/pointcloud-analysis/operations/gpu-pointcloud-coverage-validity';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../../gpu-spatial-analysis/segment-intersection/wgsl-graph-executor';
import {
  createTransientView,
  GPUCommandGraph,
  type GPUCPUVector,
  type GraphDataView,
  type GPUGraphVectorFormat
} from '@luma.gl/graph/gpu-core';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  computeTerrainFlowOnCPU,
  CPUTerrainFlowOperation,
  type CPUTerrainFlowInput,
  type CPUTerrainFlowOptions
} from '../../../src/gpu-terrain/hydrology/cpu-terrain-flow-operation';
import {
  GPUTerrainFlowOperation,
  type TerrainFlowOperationProps
} from '../../../src/gpu-terrain/hydrology/gpu-terrain-flow-operation';
import {
  getGPUTerrainFlowParameterValues,
  GPU_TERRAIN_FLOW_CELL_CLASS as CELL_CLASS,
  GPU_TERRAIN_FLOW_NONE,
  GPUTerrainFlow,
  type GPUTerrainFlowSettings
} from '../../../src/gpu-terrain/hydrology/gpu-terrain-flow';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

function makeGrid(width: number, height: number): GPURasterMetadata {
  return {width, height, affine: [1, 0, 0, 0, -1, height], pixelInterpretation: 'area'};
}

function makeInput(
  values: number[] | Float32Array,
  grid: GPURasterMetadata,
  settings: GPUTerrainFlowSettings = {}
): CPUTerrainFlowInput {
  return {
    elevation: {id: 'elevation', format: 'float32', values: Float32Array.from(values)},
    parameters: getGPUTerrainFlowParameterValues(settings, grid)
  };
}

/**
 * Independent minimax oracle: find the first elevation sublevel set that connects each cell to
 * the exterior. It never relaxes a filled height or uses a priority queue. Eight-neighbor
 * connectivity is enumerated by coordinate pairs, independently of the implementation's D8 table.
 */
function computeSublevelFill(elevation: Float32Array, grid: GPURasterMetadata): number[] {
  const levels = [...new Set(Array.from(elevation).filter(Number.isFinite))].sort(
    (first, second) => first - second
  );
  const result = new Array<number>(elevation.length).fill(NaN);
  for (let start = 0; start < elevation.length; start++) {
    if (!Number.isFinite(elevation[start])) continue;
    for (const level of levels) {
      if (level < elevation[start]) continue;
      const visited = new Set([start]);
      const queue = [start];
      let escaped = false;
      for (let head = 0; head < queue.length && !escaped; head++) {
        const cell = queue[head];
        for (let rowOffset = -1; rowOffset <= 1; rowOffset++) {
          for (let columnOffset = -1; columnOffset <= 1; columnOffset++) {
            if (rowOffset === 0 && columnOffset === 0) continue;
            const column = (cell % grid.width) + columnOffset;
            const row = Math.floor(cell / grid.width) + rowOffset;
            const neighbor = row * grid.width + column;
            if (
              row < 0 ||
              column < 0 ||
              row >= grid.height ||
              column >= grid.width ||
              !Number.isFinite(elevation[neighbor])
            ) {
              escaped = true;
              break;
            }
            if (elevation[neighbor] <= level && !visited.has(neighbor)) {
              visited.add(neighbor);
              queue.push(neighbor);
            }
          }
        }
      }
      if (escaped) {
        result[start] = level;
        break;
      }
    }
  }
  return result;
}

it('matches independent sublevel connectivity for nested depressions, negative heights and holes', () => {
  const grid = makeGrid(7, 7);
  for (let scene = 0; scene < 8; scene++) {
    const values = Float32Array.from({length: 49}, (_, cell) => {
      // Repeatable non-monotone integer surfaces with several basins, without a random stream.
      const value = ((cell * cell * 7 + cell * 13 + scene * 11) % 23) - 12;
      return scene % 2 && (cell === 16 || cell === 32) ? NaN : value;
    });
    const result = computeTerrainFlowOnCPU(makeInput(values, grid), {grid, fillDepressions: true});
    expect(Array.from(result.filledElevation)).toEqual(computeSublevelFill(values, grid));
    expect(Array.from(result.status.fillConverged)).toEqual([1]);
  }
});

it('fills to the pour point and confirms positive-epsilon filling within its explicit budget', () => {
  const grid = makeGrid(7, 7);
  const values = Float32Array.from({length: 49}, (_, cell) =>
    cell < 7 || cell >= 42 || cell % 7 === 0 || cell % 7 === 6 ? 10 : -5
  );
  for (const epsilon of [0, 0.25]) {
    const result = computeTerrainFlowOnCPU(makeInput(values, grid, {fillEpsilon: epsilon}), {
      grid,
      fillDepressions: true,
      maximumFillIterations: epsilon > 0 ? 2 : 1
    });
    expect(Array.from(result.status.fillConverged)).toEqual([1]);
    expect(Array.from(result.status.fillIterationCount)).toEqual([epsilon > 0 ? 2 : 0]);
    for (let cell = 0; cell < 49; cell++) {
      const row = Math.floor(cell / 7);
      const column = cell % 7;
      // Every eight-connected path needs at least this many steps to reach the border; a
      // cardinal path achieves it, all interior original heights are lower than the border.
      const steps = Math.min(row, column, 6 - row, 6 - column);
      expect(result.filledElevation[cell]).toBe(10 + epsilon * steps);
    }
  }
  const noRise = computeTerrainFlowOnCPU(makeInput(values, grid, {fillEpsilon: -3}), {
    grid,
    fillDepressions: true
  });
  expect(noRise.filledElevation.every(value => value === 10)).toBe(true);
  const smallRise = computeTerrainFlowOnCPU(makeInput(values, grid, {fillEpsilon: 1e-8}), {
    grid,
    fillDepressions: true
  });
  // The internal float64 slope survives even though all published elevations round to 10.
  expect(smallRise.filledElevation.every(value => value === 10)).toBe(true);
  expect(smallRise.cellClasses[24]).toBe(CELL_CLASS.draining);
});

it('routes independent eastward chains, weights them unequally and includes ground area', () => {
  const grid = {...makeGrid(4, 3), affine: [2, 0, 0, 0, -3, 9] as const};
  const input = makeInput([4, 3, 2, 1, 4, 3, 2, 1, 4, 3, 2, 1], grid, {streamThreshold: 5});
  input.runoff = Float32Array.of(1, 2, 4, 8, 3, -1, 0.5, 2, Infinity, NaN, 7, 1);
  for (const area of [false, true]) {
    const result = computeTerrainFlowOnCPU(input, {
      grid,
      accumulationUnits: area ? 'area' : 'cells',
      maximumAccumulationIterations: 1
    });
    const expected = [1, 3, 7, 15, 3, 3, 3.5, 5.5, 0, 0, 7, 8].map(value => value * (area ? 6 : 1));
    expect(Array.from(result.accumulation)).toEqual(expected);
    expect(Array.from(result.flowDirections)).toEqual([1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0]);
    expect(Array.from(result.cellClasses)).toEqual([0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3]);
    expect(Array.from(result.streams)).toEqual(expected.map(value => Number(value >= 5)));
    expect(Array.from(result.status.accumulationConverged)).toEqual([1]);
  }
  delete input.runoff;
  expect(Array.from(computeTerrainFlowOnCPU(input, {grid}).accumulation)).toEqual([
    1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4
  ]);
});

it('resolves an open plateau by shortest hop masks with first-direction ties; closed flats remain terminal', () => {
  const grid = makeGrid(5, 5);
  const input = makeInput(new Float32Array(25).fill(10), grid);
  const result = computeTerrainFlowOnCPU(input, {
    grid,
    resolveFlats: true
  });
  // Border cells are low edges, mask 2. The inner ring has mask 4, center mask 6.
  expect(result.flowDirections[12]).toBe(1);
  expect(result.flowDirections[6]).toBe(8); // SW is the first border neighbor of (1,1).
  expect(result.cellClasses.filter(value => value === CELL_CLASS.flat).length).toBe(0);
  expect(
    result.accumulation
      .filter((_, cell) => result.flowDirections[cell] === 0)
      .reduce((total, value) => total + value, 0)
  ).toBe(25);
  const closedValues = new Float32Array(25).fill(0);
  for (let cell = 0; cell < 25; cell++)
    if (cell < 5 || cell >= 20 || cell % 5 === 0 || cell % 5 === 4) closedValues[cell] = 10;
  const closed = computeTerrainFlowOnCPU(makeInput(closedValues, grid), {grid, resolveFlats: true});
  expect(closed.flowDirections[12]).toBe(0);
  expect(closed.cellClasses[12]).toBe(CELL_CLASS.flat);
  expect(closed.accumulation[12]).toBe(1);
  const pit = computeTerrainFlowOnCPU(makeInput([5, 5, 5, 5, 0, 5, 5, 5, 5], makeGrid(3, 3)), {
    grid: makeGrid(3, 3),
    resolveFlats: true
  });
  expect(pit.cellClasses[4]).toBe(CELL_CLASS.pit);
  expect(pit.accumulation[4]).toBe(9);
});

for (const routing of ['mfd-freeman', 'mfd-quinn'] as const) {
  it(`conserves a central pulse and has the hand-computed 2:1 cardinal/diagonal split for ${routing}`, () => {
    const grid = makeGrid(3, 3);
    const input = makeInput([0, 0, 0, 0, 8, 0, 0, 0, 0], grid, {
      flowExponent: routing === 'mfd-freeman' ? 2 : 0,
      streamThreshold: 2
    });
    input.runoff = Float32Array.of(0, 0, 0, 0, 12, 0, 0, 0, 0);
    const result = computeTerrainFlowOnCPU(input, {grid, flowRouting: routing});
    // Four cardinals each receive 2, four diagonals each receive 1; the center contributes 12.
    expect(Array.from(result.accumulation)).toEqual([1, 2, 1, 2, 12, 2, 1, 2, 1]);
    expect(result.flowDirections[4]).toBe(1);
    expect(Array.from(result.streams)).toEqual([0, 1, 0, 1, 1, 1, 0, 1, 0]);
    expect(
      result.accumulation.reduce((total, value, cell) => total + (cell === 4 ? 0 : value), 0)
    ).toBe(12);
  });
}

it('the higher-edge gradient breaks a lower-edge tie in the hand-computed Barnes mask', () => {
  const grid = makeGrid(7, 7);
  const values = Float32Array.from({length: 49}, (_, cell) =>
    cell % 7 === 0 ? 20 : cell % 7 === 6 ? 0 : 10
  );
  const result = computeTerrainFlowOnCPU(makeInput(values, grid), {grid, resolveFlats: true});
  // Flat columns 1..4 have higher-edge distances 1..4, component maximum 4.
  // At (row 2, col 1), lower distance = 3, mask = 2*3+(4-1) = 9.
  // North (1,1) has mask 7; NE (1,2) has mask 6 and wins. Omitting awayFromHigher
  // gives both neighbors mask 4 and incorrectly chooses north before NE.
  expect(result.flowDirections[15]).toBe(128);
  expect(result.flowDirections[23]).toBe(1);
  expect(result.cellClasses.filter(value => value === CELL_CLASS.flat).length).toBe(0);
  const terminalTotal = result.accumulation.reduce(
    (total, value, cell) => total + (result.flowDirections[cell] === 0 ? value : 0),
    0
  );
  expect(terminalTotal).toBe(49);
});

it('D-infinity splits a pulse at atan(1/2) inside E/SE, and reunites it at the lower terminal', () => {
  const grid = makeGrid(3, 3);
  const input = makeInput([20, 20, 20, 20, 10, 8, 20, 20, 7], grid);
  input.runoff = Float32Array.of(0, 0, 0, 0, 12, 0, 0, 0, 0);
  const result = computeTerrainFlowOnCPU(input, {grid, flowRouting: 'd-infinity'});
  expect(result.accumulation[4]).toBe(12);
  expect(result.accumulation[5]).toBe(Math.fround(12 * (1 - Math.atan(0.5) / (Math.PI / 4))));
  expect(result.accumulation[8]).toBe(12);
  expect(result.accumulation[5]).toBeGreaterThan(0);
  expect(result.accumulation[5]).toBeLessThan(12);
});

it('uses the same resolved-flat receiver fallback for every multiple routing', () => {
  const grid = makeGrid(5, 5);
  const input = makeInput(new Float32Array(25).fill(7), grid);
  const baseline = computeTerrainFlowOnCPU(input, {grid, resolveFlats: true});
  for (const flowRouting of ['d-infinity', 'mfd-freeman', 'mfd-quinn'] as const) {
    const result = computeTerrainFlowOnCPU(input, {grid, resolveFlats: true, flowRouting});
    expect(result.flowDirections).toEqual(baseline.flowDirections);
    expect(result.accumulation).toEqual(baseline.accumulation);
  }
});

it('handles a single usable facet endpoint and cardinal D-infinity ties', () => {
  const grid = makeGrid(3, 3);
  for (const target of [5, 8]) {
    const input = makeInput(new Float32Array(9).fill(NaN), grid);
    input.elevation.values[4] = 10;
    input.elevation.values[target] = 0;
    input.runoff = new Float32Array(9);
    input.runoff[4] = 12;
    const result = computeTerrainFlowOnCPU(input, {grid, flowRouting: 'd-infinity'});
    expect(result.accumulation[4]).toBe(12);
    expect(result.accumulation[target]).toBe(12);
  }
  const input = makeInput([0, 0, 0, 0, 8, 0, 0, 0, 0], grid);
  input.runoff = Float32Array.of(0, 0, 0, 0, 12, 0, 0, 0, 0);
  const tie = computeTerrainFlowOnCPU(input, {grid, flowRouting: 'd-infinity'});
  expect(Array.from(tie.accumulation)).toEqual([0, 0, 0, 0, 12, 12, 0, 0, 0]);
});

it('normalizes MFD before very large powers, preserving mass and dropping zero-fraction edges', () => {
  const grid = makeGrid(3, 3);
  const input = makeInput([0, 0, 0, 0, 8, 0, 0, 0, 0], grid, {flowExponent: 1e30});
  input.runoff = Float32Array.of(0, 0, 0, 0, 12, 0, 0, 0, 0);
  for (const flowRouting of ['mfd-freeman', 'mfd-quinn'] as const) {
    const result = computeTerrainFlowOnCPU(input, {grid, flowRouting});
    expect(Array.from(result.accumulation)).toEqual([0, 3, 0, 3, 12, 3, 0, 3, 0]);
    expect(result.status.accumulationConverged[0]).toBe(1);
  }
});

it('reads per-frame cell sizes from the packed parameters rather than the static grid', () => {
  const grid = makeGrid(3, 3);
  const input = makeInput([20, 20, 20, 20, 10, 8, 20, 9, 20], grid);
  expect(computeTerrainFlowOnCPU(input, {grid}).flowDirections[4]).toBe(1);
  input.parameters = getGPUTerrainFlowParameterValues({}, {...grid, affine: [4, 0, 0, 0, -1, 3]});
  expect(computeTerrainFlowOnCPU(input, {grid}).flowDirections[4]).toBe(4);
});

it('calibrates scalar bands, rounds float32 nodata, honors validity and treats a hole as boundary', () => {
  const grid = makeGrid(3, 3);
  const input = makeInput(new Float32Array(9).fill(1), grid);
  input.elevation.values[0] = Math.fround(-9999.1);
  input.elevation.noDataValue = -9999.1;
  input.elevation.validity = Uint32Array.of(1, 1, 0, 1, 1, 1, 1, 1, 1);
  input.elevation.scale = 2;
  input.elevation.offset = -1;
  const result = computeTerrainFlowOnCPU(input, {grid, fillDepressions: true, resolveFlats: true});
  expect(result.filledElevation[0]).toBeNaN();
  expect(result.filledElevation[2]).toBeNaN();
  expect(result.filledElevation[4]).toBe(1);
  expect(result.cellClasses[4]).toBe(CELL_CLASS.outlet);
  expect(result.flowDirections[4]).toBe(0);
  expect(result.flowDirections[0]).toBe(GPU_TERRAIN_FLOW_NONE);
  expect(result.accumulation[0]).toBeNaN();
  expect(result.streams[0]).toBe(0);
  for (const format of ['uint32', 'sint32'] as const) {
    const integer = {
      ...input,
      elevation: {
        ...input.elevation,
        ...(format === 'uint32'
          ? {format, noDataValue: 0xffffffff, values: new Uint32Array(9).fill(1)}
          : {format, noDataValue: -0x80000000, values: new Int32Array(9).fill(1)})
      }
    };
    expect(computeTerrainFlowOnCPU(integer, {grid}).filledElevation[4]).toBe(1);
  }
});

it('uses closed-form equatorial area for geographic and web-mercator rows', () => {
  const grid: GPURasterMetadata = {
    ...makeGrid(3, 1),
    affine: [1, 0, 0, 0, -1, 0],
    pixelInterpretation: 'point'
  };
  const input = makeInput([3, 2, 1], grid);
  const flattening = 1 / 298.257223563;
  const polarRatio = (1 - flattening) ** 2;
  for (const cellSizeMode of ['geographic', 'web-mercator'] as const) {
    const result = computeTerrainFlowOnCPU(input, {grid, cellSizeMode, accumulationUnits: 'area'});
    const area =
      polarRatio * (cellSizeMode === 'geographic' ? ((6378137 * Math.PI) / 180) ** 2 : 1);
    expect(Array.from(result.flowDirections)).toEqual([1, 1, 0]);
    for (let cell = 0; cell < 3; cell++)
      expect(result.accumulation[cell]).toBe(Math.fround(area * (cell + 1)));
  }
});

it('rounds once at output, thresholds the rounded value, and keeps long chains exact', () => {
  const grid = makeGrid(2050, 1);
  const input = makeInput(
    Float32Array.from({length: 2050}, (_, cell) => 2050 - cell),
    grid
  );
  const result = computeTerrainFlowOnCPU(input, {grid, maximumAccumulationIterations: 1});
  expect(result.accumulation[2049]).toBe(2050);
  const shortGrid = makeGrid(3, 1);
  const weighted = makeInput([3, 2, 1], shortGrid, {streamThreshold: 1});
  weighted.runoff = Float32Array.of(1, 2 ** -24, 2 ** -24);
  const rounded = computeTerrainFlowOnCPU(weighted, {grid: shortGrid});
  expect(rounded.accumulation[2]).toBe(1 + 2 ** -23);
  weighted.runoff = Float32Array.of(1 - 2 ** -24, 2 ** -25, 0);
  expect(computeTerrainFlowOnCPU(weighted, {grid: shortGrid}).streams[1]).toBe(1);
});

it('writes all-invalid and singleton layouts, statuses and nonfinite stream codes', () => {
  const grid = makeGrid(1, 1);
  const input = makeInput([NaN], grid);
  const invalid = computeTerrainFlowOnCPU(input, {grid, fillDepressions: true, resolveFlats: true});
  expect(invalid.flowDirections[0]).toBe(GPU_TERRAIN_FLOW_NONE);
  expect(invalid.cellClasses[0]).toBe(CELL_CLASS.invalid);
  expect(invalid.streams[0]).toBe(0);
  expect(invalid.accumulation[0]).toBeNaN();
  expect(Object.values(invalid.status).map(values => values[0])).toEqual([1, 1, 1, 0, 0, 0]);
  const valid = computeTerrainFlowOnCPU(makeInput([3], grid), {grid});
  expect(valid.flowDirections[0]).toBe(0);
  expect(valid.cellClasses[0]).toBe(CELL_CLASS.outlet);
  expect(valid.accumulation[0]).toBe(1);
  const hugeGrid = {...makeGrid(1, 1), affine: [1e30, 0, 0, 0, -1e30, 0] as const};
  const huge = computeTerrainFlowOnCPU(makeInput([3], hugeGrid), {
    grid: hugeGrid,
    accumulationUnits: 'area'
  });
  expect(huge.accumulation[0]).toBe(Infinity);
  expect(huge.streams[0]).toBe(0xffffffff);
});

function makeProps(): TerrainFlowOperationProps {
  return {
    id: 'flow-validation',
    grid: makeGrid(3, 3),
    elevation: {id: 'elevation', format: 'float32', storage: {kind: 'buffer', values: 'elevation'}},
    parameters: 'parameters',
    output: {flowDirections: 'directions', accumulation: 'accumulation'}
  };
}

it('validates construction of both operation declarations and CPU static options', () => {
  const props = makeProps();
  for (const Operation of [GPUTerrainFlowOperation, CPUTerrainFlowOperation]) {
    expect(() => new Operation({...props, grid: {...props.grid, width: 0}})).toThrow(
      /^flow-validation/
    );
    expect(() => new Operation({...props, output: {accumulation: 'elevation'}})).toThrow(
      /^flow-validation/
    );
    expect(
      () => new Operation({...props, output: {flowDirections: 'shared', streams: 'shared'}})
    ).toThrow(/^flow-validation/);
    expect(() => new Operation(props)).not.toThrow();
  }
  for (const maximum of [
    'maximumFillIterations',
    'maximumFlatIterations',
    'maximumAccumulationIterations'
  ] as const) {
    expect(() => new CPUTerrainFlowOperation({...props, [maximum]: 0})).toThrow(
      new RegExp(`flow-validation.*${maximum}`)
    );
    if (maximum === 'maximumFlatIterations') {
      expect(() => new CPUTerrainFlowOperation({...props, [maximum]: 1024})).not.toThrow();
      expect(() => new CPUTerrainFlowOperation({...props, [maximum]: 1025})).toThrow(
        /maximumFlatIterations.*1024/
      );
    } else expect(() => new CPUTerrainFlowOperation({...props, [maximum]: 10000})).not.toThrow();
  }
  expect(() => Reflect.construct(CPUTerrainFlowOperation, [{...props, output: {}}])).toThrow(
    /flow-validation.*output/
  );
  expect(
    () => new CPUTerrainFlowOperation({...props, output: {filledElevation: 'filled'}})
  ).toThrow(/flow-validation.*fillDepressions/);
  expect(
    () =>
      new CPUTerrainFlowOperation({
        ...props,
        output: {...props.output, status: {flatsConverged: 'flats'}}
      })
  ).toThrow(/flow-validation.*resolveFlats/);
  expect(
    () =>
      new CPUTerrainFlowOperation({
        ...props,
        output: {flowDirections: 'directions', status: {accumulationConverged: 'converged'}}
      })
  ).toThrow(/flow-validation.*accumulation/);
  expect(
    () =>
      new CPUTerrainFlowOperation({...props, grid: {...props.grid, affine: [1, 1, 0, 0, -1, 0]}})
  ).toThrow(/flow-validation.*orthogonal/);
  expect(
    () =>
      new CPUTerrainFlowOperation({
        ...props,
        grid: {...props.grid, affine: [0, 1, 0, 1, 0, 0]},
        cellSizeMode: 'geographic'
      })
  ).toThrow(/flow-validation.*axis-aligned/);
  expect(
    () =>
      new CPUTerrainFlowOperation({
        ...props,
        flowRouting: 'bad' as CPUTerrainFlowOptions['flowRouting']
      })
  ).toThrow(/flow-validation.*flowRouting/);
});

it('throws argument-specific errors for inconsistent arrays', () => {
  const grid = makeGrid(3, 3);
  const input = makeInput(new Float32Array(9), grid);
  expect(() =>
    computeTerrainFlowOnCPU(
      {...input, elevation: {...input.elevation, format: 'float32', values: new Float32Array(8)}},
      {grid}
    )
  ).toThrow(/elevation.values/);
  expect(() =>
    computeTerrainFlowOnCPU(
      {...input, elevation: {...input.elevation, validity: new Uint32Array(10)}},
      {grid}
    )
  ).toThrow(/elevation.validity/);
  expect(() =>
    computeTerrainFlowOnCPU({...input, parameters: new Float32Array(7)}, {grid})
  ).toThrow(/parameters/);
  expect(() => computeTerrainFlowOnCPU({...input, runoff: new Float32Array(10)}, {grid})).toThrow(
    /runoff/
  );
});

for (const [format, noDataValue] of [
  ['uint32', -1],
  ['uint32', 0.5],
  ['uint32', 0x100000000],
  ['uint32', NaN],
  ['uint32', Infinity],
  ['sint32', -0x80000001],
  ['sint32', 0x80000000],
  ['sint32', -0.5],
  ['sint32', NaN],
  ['sint32', Infinity],
  ['float32', Infinity],
  ['float32', -Infinity],
  ['float32', 1e39],
  ['float32', -1e39]
] as const) {
  it(`rejects ${format} no-data metadata ${noDataValue} at construction and in the pure function`, () => {
    const grid = makeGrid(1, 1);
    const props: TerrainFlowOperationProps = {
      ...makeProps(),
      id: 'review-flow',
      grid,
      elevation: {...makeProps().elevation, format, noDataValue},
      output: {accumulation: 'accumulation'}
    };
    const input = makeInput([7], grid);
    input.elevation = {
      ...input.elevation,
      noDataValue,
      ...(format === 'uint32'
        ? {format, values: Uint32Array.of(7)}
        : format === 'sint32'
          ? {format, values: Int32Array.of(7)}
          : {format, values: Float32Array.of(7)})
    };
    // An unsigned 32-bit sentinel is an integer in [0, 2^32-1], a signed one in
    // [-2^31, 2^31-1]. NaN is allowed only for float32, whose finite sentinels must
    // remain finite on rounding. None of the rows above has a representable sentinel.
    expect(() => new CPUTerrainFlowOperation(props)).toThrow(/^review-flow.*elevation.noDataValue/);
    expect(() => computeTerrainFlowOnCPU(input, {grid})).toThrow(
      /^computeTerrainFlowOnCPU.*elevation.noDataValue/
    );
  });
}

for (const [format, noDataValue] of [
  ['uint32', 0],
  ['uint32', 0xffffffff],
  ['sint32', -0x80000000],
  ['sint32', 0x7fffffff],
  ['float32', -9999.1],
  ['float32', Math.fround(3.4028234663852886e38)],
  ['float32', NaN]
] as const) {
  it(`accepts the ${format} sentinel ${noDataValue} and masks only a matching sample`, () => {
    const grid = makeGrid(1, 1);
    const props: TerrainFlowOperationProps = {
      ...makeProps(),
      grid,
      elevation: {...makeProps().elevation, format, noDataValue}
    };
    expect(() => new CPUTerrainFlowOperation(props)).not.toThrow();
    for (const missing of [false, true]) {
      const value = missing ? noDataValue : 7;
      const input = makeInput([7], grid);
      input.elevation = {
        ...input.elevation,
        noDataValue,
        ...(format === 'uint32'
          ? {format, values: Uint32Array.of(value)}
          : format === 'sint32'
            ? {format, values: Int32Array.of(value)}
            : {format, values: Float32Array.of(value)})
      };
      const result = computeTerrainFlowOnCPU(input, {grid});
      // A lone finite, unmasked cell is a terminal outlet and contributes itself once.
      // A matching sentinel (including NaN) is missing and contributes no stream cell.
      expect(result.flowDirections[0]).toBe(missing ? GPU_TERRAIN_FLOW_NONE : 0);
      expect(result.cellClasses[0]).toBe(missing ? CELL_CLASS.invalid : CELL_CLASS.outlet);
      if (missing) expect(result.accumulation[0]).toBeNaN();
      else expect(result.accumulation[0]).toBe(1);
      expect(result.streams[0]).toBe(0);
    }
  });
}

it('the GPU operation binds a real contributor and defers its domain checks until view binding', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = makeProps();
  const views: Record<string, GraphDataView> = {
    values: createTransientView(graph, 'dem', 'float32', 9),
    parameters: createTransientView(graph, 'settings', 'float32', 8),
    flowDirections: createTransientView(graph, 'directions', 'uint32', 9),
    accumulation: createTransientView(graph, 'accumulation', 'float32', 9)
  };
  const getView = <Format extends GPUGraphVectorFormat>(port: string) =>
    views[port] as GraphDataView<Format>;
  try {
    expect(new GPUTerrainFlowOperation(props).createContributor(getView)).toBeInstanceOf(
      GPUTerrainFlow
    );
    // Section 14: the GPU wrapper adds no validation; the actual contributor validates bound props.
    expect(() =>
      new GPUTerrainFlowOperation({...props, maximumAccumulationIterations: 0}).createContributor(
        getView
      )
    ).toThrow(/flow-validation.*maximumAccumulationIterations/);
    expect(() =>
      new GPUTerrainFlowOperation({
        ...props,
        grid: {...props.grid, affine: [1, 1, 0, 0, -1, 0]}
      }).createContributor(getView)
    ).toThrow(/flow-validation.*orthogonal/);
    views['parameters'] = createTransientView(graph, 'short-settings', 'float32', 7);
    expect(() => new GPUTerrainFlowOperation(props).createContributor(getView)).toThrow(
      /flow-validation.*parameters/
    );
  } finally {
    device.destroy();
  }
});

it('the CPU callback reads graph band/parameter/runoff ports and writes only requested outputs', async () => {
  const grid = makeGrid(3, 1);
  const props = {
    ...makeProps(),
    grid,
    fillDepressions: true,
    resolveFlats: true,
    runoff: 'runoff',
    output: {
      streams: 'streamMask',
      status: {
        fillConverged: 'fillDone',
        flatsConverged: 'flatDone',
        accumulationConverged: 'accDone',
        fillIterationCount: 'fillRounds',
        flatsIterationCount: 'flatRounds',
        accumulationIterationCount: 'accRounds'
      }
    }
  };
  const operation = new CPUTerrainFlowOperation(props);
  const inputs: Record<string, GPUCPUVector> = {};
  for (const [port, values] of Object.entries({
    values: Float32Array.of(3, 2, 1),
    parameters: getGPUTerrainFlowParameterValues({streamThreshold: 4}, grid),
    runoff: Float32Array.of(1, 2, 3)
  }))
    inputs[port] = {
      format: 'float32',
      length: values.length,
      chunks: [{values, length: values.length}],
      pack: () => values
    };
  const rows = await operation.execute({inputs});
  expect(Object.keys(rows).sort()).toEqual(
    [
      'streamMask',
      'fillDone',
      'flatDone',
      'accDone',
      'fillRounds',
      'flatRounds',
      'accRounds'
    ].sort()
  );
  expect(Array.from(rows['streamMask'] as Uint32Array)).toEqual([0, 0, 1]);
  for (const name of ['fillDone', 'flatDone', 'accDone'])
    expect(Array.from(rows[name] as Uint32Array)).toEqual([1]);
  for (const name of ['fillRounds', 'flatRounds', 'accRounds'])
    expect(Array.from(rows[name] as Uint32Array)).toEqual([0]);
});

it('UNI-01 uses point-cloud ground validity for receivers and preserves genuine zero-height ground', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const coverage = createTransientView(graph, 'coverage', 'uint32', 2);
  const output = {
    surfaceValidity: createTransientView(graph, 'surface-validity', 'uint32', 2),
    terrainValidity: createTransientView(graph, 'terrain-validity', 'uint32', 2),
    canopyValidity: createTransientView(graph, 'canopy-validity', 'uint32', 2)
  };
  const memory = createCPUGraphMemory();
  const converter = new GPUPointCloudCoverageValidity({coverage, output});
  const nodes = converter.getCommandNodes(graph);
  const grid = makeGrid(2, 1);
  try {
    for (const coverageRows of [Uint32Array.of(1, 3), Uint32Array.of(3, 3)]) {
      memory.write(coverage, coverageRows);
      executeCommandNodesOnCPU(nodes, memory);
      const validity = memory.readUint32(output.terrainValidity);
      const groundAtZero = coverageRows[0] === 3;
      expect(Array.from(validity)).toEqual(groundAtZero ? [1, 1] : [0, 1]);
      const input = makeInput([0, 5], grid);
      // Raw coverage marks a non-ground cell as truthy and incorrectly permits westward flow.
      if (!groundAtZero) {
        const unsafe = computeTerrainFlowOnCPU(
          {...input, elevation: {...input.elevation, validity: coverageRows}},
          {grid}
        );
        expect(Array.from(unsafe.flowDirections)).toEqual([0, 16]);
      }
      const result = computeTerrainFlowOnCPU(
        {...input, elevation: {...input.elevation, validity}},
        {grid}
      );
      // Only genuine ground is a receiver. Sea-level ground remains a usable western outlet.
      expect(Array.from(result.flowDirections)).toEqual(
        groundAtZero ? [0, 16] : [GPU_TERRAIN_FLOW_NONE, 0]
      );
      expect(result.filledElevation[1]).toBe(5);
      if (groundAtZero) expect(result.filledElevation[0]).toBe(0);
      else expect(result.filledElevation[0]).toBeNaN();
      expect(result.accumulation[1]).toBe(1);
    }
  } finally {
    device.destroy();
  }
});

it('R1-N3 distinguishes dependency convergence from numeric area overflow', () => {
  const grid = {...makeGrid(1, 1), affine: [1e20, 0, 0, 0, -1e20, 0] as const};
  const result = computeTerrainFlowOnCPU(makeInput([5], grid), {grid, accumulationUnits: 'area'});
  // A lone vertex has no pending donors, yet its area 10^40 exceeds binary32's finite range.
  expect(result.status.accumulationConverged[0]).toBe(1);
  expect(result.accumulation[0]).toBe(Infinity);
  expect(result.streams[0]).toBe(0xffffffff);
});
