// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUTerrainHorizon,
  getGPUTerrainHorizonParameterValues
} from '../../../src/gpu-terrain/terrain-illumination/gpu-terrain-horizon';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {getRasterPipelines} from '../../gpu-raster/raster-pipeline-count';
import {createBand} from '../terrain-test-utils';
import {
  computeTerrainHorizonSweepBruteForce,
  computeTerrainHorizonSweepHull
} from './terrain-horizon-sweep-oracle';

function createFixture(width = 9, height = 9) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  return {
    device,
    graph,
    props: {
      grid: makeGPURasterMetadata({width, height}),
      maximumRadius: 5,
      parameters: createTransientView(graph, 'parameters', 'float32', 8),
      elevation: createBand(graph, 'elevation', width * height),
      output: {skyViewFactor: createTransientView(graph, 'sky-view', 'float32', width * height)}
    }
  };
}

it('own, inherited and non-enumerable geographic cell sizes all refuse local march halos (raster-terrain:F3)', () => {
  const {device, props} = createFixture();
  try {
    const own = {...props, maximumRadius: 2, cellSizeMode: 'geographic' as const};
    const inherited = Object.assign(
      Object.create({
        get cellSizeMode() {
          return 'geographic';
        }
      }),
      {...props, maximumRadius: 2}
    );
    const nonEnumerable = Object.defineProperty({...props, maximumRadius: 2}, 'cellSizeMode', {
      get: () => 'geographic'
    });
    for (const [form, supplied] of [
      ['own', own],
      ['inherited', inherited],
      ['non-enumerable', nonEnumerable]
    ] as const) {
      const contributor = new GPUTerrainHorizon(supplied);
      expect(() => contributor.getHaloStage(), form).toThrow(
        /latitude-dependent halo needs parameterLayout sweep/
      );
    }
  } finally {
    device.destroy();
  }
});

it('R1 padded base settings preserve the base layout and cannot claim a geographic halo', () => {
  const {device, graph, props} = createFixture();
  try {
    expect(() =>
      getGPUTerrainHorizonParameterValues({}, props.grid, new Float32Array(32))
    ).not.toThrow();
    for (const length of [8, 32]) {
      for (const algorithm of ['march', 'sweep'] as const) {
        const contributor = new GPUTerrainHorizon({
          ...props,
          algorithm,
          cellSizeMode: 'geographic',
          parameters: createTransientView(
            graph,
            `parameters-${algorithm}-${length}`,
            'float32',
            length
          )
        });
        expect(() => contributor.getHaloStage()).toThrow();
      }
    }
    expect(
      () => new GPUTerrainHorizon({...props, algorithm: 'sweep', parameterLayout: 'sweep'})
    ).toThrow(/24/);
  } finally {
    device.destroy();
  }
});

it('R2 packed geographic rows include the same distance-limited east blocker in both windows', () => {
  const elevation = new Float32Array(99);
  elevation[3 * 9 + 5] = 10;
  const cropped = new Float32Array(25);
  cropped[1 * 5 + 3] = 10;
  const fullGrid = makeGPURasterMetadata({
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
  for (const compute of [computeTerrainHorizonSweepBruteForce, computeTerrainHorizonSweepHull]) {
    for (const [grid, values, origin, pixel] of [
      [fullGrid, elevation, [0, 0], 3 * 9 + 4],
      [windowGrid, cropped, [2, 2], 1 * 5 + 2]
    ] as const) {
      const packed = getGPUTerrainHorizonParameterValues(
        {maximumDistance: 0.19493955373764038},
        grid
      );
      const source = getGPUTerrainHorizonParameterValues({}, fullGrid);
      const result = compute({
        width: grid.width,
        height: grid.height,
        elevation: values,
        origin,
        directionCount: 4,
        maximumRadius: 1,
        cellSizeMode: 'geographic',
        referenceCellSize: [1, 1],
        cellSize: [packed[0], packed[1]],
        northEdge: packed[4],
        southEdge: packed[5],
        maximumDistance: packed[6],
        sourceGeoreference: {
          cellSize: [source[0], source[1]],
          northEdge: source[4],
          southEdge: source[5],
          height: 11,
          rowOffset: origin[1]
        }
      });
      expect(result.validity[pixel]).toBe(1);
      expect(result.horizon[pixel * 4 + 1]).toBeCloseTo(88.88322, 3);
    }
  }
});

it('R3 probe demonstrates local float32 addition changes the validity corner', () => {
  const distance = Math.fround(3 - 2 ** -20);
  expect(Math.fround(32 + distance)).toBe(35);
  expect(Math.fround(4 + distance)).toBe(6.999999046325684);
  // Taking floor of the displacement first gives the same integer and fractional offsets.
  expect(Math.floor(distance)).toBe(2);
  expect(distance - Math.floor(distance)).toBe(0.9999990463256836);
});

it('R4 identical reference-cell bytes produce the same curvature decision in both CPU oracles', () => {
  const elevation = new Float32Array(15);
  elevation[1 * 5 + 3] = 8;
  for (const compute of [computeTerrainHorizonSweepBruteForce, computeTerrainHorizonSweepHull]) {
    for (const referenceCellSize of [
      [2, 2],
      [2 - 2 ** -26, 2]
    ] as const) {
      const result = compute({
        width: 5,
        height: 3,
        elevation,
        origin: [0, 0],
        directionCount: 4,
        maximumRadius: 2,
        curvatureCoefficient: 1,
        cellSizeMode: 'geographic',
        cellSize: [0.00002, 0.00002],
        northEdge: 60,
        southEdge: 59.99994,
        referenceCellSize
      });
      expect(result.horizon[(1 * 5 + 1) * 4 + 1]).toBeCloseTo(-48.13786, 3);
    }
  }
});

it('R5 tile placement does not change sweep shader source', () => {
  const sources = [
    [0, 0],
    [1, 0]
  ].map(origin => {
    const {device, graph, props} = createFixture();
    try {
      const sourceGrid = makeGPURasterMetadata({width: 10, height: 9});
      const packed = getGPUTerrainHorizonParameterValues(
        {
          sweepOrigin: origin as [number, number],
          sweepReferenceCellSize: [1, 1],
          sweepSourceGrid: sourceGrid
        },
        props.grid
      );
      expect(Array.from(packed.slice(14, 16))).toEqual(origin);
      return getRasterPipelines(
        new GPUTerrainHorizon({
          ...props,
          algorithm: 'sweep',
          parameterLayout: 'sweep',
          parameters: createTransientView(graph, 'sweep-parameters', 'float32', 24)
        }).getCommandNodes(graph)
      ).map(pipeline => pipeline.source);
    } finally {
      device.destroy();
    }
  });
  expect(sources[1]).toEqual(sources[0]);
});

it('R6 mutating constructor props cannot shrink a captured march halo', () => {
  const {device, props} = createFixture();
  try {
    const contributor = new GPUTerrainHorizon(props);
    props.maximumRadius = 1;
    expect(Array.from(contributor.stepDistances)).toEqual([1, 2, 3, 4, 5]);
    expect(contributor.getHaloStage()).toEqual({requiredHalo: 6});
  } finally {
    device.destroy();
  }
});

it('R7 march reserves a safe integer for the extra halo cell', () => {
  const {device, props} = createFixture(1, 1);
  try {
    expect(
      new GPUTerrainHorizon({...props, maximumRadius: Number.MAX_SAFE_INTEGER - 1}).getHaloStage()
    ).toEqual({requiredHalo: Number.MAX_SAFE_INTEGER});
    expect(() => new GPUTerrainHorizon({...props, maximumRadius: Number.MAX_SAFE_INTEGER})).toThrow(
      /maximumRadius/
    );
  } finally {
    device.destroy();
  }
});
