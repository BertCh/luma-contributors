// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPUDistanceFieldParameterValues,
  GPUDistanceField,
  GPU_DISTANCE_FIELD_MAXIMUM_DIMENSION,
  GPU_DISTANCE_FIELD_NONE,
  GPU_DISTANCE_FIELD_PARAMETER_LENGTH,
  type GPUDistanceFieldProps
} from '../../../src/raster-analysis/distance-field/index';
import {GPU_DISTANCE_FIELD_MAXIMUM_REFINEMENT_COUNT} from '../../../src/raster-analysis/distance-field/distance-field-parameters';
import {getDistanceFieldJumpFloodSteps} from '../../../src/raster-analysis/distance-field/distance-field-kernels';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {getRasterPipelines} from '../raster-pipeline-count';
import {computeDistanceFieldOnCPU, getCellSeeds, getUlpDistance} from './distance-field-oracle';

const WIDTH = 20;
const HEIGHT = 10;
const CELLS = WIDTH * HEIGHT;
const NONE = GPU_DISTANCE_FIELD_NONE;

function createBaseProps(graph: GPUCommandGraph) {
  return {
    grid: makeGPURasterMetadata({width: WIDTH, height: HEIGHT}),
    parameters: createTransientView(graph, 'parameters', 'float32', 8),
    output: {
      distances: createTransientView(graph, 'distances', 'float32', CELLS)
    }
  };
}

function createPointProps(graph: GPUCommandGraph) {
  return {
    ...createBaseProps(graph),
    seedPositions: createTransientView(graph, 'positions', 'float32x2', 4)
  };
}

it('packs distance-field settings', () => {
  expect(GPU_DISTANCE_FIELD_PARAMETER_LENGTH).toBe(8);
  expect(GPU_DISTANCE_FIELD_NONE).toBe(0xffffffff);
  expect(GPU_DISTANCE_FIELD_MAXIMUM_DIMENSION).toBe(32768);
  // North-up grid: the row step is negative and the origin is the top-left corner.
  expect(
    Array.from(
      getGPUDistanceFieldParameterValues(
        {},
        makeGPURasterMetadata({width: 4, height: 3, cellSize: [2, 0.5], origin: [3, -4]})
      )
    )
  ).toEqual([3, -4, 2, 0.5, 0.5, -2, Infinity, 0]);
  // North-down grid: row 0 is the minimum-y edge.
  expect(
    Array.from(
      getGPUDistanceFieldParameterValues(
        {maximumDistance: 9, localOrigin: [1, 1]},
        makeGPURasterMetadata({
          width: 50,
          height: 4,
          cellSize: [2, 5],
          origin: [0, 10],
          rowDirection: 'north'
        })
      )
    )
  ).toEqual([-1, 9, 2, 5, 0.5, 0.20000000298023224, 9, 0]);
  const grid = makeGPURasterMetadata({width: 2, height: 2});
  expect(() =>
    getGPUDistanceFieldParameterValues({}, {...grid, affine: [0, 0, 0, 0, -1, 0]})
  ).toThrow(/positive/);
  expect(() =>
    getGPUDistanceFieldParameterValues({}, {...grid, affine: [1, 0.5, 0, 0, -1, 0]})
  ).toThrow(/axis-aligned/);
  expect(() => getGPUDistanceFieldParameterValues({}, grid, new Float32Array(4))).toThrow(
    /8 values/
  );
});

it('GPUDistanceField builds deterministic exact and jump-flood node lists', () => {
  const device = createNullWebGPUDevice();
  const getNodeIds = (mode?: GPUDistanceFieldProps['mode']) => {
    const graph = new GPUCommandGraph(device);
    const contributor = new GPUDistanceField({
      ...createPointProps(graph),
      id: 'df',
      mode
    });
    const first = contributor.getCommandNodes(graph).map(node => node.id);
    return first;
  };
  expect(getNodeIds()).toEqual([
    'df-seed-clear',
    'df-seed-scatter',
    'df-column-bands',
    'df-column-carry',
    'df-column-fill',
    'df-row-bands',
    'df-row-marker-clear',
    'df-row-start-markers',
    'df-row-marker-bands',
    'df-row-marker-carry',
    'df-row-color',
    'df-finalize'
  ]);
  // 20x10 rounds up to 32: steps 16, 8, 4, 2, 1 plus one refinement pass.
  expect(getNodeIds('jump-flood')).toEqual([
    'df-seed-clear',
    'df-seed-scatter',
    'df-jump-flood-initialize',
    'df-jump-flood-steps',
    ...[0, 1, 2, 3, 4, 5].map(step => `df-jump-flood-${step}`),
    'df-finalize'
  ]);
  const graph = new GPUCommandGraph(device);
  const contributor = new GPUDistanceField({
    ...createBaseProps(graph),
    seedMask: createTransientView(graph, 'mask', 'uint32', CELLS)
  });
  expect(contributor.id).toBe('distance-field');
  expect(contributor.getCommandNodes(graph).map(node => node.id)).toEqual([
    'distance-field-seed-clear',
    'distance-field-column-bands',
    'distance-field-column-carry',
    'distance-field-column-fill',
    'distance-field-row-bands',
    'distance-field-row-marker-clear',
    'distance-field-row-start-markers',
    'distance-field-row-marker-bands',
    'distance-field-row-marker-carry',
    'distance-field-row-color',
    'distance-field-finalize'
  ]);
});

it('getDistanceFieldJumpFloodSteps halves from the padded size and appends refinement', () => {
  expect(getDistanceFieldJumpFloodSteps({width: 1, height: 1}, 1)).toEqual([1]);
  expect(getDistanceFieldJumpFloodSteps({width: 1, height: 1}, 0)).toEqual([]);
  expect(getDistanceFieldJumpFloodSteps({width: 256, height: 3}, 0)).toEqual([
    128, 64, 32, 16, 8, 4, 2, 1
  ]);
  expect(getDistanceFieldJumpFloodSteps({width: 5, height: 9}, 2)).toEqual([8, 4, 2, 1, 2, 1]);
});

it('GPUDistanceField validates its props', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const baseProps = createPointProps(graph);
  const create = (overrides: Partial<Omit<GPUDistanceFieldProps, 'seedPositions'>>) =>
    new GPUDistanceField({...baseProps, ...overrides});
  const otherGraph = new GPUCommandGraph(device);
  expect(() => create({grid: makeGPURasterMetadata({width: 0, height: HEIGHT})})).toThrow(
    /grid dimensions/
  );
  expect(() =>
    create({
      grid: makeGPURasterMetadata({width: 1, height: GPU_DISTANCE_FIELD_MAXIMUM_DIMENSION + 1})
    })
  ).toThrow(/height/);
  expect(() => create({mode: 'nearest' as never})).toThrow(/mode/);
  expect(() => create({jumpFloodRefinementIterationCount: 3 as never})).toThrow(
    /jumpFloodRefinementIterationCount/
  );
  expect(() => create({parameters: createTransientView(graph, 's4', 'float32', 4)})).toThrow(
    /at least 8 values/
  );
  expect(() => create({seedIds: createTransientView(graph, 'ids3', 'uint32', 3)})).toThrow(
    /seedIds length/
  );
  expect(() => create({seedCount: createTransientView(graph, 'c2', 'uint32', 2)})).toThrow(
    /exactly one row/
  );
  expect(() => create({seedMask: createTransientView(graph, 'm2', 'uint32', 5)})).toThrow(
    /seedMask must contain one row per cell/
  );
  expect(() =>
    create({
      output: {distances: createTransientView(graph, 'd2', 'float32', 5)}
    })
  ).toThrow(/output.distances/);
  const distances = createTransientView(graph, 'd3', 'float32', CELLS);
  expect(() =>
    create({
      output: {
        distances,
        allocation: createTransientView(graph, 'a3', 'uint32', CELLS),
        withinDistance: graph.createDataView(distances.buffer, {
          format: 'uint32',
          length: CELLS
        })
      }
    })
  ).toThrow(/separate buffers/);
  // A distance view over a buffer that also backs the parameters must reach the alias guard,
  // so the larger distance view is allocated first and the parameters view reuses its buffer.
  const aliasedDistances = createTransientView(graph, 'd8', 'float32', CELLS);
  expect(() =>
    create({
      parameters: graph.createDataView(aliasedDistances.buffer, {format: 'float32', length: 8}),
      output: {distances: aliasedDistances}
    })
  ).toThrow(/outputs must not share buffers with inputs/);
  const foreign = new GPUDistanceField(createPointProps(otherGraph));
  expect(() => foreign.getCommandNodes(graph)).toThrow(/target graph/);
});

it('the CPU oracle snaps seeds like the GPU and breaks ties on the smallest ID', () => {
  const settings = getGPUDistanceFieldParameterValues(
    {},
    makeGPURasterMetadata({width: 5, height: 2, rowDirection: 'north'})
  );
  const positions = Float32Array.of(0.5, 0.5, 4.5, 0.5, 0.99, 0.2, -1, 0, Number.NaN, 1);
  const cellSeeds = getCellSeeds({
    width: 5,
    height: 2,
    settings,
    positions,
    ids: Uint32Array.of(4, 3, 1, 0, 0)
  });
  expect(Array.from(cellSeeds)).toEqual([1, NONE, NONE, NONE, 3, NONE, NONE, NONE, NONE, NONE]);
  const result = computeDistanceFieldOnCPU({
    width: 5,
    height: 1,
    settings,
    positions: Float32Array.of(0.5, 0.5, 4.5, 0.5),
    ids: Uint32Array.of(7, 2)
  });
  expect(Array.from(result.allocation)).toEqual([7, 7, 2, 2, 2]);
  expect(Array.from(result.distances)).toEqual([0, 1, 2, 1, 0]);
  expect(getUlpDistance(1, 1 + 2 ** -23)).toBe(1);
  expect(getUlpDistance(Infinity, Infinity)).toBe(0);
});

it('rule 12: every jump-flood step reads its offset from a step word', () => {
  const device = createNullWebGPUDevice();
  for (const [width, height] of [
    [20, 10],
    [70, 9],
    [300, 260]
  ]) {
    const graph = new GPUCommandGraph(device);
    const nodes = new GPUDistanceField({
      ...createPointProps(graph),
      grid: makeGPURasterMetadata({width, height}),
      output: {distances: createTransientView(graph, 'distances-sized', 'float32', width * height)},
      mode: 'jump-flood'
    }).getCommandNodes(graph);
    const steps = getRasterPipelines(nodes).filter(pipeline =>
      /-jump-flood-\d+$/.test(pipeline.id)
    );
    expect(steps.length).toBeGreaterThan(4);
    expect(new Set(steps.map(pipeline => pipeline.key)).size).toBe(1);
    expect(steps[0].source).not.toMatch(/const STEP:/);
  }
  device.destroy();
});

it('bounds the largest jump schedule by fifteen halvings and two refinements', () => {
  const grid = {width: 32768, height: 1};
  // 32768 = 2^15, so the exact halving schedule is 2^14 down to 2^0.
  const halves = [16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1];
  expect(getDistanceFieldJumpFloodSteps(grid, 0)).toEqual(halves);
  expect(getDistanceFieldJumpFloodSteps(grid, 2)).toEqual([...halves, 2, 1]);
  expect(GPU_DISTANCE_FIELD_MAXIMUM_REFINEMENT_COUNT).toBe(2);
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const props = createPointProps(graph);
  expect(
    () => new GPUDistanceField({...props, jumpFloodRefinementIterationCount: 2})
  ).not.toThrow();
  expect(
    () => new GPUDistanceField({...props, jumpFloodRefinementIterationCount: 3 as never})
  ).toThrow(/jumpFloodRefinementIterationCount/);
});

it('uses the named refinement capacity admission with minimum zero', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const props = createPointProps(graph);
  for (const value of [-1, 0.5, NaN, Infinity, 3]) {
    expect(
      () => new GPUDistanceField({...props, jumpFloodRefinementIterationCount: value as never})
    ).toThrow('distance-field jumpFloodRefinementIterationCount must be an integer in [0, 2]');
  }
  for (const value of [0, 1, 2] as const)
    expect(
      () => new GPUDistanceField({...props, jumpFloodRefinementIterationCount: value})
    ).not.toThrow();
});
