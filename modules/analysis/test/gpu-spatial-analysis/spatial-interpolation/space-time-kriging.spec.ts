// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {describe, expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import type {GPUTemporalContext} from '../../../src/analysis-core/contracts/temporal-context';
import type {SpaceTimeVariogramModel} from '../../../src/dataframe-analysis/pair-statistics/space-time-variogram-model';
import {
  GPUSpaceTimeKriging,
  type GPUSpaceTimeKrigingNeighborSearch
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-space-time-kriging';
import {KRIGING_PIVOT_RATIO_TOLERANCE} from '../../../src/gpu-spatial-analysis/spatial-interpolation/kriging-parameters';
import {
  getGPUSpaceTimeKrigingParameterValues,
  GPU_SPACE_TIME_KRIGING_PARAMETER_LENGTH,
  type GPUSpaceTimeKrigingSettings
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/prediction-covariance';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import gstoolsFixture from '../../oracles/fixtures/space-time-metric-gstools.json';
import {makeExtentGrid} from './extent-grid';
import {solveReferenceKriging} from './matern-kriging-reference';
import {
  getGPUKrigingParameterValues,
  GPUKriging,
  GPU_KRIGING_PARAMETER_LENGTH
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/index';
import {
  getReferenceSpaceTimeCovariance,
  selectReferenceNeighbors,
  solveReferenceSpaceTimeKriging,
  type Point,
  type ReferenceScene
} from './space-time-kriging-reference';

const TEMPORAL_CONTEXT: GPUTemporalContext = {
  unit: 'second',
  epoch: 0n,
  encoding: 'float32-relative',
  axis: {kind: 'irregular'}
};

const MODELS: SpaceTimeVariogramModel[] = [
  {
    type: 'separable',
    sill: 2,
    spatial: {model: 'exponential', nugget: 0.1, sill: 0.9, range: 2.2},
    temporal: {model: 'gaussian', nugget: 0.05, sill: 0.95, range: 9}
  },
  {
    type: 'product-sum',
    k: 0.6,
    spatial: {model: 'matern-3/2', nugget: 0.05, sill: 1, range: 2.5},
    temporal: {model: 'exponential', nugget: 0.02, sill: 0.8, range: 12}
  },
  {
    type: 'metric',
    joint: {model: 'exponential', nugget: 0.04, sill: 1.5, range: 3},
    anisotropy: 0.3
  },
  {
    type: 'sum-metric',
    spatial: {model: 'spherical', nugget: 0.02, sill: 0.6, range: 3},
    temporal: {model: 'exponential', nugget: 0.01, sill: 0.3, range: 10},
    joint: {model: 'matern-5/2', nugget: 0.03, sill: 0.8, range: 2.5},
    anisotropy: 0.25
  },
  {
    type: 'gneiting',
    nugget: 0.05,
    sill: 1.5,
    spatialRange: 2.8,
    temporalRange: 11,
    alpha: 0.5,
    gamma: 0.5,
    beta: 0.6,
    tau: 1
  }
];

function createRandom(seed: number): () => number {
  let state = seed | 0;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let mixed = Math.imul(state ^ (state >>> 15), 1 | state);
    mixed = (mixed + Math.imul(mixed ^ (mixed >>> 7), 61 | mixed)) ^ mixed;
    return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
  };
}

function createScene(seed: number, sampleCount: number) {
  const random = createRandom(seed);
  const point = (scale: number): Point => [
    Math.fround((random() - 0.5) * scale),
    Math.fround((random() - 0.5) * scale)
  ];
  const positions = Array.from({length: sampleCount}, () => point(8));
  const times = positions.map(() => Math.fround(random() * 30));
  const observations = positions.map(
    ([x, y], row) => [Math.fround(Math.sin(x) + Math.cos(y * 0.7) + 0.05 * times[row]), 0] as const
  );
  const queries = Array.from({length: 12}, () => point(6));
  const queryTimes = queries.map(() => Math.fround(5 + random() * 20));
  return {positions, times, observations, queries, queryTimes};
}

async function createHarness(props: {
  scene: ReferenceScene & {queries: readonly Point[]; queryTimes: readonly number[]};
  mode: 'simple' | 'ordinary';
  neighborCount: number;
  neighborSearch?: GPUSpaceTimeKrigingNeighborSearch;
  neighborsPerOctant?: number;
  stagedCandidateCount?: number;
}) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const {scene} = props;
  const queryCount = scene.queries.length;
  const sampleCount = scene.positions.length;
  const buffers = {
    parameters: createInputBuffer(
      device,
      new Float32Array(GPU_SPACE_TIME_KRIGING_PARAMETER_LENGTH)
    ),
    positions: createInputBuffer(device, Float32Array.from(scene.positions.flat())),
    times: createInputBuffer(device, Float32Array.from(scene.times)),
    observations: createInputBuffer(
      device,
      Float32Array.from(scene.observations.flatMap(([value, error]) => [value, error, 0, 1]))
    ),
    queries: createInputBuffer(device, Float32Array.from(scene.queries.flat())),
    queryTimes: createInputBuffer(device, Float32Array.from(scene.queryTimes)),
    values: createOutputBuffer(device, queryCount * 4),
    status: createOutputBuffer(device, queryCount)
  };
  const contributor = new GPUSpaceTimeKriging({
    temporalContext: TEMPORAL_CONTEXT,
    positions: importGraphBuffer(graph, 'positions', buffers.positions, 'float32x2', sampleCount),
    times: importGraphBuffer(graph, 'times', buffers.times, 'float32', sampleCount),
    observations: importGraphBuffer(
      graph,
      'observations',
      buffers.observations,
      'float32x4',
      sampleCount
    ),
    queries: importGraphBuffer(graph, 'queries', buffers.queries, 'float32x2', queryCount),
    queryTimes: importGraphBuffer(graph, 'queryTimes', buffers.queryTimes, 'float32', queryCount),
    mode: props.mode,
    parameters: importGraphBuffer(
      graph,
      'parameters',
      buffers.parameters,
      'float32',
      GPU_SPACE_TIME_KRIGING_PARAMETER_LENGTH
    ),
    neighborCount: props.neighborCount,
    neighborSearch: props.neighborSearch,
    neighborsPerOctant: props.neighborsPerOctant,
    stagedCandidateCount: props.stagedCandidateCount,
    output: {
      values: importGraphBuffer(graph, 'values', buffers.values, 'float32x4', queryCount),
      status: importGraphBuffer(graph, 'status', buffers.status, 'uint32', queryCount)
    }
  });
  let buildCount = 0;
  const getCommandNodes = contributor.getCommandNodes.bind(contributor);
  contributor.getCommandNodes = (target => {
    buildCount++;
    return getCommandNodes(target);
  }) as typeof contributor.getCommandNodes;
  graph.add(contributor);
  const compiled = graph.compile();
  return {
    get buildCount() {
      return buildCount;
    },
    async run(settings?: GPUSpaceTimeKrigingSettings) {
      if (settings) {
        buffers.parameters.write(getGPUSpaceTimeKrigingParameterValues(settings));
      }
      submitGraph(device, compiled, undefined);
      return {
        values: await readFloat32(buffers.values, queryCount * 4),
        status: await readUint32(buffers.status, queryCount)
      };
    },
    destroy() {
      compiled.destroy();
      Object.values(buffers).forEach(buffer => buffer.destroy());
    }
  };
}

describe('space-time covariance', () => {
  for (const model of MODELS) {
    it(`${model.type} covariance matches a binary64 evaluation of its definition`, async () => {
      // Simple kriging with one neighbor at the origin: prediction = C(h, u) / C(0, 0) * value.
      const lags: (readonly [number, number])[] = [
        [0, 0],
        [0, 3],
        [1.5, 0],
        [0.4, 1],
        [1.1, 4],
        [2.4, 9],
        [3.7, 15]
      ];
      const scene = {
        positions: [[0, 0] as Point],
        times: [0],
        observations: [[1, 0] as const],
        queries: lags.map(([h]) => [h, 0] as Point),
        queryTimes: lags.map(([, u]) => u)
      };
      const harness = await createHarness({scene, mode: 'simple', neighborCount: 1});
      try {
        const {values, status} = await harness.run({covariance: model, searchAnisotropy: 1});
        lags.forEach(([h, u], index) => {
          expect(status[index]).toBe(0);
          const expected =
            getReferenceSpaceTimeCovariance(model, h, u) /
            getReferenceSpaceTimeCovariance(model, 0, 0);
          expect(Math.abs(values[index * 4] - expected), `lag ${h},${u}`).toBeLessThan(3e-6);
        });
      } finally {
        harness.destroy();
      }
    });
  }
});

describe('local space-time kriging', () => {
  for (const [index, model] of MODELS.entries()) {
    for (const mode of ['simple', 'ordinary'] as const) {
      for (const neighborSearch of ['metric', 'octant', 'staged'] as const) {
        it(`${model.type} ${mode} ${neighborSearch} matches an independent binary64 solve`, async () => {
          const scene = createScene(101 + index, 80);
          const neighborCount = 12;
          const neighborsPerOctant = 3;
          const stagedCandidateCount = 24;
          const searchAnisotropy = 0.3;
          const harness = await createHarness({
            scene,
            mode,
            neighborCount,
            neighborSearch,
            neighborsPerOctant,
            stagedCandidateCount
          });
          try {
            const {values, status} = await harness.run({
              covariance: model,
              searchAnisotropy,
              mean: 0.4
            });
            scene.queries.forEach((query, queryIndex) => {
              expect(status[queryIndex]).toBe(0);
              const rows = selectReferenceNeighbors(
                scene,
                query,
                scene.queryTimes[queryIndex],
                model,
                {
                  search: neighborSearch,
                  neighborCount,
                  searchAnisotropy,
                  neighborsPerOctant,
                  stagedCandidateCount
                }
              );
              expect(values[queryIndex * 4 + 3]).toBe(rows.length);
              const reference = solveReferenceSpaceTimeKriging({
                scene,
                query,
                queryTime: scene.queryTimes[queryIndex],
                covariance: model,
                rows,
                mode,
                mean: 0.4
              });
              expect(Math.abs(values[queryIndex * 4] - reference.prediction)).toBeLessThan(5e-4);
              expect(Math.abs(values[queryIndex * 4 + 1] - reference.variance)).toBeLessThan(5e-4);
            });
          } finally {
            harness.destroy();
          }
        });
      }
    }
  }

  it('changes the model between frames without recompiling', async () => {
    const scene = createScene(7, 60);
    const harness = await createHarness({scene, mode: 'ordinary', neighborCount: 10});
    try {
      for (const model of MODELS) {
        const {values, status} = await harness.run({covariance: model, searchAnisotropy: 0.3});
        const rows = selectReferenceNeighbors(scene, scene.queries[0], scene.queryTimes[0], model, {
          search: 'metric',
          neighborCount: 10,
          searchAnisotropy: 0.3,
          neighborsPerOctant: 2,
          stagedCandidateCount: 20
        });
        const reference = solveReferenceSpaceTimeKriging({
          scene,
          query: scene.queries[0],
          queryTime: scene.queryTimes[0],
          covariance: model,
          rows,
          mode: 'ordinary'
        });
        expect(status[0]).toBe(0);
        expect(Math.abs(values[0] - reference.prediction)).toBeLessThan(5e-4);
      }
      expect(harness.buildCount).toBe(1);
    } finally {
      harness.destroy();
    }
  });

  it('octant search reaches a side that the metric neighbors miss', async () => {
    // Six samples in the future on one side are nearest; one past sample is farther away.
    const scene = {
      positions: [
        [0.2, 0.1],
        [0.3, 0.2],
        [0.1, 0.4],
        [0.5, 0.3],
        [0.4, 0.5],
        [0.2, 0.7],
        [-3, -3]
      ] as Point[],
      times: [1, 2, 3, 1.5, 2.5, 3.5, -8],
      observations: [
        [1, 0],
        [1.2, 0],
        [0.9, 0],
        [1.1, 0],
        [1.3, 0],
        [0.8, 0],
        [-5, 0]
      ] as const,
      queries: [[0, 0]] as Point[],
      queryTimes: [0]
    };
    const covariance = MODELS[2];
    const metric = await createHarness({scene, mode: 'ordinary', neighborCount: 4});
    const octant = await createHarness({
      scene,
      mode: 'ordinary',
      neighborCount: 4,
      neighborSearch: 'octant',
      neighborsPerOctant: 2
    });
    try {
      const metricResult = await metric.run({covariance, searchAnisotropy: 0.3});
      const octantResult = await octant.run({covariance, searchAnisotropy: 0.3});
      expect(metricResult.status[0]).toBe(0);
      expect(octantResult.status[0]).toBe(0);
      // The two searches pick different sets, including the distant past sample for octant.
      const octantRows = selectReferenceNeighbors(scene, [0, 0], 0, covariance, {
        search: 'octant',
        neighborCount: 4,
        searchAnisotropy: 0.3,
        neighborsPerOctant: 2,
        stagedCandidateCount: 8
      });
      expect(octantRows).toContain(6);
      expect(Math.abs(octantResult.values[0] - metricResult.values[0])).toBeGreaterThan(1e-3);
      const reference = solveReferenceSpaceTimeKriging({
        scene,
        query: [0, 0],
        queryTime: 0,
        covariance,
        rows: octantRows,
        mode: 'ordinary'
      });
      expect(Math.abs(octantResult.values[0] - reference.prediction)).toBeLessThan(5e-4);
    } finally {
      metric.destroy();
      octant.destroy();
    }
  });

  it('staged search prefers higher covariance over a smaller metric distance', async () => {
    // Separable with a long temporal range: a candidate far in metric terms in time can still
    // have the larger covariance than one close in the search metric.
    const covariance: SpaceTimeVariogramModel = {
      type: 'separable',
      sill: 1,
      spatial: {model: 'exponential', nugget: 0, sill: 1, range: 0.5},
      temporal: {model: 'exponential', nugget: 0, sill: 1, range: 100}
    };
    const scene = {
      positions: [
        [0.6, 0],
        [0.05, 0],
        [0.1, 0.05],
        [0.08, -0.03]
      ] as Point[],
      times: [0.5, 6, 7, 5],
      observations: [
        [5, 0],
        [1, 0],
        [1.1, 0],
        [0.9, 0]
      ] as const,
      queries: [[0, 0]] as Point[],
      queryTimes: [0]
    };
    // Metric with searchAnisotropy 1 ranks the first sample (0.78) ahead of the others (5 to 7).
    const metric = await createHarness({scene, mode: 'simple', neighborCount: 2});
    const staged = await createHarness({
      scene,
      mode: 'simple',
      neighborCount: 2,
      neighborSearch: 'staged',
      stagedCandidateCount: 4
    });
    try {
      const settings = {covariance, searchAnisotropy: 1} as const;
      const metricResult = await metric.run(settings);
      const stagedResult = await staged.run(settings);
      expect(metricResult.status[0]).toBe(0);
      expect(stagedResult.status[0]).toBe(0);
      const stagedRows = selectReferenceNeighbors(scene, [0, 0], 0, covariance, {
        search: 'staged',
        neighborCount: 2,
        searchAnisotropy: 1,
        neighborsPerOctant: 1,
        stagedCandidateCount: 4
      });
      expect(stagedRows).not.toContain(0);
      expect(Math.abs(stagedResult.values[0] - metricResult.values[0])).toBeGreaterThan(1e-3);
      const reference = solveReferenceSpaceTimeKriging({
        scene,
        query: [0, 0],
        queryTime: 0,
        covariance,
        rows: stagedRows,
        mode: 'simple'
      });
      expect(Math.abs(stagedResult.values[0] - reference.prediction)).toBeLessThan(5e-4);
    } finally {
      metric.destroy();
      staged.destroy();
    }
  });

  it('reports status 1 for an unwritten buffer and status 2 at the shared pivot limit', async () => {
    const scene = {
      positions: [
        [0, 0],
        [1e-5, 0],
        [1, 0.5],
        [-0.5, 1]
      ] as Point[],
      times: [1, 1, 2, 3],
      observations: [
        [1, 0],
        [1.1, 0],
        [0.7, 0],
        [0.2, 0]
      ] as const,
      queries: [[0.3, 0.3]] as Point[],
      queryTimes: [2]
    };
    const covariance: SpaceTimeVariogramModel = {
      type: 'metric',
      joint: {model: 'matern-5/2', nugget: 0, sill: 1, range: 2},
      anisotropy: 0.5
    };
    const harness = await createHarness({scene, mode: 'ordinary', neighborCount: 4});
    try {
      expect((await harness.run()).status).toEqual([1]);
      const singular = await harness.run({covariance});
      expect(singular.status).toEqual([2]);
      expect(singular.values[2]).toBeLessThan(KRIGING_PIVOT_RATIO_TOLERANCE);
    } finally {
      harness.destroy();
    }
  });

  for (const nuCase of Object.keys(gstoolsFixture.cases)) {
    const structure = nuCase === '0.5' ? 'exponential' : 'matern-3/2';
    const covariance: SpaceTimeVariogramModel = {
      type: 'metric',
      joint: {
        model: structure,
        nugget: gstoolsFixture.nugget,
        sill: gstoolsFixture.sill,
        range: gstoolsFixture.range
      },
      anisotropy: gstoolsFixture.anisotropy
    };
    const fixtureCase = gstoolsFixture.cases[nuCase as keyof typeof gstoolsFixture.cases];
    it(`metric correlations match GSTools 1.7 (nu ${nuCase})`, () => {
      fixtureCase.lagPairs.forEach(([h, u], index) => {
        const value =
          h === 0 && u === 0
            ? 1
            : (getReferenceSpaceTimeCovariance(covariance, h, u) -
                (h === 0 && u === 0 ? gstoolsFixture.nugget : 0)) /
              gstoolsFixture.sill;
        expect(Math.abs(value - fixtureCase.correlation[index])).toBeLessThan(1e-12);
      });
    });
    for (const mode of ['ordinary', 'simple'] as const) {
      it(`metric ${mode} kriging matches GSTools 1.7 (nu ${nuCase})`, async () => {
        const scene = {
          positions: gstoolsFixture.positions as unknown as Point[],
          times: gstoolsFixture.times,
          observations: gstoolsFixture.values.map(value => [value, 0] as const),
          queries: gstoolsFixture.queries as unknown as Point[],
          queryTimes: gstoolsFixture.queryTimes
        };
        const harness = await createHarness({scene, mode, neighborCount: 32});
        try {
          const {values, status} = await harness.run({
            covariance,
            searchAnisotropy: gstoolsFixture.anisotropy,
            mean: gstoolsFixture.mean
          });
          scene.queries.forEach((_, queryIndex) => {
            expect(status[queryIndex]).toBe(0);
            expect(
              Math.abs(values[queryIndex * 4] - fixtureCase[mode].prediction[queryIndex])
            ).toBeLessThan(1e-3);
            expect(
              Math.abs(values[queryIndex * 4 + 1] - fixtureCase[mode].variance[queryIndex])
            ).toBeLessThan(1e-3);
          });
        } finally {
          harness.destroy();
        }
      });
    }
  }
});

it('GPUKriging accepts Matern variograms and matches an independent binary64 solve', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const positions: Point[] = [
    [10, 20],
    [80, 30],
    [40, 75],
    [65, 60],
    [25, 45],
    [90, 85]
  ];
  const sampleValues = [1.2, 2.4, 0.7, 1.9, 1.5, 3.1];
  const grid = makeExtentGrid(2, 2, [0, 0, 100, 100]);
  const cells: Point[] = [
    [25, 25],
    [75, 25],
    [25, 75],
    [75, 75]
  ];
  const graph = new GPUCommandGraph(device);
  const parameterBuffer = createInputBuffer(
    device,
    getGPUKrigingParameterValues(
      {
        searchRadius: Infinity,
        neighborCount: 6,
        minimumNeighborCount: 1,
        variogram: {model: 'matern-3/2', nugget: 0, sill: 2, range: 70}
      },
      grid
    )
  );
  const positionBuffer = createInputBuffer(device, Float32Array.from(positions.flat()));
  const valueBuffer = createInputBuffer(device, Float32Array.from(sampleValues));
  const outputBuffer = createOutputBuffer(device, 4);
  const varianceBuffer = createOutputBuffer(device, 4);
  graph.add(
    new GPUKriging({
      positions: importGraphBuffer(graph, 'positions', positionBuffer, 'float32x2', 6),
      values: importGraphBuffer(graph, 'values', valueBuffer, 'float32', 6),
      parameters: importGraphBuffer(
        graph,
        'parameters',
        parameterBuffer,
        'float32',
        GPU_KRIGING_PARAMETER_LENGTH
      ),
      grid,
      indexGridSize: [4, 4],
      indexBounds: [0, 0, 100, 100],
      maximumNeighborCount: 6,
      output: {
        values: importGraphBuffer(graph, 'out', outputBuffer, 'float32', 4),
        variance: importGraphBuffer(graph, 'variance', varianceBuffer, 'float32', 4)
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const values = await readFloat32(outputBuffer, 4);
  const variance = await readFloat32(varianceBuffer, 4);
  const covariance = {model: 'matern' as const, smoothness: 1.5 as const, range: 70, sill: 2};
  cells.forEach((query, index) => {
    const reference = solveReferenceKriging({
      mode: 'ordinary',
      positions,
      observations: sampleValues.map(value => [value, 0] as const),
      query,
      covariance,
      neighborCount: 6
    });
    expect(Math.abs(values[index] - reference.prediction), `cell ${index}`).toBeLessThan(1e-3);
    expect(Math.abs(variance[index] - reference.variance), `cell ${index}`).toBeLessThan(1e-3);
  });
  compiled.destroy();
});
