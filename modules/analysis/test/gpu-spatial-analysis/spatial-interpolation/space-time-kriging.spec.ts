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
  type GPUSpaceTimeKrigingNeighborSearch,
  type GPUSpaceTimeKrigingProps
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-space-time-kriging';
import {
  KRIGING_PIVOT_RATIO_TOLERANCE,
  getGPUKrigingParameterValues,
  GPU_KRIGING_PARAMETER_LENGTH
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/kriging-parameters';
import {GPUKriging} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-kriging';
import {
  getGPUSpaceTimeKrigingParameterValues,
  GPU_SPACE_TIME_KRIGING_PARAMETER_LENGTH,
  type GPUSpaceTimeKrigingSettings
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/prediction-covariance';
import {
  createInputBuffer,
  createVectorView,
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
  getReferenceSpaceTimeCovariance,
  selectReferenceNeighbors,
  solveReferenceSpaceTimeKriging,
  type Point,
  type ReferenceScene
} from './space-time-kriging-reference';

/** Padding detects writes beyond the logical query prefix and across shared chunk boundaries. */
const OUTPUT_FLOAT_SENTINEL = -12345;
/** A distinctive uint32 padding value cannot be confused with a published status. */
const OUTPUT_STATUS_SENTINEL = 0xdeadbeef;

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
  sampleIndex?: GPUSpaceTimeKrigingProps['sampleIndex'];
  queryChunkLengths?: number[];
  outputChunkLengths?: number[];
  outputPadding?: number;
  sampleValidity?: readonly number[];
}) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const {scene} = props;
  const queryCount = scene.queries.length;
  const sampleCount = scene.positions.length;
  const outputRowCount = queryCount + (props.outputPadding ?? 0);
  const buffers = {
    parameters: createInputBuffer(
      device,
      new Float32Array(GPU_SPACE_TIME_KRIGING_PARAMETER_LENGTH)
    ),
    positions: createInputBuffer(device, Float32Array.from(scene.positions.flat())),
    times: createInputBuffer(device, Float32Array.from(scene.times)),
    observations: createInputBuffer(
      device,
      Float32Array.from(
        scene.observations.flatMap(([value, error], row) => [
          value,
          error,
          0,
          props.sampleValidity?.[row] ?? 1
        ])
      )
    ),
    queries: createInputBuffer(device, Float32Array.from(scene.queries.flat())),
    queryTimes: createInputBuffer(device, Float32Array.from(scene.queryTimes)),
    values: createOutputBuffer(device, outputRowCount * 4),
    status: createOutputBuffer(device, outputRowCount)
  };
  buffers.values.write(new Float32Array(outputRowCount * 4).fill(OUTPUT_FLOAT_SENTINEL));
  buffers.status.write(new Uint32Array(outputRowCount).fill(OUTPUT_STATUS_SENTINEL));
  const createQueryRows = <Format extends 'float32' | 'float32x2' | 'float32x4' | 'uint32'>(
    name: 'queries' | 'queryTimes' | 'values' | 'status',
    format: Format
  ) => {
    const isOutput = name === 'values' || name === 'status';
    const packed = importGraphBuffer(
      graph,
      name,
      buffers[name],
      format,
      isOutput ? outputRowCount : queryCount
    );
    const lengths = isOutput
      ? (props.outputChunkLengths ?? props.queryChunkLengths)
      : props.queryChunkLengths;
    if (!lengths) return packed;
    let rowBase = 0;
    return createVectorView(
      name,
      format,
      lengths.map(length => {
        const chunk = graph.createDataView(packed.buffer, {
          format,
          length,
          byteOffset: rowBase * packed.rowByteLength
        });
        rowBase += length;
        return chunk;
      })
    );
  };
  const contributor = (() => {
    try {
      return new GPUSpaceTimeKriging({
        temporalContext: TEMPORAL_CONTEXT,
        positions: importGraphBuffer(
          graph,
          'positions',
          buffers.positions,
          'float32x2',
          sampleCount
        ),
        times: importGraphBuffer(graph, 'times', buffers.times, 'float32', sampleCount),
        observations: importGraphBuffer(
          graph,
          'observations',
          buffers.observations,
          'float32x4',
          sampleCount
        ),
        queries: createQueryRows('queries', 'float32x2'),
        queryTimes: createQueryRows('queryTimes', 'float32'),
        mode: props.mode,
        parameters: importGraphBuffer(
          graph,
          'parameters',
          buffers.parameters,
          'float32',
          GPU_SPACE_TIME_KRIGING_PARAMETER_LENGTH
        ),
        sampleIndex: props.sampleIndex,
        neighborCount: props.neighborCount,
        neighborSearch: props.neighborSearch,
        neighborsPerOctant: props.neighborsPerOctant,
        stagedCandidateCount: props.stagedCandidateCount,
        output: {
          values: createQueryRows('values', 'float32x4'),
          status: createQueryRows('status', 'uint32')
        }
      });
    } catch (error) {
      Object.values(buffers).forEach(buffer => buffer.destroy());
      throw error;
    }
  })();
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
    writeSamples(next: ReferenceScene, validity?: readonly number[]) {
      buffers.positions.write(Float32Array.from(next.positions.flat()));
      buffers.times.write(Float32Array.from(next.times));
      buffers.observations.write(
        Float32Array.from(
          next.observations.flatMap(([value, error], row) => [
            value,
            error,
            0,
            validity?.[row] ?? 1
          ])
        )
      );
    },
    async readOutputStorage() {
      return {
        values: await readFloat32(buffers.values, outputRowCount * 4),
        status: await readUint32(buffers.status, outputRowCount)
      };
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

for (const neighborSearch of ['metric', 'octant', 'staged'] as const) {
  for (const mode of ['simple', 'ordinary'] as const) {
    it(`indexed ${neighborSearch} ${mode} predictions match a CPU solve and the scan across query chunks`, async () => {
      const scene = createScene(42, 180);
      const settings = {
        scene,
        mode,
        neighborSearch,
        neighborCount: 8,
        neighborsPerOctant: 2,
        stagedCandidateCount: 16
      };
      const indexed = await createHarness({
        ...settings,
        sampleIndex: 'grid',
        queryChunkLengths: [1, 0, 4, 7]
      });
      const reference = await createHarness({...settings, sampleIndex: 'brute-force'});
      try {
        // Both the lattice and the covariance update between frames without recompiling.
        for (const [covariance, searchAnisotropy] of [
          [MODELS[0], 0.1],
          [MODELS[2], 3]
        ] as const) {
          const parameters = {covariance, searchAnisotropy, mean: 0.4};
          const result = await indexed.run(parameters);
          const scan = await reference.run(parameters);
          expect(result.status).toEqual(scan.status);
          scene.queries.forEach((query, queryIndex) => {
            const rows = selectReferenceNeighbors(
              scene,
              query,
              scene.queryTimes[queryIndex],
              covariance,
              {
                search: neighborSearch,
                neighborCount: 8,
                neighborsPerOctant: 2,
                stagedCandidateCount: 16,
                searchAnisotropy
              }
            );
            const expected = solveReferenceSpaceTimeKriging({
              scene,
              query,
              queryTime: scene.queryTimes[queryIndex],
              covariance,
              rows,
              mode,
              mean: 0.4
            });
            expect(result.status[queryIndex]).toBe(0);
            expect(result.values[queryIndex * 4 + 3]).toBe(rows.length);
            expect(Math.abs(result.values[queryIndex * 4] - expected.prediction)).toBeLessThan(
              5e-4
            );
            expect(Math.abs(result.values[queryIndex * 4 + 1] - expected.variance)).toBeLessThan(
              5e-4
            );
            for (let component = 0; component < 4; component++)
              expect(
                Math.abs(
                  result.values[queryIndex * 4 + component] -
                    scan.values[queryIndex * 4 + component]
                )
              ).toBeLessThan(5e-4);
          });
        }
        expect(indexed.buildCount).toBe(1);
      } finally {
        indexed.destroy();
        reference.destroy();
      }
    });
  }
}

it('indexed filters exclude invalid samples and publish missing neighborhoods across chunks', async () => {
  const scene = {
    positions: [
      [NaN, 0],
      [0, 0],
      [0.5, 0],
      [1, 0],
      [2, 0]
    ] as Point[],
    times: [0, 0, 10, 0, 0],
    observations: [
      [1, 0],
      [10, 0],
      [20, 0],
      [30, 0],
      [40, -1]
    ] as const,
    queries: [
      [0, 0],
      [100, 100]
    ] as Point[],
    queryTimes: [0, 0]
  };
  const harness = await createHarness({
    scene,
    mode: 'ordinary',
    neighborCount: 8,
    sampleIndex: 'grid',
    queryChunkLengths: [1, 0, 1]
  });
  try {
    const result = await harness.run({
      covariance: MODELS[2],
      searchRadius: 0.25,
      maximumTimeLag: 0
    });
    expect(result.status).toEqual([0, 1]);
    expect(result.values[0]).toBeCloseTo(10, 5);
    expect(result.values[1]).toBeCloseTo(0, 5);
    expect(result.values[3]).toBe(1);
    expect(Number.isNaN(result.values[4])).toBe(true);
  } finally {
    harness.destroy();
  }
});

it('indexes 8M samples on default WebGPU bindings and matches closed-form one-neighbor predictions', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const sampleCount = 8 * 1024 * 1024;
  const positions = new Float32Array(sampleCount * 2);
  const times = new Float32Array(sampleCount);
  const observations = new Float32Array(sampleCount * 4);
  for (let row = 0; row < sampleCount; row++) {
    positions[row * 2] = row % 2048;
    positions[row * 2 + 1] = Math.floor(row / 2048) % 2048;
    times[row] = Math.floor(row / (2048 * 2048));
    observations[row * 4] = row;
    observations[row * 4 + 3] = 1;
  }
  const targetRows = [0, Math.floor(sampleCount / 2) + 1234, sampleCount - 1];
  const queryPositions = Float32Array.from(
    targetRows.flatMap(row => [positions[row * 2], positions[row * 2 + 1]])
  );
  const queryTimes = Float32Array.from(targetRows.map(row => times[row]));
  const queryCount = targetRows.length;
  const graph = new GPUCommandGraph(device);
  const buffers = {
    positions: createInputBuffer(device, positions),
    times: createInputBuffer(device, times),
    observations: createInputBuffer(device, observations),
    queries: createInputBuffer(device, queryPositions),
    queryTimes: createInputBuffer(device, queryTimes),
    parameters: createInputBuffer(
      device,
      getGPUSpaceTimeKrigingParameterValues({
        covariance: {
          type: 'metric',
          joint: {model: 'exponential', nugget: 0, sill: 1, range: 100},
          anisotropy: 1
        },
        mean: 0
      })
    ),
    values: createOutputBuffer(device, queryCount * 4),
    status: createOutputBuffer(device, queryCount)
  };
  const contributor = new GPUSpaceTimeKriging({
    temporalContext: TEMPORAL_CONTEXT,
    mode: 'simple',
    neighborCount: 1,
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
    parameters: importGraphBuffer(
      graph,
      'parameters',
      buffers.parameters,
      'float32',
      GPU_SPACE_TIME_KRIGING_PARAMETER_LENGTH
    ),
    output: {
      values: importGraphBuffer(graph, 'values', buffers.values, 'float32x4', queryCount),
      status: importGraphBuffer(graph, 'status', buffers.status, 'uint32', queryCount)
    }
  });
  expect(contributor.sampleIndex).toBe('grid');
  graph.add(contributor);
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const values = await readFloat32(buffers.values, queryCount * 4);
    expect(await readUint32(buffers.status, queryCount)).toEqual(targetRows.map(() => 0));
    targetRows.forEach((row, queryIndex) => {
      // Exact, zero-error hit with one neighbor: weight 1, variance 0, value is the sample ID.
      expect(values[queryIndex * 4]).toBe(row);
      expect(values[queryIndex * 4 + 1]).toBeCloseTo(0, 5);
      expect(values[queryIndex * 4 + 3]).toBe(1);
    });
  } finally {
    compiled.destroy();
    Object.values(buffers).forEach(buffer => buffer.destroy());
  }
}, 120000);

for (const neighborSearch of ['metric', 'octant', 'staged'] as const) {
  it(`preserves ${neighborSearch} row-zero underflow ties at 4097 samples`, async () => {
    const sampleCount = 4097;
    const scene = {
      positions: Array.from({length: sampleCount}, () => [0, 0] as Point),
      times: Array.from({length: sampleCount}, (_, row) => (row === 0 ? Math.fround(1e-24) : 0)),
      observations: Array.from(
        {length: sampleCount},
        (_, row) => [row === 0 ? 10 : 20, 0] as const
      ),
      queries: [[0, 0]] as Point[],
      queryTimes: [0]
    };
    const settings = {scene, mode: 'ordinary' as const, neighborSearch, neighborCount: 1};
    const indexed = await createHarness({...settings, sampleIndex: 'grid'});
    const reference = await createHarness({...settings, sampleIndex: 'brute-force'});
    try {
      const parameters = {
        covariance: {
          type: 'metric' as const,
          joint: {model: 'exponential' as const, nugget: 0, sill: 1, range: 1},
          anisotropy: 1
        }
      };
      // Explicit float32 square shows the independent expected tie; row zero must win.
      expect(Math.fround(scene.times[0] * scene.times[0])).toBe(0);
      for (const harness of [indexed, reference]) {
        const result = await harness.run(parameters);
        expect(result.status).toEqual([0]);
        expect(result.values[0]).toBeCloseTo(10, 5);
        expect(result.values[3]).toBe(1);
      }
      // With one ordinary neighbor the prediction equals its observation. Encoding source IDs
      // as values is an independent selected-row probe with no public output-layout change.
      const rowScene = {
        ...scene,
        observations: scene.observations.map((_, row) => [row, 0] as const)
      };
      for (const harness of [indexed, reference]) {
        harness.writeSamples(rowScene);
        expect((await harness.run(parameters)).values[0]).toBe(0);
      }
    } finally {
      indexed.destroy();
      reference.destroy();
    }
  });
}

it('publishes contiguous shared-buffer chunks around 32 and 256 rows without touching sentinel padding', async () => {
  const lengths = [1, 0, 31, 32, 33, 255, 256, 257, 0];
  const queryCount = lengths.reduce((total, length) => total + length, 0);
  const positions: Point[] = [
    [0, 0],
    [10, 0],
    [0, 10],
    [10, 10]
  ];
  const times = [0, 1, 2, 3];
  const observations = [
    [11, 0],
    [23, 0],
    [37, 0],
    [41, 0]
  ] as const;
  const scene = {
    positions,
    times,
    observations,
    queries: Array.from({length: queryCount}, (_, row) => positions[row % 4]),
    queryTimes: Array.from({length: queryCount}, (_, row) => times[row % 4])
  };
  const outputLengths = lengths.slice();
  outputLengths[outputLengths.length - 1] += 2;
  const harness = await createHarness({
    scene,
    mode: 'ordinary',
    neighborCount: 1,
    sampleIndex: 'grid',
    queryChunkLengths: lengths,
    outputChunkLengths: outputLengths,
    outputPadding: 2
  });
  try {
    const result = await harness.run({covariance: MODELS[2]});
    expect(result.status).toEqual(Array.from({length: queryCount}, () => 0));
    for (let row = 0; row < queryCount; row++) {
      expect(result.values[row * 4]).toBeCloseTo(observations[row % 4][0], 5);
      expect(result.values[row * 4 + 1]).toBeCloseTo(0, 5);
      expect(result.values[row * 4 + 3]).toBe(1);
    }
    const storage = await harness.readOutputStorage();
    expect(storage.values.slice(queryCount * 4)).toEqual(new Array(8).fill(OUTPUT_FLOAT_SENTINEL));
    expect(storage.status.slice(queryCount)).toEqual([
      OUTPUT_STATUS_SENTINEL,
      OUTPUT_STATUS_SENTINEL
    ]);
  } finally {
    harness.destroy();
  }
});

it('rejects an oversized intermediate output chunk before GPU execution', async () => {
  const scene = {
    positions: [[0, 0]] as Point[],
    times: [0],
    observations: [[10, 0]] as const,
    queries: [
      [0, 0],
      [0, 0]
    ] as Point[],
    queryTimes: [0, 0]
  };
  await expect(
    createHarness({
      scene,
      mode: 'ordinary',
      neighborCount: 1,
      queryChunkLengths: [1, 1],
      outputChunkLengths: [2, 1],
      outputPadding: 1
    })
  ).rejects.toThrow('non-final output chunk lengths must match queries');
});

it('rebuilds bounds and membership after positions, times, values and validity change between frames', async () => {
  const initial = {
    positions: [
      [0, 0],
      [10, 0],
      [20, 0],
      [100, 100]
    ] as Point[],
    times: [0, 0, 0, 20],
    observations: [
      [11, 0],
      [22, 0],
      [33, 0],
      [44, 0]
    ] as const,
    queries: [[0, 0]] as Point[],
    queryTimes: [0]
  };
  const harness = await createHarness({
    scene: initial,
    mode: 'ordinary',
    neighborCount: 1,
    sampleIndex: 'grid'
  });
  try {
    expect((await harness.run({covariance: MODELS[2]})).values[0]).toBeCloseTo(11, 5);
    const changed = {
      positions: [
        [1000, 1000],
        [0, 0],
        [1, 0],
        [100, 100]
      ] as Point[],
      times: [1000, 0, 100, 20],
      observations: [
        [101, 0],
        [202, 0],
        [303, 0],
        [404, 0]
      ] as const
    };
    harness.writeSamples(changed, [1, 0, 1, 1]);
    // Row 1 is now an invalid exact hit; row 2 is spatially near but outside the time window.
    // The sole eligible row is row 3, independent of either index or scan ordering.
    const result = await harness.run({
      covariance: MODELS[2],
      searchRadius: 200,
      maximumTimeLag: 30
    });
    expect(result.status).toEqual([0]);
    expect(result.values[0]).toBeCloseTo(404, 4);
    expect(result.values[3]).toBe(1);
    expect(harness.buildCount).toBe(1);
  } finally {
    harness.destroy();
  }
});
