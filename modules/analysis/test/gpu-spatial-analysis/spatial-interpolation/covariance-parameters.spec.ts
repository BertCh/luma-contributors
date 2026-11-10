// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {describe, expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPULocalKriging,
  type GPUPredictionCovariance
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-local-kriging';
import {KRIGING_PIVOT_RATIO_TOLERANCE} from '../../../src/gpu-spatial-analysis/spatial-interpolation/kriging-parameters';
import {
  getGPULocalKrigingParameterLength,
  getGPULocalKrigingParameterValues,
  MATERN_3_2_RANGE_FACTOR,
  MATERN_5_2_RANGE_FACTOR,
  type GPULocalKrigingSettings
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/prediction-covariance';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import gstoolsFixture from '../../oracles/fixtures/matern-kriging-gstools.json';
import {getReferenceCorrelation, solveReferenceKriging} from './matern-kriging-reference';

type Point = readonly [number, number];
type ObservationRow = readonly [number, number];

/** One local-kriging graph driven by a per-frame parameter buffer. */
async function createParameterHarness(props: {
  positions: readonly Point[];
  observations: readonly ObservationRow[];
  queries: readonly Point[];
  mode: 'simple' | 'ordinary' | 'co-kriging';
  neighborCount: number;
  means?: readonly number[];
  variableCount?: number;
  covarianceStructureCount?: number;
  /** Observation rows `[value, error, variable, validity]`. Replaces the default rows. */
  observationRows?: Float32Array;
  /** Compile-time covariance. Supplies `covariance` instead of `parameters`. */
  covariance?: GPUPredictionCovariance;
  coregionalization?: readonly [number, number, number, number];
}) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const queryCount = props.queries.length;
  const parameterLength = getGPULocalKrigingParameterLength(
    props.covarianceStructureCount ?? 1,
    props.variableCount ?? 1
  );
  const parameterBuffer = createInputBuffer(device, new Float32Array(parameterLength));
  const positionBuffer = createInputBuffer(device, Float32Array.from(props.positions.flat()));
  const observationBuffer = createInputBuffer(
    device,
    props.observationRows ??
      Float32Array.from(props.observations.flatMap(([value, error]) => [value, error, 0, 1]))
  );
  const queryBuffer = createInputBuffer(device, Float32Array.from(props.queries.flat()));
  const valueBuffer = createOutputBuffer(device, queryCount * 4);
  const statusBuffer = createOutputBuffer(device, queryCount);
  const weightBuffer = createOutputBuffer(device, queryCount * props.neighborCount);
  const contributor = new GPULocalKriging({
    positions: importGraphBuffer(
      graph,
      'positions',
      positionBuffer,
      'float32x2',
      props.positions.length
    ),
    observations: importGraphBuffer(
      graph,
      'observations',
      observationBuffer,
      'float32x4',
      props.positions.length
    ),
    supports: importGraphBuffer(graph, 'queries', queryBuffer, 'float32x2', queryCount),
    mode: props.mode,
    neighborCount: props.neighborCount,
    means: props.means,
    variableCount: props.variableCount,
    covarianceStructureCount: props.covarianceStructureCount,
    covariance: props.covariance,
    coregionalization: props.coregionalization,
    parameters: props.covariance
      ? undefined
      : importGraphBuffer(graph, 'parameters', parameterBuffer, 'float32', parameterLength),
    output: {
      values: importGraphBuffer(graph, 'values', valueBuffer, 'float32x4', queryCount),
      status: importGraphBuffer(graph, 'status', statusBuffer, 'uint32', queryCount),
      weights: importGraphBuffer(
        graph,
        'weights',
        weightBuffer,
        'float32',
        queryCount * props.neighborCount
      )
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
    async run(settings?: GPULocalKrigingSettings) {
      if (settings) {
        parameterBuffer.write(getGPULocalKrigingParameterValues(settings));
      }
      submitGraph(device, compiled, undefined);
      return {
        values: await readFloat32(valueBuffer, queryCount * 4),
        status: await readUint32(statusBuffer, queryCount),
        weights: await readFloat32(weightBuffer, queryCount * props.neighborCount)
      };
    },
    destroy() {
      compiled.destroy();
      for (const buffer of [
        parameterBuffer,
        positionBuffer,
        observationBuffer,
        queryBuffer,
        valueBuffer,
        statusBuffer,
        weightBuffer
      ]) {
        buffer.destroy();
      }
    }
  };
}

function createRandom(seed: number): () => number {
  let state = seed | 0;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let mixed = Math.imul(state ^ (state >>> 15), 1 | state);
    mixed = (mixed + Math.imul(mixed ^ (mixed >>> 7), 61 | mixed)) ^ mixed;
    return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
  };
}

function createScene(seed: number, sampleCount: number, queryCount: number, extent: number) {
  const random = createRandom(seed);
  const point = (scale: number): Point => [
    Math.fround((random() - 0.5) * scale),
    Math.fround((random() - 0.5) * scale)
  ];
  const positions = Array.from({length: sampleCount}, () => point(extent));
  const observations = positions.map(
    ([x, y]) => [Math.fround(Math.sin(1.7 * x) + Math.cos(y)), 0] as ObservationRow
  );
  const queries = Array.from({length: queryCount}, () => point(extent * 0.8));
  return {positions, observations, queries};
}

/** Reference and GPU values must match to `tolerance`, status 0 required. */
function expectAgreesWithReference(
  result: {values: number[]; status: number[]},
  scene: {
    positions: readonly Point[];
    observations: readonly ObservationRow[];
    queries: readonly Point[];
  },
  covariance: GPUPredictionCovariance,
  mode: 'simple' | 'ordinary',
  neighborCount: number,
  tolerance: {prediction: number; variance: number},
  mean?: number
) {
  scene.queries.forEach((query, queryIndex) => {
    expect(result.status[queryIndex]).toBe(0);
    const reference = solveReferenceKriging({
      mode,
      positions: scene.positions,
      observations: scene.observations,
      query,
      covariance,
      neighborCount,
      mean
    });
    expect(Math.abs(result.values[queryIndex * 4] - reference.prediction)).toBeLessThan(
      tolerance.prediction
    );
    expect(Math.abs(result.values[queryIndex * 4 + 1] - reference.variance)).toBeLessThan(
      tolerance.variance
    );
  });
}

describe('per-frame covariance parameters', () => {
  it('changes nugget, sill, range and model between frames without recompiling', async () => {
    const scene = createScene(1, 48, 16, 4);
    const harness = await createParameterHarness({...scene, mode: 'ordinary', neighborCount: 16});
    try {
      const frames: GPUPredictionCovariance[] = [
        {model: 'spherical', range: 2, sill: 1, nugget: 0.1},
        {model: 'exponential', range: 0.7, sill: 3, nugget: 0},
        {model: 'gaussian', range: 3, sill: 0.4, nugget: 0.01},
        {model: 'matern', smoothness: 1.5, range: 1.5, sill: 1, nugget: 0.02},
        {model: 'matern', smoothness: 2.5, range: 1.5, sill: 2, nugget: 0.02},
        {model: 'matern', smoothness: 0.5, range: 1.5, sill: 1, nugget: 0}
      ];
      for (const covariance of frames) {
        expectAgreesWithReference(
          await harness.run({covariance}),
          scene,
          covariance,
          'ordinary',
          16,
          {prediction: 2e-4, variance: 2e-4}
        );
      }
      expect(harness.buildCount).toBe(1);
    } finally {
      harness.destroy();
    }
  });

  it('produces the same values as compile-time covariance', async () => {
    const scene = createScene(2, 40, 12, 4);
    const covariance: GPUPredictionCovariance = {
      model: 'gaussian',
      range: 2,
      sill: 2,
      nugget: 0.05
    };
    const perFrame = await createParameterHarness({...scene, mode: 'ordinary', neighborCount: 16});
    const compileTime = await createParameterHarness({
      ...scene,
      mode: 'ordinary',
      neighborCount: 16,
      covariance
    });
    try {
      const parametric = await perFrame.run({covariance});
      const literal = await compileTime.run();
      expect(parametric.status).toEqual(literal.status);
      parametric.values.forEach((value, index) => {
        expect(Math.abs(value - literal.values[index])).toBeLessThan(2e-6);
      });
    } finally {
      perFrame.destroy();
      compileTime.destroy();
    }
  });

  it('reports status 1 for every query when the parameter buffer was never written', async () => {
    const scene = createScene(3, 12, 4, 4);
    const harness = await createParameterHarness({...scene, mode: 'ordinary', neighborCount: 8});
    try {
      const result = await harness.run();
      expect(result.status).toEqual([1, 1, 1, 1]);
    } finally {
      harness.destroy();
    }
  });

  it('supports per-frame co-kriging structures and loadings', async () => {
    const random = createRandom(4);
    const positions = Array.from(
      {length: 24},
      () => [Math.fround(random() * 4 - 2), Math.fround(random() * 4 - 2)] as Point
    );
    const observationRows = Float32Array.from(
      positions.flatMap(([x, y], rowIndex) => [Math.sin(x) + y * 0.2, 0, rowIndex % 2, 1])
    );
    const queries: Point[] = [
      [0.3, 0.4],
      [-1.2, 0.9]
    ];
    const observations = positions.map(() => [0, 0] as ObservationRow);
    const coregionalization = [1, 0, 0.4, 0.8] as const;
    const covariance: GPUPredictionCovariance = {
      model: 'exponential',
      range: 1.5,
      sill: 1,
      nugget: 0.05
    };
    const perFrame = await createParameterHarness({
      positions,
      observations,
      queries,
      mode: 'co-kriging',
      neighborCount: 12,
      variableCount: 2,
      means: [0.5, -0.5],
      observationRows
    });
    const compileTime = await createParameterHarness({
      positions,
      observations,
      queries,
      mode: 'co-kriging',
      neighborCount: 12,
      variableCount: 2,
      means: [0.5, -0.5],
      observationRows,
      covariance,
      coregionalization
    });
    try {
      const parametric = await perFrame.run({covariance, coregionalization});
      const literal = await compileTime.run();
      expect(parametric.status).toEqual([0, 0]);
      parametric.values.forEach((value, index) => {
        expect(Math.abs(value - literal.values[index])).toBeLessThan(2e-6);
      });
      expect(perFrame.buildCount).toBe(1);
    } finally {
      perFrame.destroy();
      compileTime.destroy();
    }
  });
});

describe('Matern covariance', () => {
  const smoothnessCases = [
    {smoothness: 0.5 as const, factor: 3},
    {smoothness: 1.5 as const, factor: MATERN_3_2_RANGE_FACTOR},
    {smoothness: 2.5 as const, factor: MATERN_5_2_RANGE_FACTOR}
  ];

  it('Matern 1/2 equals the exponential model bit for bit', async () => {
    const scene = createScene(5, 32, 8, 4);
    const harness = await createParameterHarness({...scene, mode: 'ordinary', neighborCount: 12});
    try {
      const exponential = await harness.run({
        covariance: {model: 'exponential', range: 1.4, sill: 1.2, nugget: 0.03}
      });
      const matern = await harness.run({
        covariance: {model: 'matern', smoothness: 0.5, range: 1.4, sill: 1.2, nugget: 0.03}
      });
      expect(matern.values).toEqual(exponential.values);
    } finally {
      harness.destroy();
    }
  });

  for (const {smoothness, factor} of smoothnessCases) {
    it(`Matern ${smoothness} correlations match the double-precision closed form, with effective range`, async () => {
      // Simple kriging with one neighbor at the origin: the weight is the correlation itself.
      const range = 2;
      const distances = Array.from({length: 96}, (_, sample) => {
        const argumentValue = sample < 64 ? 2 ** (-14 + sample / 8) : 0.1 * 1.1 ** (sample - 64);
        return (argumentValue / factor) * range;
      }).filter(separation => separation > 0);
      const queries: Point[] = distances.map(separation => [Math.fround(separation), 0] as Point);
      const covariance: GPUPredictionCovariance = {model: 'matern', smoothness, range, sill: 1};
      const harness = await createParameterHarness({
        positions: [[0, 0]],
        observations: [[1, 0]],
        queries,
        mode: 'simple',
        neighborCount: 1
      });
      try {
        const {weights, status} = await harness.run({covariance});
        status.forEach(value => expect(value).toBe(0));
        const tolerance = smoothness === 0.5 ? 1.2e-7 : 1.8e-7;
        weights.forEach((weight, index) => {
          const separation = queries[index][0];
          const expected = getReferenceCorrelation(covariance, separation);
          expect(Math.abs(weight - expected), `separation ${separation}`).toBeLessThanOrEqual(
            Math.max(tolerance, tolerance * expected)
          );
          // Near zero, float32 rounding must not raise the correlation above 1.
          expect(weight).toBeLessThanOrEqual(1);
          if (index > 0) {
            // Separations increase, so the correlation must not increase except by rounding.
            expect(weight).toBeLessThanOrEqual(weights[index - 1] + 1.2e-7);
          }
        });
        if (smoothness !== 0.5) {
          // The Matern 3/2 and 5/2 series keeps a deterministic staircase near zero distance.
          expect(
            weights
              .slice(0, 16)
              .every((weight, index, list) => index === 0 || weight <= list[index - 1])
          ).toBe(true);
        }
        const atRange = getReferenceCorrelation(covariance, range);
        expect(atRange).toBeCloseTo(Math.exp(-3), 6);
      } finally {
        harness.destroy();
      }
    });

    for (const mode of ['simple', 'ordinary'] as const) {
      for (const nugget of [0, 0.05]) {
        it(`Matern ${smoothness} ${mode} kriging matches an independent double-precision solve (nugget ${nugget})`, async () => {
          const scene = createScene(11 + nugget * 100, 40, 12, 3);
          const covariance: GPUPredictionCovariance = {
            model: 'matern',
            smoothness,
            range: 1.5,
            sill: 1.5,
            nugget
          };
          const harness = await createParameterHarness({
            ...scene,
            mode,
            neighborCount: 16,
            means: mode === 'simple' ? [0.25] : undefined
          });
          try {
            expectAgreesWithReference(
              await harness.run({covariance}),
              scene,
              covariance,
              mode,
              16,
              {prediction: 1e-3, variance: 3e-4},
              0.25
            );
          } finally {
            harness.destroy();
          }
        });
      }
    }
  }

  for (const nuCase of Object.keys(gstoolsFixture.cases)) {
    const smoothness = Number(nuCase) as GPUPredictionCovariance['smoothness'] & number;
    const covariance: GPUPredictionCovariance = {
      model: 'matern',
      smoothness,
      range: gstoolsFixture.range,
      sill: gstoolsFixture.sill,
      nugget: gstoolsFixture.nugget
    };
    const fixtureCase = gstoolsFixture.cases[nuCase as keyof typeof gstoolsFixture.cases];
    const positions = gstoolsFixture.positions as unknown as Point[];
    const queries = gstoolsFixture.queries as unknown as Point[];
    const observations = gstoolsFixture.values.map(value => [value, 0] as ObservationRow);

    it(`Matern ${nuCase} correlations match GSTools 1.7 with the documented length-scale mapping`, async () => {
      const distances = gstoolsFixture.distances;
      const harness = await createParameterHarness({
        positions: [[0, 0]],
        observations: [[1, 0]],
        queries: distances.map(separation => [separation, 0] as Point),
        mode: 'simple',
        neighborCount: 1
      });
      try {
        const {weights} = await harness.run({
          covariance: {model: 'matern', smoothness, range: gstoolsFixture.range, sill: 1}
        });
        weights.forEach((weight, index) => {
          expect(Math.abs(weight - fixtureCase.correlation[index])).toBeLessThan(2e-7);
        });
      } finally {
        harness.destroy();
      }
    });

    for (const mode of ['ordinary', 'simple'] as const) {
      it(`Matern ${nuCase} ${mode} predictions and variances match GSTools 1.7 kriging`, async () => {
        const harness = await createParameterHarness({
          positions,
          observations,
          queries,
          mode,
          neighborCount: 32,
          means: mode === 'simple' ? [gstoolsFixture.mean] : undefined
        });
        try {
          const result = await harness.run({covariance});
          queries.forEach((_, queryIndex) => {
            expect(result.status[queryIndex]).toBe(0);
            expect(
              Math.abs(result.values[queryIndex * 4] - fixtureCase[mode].prediction[queryIndex])
            ).toBeLessThan(5e-4);
            expect(
              Math.abs(result.values[queryIndex * 4 + 1] - fixtureCase[mode].variance[queryIndex])
            ).toBeLessThan(5e-4);
            // The double-precision reference agrees with GSTools, so it is not fitted to the GPU.
            const reference = solveReferenceKriging({
              mode,
              positions,
              observations,
              query: queries[queryIndex],
              covariance,
              neighborCount: 32,
              mean: gstoolsFixture.mean
            });
            expect(
              Math.abs(reference.prediction - fixtureCase[mode].prediction[queryIndex])
            ).toBeLessThan(1e-9);
            expect(
              Math.abs(reference.variance - fixtureCase[mode].variance[queryIndex])
            ).toBeLessThan(1e-9);
          });
        } finally {
          harness.destroy();
        }
      });
    }
  }
});

describe('conditioning (one rule: KRIGING_PIVOT_RATIO_TOLERANCE)', () => {
  const models: GPUPredictionCovariance[] = [
    {model: 'matern', smoothness: 1.5, range: 0.6, sill: 1},
    {model: 'matern', smoothness: 2.5, range: 0.6, sill: 1},
    {model: 'gaussian', range: 0.6, sill: 1}
  ];
  it('accepts only systems above the shared pivot-ratio limit, and they match double precision', async () => {
    const scene = createScene(21, 64, 32, 1);
    const harness = await createParameterHarness({...scene, mode: 'ordinary', neighborCount: 32});
    try {
      let sawRejection = false;
      for (const covariance of models) {
        for (const nugget of [0, 1e-5, 1e-3]) {
          const frame = {...covariance, nugget};
          const {values, status} = await harness.run({covariance: frame});
          scene.queries.forEach((query, queryIndex) => {
            const pivotRatio = values[queryIndex * 4 + 2];
            if (status[queryIndex] === 0) {
              expect(pivotRatio).toBeGreaterThanOrEqual(KRIGING_PIVOT_RATIO_TOLERANCE);
              const reference = solveReferenceKriging({
                mode: 'ordinary',
                positions: scene.positions,
                observations: scene.observations,
                query,
                covariance: frame,
                neighborCount: 32
              });
              expect(Math.abs(values[queryIndex * 4] - reference.prediction)).toBeLessThan(3e-3);
              expect(Math.abs(values[queryIndex * 4 + 1] - reference.variance)).toBeLessThan(1e-3);
            } else {
              expect(status[queryIndex]).toBe(2);
              expect(pivotRatio).toBeLessThan(KRIGING_PIVOT_RATIO_TOLERANCE);
              sawRejection = true;
            }
          });
        }
      }
      expect(sawRejection).toBe(true);
    } finally {
      harness.destroy();
    }
  });

  for (const smoothness of [1.5, 2.5] as const) {
    it(`Matern ${smoothness} reports status 2 for a near-duplicate pair with no nugget, and status 0 with a nugget`, async () => {
      const positions: Point[] = [
        [0, 0],
        [1e-5, 0],
        [1, 0.5],
        [-0.5, 1],
        [0.8, -1]
      ];
      const observations = positions.map(([x, y]) => [x + y, 0] as ObservationRow);
      const queries: Point[] = [[0.3, 0.3]];
      const covariance: GPUPredictionCovariance = {model: 'matern', smoothness, range: 2, sill: 1};
      const harness = await createParameterHarness({
        positions,
        observations,
        queries,
        mode: 'ordinary',
        neighborCount: 5
      });
      try {
        const singular = await harness.run({covariance: {...covariance, nugget: 0}});
        expect(singular.status).toEqual([2]);
        expect(singular.values[2]).toBeLessThan(KRIGING_PIVOT_RATIO_TOLERANCE);
        const regular = await harness.run({covariance: {...covariance, nugget: 1e-3}});
        expect(regular.status).toEqual([0]);
      } finally {
        harness.destroy();
      }
    });
  }
});
