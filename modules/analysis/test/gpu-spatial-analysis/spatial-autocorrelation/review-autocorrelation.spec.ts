// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPULocalSpatialHeteroscedasticity,
  GPUSpatialPearson,
  GPUMultivariateLocalGeary,
  GPUMultivariateLocalGearyPermutationTest,
  GPULocalMoran,
  GPUHotSpotAnalysis,
  GPUGammaStatistic,
  getGPUSpatialAutocorrelationParameterValues
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';
import {GPUGlobalSpatialStatistics} from '../../../src/gpu-spatial-analysis/global-spatial-statistics/index';
import {GPUQuantileLISA} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {getGPUPermutationParameterValues} from '../../../src/gpu-spatial-analysis/permutation-inference/index';

function compare(actual: readonly number[], expected: readonly number[], tolerance = 2e-5) {
  expect(actual.length).toBe(expected.length);
  for (const [index, value] of expected.entries()) {
    if (Number.isNaN(value)) expect(actual[index]).toBeNaN();
    else {
      expect(Number.isFinite(actual[index])).toBe(true);
      expect(Math.abs(actual[index] - value)).toBeLessThanOrEqual(tolerance);
    }
  }
}

it('review2 R5a constant labels give squared Gamma zero rather than a defined Geary mapping', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device),
    statistic = rig.output('float32', 1);
  try {
    rig.run(
      new GPUGammaStatistic({
        operation: 'squared-difference',
        values: rig.input(new Float32Array(8).fill(2), 'float32', 8),
        weights: rig.uploadWeights({
          offsets: Array.from({length: 9}, (_, row) => row * 7),
          neighbors: Array.from({length: 8}, (_, row) =>
            Array.from({length: 8}, (_, other) => other).filter(other => other !== row)
          ).flat(),
          weights: new Array(56).fill(1),
          distances: []
        }),
        output: {statistic: statistic.view}
      })
    );
    // Every squared difference is exactly 0, independently of permutation order.
    compare(await readFloat32(statistic.buffer, 1), [0], 0);
  } finally {
    rig.destroy();
  }
});

it('review2 R8 effective symmetric pair deviations distinguish scaled and near-invariant Geary', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const mode of ['reference', 'tiled'] as const)
    for (const fixture of [
      {perturbed: false, chunked: false, scale: 1},
      {perturbed: true, chunked: false, scale: 1},
      {perturbed: false, chunked: true, scale: 1},
      {perturbed: true, chunked: true, scale: 1},
      {perturbed: true, chunked: false, scale: 2 ** 70},
      {perturbed: true, chunked: false, scale: 2 ** -70}
    ]) {
      const {perturbed, chunked} = fixture;
      const rows = perturbed ? 9 : 8,
        rig = new WeightsRig(device),
        summary = rig.output('float32', 52),
        weights = new Array(rows * (rows - 1)).fill(perturbed ? 1 : Math.fround(0.1));
      if (perturbed) weights[0] = 1 + 2 ** -23;
      const uploadRows = (start: number, count: number) =>
        rig.uploadWeights({
          offsets: Array.from({length: count + 1}, (_, row) => row * (rows - 1)),
          neighbors: Array.from({length: count}, (_, local) =>
            Array.from({length: rows}, (_, other) => other).filter(other => other !== local + start)
          ).flat(),
          weights: weights.slice(start * (rows - 1), (start + count) * (rows - 1)),
          distances: []
        });
      try {
        rig.run(
          new GPUGlobalSpatialStatistics({
            mode,
            statistics: ['geary'],
            values: rig.input(
              Float32Array.from({length: rows}, (_, row) => row * fixture.scale),
              'float32',
              rows
            ),
            weights: chunked
              ? {
                  rowCount: rows,
                  chunks: [
                    {rowStart: 0, rowCount: 3, weights: uploadRows(0, 3)},
                    {rowStart: 3, rowCount: rows - 3, weights: uploadRows(3, rows - 3)}
                  ]
                }
              : uploadRows(0, rows),
            output: {summary: summary.view}
          })
        );
        const result = await readFloat32(summary.buffer, 52);
        if (!perturbed) {
          expect(result[18]).toBe(0);
          expect(result[21]).toBe(0);
          for (const field of [19, 20, 22, 23]) expect(result[field]).toBeNaN();
        } else {
          expect(Number.isFinite(result[21])).toBe(true);
          expect(result[21]).toBeGreaterThan(0);
          expect(Math.abs(result[21] / 3.0702463789201337e-18 - 1)).toBeLessThan(2e-4);
          compare([result[22]], [-0.8819171037], 2e-4);
          compare([result[23]], [0.37782163710006383], 2e-4);
          expect(Number.isFinite(result[18])).toBe(true);
          expect(Math.abs(result[18] / 3.8378079736501665e-18 - 1)).toBeLessThan(2e-4);
          compare([result[19], result[20]], [-0.7888106377466154, 0.4302226911321914], 2e-4);
        }
      } finally {
        rig.destroy();
      }
    }
});

it('review2 R4 complete equal-weight conditional lags have no z or p in Moran and Gi', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const model of ['moran', 'gi']) {
    const rig = new WeightsRig(device),
      z = rig.output('float32', 6),
      p = rig.output('float32', 6),
      localI = rig.output('float32', 6);
    const props = {
      values: rig.input(Float32Array.of(0, 1, 2, 3, 4, 5), 'float32', 6),
      parameters: rig.input(getGPUSpatialAutocorrelationParameterValues(), 'float32', 8),
      weights: {
        ...rig.uploadWeights({
          offsets: Array.from({length: 7}, (_, row) => row * 5),
          neighbors: Array.from({length: 6}, (_, row) =>
            Array.from({length: 6}, (_, other) => other).filter(other => other !== row)
          ).flat(),
          weights: new Array(30).fill(0.2),
          distances: []
        }),
        information: {rowCount: 6, transform: 'row' as const, symmetric: true}
      },
      output: {zScores: z.view, pValues: p.view}
    };
    try {
      rig.run(
        model === 'moran'
          ? new GPULocalMoran({...props, output: {...props.output, localI: localI.view}})
          : new GPUHotSpotAnalysis({...props, selfWeight: 0})
      );
      compare(await readFloat32(z.buffer, 6), new Array(6).fill(NaN));
      compare(await readFloat32(p.buffer, 6), new Array(6).fill(NaN));
      if (model === 'moran') compare((await readFloat32(localI.buffer, 6)).slice(0, 1), [-5 / 14]);
    } finally {
      rig.destroy();
    }
  }
});

it('review2 R3a original close donors retain observed distance and conditional tail in both strategies', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Exact rational population variance and independent integer Philox oracle: upper=9 of 19.
  const expected = 2.0301221281e-15;
  for (const strategy of ['exact', 'tiled'] as const) {
    const rig = new WeightsRig(device),
      statistic = rig.output('float32', 8),
      pValues = rig.output('float32', 8),
      classes = rig.output('uint32', 8);
    try {
      rig.run(
        new GPUMultivariateLocalGearyPermutationTest({
          strategy,
          variableCount: 1,
          permutationCount: 19,
          seed: [0, 0],
          tail: 'esda-min-tail',
          values: rig.input(
            Float32Array.of(1, 0, 2 ** -26, 0, 2 ** -26, 0, 2 ** -26, 0),
            'float32',
            8
          ),
          weights: rig.uploadWeights({
            offsets: [0, 1, 2, 3, 4, 5, 6, 7, 8],
            neighbors: [1, 2, 3, 4, 5, 6, 7, 0],
            weights: new Array(8).fill(1),
            distances: []
          }),
          output: {
            statistic: statistic.view,
            pseudoPValues: pValues.view,
            classes: classes.view
          }
        })
      );
      const actual = (await readFloat32(statistic.buffer, 8))[1];
      expect(Number.isFinite(actual)).toBe(true);
      expect(actual).toBeGreaterThan(0);
      expect(Math.abs(actual - expected) / expected).toBeLessThan(2e-5);
      compare((await readFloat32(pValues.buffer, 8)).slice(1, 2), [0.5]);
      expect((await readUint32(classes.buffer, 8))[1]).toBe(0);
    } finally {
      rig.destroy();
    }
  }
});

it('review2 R1 four balanced neighbors retain undefined zero-field LOSH in both layouts and powers', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Stored f32 .1 occurs with both signs: 1+w-1-w is exactly zero over the reals.
  for (const chunked of [false, true])
    for (const residualPower of [1, 2] as const) {
      const rig = new WeightsRig(device);
      const layout = {
        offsets: [0, 4, 5, 6, 7, 8, 9, 10, 11],
        neighbors: [1, 2, 3, 4, 2, 1, 4, 3, 0, 5, 6],
        weights: [1, 0.1, 1, 0.1, 1, 1, 1, 1, 1, 1, 1],
        distances: []
      };
      const weights = chunked
        ? {
            rowCount: 8,
            chunks: [
              {
                rowStart: 0,
                rowCount: 1,
                weights: rig.uploadWeights({
                  ...layout,
                  offsets: [0, 4],
                  neighbors: layout.neighbors.slice(0, 4),
                  weights: layout.weights.slice(0, 4)
                })
              },
              {
                rowStart: 1,
                rowCount: 7,
                weights: rig.uploadWeights({
                  ...layout,
                  offsets: [0, 1, 2, 3, 4, 5, 6, 7],
                  neighbors: layout.neighbors.slice(4),
                  weights: layout.weights.slice(4)
                })
              }
            ]
          }
        : rig.uploadWeights(layout);
      const statistic = rig.output('float32', 8),
        normalized = rig.output('float32', 8),
        status = rig.output('uint32', 8),
        count = rig.output('uint32', 1);
      try {
        rig.run(
          new GPULocalSpatialHeteroscedasticity({
            weights,
            residualPower,
            values: rig.input(Float32Array.of(2, 3, 3, 1, 1, 2, 2, 2), 'float32', 8),
            output: {
              statistic: statistic.view,
              status: status.view,
              normalizedResiduals: normalized.view,
              residualCount: count.view
            }
          })
        );
        compare(await readFloat32(statistic.buffer, 8), new Array(8).fill(NaN));
        compare(await readFloat32(normalized.buffer, 8), new Array(8).fill(NaN));
        expect(await readUint32(count.buffer, 1)).toEqual([8]);
        expect(await readUint32(status.buffer, 8)).toEqual(new Array(8).fill(2));
      } finally {
        rig.destroy();
      }
    }
});

it('R1 harmonic zero residual field is undefined in whole/chunked LOSH for both powers', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Every exact weighted neighbor mean equals its focus (including row 0: (2+4)/2=3).
  for (const chunked of [false, true])
    for (const residualPower of [1, 2] as const) {
      const rig = new WeightsRig(device);
      const layout = {
        offsets: [0, 2, 3, 4, 5, 6, 7, 8, 9],
        neighbors: [1, 2, 3, 4, 1, 2, 0, 5, 6],
        weights: [0.9, 0.9, 1, 1, 1, 1, 1, 1, 1],
        distances: []
      };
      const weights = chunked
        ? {
            rowCount: 8,
            chunks: [
              {
                rowStart: 0,
                rowCount: 1,
                weights: rig.uploadWeights({
                  ...layout,
                  offsets: [0, 2],
                  neighbors: [1, 2],
                  weights: [0.9, 0.9]
                })
              },
              {
                rowStart: 1,
                rowCount: 7,
                weights: rig.uploadWeights({
                  ...layout,
                  offsets: [0, 1, 2, 3, 4, 5, 6, 7],
                  neighbors: layout.neighbors.slice(2),
                  weights: layout.weights.slice(2)
                })
              }
            ]
          }
        : rig.uploadWeights(layout);
      const statistic = rig.output('float32', 8),
        normalized = rig.output('float32', 8),
        status = rig.output('uint32', 8),
        count = rig.output('uint32', 1);
      try {
        rig.run(
          new GPULocalSpatialHeteroscedasticity({
            weights,
            values: rig.input(Float32Array.of(3, 2, 4, 2, 4, 3, 3, 3), 'float32', 8),
            residualPower,
            output: {
              statistic: statistic.view,
              status: status.view,
              normalizedResiduals: normalized.view,
              residualCount: count.view
            }
          })
        );
        compare(await readFloat32(statistic.buffer, 8), new Array(8).fill(NaN));
        compare(await readFloat32(normalized.buffer, 8), new Array(8).fill(NaN));
        expect(await readUint32(status.buffer, 8)).toEqual(new Array(8).fill(2));
        expect(await readUint32(count.buffer, 1)).toEqual([8]);
      } finally {
        rig.destroy();
      }
    }
});

it('R9 and review2 R9 integer type-five ranks retain literal boundary members', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const fixture of [
    {rows: 25, quantiles: 10, selected: 4, bounds: [7, 9.5], members: [7, 8, 9]},
    {rows: 21, quantiles: 14, selected: 10, bounds: [13, 14.5], members: [13, 14]}
  ]) {
    const rows = fixture.rows,
      rig = new WeightsRig(device),
      membership = rig.output('uint32', rows),
      boundaries = rig.output('float32', 2),
      validCount = rig.output('uint32', 1),
      invalid = rig.output('uint32', 1);
    try {
      rig.run(
        new GPUQuantileLISA({
          alternative: 'geoda-legacy',
          values: rig.input(
            Float32Array.from({length: rows}, (_, row) => row),
            'float32',
            rows
          ),
          quantileParameters: rig.input(
            Uint32Array.of(fixture.quantiles, fixture.selected),
            'uint32',
            2
          ),
          parameters: rig.input(
            getGPUPermutationParameterValues({permutationCount: 9, seed: 43}),
            'uint32',
            4
          ),
          maximumPermutations: 9,
          weights: rig.uploadWeights({
            offsets: Array.from({length: rows + 1}, (_, row) => row),
            neighbors: Array.from({length: rows}, (_, row) => (row + 1) % rows),
            weights: new Array(rows).fill(1),
            distances: []
          }),
          output: {
            membership: membership.view,
            boundaries: boundaries.view,
            validCount: validCount.view,
            invalidQuantile: invalid.view,
            classes: rig.output('uint32', rows).view,
            exceedances: rig.output('uint32', rows).view,
            pseudoPValues: rig.output('float32', rows).view,
            status: {overflow: rig.output('uint32', 1).view}
          }
        })
      );
      compare(await readFloat32(boundaries.buffer, 2), fixture.bounds, 0);
      expect(await readUint32(membership.buffer, rows)).toEqual(
        Array.from({length: rows}, (_, row) => Number(fixture.members.includes(row)))
      );
      expect(await readUint32(validCount.buffer, 1)).toEqual([rows]);
      expect(await readUint32(invalid.buffer, 1)).toEqual([0]);
    } finally {
      rig.destroy();
    }
  }
});

it('R8 complete graphs have zero Geary variance and undefined analytic inference in both modes', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // For complete unit weights sum_ij(xi-xj)^2=2*n*M2, so C=1 for every permutation.
  for (const rows of [8, 9])
    for (const mode of ['reference', 'tiled'] as const) {
      const rig = new WeightsRig(device),
        summary = rig.output('float32', 52);
      try {
        rig.run(
          new GPUGlobalSpatialStatistics({
            mode,
            statistics: ['geary'],
            values: rig.input(
              Float32Array.from({length: rows}, (_, row) => row),
              'float32',
              rows
            ),
            weights: rig.uploadWeights({
              offsets: Array.from({length: rows + 1}, (_, row) => row * (rows - 1)),
              neighbors: Array.from({length: rows}, (_, row) =>
                Array.from({length: rows}, (_, neighbor) => neighbor).filter(
                  neighbor => neighbor !== row
                )
              ).flat(),
              weights: new Array(rows * (rows - 1)).fill(1),
              distances: []
            }),
            output: {summary: summary.view}
          })
        );
        const result = await readFloat32(summary.buffer, 52);
        expect(result[18]).toBe(0);
        expect(result[21]).toBe(0);
        compare(result.slice(16, 24), [1, 1, 0, NaN, NaN, 0, NaN, NaN]);
      } finally {
        rig.destroy();
      }
    }
});

it('R6 ten tied Geary draws retain dissimilarity tie class two in exact and tiled inference', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const strategy of ['exact', 'tiled'] as const) {
    const rig = new WeightsRig(device),
      statistic = rig.output('float32', 8),
      probabilities = rig.output('float32', 8),
      classes = rig.output('uint32', 8);
    const simulations = strategy === 'exact' ? rig.output('float32', 80) : undefined;
    try {
      rig.run(
        new GPUMultivariateLocalGearyPermutationTest({
          strategy,
          variableCount: 1,
          permutationCount: 10,
          significanceLevel: 0.15,
          tail: 'esda-min-tail',
          rowTileSize: 3,
          permutationTileSize: 4,
          values: rig.input(Float32Array.of(0, 2, NaN, NaN, NaN, NaN, NaN, NaN), 'float32', 8),
          weights: rig.uploadWeights({
            offsets: [0, 1, 1, 1, 1, 1, 1, 1, 1],
            neighbors: [1],
            weights: [1],
            distances: []
          }),
          output: {
            statistic: statistic.view,
            pseudoPValues: probabilities.view,
            classes: classes.view,
            simulations: simulations?.view
          }
        })
      );
      compare(await readFloat32(statistic.buffer, 8), [4, ...new Array(7).fill(NaN)]);
      compare(await readFloat32(probabilities.buffer, 8), [1 / 11, ...new Array(7).fill(NaN)]);
      expect(await readUint32(classes.buffer, 8)).toEqual([2, ...new Array(7).fill(0xffffffff)]);
      if (simulations)
        compare(await readFloat32(simulations.buffer, 80), [
          ...new Array(10).fill(4),
          ...new Array(70).fill(NaN)
        ]);
    } finally {
      rig.destroy();
    }
  }
});

it('R4 Moran and Gi leave a constant conditional null undefined while retaining local I', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const model of ['moran', 'gi']) {
    const rig = new WeightsRig(device);
    const zScores = rig.output('float32', 9),
      pValues = rig.output('float32', 9),
      localI = rig.output('float32', 9);
    const props = {
      values: rig.input(Float32Array.of(1, 0, 0, 0, 0, 0, 0, 0, 0), 'float32', 9),
      parameters: rig.input(getGPUSpatialAutocorrelationParameterValues(), 'float32', 8),
      weights: rig.uploadWeights({
        offsets: Array.from({length: 10}, (_, row) => row),
        neighbors: Array.from({length: 9}, (_, row) => (row + 1) % 9),
        weights: new Array(9).fill(1),
        distances: []
      }),
      output: {zScores: zScores.view, pValues: pValues.view}
    };
    try {
      rig.run(
        model === 'moran'
          ? new GPULocalMoran({...props, output: {...props.output, localI: localI.view}})
          : new GPUHotSpotAnalysis({...props, selfWeight: 0})
      );
      expect((await readFloat32(zScores.buffer, 9))[0]).toBeNaN();
      expect((await readFloat32(pValues.buffer, 9))[0]).toBeNaN();
      if (model === 'moran') compare((await readFloat32(localI.buffer, 9)).slice(0, 1), [-8 / 9]);
    } finally {
      rig.destroy();
    }
  }
});

it('R3 translated adjacent float32 groups retain Geary distance four in all four paths', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Population variance=d²/4, crossing distance=d: squared standardized distance is exactly 4.
  for (const path of ['reference', 'tiled', 'exact-inference', 'tiled-inference'] as const)
    for (const shift of [0, 1]) {
      const rig = new WeightsRig(device);
      const weights = rig.uploadWeights({
        offsets: [0, 1, 2, 3, 4, 5, 6, 7, 8],
        neighbors: [1, 2, 3, 4, 5, 6, 7, 0],
        weights: new Array(8).fill(1),
        distances: []
      });
      const statistic = rig.output('float32', 8),
        status = rig.output('uint32', 8);
      const common = {
        weights,
        variableCount: 1,
        values: rig.input(
          Float32Array.from({length: 8}, (_, row) => shift + (row < 4 ? 0 : 2 ** -23)),
          'float32',
          8
        )
      };
      try {
        rig.run(
          path === 'reference' || path === 'tiled'
            ? new GPUMultivariateLocalGeary({
                ...common,
                mode: path,
                output: {statistic: statistic.view, status: status.view}
              })
            : new GPUMultivariateLocalGearyPermutationTest({
                ...common,
                strategy: path === 'exact-inference' ? 'exact' : 'tiled',
                permutationCount: 10,
                output: {
                  statistic: statistic.view,
                  pseudoPValues: rig.output('float32', 8).view,
                  classes: status.view
                }
              })
        );
        compare(await readFloat32(statistic.buffer, 8), [0, 0, 0, 4, 0, 0, 0, 4]);
      } finally {
        rig.destroy();
      }
    }
});

it('R2 weighted Pearson has undefined variance for either constant edge marginal on resubmission', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  const values = rig.input(new Float32Array(8).fill(0.1), 'float32', 8),
    second = rig.input(new Float32Array(8).fill(0.1), 'float32', 8),
    result = rig.output('float32', 1);
  rig.graph.add(
    new GPUSpatialPearson({
      values,
      secondValues: second,
      weights: rig.uploadWeights({
        offsets: [0, 1, 2, 3, 4, 5, 6, 7, 8],
        neighbors: [1, 2, 3, 4, 5, 6, 7, 0],
        weights: [1, 1, 1, 1, 1, 1, 1, 2],
        distances: []
      }),
      output: {statistic: result.view}
    })
  );
  const compiled = rig.graph.compile();
  try {
    // Constant either marginal gives exactly zero variance, hence undefined correlation.
    for (const constant of ['both', 'first', 'second']) {
      rig
        .bufferOf(values)
        .write(Float32Array.from({length: 8}, (_, row) => (constant === 'second' ? row : 0.1)));
      rig
        .bufferOf(second)
        .write(Float32Array.from({length: 8}, (_, row) => (constant === 'first' ? row : 0.1)));
      submitGraph(device, compiled, undefined);
      compare(await readFloat32(result.buffer, 1), [NaN]);
    }
  } finally {
    compiled.destroy();
    rig.destroy();
  }
});
