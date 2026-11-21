// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPUGeographicallyWeightedRegressionNonstationarityTest} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-geographically-weighted-regression-nonstationarity-test';
import {getGPUPermutationParameterValues} from '../../../src/gpu-spatial-analysis/permutation-inference/permutation-parameters';
import {GPUGeographicallyWeightedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-geographically-weighted-regression';
import {getGPUGeographicallyWeightedRegressionParameterValues} from '../../../src/gpu-spatial-analysis/spatial-regression/geographically-weighted-regression-parameters';
import {
  GPUGeographicallyTemporallyWeightedRegression,
  getGPUGTWRParameterValues
} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-geographically-temporally-weighted-regression';
import {readFloat32, readUint32} from '../../utils/gpu-contributor-test-utils';
import {createRegressionTestContext} from './math-review-regression-test-utils';

// Independent float64 WLS on the actual uploaded float32 x, y and positions:
// w = exp(-0.5*((position-position[focus])/30)**2)
// z = x - sum(w*x)/sum(w)
// slope = sum(w*z*(y-y[focus])) / sum(w*z*z).
// Computed with /usr/bin/python3 + NumPy, not the contributor's CPU twin.
const RESPONSE_REFERENCES = [
  {
    offset: 0,
    slopes: [1.955305524449208, 1.9851512203827477, 2.0166751299649177],
    intercepts: [9.994568230939121, 10.0001453468095, 9.997024708865046],
    residuals: [-0.8748027697127646, 0.516159172501882, 0.49740802654676447],
    localR2: [0.8288346586803108, 0.8411571472765988, 0.848313777740821]
  },
  {
    offset: 1e6,
    slopes: [1.9553754305554694, 1.9852616887027106, 2.016346515279667],
    intercepts: [1000009.9942705523, 1000009.9999012032, 1000009.9969918935],
    residuals: [-0.868684997235551, 0.5326781763254657, 0.48129650261546897],
    localR2: [0.829330072379356, 0.8414336097277564, 0.8483986593265088]
  }
];
const FOCAL_ROWS = [0, 1999, 3999];
const ROW_COUNT = 4000;

for (const route of ['gwr', 'gtwr-dense', 'gtwr-grid'] as const) {
  for (const reference of RESPONSE_REFERENCES) {
    it(`${route} response offset ${reference.offset} preserves float64 local slopes`, {
      timeout: 120000
    }, async () => {
      const context = await createRegressionTestContext();
      requireTestDevice(context.device);
      try {
        let state = 12345;
        const random = () => {
          state = (Math.imul(1664525, state) + 1013904223) >>> 0;
          return state / 2 ** 32;
        };
        const predictors = new Float32Array(ROW_COUNT);
        const response = new Float32Array(ROW_COUNT);
        const positions = new Float32Array(ROW_COUNT * 2);
        for (let row = 0; row < ROW_COUNT; row++) {
          predictors[row] = 2 * random() - 1;
          const noise = (random() - 0.5) * Math.sqrt(3);
          response[row] = reference.offset + 10 + 2 * predictors[row] + noise;
          positions[row * 2] = row / 40;
        }
        const coordinates = context.input('float32', positions);
        const positionView = context.graph.createDataView(coordinates.buffer, {
          format: 'float32x2',
          byteOffset: coordinates.byteOffset,
          length: ROW_COUNT
        });
        const coefficients = context.output('float32', ROW_COUNT * 2);
        const localStatus = context.output('uint32', ROW_COUNT);
        const fitted = context.output('float32', ROW_COUNT);
        const hatDiagonal = context.output('float32', ROW_COUNT);
        const summary = context.output('float32', 12);
        const residuals = context.output('float32', ROW_COUNT);
        const localR2 = context.output('float32', ROW_COUNT);
        const common = {
          positions: positionView,
          predictors: context.input('float32', predictors),
          predictorCount: 1,
          response: context.input('float32', response),
          output: {
            coefficients: coefficients.view,
            localStatus: localStatus.view,
            fitted: fitted.view,
            hatDiagonal: hatDiagonal.view,
            summary: summary.view
          }
        };
        if (route === 'gwr') {
          context.graph.add(
            new GPUGeographicallyWeightedRegression({
              ...common,
              output: {...common.output, residuals: residuals.view, localR2: localR2.view},
              indexGridSize: false,
              refinement: false,
              maximumBandwidthCount: 1,
              parameters: context.input(
                'float32',
                getGPUGeographicallyWeightedRegressionParameterValues(
                  {kernel: 'gaussian', bandwidths: [30]},
                  1
                )
              )
            })
          );
        } else {
          context.graph.add(
            new GPUGeographicallyTemporallyWeightedRegression({
              ...common,
              times: context.input('float32', new Float32Array(ROW_COUNT)),
              temporalContext: {
                unit: 'second',
                epoch: 0n,
                encoding: 'float32-relative',
                axis: {kind: 'irregular'}
              },
              kernel: 'gaussian',
              searchStructure: route === 'gtwr-grid' ? 'grid' : 'brute-force',
              // Maximum distance < 100 < 4*30, so no Gaussian weight is truncated.
              gaussianCutoff: route === 'gtwr-grid' ? 4 : undefined,
              parameters: context.input(
                'float32',
                getGPUGTWRParameterValues({bandwidth: 30, timeScale: 0})
              )
            })
          );
        }
        context.run();
        const statuses = await readUint32(localStatus.buffer, ROW_COUNT);
        const values = await readFloat32(coefficients.buffer, ROW_COUNT * 2);
        expect(statuses).toHaveLength(ROW_COUNT);
        expect(values).toHaveLength(ROW_COUNT * 2);
        const residualValues =
          route === 'gwr' ? await readFloat32(residuals.buffer, ROW_COUNT) : [];
        const rSquaredValues = route === 'gwr' ? await readFloat32(localR2.buffer, ROW_COUNT) : [];
        const fittedValues = await readFloat32(fitted.buffer, ROW_COUNT);
        FOCAL_ROWS.forEach((row, index) => {
          expect(statuses[row], `fit status at ${row}`).toBe(0);
          const slope = values[row * 2 + 1];
          expect(Number.isFinite(slope), `finite slope at ${row}`).toBe(true);
          expect(Math.abs(slope - reference.slopes[index]), `slope at ${row}`).toBeLessThan(2e-3);
          // Float32 publication at 1e6 rounds fitted values/intercepts by up to 1/32.
          expect(Number.isFinite(values[row * 2])).toBe(true);
          expect(Math.abs(values[row * 2] - reference.intercepts[index])).toBeLessThan(
            reference.offset ? 0.04 : 2e-3
          );
          expect(Number.isFinite(fittedValues[row])).toBe(true);
          expect(
            Math.abs(fittedValues[row] - (response[row] - reference.residuals[index]))
          ).toBeLessThan(0.04);
          if (route === 'gwr') {
            expect(Number.isFinite(residualValues[row])).toBe(true);
            expect(Math.abs(residualValues[row] - reference.residuals[index])).toBeLessThan(2e-3);
            expect(Number.isFinite(rSquaredValues[row])).toBe(true);
            expect(Math.abs(rSquaredValues[row] - reference.localR2[index])).toBeLessThan(2e-3);
          }
        });
      } finally {
        context.destroy();
      }
    });
  }
}

for (const indexed of [false, true]) {
  it(`GWR nonstationarity anchored Gaussian refits have zero spread for global support: indexed=${indexed}`, {
    timeout: 120000
  }, async () => {
    const context = await createRegressionTestContext();
    requireTestDevice(context.device);
    try {
      // Coincident positions give unit Gaussian weights in every local fit, under every
      // permutation. OLS on x=-3..4 and alternating +/-1/4 noise gives slope 85/42 and
      // intercept 1e6-1/84. Every location must recover the same model, so both spreads are 0.
      const rows = 8;
      const predictors = Float32Array.from({length: rows}, (_, row) => row - 3);
      const response = Float32Array.from(
        predictors,
        (value, row) => 1e6 + 2 * value + (row % 2 ? 0.25 : -0.25)
      );
      const coordinates = context.input('float32', new Float32Array(rows * 2));
      const positions = context.graph.createDataView(coordinates.buffer, {
        format: 'float32x2',
        byteOffset: coordinates.byteOffset,
        length: rows
      });
      const standardDeviations = context.output('float32', 4);
      const summary = context.output('float32', 4);
      context.graph.add(
        new GPUGeographicallyWeightedRegressionNonstationarityTest({
          positions,
          predictors: context.input('float32', predictors),
          predictorCount: 1,
          response: context.input('float32', response),
          bandwidthParameters: context.input(
            'float32',
            getGPUGeographicallyWeightedRegressionParameterValues(
              {kernel: 'gaussian', bandwidths: [30]},
              1
            )
          ),
          selectedBandwidth: context.input('float32', [0, 30]),
          coefficients: context.input(
            'float32',
            Float32Array.from({length: rows * 2}, (_, index) =>
              index % 2 ? 85 / 42 : 1e6 - 1 / 84
            )
          ),
          parameters: context.input(
            'uint32',
            getGPUPermutationParameterValues({seed: 12345, permutationCount: 1})
          ),
          maximumPermutations: 1,
          maximumBandwidthCount: 1,
          indexGridSize: indexed ? [2, 2] : false,
          gaussianCutoff: indexed ? 4 : undefined,
          output: {
            table: context.output('float32', 12).view,
            summary: summary.view,
            standardDeviations: standardDeviations.view
          }
        })
      );
      context.run();
      const spreads = await readFloat32(standardDeviations.buffer, 4);
      expect(spreads).toHaveLength(4);
      for (const value of spreads) {
        expect(Number.isFinite(value)).toBe(true);
        expect(Math.abs(value)).toBeLessThan(2e-3);
      }
      expect(await readFloat32(summary.buffer, 4)).toEqual([1, 8, 8, 0]);
    } finally {
      context.destroy();
    }
  });
}

it('indexed GTWR publishes model statuses consumed by its summary', async () => {
  const context = await createRegressionTestContext();
  requireTestDevice(context.device);
  try {
    // First two rows span a full-rank neighborhood; row 2 is isolated (rank one), row 3 masked.
    const coordinates = context.input('float32', [0, 0, 0.1, 0, 10, 0, 20, 0]);
    const positions = context.graph.createDataView(coordinates.buffer, {
      format: 'float32x2',
      byteOffset: coordinates.byteOffset,
      length: 4
    });
    const localStatus = context.output('uint32', 4);
    const summary = context.output('float32', 12);
    context.graph.add(
      new GPUGeographicallyTemporallyWeightedRegression({
        positions,
        times: context.input('float32', [0, 0, 0, 0]),
        temporalContext: {
          unit: 'second',
          epoch: 0n,
          encoding: 'float32-relative',
          axis: {kind: 'irregular'}
        },
        predictors: context.input('float32', [0, 1, 2, 3]),
        predictorCount: 1,
        response: context.input('float32', [1, 3, 5, 7]),
        mask: context.input('uint32', [1, 1, 1, 0]),
        parameters: context.input(
          'float32',
          getGPUGTWRParameterValues({bandwidth: 1, timeScale: 0})
        ),
        kernel: 'bisquare',
        searchStructure: 'grid',
        output: {
          coefficients: context.output('float32', 8).view,
          fitted: context.output('float32', 4).view,
          hatDiagonal: context.output('float32', 4).view,
          localStatus: localStatus.view,
          summary: summary.view
        }
      })
    );
    context.run();
    expect(await readUint32(localStatus.buffer, 4)).toEqual([0, 0, 1, 2]);
    const values = await readFloat32(summary.buffer, 12);
    expect(values[4]).toBe(2);
    expect(values[6]).toBe(1);
    expect(values[11]).toBe(3);
    expect(Number.isNaN(values[5])).toBe(true);
  } finally {
    context.destroy();
  }
});

for (const route of ['gwr', 'gtwr-dense', 'gtwr-grid'] as const) {
  it(`${route} publishes a representable small intercept after cancelling large origins`, {
    timeout: 120000
  }, async () => {
    const context = await createRegressionTestContext();
    requireTestDevice(context.device);
    try {
      const rows = 16;
      const predictors = Float32Array.from(
        {length: rows},
        (_, row) => 1e6 + (row < 2 ? -1 : row < 4 ? 1 : 0)
      );
      const response = Float32Array.from(
        predictors,
        (value, row) => value + (row >= 12 ? 1 / 16 : 0)
      );
      const coordinates = context.input('float32', new Float32Array(rows * 2));
      const positions = context.graph.createDataView(coordinates.buffer, {
        format: 'float32x2',
        byteOffset: coordinates.byteOffset,
        length: rows
      });
      const coefficients = context.output('float32', rows * 2);
      const common = {
        positions,
        predictors: context.input('float32', predictors),
        predictorCount: 1,
        response: context.input('float32', response),
        output: {
          coefficients: coefficients.view,
          fitted: context.output('float32', rows).view,
          hatDiagonal: context.output('float32', rows).view,
          localStatus: context.output('uint32', rows).view,
          summary: context.output('float32', 12).view
        }
      };
      if (route === 'gwr')
        context.graph.add(
          new GPUGeographicallyWeightedRegression({
            ...common,
            indexGridSize: false,
            refinement: false,
            maximumBandwidthCount: 1,
            parameters: context.input(
              'float32',
              getGPUGeographicallyWeightedRegressionParameterValues(
                {kernel: 'gaussian', bandwidths: [30]},
                1
              )
            )
          })
        );
      else
        context.graph.add(
          new GPUGeographicallyTemporallyWeightedRegression({
            ...common,
            kernel: 'gaussian',
            times: context.input('float32', new Float32Array(rows)),
            temporalContext: {
              unit: 'second',
              epoch: 0n,
              encoding: 'float32-relative',
              axis: {kind: 'irregular'}
            },
            searchStructure: route === 'gtwr-grid' ? 'grid' : 'brute-force',
            gaussianCutoff: route === 'gtwr-grid' ? 4 : undefined,
            parameters: context.input(
              'float32',
              getGPUGTWRParameterValues({bandwidth: 30, timeScale: 0})
            )
          })
        );
      context.run();
      const values = await readFloat32(coefficients.buffer, rows * 2);
      // At an unperturbed x=1e6 focus: X'WX=diag(16,4), X'W(y-y_i)=[1/4,4].
      expect(Number.isFinite(values[8])).toBe(true);
      expect(Math.abs(values[8] - 1 / 64)).toBeLessThan(2e-5);
    } finally {
      context.destroy();
    }
  });
}

for (const route of ['dense', 'indexed', 'search'] as const) {
  it(`GTWR ${route} retains centered residuals when every fitted response rounds to observed`, {
    timeout: 120000
  }, async () => {
    const context = await createRegressionTestContext();
    requireTestDevice(context.device);
    try {
      const coordinates = context.input('float32', new Float32Array(16));
      const positions = context.graph.createDataView(coordinates.buffer, {
        format: 'float32x2',
        byteOffset: coordinates.byteOffset,
        length: 8
      });
      const summary = context.output('float32', 12);
      const residuals = context.output('float32', 8);
      context.graph.add(
        new GPUGeographicallyTemporallyWeightedRegression({
          positions,
          times: context.input('float32', new Float32Array(8)),
          temporalContext: {
            unit: 'second',
            epoch: 0n,
            encoding: 'float32-relative',
            axis: {kind: 'irregular'}
          },
          predictors: context.input('float32', [0, 1, 2, 3, 9, 10, 11, 12]),
          predictorCount: 1,
          response: context.input('float32', [
            1e6,
            1e6,
            1e6,
            1e6,
            1e6 + 1 / 16,
            1e6 + 1 / 16,
            1e6 + 1 / 16,
            1e6 + 1 / 16
          ]),
          kernel: 'gaussian',
          searchStructure: route === 'indexed' ? 'grid' : 'brute-force',
          gaussianCutoff: route === 'indexed' ? 4 : undefined,
          bandwidthSearch:
            route === 'search'
              ? {
                  parameters: context.input('float32', [1, 2, 0, 0, 0.0001, 0.0001]),
                  maximumIterations: 1
                }
              : undefined,
          parameters: context.input(
            'float32',
            getGPUGTWRParameterValues({bandwidth: 30, timeScale: 0})
          ),
          output: {
            coefficients: context.output('float32', 16).view,
            fitted: context.output('float32', 8).view,
            hatDiagonal: context.output('float32', 8).view,
            localStatus: context.output('uint32', 8).view,
            summary: summary.view,
            residuals: residuals.view
          }
        })
      );
      context.run();
      const result = await readFloat32(summary.buffer, 12);
      const residual = await readFloat32(residuals.buffer, 8);
      // Rational OLS, x as above: slope=9/1376, centered intercept=-11/1376, RSS=5/11008.
      expect(Number.isFinite(result[0])).toBe(true);
      expect(Math.abs(result[0] - 5 / 11008)).toBeLessThan(2e-7);
      expect(Number.isFinite(result[10])).toBe(true);
      expect(Math.abs(result[10] - 27.7540165)).toBeLessThan(2e-3);
      const expected = [11, 2, -7, -16, 16, 7, -2, -11].map(value => value / 1376);
      residual.forEach((value, index) => {
        expect(Number.isFinite(value)).toBe(true);
        expect(Math.abs(value - expected[index])).toBeLessThan(2e-5);
      });
    } finally {
      context.destroy();
    }
  });
}

it('indexed localized Gaussian nonstationarity refits retain independently nonzero spreads', {
  timeout: 120000
}, async () => {
  const context = await createRegressionTestContext();
  requireTestDevice(context.device);
  try {
    const rows = 12;
    const coordinates = context.input(
      'float32',
      Float32Array.from({length: 24}, (_, index) =>
        index % 2 ? 0 : index / 2 < 6 ? index / 2 : index / 2 + 4
      )
    );
    const positions = context.graph.createDataView(coordinates.buffer, {
      format: 'float32x2',
      byteOffset: coordinates.byteOffset,
      length: rows
    });
    const predictors = Float32Array.from({length: rows}, (_, row) => (row % 6) - 2);
    const response = Float32Array.from(predictors, (value, row) =>
      row < 6 ? 10 + 2 * value : -5 + 0.5 * value
    );
    const deviations = context.output('float32', 4);
    context.graph.add(
      new GPUGeographicallyWeightedRegressionNonstationarityTest({
        positions,
        predictors: context.input('float32', predictors),
        predictorCount: 1,
        response: context.input('float32', response),
        bandwidthParameters: context.input(
          'float32',
          getGPUGeographicallyWeightedRegressionParameterValues(
            {kernel: 'gaussian', bandwidths: [2]},
            1
          )
        ),
        selectedBandwidth: context.input('float32', [0, 2]),
        coefficients: context.input(
          'float32',
          Float32Array.from({length: 24}, (_, index) =>
            index < 12 ? (index % 2 ? 2 : 10) : index % 2 ? 0.5 : -5
          )
        ),
        parameters: context.input(
          'uint32',
          getGPUPermutationParameterValues({seed: 12345, permutationCount: 1})
        ),
        maximumPermutations: 1,
        maximumBandwidthCount: 1,
        indexGridSize: [4, 2],
        gaussianCutoff: 2,
        output: {
          table: context.output('float32', 12).view,
          summary: context.output('float32', 4).view,
          standardDeviations: deviations.view
        }
      })
    );
    context.run();
    const values = await readFloat32(deviations.buffer, 4);
    // Independent NumPy weighted fits, exp(-.5*(distance/2)^2) for distance<=4.
    // Swap-or-not primitive assignment for seed12345/run1: [2,11,9,0,6,1,7,8,10,3,5,4].
    const expected = [7.5, 0.75, 1.9430792300432143, 2.6950221163588304];
    expect(values).toHaveLength(4);
    values.forEach((value, index) => {
      expect(Number.isFinite(value)).toBe(true);
      expect(Math.abs(value - expected[index])).toBeLessThan(2e-3);
    });
  } finally {
    context.destroy();
  }
});

for (const indexed of [false, true]) {
  it(`GTWR publishes explicit exact-fit infinities on device: indexed=${indexed}`, {
    timeout: 120000
  }, async () => {
    const context = await createRegressionTestContext();
    requireTestDevice(context.device);
    try {
      const coordinates = context.input('float32', new Float32Array(16));
      const positions = context.graph.createDataView(coordinates.buffer, {
        format: 'float32x2',
        byteOffset: coordinates.byteOffset,
        length: 8
      });
      const summary = context.output('float32', 12);
      context.graph.add(
        new GPUGeographicallyTemporallyWeightedRegression({
          positions,
          times: context.input('float32', new Float32Array(8)),
          temporalContext: {
            unit: 'second',
            epoch: 0n,
            encoding: 'float32-relative',
            axis: {kind: 'irregular'}
          },
          predictors: context.input('float32', [0, 1, 2, 3, 4, 5, 6, 7]),
          predictorCount: 1,
          response: context.input('float32', new Float32Array(8).fill(10)),
          kernel: 'gaussian',
          searchStructure: indexed ? 'grid' : 'brute-force',
          gaussianCutoff: indexed ? 4 : undefined,
          parameters: context.input(
            'float32',
            getGPUGTWRParameterValues({bandwidth: 30, timeScale: 0})
          ),
          output: {
            coefficients: context.output('float32', 16).view,
            fitted: context.output('float32', 8).view,
            hatDiagonal: context.output('float32', 8).view,
            localStatus: context.output('uint32', 8).view,
            summary: summary.view
          }
        })
      );
      context.run();
      const values = await readFloat32(summary.buffer, 12);
      expect(values[0]).toBe(0);
      expect(values[10]).toBe(Infinity);
      expect(values[2]).toBe(-Infinity);
      expect(Number.isNaN(values[5])).toBe(true);
    } finally {
      context.destroy();
    }
  });
}
