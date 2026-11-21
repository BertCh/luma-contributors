// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUGammaStatistic,
  GPUSpatialPearson,
  GPULocalGeary,
  getGPUSpatialAutocorrelationParameterValues
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';
import {
  AUDIT_VALUES,
  AUDIT_MASK,
  AUDIT_WEIGHTS,
  AUDIT_REFERENCE
} from './local-autocorrelation-audit-fixture';

// Independent literals from edge-association-audit.oracle.py: integer edge weights expanded
// into repeated observations, numpy.cov/corrcoef and mean differences. No GPU/TS snapshots.
const EDGE_REFERENCE = [
  {
    second: AUDIT_VALUES,
    crossProduct: 7,
    pearson: 0.4293151898304457,
    squaredDifference: 19.142857142857142,
    absoluteDifference: 3.142857142857143
  },
  {
    second: [2, 5, 1, 7, 4, 3, 11, 999, 9, NaN],
    crossProduct: 1.4285714285714284,
    pearson: 0.11530909949342485,
    squaredDifference: 24.61904761904762,
    absoluteDifference: 3.380952380952381
  }
];
const GEARY_REFERENCE = [
  [
    0.1047008547008547,
    0.47115384615384615,
    0.47115384615384615,
    1.6228632478632476,
    1.4134615384615383,
    1.4134615384615383,
    15.548076923076923,
    NaN,
    NaN,
    NaN
  ],
  [
    0.0349002849002849,
    0.23557692307692307,
    0.15705128205128205,
    0.4057158119658119,
    0.4711538461538461,
    0.4711538461538461,
    5.1826923076923075,
    NaN,
    NaN,
    NaN
  ],
  [
    0.1195144724556489,
    0.5378151260504203,
    0.5378151260504201,
    1.8524743230625587,
    1.6134453781512608,
    1.61344537815126,
    17.747899159663866,
    NaN,
    0,
    NaN
  ],
  [
    0.039838157485216304,
    0.26890756302521013,
    0.1792717086834734,
    0.46311858076563966,
    0.5378151260504203,
    0.53781512605042,
    5.915966386554622,
    NaN,
    0,
    NaN
  ]
];

function compare(actual: readonly number[], expected: readonly number[], label: string) {
  expect(actual.length).toBe(expected.length);
  for (const [index, value] of expected.entries()) {
    if (Number.isNaN(value)) expect(actual[index], `${label}[${index}]`).toBeNaN();
    else {
      expect(Number.isFinite(actual[index]), `${label}[${index}]`).toBe(true);
      // Short sums over ten rows; allows float32 centering/reduction/division rounding.
      expect(Math.abs(actual[index] - value), `${label}[${index}]`).toBeLessThanOrEqual(
        2e-5 + 4e-6 * Math.abs(value)
      );
    }
  }
}

for (const chunked of [false, true]) {
  it(`edge statistics ${chunked ? 'chunked' : 'whole'} match expanded-edge oracle and suppress either loss flag`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    for (const reference of EDGE_REFERENCE) {
      const rig = new WeightsRig(device);
      try {
        const overflow = rig.input(Uint32Array.of(0), 'uint32', 1);
        const candidateOverflow = rig.input(Uint32Array.of(0), 'uint32', 1);
        const chunks = [0, 3].map((rowStart, index) => {
          const rowEnd = index === 0 ? 3 : 10;
          const start = AUDIT_WEIGHTS.offsets[rowStart];
          const end = AUDIT_WEIGHTS.offsets[rowEnd];
          return {
            rowStart,
            rowCount: rowEnd - rowStart,
            weights: rig.uploadWeights({
              offsets: AUDIT_WEIGHTS.offsets
                .slice(rowStart, rowEnd + 1)
                .map(offset => offset - start),
              neighbors: AUDIT_WEIGHTS.neighbors.slice(start, end),
              weights: AUDIT_WEIGHTS.weights.slice(start, end),
              distances: []
            })
          };
        });
        const weights = {
          ...(chunked ? {rowCount: 10, chunks} : rig.uploadWeights(AUDIT_WEIGHTS)),
          status: {overflow, candidateOverflow}
        };
        const values = rig.input(Float32Array.from(AUDIT_VALUES), 'float32', 10);
        const secondValues = rig.input(Float32Array.from(reference.second), 'float32', 10);
        const mask = rig.input(Uint32Array.from(AUDIT_MASK), 'uint32', 10);
        const outputs = [0, 1, 2, 3].map(() => rig.output('float32', 1));
        for (const [index, operation] of [
          'cross-product',
          'squared-difference',
          'absolute-difference'
        ].entries()) {
          rig.graph.add(
            new GPUGammaStatistic({
              id: `gamma-${index}`,
              values,
              secondValues,
              mask,
              weights,
              operation: operation as
                | 'cross-product'
                | 'squared-difference'
                | 'absolute-difference',
              output: {statistic: outputs[index].view}
            })
          );
        }
        rig.graph.add(
          new GPUSpatialPearson({
            weights,
            values,
            secondValues,
            mask,
            output: {statistic: outputs[3].view}
          })
        );
        const compiled = rig.graph.compile();
        try {
          for (const scale of [1, 2]) {
            rig
              .bufferOf(values)
              .write(
                Float32Array.from(AUDIT_VALUES, value => scale * value + (scale === 2 ? 32 : 0))
              );
            rig
              .bufferOf(secondValues)
              .write(
                Float32Array.from(reference.second, value => scale * value + (scale === 2 ? 32 : 0))
              );
            submitGraph(device, compiled, undefined);
            const expected = [
              reference.crossProduct * scale * scale,
              reference.squaredDifference * scale * scale,
              reference.absoluteDifference * scale,
              reference.pearson
            ];
            for (const [index, output] of outputs.entries())
              compare(await readFloat32(output.buffer, 1), [expected[index]], `operation ${index}`);
          }
          for (const flags of [
            [9, 0],
            [0, 7],
            [0, 0]
          ]) {
            rig.bufferOf(overflow).write(Uint32Array.of(flags[0]));
            rig.bufferOf(candidateOverflow).write(Uint32Array.of(flags[1]));
            submitGraph(device, compiled, undefined);
            const expected = [
              reference.crossProduct * 4,
              reference.squaredDifference * 4,
              reference.absoluteDifference * 2,
              reference.pearson
            ];
            for (const [index, output] of outputs.entries())
              compare(
                await readFloat32(output.buffer, 1),
                [flags.some(Boolean) ? NaN : expected[index]],
                `loss operation ${index}`
              );
          }
          rig.bufferOf(mask).write(new Uint32Array(10));
          submitGraph(device, compiled, undefined);
          for (const output of outputs)
            compare(await readFloat32(output.buffer, 1), [NaN], 'empty edge population');
        } finally {
          compiled.destroy();
        }
      } finally {
        rig.destroy();
      }
    }
  });
}

it('Pearson preserves perfect and nonperfect correlations in large and small units', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  const values = rig.input(new Float32Array(8), 'float32', 8);
  const secondValues = rig.input(new Float32Array(8), 'float32', 8);
  const result = rig.output('float32', 1);
  rig.graph.add(
    new GPUSpatialPearson({
      values,
      secondValues,
      weights: rig.uploadWeights({
        offsets: [0, 1, 2, 3, 4, 5, 6, 7, 8],
        neighbors: [1, 2, 3, 4, 5, 6, 7, 0],
        weights: new Array(8).fill(1),
        distances: []
      }),
      output: {statistic: result.view}
    })
  );
  const compiled = rig.graph.compile();
  try {
    for (const scale of [1, 2 ** 32, 2 ** -48]) {
      rig.bufferOf(values).write(Float32Array.from({length: 8}, (_, index) => (index + 1) * scale));
      for (const swapLastTwo of [false, true]) {
        rig.bufferOf(secondValues).write(
          Float32Array.from({length: 8}, (_, index) => {
            const value = index === 0 ? 8 : index;
            return (swapLastTwo && value >= 7 ? 15 - value : value) * scale;
          })
        );
        submitGraph(device, compiled, undefined);
        // Both marginals are 1..8, with square sum 42. Swapping 7 and 8 lowers their cross sum
        // by 1, hence r=41/42. At small units the old denominator underflows and clamps r to 1.
        compare(await readFloat32(result.buffer, 1), [swapLastTwo ? 41 / 42 : 1], `units ${scale}`);
      }
    }
  } finally {
    compiled.destroy();
    rig.destroy();
  }
});

it('local Geary publishes all outputs against population-standardized distances', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const [index, reference] of AUDIT_REFERENCE.entries()) {
    const rig = new WeightsRig(device);
    const values = rig.input(Float32Array.from(AUDIT_VALUES), 'float32', 10);
    const result = rig.output('float32', 10);
    const globalStatistics = rig.output('float32', 5);
    const globalCounts = rig.output('uint32', 2);
    const invalidCount = rig.output('uint32', 1);
    const overflow = rig.output('uint32', 1);
    const candidateOverflow = rig.output('uint32', 1);
    const inputOverflow = rig.input(Uint32Array.of(0), 'uint32', 1);
    const inputCandidateOverflow = rig.input(Uint32Array.of(0), 'uint32', 1);
    rig.graph.add(
      new GPULocalGeary({
        values,
        weights: {
          ...rig.uploadWeights(AUDIT_WEIGHTS),
          status: {overflow: inputOverflow, candidateOverflow: inputCandidateOverflow},
          information: reference.rowStandardized
            ? {rowCount: 10, transform: 'row', symmetric: 'unknown'}
            : undefined
        },
        mask: rig.input(Uint32Array.from(AUDIT_MASK), 'uint32', 10),
        islands: reference.islands,
        parameters: rig.input(getGPUSpatialAutocorrelationParameterValues(), 'float32', 8),
        output: {
          statistic: result.view,
          globalStatistics: globalStatistics.view,
          globalCounts: globalCounts.view,
          status: {
            invalidCount: invalidCount.view,
            overflow: overflow.view,
            candidateOverflow: candidateOverflow.view
          }
        }
      })
    );
    const compiled = rig.graph.compile();
    try {
      for (const scale of [1, 2]) {
        rig
          .bufferOf(values)
          .write(Float32Array.from(AUDIT_VALUES, value => scale * value + (scale === 2 ? 32 : 0)));
        submitGraph(device, compiled, undefined);
        compare(await readFloat32(result.buffer, 10), GEARY_REFERENCE[index], 'local Geary');
        compare(
          await readFloat32(globalStatistics.buffer, 5),
          [
            reference.globalStatistics[0],
            reference.globalStatistics[1] * scale + (scale === 2 ? 32 : 0),
            reference.globalStatistics[2] * scale * scale,
            reference.globalStatistics[3] * scale,
            1
          ],
          'Geary moments'
        );
        expect(await readUint32(globalCounts.buffer, 2)).toEqual(reference.globalCounts);
        expect(await readUint32(invalidCount.buffer, 1)).toEqual([1]);
        expect(await readUint32(overflow.buffer, 1)).toEqual([0]);
        expect(await readUint32(candidateOverflow.buffer, 1)).toEqual([0]);
      }
      for (const flags of [
        [9, 0],
        [0, 7]
      ]) {
        rig.bufferOf(inputOverflow).write(Uint32Array.of(flags[0]));
        rig.bufferOf(inputCandidateOverflow).write(Uint32Array.of(flags[1]));
        submitGraph(device, compiled, undefined);
        compare(await readFloat32(result.buffer, 10), new Array(10).fill(NaN), 'incomplete Geary');
        expect(await readUint32(overflow.buffer, 1)).toEqual([Number(flags[0] !== 0)]);
        expect(await readUint32(candidateOverflow.buffer, 1)).toEqual([Number(flags[1] !== 0)]);
      }
    } finally {
      compiled.destroy();
      rig.destroy();
    }
  }
});

it('FLT-05 reversed edge Pearson and Gamma are invariant under a 10000000 translation', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const baseline of [0, 10_000_000]) {
    const rig = new WeightsRig(device);
    try {
      const values = rig.input(Float32Array.of(baseline, baseline + 1), 'float32', 2);
      const weights = rig.uploadWeights({
        offsets: [0, 1, 2],
        neighbors: [1, 0],
        weights: [1, 1],
        distances: []
      });
      const pearson = rig.output('float32', 1);
      const gamma = rig.output('float32', 1);
      rig.graph.add(
        new GPUSpatialPearson({
          weights,
          values,
          secondValues: values,
          output: {statistic: pearson.view}
        })
      );
      rig.graph.add(
        new GPUGammaStatistic({
          weights,
          values,
          secondValues: values,
          operation: 'cross-product',
          output: {statistic: gamma.view}
        })
      );
      const compiled = rig.graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        // The two centered pairs are (-.5,.5),(.5,-.5): covariance=-.25, r=-1.
        const actualPearson = (await readFloat32(pearson.buffer, 1))[0];
        const actualGamma = (await readFloat32(gamma.buffer, 1))[0];
        expect(Number.isFinite(actualPearson)).toBe(true);
        expect(Number.isFinite(actualGamma)).toBe(true);
        expect(actualPearson).toBeCloseTo(-1, 6);
        expect(actualGamma).toBeCloseTo(-0.25, 6);
      } finally {
        compiled.destroy();
      }
    } finally {
      rig.destroy();
    }
  }
});

it('B1 a distant minimum preserves the directed-ring cross-product', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const values = rig.input(Float32Array.of(-16777216, 0, 1), 'float32', 3);
    const secondValues = rig.input(Float32Array.of(3, 0, -3), 'float32', 3);
    const weights = rig.uploadWeights({
      offsets: [0, 1, 2, 3],
      neighbors: [1, 2, 0],
      weights: [1, 1, 1],
      distances: []
    });
    const gamma = rig.output('float32', 1);
    rig.graph.add(
      new GPUGammaStatistic({
        weights,
        values,
        secondValues,
        operation: 'cross-product',
        output: {statistic: gamma.view}
      })
    );
    const compiled = rig.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const result = await readFloat32(gamma.buffer, 1);
      expect(result).toHaveLength(1);
      expect(Number.isFinite(result[0])).toBe(true);
      // Destination mean is zero; covariance = ((-16777216)*0 + 0*(-3) + 1*3)/3.
      expect(result[0]).toBe(1);
    } finally {
      compiled.destroy();
    }
  } finally {
    rig.destroy();
  }
});
