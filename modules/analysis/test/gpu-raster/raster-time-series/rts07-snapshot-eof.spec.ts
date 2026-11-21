// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {
  GPURasterEOFProjection,
  GPURasterLaggedCorrelation,
  GPURasterSnapshotCovariance,
  getGPURasterLaggedCorrelationParameterValues,
  getGPURasterSnapshotCovarianceMatrix,
  getGPURasterSnapshotCovarianceParameterValues,
  getGPURasterSnapshotEOFDecomposition
} from '@luma-contributors/analysis/raster-analysis';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  createRasterTimeSeriesHarness,
  getMaximumAbsoluteDifference
} from './raster-time-series-test-utils';
import reference from '../../oracles/fixtures/rts07-snapshot-eof-numpy.json';

const toFloat = (values: (number | null)[]) => values.map(value => value ?? Number.NaN);
const VALUE_MAGNITUDE = 16;

async function getDevice() {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  return device;
}

type FieldOptions = {values?: number[]; weights?: number[]; magnitude?: number};

/** Runs the two GPU steps and the CPU eigen step between them, the documented readback flow. */
async function runSnapshotEOF(
  device: Awaited<ReturnType<typeof getDevice>>,
  options: FieldOptions = {}
) {
  const {cellCount, sliceCount, modeCount} = reference;
  const values = options.values ?? toFloat(reference.values);
  const weights = options.weights ?? reference.weights;
  const parameters = getGPURasterSnapshotCovarianceParameterValues(
    {maximumValueMagnitude: options.magnitude ?? VALUE_MAGNITUDE},
    cellCount
  );
  const harness = createRasterTimeSeriesHarness(device);
  const entries = (sliceCount * (sliceCount + 1)) / 2;
  const covariance = await harness.run(
    new GPURasterSnapshotCovariance({
      cellCount,
      sliceCount,
      values: harness.floats(values),
      weights: harness.floats(weights),
      parameters: harness.floats(parameters),
      output: {
        covarianceWords: harness.output('words', 'uint32', entries * 3),
        keptCellCount: harness.output('kept', 'uint32', 1)
      }
    })
  );
  const matrix = getGPURasterSnapshotCovarianceMatrix(
    covariance['words'],
    sliceCount,
    parameters[0]
  );
  const decomposition = getGPURasterSnapshotEOFDecomposition(matrix, sliceCount, modeCount);
  const projectionHarness = createRasterTimeSeriesHarness(device);
  const projection = await projectionHarness.run(
    new GPURasterEOFProjection({
      cellCount,
      sliceCount,
      modeCount,
      values: projectionHarness.floats(values),
      weights: projectionHarness.floats(weights),
      parameters: projectionHarness.floats(parameters),
      temporalModes: projectionHarness.floats(decomposition.temporalModes),
      singularValues: projectionHarness.floats(decomposition.singularValues),
      output: {
        eofs: projectionHarness.output('eofs', 'float32', cellCount * modeCount),
        status: projectionHarness.output('status', 'uint32', cellCount)
      }
    })
  );
  return {matrix, decomposition, covariance, projection};
}

function matrixError(matrix: ArrayLike<number>, expected: ArrayLike<number>) {
  let largest = 0;
  for (let index = 0; index < expected.length; index++)
    largest = Math.max(largest, Math.abs(expected[index]));
  return getMaximumAbsoluteDifference(matrix, expected) / largest;
}

// float32 inputs and products against binary64: the Gram matrix agrees to about 1e-6 relative.
const GRAM_TOLERANCE = 1e-5;

it('RTS-07 snapshot covariance matches the weighted Gram matrix', async () => {
  const device = await getDevice();
  const {matrix, covariance} = await runSnapshotEOF(device);
  expect(covariance['kept'][0]).toBe(reference.keptCellCount);
  expect(matrixError(matrix, reference.gram)).toBeLessThan(GRAM_TOLERANCE);
});

it('RTS-07 snapshot covariance is reproducible across runs', async () => {
  const device = await getDevice();
  const first = await runSnapshotEOF(device);
  const second = await runSnapshotEOF(device);
  expect(first.covariance['words']).toEqual(second.covariance['words']);
});

it('RTS-07 eigen step reproduces the NumPy SVD and the sign convention', async () => {
  const device = await getDevice();
  const {decomposition} = await runSnapshotEOF(device);
  const {singularValues, temporalModes, explainedVariance, explainedVarianceRatio} = decomposition;
  expect(getMaximumAbsoluteDifference(singularValues, reference.singularValues)).toBeLessThan(5e-4);
  expect(getMaximumAbsoluteDifference(explainedVariance, reference.explainedVariance)).toBeLessThan(
    2e-3
  );
  expect(
    getMaximumAbsoluteDifference(explainedVarianceRatio, reference.explainedVarianceRatio)
  ).toBeLessThan(1e-5);
  expect(getMaximumAbsoluteDifference(temporalModes, reference.temporalModes)).toBeLessThan(2e-3);
  const {sliceCount, modeCount} = reference;
  for (let mode = 0; mode < modeCount; mode++) {
    let largest = 0;
    for (let slice = 0; slice < sliceCount; slice++) {
      if (
        Math.abs(temporalModes[mode * sliceCount + slice]) >
        Math.abs(temporalModes[mode * sliceCount + largest])
      ) {
        largest = slice;
      }
    }
    expect(temporalModes[mode * sliceCount + largest]).toBeGreaterThan(0);
  }
});

it('RTS-07 spatial EOFs match the NumPy singular vectors, excluding cells', async () => {
  const device = await getDevice();
  const {projection} = await runSnapshotEOF(device);
  const expected = toFloat(reference.weightedEOFs);
  expect(getMaximumAbsoluteDifference(projection['eofs'], expected)).toBeLessThan(2e-3);
  const excluded = projection['status'].flatMap((value, cell) => (value === 1 ? [cell] : []));
  // The fixture excludes one cell for a missing sample and one for a zero area weight.
  expect(excluded).toEqual([7, 13]);
  expect(Number.isNaN(projection['eofs'][7 * reference.modeCount])).toBe(true);
});

it('RTS-07 sign convention follows the temporal mode when the data sign flips', async () => {
  const device = await getDevice();
  const original = await runSnapshotEOF(device);
  const negated = await runSnapshotEOF(device, {
    values: toFloat(reference.values).map(value => (Number.isNaN(value) ? value : -value))
  });
  // Negating the data leaves the temporal mode (fixed by the convention) and negates the EOF.
  expect(
    getMaximumAbsoluteDifference(
      original.decomposition.temporalModes,
      negated.decomposition.temporalModes
    )
  ).toBeLessThan(1e-4);
  expect(
    getMaximumAbsoluteDifference(
      negated.projection['eofs'],
      original.projection['eofs'].map(value => -value)
    )
  ).toBeLessThan(1e-3);
});

it('RTS-07 area weighting changes the EOF and matches the weighted and unweighted references', async () => {
  const device = await getDevice();
  const {cellCount, modeCount} = reference;
  const unitWeights = reference.weights.map(weight => (weight === 0 ? 0 : 1));
  const unweighted = await runSnapshotEOF(device, {weights: unitWeights});
  const expected = toFloat(reference.unweighted.eofs);
  expect(getMaximumAbsoluteDifference(unweighted.projection['eofs'], expected)).toBeLessThan(2e-3);
  expect(
    getMaximumAbsoluteDifference(
      unweighted.decomposition.singularValues,
      reference.unweighted.singularValues
    )
  ).toBeLessThan(5e-4);
  const weighted = await runSnapshotEOF(device);
  let gap = 0;
  for (let index = 0; index < cellCount * modeCount; index++) {
    const left = weighted.projection['eofs'][index];
    const right = unweighted.projection['eofs'][index];
    if (!Number.isNaN(left) && !Number.isNaN(right)) gap = Math.max(gap, Math.abs(left - right));
  }
  expect(gap).toBeGreaterThan(1e-3);
});

it('RTS-07 value-space EOFs undo the area weight', async () => {
  const device = await getDevice();
  const {cellCount, sliceCount, modeCount} = reference;
  const values = toFloat(reference.values);
  const parameters = getGPURasterSnapshotCovarianceParameterValues(
    {maximumValueMagnitude: VALUE_MAGNITUDE},
    cellCount
  );
  const {decomposition} = await runSnapshotEOF(device);
  const harness = createRasterTimeSeriesHarness(device);
  const result = await harness.run(
    new GPURasterEOFProjection({
      cellCount,
      sliceCount,
      modeCount,
      values: harness.floats(values),
      weights: harness.floats(reference.weights),
      parameters: harness.floats(parameters),
      temporalModes: harness.floats(decomposition.temporalModes),
      singularValues: harness.floats(decomposition.singularValues),
      componentSpace: 'value',
      output: {
        eofs: harness.output('eofs', 'float32', cellCount * modeCount),
        status: harness.output('status', 'uint32', cellCount)
      }
    })
  );
  expect(getMaximumAbsoluteDifference(result['eofs'], toFloat(reference.valueEOFs))).toBeLessThan(
    5e-3
  );
});

it('RTS-07 a zero-variance field has zero singular values and NaN EOFs', async () => {
  const device = await getDevice();
  const {cellCount, sliceCount} = reference;
  const flat = new Array<number>(cellCount * sliceCount).fill(1.5);
  const {decomposition, projection} = await runSnapshotEOF(device, {
    values: flat,
    weights: new Array<number>(cellCount).fill(1)
  });
  expect(Array.from(decomposition.singularValues)).toEqual([0, 0, 0]);
  expect(projection['eofs'].every(Number.isNaN)).toBe(true);
});

it('RTS-07 a sample above the declared magnitude excludes its cell', async () => {
  const device = await getDevice();
  const {cellCount, sliceCount} = reference;
  const values = toFloat(reference.values);
  const limit = 2;
  let expectedKept = 0;
  for (let cell = 0; cell < cellCount; cell++) {
    const series = values.slice(cell * sliceCount, (cell + 1) * sliceCount);
    if (
      reference.weights[cell] > 0 &&
      series.every(value => Number.isFinite(value) && Math.abs(value) <= limit)
    ) {
      expectedKept++;
    }
  }
  const {covariance} = await runSnapshotEOF(device, {magnitude: limit});
  expect(covariance['kept'][0]).toBe(expectedKept);
  expect(expectedKept).toBeLessThan(reference.keptCellCount);
});

it('RTS-07 snapshot covariance refuses an oversized accumulation', async () => {
  const device = await getDevice();
  const harness = createRasterTimeSeriesHarness(device);
  const sliceCount = 256;
  const cellCount = 256 * 1024;
  const parameters = harness.floats(new Float32Array(2));
  // Views are not allocated by a constructor check; only lengths are compared.
  expect(
    () =>
      new GPURasterSnapshotCovariance({
        cellCount,
        sliceCount,
        values: harness.floats(new Float32Array(4)),
        parameters,
        output: {
          covarianceWords: harness.output('w', 'uint32', 3),
          keptCellCount: harness.output('k', 'uint32', 1)
        }
      })
  ).toThrow(/raster-snapshot-covariance accumulation invocations .*must be at most 16776960/);
});

it('RTS-07 lagged correlation matches the NumPy pairwise reference', async () => {
  const device = await getDevice();
  const {cellCount, sliceCount, lags, index, values, correlations, pairCounts} =
    reference.laggedCorrelation;
  const harness = createRasterTimeSeriesHarness(device);
  const entries = cellCount * lags.length;
  const result = await harness.run(
    new GPURasterLaggedCorrelation({
      cellCount,
      sliceCount,
      values: harness.floats(toFloat(values)),
      referenceIndex: harness.floats(toFloat(index)),
      parameters: harness.floats(getGPURasterLaggedCorrelationParameterValues()),
      lags,
      output: {
        correlations: harness.output('correlations', 'float32', entries),
        status: harness.output('status', 'uint32', entries),
        pairCount: harness.output('pairs', 'uint32', entries)
      }
    })
  );
  expect(result['status'].every(value => value === 0)).toBe(true);
  expect(result['pairs']).toEqual(pairCounts);
  expect(getMaximumAbsoluteDifference(result['correlations'], toFloat(correlations))).toBeLessThan(
    1e-3
  );
  // Cell 4 is the index delayed by two slices, so lag +2 (index leads) has the strongest value.
  const cellFour = result['correlations'].slice(4 * lags.length, 5 * lags.length);
  expect(lags[cellFour.indexOf(Math.max(...cellFour))]).toBe(2);
});

it('RTS-07 lagged correlation reports short support and zero variance', async () => {
  const device = await getDevice();
  const harness = createRasterTimeSeriesHarness(device);
  const sliceCount = 6;
  const result = await harness.run(
    new GPURasterLaggedCorrelation({
      cellCount: 2,
      sliceCount,
      values: harness.floats([
        1,
        Number.NaN,
        Number.NaN,
        Number.NaN,
        Number.NaN,
        2,
        3,
        3,
        3,
        3,
        3,
        3
      ]),
      referenceIndex: harness.floats([1, 2, 3, 4, 5, 6]),
      parameters: harness.floats(
        getGPURasterLaggedCorrelationParameterValues({minimumPairCount: 3})
      ),
      lags: [0],
      output: {
        correlations: harness.output('correlations', 'float32', 2),
        status: harness.output('status', 'uint32', 2)
      }
    })
  );
  // Cell 0 has two valid pairs, below the minimum; cell 1 is constant.
  expect(result['status']).toEqual([1, 2]);
  expect(result['correlations'].every(Number.isNaN)).toBe(true);
});

it('RT-132 RT-133 lagged correlation is scale and translation invariant on the device', async () => {
  const device = await getDevice();
  const harness = createRasterTimeSeriesHarness(device);
  // The index is [0, 1, 2, 4] translated by 1e6. Cells 0-4 are positive-slope affine images of
  // [0, 1, 2, 4] at magnitudes 1e-20 to 1e20 (r = 1 exactly); cell 5 is [1, 3, 2, 5], and
  // /usr/bin/python3 np.corrcoef([0, 1, 2, 4], [1, 3, 2, 5])[0, 1] = 0.8857142857142857.
  const index = [1e6, 1e6 + 1, 1e6 + 2, 1e6 + 4];
  const cells = [
    [1e-20, 2e-20, 3e-20, 5e-20],
    [1, 2, 3, 5],
    [1e10, 2e10, 3e10, 5e10],
    [1e20, 2e20, 3e20, 5e20],
    [1e6, 1e6 + 1, 1e6 + 2, 1e6 + 4],
    [1, 3, 2, 5]
  ];
  const cellCount = cells.length;
  const sliceCount = index.length;
  const result = await harness.run(
    new GPURasterLaggedCorrelation({
      cellCount,
      sliceCount,
      values: harness.floats(cells.flat()),
      referenceIndex: harness.floats(index),
      parameters: harness.floats(getGPURasterLaggedCorrelationParameterValues()),
      lags: [0],
      output: {
        correlations: harness.output('correlations', 'float32', cellCount),
        status: harness.output('status', 'uint32', cellCount),
        pairCount: harness.output('pairs', 'uint32', cellCount)
      }
    })
  );
  expect(result['status']).toEqual([0, 0, 0, 0, 0, 0]);
  expect(result['pairs']).toEqual([4, 4, 4, 4, 4, 4]);
  const expected = [1, 1, 1, 1, 1, 0.8857142857142857];
  for (let cell = 0; cell < cellCount; cell++) {
    expect(Number.isFinite(result['correlations'][cell])).toBe(true);
    expect(Math.abs(result['correlations'][cell] - expected[cell])).toBeLessThan(1e-5);
  }
});

it('FLT-06 decodes the centred two-slice Gram matrix before eigendecomposition', async () => {
  const device = await getDevice();
  const harness = createRasterTimeSeriesHarness(device);
  const parameters = getGPURasterSnapshotCovarianceParameterValues(
    {maximumValueMagnitude: 10000001},
    1
  );
  const covariance = await harness.run(
    new GPURasterSnapshotCovariance({
      cellCount: 1,
      sliceCount: 2,
      values: harness.floats([10000000, 10000001]),
      parameters: harness.floats(parameters),
      output: {
        covarianceWords: harness.output('gram', 'uint32', 9),
        keptCellCount: harness.output('kept', 'uint32', 1)
      }
    })
  );
  // X=[-.5,.5] exactly, so X'X=[.25,-.25;-.25,.25], independent of either contributor.
  const matrix = getGPURasterSnapshotCovarianceMatrix(covariance['gram'], 2, parameters[0]);
  expect(Array.from(matrix)).toEqual([0.25, -0.25, -0.25, 0.25]);
  expect(covariance['kept']).toEqual([1]);
  const decomposition = getGPURasterSnapshotEOFDecomposition(matrix, 2, 1);
  const projectionHarness = createRasterTimeSeriesHarness(device);
  const projection = await projectionHarness.run(
    new GPURasterEOFProjection({
      cellCount: 1,
      sliceCount: 2,
      modeCount: 1,
      values: projectionHarness.floats([10000000, 10000001]),
      parameters: projectionHarness.floats(parameters),
      temporalModes: projectionHarness.floats(decomposition.temporalModes),
      singularValues: projectionHarness.floats(decomposition.singularValues),
      output: {
        eofs: projectionHarness.output('eofs', 'float32', 1),
        status: projectionHarness.output('code', 'uint32', 1)
      }
    })
  );
  expect(projection['code']).toEqual([0]);
  expect(Number.isFinite(projection['eofs'][0])).toBe(true);
  expect(Math.abs(projection['eofs'][0] + 1)).toBeLessThan(1e-6);
});

it('review B3 preserves the small signed-field covariance entry', async () => {
  const harness = createRasterTimeSeriesHarness(await getDevice());
  const parameters = getGPURasterSnapshotCovarianceParameterValues(
    {maximumValueMagnitude: 32768},
    1
  );
  const result = await harness.run(
    new GPURasterSnapshotCovariance({
      cellCount: 1,
      sliceCount: 3,
      values: harness.floats([32768, -32768, 1 / 1024]),
      parameters: harness.floats(parameters),
      output: {
        covarianceWords: harness.output('words', 'uint32', 18),
        keptCellCount: harness.output('kept', 'uint32', 1)
      }
    })
  );
  const matrix = getGPURasterSnapshotCovarianceMatrix(result['words'], 3, parameters[0]);
  // Exact mean 1/3072; (32768 - 1/3072) * (1/1024 - 1/3072) = 21.333333121405707.
  expect(parameters[0]).toBe(-61);
  expect(result['kept']).toEqual([1]);
  expect(Number.isFinite(matrix[2])).toBe(true);
  expect(Math.abs(matrix[2] - 21.333333121405707)).toBeLessThan(2e-6);
});

it('review B7 constant six-slice cells have exactly zero EOF projection', async () => {
  const harness = createRasterTimeSeriesHarness(await getDevice());
  // Reference field [1,2,-3,0,0,0]/1024 has zero mean, singular sqrt(14)/1024,
  // and unit temporal mode [1,2,-3,0,0,0]/sqrt(14). Its own projection is 1.
  const result = await harness.run(
    new GPURasterEOFProjection({
      cellCount: 2,
      sliceCount: 6,
      modeCount: 1,
      values: harness.floats([
        10000002,
        10000002,
        10000002,
        10000002,
        10000002,
        10000002,
        1 / 1024,
        2 / 1024,
        -3 / 1024,
        0,
        0,
        0
      ]),
      parameters: harness.floats(
        getGPURasterSnapshotCovarianceParameterValues({maximumValueMagnitude: 10000002}, 2)
      ),
      temporalModes: harness.floats([
        1 / Math.sqrt(14),
        2 / Math.sqrt(14),
        -3 / Math.sqrt(14),
        0,
        0,
        0
      ]),
      singularValues: harness.floats([Math.sqrt(14) / 1024]),
      output: {
        eofs: harness.output('eofs', 'float32', 2),
        status: harness.output('code', 'uint32', 2)
      }
    })
  );
  expect(result['code']).toEqual([0, 0]);
  expect(result['eofs'][0]).toBe(0);
  expect(Number.isFinite(result['eofs'][1])).toBe(true);
  expect(Math.abs(result['eofs'][1] - 1)).toBeLessThan(1e-6);
});
