// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {isGeodaTailSignificant} from './geoda-tail-significance';
import {GPUQuantileLISA} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/gpu-quantile-lisa';
import {getGPUPermutationParameterValues} from '../../../src/gpu-spatial-analysis/permutation-inference/index';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  computeLocalPermutationOracle,
  createWeights,
  type CPUSpatialWeights
} from '../permutation-inference/permutation-oracle';
import {
  QUANTILE_LISA_LEGACY_INFERENCE,
  computeQuantileLISAReference,
  QUANTILE_LISA_REFERENCE_CASES
} from './quantile-lisa-reference';

import MULTIVARIATE_REFERENCE from '../../oracles/fixtures/multivariate-quantile-lisa-numpy.json';
import GEODA_REFERENCE from '../../oracles/fixtures/quantile-lisa-geoda.json';

const SEED = 314159;
const PERMUTATIONS = 199;
const LEVEL = 0.1;
const ring = createWeights(
  Array.from(
    {length: 8},
    (_, row) =>
      [
        [(row + 1) % 8, 1],
        [(row + 7) % 8, 1]
      ].sort((left, right) => left[0] - right[0]) as [number, number][]
  )
);

function createHarness(
  device: Device,
  weights: CPUSpatialWeights = ring,
  maximumNeighbors = 2,
  variableCount = 1
) {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const input = <Format extends 'uint32' | 'float32'>(
    name: string,
    format: Format,
    values: Uint32Array | Float32Array
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, format, values.length)};
  };
  const output = <Format extends 'uint32' | 'float32'>(
    name: string,
    format: Format,
    length: number
  ) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, format, length)};
  };
  const values = input('values', 'float32', new Float32Array(8));
  const additionalVariables = Array.from({length: variableCount - 1}, (_, variableIndex) => ({
    values: input(`values-${variableIndex}`, 'float32', new Float32Array(8)),
    quantileParameters: input(`quantiles-${variableIndex}`, 'uint32', new Uint32Array([2, 2]))
  }));
  const mask = input('mask', 'uint32', new Uint32Array(8).fill(1));
  const quantileParameters = input('quantile-parameters', 'uint32', new Uint32Array([4, 1]));
  const parameters = input(
    'inference-parameters',
    'uint32',
    getGPUPermutationParameterValues({
      seed: SEED,
      permutationCount: PERMUTATIONS,
      significanceLevel: LEVEL
    })
  );
  const outputs = {
    membership: output('membership', 'uint32', 8),
    boundaries: output('boundaries', 'float32', 2 * variableCount),
    validCount: output('valid-count', 'uint32', variableCount),
    classes: output('classes', 'uint32', 8),
    invalidQuantile: output('invalid-quantile', 'uint32', 1),
    exceedances: output('exceedances', 'uint32', 8),
    pseudoPValues: output('pseudo-p-values', 'float32', 8),
    observed: output('observed', 'float32', 8),
    significant: output('significant', 'uint32', 8),
    overflow: output('overflow', 'uint32', 1),
    invalidCount: output('invalid-count', 'uint32', 1),
    islandCount: output('island-count', 'uint32', 1)
  };
  graph.add(
    new GPUQuantileLISA({
      ...QUANTILE_LISA_LEGACY_INFERENCE,
      values: values.view,
      additionalVariables: additionalVariables.map(variable => ({
        values: variable.values.view,
        quantileParameters: variable.quantileParameters.view
      })),
      mask: mask.view,
      quantileParameters: quantileParameters.view,
      parameters: parameters.view,
      maximumPermutations: PERMUTATIONS,
      maximumNeighbors,
      weights: {
        offsets: input('offsets', 'uint32', weights.offsets).view,
        neighbors: input('neighbors', 'uint32', weights.neighbors).view,
        weights: input('weights', 'float32', weights.weights).view
      },
      output: {
        membership: outputs.membership.view,
        boundaries: outputs.boundaries.view,
        validCount: outputs.validCount.view,
        classes: outputs.classes.view,
        invalidQuantile: outputs.invalidQuantile.view,
        exceedances: outputs.exceedances.view,
        pseudoPValues: outputs.pseudoPValues.view,
        observed: outputs.observed.view,
        significant: outputs.significant.view,
        islandCount: outputs.islandCount.view,
        status: {overflow: outputs.overflow.view, invalidCount: outputs.invalidCount.view}
      }
    })
  );
  const compiled = graph.compile();
  return {
    async run(
      source: Float32Array,
      quantileCount: number,
      selectedQuantile: number,
      rowMask = new Uint32Array(8).fill(1),
      permutationCount = PERMUTATIONS,
      additional: {values: Float32Array; settings: number[]}[] = []
    ) {
      const permutationSettings = getGPUPermutationParameterValues({
        seed: SEED,
        permutationCount: 1,
        significanceLevel: LEVEL
      });
      permutationSettings[2] = permutationCount;
      parameters.buffer.write(permutationSettings);
      values.buffer.write(source);
      for (const [variableIndex, variable] of additional.entries()) {
        additionalVariables[variableIndex].values.buffer.write(variable.values);
        additionalVariables[variableIndex].quantileParameters.buffer.write(
          Uint32Array.from(variable.settings)
        );
      }
      mask.buffer.write(rowMask);
      quantileParameters.buffer.write(new Uint32Array([quantileCount, selectedQuantile]));
      submitGraph(device, compiled, undefined);
      const [
        membership,
        boundaries,
        validCount,
        classes,
        invalidQuantile,
        exceedances,
        pseudoPValues,
        observed,
        significant,
        overflow,
        invalidCount,
        islandCount
      ] = await Promise.all([
        readUint32(outputs.membership.buffer, 8),
        readFloat32(outputs.boundaries.buffer, 2 * variableCount),
        readUint32(outputs.validCount.buffer, variableCount),
        readUint32(outputs.classes.buffer, 8),
        readUint32(outputs.invalidQuantile.buffer, 1),
        readUint32(outputs.exceedances.buffer, 8),
        readFloat32(outputs.pseudoPValues.buffer, 8),
        readFloat32(outputs.observed.buffer, 8),
        readUint32(outputs.significant.buffer, 8),
        readUint32(outputs.overflow.buffer, 1),
        readUint32(outputs.invalidCount.buffer, 1),
        readUint32(outputs.islandCount.buffer, 1)
      ]);
      return {
        membership,
        boundaries,
        validCount: validCount[0],
        validCounts: validCount,
        classes,
        invalidQuantile: invalidQuantile[0],
        exceedances,
        pseudoPValues,
        observed,
        significant,
        overflow: overflow[0],
        invalidCount: invalidCount[0],
        islandCount: islandCount[0]
      };
    },
    destroy() {
      compiled.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  };
}

function computeGeodaTailOracle(input: Parameters<typeof computeLocalPermutationOracle>[0]) {
  const result = computeLocalPermutationOracle(input);
  result.exceedances = result.exceedances.map(count =>
    count === 0xffffffff ? count : Math.min(count, input.permutationCount - count)
  );
  result.pseudoPValues = result.exceedances.map(count =>
    count === 0xffffffff ? NaN : Math.fround((count + 1) / (input.permutationCount + 1))
  );
  result.significant = result.exceedances.map(count =>
    isGeodaTailSignificant(count, input.permutationCount, input.significanceLevel) ? 1 : 0
  );
  return result;
}

it('the Geoda tail oracle preserves exact ratio boundaries', () => {
  // 1/25 exceeds the stored .04 cutoff; 1/32 equals the exactly representable .03125.
  expect(isGeodaTailSignificant(0, 24, 0.04)).toBe(false);
  expect(isGeodaTailSignificant(0, 31, 0.03125)).toBe(true);
});

function expectNumbers(actual: number[], expected: readonly number[]) {
  expect(actual.length).toBe(expected.length);
  for (let row = 0; row < actual.length; row++) {
    if (!Number.isFinite(expected[row])) expect(actual[row]).toBe(expected[row]);
    else expect(actual[row]).toBeCloseTo(expected[row], 5);
  }
}

it('quantile LISA executes pinned boundaries, membership and conditional inference with resident replay', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createHarness(device);
  try {
    for (const fixture of QUANTILE_LISA_REFERENCE_CASES) {
      const values = Float32Array.from(fixture.values);
      const reference = computeQuantileLISAReference(
        values,
        fixture.quantileCount,
        fixture.selectedQuantile
      );
      const expected = computeGeodaTailOracle({
        weights: ring,
        values: Float32Array.from(reference.membership),
        mask: reference.validMask,
        statistic: 'localJoinCount',
        alternative: 'greater',
        islands: 'exclude',
        seed: SEED,
        permutationCount: PERMUTATIONS,
        significanceLevel: LEVEL,
        maximumNeighbors: 2
      });
      const actual = await harness.run(values, fixture.quantileCount, fixture.selectedQuantile);
      expect(actual.membership, fixture.name).toEqual(fixture.membership);
      expectNumbers(actual.boundaries, fixture.boundaries);
      expect(actual.validCount).toBe(8);
      expect(actual.overflow).toBe(0);
      expect(actual.invalidQuantile).toBe(0);
      const geoda = GEODA_REFERENCE.cases.find(
        referenceCase => referenceCase.name === fixture.name
      )!;
      expectNumbers(actual.observed, geoda.observed);
      // Broad cross-library comparison only: upstream rounded-index sampling is biased at endpoints.
      // The independent uniform Philox oracle below provides the exact inference assertion.
      const referenceTolerance = 0.2;
      for (let row = 0; row < 8; row++) {
        const referencePValue = geoda.pseudoPValues[row];
        if (referencePValue !== null)
          expect(Math.abs(actual.pseudoPValues[row] - referencePValue)).toBeLessThan(
            referenceTolerance
          );
      }
      expect(actual.exceedances).toEqual(expected.exceedances);
      expectNumbers(actual.pseudoPValues, expected.pseudoPValues);
      expectNumbers(actual.observed, expected.observed);
      expect(actual.classes).toEqual(expected.significant);
      expect(await harness.run(values, fixture.quantileCount, fixture.selectedQuantile)).toEqual(
        actual
      );
    }
  } finally {
    harness.destroy();
  }
});

it('quantile LISA omits missing rows, exposes islands, invalid settings and neighbor overflow', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const weights = createWeights([
    [[1, 1]],
    [
      [0, 1],
      [2, 1]
    ],
    [
      [1, 1],
      [3, 1]
    ],
    [[2, 1]],
    [],
    [[4, 1]],
    [[7, 1]],
    [[6, 1]]
  ]);
  const harness = createHarness(device, weights);
  try {
    const values = new Float32Array([-5, -5, 0, 2, 4, 6, NaN, Infinity]);
    const mask = new Uint32Array([1, 1, 1, 0, 1, 1, 1, 1]);
    const reference = computeQuantileLISAReference(values, 3, 3, mask);
    const actual = await harness.run(values, 3, 3, mask);
    expect(actual.membership).toEqual(Array.from(reference.membership));
    expect(actual.validCount).toBe(5);
    expectNumbers(actual.boundaries, reference.boundaries);
    expect(actual.classes[3]).toBe(2);
    expect(actual.classes[4]).toBe(3);
    expect(actual.classes[6]).toBe(2);
    expect(actual.classes[7]).toBe(2);
    const expected = computeGeodaTailOracle({
      weights,
      values: Float32Array.from(reference.membership),
      mask: reference.validMask,
      statistic: 'localJoinCount',
      alternative: 'greater',
      islands: 'exclude',
      seed: SEED,
      permutationCount: PERMUTATIONS,
      significanceLevel: LEVEL,
      maximumNeighbors: 2
    });
    expect(actual.exceedances).toEqual(expected.exceedances);
    expectNumbers(actual.pseudoPValues, expected.pseudoPValues);
    const empty = await harness.run(new Float32Array(8).fill(NaN), 4, 4);
    expect(empty.membership).toEqual(new Array(8).fill(0));
    expect(empty.validCount).toBe(0);
    expect(empty.classes).toEqual(new Array(8).fill(2));
    expect(empty.pseudoPValues.every(Number.isNaN)).toBe(true);
    for (const finiteCount of [1, 2]) {
      const sparse = new Float32Array(8).fill(NaN);
      sparse[0] = -4;
      if (finiteCount === 2) sparse[1] = 6;
      const sparseReference = computeQuantileLISAReference(sparse, 4, 4);
      const sparseActual = await harness.run(sparse, 4, 4);
      expect(sparseActual.membership).toEqual(Array.from(sparseReference.membership));
      expect(sparseActual.validCount).toBe(finiteCount);
      expectNumbers(sparseActual.boundaries, sparseReference.boundaries);
    }
    for (const permutationCount of [0, PERMUTATIONS + 1]) {
      const invalid = await harness.run(
        new Float32Array(8).fill(5),
        4,
        4,
        new Uint32Array(8).fill(1),
        permutationCount
      );
      expect(invalid.overflow).toBe(1);
      expect(invalid.invalidQuantile).toBe(0);
      expect(invalid.pseudoPValues.every(Number.isNaN)).toBe(true);
    }
    for (const [quantileCount, selectedQuantile] of [
      [0, 1],
      [4, 0],
      [4, 5],
      [8, 1]
    ]) {
      const invalid = await harness.run(
        new Float32Array(8).fill(5),
        quantileCount,
        selectedQuantile
      );
      expect(invalid.invalidQuantile).toBe(1);
      expect(invalid.membership).toEqual(new Array(8).fill(0));
      expect(invalid.pseudoPValues.every(Number.isNaN)).toBe(true);
    }
  } finally {
    harness.destroy();
  }
  const limited = createHarness(device, ring, 1);
  try {
    const actual = await limited.run(new Float32Array(8).fill(5), 4, 4);
    expect(actual.overflow).toBe(1);
    expect(actual.pseudoPValues.every(Number.isNaN)).toBe(true);
    expect(actual.classes).toEqual(new Array(8).fill(0));
  } finally {
    limited.destroy();
  }
});

it('multivariate quantile LISA matches independent NumPy intersections and conditional inference', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const fixture of MULTIVARIATE_REFERENCE.cases) {
    // Extra slots exercise self edges, masked padding, out-of-range IDs and nonpositive weights.
    const weights = createWeights(
      fixture.neighbors.map((neighbors, row) => [
        ...neighbors.map(neighbor => [neighbor, 3] as [number, number]),
        [row, 1] as [number, number],
        [0xffffffff, 1] as [number, number],
        [0, 0] as [number, number],
        [1, -1] as [number, number],
        [2, NaN] as [number, number]
      ])
    );
    const harness = createHarness(device, weights, 2, fixture.columns.length);
    const columns = fixture.columns.map(column =>
      Float32Array.from(column, value => (value === null ? NaN : value))
    );
    const additional = columns
      .slice(1)
      .map((values, variableIndex) => ({values, settings: fixture.settings[variableIndex + 1]}));
    const run = (
      settings = fixture.settings[0],
      variables = additional,
      permutationCount = PERMUTATIONS
    ) =>
      harness.run(
        columns[0],
        settings[0],
        settings[1],
        Uint32Array.from(fixture.mask),
        permutationCount,
        variables
      );
    try {
      const actual = await run();
      expect(actual.membership, fixture.name).toEqual(fixture.membership);
      expect(actual.validCounts, fixture.name).toEqual(fixture.validCounts);
      expectNumbers(
        actual.boundaries,
        fixture.boundaries.map(boundary =>
          typeof boundary === 'number' ? boundary : Number(boundary)
        )
      );
      expectNumbers(
        actual.observed,
        fixture.observed.map(value => (value === null ? NaN : value))
      );
      expect(actual.islandCount, fixture.name).toBe(fixture.islandCount);
      expect(actual.overflow).toBe(0);
      expect(actual.invalidCount).toBe(0);
      expect(actual.invalidQuantile).toBe(0);
      const expected = computeGeodaTailOracle({
        weights,
        values: Float32Array.from(fixture.membership),
        mask: Uint32Array.from(fixture.validMask),
        statistic: 'localJoinCount',
        alternative: 'greater',
        islands: 'exclude',
        seed: SEED,
        permutationCount: PERMUTATIONS,
        significanceLevel: LEVEL,
        maximumNeighbors: 2
      });
      expect(actual.exceedances, fixture.name).toEqual(expected.exceedances);
      expectNumbers(actual.pseudoPValues, expected.pseudoPValues);
      expect(actual.significant).toEqual(expected.significant);
      expect(actual.classes).toEqual(
        fixture.baseClasses.map(
          (classification, row) => classification || expected.significant[row]
        )
      );
      // Independently enumerated conditional distributions also constrain the sampled result.
      for (const [row, probability] of fixture.upperTailProbabilities.entries()) {
        if (probability !== null) {
          const expectedTail = Math.min(probability, 1 - probability);
          expect(Math.abs(actual.pseudoPValues[row] - expectedTail)).toBeLessThan(0.15);
        }
      }
      expect(await run()).toEqual(actual);
      // Resident changes must replace previous intersections and clear aggregated invalid settings.
      for (const variableIndex of [0, fixture.columns.length - 1]) {
        const invalidVariables = additional.map(variable => ({...variable}));
        if (variableIndex > 0)
          invalidVariables[variableIndex - 1] = {
            ...invalidVariables[variableIndex - 1],
            settings: [2, 0]
          };
        const invalid = await run(
          variableIndex === 0 ? [2, 0] : fixture.settings[0],
          invalidVariables
        );
        expect(invalid.invalidQuantile).toBe(1);
        expect(invalid.membership).toEqual(new Array(8).fill(0));
        expect(invalid.classes).toEqual(new Array(8).fill(2));
        expect(invalid.exceedances).toEqual(new Array(8).fill(0xffffffff));
        expect(invalid.pseudoPValues.every(Number.isNaN)).toBe(true);
        expect(await run()).toEqual(actual);
      }
      const invalidPermutation = await run(fixture.settings[0], additional, 0);
      expect(invalidPermutation.overflow).toBe(1);
      expect(invalidPermutation.exceedances).toEqual(new Array(8).fill(0xffffffff));
    } finally {
      harness.destroy();
    }
  }
  const limited = createHarness(device, ring, 1, 2);
  try {
    const actual = await limited.run(
      new Float32Array(8).fill(5),
      4,
      4,
      new Uint32Array(8).fill(1),
      PERMUTATIONS,
      [{values: new Float32Array(8).fill(7), settings: [2, 2]}]
    );
    expect(actual.overflow).toBe(1);
    expect(actual.exceedances).toEqual(new Array(8).fill(0xffffffff));
    expect(actual.pseudoPValues.every(Number.isNaN)).toBe(true);
  } finally {
    limited.destroy();
  }
});
