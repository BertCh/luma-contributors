// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUQuantileLISA} from '../../../src/gpu-spatial-analysis/recipes/quantile-lisa-recipe';
import {getGPUPermutationParameterValues} from '../../../src/gpu-spatial-analysis/permutation-inference/index';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  computeLocalPermutationOracle,
  createWeights,
  type CPUSpatialWeights
} from '../permutation-inference/permutation-oracle';
import {
  computeQuantileLISAReference,
  QUANTILE_LISA_REFERENCE_CASES
} from './quantile-lisa-reference';

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

function createHarness(device: Device, weights: CPUSpatialWeights = ring, maximumNeighbors = 2) {
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
    boundaries: output('boundaries', 'float32', 2),
    validCount: output('valid-count', 'uint32', 1),
    classes: output('classes', 'uint32', 8),
    invalidQuantile: output('invalid-quantile', 'uint32', 1),
    exceedances: output('exceedances', 'uint32', 8),
    pseudoPValues: output('pseudo-p-values', 'float32', 8),
    observed: output('observed', 'float32', 8),
    significant: output('significant', 'uint32', 8),
    overflow: output('overflow', 'uint32', 1)
  };
  graph.add(
    new GPUQuantileLISA({
      values: values.view,
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
        status: {overflow: outputs.overflow.view}
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
      permutationCount = PERMUTATIONS
    ) {
      const permutationSettings = getGPUPermutationParameterValues({
        seed: SEED,
        permutationCount: 1,
        significanceLevel: LEVEL
      });
      permutationSettings[2] = permutationCount;
      parameters.buffer.write(permutationSettings);
      values.buffer.write(source);
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
        overflow
      ] = await Promise.all([
        readUint32(outputs.membership.buffer, 8),
        readFloat32(outputs.boundaries.buffer, 2),
        readUint32(outputs.validCount.buffer, 1),
        readUint32(outputs.classes.buffer, 8),
        readUint32(outputs.invalidQuantile.buffer, 1),
        readUint32(outputs.exceedances.buffer, 8),
        readFloat32(outputs.pseudoPValues.buffer, 8),
        readFloat32(outputs.observed.buffer, 8),
        readUint32(outputs.significant.buffer, 8),
        readUint32(outputs.overflow.buffer, 1)
      ]);
      return {
        membership,
        boundaries,
        validCount: validCount[0],
        classes,
        invalidQuantile: invalidQuantile[0],
        exceedances,
        pseudoPValues,
        observed,
        significant,
        overflow: overflow[0]
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
    count !== 0xffffffff &&
    count + 1 <= Math.fround(Math.fround(input.significanceLevel) * (input.permutationCount + 1))
      ? 1
      : 0
  );
  return result;
}

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
