// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  computeWeightsAlgebra,
  PLATFORM4_WEIGHTS_BINDINGS
} from '../src/analysis/catalog-platform4-weights';
import type {WeightsSetOperation} from '../src/analysis/catalog-platform4-weights';
import {normalizeCatalogParameters} from '../src/analysis/catalog-bindings';

test('canonical XOR preserves left-only and right-only edges and drops the shared edge', () => {
  // Row zero: left={1:2,2:3}, right={2:5,3:7}. XOR is exactly {1:2,3:7}.
  const left = {
    offsets: Uint32Array.of(0, 2, 2, 2, 2),
    neighbors: Uint32Array.of(1, 2),
    weights: Float32Array.of(2, 3)
  };
  const right = {
    offsets: Uint32Array.of(0, 2, 2, 2, 2),
    neighbors: Uint32Array.of(2, 3),
    weights: Float32Array.of(5, 7)
  };
  const operation = 'symmetric-difference';
  const result = computeWeightsAlgebra(left, right, operation, 'sum');
  expect(Array.from(result.offsets)).toEqual([0, 2, 2, 2, 2]);
  expect(Array.from(result.neighbors)).toEqual([1, 3]);
  expect(Array.from(result.weights)).toEqual([2, 7]);
  expect(() => computeWeightsAlgebra(left, right, 'xor' as WeightsSetOperation, 'sum')).toThrow(
    /operation.*union.*symmetric-difference.*xor/
  );
  const binding = PLATFORM4_WEIGHTS_BINDINGS.find(binding => binding.id === 'weights-algebra')!;
  expect(normalizeCatalogParameters(binding, {operation})['operation']).toBe(operation);
  const removedOperation = 'symmetricDifference';
  expect(() =>
    computeWeightsAlgebra(left, right, removedOperation as WeightsSetOperation, 'sum')
  ).toThrow(/operation.*symmetric-difference/);
  expect(() => normalizeCatalogParameters(binding, {operation: removedOperation})).toThrow(
    /operation.*symmetric-difference/
  );
});
