// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  createTransientView,
  GPUCommandGraph,
  type GraphDataView,
  type GPUGraphVectorFormat
} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  CPUMultipleTestingAdjustmentOperation,
  computeMultipleTestingAdjustmentOnCPU
} from '../../../src/gpu-spatial-analysis/multiple-testing/cpu-multiple-testing-adjustment-operation';
import {GPUMultipleTestingAdjustmentOperation} from '../../../src/gpu-spatial-analysis/multiple-testing/gpu-multiple-testing-adjustment-operation';
import {MULTIPLE_TESTING_FIXTURES} from './multiple-testing-fixtures';

it('delegates GPU method validation to contributor construction at lowering', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const getView = <Format extends GPUGraphVectorFormat>(port: string): GraphDataView<Format> =>
      createTransientView(graph, port, 'float32', 3) as GraphDataView<Format>;
    const operation = new GPUMultipleTestingAdjustmentOperation({
      id: 'gpu-adjust',
      pValues: 'pValues',
      rowCount: 3,
      method: 'invalid' as 'holm',
      output: {adjustedPValues: 'adjusted'}
    });
    expect(() => operation.createContributor(getView)).toThrow(/gpu-adjust method/);
  } finally {
    device.destroy();
  }
});

it('matches independent statsmodels fixtures, including missing rows and pinned families', () => {
  for (const fixture of MULTIPLE_TESTING_FIXTURES) {
    for (const method of [
      'bonferroni',
      'holm',
      'benjamini-hochberg',
      'benjamini-yekutieli'
    ] as const) {
      const result = computeMultipleTestingAdjustmentOnCPU(
        Float32Array.from(fixture.pValues),
        method,
        fixture.minimumFamilySize
      );
      result.adjustedPValues.forEach((value, row) => {
        const expected = fixture.expected[method][row];
        if (Number.isNaN(expected)) {
          expect(value).toBeNaN();
        } else {
          expect(Math.abs(value - expected)).toBeLessThan(2e-6);
        }
      });
      const tested = fixture.pValues.filter(
        value => Number.isFinite(value) && value >= 0 && value <= 1
      ).length;
      expect(result.testedCount[0]).toBe(Math.max(tested, fixture.minimumFamilySize));
    }
  }
});

it('none preserves tested values, normalizes negative zero and excludes invalid values', () => {
  const result = computeMultipleTestingAdjustmentOnCPU(
    Float32Array.of(-0, 1, NaN, -1, Infinity),
    'none',
    8
  );
  expect(Array.from(result.adjustedPValues)).toEqual([0, 1, NaN, NaN, NaN]);
  expect(result.testedCount[0]).toBe(8);
  expect(
    computeMultipleTestingAdjustmentOnCPU(Float32Array.of(0.25), 'none', 2 ** 24).testedCount[0]
  ).toBe(2 ** 24);
  expect(computeMultipleTestingAdjustmentOnCPU(new Float32Array(), 'holm').testedCount[0]).toBe(0);
});

it('preserves exact zeros and hand-derived adjustments for tiny normal p-values', () => {
  const smallest = 2 ** -80;
  const pValues = Float32Array.of(0, smallest, 2 * smallest, 4 * smallest);
  // m=80. Holm's running maxima are [0,79a,156a,308a]. BH's reverse minima are
  // [0,40a,160a/3,80a]. BY multiplies those by H_80 before final rounding.
  let harmonic = 0;
  for (let denominator = 80; denominator >= 1; denominator--) harmonic += 1 / denominator;
  const expectedValues = {
    none: [0, smallest, 2 * smallest, 4 * smallest],
    bonferroni: [0, 80 * smallest, 160 * smallest, 320 * smallest],
    holm: [0, 79 * smallest, 156 * smallest, 308 * smallest],
    'benjamini-hochberg': [0, 40 * smallest, (160 * smallest) / 3, 80 * smallest],
    'benjamini-yekutieli': [
      0,
      40 * smallest * harmonic,
      (160 * smallest * harmonic) / 3,
      80 * smallest * harmonic
    ]
  };
  for (const method of [
    'none',
    'bonferroni',
    'holm',
    'benjamini-hochberg',
    'benjamini-yekutieli'
  ] as const) {
    const result = computeMultipleTestingAdjustmentOnCPU(pValues, method, 80);
    expect(Array.from(result.adjustedPValues)).toEqual(
      Array.from(Float32Array.from(expectedValues[method]))
    );
    expect(result.testedCount[0]).toBe(80);
  }
});

it('validates construction and declares identical graph outputs', () => {
  const props = {
    id: 'adjust',
    pValues: 'pValues',
    rowCount: 3,
    method: 'holm' as const,
    output: {adjustedPValues: 'adjusted', testedCount: 'tested'}
  };
  expect(() => new CPUMultipleTestingAdjustmentOperation({...props, rowCount: 0})).toThrow(
    /adjust rowCount/
  );
  expect(
    () => new CPUMultipleTestingAdjustmentOperation({...props, minimumFamilySize: -1})
  ).toThrow(/adjust minimumFamilySize/);
  expect(
    () => new CPUMultipleTestingAdjustmentOperation({...props, method: 'invalid' as 'holm'})
  ).toThrow(/adjust method/);
  expect(() => new GPUMultipleTestingAdjustmentOperation({...props, rowCount: 0})).toThrow(
    /adjust outputs.adjustedPValues.length/
  );
  expect(new CPUMultipleTestingAdjustmentOperation(props).contract.outputs).toEqual(
    new GPUMultipleTestingAdjustmentOperation(props).contract.outputs
  );
});
