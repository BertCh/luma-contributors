// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, expectTypeOf, it, vi} from 'vitest';
import {
  createTransientView,
  GPUCommandGraph,
  type GraphDataView,
  type GPUGraphVectorFormat
} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  CPUMultipleTestingAdjustmentOperation,
  computeMultipleTestingAdjustmentOnCPU,
  type CPUMultipleTestingAdjustmentResult
} from '../../../src/gpu-spatial-analysis/multiple-testing/cpu-multiple-testing-adjustment-operation';
import {GPUMultipleTestingAdjustmentOperation} from '../../../src/gpu-spatial-analysis/multiple-testing/gpu-multiple-testing-adjustment-operation';
import {MULTIPLE_TESTING_FIXTURES} from './multiple-testing-fixtures';
import {
  getMultipleTestingAdjustedPValues,
  getFamilyHarmonicNumber,
  GPUMultipleTestingAdjustment,
  MAXIMUM_MULTIPLE_TESTING_EXACT_HARMONIC_TERMS
} from '../../../src/gpu-spatial-analysis/multiple-testing/gpu-multiple-testing-adjustment';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';

it('TYP-16 exposes the named complete adjustment result type', () => {
  expectTypeOf<
    ReturnType<typeof computeMultipleTestingAdjustmentOnCPU>
  >().toEqualTypeOf<CPUMultipleTestingAdjustmentResult>();
});

it('CMP-14 GPU operations bind independently selected counts and capture settings', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const port of ['finiteTestCount', 'effectiveFamilySize'] as const) {
      const graph = new GPUCommandGraph(device);
      const views = {
        pValues: createTransientView(graph, 'p', 'float32', 2),
        adjustedPValues: createTransientView(graph, 'adjusted', 'float32', 2),
        [port]: createTransientView(graph, 'count', 'uint32', 1)
      };
      const props = {
        pValues: 'probabilities',
        rowCount: 2,
        minimumFamilySize: 10,
        method: 'bonferroni' as 'bonferroni' | 'none',
        output: {adjustedPValues: 'adjusted', [port]: 'count'}
      };
      const operation = new GPUMultipleTestingAdjustmentOperation(props);
      props.method = 'none';
      props.minimumFamilySize = 1;
      props.output[port] = 'redirected';
      const getView = <Format extends GPUGraphVectorFormat>(name: string): GraphDataView<Format> =>
        views[name as keyof typeof views] as GraphDataView<Format>;
      const contributor = operation.createContributor(getView) as GPUMultipleTestingAdjustment;
      expect(contributor.props.method).toBe('bonferroni');
      expect(contributor.props.minimumFamilySize).toBe(10);
      expect(contributor.props.output[port]).toBe(views[port]);
      contributor.getCommandNodes(graph);
    }
  } finally {
    device.destroy();
  }
});

it('W3-01 shares the exact harmonic bound and preserves both sides of the transition', () => {
  expect(MAXIMUM_MULTIPLE_TESTING_EXACT_HARMONIC_TERMS).toBe(64);
  // /usr/bin/python3: float(sum(Fraction(1,k) for k in range(1,m+1))), then H_m*m/2**16.
  for (const [familySize, harmonic, expected] of [
    [63, 4.728265903705769, 0.004545299559531608],
    [64, 4.743890903705769, 0.004632705960650165],
    [65, 4.759275519090385, 0.004720350780347824]
  ]) {
    const value = getFamilyHarmonicNumber(familySize);
    expect(Number.isFinite(value)).toBe(true);
    expect(Math.abs(value - harmonic)).toBeLessThan(2e-15);
    const result = computeMultipleTestingAdjustmentOnCPU(
      Float32Array.of(2 ** -16),
      'benjamini-yekutieli',
      familySize
    );
    expect(result.adjustedPValues).toEqual(Float32Array.of(expected));
  }
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    recordWGSLKernelNodes(graph);
    new GPUMultipleTestingAdjustment({
      id: 'harmonic-bound',
      method: 'benjamini-yekutieli',
      pValues: createTransientView(graph, 'p', 'float32', 2),
      output: {adjustedPValues: createTransientView(graph, 'adjusted', 'float32', 2)}
    }).getCommandNodes(graph);
    const record = getWGSLKernelNodeRecord(graph, 'harmonic-bound-family');
    expect(record?.declarations).toContain(
      `const EXACT_SUM_MAXIMUM_TERMS: u32 = ${MAXIMUM_MULTIPLE_TESTING_EXACT_HARMONIC_TERMS}u;`
    );
    expect(record?.declarations).toContain('m <= EXACT_SUM_MAXIMUM_TERMS');
  } finally {
    device.destroy();
  }
});

it('rejects destination aliases and empty names before building CPU descriptors (FRM-01)', () => {
  for (const output of [
    {adjustedPValues: 'count', effectiveFamilySize: 'count'},
    {adjustedPValues: ''},
    {adjustedPValues: 'probabilities'}
  ]) {
    expect(
      () =>
        new CPUMultipleTestingAdjustmentOperation({
          id: 'alias-adjust',
          pValues: 'probabilities',
          rowCount: 2,
          method: 'bonferroni',
          output
        })
    ).toThrow(/alias-adjust/);
  }
});

it('captures the method, sizes and destinations at construction (FRM-04)', async () => {
  const props = {
    pValues: 'probabilities',
    rowCount: 2,
    method: 'bonferroni' as 'bonferroni' | 'none',
    minimumFamilySize: 0,
    output: {adjustedPValues: 'adjusted', effectiveFamilySize: 'family'}
  };
  const operation = new CPUMultipleTestingAdjustmentOperation(props);
  props.method = 'none';
  props.output.adjustedPValues = 'redirected';
  props.output.effectiveFamilySize = 'redirected-count';
  props.minimumFamilySize = 10;
  props.rowCount = 3;
  const pValues = Float32Array.of(0.02, 0.2);
  const outputs = await operation.execute({
    inputs: {
      pValues: {
        format: 'float32',
        length: 2,
        chunks: [{values: pValues, length: 2}],
        pack: () => pValues
      }
    }
  });
  // Bonferroni with two tests doubles each probability, independent of later props edits.
  expect(outputs).toEqual({adjusted: Float32Array.of(0.04, 0.4), family: Uint32Array.of(2)});
});

it('separates observed tests from the effective family and publishes each count independently (CMP-14)', async () => {
  const pValues = Float32Array.of(0.02, NaN);
  const result = computeMultipleTestingAdjustmentOnCPU(pValues, 'bonferroni', 10);
  // One admissible observed p-value; nine implicit p=1 rows pad the denominator to ten.
  expect(result.finiteTestCount).toEqual(Uint32Array.of(1));
  expect(result.effectiveFamilySize).toEqual(Uint32Array.of(10));
  expect(result.adjustedPValues).toHaveLength(2);
  expect(Number.isFinite(result.adjustedPValues[0])).toBe(true);
  expect(result.adjustedPValues[0]).toBeCloseTo(0.2, 7);
  expect(result.adjustedPValues[1]).toBeNaN();
  for (const [port, expected] of [
    ['finiteTestCount', 1],
    ['effectiveFamilySize', 10]
  ] as const) {
    const props = {
      pValues: 'probabilities',
      rowCount: 2,
      method: 'bonferroni' as const,
      minimumFamilySize: 10,
      output: {adjustedPValues: 'adjusted', [port]: 'count'}
    };
    const operation = new CPUMultipleTestingAdjustmentOperation(props);
    expect(operation.contract.outputs).toEqual(
      new GPUMultipleTestingAdjustmentOperation(props).contract.outputs
    );
    const outputs = await operation.execute({
      inputs: {
        pValues: {
          format: 'float32',
          length: 2,
          chunks: [{values: pValues, length: 2}],
          pack: () => pValues
        }
      }
    });
    expect(outputs['count']).toEqual(Uint32Array.of(expected));
  }
});

it('allocates only observed rows for a family above the GPU size boundary', async () => {
  const OriginalFloat64Array = Float64Array;
  vi.stubGlobal(
    'Float64Array',
    new Proxy(OriginalFloat64Array, {
      construct(target, argumentsList) {
        if (typeof argumentsList[0] === 'number' && argumentsList[0] > 16) {
          throw new Error(`logical family allocation: ${argumentsList[0]} rows`);
        }
        return Reflect.construct(target, argumentsList);
      }
    })
  );
  try {
    // Independent SciPy digamma(m+1) + Euler-gamma fixtures; no family-sized reference loops.
    for (const [familySize, harmonic] of [
      [2 ** 31 - 1, 22.06477826202701],
      [2 ** 31, 22.06477826249267],
      [2 ** 31 + 1, 22.06477826295833],
      [0xffffffff, 22.757925442703367]
    ]) {
      for (const method of [
        'none',
        'bonferroni',
        'holm',
        'benjamini-hochberg',
        'benjamini-yekutieli'
      ] as const) {
        const result = computeMultipleTestingAdjustmentOnCPU(
          Float32Array.of(2 ** -40),
          method,
          familySize
        );
        const expected =
          2 ** -40 *
          (method === 'none' ? 1 : familySize) *
          (method === 'benjamini-yekutieli' ? harmonic : 1);
        expect(result.adjustedPValues).toEqual(Float32Array.of(expected));
        expect(result.effectiveFamilySize[0]).toBe(familySize);
        const pValues = Float32Array.of(2 ** -40);
        const operation = new CPUMultipleTestingAdjustmentOperation({
          pValues: 'probabilities',
          rowCount: 1,
          method,
          minimumFamilySize: familySize,
          output: {adjustedPValues: 'adjusted', effectiveFamilySize: 'family'}
        });
        const outputs = await operation.execute({
          inputs: {
            pValues: {
              format: 'float32',
              length: 1,
              chunks: [{values: pValues, length: 1}],
              pack: () => pValues
            }
          }
        });
        expect(outputs).toEqual({
          adjusted: Float32Array.of(expected),
          family: Uint32Array.of(familySize)
        });
      }
    }
  } finally {
    vi.unstubAllGlobals();
  }
});

it('rejects unknown methods at construction with the accepted values in both forms', () => {
  for (const Operation of [
    CPUMultipleTestingAdjustmentOperation,
    GPUMultipleTestingAdjustmentOperation
  ]) {
    expect(
      () =>
        new Operation({
          id: 'gpu-adjust',
          pValues: 'pValues',
          rowCount: 3,
          method: 'invalid' as 'holm',
          output: {adjustedPValues: 'adjusted'}
        })
    ).toThrow(
      /gpu-adjust method.*none.*bonferroni.*holm.*benjamini-hochberg.*benjamini-yekutieli.*invalid/
    );
  }
});

it('matches independent statsmodels fixtures, including missing rows and pinned families', () => {
  let maximumDoubleDifference = 0;
  let maximumFloatDifference = 0;
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
      const doubleResult = getMultipleTestingAdjustedPValues(
        fixture.pValues,
        method,
        fixture.minimumFamilySize
      );
      result.adjustedPValues.forEach((value, row) => {
        const expected = fixture.expected[method][row];
        if (Number.isNaN(expected)) {
          expect(value).toBeNaN();
        } else {
          expect(Math.abs(value - expected)).toBeLessThan(2e-6);
          maximumFloatDifference = Math.max(maximumFloatDifference, Math.abs(value - expected));
          maximumDoubleDifference = Math.max(
            maximumDoubleDifference,
            Math.abs(doubleResult[row] - expected)
          );
        }
      });
      const tested = fixture.pValues.filter(
        value => Number.isFinite(value) && value >= 0 && value <= 1
      ).length;
      expect(result.effectiveFamilySize[0]).toBe(Math.max(tested, fixture.minimumFamilySize));
    }
  }
  expect(maximumDoubleDifference).toBeLessThanOrEqual(2e-12);
  console.info('Multiple-testing audit maximum absolute differences', {
    adjustedFloat64: maximumDoubleDifference,
    adjustedFloat32: maximumFloatDifference,
    effectiveFamilySize: 0
  });
});

it('none preserves tested values, normalizes negative zero and excludes invalid values', () => {
  const result = computeMultipleTestingAdjustmentOnCPU(
    Float32Array.of(-0, 1, NaN, -1, Infinity),
    'none',
    8
  );
  expect(Array.from(result.adjustedPValues)).toEqual([0, 1, NaN, NaN, NaN]);
  expect(result.effectiveFamilySize[0]).toBe(8);
  expect(
    computeMultipleTestingAdjustmentOnCPU(Float32Array.of(0.25), 'none', 2 ** 24)
      .effectiveFamilySize[0]
  ).toBe(2 ** 24);
  expect(
    computeMultipleTestingAdjustmentOnCPU(new Float32Array(), 'holm').effectiveFamilySize[0]
  ).toBe(0);
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
    expect(result.effectiveFamilySize[0]).toBe(80);
  }
});

it('validates construction and declares identical graph outputs', () => {
  const props = {
    id: 'adjust',
    pValues: 'pValues',
    rowCount: 3,
    method: 'holm' as const,
    output: {adjustedPValues: 'adjusted', effectiveFamilySize: 'tested'}
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
    /adjust rowCount/
  );
  expect(new CPUMultipleTestingAdjustmentOperation(props).contract.outputs).toEqual(
    new GPUMultipleTestingAdjustmentOperation(props).contract.outputs
  );
});
