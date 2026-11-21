// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// TOOL-109: mutation checks of the comparison helpers inside headless specs. Each helper is parsed
// out of its spec and called on a supplied result, so NaN, missing and extra elements are shown to
// fail without acquiring a device or registering the headless cases here. The network statistics
// comparison is checked in `network-statistics-reference.node.spec.ts`.

import {readFileSync} from 'node:fs';
import typescript from 'typescript';
import {expect, it} from 'vitest';
import {DENSE_SOLVE_STATUS} from '../../src/analysis-core/kernels/dense-solve-wgsl';
import {expectNumericArrayClose} from '../../src/testing/expect-numeric-array-close';

type SpecHelper = (...parameters: never[]) => unknown;

/**
 * Transpiles the named top-level functions and constants of a spec and returns the functions.
 * `dependencies` supplies the imported symbols they use.
 */
function loadSpecHelpers(
  relativePath: string,
  names: string[],
  dependencies: Record<string, unknown> = {}
): Record<string, SpecHelper> {
  const source = readFileSync(new URL(relativePath, import.meta.url), 'utf8');
  const parsed = typescript.createSourceFile(
    'helpers.ts',
    source,
    typescript.ScriptTarget.Latest,
    true
  );
  const texts = names.map(name => {
    const statement = parsed.statements.find(
      candidate =>
        (typescript.isFunctionDeclaration(candidate) && candidate.name?.text === name) ||
        (typescript.isVariableStatement(candidate) &&
          candidate.declarationList.declarations.some(
            declaration =>
              typescript.isIdentifier(declaration.name) && declaration.name.text === name
          ))
    );
    if (!statement) {
      throw new Error(`${relativePath} has no top-level ${name}`);
    }
    return statement.getText(parsed);
  });
  const compiled = typescript.transpileModule(texts.join('\n'), {
    compilerOptions: {target: typescript.ScriptTarget.ES2022}
  }).outputText;
  const dependencyNames = Object.keys(dependencies);
  return new Function(...dependencyNames, `${compiled}; return {${names.join(', ')}};`)(
    ...dependencyNames.map(name => dependencies[name])
  );
}

it('the Vincenty final bearing comparison rejects NaN, missing and extra bearings', () => {
  const {getFinalBearingFailures} = loadSpecHelpers(
    '../gpu-spatial-analysis/geometry-measures/geodesic-direct-final-bearing.spec.ts',
    ['getBearingError', 'getFinalBearingFailures']
  ) as {getFinalBearingFailures: (...parameters: unknown[]) => string[]};
  expect(getFinalBearingFailures([90], [90], ['east'])).toEqual([]);
  // The 1e-3 degree budget still wraps across 0/360.
  expect(getFinalBearingFailures([359.9995], [0], ['north'])).toEqual([]);
  expect(getFinalBearingFailures([NaN], [90], ['east'])).not.toEqual([]);
  expect(getFinalBearingFailures([], [90], ['east'])).not.toEqual([]);
  expect(getFinalBearingFailures([90, 90], [90], ['east'])).not.toEqual([]);
  expect(getFinalBearingFailures([90.002], [90], ['east'])).not.toEqual([]);
});

it('the planar heading comparison rejects a NaN northward heading, missing and extra rows', () => {
  const {getHeadingFailures} = loadSpecHelpers(
    '../gpu-spatial-analysis/trajectory-analysis/trajectory-heading-signed-zero.spec.ts',
    ['getHeadingFailures']
  ) as {getHeadingFailures: (...parameters: unknown[]) => string[]};
  const positions = [
    [0, 0],
    [-0, 1]
  ];
  // Due north from x = +0 to x = -0: atan2(1, +0) = pi / 2.
  expect(getHeadingFailures(positions, [0, Math.PI / 2])).toEqual([]);
  expect(getHeadingFailures(positions, [0, NaN])).not.toEqual([]);
  expect(getHeadingFailures(positions, [0])).not.toEqual([]);
  expect(getHeadingFailures(positions, [0, Math.PI / 2, 0])).not.toEqual([]);
  expect(getHeadingFailures(positions, [0, Math.PI / 2 + 2e-6])).not.toEqual([]);
});

it('the H3 unit-vector longitude/latitude comparison rejects NaN, missing and extra points', () => {
  const {getUnitVectorLngLatMismatches} = loadSpecHelpers(
    '../gpu-spatial-analysis/cell-indexing/h3-boundary-wgsl.spec.ts',
    ['MAXIMUM_DEGREE_ERROR', 'getUnitVectorLngLatMismatches']
  ) as {getUnitVectorLngLatMismatches: (...parameters: unknown[]) => string[]};
  // [0, 1, 0] is longitude 90, latitude 0.
  const vectors = [[0, 1, 0]];
  expect(getUnitVectorLngLatMismatches(vectors, [90, 0])).toEqual([]);
  expect(getUnitVectorLngLatMismatches(vectors, [90 + 2e-5, 0])).toEqual([]);
  expect(getUnitVectorLngLatMismatches(vectors, [NaN, NaN])).not.toEqual([]);
  expect(getUnitVectorLngLatMismatches(vectors, [90, NaN])).not.toEqual([]);
  expect(getUnitVectorLngLatMismatches(vectors, [])).not.toEqual([]);
  expect(getUnitVectorLngLatMismatches(vectors, [90, 0, 0])).not.toEqual([]);
  expect(getUnitVectorLngLatMismatches(vectors, [90 + 1e-4, 0])).not.toEqual([]);
});

it('the Lee L distribution comparison rejects a NaN draw, missing and extra draws', () => {
  const {expectLeeLDistributionMatchesTwin} = loadSpecHelpers(
    '../gpu-spatial-analysis/permutation-inference/new-statistics.spec.ts',
    ['expectLeeLDistributionMatchesTwin'],
    {expectNumericArrayClose}
  ) as {expectLeeLDistributionMatchesTwin: (...parameters: unknown[]) => void};
  expectLeeLDistributionMatchesTwin([0.5, 0.25 + 0.5e-4], [0.5, 0.25]);
  expect(() => expectLeeLDistributionMatchesTwin([NaN, 0.25], [0.5, 0.25])).toThrow();
  expect(() => expectLeeLDistributionMatchesTwin([0.5], [0.5, 0.25])).toThrow();
  expect(() => expectLeeLDistributionMatchesTwin([0.5, 0.25, 0], [0.5, 0.25])).toThrow();
  expect(() => expectLeeLDistributionMatchesTwin([0.5, 0.25 + 2e-4], [0.5, 0.25])).toThrow();
});

it('the symmetric eigen comparison rejects NaN eigenpairs, missing and extra values', () => {
  const {getEigenMismatches} = loadSpecHelpers(
    './symmetric-eigen-wgsl.spec.ts',
    ['getReferenceEigenvalues', 'getEigenMismatches'],
    {DENSE_SOLVE_STATUS}
  ) as {getEigenMismatches: (...parameters: unknown[]) => string[]};
  const cases = [
    {
      matrix: [
        [2, 0],
        [0, 3]
      ],
      maximumSweeps: 16
    }
  ];
  const identity = [
    [1, 0],
    [0, 1]
  ];
  const result = (values: number[], vectors: number[][]) => [
    {status: DENSE_SOLVE_STATUS.OK, sweeps: 0, values, vectors}
  ];
  expect(getEigenMismatches(cases, result([2, 3], identity))).toEqual([]);
  expect(
    getEigenMismatches(
      cases,
      result(
        [NaN, NaN],
        [
          [NaN, NaN],
          [NaN, NaN]
        ]
      )
    )
  ).not.toEqual([]);
  expect(getEigenMismatches(cases, result([2, NaN], identity))).not.toEqual([]);
  expect(
    getEigenMismatches(
      cases,
      result(
        [2, 3],
        [
          [1, 0],
          [0, NaN]
        ]
      )
    )
  ).not.toEqual([]);
  expect(getEigenMismatches(cases, result([2], identity))).not.toEqual([]);
  expect(getEigenMismatches(cases, result([2, 3, 4], identity))).not.toEqual([]);
  expect(getEigenMismatches(cases, result([2, 3], [[1, 0]]))).not.toEqual([]);
  expect(getEigenMismatches(cases, [])).not.toEqual([]);
  expect(getEigenMismatches(cases, result([2, 3 + 1e-4], identity))).not.toEqual([]);
});

it('the dense solve comparison rejects a NaN solution, inverse or log-determinant on an OK status (CORE-117)', () => {
  const {getSolutionMismatches} = loadSpecHelpers(
    './dense-solve-wgsl.spec.ts',
    ['solveReference', 'getRelativeError', 'getSolutionMismatches'],
    {DENSE_SOLVE_STATUS}
  ) as {getSolutionMismatches: (...parameters: unknown[]) => string[]};
  // Hand-solved: A = [[0, 2, 1], [1, 1, 0], [3, 0, 1]], b = [1, 2, 3] gives x = [6/5, 4/5, -3/5],
  // det A = -5 and inverse [[-1, 2, 1], [1, 3, -1], [3, -6, 2]] / 5 (NumPy agrees).
  const cases = [
    {
      name: 'hand 3x3',
      matrix: [
        [0, 2, 1],
        [1, 1, 0],
        [3, 0, 1]
      ],
      vector: [1, 2, 3],
      mode: 0,
      minimumPivotRatio: 1e-6,
      expectedStatus: DENSE_SOLVE_STATUS.OK
    }
  ];
  const inverse = [
    [-1, 2, 1],
    [1, 3, -1],
    [3, -6, 2]
  ].map(row => row.map(value => value / 5));
  const result = (
    changes: Partial<Record<'log' | 'sign', number>> & {
      solution?: number[];
      inverse?: number[][];
    } = {}
  ) => [
    {
      status: DENSE_SOLVE_STATUS.OK,
      log: Math.log(5),
      sign: -1,
      solution: [6 / 5, 4 / 5, -3 / 5],
      inverse,
      ...changes
    }
  ];
  expect(getSolutionMismatches(cases, result(), 1e-6)).toEqual([]);
  expect(getSolutionMismatches(cases, result({solution: [NaN, 4 / 5, -3 / 5]}), 1e-6)).not.toEqual(
    []
  );
  expect(
    getSolutionMismatches(cases, result({inverse: [inverse[0], inverse[1], [NaN, NaN, NaN]]}), 1e-6)
  ).not.toEqual([]);
  expect(getSolutionMismatches(cases, result({log: NaN}), 1e-6)).not.toEqual([]);
  expect(getSolutionMismatches(cases, result({log: Infinity}), 1e-6)).not.toEqual([]);
  expect(getSolutionMismatches(cases, result({solution: [6 / 5, 4 / 5]}), 1e-6)).not.toEqual([]);
  expect(getSolutionMismatches(cases, result({solution: [6 / 5, 4 / 5, -0.59]}), 1e-6)).not.toEqual(
    []
  );
  expect(getSolutionMismatches(cases, [], 1e-6)).not.toEqual([]);
  // A non-OK status is checked by checkStatuses, not here.
  expect(
    getSolutionMismatches(
      cases,
      [{...result({solution: [NaN, NaN, NaN]})[0], status: DENSE_SOLVE_STATUS.NONFINITE}],
      1e-6
    )
  ).toEqual([]);
});

it('the float compensation classification accepts an erasing or a preserving compiler and nothing else (CORE-120)', () => {
  const {getFloatCompensationBehaviour} = loadSpecHelpers(
    './addition-rounding-error-wgsl.spec.ts',
    ['getFloatCompensationBehaviour']
  ) as {getFloatCompensationBehaviour: (...parameters: unknown[]) => string};
  // True sum 2000; the IEEE serial running sum is 1990 and its correction 9.
  const exact = 2000;
  const serial = 1990;
  // Erased (Metal fast math): correction 0 and the plain sum, far from the true sum.
  expect(
    getFloatCompensationBehaviour({plain: 100, floatSum: 100, floatCompensation: 0}, exact, serial)
  ).toBe('erased');
  // Preserved (IEEE as written): the serial sum plus a correction within 4 of the true sum.
  expect(
    getFloatCompensationBehaviour({plain: 100, floatSum: 1990, floatCompensation: 9}, exact, serial)
  ).toBe('preserved');
  // A partly folded, non-finite or wrong correction is neither.
  for (const result of [
    {plain: 100, floatSum: 1990, floatCompensation: 500},
    {plain: 100, floatSum: 1990, floatCompensation: NaN},
    {plain: 100, floatSum: 100, floatCompensation: 1900},
    {plain: 1999, floatSum: 1999, floatCompensation: 0}
  ]) {
    expect(getFloatCompensationBehaviour(result, exact, serial)).toMatch(/^neither/);
  }
});
