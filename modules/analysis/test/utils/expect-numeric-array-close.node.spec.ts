// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {readFileSync} from 'node:fs';
import typescript from 'typescript';
import {expectNumericArrayClose} from '../../src/testing/expect-numeric-array-close';

it('the load-balanced row-sum assertion rejects a one-row NaN result', () => {
  // Exercise the assertion used by the headless spec without acquiring a device or registering it.
  const source = readFileSync(new URL('./load-balanced-for-each.spec.ts', import.meta.url), 'utf8');
  const parsed = typescript.createSourceFile(
    'assertion.ts',
    source,
    typescript.ScriptTarget.Latest,
    true
  );
  const declaration = parsed.statements.find(
    statement =>
      typescript.isFunctionDeclaration(statement) && statement.name?.text === 'expectRowSums'
  );
  expect(declaration).toBeDefined();
  const compiled = typescript.transpileModule(declaration!.getText(parsed), {
    compilerOptions: {target: typescript.ScriptTarget.ES2022}
  }).outputText;
  const assertRows = new Function(
    'expect',
    'expectNumericArrayClose',
    `${compiled}; return expectRowSums;`
  )(expect, expectNumericArrayClose);
  expect(() => assertRows([NaN], {sums: [1], magnitudes: [1]})).toThrow();
  assertRows([1 + 0.5e-5], {sums: [1], magnitudes: [1]});
  expect(() => assertRows([1 + 2e-5], {sums: [1], magnitudes: [1]})).toThrow();
});

it('the spatial-lag hub row assertion rejects NaN and preserves its magnitude tolerance', () => {
  const source = readFileSync(
    new URL('../gpu-spatial-analysis/spatial-weights/spatial-lag-hub.spec.ts', import.meta.url),
    'utf8'
  );
  const block = source.slice(source.indexOf('  // Tolerance:'), source.indexOf('  // A/B report'));
  expect(block.length).toBeGreaterThan(0);
  const compiled = typescript.transpileModule(block, {
    compilerOptions: {target: typescript.ScriptTarget.ES2022}
  }).outputText;
  const assertRows = new Function(
    'expect',
    'expectNumericArrayClose',
    'balanced',
    'expected',
    'csr',
    'values',
    'ROWS',
    compiled
  );
  const run = (value: number) =>
    assertRows(
      expect,
      expectNumericArrayClose,
      {lag: [value]},
      [1],
      {offsets: [0, 1], weights: [1], neighbors: [0]},
      [1],
      1
    );
  expect(() => run(NaN)).toThrow();
  run(1 + 0.5e-5);
  expect(() => run(1 + 2e-5)).toThrow();
});

it('rejects NaN and missing elements against finite expectations', () => {
  expect(() => expectNumericArrayClose([NaN], [1], {absoluteTolerance: 1e-4})).toThrow();
  expect(() => expectNumericArrayClose([], [1])).toThrow();
  expect(() => expectNumericArrayClose(new Array<number>(1), [1])).toThrow();
});

it('compares nonfinite values explicitly', () => {
  expectNumericArrayClose([NaN, Infinity, -Infinity], [NaN, Infinity, -Infinity]);
  expect(() => expectNumericArrayClose([1], [NaN])).toThrow();
  expect(() => expectNumericArrayClose([Infinity], [-Infinity])).toThrow();
  expect(() => expectNumericArrayClose([Infinity], [1])).toThrow();
});

it('uses an absolute plus expected-relative tolerance and rejects values outside it', () => {
  const options = {absoluteTolerance: 0.25, relativeTolerance: 0.01};
  expectNumericArrayClose([0.25, 101.25], [0, 100], options);
  expect(() => expectNumericArrayClose([101.251], [100], options)).toThrow();
  expectNumericArrayClose([100.1], [100], {absoluteTolerance: value => value / 100});
  expect(() => expectNumericArrayClose([2], [1])).toThrow();
});
