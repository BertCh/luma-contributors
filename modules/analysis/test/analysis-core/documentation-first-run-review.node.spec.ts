// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {join} from 'node:path';
import {runInNewContext} from 'node:vm';
import typescript from 'typescript';
import {expect, it} from 'vitest';
import {REPOSITORY_ROOT, readDocPage} from './doc-program';

it.each([
  -1,
  NaN,
  Infinity
])('REVIEW B1 reaches the actual variogram residual assertion for %s', residual => {
  const parameters = new Float32Array(12);
  parameters.set([0.2, 1.5, 2.2, residual], 4);
  const dependencies: Record<string, unknown> = {
    'node:fs': {readFileSync},
    'node:path': {join},
    'node:console': {
      Console: class {
        log() {}
      }
    },
    '@luma-contributors/analysis/gpu-network': {},
    '@luma-contributors/analysis/dataframe-analysis': {
      computeVariogramModelFitOnCPU: () => ({parameters, status: Uint32Array.of(0, 0, 0)}),
      fitVariogramModel: () => ({nugget: 0.2}),
      fitVariogramModelWithDiagnostics: () => ({
        model: {nugget: 0.2},
        atBoundary: false,
        logRangeBracket: 0,
        converged: true
      })
    },
    typescript,
    './first-run-readme-links.test-utils': {},
    './doc-program': {REPOSITORY_ROOT, readDocPage}
  };
  const tests: {name: string; action: () => unknown}[] = [];
  dependencies['vitest'] = {
    expect,
    it: (name: string, action: () => unknown) => tests.push({name, action})
  };
  const source = readFileSync(
    join(REPOSITORY_ROOT, 'modules/analysis/test/analysis-core/first-run-docs.node.spec.ts'),
    'utf8'
  );
  const compiled = typescript.transpileModule(source, {
    compilerOptions: {
      module: typescript.ModuleKind.CommonJS,
      target: typescript.ScriptTarget.ES2022
    }
  }).outputText;
  runInNewContext(compiled, {
    exports: {},
    require: (specifier: string) => {
      if (!Object.hasOwn(dependencies, specifier))
        throw new Error(`Unexpected fixture import: ${specifier}`);
      return dependencies[specifier];
    },
    process
  });
  const test = tests.find(candidate => candidate.name.includes('CPU variogram walkthrough'));
  expect(test).toBeDefined();
  expect(() => test!.action()).toThrow(
    residual < 0 ? 'to be greater than or equal to 0' : 'expected false to be true'
  );
});
