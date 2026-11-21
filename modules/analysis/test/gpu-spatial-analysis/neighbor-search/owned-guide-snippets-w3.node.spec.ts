// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import typescript from 'typescript';
import {getGPUKrigingParameterValues} from '../../../src/gpu-spatial-analysis/spatial-interpolation/kriging-parameters';
import {
  extractSnippets,
  getSnippetDiagnostics
} from '../../analysis-core/doc-snippet-diagnostics.test-utils';

it('DOC-110 declares public imports and typed inputs in both owned guides', () => {
  const snippets = ['operators-density', 'operators-weights'].flatMap(page =>
    extractSnippets(`gpu-spatial-analysis/${page}.md`)
  );
  expect(getSnippetDiagnostics(snippets)).toEqual([]);
}, 120000);

function executeFragment(className: string, includeOptional = false) {
  const page = className === 'GPUMapColoring' ? 'operators-weights' : 'operators-density';
  const snippet = extractSnippets(`gpu-spatial-analysis/${page}.md`).find(fragment =>
    fragment.code.includes(`new ${className}(`)
  )!;
  const parameterBuffers: {
    values?: Float32Array | Uint32Array;
    write: (values: Float32Array | Uint32Array) => void;
    importToGraph: () => unknown;
  }[] = [];
  const gpuModule = new Proxy({getGPUKrigingParameterValues} as Record<string, unknown>, {
    get(target, key: string) {
      if (key in target) return target[key];
      if (key.startsWith('get')) return () => new Float32Array(12);
      return class {
        outputWeights = {};
        constructor(readonly props: unknown) {}
      };
    }
  });
  class ParameterBuffer {
    values?: Float32Array | Uint32Array;
    constructor(_device: unknown, props: {values?: Float32Array | Uint32Array}) {
      this.values = props.values;
      parameterBuffers.push(this);
    }
    write(values: Float32Array | Uint32Array) {
      this.values = values;
    }
    importToGraph() {
      return this;
    }
  }
  const javascript = typescript.transpileModule(snippet.code, {
    compilerOptions: {
      module: typescript.ModuleKind.CommonJS,
      target: typescript.ScriptTarget.ES2022
    }
  }).outputText;
  const inputs = {
    positions: {},
    values: {},
    grid: {width: 1, height: 1, affine: [1, 0, 0, 0, -1, 1]},
    output: {
      values: {},
      positions: {},
      colors: {},
      dots: {ids: {}, status: {count: {}, overflow: {}}},
      ...(includeOptional ? {status: {}} : {})
    }
  };
  new Function(
    'require',
    'exports',
    'inputs',
    'adjacency',
    'graph',
    'device',
    'peoplePerDot',
    'x0',
    'y0',
    'cw',
    'ch',
    'minX',
    'minY',
    'maxX',
    'maxY',
    javascript
  )(
    (name: string) =>
      name.endsWith('/analysis-core') ? {GPUParameterBuffer: ParameterBuffer} : gpuModule,
    {},
    inputs,
    {},
    {add() {}},
    {},
    500,
    0,
    0,
    1,
    -1,
    0,
    0,
    1,
    1
  );
  return parameterBuffers;
}

it('B6 initializes the kriging example with a valid packed variogram', () => {
  const [parameters] = executeFragment('GPUKriging', true);
  expect(parameters.values).toBeDefined();
  expect(parameters.values![5]).toBeGreaterThan(0);
  expect(parameters.values![8]).toBeGreaterThan(0);
  expect(parameters.values![9]).toBeGreaterThan(0);
});

it('B7 runs the guide fragments with optional status and mask ports omitted', () => {
  for (const name of ['GPUKernelDensity', 'GPUKriging', 'GPUDotDensity', 'GPUMapColoring']) {
    expect(() => executeFragment(name), name).not.toThrow();
  }
});
