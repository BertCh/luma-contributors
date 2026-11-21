// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {join} from 'node:path';
import * as core from '@luma.gl/core';
import * as graph from '@luma.gl/graph/gpu-core';
import typescript from 'typescript';
import * as spatial from '../../../src/gpu-spatial-analysis/index';
import {decodeGPUGroupSum} from '../../../src/dataframe-analysis/index';
import {GPUParameterBuffer} from '../../../src/analysis-core/index';
import {DOCS_ROOT} from '../../analysis-core/doc-program';
import {extractSnippets} from '../../analysis-core/doc-snippet-diagnostics.test-utils';

export function readGuide(name: string): string {
  return readFileSync(join(DOCS_ROOT, 'gpu-spatial-analysis', `${name}.md`), 'utf8');
}

/** Execute the published fence itself, so a guide-only regression cannot evade declaration tests. */
export function loadGuideFunction<FunctionType>(page: string, functionName: string): FunctionType {
  const snippets = extractSnippets(`${page}.md`, readGuide(page));
  const matches = snippets.filter(snippet => snippet.code.includes(`function ${functionName}(`));
  if (matches.length !== 1) throw new Error(`${page}: expected one ${functionName} fence`);
  const javascript = typescript.transpileModule(matches[0].code, {
    compilerOptions: {
      module: typescript.ModuleKind.CommonJS,
      target: typescript.ScriptTarget.ES2022
    }
  }).outputText;
  const modules: Record<string, unknown> = {
    '@luma.gl/core': core,
    '@luma.gl/graph/gpu-core': graph,
    '@luma-contributors/analysis/gpu-spatial-analysis': spatial,
    '@luma-contributors/analysis/dataframe-analysis': {decodeGPUGroupSum},
    '@luma-contributors/analysis/analysis-core': {GPUParameterBuffer}
  };
  const exported: Record<string, unknown> = {};
  new Function('require', 'exports', javascript)((name: string) => {
    if (!(name in modules)) throw new Error(`Unexpected guide import ${name}`);
    return modules[name];
  }, exported);
  if (typeof exported[functionName] !== 'function') throw new Error(`Missing ${functionName}`);
  return exported[functionName] as FunctionType;
}
