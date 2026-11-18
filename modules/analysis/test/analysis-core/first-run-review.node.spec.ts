// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {join} from 'node:path';
import {runInNewContext} from 'node:vm';
import typescript from 'typescript';
import {expect, it} from 'vitest';
import {
  DOCS_ROOT,
  REPOSITORY_ROOT,
  createVirtualProgram,
  formatDiagnostic,
  readDocPage
} from './doc-program';
import {getReadmeLinkIssues} from './first-run-readme-links.test-utils';

const FAMILY_PAGES = ['gpu-terrain/README.md', 'gpu-raster/README.md', 'gpu-dataframe-analysis.md'];

it('the browser entry declares every iframe dependency before Vite starts executing tests', () => {
  const getImports = (name: string) => {
    const path = join(REPOSITORY_ROOT, 'modules/analysis/test/analysis-core', name);
    const source = typescript.createSourceFile(
      path,
      readFileSync(path, 'utf8'),
      typescript.ScriptTarget.Latest,
      true
    );
    return source.statements.flatMap(statement =>
      typescript.isImportDeclaration(statement) &&
      !statement.importClause?.isTypeOnly &&
      typescript.isStringLiteral(statement.moduleSpecifier)
        ? [statement.moduleSpecifier.text]
        : []
    );
  };
  const exampleImports = getImports('point-density-first-run.test-utils.ts');
  const entryImports = getImports('point-density-first-run.spec.ts');
  expect(exampleImports).toContain('@luma.gl/core/webgpu');
  expect(exampleImports.filter(specifier => !entryImports.includes(specifier))).toEqual([]);
});

it('the fresh iframe provides the browser runner import bridge without preparing GPU state', async () => {
  const html = readFileSync(
    join(REPOSITORY_ROOT, 'modules/analysis/test/analysis-core/point-density-first-run.html'),
    'utf8'
  );
  const bootstrap = /<script>\s*([\s\S]*?)<\/script>/.exec(html)?.[1];
  expect(bootstrap, 'fresh-realm import bootstrap before the module script').toBeDefined();
  const realm: {
    __vitest_browser_runner__?: {
      wrapDynamicImport: <Value>(factory: () => Promise<Value>) => Promise<Value>;
    };
  } = {};
  runInNewContext(bootstrap!, realm);
  expect(Object.keys(realm)).toEqual(['__vitest_browser_runner__']);
  const load = realm.__vitest_browser_runner__!.wrapDynamicImport;
  let calls = 0;
  expect(
    await load(async () => {
      calls++;
      return 7;
    })
  ).toBe(7);
  expect(calls).toBe(1);
  const failure = new Error('module failed');
  await expect(load(() => Promise.reject(failure))).rejects.toBe(failure);
  expect(html.indexOf('<script>')).toBeLessThan(html.indexOf('<script type="module">'));
});

it('rejects misspelled README repository paths and heading anchors offline', () => {
  const manifest = JSON.parse(
    readFileSync(join(REPOSITORY_ROOT, 'modules/analysis/package.json'), 'utf8')
  );
  const repository = manifest.repository.url;
  const target = `${repository}/blob/master/docs/api/operation-forms-and-cpu-twins.md`;
  const check = (url: string) =>
    getReadmeLinkIssues(`[first run](${url})`, repository, manifest.files);
  expect(check(`${target}#point-density-from-device-to-readback`)).toEqual([]);
  expect(check(`${target}#point-density-from-device-to-readbak`)[0]).toContain(
    'heading anchor does not exist'
  );
  expect(check(target.replace('.md', '-misspelled.md'))[0]).toContain(
    'repository path does not exist'
  );
  expect(check(target.replace('/luma-contributors/', '/luma-contributorz/'))[0]).toContain(
    'repository or branch'
  );
  expect(check(target.replace('github.com', 'githab.com'))[0]).toContain(
    'unrecognized external origin'
  );
});

function getQuickStartCode(page: string): string {
  const section = readDocPage(page).split('## Quick start')[1]?.split('\n## ')[0];
  const code = /^```ts\n([\s\S]*?)^```/m.exec(section ?? '')?.[1];
  expect(code, `${page}: checked domain-specific construction`).toBeDefined();
  return code!;
}

function getDiagnostics(codes: Map<string, string>) {
  const files = new Map(
    [...codes].map(([name, code]) => [join(DOCS_ROOT, '__doc_snippets__', name), code])
  );
  const program = createVirtualProgram(files);
  return [...files.keys()].flatMap(path => {
    const source = program.getSourceFile(path);
    return [
      ...program.getSyntacticDiagnostics(source),
      ...program.getSemanticDiagnostics(source)
    ].map(diagnostic => ({path, code: diagnostic.code, message: formatDiagnostic(diagnostic)}));
  });
}

it('the former terrain band and parameter props reproduce the published-type errors', () => {
  const errors = getDiagnostics(
    new Map([
      [
        'former-first-run.ts',
        `
import type {Device} from '@luma.gl/core';
import type {GraphDataView} from '@luma.gl/graph';
import type {GPURasterBand} from '@luma.gl/graph/gpu-raster';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
declare const device: Device;
declare const elevationView: GraphDataView<'float32'>;
const elevation: GPURasterBand = {id: 'elevation', format: 'float32', storage: {kind: 'buffer', view: elevationView}};
new GPUParameterBuffer(device, {format: 'float32', length: 8, values: new Float32Array(8)});
`
      ]
    ])
  );
  expect(errors.map(error => error.code)).toEqual(expect.arrayContaining([2353, 2345]));
});

it('family quick starts construct bands and parameter buffers without implicit inputs', () => {
  const codes = new Map(
    FAMILY_PAGES.map((page, index) => [`family-${index}.ts`, getQuickStartCode(page)])
  );
  expect(getDiagnostics(codes)).toEqual([]);
}, 120_000);

it('restoring storage.view or removing parameter ids fails the strict documentation check', () => {
  const terrain = getQuickStartCode(FAMILY_PAGES[0]);
  const incorrectBand = terrain.replace(
    "makeGPURasterBand(elevationView, grid, {id: 'elevation'})",
    "{id: 'elevation', format: 'float32', storage: {kind: 'buffer', view: elevationView}}"
  );
  expect(incorrectBand).not.toBe(terrain);
  const codes = new Map([['incorrect-band.ts', incorrectBand]]);
  for (const [index, page] of FAMILY_PAGES.entries()) {
    const correct = getQuickStartCode(page);
    const withoutId = correct.replace(/^\s+id: '[^']+',\n/m, '\n');
    expect(withoutId).not.toBe(correct);
    codes.set(`missing-id-${index}.ts`, withoutId);
  }
  const errors = getDiagnostics(codes);
  expect(errors.some(error => error.path.endsWith('incorrect-band.ts'))).toBe(true);
  for (let index = 0; index < FAMILY_PAGES.length; index++) {
    expect(
      errors.some(error => error.path.endsWith(`missing-id-${index}.ts`) && error.code === 2345)
    ).toBe(true);
  }
}, 120_000);

it('the GPU source includes both displayed fences and its spec has no test-device import', () => {
  const section = readDocPage('operation-forms-and-cpu-twins.md').split(
    '## Point density from device to readback'
  )[1];
  const fences = [...section.matchAll(/^```ts\n([\s\S]*?)^```/gm)].map(match => match[1].trim());
  const source = readFileSync(
    join(
      REPOSITORY_ROOT,
      'modules/analysis/test/analysis-core/point-density-first-run.test-utils.ts'
    ),
    'utf8'
  )
    .replace(/^\/\/ SPDX-.*\n/gm, '')
    .trim();
  expect(source).toBe(fences.join('\n\n'));
  const spec = readFileSync(
    join(REPOSITORY_ROOT, 'modules/analysis/test/analysis-core/point-density-first-run.spec.ts'),
    'utf8'
  );
  expect(spec).not.toContain('@luma.gl/core/test-utils');
});
