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
  getFirstRunFences,
  getPairedFirstRunFences,
  readDocPage
} from './doc-program';
import {getSnippetDiagnostics} from './doc-snippet-diagnostics.test-utils';
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
  const familyImports = ['terrain', 'distance', 'time', 'pointcloud'].flatMap(name =>
    getImports(`family-first-run-${name}.test-utils.ts`)
  );
  const familyEntryImports = getImports('family-first-run.spec.ts');
  expect(familyImports.filter(specifier => !familyEntryImports.includes(specifier))).toEqual([]);
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
  const unpaired = getDiagnostics(
    new Map(
      getFirstRunFences()
        .slice(1)
        .map((fence, index) => [`unpaired-${index}.ts`, fence.code])
    )
  );
  expect(unpaired.filter(error => error.code === 7031)).toHaveLength(11);
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

function getPointDensityFence(markdown: string): string {
  const section = markdown.split('## Point density from device to readback')[1]?.split('\n## ')[0];
  expect(section, 'point-density section').toBeDefined();
  const fences = [...section!.matchAll(/^```ts\n([\s\S]*?)^```/gm)];
  expect(fences, 'exactly one displayed GPU fence').toHaveLength(1);
  return fences[0][1].trim();
}

it('FST-11 the GPU source equals the single displayed fence and its spec has no test-device import', () => {
  const fence = getPointDensityFence(readDocPage('operation-forms-and-cpu-twins.md'));
  const source = readFileSync(
    join(
      REPOSITORY_ROOT,
      'modules/analysis/test/analysis-core/point-density-first-run.test-utils.ts'
    ),
    'utf8'
  )
    .split('// BEGIN LITERAL FENCES\n')[1]
    ?.split('\n// END LITERAL FENCES')[0]
    .trim();
  expect(source).toBe(fence);
  const spec = readFileSync(
    join(REPOSITORY_ROOT, 'modules/analysis/test/analysis-core/point-density-first-run.spec.ts'),
    'utf8'
  );
  expect(spec).not.toContain('@luma.gl/core/test-utils');
});

it('FST-11 rejects an accidental split of the displayed GPU fence', () => {
  const original = readDocPage('operation-forms-and-cpu-twins.md');
  const changed = original.replace(
    'export async function runPointDensityExample()',
    '```\n\n```ts\nexport async function runPointDensityExample()'
  );
  expect(changed).not.toBe(original);
  expect(() => getPointDensityFence(changed)).toThrow('exactly one displayed GPU fence');
});

it('DOC-111 pairs the literal shared lifecycle with every invoked README example', () => {
  const paired = getPairedFirstRunFences();
  expect(paired).toHaveLength(4);
  expect(
    getDiagnostics(new Map(paired.map((snippet, index) => [`paired-${index}.ts`, snippet.code])))
  ).toEqual([]);
  for (const [index, name] of ['terrain', 'distance', 'time', 'pointcloud'].entries()) {
    const source = readFileSync(
      join(
        REPOSITORY_ROOT,
        `modules/analysis/test/analysis-core/family-first-run-${name}.test-utils.ts`
      ),
      'utf8'
    );
    const literal = source
      .split('// BEGIN LITERAL FENCES\n')[1]
      ?.split('\n// END LITERAL FENCES')[0];
    expect(literal, name).toBe(paired[index].code);
  }
  const controlSource = readFileSync(
    join(
      REPOSITORY_ROOT,
      'modules/analysis/test/analysis-core/family-first-run-time-control.test-utils.ts'
    ),
    'utf8'
  );
  const controlLiteral = controlSource
    .split('// BEGIN LITERAL FENCES\n')[1]
    ?.split('\n// END LITERAL FENCES')[0];
  const changed = paired[2].code.replace(
    'start: (t + 1000) - t, end: (t + 1000) - t',
    'start: (t + 1001) - t, end: (t + 1999) - t'
  );
  expect(changed).not.toBe(paired[2].code);
  expect(controlLiteral).toBe(changed.replace('// [1], 1, 1, 0', '// [], 0, 0, 0'));
  expect(controlLiteral).not.toBe(paired[2].code);
}, 120_000);

it('DOC-111 rejects an invalid typed prop and attributes both halves to their Markdown lines', () => {
  const original = readDocPage('README.md');
  const invalid = original.replace("mode: 'exact'", "mode: 'misspelled'");
  expect(invalid).not.toBe(original);
  const paired = getPairedFirstRunFences(invalid);
  const snippet = paired[1];
  const errors = getSnippetDiagnostics([snippet]);
  const expectedExampleLine =
    invalid.split('\n').findIndex(line => line.includes("mode: 'misspelled'")) + 1;
  expect(errors).toContainEqual(
    expect.objectContaining({page: 'README.md', line: expectedExampleLine, code: 2322})
  );
  const sharedInvalid = original.replace('buffers.push(buffer);', 'buffers.push(123);');
  const sharedErrors = getSnippetDiagnostics([getPairedFirstRunFences(sharedInvalid)[0]]);
  const expectedSharedLine =
    sharedInvalid.split('\n').findIndex(line => line.includes('buffers.push(123)')) + 1;
  expect(sharedErrors).toContainEqual(
    expect.objectContaining({page: 'README.md', line: expectedSharedLine, code: 2345})
  );
  // Independent inclusive-window calculation in milliseconds relative to Date.UTC(2026,10,19).
  expect(
    [0, 1000, 2000].flatMap((time, index) => (time >= 1000 && time <= 1000 ? [index] : []))
  ).toEqual([1]);
  expect(
    [0, 1000, 2000].flatMap((time, index) => (time >= 1001 && time <= 1999 ? [index] : []))
  ).toEqual([]);
  // 1750 is inside the requested negative-control interval but outside the old [1001,1500].
  expect([1750].filter(time => time >= 1001 && time <= 1999)).toEqual([1750]);
  expect([1750].filter(time => time >= 1001 && time <= 1500)).toEqual([]);
}, 120_000);

function getDisplayedPointCloudConsumer(displayed: unknown[][], onFreshCall: () => void) {
  const callbacks: {name: string; action: () => Promise<void>}[] = [];
  const source = readFileSync(
    join(REPOSITORY_ROOT, 'modules/analysis/test/analysis-core/family-first-run.spec.ts'),
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
    navigator: {gpu: {}},
    require: (specifier: string) => {
      if (specifier === 'vitest')
        return {
          expect,
          it: (name: string, action: () => Promise<void>) => callbacks.push({name, action})
        };
      if (specifier === './family-first-run.test-utils')
        return {
          loadPointCloudExample: async () => ({
            displayedPointCloudResults: displayed,
            async runPointCloudStatisticalFilterExample(multiplier: number) {
              onFreshCall();
              return {
                meanDistances: Float32Array.of(1, 1, 10, -1),
                validity: Uint32Array.from(multiplier === 0 ? [1, 1, 0, 0] : [1, 1, 1, 0])
              };
            }
          })
        };
      if (specifier.includes('require-test-device')) return {requireTestDevice: () => {}};
      return {};
    }
  });
  return callbacks.find(callback => callback.name.startsWith('COMMON point-cloud'))!.action;
}

type DisplayedTimeWindow = {
  rows: Uint32Array;
  count: number;
  requiredCount: number;
  overflow: number;
};

async function checkDisplayedTimeResults(
  positive: DisplayedTimeWindow,
  negative: DisplayedTimeWindow
) {
  const callbacks: {name: string; action: () => Promise<void>}[] = [];
  const source = readFileSync(
    join(REPOSITORY_ROOT, 'modules/analysis/test/analysis-core/family-first-run.spec.ts'),
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
    navigator: {gpu: {}},
    require: (specifier: string) => {
      if (specifier === 'vitest')
        return {
          expect,
          it: (name: string, action: () => Promise<void>) => callbacks.push({name, action})
        };
      if (specifier === './family-first-run.test-utils')
        return {
          loadTimeExample: async () => ({timeWindow: positive}),
          loadChangedTimeExample: async () => ({timeWindow: negative})
        };
      if (specifier.includes('require-test-device')) return {requireTestDevice: () => {}};
      return {};
    }
  });
  await callbacks.find(callback => callback.name.startsWith('COMMON time'))!.action();
}

it('UNI-17 pins displayed positive and negative rows, exact counts and clear overflow', async () => {
  // Inclusive [1000,1000] selects only row 1; [1001,1999] selects none of [0,1000,2000].
  const positive = {rows: Uint32Array.of(1), count: 1, requiredCount: 1, overflow: 0};
  const negative = {rows: new Uint32Array(), count: 0, requiredCount: 0, overflow: 0};
  await checkDisplayedTimeResults(positive, negative);
  // Mutation controls ensure the GPU consumer cannot silently drop any of these assertions.
  for (const key of ['count', 'requiredCount', 'overflow'] as const) {
    await expect(
      checkDisplayedTimeResults({...positive, [key]: positive[key] + 1}, negative)
    ).rejects.toThrow();
    await expect(checkDisplayedTimeResults(positive, {...negative, [key]: 1})).rejects.toThrow();
  }
  await expect(
    checkDisplayedTimeResults({...positive, rows: new Uint32Array()}, negative)
  ).rejects.toThrow();
  await expect(
    checkDisplayedTimeResults(positive, {...negative, rows: Uint32Array.of(1)})
  ).rejects.toThrow();
});

it('REVIEW B8 rejects a bad displayed multiplier-zero result even when fresh calls would pass', async () => {
  // Closed form: supported means 1,1,10 have mean 4 and variance 18; multiplier 0 rejects 10.
  const displayed = [
    [2, [1, 1, 10, -1], [1, 1, 1, 0]],
    [0, [1, 1, 10, -1], [1, 1, 1, 0]],
    [1.5, [1, 1, 10, -1], [1, 1, 1, 0]]
  ];
  await expect(getDisplayedPointCloudConsumer(displayed, () => {})()).rejects.toThrow();
});

it('REVIEW B8 consumes the three displayed results without fresh GPU example calls', async () => {
  const displayed = [
    [2, [1, 1, 10, -1], [1, 1, 1, 0]],
    [0, [1, 1, 10, -1], [1, 1, 0, 0]],
    [1.5, [1, 1, 10, -1], [1, 1, 1, 0]]
  ];
  let freshCalls = 0;
  await getDisplayedPointCloudConsumer(displayed, () => {
    freshCalls++;
  })();
  expect(freshCalls).toBe(0);
});
