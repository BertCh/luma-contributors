// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {spawnSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {dirname, join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {afterEach, describe, expect, it} from 'vitest';

const CHECKER_PATH = resolve(
  dirname(fileURLToPath(import.meta.url)),
  '../../../../scripts/check-contributor-contract.mjs'
);
const SOURCE_ROOT = 'modules/analysis/src';
const CONTRIBUTOR_PATH = `${SOURCE_ROOT}/analysis-core/fixture/gpu-fixture.ts`;
const BASELINE_PATH = 'scripts/contributor-contract-baseline.json';
const fixtureDirectories: string[] = [];

type Baseline = Record<string, Record<string, number | string[]>>;

function createFixture(files: Record<string, string>, baseline: Baseline = {}): string {
  const directory = mkdtempSync(join(tmpdir(), 'contributor-contract-checker-'));
  fixtureDirectories.push(directory);
  const fixtureFiles = {
    [`${SOURCE_ROOT}/gpu-spatial-analysis/ontology/index.ts`]: '',
    'modules/analysis/package.json': JSON.stringify({
      exports: Object.fromEntries(
        [
          '.',
          './analysis-core',
          './gpu-network',
          './gpu-terrain',
          './raster-analysis',
          './dataframe-analysis',
          './gpu-spatial-analysis',
          './gpu-spatial-analysis/ontology',
          './pointcloud-analysis',
          './geospatial-analysis',
          './computation-record'
        ].map(entry => [
          entry,
          entry === '.' ? './dist/index.js' : `./dist/${entry.slice(2)}/index.js`
        ])
      )
    }),
    [BASELINE_PATH]: JSON.stringify(baseline),
    ...files
  };
  for (const [file, text] of Object.entries(fixtureFiles)) {
    const filePath = join(directory, file);
    mkdirSync(dirname(filePath), {recursive: true});
    writeFileSync(filePath, text);
  }
  return directory;
}

function runChecker(directory: string, ...arguments_: string[]) {
  const result = spawnSync(process.execPath, [CHECKER_PATH, ...arguments_], {
    cwd: directory,
    encoding: 'utf8',
    timeout: 30_000
  });
  return {exitCode: result.status, output: `${result.stdout}${result.stderr}`};
}

function kernel(body: string, properties = 'invocationCount: 1,'): string {
  return `createWGSLKernelNode(graph, {${properties} body: \`${body}\`});`;
}

afterEach(() => {
  for (const directory of fixtureDirectories.splice(0)) {
    rmSync(directory, {recursive: true, force: true});
  }
});

describe('contributor contract checker', () => {
  it.each([
    '--update-baseline',
    '--migrate-baseline'
  ])('B2 rejects partial %s without erasing reviewed allowances', mode => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: kernel('for(var row=0u;row<count;row++){}')
    });
    expect(
      runChecker(
        directory,
        '--update-baseline',
        '--reason',
        'Reviewed loop, unchanged by selected rules'
      ).exitCode
    ).toBe(0);
    if (mode === '--migrate-baseline')
      writeFileSync(
        join(directory, BASELINE_PATH),
        JSON.stringify({
          'single-invocation-data-loop': {[CONTRIBUTOR_PATH]: 1},
          _reasons: {
            'single-invocation-data-loop': {[CONTRIBUTOR_PATH]: 'Reviewed legacy loop retained'}
          }
        })
      );
    const original = readFileSync(join(directory, BASELINE_PATH), 'utf8');
    expect(JSON.parse(original)['single-invocation-data-loop'][CONTRIBUTOR_PATH]).toBeTruthy();
    const result = runChecker(directory, '--rule', 'bare-assert', mode);
    expect(result.exitCode, result.output).toBe(2);
    expect(result.output).toContain('--rule cannot be combined');
    expect(readFileSync(join(directory, BASELINE_PATH), 'utf8')).toBe(original);
  });

  it('B2 rejects a missing rule value without touching the baseline', () => {
    const directory = createFixture({[CONTRIBUTOR_PATH]: ''});
    const before = readFileSync(join(directory, BASELINE_PATH), 'utf8');
    for (const mode of ['--update-baseline', '--migrate-baseline']) {
      const result = runChecker(directory, mode, '--rule');
      expect(result.exitCode, result.output).toBe(2);
      expect(result.output).toContain('--rule cannot be combined');
      expect(readFileSync(join(directory, BASELINE_PATH), 'utf8')).toBe(before);
    }
  });

  it('B4 invalidates checked bindings after writes and accepts dominating rejecting guards', () => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: `
      function assigned(view, replacement) {
        validateSinglePackedView(view, ['float32'], 'checked');
        view = replacement;
        validateSinglePackedView(view, [view.format], 'assigned');
      }
      function guarded(view) {
        if (view.format !== 'float32') throw new Error('format');
        validateSinglePackedView(view, [view.format], 'guarded');
      }
    `
    });
    const result = runChecker(directory, '--rule', 'self-format-validation');
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toMatch(/^self-format-validation\s+1\s+1$/m);
    const source = readFileSync(join(directory, CONTRIBUTOR_PATH), 'utf8');
    expect(result.output).toContain(
      `${CONTRIBUTOR_PATH}:${source.slice(0, source.indexOf("validateSinglePackedView(view, [view.format], 'assigned')")).split('\n').length}`
    );
  });

  it('B4 accepts checked collections and rejects a replaced member', () => {
    const file = [
      'function layouts(id, inputs, outputs) {',
      ' for (const view of [...inputs, ...outputs]) validateSinglePackedView(view, [view.format], id);',
      '}',
      'function fixed(props: {values: GraphDataView; output: {status: GraphDataView}}) {',
      " for (const [view, format] of [[props.values, 'float32'], [props.output.status, 'uint32']] as const) {",
      '  validateSinglePackedView(view, [format], props.id);',
      ' }',
      ' layouts(props.id, [props.values], Object.values(props.output));',
      '}'
    ].join('\n');
    const directory = createFixture({[CONTRIBUTOR_PATH]: file});
    const valid = runChecker(directory, '--rule', 'self-format-validation');
    expect(valid.exitCode, valid.output).toBe(0);
    writeFileSync(
      join(directory, CONTRIBUTOR_PATH),
      file.replace(' layouts(props.id', ' props.values = replacement;\n layouts(props.id')
    );
    const invalid = runChecker(directory, '--rule', 'self-format-validation');
    expect(invalid.exitCode, invalid.output).toBe(1);
  });

  it('B4 tracks property writes and shadowed bindings independently', () => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: `
      function changed(props, replacement) {
        validateSinglePackedView(props.values, ['float32'], 'old');
        props.values = replacement;
        validateSinglePackedView(props.values, [props.values.format], 'changed');
      }
      function shadowed(view, replacement) {
        validateSinglePackedView(view, ['float32'], 'outer');
        { const view = replacement; validateSinglePackedView(view, [view.format], 'inner'); }
      }
    `
    });
    const result = runChecker(directory, '--rule', 'self-format-validation');
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toMatch(/^self-format-validation\s+2\s+1$/m);
  });

  it('B4 refuses cyclic collection proofs without recursing indefinitely', () => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: `
      function layouts(inputs) {
        for (const view of inputs) validateSinglePackedView(view, [view.format], 'cyclic');
        layouts(inputs);
      }
    `
    });
    const result = runChecker(directory, '--rule', 'self-format-validation');
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain('NEW self-format-validation');
    expect(result.output).not.toContain('RangeError');
  });

  it('B5 treats source-chunk format checks as self checks and keeps independent formats valid', () => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: `
      function unchecked(view) {
        for (const chunk of getGraphViewChunks(view)) validatePackedView(chunk, [view.format], 'unchecked');
      }
      function checked(view) {
        for (const chunk of getGraphViewChunks(view)) validatePackedView(chunk, ['float32'], 'checked');
      }
    `
    });
    const result = runChecker(directory, '--rule', 'self-format-validation');
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toMatch(/^self-format-validation\s+1\s+1$/m);
  });

  it('VAL-01 flags self-format validation at fixed ports without baselining it', () => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: `
      function fixed(props) { validateSinglePackedView(props.values, [props.values.format], 'demo values'); }
    `
    });
    const result = runChecker(directory, '--rule', 'self-format-validation');
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain('NEW self-format-validation');
    expect(result.output).toContain(CONTRIBUTOR_PATH);
    const baseline = runChecker(
      directory,
      '--update-baseline',
      '--reason',
      'cannot excuse fixed ports'
    );
    expect(baseline.exitCode, baseline.output).toBe(1);
    expect(baseline.output).toContain('self-format-validation');
  });

  it('VAL-01 accepts layout helpers only after a real format check on every call', () => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: `
      function layout(view) { validateSinglePackedView(view, [view.format], 'demo values'); }
      function fixed(props) {
        validateSinglePackedView(props.values, ['float32'], 'demo values');
        layout(props.values);
      }
      function dynamic(view) {
        validateRequirement('demo', /^(float32|uint32)$/.test(view.format), 'view format');
        validateSinglePackedView(view, [view.format], 'demo view');
      }
    `
    });
    const result = runChecker(directory, '--rule', 'self-format-validation');
    expect(result.exitCode, result.output).toBe(0);
    writeFileSync(
      join(directory, CONTRIBUTOR_PATH),
      `
      function layout(view) { validateSinglePackedView(view, [view.format], 'demo values'); }
      function fixed(props) {
        validateSinglePackedView(props.values, ['float32'], 'demo values');
        layout(props.values);
      }
      function unchecked(props) { layout(props.values); }
    `
    );
    const unchecked = runChecker(directory, '--rule', 'self-format-validation');
    expect(unchecked.exitCode, unchecked.output).toBe(1);
    expect(unchecked.output).toContain('NEW self-format-validation');
  });

  it('VAL-01 does not confuse a conditional or unrelated check with a format preflight', () => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: `
      function conditional(view, other, enabled) {
        if (enabled) validateSinglePackedView(view, ['float32'], 'demo view');
        validateSinglePackedView(other, ['float32'], 'demo other');
        validateRequirement('demo', true, 'view.format');
        validateSinglePackedView(view, [view.format], 'demo view');
      }
    `
    });
    const result = runChecker(directory, '--rule', 'self-format-validation');
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain('NEW self-format-validation');
  });

  it('keeps catalog admission entry-qualified for shared and same-name declarations (ERG-106)', () => {
    const directory = createFixture({
      'modules/analysis/package.json': JSON.stringify({
        name: '@luma-contributors/analysis',
        exports: {
          './first': './dist/analysis-core/first/index.js',
          './second': './dist/analysis-core/second/index.js'
        }
      }),
      [`${SOURCE_ROOT}/analysis-core/fixture/shared.ts`]: 'export function addSharedRecipe() {}',
      [`${SOURCE_ROOT}/analysis-core/first/index.ts`]:
        "export {addSharedRecipe} from '../fixture/shared';\nexport function addDuplicateRecipe() {}",
      [`${SOURCE_ROOT}/analysis-core/second/index.ts`]:
        "export {addSharedRecipe} from '../fixture/shared';\nexport function addDuplicateRecipe() {}",
      [`${SOURCE_ROOT}/gpu-spatial-analysis/ontology/index.ts`]: `export const ALL_OPERATORS = [
        {package: '@luma-contributors/analysis/first', exportName: 'addSharedRecipe'},
        {package: '@luma-contributors/analysis/first', exportName: 'addDuplicateRecipe'}
      ];`
    });
    const missing = runChecker(directory, '--rule', 'unregistered-contributor');
    expect(missing.exitCode, missing.output).toBe(1);
    expect(missing.output).toContain('(@luma-contributors/analysis/second) addSharedRecipe');
    expect(missing.output).toContain('(@luma-contributors/analysis/second) addDuplicateRecipe');
    // Use literal public records: the catalog evaluator intentionally does not execute callbacks.
    writeFileSync(
      join(directory, SOURCE_ROOT, 'gpu-spatial-analysis/ontology/index.ts'),
      'export const ALL_OPERATORS = ' +
        JSON.stringify(
          ['first', 'second'].flatMap(entry =>
            ['addSharedRecipe', 'addDuplicateRecipe'].map(exportName => ({
              package: '@luma-contributors/analysis/' + entry,
              exportName
            }))
          )
        ) +
        ';'
    );
    const complete = runChecker(directory, '--rule', 'unregistered-contributor');
    expect(complete.exitCode, complete.output).toBe(0);
  });

  it('round 1 [B1] terminates on an external namespace without modifying local exports', () => {
    const directory = createFixture({
      [`${SOURCE_ROOT}/index.ts`]:
        "export const version = 1;\nexport * as arrow from 'apache-arrow';"
    });
    const result = spawnSync(
      process.execPath,
      ['--max-old-space-size=128', CHECKER_PATH, '--rule', 'internal-export'],
      {cwd: directory, encoding: 'utf8', timeout: 2000}
    );
    expect(result.status, `${result.error}\n${result.stdout}${result.stderr}`).toBe(0);
    expect(result.stdout).toContain('internal-export                    0    0');
    // The unrelated public local declaration must keep its own origin as well.
    writeFileSync(
      join(directory, SOURCE_ROOT, 'index.ts'),
      "/** @internal */\nexport const version = 1;\nexport * as arrow from 'apache-arrow';"
    );
    const internal = runChecker(directory, '--rule', 'internal-export');
    expect(internal.output).toContain(' version');
    expect(internal.output).not.toContain('arrow.version');
  });

  it.each([
    false,
    true
  ])('round 1 [B2] registers only emitted records, original spread = %s', spread => {
    const originals = [
      {exportName: 'addSourceRecipe', package: '@luma-contributors/analysis/gpu-spatial-analysis'}
    ];
    const expression = `[${spread ? '...GPU_SPATIAL_ANALYSIS_OPERATORS,' : ''} {...GPU_SPATIAL_ANALYSIS_OPERATORS.find(operator => operator.exportName === 'addSourceRecipe'), package: '@luma-contributors/analysis/analysis-core', exportName: 'addFixtureRecipe'}]`;
    // This same expression is used by JavaScript and by the checker fixture. A find reads one
    // source record; only the explicit spread emits the original spatial identity.
    const actual = new Function('GPU_SPATIAL_ANALYSIS_OPERATORS', `return ${expression};`)(
      originals
    ) as typeof originals;
    expect(actual.map(record => record.exportName)).toEqual(
      spread ? ['addSourceRecipe', 'addFixtureRecipe'] : ['addFixtureRecipe']
    );
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: 'export function addFixtureRecipe() {}',
      [`${SOURCE_ROOT}/gpu-spatial-analysis/fixture/source.ts`]:
        'export function addSourceRecipe() {}',
      [`${SOURCE_ROOT}/gpu-spatial-analysis/index.ts`]:
        "export {addSourceRecipe} from './fixture/source';",
      [`${SOURCE_ROOT}/gpu-spatial-analysis/ontology/index.ts`]: `const GPU_SPATIAL_ANALYSIS_OPERATORS = ${JSON.stringify(originals)}; export const ALL_OPERATORS = ${expression};`
    });
    const result = runChecker(directory, '--rule', 'unregistered-contributor');
    expect(result.exitCode, result.output).toBe(spread ? 0 : 1);
    expect(result.output.includes(' addSourceRecipe')).toBe(!spread);
    expect(result.output).not.toContain(' addFixtureRecipe');
  });

  it('round 1 [B3] ignores private ALL_OPERATORS when the public getter returns an empty array', () => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: 'export function addFixtureRecipe() {}',
      [`${SOURCE_ROOT}/gpu-spatial-analysis/ontology/index.ts`]:
        "const ALL_OPERATORS = [{exportName: 'addFixtureRecipe', package: '@luma-contributors/analysis/analysis-core'}]; export function getGPUAnalysisOperatorCatalog() {return [];}"
    });
    const result = runChecker(directory, '--rule', 'unregistered-contributor');
    expect(result.exitCode, result.output).toBe(1);
  });

  it('round 1 [B3] resolves a record variable emitted by the public catalog', () => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: 'export function addFixtureRecipe() {}',
      [`${SOURCE_ROOT}/gpu-spatial-analysis/ontology/index.ts`]:
        "const operator = {exportName: 'addFixtureRecipe', package: '@luma-contributors/analysis/analysis-core'}; const rows = [operator]; export function getGPUAnalysisOperatorCatalog() {return rows;}"
    });
    const result = runChecker(directory, '--rule', 'unregistered-contributor');
    expect(result.exitCode, result.output).toBe(0);
  });

  it.each([
    null,
    './dist/analysis-core/api/public.js'
  ])('round 1 [B4] honors the winning exact export target %s', target => {
    const directory = createFixture({
      'modules/analysis/package.json': JSON.stringify({
        exports: {'./api/*': './dist/analysis-core/api/*.js', './api/private': target}
      }),
      [`${SOURCE_ROOT}/analysis-core/api/private.ts`]: '/** @internal */\nexport const secret = 1;',
      [`${SOURCE_ROOT}/analysis-core/api/public.ts`]: 'export const visible = 1;'
    });
    const result = runChecker(directory, '--rule', 'internal-export');
    expect(result.exitCode, result.output).toBe(0);
  });

  it('round 1 [B4] honors a more-specific wildcard redirect', () => {
    const directory = createFixture({
      'modules/analysis/package.json': JSON.stringify({
        exports: {
          './api/*': './dist/analysis-core/api/*.js',
          './api/private/*': './dist/analysis-core/public/*.js'
        }
      }),
      [`${SOURCE_ROOT}/analysis-core/api/private/data.ts`]:
        '/** @internal */\nexport const secret = 1;',
      [`${SOURCE_ROOT}/analysis-core/public/data.ts`]: 'export const visible = 1;'
    });
    expect(runChecker(directory, '--rule', 'internal-export').exitCode).toBe(0);
  });

  it('round 1 [B5] resolves a cjs import to cts ahead of a ts sibling', () => {
    const directory = createFixture({
      [`${SOURCE_ROOT}/index.ts`]: "export * as sub from './analysis-core/fixture/data.cjs';",
      [`${SOURCE_ROOT}/analysis-core/fixture/data.ts`]: 'export const visible = 1;',
      [`${SOURCE_ROOT}/analysis-core/fixture/data.cts`]:
        '/** @internal */\nexport const secret = 1;'
    });
    const result = runChecker(directory, '--rule', 'internal-export');
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain('sub.secret');
  });

  it.each([
    'cjs',
    'mjs'
  ])('round 1 [B5] preserves manifest module format for %s targets', extension => {
    const ownExtension = extension === 'cjs' ? 'cts' : 'mts';
    const otherExtension = extension === 'cjs' ? 'mts' : 'cts';
    const directory = createFixture({
      'modules/analysis/package.json': JSON.stringify({
        exports: {'./api': `./dist/analysis-core/fixture/main.${extension}`}
      }),
      [`${SOURCE_ROOT}/analysis-core/fixture/main.${ownExtension}`]: 'export const visible = 1;',
      [`${SOURCE_ROOT}/analysis-core/fixture/main.${otherExtension}`]:
        '/** @internal */\nexport const secret = 1;'
    });
    expect(runChecker(directory, '--rule', 'internal-export').exitCode).toBe(0);
  });

  it('round 1 [B5] checks an authored declaration-only public types target', () => {
    const file = `${SOURCE_ROOT}/analysis-core/fixture/public.d.ts`;
    const directory = createFixture({
      'modules/analysis/package.json': JSON.stringify({
        exports: {'./api': {types: './dist/analysis-core/fixture/public.d.ts'}}
      }),
      [file]: '/** @internal */\nexport declare const secret: number;'
    });
    const result = runChecker(directory, '--rule', 'internal-export');
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain(`NEW internal-export ${file}`);
  });

  it.each([
    '(secret)',
    'api.secret',
    "api['secret']"
  ])('round 1 [B6] follows default expression %s', expression => {
    const directory = createFixture({
      [`${SOURCE_ROOT}/index.ts`]: `import {secret} from './analysis-core/fixture/data'; import * as api from './analysis-core/fixture/data'; export default ${expression};`,
      [`${SOURCE_ROOT}/analysis-core/fixture/data.ts`]: '/** @internal */\nexport const secret = 1;'
    });
    const result = runChecker(directory, '--rule', 'internal-export');
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain(' default');
  });

  it('round 1 [B6] finds the tag on a default function exposed through a namespace', () => {
    const directory = createFixture({
      [`${SOURCE_ROOT}/index.ts`]: "export * as api from './analysis-core/fixture/data';",
      [`${SOURCE_ROOT}/analysis-core/fixture/data.ts`]:
        '/** @internal */\nexport default function reader() {}'
    });
    const result = runChecker(directory, '--rule', 'internal-export');
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain(' api.default');
  });

  it('round 1 [NOTE declaration] keeps a nested internal declaration separate from its public namesake', () => {
    const directory = createFixture({
      [`${SOURCE_ROOT}/index.ts`]:
        'export const foo = 1;\nfunction helper() {\n/** @internal */\nconst foo = 2;\nreturn foo;\n}'
    });
    const result = runChecker(directory, '--rule', 'internal-export');
    expect(result.exitCode, result.output).toBe(0);
  });

  it('round 1 [B7] follows indexed props through a default type import', () => {
    const directory = createFixture({
      [`${SOURCE_ROOT}/analysis-core/index.ts`]:
        "import type Entry from './fixture/data'; export type PublicProps = Entry['visible'];",
      [`${SOURCE_ROOT}/analysis-core/fixture/data.ts`]:
        'export default interface Entry {visible: {\n/** @internal */\nsecret?: number;\n}}'
    });
    const result = runChecker(directory, '--rule', 'internal-export');
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain('PublicProps.secret');
  });

  it.each([
    [
      'method',
      "interface Group {transform(options: Options): number;} export type PublicProps = Group['transform'];",
      true
    ],
    [
      'instance',
      "class Group {a = 1; static a: Options;} export type PublicProps = Group['a'];",
      false
    ],
    ['record', "export type PublicProps = Record<'palette', Options>['palette'];", true]
  ] as const)('round 1 [B8] selects the %s value and its full signature', (_, body, internal) => {
    const directory = createFixture({
      [`${SOURCE_ROOT}/analysis-core/index.ts`]:
        'interface Options {\n/** @internal */\nsecret?: number;\n}\n' + body
    });
    const result = runChecker(directory, '--rule', 'internal-export');
    expect(result.exitCode, result.output).toBe(internal ? 1 : 0);
    if (internal) expect(result.output).toContain('secret');
  });

  it.each([
    [
      "import {assert as check} from '@luma.gl/core'; for (const check of [Boolean]) check(4);",
      false
    ],
    ["import {assert as check} from '@luma.gl/core'; type check = number; check(true);", true],
    ["import * as core from '@luma.gl/core'; core['assert'](true);", true]
  ] as const)('round 1 [B9] resolves value-namespace assert calls: %s', (text, violation) => {
    const result = runChecker(createFixture({[CONTRIBUTOR_PATH]: text}), '--rule', 'bare-assert');
    expect(result.exitCode, result.output).toBe(violation ? 1 : 0);
  });

  it.each([
    [
      "import * as luma from '@luma.gl/core'; const graphics = luma; export default graphics.Device;",
      true
    ],
    [
      "import {Device as DEVICE} from '@luma.gl/core'; export const ANALYSIS_DEVICE = DEVICE;",
      true
    ],
    [
      "import {GPU_FFT2D_MAX_DIMENSION as limit} from '@luma.gl/graph/gpu-core'; export const ANALYSIS_LIMIT = (limit);",
      false
    ]
  ] as const)('round 1 [B10] retains originating names and namespace identity: %s', (text, violation) => {
    const result = runChecker(createFixture({[CONTRIBUTOR_PATH]: text}), '--rule', 'luma-reexport');
    expect(result.exitCode, result.output).toBe(violation ? 1 : 0);
  });

  it('round 1 [NOTE allowance] requires a stated bound rather than arbitrary explanation text', () => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]:
        '// contract-allow: single-invocation -- TODO review later\n' +
        kernel('for(var row=0u;row<rowCount;row++){}')
    });
    const result = runChecker(directory, '--rule', 'single-invocation-data-loop');
    expect(result.exitCode, result.output).toBe(1);
  });
  it('checks a conditional native CommonJS source target (CORE-119)', () => {
    const file = `${SOURCE_ROOT}/analysis-core/nested/entry.cts`;
    const directory = createFixture({
      'modules/analysis/package.json': JSON.stringify({
        exports: {'./nested': {require: './dist/analysis-core/nested/entry.cjs'}}
      }),
      [file]: '/** @internal */\nexport const hidden = 1;'
    });
    const result = runChecker(directory, '--rule', 'internal-export');
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain(`NEW internal-export ${file}`);
  });

  it.each([
    [
      './nested',
      {
        import: {
          types: './dist/analysis-core/nested/entry.d.ts',
          default: './dist/analysis-core/nested/entry.js'
        }
      }
    ],
    ['./nested/*', {import: './dist/analysis-core/*/entry.js'}]
  ])('checks internal exports from manifest target %s (CORE-119)', (entry, target) => {
    const file = `${SOURCE_ROOT}/analysis-core/nested/entry.ts`;
    const directory = createFixture({
      'modules/analysis/package.json': JSON.stringify({exports: {[entry]: target}}),
      [file]: '/** @internal */\nexport const hidden = 1;',
      [`${SOURCE_ROOT}/analysis-core/private/index.ts`]:
        '/** @internal */\nexport const privateControl = 2;'
    });
    const result = runChecker(directory, '--rule', 'internal-export');
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain(`NEW internal-export ${file}`);
    expect(result.output).toContain('hidden');
    expect(result.output).not.toContain('privateControl');
  });

  it.each([
    "import {Device as Renamed} from '@luma.gl/core'; export {Renamed};",
    "import * as core from '@luma.gl/core'; export const Renamed = core.Device;",
    "import {Device} from '@luma.gl/core'; export default Device;",
    "import {Device} from '@luma.gl/core'; const First = Device; const Second = First; export {Second};"
  ])('resolves luma re-export binding origins: %s (CORE-119)', text => {
    const result = runChecker(createFixture({[CONTRIBUTOR_PATH]: text}), '--rule', 'luma-reexport');
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain(`NEW luma-reexport ${CONTRIBUTOR_PATH}`);
  });

  it('permits domain type aliases and named domain limits (CORE-119)', () => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: `
      import type {Device} from '@luma.gl/core';
      import {GPU_FFT2D_MAX_DIMENSION} from '@luma.gl/graph/gpu-core';
      export type FixtureDevice = Device;
      export const MAXIMUM_SPECTRAL_DIMENSION = GPU_FFT2D_MAX_DIMENSION;
    `
    });
    expect(runChecker(directory, '--rule', 'luma-reexport').exitCode).toBe(0);
  });

  it.each([
    "import {assert as check} from '@luma.gl/core';\nconst description = 'check(true)';",
    "import {assert as check} from '@luma.gl/core';\nfunction local(check: (value: boolean) => void) {check(true);}",
    "import {assert as check} from '@luma.gl/core';\nfunction local() {function check(value: boolean) {return value;} check(true);}",
    "import * as core from '@luma.gl/core';\nfunction local(core: {assert(value: boolean): void}) {core.assert(true);}"
  ])('ignores text and shadowed assert bindings: %s (CORE-119)', text => {
    const result = runChecker(createFixture({[CONTRIBUTOR_PATH]: text}), '--rule', 'bare-assert');
    expect(result.exitCode, result.output).toBe(0);
  });

  it('reports the callee source line of a multiline imported assert call (CORE-119)', () => {
    const result = runChecker(
      createFixture({
        [CONTRIBUTOR_PATH]: "import {assert as check} from '@luma.gl/core';\ncheck\n(\n  true\n);"
      }),
      '--rule',
      'bare-assert'
    );
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain(`${CONTRIBUTOR_PATH}:2`);
  });

  it.each([
    'default',
    'namespace'
  ])('follows %s exports to internal and public declarations (TOOL-117)', route => {
    const module = `${SOURCE_ROOT}/analysis-core/fixture/hidden.ts`;
    const index = `${SOURCE_ROOT}/index.ts`;
    const text = '/** @internal */\nexport const hidden = 1;\nexport const visible = 2;';
    const internalRoute =
      route === 'default'
        ? "import {hidden} from './analysis-core/fixture/hidden'; export default hidden;"
        : "export * as exposed from './analysis-core/fixture/hidden';";
    const directory = createFixture({[module]: text, [index]: internalRoute});
    const result = runChecker(directory, '--rule', 'internal-export');
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain(`NEW internal-export ${index}`);
    if (route === 'namespace') expect(result.output).toContain(' exposed.hidden');
    expect(result.output).not.toContain('exposed.visible');
    writeFileSync(
      join(directory, index),
      route === 'default'
        ? "import {visible} from './analysis-core/fixture/hidden'; export default visible;"
        : "export * as exposed from './analysis-core/fixture/public';"
    );
    writeFileSync(
      join(directory, SOURCE_ROOT, 'analysis-core/fixture/public.ts'),
      'export const visible = 2;'
    );
    expect(runChecker(directory, '--rule', 'internal-export').exitCode).toBe(0);
  });

  it.each([
    'local',
    'imported'
  ])('follows selected indexed-access props through %s aliases (TOOL-117)', route => {
    const types = `
      type Key = 'options';
      interface Branches {options: {
        /** @internal */ hidden?: number
      }; public: {visible: number};}
      export type Selected<T extends Branches = Branches> = T[Key];
      export type PublicSelected = Branches['public'];
    `;
    const entry = `${SOURCE_ROOT}/analysis-core/index.ts`;
    const directory = createFixture({
      [entry]:
        (route === 'local'
          ? types
          : "import type {Selected, PublicSelected} from './fixture/types';") +
        '\nexport type PublicProps = Selected;',
      [`${SOURCE_ROOT}/analysis-core/fixture/types.ts`]: types
    });
    const result = runChecker(directory, '--rule', 'internal-export');
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain('PublicProps.hidden');
    writeFileSync(
      join(directory, entry),
      (route === 'local' ? types : "import type {PublicSelected} from './fixture/types';") +
        '\nexport type PublicProps = PublicSelected;'
    );
    expect(runChecker(directory, '--rule', 'internal-export').exitCode).toBe(0);
  });

  it('credits only catalog records connected to the exported construction (TOOL-117)', () => {
    const catalog = `${SOURCE_ROOT}/gpu-spatial-analysis/ontology/index.ts`;
    const orphan =
      "const REUSED_ANALYSIS_OPERATORS = [{exportName: 'addFixtureRecipe', package: '@luma-contributors/analysis/analysis-core'}];";
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: 'export function addFixtureRecipe() {}',
      [catalog]: orphan + '\nexport const ALL_OPERATORS = [];'
    });
    const missing = runChecker(directory, '--rule', 'unregistered-contributor');
    expect(missing.exitCode, missing.output).toBe(1);
    writeFileSync(
      join(directory, catalog),
      orphan +
        '\nconst family = [...REUSED_ANALYSIS_OPERATORS];\nexport const ALL_OPERATORS = [...family];'
    );
    const connected = runChecker(directory, '--rule', 'unregistered-contributor');
    expect(connected.exitCode, connected.output).toBe(0);
  });
  it.each([
    'GPU_SPATIAL_ANALYSIS_OPERATORS',
    'GPU_SPATIAL_ANALYSIS_OPERATOR_DESCRIPTORS'
  ])('traces imported families, spreads and cloned %s records (TOOL-117)', collection => {
    const catalog = `${SOURCE_ROOT}/gpu-spatial-analysis/ontology/index.ts`;
    const family = `${SOURCE_ROOT}/gpu-spatial-analysis/ontology/family.ts`;
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: 'export function addFixtureRecipe() {}',
      [family]: `export const ${collection} = [{exportName: 'addSourceRecipe', package: '@luma-contributors/analysis/gpu-spatial-analysis'}];`,
      [catalog]: `import {${collection}} from './family';
          const family = [...${collection}];
          export const ALL_OPERATORS = [...family, {
            ...${collection}.find(operator => operator.exportName === 'addSourceRecipe')!,
            package: '@luma-contributors/analysis/analysis-core', exportName: 'addFixtureRecipe'
          }];`
    });
    const connected = runChecker(directory, '--rule', 'unregistered-contributor');
    expect(connected.exitCode, connected.output).toBe(0);
    writeFileSync(join(directory, family), `export const ${collection} = [];`);
    const missing = runChecker(directory, '--rule', 'unregistered-contributor');
    expect(missing.exitCode, missing.output).toBe(1);
  });

  it('keeps internal site IDs and acceptance independent of directory enumeration order', () => {
    const file = `${SOURCE_ROOT}/analysis-core/shared/base.ts`;
    const directory = createFixture({
      [file]: 'export class GPUSharedFixture {\n/** @internal */\nhidden = 1;\n}',
      [`${SOURCE_ROOT}/gpu-spatial-analysis/index.ts`]:
        "export {GPUSharedFixture as GPUAlpha} from '../analysis-core/shared/base';",
      [`${SOURCE_ROOT}/gpu-terrain/index.ts`]:
        "export {GPUSharedFixture as GPUZulu} from '../analysis-core/shared/base';",
      'enumeration.cjs': `const fs = require('node:fs');
const original = fs.readdirSync;
fs.readdirSync = function(directory, ...arguments_) {
  const entries = original.call(this, directory, ...arguments_);
  return directory === 'modules/analysis/src' && process.env.CONTRACT_REVERSE_ENTRIES === '1'
    ? entries.reverse() : entries;
};`
    });
    const description = 'GPUAlpha.hidden';
    const id = `${createHash('sha256').update(description).digest('hex')}:1`;
    writeFileSync(
      join(directory, BASELINE_PATH),
      JSON.stringify({
        _version: 2,
        'internal-export': {
          [file]: [
            {
              id,
              description,
              owner: 'fixture',
              legacy: 'count-baseline-2026-11-17',
              reason: 'Reviewed shared member fixture'
            }
          ]
        }
      })
    );
    const results = ['0', '1'].map(reverse =>
      spawnSync(
        process.execPath,
        [
          '--require',
          join(directory, 'enumeration.cjs'),
          CHECKER_PATH,
          '--rule',
          'internal-export',
          '--all',
          '--check-stale'
        ],
        {cwd: directory, encoding: 'utf8', env: {...process.env, CONTRACT_REVERSE_ENTRIES: reverse}}
      )
    );
    for (const result of results) expect(result.status, result.stdout + result.stderr).toBe(0);
    expect(results[1].stdout).toBe(results[0].stdout);
    expect(results[0].stdout).toContain(description);
    expect(results[1].stdout).not.toContain(`STALE internal-export ${file} ${id}`);
  });

  it('requires separate qualified registrations for same-name visibility recipes', () => {
    const spatial = `${SOURCE_ROOT}/gpu-spatial-analysis/visibility/recipe.ts`;
    const network = `${SOURCE_ROOT}/gpu-network/visibility/recipe.ts`;
    const catalog = `${SOURCE_ROOT}/gpu-spatial-analysis/ontology/index.ts`;
    const spatialRegistration =
      "export const GPU_SPATIAL_ANALYSIS_CAPABILITIES = [{operators: ['addVisibilityGraphRecipe']}];" +
      "const GPU_SPATIAL_ANALYSIS_OPERATORS = GPU_SPATIAL_ANALYSIS_CAPABILITIES.flatMap(capability => capability.operators.map(exportName => ({exportName, package: '@luma-contributors/analysis/gpu-spatial-analysis'})));";
    const directory = createFixture({
      [spatial]: 'export function addVisibilityGraphRecipe() {}',
      [network]: 'export function addVisibilityGraphRecipe() {}',
      [`${SOURCE_ROOT}/gpu-spatial-analysis/index.ts`]:
        "export {addVisibilityGraphRecipe} from './visibility/recipe';",
      [`${SOURCE_ROOT}/gpu-network/index.ts`]:
        "export {addVisibilityGraphRecipe} from './visibility/recipe';",
      [catalog]: spatialRegistration
    });
    const missing = runChecker(directory, '--rule', 'unregistered-contributor');
    expect(missing.exitCode, missing.output).toBe(1);
    expect(missing.output).toContain(network);
    expect(missing.output).toContain('@luma-contributors/analysis/gpu-network');
    expect(missing.output).not.toContain(`NEW unregistered-contributor ${spatial}`);
    writeFileSync(
      join(directory, catalog),
      spatialRegistration +
        "\nexport const ALL_OPERATORS = [...GPU_SPATIAL_ANALYSIS_OPERATORS, {...GPU_SPATIAL_ANALYSIS_OPERATORS.find(operator => operator.exportName === 'addVisibilityGraphRecipe')!, package: '@luma-contributors/analysis/gpu-network'}];"
    );
    const complete = runChecker(directory, '--rule', 'unregistered-contributor');
    expect(complete.exitCode, complete.output).toBe(0);
    const copy =
      "...GPU_SPATIAL_ANALYSIS_OPERATORS.find(operator => operator.exportName === 'addVisibilityGraphRecipe')!";
    for (const properties of [
      `${copy}, package: '@luma-contributors/analysis/gpu-network', exportName: 'addOtherRecipe'`,
      `${copy}, exportName: 'addVisibilityGraphRecipe', exportName: 'addOtherRecipe', package: '@luma-contributors/analysis/gpu-network'`,
      `package: '@luma-contributors/analysis/gpu-network', ${copy}`
    ]) {
      writeFileSync(
        join(directory, catalog),
        spatialRegistration +
          `\nexport const ALL_OPERATORS = [...GPU_SPATIAL_ANALYSIS_OPERATORS, {${properties}}];`
      );
      const overwritten = runChecker(directory, '--rule', 'unregistered-contributor');
      expect(overwritten.exitCode, overwritten.output).toBe(1);
      expect(overwritten.output).toContain(`NEW unregistered-contributor ${network}`);
      expect(overwritten.output).toContain('@luma-contributors/analysis/gpu-network');
    }
    writeFileSync(
      join(directory, catalog),
      spatialRegistration +
        `\nexport const ALL_OPERATORS = [...GPU_SPATIAL_ANALYSIS_OPERATORS, {exportName: 'addOtherRecipe', ${copy}, package: '@luma-contributors/analysis/gpu-network'}];`
    );
    const restored = runChecker(directory, '--rule', 'unregistered-contributor');
    expect(restored.exitCode, restored.output).toBe(0);
    writeFileSync(
      join(directory, `${SOURCE_ROOT}/gpu-network/index.ts`),
      "export {addVisibilityGraphRecipe as addOtherRecipe} from './visibility/recipe';"
    );
    writeFileSync(
      join(directory, catalog),
      spatialRegistration +
        `\nexport const ALL_OPERATORS = [...GPU_SPATIAL_ANALYSIS_OPERATORS, {${copy}, package: '@luma-contributors/analysis/gpu-network', exportName: 'addOtherRecipe'}];`
    );
    const renamed = runChecker(directory, '--rule', 'unregistered-contributor');
    expect(renamed.exitCode, renamed.output).toBe(0);
  });
  it.each([
    ["['exportName']: 'addRenamedVisibilityRecipe'", false],
    ["...{summary: 'updated description'}", true],
    ["...{['exportName']: 'addRenamedVisibilityRecipe'}", false],
    [
      "['exportName']: 'addRenamedVisibilityRecipe', ...{exportName: 'addVisibilityGraphRecipe'}",
      true
    ],
    ["...{...{summary: 'nested updated description'}}", true],
    ["...{[0]: 'metadata', ['summary']: 'updated description'}", true],
    [
      "...{package: '@luma-contributors/analysis/gpu-network', exportName: 'addVisibilityGraphRecipe'}, ['exportName']: 'addRenamedVisibilityRecipe'",
      false
    ]
  ] as const)('matches runtime qualified lookups after catalog edit %s', (edit, expected) => {
    const spatial = `${SOURCE_ROOT}/gpu-spatial-analysis/visibility/recipe.ts`;
    const network = `${SOURCE_ROOT}/gpu-network/visibility/recipe.ts`;
    const catalog = `${SOURCE_ROOT}/gpu-spatial-analysis/ontology/index.ts`;
    const copy =
      "...GPU_SPATIAL_ANALYSIS_OPERATORS.find(operator => operator.exportName === 'addVisibilityGraphRecipe')";
    const properties = `${copy}, package: '@luma-contributors/analysis/gpu-network', ${edit}`;
    const spatialOperators = [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'addVisibilityGraphRecipe'
      }
    ];
    // Execute only this controlled fixture expression, independently of the checker's parser.
    const runtime = new Function(
      'GPU_SPATIAL_ANALYSIS_OPERATORS',
      `return [...GPU_SPATIAL_ANALYSIS_OPERATORS, {${properties}}];`
    )(spatialOperators) as typeof spatialOperators;
    const lookup = (entryPoint: string) =>
      runtime.find(
        operator =>
          operator.package === entryPoint && operator.exportName === 'addVisibilityGraphRecipe'
      );
    expect(lookup('@luma-contributors/analysis/gpu-spatial-analysis')).toBeDefined();
    expect(Boolean(lookup('@luma-contributors/analysis/gpu-network'))).toBe(expected);
    const directory = createFixture({
      [spatial]: 'export function addVisibilityGraphRecipe() {}',
      [network]: 'export function addVisibilityGraphRecipe() {}',
      [`${SOURCE_ROOT}/gpu-spatial-analysis/index.ts`]:
        "export {addVisibilityGraphRecipe} from './visibility/recipe';",
      [`${SOURCE_ROOT}/gpu-network/index.ts`]:
        "export {addVisibilityGraphRecipe} from './visibility/recipe';",
      [catalog]:
        "export const GPU_SPATIAL_ANALYSIS_CAPABILITIES = [{operators: ['addVisibilityGraphRecipe']}];" +
        "const GPU_SPATIAL_ANALYSIS_OPERATORS = GPU_SPATIAL_ANALYSIS_CAPABILITIES.flatMap(capability => capability.operators.map(exportName => ({exportName, package: '@luma-contributors/analysis/gpu-spatial-analysis'})));" +
        `\nexport const ALL_OPERATORS = [...GPU_SPATIAL_ANALYSIS_OPERATORS, {${properties}}];`
    });
    const result = runChecker(directory, '--rule', 'unregistered-contributor');
    expect(result.exitCode, result.output).toBe(expected ? 0 : 1);
    expect(result.output.includes(`NEW unregistered-contributor ${network}`)).toBe(!expected);
    expect(result.output).not.toContain(`NEW unregistered-contributor ${spatial}`);
  });

  it('qualifies recipes by public barrel aliases rather than implementation directories', () => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: 'export function addFixtureRecipe() {}',
      [`${SOURCE_ROOT}/gpu-network/index.ts`]:
        "export {addFixtureRecipe as addVisibilityGraphRecipe} from '../analysis-core/fixture/gpu-fixture';",
      [`${SOURCE_ROOT}/gpu-spatial-analysis/index.ts`]:
        "export {addFixtureRecipe as addVisibilityGraphRecipe} from '../analysis-core/fixture/gpu-fixture';",
      [`${SOURCE_ROOT}/gpu-spatial-analysis/ontology/index.ts`]:
        "export const GPU_SPATIAL_ANALYSIS_CAPABILITIES = [{operators: ['addVisibilityGraphRecipe']}];"
    });
    const missing = runChecker(directory, '--rule', 'unregistered-contributor');
    expect(missing.exitCode, missing.output).toBe(1);
    expect(missing.output).toContain('unregistered-contributor           1    1');
    expect(missing.output).toContain('@luma-contributors/analysis/gpu-network');
    writeFileSync(
      join(directory, `${SOURCE_ROOT}/gpu-spatial-analysis/ontology/index.ts`),
      "export const ALL_OPERATORS = [{exportName: 'addVisibilityGraphRecipe', package: '@luma-contributors/analysis/gpu-network'}, {exportName: 'addVisibilityGraphRecipe', package: '@luma-contributors/analysis/gpu-spatial-analysis'}];"
    );
    const complete = runChecker(directory, '--rule', 'unregistered-contributor');
    expect(complete.exitCode, complete.output).toBe(0);
  });
  it.each([
    'constant',
    'transitive constant',
    'kernel identity'
  ])('expires an allowance after changing the %s', replacement => {
    const source = (maximum: number, id: string) => `function addKernel() {
      createWGSLKernelNode(graph, {id: '${id}', invocationCount: 1,
        declarations: '${replacement === 'transitive constant' ? `const BASE_COUNT = ${maximum}u; const MAX_COUNT = BASE_COUNT;` : `const MAX_COUNT = ${maximum}u;`}',
        body: 'for(var row=0u;row<MAX_COUNT;row++){}'});
    }`;
    const directory = createFixture({[CONTRIBUTOR_PATH]: source(4, 'bounded-kernel')});
    expect(
      runChecker(directory, '--update-baseline', '--reason', 'Reviewed four-step kernel').exitCode
    ).toBe(0);
    writeFileSync(
      join(directory, CONTRIBUTOR_PATH),
      replacement !== 'kernel identity'
        ? source(1000000, 'bounded-kernel')
        : source(4, 'replacement-kernel')
    );
    const result = runChecker(directory, '--check-stale');
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain('NEW single-invocation-data-loop');
    expect(result.output).toContain('STALE single-invocation-data-loop');
  });

  it('keeps an allowance when only an unreferenced module constant changes', () => {
    const source = (unrelated: number) =>
      `createWGSLKernelNode(graph, {id: 'bounded', invocationCount: 1, declarations: 'const MAX_COUNT = 4u; const UNRELATED = ${unrelated}u;', body: 'for(var row=0u;row<MAX_COUNT;row++){}'});`;
    const directory = createFixture({[CONTRIBUTOR_PATH]: source(1)});
    expect(
      runChecker(directory, '--update-baseline', '--reason', 'Reviewed four-step kernel').exitCode
    ).toBe(0);
    writeFileSync(join(directory, CONTRIBUTOR_PATH), source(1000000));
    const result = runChecker(directory, '--check-stale');
    expect(result.exitCode, result.output).toBe(0);
  });

  it('expires an allowance when a referenced pipeline override changes', () => {
    const source = (maximum: number) => `const maximumCount = ${maximum};
      createWGSLKernelNode(graph, {id: 'bounded-override', invocationCount: 1,
        constants: {MAX_COUNT: maximumCount}, declarations: 'override MAX_COUNT: u32 = 4u;',
        body: 'for(var row=0u;row<MAX_COUNT;row++){}'});`;
    const directory = createFixture({[CONTRIBUTOR_PATH]: source(4)});
    expect(
      runChecker(directory, '--update-baseline', '--reason', 'Reviewed four-step override').exitCode
    ).toBe(0);
    writeFileSync(join(directory, CONTRIBUTOR_PATH), source(1000000));
    const result = runChecker(directory, '--check-stale');
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain('NEW single-invocation-data-loop');
    expect(result.output).toContain('STALE single-invocation-data-loop');
  });

  it('does not register a recipe mentioned only in the dimensionality tag set', () => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: 'export function addFixtureRecipe() {}',
      [`${SOURCE_ROOT}/gpu-spatial-analysis/ontology/index.ts`]:
        "export const SPATIOTEMPORAL_OPERATOR_NAMES = new Set(['addFixtureRecipe']);"
    });
    const result = runChecker(directory, '--rule', 'unregistered-contributor');
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain('addFixtureRecipe');
  });
  it.each([
    'class instance',
    'callback return'
  ])('finds internal fields in a reachable %s', route => {
    const directory = createFixture({
      [`${SOURCE_ROOT}/index.ts`]: `export type {PublicProps} from './analysis-core/fixture/gpu-fixture';`,
      [CONTRIBUTOR_PATH]: `
        class Configuration {
          /** @internal */ batchReplicates?: number;
          /** @internal */ private privateControl?: number;
          /** @internal */ static staticControl?: number;
        }
        interface Result {
          /** @internal */ fittedScale?: 'link';
        }
        export type PublicProps = {input: ${route === 'class instance' ? 'Configuration' : '() => Result'}};
      `
    });
    const result = runChecker(directory, '--rule', 'internal-export');
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain(route === 'class instance' ? 'batchReplicates' : 'fittedScale');
    expect(result.output).not.toContain('privateControl');
    expect(result.output).not.toContain('staticControl');
  });

  it('keeps cell and centrality composition preflights off public classes', () => {
    const result = runChecker(
      resolve(dirname(CHECKER_PATH), '..'),
      '--rule',
      'internal-export',
      '--paths',
      `${SOURCE_ROOT}/gpu-spatial-analysis/cell-aggregation/gpu-cell-aggregation.ts`,
      `${SOURCE_ROOT}/gpu-network/network-centrality/gpu-network-centrality.ts`
    );
    expect(result.exitCode, result.output).toBe(0);
    expect(result.output).toMatch(/^internal-export\s+0\s+0$/m);
  });

  it('rejects internal members reached through public props aliases and parameter types', () => {
    const directory = createFixture({
      [`${SOURCE_ROOT}/index.ts`]: `export {GPUFixture, PublicProps} from './analysis-core/fixture/gpu-fixture';`,
      [`${SOURCE_ROOT}/analysis-core/fixture/shared.ts`]: `
        export interface Shared {
          /** @internal */ extraCounts?: number[]; visible: number;
        }
        export type Options = {
          /** @internal */ gate?: number
        };
      `,
      [CONTRIBUTOR_PATH]: `
        import type {Shared as Parent, Options} from './shared';
        type InternalBranch = {
          /** @internal */ fittedScale?: 'link'
        };
        export type PublicProps = Parent & (InternalBranch | {mode: 'ordinary'});
        export class GPUFixture {
          constructor(props: PublicProps, /** @internal */ separateStatus = false) {}
          getCommandNodes(graph: unknown, options: Options) {return [];}
        }
      `
    });
    const result = runChecker(directory, '--rule', 'internal-export');
    expect(result.exitCode, result.output).toBe(1);
    for (const member of ['extraCounts', 'fittedScale', 'separateStatus', 'gate']) {
      expect(result.output).toContain(member);
    }
  });

  it('checks generics, nested props and interface inheritance without revisiting cycles', () => {
    const directory = createFixture({
      [`${SOURCE_ROOT}/index.ts`]: `export type {PublicProps} from './analysis-core/fixture/gpu-fixture';`,
      [CONTRIBUTOR_PATH]: `
        interface Hidden {
          /** @internal */ batchReplicates?: number;
        }
        type Container<Value, Default = Value> = {options: Default};
        interface Base extends Container<Hidden> {next?: PublicProps;}
        export interface PublicProps extends Base {values: number[];}
      `
    });
    const result = runChecker(directory, '--rule', 'internal-export');
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain('batchReplicates');
    const baselineBefore = readFileSync(join(directory, BASELINE_PATH), 'utf8');
    const update = runChecker(directory, '--update-baseline');
    expect(update.exitCode, update.output).toBe(1);
    expect(update.output).toContain('cannot be baselined');
    expect(readFileSync(join(directory, BASELINE_PATH), 'utf8')).toBe(baselineBefore);
  });

  it('resolves namespace props and renamed public exports', () => {
    const directory = createFixture({
      [`${SOURCE_ROOT}/index.ts`]: `export {FixtureProps as Settings} from './analysis-core/fixture/gpu-fixture';`,
      [`${SOURCE_ROOT}/analysis-core/fixture/shared.ts`]: `export interface Options {
        /** @internal */ sampleIndexChunkQueryCount?: number;
      }`,
      [CONTRIBUTOR_PATH]: `import type * as shared from './shared';
        export type FixtureProps = shared.Options;`
    });
    const result = runChecker(directory, '--rule', 'internal-export');
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain('sampleIndexChunkQueryCount');
  });

  it('ignores omitted props, private methods and erased implementation signatures', () => {
    const directory = createFixture({
      [`${SOURCE_ROOT}/index.ts`]: `export {GPUFixture, PublicProps} from './analysis-core/fixture/gpu-fixture';`,
      [CONTRIBUTOR_PATH]: `
        interface Shared {
          /** @internal */ extraCounts?: number[]; visible: number;
        }
        type Options = {
          /** @internal */ gate?: number
        };
        export type PublicProps = Omit<Shared, 'extraCounts'> & Pick<Shared, 'visible'>;
        export class GPUFixture {
          constructor(props: PublicProps);
          constructor(props: PublicProps, /** @internal */ extraCounts?: number[]) {}
          getCommandNodes(graph: unknown): unknown[];
          getCommandNodes(graph: unknown, /** @internal */ options?: Options) {return [];}
          private compose(options: Options) {}
          protected prepare(options: Options) {}
        }
        class Unexported {constructor(/** @internal */ hidden?: number) {}}
      `
    });
    const result = runChecker(directory, '--rule', 'internal-export');
    expect(result.exitCode, result.output).toBe(0);
  });

  it('rejects a contributor mentioned only in an ontology comment or unrelated string', () => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: 'export function addFixtureRecipe() {}',
      [`${SOURCE_ROOT}/gpu-spatial-analysis/ontology/index.ts`]:
        "// addFixtureRecipe\nconst description = 'addFixtureRecipe';"
    });
    const result = runChecker(directory, '--rule', 'unregistered-contributor');
    expect(result.exitCode, result.output).toBe(1);
  });

  it.each([
    "const PACKAGE = '@luma-contributors/analysis'; export const ENTRY_POINT_OPERATOR_DETAILS = {addFixtureRecipe: {package: `${PACKAGE}/analysis-core`}}; export const ENTRY_POINT_CAPABILITIES = [{operators: ['addFixtureRecipe']}];",
    "export const REUSED_ANALYSIS_OPERATORS = [{exportName: 'addFixtureRecipe', package: '@luma-contributors/analysis/analysis-core'}];"
  ])('accepts parsed catalog construction inputs: %s', catalog => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: 'export function addFixtureRecipe() {}',
      [`${SOURCE_ROOT}/gpu-spatial-analysis/ontology/index.ts`]: catalog
    });
    const result = runChecker(directory, '--rule', 'unregistered-contributor');
    expect(result.exitCode, result.output).toBe(0);
  });

  it('rejects an annotation key that never enters the operator catalog', () => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: 'export function addFixtureRecipe() {}',
      [`${SOURCE_ROOT}/gpu-spatial-analysis/ontology/index.ts`]:
        'export const OPERATOR_ANNOTATIONS = {addFixtureRecipe: {tier: "scale"}};'
    });
    expect(runChecker(directory, '--rule', 'unregistered-contributor').exitCode).toBe(1);
  });

  it.each([
    'analysis-core',
    'pointcloud-analysis',
    'geospatial-analysis'
  ])('checks direct source files in %s', family => {
    const directory = createFixture({
      [`${SOURCE_ROOT}/${family}/loose.ts`]: 'export const value = 1;'
    });
    const result = runChecker(directory, '--rule', 'root-file');
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain(`root-file ${SOURCE_ROOT}/${family}/loose.ts`);
  });

  it('checks every published entry directory including nested entries', () => {
    const directory = createFixture({
      'modules/analysis/package.json': JSON.stringify({
        exports: {
          './new-family': './dist/new-family/index.js',
          './new-family/nested': './dist/new-family/nested/index.js'
        }
      }),
      [`${SOURCE_ROOT}/new-family/loose.ts`]: 'export const first = 1;',
      [`${SOURCE_ROOT}/new-family/nested/loose.ts`]: 'export const second = 2;',
      [`${SOURCE_ROOT}/new-family/nested/index.ts`]: 'export const third = 3;'
    });
    const result = runChecker(directory, '--rule', 'root-file');
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain(`root-file ${SOURCE_ROOT}/new-family/loose.ts`);
    expect(result.output).toContain(`root-file ${SOURCE_ROOT}/new-family/nested/loose.ts`);
    expect(result.output).not.toContain('/nested/index.ts');
  });

  it.each([
    ['./families/*', './dist/families/*/index.js', 'families/a'],
    ['./f', './dist/f/public.js', 'f'],
    ['./f', './dist/f/index.mjs', 'f']
  ])('resolves public root coverage for %s targeting %s', (entry, target, directory) => {
    const fixture = createFixture({
      'modules/analysis/package.json': JSON.stringify({exports: {[entry]: target}}),
      [`${SOURCE_ROOT}/${directory}/index.ts`]: '',
      [`${SOURCE_ROOT}/${directory}/loose.ts`]: 'export const value = 1;'
    });
    const result = runChecker(fixture, '--rule', 'root-file');
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain(`root-file ${SOURCE_ROOT}/${directory}/loose.ts`);
  });

  it.each([
    './assets/schema.json',
    './dist/f/',
    './dist/missing/*.js'
  ])('refuses an unresolved or unsupported export target %s instead of silently disabling coverage', target => {
    const fixture = createFixture({
      'modules/analysis/package.json': JSON.stringify({exports: {'./f': target}}),
      [`${SOURCE_ROOT}/f/index.ts`]: '',
      [`${SOURCE_ROOT}/f/loose.ts`]: 'export const value = 1;'
    });
    const result = runChecker(fixture, '--rule', 'root-file');
    expect(result.exitCode, result.output).toBe(2);
    expect(result.output).toContain('Cannot resolve root-file export');
    expect(result.output).toContain(target);
  });

  it('uses one wildcard substitution at every target occurrence and checks every matching root', () => {
    const fixture = createFixture({
      'modules/analysis/package.json': JSON.stringify({
        exports: {
          './families/*': './dist/families/*/*/index.js'
        }
      }),
      [`${SOURCE_ROOT}/families/a/a/index.ts`]: '',
      [`${SOURCE_ROOT}/families/a/a/loose.ts`]: '',
      [`${SOURCE_ROOT}/families/b/b/index.mts`]: '',
      [`${SOURCE_ROOT}/families/b/b/loose.cts`]: '',
      [`${SOURCE_ROOT}/families/a/b/index.ts`]: '',
      [`${SOURCE_ROOT}/families/a/b/loose.ts`]: ''
    });
    const result = runChecker(fixture, '--rule', 'root-file');
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain(`root-file ${SOURCE_ROOT}/families/a/a/loose.ts`);
    expect(result.output).toContain(`root-file ${SOURCE_ROOT}/families/b/b/loose.cts`);
    expect(result.output).not.toContain(`root-file ${SOURCE_ROOT}/families/a/b/loose.ts`);
  });

  it.each([
    'cts',
    'mts',
    'tsx',
    'js',
    'jsx',
    'mjs',
    'cjs'
  ])('checks a loose .%s implementation independently of the TypeScript contract traversal', extension => {
    const fixture = createFixture({
      'modules/analysis/package.json': JSON.stringify({exports: {'./f': './dist/f/index.js'}}),
      [`${SOURCE_ROOT}/f/index.ts`]: '',
      [`${SOURCE_ROOT}/f/loose.${extension}`]: 'export const value = 1;'
    });
    const result = runChecker(fixture, '--rule', 'root-file');
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain(`root-file ${SOURCE_ROOT}/f/loose.${extension}`);
  });

  it('checks a nested public entry below the exact computation-record exception', () => {
    const fixture = createFixture({
      'modules/analysis/package.json': JSON.stringify({
        exports: {
          './computation-record': './dist/computation-record/index.js',
          './computation-record/analysis': './dist/computation-record/analysis/index.js'
        }
      }),
      [`${SOURCE_ROOT}/computation-record/infrastructure.ts`]: '',
      [`${SOURCE_ROOT}/computation-record/analysis/index.ts`]: '',
      [`${SOURCE_ROOT}/computation-record/analysis/loose.ts`]: 'export const value = 1;'
    });
    const result = runChecker(fixture, '--rule', 'root-file');
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain(
      `root-file ${SOURCE_ROOT}/computation-record/analysis/loose.ts`
    );
    expect(result.output).not.toContain('computation-record/infrastructure.ts');
  });

  it('checks the source directory behind a conditional public export alias', () => {
    const directory = createFixture({
      'modules/analysis/package.json': JSON.stringify({
        exports: {
          './public-alias': {
            types: './dist/actual-family/index.d.ts',
            import: './dist/actual-family/index.js',
            require: './dist/actual-family/index.cjs'
          }
        }
      }),
      [`${SOURCE_ROOT}/actual-family/loose.ts`]: 'export const first = 1;'
    });
    const result = runChecker(directory, '--rule', 'root-file');
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain(`root-file ${SOURCE_ROOT}/actual-family/loose.ts`);
  });

  it('exempts named infrastructure entries and ignores unpublished directories', () => {
    const directory = createFixture({
      'modules/analysis/package.json': JSON.stringify({
        exports: {
          './computation-record': './dist/computation-record/index.js',
          './gpu-spatial-analysis/ontology': './dist/gpu-spatial-analysis/ontology/index.js'
        }
      }),
      [`${SOURCE_ROOT}/computation-record/loose.ts`]: 'export const first = 1;',
      [`${SOURCE_ROOT}/gpu-spatial-analysis/ontology/loose.ts`]: 'export const second = 2;',
      [`${SOURCE_ROOT}/gpu-network/unpublished.ts`]: 'export const third = 3;'
    });
    const result = runChecker(directory, '--rule', 'root-file');
    expect(result.exitCode, result.output).toBe(0);
    expect(result.output).toContain('root-file');
    expect(result.output).not.toContain('NEW root-file');
  });

  it('keeps a loop allowance stable through whitespace and line shifts', () => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: kernel('for(var row=0u;row<rowCount;row++){}')
    });
    expect(runChecker(directory, '--update-baseline', '--reason', 'Reviewed loop').exitCode).toBe(
      0
    );
    writeFileSync(
      join(directory, CONTRIBUTOR_PATH),
      `// new introductory comment\n\n${kernel('for (var row = 0u; row < rowCount; row++) { }')}`
    );
    const result = runChecker(directory, '--check-stale');
    expect(result.exitCode, result.output).toBe(0);
  });

  it('does not let a replacement loop spend a removed loop allowance', () => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: kernel('for(var row=0u;row<rowCount;row++){}')
    });
    expect(
      runChecker(directory, '--update-baseline', '--reason', 'Reviewed legacy site').exitCode
    ).toBe(0);
    writeFileSync(
      join(directory, CONTRIBUTOR_PATH),
      kernel('for(var pair=0u;pair<pairCount;pair++){}')
    );
    const result = runChecker(directory);
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain('NEW single-invocation-data-loop');
  });

  it('invalidates a loop allowance when only the host bound its reason cites changes (CORE-118)', () => {
    const source = (bound: string) =>
      `export ${bound};\n${kernel('for(var row=0u;row<rowCount;row++){}')}`;
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: source('const MAXIMUM_FIXTURE_ROW_COUNT = 64')
    });
    expect(
      runChecker(
        directory,
        '--update-baseline',
        '--reason',
        'Rows capped by MAXIMUM_FIXTURE_ROW_COUNT = 64 at construction.'
      ).exitCode
    ).toBe(0);
    const accepted = runChecker(directory);
    expect(accepted.exitCode, accepted.output).toBe(0);
    // The shader and site identity are unchanged; only the host bound moves.
    writeFileSync(
      join(directory, CONTRIBUTOR_PATH),
      source('const MAXIMUM_FIXTURE_ROW_COUNT = 65536')
    );
    const raised = runChecker(directory);
    expect(raised.exitCode, raised.output).toBe(1);
    expect(raised.output).toContain('NEW single-invocation-data-loop');
    expect(raised.output).toContain(
      'cites MAXIMUM_FIXTURE_ROW_COUNT = 64, but the source value is 65536'
    );
    writeFileSync(join(directory, CONTRIBUTOR_PATH), source('const MAXIMUM_ROW_LIMIT = 64'));
    const renamed = runChecker(directory);
    expect(renamed.exitCode, renamed.output).toBe(1);
    expect(renamed.output).toContain('cites MAXIMUM_FIXTURE_ROW_COUNT, which no analysis source');
  });

  it.each([
    ['MGWR_INFERENCE_MAXIMUM_ROW_COUNT', 128, 129],
    ['MGWR_INFERENCE_MAXIMUM_WORK', 268435456, 268435457]
  ] as const)('W2-03 reports a new gate finding when %s changes', (name, before, after) => {
    const repository = resolve(dirname(CHECKER_PATH), '..');
    const inferencePath = `${SOURCE_ROOT}/gpu-spatial-analysis/spatial-regression/multiscale-inference.ts`;
    const baseline = readFileSync(join(repository, BASELINE_PATH), 'utf8');
    const source = readFileSync(join(repository, inferencePath), 'utf8');
    const directory = createFixture({[inferencePath]: source, [BASELINE_PATH]: baseline});
    const accepted = runChecker(directory, '--rule', 'single-invocation-data-loop');
    expect(accepted.exitCode, accepted.output).toBe(0);

    const declaration = `export const ${name} = ${before};`;
    expect(source).toContain(declaration);
    writeFileSync(
      join(directory, inferencePath),
      source.replace(declaration, `export const ${name} = ${after};`)
    );
    const raised = runChecker(directory, '--rule', 'single-invocation-data-loop');
    expect(raised.exitCode, raised.output).toBe(1);
    expect(raised.output).toMatch(/^single-invocation-data-loop\s+1\s+1$/m);
    expect(raised.output).toContain(`NEW single-invocation-data-loop ${inferencePath}`);
  });

  it('resolves an unshadowed constant Math.floor bound and rejects its changed value (review numeric policy)', () => {
    const source = (expression: string) =>
      `export const MAXIMUM_FIXTURE_ROW_COUNT = ${expression};\n// contract-allow: single-invocation -- MAXIMUM_FIXTURE_ROW_COUNT = 1073741823 rows\n${kernel('for(var row=0u;row<rowCount;row++){}')}`;
    const directory = createFixture({[CONTRIBUTOR_PATH]: source('Math.floor(0xffffffff / 4)')});
    const original = runChecker(directory, '--rule', 'single-invocation-data-loop');
    expect(original.exitCode, original.output).toBe(0);
    writeFileSync(join(directory, CONTRIBUTOR_PATH), source('Math.floor((0xffffffff - 4) / 4)'));
    const changed = runChecker(directory, '--rule', 'single-invocation-data-loop');
    expect(changed.exitCode, changed.output).toBe(1);
    writeFileSync(
      join(directory, CONTRIBUTOR_PATH),
      'const Math = {floor: (value: number) => value};\n' + source('Math.floor(0xffffffff / 4)')
    );
    const shadowed = runChecker(directory, '--rule', 'single-invocation-data-loop');
    expect(shadowed.exitCode, shadowed.output).toBe(1);
  });

  it('rejects an unevaluable numeric host citation instead of certifying it (review earlier numeric policy)', () => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]:
        'export const MAXIMUM_FIXTURE_ROW_COUNT = 2 ** 16;\n' +
        kernel('for(var row=0u;row<rowCount;row++){}')
    });
    expect(
      runChecker(
        directory,
        '--update-baseline',
        '--reason',
        'MAXIMUM_FIXTURE_ROW_COUNT = 65536 rows.'
      ).exitCode
    ).toBe(0);
    writeFileSync(
      join(directory, CONTRIBUTOR_PATH),
      'export const MAXIMUM_FIXTURE_ROW_COUNT = 1 << 18;\n' +
        kernel('for(var row=0u;row<rowCount;row++){}')
    );
    const changed = runChecker(directory, '--rule', 'single-invocation-data-loop');
    expect(changed.exitCode, changed.output).toBe(1);
    expect(changed.output).toContain('cannot statically resolve');
  });

  it('accepts an inline loop allowance only with a stated bound that still holds (CORE-118)', () => {
    const loop = kernel('for(var row=0u;row<rowCount;row++){}');
    const check = (allowance: string) => {
      const directory = createFixture({
        [CONTRIBUTOR_PATH]: `export const MAXIMUM_FIXTURE_ROW_COUNT = 64;\n${allowance}\n${loop}`
      });
      return runChecker(directory, '--rule', 'single-invocation-data-loop');
    };
    const counted = (allowance: string) =>
      /^single-invocation-data-loop\s+(\d+)/m.exec(check(allowance).output)?.[1];
    expect(counted('// contract-allow: single-invocation')).toBe('1');
    expect(counted('// contract-allow: single-invocation --')).toBe('1');
    expect(counted('// contract-allow: single-invocation -- at most 64 rows')).toBe('0');
    expect(
      counted('// contract-allow: single-invocation -- exactly eight fixed summary words')
    ).toBe('0');
    expect(
      counted('// contract-allow: single-invocation -- four bounded order-four PSD projections')
    ).toBe('0');
    expect(counted('// At most 64 rows.\n// contract-allow: single-invocation')).toBe('0');
    expect(
      counted('// contract-allow: single-invocation -- MAXIMUM_FIXTURE_ROW_COUNT = 64 rows')
    ).toBe('0');
    expect(
      counted('// contract-allow: single-invocation -- MAXIMUM_FIXTURE_ROW_COUNT = 32 rows')
    ).toBe('1');
    expect(counted('// contract-allow: single-invocation -- MAXIMUM_REMOVED_ROW_COUNT rows')).toBe(
      '1'
    );
  });

  it('reports stale baseline entries without rewriting them', () => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: kernel('for(var row=0u;row<rowCount;row++){}')
    });
    expect(
      runChecker(directory, '--update-baseline', '--reason', 'Reviewed legacy site').exitCode
    ).toBe(0);
    const baseline = readFileSync(join(directory, BASELINE_PATH), 'utf8');
    writeFileSync(join(directory, CONTRIBUTOR_PATH), kernel(''));
    const result = runChecker(directory, '--check-stale');
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain('STALE single-invocation-data-loop');
    expect(readFileSync(join(directory, BASELINE_PATH), 'utf8')).toBe(baseline);
  });

  it('requires a reason before accepting a new baseline site', () => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: kernel('for(var row=0u;row<rowCount;row++){}')
    });
    const result = runChecker(directory, '--update-baseline');
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain('reason');
  });

  it.each([
    ['bare-assert', "import {assert} from '@luma.gl/core';\nassert(true);", 1],
    [
      'missing-producer-declaration',
      'export class GPUFixture {getCommandNodes() {return [];}}',
      ['GPUFixture']
    ],
    ['luma-reexport', "export {Device} from '@luma.gl/core';", 1],
    ['duplicate-wgsl-declaration', kernel('let row=0u; let row=1u;'), ['row']],
    ['internal-export', '/** @internal */\nexport const hidden = 1;\n', ['hidden']],
    ['unregistered-contributor', 'export function addFixtureRecipe() {}', ['addFixtureRecipe']]
  ])('rejects baselines and baseline updates for the clean rule %s', (rule, text, recorded) => {
    const file = rule === 'internal-export' ? `${SOURCE_ROOT}/index.ts` : CONTRIBUTOR_PATH;
    const directory = createFixture({[file]: text}, {[rule]: {[file]: recorded}});
    const baselineBefore = readFileSync(join(directory, BASELINE_PATH), 'utf8');
    const check = runChecker(directory, '--rule', rule);
    expect(check.exitCode, check.output).toBe(1);
    expect(check.output).toContain(`NEW ${rule} ${file}:`);
    const update = runChecker(directory, '--update-baseline');
    expect(update.exitCode, update.output).toBe(1);
    expect(update.output).toContain('cannot be baselined');
    expect(readFileSync(join(directory, BASELINE_PATH), 'utf8')).toBe(baselineBefore);
  });

  it.each([
    ['aliased', "import {assert as check} from '@luma.gl/core';\ncheck(true);"],
    ['namespace', "import * as core from '@luma.gl/core';\ncore.assert(true);"],
    ['multiline', "import {\n  type Device,\n  assert\n} from '@luma.gl/core';\nassert(true);"]
  ])('rejects a bare assert imported %s (TOOL-117)', (_, text) => {
    const directory = createFixture({[CONTRIBUTOR_PATH]: text});
    const result = runChecker(directory, '--rule', 'bare-assert');
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain(`NEW bare-assert ${CONTRIBUTOR_PATH}:`);
  });

  it('accepts a local assert and a luma.gl import without assert', () => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]:
        "import {log} from '@luma.gl/core';\nconst assert = (value: boolean) => value;\nassert(true);\nlog.assert(true);"
    });
    expect(runChecker(directory, '--rule', 'bare-assert').exitCode).toBe(0);
  });

  it.each([
    [
      'an imported internal re-exported by name',
      {
        [`${SOURCE_ROOT}/analysis-core/fixture/hidden.ts`]:
          '/** @internal */\nexport const hidden = 1;\n',
        [`${SOURCE_ROOT}/index.ts`]:
          "import {hidden} from './analysis-core/fixture/hidden';\nexport {hidden};\n"
      }
    ],
    [
      'a renamed imported internal',
      {
        [`${SOURCE_ROOT}/analysis-core/fixture/hidden.ts`]:
          '/** @internal */\nexport const hidden = 1;\n',
        [`${SOURCE_ROOT}/index.ts`]:
          "import {hidden as visible} from './analysis-core/fixture/hidden';\nexport {visible as shown};\n"
      }
    ],
    [
      'a local internal exported by a clause',
      {[`${SOURCE_ROOT}/index.ts`]: '/** @internal */\nconst hidden = 1;\nexport {hidden};\n'}
    ]
  ])('rejects %s (TOOL-117)', (_, files) => {
    const result = runChecker(createFixture(files), '--rule', 'internal-export');
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain(`NEW internal-export ${SOURCE_ROOT}/index.ts`);
  });

  it('retains explicitly reviewed legacy internal sites without accepting new members', () => {
    const file = `${SOURCE_ROOT}/index.ts`;
    const source = '/** @internal */\nexport const hidden = 1;\n';
    const directory = createFixture({[file]: source});
    const entry = {
      id: `${createHash('sha256').update('hidden').digest('hex')}:1`,
      description: 'hidden',
      reason: 'Pre-pass internal view used by existing consumers; owner: core-contracts.',
      owner: 'core-contracts',
      legacy: 'count-baseline-2026-11-17'
    };
    const baseline = {_version: 2, 'internal-export': {[file]: [entry]}};
    writeFileSync(join(directory, BASELINE_PATH), JSON.stringify(baseline));
    const accepted = runChecker(directory, '--rule', 'internal-export', '--check-stale');
    expect(accepted.exitCode, accepted.output).toBe(0);
    expect(runChecker(directory, '--update-baseline').exitCode).toBe(0);
    expect(JSON.parse(readFileSync(join(directory, BASELINE_PATH), 'utf8'))).toEqual(baseline);
    writeFileSync(
      join(directory, file),
      source + '/** @internal */\nexport const replacement = 2;\n'
    );
    const changed = runChecker(directory, '--rule', 'internal-export');
    expect(changed.exitCode, changed.output).toBe(1);
    expect(changed.output).toContain(' replacement');
    expect(
      runChecker(directory, '--update-baseline', '--reason', 'Cannot add a strict site').exitCode
    ).toBe(1);
    for (const invalid of [
      {...entry, owner: ''},
      {...entry, legacy: 'unreviewed-baseline'}
    ]) {
      writeFileSync(join(directory, file), source);
      writeFileSync(
        join(directory, BASELINE_PATH),
        JSON.stringify({_version: 2, 'internal-export': {[file]: [invalid]}})
      );
      expect(runChecker(directory, '--rule', 'internal-export').exitCode).toBe(1);
    }
  });

  it('keeps site allowances for unfinished rules and rejects additional sites', () => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: kernel('for (var row = 0u; row < rowCount; row++) {}')
    });
    expect(
      runChecker(directory, '--update-baseline', '--reason', 'Reviewed legacy site').exitCode
    ).toBe(0);
    expect(runChecker(directory).exitCode).toBe(0);
    writeFileSync(
      join(directory, CONTRIBUTOR_PATH),
      kernel('for (var row = 0u; row < rowCount; row++) {}').repeat(2)
    );
    const check = runChecker(directory);
    expect(check.exitCode, check.output).toBe(1);
    expect(check.output).toContain(`single-invocation-data-loop ${CONTRIBUTOR_PATH}:1 x2`);
  });

  it('detects while and loop syntax through local aliases and constant invocation counts', () => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: `
        const ONE_INVOCATION = 1;
        const shaderBody = 'while (row < rowCount) {row++;}';
        const body = shaderBody;
        createWGSLKernelNode(graph, {invocationCount: (ONE_INVOCATION), body});
        ${kernel('loop { if (row >= rowCount) { break; } row++; }')}
      `
    });
    const check = runChecker(directory);
    expect(check.exitCode, check.output).toBe(1);
    expect(check.output).toMatch(/single-invocation-data-loop .* x2/);
  });

  it('checks generic kernel calls and referenced property objects', () => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: `
        const properties = {invocationCount: 2 - 1, body: 'for(var row=0u;row<count;row++){}'};
        createWGSLKernelNode<Parameters>(graph, properties);
      `
    });
    const check = runChecker(directory);
    expect(check.exitCode, check.output).toBe(1);
    expect(check.output).toContain('single-invocation-data-loop');
  });

  it('checks single-invocation loops through local kernel wrappers at each call site', () => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: `
        const declarations = 'fn scan() {for(var row=0u;row<1024u;row++){}}';
        const kernel = (name, count, bindings, body, extra = '') =>
          nodes.push(createWGSLKernelNode(graph, {
            id: name, invocationCount: count, bindings,
            declarations: declarations + extra, body
          }));
        const ONE_INVOCATION = 1;
        kernel('pivot', ONE_INVOCATION, [], 'for(var row=0u;row<1024u;row++){}');
        kernel('summary', 1, [], 'scan();');
        kernel('parallel', 16, [], 'scan();');
        kernel('unknown', rowCount + 1, [], 'scan();');
      `
    });
    const check = runChecker(directory, '--rule', 'single-invocation-data-loop');
    expect(check.exitCode, check.output).toBe(1);
    expect(check.output).toMatch(/NEW single-invocation-data-loop .* x2/);
  });

  it('preserves a direct scalar kernel beside an expanded forwarding kernel', () => {
    const source = `
      function kernel(count, body) {
        createWGSLKernelNode(graph, {invocationCount: count, body});
        createWGSLKernelNode(graph, {
          invocationCount: 1,
          body: 'for(var row=0u;row<rowCount;row++){}'
        });
      }
      kernel(32, 'let value=0u;');
    `;
    const directory = createFixture({[CONTRIBUTOR_PATH]: source});
    const check = runChecker(directory, '--rule', 'single-invocation-data-loop');
    expect(check.exitCode, check.output).toBe(1);
    expect(check.output).toMatch(/^single-invocation-data-loop\s+1\s+1$/m);
    const line = source.slice(0, source.indexOf('invocationCount: 1')).split('\n').length;
    expect(check.output).toContain(`${CONTRIBUTOR_PATH}:${line}\n`);
  });

  it('accepts the reviewed bounded tiled-panel wrapper sites without a baseline entry', () => {
    const check = runChecker(
      resolve(dirname(CHECKER_PATH), '..'),
      '--all',
      '--rule',
      'single-invocation-data-loop',
      '--paths',
      `${SOURCE_ROOT}/gpu-spatial-analysis/spatial-regression/panel-lag-tiled-nodes.ts`
    );
    expect(check.exitCode, check.output).toBe(0);
    expect(check.output).toMatch(/^single-invocation-data-loop\s+0\s+0$/m);
  });

  it('accepts the isolines logarithmic capacity search without a baseline entry', () => {
    const check = runChecker(
      resolve(dirname(CHECKER_PATH), '..'),
      '--all',
      '--rule',
      'single-invocation-data-loop',
      '--paths',
      `${SOURCE_ROOT}/raster-analysis/isolines/gpu-isolines.ts`
    );
    expect(check.exitCode, check.output).toBe(0);
    expect(check.output).toMatch(/^single-invocation-data-loop\s+0\s+0$/m);
  });

  it('accepts reviewed GWR search and MGWR convergence capacities without a baseline entry', () => {
    const check = runChecker(
      resolve(dirname(CHECKER_PATH), '..'),
      '--all',
      '--rule',
      'single-invocation-data-loop',
      '--paths',
      `${SOURCE_ROOT}/gpu-spatial-analysis/spatial-regression/gwr-golden-section-nodes.ts`,
      `${SOURCE_ROOT}/gpu-spatial-analysis/spatial-regression/gpu-multiscale-geographically-weighted-regression.ts`
    );
    expect(check.exitCode, check.output).toBe(0);
    expect(check.output).toMatch(/^single-invocation-data-loop\s+0\s+0$/m);
  });

  it('keeps wrapper arguments, defaults, guards and lexical scopes independent', () => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: `
        const count = 1;
        const body = 'for(var row=0u;row<count;row++){}';
        const enabled = dynamicFlag;
        function kernel(enabled, count, body, extra = 'fn scan() {while(true){break;}}') {
          if (enabled) return createWGSLKernelNode(graph, {
            invocationCount: count, declarations: extra, body
          });
        }
        const build = kernel;
        build(false, 1, body);
        build(true, 16, body);
        if (!enabled) build(enabled, 1, body);
        build(true, 1, 'scan();', 'fn scan() {}');
        build(true, 1, 'scan();');
        // contract-allow: single-invocation -- reviewed scan of at most 64 rows
        build(true, 1, body);
        function other() {
          const build = (enabled, count, body) => unrelated(count, body);
          build(true, 1, body);
        }
      `
    });
    const check = runChecker(directory, '--rule', 'single-invocation-data-loop');
    expect(check.exitCode, check.output).toBe(1);
    expect(check.output).toMatch(/^single-invocation-data-loop\s+1\s+1$/m);
  });

  it('checks duplicate declarations in each resolved wrapper shader without mixing callers', () => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: `
        const kernel = (count, body, extra = '') => createWGSLKernelNode(graph, {
          invocationCount: count, declarations: extra, body
        });
        kernel(16, 'let duplicate=0u; let duplicate=1u;');
        kernel(16, 'let separate=0u;');
        kernel(16, 'let separate=0u;');
      `
    });
    const check = runChecker(directory, '--rule', 'duplicate-wgsl-declaration');
    expect(check.exitCode, check.output).toBe(1);
    expect(check.output).toContain(' duplicate\n');
    expect(check.output).not.toContain(' separate\n');
  });

  it('does not substitute overwritten wrapper parameters or mutable factory aliases', () => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: `
        function kernel(count, body) {
          count = 16;
          return createWGSLKernelNode(graph, {invocationCount: count, body});
        }
        kernel(1, 'for(var row=0u;row<count;row++){}');
        const forward = (count, body) => createWGSLKernelNode(graph, {invocationCount: count, body});
        let build = forward;
        build = unrelated;
        build(1, 'for(var row=0u;row<count;row++){}');
      `
    });
    const check = runChecker(directory, '--rule', 'single-invocation-data-loop');
    expect(check.exitCode, check.output).toBe(0);
  });

  it('resolves imported shader helpers and aliased kernel factories', () => {
    const directory = createFixture({
      [`${SOURCE_ROOT}/analysis-core/fixture/helpers.ts`]:
        "export const LOOP_WGSL = 'fn scan() {for(var row=0u;row<count;row++){}}';",
      [CONTRIBUTOR_PATH]: `
        import {createWGSLKernelNode as createKernel} from '../kernels/wgsl-kernel-nodes';
        import {LOOP_WGSL as declarations} from './helpers';
        createKernel(graph, {invocationCount: 1, declarations, body: 'scan();'});
      `
    });
    const check = runChecker(directory);
    expect(check.exitCode, check.output).toBe(1);
    expect(check.output).toContain(`single-invocation-data-loop ${CONTRIBUTOR_PATH}:`);
  });

  it('ignores comment loops, CPU loops, multi-invocation kernels and explicit allowances', () => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: `
        // ${kernel('for(;;) {}')}
        const example = '${kernel('for(;;) {}')}';
        for (const item of items) { ${kernel('// for(;;) {}')} }
        ${kernel('/* while (true) {} */ let row = 0u;')}
        ${kernel('for(var row=0u;row<count;row++){}', 'invocationCount: 16,')}
        ${kernel('for(var row=0u;row<count;row++){}', 'invocationCount: rowCount + 1,')}
        ${kernel('for(var row=0u;row<count;row++){}', 'invocationCount: 4 * rowCount + 1,')}
        let changingCount = 1;
        changingCount = 16;
        ${kernel('for(var row=0u;row<count;row++){}', 'invocationCount: changingCount,')}
        createWGSLKernelNode(graph, {
          // contract-allow: single-invocation -- bounded dense system of at most 4 rows
          invocationCount: 1,
          body: 'for(var row=0u;row<4u;row++){}'
        });
      `
    });
    const check = runChecker(directory);
    expect(check.exitCode, check.output).toBe(0);
  });

  it('checks only actual class members, including arrow properties and multiline exports', () => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: `
        export class GPUPlain {}
        function unrelated() { return other.getCommandNodes(); }
        export\nclass GPUFixture { getCommandNodes = () => []; }
      `
    });
    const check = runChecker(directory, '--rule', 'missing-producer-declaration');
    expect(check.exitCode, check.output).toBe(1);
    expect(check.output).toContain(' GPUFixture\n');
    expect(check.output).not.toContain(' GPUPlain\n');
  });

  it('follows aliased producer imports and inheritance without a depth limit or class-name collisions', () => {
    const basePath = `${SOURCE_ROOT}/analysis-core/fixture/base.ts`;
    const hierarchy = Array.from(
      {length: 12},
      (_, index) => `class Stage${index} extends ${index === 0 ? 'Base' : `Stage${index - 1}`} {}`
    ).join('\n');
    const directory = createFixture({
      [basePath]: `import type {GPUCommandNodeProducer as Producer} from '@luma.gl/graph/gpu-core';
        export class Base implements Producer {getCommandNodes() {return [];}}`,
      [`${SOURCE_ROOT}/analysis-core/fixture/barrel.ts`]: `
        export {Base as PublicBase} from './base';`,
      [CONTRIBUTOR_PATH]: `import {PublicBase as Parent} from './barrel';
        class Base extends Parent {}
        ${hierarchy}
        export class GPUFixture extends Stage11 {getCommandNodes() {return [];}}`,
      [`${SOURCE_ROOT}/analysis-core/fixture/other.ts`]:
        'class Base {}\nexport class GPUOther extends Base {getCommandNodes() {return [];}}'
    });
    const check = runChecker(directory, '--rule', 'missing-producer-declaration');
    expect(check.exitCode, check.output).toBe(1);
    expect(check.output).toContain(' GPUOther\n');
    expect(check.output).not.toContain(' GPUFixture\n');
  });

  it('reports inherited contributors in the registry and terminates inheritance cycles', () => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: `
        class Base implements GPUCommandNodeProducer {getCommandNodes() {return [];}}
        export class GPUFixture extends Base {}
        export class GPUCycle extends GPUCycle {}
      `
    });
    const check = runChecker(directory, '--rule', 'unregistered-contributor');
    expect(check.exitCode, check.output).toBe(1);
    expect(check.output).toContain(' GPUFixture\n');
    expect(check.output).not.toContain(' GPUCycle\n');
  });

  it('finds duplicate module declarations assembled from imported and local WGSL snippets', () => {
    const directory = createFixture({
      [`${SOURCE_ROOT}/analysis-core/fixture/helpers.ts`]:
        "export const HELPERS = 'fn shared() {} struct State {value: u32}';",
      [CONTRIBUTOR_PATH]: `
        import {HELPERS} from './helpers';
        const declarations = \`\${HELPERS}\nfn shared() {} alias State = u32;\`;
        createWGSLKernelNode(graph, {invocationCount: 16, declarations, body: ''});
      `
    });
    const check = runChecker(directory, '--rule', 'duplicate-wgsl-declaration');
    expect(check.exitCode, check.output).toBe(1);
    expect(check.output).toContain(' shared\n');
    expect(check.output).toContain(' State\n');
  });

  it('tracks WGSL scopes, nested comments, var qualifiers and separate shaders', () => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: `
        createWGSLKernelNode(graph, {
          invocationCount: 16,
          declarations: \`// fn same() {}\n/* outer /* fn same() {} */ fn same() {} */
            fn same() {let value=1u; {let value=2u;}}
            fn other() {let value=3u;}
            var<private> state: u32; const state = 1u;\`,
          body: 'let value=4u; {let value=5u;}'
        });
        createWGSLKernelNode(graph, {invocationCount: 16, declarations: 'fn same() {}', body: ''});
      `
    });
    const check = runChecker(directory, '--rule', 'duplicate-wgsl-declaration');
    expect(check.exitCode, check.output).toBe(1);
    expect(check.output).toContain(' state\n');
    expect(check.output).not.toContain(' same\n');
    expect(check.output).not.toContain(' value\n');
  });

  it('finds duplicate local declarations in the same WGSL scope', () => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: kernel('let row=0u; var row=1u;', 'invocationCount: 16,')
    });
    const check = runChecker(directory, '--rule', 'duplicate-wgsl-declaration');
    expect(check.exitCode, check.output).toBe(1);
    expect(check.output).toContain(' row\n');
  });

  it('checks unconditional shader appends without merging conditional alternatives or later appends', () => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: `
        let declarations = 'fn shared() {}';
        declarations += 'fn shared() {}';
        createWGSLKernelNode(graph, {invocationCount: 16, declarations, body: ''});
        let alternatives = '';
        if (choice) { alternatives += 'fn chosen() {}'; }
        else { alternatives += 'fn chosen() {}'; }
        createWGSLKernelNode(graph, {invocationCount: 16, declarations: alternatives, body: ''});
        alternatives += 'fn later() {} fn later() {}';
      `
    });
    const check = runChecker(directory, '--rule', 'duplicate-wgsl-declaration');
    expect(check.exitCode, check.output).toBe(1);
    expect(check.output).toContain(' shared\n');
    expect(check.output).not.toContain(' chosen\n');
    expect(check.output).not.toContain(' later\n');
  });

  it('ignores unused loop helpers but follows nested helper calls', () => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: `
        const declarations = 'fn scan(){loop {break;}} fn helper(){scan();}';
        createWGSLKernelNode(graph, {invocationCount: 1, declarations, body: ''});
        createWGSLKernelNode(graph, {invocationCount: 1, declarations, body: 'helper();'});
      `
    });
    const check = runChecker(directory);
    expect(check.exitCode, check.output).toBe(1);
    expect(check.output).toContain('single-invocation-data-loop');
    expect(check.output).not.toContain(' x2');
  });

  it.each([
    'true',
    'unknownChoice'
  ])('detects loops in conditional shader bodies (%s)', condition => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: `
        const enabled = ${condition};
        const body = enabled ? 'for(var row=0u;row<rowCount;row++){}' : '';
        createWGSLKernelNode(graph, {invocationCount: 1, body});
      `
    });
    const check = runChecker(directory, '--rule', 'single-invocation-data-loop');
    expect(check.exitCode, check.output).toBe(1);
    expect(check.output).toContain(`NEW single-invocation-data-loop ${CONTRIBUTOR_PATH}:`);
  });

  it('checks duplicates within conditional branches without combining alternatives', () => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: `
        const declarations = choice ? 'fn shared() {}' : 'fn shared() {}';
        createWGSLKernelNode(graph, {
          invocationCount: 16, declarations,
          body: choice ? 'let row=0u; let row=1u;' : ''
        });
      `
    });
    const check = runChecker(directory, '--rule', 'duplicate-wgsl-declaration');
    expect(check.exitCode, check.output).toBe(1);
    expect(check.output).toContain(' row\n');
    expect(check.output).not.toContain(' shared\n');
  });

  it('preserves conditional alternatives inside templates and known false conditions', () => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: `
        const enabled = false;
        createWGSLKernelNode(graph, {
          invocationCount: 1,
          body: enabled ? 'for(;;){} let row=0u; let row=1u;' : ''
        });
        createWGSLKernelNode(graph, {
          invocationCount: 16,
          declarations: \`fn shared() {} \${choice ? 'fn shared() {}' : ''}\`,
          body: ''
        });
      `
    });
    const check = runChecker(directory);
    expect(check.exitCode, check.output).toBe(1);
    expect(check.output).toContain(' shared\n');
    expect(check.output).not.toContain(' row\n');
    expect(check.output).not.toContain('NEW single-invocation-data-loop');
  });

  it('keeps a predicate and its negation mutually exclusive across template fragments', () => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: `
        const enabled = readChoice();
        const declarations = \`\${enabled ? 'fn shared() {}' : ''}\${!enabled ? 'fn shared() {}' : ''}\`;
        createWGSLKernelNode(graph, {invocationCount: 16, declarations, body: ''});
      `
    });
    const check = runChecker(directory);
    expect(check.exitCode, check.output).toBe(0);
  });

  it('applies shader assignments in order before appends and snapshots aliases at their use', () => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: `
        let declarations='fn shared() {}';
        declarations='';
        declarations+='fn shared() {}';
        const snapshot = declarations;
        declarations='fn later() {} fn later() {}';
        createWGSLKernelNode(graph, {invocationCount: 16, declarations: snapshot, body: ''});
        let body = 'for(;;){}';
        body = '';
        createWGSLKernelNode(graph, {invocationCount: 1, body});
      `
    });
    const check = runChecker(directory);
    expect(check.exitCode, check.output).toBe(0);
  });

  it('still reports duplicates introduced by a replacement assignment', () => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: `
        let declarations='';
        declarations='fn shared() {} fn shared() {}';
        createWGSLKernelNode(graph, {invocationCount: 16, declarations, body: ''});
      `
    });
    const check = runChecker(directory, '--rule', 'duplicate-wgsl-declaration');
    expect(check.exitCode, check.output).toBe(1);
    expect(check.output).toContain(' shared\n');
  });

  it.each([
    'fn helper${suffix}() {} fn helper() {}',
    'fn ${suffix}helper() {} fn helper() {}',
    'fn helper${suffix}Suffix() {} fn helperSuffix() {}'
  ])('keeps unknown interpolation inside the identifier uncertain: %s', declarations => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: `
        const suffix = ['A'].join('');
        createWGSLKernelNode(graph, {invocationCount: 16, declarations: \`${declarations}\`, body: ''});
      `
    });
    const check = runChecker(directory);
    expect(check.exitCode, check.output).toBe(0);
  });

  it('keeps unknown concatenated names uncertain while detecting independent duplicates', () => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: `
        const suffix = ['A'].join('');
        const declarations = 'fn helper' + suffix + '() {} fn helper() {} fn known() {} fn known() {}';
        createWGSLKernelNode(graph, {invocationCount: 16, declarations, body: ''});
      `
    });
    const check = runChecker(directory, '--rule', 'duplicate-wgsl-declaration');
    expect(check.exitCode, check.output).toBe(1);
    expect(check.output).toContain(' known\n');
    expect(check.output).not.toContain(' helper\n');
  });

  it('handles line-comment tokens inside nested block comments without hiding later WGSL', () => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: `
        createWGSLKernelNode(graph, {
          invocationCount: 1,
          declarations: '/* note // */ fn shared() {} fn shared() {}',
          body: '/* outer /* nested // */ still outer // */ for(var row=0u;row<count;row++){}'
        });
      `
    });
    const check = runChecker(directory);
    expect(check.exitCode, check.output).toBe(1);
    expect(check.output).toContain(' shared\n');
    expect(check.output).toContain(`NEW single-invocation-data-loop ${CONTRIBUTOR_PATH}:`);
  });

  it.each([
    'fn helper(value:u32,value:u32) {}',
    'fn helper(@location(0) value:vec2<u32>, @location(1) value:u32) {}',
    'struct State { value:u32, value:f32 }',
    'struct State { @align(16) value:array<vec2<u32>, 4>, @size(16) value:f32 }'
  ])('detects duplicate WGSL parameters and members: %s', declarations => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: `createWGSLKernelNode(graph, {
        invocationCount: 16, declarations: '${declarations}', body: ''
      });`
    });
    const check = runChecker(directory, '--rule', 'duplicate-wgsl-declaration');
    expect(check.exitCode, check.output).toBe(1);
    expect(check.output).toContain(' value\n');
  });

  it('separates member and parameter lists and ignores attributes and nested type commas', () => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: `createWGSLKernelNode(graph, {
        invocationCount: 16,
        declarations: 'struct State { @align(16) value:array<vec2<u32>, 4>, other:f32 } struct Other { value:u32 } fn helper(@location(0) value:vec2<u32>, other:u32) {} fn second(value:u32) {}',
        body: ''
      });`
    });
    const check = runChecker(directory);
    expect(check.exitCode, check.output).toBe(0);
  });

  it('does not turn an opaque identifier prefix into a WGSL loop keyword', () => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: `
        const prefix = ['custom'].join('');
        createWGSLKernelNode(graph, {invocationCount: 1, body: \`\${prefix}while(rowCount) {}\`});
      `
    });
    const check = runChecker(directory);
    expect(check.exitCode, check.output).toBe(0);
  });

  it('correlates repeated unchanged destructured conditions without multiplying branches', () => {
    const fragments = Array.from(
      {length: 20},
      (_, index) =>
        '${enabled ? ' +
        JSON.stringify(`let value${index}=0u;`) +
        ':' +
        JSON.stringify(`let value${index}=1u;`) +
        '}'
    ).join('\n');
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: `function addKernel({enabled}) {
        createWGSLKernelNode(graph, {invocationCount: 16, body: \`${fragments}\`});
      }`
    });
    const check = runChecker(directory);
    expect(check.exitCode, check.output).toBe(0);
  });

  it('reports a named budget finding instead of silently skipping excessive shader branches', () => {
    const conditions = Array.from(
      {length: 10},
      (_, index) => `const choice${index} = readChoice();`
    ).join('\n');
    const fragments = Array.from(
      {length: 10},
      (_, index) =>
        '${choice' +
        index +
        ' ? ' +
        JSON.stringify(`let value${index}=0u;`) +
        ':' +
        JSON.stringify(`let value${index}=1u;`) +
        '}'
    ).join('\n');
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: `${conditions}
        createWGSLKernelNode(graph, {invocationCount: 16, body: \`${fragments}\`});`
    });
    const check = runChecker(directory, '--rule', 'shader-analysis-limit');
    expect(check.exitCode, check.output).toBe(1);
    expect(check.output).toContain(`NEW shader-analysis-limit ${CONTRIBUTOR_PATH}:`);
    expect(
      runChecker(directory, '--update-baseline', '--reason', 'Reviewed branch budget').exitCode
    ).toBe(0);
    expect(runChecker(directory).exitCode).toBe(0);
  });

  it('constrains declarations to the enclosing kernel-call branch', () => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: `function addKernel(enabled) {
        let declarations = 'fn shared() {}';
        if (enabled) { declarations += 'fn shared() {}'; }
        else { createWGSLKernelNode(graph, {invocationCount: 16, declarations, body: ''}); }
      }`
    });
    const check = runChecker(directory);
    expect(check.exitCode, check.output).toBe(0);
  });

  it('constrains loops to call guards, including ternary and logical call expressions', () => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: `function addKernel(enabled) {
        const body = enabled ? 'for(var row=0u;row<rowCount;row++){}' : '';
        if (!enabled) { createWGSLKernelNode(graph, {invocationCount: 1, body}); }
        enabled || createWGSLKernelNode(graph, {invocationCount: 1, body});
        enabled ?? createWGSLKernelNode(graph, {invocationCount: 1, body});
        enabled ? null : createWGSLKernelNode(graph, {invocationCount: 1, body});
      }`
    });
    const check = runChecker(directory);
    expect(check.exitCode, check.output).toBe(0);
  });

  it('still detects violations in a compatible guarded call', () => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: `function addKernel(enabled) {
        let declarations = 'fn shared() {}';
        if (enabled) { declarations += 'fn shared() {}'; }
        if (enabled) { createWGSLKernelNode(graph, {invocationCount: 16, declarations, body: ''}); }
      }`
    });
    const check = runChecker(directory, '--rule', 'duplicate-wgsl-declaration');
    expect(check.exitCode, check.output).toBe(1);
    expect(check.output).toContain(' shared\n');
  });

  it.each([
    ['fn helper(value: u32) { let value = 1u; }', 'value'],
    [
      'fn helper(@location(0) value: vec2<u32>) -> @location(0) u32 { let value = 1u; return value; }',
      'value'
    ],
    ['fn helper() { for (var row = 0u; row < 4u; row++) { let row = 1u; } }', 'row']
  ])('retains header declarations through the WGSL body: %s', (declarations, name) => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: `createWGSLKernelNode(graph, {
        invocationCount: 16, declarations: '${declarations}', body: ''
      });`
    });
    const check = runChecker(directory, '--rule', 'duplicate-wgsl-declaration');
    expect(check.exitCode, check.output).toBe(1);
    expect(check.output).toContain(` ${name}\n`);
  });

  it('allows nested-block shadowing and expires header names after the body', () => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: `createWGSLKernelNode(graph, {
        invocationCount: 16,
        declarations: 'fn helper(value:u32) { {let value=1u;} for(var row=0u;row<4u;row++){ {let row=1u;} } let row=1u; for(var row=0u;row<4u;row++){} } fn second(value:u32) {}',
        body: ''
      });`
    });
    const check = runChecker(directory);
    expect(check.exitCode, check.output).toBe(0);
  });

  it.each([
    'false &&',
    'true ||',
    'false ??',
    '0 &&',
    "'' &&"
  ])('does not execute a short-circuited replacement (%s)', expression => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: `
        let body = 'let row=0u; let row=1u;';
        ${expression} (body = '');
        createWGSLKernelNode(graph, {invocationCount: 16, body});
      `
    });
    const check = runChecker(directory, '--rule', 'duplicate-wgsl-declaration');
    expect(check.exitCode, check.output).toBe(1);
    expect(check.output).toContain(' row\n');
  });

  it('does not erase a loop behind a short-circuited replacement', () => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: `
        let body = 'for(var row=0u;row<rowCount;row++){}';
        false && (body = '');
        createWGSLKernelNode(graph, {invocationCount: 1, body});
      `
    });
    const check = runChecker(directory, '--rule', 'single-invocation-data-loop');
    expect(check.exitCode, check.output).toBe(1);
  });

  it.each([
    'true &&',
    'false ||',
    'null ??'
  ])('applies a replacement when its logical guard executes (%s)', expression => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: `
        let body = 'let row=0u; let row=1u;';
        ${expression} (body = '');
        createWGSLKernelNode(graph, {invocationCount: 16, body});
      `
    });
    const check = runChecker(directory);
    expect(check.exitCode, check.output).toBe(0);
  });

  it.each([
    'as const',
    'as KernelProps',
    'satisfies KernelProps'
  ])('unwraps kernel property objects using %s', wrapper => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: `
        const props = ({invocationCount: 1, body: 'for(var row=0u;row<rowCount;row++){}'} ${wrapper});
        createWGSLKernelNode(graph, props);
        createWGSLKernelNode(graph, ({invocationCount: 16, body: 'let row=0u; let row=1u;'} ${wrapper}));
      `
    });
    const check = runChecker(directory);
    expect(check.exitCode, check.output).toBe(1);
    expect(check.output).toContain(`NEW single-invocation-data-loop ${CONTRIBUTOR_PATH}:`);
    expect(check.output).toContain(' row\n');
  });

  it.each([
    'for(let index=0;index<0;index++)',
    'while(false)'
  ])('does not treat opaque control flow as a guaranteed replacement (%s)', control => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: `
          let body = 'let row=0u; let row=1u;';
          ${control} {body = '';}
          createWGSLKernelNode(graph, {invocationCount: 16, body});
        `
    });
    const check = runChecker(directory, '--rule', 'duplicate-wgsl-declaration');
    expect(check.exitCode, check.output).toBe(1);
    expect(check.output).toContain(' row\n');
  });

  it('preserves explanations when updating the baseline', () => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: kernel('for(var row=0u;row<count;row++){}')
    });
    const reason = 'Existing serial loop, retained for a separate contributor fix.';
    writeFileSync(
      join(directory, BASELINE_PATH),
      JSON.stringify({
        'single-invocation-data-loop': {[CONTRIBUTOR_PATH]: 1},
        _reasons: {'single-invocation-data-loop': {[CONTRIBUTOR_PATH]: reason}}
      })
    );
    const check = runChecker(directory, '--migrate-baseline');
    expect(check.exitCode, check.output).toBe(0);
    const baseline = JSON.parse(readFileSync(join(directory, BASELINE_PATH), 'utf8'));
    expect(baseline._version).toBe(2);
    expect(baseline['single-invocation-data-loop'][CONTRIBUTOR_PATH][0].reason).toBe(reason);
    expect(runChecker(directory, '--update-baseline').exitCode).toBe(0);
    expect(JSON.parse(readFileSync(join(directory, BASELINE_PATH), 'utf8'))).toEqual(baseline);
  });

  it('keeps baseline output stable and rejects unknown rules', () => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: kernel('for(var row=0u;row<count;row++){}')
    });
    expect(
      runChecker(directory, '--update-baseline', '--reason', 'Reviewed legacy site').exitCode
    ).toBe(0);
    const baseline = readFileSync(join(directory, BASELINE_PATH), 'utf8');
    expect(runChecker(directory, '--update-baseline').exitCode).toBe(0);
    expect(readFileSync(join(directory, BASELINE_PATH), 'utf8')).toBe(baseline);
    expect(runChecker(directory).exitCode).toBe(0);
    expect(runChecker(directory, '--all').output).toContain('single-invocation-data-loop');
    expect(runChecker(directory, '--rule', 'unknown').exitCode).toBe(2);
  });
});

describe('storage binding preflight audit', () => {
  const declaration = (body: string, extra = '') => `
    import {createTransientView as scratch} from '@luma.gl/graph/gpu-core';
    import {validateGPUStorageBindingByteLength as checkBytes} from '../kernels/gpu-contributor-utils';
    export class GPUFixture implements GPUCommandNodeProducer {
      getCommandNodes(graph) {${body}}
      ${extra}
    }
  `;
  const allocation = "const values = scratch(graph, 'rows', 'uint32', this.props.rows);";
  const check = "checkBytes(graph.device, this.id, 'rows', this.props.rows * 4, 'reduce rows');";

  it.each([
    allocation,
    `${allocation} ${check}`,
    `if (enabled) {${check}} ${allocation}`,
    `enabled && ${check} ${allocation}`,
    `for (const view of views) {${check}} ${allocation}`,
    `const later = () => {${check}}; ${allocation}`
  ])('flags missing, late, conditional and deferred checks (%s)', body => {
    const directory = createFixture({[CONTRIBUTOR_PATH]: declaration(body)});
    const result = runChecker(directory, '--rule', 'storage-binding-preflight');
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain(`storage-binding-preflight ${CONTRIBUTOR_PATH}:`);
    expect(result.output).toContain(' GPUFixture');
  });

  it.each([
    `${check} ${allocation}`,
    `if (enabled) {${check}} else {${check}} ${allocation}`,
    `if (empty) return []; ${check} ${allocation}`,
    `this.preflight(graph); ${allocation}`
  ])('accepts checks that precede declarations on every path (%s)', body => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: declaration(body, `preflight(graph) {${check}}`)
    });
    const result = runChecker(directory, '--rule', 'storage-binding-preflight');
    expect(result.exitCode, result.output).toBe(0);
  });

  it('checks row bindings even in a serial reference kernel', () => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: declaration(`
      return [createWGSLKernelNode(graph, {invocationCount: 1,
        bindings: [{name: 'rows', view: this.props.rows, type: 'f32', access: 'read'}], body: ''})];
    `)
    });
    const result = runChecker(directory, '--rule', 'storage-binding-preflight');
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain(' GPUFixture');
  });

  it('follows imported preflight helpers and keeps fixed-size-only storage outside this rule', () => {
    const directory = createFixture({
      [`${SOURCE_ROOT}/analysis-core/fixture/preflight.ts`]: `
        import {validateGraphChunkBindingRanges} from '../kernels/graph-chunk-dispatch';
        export function preflight(graph, views) {validateGraphChunkBindingRanges(graph, 'rows', views);}
      `,
      [CONTRIBUTOR_PATH]: `import {preflight as verify} from './preflight';
        ${declaration(`verify(graph, this.props); ${allocation}`)}
        export class GPUFixed { getCommandNodes(graph) {
          createTransientView(graph, 'scalar', 'uint32', 1);
        }}
      `
    });
    const result = runChecker(directory, '--rule', 'storage-binding-preflight');
    expect(result.exitCode, result.output).toBe(0);
  });

  it('finds unchecked scratch in called helpers and inherited methods', () => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: `
        function allocate(graph, rows) {return createTransientView(graph, 'rows', 'uint32', rows);}
        class Parent {getCommandNodes(graph) {return allocate(graph, this.props.rows);}}
        export class GPUFixture extends Parent {}
      `
    });
    const result = runChecker(directory, '--rule', 'storage-binding-preflight');
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain(' GPUFixture');
  });
});

it.each([
  'for (let round = 0; round < this.props.rounds; round++)',
  'for (const chunk of this.props.chunks)',
  'while (this.props.hasRows)',
  'do'
])('finds unchecked row allocations in %s loop bodies', loop => {
  const suffix = loop === 'do' ? 'while (this.props.hasRows);' : '';
  const directory = createFixture({
    [CONTRIBUTOR_PATH]: `export class GPUFixture {
    getCommandNodes(graph) { ${loop} {
      createTransientView(graph, 'rows', 'uint32', this.props.rows);
    } ${suffix} }
  }`
  });
  const result = runChecker(directory, '--all', '--rule', 'storage-binding-preflight');
  expect(result.exitCode, result.output).toBe(1);
  expect(result.output).toContain(' GPUFixture');
});

it('filters --all findings by exact file or directory, retaining shared helper resolution', () => {
  const family = `${SOURCE_ROOT}/raster-analysis/fixture`;
  const directory = createFixture({
    [CONTRIBUTOR_PATH]:
      'export class GPUOther {getCommandNodes(graph) {createTransientView(graph, "rows", "uint32", rows);}}',
    [`${family}/gpu-fixture.ts`]:
      'export class GPUFixture {getCommandNodes(graph) {createTransientView(graph, "rows", "uint32", rows);}}',
    [`${family}-other/gpu-fixture.ts`]:
      'export class GPUUnrelated {getCommandNodes(graph) {createTransientView(graph, "rows", "uint32", rows);}}'
  });
  for (const selected of [family, `${family}/gpu-fixture.ts`]) {
    const result = runChecker(
      directory,
      '--all',
      '--rule',
      'storage-binding-preflight',
      '--paths',
      selected
    );
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain(' GPUFixture');
    expect(result.output).not.toContain(' GPUOther');
    expect(result.output).not.toContain(' GPUUnrelated');
  }
  expect(runChecker(directory, '--paths').exitCode).toBe(2);
  expect(runChecker(directory, '--paths', family, '--update-baseline').exitCode).toBe(2);
});

it('recognises imported named device-capacity preflights through wrapper return chains', () => {
  const directory = createFixture({
    [`${SOURCE_ROOT}/analysis-core/fixture/preflight.ts`]: `
      import {getStorageBindingElementLimit, validateCapacity} from '../kernels/gpu-contributor-utils';
      function maximumRows(device) {return Math.min(getStorageBindingElementLimit(device, 16), 1000000);}
      export function validateRows(id, device, rows) {validateCapacity(id, 'rows', rows, 0, maximumRows(device));}
    `,
    [CONTRIBUTOR_PATH]: `import {validateRows as check} from './preflight';
      export class GPUFixture {getCommandNodes(graph) {
        check(this.id, graph.device, this.props.rows);
        createTransientView(graph, 'scratch', 'float32', this.props.rows);
      }}`
  });
  const result = runChecker(
    directory,
    '--all',
    '--rule',
    'storage-binding-preflight',
    '--paths',
    CONTRIBUTOR_PATH
  );
  expect(result.exitCode, result.output).toBe(0);
});

it('recognises preflight loops over nonempty scratch lists and imported helper parameters', () => {
  const directory = createFixture({
    [`${SOURCE_ROOT}/analysis-core/fixture/preflight.ts`]: `
      import {validateGPUStorageBindingByteLength} from '../kernels/gpu-contributor-utils';
      export function validateBindings(graph, views) {
        for (const view of views) {
          validateGPUStorageBindingByteLength(graph.device, 'rows', 'input', view.length * 4, 'reduce rows');
        }
      }`,
    [CONTRIBUTOR_PATH]: `import {validateBindings} from './preflight';
      export class GPUFixture {getCommandNodes(graph) {
        validateBindings(graph, [this.props.input]);
        for (const bytes of [this.props.rows * 4, this.props.capacity * 8]) {
          validateGPUStorageBindingByteLength(graph.device, 'rows', 'scratch', bytes, 'reduce rows');
        }
        createTransientView(graph, 'scratch', 'float32', this.props.rows);
      }}`
  });
  const result = runChecker(directory, '--all', '--rule', 'storage-binding-preflight');
  expect(result.exitCode, result.output).toBe(0);
});

it('recognises shared aligned-range preflight loops with optional-view guards', () => {
  const directory = createFixture({
    [`${SOURCE_ROOT}/analysis-core/fixture/preflight.ts`]: `
      export function validateViews(graph, views) {
        for (const view of views) {
          if (view) validateGPUStorageBindingByteLength(graph.device, 'rows', 'view', getViewBindingRange(view).size, 'reduce rows');
        }
      }`,
    [CONTRIBUTOR_PATH]: `import {validateViews} from './preflight';
      export class GPUFixture {getCommandNodes(graph) {
        validateViews(graph, this.props.views);
        createTransientView(graph, 'scratch', 'float32', this.props.rows);
      }}`
  });
  const result = runChecker(directory, '--all', '--rule', 'storage-binding-preflight');
  expect(result.exitCode, result.output).toBe(0);
});

it('does not recognise an aligned-range loop guarded by an unrelated condition', () => {
  const directory = createFixture({
    [CONTRIBUTOR_PATH]: `
    export class GPUFixture {getCommandNodes(graph) {
      for (const view of this.props.views) {
        if (enabled) validateGPUStorageBindingByteLength(graph.device, 'rows', 'view', getViewBindingRange(view).size, 'reduce rows');
      }
      createTransientView(graph, 'scratch', 'float32', this.props.rows);
    }}`
  });
  const result = runChecker(directory, '--all', '--rule', 'storage-binding-preflight');
  expect(result.exitCode, result.output).toBe(1);
});

it('follows a family metadata-visitor preflight without confusing Set.add with graph.add', () => {
  const directory = createFixture({
    [`${SOURCE_ROOT}/gpu-spatial-analysis/spatial-weights/spatial-statistics-bindings.ts`]: `
      import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
      export function validateSpatialStatisticsBindings(graph, id, props) {
        const seen = new Set();
        const fields = {};
        const visit = value => {
          if (!value || seen.has(value)) return;
          seen.add(value);
          fields.input = value;
        };
        visit(props);
        validateGraphChunkBindingRanges(graph, id, fields);
      }`,
    [`${SOURCE_ROOT}/gpu-spatial-analysis/spatial-autocorrelation/gpu-fixture.ts`]: `
      import {validateSpatialStatisticsBindings} from '../spatial-weights/spatial-statistics-bindings';
      export class GPUFixture {getCommandNodes(graph) {
        validateSpatialStatisticsBindings(graph, this.id, this.props);
        return [createWGSLKernelNode(graph, {invocationCount: rows, bindings: [], body: ''})];
      }}`
  });
  const result = runChecker(
    directory,
    '--all',
    '--rule',
    'storage-binding-preflight',
    '--paths',
    `${SOURCE_ROOT}/gpu-spatial-analysis/spatial-autocorrelation`
  );
  expect(result.exitCode, result.output).toBe(0);
});
