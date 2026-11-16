// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {spawnSync} from 'node:child_process';
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

  it('keeps count budgets for unfinished rules and rejects increases', () => {
    const directory = createFixture(
      {[CONTRIBUTOR_PATH]: kernel('for (var row = 0u; row < rowCount; row++) {}')},
      {'single-invocation-data-loop': {[CONTRIBUTOR_PATH]: 1}}
    );
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
          // contract-allow: single-invocation -- bounded dense system
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
    writeFileSync(
      join(directory, BASELINE_PATH),
      JSON.stringify({
        'shader-analysis-limit': {[CONTRIBUTOR_PATH]: 1}
      })
    );
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
        _reasons: {'single-invocation-data-loop': {[CONTRIBUTOR_PATH]: reason}}
      })
    );
    const check = runChecker(directory, '--update-baseline');
    expect(check.exitCode, check.output).toBe(0);
    expect(JSON.parse(readFileSync(join(directory, BASELINE_PATH), 'utf8'))._reasons).toEqual({
      'single-invocation-data-loop': {[CONTRIBUTOR_PATH]: reason}
    });
  });

  it('keeps baseline output stable and rejects unknown rules', () => {
    const directory = createFixture({
      [CONTRIBUTOR_PATH]: kernel('for(var row=0u;row<count;row++){}')
    });
    expect(runChecker(directory, '--update-baseline').exitCode).toBe(0);
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
