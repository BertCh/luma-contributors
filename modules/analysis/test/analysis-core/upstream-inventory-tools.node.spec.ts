// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {spawnSync} from 'node:child_process';
import {mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join, resolve} from 'node:path';
import {expect, it} from 'vitest';

const REPOSITORY_ROOT = resolve(import.meta.dirname, '../../../..');

function createToolFixture() {
  const directory = mkdtempSync(join(tmpdir(), 'upstream-inventory-review-'));
  const write = (path: string, contents: string) => {
    const destination = join(directory, path);
    mkdirSync(resolve(destination, '..'), {recursive: true});
    writeFileSync(destination, contents);
  };
  const run = (name: string, argumentsList: string[]) => {
    const result = spawnSync(
      process.execPath,
      [join(REPOSITORY_ROOT, `scripts/upstream/${name}.mjs`), ...argumentsList],
      {cwd: directory, encoding: 'utf8'}
    );
    expect(result.status, result.stderr).toBe(0);
    return result.stdout;
  };
  return {directory, write, run, destroy: () => rmSync(directory, {recursive: true, force: true})};
}

it('censuses adjacent math and luma imports as separate declarations', () => {
  const fixture = createToolFixture();
  try {
    fixture.write('exports.json', JSON.stringify({'@luma.gl/core': {log: {kind: 'const'}}}));
    fixture.write(
      'modules/analysis/src/imports.ts',
      "import {toRadians} from '@math.gl/core';\nimport {log} from '@luma.gl/core';\n"
    );
    fixture.run('usage', ['exports.json', 'usage.json', 'report.json']);
    expect(JSON.parse(readFileSync(join(fixture.directory, 'usage.json'), 'utf8')).usage).toEqual({
      '@luma.gl/core#log': {'analysis-src': {value: 1, type: 0}}
    });
    fixture.write(
      'docs/example.md',
      "```sh\nimport {shellOnly} from '@luma.gl/core';\n```\n```ts\nimport type {Device} from '@luma.gl/core';\n```\n"
    );
    fixture.run('usage', ['exports.json', 'usage.json', 'report.json']);
    expect(JSON.parse(readFileSync(join(fixture.directory, 'usage.json'), 'utf8')).usage).toEqual({
      '@luma.gl/core#log': {'analysis-src': {value: 1, type: 0}},
      '@luma.gl/core#Device': {docs: {value: 0, type: 1}}
    });
  } finally {
    fixture.destroy();
  }
});

it('review LUMA-105: attributes usage to the exact entry point, not to every entry with the name', () => {
  const fixture = createToolFixture();
  try {
    // The graph root re-exports `GPUScan` from the same declaration file as `gpu-core`.
    fixture.write(
      'exports.json',
      JSON.stringify({
        '@luma.gl/graph': {
          GPUScan: {kind: 'class', file: 'dist/gpu-core/gpu-scan.d.ts'},
          GPUSort: {kind: 'class', file: 'dist/gpu-core/gpu-sort.d.ts'}
        },
        '@luma.gl/graph/gpu-core': {
          GPUScan: {kind: 'class', file: 'dist/gpu-core/gpu-scan.d.ts'},
          GPUSort: {kind: 'class', file: 'dist/gpu-core/gpu-sort.d.ts'}
        }
      })
    );
    fixture.write(
      'modules/analysis/src/imports.ts',
      "import {GPUScan} from '@luma.gl/graph/gpu-core';\nimport {GPUSort} from '@luma.gl/graph';\n"
    );
    fixture.run('usage', ['exports.json', 'usage.json', 'report.json']);
    const report = JSON.parse(readFileSync(join(fixture.directory, 'report.json'), 'utf8'));
    // Before the fix both entries reported usedInSrc 2: usage was joined by bare name.
    expect(report['@luma.gl/graph']).toEqual({
      values: 2,
      usedInSrc: 1,
      unused: ['GPUScan'],
      testOnly: [],
      usedThroughOtherEntries: {GPUScan: ['@luma.gl/graph/gpu-core']}
    });
    expect(report['@luma.gl/graph/gpu-core']).toEqual({
      values: 2,
      usedInSrc: 1,
      unused: ['GPUSort'],
      testOnly: [],
      usedThroughOtherEntries: {GPUSort: ['@luma.gl/graph']}
    });
  } finally {
    fixture.destroy();
  }
});

it('review LUMA-112: diff reports changed declarations and string values under unchanged names', () => {
  const fixture = createToolFixture();
  try {
    const writeVersion = (root: string, signature: string, wgsl: string) => {
      for (const name of ['core', 'graph', 'shaders', 'engine']) {
        fixture.write(
          `${root}/${name}/package.json`,
          JSON.stringify({exports: {'.': {types: './index.d.ts', import: './index.js'}}})
        );
        fixture.write(
          `${root}/${name}/index.d.ts`,
          `export declare function getCell(${signature}): number;\nexport declare const GRID_WGSL: string;\nexport declare const STABLE_WGSL: string;\n`
        );
        fixture.write(
          `${root}/${name}/index.js`,
          `export function getCell() { return 0; }\nexport const GRID_WGSL = ${JSON.stringify(wgsl)};\nexport const STABLE_WGSL = 'fn stable() {}';\n`
        );
      }
    };
    writeVersion('old', 'x: number', 'fn cell(x: f32) -> f32 { return floor(x / size); }');
    writeVersion(
      'new',
      'x: number, size: number',
      'fn cell(x: f32) -> f32 { return floor(x * inverseSize); }'
    );
    fixture.run('exports', ['old', 'old.json']);
    fixture.run('exports', ['new', 'new.json']);
    // Names and kinds are identical, so the name-only diff (the only diff before) prints nothing.
    expect(fixture.run('diff', ['old.json', 'new.json'])).toBe('');
    const output = fixture.run('diff', ['old.json', 'new.json', 'old', 'new']);
    expect(output).toMatch(/^CHANGED DECLARATIONS 4\n/);
    expect(output).toMatch(
      /^ {2}@luma\.gl\/graph getCell\(function\) [0-9a-f]{12} -> [0-9a-f]{12}$/m
    );
    expect(output).toContain('CHANGED STRING VALUES 4\n');
    expect(output).toMatch(/^ {2}@luma\.gl\/graph GRID_WGSL [0-9a-f]{12} -> [0-9a-f]{12}$/m);
    expect(output).not.toContain('STABLE_WGSL');
    expect(output).not.toContain('UNLOADED ENTRY');
    expect(fixture.run('diff', ['old.json', 'old.json', 'old', 'old'])).toBe(
      'CHANGED DECLARATIONS 0\nCHANGED STRING VALUES 0\n'
    );
  } finally {
    fixture.destroy();
  }
});

it('keeps equivalent conditional and string export targets and reports unresolved public entries', () => {
  const fixture = createToolFixture();
  try {
    for (const name of ['core', 'graph', 'shaders', 'engine']) {
      fixture.write(
        `packages/${name}/package.json`,
        JSON.stringify({exports: {'.': {types: './dist/index.d.ts', import: './dist/index.js'}}})
      );
      fixture.write(
        `packages/${name}/dist/index.d.ts`,
        'export declare const publicValue: number;\n'
      );
      fixture.write(`packages/${name}/dist/index.js`, 'export const publicValue = 1;\n');
    }
    fixture.run('exports', ['packages', 'conditional.json']);
    fixture.write(
      'packages/core/package.json',
      JSON.stringify({exports: {'.': './dist/index.js', './missing': './missing.js'}})
    );
    const output = fixture.run('exports', ['packages', 'string.json']);
    const previous = JSON.parse(readFileSync(join(fixture.directory, 'conditional.json'), 'utf8'));
    const current = JSON.parse(readFileSync(join(fixture.directory, 'string.json'), 'utf8'));
    expect(current['@luma.gl/core']).toEqual(previous['@luma.gl/core']);
    expect(current).toHaveProperty('@luma.gl/core/missing', null);
    expect(output).toContain('@luma.gl/core/missing unresolved');
    expect(fixture.run('diff', ['conditional.json', 'string.json'])).not.toContain(
      'REMOVED ENTRY @luma.gl/core'
    );
    fixture.write('packages/core/package.json', JSON.stringify({exports: {'.': './absent.js'}}));
    fixture.run('exports', ['packages', 'unresolved.json']);
    const unresolvedDifference = fixture.run('diff', ['conditional.json', 'unresolved.json']);
    expect(unresolvedDifference).toContain('UNRESOLVED ENTRY @luma.gl/core');
    expect(unresolvedDifference).not.toContain('REMOVED ENTRY @luma.gl/core');
  } finally {
    fixture.destroy();
  }
});

it('runs the maintained export, usage and diff tools on public entries', () => {
  const fixtureDirectory = mkdtempSync(join(tmpdir(), 'upstream-inventory-'));
  const writeFixture = (path: string, contents: string) => {
    const destination = join(fixtureDirectory, path);
    mkdirSync(resolve(destination, '..'), {recursive: true});
    writeFileSync(destination, contents);
  };
  const runTool = (name: string, argumentsList: string[]) => {
    const result = spawnSync(
      process.execPath,
      [join(REPOSITORY_ROOT, `scripts/upstream/${name}.mjs`), ...argumentsList],
      {
        cwd: fixtureDirectory,
        encoding: 'utf8'
      }
    );
    expect(result.status, result.stderr).toBe(0);
    return result.stdout;
  };
  try {
    for (const packageName of ['core', 'graph', 'shaders', 'engine']) {
      writeFixture(
        `packages/${packageName}/package.json`,
        JSON.stringify({
          exports: {
            '.': {types: './index.d.ts'},
            './_internal': {types: './private.d.ts'}
          }
        })
      );
      writeFixture(
        `packages/${packageName}/index.d.ts`,
        'export declare class PublicValue {}\nexport type PublicType = number;\n'
      );
      writeFixture(`packages/${packageName}/private.d.ts`, 'export declare class HiddenValue {}\n');
    }
    const exportPath = join(fixtureDirectory, 'exports.json');
    runTool('exports', [join(fixtureDirectory, 'packages'), exportPath]);
    const inventory = JSON.parse(readFileSync(exportPath, 'utf8'));
    expect(Object.keys(inventory).sort()).toEqual([
      '@luma.gl/core',
      '@luma.gl/engine',
      '@luma.gl/graph',
      '@luma.gl/shaders'
    ]);
    expect(inventory['@luma.gl/core']).toEqual({
      PublicValue: {kind: 'class', file: 'index.d.ts'},
      PublicType: {kind: 'type', file: 'index.d.ts'}
    });
    writeFixture(
      'modules/analysis/src/example.ts',
      "import {PublicValue, type PublicType} from '@luma.gl/core';\n"
    );
    const usagePath = join(fixtureDirectory, 'usage.json');
    const reportPath = join(fixtureDirectory, 'report.json');
    runTool('usage', [exportPath, usagePath, reportPath]);
    expect(JSON.parse(readFileSync(usagePath, 'utf8')).usage).toEqual({
      '@luma.gl/core#PublicValue': {'analysis-src': {value: 1, type: 0}},
      '@luma.gl/core#PublicType': {'analysis-src': {value: 0, type: 1}}
    });
    expect(JSON.parse(readFileSync(reportPath, 'utf8'))['@luma.gl/core']).toEqual({
      values: 1,
      usedInSrc: 1,
      unused: [],
      testOnly: [],
      usedThroughOtherEntries: {}
    });
    writeFixture(
      'previous.json',
      JSON.stringify({'@luma.gl/core': {RemovedValue: {kind: 'const'}}})
    );
    expect(runTool('diff', [join(fixtureDirectory, 'previous.json'), exportPath])).toContain(
      '+ PublicValue(class), PublicType(type)\n  - RemovedValue'
    );
  } finally {
    rmSync(fixtureDirectory, {recursive: true, force: true});
  }
});
