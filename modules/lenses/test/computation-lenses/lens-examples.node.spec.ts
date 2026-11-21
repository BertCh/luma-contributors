// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {existsSync, readFileSync} from 'node:fs';
import {describe, expect, test} from 'vitest';
import {
  extractLensExample,
  findNonPublicImports,
  getLensExamplePath,
  LENS_EXAMPLE_MAX_LINES
} from '../../src/computation-lenses/lens-example';
import {listLensScenarios} from '../../src/computation-lenses/index';

const region = (code: string) => `// #region example\n${code}\n// #endregion example`;

describe('lens example mechanism', () => {
  test('extracts and dedents only the shown block, including CRLF', () => {
    expect(
      extractLensExample(
        `import {Device} from '@luma.gl/core';\r\n${region('    const result = await run();\n      return result;').replace(/\n/g, '\r\n')}`
      )
    ).toEqual({
      code: 'const result = await run();\n  return result;',
      language: 'ts',
      lineCount: 2
    });
    expect(
      extractLensExample(region(Array.from({length: 25}, () => 'run();').join('\n'))).lineCount
    ).toBe(LENS_EXAMPLE_MAX_LINES);
  });
  test('rejects missing, duplicated, empty, reversed and overlong regions', () => {
    expect(() => extractLensExample('const value = 1;')).toThrow(/missing/i);
    expect(() => extractLensExample(`${region('run();')}\n${region('run();')}`)).toThrow(
      /duplicated/i
    );
    expect(() =>
      extractLensExample(region(Array.from({length: 26}, () => 'run();').join('\n')))
    ).toThrow(/25 lines/);
    expect(() => extractLensExample(region(''))).toThrow(/empty/i);
    expect(() => extractLensExample(region(' \n  '))).toThrow(/empty/i);
    expect(() => extractLensExample(region('\nrun();'))).toThrow(/blank first or last/);
    expect(() => extractLensExample(region('run();\n'))).toThrow(/blank first or last/);
    expect(() => extractLensExample('// #endregion example\n// #region example')).toThrow(
      /precedes/
    );
  });
  test('accepts only public entry imports, including dynamic imports and re-exports', () => {
    for (const kind of ['import', 'require']) {
      expect(findNonPublicImports(`${kind}('@luma.gl/graph/no-such-entry');`)).toEqual([
        '@luma.gl/graph/no-such-entry'
      ]);
      expect(findNonPublicImports(`${kind}('@luma.gl/graph/gpu-core');`)).toEqual([]);
    }
    expect(
      findNonPublicImports(`
      import '@luma-contributors/lenses/bogus/computation-pack/index';
      import '@luma-contributors/lenses/computation-lenses/data/example.json';
      require('@luma-contributors/lenses/computation-lenses/registry');
    `)
    ).toEqual([
      '@luma-contributors/lenses/bogus/computation-pack/index',
      '@luma-contributors/lenses/computation-lenses/data/example.json',
      '@luma-contributors/lenses/computation-lenses/registry'
    ]);
    expect(
      findNonPublicImports(`
      import '@luma-contributors/analysis/gpu-spatial-analysis/not-published';
      export * from '@luma-contributors/lenses/computation-views/hosts/mount-lens';
      import '@luma-contributors/analysis/not-an-entry';
    `)
    ).toEqual([
      '@luma-contributors/analysis/gpu-spatial-analysis/not-published',
      '@luma-contributors/lenses/computation-views/hosts/mount-lens',
      '@luma-contributors/analysis/not-an-entry'
    ]);
    expect(
      findNonPublicImports(`
      import '@luma-contributors/lenses/gpu-spatial-analysis/computation-pack/index';
      import '@luma-contributors/lenses/computation-lenses/data/s1-chicago-tracts.json' with {type: 'json'};
    `)
    ).toEqual([]);
    expect(findNonPublicImports('import(`./private`);')).toEqual(['./private']);
    expect(
      findNonPublicImports(`import type {Device} from '@luma.gl/core';
      import {Model} from '@luma.gl/engine';
      import '@luma.gl/graph';
      import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
      export {Result} from '@luma-contributors/analysis/computation-record';
      const module = import('@luma.gl/graph/gpu-raster');`)
    ).toEqual([]);
    expect(
      findNonPublicImports(`import './local';
      import type {Hidden} from '@luma.gl/graph/src/private';
      export * from '@luma.gl/graph/dist/index';
      import('@luma.gl/graph/test/fixture');
      import {readFile} from 'node:fs';
      import '@luma.gl/gpgpu';
      import '/absolute';
      const local = require('../private');`)
    ).toEqual([
      './local',
      '@luma.gl/graph/src/private',
      '@luma.gl/graph/dist/index',
      '@luma.gl/graph/test/fixture',
      'node:fs',
      '@luma.gl/gpgpu',
      '/absolute',
      '../private'
    ]);
    expect(
      findNonPublicImports(
        `// import './comment';\n/* import './block'; */\nconst text = "import './string';";`
      )
    ).toEqual([]);
  });
  test('R8 checked module specifiers must be single literals, including dynamic import and require', () => {
    for (const loader of ['import', 'require']) {
      expect(findNonPublicImports(`${loader}('@luma.gl/graph/gpu-core');`)).toEqual([]);
      expect(
        findNonPublicImports(`${loader}('@luma.gl/graph/gpu-core' + '/no-such-entry');`)
      ).toEqual(['@luma.gl/graph/gpu-core (module specifier must be a single literal)']);
      expect(findNonPublicImports(`${loader}(moduleName);`)).toEqual([
        'module specifier must be a single literal'
      ]);
    }
  });
  test('derives sibling paths from entry and element, ignoring variant', () => {
    expect(getLensExamplePath('gpu-spatial-analysis.local-moran.variant')).toBe(
      'scenarios/gpu-spatial-analysis/local-moran.example.ts'
    );
    expect(() => getLensExamplePath('../private')).toThrow('Invalid scenario id');
  });
  test('every registered scenario has a checked public-API example', async () => {
    const root = new URL('../../src/computation-lenses/', import.meta.url);
    const index = new URL('scenarios/index.ts', root);
    if (existsSync(index)) await import(index.href);
    const scenarios = listLensScenarios();
    expect(Array.isArray(scenarios)).toBe(true);
    for (const scenario of scenarios) {
      const path = new URL(scenario.example?.path ?? getLensExamplePath(scenario.id), root);
      expect(existsSync(path), `${scenario.id}: ${path.pathname}`).toBe(true);
      const source = readFileSync(path, 'utf8');
      expect(extractLensExample(source).lineCount, scenario.id).toBeLessThanOrEqual(
        LENS_EXAMPLE_MAX_LINES
      );
      expect(findNonPublicImports(source), scenario.id).toEqual([]);
      expect(source, scenario.id).toContain('export async function runExample');
    }
  });
});
