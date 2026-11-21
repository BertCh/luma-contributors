// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {mkdtempSync, rmSync, writeFileSync} from 'node:fs';
import {createRequire, SourceMap} from 'node:module';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {pathToFileURL} from 'node:url';
import typescript from 'typescript';
import {afterEach, describe, expect, it} from 'vitest';
import {
  rewriteRelativeModuleSpecifiers,
  transformCommonJSModule
} from '../../../../scripts/build-commonjs.mjs';

const directories: string[] = [];

afterEach(() => {
  for (const directory of directories.splice(0)) rmSync(directory, {recursive: true, force: true});
});

// Values that look like imports. The expected values are the literals themselves.
const IMPORT_LIKE_VALUES = {
  doubleQuoted: "import x from './y.js'",
  extensionless: "import x from './y'",
  sideEffect: "import './sibling.js'",
  dynamic: "await import('./sibling.js')",
  template: "export {x} from '../z.js'; let s = require('./w.js');"
};

const ESM_SOURCE = `import {marker} from './sibling.js';
export {marker as reexported} from './sibling.js';
export const doubleQuoted = ${JSON.stringify(IMPORT_LIKE_VALUES.doubleQuoted)};
export const extensionless = "import x from './y'";
export const sideEffect = "import './sibling.js'";
export const dynamic = "await import('./sibling.js')";
export const template = \`export {x} from '../z.js'; let s = require('./w.js');\`;
export const imported = marker;
export async function loadMarker() {
  return (await import('./sibling.js')).marker;
}
`;

describe('CommonJS build transform', () => {
  it('TOOL-101 rewrites parsed module specifiers only, leaving import-like strings byte-for-byte', async () => {
    const directory = mkdtempSync(join(tmpdir(), 'build-commonjs-'));
    directories.push(directory);
    writeFileSync(join(directory, 'package.json'), '{"type":"module"}');
    // Different markers prove which sibling each format actually loaded.
    writeFileSync(join(directory, 'sibling.js'), "export const marker = 'esm';");
    writeFileSync(join(directory, 'sibling.cjs'), "exports.marker = 'cjs';");
    writeFileSync(join(directory, 'probe.js'), ESM_SOURCE);
    const {code} = await transformCommonJSModule(ESM_SOURCE, join(directory, 'probe.js'));
    writeFileSync(join(directory, 'probe.cjs'), code);

    const esm = await import(pathToFileURL(join(directory, 'probe.js')).href);
    const commonjs = createRequire(import.meta.url)(join(directory, 'probe.cjs'));
    for (const module of [esm, commonjs]) {
      for (const [name, value] of Object.entries(IMPORT_LIKE_VALUES)) {
        expect(module[name], name).toBe(value);
      }
    }
    expect([esm.imported, esm.reexported, await esm.loadMarker()]).toEqual(['esm', 'esm', 'esm']);
    expect([commonjs.imported, commonjs.reexported, await commonjs.loadMarker()]).toEqual([
      'cjs',
      'cjs',
      'cjs'
    ]);
  });

  it('TOOL-101 keeps quote style and leaves bare and non-JavaScript specifiers alone', () => {
    const source = [
      "import a from './a.js';",
      'import b from "../b.js";',
      "import c from 'package/c.js';",
      "import d from './d.json' with {type: 'json'};",
      "const e = import('./e.js');",
      "const f = `import('./f.js')`;"
    ].join('\n');
    expect(rewriteRelativeModuleSpecifiers(source, 'probe.js')).toBe(
      [
        "import a from './a.cjs';",
        'import b from "../b.cjs";',
        "import c from 'package/c.js';",
        "import d from './d.json' with {type: 'json'};",
        "const e = import('./e.cjs');",
        "const f = `import('./f.js')`;"
      ].join('\n')
    );
  });

  it('composes a linked CommonJS map through TypeScript and a rewritten dynamic import', async () => {
    const source = [
      "import {marker} from './sibling.js';",
      'export async function probe(value: number) {',
      "  return (await import('./sibling.js')).marker + value;",
      '}',
      'export const result = marker;'
    ].join('\n');
    const emitted = typescript.transpileModule(source, {
      fileName: 'probe.ts',
      compilerOptions: {
        target: typescript.ScriptTarget.ES2022,
        module: typescript.ModuleKind.ES2022,
        sourceMap: true,
        inlineSources: true
      }
    });
    expect(emitted.sourceMapText).toBeDefined();
    const {code, map} = await transformCommonJSModule(
      emitted.outputText,
      'probe.js',
      emitted.sourceMapText
    );
    expect(code).toContain('require("./sibling.cjs")');
    expect(code).toContain('import("./sibling.cjs")');
    expect(code).toContain('//# sourceMappingURL=probe.cjs.map');
    const payload = JSON.parse(map);
    expect(payload.sources).toEqual(['probe.ts']);
    expect(payload.sourcesContent).toEqual([source]);
    const generatedLines = code.split('\n');
    const generatedLine = generatedLines.findIndex(line => line.includes('.marker + value'));
    expect(generatedLine).toBeGreaterThanOrEqual(0);
    const generatedColumn = generatedLines[generatedLine].lastIndexOf('value');
    const originalColumn = source.split('\n')[2].lastIndexOf('value');
    expect(new SourceMap(payload).findEntry(generatedLine, generatedColumn)).toMatchObject({
      originalLine: 2,
      originalColumn
    });
  });
});
