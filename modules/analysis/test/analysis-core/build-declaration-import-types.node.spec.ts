// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {dirname, join} from 'node:path';
import {afterEach, expect, it} from 'vitest';
import {checkDeclarationEdges} from '../../../../scripts/check-packed-consumer.mjs';
import {rewriteDeclarationImportTypes} from '../../../../scripts/typescript-project-inputs.mjs';

const directories: string[] = [];
const configuration = JSON.parse(
  readFileSync(new URL('../../../../tsconfig.json', import.meta.url), 'utf8')
);
const extensions: string[] = configuration.compilerOptions.plugins.find(
  (plugin: {afterDeclarations?: boolean}) => plugin.afterDeclarations
).extensions;

function write(directory: string, file: string, source = 'export type Marker = true;') {
  const target = join(directory, file);
  mkdirSync(dirname(target), {recursive: true});
  writeFileSync(target, source);
  return target;
}

function fixture() {
  const directory = mkdtempSync(join(tmpdir(), 'declaration-import-types-'));
  directories.push(directory);
  write(directory, 'package.json', '{"type":"module"}');
  return directory;
}

afterEach(() => {
  for (const directory of directories.splice(0)) rmSync(directory, {recursive: true, force: true});
});

it('TOOL-102 qualifies relative import types that have a real file or index declaration', () => {
  const directory = fixture();
  for (const target of [
    'dist/index.d.ts',
    'dist/entry/index.d.ts',
    'dist/entry/file.d.ts',
    'dist/entry/file/index.d.ts',
    'dist/entry/folder/index.d.ts',
    'dist/entry/capture.lens.d.ts',
    'dist/entry/demo.example.d.ts',
    'dist/entry/start.run.d.ts',
    'dist/entry/presets.scenarios.d.ts'
  ]) {
    write(directory, target);
  }
  const source = [
    'export type File = import("./file").Marker;',
    "export type Folder = import('./folder').Marker;",
    'export type Here = import(".").Marker;',
    'export type Parent = import("..").Marker;',
    'export type Lens = import("./capture.lens").Marker;',
    "export type Example = import('./demo.example').Marker;",
    'export type Run = import("./start.run").Marker;',
    'export type Scenarios = import("./presets.scenarios").Marker;',
    "export type Extended = import('./file.js').Marker;",
    'export type Package = import("@scope/package/file").Marker;',
    'export type Text = \'import("./file")\';'
  ].join('\n');
  const declaration = write(directory, 'dist/entry/barrel.d.ts', source);

  expect(() => checkDeclarationEdges(directory)).toThrow('dist/entry/barrel.d.ts:2 "./folder"');
  const rewritten = rewriteDeclarationImportTypes(source, declaration, extensions);
  expect(rewritten).toBe(
    [
      'export type File = import("./file.js").Marker;',
      "export type Folder = import('./folder/index.js').Marker;",
      'export type Here = import("./index.js").Marker;',
      'export type Parent = import("../index.js").Marker;',
      'export type Lens = import("./capture.lens.js").Marker;',
      "export type Example = import('./demo.example.js').Marker;",
      'export type Run = import("./start.run.js").Marker;',
      'export type Scenarios = import("./presets.scenarios.js").Marker;',
      "export type Extended = import('./file.js').Marker;",
      'export type Package = import("@scope/package/file").Marker;',
      'export type Text = \'import("./file")\';'
    ].join('\n')
  );
  expect(rewriteDeclarationImportTypes(rewritten, declaration, extensions)).toBe(rewritten);
  writeFileSync(declaration, rewritten);
  expect(checkDeclarationEdges(directory).edgeCount).toBe(9);

  const unresolved = 'export type Missing = import("./missing").Marker;';
  expect(rewriteDeclarationImportTypes(unresolved, declaration, extensions)).toBe(unresolved);
});

it.each([
  ['cts', 'cjs'],
  ['mts', 'mjs']
])('TOOL-102 selects native .d.%s targets from disk', (declarationSuffix, runtimeSuffix) => {
  const directory = fixture();
  write(directory, `dist/file.d.${declarationSuffix}`);
  write(directory, `dist/folder/index.d.${declarationSuffix}`);
  write(directory, `dist/regular.d.${declarationSuffix}`);
  write(directory, 'dist/regular.d.ts');
  const source =
    "export type Native = [import('./file').Marker, import('./folder/').Marker, import('./regular').Marker];";
  const declaration = write(directory, `dist/index.d.${declarationSuffix}`, source);
  const rewritten = rewriteDeclarationImportTypes(source, declaration, extensions);
  expect(rewritten).toBe(
    `export type Native = [import('./file.${runtimeSuffix}').Marker, import('./folder/index.${runtimeSuffix}').Marker, import('./regular.js').Marker];`
  );
  writeFileSync(declaration, rewritten);
  expect(checkDeclarationEdges(directory).edgeCount).toBe(3);
});
