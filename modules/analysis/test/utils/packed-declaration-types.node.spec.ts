// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {afterEach, expect, it} from 'vitest';
import typescript from 'typescript';
import {checkDeclarationEdges} from '../../../../scripts/check-packed-consumer.mjs';

const directories: string[] = [];
afterEach(() => {
  for (const directory of directories.splice(0)) rmSync(directory, {recursive: true, force: true});
});

function createPackedFixture() {
  const directory = mkdtempSync(join(tmpdir(), 'packed-types-'));
  directories.push(directory);
  const packed = join(directory, 'package');
  mkdirSync(packed);
  writeFileSync(join(packed, 'package.json'), '{"type":"module"}');
  return {directory, packed};
}

it.each([
  'present',
  'missing',
  'escaping',
  'esm'
])('W2-01 checks relative type-reference directives (%s)', state => {
  const {directory, packed} = createPackedFixture();
  const extension = state === 'esm' ? 'mts' : 'cts';
  const declaration = `index.d.${extension}`;
  writeFileSync(
    join(packed, declaration),
    '// fixture\n/// <reference types="./ambient" />\nexport {};'
  );
  if (state === 'present' || state === 'esm')
    writeFileSync(join(packed, 'ambient.d.ts'), 'export {};');
  if (state === 'escaping') {
    writeFileSync(join(directory, 'outside.d.ts'), 'export {};');
    symlinkSync(join(directory, 'outside.d.ts'), join(packed, 'ambient.d.ts'));
  }
  // CommonJS type references permit extensionless .d.ts targets; native ESM does not.
  // Each fixture has one directive, independently of its resolution or containment outcome.
  if (state === 'present') {
    expect(checkDeclarationEdges(packed)).toEqual({declarationCount: 2, edgeCount: 1});
  } else {
    expect(() => checkDeclarationEdges(packed)).toThrow('1 of 1 relative declaration references');
    expect(() => checkDeclarationEdges(packed)).toThrow(
      `${declaration}:2 reference types "./ambient"`
    );
  }
});

it('B1 resolves a relative type-name collision inside the isolated package', () => {
  const {packed} = createPackedFixture();
  writeFileSync(join(packed, 'index.d.cts'), '/// <reference types="./node" />\nexport {};');
  writeFileSync(join(packed, 'node.d.ts'), 'export {};');
  expect(checkDeclarationEdges(packed)).toEqual({declarationCount: 2, edgeCount: 1});
});

it('B1 uses the native consumer root rather than package-local ambient type roots', () => {
  const {directory, packed} = createPackedFixture();
  writeFileSync(join(packed, 'index.d.cts'), '/// <reference types="./node" />\nexport {};');
  writeFileSync(join(packed, 'node.d.ts'), 'export {};');
  writeFileSync(join(directory, 'outside.d.ts'), 'export {};');
  mkdirSync(join(packed, 'node_modules/@types/node'), {recursive: true});
  symlinkSync(join(directory, 'outside.d.ts'), join(packed, 'node_modules/@types/node/index.d.ts'));
  expect(checkDeclarationEdges(packed, directory)).toEqual({declarationCount: 3, edgeCount: 1});
});

it('S3 respects an explicit require resolution mode in an ESM declaration', () => {
  const {packed} = createPackedFixture();
  writeFileSync(
    join(packed, 'index.d.mts'),
    '/// <reference types="./ambient" resolution-mode="require" />\nexport {};'
  );
  writeFileSync(join(packed, 'ambient.d.ts'), 'export {};');
  expect(checkDeclarationEdges(packed)).toEqual({declarationCount: 2, edgeCount: 1});
});

it.each([
  ['types', '/// <reference types="./..units" />\nexport {};'],
  ['module', "export type Unit = import('./..units.js').Unit;"],
  ['path', '/// <reference path="./..units.d.ts" />\nexport {};']
])('B5 accepts a two-dot filename in a relative %s edge', (_kind, declaration) => {
  const {packed} = createPackedFixture();
  writeFileSync(join(packed, 'index.d.cts'), declaration);
  writeFileSync(join(packed, '..units.d.ts'), 'export interface Unit {value: number}');
  expect(checkDeclarationEdges(packed)).toEqual({declarationCount: 2, edgeCount: 1});
});

it.each([
  'extensionless',
  'unsupported',
  'symlink escape'
])('EARLIER-1 uses native TypeScript path-reference rules (%s)', state => {
  const {directory, packed} = createPackedFixture();
  const reference =
    state === 'extensionless'
      ? './geometry'
      : state === 'unsupported'
        ? './readme.md'
        : './outside.d.ts';
  writeFileSync(join(packed, 'index.d.cts'), `/// <reference path="${reference}" />\nexport {};`);
  if (state === 'extensionless') writeFileSync(join(packed, 'geometry.d.ts'), 'export {};');
  if (state === 'unsupported') writeFileSync(join(packed, 'readme.md'), 'plain text');
  if (state === 'symlink escape') {
    writeFileSync(join(directory, 'outside.d.ts'), 'export {};');
    symlinkSync(join(directory, 'outside.d.ts'), join(packed, 'outside.d.ts'));
  }
  // A literal fixture has one reference. TypeScript appends .ts/.tsx/.d.ts to extensionless
  // path references, rejects Markdown, and package containment must use the resolved real file.
  if (state === 'extensionless')
    expect(checkDeclarationEdges(packed)).toEqual({declarationCount: 2, edgeCount: 1});
  else
    expect(() => checkDeclarationEdges(packed)).toThrow(
      `index.d.cts:1 reference ${JSON.stringify(reference)}`
    );
});

it('W3-03 rejects a missing declaration sidecar and preserves the invented-property sentinel with its sidecar', () => {
  const {directory, packed} = createPackedFixture();
  writeFileSync(join(packed, 'index.d.cts'), "export type Unit = import('./unit.js').Unit;");
  writeFileSync(join(packed, 'unit.js'), 'exports.Unit = {};');
  const consumer = join(directory, 'consumer.mts');
  writeFileSync(
    consumer,
    `import type {Unit} from './package/index.cjs';
    declare const unit: Unit;
    const value: number = unit.value; void value;
    // @ts-expect-error A packed Unit must not degrade to any.
    unit.inventedProperty;`
  );
  function diagnostics() {
    const options = {
      ...{
        module: typescript.ModuleKind.NodeNext,
        moduleResolution: typescript.ModuleResolutionKind.NodeNext
      },
      skipLibCheck: true,
      strict: true,
      noEmit: true,
      types: []
    };
    const host = typescript.createCompilerHost(options);
    host.getCurrentDirectory = () => directory;
    return typescript
      .getPreEmitDiagnostics(typescript.createProgram([consumer], options, host))
      .map(diagnostic => diagnostic.code);
  }
  // Independent native consumer oracle: skipLibCheck hides the missing sidecar, but any leaves
  // the invented-property directive unused (TS2578). With Unit {value: number}, it is consumed.
  expect(diagnostics()).toContain(2578);
  expect(() => checkDeclarationEdges(packed)).toThrow('index.d.cts:1 "./unit.js"');
  writeFileSync(join(packed, 'unit.d.ts'), 'export interface Unit {value: number}');
  expect(checkDeclarationEdges(packed)).toEqual({declarationCount: 2, edgeCount: 1});
  expect(diagnostics()).toEqual([]);
});

it.each([
  ['types', '/// <reference types="../outside" />\nexport {};'],
  ['module', "export type Unit = import('../outside.js').Unit;"],
  ['path', '/// <reference path="../outside.d.ts" />\nexport {};']
])('W3-05 rejects a true parent escape for a relative %s edge', (_kind, declaration) => {
  const {directory, packed} = createPackedFixture();
  writeFileSync(join(packed, 'index.d.cts'), declaration);
  writeFileSync(join(directory, 'outside.d.ts'), 'export interface Unit {value: number}');
  expect(() => checkDeclarationEdges(packed)).toThrow('1 of 1 relative declaration references');
});

it.each([
  false,
  true
])('review B5 follows the host filename case policy for path directives (sensitive=%s)', caseSensitive => {
  const {directory, packed} = createPackedFixture();
  const declaration = join(packed, 'index.d.cts');
  writeFileSync(declaration, '/// <reference path="./rows.D.TS" />\nexport {};');
  writeFileSync(join(packed, 'rows.D.TS'), 'export interface Row {value: number}');
  const options = {
    module: typescript.ModuleKind.NodeNext,
    moduleResolution: typescript.ModuleResolutionKind.NodeNext,
    noEmit: true,
    types: []
  };
  const host = typescript.createCompilerHost(options);
  host.getCurrentDirectory = () => directory;
  host.useCaseSensitiveFileNames = () => caseSensitive;
  host.getCanonicalFileName = file => (caseSensitive ? file : file.toLowerCase());
  const nativeDiagnostics = typescript.getPreEmitDiagnostics(
    typescript.createProgram([declaration], options, host)
  );
  const system = {...typescript.sys, useCaseSensitiveFileNames: caseSensitive};
  if (caseSensitive) {
    // A supported extension is recognized after the compiler canonicalizes the filename.
    expect(nativeDiagnostics.map(diagnostic => diagnostic.code)).toContain(6054);
    expect(() => checkDeclarationEdges(packed, directory, system)).toThrow(
      'index.d.cts:1 reference "./rows.D.TS"'
    );
  } else {
    expect(nativeDiagnostics).toEqual([]);
    expect(checkDeclarationEdges(packed, directory, system)).toEqual({
      declarationCount: 2,
      edgeCount: 1
    });
  }
});

it.each([
  false,
  true
])('review B5 canonicalizes the declaration inventory (sensitive=%s)', caseSensitive => {
  const {directory, packed} = createPackedFixture();
  writeFileSync(join(packed, 'INDEX.D.CTS'), '/// <reference path="./rows.d.ts" />\nexport {};');
  writeFileSync(join(packed, 'rows.d.ts'), 'export {};');
  expect(
    checkDeclarationEdges(packed, directory, {
      ...typescript.sys,
      useCaseSensitiveFileNames: caseSensitive
    })
  ).toEqual({
    declarationCount: caseSensitive ? 1 : 2,
    edgeCount: caseSensitive ? 0 : 1
  });
});
