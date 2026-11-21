// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createHash} from 'node:crypto';
import {mkdirSync, mkdtempSync, rmSync, writeFileSync} from 'node:fs';
import {dirname, join} from 'node:path';
import {fileURLToPath} from 'node:url';
import typescript from 'typescript';
import {afterEach, expect, it} from 'vitest';
import {
  captureTypeScriptProjectInputs,
  getTypeScriptProjectInputs,
  listTypeScriptDependencyInputs
} from '../../../../scripts/typescript-project-inputs.mjs';
import {captureBuildInputs, checkBuildRecord} from '../../../../scripts/build-record.mjs';

const directories: string[] = [];
const scratch = fileURLToPath(
  new URL('../../../../.typecheck/r19/scratch/forwards6-tooling/', import.meta.url)
);
afterEach(() => {
  for (const directory of directories.splice(0)) rmSync(directory, {recursive: true, force: true});
});
function write(directory: string, name: string, contents: string) {
  mkdirSync(dirname(join(directory, name)), {recursive: true});
  writeFileSync(join(directory, name), contents);
}
function fixture(noResolve: boolean) {
  mkdirSync(scratch, {recursive: true});
  const directory = mkdtempSync(join(scratch, 'references-'));
  directories.push(directory);
  write(directory, 'package.json', '{"type":"module"}');
  write(
    directory,
    'modules/analysis/package.json',
    '{"name":"reference-input-fixture","type":"module"}'
  );
  write(
    directory,
    'modules/analysis/tsconfig.json',
    JSON.stringify({
      compilerOptions: {module: 'NodeNext', moduleResolution: 'NodeNext', types: [], noResolve},
      files: ['src/entry.d.cts']
    })
  );
  write(
    directory,
    'node_modules/conditional/package.json',
    JSON.stringify({
      name: 'conditional',
      exports: {'.': {import: {types: './esm.d.mts'}, require: {types: './cjs.d.cts'}}}
    })
  );
  write(directory, 'node_modules/conditional/esm.d.mts', 'export type Marker = 11;');
  write(directory, 'node_modules/conditional/cjs.d.cts', 'export type Marker = 22;');
  write(directory, 'modules/analysis/src/companion.d.ts', 'export type Companion = 33;');
  write(
    directory,
    'modules/analysis/src/entry.d.cts',
    [
      '/// <reference path="./companion.d.ts" />',
      '/// <reference types="conditional" resolution-mode="import" />',
      'export declare const value: number;'
    ].join('\n')
  );
  return directory;
}

it.each([
  false,
  true
])('R6 source references match compiler selection with noResolve=%s', noResolve => {
  const directory = fixture(noResolve);
  const configuration = join(directory, 'modules/analysis/tsconfig.json');
  const project = getTypeScriptProjectInputs(configuration);
  const program = typescript.createProgram(project.files, project.options);
  // The actual compiler is the independent selection oracle, including explicit import mode.
  expect(typescript.getPreEmitDiagnostics(program).map(diagnostic => diagnostic.code)).toEqual([]);
  const selected = program
    .getSourceFiles()
    .map(source => source.fileName)
    .filter(file => file.startsWith(directory))
    .sort();
  expect(selected).toContain(join(directory, 'modules/analysis/src/entry.d.cts'));
  expect(selected.includes(join(directory, 'node_modules/conditional/esm.d.mts'))).toBe(!noResolve);
  expect(selected.includes(join(directory, 'modules/analysis/src/companion.d.ts'))).toBe(
    !noResolve
  );
  expect(selected).not.toContain(join(directory, 'node_modules/conditional/cjs.d.cts'));
  expect(listTypeScriptDependencyInputs(project)).toEqual(selected);
  const before = captureTypeScriptProjectInputs(configuration).fingerprint;
  write(directory, 'node_modules/conditional/esm.d.mts', 'export type Marker = 44;');
  write(directory, 'modules/analysis/src/companion.d.ts', 'export type Companion = 55;');
  const after = captureTypeScriptProjectInputs(configuration).fingerprint;
  expect(after === before).toBe(noResolve);
});

it('R6 ignored declaration JSDoc leaves watch fingerprints and build records valid', () => {
  const directory = fixture(false);
  const moduleDirectory = join(directory, 'modules/analysis');
  const configuration = join(moduleDirectory, 'tsconfig.json');
  write(
    directory,
    'modules/analysis/src/entry.d.cts',
    '/** @type {import("conditional").Marker} */\nexport declare const value: number;'
  );
  const project = getTypeScriptProjectInputs(configuration);
  const program = typescript.createProgram(project.files, project.options);
  expect(typescript.getPreEmitDiagnostics(program).map(diagnostic => diagnostic.code)).toEqual([]);
  expect(
    program
      .getSourceFiles()
      .filter(source => source.fileName.startsWith(directory))
      .map(source => source.fileName)
  ).toEqual([join(moduleDirectory, 'src/entry.d.cts')]);
  const before = captureTypeScriptProjectInputs(configuration);
  const inputs = captureBuildInputs(moduleDirectory, project);
  const output = 'export const value = 1;';
  write(directory, 'modules/analysis/dist/index.js', output);
  write(
    directory,
    'modules/analysis/dist/.build-record.json',
    JSON.stringify({
      version: 1,
      inputs,
      outputs: {'dist/index.js': createHash('sha256').update(output).digest('hex')}
    })
  );
  expect(checkBuildRecord(moduleDirectory, project).valid).toBe(true);
  for (const branch of ['esm.d.mts', 'cjs.d.cts']) {
    write(directory, `node_modules/conditional/${branch}`, 'export type Marker = 66;');
    expect(captureTypeScriptProjectInputs(configuration)).toEqual(before);
    expect(checkBuildRecord(moduleDirectory, project).valid).toBe(true);
  }
  // A real compiler input must still invalidate both consumers.
  write(directory, 'modules/analysis/src/entry.d.cts', 'export declare const value: string;');
  expect(captureTypeScriptProjectInputs(configuration).fingerprint).not.toBe(before.fingerprint);
  expect(checkBuildRecord(moduleDirectory, project).valid).toBe(false);
});
