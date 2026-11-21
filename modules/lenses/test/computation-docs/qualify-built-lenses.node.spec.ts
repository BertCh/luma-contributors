// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {mkdtempSync, mkdirSync, readFileSync, writeFileSync, rmSync, cpSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join, dirname} from 'node:path';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
import {describe, expect, test} from 'vitest';
import {captureBuildInputs, writeBuildRecord} from '../../../../scripts/build-record.mjs';
import {getTypeScriptProjectInputs} from '../../../../scripts/typescript-project-inputs.mjs';
import {
  BUILT_LENS_CASE,
  BUILT_LENS_SPEC,
  BUILT_LENS_FILTER,
  requireBuiltLensCase,
  requireCurrentLensBuild
} from './qualify-built-lenses.mjs';

describe('lenses package registration', () => {
  test(BUILT_LENS_CASE, async ({task}) => {
    // Exercise the installed CLI task-name helper, independently of JSON reporter fullName.
    const require = createRequire(import.meta.url);
    const helper = await import(
      pathToFileURL(join(dirname(require.resolve('vitest/package.json')), 'dist/task-utils.js'))
        .href
    );
    const taskName = helper.getTestName(task);
    expect(taskName).toBe(`lenses package registration > ${BUILT_LENS_CASE}`);
    expect(new RegExp(BUILT_LENS_FILTER).test(taskName)).toBe(true);
    expect(new RegExp(BUILT_LENS_FILTER).test(`${taskName} extra`)).toBe(false);
  });
});

test('TOOL-119: qualification requires the exact passing built case, rejecting missing, skipped and failed role loads', () => {
  const root = '/fixture';
  const assertion = {
    fullName: `lenses package registration ${BUILT_LENS_CASE}`,
    title: BUILT_LENS_CASE,
    status: 'passed'
  };
  const report = {
    success: true,
    testResults: [{name: join(root, BUILT_LENS_SPEC), assertionResults: [assertion]}]
  };
  expect(() => requireBuiltLensCase(report, root)).not.toThrow();
  for (const status of ['pending', 'skipped', 'failed', 'todo']) {
    expect(() =>
      requireBuiltLensCase(
        {
          ...report,
          testResults: [{...report.testResults[0], assertionResults: [{...assertion, status}]}]
        },
        root
      )
    ).toThrow('exactly one passing case');
  }
  for (const changed of [
    {...report, success: false},
    {...report, testResults: []},
    {...report, testResults: [{...report.testResults[0], assertionResults: []}]},
    {...report, testResults: [{...report.testResults[0], name: '/fixture/wrong.node.spec.ts'}]},
    {
      ...report,
      testResults: [
        {...report.testResults[0], assertionResults: [{...assertion, title: 'renamed'}]}
      ]
    }
  ])
    expect(() => requireBuiltLensCase(changed, root)).toThrow('exactly one passing case');
});

test('TOOL-119: qualification rejects missing/stale/corrupt outputs and incomplete build inventories', () => {
  const root = mkdtempSync(join(tmpdir(), 'lens-qualification-test-'));
  try {
    for (const name of ['analysis', 'lenses']) {
      const directory = join(root, 'modules', name);
      mkdirSync(join(directory, 'src'), {recursive: true});
      mkdirSync(join(directory, 'dist'));
      writeFileSync(join(directory, 'package.json'), JSON.stringify({private: true}));
      writeFileSync(
        join(directory, 'tsconfig.json'),
        JSON.stringify({compilerOptions: {rootDir: './src', declaration: true}, include: ['src']})
      );
      writeFileSync(join(directory, 'src/index.ts'), 'export const value = 1;');
      for (const extension of ['js', 'cjs', 'd.ts'])
        writeFileSync(join(directory, `dist/index.${extension}`), 'fixture');
    }
    expect(() => requireCurrentLensBuild(root)).toThrow('missing or stale build evidence');
    const record = (name: string, includeObsoleteOutput = false) => {
      const directory = join(root, 'modules', name);
      const project = getTypeScriptProjectInputs(join(directory, 'tsconfig.json'));
      writeBuildRecord(
        directory,
        project,
        new Set([
          'index.js',
          'index.cjs',
          'index.d.ts',
          ...(includeObsoleteOutput ? ['obsolete.cjs'] : [])
        ]),
        captureBuildInputs(directory, project)
      );
    };
    record('analysis');
    record('lenses');
    expect(() => requireCurrentLensBuild(root)).not.toThrow();
    for (const name of ['analysis', 'lenses']) {
      const extra = join(root, 'modules', name, 'dist/obsolete.cjs');
      writeFileSync(extra, 'stale unrecorded output');
      expect(() => requireCurrentLensBuild(root)).toThrow(
        'unexpected emitted output dist/obsolete.cjs'
      );
      record(name, true);
      expect(() => requireCurrentLensBuild(root)).toThrow(
        'unexpected emitted output dist/obsolete.cjs'
      );
      rmSync(extra);
      record(name);
    }
    const source = join(root, 'modules/lenses/src/index.ts');
    writeFileSync(source, 'export const value = 2;');
    expect(() => requireCurrentLensBuild(root)).toThrow('missing or stale build evidence');
    record('lenses');
    const output = join(root, 'modules/lenses/dist/index.cjs');
    writeFileSync(output, 'corrupt');
    expect(() => requireCurrentLensBuild(root)).toThrow('missing or stale build evidence');
    rmSync(output);
    expect(() => requireCurrentLensBuild(root)).toThrow('missing or stale build evidence');
    writeFileSync(output, 'fixture');
    record('lenses');
    const recordFile = join(root, 'modules/lenses/dist/.build-record.json');
    const evidence = JSON.parse(readFileSync(recordFile, 'utf8'));
    delete evidence.outputs['dist/index.cjs'];
    writeFileSync(recordFile, JSON.stringify(evidence));
    expect(() => requireCurrentLensBuild(root)).toThrow(
      'missing build evidence for dist/index.cjs'
    );
  } finally {
    rmSync(root, {recursive: true, force: true});
  }
});

test('TOOL-119 qualifies React Native JSX outputs and relocated promoted records', () => {
  const temporary = mkdtempSync(join(tmpdir(), 'lens-native-qualification-'));
  const staged = join(temporary, 'stage');
  const promoted = join(temporary, 'promoted');
  try {
    for (const name of ['analysis', 'lenses']) {
      const directory = join(staged, 'modules', name);
      mkdirSync(join(directory, 'src'), {recursive: true});
      mkdirSync(join(directory, 'dist'));
      writeFileSync(join(directory, 'package.json'), JSON.stringify({private: true}));
      writeFileSync(
        join(directory, 'tsconfig.json'),
        JSON.stringify({
          compilerOptions: {
            rootDir: './src',
            outDir: './dist',
            jsx: 'react-native',
            declaration: true
          },
          include: ['src']
        })
      );
      writeFileSync(join(directory, 'src/index.tsx'), 'export const element = <View />;');
      // TypeScript React Native JSX emits .js, and the shared build transforms it to .cjs.
      for (const extension of ['js', 'cjs', 'd.ts'])
        writeFileSync(join(directory, `dist/index.${extension}`), 'fixture');
      const project = getTypeScriptProjectInputs(join(directory, 'tsconfig.json'));
      writeBuildRecord(
        directory,
        project,
        new Set(['index.js', 'index.cjs', 'index.d.ts']),
        captureBuildInputs(directory, project)
      );
    }
    expect(() => requireCurrentLensBuild(staged)).not.toThrow();
    cpSync(join(staged, 'modules'), join(promoted, 'modules'), {recursive: true});
    rmSync(staged, {recursive: true});
    expect(() => requireCurrentLensBuild(promoted)).not.toThrow();
  } finally {
    rmSync(temporary, {recursive: true, force: true});
  }
});
