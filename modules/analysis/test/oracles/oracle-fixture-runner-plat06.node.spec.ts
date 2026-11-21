// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync} from 'node:fs';
import {spawnSync} from 'node:child_process';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {expect, it} from 'vitest';
import typescript from 'typescript';
import {
  classifyFailure,
  diffJson,
  flattenJson,
  parseExpectedFixtures,
  resolveGenerator
} from '../../../../scripts/run-oracle-fixture.mjs';

it('diffJson reports changed, added and removed leaves and the largest numeric difference', () => {
  const diff = diffJson(
    {versions: {numpy: '2.5.4'}, values: [1, 2, 3], gone: true},
    {versions: {numpy: '2.5.5'}, values: [1, 2.5, 3], fresh: 1}
  );
  expect(diff.changed.map(item => item.path)).toEqual(['versions.numpy', 'values[1]']);
  expect(diff.changedNumberCount).toBe(1);
  expect(diff.maximumAbsoluteDifference).toBe(0.5);
  expect(diff.added).toEqual(['fresh']);
  expect(diff.removed).toEqual(['gone']);
  expect(diff.metadataOnly).toBe(false);
});

it('diffJson recognises version-only drift and treats NaN as equal to NaN', () => {
  const diff = diffJson(
    {versions: {scipy: '1'}, x: [Number.NaN]},
    {versions: {scipy: '2'}, x: [Number.NaN]}
  );
  expect(diff.metadataOnly).toBe(true);
  expect(diffJson({a: [1, {b: 2}]}, {a: [1, {b: 2}]}).changed).toEqual([]);
  expect([...flattenJson({a: [], b: {}}).keys()]).toEqual(['a[]', 'b{}']);
});

it('classifyFailure names a missing Python or R package and ignores ordinary failures', () => {
  expect(classifyFailure('.py', "ModuleNotFoundError: No module named 'esda.moran'")).toContain(
    'missing Python package: esda'
  );
  expect(
    classifyFailure('.R', 'Error in library(gstat) : there is no package called ‘gstat’')
  ).toBe('missing R package: gstat');
  expect(classifyFailure('.py', 'AssertionError: wrong version')).toBeUndefined();
});

it('parseExpectedFixtures reads GENERATORS.md sections and resolveGenerator finds scripts', () => {
  const expected = parseExpectedFixtures(
    '## Literal mappings\n\n### `a-b.py`\n\n`a-b.json`, `a-c.json` (2)\n\n## Counts\n\n`ignored.json`\n'
  );
  expect(expected.get('a-b.py')).toEqual(['a-b.json', 'a-c.json']);
  const directory = mkdtempSync(join(tmpdir(), 'plat06-'));
  try {
    for (const file of ['one.py', 'two.R', 'two.py', 'three.cpp'])
      writeFileSync(join(directory, file), '');
    expect(resolveGenerator('one', directory)).toEqual({file: 'one.py'});
    expect(resolveGenerator('two', directory).error).toContain('ambiguous');
    expect(resolveGenerator('two.R', directory)).toEqual({file: 'two.R'});
    expect(resolveGenerator('three', directory).error).toContain('not runnable');
    expect(resolveGenerator('missing', directory).error).toContain('no generator');
  } finally {
    rmSync(directory, {recursive: true, force: true});
  }
});

it('records missing, partial, differing and matching regeneration without inferring comparisons', async () => {
  const {createOracleRunEvidence} = await import('../../../../scripts/run-oracle-fixture.mjs');
  const base = {
    runId: 'miniature-run',
    startedAt: '2026-11-19T00:00:00.000Z',
    completedAt: '2026-11-19T00:00:01.000Z',
    generator: 'first.py',
    generatorBytes: Buffer.from('abc'),
    declaredOutputs: ['a.json', 'b.json']
  };
  const missing = createOracleRunEvidence({...base, outputs: []});
  expect(missing.outputs.map(output => [output.regenerated, output.comparison])).toEqual([
    [false, 'not-compared'],
    [false, 'not-compared']
  ]);
  const partial = createOracleRunEvidence({
    ...base,
    outputs: [
      {path: 'a.json', committedBytes: Buffer.from('abc'), regeneratedBytes: Buffer.from('abc')}
    ]
  });
  expect(partial.outputs.map(output => [output.regenerated, output.comparison])).toEqual([
    [true, 'not-compared'],
    [false, 'not-compared']
  ]);
  const complete = createOracleRunEvidence({
    ...base,
    outputs: [
      {path: 'a.json', committedBytes: Buffer.from('abc'), regeneratedBytes: Buffer.from('abc')},
      {path: 'b.json', committedBytes: Buffer.from('abc'), regeneratedBytes: Buffer.from('def')}
    ]
  });
  expect(complete.outputs.map(output => output.comparison)).toEqual(['matching', 'differing']);
  // SHA-256('abc'), the standard test vector, fixes the independent hash expectation.
  expect(complete.generatorSha256).toBe(
    'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
  );
  expect(complete.outputs[0]?.committedSha256).toBe(complete.generatorSha256);
  expect(complete.outputs[0]?.regeneratedSha256).toBe(complete.generatorSha256);
});

it('writes identified receipts from actual miniature runner comparisons, including incomplete runs', () => {
  for (const mode of ['missing', 'partial', 'differing', 'matching', 'accepted']) {
    const root = realpathSync(mkdtempSync(join(tmpdir(), 'oracle-evidence-')));
    try {
      const scriptDirectory = join(root, 'scripts');
      const oracleDirectory = join(root, 'modules/analysis/test/oracles');
      mkdirSync(scriptDirectory, {recursive: true});
      mkdirSync(join(oracleDirectory, 'fixtures'), {recursive: true});
      writeFileSync(
        join(scriptDirectory, 'run-oracle-fixture.mjs'),
        readFileSync(new URL('../../../../scripts/run-oracle-fixture.mjs', import.meta.url))
      );
      writeFileSync(
        join(scriptDirectory, 'oracle-fixtures.json'),
        JSON.stringify({
          version: 1,
          profiles: {node: {runtime: 'node'}},
          generators: {
            'miniature.mjs': {
              kind: 'fixture',
              profile: 'node',
              outputs: ['oracles/fixtures/a.json', 'oracles/fixtures/b.json']
            }
          }
        })
      );
      for (const name of ['a', 'b'])
        writeFileSync(join(oracleDirectory, `fixtures/${name}.json`), '1');
      const source = ["import fs from 'node:fs';"];
      if (mode !== 'missing')
        source.push(
          `fs.writeFileSync('modules/analysis/test/oracles/fixtures/a.json', '${mode === 'partial' ? '9' : '1'}');`
        );
      if (!['missing', 'partial'].includes(mode))
        source.push(
          `fs.writeFileSync('modules/analysis/test/oracles/fixtures/b.json', '${mode === 'matching' ? '1' : '2'}');`
        );
      writeFileSync(join(oracleDirectory, 'miniature.mjs'), source.join('\n'));
      const receiptPath = join(root, 'receipt.json');
      const run = spawnSync(
        process.execPath,
        [
          join(scriptDirectory, 'run-oracle-fixture.mjs'),
          'miniature',
          '--evidence',
          receiptPath,
          ...(['accepted', 'partial'].includes(mode) ? ['--write'] : [])
        ],
        {encoding: 'utf8'}
      );
      expect(run.status, `${mode}: ${run.stderr}`).toBe(
        ['missing', 'partial'].includes(mode) ? 3 : mode === 'differing' ? 1 : 0
      );
      const receipt = JSON.parse(readFileSync(receiptPath, 'utf8')) as {
        runId: string;
        startedAt: string;
        completedAt: string;
        generator: string;
        outputs: {path: string; regenerated: boolean; comparison: string}[];
      };
      expect(receipt.runId).toMatch(/^[a-f0-9-]{36}$/);
      expect(receipt.generator).toBe('miniature.mjs');
      expect(Date.parse(receipt.completedAt)).toBeGreaterThanOrEqual(Date.parse(receipt.startedAt));
      expect(receipt.outputs.map(output => [output.path, output.regenerated])).toEqual([
        ['oracles/fixtures/a.json', mode !== 'missing'],
        ['oracles/fixtures/b.json', !['missing', 'partial'].includes(mode)]
      ]);
      expect(readFileSync(join(oracleDirectory, 'fixtures/a.json'), 'utf8')).toBe('1');
      expect(receipt.outputs.map(output => output.comparison)).toEqual(
        ['missing', 'partial'].includes(mode)
          ? ['not-compared', 'not-compared']
          : ['matching', mode === 'matching' ? 'matching' : 'differing']
      );
      expect(readFileSync(join(oracleDirectory, 'fixtures/b.json'), 'utf8')).toBe(
        mode === 'accepted' ? '2' : '1'
      );
    } finally {
      rmSync(root, {recursive: true, force: true});
    }
  }
});

it('emits a concrete createOracleRunEvidence boundary for strict TypeScript consumers (B6)', () => {
  // Declaration-only transform, without typechecking a project or running a compiler program.
  const fileName = 'scripts/run-oracle-fixture.mjs';
  const result = typescript.transpileDeclaration(
    readFileSync(new URL('../../../../scripts/run-oracle-fixture.mjs', import.meta.url), 'utf8'),
    {fileName, compilerOptions: {allowJs: true}}
  );
  const declaration = typescript.createSourceFile(
    'boundary.d.ts',
    result.outputText,
    typescript.ScriptTarget.Latest,
    true
  );
  const boundary = declaration.statements.find(
    statement =>
      typescript.isFunctionDeclaration(statement) &&
      statement.name?.text === 'createOracleRunEvidence'
  ) as typescript.FunctionDeclaration;
  expect(boundary.type?.getText(declaration)).toBe('OracleRunEvidence');
});
