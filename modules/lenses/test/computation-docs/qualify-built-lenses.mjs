// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {execFileSync} from 'node:child_process';
import {mkdtempSync, readFileSync, readdirSync, statSync, rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join, resolve, sep} from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {checkBuildRecord} from '../../../../scripts/build-record.mjs';
import {getTypeScriptProjectInputs, getTypeScriptOutputPlan} from '../../../../scripts/typescript-project-inputs.mjs';

export const BUILT_LENS_CASE =
  'checks emitted relative links and public role loading after a build';
export const BUILT_LENS_SPEC = 'modules/lenses/test/computation-docs/lens-package.node.spec.ts';
// Vitest's CLI task names use " > "; its JSON assertion fullName uses spaces.
export const BUILT_LENS_FILTER = `^lenses package registration > ${BUILT_LENS_CASE}$`;
const REPOSITORY_ROOT = fileURLToPath(new URL('../../../../', import.meta.url));

/** Require current source/configuration and emitted-content hashes, including every source output. */
export function requireCurrentLensBuild(repositoryRoot = REPOSITORY_ROOT) {
  const evidence = {};
  const manifest = JSON.parse(
    readFileSync(join(repositoryRoot, 'modules/lenses/package.json'), 'utf8')
  );
  if (manifest.private !== true)
    throw new Error('Lens qualification requires lenses to remain private');
  // Analysis is a runtime dependency of lenses; kepler is not part of this lens-local command.
  for (const name of ['analysis', 'lenses']) {
    const directory = join(repositoryRoot, 'modules', name);
    const project = getTypeScriptProjectInputs(join(directory, 'tsconfig.json'), [
      join(directory, 'src')
    ]);
    const record = checkBuildRecord(directory, project);
    if (!record.present || !record.valid)
      throw new Error(`${name}: missing or stale build evidence; run yarn build first`);
    const outputPlan = getTypeScriptOutputPlan(project, directory);
    const sourceCount = outputPlan.length;
    const expectedOutputs = new Set();
    for (const entry of outputPlan) {
      for (const output of [...entry.compilerOutputs, ...entry.transformedOutputs]) {
        const filename = `dist/${output.split(sep).join('/')}`;
        expectedOutputs.add(filename);
        if (!record.outputFiles.includes(filename))
          throw new Error(`${name}: missing build evidence for ${filename}; run yarn build first`);
      }
    }
    if (!sourceCount) throw new Error(`${name}: no current sources to qualify`);
    const actualOutputs = readdirSync(join(directory, 'dist'), {
      recursive: true,
      encoding: 'utf8'
    })
      .filter(
        filename =>
          filename !== '.build-record.json' &&
          statSync(join(directory, 'dist', filename)).isFile()
      )
      .map(filename => `dist/${filename.split(sep).join('/')}`);
    for (const output of actualOutputs)
      if (!expectedOutputs.has(output))
        throw new Error(`${name}: unexpected emitted output ${output}; run yarn build first`);
    const actualOutputSet = new Set(actualOutputs);
    for (const output of expectedOutputs)
      if (!actualOutputSet.has(output))
        throw new Error(`${name}: missing emitted output ${output}; run yarn build first`);
    for (const output of record.outputFiles)
      if (!expectedOutputs.has(output))
        throw new Error(`${name}: unexpected recorded output ${output}; run yarn build first`);
    // Expected-to-record was checked above; hashing checks record-to-disk, including deletions.
    evidence[name] = JSON.parse(readFileSync(join(directory, 'dist/.build-record.json'), 'utf8'));
  }
  return evidence;
}

/** Exit status alone cannot prove the intended test existed, ran, or avoided a skip. */
export function requireBuiltLensCase(report, repositoryRoot = REPOSITORY_ROOT) {
  const suites = report.testResults ?? [];
  const assertions = suites.flatMap(suite => suite.assertionResults ?? []);
  const matching = assertions.filter(
    assertion =>
      assertion.fullName === `lenses package registration ${BUILT_LENS_CASE}` &&
      assertion.title === BUILT_LENS_CASE
  );
  if (
    report.success !== true ||
    suites.length !== 1 ||
    resolve(suites[0].name) !== resolve(repositoryRoot, BUILT_LENS_SPEC) ||
    matching.length !== 1 ||
    matching[0].status !== 'passed' ||
    assertions.filter(assertion => assertion.status === 'passed').length !== 1 ||
    assertions.some(assertion => assertion.status === 'failed')
  )
    throw new Error(`Lens qualification requires exactly one passing case: ${BUILT_LENS_CASE}`);
}

export function qualifyBuiltLenses(repositoryRoot = REPOSITORY_ROOT) {
  const before = requireCurrentLensBuild(repositoryRoot);
  const temporary = mkdtempSync(join(tmpdir(), 'qualify-built-lenses-'));
  try {
    const reportFile = join(temporary, 'vitest.json');
    execFileSync(
      process.execPath,
      [
        join(repositoryRoot, 'node_modules/vitest/vitest.mjs'),
        'run',
        '--project',
        'node',
        BUILT_LENS_SPEC,
        '--testNamePattern',
        BUILT_LENS_FILTER,
        '--maxWorkers=1',
        '--reporter=json',
        '--outputFile',
        reportFile
      ],
      {cwd: repositoryRoot, env: {...process.env, LENS_VERIFY_BUILT: '1'}, stdio: 'inherit'}
    );
    requireBuiltLensCase(JSON.parse(readFileSync(reportFile, 'utf8')), repositoryRoot);
    const after = requireCurrentLensBuild(repositoryRoot);
    if (JSON.stringify(before) !== JSON.stringify(after))
      throw new Error('Build evidence changed during lens qualification; rebuild and retry');
    console.log(`Qualified private lenses: ${BUILT_LENS_CASE}`);
  } finally {
    rmSync(temporary, {recursive: true, force: true});
  }
}

if (process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url) {
  try {
    qualifyBuiltLenses();
  } catch (error) {
    console.error(error instanceof Error ? error.message : error);
    process.exitCode = 1;
  }
}
