// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {spawnSync} from 'node:child_process';
import {mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import typescript from 'typescript';
import {afterEach, expect, it} from 'vitest';
import {getVerificationBrowserConfiguration} from '../../../../scripts/verification-policy.mjs';

const ROOT = fileURLToPath(new URL('../../../../', import.meta.url));
const directories: string[] = [];
afterEach(() => {
  for (const directory of directories.splice(0)) rmSync(directory, {recursive: true, force: true});
});

it.each([
  ['linux', 'nonempty', false],
  ['linux', '', false],
  ['darwin', 'nonempty', false],
  ['linux', 'nonempty', true]
] as const)('W2-03 launches with receipt configuration (%s, CI=%s, changed helper=%s)', (platform, continuousIntegration, changedHelper) => {
  const directory = mkdtempSync(join(tmpdir(), 'browser-configuration-'));
  directories.push(directory);
  mkdirSync(join(directory, 'scripts'));
  const policyURL = JSON.stringify(
    pathToFileURL(join(ROOT, 'scripts/verification-policy.mjs')).href
  );
  // A distinct helper result pins the dependency on the shared policy, rather than coincidental
  // agreement between two copies of today's literals. No browser is launched by this fixture.
  writeFileSync(
    join(directory, 'scripts/verification-policy.mjs'),
    `
    export * from ${policyURL};
    import {getVerificationBrowserConfiguration as original} from ${policyURL};
    export function getVerificationBrowserConfiguration(...args) {
      const configuration = original(...args);
      return ${changedHelper ? "{...configuration, channel: 'receipt-fixture', launchArguments: [...configuration.launchArguments, '--receipt-fixture']}" : 'configuration'};
    }
  `
  );
  mkdirSync(join(directory, 'node_modules/@vis.gl/dev-tools'), {recursive: true});
  writeFileSync(
    join(directory, 'node_modules/@vis.gl/dev-tools/package.json'),
    '{"type":"module","exports":"./index.js"}'
  );
  writeFileSync(
    join(directory, 'node_modules/@vis.gl/dev-tools/index.js'),
    'export const getVitestConfig = options => options;'
  );
  mkdirSync(join(directory, 'test'));
  writeFileSync(
    join(directory, 'test/kepler-compute-example-specs.json'),
    '{"excludedFromRuns":{}}'
  );
  const source = readFileSync(join(ROOT, 'vitest.config.ts'), 'utf8');
  writeFileSync(
    join(directory, 'vitest.config.mjs'),
    typescript.transpileModule(source, {
      compilerOptions: {
        module: typescript.ModuleKind.ES2022,
        target: typescript.ScriptTarget.ES2022
      }
    }).outputText
  );
  const result = spawnSync(
    process.execPath,
    [
      '--input-type=module',
      '-e',
      `
    Object.defineProperty(process, 'platform', {value: ${JSON.stringify(platform)}});
    const {default: configuration} = await import('./vitest.config.mjs');
    console.log(JSON.stringify(configuration.launchOptions));
  `
    ],
    {
      cwd: directory,
      encoding: 'utf8',
      timeout: 10000,
      env: {...process.env, CI: continuousIntegration, NODE_OPTIONS: '--max-old-space-size=256'}
    }
  );
  expect(result.status, result.stderr).toBe(0);
  const receiptConfiguration = getVerificationBrowserConfiguration(
    {CI: continuousIntegration},
    platform
  );
  const expectedArguments = [...receiptConfiguration.launchArguments];
  if (changedHelper) expectedArguments.push('--receipt-fixture');
  expect(JSON.parse(result.stdout)).toEqual({
    channel: changedHelper ? 'receipt-fixture' : receiptConfiguration.channel,
    args: expectedArguments
  });
  expect(expectedArguments.includes('--use-angle=swiftshader')).toBe(
    platform === 'linux' && Boolean(continuousIntegration)
  );
  expect(receiptConfiguration.observedAdapter).toBe(null);
});
