// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {spawnSync} from 'node:child_process';
import {
  chmodSync,
  copyFileSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync
} from 'node:fs';
import {dirname, join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {afterEach, expect, it} from 'vitest';
import * as policy from '../../../../scripts/verification-policy.mjs';
import {CHECKED_TARBALL_BASE64} from '../analysis-core/checked-release-archive-fixture';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../../../..');
const directories: string[] = [];
function write(root: string, file: string, contents: string) {
  mkdirSync(dirname(join(root, file)), {recursive: true});
  writeFileSync(join(root, file), contents);
}
function fixture() {
  const parent = join(ROOT, '.typecheck/r19/scratch/tooling-tests-release-qualification-receipt');
  mkdirSync(parent, {recursive: true});
  const root = mkdtempSync(join(parent, 'fixture-'));
  directories.push(root);
  write(
    root,
    'package.json',
    '{"type":"module","peerDependencies":{"@luma.gl/core":"~10.0.0-alpha.16"}}'
  );
  write(root, 'yarn.lock', 'original lock');
  write(root, 'modules/analysis/src/value.ts', 'export const value = 1;');
  write(root, 'modules/analysis/test/value.node.spec.ts', '// node fixture');
  write(root, 'modules/analysis/test/value.spec.ts', '// browser fixture');
  write(
    root,
    'node_modules/@luma.gl/core/package.json',
    '{"name":"@luma.gl/core","version":"10.0.0-alpha.16","main":"index.js"}'
  );
  write(root, 'node_modules/@luma.gl/core/index.js', '');
  return root;
}
afterEach(() => {
  for (const root of directories.splice(0)) rmSync(root, {recursive: true, force: true});
});
const nodeFile = 'modules/analysis/test/value.node.spec.ts';
function report(root: string, statuses = ['passed']) {
  return {
    success: true,
    testResults: [
      {
        name: join(root, nodeFile),
        status: 'passed',
        assertionResults: statuses.map(status => ({title: status, status}))
      }
    ]
  };
}

it.each([
  'missing',
  'unexpected',
  'duplicate',
  'failed',
  'empty',
  'unknown',
  'failed assertion'
])('TOOL-111 rejects a %s Node report', mutation => {
  const root = fixture();
  const result = report(root);
  if (mutation === 'missing') result.testResults = [];
  if (mutation === 'unexpected') {
    write(root, 'other.node.spec.ts', '');
    result.testResults.push({...result.testResults[0]!, name: join(root, 'other.node.spec.ts')});
    expect(() => policy.reconcileTestReport(root, [nodeFile], result, 'node')).toThrow(
      'Node inventory mismatch: missing []; unexpected [other.node.spec.ts]'
    );
    return;
  }
  if (mutation === 'duplicate') result.testResults.push(result.testResults[0]!);
  if (mutation === 'failed') result.testResults[0]!.status = 'failed';
  if (mutation === 'empty') result.testResults[0]!.assertionResults = [];
  if (mutation === 'unknown') result.testResults[0]!.assertionResults[0]!.status = 'mystery';
  if (mutation === 'failed assertion')
    result.testResults[0]!.assertionResults[0]!.status = 'failed';
  expect(policy.reconcileTestReport).toBeTypeOf('function');
  expect(() => policy.reconcileTestReport(root, [nodeFile], result, 'node')).toThrow();
});
it('TOOL-111 records explicit Node skips but requires a passing case in every headless file', () => {
  const root = fixture();
  const reconciled = policy.reconcileTestReport(
    root,
    [nodeFile],
    report(root, ['skipped', 'todo']),
    'node'
  );
  expect(reconciled.suites[0].skipped).toHaveLength(2);
  expect(() =>
    policy.reconcileTestReport(root, [nodeFile], report(root, ['skipped', 'todo']), 'headless')
  ).toThrow('No passing test');
});
it.each([
  'source',
  'test',
  'lock',
  'installed version'
])('TOOL-105 independently identifies a %s edit', change => {
  const root = fixture();
  const before = policy.captureQualificationIdentity(root);
  const files: Record<string, string> = {
    source: 'modules/analysis/src/value.ts',
    test: nodeFile,
    lock: 'yarn.lock',
    'installed version': 'node_modules/@luma.gl/core/package.json'
  };
  write(
    root,
    files[change]!,
    change === 'installed version'
      ? '{"name":"@luma.gl/core","version":"10.0.0-alpha.17","main":"index.js"}'
      : 'changed bytes'
  );
  const after = policy.captureQualificationIdentity(root);
  expect(after.identitySha256).not.toBe(before.identitySha256);
  if (change === 'lock') expect(after.lockfileSha256).not.toBe(before.lockfileSha256);
  if (change === 'installed version') {
    expect(after.resolvedDependencies['@luma.gl/core']).toBe('10.0.0-alpha.17');
    expect(readFileSync(join(root, 'package.json'), 'utf8')).toContain('~10.0.0-alpha.16');
  }
});
function verificationFixture() {
  const root = fixture();
  for (const file of ['verify.mjs', 'verification-policy.mjs']) {
    write(root, `scripts/${file}`, '');
    copyFileSync(join(ROOT, 'scripts', file), join(root, 'scripts', file));
  }
  write(
    root,
    'disk.cjs',
    "require('node:fs').statfsSync = () => ({bsize:1024,bavail:16*1024**2}); require('node:module').syncBuiltinESMExports();"
  );
  write(
    root,
    'node_modules/.bin/ps',
    `#!${process.execPath}\nprocess.stdout.write(process.ppid + ' 1 1024\\n');`
  );
  write(
    root,
    'node_modules/.bin/yarn',
    `#!/bin/sh\nexec "${process.execPath}" "${join(root, 'runner.cjs')}" "$@"\n`
  );
  write(root, 'node_modules/.bin/npm', `#!${process.execPath}\nprocess.stdout.write('10.0.0');`);
  for (const file of ['ps', 'yarn', 'npm']) chmodSync(join(root, 'node_modules/.bin', file), 0o755);
  write(
    root,
    'runner.cjs',
    `
const fs = require('node:fs'); const path = require('node:path'); const args = process.argv.slice(2);
if (args[0] === '--version') { process.stdout.write('4.13.0'); process.exit(0); }
if (args[0] === 'lint' && process.env.MODE === 'fail') process.exit(7);
if (args[0] === 'lint' && process.env.MODE === 'interrupt') { process.kill(process.ppid, 'SIGTERM'); setInterval(() => {}, 1000); }
if (args[0] === 'build' && process.env.MODE === 'drift') fs.appendFileSync('modules/analysis/src/value.ts', '// drift');
if (args[0] === 'test-node' || args[0] === 'test-headless') {
 const files = args[0] === 'test-node' ? ['${nodeFile}'] : args.filter(value => value.endsWith('.spec.ts'));
 const suites = files.map(file => ({name:path.resolve(file), status:process.env.MODE === 'failed-report' ? 'failed' : 'passed', assertionResults:[{title:'fixture',status:process.env.MODE === 'failed-report' ? 'failed' : 'passed'}]}));
 const output = args.find(value => value.startsWith('--outputFile.json='));
 if (output && process.env.MODE !== 'stale') fs.writeFileSync(output.slice('--outputFile.json='.length), JSON.stringify({success:process.env.MODE !== 'failed-report',testResults:suites}));
 if (args[0] === 'test-node' && process.env.MODE === 'failed-report') process.exit(1);
 if (args[0] === 'test-node' && process.env.MODE === 'edit-restore') fs.writeFileSync('modules/analysis/src/value.ts', 'temporary edit');
 if (args[0] === 'test-headless' && process.env.MODE === 'edit-restore') fs.writeFileSync('modules/analysis/src/value.ts', 'export const value = 1;');
 if (args[0] === 'test-node' && process.env.MODE === 'membership') fs.writeFileSync('modules/analysis/test/added.node.spec.ts', '');
}
`
  );
  return root;
}
function run(root: string, mode = '', arguments_: string[] = []) {
  return spawnSync(process.execPath, [join(root, 'scripts/verify.mjs'), ...arguments_], {
    cwd: root,
    encoding: 'utf8',
    timeout: 15000,
    env: {
      ...process.env,
      NODE_OPTIONS: `--require=${join(root, 'disk.cjs')} --max-old-space-size=256`,
      PATH: `${join(root, 'node_modules/.bin')}:${process.env['PATH']}`,
      MODE: mode
    }
  });
}
function receipt(root: string) {
  return JSON.parse(readFileSync(join(root, '.typecheck/verify/latest.json'), 'utf8'));
}
it.each([
  '',
  'fail',
  'interrupt',
  'drift',
  'partial'
])('TOOL-105 publishes durable evidence for %s runs', mode => {
  const root = verificationFixture();
  const result = run(root, mode, mode === 'partial' ? ['--only', 'lint'] : []);
  expect(result.status, `${result.stdout}${result.stderr}`).toBe(
    mode === '' || mode === 'partial' ? 0 : 1
  );
  const record = receipt(root);
  expect(record.qualification).toBe(
    mode === '' ? 'full' : mode === 'partial' ? 'partial' : 'failed'
  );
  expect(record.options.maxWorkers).toBe(1);
  expect(record.commands.length).toBeGreaterThan(0);
  expect(Number.isFinite(record.commands[0].peakRSSBytes)).toBe(true);
  if (mode === '') {
    expect(record.inventories.node.selected).toEqual([nodeFile]);
    expect(record.inventories.headless.selected).toEqual(['modules/analysis/test/value.spec.ts']);
    expect(record.executedSteps).toHaveLength(10);
    expect(policy.isFullQualificationReceipt(record, record.startedIdentity)).toBe(true);
    const association = policy.associateReleaseQualification(record, record.startedIdentity, {
      sha256: 'abc',
      byteLength: 13
    });
    expect(association).toMatchObject({kind: 'full', tarballSha256: 'abc', tarballByteLength: 13});
    for (const qualification of ['running', 'partial', 'failed']) {
      const unfinished = {...record, qualification};
      expect(
        policy.associateReleaseQualification(unfinished, record.startedIdentity, {
          sha256: 'abc',
          byteLength: 13
        })
      ).toEqual({kind: 'build-package'});
    }
    for (const mutation of [
      'missing command',
      'empty headless',
      'duplicate suite',
      'nonfinite count',
      'missing finish'
    ]) {
      const incomplete = structuredClone(record);
      if (mutation === 'missing command') incomplete.commands.splice(1, 1);
      if (mutation === 'empty headless') incomplete.inventories.headless.suites[0].passed = 0;
      if (mutation === 'duplicate suite')
        incomplete.inventories.node.suites.push(incomplete.inventories.node.suites[0]);
      if (mutation === 'nonfinite count') incomplete.inventories.node.suites[0].passed = Number.NaN;
      if (mutation === 'missing finish') incomplete.finishedAt = null;
      expect(policy.isFullQualificationReceipt(incomplete, record.startedIdentity), mutation).toBe(
        false
      );
    }
    const previous = structuredClone(record);
    previous.commands[0].outcome = 'failed';
    expect(policy.isFullQualificationReceipt(previous, record.startedIdentity)).toBe(false);
    write(root, 'modules/analysis/src/value.ts', 'different stage');
    const differentStage = policy.captureQualificationIdentity(root);
    expect(policy.isFullQualificationReceipt(record, differentStage)).toBe(false);
    expect(
      policy.associateReleaseQualification(record, differentStage, {sha256: 'def', byteLength: 14})
    ).toEqual({kind: 'build-package'});
  }
});
it.each([
  'membership',
  'stale'
])('TOOL-111 rejects %s Node evidence despite process success', mode => {
  const root = verificationFixture();
  write(root, '.typecheck/verify/node.json', JSON.stringify(report(root)));
  const result = run(root, mode, ['--only', 'test-node']);
  expect(result.status, `${result.stdout}${result.stderr}`).toBe(1);
  if (mode === 'membership')
    expect(`${result.stdout}${result.stderr}`).toContain(
      'Node files changed during verification; rerun test-node'
    );
});

function releaseFixture() {
  const root = verificationFixture();
  copyFileSync(join(ROOT, 'scripts/build-release.mjs'), join(root, 'scripts/build-release.mjs'));
  copyFileSync(
    join(ROOT, 'scripts/packed-file-inventory.mjs'),
    join(root, 'scripts/packed-file-inventory.mjs')
  );
  for (const file of ['tsconfig.json', 'tsconfig.build.json']) write(root, file, '{}');
  for (const name of ['analysis', 'lenses', 'kepler']) {
    write(
      root,
      `modules/${name}/package.json`,
      JSON.stringify({name: `@luma-contributors/${name}`})
    );
    write(root, `modules/${name}/src/index.ts`, 'export const value = 1;');
  }
  write(
    root,
    'scripts/build.mjs',
    `
import {mkdirSync, writeFileSync} from 'node:fs'; import {join} from 'node:path';
export async function buildPackages(root) {
 for (const name of ['analysis', 'lenses', 'kepler']) {
  const output = join(root, 'modules', name, 'dist'); mkdirSync(output, {recursive:true});
  writeFileSync(join(output, 'index.js'), 'export const value = 1;');
 }
}`
  );
  write(
    root,
    'scripts/build-record.mjs',
    `import {writeFileSync,renameSync} from 'node:fs'; export function writeFileAtomic(file, text) {writeFileSync(file+'.tmp',text);renameSync(file+'.tmp',file);}`
  );
  write(root, 'scripts/typescript-project-inputs.mjs', '');
  write(root, 'scripts/verify-package-exports.mjs', '');
  write(
    root,
    'scripts/check-packed-consumer.mjs',
    `
import {createHash} from 'node:crypto';
import {writeFileSync} from 'node:fs';
import {join} from 'node:path';
const file = join(process.env.TMPDIR, 'checked.tgz');
const archive = Buffer.from(${JSON.stringify(CHECKED_TARBALL_BASE64)}, 'base64');
writeFileSync(file, archive);
console.log('checked tarball: ' + file);
if (process.env.MODE !== 'path-only') {
  const sha256 = createHash('sha256').update(archive).digest('hex');
  console.log('checked tarball identity: ' + JSON.stringify({path: file, sha256, byteLength: archive.byteLength}));
}
if (process.env.MODE === 'tamper-handoff') writeFileSync(file, 'replacement bytes');
`
  );
  return root;
}
function runRelease(root: string, mode = '') {
  return spawnSync(process.execPath, [join(root, 'scripts/build-release.mjs')], {
    cwd: root,
    encoding: 'utf8',
    timeout: 15000,
    env: {
      ...process.env,
      NODE_OPTIONS: '--max-old-space-size=256',
      MODE: mode,
      PATH: `${join(root, 'node_modules/.bin')}:${process.env['PATH']}`
    }
  });
}
it('TOOL-105 retains matching full receipt and archive identity, and leaves a different stage unqualified', () => {
  const root = releaseFixture();
  const verification = run(root);
  expect(verification.status, `${verification.stdout}${verification.stderr}`).toBe(0);
  const qualified = runRelease(root);
  expect(qualified.status, `${qualified.stdout}${qualified.stderr}`).toBe(0);
  const stage = JSON.parse(
    readFileSync(join(root, '.typecheck/release-build-latest.json'), 'utf8')
  );
  expect(stage.qualification.kind).toBe('full');
  // SHA-256 and byte length of the fixed gzip/ustar fixture.
  expect(stage.tarballSha256).toBe(
    'bba5a593cacca1cb5c321154c4d29c33c91f6f8cd1e1d14b04af301adc296874'
  );
  expect(stage.tarballByteLength).toBe(129);
  expect(stage.qualification.tarballSha256).toBe(stage.tarballSha256);
  expect(stage.qualification.tarballByteLength).toBe(stage.tarballByteLength);
  const retained = JSON.parse(readFileSync(stage.qualification.receipt.path, 'utf8'));
  expect(retained.id).toBe(receipt(root).id);
  write(root, 'modules/analysis/src/index.ts', 'export const value = 2;');
  const independent = runRelease(root);
  expect(independent.status, `${independent.stdout}${independent.stderr}`).toBe(0);
  const different = JSON.parse(
    readFileSync(join(root, '.typecheck/release-build-latest.json'), 'utf8')
  );
  expect(different.directory).not.toBe(stage.directory);
  expect(different.qualification).toEqual({kind: 'build-package'});
});

it('TOOL-111 preserves deliberate runtime exclusion opt-ins for Node membership', () => {
  const root = fixture();
  write(
    root,
    'test/kepler-compute-example-specs.json',
    JSON.stringify({excludedFromRuns: {[nodeFile]: 'optional integration'}})
  );
  expect(policy.listTestFiles(root, 'node', {})).toEqual([]);
  expect(
    policy.listTestFiles(root, 'node', {LUMA_CONTRIBUTORS_RUN_EXCLUDED_SPECS: 'true'})
  ).toEqual([nodeFile]);
});

it('TOOL-105 records a revision independently of same-name working-tree input bytes', () => {
  const root = fixture();
  const revision = '1111111111111111111111111111111111111111';
  write(
    root,
    'node_modules/.bin/git',
    `#!${process.execPath}\nconst args=process.argv.slice(2); process.stdout.write(args.includes('--show-toplevel') ? ${JSON.stringify(root)}+'\\n' : args.includes('HEAD') ? '${revision}\\n' : ${JSON.stringify(['package.json', 'yarn.lock', 'modules/analysis/src/value.ts', nodeFile].join('\0'))});`
  );
  chmodSync(join(root, 'node_modules/.bin/git'), 0o755);
  const originalPath = process.env['PATH'];
  process.env['PATH'] = `${join(root, 'node_modules/.bin')}:${originalPath}`;
  try {
    const before = policy.captureQualificationIdentity(root);
    expect(before.revision).toBe(revision);
    write(root, 'modules/analysis/src/value.ts', 'export const value = 7;');
    const after = policy.captureQualificationIdentity(root);
    expect(after.revision).toBe(revision);
    expect(after.contentSha256).not.toBe(before.contentSha256);
  } finally {
    process.env['PATH'] = originalPath;
  }
});

it('B1 rejects pending assertions that the installed reporter uses for unexecuted tests', () => {
  const root = fixture();
  expect(() =>
    policy.reconcileTestReport(root, [nodeFile], report(root, ['pending']), 'node')
  ).toThrow('nonpassing assertion');
});
function installGitStandIn(root: string, files: string[]) {
  write(
    root,
    'node_modules/.bin/git',
    `#!${process.execPath}\nconst argumentsList=process.argv.slice(2); process.stdout.write(argumentsList.includes('--show-toplevel') ? ${JSON.stringify(root)}+'\\n' : argumentsList.includes('HEAD') ? '1111111111111111111111111111111111111111\\n' : ${JSON.stringify(files.join('\0'))});`
  );
  chmodSync(join(root, 'node_modules/.bin/git'), 0o755);
}
function withFixturePath<T>(root: string, callback: () => T): T {
  const originalPath = process.env['PATH'];
  process.env['PATH'] = `${join(root, 'node_modules/.bin')}:${originalPath}`;
  try {
    return callback();
  } finally {
    process.env['PATH'] = originalPath;
  }
}
it.each([
  'cached Arrow',
  'excluded source',
  'excluded spec'
])('B2 identities include %s bytes omitted by Git', input => {
  const root = fixture();
  installGitStandIn(root, ['package.json', 'yarn.lock']);
  const files: Record<string, string> = {
    'cached Arrow':
      'examples/deck/kepler-compute/scripts/kepler-demo-cache/samples/nyc_earnings/data.arrow',
    'excluded source': 'modules/analysis/src/value.ts',
    'excluded spec': nodeFile
  };
  write(root, files[input]!, 'before');
  withFixturePath(root, () => {
    const before = policy.captureQualificationIdentity(root);
    write(root, files[input]!, 'after');
    expect(policy.captureQualificationIdentity(root).identitySha256).not.toBe(
      before.identitySha256
    );
  });
});
it('B3 refuses an edit later restored by the next test project', () => {
  const root = verificationFixture();
  const result = run(root, 'edit-restore');
  expect(result.status, `${result.stdout}${result.stderr}`).toBe(1);
  expect(`${result.stdout}${result.stderr}`).toContain('inputs changed');
  expect(receipt(root).qualification).toBe('failed');
});
it.each([
  '@vis.gl/dev-tools',
  'adapter playwright',
  'npm'
])('B4 identities include the actually invoked %s version', tool => {
  const root = fixture();
  const packagePath =
    tool === 'adapter playwright'
      ? 'node_modules/@vitest/browser-playwright/node_modules/playwright'
      : 'node_modules/@vis.gl/dev-tools';
  write(
    root,
    'node_modules/@vitest/browser-playwright/package.json',
    '{"name":"@vitest/browser-playwright","version":"5.0.2","main":"index.js"}'
  );
  write(root, 'node_modules/@vitest/browser-playwright/index.js', '');
  write(root, `${packagePath}/index.js`, '');
  const packageName = tool === 'adapter playwright' ? 'playwright' : '@vis.gl/dev-tools';
  write(
    root,
    `${packagePath}/package.json`,
    JSON.stringify({name: packageName, version: '1.0.0', main: 'index.js'})
  );
  write(root, '.typecheck/npm-version', '10.0.0');
  write(
    root,
    'node_modules/.bin/npm',
    `#!${process.execPath}\nprocess.stdout.write(require('node:fs').readFileSync(${JSON.stringify(join(root, '.typecheck/npm-version'))},'utf8'));`
  );
  chmodSync(join(root, 'node_modules/.bin/npm'), 0o755);
  withFixturePath(root, () => {
    const before = policy.captureQualificationIdentity(root);
    if (tool === 'npm') write(root, '.typecheck/npm-version', '10.0.1');
    else
      write(
        root,
        `${packagePath}/package.json`,
        JSON.stringify({name: packageName, version: '1.0.1', main: 'index.js'})
      );
    const after = policy.captureQualificationIdentity(root);
    expect(after.lockfileSha256).toBe(before.lockfileSha256);
    expect(after.identitySha256).not.toBe(before.identitySha256);
  });
});
it('B5 refuses full qualification when staged packages resolve a different local peer', () => {
  const root = releaseFixture();
  write(
    root,
    'modules/analysis/node_modules/@luma.gl/core/package.json',
    '{"name":"@luma.gl/core","version":"10.0.0-alpha.17","main":"index.js"}'
  );
  write(root, 'modules/analysis/node_modules/@luma.gl/core/index.js', '');
  expect(run(root).status).toBe(0);
  const result = runRelease(root);
  expect(result.status, `${result.stdout}${result.stderr}`).toBe(0);
  const stage = JSON.parse(
    readFileSync(join(root, '.typecheck/release-build-latest.json'), 'utf8')
  );
  expect(stage.qualification.kind).toBe('build-package');
  expect(stage.stagedResolvedWorkspaceDependencies.analysis['@luma.gl/core']).toBe(
    '10.0.0-alpha.16'
  );
});
it('B6 requires every headless chunk and safely rejects malformed nested receipt evidence', () => {
  const root = verificationFixture();
  write(root, 'modules/analysis/test/second/second.spec.ts', '');
  expect(run(root).status).toBe(0);
  const valid = receipt(root);
  for (const mutation of [
    'missing chunk',
    'null suite',
    'null command',
    'missing id',
    'wrong files'
  ]) {
    const broken = structuredClone(valid);
    if (mutation === 'missing chunk') broken.commands.pop();
    if (mutation === 'null suite') broken.inventories.node.suites[0] = null;
    if (mutation === 'null command') broken.commands[0] = null;
    if (mutation === 'missing id') delete broken.id;
    if (mutation === 'wrong files') broken.commands.at(-1).arguments[1] = 'unselected.spec.ts';
    expect(
      policy.associateReleaseQualification(broken, valid.startedIdentity, {
        sha256: 'abc',
        byteLength: 13
      }),
      mutation
    ).toEqual({kind: 'build-package'});
  }
});
it('B7 retains selection, attempted steps and a failed fresh report after a subsequent run', () => {
  const root = verificationFixture();
  expect(run(root, 'failed-report').status).toBe(1);
  const failed = receipt(root);
  expect(failed.attemptedSteps).toContain('test-node');
  expect(failed.inventories.node.selected).toEqual([nodeFile]);
  const failedReport = failed.inventories.node.reports[0].file;
  expect(JSON.parse(readFileSync(failedReport, 'utf8')).success).toBe(false);
  expect(run(root).status).toBe(0);
  expect(receipt(root).inventories.node.reports[0].file).not.toBe(failedReport);
  expect(JSON.parse(readFileSync(failedReport, 'utf8')).success).toBe(false);
});
it('S1 records the configured Linux CI software-renderer request separately from observations', () => {
  const configuration = policy.getVerificationBrowserConfiguration({CI: 'true'}, 'linux');
  expect(configuration.browser).toBe('chromium');
  expect(configuration.launchArguments).toEqual([
    '--enable-unsafe-webgpu',
    '--ignore-gpu-blocklist',
    '--use-angle=swiftshader',
    '--enable-unsafe-swiftshader'
  ]);
  expect(configuration.observedAdapter).toBe(null);
});
it('S2 permits full execution from the first step and documents the exception', () => {
  const root = verificationFixture();
  expect(run(root, '', ['--from', 'lint']).status).toBe(0);
  expect(receipt(root).qualification).toBe('full');
  expect(readFileSync(join(ROOT, 'dev-docs/testing-gates.md'), 'utf8')).toContain('`--from lint`');
});
it('heap parsing refuses an entire malformed value even when a later CLI value overrides it', () => {
  expect(() =>
    policy.getChildNodeOptions('--max-old-space-size=16384=oops', ['--max-old-space-size=256'])
  ).toThrow('at most 8192 MiB');
});
it('explicit partial capture remains successful while changed inputs cannot qualify', () => {
  const root = verificationFixture();
  const result = run(root, 'drift', ['--only', 'build']);
  expect(result.status, `${result.stdout}${result.stderr}`).toBe(0);
  expect(receipt(root).qualification).toBe('partial');
  expect(receipt(root).inputChanges.length).toBeGreaterThan(0);
});

it('archive handoff verifies the digest reported for extracted bytes before retaining replacements', () => {
  const root = releaseFixture();
  const result = runRelease(root, 'tamper-handoff');
  expect(result.status, `${result.stdout}${result.stderr}`).toBe(1);
  expect(`${result.stdout}${result.stderr}`).toContain('differs from the checked tarball');
});
it('path-only packed-checker handoffs remain independently usable build-package evidence', () => {
  const root = releaseFixture();
  expect(run(root).status).toBe(0);
  const result = runRelease(root, 'path-only');
  expect(result.status, `${result.stdout}${result.stderr}`).toBe(0);
  const stage = JSON.parse(
    readFileSync(join(root, '.typecheck/release-build-latest.json'), 'utf8')
  );
  expect(stage.qualification.kind).toBe('build-package');
  expect(stage.archiveHandoff.kind).toBe('path-only');
});

it('B7 retains fresh failed headless reports using run-specific paths', () => {
  const root = verificationFixture();
  const failed = run(root, 'failed-report', ['--only', 'test-headless']);
  expect(failed.status).toBe(1);
  const previous = receipt(root);
  expect(previous.attemptedSteps).toEqual(['test-headless']);
  expect(previous.executedSteps).toEqual([]);
  expect(previous.inventories.headless.selected).toEqual(['modules/analysis/test/value.spec.ts']);
  const file = previous.inventories.headless.reports[0].file;
  expect(JSON.parse(readFileSync(file, 'utf8')).success).toBe(false);
  expect(run(root, '', ['--only', 'test-headless']).status).toBe(0);
  expect(receipt(root).inventories.headless.reports[0].file).not.toBe(file);
  expect(JSON.parse(readFileSync(file, 'utf8')).success).toBe(false);
});
