// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';
import {closeSync, openSync} from 'node:fs';
import {mkdir, writeFile} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

// Run only while holding the shared GPU validation slot. Each child closes its browser before
// the next starts. A bundle digest binds every report to the frozen served production artifacts.
const baseUrl = process.argv[2] ?? 'http://127.0.0.1:5313/';
const outputDirectory = path.resolve(process.argv[3] ?? '/tmp/geometry-observation-batch');
const requestedSuites = process.argv.slice(4);
const suites = [
  ['coverage-audit', 'check-coverage-audit-reference.mjs'],
  ['county-linework', 'check-county-linework-reference.mjs'],
  ['footprint-shape', 'check-footprint-shape-reference.mjs'],
  ['sampling-frames', 'check-sampling-frames-reference.mjs'],
  ['block-relationships', 'check-block-relationships-reference.mjs'],
  ['block-smoke', 'check-block-relationships-reference.mjs', ['--smoke']],
  ['earthquake-inhomogeneous', 'check-earthquake-inhomogeneous-reference.mjs']
].filter(([name]) => requestedSuites.length === 0 ? name !== 'block-smoke' : requestedSuites.includes(name));
assert.ok(suites.length > 0, 'Select at least one known numerical suite');
await mkdir(outputDirectory, {recursive: true});
const readBuildEvidence = async () => {
  const response = await fetch(new URL('build-evidence.json', baseUrl), {cache: 'no-store', headers: {connection: 'close'}});
  assert.ok(response.ok, 'Production build evidence must be served');
  const evidence = await response.json();
  assert.match(evidence.bundleSha256, /^[a-f0-9]{64}$/);
  return evidence;
};
const initialEvidence = await readBuildEvidence();
await writeFile(path.join(outputDirectory, 'build-evidence.json'), `${JSON.stringify(initialEvidence, null, 2)}\n`);
const results = [];
let bundleUnchanged = true;
for (const [name, script, extraArguments = []] of suites) {
  const logPath = path.join(outputDirectory, `${name}.log`);
  const reportPath = path.join(outputDirectory, `${name}.json`);
  const logDescriptor = openSync(logPath, 'w');
  let result;
  try {
    result = spawnSync(process.execPath, [fileURLToPath(new URL(script, import.meta.url)), baseUrl, reportPath, ...extraArguments], {
      stdio: ['ignore', logDescriptor, logDescriptor],
      timeout: 15 * 60 * 1000
    });
  } finally {
    closeSync(logDescriptor);
  }
  const currentEvidence = await readBuildEvidence();
  bundleUnchanged &&= currentEvidence.bundleSha256 === initialEvidence.bundleSha256;
  results.push({name, passed: result.status === 0 && !result.error, exitCode: result.status,
    error: result.error ? String(result.error) : null, logPath, reportPath,
    bundleSha256: initialEvidence.bundleSha256, bundleUnchanged});
  await writeFile(path.join(outputDirectory, 'report.json'), `${JSON.stringify({baseUrl, bundleSha256: initialEvidence.bundleSha256, bundleUnchanged, results}, null, 2)}\n`);
  console.log(`${result.status === 0 ? 'PASS' : 'FAIL'} ${name}; bundle unchanged: ${bundleUnchanged}`);
  // A killed browser process may retain GPU resources; stop rather than start another suite.
  if (result.error || !bundleUnchanged) break;
}
const finalEvidence = await readBuildEvidence();
await writeFile(path.join(outputDirectory, 'build-evidence-after.json'), `${JSON.stringify(finalEvidence, null, 2)}\n`);
assert.equal(finalEvidence.bundleSha256, initialEvidence.bundleSha256, 'Production bundle changed during validation');
assert.equal(results.length, suites.length, 'All selected suites must complete');
assert.ok(results.every(result => result.passed), `Numerical failures are recorded in ${outputDirectory}`);
