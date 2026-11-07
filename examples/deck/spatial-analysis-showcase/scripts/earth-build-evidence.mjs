// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import assert from 'node:assert/strict';
import {readFile, writeFile} from 'node:fs/promises';

/** Bind numerical evidence to one served production bundle, including failed runs. */
export async function captureBuildEvidence(baseUrl, outputDirectory) {
  const evidenceUrl = new URL('build-evidence.json', baseUrl);
  const initialResponse = await fetch(evidenceUrl, {cache: 'no-store'});
  assert.ok(initialResponse.ok, 'The served production build must provide build-evidence.json');
  const initialEvidence = await initialResponse.json();
  assert.match(initialEvidence.bundleSha256, /^[a-f0-9]{64}$/);
  await writeFile(`${outputDirectory}/build-evidence.json`, `${JSON.stringify(initialEvidence, null, 2)}\n`);
  return async () => {
    const finalResponse = await fetch(evidenceUrl, {cache: 'no-store'});
    assert.ok(finalResponse.ok, 'The build evidence must remain available after verification');
    const finalEvidence = await finalResponse.json();
    const bundleUnchanged = finalEvidence.bundleSha256 === initialEvidence.bundleSha256;
    const verification = {
      bundleSha256: initialEvidence.bundleSha256,
      finalBundleSha256: finalEvidence.bundleSha256,
      bundleUnchanged
    };
    await writeFile(`${outputDirectory}/build-verification.json`, `${JSON.stringify(verification, null, 2)}\n`);
    let report;
    try {
      report = JSON.parse(await readFile(`${outputDirectory}/report.json`, 'utf8'));
    } catch (error) {
      if (error.code !== 'ENOENT') throw error;
      report = {passed: false, error: 'The numerical script did not produce its report'};
    }
    Object.assign(report, verification);
    if (!bundleUnchanged) report.passed = false;
    await writeFile(`${outputDirectory}/report.json`, `${JSON.stringify(report, null, 2)}\n`);
    assert.ok(bundleUnchanged, 'The served bundle changed during numerical verification');
  };
}
