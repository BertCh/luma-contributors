// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import assert from 'node:assert/strict';
import {chromium} from 'playwright';
const browser = await chromium.launch({headless: true});
const page = await browser.newPage();
try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/scripts/fixtures/sandbox-share.html');
  await page.waitForFunction(() => window.shareChecks);
  const result = await page.evaluate(() => {
    const {createSandboxShareLink, readSandboxShareLink} = window.shareChecks;
    const state = {
      schemaVersion: 1,
      sources: [{dataset: {datasetId: 'resident-observations', revision: 1}, reference: {kind: 'fixture', id: 'luma-kepler-observations', version: 1, parameters: {seed: 42, rowCount: 120}}, schema: [{name: 'cost € 🌍', type: 'Float32', nullable: true}], normalization: {}}],
      nativeConfig: {config: {visState: {filters: [], layers: [{config: {dataId: 'resident-observations', label: 'München 🌍'}}]}}},
      selections: [{source: {datasetId: 'resident-observations', revision: 1}, revision: 1, value: {kind: 'native-kepler-config', filters: []}}], analyses: []
    };
    const link = createSandboxShareLink(JSON.stringify(state), 'https://example.com/sandbox/?secret=remove#old');
    const rejects = operation => {try {operation(); return false;} catch {return true;}};
    return {
      link,
      restored: JSON.parse(readSandboxShareLink(link)),
      unrelated: readSandboxShareLink('https://example.com/#map'),
      malformed: ['v2.abc', 'v1.!bad', 'v1.a', 'v1._w'].every(value => rejects(() => readSandboxShareLink(`https://example.com/#sandbox=${value}`))),
      oversized: rejects(() => createSandboxShareLink(JSON.stringify({...state, nativeConfig: {text: 'a'.repeat(50000)}}), 'https://example.com/')),
      missingSource: rejects(() => createSandboxShareLink(JSON.stringify({...state, nativeConfig: {dataId: 'missing'}}), 'https://example.com/')),
      localSource: rejects(() => createSandboxShareLink(JSON.stringify({...state, sources: [{...state.sources[0], reference: {kind: 'application', id: 'local-file', version: 1, parameters: null}}]}), 'https://example.com/')),
      credentials: rejects(() => createSandboxShareLink(JSON.stringify(state), 'https://user:password@example.com/')),
      unsupportedProtocol: rejects(() => createSandboxShareLink(JSON.stringify(state), 'file:///sandbox/')),
      expected: state
    };
  });
  assert.equal(new URL(result.link).search, '?rows=120');
  assert.equal(new URL(result.link).pathname, '/sandbox/');
  assert.deepEqual(result.restored, result.expected);
  assert.equal(result.unrelated, null);
  for (const key of ['malformed', 'oversized', 'missingSource', 'localSource', 'credentials', 'unsupportedProtocol']) assert.equal(result[key], true, key);
  console.log('Sandbox share checks passed: Unicode round trip, bounded decoding, source coverage, fixture-only sharing and clean destination URL.');
} finally {await browser.close();}
