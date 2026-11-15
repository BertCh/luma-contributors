// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Mid-body CSV remote import resume (?automatic=retry): a local server cuts the first response
// inside a quoted multi-line field; the importer resumes with Range/If-Range (or restarts and
// discards the delivered prefix) and must load exactly the original rows, header once.
//
// Usage: node scripts/csv-resume.mjs [url] [range|norange]
import assert from 'node:assert/strict';
import http from 'node:http';
import {chromium} from 'playwright';
import {getSourceCheckUrl} from './source-check-utils.mjs';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {openNativeData, closeNativeDialog} from './native-navigation.mjs';

const mode = process.argv.includes('norange') ? 'norange' : 'range';
const rowCount = 40;
const text =
  'id,note,value\n' +
  Array.from(
    {length: rowCount},
    (_, index) => `${index},"line ${index}\nsecond, line",${index * 2}`
  ).join('\n') +
  '\n';
const bytes = Buffer.from(text);
const cutOffset = text.indexOf('second', Math.floor(text.length / 2)) + 3;
const etag = '"csv-resume-fixture"';
const requests = [];
const server = http.createServer((request, response) => {
  const cors = {
    'access-control-allow-origin': '*',
    'access-control-allow-headers': 'range, if-range',
    'access-control-expose-headers': 'etag, accept-ranges, content-range, content-length'
  };
  if (request.method === 'OPTIONS') return void response.writeHead(204, cors).end();
  const range = /bytes=(\d+)-/.exec(request.headers.range ?? '')?.[1];
  requests.push({
    range: request.headers.range ?? null,
    ifRange: request.headers['if-range'] ?? null
  });
  const first = requests.length === 1;
  const start = mode === 'range' && range ? Number(range) : 0;
  const body = bytes.subarray(start);
  response.writeHead(start ? 206 : 200, {
    ...cors,
    'content-type': 'text/csv',
    etag,
    'content-length': body.length,
    // Both the old and current importer optimistically try hidden cross-origin ranges.
    // Advertise an explicit refusal so norange tests restart without a Range request.
    'accept-ranges': mode === 'range' ? 'bytes' : 'none',
    ...(start ? {'content-range': `bytes ${start}-${bytes.length - 1}/${bytes.length}`} : {})
  });
  if (first) {
    response.write(body.subarray(0, cutOffset));
    setTimeout(() => response.destroy(), 50);
  } else response.end(body);
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const csvUrl = `http://127.0.0.1:${server.address().port}/resume.csv`;

const browser = await chromium.launch(
  getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'})
);
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
page.setDefaultTimeout(30000);
const watchdog = setTimeout(() => {
  console.error('CSV resume check exceeded 140 seconds');
  void browser.close();
  server.close();
  process.exit(1);
}, 140000);
await pinViteModules(page);
const failures = [];
page.on('pageerror', error => failures.push(error.message));
try {
  const base = getSourceCheckUrl(process.argv[2], {rows: 256, retry: true});
  await page.goto(base, {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(
    () => window.__keplerComputeTest?.getState().selectedCount === 256,
    undefined,
    {timeout: 60000}
  );
  await openNativeData(page);
  const control = page.getByTestId('remote-source-control');
  await control.getByLabel('Source URL').fill(csvUrl);
  await control.getByRole('button', {name: 'Open URL'}).click();
  await page
    .getByTestId('remote-source-status')
    .filter({hasText: `${rowCount} rows`})
    .waitFor({timeout: 30000});
  assert(requests.length >= 2, 'the cut body must trigger a second request');
  if (mode === 'range') {
    assert(requests[1].range?.startsWith('bytes='), 'second request must use Range');
    assert.equal(requests[1].ifRange, etag, 'resume must carry the original validator');
  } else assert.equal(requests[1].range, null);
  await closeNativeDialog(page);
  assert.deepEqual(failures, []);
  console.log(JSON.stringify({passed: true, mode, rowCount, requests}));
} finally {
  clearTimeout(watchdog);
  await browser.close();
  server.close();
}
