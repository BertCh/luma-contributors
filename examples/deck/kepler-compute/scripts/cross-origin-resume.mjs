// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Cross-origin mid-body resume (SHARING5). The remote server sits on another origin and, unlike
// csv-resume.mjs, exposes none of ETag/Accept-Ranges/Content-Range to the page:
//   hidden    - Last-Modified + Content-Length only; the importer must still resume by Range.
//   preflight - the CORS preflight rejects Range/If-Range; the importer must restart and discard.
//
// Usage: node scripts/cross-origin-resume.mjs [url] [hidden|preflight]
import assert from 'node:assert/strict';
import http from 'node:http';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {openNativeData, closeNativeDialog} from './native-navigation.mjs';

const mode = process.argv.includes('preflight') ? 'preflight' : 'hidden';
const lastModified = 'Tue, 01 Jan 2030 00:00:00 GMT';
const rowCount = 40;
const text =
  'id,note,value\n' +
  Array.from({length: rowCount}, (_, index) => `${index},"line ${index}\nsecond, line",${index * 2}`).join('\n') +
  '\n';
const bytes = Buffer.from(text);
const cutOffset = text.indexOf('second', Math.floor(text.length / 2)) + 3;
const requests = [];
const server = http.createServer((request, response) => {
  const cors = {
    'access-control-allow-origin': '*',
    // Nothing else is exposed: only CORS-safelisted response headers reach the page.
    ...(mode === 'preflight' ? {} : {'access-control-allow-headers': 'range, if-range'})
  };
  if (request.method === 'OPTIONS')
    return void response.writeHead(mode === 'preflight' ? 403 : 204, cors).end();
  const range = /bytes=(\d+)-/.exec(request.headers.range ?? '')?.[1];
  requests.push({range: request.headers.range ?? null, ifRange: request.headers['if-range'] ?? null});
  const first = requests.length === 1;
  const start = range ? Number(range) : 0;
  const body = bytes.subarray(start);
  response.writeHead(start ? 206 : 200, {
    ...cors,
    'content-type': 'text/csv',
    'last-modified': lastModified,
    'content-length': body.length,
    'accept-ranges': 'bytes',
    ...(start ? {'content-range': `bytes ${start}-${bytes.length - 1}/${bytes.length}`} : {})
  });
  if (first) {
    response.write(body.subarray(0, cutOffset));
    setTimeout(() => response.destroy(), 50);
  } else response.end(body);
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const csvUrl = `http://127.0.0.1:${server.address().port}/resume.csv`;

const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
page.setDefaultTimeout(30000);
const watchdog = setTimeout(() => { console.error('Cross-origin resume check exceeded 140 seconds'); void browser.close(); server.close(); process.exit(1); }, 140000);
await pinViteModules(page);
const failures = [];
page.on('pageerror', error => failures.push(error.message));
try {
  const given = process.argv[2]?.startsWith('http') ? process.argv[2] : 'http://127.0.0.1:5318/?rows=256';
  const base = given.includes('automatic=') ? given : `${given}${given.includes('?') ? '&' : '?'}automatic=retry`;
  await page.goto(base, {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
  await openNativeData(page);
  const control = page.getByTestId('remote-source-control');
  await control.getByLabel('Source URL').fill(csvUrl);
  await control.getByRole('button', {name: 'Open URL'}).click();
  await page.getByTestId('remote-source-status').filter({hasText: `${rowCount} rows`}).waitFor({timeout: 30000});
  assert(requests.length >= 2, 'the cut body must trigger a second request');
  if (mode === 'hidden') {
    assert(requests[1].range?.startsWith('bytes='), 'second request must use Range');
    assert.equal(requests[1].ifRange, lastModified, 'If-Range must carry Last-Modified');
  } else assert(requests.at(-1).range === null, 'after a rejected preflight the importer restarts without Range');
  await closeNativeDialog(page);
  assert.deepEqual(failures, []);
  console.log(JSON.stringify({passed: true, mode, rowCount, requests}));
} finally {
  clearTimeout(watchdog);
  await browser.close();
  server.close();
}
