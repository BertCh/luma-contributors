// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Mid-body remote stream resume (?automatic=retry): a local server cuts the first response part way
// through the body; the stream resumes with Range/If-Range (or restarts and discards the delivered
// prefix) and must admit exactly the original batches and rows, never duplicated or relabelled.
//
// Usage: node scripts/stream-resume.mjs [url] [range|norange]
import assert from 'node:assert/strict';
import http from 'node:http';
import * as arrow from 'apache-arrow';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {openNativeData, closeNativeDialog} from './native-navigation.mjs';

const mode = process.argv.includes('norange') ? 'norange' : 'range';
const batches = [[1, 2], [], [3, 4, 5]].map(values =>
  arrow.tableFromIPC(arrow.tableToIPC(new arrow.Table({amount: arrow.vectorFromArray(values, new arrow.Float32())}), 'stream')).batches[0]
);
const bytes = Buffer.from(arrow.tableToIPC(new arrow.Table(batches[0].schema, batches), 'stream'));
const etag = '"stream-resume-fixture"';
const requests = [];
const server = http.createServer((request, response) => {
  const cors = {
    'access-control-allow-origin': '*',
    'access-control-allow-headers': 'range, if-range',
    'access-control-expose-headers': 'etag, accept-ranges, content-range, content-length'
  };
  if (request.method === 'OPTIONS') return void response.writeHead(204, cors).end();
  const range = /bytes=(\d+)-/.exec(request.headers.range ?? '')?.[1];
  requests.push({range: request.headers.range ?? null, ifRange: request.headers['if-range'] ?? null});
  const first = requests.length === 1;
  const start = mode === 'range' && range ? Number(range) : 0;
  const body = bytes.subarray(start);
  response.writeHead(start ? 206 : 200, {
    ...cors,
    'content-type': 'application/vnd.apache.arrow.stream',
    etag,
    'content-length': body.length,
    ...(mode === 'range' ? {'accept-ranges': 'bytes'} : {}),
    ...(start ? {'content-range': `bytes ${start}-${bytes.length - 1}/${bytes.length}`} : {})
  });
  if (first) {
    // Send roughly half of the body, then drop the connection mid-body.
    response.write(body.subarray(0, Math.floor(body.length / 2)));
    setTimeout(() => response.destroy(), 50);
  } else response.end(body);
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const streamUrl = `http://127.0.0.1:${server.address().port}/resume.arrows`;

const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
page.setDefaultTimeout(30000);
const watchdog = setTimeout(() => { console.error('Stream resume check exceeded 140 seconds'); void browser.close(); server.close(); process.exit(1); }, 140000);
await pinViteModules(page);
const failures = [];
page.on('pageerror', error => failures.push(error.message));
try {
  const given = process.argv[2]?.startsWith('http') ? process.argv[2] : 'http://127.0.0.1:5318/?rows=256';
  const base = given.includes('automatic=') ? given : `${given}${given.includes('?') ? '&' : '?'}automatic=retry`;
  await page.goto(base, {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
  await openNativeData(page);
  const panel = page.getByTestId('arrow-stream-panel');
  await panel.getByLabel(/Remote Arrow IPC stream URL/).fill(streamUrl);
  await panel.getByRole('button', {name: 'Open stream', exact: true}).click();
  await panel.getByRole('status').filter({hasText: 'paused: 1 batches, 2 admitted rows'}).waitFor();
  await panel.getByRole('button', {name: 'Continue stream', exact: true}).click();
  await panel.getByRole('status').filter({hasText: 'exhausted: 3 batches, 5 admitted rows'}).waitFor();
  await panel.getByRole('button', {name: 'Finish stream', exact: true}).click();
  await panel.getByRole('status').filter({hasText: 'complete: 3 batches, 5 admitted rows'}).waitFor();
  assert.match(await panel.innerText(), /Connection (resumed|restarted) after 1 interruption/);
  assert(requests.length >= 2, 'the cut body must trigger a second request');
  if (mode === 'range') assert(requests[1].range?.startsWith('bytes='), 'second request must use Range');
  else assert.equal(requests[1].range, null);
  await panel.getByRole('button', {name: 'Add completed stream', exact: true}).click();
  await page.locator('.ReactModal__Overlay').waitFor({state: 'hidden'});
  await page.waitForFunction(() => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 5);
  assert.deepEqual((await page.evaluate(() => window.__keplerComputeTest.getState().publications[0])).sourceBatches, [2, 0, 3]);
  await closeNativeDialog(page);
  assert.deepEqual(failures, []);
  console.log(JSON.stringify({passed: true, mode, requests}));
} finally {
  clearTimeout(watchdog);
  await browser.close();
  server.close();
}
