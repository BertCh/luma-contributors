// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Browser check (not run by authors): shared consumer notices and analysis legend/tooltip.
// Usage: node scripts/legend-analysis-consumers.mjs [url]
import {chromium} from 'playwright';

const url = process.argv[2] ?? 'http://localhost:5318/';
const browser = await chromium.launch({args: ['--enable-unsafe-webgpu']});
const page = await browser.newPage();
await page.goto(url);
await page.waitForSelector('[data-resident-style-legend], .map-control', {timeout: 120000});
const report = await page.evaluate(() => ({
  consumerStates: [...document.querySelectorAll('[data-resident-consumer-state]')].map(node => ({
    consumer: node.getAttribute('data-resident-consumer'),
    state: node.getAttribute('data-resident-consumer-state')
  })),
  analysisLegend: Boolean(document.querySelector('[data-resident-analysis-legend]')),
  channelLegends: [...document.querySelectorAll('[data-resident-channel-legend]')].map(node =>
    node.getAttribute('data-resident-channel-legend')
  )
}));
console.log(JSON.stringify(report, null, 2));
await browser.close();
