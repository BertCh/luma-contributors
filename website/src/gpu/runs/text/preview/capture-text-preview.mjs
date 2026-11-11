// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Photographs text-preview.html for every case, theme and size on WebGPU, writes one PNG per
// picture, the poster scaled to card size the way the site shows it, and a contact sheet.
// Needs a Vite server for the website folder (the run harness server on port 3301 serves it):
//
//   node website/src/gpu/runs/text/preview/capture-text-preview.mjs <output folder> [--port 3301]

import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';

const directory = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(path.resolve(directory, '../../../../../../package.json'));
const {chromium} = require('playwright');

const argumentsList = process.argv.slice(2);
const portIndex = argumentsList.indexOf('--port');
const port = portIndex >= 0 ? Number(argumentsList[portIndex + 1]) : 3301;
const output = path.resolve(
  argumentsList.find(
    (argument, index) => !argument.startsWith('--') && (portIndex < 0 || index !== portIndex + 1)
  ) ?? 'text-preview'
);
fs.mkdirSync(output, {recursive: true});

const CASES = ['axes', 'bars', 'legend', 'matrix'];
const THEMES = ['dark', 'light'];
/** Device pixel ratio per size: cards and hero on a 2x display; the poster harness at 1x. */
const SIZES = {card: 2, card1: 1, hero: 2, poster: 1};
const CARD_WIDTH = 265;

const browser = await chromium.launch({
  args: [
    '--enable-unsafe-webgpu',
    '--ignore-gpu-blocklist',
    '--enable-features=Vulkan,WebGPUService',
    ...(process.platform === 'darwin' ? ['--use-angle=metal'] : [])
  ]
});
const results = [];
try {
  for (const [size, ratio] of Object.entries(SIZES)) {
    const context = await browser.newContext({
      viewport: {width: 1000, height: 640},
      deviceScaleFactor: ratio
    });
    for (const previewCase of CASES) {
      for (const theme of THEMES) {
        const page = await context.newPage();
        const messages = [];
        page.on('console', message => {
          if (message.type() === 'warning' || message.type() === 'error') messages.push(message.text());
        });
        // Every other picture takes the lazy path: layers first, atlas later, then a redraw.
        const lazy = (CASES.indexOf(previewCase) + THEMES.indexOf(theme)) % 2 === 1 ? '1' : '0';
        const query = new URLSearchParams({case: previewCase, theme, size, lazy});
        await page.goto(`http://localhost:${port}/src/gpu/runs/text/preview/text-preview.html?${query}`);
        await page.waitForFunction(() => window.textPreview?.ready, null, {timeout: 60000});
        const state = await page.evaluate(() => window.textPreview);
        const file = path.join(output, `${previewCase}-${theme}-${size}.png`);
        await page.locator('canvas').screenshot({path: file});
        results.push({previewCase, theme, size, file, lazy, ...state, messages});
        await page.close();
      }
    }
    await context.close();
  }
  // Posters as a card shows them: 640 px WebP, then the card's 265 CSS pixels at 2x.
  const converter = await (await browser.newContext({deviceScaleFactor: 1})).newPage();
  for (const result of results.filter(item => item.size === 'poster')) {
    const png = fs.readFileSync(result.file).toString('base64');
    const scaled = await converter.evaluate(
      async ([data, width]) => {
        const image = new Image();
        image.src = `data:image/png;base64,${data}`;
        await image.decode();
        const resize = (source, targetWidth) => {
          const canvas = document.createElement('canvas');
          canvas.width = targetWidth;
          canvas.height = Math.round((targetWidth * source.height) / source.width);
          const context = canvas.getContext('2d');
          context.imageSmoothingQuality = 'high';
          context.drawImage(source, 0, 0, canvas.width, canvas.height);
          return canvas;
        };
        const webp = resize(image, 640);
        const encoded = new Image();
        encoded.src = webp.toDataURL('image/webp', 0.62);
        await encoded.decode();
        return resize(encoded, width).toDataURL('image/png').split(',')[1];
      },
      [png, CARD_WIDTH * 2]
    );
    const file = result.file.replace('-poster.png', '-poster-on-card.png');
    fs.writeFileSync(file, Buffer.from(scaled, 'base64'));
    results.push({...result, size: 'poster-on-card', file});
  }
  // Contact sheet: per case and theme, card (2x), poster as shown on the card, hero (2x).
  const sheet = await (await browser.newContext({deviceScaleFactor: 1, viewport: {width: 3200, height: 800}})).newPage();
  const image = file => `data:image/png;base64,${fs.readFileSync(file).toString('base64')}`;
  const rows = CASES.flatMap(previewCase =>
    THEMES.map(theme => {
      const find = size => results.find(item => item.previewCase === previewCase && item.theme === theme && item.size === size);
      const cell = (size, width) => {
        const item = find(size);
        return `<figure><img src="${image(item.file)}" style="width:${width}px"><figcaption>${previewCase} · ${theme} · ${size} · ${item.width ?? ''}×${item.height ?? ''} · drawn ${item.placed?.length ?? '?'} dropped ${item.dropped?.length ?? '?'}</figcaption></figure>`;
      };
      return `<div class="row">${cell('card1', 265)}${cell('card', 530)}${cell('poster-on-card', 530)}${cell('hero', 1774)}</div>`;
    })
  );
  await sheet.setContent(`<!doctype html><html><head><style>
    body { margin: 12px; background: #777; font: 14px sans-serif; color: #fff; }
    .row { display: flex; gap: 16px; align-items: flex-start; margin-bottom: 16px; }
    figure { margin: 0; } img { display: block; }
  </style></head><body>${rows.join('')}</body></html>`);
  await sheet.screenshot({path: path.join(output, 'contact-sheet.png'), fullPage: true});
} finally {
  await browser.close();
}
fs.writeFileSync(
  path.join(output, 'results.json'),
  JSON.stringify(
    results.map(({previewCase, theme, size, file, lazy, errors, messages, placed, dropped, instanceCount, layoutMilliseconds, textMilliseconds, atlasMilliseconds, firstFrameMilliseconds}) => ({
      previewCase,
      theme,
      size,
      file: path.basename(file),
      lazy,
      errors,
      messages,
      placed,
      dropped,
      instanceCount,
      layoutMilliseconds,
      textMilliseconds,
      atlasMilliseconds,
      firstFrameMilliseconds
    })),
    null,
    2
  )
);
const failures = results.filter(item => item.errors?.length);
console.log(`${results.length} pictures in ${output}; ${failures.length} with errors`);
for (const failure of failures) console.log(failure.previewCase, failure.theme, failure.size, failure.errors);
