// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Run against a built site or dev server: node scripts/check-card-interactions.mjs [base URL]
// Checks one catalog card per pointer gesture (hover, keyboard, navigation), then the poster-only,
// no-GPU and touch paths on the first entry.
import assert from 'node:assert/strict';
import {chromium} from 'playwright';

const base = process.argv[2] ?? 'http://localhost:3000';
// One entry per gesture kind, plus a flagship demo.
const ENTRIES = [
  'raster-arithmetic',
  'viewshed', // place
  'slope-aspect-hillshade', // aim
  'buffer-surface', // reach
  'raster-conditional', // level
  'rectangle-clip', // window
  'fill-nodata', // swipe
  'neighbor-search', // probe
  'polygon-overlay', // drag
  'trajectory-playhead', // time
  'relief-blend', // pad
  'kernel-density' // flagship demo
];
const getUrl = entry => new URL(`/explore?q=${entry}`, base).href;
const getCardSelector = entry => `.entry-card:has(a[href="/fn/${entry}"])`;
const browser = await chromium.launch({
  args: [
    '--enable-unsafe-webgpu',
    '--ignore-gpu-blocklist',
    ...(process.platform === 'darwin' ? ['--use-angle=metal'] : [])
  ]
});

async function openCard(page, entry) {
  await page.goto(getUrl(entry));
  const card = page.locator(getCardSelector(entry)).first();
  // Center the card: at the top of the viewport the sticky filter bar would cover it.
  // Instant, so the card's position is final before it is measured.
  await card.evaluate(element => element.scrollIntoView({block: 'center', behavior: 'instant'}));
  await card.locator('.run-snapshot img:visible').evaluate(image => image.decode());
  assert.equal(await card.locator('canvas').count(), 0, 'Idle cards must have only a poster.');
  return card;
}

/** The element's box once the page has stopped moving it (filtering and scrolling settle late). */
async function getStableBox(page, locator) {
  let previous = await locator.boundingBox();
  for (let attempt = 0; attempt < 20; attempt++) {
    await page.waitForTimeout(150);
    const next = await locator.boundingBox();
    if (next.x === previous.x && next.y === previous.y) return next;
    previous = next;
  }
  return previous;
}

/** Screenshots until the canvas differs from `before`; false when it never does. */
async function waitForChange(canvas, before) {
  for (let attempt = 0; attempt < 30; attempt++) {
    if (!(await canvas.screenshot()).equals(before)) return true;
  }
  return false;
}

async function verifyLiveCard(page, entry) {
  const card = await openCard(page, entry);
  const link = card.locator('a');
  const media = card.locator('.card__media');
  const ready = card.locator('.gpu-canvas[data-ready=true]');
  const canvas = ready.locator('canvas');
  const hint = ready.locator('.gpu-canvas-hint');
  const bounds = await getStableBox(page, media);
  await page.mouse.move(bounds.x + bounds.width * 0.24, bounds.y + bounds.height / 2);
  await canvas.waitFor();
  // Compare the settled canvas itself, after its initial fade from the poster.
  await canvas.evaluate(async element => {
    await Promise.all(element.parentElement.getAnimations().map(animation => animation.finished));
  });
  assert.ok((await hint.textContent()).length > 0, `${entry}: the chip should name the gesture.`);
  // Move again after startup so the result tests the stretched link's pointer surface.
  await page.mouse.move(bounds.x + bounds.width * 0.21, bounds.y + bounds.height / 2);
  await page.waitForTimeout(250);
  const first = await canvas.screenshot();
  await page.mouse.move(bounds.x + bounds.width * 0.78, bounds.y + bounds.height * 0.32);
  assert.ok(
    await waitForChange(canvas, first),
    `${entry}: moving over the card should change the rendered pixels.`
  );
  await page.mouse.move(0, 0);
  await canvas.waitFor({state: 'detached'});

  await page.keyboard.press('Tab');
  await link.focus();
  await canvas.waitFor();
  await canvas.evaluate(async element => {
    await Promise.all(element.parentElement.getAnimations().map(animation => animation.finished));
  });
  const focused = await canvas.screenshot();
  for (let press = 0; press < 3; press++) await page.keyboard.press('ArrowRight');
  await page.keyboard.press('ArrowUp');
  assert.ok(
    await waitForChange(canvas, focused),
    `${entry}: arrow keys should change the rendered pixels.`
  );
  await page.keyboard.press('Enter');
  await page.waitForURL('**/fn/' + entry);
}

try {
  for (const [index, entry] of ENTRIES.entries()) {
    // The first entry runs in both themes with reduced motion; the rest in dark with motion.
    const variants =
      index === 0
        ? [
            {colorScheme: 'dark', reducedMotion: 'reduce'},
            {colorScheme: 'light', reducedMotion: 'reduce'}
          ]
        : [{colorScheme: 'dark'}];
    for (const variant of variants) {
      const page = await browser.newPage(variant);
      try {
        await verifyLiveCard(page, entry);
        console.log(`ok ${entry} ${variant.colorScheme}${variant.reducedMotion ? ' reduced motion' : ''}`);
      } finally {
        await page.close();
      }
    }
  }

  const unsupported = await browser.newPage();
  await unsupported.addInitScript(() => {
    Object.defineProperty(navigator, 'gpu', {value: undefined});
    window.failedContextAttempts = 0;
    const getContext = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function (type, ...options) {
      if (/^(webgl|experimental-webgl)/.test(type)) {
        window.failedContextAttempts++;
        return null;
      }
      return getContext.call(this, type, ...options);
    };
  });
  const fallback = await openCard(unsupported, ENTRIES[0]);
  await fallback.hover();
  await fallback.locator('.gpu-canvas').waitFor();
  await unsupported.waitForFunction(() => window.failedContextAttempts > 0);
  await fallback.locator('.run-snapshot img:visible').evaluate(image => image.decode());
  assert.equal(await fallback.locator('[data-ready=true]').count(), 0);
  await unsupported.close();

  const touch = await browser.newPage({hasTouch: true, isMobile: true, viewport: {width: 390, height: 844}});
  const touchCard = await openCard(touch, ENTRIES[0]);
  await touchCard.locator('a').tap();
  await touch.waitForURL('**/fn/' + ENTRIES[0]);
  await touch.close();
  console.log('Card poster, hover pixels, keyboard, theme, reduced-motion, fallback and touch checks passed.');
} finally {
  await browser.close();
}
