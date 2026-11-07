// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Live basemap tile occlusion on the globe. Kepler's globe draws CARTO vector tiles (water fill
// and borders) over a depth disk. The script rotates so the tile-covered near hemisphere faces the
// camera, then the antipode, and samples the canvas: pixels inside the globe disc must show tile
// colours (not the bare disk or space colours), and pixels outside the disc must show only
// background colours. The near-side window is centred on the globe, which is the canvas centre (the
// sidebar overlays the full-size canvas rather than shrinking it). Skips when the tile server is
// unreachable.
//
// Usage: node scripts/globe-tiles.mjs [url]
import assert from 'node:assert/strict';
import {chromium} from 'playwright';
import {PNG} from 'pngjs';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';

const url = process.argv.slice(2).find(value => value.startsWith('http')) || 'http://127.0.0.1:5318/?rows=256';
const TILE_PROBE_URL = 'https://tiles-a.basemaps.cartocdn.com/vectortiles/carto.streets/v1/0/0/0.mvt';
const NEAR = {longitude: -30, latitude: 20};
const FAR = {longitude: 150, latitude: -20};

// Bare globe disk (DEFAULT_BASEMAP_COLOR.backgroundFillColor) and the space behind it. Tile water fill
// is [17, 35, 48] and tile lines [40, 63, 93], both further than the tolerance from these.
const BACKGROUND_COLORS = [[9, 16, 29], [4, 9, 17]];
const BACKGROUND_TOLERANCE = 30;
const isBackground = (red, green, blue) =>
  BACKGROUND_COLORS.some(
    ([r, g, b]) => Math.abs(red - r) + Math.abs(green - g) + Math.abs(blue - b) < BACKGROUND_TOLERANCE
  );

async function isTileServerReachable() {
  try {
    const response = await fetch(TILE_PROBE_URL, {signal: AbortSignal.timeout(8000)});
    return response.ok;
  } catch {
    return false;
  }
}

if (!(await isTileServerReachable())) {
  console.log(JSON.stringify({passed: true, skipped: `tile server unreachable: ${TILE_PROBE_URL}`}));
  process.exit(0);
}

// The starfield behind the globe has grey pixels of every brightness, so leakage outside the disc is
// judged by the tile water fill colour only.
const WATER_FILL = [17, 35, 48];
const isWaterFill = (red, green, blue) =>
  Math.abs(red - WATER_FILL[0]) + Math.abs(green - WATER_FILL[1]) + Math.abs(blue - WATER_FILL[2]) < 12;

/** Counts tile pixels (non-background, or water fill only with `waterOnly`) in a screenshot rectangle. */
function countTilePixels(image, region, waterOnly = false) {
  let tilePixels = 0;
  let total = 0;
  for (let vertical = region.top; vertical < region.bottom; vertical++) {
    for (let horizontal = region.left; horizontal < region.right; horizontal++) {
      const offset = (vertical * image.width + horizontal) * 4;
      total++;
      const [red, green, blue] = [image.data[offset], image.data[offset + 1], image.data[offset + 2]];
      if (waterOnly ? isWaterFill(red, green, blue) : !isBackground(red, green, blue)) tilePixels++;
    }
  }
  return {tilePixels, total};
}

const browser = await chromium.launch(getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'}));
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
await pinViteModules(page);
const errors = [];
page.on('pageerror', error => !/send was called before connect/.test(error.message) && errors.push(error.message));

async function sampleGlobe(config, center, name) {
  config.config.mapState = {...config.config.mapState, ...center, zoom: 1.2, pitch: 0, bearing: 0,
    mapViewMode: 'MODE_GLOBE', globe: {...(config.config.mapState.globe ?? {}), enabled: true}};
  await page.evaluate(value => window.__keplerComputeTest.applyNativeConfig(value), config);
  await page.evaluate(() => new Promise(resolve => setTimeout(resolve, 4000)));
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  return PNG.sync.read(await page.screenshot({path: `/tmp/luma-kepler-globe-tiles-${name}.png`}));
}

let result;
try {
  await page.goto(url, {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => window.__keplerComputeTest?.getState().selectedCount === 256, undefined, {timeout: 60000});
  const config = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  // Hide data layers so only the basemap and depth disk are on the canvas.
  for (const layer of config.config.visState.layers) layer.config.isVisible = false;

  const nearImage = await sampleGlobe(config, NEAR, 'near');
  // The globe is centred in the full canvas; the sidebar floats over it.
  const centreX = Math.round(nearImage.width / 2);
  const centreY = Math.round(nearImage.height / 2);
  const half = 60;
  const near = countTilePixels(nearImage, {left: centreX - half, right: centreX + half, top: centreY - half, bottom: centreY + half});
  assert(near.tilePixels / near.total > 0.5, `near side must show tiles (${near.tilePixels}/${near.total})`);

  // Facing the antipode, the near-side tiles are now on the far side: the globe centre shows the
  // far hemisphere, and the strip just outside the limb must be background. No tile colour may
  // leak through anywhere that the disk should have hidden: sample the corners (outside the disc).
  const farImage = await sampleGlobe(config, FAR, 'far');
  const corner = 40;
  const corners = [
    {left: 340 + 10, right: 340 + 10 + corner, top: 70, bottom: 70 + corner},
    {left: farImage.width - 10 - corner, right: farImage.width - 10, top: 70, bottom: 70 + corner},
    {left: 340 + 10, right: 340 + 10 + corner, top: farImage.height - 50 - corner, bottom: farImage.height - 50},
    {left: farImage.width - 10 - corner, right: farImage.width - 10, top: farImage.height - 50 - corner, bottom: farImage.height - 50}
  ];
  for (const [index, region] of corners.entries()) {
    const leaked = countTilePixels(farImage, region, true);
    assert.equal(leaked.tilePixels, 0, `corner ${index} outside the globe must be background (${leaked.tilePixels} tile pixels)`);
  }
  // The far hemisphere is also tiled, so it must show tiles from the front only: compare against
  // the near-facing image, whose centre content must differ (different hemisphere, not mirrored through).
  const faceCentre = countTilePixels(farImage, {left: centreX - half, right: centreX + half, top: centreY - half, bottom: centreY + half});
  assert(faceCentre.tilePixels / faceCentre.total > 0.5, `rotated side must show its own tiles (${faceCentre.tilePixels}/${faceCentre.total})`);
  // Compare the whole disc interior (radius 177 at zoom 1.2): the two window centres are both ocean,
  // so land and borders away from the centre are what differ.
  const comparisonHalf = 150;
  let differing = 0;
  for (let vertical = centreY - comparisonHalf; vertical < centreY + comparisonHalf; vertical += 4) {
    for (let horizontal = centreX - comparisonHalf; horizontal < centreX + comparisonHalf; horizontal += 4) {
      const offset = (vertical * nearImage.width + horizontal) * 4;
      if (Math.abs(nearImage.data[offset] - farImage.data[offset]) + Math.abs(nearImage.data[offset + 1] - farImage.data[offset + 1]) > 10) differing++;
    }
  }
  assert(differing > 50, `rotating must change the visible hemisphere (${differing} sampled pixels differ)`);
  assert.deepEqual(errors, []);
  result = {passed: true, near, faceCentre, differing};
} catch (error) {
  await page.screenshot({path: '/tmp/luma-kepler-globe-tiles-failure.png'}).catch(() => {});
  result = {passed: false, message: error.message, errors};
} finally {
  await browser.close();
}
console.log(JSON.stringify(result, null, 1));
if (!result.passed) process.exit(1);
