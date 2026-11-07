// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Default single-Deck two-viewport Dual and Swipe maps, driven through Kepler's split-map control.
import assert from 'node:assert/strict';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';

const baseUrl = process.env.KEPLER_COMPUTE_URL ?? 'http://127.0.0.1:5318/';
import {pinViteModules} from './pin-vite-modules.mjs';
const browser = await chromium.launch(
  getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'})
);
const context = await browser.newContext({viewport: {width: 1440, height: 1000}, hasTouch: true});
const page = await context.newPage();
await pinViteModules(page);
const failures = [];
page.on('pageerror', error => failures.push(error.message));
page.on('console', message => {
  if (message.type() === 'error' && /WGSL|validation error|device.*lost/i.test(message.text()))
    failures.push(message.text());
});

/** Visible Deck canvases; basemap canvases and the empty second map's hidden canvas do not count. */
async function canvasCount() {
  return page.evaluate(
    () =>
      [...document.querySelectorAll('canvas')].filter(
        canvas =>
          !canvas.classList.contains('maplibregl-canvas') &&
          getComputedStyle(canvas).visibility !== 'hidden'
      ).length
  );
}
/** Open Kepler's split-map toolbar and choose Single, Dual or Swipe. */
async function chooseSplitMode(title) {
  await page.locator('.map-control-button.split-map').click();
  await page.locator('.toolbar-item', {hasText: new RegExp(`^${title}$`)}).click();
}
async function getMapState() {
  return page.evaluate(() => {
    const {mapState} = window.__keplerComputeTest.saveNativeConfig().config;
    return mapState;
  });
}
async function waitForMode(mode) {
  await page.waitForFunction(
    expected =>
      window.__keplerComputeTest.saveNativeConfig().config.mapState.mapSplitMode === expected,
    mode
  );
  await page.waitForTimeout(2000);
}
async function setLayerMaps(left, right) {
  const {splitMaps} = await page.evaluate(
    () => window.__keplerComputeTest.saveNativeConfig().config.visState
  );
  const has = index => Boolean(splitMaps[index]?.layers?.['resident-points']);
  if (has(0) !== left)
    await page.evaluate(() => window.__keplerComputeTest.toggleLayerForMap(0, 'resident-points'));
  if (has(1) !== right)
    await page.evaluate(() => window.__keplerComputeTest.toggleLayerForMap(1, 'resident-points'));
  await page.waitForTimeout(1500);
}
/** Read pixels from a fresh screenshot: [r, g, b] per requested point. */
async function readPixels(points) {
  const base64 = (await page.screenshot()).toString('base64');
  return page.evaluate(
    async ({base64, points}) => {
      const image = await createImageBitmap(await (await fetch(`data:image/png;base64,${base64}`)).blob());
      const canvas = new OffscreenCanvas(image.width, image.height);
      const context = canvas.getContext('2d');
      context.drawImage(image, 0, 0);
      const scale = image.width / window.innerWidth;
      return points.map(([x, y]) => [...context.getImageData(Math.round(x * scale), Math.round(y * scale), 1, 1).data].slice(0, 3));
    },
    {base64, points}
  );
}
const isBasemap = ([r, g, b]) => Math.abs(r - 16) < 8 && Math.abs(g - 29) < 8 && Math.abs(b - 43) < 8;
const isPoint = ([r, g, b]) => b > 150 && r < 140 && b - r > 80;
const describe = pixel => `rgb(${pixel.join(',')})`;
async function expectPixels(label, checks) {
  const pixels = await readPixels(checks.map(check => check.at));
  checks.forEach((check, index) =>
    assert.ok(check.test(pixels[index]), `${label}: ${check.name} at ${check.at} was ${describe(pixels[index])}`)
  );
}

try {
  // `?views=1` forces single view: split stays rejected exactly as before.
  await page.goto(`${baseUrl}?views=1`);
  await page.locator('canvas').first().waitFor();
  await chooseSplitMode('Dual');
  await page.getByText(/supports one map view/i).first().waitFor({timeout: 10_000});

  // Default: Dual opens two viewports in one Deck on one device (no additional visible Deck canvas).
  await page.goto(baseUrl);
  await page.locator('.maplibregl-canvas').waitFor();
  await page.waitForTimeout(2000);
  const canvasesBefore = await canvasCount();
  await chooseSplitMode('Dual');
  await waitForMode('DUAL_MAP');
  assert.equal(await canvasCount(), canvasesBefore, 'split must not create a second visible Deck canvas');
  assert.equal(await page.getByText(/supports one map view/i).count(), 0, 'no single-view rejection under the flag');

  // The right half keeps its own basemap (the right map is empty by native default) ...
  assert.equal(await page.locator('.maplibregl-map').count(), 2, 'each half mounts its own basemap');
  await expectPixels('dual left-only layer', [
    {name: 'right basemap', at: [1300, 100], test: isBasemap},
    {name: 'right basemap lower', at: [1000, 900], test: isBasemap},
    {name: 'right half has no points', at: [1080, 400], test: isBasemap},
    {name: 'left points', at: [450, 400], test: isPoint}
  ]);
  // ... and draws layers from the shared buffers once the layer is enabled there too.
  await setLayerMaps(true, true);
  await expectPixels('dual both layers', [
    {name: 'right basemap', at: [1300, 100], test: isBasemap},
    {name: 'right points', at: [1000, 400], test: isPoint},
    {name: 'left points', at: [450, 400], test: isPoint}
  ]);
  await setLayerMaps(false, true);
  await expectPixels('dual right-only layer', [
    {name: 'left basemap without points', at: [450, 400], test: isBasemap},
    {name: 'right points', at: [1000, 400], test: isPoint}
  ]);
  await setLayerMaps(true, false);

  // Swipe: both maps are full size; the divider chooses which map each side shows.
  await chooseSplitMode('Swipe');
  await waitForMode('SWIPE_COMPARE');
  assert.equal(await page.getByText(/supports one map view/i).count(), 0);
  assert.equal(await canvasCount(), canvasesBefore, 'swipe must not create a second Deck canvas');
  const swipeState = await getMapState();
  assert.equal(swipeState.swipeComparePercentage, 50);
  // Layer on the left map only: points left of the 720px divider, basemap right of it.
  await expectPixels('swipe left layer', [
    {name: 'left side shows points', at: [600, 400], test: isPoint},
    {name: 'right side clipped to basemap', at: [800, 400], test: isBasemap},
    {name: 'right basemap', at: [1300, 100], test: isBasemap},
    {name: 'left basemap', at: [400, 100], test: isBasemap}
  ]);
  // Layer on the right map only; dragging the divider left uncovers more of it.
  await setLayerMaps(false, true);
  await expectPixels('swipe right layer', [
    {name: 'left side clipped', at: [600, 400], test: isBasemap},
    {name: 'right side shows points', at: [800, 400], test: isPoint}
  ]);
  await page.mouse.move(720, 500);
  await page.mouse.down();
  await page.mouse.move(600, 500, {steps: 4});
  await page.mouse.move(500, 500, {steps: 4});
  await page.waitForTimeout(800);
  // While dragging, the Deck views follow before the store commits.
  await expectPixels('swipe dragging', [
    {name: 'points uncovered left of the old divider', at: [600, 400], test: isPoint},
    {name: 'basemap left of the divider', at: [420, 400], test: isBasemap}
  ]);
  await page.mouse.up();
  await page.waitForFunction(() => {
    const percentage = window.__keplerComputeTest.saveNativeConfig().config.mapState.swipeComparePercentage;
    return percentage > 30 && percentage < 40;
  });
  await page.waitForTimeout(800);
  await expectPixels('swipe released', [
    {name: 'points right of the divider', at: [600, 400], test: isPoint},
    {name: 'basemap left of the divider', at: [420, 400], test: isBasemap}
  ]);
  assert.equal(await canvasCount(), canvasesBefore);

  // Back to Dual keeps the same Deck, then closing view 1 keeps the shared Deck, device and runtime.
  await chooseSplitMode('Dual');
  await waitForMode('DUAL_MAP');
  assert.equal(await canvasCount(), canvasesBefore);
  assert.equal(await page.evaluate(() => window.__keplerComputeTest.getState().ready), true);
  await chooseSplitMode('Single');
  await waitForMode('SINGLE_MAP');
  assert.equal(await canvasCount(), canvasesBefore);
  assert.equal(await page.locator('.map-control-button.close-map').count(), 0);
  assert.equal(await page.evaluate(() => window.__keplerComputeTest.getState().ready), true);
  await setLayerMaps(true, true).catch(() => {});
  // Closing from Swipe also returns to one live view.
  await chooseSplitMode('Swipe');
  await waitForMode('SWIPE_COMPARE');
  await chooseSplitMode('Single');
  await waitForMode('SINGLE_MAP');
  assert.equal(await canvasCount(), canvasesBefore);
  assert.equal(await page.evaluate(() => window.__keplerComputeTest.getState().ready), true);

  // ---- Per-view picking and hover ----
  async function findRow(viewId, [minX, maxX]) {
    return page.evaluate(
      ({viewId, minX, maxX}) => {
        for (let index = 0; index < 20000; index += 7) {
          const position = window.__keplerComputeTest.projectSourceRowInView(viewId, index);
          if (position && position[0] > minX && position[0] < maxX && position[1] > 300 && position[1] < 700)
            return {index, position};
        }
        return null;
      },
      {viewId, minX, maxX}
    );
  }
  async function hoverAt(position) {
    await page.evaluate(() => window.__keplerComputeTest.clearHoverInfo());
    await page.waitForFunction(() => !window.__keplerComputeTest.getHoverInfo()?.layerId);
    await page.mouse.move(position[0] - 3, position[1] - 3);
    await page.mouse.move(position[0], position[1], {steps: 3});
    await page.waitForFunction(
      () => window.__keplerComputeTest.getHoverInfo()?.layerId,
      undefined,
      {timeout: 8000}
    );
    return page.evaluate(() => window.__keplerComputeTest.getHoverInfo());
  }
  async function clearHover() {
    await page.mouse.move(1000, 950);
    await page.mouse.move(1010, 960);
    await page.waitForTimeout(800);
  }
  await chooseSplitMode('Dual');
  await waitForMode('DUAL_MAP');
  await setLayerMaps(true, true);
  const dualLeft = await findRow('view-0', [420, 600]);
  const dualRight = await findRow('view-1', [1000, 1200]);
  assert.ok(dualLeft && dualRight, 'dual rows are visible in each half');
  assert.equal((await hoverAt(dualLeft.position)).mapIndex, 0, 'dual left hover routes to map 0');
  await clearHover();
  assert.equal((await hoverAt(dualRight.position)).mapIndex, 1, 'dual right hover routes to map 1');
  await clearHover();

  // ---- View 1 pan and zoom (synchronized, then independent) ----
  const getViewports = () =>
    page.evaluate(() => {
      const {mapState} = window.__keplerComputeTest.saveNativeConfig().config;
      return {mapState, viewports: mapState.splitMapViewports};
    });
  const dragView = async (from, to) => {
    await page.mouse.move(...from);
    await page.mouse.down();
    await page.mouse.move(to[0], to[1], {steps: 6});
    await page.mouse.up();
    await page.waitForTimeout(800);
  };
  const synced = await getViewports();
  await dragView([1000, 600], [900, 600]);
  const afterSyncedDrag = await getViewports();
  assert.notEqual(afterSyncedDrag.mapState.longitude, synced.mapState.longitude, 'view 1 drag pans the synchronized map');
  await page.evaluate(() => window.__keplerComputeTest.toggleSplitMapViewport({isViewportSynced: false}));
  await page.waitForTimeout(1500);
  const unsynced = await getViewports();
  assert.equal(unsynced.mapState.isViewportSynced, false);
  assert.equal(unsynced.viewports.length, 2);
  await dragView([1000, 600], [940, 600]);
  const afterRightDrag = await getViewports();
  assert.equal(afterRightDrag.viewports[0].longitude, unsynced.viewports[0].longitude, 'view 0 unchanged by view 1 drag');
  assert.notEqual(afterRightDrag.viewports[1].longitude, unsynced.viewports[1].longitude, 'view 1 pans independently');
  await page.mouse.move(1000, 600);
  await page.mouse.wheel(0, -400);
  await page.waitForTimeout(1200);
  const afterZoom = await getViewports();
  assert.ok(afterZoom.viewports[1].zoom > afterRightDrag.viewports[1].zoom, 'view 1 wheel zoom');
  assert.equal(afterZoom.viewports[0].zoom, afterRightDrag.viewports[0].zoom, 'view 0 zoom unchanged');
  // The shared Deck draws each half from its own view state.
  const spans = await page.evaluate(() => {
    const api = window.__keplerComputeTest;
    const span = viewId => {
      const first = api.projectSourceRowInView(viewId, 3);
      const second = api.projectSourceRowInView(viewId, 4000);
      return Math.hypot(first[0] - second[0], first[1] - second[1]);
    };
    return [span('view-0'), span('view-1')];
  });
  assert.ok(spans[1] > spans[0] * 1.3, `view 1 is zoomed independently of view 0: ${spans}`);
  // Independent hover still routes to the right map.
  const unsyncedRight = await findRow('view-1', [900, 1300]);
  assert.ok(unsyncedRight, 'a row is visible in the independent right view');
  assert.equal((await hoverAt(unsyncedRight.position)).mapIndex, 1);
  await clearHover();

  // ---- Swipe: hover on each side, independent view-1 pan, saved-config round trip ----
  await chooseSplitMode('Swipe');
  await waitForMode('SWIPE_COMPARE');
  await page.evaluate(() => window.__keplerComputeTest.toggleSplitMapViewport({isViewportSynced: true}));
  await page.waitForTimeout(1500);
  await setLayerMaps(true, true);
  // The divider keeps its earlier position; put it back at 50%.
  const dividerX = await page.evaluate(
    () => (window.__keplerComputeTest.saveNativeConfig().config.mapState.swipeComparePercentage * 1440) / 100
  );
  if (Math.abs(dividerX - 720) > 2) {
    await page.mouse.move(dividerX, 500);
    await page.mouse.down();
    await page.mouse.move(720, 500, {steps: 6});
    await page.mouse.up();
    await page.waitForTimeout(1200);
  }
  const swipeLeft = await findRow('view-0', [540, 700]);
  const swipeRight = await findRow('view-1', [760, 880]);
  assert.ok(swipeLeft && swipeRight, 'swipe rows are visible on each side');
  const swipeLeftInfo = await hoverAt(swipeLeft.position);
  assert.equal(swipeLeftInfo.mapIndex, 0, `swipe left hover routes to map 0 ${JSON.stringify([swipeLeftInfo, swipeLeft])}`);
  await clearHover();
  assert.equal((await hoverAt(swipeRight.position)).mapIndex, 1, 'swipe right hover routes to map 1');
  await clearHover();
  const swipeBefore = await getViewports();
  await dragView([800, 600], [750, 600]);
  assert.notEqual((await getViewports()).mapState.longitude, swipeBefore.mapState.longitude, 'swipe right side pans');
  // Divider to 35%, then round-trip the native configuration through Single.
  await page.mouse.move(720, 500);
  await page.mouse.down();
  await page.mouse.move(504, 500, {steps: 6});
  await page.mouse.up();
  await page.waitForFunction(() => {
    const percentage = window.__keplerComputeTest.saveNativeConfig().config.mapState.swipeComparePercentage;
    return percentage > 30 && percentage < 40;
  });
  const savedConfig = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  await chooseSplitMode('Single');
  await waitForMode('SINGLE_MAP');
  await page.evaluate(config => window.__keplerComputeTest.applyNativeConfig(config), savedConfig);
  await page.waitForFunction(
    () => window.__keplerComputeTest.saveNativeConfig().config.mapState.mapSplitMode === 'SWIPE_COMPARE'
  );
  await page.waitForTimeout(1500);
  const restored = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  assert.equal(restored.config.mapState.swipeComparePercentage, savedConfig.config.mapState.swipeComparePercentage);
  assert.deepEqual(restored.config.visState.splitMaps, savedConfig.config.visState.splitMaps);
  assert.equal(await page.evaluate(() => window.__keplerComputeTest.getState().ready), true);
  const restoredDivider = await page.evaluate(() => {
    const element = document.querySelector('.layout-splitter .splitter');
    return element ? element.getBoundingClientRect().x : null;
  });
  assert.ok(restoredDivider > 480 && restoredDivider < 530, `restored divider position ${restoredDivider}`);

  // ---- Closing map 0 keeps the shared Deck and device; map 1 becomes the only view ----
  await chooseSplitMode('Dual');
  await waitForMode('DUAL_MAP');
  await setLayerMaps(false, true);
  await page.evaluate(() => {
    document.querySelectorAll('canvas').forEach(canvas => {
      if (!canvas.classList.contains('maplibregl-canvas') && getComputedStyle(canvas).visibility !== 'hidden')
        canvas.dataset.lumaSurvivor = 'true';
    });
  });
  await page.evaluate(() => window.__keplerComputeTest.toggleSplitMap(0));
  await waitForMode('SINGLE_MAP');
  assert.equal(await page.locator('canvas[data-luma-survivor]').count(), 1, 'the same Deck canvas survives closing map 0');
  assert.equal(await page.evaluate(() => window.__keplerComputeTest.getState().ready), true);
  assert.equal(await canvasCount(), canvasesBefore);
  const survivingRow = await page.evaluate(() => {
    for (let index = 0; index < 20000; index += 11) {
      const position = window.__keplerComputeTest.projectSourceRow(index);
      if (position && position[0] > 450 && position[0] < 1300 && position[1] > 150 && position[1] < 800)
        return position;
    }
    return null;
  });
  assert.ok(survivingRow, 'surviving single map has resident rows on screen');
  await expectPixels('after closing map 0', [
    {name: 'surviving map draws the layer', at: survivingRow, test: isPoint}
  ]);

  // ---- Edge behaviors: wheel pivot, 3D, globe and touch in both split modes ----
  const unprojectIn = (viewId, x, y) =>
    page.evaluate(([id, a, b]) => window.__keplerComputeTest.unprojectPositionInView(id, a, b), [viewId, x, y]);
  const projectIn = (viewId, longitude, latitude) =>
    page.evaluate(([id, a, b]) => window.__keplerComputeTest.projectPositionInView(id, a, b), [viewId, longitude, latitude]);
  const getZoom = () => page.evaluate(() => window.__keplerComputeTest.saveNativeConfig().config.mapState.zoom);
  /** The geography under the pointer must stay under the pointer, exactly like a native map. */
  async function measureWheelPivot(viewId, x, y) {
    await page.mouse.move(x - 5, y - 5);
    await page.mouse.move(x, y);
    await page.waitForTimeout(300);
    const anchor = await unprojectIn(viewId, x, y);
    const zoomBefore = await getZoom();
    await page.mouse.wheel(0, -300);
    await page.waitForFunction(before => window.__keplerComputeTest.saveNativeConfig().config.mapState.zoom > before + 0.3, zoomBefore, {timeout: 8000});
    await page.waitForTimeout(1200);
    const [px, py] = await projectIn(viewId, anchor[0], anchor[1]);
    return [px - x, py - y];
  }
  async function expectWheelPivot(label, viewId, x, y) {
    const drift = await measureWheelPivot(viewId, x, y);
    assert.ok(Math.hypot(...drift) < 0.5, `${label}: wheel zoom pivot drifted to ${drift}`);
  }
  async function enterSplit(title) {
    await chooseSplitMode(title);
    await waitForMode(title === 'Swipe' ? 'SWIPE_COMPARE' : 'DUAL_MAP');
    await setLayerMaps(true, true);
    const percentage = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig().config.mapState.swipeComparePercentage);
    if (title === 'Swipe' && Math.abs(percentage - 50) > 1) {
      await page.mouse.move((percentage * 1440) / 100, 500);
      await page.mouse.down();
      await page.mouse.move(720, 500, {steps: 6});
      await page.mouse.up();
      await page.waitForTimeout(1000);
    }
  }
  async function leaveSplit() {
    await chooseSplitMode('Single');
    await waitForMode('SINGLE_MAP');
  }
  const failuresBefore = failures.length;
  await page.goto(baseUrl);
  await page.locator('.maplibregl-canvas').waitFor();
  await page.waitForTimeout(2000);
  const swipePoints = [['swipe view 0', 'view-0', 500, 700], ['swipe view 1', 'view-1', 1000, 300], ['swipe view 1 near divider', 'view-1', 780, 800]];
  const dualPoints = [['dual view 0', 'view-0', 450, 300], ['dual view 1', 'view-1', 1100, 700]];
  for (const [title, points] of [['Swipe', swipePoints], ['Dual', dualPoints]]) {
    await enterSplit(title);
    for (const [label, viewId, x, y] of points) await expectWheelPivot(label, viewId, x, y);
    await leaveSplit();
  }

  // 3D: the pitched, rotated map slices exactly like the full-size map in Swipe and keeps pivoting in Dual.
  await page.evaluate(() => window.__keplerComputeTest.setMapViewport({pitch: 55, bearing: 25, zoom: 11}));
  await page.waitForTimeout(1500);
  await enterSplit('Swipe');
  const pitchedSwipe = await page.evaluate(() => [0, 1].map(index => window.__keplerComputeTest.projectSourceRowInView(`view-${index}`, 3)));
  assert.ok(Math.hypot(pitchedSwipe[0][0] - pitchedSwipe[1][0], pitchedSwipe[0][1] - pitchedSwipe[1][1]) < 0.01, `pitched swipe views disagree: ${pitchedSwipe}`);
  const pitchedRow = await findRow('view-1', [760, 1300]).then(row => row ?? findRow('view-1', [740, 1400]));
  assert.ok(pitchedRow, 'a pitched row is visible right of the divider');
  await expectPixels('pitched swipe', [{name: 'pitched points right of the divider', at: pitchedRow.position, test: isPoint}]);
  assert.equal((await hoverAt(pitchedRow.position)).mapIndex, 1, 'pitched swipe hover routes to map 1');
  await clearHover();
  await expectWheelPivot('pitched swipe view 1', 'view-1', 1000, 350);
  await leaveSplit();
  await enterSplit('Dual');
  const pitchedDualRow = await findRow('view-1', [900, 1300]);
  assert.ok(pitchedDualRow, 'a pitched row is visible in the Dual right view');
  await expectPixels('pitched dual', [{name: 'pitched points in view 1', at: pitchedDualRow.position, test: isPoint}]);
  assert.equal((await hoverAt(pitchedDualRow.position)).mapIndex, 1, 'pitched dual hover routes to map 1');
  await clearHover();
  await expectWheelPivot('pitched dual view 1', 'view-1', 1100, 350);
  await leaveSplit();

  // Globe: Swipe shows one seamless globe; Dual shows one globe per view; no GPU validation errors.
  await page.goto(baseUrl);
  await page.locator('.maplibregl-canvas').waitFor();
  await page.waitForTimeout(2000);
  const globeConfig = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  globeConfig.config.mapState = {
    ...globeConfig.config.mapState,
    mapViewMode: 'MODE_GLOBE',
    zoom: 1.6,
    longitude: -74,
    latitude: 30,
    pitch: 0,
    bearing: 0,
    globe: {...(globeConfig.config.mapState.globe ?? {}), enabled: true}
  };
  await page.evaluate(config => window.__keplerComputeTest.applyNativeConfig(config), globeConfig);
  await page.waitForTimeout(2500);
  const globeNear = [-74, 40.75];
  const nearGlobePoint = view => projectIn(view, ...globeNear);
  const isGlobePoint = ([r, g, b]) => b > 150 && g > 120 && r < 160;
  // Globe wheel zoom pivots exactly at the cursor (ExactGlobeController), the same bound as planar views.
  const GLOBE_PIVOT_PIXELS = 0.5;
  // Deck's globe controller clamps the zoom to minZoom (2 plus the latitude offset) on the first wheel
  // event, which would move the geography by itself, so pivots are measured from inside that range.
  const setGlobeView = async zoom => {
    await page.evaluate(zoom => window.__keplerComputeTest.setMapViewport({zoom, longitude: -74, latitude: 30, bearing: 0, pitch: 0}), zoom);
    await page.waitForTimeout(1500);
  };
  await enterSplit('Swipe');
  const swipeGlobe = [await nearGlobePoint('view-0'), await nearGlobePoint('view-1')];
  assert.ok(Math.hypot(swipeGlobe[0][0] - swipeGlobe[1][0], swipeGlobe[0][1] - swipeGlobe[1][1]) < 0.5, `swipe globes disagree: ${swipeGlobe}`);
  assert.ok(await page.evaluate(() => window.__keplerComputeTest.getState().ready));
  await setGlobeView(3.2);
  for (const [viewId, x] of [['view-0', 640], ['view-1', 800]]) {
    const drift = await measureWheelPivot(viewId, x, 430);
    assert.ok(Math.hypot(...drift) < GLOBE_PIVOT_PIXELS, `swipe globe ${viewId} wheel pivot drifted ${drift}`);
  }
  await leaveSplit();
  await setGlobeView(1.6);
  await enterSplit('Dual');
  const dualGlobe = [await nearGlobePoint('view-0'), await nearGlobePoint('view-1')];
  assert.ok(dualGlobe.every(position => position && Number.isFinite(position[0])), `dual globe projections ${dualGlobe}`);
  await expectPixels('dual globe', [
    {name: 'view 0 globe point', at: [dualGlobe[0][0], dualGlobe[0][1]], test: isGlobePoint},
    {name: 'view 1 globe point', at: [dualGlobe[1][0], dualGlobe[1][1]], test: isGlobePoint}
  ]);
  await setGlobeView(3.2);
  for (const [viewId, x] of [['view-0', 560], ['view-1', 1100]]) {
    const drift = await measureWheelPivot(viewId, x, 430);
    assert.ok(Math.hypot(...drift) < GLOBE_PIVOT_PIXELS, `dual globe ${viewId} wheel pivot drifted ${drift}`);
  }
  await leaveSplit();
  assert.deepEqual(failures.slice(failuresBefore), [], 'split 3D and globe raise no GPU errors');

  // Touch: one-finger pan and two-finger pinch on view 1 match the single-view behavior.
  await page.goto(baseUrl);
  await page.locator('.maplibregl-canvas').waitFor();
  await page.waitForTimeout(2000);
  const devtools = await context.newCDPSession(page);
  const sendTouch = (type, points) =>
    devtools.send('Input.dispatchTouchEvent', {type, touchPoints: points.map(([x, y], id) => ({x, y, id}))});
  const readViewState = () => page.evaluate(() => window.__keplerComputeTest.saveNativeConfig().config.mapState);
  /** Pixels the geography under the finger moved, which is independent of the current zoom. */
  async function touchPan(viewId, x, y) {
    const anchor = await unprojectIn(viewId, x, y);
    await sendTouch('touchStart', [[x, y]]);
    for (let step = 1; step <= 8; step++) await sendTouch('touchMove', [[x - step * 10, y]]);
    await sendTouch('touchEnd', []);
    await page.waitForTimeout(1200);
    const [projectedX] = await projectIn(viewId, anchor[0], anchor[1]);
    return x - projectedX;
  }
  async function touchPinch(viewId, x, y) {
    const zoomBefore = (await readViewState()).zoom;
    const anchor = await unprojectIn(viewId, x, y);
    await sendTouch('touchStart', [[x - 40, y], [x + 40, y]]);
    for (let step = 1; step <= 10; step++)
      await sendTouch('touchMove', [[x - 40 - step * 8, y], [x + 40 + step * 8, y]]);
    await sendTouch('touchEnd', []);
    await page.waitForTimeout(1500);
    const [projectedX, projectedY] = await projectIn(viewId, anchor[0], anchor[1]);
    return {zoomDelta: (await readViewState()).zoom - zoomBefore, drift: [projectedX - x, projectedY - y]};
  }
  const singlePan = await touchPan(null, 1000, 500);
  const singlePinch = await touchPinch(null, 1000, 500);
  assert.ok(singlePan > 20 && singlePinch.zoomDelta > 0.5, `single-view touch baseline ${singlePan} ${singlePinch}`);
  await page.goto(baseUrl);
  await page.locator('.maplibregl-canvas').waitFor();
  await page.waitForTimeout(2000);
  for (const title of ['Swipe', 'Dual']) {
    await enterSplit(title);
    const panDelta = await touchPan('view-1', 1000, 500);
    // Touch recognition thresholds vary by a few pixels between gestures.
    assert.ok(panDelta > 20 && Math.abs(panDelta - singlePan) < 25, `${title} view 1 touch pan ${panDelta} matches single ${singlePan}`);
    const pinch = await touchPinch('view-1', 1000, 500);
    assert.ok(pinch.zoomDelta > 1 && Math.abs(pinch.zoomDelta - singlePinch.zoomDelta) < 0.25, `${title} view 1 pinch zoom ${pinch.zoomDelta} matches single ${singlePinch.zoomDelta}`);
    // Native pinch has its own recognition offset; the split view must reproduce it exactly.
    assert.ok(Math.hypot(pinch.drift[0] - singlePinch.drift[0], pinch.drift[1] - singlePinch.drift[1]) < 4, `${title} view 1 pinch pivot ${pinch.drift} matches single ${singlePinch.drift}`);
    await leaveSplit();
  }

  // Editor: polygon draw and select operate in whichever view receives the pointer.
  const readEditor = () => page.evaluate(() => window.__keplerComputeTest.nativeEditor());
  async function chooseEditorTool(name) {
    const tool = page.getByText(name, {exact: true}).first();
    if (!(await tool.isVisible().catch(() => false))) await page.locator('.map-control-button.map-draw').first().click();
    await tool.click();
  }
  async function drawPolygonIn(label, viewId, [x0, y0, x1, y1]) {
    await chooseEditorTool('Polygon');
    const previousCount = (await readEditor()).features.length;
    for (const [x, y] of [[x0, y0], [x1, y0], [x1, y1]]) {
      await page.mouse.move(x, y);
      await page.waitForTimeout(700);
      await page.mouse.click(x, y);
      await page.waitForTimeout(900);
    }
    await page.mouse.move(x0, y1);
    await page.waitForTimeout(700);
    await page.mouse.click(x0, y1);
    await page.waitForTimeout(900);
    await page.mouse.dblclick(x0, y1);
    await page.waitForFunction(count => window.__keplerComputeTest.nativeEditor().features.length === count + 1, previousCount, {timeout: 10000});
    const feature = (await readEditor()).features.at(-1);
    const ring = feature.geometry.coordinates[0];
    const expected = await Promise.all([[x0, y0], [x1, y0], [x1, y1]].map(([x, y]) => unprojectIn(viewId, x, y)));
    // The native editor may reverse the ring winding, so match vertices as a set.
    expected.forEach((position, index) => {
      assert.ok(
        ring.some(vertex => Math.abs(vertex[0] - position[0]) < 1e-6 && Math.abs(vertex[1] - position[1]) < 1e-6),
        `${label}: click ${index} was not drawn at ${viewId} ${position}; ring ${JSON.stringify(ring)}`
      );
    });
    return feature;
  }
  const regions = {
    Swipe: {'view-0': [350, 790, 650, 900], 'view-1': [950, 790, 1150, 900]},
    Dual: {'view-0': [350, 790, 650, 900], 'view-1': [950, 790, 1150, 900]}
  };
  for (const title of ['Swipe', 'Dual']) {
    await page.goto(baseUrl);
    await page.locator('.maplibregl-canvas').waitFor();
    await page.waitForTimeout(2000);
    await enterSplit(title);
    for (const viewId of ['view-1', 'view-0']) {
      const feature = await drawPolygonIn(`${title} ${viewId} draw`, viewId, regions[title][viewId]);
      // A drawn feature becomes the selected feature of the one native editor.
      assert.equal((await readEditor()).selectedFeature?.id, feature.id, `${title} ${viewId} drawn feature is selected`);
    }
    await page.screenshot({path: `/tmp/luma-kepler-views-editor-${title}.png`});
    await leaveSplit();
  }
  assert.deepEqual(failures, [], failures.join('\n'));
  console.log('views.mjs checks completed');
} finally {
  await browser.close();
}
