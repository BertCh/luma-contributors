// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Touch coverage for the Dual and Swipe maps on an emulated phone: one-finger pan, two-finger
// pinch and rotate, and swipe divider drag by touch. Unlike views.mjs this is not limited to
// headless Chromium.
//
// Environment:
//   KEPLER_COMPUTE_URL      app URL (default http://127.0.0.1:5318/)
//   KEPLER_TOUCH_BROWSER    chromium (default) | chrome | msedge | webkit
//   KEPLER_TOUCH_HEADED     1 to run headed (WebGPU needs a real GPU or a headed window on some hosts)
//   KEPLER_TOUCH_DEVICE     Playwright device descriptor name (default 'Pixel 7'; try 'iPhone 14')
//   KEPLER_TOUCH_EXPECT_ROTATE  0 to skip the bearing assertion. The app enables Deck's native
//                           touchRotate (withTouchRotate in globe-wheel-controller.ts), so it is on by default.
//
// Chromium-family browsers drive real touch input through CDP Input.dispatchTouchEvent. WebKit has
// no CDP, so gestures are dispatched as synthetic Pointer Events, which exercises the divider
// (Pointer Events) but only soft-checks map gestures. Firefox is not supported (no WebGPU).
import assert from 'node:assert/strict';
import {chromium, webkit, devices} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';

const baseUrl = process.env.KEPLER_COMPUTE_URL ?? 'http://127.0.0.1:5318/';
const browserName = process.env.KEPLER_TOUCH_BROWSER ?? 'chromium';
const headed = process.env.KEPLER_TOUCH_HEADED === '1';
const deviceName = process.env.KEPLER_TOUCH_DEVICE ?? 'Pixel 7';
const expectRotate = process.env.KEPLER_TOUCH_EXPECT_ROTATE !== '0';
const useCdp = browserName !== 'webkit';
const device = devices[deviceName];
assert.ok(device, `unknown Playwright device ${deviceName}`);

const browser = useCdp
  ? await chromium.launch(
      getPlaywrightLaunchOptions({
        headless: !headed,
        backend: 'webgpu-core',
        channel: browserName === 'chromium' ? undefined : browserName
      })
    )
  : await webkit.launch({headless: !headed});
const context = await browser.newContext({...device, hasTouch: true, isMobile: true});
const page = await context.newPage();
await pinViteModules(page);
const failures = [];
// The synthetic pointer sequences below use pointer ids the browser never issued, so the divider's
// setPointerCapture rejects them; real pointers are always active and do not hit this.
page.on('pageerror', error => {
  if (!/setPointerCapture.*No active pointer/.test(error.message)) failures.push(error.message);
});
page.on('console', message => {
  if (message.type() === 'error' && /WGSL|validation error|device.*lost/i.test(message.text()))
    failures.push(message.text());
});

const viewport = page.viewportSize();
const devtools = useCdp ? await context.newCDPSession(page) : null;

const readMapState = () =>
  page.evaluate(() => window.__keplerComputeTest.saveNativeConfig().config.mapState);
const unprojectIn = (viewId, x, y) =>
  page.evaluate(([id, a, b]) => window.__keplerComputeTest.unprojectPositionInView(id, a, b), [viewId, x, y]);
const projectIn = (viewId, longitude, latitude) =>
  page.evaluate(([id, a, b]) => window.__keplerComputeTest.projectPositionInView(id, a, b), [viewId, longitude, latitude]);

async function chooseSplitMode(title) {
  await page.locator('.map-control-button.split-map').click();
  await page.locator('.toolbar-item', {hasText: new RegExp(`^${title}$`)}).click();
}
async function waitForMode(mode) {
  await page.waitForFunction(
    expected => window.__keplerComputeTest.saveNativeConfig().config.mapState.mapSplitMode === expected,
    mode
  );
  await page.waitForTimeout(2000);
}
async function enterSplit(title) {
  await chooseSplitMode(title);
  await waitForMode(title === 'Swipe' ? 'SWIPE_COMPARE' : 'DUAL_MAP');
}
async function leaveSplit() {
  await chooseSplitMode('Single');
  await waitForMode('SINGLE_MAP');
}
async function reload() {
  await page.goto(baseUrl);
  await page.locator('.maplibregl-canvas').waitFor();
  // On a phone viewport Kepler's side panel covers the whole map, so gestures would land on it.
  const closeButton = page.locator('.side-bar__close');
  if (await closeButton.isVisible()) await closeButton.click();
  await page.waitForTimeout(2000);
}

/** Touch points are [[x, y], ...] in CSS pixels; each finger keeps its index as its id. */
async function sendTouch(type, points) {
  await devtools.send('Input.dispatchTouchEvent', {
    type,
    touchPoints: points.map(([x, y], id) => ({x, y, id}))
  });
}

/** Synthetic Pointer Events for engines without CDP; one pointer per finger. */
async function sendPointer(type, target, [x, y], pointerId) {
  await target.dispatchEvent(type, {
    pointerId,
    pointerType: 'touch',
    isPrimary: pointerId === 1,
    button: 0,
    buttons: type === 'pointerup' || type === 'pointercancel' ? 0 : 1,
    clientX: x,
    clientY: y,
    bubbles: true
  });
}

async function touchPan(viewId, x, y) {
  assert.ok(useCdp, 'map touch gestures need CDP');
  const anchor = await unprojectIn(viewId, x, y);
  await sendTouch('touchStart', [[x, y]]);
  for (let step = 1; step <= 8; step++) await sendTouch('touchMove', [[x - step * 10, y]]);
  await sendTouch('touchEnd', []);
  await page.waitForTimeout(1200);
  const [projectedX] = await projectIn(viewId, anchor[0], anchor[1]);
  return x - projectedX;
}
async function touchPinch(viewId, x, y) {
  assert.ok(useCdp, 'map touch gestures need CDP');
  const zoomBefore = (await readMapState()).zoom;
  await sendTouch('touchStart', [[x - 40, y], [x + 40, y]]);
  for (let step = 1; step <= 10; step++)
    await sendTouch('touchMove', [[x - 40 - step * 8, y], [x + 40 + step * 8, y]]);
  await sendTouch('touchEnd', []);
  await page.waitForTimeout(1500);
  return (await readMapState()).zoom - zoomBefore;
}
/** Rotate the finger pair around (x, y) by `degrees` while keeping their distance. */
async function touchRotate(x, y, degrees) {
  assert.ok(useCdp, 'map touch gestures need CDP');
  const bearingBefore = (await readMapState()).bearing ?? 0;
  const radius = 60;
  const place = angle => {
    const radians = (angle * Math.PI) / 180;
    return [
      [x - radius * Math.cos(radians), y - radius * Math.sin(radians)],
      [x + radius * Math.cos(radians), y + radius * Math.sin(radians)]
    ];
  };
  await sendTouch('touchStart', place(0));
  const steps = 12;
  for (let step = 1; step <= steps; step++) await sendTouch('touchMove', place((degrees * step) / steps));
  await sendTouch('touchEnd', []);
  await page.waitForTimeout(1500);
  return ((await readMapState()).bearing ?? 0) - bearingBefore;
}

async function getDividerX() {
  return page.evaluate(() => {
    const element = document.querySelector('.layout-splitter .splitter');
    const bounds = element?.getBoundingClientRect();
    return bounds ? bounds.x + bounds.width / 2 : null;
  });
}
const getStoredPercentage = async () => (await readMapState()).swipeComparePercentage;

/** Drag the divider with one finger to targetX; returns the stored percentage after release. */
async function touchDragDivider(targetX, y = Math.round(viewport.height / 2)) {
  const startX = await getDividerX();
  assert.ok(startX !== null, 'swipe divider is mounted');
  if (useCdp) {
    await sendTouch('touchStart', [[startX, y]]);
    for (let step = 1; step <= 8; step++)
      await sendTouch('touchMove', [[startX + ((targetX - startX) * step) / 8, y]]);
    await sendTouch('touchEnd', []);
  } else {
    const divider = page.locator('.layout-splitter');
    await sendPointer('pointerdown', divider, [startX, y], 1);
    for (let step = 1; step <= 8; step++)
      await sendPointer('pointermove', divider, [startX + ((targetX - startX) * step) / 8, y], 1);
    await sendPointer('pointerup', divider, [targetX, y], 1);
  }
  await page.waitForTimeout(800);
  return getStoredPercentage();
}

const results = [];
const record = (name, value) => {
  results.push({name, value});
  console.log(`${name}: ${JSON.stringify(value)}`);
};

try {
  await reload();
  const touchAction = await page.evaluate(() => getComputedStyle(document.body).touchAction);
  record('device', {browserName, deviceName, headed, viewport, bodyTouchAction: touchAction});

  // Single view baseline for the map gestures.
  let singlePan = null;
  if (useCdp) {
    const centerX = Math.round(viewport.width / 2);
    const centerY = Math.round(viewport.height / 2);
    singlePan = await touchPan(null, centerX, centerY);
    const singleZoom = await touchPinch(null, centerX, centerY);
    assert.ok(singlePan > 20 && singleZoom > 0.3, `single touch baseline pan ${singlePan} zoom ${singleZoom}`);
    const rotation = await touchRotate(centerX, centerY, 60);
    record('single', {singlePan, singleZoom, rotation});
    if (expectRotate) assert.ok(Math.abs(rotation) > 10, `two-finger rotate changed bearing by ${rotation}`);
  }

  // Dual: each view takes gestures independently of the other.
  await reload();
  await enterSplit('Dual');
  if (useCdp) {
    const rightX = Math.round(viewport.width * 0.75);
    const centerY = Math.round(viewport.height / 2);
    const before = await readMapState();
    const pan = await touchPan('view-1', rightX, centerY);
    const zoomDelta = await touchPinch('view-1', rightX, centerY);
    const after = await readMapState();
    assert.ok(pan > 10, `Dual view 1 touch pan moved ${pan}`);
    assert.ok(zoomDelta > 0.3, `Dual view 1 pinch zoom ${zoomDelta}`);
    assert.notEqual(after.longitude, before.longitude, 'Dual touch pan changed the viewport');
    record('dual', {pan, zoomDelta});
  }
  await leaveSplit();

  // Swipe: divider drag by touch, then map gestures on the right side.
  await reload();
  await enterSplit('Swipe');
  assert.equal(await getStoredPercentage(), 50, 'swipe starts centered');
  assert.equal(
    await page.evaluate(() => getComputedStyle(document.querySelector('.layout-splitter')).touchAction),
    'none',
    'divider disables browser touch gestures'
  );
  const leftTarget = Math.round(viewport.width * 0.3);
  const leftPercentage = await touchDragDivider(leftTarget);
  assert.ok(Math.abs(leftPercentage - 30) < 3, `touch drag stored ${leftPercentage} near 30`);
  const rightPercentage = await touchDragDivider(Math.round(viewport.width * 0.8));
  assert.ok(Math.abs(rightPercentage - 80) < 3, `touch drag stored ${rightPercentage} near 80`);
  // Dragging past the edge clamps to the view.
  const clampedPercentage = await touchDragDivider(viewport.width - 1);
  assert.ok(clampedPercentage <= 100 && clampedPercentage > 90, `edge drag clamped to ${clampedPercentage}`);
  record('swipeDivider', {leftPercentage, rightPercentage, clampedPercentage});

  // pointercancel commits the last position instead of jumping (synthetic, any engine).
  const divider = page.locator('.layout-splitter');
  await touchDragDivider(Math.round(viewport.width * 0.5));
  const startX = await getDividerX();
  const middleY = Math.round(viewport.height / 2);
  await sendPointer('pointerdown', divider, [startX, middleY], 7);
  await sendPointer('pointermove', divider, [Math.round(viewport.width * 0.4), middleY], 7);
  await sendPointer('pointercancel', divider, [0, 0], 7);
  await page.waitForTimeout(800);
  const cancelledPercentage = await getStoredPercentage();
  assert.ok(Math.abs(cancelledPercentage - 40) < 3, `pointercancel kept ${cancelledPercentage}, not a jump to 0`);
  // A second pointer must not hijack a drag in progress.
  await sendPointer('pointerdown', divider, [Math.round(viewport.width * 0.4), middleY], 8);
  await sendPointer('pointerdown', divider, [Math.round(viewport.width * 0.9), middleY], 9);
  await sendPointer('pointermove', divider, [Math.round(viewport.width * 0.9), middleY], 9);
  await sendPointer('pointerup', divider, [Math.round(viewport.width * 0.9), middleY], 9);
  await sendPointer('pointerup', divider, [Math.round(viewport.width * 0.45), middleY], 8);
  await page.waitForTimeout(800);
  const secondPointerPercentage = await getStoredPercentage();
  assert.ok(Math.abs(secondPointerPercentage - 45) < 3, `second pointer ignored, stored ${secondPointerPercentage}`);
  record('swipePointers', {cancelledPercentage, secondPointerPercentage});

  if (useCdp) {
    // Map gestures right of the divider while swiping; the divider must not move.
    await touchDragDivider(Math.round(viewport.width * 0.3));
    const percentageBefore = await getStoredPercentage();
    const rightX = Math.round(viewport.width * 0.7);
    const pan = await touchPan('view-1', rightX, middleY);
    const zoomDelta = await touchPinch('view-1', rightX, middleY);
    assert.ok(pan > 10 && zoomDelta > 0.3, `swipe right side pan ${pan} pinch ${zoomDelta}`);
    assert.equal(await getStoredPercentage(), percentageBefore, 'map gestures leave the divider alone');
    record('swipeMap', {pan, zoomDelta});
  }
  await leaveSplit();

  assert.deepEqual(failures, [], 'touch gestures raise no GPU or page errors');
  console.log(`views-touch passed on ${browserName} / ${deviceName}`);
} finally {
  await browser.close();
}
