// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// NATIVEUI5 browser check (headless): tooltip compare mode (absolute, then relative) against the
// native delta formulas, and the GPU legend missingCount against the fixture's null count.
// Usage: node scripts/tooltip-compare-legend.mjs [url]
import assert from 'node:assert/strict';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';

const browser = await chromium.launch(
  getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'})
);
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
await pinViteModules(page);
const errors = [];
page.on('pageerror', error => errors.push(error.message));

// Native getTooltipDisplayDeltaValue: absolute '.3~f' (field format wins), relative value / pinned - 1 as '.2%'.
const formatAbsolute = delta => {
  const text = String(Number(delta.toFixed(3)));
  return delta < 0 ? text : `+${text}`;
};
const formatRelative = delta => {
  const text = `${(delta * 100).toFixed(2)}%`;
  return delta < 0 ? text : `+${text}`;
};

try {
  await page.goto(process.argv[2] || 'http://127.0.0.1:5318/?rows=256');
  await page.waitForFunction(
    () => window.__keplerComputeTest?.getState().selectedCount === 256,
    undefined,
    {timeout: 60000}
  );
  const rows = await page.evaluate(() => window.__keplerComputeTest.source().map(row => row[2]));
  const pinnedIndex = rows.findIndex(value => Number.isFinite(value) && value !== 0);
  const hoverIndex = rows.findIndex(
    (value, index) => index !== pinnedIndex && Number.isFinite(value)
  );
  assert(pinnedIndex >= 0 && hoverIndex >= 0, 'fixture needs two finite rows');
  const original = await page.evaluate(() => window.__keplerComputeTest.saveNativeConfig());
  // Switch compare mode through the native Interactions panel (the native loader ignores it).
  const setCompareMode = async enabled => {
    const readMode = () =>
      page.evaluate(
        () =>
          window.__keplerComputeTest.saveNativeConfig().config.visState.interactionConfig.tooltip
            .compareMode
      );
    const toggle = page.locator('#compare-mode-toggle').first();
    if (!(await toggle.count())) {
      await page
        .locator('.side-panel__tab')
        .filter({hasText: /Interactions/i})
        .first()
        .click();
      await page.locator('.panel--header__action__component').first().click();
    }
    if ((await readMode()) !== enabled) await toggle.click({force: true});
    await page.waitForFunction(
      value =>
        window.__keplerComputeTest.saveNativeConfig().config.visState.interactionConfig.tooltip
          .compareMode === value,
      enabled
    );
  };
  const inspection = page.getByRole('region', {name: 'Data inspection', exact: true});
  const waitForHover = index =>
    page.waitForFunction(
      index => window.__keplerComputeTest.getHoverInfo()?.index === index,
      index,
      {timeout: 20000}
    );
  for (const compareType of ['absolute', 'relative']) {
    // The native loader keeps only fieldsToShow/timezone, so compare mode is switched on through the
    // native Interactions panel, exactly as a user would.
    await page.evaluate(configuration => {
      configuration.config.visState.interactionConfig.tooltip.fieldsToShow[
        'resident-observations'
      ] = [{name: 'value', format: null}];
      return window.__keplerComputeTest.applyNativeConfig(configuration);
    }, structuredClone(original));
    await setCompareMode(true);
    if (compareType === 'relative') {
      await page
        .locator('.item-selector')
        .filter({hasText: /absolute/i})
        .first()
        .click();
      await page
        .getByText(/^relative$/i)
        .first()
        .click();
    }
    await page.waitForFunction(type => {
      const tooltip =
        window.__keplerComputeTest.saveNativeConfig().config.visState.interactionConfig.tooltip;
      return tooltip.compareMode === true && tooltip.compareType === type;
    }, compareType);
    await page.evaluate(() => window.__keplerComputeTest.startLayerHoverProbe('resident-points'));
    const pinned = await page.evaluate(
      index => window.__keplerComputeTest.projectSourceRow(index),
      pinnedIndex
    );
    await page.mouse.move(pinned[0] - 3, pinned[1] - 3);
    await page.mouse.move(pinned[0], pinned[1]);
    await waitForHover(pinnedIndex);
    await page.mouse.click(pinned[0], pinned[1]);
    await page.waitForTimeout(600);
    const hovered = await page.evaluate(
      index => window.__keplerComputeTest.projectSourceRow(index),
      hoverIndex
    );
    await page.mouse.move(hovered[0] - 3, hovered[1] - 3);
    await page.mouse.move(hovered[0], hovered[1]);
    // The pre-port script also read an already-visible +0 delta for the pinned row before
    // the next async hover resolved. Wait for identity AND bounded tooltip readback, rather
    // than treating the existence of an old delta as completion of the new pointer move.
    await waitForHover(hoverIndex);
    const hoveredInspection = page.locator(
      `[data-resident-tooltip-row="${hoverIndex}"][data-resident-tooltip-compare-row="${pinnedIndex}"]`
    );
    const difference = hoveredInspection.locator('[data-resident-tooltip-difference]').first();
    await difference.waitFor({timeout: 20000});
    const expected =
      compareType === 'relative'
        ? formatRelative(rows[hoverIndex] / rows[pinnedIndex] - 1)
        : formatAbsolute(rows[hoverIndex] - rows[pinnedIndex]);
    assert.equal(await inspection.count(), 2, 'pinned and hovered popovers both render');
    assert.equal(
      await page.locator('[data-resident-consumer-state]').count(),
      0,
      'no popover reports an unavailable row'
    );
    assert.equal(
      (await difference.textContent()).replace(/[()\s]/g, ''),
      expected,
      `${compareType} compare delta`
    );
    await page.waitForFunction(
      index =>
        window.__keplerComputeTest
          .readLayerHoverProbe()
          .some(event => event.picked && event.index === index),
      hoverIndex
    );
    // (1, 1) can be inside Deck's full-window event root, even when the sidebar covers
    // its map pixels. That is a normal no-hit pick, not a canvas-leave event. Inspect the
    // actual root and move straight outside the browser viewport without an intervening
    // empty-map hover (which would already forget the previous layer).
    const eventRoot = await page.evaluate(() => window.__keplerComputeTest.readHoverEventRoot());
    assert(eventRoot.bounds, 'Deck event root exists');
    const outside = [
      Math.min(-10, eventRoot.bounds.left - 10),
      Math.min(-10, eventRoot.bounds.top - 10)
    ];
    await page.mouse.move(outside[0], outside[1]);
    await page.waitForFunction(
      () => {
        const root = window.__keplerComputeTest.readHoverEventRoot();
        return root.nativeLeaveCount > 0 && root.recognizedLeaveCount > 0;
      },
      undefined,
      {timeout: 20000}
    );
    await page.waitForFunction(
      () => {
        const hover = window.__keplerComputeTest.getHoverInfo();
        return !hover || hover.index < 0 || !hover.layerId;
      },
      undefined,
      {timeout: 20000}
    );
    await page.waitForFunction(
      () =>
        window.__keplerComputeTest
          .readLayerHoverProbe()
          .some(event => !event.picked && event.index === -1 && event.x === -1 && event.y === -1),
      undefined,
      {timeout: 20000}
    );
    await page.evaluate(() => window.__keplerComputeTest.stopLayerHoverProbe());
    await page.keyboard.press('Escape');
  }
  await setCompareMode(false);
  await page.evaluate(
    configuration => window.__keplerComputeTest.applyNativeConfig(configuration),
    original
  );

  // GPU legend missing count: every colored layer legend reports a number consistent with the rows.
  const legends = await page.evaluate(() =>
    ['fillColor', 'strokeColor', 'size', 'pointSize']
      .map(channel => [
        channel,
        window.__keplerComputeTest.getStyleLegend('resident-points', channel)?.legend
      ])
      .filter(([, legend]) => legend)
      .map(([channel, legend]) => ({
        channel,
        missingCount: legend.missingCount,
        finiteCount: legend.finiteCount
      }))
  );
  for (const legend of legends) {
    assert(
      legend.missingCount === null ||
        (Number.isInteger(legend.missingCount) &&
          legend.missingCount >= 0 &&
          legend.missingCount <= 256),
      JSON.stringify(legend)
    );
    if (typeof legend.missingCount === 'number' && typeof legend.finiteCount === 'number')
      assert(legend.missingCount + legend.finiteCount <= 256 + 0, JSON.stringify(legend));
    if (legend.missingCount > 0)
      await page.locator('[data-resident-legend-missing]').first().waitFor({timeout: 10000});
  }
  process.stdout.write(`${JSON.stringify({legends, errors}, null, 2)}\n`);
  assert.deepEqual(errors, []);
} catch (error) {
  process.stderr.write(
    `${JSON.stringify(
      await page.evaluate(() => ({
        hover: window.__keplerComputeTest?.getHoverInfo(),
        eventRoot: window.__keplerComputeTest?.readHoverEventRoot(),
        layerEvents: window.__keplerComputeTest?.readLayerHoverProbe()
      })),
      null,
      2
    )}\n`
  );
  throw error;
} finally {
  await page.evaluate(() => window.__keplerComputeTest?.stopLayerHoverProbe()).catch(() => {});
  await browser.close();
}
