// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Shared helpers for the raster panel browser scripts (raster-zonal-panel.mjs, raster-terrain-gpu.mjs).
// They need a running dev server (port 5318) and default headless Playwright only.
import * as arrow from 'apache-arrow';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';
import {openNativeData, closeNativeDialog} from './native-navigation.mjs';

export const RASTER_TILE_INDEX = {x: 301, y: 384, z: 10};
export const RASTER_TILE_SIZE = 32;

export async function openRasterPage(url, watchdogMilliseconds = 170000) {
  const browser = await chromium.launch(
    getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'})
  );
  const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
  await pinViteModules(page);
  const failures = [];
  page.on('pageerror', error => failures.push(error.message));
  page.on('console', message => {
    if (
      message.type() === 'error' &&
      /WGSL|Invalid.*(?:Pipeline|Buffer|Texture)|validation error/i.test(message.text())
    )
      failures.push(message.text());
  });
  const watchdog = setTimeout(() => {
    console.error(`Script exceeded ${watchdogMilliseconds / 1000} seconds`);
    void browser.close();
  }, watchdogMilliseconds);
  await page.goto(url || 'http://127.0.0.1:5318/?rows=256', {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(
    () => window.__keplerComputeTest?.getState().selectedCount === 256,
    undefined,
    {timeout: 60000}
  );
  return {browser, page, failures, close: async () => {
    clearTimeout(watchdog);
    await browser.close();
  }};
}

/** Injects a synthetic Kepler-shaped raster tile layer: a 32x32 elevation ramp with a nodata hole. */
export async function injectElevationRaster(page, layerId, label) {
  await page.evaluate(
    ({layerId, label, index, size}) => {
      const elevation = new Float32Array(size * size);
      const mask = new Float32Array(size * size).fill(1);
      for (let row = 0; row < size; row++)
        for (let column = 0; column < size; column++) {
          elevation[row * size + column] = 200 * column + 120 * row + 50 * Math.sin(column + row);
          // A nodata block exercises null rows and validity handling.
          if (column >= 4 && column < 8 && row >= 4 && row < 8) mask[row * size + column] = 0;
        }
      window.__keplerComputeTest.injectRasterTestLayer(layerId, label, [
        {
          index,
          data: {
            images: {
              imageBands: [{data: elevation, width: size, height: size}],
              imageMask: {data: mask, width: size, height: size}
            }
          }
        }
      ]);
    },
    {layerId, label, index: RASTER_TILE_INDEX, size: RASTER_TILE_SIZE}
  );
}

/** Extent [west, south, east, north] of the injected tile in longitude/latitude, from the app's own tile math. */
export async function getRasterTileExtent(page) {
  return page.evaluate(async index => {
    const ingest = await import('/src/raster-tile-ingest.ts');
    const reprojection = await import('/src/raster-reprojection.ts');
    const [west, south, east, north] = ingest.getXyzTileExtent(index);
    const plan = reprojection.planRasterReprojection('EPSG:3857', 'EPSG:4326');
    return [...plan.transformSourceToTarget(west, south), ...plan.transformSourceToTarget(east, north)];
  }, RASTER_TILE_INDEX);
}

/** Publishes two GeoJSON polygons inside the injected tile through the Prepare data dialog. */
export async function publishTilePolygons(page, extent) {
  const [west, south, east, north] = extent;
  const width = east - west;
  const height = north - south;
  const ring = (x0, y0, x1, y1) => [
    [west + width * x0, south + height * y0],
    [west + width * x1, south + height * y0],
    [west + width * x1, south + height * y1],
    [west + width * x0, south + height * y1],
    [west + width * x0, south + height * y0]
  ];
  const polygon = (x0, y0, x1, y1) =>
    JSON.stringify({
      type: 'Feature',
      geometry: {type: 'Polygon', coordinates: [ring(x0, y0, x1, y1)]},
      properties: {}
    });
  const source = arrow.tableFromArrays({
    geometry: [polygon(0.1, 0.1, 0.45, 0.45), polygon(0.55, 0.5, 0.9, 0.95)],
    zoneValue: new Float32Array([1, 2])
  });
  await openNativeData(page);
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await workspace.locator('input[type=file]').setInputFiles({
    name: 'raster-zones.arrow',
    mimeType: 'application/vnd.apache.arrow.stream',
    buffer: Buffer.from(arrow.tableToIPC(source, 'stream'))
  });
  await workspace.getByLabel('Source geometry column').selectOption('geometry');
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
  await page.waitForFunction(
    () => window.__keplerComputeTest.getState().publications[0]?.selectedCount === 2,
    undefined,
    {timeout: 45000}
  );
  await closeNativeDialog(page);
  return (await page.evaluate(() => window.__keplerComputeTest.getState())).publications[0].id;
}
