// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Shared fixtures, publication helpers and CPU references for the resident Spatial Join browser checks.
import * as arrow from 'apache-arrow';
import {openNativeData, closeNativeDialog, openNativeExport} from './native-navigation.mjs';

export const ERROR_PATTERN = /WGSL|Invalid.*(?:Pipeline|Buffer|BindGroup)|validation error|exceeds the maximum|Uncaught|destroyed/i;

export function createPolygonFeature(rings) {
  return JSON.stringify({type: 'Feature', properties: {}, geometry: {type: 'Polygon', coordinates: rings}});
}
export function createMultiPolygonFeature(polygons) {
  return JSON.stringify({type: 'Feature', properties: {}, geometry: {type: 'MultiPolygon', coordinates: polygons}});
}
export function makeSquareRing(west, south, east, north) {
  return [[west, south], [east, south], [east, north], [west, north], [west, south]];
}

/** Polygons: overlap, hole, multipolygon, null/duplicate values. Rings are given as closed GeoJSON rings. */
export const polygonRows = [
  {rings: [makeSquareRing(0, 0, 10, 10)], weight: 1.5, rank: 3},
  {rings: [makeSquareRing(5, 5, 15, 15)], weight: 2.25, rank: -2},
  {rings: [makeSquareRing(20, 0, 30, 10), makeSquareRing(24, 4, 26, 6)], weight: null, rank: null},
  {rings: [makeSquareRing(40, 0, 42, 2), makeSquareRing(44, 0, 46, 2)], multi: true, weight: 4, rank: 7}
];
export function createPolygonTable() {
  const geometry = polygonRows.map(row =>
    row.multi
      ? createMultiPolygonFeature([[row.rings[0]], [row.rings[1]]])
      : createPolygonFeature(row.rings)
  );
  return new arrow.Table({
    geometry: arrow.vectorFromArray(geometry, new arrow.Utf8()),
    weight: arrow.vectorFromArray(polygonRows.map(row => row.weight), new arrow.Float32()),
    rank: arrow.vectorFromArray(polygonRows.map(row => row.rank), new arrow.Int32())
  });
}

function isInsideRing(x, y, ring) {
  // Returns 'in', 'out' or 'edge' using exact double arithmetic on Float32-rounded coordinates.
  let inside = false;
  for (let index = 0, previous = ring.length - 1; index < ring.length; previous = index++) {
    const [ax, ay] = ring[previous].map(Math.fround);
    const [bx, by] = ring[index].map(Math.fround);
    const cross = (x - ax) * (by - ay) - (y - ay) * (bx - ax);
    if (cross === 0 && x >= Math.min(ax, bx) && x <= Math.max(ax, bx) && y >= Math.min(ay, by) && y <= Math.max(ay, by))
      return 'edge';
    if (ay > y !== by > y && x < ((bx - ax) * (y - ay)) / (by - ay) + ax) inside = !inside;
  }
  return inside ? 'in' : 'out';
}
/** Polygon parts of a row: list of rings arrays (outer + holes). */
function getParts(row) {
  return row.multi ? [[row.rings[0]], [row.rings[1]]] : [row.rings];
}
export function classifyPointForRow(x, y, row) {
  // 'in' interior, 'edge' boundary only, 'out'.
  let result = 'out';
  for (const part of getParts(row)) {
    const outer = isInsideRing(x, y, part[0]);
    if (outer === 'out') continue;
    let state = outer;
    for (const hole of part.slice(1)) {
      const holeState = isInsideRing(x, y, hole);
      if (holeState === 'in') state = 'out';
      else if (holeState === 'edge') state = 'edge';
    }
    if (state === 'in') return 'in';
    if (state === 'edge') result = 'edge';
  }
  return result;
}
export function matchPolygons(x, y, predicate) {
  const fx = Math.fround(x);
  const fy = Math.fround(y);
  return polygonRows.flatMap((row, index) => {
    const state = classifyPointForRow(fx, fy, row);
    return state === 'in' || (predicate === 'intersects' && state === 'edge') ? [index] : [];
  });
}
/** Native aggregateValues: count includes nulls, others skip nulls and give null if none defined. */
export function aggregateValues(values, technique) {
  if (technique === 'count') return values.length;
  const defined = values.filter(value => value !== null);
  if (!defined.length) return null;
  const sum = defined.reduce((total, value) => total + value, 0);
  return technique === 'sum' ? sum : technique === 'average' ? sum / defined.length
    : technique === 'maximum' ? Math.max(...defined) : Math.min(...defined);
}

export async function publishTable(page, name, table, {geometry = '', longitude = '', latitude = ''} = {}) {
  await openNativeData(page);
  const workspace = page.getByRole('region', {name: 'Resident datasets', exact: true});
  await workspace.locator('input[type=file]').setInputFiles({
    name, mimeType: 'application/vnd.apache.arrow.stream', buffer: Buffer.from(arrow.tableToIPC(table, 'stream'))
  });
  await workspace.getByLabel('Source geometry column', {exact: true}).selectOption(geometry);
  if (!geometry) {
    await workspace.getByLabel('Source longitude column', {exact: true}).selectOption(longitude);
    await workspace.getByLabel('Source latitude column', {exact: true}).selectOption(latitude);
  }
  await workspace.getByRole('button', {name: 'Check schema and precision', exact: true}).click();
  await workspace.getByRole('button', {name: 'Publish selected fields', exact: true}).click();
}
export async function selectNativeOption(page, selector, value) {
  await selector.locator('.item-selector__dropdown').click();
  await page.locator('.list-selector:visible').getByText(value, {exact: true}).click();
}
export async function exportArrow(page, datasetId) {
  await openNativeExport(page);
  await page.getByLabel('Export dataset', {exact: true}).selectOption(datasetId);
  await page.getByLabel('Export format', {exact: true}).selectOption('arrow');
  const download = page.waitForEvent('download');
  await page.getByRole('button', {name: 'Export all selected', exact: true}).click();
  const stream = await (await download).createReadStream();
  const chunks = [];
  for await (const chunk of stream) chunks.push(chunk);
  await closeNativeDialog(page);
  return arrow.tableFromIPC(Buffer.concat(chunks));
}
export function attachErrorCollectors(page, errors) {
  page.on('pageerror', error => errors.push(`pageerror: ${error.message}`));
  page.on('console', message => {
    if (message.type() === 'error' && ERROR_PATTERN.test(message.text())) errors.push(message.text());
  });
}
