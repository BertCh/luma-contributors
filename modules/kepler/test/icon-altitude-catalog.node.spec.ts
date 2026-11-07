// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {isIconCoordinateVisible} from '../../../examples/deck/kepler-compute/src/icon-altitude';
import {
  ICON_CATALOG_FORMAT,
  makeIconCatalogDocument,
  parseIconCatalogDocument,
  selectEmbeddedIconCatalog
} from '../../../examples/deck/kepler-compute/src/icon-catalog-assets';
import type {NativeIconDefinition} from '../../../examples/deck/kepler-compute/src/icon-source';

const triangle: NativeIconDefinition = {
  id: 'custom-triangle',
  mesh: {
    positions: [
      [-1, 0.3, 0],
      [1, 0.3, 0],
      [0, -1, 0.25]
    ],
    cells: [[0, 1, 2]]
  }
};

test('icon altitude visibility treats null as zero and hides nonfinite values', () => {
  expect(isIconCoordinateVisible(1, 2, null, true)).toBe(true);
  expect(isIconCoordinateVisible(1, 2, undefined, true)).toBe(true);
  expect(isIconCoordinateVisible(1, 2, 5, true)).toBe(true);
  expect(isIconCoordinateVisible(1, 2, Number.NaN, true)).toBe(false);
  expect(isIconCoordinateVisible(1, 2, Number.POSITIVE_INFINITY, true)).toBe(false);
  expect(isIconCoordinateVisible(1, 2, '3', true)).toBe(false);
  expect(isIconCoordinateVisible(1, 2, Number.NaN, false)).toBe(true);
  expect(isIconCoordinateVisible(null, 2, 0, true)).toBe(false);
  expect(isIconCoordinateVisible(1, Number.NaN, 0, false)).toBe(false);
});

test('custom icon catalog round-trips through JSON and reports malformed meshes', () => {
  const document = makeIconCatalogDocument([triangle]);
  expect(document.format).toBe(ICON_CATALOG_FORMAT);
  const restored = parseIconCatalogDocument(JSON.parse(JSON.stringify(document)));
  expect(restored).toEqual([triangle]);
  expect(selectEmbeddedIconCatalog(undefined)).toBeUndefined();
  expect(selectEmbeddedIconCatalog([])).toBeUndefined();
  expect(selectEmbeddedIconCatalog([triangle])?.icons).toHaveLength(1);
  expect(() => parseIconCatalogDocument({format: 'other', version: 1, icons: []})).toThrow();
  expect(() =>
    parseIconCatalogDocument({
      ...document,
      icons: [{id: 'bad', mesh: {positions: [[0, 0, 0]], cells: [[0, 1, 2]]}}]
    })
  ).toThrow(/bad/);
});

test('later duplicate custom icon ids override earlier ones', () => {
  const replacement: NativeIconDefinition = {
    id: triangle.id,
    mesh: {...triangle.mesh, positions: triangle.mesh.positions.map(p => [p[0] * 2, p[1], p[2]])}
  };
  const document = makeIconCatalogDocument([triangle, replacement]);
  expect(document.icons).toHaveLength(1);
  expect(document.icons[0].mesh.positions[0][0]).toBe(-2);
});
