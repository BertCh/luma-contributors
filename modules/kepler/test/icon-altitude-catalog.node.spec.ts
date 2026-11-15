// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {isIconCoordinateVisible} from '../../../examples/deck/kepler-compute/src/icon-altitude';
import {
  ICON_CATALOG_FORMAT,
  DEFAULT_ICON_CATALOG_LIMITS,
  makeIconCatalogDocument,
  parseIconCatalogDocument,
  selectEmbeddedIconCatalog
} from '../../../examples/deck/kepler-compute/src/icon-catalog-assets';
import {
  prepareNativeIconCatalog,
  prepareNativeIconSource,
  type NativeIconDefinition
} from '../../../examples/deck/kepler-compute/src/icon-source';
import * as arrow from 'apache-arrow';
import {
  CUSTOM_ICON_DEFINITIONS,
  makeIconAltitudeTable
} from '../../../examples/deck/kepler-compute/scripts/icons-altitude-labels-fixture.mjs';

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

test('portable icon catalogs reject malformed or unused nonfinite coordinates before JSON rewrites them', () => {
  for (const position of [
    [NaN, 0, 0],
    [Infinity, 0, 0],
    [0, 0],
    [1e100, 0, 0]
  ]) {
    expect(() =>
      makeIconCatalogDocument([
        {...triangle, mesh: {...triangle.mesh, positions: [...triangle.mesh.positions, position]}}
      ])
    ).toThrow(/custom-triangle.*coordinates/);
  }
  expect(() =>
    parseIconCatalogDocument({
      format: ICON_CATALOG_FORMAT,
      version: 1,
      icons: [{id: 'broken', mesh: {positions: [[0, 0, 0]], cells: [null]}}]
    })
  ).toThrow(/broken.*triangles/);
  expect(() =>
    makeIconCatalogDocument([triangle], {...DEFAULT_ICON_CATALOG_LIMITS, maximumDocumentBytes: 1})
  ).toThrow(/document budget/);
});

test('portable browser fixture consumes only its embedded custom mesh with source-aligned picking and no defaults', () => {
  const original = makeIconAltitudeTable();
  const source = arrow.tableFromIPC(arrow.tableToIPC(original, 'stream'));
  const symbols = source.getChild('symbol')!;
  const icons = Array.from({length: source.numRows}, (_, index) => symbols.get(index));
  expect(icons).toEqual(Array(5).fill('custom-square'));
  const embedded = parseIconCatalogDocument(
    JSON.parse(JSON.stringify(makeIconCatalogDocument(CUSTOM_ICON_DEFINITIONS)))
  );
  const catalog = prepareNativeIconCatalog([], embedded);
  const validity = Array.from({length: source.numRows}, (_, index) =>
    Number(
      isIconCoordinateVisible(
        source.getChild('east')!.get(index),
        source.getChild('north')!.get(index),
        source.getChild('height')!.get(index),
        true
      )
    )
  );
  const prepared = prepareNativeIconSource([{id: 'restored', icons, validity}], catalog);
  expect(prepared.groups).toHaveLength(1);
  const group = prepared.groups[0];
  expect(group.iconId).toBe('custom-square');
  expect([...group.vertices]).toEqual([-1, 1, 0, 1, 1, 0, 1, -1, 0, -1, 1, 0, 1, -1, 0, -1, -1, 0]);
  expect([...group.sourceRowIds[0]]).toEqual([0, 1, 2, 3]);
  expect(group.resolveSourceRow(1)).toBe(1);
  expect(group.resolveSourceRow(3)).toBe(3);
  expect(group.resolveSourceRow(4)).toBe(-1);
  expect(() =>
    prepareNativeIconSource(
      [{id: 'without-embedded', icons, validity}],
      prepareNativeIconCatalog([])
    )
  ).toThrow('custom-square');
});
