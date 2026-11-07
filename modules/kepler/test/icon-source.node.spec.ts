// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import {dirname, join} from 'node:path';
import {runInNewContext} from 'node:vm';
import {expect, test, vi} from 'vitest';
import type {Device} from '@luma.gl/core';
import {
  prepareNativeIconCatalog,
  prepareNativeIconGeometry,
  prepareNativeIconSource,
  loadNativeIconCatalog,
  type NativeIconDefinition
} from '../../../examples/deck/kepler-compute/src/icon-source';
import {makeGPUVectorsFromNativeIcons} from '../../../examples/deck/kepler-compute/src/icon-runtime';

const triangle: NativeIconDefinition = {
  id: 'triangle',
  mesh: {
    positions: [
      [-1, 0.3, 0],
      [1, 0.3, 0],
      [0, -1, 0.25]
    ],
    cells: [[0, 1, 2]]
  }
};

test('native icon preparation matches the pinned mesh flattener without rasterization or normalization', () => {
  const require = createRequire(import.meta.url);
  const source = readFileSync(
    join(dirname(require.resolve('@kepler.gl/layers')), 'icon-layer/icon-layer.js'),
    'utf8'
  );
  const nativeFlattener = source.match(/function flatterIconPositions\(icon\) \{[\s\S]*?\n\}/)?.[0];
  expect(nativeFlattener).toBeTruthy();
  const native = runInNewContext(`${nativeFlattener}; flatterIconPositions(icon)`, {
    icon: triangle
  });
  const catalog = prepareNativeIconCatalog([triangle]);
  expect(catalog.meshes.get('triangle')).toEqual(new Float32Array(native));
  const copied = prepareNativeIconGeometry({triangle: native});
  expect(copied.meshes.get('triangle')).toEqual(catalog.meshes.get('triangle'));
  expect(copied.meshes.get('triangle')).not.toBe(catalog.meshes.get('triangle'));
  expect(copied.typedStorageByteLength).toBe(36);
});

test('custom icons override by ID and immutable per-icon row maps preserve source batches, nulls and coordinate validity', () => {
  const replacement = {...triangle, mesh: {...triangle.mesh, cells: [[2, 1, 0]]}};
  const catalog = prepareNativeIconCatalog([triangle, {...triangle, id: 'other'}], [replacement]);
  expect(Array.from(catalog.meshes.keys())).toEqual(['triangle', 'other']);
  expect(catalog.meshes.get('triangle')?.slice(0, 3)).toEqual(new Float32Array([0, 1, 0.25]));
  const source = prepareNativeIconSource(
    [
      {id: 'first', icons: ['triangle', null, 'other']},
      {id: 'empty', icons: []},
      {id: 'last', icons: ['other', 'triangle', 'unknown', 7], validity: [1, 1, 0, 1]}
    ],
    catalog
  );
  expect(source.batches.map(batch => [batch.id, batch.sourceRowOffset, batch.rowCount])).toEqual([
    ['first', 0, 3],
    ['empty', 3, 0],
    ['last', 3, 4]
  ]);
  expect(source.groups.map(group => group.sourceRowIds.map(chunk => Array.from(chunk)))).toEqual([
    [[0], [], [4]],
    [[2], [], [3]]
  ]);
  expect(source.groups[0].resolveSourceRow(1)).toBe(4);
  expect(source.groups[0].resolveSourceRow(2)).toBe(-1);
  expect(source.groups[0].resolveSourceRow(-1)).toBe(-1);
  expect(source.iconRowCount).toBe(4);
  expect(source.omittedRowCount).toBe(3);
  expect(source.typedStorageByteLength).toBe(88);
});

test('missing markers and invalid meshes fail explicitly before upload; source budgets admit no allocation', () => {
  const catalog = prepareNativeIconCatalog([triangle]);
  expect(() => prepareNativeIconSource([{id: 'rows', icons: ['missing']}], catalog)).toThrow(
    'source row 0 is unavailable'
  );
  expect(() =>
    prepareNativeIconCatalog([{...triangle, mesh: {...triangle.mesh, cells: [[0, 1]]}}])
  ).toThrow('triangles');
  expect(() => prepareNativeIconGeometry({bad: [1, 2, NaN, 0, 0, 0, 1, 1, 1]})).toThrow(
    'invalid triangle coordinates'
  );
  expect(() => prepareNativeIconCatalog([triangle], [], {maximumVertexCount: 2})).toThrow(
    'geometry limits'
  );
  const prepared = prepareNativeIconSource([{id: 'rows', icons: ['triangle']}], catalog);
  const createBuffer = vi.fn();
  expect(() =>
    makeGPUVectorsFromNativeIcons({createBuffer} as unknown as Device, prepared, {
      workingMemoryBudgetBytes: 0
    })
  ).toThrow('admitted memory');
  expect(createBuffer).not.toHaveBeenCalled();
});

test('catalog fetch/parse failures stay visible and local custom definitions win over remote definitions', async () => {
  const request = vi
    .fn()
    .mockResolvedValueOnce({ok: true, json: async () => ({svgIcons: [triangle]})})
    .mockResolvedValueOnce({
      ok: true,
      json: async () => ({svgIcons: [{...triangle, mesh: {...triangle.mesh, cells: [[2, 1, 0]]}}]})
    });
  const catalog = await loadNativeIconCatalog({
    defaultUrl: 'https://example.test/default',
    customUrl: 'https://example.test/custom',
    customDefinitions: [triangle],
    fetch: request
  });
  expect(catalog.meshes.get('triangle')).toEqual(
    prepareNativeIconCatalog([triangle]).meshes.get('triangle')
  );
  expect(request.mock.calls[0][1]).toMatchObject({mode: 'cors', cache: 'no-cache'});
  await expect(
    loadNativeIconCatalog({
      defaultUrl: 'missing',
      fetch: vi.fn().mockResolvedValue({ok: false, status: 404})
    })
  ).rejects.toThrow('404');
  await expect(
    loadNativeIconCatalog({
      defaultUrl: 'malformed',
      fetch: vi.fn().mockResolvedValue({ok: true, json: async () => ({icons: []})})
    })
  ).rejects.toThrow('svgIcons array');
});
