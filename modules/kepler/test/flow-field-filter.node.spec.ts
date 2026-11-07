// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {buildFlowFieldGrid} from '../../../examples/deck/kepler-compute/src/flow-field-grid';
import {
  FlowFieldSelectionCache,
  getRemovedLayerIds,
  readSelectionWords
} from '../../../examples/deck/kepler-compute/src/flow-field-filter';

function makeSamples(selected?: number[]) {
  const longitudes: number[] = [];
  const latitudes: number[] = [];
  for (let row = 0; row < 8; row++) {
    longitudes.push(row % 4);
    latitudes.push(Math.floor(row / 4));
  }
  return {
    longitudes,
    latitudes,
    first: longitudes.map(() => 1),
    second: longitudes.map(() => 0),
    selected
  };
}

function makeMask(words: number[], chunkSize = 4) {
  const data = [];
  for (let start = 0; start < words.length; start += chunkSize) {
    const chunkWords = new Uint32Array(words.slice(start, start + chunkSize));
    data.push({
      byteOffset: 0,
      length: chunkWords.length,
      buffer: {readAsync: async () => new Uint8Array(chunkWords.buffer)}
    });
  }
  return {length: words.length, data} as never;
}

const options = {mode: 'UV', gridResolution: 8, smoothing: 0} as const;

test('field grid excludes rows removed by the layer filter', () => {
  const unfiltered = buildFlowFieldGrid(makeSamples(), options);
  expect(unfiltered?.sampleCount).toBe(8);
  const filtered = buildFlowFieldGrid(makeSamples([1, 1, 1, 1, 1, 0, 0, 0]), options);
  expect(filtered?.sampleCount).toBe(5);
  expect(filtered?.skippedCount).toBe(0);
  // Fewer than four selected samples cannot form a field, like native.
  expect(buildFlowFieldGrid(makeSamples([1, 1, 0, 0, 0, 0, 0, 1]), options)).toBeNull();
});

test('selection words are read across chunks', async () => {
  const words = await readSelectionWords(makeMask([1, 0, 1, 1, 0, 1]));
  expect([...words]).toEqual([1, 0, 1, 1, 0, 1]);
});

test('selection cache applies a revision once and releases with the layer', async () => {
  const cache = new FlowFieldSelectionCache();
  let revision = 1;
  let updates = 0;
  const context = {
    mask: makeMask([1, 1, 0, 1]),
    getRevision: () => revision,
    requestUpdate: () => updates++
  };
  expect(cache.resolve('layer', context)).toMatchObject({words: undefined, key: 'filter:pending'});
  await new Promise(resolve => setTimeout(resolve, 0));
  const applied = cache.resolve('layer', context);
  expect(applied.key).toBe('filter:1');
  expect([...applied.words!]).toEqual([1, 1, 0, 1]);
  expect(updates).toBe(1);
  revision = 2;
  // The previous words keep serving until the new revision has been read back.
  expect(cache.resolve('layer', context).key).toBe('filter:1');
  await new Promise(resolve => setTimeout(resolve, 0));
  expect(cache.resolve('layer', context).key).toBe('filter:2');
  expect(cache.resolve('layer', undefined).key).toBe('unfiltered');
  cache.destroy();
});

test('removed layers are those cached but no longer active', () => {
  expect(getRemovedLayerIds(['a', 'b', 'c'], new Set(['b']))).toEqual(['a', 'c']);
});
