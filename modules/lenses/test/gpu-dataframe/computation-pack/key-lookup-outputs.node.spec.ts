// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {GROUPS_JOINS_OPERATORS} from '../../../src/gpu-dataframe/computation-pack/operators/groups-joins';
import {createRandom} from '../../../../../website/src/gpu/previews/data';
import {DATAFRAME_POINTCLOUD_RECORDS_PREVIEWS} from '../../../../../website/src/gpu/previews/entries/dataframe-pointcloud-records';

test('the key lookup pack describes left rows and right keys and keeps its recorded formula ID', () => {
  const lookup = GROUPS_JOINS_OPERATORS['GPUKeyLookup']!;
  expect(lookup.datumKinds).toEqual([
    {kind: 'row', label: 'Left rows', indexSpace: '1d'},
    {kind: 'group', label: 'Unique right keys', indexSpace: '1d'}
  ]);
  expect(lookup.formula).toMatchObject({id: 'gpu-dataframe:GPUKeyJoin'});
});

test('key lookup compaction adds a list in left-row order and preserves aligned slots', () => {
  const preview = DATAFRAME_POINTCLOUD_RECORDS_PREVIEWS['key-join']!;
  const build = (compactMatchedLeftRows: number) => {
    const random = createRandom(1);
    const keys = [3, 7, 1, 3, 6, 0, 5, 7, 2].values();
    return preview.build({
      random,
      fork: () => ({...random, integer: () => keys.next().value!}),
      p: {compactMatchedLeftRows},
      pointer: null
    });
  };
  expect(preview.parameters).toMatchObject([
    {id: 'compactMatchedLeftRows', options: ['off', 'on']}
  ]);
  const aligned = build(0);
  const compact = build(1);
  expect(compact.slice(0, aligned.length)).toEqual(aligned);
  const alignedRows = aligned.flatMap(layer =>
    layer.type === 'sprites' ? layer.items.filter(marker => marker.x === 76) : []
  );
  expect(alignedRows.map(marker => marker.y).sort((first, second) => first - second)).toEqual([
    14, 23, 32, 41, 50, 59, 68, 77, 86
  ]);
  const compactColumns = compact
    .slice(aligned.length)
    .flatMap(layer => (layer.type === 'sprites' ? layer.items : []));
  // Left rows 0, 2, 3, 5, 6, and 8 match. The repeated key 3 appears once per left row.
  expect(compactColumns.map(marker => marker.phase)).toEqual(
    [0, 2, 3, 5, 6, 8].map(row => row / 9)
  );
  expect(compactColumns.map(marker => [marker.x, marker.y, marker.role])).toEqual([
    [152, 14, 'cool'],
    [152, 23, 'warm'],
    [152, 32, 'cool'],
    [152, 41, 'primary'],
    [152, 50, 'primary'],
    [152, 59, 'hot']
  ]);
});
