// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, expectTypeOf, it} from 'vitest';
import {readFileSync} from 'node:fs';
import {entries} from '../../../../../website/src/catalog/entries/cells-index-join';
import type {
  GPUCellAggregationFamily,
  GPUCellAggregationProps,
  GPUCellRollupProps
} from '../../../src/gpu-spatial-analysis/cell-aggregation/index';
import type {
  GPUCellIndexFamily,
  GPUCellTopologyFamily
} from '../../../src/gpu-spatial-analysis/cell-indexing/index';
import type {
  GPUCellCompactionProps,
  GPUCellGridPathProps,
  GPUCellTopologyProps
} from '../../../src/gpu-spatial-analysis/cell-topology/index';
import type {GPUCellCoverProps} from '../../../src/gpu-spatial-analysis/cell-cover/index';
import type {GPUCellSetOutlineFamily} from '../../../src/gpu-spatial-analysis/cell-set-outline/index';

it('catalog family descriptions include the public S2 subsets and generated path family', () => {
  for (const symbol of ['GPUCellGridPath', 'GPUCellCover', 'GPUCellSetOutline']) {
    const entry = entries.find(candidate => candidate.symbol === symbol)!;
    expect(entry.summary, symbol).toContain('S2');
    const family = entry.params?.find(parameter => parameter.name === 'family');
    expect(family?.description, symbol).toContain("'s2'");
    if (symbol === 'GPUCellCover') expect(family?.description).toContain('15');
    if (symbol === 'GPUCellSetOutline') expect(family?.description).toContain('one level');
  }
  const inventory = JSON.parse(
    readFileSync(
      new URL('../../../../../website/src/catalog/generated/inventory.json', import.meta.url),
      'utf8'
    )
  ) as {
    symbols: {symbol: string; props: {name: string; type: string}[]}[];
  };
  const gridPath = inventory.symbols.find(entry => entry.symbol === 'GPUCellGridPath');
  expect(gridPath?.props.find(prop => prop.name === 'family')?.type).toBe('GPUCellTopologyFamily');
});

it('exports separate aggregation and topology capability types including S2 only for topology', () => {
  expectTypeOf<GPUCellAggregationFamily>().toEqualTypeOf<'quadbin' | 'h3'>();
  expectTypeOf<GPUCellTopologyFamily>().toEqualTypeOf<'quadbin' | 'h3' | 's2'>();
  expectTypeOf<GPUCellIndexFamily>().toEqualTypeOf<GPUCellTopologyFamily | 'quadkey' | 'geohash'>();
  expectTypeOf<GPUCellAggregationProps['family']>().toEqualTypeOf<GPUCellAggregationFamily>();
  expectTypeOf<GPUCellRollupProps['family']>().toEqualTypeOf<GPUCellAggregationFamily>();
  expectTypeOf<GPUCellTopologyProps['family']>().toEqualTypeOf<GPUCellTopologyFamily>();
  expectTypeOf<GPUCellCompactionProps['family']>().toEqualTypeOf<GPUCellTopologyFamily>();
  expectTypeOf<GPUCellGridPathProps['family']>().toEqualTypeOf<GPUCellTopologyFamily>();
  expectTypeOf<GPUCellCoverProps['family']>().toEqualTypeOf<GPUCellTopologyFamily>();
  expectTypeOf<GPUCellSetOutlineFamily>().toEqualTypeOf<GPUCellTopologyFamily>();
});
