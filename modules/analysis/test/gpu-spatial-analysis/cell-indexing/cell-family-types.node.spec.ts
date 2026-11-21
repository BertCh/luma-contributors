// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {
  GPUCellMeasures,
  type GPUCellMeasuresOutput
} from '../../../src/gpu-spatial-analysis/cell-indexing/gpu-cell-measures';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {expect, expectTypeOf, it} from 'vitest';
import {GPU_CELL_MAXIMUM_RESOLUTION} from '../../../src/gpu-spatial-analysis/cell-aggregation/cell-keys';
import {
  GPU_CELL_INDEX_FAMILIES,
  GPU_CELL_INDEX_RESOLUTION_RANGES
} from '../../../src/gpu-spatial-analysis/cell-indexing/cell-index-families';
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

it('library family contracts pin their public ranges without website dependencies', () => {
  expect(GPU_CELL_INDEX_FAMILIES).toEqual(['quadbin', 'h3', 'quadkey', 'geohash', 's2']);
  expect(GPU_CELL_INDEX_RESOLUTION_RANGES).toEqual({
    quadbin: {minimum: 0, maximum: 26},
    h3: {minimum: 0, maximum: 15},
    quadkey: {minimum: 1, maximum: 29},
    geohash: {minimum: 1, maximum: 12},
    s2: {minimum: 0, maximum: 30}
  });
  expect(GPU_CELL_MAXIMUM_RESOLUTION).toEqual({quadbin: 26, h3: 15});
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

it('W3-02 accepts each minimal terminal measure and rejects an empty selection', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const cells = createTransientView(graph, 'cells', 'uint32x2', 1);
    const scalar = createTransientView(graph, 'scalar', 'float32', 1);
    const edges = createTransientView(graph, 'edges', 'float32', 6);
    const integer = createTransientView(graph, 'integer', 'uint32', 1);
    const selections: GPUCellMeasuresOutput[] = [
      {areas: scalar},
      {perimeters: scalar},
      {edgeLengths: edges},
      {edgeCounts: integer},
      {pentagons: integer}
    ];
    for (const output of selections)
      expect(() => new GPUCellMeasures({family: 'h3', cells, output})).not.toThrow();
    // @ts-expect-error At least one actual terminal measure must be selected.
    const empty: GPUCellMeasuresOutput = {};
    expect(() => new GPUCellMeasures({family: 'h3', cells, output: empty})).toThrow(/at least one/);
  } finally {
    device.destroy();
  }
});
