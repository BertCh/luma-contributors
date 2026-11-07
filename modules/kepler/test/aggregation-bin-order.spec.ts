// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {Buffer, type Device} from '@luma.gl/core';
import {GPUVector} from '@luma-contributors/graph/gpu-data';
import {KeplerDatasetRegistry} from '@luma-contributors/kepler';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '../../graph/test/utils/require-test-device';
import {
  AggregationRuntime,
  type NativeAggregationLayer
} from '../../../examples/deck/kepler-compute/src/aggregation-runtime';
import {getFirstAppearanceIndices} from '../../../examples/deck/kepler-compute/src/aggregation-scale-lookup';

function makeVector<Format extends 'uint32' | 'float32'>(
  device: Device,
  format: Format,
  values: Uint32Array | Float32Array
): GPUVector<Format> {
  return new GPUVector({
    type: 'buffer',
    name: 'bin-order-column',
    format,
    length: values.length,
    buffer: device.createBuffer({
      data: values,
      usage: Buffer.STORAGE | Buffer.VERTEX | Buffer.COPY_SRC | Buffer.COPY_DST
    }),
    ownsBuffer: true
  });
}

type StatisticsProbe = {
  programs: Map<string, {statistics?: {readScaleInput(): Promise<Float32Array>}}>;
};

type BinOrderCase = {
  codes: number[];
  longitudeOffsets: number[];
  dictionarySize: number;
  /** Rows visible to values (GPU filters); defaults to all rows. */
  visible?: number[];
  /** Rows that form bins (native CPU-filtered topology); defaults to `visible`. */
  topology?: number[];
};

async function readCellIndices(kind: 'grid' | 'hexagon', testCase: BinOrderCase) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {codes, longitudeOffsets, dictionarySize} = testCase;
  const visible = testCase.visible ?? codes.map(() => 1);
  const mask = makeVector(device, 'uint32', Uint32Array.from(visible));
  const topologyMask = testCase.topology
    ? makeVector(device, 'uint32', Uint32Array.from(testCase.topology))
    : undefined;
  const category = makeVector(device, 'uint32', Uint32Array.from(codes));
  const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 64 * 1024 * 1024});
  const labels = Array.from({length: dictionarySize}, (_, index) => `label-${index}`);
  const runtime = new AggregationRuntime({
    device,
    memoryRegistry: registry,
    longitude: longitudeOffsets.map(offset => -74 + offset * 0.1),
    latitude: codes.map(() => 40.7),
    selectionMask: mask,
    columns: {category: {values: category}},
    getCategoryLabels: column => (column === 'category' ? labels : undefined)
  });
  try {
    const layer: NativeAggregationLayer = {
      id: kind,
      type: kind,
      config: {
        columns: {lat: {value: 'latitude'}, lng: {value: 'longitude'}},
        colorField: {name: 'category', type: 'string'},
        colorScale: 'linear',
        visConfig: {
          worldUnitSize: 2,
          coverage: 1,
          colorAggregation: 'mode',
          colorRange: {colors: ['#000000', '#ffffff']}
        }
      }
    };
    expect(runtime.resolve(layer, mask, 0, topologyMask).reason).toBeUndefined();
    const encoder = device.createCommandEncoder();
    runtime.encode(encoder);
    device.submit(encoder.finish());
    const statistics = (runtime as unknown as StatisticsProbe).programs.get(kind)!.statistics!;
    return Array.from(await statistics.readScaleInput());
  } finally {
    const fence = device.createFence();
    await runtime.destroy(fence.signaled);
    fence.destroy();
    mask.destroy();
    topologyMask?.destroy();
    category.destroy();
  }
}

/** One point per cell, source rows deliberately out of spatial order: cell-id order != bin order. */
it.each([
  'grid',
  'hexagon'
] as const)('categorical mode %s cells are indexed by first source row (native bin order) on the GPU', async kind => {
  const codes = [2, 0, 2, 1, 0, 1];
  const longitudeOffsets = [3, 0, 5, 1, 4, 2];
  const cellValues = (
    await readCellIndices(kind, {codes, longitudeOffsets, dictionarySize: 3})
  ).filter(Number.isFinite);
  // Expected: rows ordered by longitude give the cell-id order; value = first-appearance index.
  const expectedIndices = getFirstAppearanceIndices(
    codes.map((_, row) => row),
    codes
  );
  const rowsByCell = longitudeOffsets
    .map((offset, row) => ({offset, row}))
    .sort((first, second) => first.offset - second.offset);
  const expected = rowsByCell.map(({row}) => expectedIndices.get(codes[row])!);
  expect(cellValues.length).toBe(expected.length);
  expect([...cellValues].sort()).toEqual([...expected].sort());
  if (kind === 'grid') expect(cellValues).toEqual(expected);
});

/**
 * Native forms bins from the CPU-filtered rows and only the per-bin values honour GPU range
 * filters: a bin whose first row is hidden by a GPU filter keeps its position in bin order.
 */
it('bin order follows the topology rows when a filter hides a row inside a bin', async () => {
  // Cells A (offset 0), B (1), C (2). Row 0 is in cell A but hidden from values.
  const codes = [0, 1, 2, 1];
  const longitudeOffsets = [0, 1, 0, 2];
  const visible = [0, 1, 1, 1];
  const withTopology = (
    await readCellIndices('grid', {
      codes,
      longitudeOffsets,
      dictionarySize: 3,
      visible,
      topology: [1, 1, 1, 1]
    })
  ).filter(Number.isFinite);
  // Bin order A(row 0), B(row 1), C(row 3); values A=code 2, B=1, C=1 -> indices 0, 1, 1.
  expect(withTopology).toEqual([0, 1, 1]);
  const visibleOnly = (
    await readCellIndices('grid', {codes, longitudeOffsets, dictionarySize: 3, visible})
  ).filter(Number.isFinite);
  // Without the topology mask bins start at their first visible row: B, A, C -> 1, 0, 0.
  expect(visibleOnly).toEqual([1, 0, 0]);
});

it('first-appearance indices work for dictionaries above 16,384 labels', async () => {
  const codes = [19999, 17000, 19999, 40, 17000, 5];
  const longitudeOffsets = [3, 0, 5, 1, 4, 2];
  const cellValues = (
    await readCellIndices('grid', {codes, longitudeOffsets, dictionarySize: 20000})
  ).filter(Number.isFinite);
  const expectedIndices = getFirstAppearanceIndices(
    codes.map((_, row) => row),
    codes
  );
  const rowsByCell = longitudeOffsets
    .map((offset, row) => ({offset, row}))
    .sort((first, second) => first.offset - second.offset);
  expect(cellValues).toEqual(rowsByCell.map(({row}) => expectedIndices.get(codes[row])!));
});
