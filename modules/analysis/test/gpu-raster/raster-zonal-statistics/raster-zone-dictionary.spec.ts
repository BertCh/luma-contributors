// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it('publishes source IDs 100 and 900 separately from denseZoneRows 0 and 1', async () => {
  const {GPURasterZonalStatistics} = await import(
    '../../../src/raster-analysis/raster-zonal-statistics/gpu-raster-zonal-statistics'
  );
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const dictionary = createInputBuffer(device, Uint32Array.of(100, 900));
  const zones = createInputBuffer(device, Uint32Array.of(900, 100));
  const values = createInputBuffer(device, Float32Array.of(1, 2));
  const sourceZoneIds = createOutputBuffer(device, 2);
  const denseZoneRows = createOutputBuffer(device, 2);
  const sums = createOutputBuffer(device, 2);
  const invalidCount = createOutputBuffer(device, 1);
  graph.add(
    new GPURasterZonalStatistics({
      grid: {width: 2, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
      zones: importGraphBuffer(graph, 'zones', zones, 'uint32', 2),
      values: {
        id: 'values',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: importGraphBuffer(graph, 'values', values, 'float32', 2)
        }
      },
      zoneCapacity: 2,
      zoneDictionary: importGraphBuffer(graph, 'dictionary', dictionary, 'uint32', 2),
      output: {
        sums: importGraphBuffer(graph, 'sums', sums, 'float32', 2),
        sourceZoneIds: importGraphBuffer(graph, 'source-ids', sourceZoneIds, 'uint32', 2),
        denseZoneRows: importGraphBuffer(graph, 'dense-rows', denseZoneRows, 'uint32', 2),
        status: {invalidCount: importGraphBuffer(graph, 'invalid', invalidCount, 'uint32', 1)}
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await readUint32(sourceZoneIds, 2)).toEqual([100, 900]);
    expect(await readUint32(denseZoneRows, 2)).toEqual([0, 1]);
    expect(await readFloat32(sums, 2)).toEqual([2, 1]);
    expect(await readUint32(invalidCount, 1)).toEqual([0]);
  } finally {
    compiled.destroy();
    for (const buffer of [
      dictionary,
      zones,
      values,
      sourceZoneIds,
      denseZoneRows,
      sums,
      invalidCount
    ])
      buffer.destroy();
  }
});

it('aggregates sparse numeric zone IDs and publishes the exact ID-to-dense mapping', async () => {
  const {GPURasterZonalStatistics} = await import(
    '../../../src/raster-analysis/raster-zonal-statistics/gpu-raster-zonal-statistics'
  );
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const ignoredZone of [0xffffffff, 99]) {
    for (const invalidDictionary of [false, true]) {
      const dictionaryValues = Uint32Array.from(
        ignoredZone === 99 ? [7, 4000000000, 0xffffffff] : [7, 2147483648, 4000000000]
      );
      if (invalidDictionary) dictionaryValues.reverse();
      const sourceZones = Uint32Array.of(
        4000000000,
        7,
        4000000000,
        ignoredZone === 99 ? 0xffffffff : 2147483648,
        99,
        0xffffffff
      );
      const zones = createInputBuffer(device, sourceZones);
      const dictionary = createInputBuffer(device, dictionaryValues);
      const values = createInputBuffer(device, Float32Array.of(1, 2, 3, 4, 5, 6));
      const cellCounts = createOutputBuffer(device, 3);
      const sums = createOutputBuffer(device, 3);
      const sourceZoneIds = createOutputBuffer(device, 3);
      const denseZoneRows = createOutputBuffer(device, 3);
      const invalidCount = createOutputBuffer(device, 1);
      const graph = new GPUCommandGraph(device);
      graph.add(
        new GPURasterZonalStatistics({
          grid: {width: 6, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
          zones: importGraphBuffer(graph, 'zones', zones, 'uint32', 6),
          values: {
            id: 'values',
            format: 'float32',
            storage: {
              kind: 'buffer',
              values: importGraphBuffer(graph, 'values', values, 'float32', 6)
            }
          },
          zoneCapacity: 3,
          ignoredZone,
          zoneDictionary: importGraphBuffer(graph, 'dictionary', dictionary, 'uint32', 3),
          output: {
            cellCounts: importGraphBuffer(graph, 'counts', cellCounts, 'uint32', 3),
            sums: importGraphBuffer(graph, 'sums', sums, 'float32', 3),
            sourceZoneIds: importGraphBuffer(graph, 'source-ids', sourceZoneIds, 'uint32', 3),
            denseZoneRows: importGraphBuffer(graph, 'dense-rows', denseZoneRows, 'uint32', 3),
            status: {invalidCount: importGraphBuffer(graph, 'invalid', invalidCount, 'uint32', 1)}
          }
        })
      );
      const compiled = graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        expect(await readUint32(sourceZoneIds, 3)).toEqual(Array.from(dictionaryValues));
        expect(await readUint32(denseZoneRows, 3)).toEqual(
          invalidDictionary ? [0xffffffff, 0xffffffff, 0xffffffff] : [0, 1, 2]
        );
        expect(await readUint32(cellCounts, 3)).toEqual(
          invalidDictionary ? [0, 0, 0] : ignoredZone === 99 ? [1, 2, 2] : [1, 1, 2]
        );
        expect(await readFloat32(sums, 3)).toEqual(
          invalidDictionary ? [0, 0, 0] : ignoredZone === 99 ? [2, 4, 10] : [2, 4, 4]
        );
        expect(await readUint32(invalidCount, 1)).toEqual([
          invalidDictionary ? 2 : ignoredZone === 99 ? 0 : 1
        ]);
      } finally {
        compiled.destroy();
        for (const buffer of [
          zones,
          dictionary,
          values,
          cellCounts,
          sums,
          sourceZoneIds,
          denseZoneRows,
          invalidCount
        ])
          buffer.destroy();
      }
    }
  }
});
