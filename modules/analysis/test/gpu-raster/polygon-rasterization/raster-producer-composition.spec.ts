// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPURasterJoin} from '../../../src/raster-analysis/polygon-rasterization';
import {GPURasterZonalStatistics} from '../../../src/raster-analysis/raster-zonal-statistics';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {createPolygons} from './polygon-rasterization-oracle';
import {
  createRasterizationFixture,
  destroyRasterizationFixture,
  makeExtentGrid
} from './polygon-rasterization-fixture';

it('RT-137 overflow suppresses join and zonal outputs and resets on the same graph', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rectangle = (height: number) =>
    createPolygons([
      [
        [
          [
            [0, 0],
            [1, 0],
            [1, height],
            [0, height]
          ]
        ]
      ]
    ]);
  const fixture = createRasterizationFixture(device, {
    polygons: rectangle(4),
    width: 1,
    height: 4,
    extent: [0, 0, 1, 1],
    crossingCapacity: 2
  });
  const {graph} = fixture;
  const owned: ReturnType<typeof createOutputBuffer>[] = [];
  const output = <Format extends 'uint32' | 'float32'>(
    name: string,
    format: Format,
    length: number
  ) => {
    const buffer = createOutputBuffer(device, length);
    owned.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, format, length)};
  };
  const points = createInputBuffer(device, new Float32Array([0.5, 0.5, 0.5, 1.5]));
  const pointValues = createInputBuffer(device, new Float32Array([2, 7]));
  const cellValues = createInputBuffer(device, new Float32Array([3, 5, 7, 11]));
  const counts = output('counts', 'uint32', 1);
  const sums = output('sums', 'float32', 1);
  const pointZones = output('point-zones', 'uint32', 2);
  const incomplete = output('incomplete', 'uint32', 1);
  const cellCounts = output('cell-counts', 'uint32', 1);
  const cellSums = output('cell-sums', 'float32', 1);
  const zonalIncomplete = output('zonal-incomplete', 'uint32', 1);
  const producerOverflow = fixture.contributor.props.output.status.overflow;
  const grid = makeExtentGrid(1, 4, [0, 0, 1, 1]);
  const pointsView = importGraphBuffer(graph, 'points', points, 'float32x2', 2);
  const pointValuesView = importGraphBuffer(graph, 'point-values', pointValues, 'float32', 2);
  const cellValuesView = importGraphBuffer(graph, 'cell-values', cellValues, 'float32', 4);
  graph.add(
    new GPURasterJoin({
      grid,
      parameters: fixture.extentView,
      zones: fixture.zonesView,
      producerOverflow,
      points: pointsView,
      values: pointValuesView,
      zoneCount: 1,
      output: {
        counts: counts.view,
        sums: sums.view,
        pointZones: pointZones.view,
        status: {incomplete: incomplete.view}
      }
    })
  );
  graph.add(
    new GPURasterZonalStatistics({
      grid,
      zones: fixture.zonesView,
      producerOverflow,
      zoneCapacity: 1,
      values: {
        id: 'cell-values',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: cellValuesView
        }
      },
      output: {
        cellCounts: cellCounts.view,
        sums: cellSums.view,
        status: {incomplete: zonalIncomplete.view}
      }
    })
  );
  const countOnly = output('count-only', 'uint32', 1);
  const countOnlyIncomplete = output('count-only-incomplete', 'uint32', 1);
  const pointOnly = output('point-only', 'uint32', 2);
  const pointOnlyIncomplete = output('point-only-incomplete', 'uint32', 1);
  const zonalCountOnly = output('zonal-count-only', 'uint32', 1);
  const zonalCountOnlyIncomplete = output('zonal-count-only-incomplete', 'uint32', 1);
  for (const [id, columns, incompleteOutput] of [
    ['counts-only', {counts: countOnly.view}, countOnlyIncomplete.view],
    ['points-only', {pointZones: pointOnly.view}, pointOnlyIncomplete.view]
  ] as const) {
    graph.add(
      new GPURasterJoin({
        id,
        grid,
        parameters: fixture.extentView,
        zones: fixture.zonesView,
        producerOverflow,
        points: pointsView,
        zoneCount: 1,
        output: {...columns, status: {incomplete: incompleteOutput}}
      })
    );
  }
  graph.add(
    new GPURasterZonalStatistics({
      id: 'zonal-counts-only',
      grid,
      zones: fixture.zonesView,
      producerOverflow,
      zoneCapacity: 1,
      values: {
        id: 'cell-values',
        format: 'float32',
        storage: {kind: 'buffer', values: cellValuesView}
      },
      output: {cellCounts: zonalCountOnly.view, status: {incomplete: zonalCountOnlyIncomplete.view}}
    })
  );
  const atomicCounts = output('atomic-counts', 'uint32', 1);
  const atomicSums = output('atomic-sums', 'float32', 1);
  const atomicIncomplete = output('atomic-incomplete', 'uint32', 1);
  graph.add(
    new GPURasterZonalStatistics({
      id: 'zonal-atomic',
      grid,
      zones: fixture.zonesView,
      producerOverflow,
      zoneCapacity: 1,
      sumOrder: 'atomic',
      values: {
        id: 'cell-values',
        format: 'float32',
        storage: {kind: 'buffer', values: cellValuesView}
      },
      output: {
        cellCounts: atomicCounts.view,
        sums: atomicSums.view,
        status: {incomplete: atomicIncomplete.view}
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(Array.from(await readUint32(incomplete.buffer))).toEqual([1]);
    expect(Array.from(await readUint32(zonalIncomplete.buffer))).toEqual([1]);
    expect(Array.from(await readUint32(counts.buffer))).toEqual([0xffffffff]);
    expect(Array.from(await readUint32(cellCounts.buffer))).toEqual([0xffffffff]);
    expect(Number.isNaN((await readFloat32(sums.buffer))[0])).toBe(true);
    expect(Number.isNaN((await readFloat32(cellSums.buffer))[0])).toBe(true);
    expect(Array.from(await readUint32(pointZones.buffer))).toEqual([0xffffffff, 0xffffffff]);
    for (const result of [countOnly, zonalCountOnly, atomicCounts]) {
      expect(await readUint32(result.buffer)).toEqual([0xffffffff]);
    }
    expect(await readUint32(pointOnly.buffer)).toEqual([0xffffffff, 0xffffffff]);
    expect(Number.isNaN((await readFloat32(atomicSums.buffer))[0])).toBe(true);
    for (const result of [
      countOnlyIncomplete,
      pointOnlyIncomplete,
      zonalCountOnlyIncomplete,
      atomicIncomplete
    ]) {
      expect(await readUint32(result.buffer)).toEqual([1]);
    }
    fixture.polygonBuffers[0].write(rectangle(1).positions);
    submitGraph(device, compiled, undefined);
    // Hand fixture: the smaller rectangle covers just cell 0 and the first point.
    expect(Array.from(await readUint32(counts.buffer))).toEqual([1]);
    expect(Array.from(await readFloat32(sums.buffer))).toEqual([2]);
    expect(Array.from(await readUint32(cellCounts.buffer))).toEqual([1]);
    expect(Array.from(await readFloat32(cellSums.buffer))).toEqual([3]);
    expect(Array.from(await readUint32(pointZones.buffer))).toEqual([0, 0xffffffff]);
    expect(Array.from(await readUint32(incomplete.buffer))).toEqual([0]);
    expect(Array.from(await readUint32(zonalIncomplete.buffer))).toEqual([0]);
    for (const result of [countOnly, zonalCountOnly, atomicCounts]) {
      expect(await readUint32(result.buffer)).toEqual([1]);
    }
    expect(await readUint32(pointOnly.buffer)).toEqual([0, 0xffffffff]);
    expect(await readFloat32(atomicSums.buffer)).toEqual([3]);
    for (const result of [
      countOnlyIncomplete,
      pointOnlyIncomplete,
      zonalCountOnlyIncomplete,
      atomicIncomplete
    ]) {
      expect(await readUint32(result.buffer)).toEqual([0]);
    }
  } finally {
    compiled.destroy();
    for (const buffer of [...owned, points, pointValues, cellValues]) buffer.destroy();
    destroyRasterizationFixture(fixture);
  }
});

it('B4 populated point assignments are suppressed and restored when only the producer guard changes', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const grid = makeExtentGrid(1, 1, [0, 0, 1, 1]);
  const owned = [
    createInputBuffer(device, new Uint32Array([0])),
    createInputBuffer(device, new Uint32Array([0])),
    createInputBuffer(device, new Float32Array([0.5, 0.5])),
    createInputBuffer(device, new Float32Array([0, 0, 1, 1])),
    createOutputBuffer(device, 1),
    createOutputBuffer(device, 1)
  ];
  graph.add(
    new GPURasterJoin({
      grid,
      zoneCount: 1,
      zones: importGraphBuffer(graph, 'populated-zones', owned[0], 'uint32', 1),
      producerOverflow: importGraphBuffer(graph, 'guard', owned[1], 'uint32', 1),
      points: importGraphBuffer(graph, 'points', owned[2], 'float32x2', 1),
      parameters: importGraphBuffer(graph, 'parameters', owned[3], 'float32', 4),
      output: {
        pointZones: importGraphBuffer(graph, 'assignments', owned[4], 'uint32', 1),
        status: {incomplete: importGraphBuffer(graph, 'incomplete', owned[5], 'uint32', 1)}
      }
    })
  );
  const compiled = graph.compile();
  try {
    for (const guard of [0, 1, 0]) {
      owned[1].write(new Uint32Array([guard]));
      submitGraph(device, compiled, undefined);
      expect(await readUint32(owned[4], 1)).toEqual([guard === 0 ? 0 : 0xffffffff]);
      expect(await readUint32(owned[5], 1)).toEqual([guard]);
      expect(await readUint32(owned[0], 1)).toEqual([0]);
    }
  } finally {
    compiled.destroy();
    for (const buffer of owned) buffer.destroy();
  }
});
