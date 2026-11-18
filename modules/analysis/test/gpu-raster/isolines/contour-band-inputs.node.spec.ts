// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Texture} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import type {GPURasterBand} from '@luma.gl/graph/gpu-raster';
import {expect, it, vi} from 'vitest';
import {GPUIsolines} from '../../../src/raster-analysis/isolines/gpu-isolines';
import {GPUIsobands} from '../../../src/raster-analysis/isolines/gpu-isobands';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import {
  createNullWebGPUDevice,
  getParsedKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import {buildCellBandPieces, fanTriangulate, getPolygonArea} from './isobands-oracle';

function createFixture(kind: 'isolines' | 'isobands', width = 3) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const grid = makeGPURasterMetadata({width, height: width});
  let sequence = 0;
  const view = <Format extends 'float32' | 'uint32' | 'float32x4'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${sequence++}`, format, length);
  const band: GPURasterBand = {
    id: 'input',
    format: 'uint32',
    storage: {kind: 'buffer', values: view('uint32', width * width + 3)},
    validity: view('uint32', width * width + 2),
    noDataValue: 99,
    scale: 2,
    offset: 10
  };
  const textureBand = (size = width, owner = graph): GPURasterBand => ({
    id: 'texture',
    format: 'uint32',
    storage: {
      kind: 'texture',
      view: owner.createTextureView(
        owner.createTransientTexture({
          id: `texture-${sequence++}`,
          format: 'r32uint',
          width: size,
          height: size,
          usage: Texture.SAMPLE
        })
      )
    }
  });
  const parameters = view('float32', 12);
  const breaks = view('float32', 2);
  const segments = view('float32x4', 1);
  const classes = view('uint32', width * width + 2);
  const count = view('uint32', 1);
  const overflow = view('uint32', 1);
  const create = (
    values: GPURasterBand | GraphDataView<'float32'> = band,
    size = width,
    alias?: GraphDataView<'uint32'>
  ) => {
    const shared = {
      id: 'contour',
      grid: {...grid, width: size, height: size},
      values,
      parameters,
      mask: undefined,
      noDataValue: 10
    };
    const classOutput =
      alias ??
      graph.createDataView(classes.buffer, {
        format: 'uint32',
        length: kind === 'isolines' ? 1 : Math.min(size * size, classes.length)
      });
    return kind === 'isolines'
      ? new GPUIsolines({
          ...shared,
          levels: breaks,
          output: {segments, segmentLevels: classOutput, status: {count, overflow}}
        })
      : new GPUIsobands({...shared, breaks, output: {bandClasses: classOutput}});
  };
  return {device, graph, grid, band, view, textureBand, create};
}

it('accounts for degenerate triangle records at a calibrated break equality', () => {
  const geometry = {
    cellColumn: 0,
    cellRow: 0,
    minX: 0,
    minY: 0,
    cellWidth: 1,
    cellHeight: 1
  };
  // The sole valid cell of the headless band fixture has these calibrated corners, CCW.
  const corners = [10, 12, 12, 10];
  const boundaries = [
    [null, 11],
    [11, 12],
    [12, null]
  ] as const;
  const triangles = boundaries.map(([lower, upper]) =>
    buildCellBandPieces(corners, geometry, lower, upper).flatMap(fanTriangulate)
  );
  expect(triangles.map(records => records.length)).toEqual([2, 2, 2]);
  expect(triangles.map(records => records.map(getPolygonArea))).toEqual([
    [0.25, 0.25],
    [0.25, 0.25],
    [0, 0]
  ]);
  expect(triangles[2].map(record => record.map(point => [point.x, point.y]))).toEqual([
    [
      [1.5, 0.5],
      [1.5, 0.5],
      [1.5, 1.5]
    ],
    [
      [1.5, 0.5],
      [1.5, 1.5],
      [1.5, 1.5]
    ]
  ]);
});

it.each(
  (['isolines', 'isobands'] as const).flatMap(kind =>
    (['buffer', 'texture'] as const).map(storageKind => ({kind, storageKind}))
  )
)('$kind accepts published $storageKind bands through public conversions', ({
  kind,
  storageKind
}) => {
  const fixture = createFixture(kind);
  try {
    const source = storageKind === 'buffer' ? fixture.band : fixture.textureBand();
    const nodes = fixture.create(source).getCommandNodes(fixture.graph);
    const kernels = getParsedKernelSources(nodes);
    expect([...kernels.keys()].some(name => name.includes('elevation'))).toBe(true);
    for (const {source} of kernels.configurations)
      expect(getStorageBindingCount(source)).toBeLessThanOrEqual(8);
    for (const node of nodes) fixture.graph.add(node);
  } finally {
    fixture.device.destroy();
  }
});

it.each([
  'isolines',
  'isobands'
] as const)('%s validates metadata and all original band aliases at construction', kind => {
  const fixture = createFixture(kind);
  try {
    const {band, create, view} = fixture;
    expect(() => create({...band, scale: Infinity})).toThrow(/contour values.scale/);
    expect(() => create({...band, validity: view('uint32', 8)})).toThrow(
      /values.validity.*at least 9/
    );
    expect(() => create({...band, noDataValue: -1})).toThrow(/values.noDataValue/);
    expect(() => create(band, 4)).toThrow(/values/);
    expect(() => create(band, 3, band.validity)).toThrow(/outputs must not share buffers/);
    if (band.storage.kind === 'buffer') {
      const aliasedValues = fixture.graph.createDataView(band.storage.values.buffer, {
        format: 'uint32',
        length: 9
      });
      expect(() => create(band, 3, aliasedValues)).toThrow(/outputs must not share buffers/);
    }
    const legacy = view('float32', 9);
    expect(() => create(legacy)).not.toThrow();
  } finally {
    fixture.device.destroy();
  }
});

it.each([
  'isolines',
  'isobands'
] as const)('%s rejects foreign textures before allocation and permits a corrected retry', kind => {
  const fixture = createFixture(kind);
  const foreignGraph = new GPUCommandGraph(fixture.device);
  const invalid = fixture.create(fixture.textureBand(3, foreignGraph));
  const corrected = fixture.create();
  const allocations = vi.spyOn(fixture.graph, 'createTransientBuffer');
  try {
    expect(() => invalid.getCommandNodes(fixture.graph)).toThrow(/target graph/);
    const failedAllocations = allocations.mock.calls.length;
    expect(() => corrected.getCommandNodes(fixture.graph)).not.toThrow();
    expect(failedAllocations).toBe(0);
  } finally {
    allocations.mockRestore();
    fixture.device.destroy();
  }
});

it.each(
  (['isolines', 'isobands'] as const).flatMap(kind => [64, 80].map(limit => ({kind, limit})))
)('$kind checks full-grid conversion scratch at the $limit-byte limit before allocation', ({
  kind,
  limit
}) => {
  const fixture = createFixture(kind, 5);
  Object.defineProperty(fixture.device, 'limits', {
    value: {...fixture.device.limits, maxStorageBufferBindingSize: limit}
  });
  const invalid = fixture.create(fixture.textureBand());
  const corrected = fixture.create(fixture.textureBand(4), 4);
  const allocations = vi.spyOn(fixture.graph, 'createTransientBuffer');
  try {
    expect(() => invalid.getCommandNodes(fixture.graph)).toThrow(
      new RegExp(`^contour.*100 bytes.*limit ${limit}`)
    );
    const failedAllocations = allocations.mock.calls.length;
    expect(() => corrected.getCommandNodes(fixture.graph)).not.toThrow();
    expect(failedAllocations).toBe(0);
  } finally {
    allocations.mockRestore();
    fixture.device.destroy();
  }
});
