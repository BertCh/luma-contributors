// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Texture} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import type {GPURasterBand} from '@luma.gl/graph/gpu-raster';
import {expect, it, vi} from 'vitest';
import {
  GPURasterCellStatistics,
  type GPURasterCellStatisticsProps
} from '../../../src/raster-analysis/raster-algebra/gpu-raster-cell-statistics';
import {
  GPUWeightedOverlay,
  type GPUWeightedOverlayProps
} from '../../../src/raster-analysis/raster-algebra/gpu-weighted-overlay';
import {
  createNullWebGPUDevice,
  getParsedKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';

function createFixture(cellCount = 9) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const grid = {
    width: cellCount,
    height: 1,
    affine: [1, 0, 0, 0, 1, 0] as const,
    pixelInterpretation: 'area' as const
  };
  let sequence = 0;
  const view = <Format extends 'float32' | 'uint32' | 'sint32'>(
    format: Format,
    length = cellCount
  ) => createTransientView(graph, `view-${sequence++}`, format, length);
  const band: GPURasterBand = {
    id: 'band',
    format: 'uint32',
    storage: {kind: 'buffer', values: view('uint32', cellCount + 3)},
    validity: view('uint32', cellCount + 2),
    scale: 2,
    offset: 10,
    noDataValue: 99
  };
  const statistics = (
    stack: readonly GPURasterBand[] = [band, band],
    overrides: Partial<GPURasterCellStatisticsProps> = {}
  ) =>
    new GPURasterCellStatistics({
      id: 'band-statistics',
      grid,
      cellCount,
      layerCount: stack.length,
      stack,
      parameters: view('float32', 4),
      output: {
        mean: view('float32'),
        counts: view('uint32'),
        minimum: view('float32'),
        majority: view('float32')
      },
      ...overrides
    });
  const overlay = (
    stack: readonly GPURasterBand[] = [band, band],
    overrides: Partial<GPUWeightedOverlayProps> = {}
  ) =>
    new GPUWeightedOverlay({
      id: 'band-overlay',
      grid,
      cellCount,
      layerCount: stack.length,
      stack,
      parameters: view('float32', 8 + 8 * stack.length),
      output: {score: view('float32'), validity: view('uint32'), scoreRange: view('float32', 2)},
      ...overrides
    });
  return {device, graph, grid, view, band, statistics, overlay};
}

it.each([
  'buffer',
  'texture'
] as const)('accepts %s band stacks and keeps core binding limits', storageKind => {
  const fixture = createFixture();
  try {
    const {graph, grid, band, view, statistics, overlay} = fixture;
    const second = {
      ...band,
      id: 'second',
      format: 'sint32',
      noDataValue: -99,
      storage:
        storageKind === 'buffer'
          ? {kind: 'buffer', values: view('sint32')}
          : {
              kind: 'texture',
              view: graph.createTextureView(
                graph.createTransientTexture({
                  id: 'signed-texture',
                  format: 'r32sint',
                  width: grid.width,
                  height: grid.height,
                  usage: Texture.SAMPLE
                })
              )
            }
    };
    for (const create of [statistics, overlay]) {
      const contributor = create([band, second as GPURasterBand]);
      const nodes = contributor.getCommandNodes(graph);
      const sources = getParsedKernelSources(nodes);
      expect([...sources.values()].some(source => source.includes('sourceValidity'))).toBe(true);
      expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
      for (const source of sources.values())
        expect(getStorageBindingCount(source)).toBeLessThanOrEqual(8);
      for (const node of nodes) graph.add(node);
    }
  } finally {
    fixture.device.destroy();
  }
});

it('validates stack shape, band metadata and aliases at construction', () => {
  const {device, graph, band, view, statistics, overlay} = createFixture();
  try {
    for (const create of [statistics, overlay]) {
      expect(() => create([band, band], {grid: undefined})).toThrow(/stack requires grid/);
      expect(() => create([], {layerCount: 1})).toThrow(/stack must contain exactly 1 bands/);
      expect(() => create([view('float32') as unknown as GPURasterBand])).toThrow(
        /stack\[0\] must be a raster band/
      );
      expect(() => create([band], {layerCount: 2})).toThrow(/stack must contain exactly 2 bands/);
      expect(() => create([band, band], {cellCount: 8})).toThrow(
        /stack grid must contain exactly 8 cells/
      );
      expect(() => create([band, {...band, validity: view('uint32', 8)}])).toThrow(
        /stack\[1\].validity.*at least 9/
      );
      expect(() => create([band, {...band, scale: Infinity}])).toThrow(/stack\[1\].scale/);
      expect(() => create([band, {...band, noDataValue: -1}])).toThrow(/stack\[1\].noDataValue/);
    }
    const aliased = graph.createDataView(band.validity!.buffer, {format: 'float32', length: 9});
    expect(() => statistics(undefined, {output: {mean: aliased}})).toThrow(
      /outputs must not share buffers with inputs/
    );
    expect(() => overlay(undefined, {output: {score: aliased}})).toThrow(
      /outputs must not share buffers with inputs/
    );
  } finally {
    device.destroy();
  }
});

it('keeps empty legacy stacks on the existing path without packing scratch', () => {
  const fixture = createFixture();
  try {
    for (const create of [fixture.statistics, fixture.overlay]) {
      const contributor = create([], {
        stack: fixture.view('float32', 0),
        cellCount: 0,
        layerCount: 1,
        parameters: fixture.view('float32', 16),
        grid: undefined
      });
      const allocations = vi.spyOn(fixture.graph, 'createTransientBuffer');
      const nodes = contributor.getCommandNodes(fixture.graph);
      // Overlay's existing range reduction owns two uint32 words; no band stack is allocated.
      expect(allocations.mock.results.every(result => result.value.byteLength === 8)).toBe(true);
      expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
      expect(getParsedKernelSources(nodes).size).toBeGreaterThan(0);
      allocations.mockRestore();
    }
  } finally {
    fixture.device.destroy();
  }
});

it.each([
  255, 256
])('checks packed stack scratch against the %i-byte binding limit before allocation', limit => {
  const {device, graph, statistics, overlay} = createFixture(32);
  Object.defineProperty(device, 'limits', {
    value: {...device.limits, maxStorageBufferBindingSize: limit}
  });
  try {
    for (const create of [statistics, overlay]) {
      const contributor = create();
      const allocations = vi.spyOn(graph, 'createTransientBuffer');
      if (limit < 256) {
        expect(() => contributor.getCommandNodes(graph)).toThrow(
          /stack needs 256 bytes in one binding \(limit 255\)/
        );
        expect(allocations).not.toHaveBeenCalled();
      } else {
        expect(() => contributor.getCommandNodes(graph)).not.toThrow();
      }
      allocations.mockRestore();
    }
  } finally {
    device.destroy();
  }
});

it('rejects foreign texture storage before creating stack nodes', () => {
  const fixture = createFixture();
  try {
    const foreign = new GPUCommandGraph(fixture.device);
    const band: GPURasterBand = {
      ...fixture.band,
      validity: undefined,
      storage: {
        kind: 'texture',
        view: foreign.createTextureView(
          foreign.createTransientTexture({
            id: 'foreign',
            format: 'r32uint',
            width: 9,
            height: 1,
            usage: Texture.SAMPLE
          })
        )
      }
    };
    for (const create of [fixture.statistics, fixture.overlay]) {
      const contributor = create([fixture.band, band]);
      const allocations = vi.spyOn(fixture.graph, 'createTransientBuffer');
      expect(() => contributor.getCommandNodes(fixture.graph)).toThrow(/target graph/);
      expect(allocations).not.toHaveBeenCalled();
      allocations.mockRestore();
    }
  } finally {
    fixture.device.destroy();
  }
});

it.each([1, 16, 64])('bounds node count for %i published bands', layerCount => {
  const fixture = createFixture(1);
  try {
    const stack = Array.from({length: layerCount}, () => fixture.band);
    const contributors = [
      fixture.statistics(stack),
      ...(layerCount <= 16 ? [fixture.overlay(stack)] : [])
    ];
    for (const contributor of contributors) {
      const nodes = contributor.getCommandNodes(fixture.graph);
      expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
      const sources = getParsedKernelSources(nodes);
      expect(sources.size).toBeGreaterThan(layerCount);
      for (const source of sources.values())
        expect(getStorageBindingCount(source)).toBeLessThanOrEqual(8);
      for (const node of nodes) fixture.graph.add(node);
    }
  } finally {
    fixture.device.destroy();
  }
});
