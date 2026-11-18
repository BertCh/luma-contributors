// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {expect, it, vi} from 'vitest';
import {
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';
import {
  stripWGSLComments,
  WGSL_RESERVED_WORDS
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

import {createRasterContributorFixture} from '../raster-grid/raster-contributor-test-fixture';

it('maps sparse uint32 zones without per-row shader variants and rejects malformed dictionaries at construction', async () => {
  const {GPURasterZonalStatistics} = await import(
    '../../../src/raster-analysis/raster-zonal-statistics/gpu-raster-zonal-statistics'
  );
  const counts = [5, 17].map(width => {
    const {graph, grid, view, band} = createRasterContributorFixture(width);
    const contributor = new GPURasterZonalStatistics({
      grid,
      zones: view('uint32'),
      values: band,
      zoneCapacity: 3,
      zoneDictionary: view('uint32', 3),
      output: {
        cellCounts: view('uint32', 3),
        sourceZoneIds: view('uint32', 3),
        denseZoneRows: view('uint32', 3),
        status: {invalidCount: view('uint32', 1)}
      }
    });
    const nodes = contributor.getCommandNodes(graph);
    const sources = getGPUCommandNodeKernelSources(nodes);
    for (const {id, source} of sources) {
      const stripped = stripWGSLComments(source);
      const tokens = new Set(stripped.match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? []);
      expect(
        [...WGSL_RESERVED_WORDS].filter(word => tokens.has(word)),
        id
      ).toEqual([]);
    }
    expect(nodes.some(node => node.id.endsWith('-dictionary-map'))).toBe(true);
    return {
      shaders: new Set(sources.map(source => source.source)).size,
      pipelines: getGPUCommandNodePipelineKeys(nodes).size
    };
  });
  expect(counts[1]).toEqual(counts[0]);
  const {grid, view, band} = createRasterContributorFixture();
  expect(
    () =>
      new GPURasterZonalStatistics({
        grid,
        zones: view('uint32'),
        values: band,
        zoneCapacity: 3,
        zoneDictionary: view('uint32', 2),
        output: {
          cellCounts: view('uint32', 3),
          sourceZoneIds: view('uint32', 3),
          denseZoneRows: view('uint32', 3),
          status: {invalidCount: view('uint32', 1)}
        }
      })
  ).toThrow(/zoneDictionary.*zoneCapacity/);
});
it('preflights nested sparse-zone analysis before mapping allocations and accepts packed mapping outputs', async () => {
  const {GPURasterZonalStatistics} = await import(
    '../../../src/raster-analysis/raster-zonal-statistics/gpu-raster-zonal-statistics'
  );
  for (const badUsage of [false, true]) {
    const {graph, grid, view, band} = createRasterContributorFixture();
    const packed = graph.createTransientBuffer({
      id: 'packed-zone-output',
      byteLength: 768,
      usage: Buffer.STORAGE
    });
    const outputView = (byteOffset: number, length: number) =>
      graph.createDataView(packed, {format: 'uint32', byteOffset, length});
    const dictionaryBuffer = graph.createTransientBuffer({
      id: 'dictionary',
      byteLength: 12,
      usage: badUsage ? Buffer.COPY_DST : Buffer.STORAGE
    });
    const contributor = new GPURasterZonalStatistics({
      grid,
      zones: view('uint32'),
      values: band,
      zoneCapacity: 3,
      zoneDictionary: graph.createDataView(dictionaryBuffer, {format: 'uint32', length: 3}),
      output: {
        cellCounts: outputView(0, 3),
        sourceZoneIds: outputView(12, 3),
        denseZoneRows: outputView(256, 3),
        status: {invalidCount: outputView(512, 1)}
      }
    });
    if (badUsage) {
      const resources = vi.spyOn(graph, 'createTransientBuffer');
      const nodes = vi.spyOn(graph, 'addComputePass');
      expect(() => contributor.getCommandNodes(graph)).toThrow(/STORAGE/);
      expect(resources).not.toHaveBeenCalled();
      expect(nodes).not.toHaveBeenCalled();
    } else {
      expect(() => graph.add(contributor)).not.toThrow();
    }
  }
});
