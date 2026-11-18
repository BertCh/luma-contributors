// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {
  makeGPURasterBand,
  makeGPURasterMetadata
} from '../../src/raster-analysis/raster-grid/index';
import {GPURasterEmbeddingKMeans} from '../../src/raster-analysis/raster-embeddings/gpu-raster-embedding-kmeans';
import {GPURasterEmbeddingNearestPixels} from '../../src/raster-analysis/raster-embeddings/gpu-raster-embedding-nearest-pixels';
import {createNullWebGPUDevice} from '../utils/gpu-contributor-test-utils';
import {createSpectralIndex} from './raster-spectral-indices-test-utils';

it('wraps only the grid prefix of padded raster outputs, preserving offsets and metadata', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const grid = makeGPURasterMetadata({width: 3, height: 1});
    const backing = createTransientView(graph, 'backing', 'float32', 20);
    const values = graph.createDataView(backing.buffer, {
      format: 'float32',
      byteOffset: 8,
      length: 5
    });
    const validity = graph.createDataView(backing.buffer, {
      format: 'uint32',
      byteOffset: 40,
      length: 7
    });
    const band = makeGPURasterBand(values, grid, {
      id: 'pooled',
      validity,
      scale: 2,
      offset: 3,
      noDataValue: -99,
      verticalDatum: 'ellipsoidal'
    });
    expect(band.storage.values.length).toBe(3);
    expect(band.storage.values.buffer).toBe(backing.buffer);
    expect(band.storage.values.byteOffset).toBe(8);
    expect(band.validity?.length).toBe(3);
    expect(band.validity?.buffer).toBe(backing.buffer);
    expect(band.validity?.byteOffset).toBe(40);
    expect(band).toMatchObject({
      id: 'pooled',
      scale: 2,
      offset: 3,
      noDataValue: -99,
      verticalDatum: 'ellipsoidal'
    });
    expect(values.length).toBe(5);
    expect(validity.length).toBe(7);
    expect(() => makeGPURasterBand(values, {...grid, width: 6})).toThrow(/grid pixel/);
    expect(() =>
      makeGPURasterBand(values, grid, {
        validity: graph.createDataView(backing.buffer, {format: 'uint32', length: 2})
      })
    ).toThrow(/grid pixel/);
  } finally {
    device.destroy();
  }
});

it.each([
  0, 1, 2, 3, 4, 5
])('raster producer %s can be inspected before graph.add schedules it once', producerIndex => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let sequence = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `view-${sequence++}`, format, length);
  try {
    const band = makeGPURasterBand(
      view('float32', 2),
      makeGPURasterMetadata({width: 2, height: 1})
    );
    const props = {
      width: 2,
      height: 1,
      output: view('float32', 2),
      outputValidity: view('uint32', 2)
    };
    const bands = {
      green: band,
      nearInfrared: band,
      shortwaveInfrared: band,
      shortwaveInfrared2: band,
      red: band,
      blue: band
    };
    const producers = [
      ...(['ndwi-gao', 'nbr', 'evi', 'savi'] as const).map(index =>
        createSpectralIndex(index, props, bands)
      ),
      new GPURasterEmbeddingKMeans({
        pixelCount: 2,
        dimensions: 2,
        vectors: view('float32', 4),
        clusterCount: 1,
        maximumIterations: 1,
        output: {
          labels: view('uint32', 2),
          centroids: view('float32', 2),
          counts: view('uint32', 1)
        }
      }),
      new GPURasterEmbeddingNearestPixels({
        pixelCount: 2,
        dimensions: 2,
        vectors: view('float32', 4),
        queryCount: 1,
        queries: view('float32', 2),
        resultCount: 1,
        output: {
          pixelRows: view('uint32', 1),
          scores: view('float32', 1),
          resultCounts: view('uint32', 1)
        }
      })
    ];
    for (const producer of [producers[producerIndex]]) {
      const scheduled: unknown[] = [];
      const addPass = vi.spyOn(graph, 'addComputePass').mockImplementation(node => {
        scheduled.push(node);
        return graph;
      });
      try {
        const nodes = producer.getCommandNodes(graph);
        expect(nodes.length).toBeGreaterThan(0);
        expect(scheduled).toEqual([]);
        expect(producer.getCommandNodes(graph)).toBe(nodes);
        const originalNodes = [...nodes];
        try {
          Array.prototype.pop.call(nodes);
        } catch (error) {
          expect(error).toBeInstanceOf(TypeError);
        }
        expect(producer.getCommandNodes(graph).map(node => node.id)).toEqual(
          originalNodes.map(node => node.id)
        );
        graph.add(producer);
        expect(scheduled).toHaveLength(nodes.length);
        expect(scheduled.map(node => (node as {id: string}).id)).toEqual(
          nodes.map(node => node.id)
        );
        if (producerIndex < 4) expect(nodes).toHaveLength([1, 1, 6, 3][producerIndex]);
      } finally {
        addPass.mockRestore();
      }
    }
  } finally {
    device.destroy();
  }
});
