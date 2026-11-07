// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUHotSpotAnalysis,
  GPULocalMoran,
  GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH,
  getGPUSpatialAutocorrelationParameterValues
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {addKeplerPointWeights} from '../src/analysis/point-weights';
import {
  buildKeplerStreamedPointWeights,
  importKeplerStreamedPointWeights
} from '../src/analysis/point-weights-streamed';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../analysis/src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../analysis/test/utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';

const settings = {
  kind: 'distance-band' as const,
  longitudeColumn: 'longitude',
  latitudeColumn: 'latitude',
  coordinateSpace: 'longitude-latitude' as const,
  metric: 'great-circle' as const,
  units: 'meters' as const,
  radiusMeters: 3000,
  neighborCapacity: 65536
};

/** Chunked point weights (forced small chunks) give the single-CSR hotspot and Local Moran results. */
test('streamed point weights drive Gi* and Local Moran to the single-CSR results', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rowCount = 200;
  const longitudes = Float32Array.from(
    {length: rowCount},
    (_, index) => -74 + 0.005 * (index % 20) + 0.0003 * Math.sin(index)
  );
  const latitudes = Float32Array.from(
    {length: rowCount},
    (_, index) => 40.7 + 0.005 * Math.floor(index / 20) + 0.0003 * Math.cos(index)
  );
  const values = Float32Array.from(
    {length: rowCount},
    (_, index) => Math.sin(index / 7) + (index % 20) / 20
  );
  const longitudeBuffer = createInputBuffer(device, longitudes);
  const latitudeBuffer = createInputBuffer(device, latitudes);

  const run = async (chunked: boolean) => {
    const graph = new GPUCommandGraph(device, {id: `streamed-consumers-${chunked}`});
    const parameters = new GPUParameterBuffer(device, {
      id: 'parameters',
      format: 'float32',
      length: GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH,
      values: getGPUSpatialAutocorrelationParameterValues({significanceLevel: 0.05})
    });
    const mask = importGraphBuffer(
      graph,
      'mask',
      createInputBuffer(device, new Uint32Array(rowCount).fill(1)),
      'uint32',
      rowCount
    );
    const valuesView = importGraphBuffer(
      graph,
      'values',
      createInputBuffer(device, values),
      'float32',
      rowCount
    );
    let weights: ConstructorParameters<typeof GPUHotSpotAnalysis>[0]['weights'];
    let cleanup: () => void;
    let whole: ReturnType<typeof addKeplerPointWeights> | undefined;
    const wholeGraph = new GPUCommandGraph(device, {id: 'streamed-consumers-whole-build'});
    if (chunked) {
      const streamed = await buildKeplerStreamedPointWeights({
        device,
        rowCount,
        firstCoordinate: longitudeBuffer,
        secondCoordinate: latitudeBuffer,
        settings,
        maximumSlotsPerChunk: 600
      });
      expect(streamed.chunks.length).toBeGreaterThan(1);
      weights = importKeplerStreamedPointWeights(graph, streamed);
      cleanup = () => streamed.destroy();
    } else {
      // The consumer graph imports the CSR the whole build wrote.
      const offsets = createOutputBuffer(device, rowCount + 1);
      const neighbors = createOutputBuffer(device, settings.neighborCapacity);
      const csrWeights = createOutputBuffer(device, settings.neighborCapacity);
      const status = [0, 1, 2].map(() => createOutputBuffer(device, 1));
      whole = addKeplerPointWeights(wholeGraph, {
        source: {datasetId: 'points', revision: 1},
        settings,
        longitude: importGraphBuffer(wholeGraph, 'longitude', longitudeBuffer, 'float32', rowCount),
        latitude: importGraphBuffer(wholeGraph, 'latitude', latitudeBuffer, 'float32', rowCount),
        output: {
          offsets: importGraphBuffer(wholeGraph, 'offsets', offsets, 'uint32', rowCount + 1),
          neighbors: importGraphBuffer(
            wholeGraph,
            'neighbors',
            neighbors,
            'uint32',
            settings.neighborCapacity
          ),
          weights: importGraphBuffer(
            wholeGraph,
            'weights',
            csrWeights,
            'float32',
            settings.neighborCapacity
          ),
          status: {
            overflow: importGraphBuffer(wholeGraph, 'overflow', status[0], 'uint32', 1),
            requiredCount: importGraphBuffer(wholeGraph, 'required', status[1], 'uint32', 1),
            invalidCount: importGraphBuffer(wholeGraph, 'invalid', status[2], 'uint32', 1)
          }
        }
      });
      const buildCompiled = wholeGraph.compile();
      submitGraph(device, buildCompiled, undefined);
      weights = {
        offsets: importGraphBuffer(graph, 'c-offsets', offsets, 'uint32', rowCount + 1),
        neighbors: importGraphBuffer(
          graph,
          'c-neighbors',
          neighbors,
          'uint32',
          settings.neighborCapacity
        ),
        weights: importGraphBuffer(
          graph,
          'c-weights',
          csrWeights,
          'float32',
          settings.neighborCapacity
        ),
        information: whole.weights.information,
        spatialContext: whole.weights.spatialContext
      };
      cleanup = () => {
        buildCompiled.destroy();
        [offsets, neighbors, csrWeights, ...status].forEach(buffer => buffer.destroy());
      };
    }
    const outputs = {
      giZ: createOutputBuffer(device, rowCount),
      giCounts: createOutputBuffer(device, rowCount),
      moranZ: createOutputBuffer(device, rowCount),
      moranI: createOutputBuffer(device, rowCount),
      moranCounts: createOutputBuffer(device, rowCount)
    };
    const parametersView = parameters.importToGraph(graph);
    graph.add(
      new GPUHotSpotAnalysis({
        id: 'gi',
        weights,
        values: valuesView,
        mask,
        parameters: parametersView,
        output: {
          zScores: importGraphBuffer(graph, 'gi-z', outputs.giZ, 'float32', rowCount),
          neighborCounts: importGraphBuffer(graph, 'gi-n', outputs.giCounts, 'uint32', rowCount)
        }
      })
    );
    graph.add(
      new GPULocalMoran({
        id: 'moran',
        weights,
        values: valuesView,
        mask,
        parameters: parametersView,
        output: {
          zScores: importGraphBuffer(graph, 'm-z', outputs.moranZ, 'float32', rowCount),
          localI: importGraphBuffer(graph, 'm-i', outputs.moranI, 'float32', rowCount),
          neighborCounts: importGraphBuffer(graph, 'm-n', outputs.moranCounts, 'uint32', rowCount)
        }
      })
    );
    const compiled = graph.compile();
    submitGraph(device, compiled, undefined);
    const result = {
      giZ: Array.from(await readFloat32(outputs.giZ, rowCount)),
      giCounts: Array.from(await readUint32(outputs.giCounts, rowCount)),
      moranZ: Array.from(await readFloat32(outputs.moranZ, rowCount)),
      moranI: Array.from(await readFloat32(outputs.moranI, rowCount)),
      moranCounts: Array.from(await readUint32(outputs.moranCounts, rowCount))
    };
    compiled.destroy();
    whole?.destroyParameters();
    cleanup();
    return result;
  };

  const single = await run(false);
  const streamedResult = await run(true);
  // The cross join lists every row's own slot, which the consumers skip: counts and statistics match.
  expect(streamedResult.giCounts).toEqual(single.giCounts);
  expect(streamedResult.moranCounts).toEqual(single.moranCounts);
  for (const key of ['giZ', 'moranZ', 'moranI'] as const) {
    expect(streamedResult[key], key).toEqual(single[key]);
  }
  expect(single.giCounts.some(count => count > 0)).toBe(true);
  longitudeBuffer.destroy();
  latitudeBuffer.destroy();
});
