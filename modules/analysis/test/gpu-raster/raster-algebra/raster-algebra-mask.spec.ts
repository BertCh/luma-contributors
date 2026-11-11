// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  getGPURasterArithmeticParameterValues,
  getGPURasterCellStatisticsParameterValues,
  getGPUWeightedOverlayParameterLength,
  getGPUWeightedOverlayParameterValues,
  GPURasterArithmetic,
  GPURasterCellStatistics,
  GPUWeightedOverlay
} from '../../../src/raster-analysis/raster-algebra/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

// Cell 0 kept, 1 masked, 2 kept, 3 masked, 4 kept (also nodata), 5 kept.
const MASK = new Uint32Array([1, 0, 7, 0, 1, 1]);
const CELL_COUNT = MASK.length;

it('GPURasterArithmetic treats a zero mask cell as no-data and leaves kept cells unchanged', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const a = new Float32Array([1, 2, 3, 4, NaN, 6]);
  const b = new Float32Array([10, 20, 30, 40, 50, 60]);
  const parameters = new GPUParameterBuffer(device, {
    id: 'mask-arithmetic-parameters',
    format: 'float32',
    length: 8
  });
  parameters.write(getGPURasterArithmeticParameterValues({operation: 'add'}));
  const maskedOut = createOutputBuffer(device, CELL_COUNT);
  const plainOut = createOutputBuffer(device, CELL_COUNT);
  const graph = new GPUCommandGraph(device, {id: 'mask-arithmetic-graph'});
  const aView = importGraphBuffer(graph, 'a', createInputBuffer(device, a), 'float32', CELL_COUNT);
  const bView = importGraphBuffer(graph, 'b', createInputBuffer(device, b), 'float32', CELL_COUNT);
  graph.add(
    new GPURasterArithmetic({
      id: 'masked',
      cellCount: CELL_COUNT,
      a: aView,
      b: bView,
      mask: importGraphBuffer(graph, 'mask', createInputBuffer(device, MASK), 'uint32', CELL_COUNT),
      parameters: parameters.importToGraph(graph),
      output: {values: importGraphBuffer(graph, 'masked-out', maskedOut, 'float32', CELL_COUNT)}
    })
  );
  graph.add(
    new GPURasterArithmetic({
      id: 'plain',
      cellCount: CELL_COUNT,
      a: aView,
      b: bView,
      parameters: parameters.importToGraph(graph, 'plain-parameters'),
      output: {values: importGraphBuffer(graph, 'plain-out', plainOut, 'float32', CELL_COUNT)}
    })
  );
  submitGraph(device, graph.compile(), undefined);
  const masked = await readFloat32(maskedOut, CELL_COUNT);
  const plain = await readFloat32(plainOut, CELL_COUNT);
  expect(Array.from(plain)).toEqual([11, 22, 33, 44, NaN, 66]);
  expect(Array.from(masked)).toEqual([11, NaN, 33, NaN, NaN, 66]);
});

it('GPURasterCellStatistics treats a zero mask cell as no-data across all layers', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const layerCount = 3;
  const stack = new Float32Array([
    ...[1, 2, 3, 4, 5, 6],
    ...[2, 3, 4, 5, 6, 7],
    ...[3, 4, 5, 6, 7, 8]
  ]);
  const parameters = new GPUParameterBuffer(device, {
    id: 'mask-statistics-parameters',
    format: 'float32',
    length: 4
  });
  parameters.write(getGPURasterCellStatisticsParameterValues());
  const meanOut = createOutputBuffer(device, CELL_COUNT);
  const majorityOut = createOutputBuffer(device, CELL_COUNT);
  const varietyOut = createOutputBuffer(device, CELL_COUNT);
  const countsOut = createOutputBuffer(device, CELL_COUNT);
  const graph = new GPUCommandGraph(device, {id: 'mask-statistics-graph'});
  graph.add(
    new GPURasterCellStatistics({
      id: 'masked-statistics',
      stack: importGraphBuffer(graph, 'stack', createInputBuffer(device, stack), 'float32', 18),
      layerCount,
      cellCount: CELL_COUNT,
      mask: importGraphBuffer(graph, 'mask', createInputBuffer(device, MASK), 'uint32', CELL_COUNT),
      parameters: parameters.importToGraph(graph),
      output: {
        mean: importGraphBuffer(graph, 'mean', meanOut, 'float32', CELL_COUNT),
        majority: importGraphBuffer(graph, 'majority', majorityOut, 'float32', CELL_COUNT),
        variety: importGraphBuffer(graph, 'variety', varietyOut, 'uint32', CELL_COUNT),
        counts: importGraphBuffer(graph, 'counts', countsOut, 'uint32', CELL_COUNT)
      }
    })
  );
  submitGraph(device, graph.compile(), undefined);
  expect(Array.from(await readFloat32(meanOut, CELL_COUNT))).toEqual([2, NaN, 4, NaN, 6, 7]);
  expect(Array.from(await readFloat32(majorityOut, CELL_COUNT))).toEqual([1, NaN, 3, NaN, 5, 6]);
  expect(await readUint32(varietyOut, CELL_COUNT)).toEqual([3, 0, 3, 0, 3, 3]);
  expect(await readUint32(countsOut, CELL_COUNT)).toEqual([3, 0, 3, 0, 3, 3]);
});

it('GPUWeightedOverlay marks zero mask cells no-data in the score and the validity output', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const layerCount = 2;
  const stack = new Float32Array([...[0, 10, 20, 30, 40, 50], ...[50, 40, 30, 20, 10, 0]]);
  const parameters = new GPUParameterBuffer(device, {
    id: 'mask-overlay-parameters',
    format: 'float32',
    length: getGPUWeightedOverlayParameterLength(layerCount)
  });
  parameters.write(
    getGPUWeightedOverlayParameterValues({
      layers: [
        {weight: 1, inputMinimum: 0, inputMaximum: 50},
        {weight: 1, inputMinimum: 0, inputMaximum: 50}
      ]
    })
  );
  const scoreOut = createOutputBuffer(device, CELL_COUNT);
  const validityOut = createOutputBuffer(device, CELL_COUNT);
  const rangeOut = createOutputBuffer(device, 2);
  const graph = new GPUCommandGraph(device, {id: 'mask-overlay-graph'});
  graph.add(
    new GPUWeightedOverlay({
      id: 'masked-overlay',
      stack: importGraphBuffer(graph, 'stack', createInputBuffer(device, stack), 'float32', 12),
      layerCount,
      cellCount: CELL_COUNT,
      mask: importGraphBuffer(graph, 'mask', createInputBuffer(device, MASK), 'uint32', CELL_COUNT),
      parameters: parameters.importToGraph(graph),
      output: {
        score: importGraphBuffer(graph, 'score', scoreOut, 'float32', CELL_COUNT),
        validity: importGraphBuffer(graph, 'validity', validityOut, 'uint32', CELL_COUNT),
        scoreRange: importGraphBuffer(graph, 'range', rangeOut, 'float32', 2)
      }
    })
  );
  submitGraph(device, graph.compile(), undefined);
  // Every kept cell scores 1 (the two layers sum to 50 per cell), within float32 rounding.
  const score = Array.from(await readFloat32(scoreOut, CELL_COUNT));
  expect(score.map(value => (Number.isNaN(value) ? 'nan' : Math.round(value * 1e5) / 1e5))).toEqual(
    [1, 'nan', 1, 'nan', 1, 1]
  );
  expect(await readUint32(validityOut, CELL_COUNT)).toEqual([1, 0, 1, 0, 1, 1]);
  const range = Array.from(await readFloat32(rangeOut, 2));
  expect(range[0]).toBeCloseTo(1, 5);
  expect(range[1]).toBeCloseTo(1, 5);
});
