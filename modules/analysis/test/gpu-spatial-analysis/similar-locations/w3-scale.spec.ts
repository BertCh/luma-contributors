// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  GPUSimilarLocations,
  getGPUSimilarLocationsParameterValues
} from '../../../src/gpu-spatial-analysis/similar-locations';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

it('preserves z-score distances and ordering when units scale by two to minus eighty', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const standardization of ['z-score'] as const) {
    for (const scale of [1, 2 ** -80, 2 ** 127]) {
      const graph = new GPUCommandGraph(device);
      const values = createInputBuffer(
        device,
        scale === 2 ** 127
          ? Float32Array.of(-scale, scale, 0)
          : Float32Array.of(0, 2 * scale, scale)
      );
      const selection = createInputBuffer(device, Uint32Array.of(1, 0, 0));
      const parameters = createInputBuffer(
        device,
        getGPUSimilarLocationsParameterValues({resultCount: 2}, 1)
      );
      const ranks = createOutputBuffer(device, 3);
      const distances = createOutputBuffer(device, 3);
      const top = createOutputBuffer(device, 2);
      const count = createOutputBuffer(device, 1);
      graph.add(
        new GPUSimilarLocations({
          values: importGraphBuffer(graph, 'values', values, 'float32', 3),
          columnCount: 1,
          referenceMask: importGraphBuffer(graph, 'selection', selection, 'uint32', 3),
          parameters: importGraphBuffer(graph, 'parameters', parameters, 'float32', 5),
          standardization,
          resultCapacity: 2,
          output: {
            ranks: importGraphBuffer(graph, 'ranks', ranks, 'uint32', 3),
            distances: importGraphBuffer(graph, 'distances', distances, 'float32', 3),
            topIds: importGraphBuffer(graph, 'top', top, 'uint32', 2),
            status: {count: importGraphBuffer(graph, 'count', count, 'uint32', 1)}
          }
        })
      );
      const compiled = graph.compile();
      submitGraph(device, compiled, undefined);
      expect(Array.from(await readUint32(top, 2)), `${standardization}, scale=${scale}`).toEqual([
        2, 1
      ]);
      const result = await readFloat32(distances, 3);
      // Population variance of [0, 2d, d] is 2d²/3; distances from zero are sqrt(3/2), sqrt(6).
      for (const [row, expected] of [
        [2, 1.2247449],
        [1, 2.4494897]
      ]) {
        expect(Number.isFinite(result[row])).toBe(true);
        expect(result[row]).toBeCloseTo(expected, 5);
      }
      compiled.destroy();
      for (const buffer of [values, selection, parameters, ranks, distances, top, count])
        buffer.destroy();
    }
  }
});
