// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {makeGPURasterMetadata} from '../../src/raster-analysis/raster-grid/gpu-raster-grid';
import {GPURasterWatershedSegmentation} from '../../src/raster-analysis/gpu-raster-watershed-segmentation';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../utils/require-test-device';

it('GPURasterWatershedSegmentation publishes iterationLimitReached beside converged', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const width = 6;
  const count = width;
  const grid = makeGPURasterMetadata({width, height: 1});
  const elevationBuffer = createInputBuffer(device, Float32Array.of(0, 1, 2, 2, 1, 0));
  const markerBuffer = createInputBuffer(device, Uint32Array.of(1, 0, 0, 0, 0, 2));
  const cases = [
    {method: 'fifo', maximumSteps: undefined, converged: 1},
    {method: 'fifo', maximumSteps: 2, converged: 0},
    {method: 'minimum-spanning-forest', maximumSteps: undefined, converged: 1}
  ] as const;
  for (const testCase of cases) {
    const graph = new GPUCommandGraph(device, {id: 'watershed-status'});
    const labels = createOutputBuffer(device, count);
    const converged = createOutputBuffer(device, 1);
    const iterationLimitReached = createOutputBuffer(device, 1);
    graph.add(
      new GPURasterWatershedSegmentation({
        grid,
        input: {
          id: 'surface',
          format: 'float32',
          storage: {
            kind: 'buffer',
            values: importGraphBuffer(graph, 'surface', elevationBuffer, 'float32', count)
          }
        },
        markers: importGraphBuffer(graph, 'markers', markerBuffer, 'uint32', count),
        method: testCase.method,
        maximumSteps: testCase.maximumSteps,
        output: {
          labels: importGraphBuffer(graph, 'labels', labels, 'uint32', count),
          status: {
            converged: importGraphBuffer(graph, 'converged', converged, 'uint32', 1),
            iterationLimitReached: importGraphBuffer(
              graph,
              'iteration-limit-reached',
              iterationLimitReached,
              'uint32',
              1
            )
          }
        }
      })
    );
    const compiled = graph.compile();
    submitGraph(device, compiled, undefined);
    expect({
      ...testCase,
      converged: (await readUint32(converged, 1))[0],
      iterationLimitReached: (await readUint32(iterationLimitReached, 1))[0]
    }).toEqual({...testCase, iterationLimitReached: 1 - testCase.converged});
    if (testCase.converged) {
      // Plateau ownership differs between methods; a converged run labels every cell.
      const labelled = Array.from(await readUint32(labels, count));
      expect(labelled.every(label => label === 1 || label === 2)).toBe(true);
    }
    compiled.destroy();
    [labels, converged, iterationLimitReached].forEach(buffer => buffer.destroy());
  }
  [elevationBuffer, markerBuffer].forEach(buffer => buffer.destroy());
});
