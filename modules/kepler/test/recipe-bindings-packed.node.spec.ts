// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createRequire} from 'node:module';
import {expect, it, vi} from 'vitest';
import {GPUCommandGraph, GraphVectorView, createTransientView} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../analysis/src/testing/gpu-contributor-test-utils';
import {addKeplerAnalysisRecipe} from '../src/analysis/recipe-bindings';
import * as analysisBindings from '../src/analysis/analysis-bindings';

const foreign = createRequire(import.meta.url)('@luma.gl/graph/gpu-core') as {
  GraphVectorView: typeof GraphVectorView;
};

it.each([
  1, 2
])('excess-risk unwraps precisely one foreign vector chunk (chunks=%s)', chunkCount => {
  const device = createNullWebGPUDevice();
  const resolve = vi.spyOn(analysisBindings, 'addKeplerExcessRisk');
  try {
    const graph = new GPUCommandGraph(device);
    const data = Array.from({length: chunkCount}, (_, index) =>
      createTransientView(graph, `events-${index}`, 'float32', 2)
    );
    const events = new foreign.GraphVectorView({
      id: 'events',
      name: 'events',
      format: 'float32',
      length: chunkCount * 2,
      valueLength: chunkCount * 2,
      stride: 1,
      byteStride: 4,
      rowByteLength: 4,
      data
    });
    expect(events).not.toBeInstanceOf(GraphVectorView);
    const source = {datasetId: 'source', revision: 1};
    const run = () =>
      addKeplerAnalysisRecipe(
        graph,
        {
          schemaVersion: 1,
          algorithmVersion: 1,
          id: 'risk',
          source,
          population: 'all',
          backendPolicy: 'require-gpu',
          operation: {kind: 'excess-risk', events: 'events', populations: 'populations'}
        },
        {
          source,
          columns: {
            events,
            populations: createTransientView(graph, 'populations', 'float32', chunkCount * 2)
          }
        }
      );
    if (chunkCount === 2) {
      expect(run).toThrow('This analysis requires explicit packing before execution');
      expect(resolve).not.toHaveBeenCalled();
    } else {
      const result = run();
      expect(resolve.mock.calls[0][1].events).toBe(data[0]);
      result.destroyParameters();
    }
  } finally {
    resolve.mockRestore();
    device.destroy();
  }
});
