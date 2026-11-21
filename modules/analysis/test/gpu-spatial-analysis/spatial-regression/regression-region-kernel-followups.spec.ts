// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createGTWRIndexedSummaryNodes} from '../../../src/gpu-spatial-analysis/spatial-regression/gtwr-indexed-summary';
import {GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS} from '../../../src/gpu-spatial-analysis/spatial-regression/geographically-weighted-regression-parameters';
import {createGridIndexNodes} from '../../../src/gpu-spatial-analysis/spatial-regression/gwr-grid-index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

it('indexed GTWR summary distinguishes clean, singular and excluded rows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: ReturnType<typeof createInputBuffer>[] = [];
  const input = <Format extends 'float32' | 'uint32'>(
    name: string,
    format: Format,
    data: Float32Array | Uint32Array
  ) => {
    const buffer = createInputBuffer(device, data);
    buffers.push(buffer);
    return importGraphBuffer(graph, name, buffer, format, data.length);
  };
  const summaryBuffer = createOutputBuffer(device, 12);
  buffers.push(summaryBuffer);
  const summary = importGraphBuffer(graph, 'summary', summaryBuffer, 'float32', 12);
  const nodes = createGTWRIndexedSummaryNodes(graph, 'mixed-status', {
    positions: createTransientView(graph, 'positions', 'float32x2', 4),
    times: createTransientView(graph, 'times', 'float32', 4),
    temporalContext: {
      unit: 'second',
      epoch: 0n,
      encoding: 'float32-relative',
      axis: {kind: 'irregular'}
    },
    predictors: createTransientView(graph, 'predictors', 'float32', 4),
    predictorCount: 1,
    parameters: createTransientView(graph, 'parameters', 'float32', 4),
    response: input('response', 'float32', Float32Array.of(10, 14, 20, 30)),
    output: {
      coefficients: createTransientView(graph, 'coefficients', 'float32', 8),
      fitted: input('fitted', 'float32', Float32Array.of(9, 12, NaN, NaN)),
      residuals: input('residuals', 'float32', Float32Array.of(1, 2, NaN, NaN)),
      hatDiagonal: input('hat', 'float32', Float32Array.of(0.25, 0.5, NaN, NaN)),
      localStatus: input(
        'status',
        'uint32',
        Uint32Array.of(
          GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS.OK,
          GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS.OK,
          GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS.SINGULAR,
          GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS.EXCLUDED
        )
      ),
      summary
    }
  });
  for (const node of nodes) graph.add(node);
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const values = await readFloat32(summaryBuffer, 12);
    // Two usable fits: RSS=(10-9)^2+(14-12)^2=5, trace=3/4; three included, one singular.
    for (const [slot, expected] of [
      [0, 5],
      [1, 0.75],
      // Usable response mean is 12, TSS=(10-12)^2+(14-12)^2=8: R²=1-5/8.
      [3, 0.375],
      [4, 2],
      [6, 1],
      [11, 3]
    ]) {
      expect(Number.isFinite(values[slot])).toBe(true);
      expect(values[slot]).toBe(expected);
    }
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});

for (const fixture of [
  {label: 'finite rows', positions: [3, -2, -5, 7], rowCount: 2, extent: [-5, -2, 3, 7]},
  {label: 'nonfinite rows', positions: [NaN, 1, 2, NaN], rowCount: 2, extent: [NaN, NaN, NaN, NaN]},
  // The physical buffer has one point, but the imported view and normalized view have zero rows.
  {label: 'zero rows', positions: [0, 0], rowCount: 0, extent: [NaN, NaN, NaN, NaN]},
  {
    label: 'positive extreme',
    positions: [3.1e38, -2, 3.2e38, 7],
    rowCount: 2,
    extent: [3.1e38, -2, 3.2e38, 7]
  },
  {
    label: 'negative extreme',
    positions: [-3.1e38, -2, -3.2e38, 7],
    rowCount: 2,
    extent: [-3.2e38, -2, -3.1e38, 7]
  }
]) {
  it(`GWR publishes ordered extent components and preserves empty extent: ${fixture.label}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const graph = new GPUCommandGraph(device);
    const positionData = Float32Array.from(fixture.positions);
    const positionBuffer = createInputBuffer(device, positionData);
    const positions = importGraphBuffer(
      graph,
      'positions',
      positionBuffer,
      'float32x2',
      fixture.rowCount
    );
    const grid = createGridIndexNodes(graph, {
      id: 'ordered-extent',
      operation: 'extent-fixture',
      positions,
      rowCount: fixture.rowCount,
      gridSize: [1, 1]
    });
    const outputBuffer = createOutputBuffer(device, 4);
    const output = importGraphBuffer(graph, 'extent-output', outputBuffer, 'float32', 4);
    for (const node of grid.nodes) graph.add(node);
    graph.add(
      createWGSLKernelNode(graph, {
        id: 'copy-extent',
        operation: 'extent-fixture',
        invocationCount: 4,
        bindings: [
          {name: 'sortedIds', view: grid.sortedIds, type: 'u32', access: 'read'},
          {name: 'output', view: output, type: 'f32', access: 'read_write'}
        ],
        body: `output[outputOffset + index] = bitcast<f32>(sortedIds[sortedIdsOffset + ${fixture.rowCount}u + index]);`
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const extent = await readFloat32(outputBuffer, 4);
      expect(extent).toHaveLength(4);
      for (const [component, expected] of fixture.extent.entries()) {
        if (Number.isNaN(expected)) {
          expect(Number.isNaN(extent[component])).toBe(true);
        } else {
          expect(Number.isFinite(extent[component])).toBe(true);
          // The extrema of these two-point fixtures are literal input endpoints, rounded to f32.
          expect(extent[component]).toBe(Math.fround(expected));
        }
      }
    } finally {
      compiled.destroy();
      positionBuffer.destroy();
      outputBuffer.destroy();
    }
  });
}
