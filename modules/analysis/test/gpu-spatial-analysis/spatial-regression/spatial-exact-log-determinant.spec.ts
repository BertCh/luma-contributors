// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUSpatialErrorMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-maximum-likelihood';
import {GPUSpatialLagMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-lag-maximum-likelihood';
import {getSpatialWeightsEigenvalues} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-exact-log-determinant';
import {
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LOG_DETERMINANT,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PARAMETER,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_TRUNCATION_BOUND
} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-maximum-likelihood-parameters';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {SPATIAL_MAXIMUM_LIKELIHOOD_SCENES} from './spatial-maximum-likelihood-reference';

type Scene = {
  weights: {offsets: Uint32Array; neighbors: Uint32Array; weights: Float32Array};
  predictors: Float32Array;
  response: Float32Array;
  predictorCount: number;
};

async function runMaximumLikelihood(
  device: Device,
  kind: 'lag' | 'error',
  scene: Scene,
  exact: boolean
) {
  const rows = scene.response.length;
  const k = scene.predictorCount;
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const graph = new GPUCommandGraph(device, {id: `${kind}-exact-graph`});
  const weights = scene.weights;
  const tableBuffer = track(createOutputBuffer(device, (k + 2) * 4));
  const summaryBuffer = track(
    createOutputBuffer(device, GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH)
  );
  const statusBuffer = track(createOutputBuffer(device, 1));
  const eigenvalues = exact
    ? importGraphBuffer(
        graph,
        'eigenvalues',
        track(
          createInputBuffer(
            device,
            Float32Array.from(
              getSpatialWeightsEigenvalues(
                weights.offsets,
                weights.neighbors,
                weights.weights,
                rows
              )
            )
          )
        ),
        'float32',
        rows
      )
    : undefined;
  const props = {
    id: `${kind}-exact`,
    weights: {
      offsets: importGraphBuffer(
        graph,
        'offsets',
        track(createInputBuffer(device, weights.offsets)),
        'uint32',
        weights.offsets.length
      ),
      neighbors: importGraphBuffer(
        graph,
        'neighbors',
        track(createInputBuffer(device, weights.neighbors)),
        'uint32',
        weights.neighbors.length
      ),
      weights: importGraphBuffer(
        graph,
        'weights',
        track(createInputBuffer(device, weights.weights)),
        'float32',
        weights.weights.length
      )
    },
    predictors: importGraphBuffer(
      graph,
      'predictors',
      track(createInputBuffer(device, scene.predictors)),
      'float32',
      rows * k
    ),
    response: importGraphBuffer(
      graph,
      'response',
      track(createInputBuffer(device, scene.response)),
      'float32',
      rows
    ),
    predictorCount: k,
    logDeterminant: exact ? ('exact' as const) : undefined,
    eigenvalues,
    output: {
      table: importGraphBuffer(graph, 'table', tableBuffer, 'float32', (k + 2) * 4),
      summary: importGraphBuffer(
        graph,
        'summary',
        summaryBuffer,
        'float32',
        GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH
      ),
      status: {code: importGraphBuffer(graph, 'status', statusBuffer, 'uint32', 1)}
    }
  };
  graph.add(
    kind === 'lag'
      ? new GPUSpatialLagMaximumLikelihood(props)
      : new GPUSpatialErrorMaximumLikelihood(props)
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    status: (await readUint32(statusBuffer, 1))[0],
    table: Float32Array.from(await readFloat32(tableBuffer, (k + 2) * 4)),
    summary: Float32Array.from(
      await readFloat32(summaryBuffer, GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH)
    )
  };
  compiled.destroy();
  for (const buffer of buffers) buffer.destroy();
  return result;
}

for (const [kind, sceneName] of [
  ['lag', 'lag_12_21'],
  ['lag', 'knn_60_13'],
  ['error', 'err_lattice_400_31']
] as const) {
  it(`${kind} maximum likelihood with an exact log-determinant matches the series on ${sceneName}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const scene = SPATIAL_MAXIMUM_LIKELIHOOD_SCENES[sceneName]() as Scene;
    // knn weights are not symmetric-pattern: the exact path must refuse them up front.
    if (sceneName === 'knn_60_13') {
      expect(() =>
        getSpatialWeightsEigenvalues(
          scene.weights.offsets,
          scene.weights.neighbors,
          scene.weights.weights,
          scene.response.length
        )
      ).toThrow();
      return;
    }
    const series = await runMaximumLikelihood(device, kind, scene, false);
    const exact = await runMaximumLikelihood(device, kind, scene, true);
    expect(series.status).toBe(0);
    expect(exact.status).toBe(0);
    const parameter = GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PARAMETER;
    expect(Math.abs(exact.summary[parameter] - series.summary[parameter])).toBeLessThan(2e-3);
    expect(
      Math.abs(
        exact.summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LOG_DETERMINANT] -
          series.summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LOG_DETERMINANT]
      )
    ).toBeLessThan(1e-2);
    expect(exact.summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_TRUNCATION_BOUND]).toBe(0);
    for (let row = 0; row < scene.predictorCount + 2; row++) {
      expect(Math.abs(exact.table[row * 4] - series.table[row * 4])).toBeLessThan(
        2e-3 * Math.max(1, Math.abs(series.table[row * 4]))
      );
    }
  });
}
