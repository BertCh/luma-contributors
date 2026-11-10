// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {GPU_LONGITUDE_LATITUDE_SPATIAL_CONTEXT} from '../../../src/analysis-core/contracts/spatial-context';
import {GPUContinuousSpatialSample} from '../../../src/gpu-spatial-analysis/point-process/gpu-continuous-spatial-sample';
import {GPUSpatialBalancedSample} from '../../../src/gpu-spatial-analysis/point-process/gpu-spatial-balanced-sample';
import {GPUSpatialCrossValidation} from '../../../src/gpu-spatial-analysis/spatial-validation/gpu-spatial-cross-validation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

describe('point-process and validation spatial context', () => {
  it('rejects longitude-latitude positions in sampling and cross validation', () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const view = <Format extends 'uint32' | 'float32' | 'float32x2' | 'uint32x4'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, name, format, length);
    expect(
      () =>
        new GPUSpatialBalancedSample({
          id: 'balanced',
          spatialContext: GPU_LONGITUDE_LATITUDE_SPATIAL_CONTEXT,
          positions: view('b-positions', 'float32x2', 4),
          frameIds: view('b-frame-ids', 'uint32', 4),
          allocations: [{stratumId: 0, sampleCount: 2}],
          bounds: [0, 0, 1, 1],
          seed: 1,
          output: {
            frameIds: view('b-ids', 'uint32', 2),
            status: view('b-status', 'uint32', 2),
            inclusionProbabilities: view('b-probabilities', 'float32', 2),
            designWeights: view('b-weights', 'float32', 2)
          }
        })
    ).toThrow('balanced does not support longitude-latitude coordinates');
    expect(
      () =>
        new GPUContinuousSpatialSample({
          id: 'continuous',
          spatialContext: GPU_LONGITUDE_LATITUDE_SPATIAL_CONTEXT,
          frame: {type: 'line', positions: view('c-positions', 'float32x2', 3)},
          spacing: 1,
          seed: 1,
          capacity: 4,
          output: {
            positions: view('c-out-positions', 'float32x2', 4),
            designWeights: view('c-weights', 'float32', 4),
            provenance: view('c-provenance', 'uint32x4', 4),
            status: view('c-status', 'uint32x4', 1)
          }
        })
    ).toThrow('continuous does not support longitude-latitude coordinates');
    expect(
      () =>
        new GPUSpatialCrossValidation({
          id: 'validation',
          spatialContext: GPU_LONGITUDE_LATITUDE_SPATIAL_CONTEXT,
          positions: view('v-positions', 'float32x2', 4),
          blocks: view('v-blocks', 'uint32', 4),
          foldCount: 2,
          output: {
            folds: view('v-folds', 'uint32', 4),
            training: view('v-training', 'uint32', 8),
            testing: view('v-testing', 'uint32', 8),
            counts: view('v-counts', 'uint32', 6),
            status: view('v-status', 'uint32', 2)
          }
        })
    ).toThrow('validation does not support longitude-latitude coordinates');
  });

  it('names the limit in sampling validation messages', () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    expect(
      () =>
        new GPUSpatialBalancedSample({
          id: 'balanced',
          positions: createTransientView(graph, 'positions', 'float32x2', 4097),
          frameIds: createTransientView(graph, 'frame-ids', 'uint32', 4097),
          allocations: [{stratumId: 0, sampleCount: 2}],
          bounds: [0, 0, 1, 1],
          seed: 1,
          output: {
            frameIds: createTransientView(graph, 'ids', 'uint32', 2),
            status: createTransientView(graph, 'status', 'uint32', 2),
            inclusionProbabilities: createTransientView(graph, 'probabilities', 'float32', 2),
            designWeights: createTransientView(graph, 'weights', 'float32', 2)
          }
        })
    ).toThrow(/^balanced positions rows/);
  });
});
