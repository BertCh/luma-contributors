// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {NullDevice} from '@luma.gl/core/test-utils';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {
  GPU_PERMUTATION_PARAMETER_LENGTH,
  GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH,
  type GPUHotSpotAnalysisRecipeProps
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {expect, test} from 'vitest';
import {addKeplerHotSpots} from '../src/analysis/analysis-bindings';

const WIDTH = 4;
const HEIGHT = 4;
const provenance = {source: {datasetId: 'lattice', revision: 1}, population: 'all' as const};

function createAnalysis(
  selfWeight: number | undefined,
  statistic?: 'localG' | 'localGStar' | 'localMoran'
): {graph: GPUCommandGraph; analysis: GPUHotSpotAnalysisRecipeProps} {
  const device = new NullDevice({});
  Object.defineProperty(device, 'type', {value: 'webgpu'});
  Object.defineProperty(device, 'limits', {
    value: {
      ...device.limits,
      maxComputeWorkgroupsPerDimension: 65535,
      maxStorageBufferBindingSize: 134217728,
      maxStorageBuffersPerShaderStage: 8,
      maxComputeWorkgroupStorageSize: 16384,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupSizeY: 256,
      maxComputeWorkgroupSizeZ: 64
    }
  });
  const graph = new GPUCommandGraph(device, {id: 'kepler-hot-spots-self-weight'});
  const analysis: GPUHotSpotAnalysisRecipeProps = {
    id: 'kepler-hot-spots',
    source: {
      kind: 'lattice',
      values: createTransientView(graph, 'values', 'float32', WIDTH * HEIGHT),
      width: WIDTH,
      height: HEIGHT,
      neighborCapacity: WIDTH * HEIGHT * 8
    },
    parameters: new GPUParameterBuffer(device, {
      id: 'parameters',
      format: 'float32',
      length: GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
    }),
    ...(selfWeight === undefined ? {} : {selfWeight}),
    permutation: {
      parameters: new GPUParameterBuffer(device, {
        id: 'permutation-parameters',
        format: 'uint32',
        length: GPU_PERMUTATION_PARAMETER_LENGTH
      }),
      maximumPermutations: 99,
      ...(statistic ? {statistic} : {})
    }
  };
  return {graph, analysis};
}

// The recipe's permutation confirmation has a fixed focal weight (Gi*: 1, Gi: 0). Kepler hosts
// accept any non-negative self weight, so the binding rejects the combinations the recipe cannot
// confirm in host terms, before any graph work.
test('addKeplerHotSpots rejects a permutation request whose selfWeight the test cannot reproduce', () => {
  for (const [selfWeight, statistic, required] of [
    [0.5, undefined, 1],
    [2, 'localGStar', 1],
    [1, 'localG', 0],
    [0, 'localGStar', 1]
  ] as const) {
    const {graph, analysis} = createAnalysis(selfWeight, statistic);
    expect(() => addKeplerHotSpots(graph, {...provenance, analysis})).toThrow(
      new RegExp(
        `^kepler-hot-spots analysis\\.selfWeight must be ${required} when analysis\\.permutation uses '${statistic ?? 'localGStar'}', not ${selfWeight};`
      )
    );
  }
});

test('addKeplerHotSpots forwards self weights the permutation statistic reproduces', () => {
  for (const [selfWeight, statistic] of [
    [undefined, undefined],
    [1, undefined],
    [0, undefined],
    [1, 'localGStar'],
    [0, 'localG'],
    [0.5, 'localMoran']
  ] as const) {
    const {graph, analysis} = createAnalysis(selfWeight, statistic);
    const result = addKeplerHotSpots(graph, {...provenance, analysis});
    expect(result.operation).toBe('hot-spots');
  }
});
