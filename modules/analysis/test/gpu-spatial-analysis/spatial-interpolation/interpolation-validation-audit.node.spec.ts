// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  getGPULocalKrigingParameterValues,
  getGPUSpaceTimeKrigingParameterValues
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/prediction-covariance';
import {expect, it} from 'vitest';
import {
  GPULocalKriging,
  type GPULocalKrigingProps
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-local-kriging';
import {GPULocalRBF} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-local-rbf';
import {GPUNaturalNeighbor} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-natural-neighbor';
import {GPUConditionalGaussianSimulation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-conditional-gaussian-simulation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('PTS-137 construction diagnostics name the contributor and rejected fields/limits', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const positions = createTransientView(graph, 'positions', 'float32x2', 3);
  const observations = createTransientView(graph, 'observations', 'float32x4', 3);
  const queries = createTransientView(graph, 'queries', 'float32x2', 1);
  const values = createTransientView(graph, 'values', 'float32', 3);
  const output = {
    values: createTransientView(graph, 'output', 'float32x4', 1),
    status: createTransientView(graph, 'status', 'uint32', 1)
  };
  const props: GPULocalKrigingProps = {
    id: 'validation-local',
    positions,
    observations,
    supports: queries,
    mode: 'ordinary',
    covariance: {model: 'exponential', range: 3, sill: 1},
    output
  };
  expect(() => new GPULocalKriging({...props, driftCount: 9, mode: 'external-drift'})).toThrow(
    /^validation-local driftCount.*\[0, 8\]/
  );
  expect(
    () => new GPULocalKriging({...props, covariance: {...props.covariance!, range: -1}})
  ).toThrow(/^validation-local covariance.range.*positive/);
  expect(
    () =>
      new GPULocalKriging({
        ...props,
        observations: createTransientView(graph, 'short', 'float32x4', 1)
      })
  ).toThrow(/^validation-local .*observations.length.*positions.length/);
  expect(
    () =>
      new GPULocalRBF({
        id: 'validation-rbf',
        positions,
        values,
        queries,
        degree: -1,
        kernel: 'quintic',
        output
      })
  ).toThrow(/^validation-rbf kernel and degree.*quintic = 2/);
  expect(
    () =>
      new GPUNaturalNeighbor({
        id: 'validation-natural',
        positions,
        values,
        queries,
        maximumVertexCount: 65,
        output
      })
  ).toThrow(/^validation-natural maximumVertexCount.*\[3, 64\]/);
  expect(
    () =>
      new GPUConditionalGaussianSimulation({
        id: 'validation-simulation',
        positions,
        observations,
        queries,
        covariance: props.covariance!,
        seed: -1,
        replicateCount: 1,
        output: {
          values: createTransientView(graph, 'draws', 'float32', 1),
          moments: createTransientView(graph, 'moments', 'float32x2', 1),
          status: output.status
        }
      })
  ).toThrow(/^validation-simulation .*seed must be uint32/);
});

it('S5 parameter packers identify their calling function and the specific covariance field', () => {
  expect(() =>
    getGPULocalKrigingParameterValues({covariance: {model: 'exponential', range: 0, sill: 1}})
  ).toThrow(/^getGPULocalKrigingParameterValues covariance.range/);
  expect(() =>
    getGPULocalKrigingParameterValues({
      variableCount: 2,
      coregionalizationModels: [
        {covariance: {model: 'exponential', range: 0, sill: 1}, loadings: [1, 0, 0, 1]}
      ]
    })
  ).toThrow(/^getGPULocalKrigingParameterValues coregionalizationModels\[0\].covariance.range/);
  expect(() =>
    getGPUSpaceTimeKrigingParameterValues({
      covariance: {
        type: 'metric',
        joint: {model: 'exponential', range: 0, nugget: 0, sill: 1},
        anisotropy: 1
      }
    })
  ).toThrow(/^getGPUSpaceTimeKrigingParameterValues covariance.joint.range/);
  expect(() =>
    getGPUSpaceTimeKrigingParameterValues({
      covariance: {
        type: 'separable',
        spatial: {model: 'exponential', range: 1, nugget: 0, sill: 1},
        temporal: {model: 'exponential', range: 1, nugget: 0, sill: 1},
        sill: 0
      },
      searchAnisotropy: 1
    })
  ).toThrow(/^getGPUSpaceTimeKrigingParameterValues covariance.sill/);
});
