// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUInstrumentalVariablesCovariance,
  type GPUInstrumentalVariablesCovarianceProps
} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-instrumental-variables-covariance';
import {GPUSpatialTwoStageLeastSquares} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-two-stage-least-squares';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('validates covariance options and named limits at construction, including through spatial 2SLS', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `view-${serial++}`, format, length);
    const output = {covariance: view('float32', 9), status: {code: view('uint32', 1)}};
    const props: GPUInstrumentalVariablesCovarianceProps = {
      id: 'iv-limits',
      design: view('float32', 16),
      coefficientCount: 2,
      instruments: view('float32', 24),
      instrumentCount: 3,
      residuals: view('float32', 8),
      output
    };
    const clusters = view('uint32', 8);
    for (const [overrides, message] of [
      [{method: 'invalid'}, /iv-limits.*method/],
      [{method: 'HAC'}, /iv-limits.*kernel/],
      [{method: 'CLUSTER'}, /iv-limits.*clusters/],
      [{method: 'HC2', clusters}, /iv-limits.*clusters/],
      [{method: 'CLUSTER', clusters, clusterCount: 0}, /iv-limits.*clusterCount/],
      [{method: 'CLUSTER', clusters, clusterCount: 9}, /iv-limits.*clusterCount/],
      [{method: 'CLUSTER', clusters, clusterCount: 1.5}, /iv-limits.*clusterCount/],
      [
        {method: 'CLUSTER', clusters: view('uint32', 7), clusterCount: 2},
        /iv-limits.*clusters.*length/
      ],
      [{coefficientCount: 16}, /iv-limits.*coefficientCount/],
      [{instrumentCount: 17}, /iv-limits.*instrumentCount/]
    ] as const)
      expect(
        () =>
          new GPUInstrumentalVariablesCovariance({
            ...props,
            ...overrides
          } as GPUInstrumentalVariablesCovarianceProps)
      ).toThrow(message);
    for (const method of ['HC2', 'HC3', 'CLUSTER'] as const) {
      const contributor = new GPUInstrumentalVariablesCovariance({
        ...props,
        id: `iv-limits-${method}`,
        method,
        ...(method === 'CLUSTER' ? {clusters, clusterCount: 6} : {})
      });
      const nodes = contributor.getCommandNodes(graph);
      expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
      expect(nodes.some(node => node.id.endsWith('-finish'))).toBe(true);
      const estimator = new GPUSpatialTwoStageLeastSquares({
        id: `spatial-${method}`,
        predictors: view('float32', 8),
        response: view('float32', 8),
        predictorCount: 1,
        weights: {
          offsets: view('uint32', 9),
          neighbors: view('uint32', 8),
          weights: view('float32', 8)
        },
        robust: {method, ...(method === 'CLUSTER' ? {clusters, clusterCount: 6} : {}), output},
        output: {
          table: view('float32', 12),
          summary: view('float32', 16),
          status: {code: view('uint32', 1)}
        }
      });
      expect(estimator.getCommandNodes(graph).length).toBeLessThanOrEqual(
        estimator.estimatedCommandNodeCount
      );
    }
    expect(
      () =>
        new GPUSpatialTwoStageLeastSquares({
          id: 'spatial-limits',
          predictors: view('float32', 8),
          response: view('float32', 8),
          predictorCount: 1,
          weights: {
            offsets: view('uint32', 9),
            neighbors: view('uint32', 8),
            weights: view('float32', 8)
          },
          robust: {method: 'CLUSTER', clusters, clusterCount: 9, output},
          output: {
            table: view('float32', 12),
            summary: view('float32', 16),
            status: {code: view('uint32', 1)}
          }
        })
    ).toThrow(/spatial-limits.*clusterCount/);
  } finally {
    device.destroy();
  }
});

it('bounds the actual clustered command count at the maximum dense-system and row shape', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const rowCount = 2 ** 24;
    const view = <Format extends 'float32' | 'uint32'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, name, format, length);
    const contributor = new GPUInstrumentalVariablesCovariance({
      design: view('design', 'float32', rowCount * 15),
      coefficientCount: 15,
      instruments: view('instruments', 'float32', rowCount * 16),
      instrumentCount: 16,
      residuals: view('residuals', 'float32', rowCount),
      method: 'CLUSTER',
      clusters: view('clusters', 'uint32', rowCount),
      clusterCount: rowCount,
      output: {
        covariance: view('covariance', 'float32', 256),
        status: {code: view('status', 'uint32', 1)}
      }
    });
    // The null adapter has the same storage-binding cap as a core adapter. Use its largest
    // permitted score binding; no storage is allocated and no shader is executed.
    const maximumRows = Math.floor(device.limits.maxStorageBufferBindingSize / 64);
    const rows = Math.min(rowCount, maximumRows);
    const bounded = new GPUInstrumentalVariablesCovariance({
      ...contributor.props,
      design: view('bounded-design', 'float32', rows * 15),
      instruments: view('bounded-instruments', 'float32', rows * 16),
      residuals: view('bounded-residuals', 'float32', rows),
      clusters: view('bounded-clusters', 'uint32', rows),
      clusterCount: rows
    });
    const nodes = bounded.getCommandNodes(graph);
    expect(nodes.length).toBeLessThanOrEqual(bounded.estimatedCommandNodeCount);
  } finally {
    device.destroy();
  }
});
