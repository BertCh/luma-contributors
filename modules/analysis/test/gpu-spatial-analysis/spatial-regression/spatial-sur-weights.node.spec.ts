// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUSeeminglyUnrelatedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-seemingly-unrelated-regression';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

it('spatial SUR validates weights once with row tiles instead of once per equation in one thread', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const rowCount = 65536;
    let serial = 0;
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `spatial-sur-${serial++}`, format, length);
    const sources = getGPUCommandNodeKernelSources(
      new GPUSeeminglyUnrelatedRegression({
        rowCount,
        equationCount: 2,
        predictorCount: 1,
        predictors: view('float32', 2 * rowCount),
        responses: view('float32', 2 * rowCount),
        spatial: {
          weights: {
            offsets: view('uint32', rowCount + 1),
            neighbors: view('uint32', rowCount * 2),
            weights: view('float32', rowCount * 2)
          },
          output: {parameters: view('float32', 2), status: view('uint32', 2)}
        },
        output: {
          coefficients: view('float32', 4),
          covariance: view('float32', 16),
          residualCovariance: view('float32', 4),
          residuals: view('float32', rowCount * 2),
          status: view('uint32', 1)
        }
      }).getCommandNodes(graph)
    );
    const validation = sources.filter(kernel => kernel.id.includes('-validate-weights-'));
    expect(validation).toHaveLength(2);
    for (const kernel of validation)
      expect(auditWGSLModule(kernel.source, kernel.constants)).toEqual([]);
    const parameters = sources.filter(kernel => /-parameter-\d+$/.test(kernel.id));
    expect(parameters).toHaveLength(2);
    for (const kernel of parameters) {
      expect(kernel.source).toContain('invalidWeights[invalidWeightsOffset]');
      expect(kernel.source).not.toContain('var row');
    }
  } finally {
    device.destroy();
  }
});
