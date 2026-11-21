// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUSeeminglyUnrelatedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-seemingly-unrelated-regression';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

it('t1 maximum SUR GLS shader refines coefficients and covariance within shader limits', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `sur-refine-${serial++}`, format, length);
    const model = new GPUSeeminglyUnrelatedRegression({
      rowCount: 8,
      equationCount: 4,
      predictorCount: 3,
      predictors: view('float32', 96),
      responses: view('float32', 32),
      output: {
        coefficients: view('float32', 16),
        covariance: view('float32', 256),
        residualCovariance: view('float32', 16),
        residuals: view('float32', 32),
        status: view('uint32', 1)
      }
    });
    const solve = getGPUCommandNodeKernelSources(model.getCommandNodes(graph)).find(kernel =>
      kernel.id.endsWith('-solve-gls')
    )!;
    expect(solve.source).toContain(
      'refineSURSolution_16(&originalNormal, &originalRight, &normal, &scales, &solution)'
    );
    expect(solve.source).toContain(
      'refineSURSolution_16(&originalNormal, &covarianceRight, &normal, &scales, &transform)'
    );
    expect(auditWGSLModule(solve.source, solve.constants)).toEqual([]);
  } finally {
    device.destroy();
  }
});
