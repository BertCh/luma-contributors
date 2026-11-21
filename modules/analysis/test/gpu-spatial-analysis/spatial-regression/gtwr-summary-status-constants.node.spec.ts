// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS} from '../../../src/gpu-spatial-analysis/spatial-regression/geographically-weighted-regression-parameters';
import type {GPUGeographicallyTemporallyWeightedRegressionProps} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-geographically-temporally-weighted-regression';
import {createGTWRIndexedNodes} from '../../../src/gpu-spatial-analysis/spatial-regression/gtwr-indexed-nodes';
import {createGTWRIndexedSummaryNodes} from '../../../src/gpu-spatial-analysis/spatial-regression/gtwr-indexed-summary';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

it('indexed GTWR summary generates its usable and included predicates from model statuses', () => {
  const device = createNullWebGPUDevice();
  // Both builders read the shared status object during generation. Change its values in place
  // for this synchronous test so already-loaded builders see the same codes.
  const modelStatus = GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS;
  const originalStatus = {...modelStatus};
  try {
    Object.assign(modelStatus, {OK: 7, SINGULAR: 11, EXCLUDED: 13});
    const graph = new GPUCommandGraph(device);
    const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, name, format, length);
    const props: GPUGeographicallyTemporallyWeightedRegressionProps = {
      positions: view('positions', 'float32x2', 3),
      times: view('times', 'float32', 3),
      temporalContext: {
        unit: 'second',
        epoch: 0n,
        encoding: 'float32-relative',
        axis: {kind: 'irregular'}
      },
      predictors: view('predictors', 'float32', 3),
      predictorCount: 1,
      response: view('response', 'float32', 3),
      parameters: view('parameters', 'float32', 4),
      output: {
        coefficients: view('coefficients', 'float32', 6),
        fitted: view('fitted', 'float32', 3),
        hatDiagonal: view('hat', 'float32', 3),
        localStatus: view('codes', 'uint32', 3),
        summary: view('summary', 'float32', 12),
        residuals: view('residuals', 'float32', 3)
      }
    };
    const nodes = createGTWRIndexedSummaryNodes(
      graph,
      'summary-status',
      props,
      props.output.residuals!
    );
    const writerKernels = getGPUCommandNodeKernelSources(
      createGTWRIndexedNodes(graph, 'writer-status', props)
    );
    const fit = writerKernels.find(kernel => kernel.id === 'writer-status-fit')!;
    expect(fit.source).toContain('var code = 11u;');
    expect(fit.source).toContain('code = 13u;');
    expect(fit.source).toContain('code = 7u;');
    const kernels = getGPUCommandNodeKernelSources(nodes);
    const anchor = kernels.find(kernel => kernel.id === 'summary-status-response-anchor-bounds')!;
    const variance = kernels.find(kernel => kernel.id === 'summary-status-variance-partial')!;
    const statistics = kernels.find(kernel => kernel.id === 'summary-status-statistics-partial')!;
    expect(anchor.source).toContain('localStatus[localStatusOffset + row] == 7u');
    expect(variance.source).toContain('localStatus[localStatusOffset + row] == 7u');
    expect(statistics.source).toContain('if (code != 13u)');
    expect(statistics.source).toContain('if (code == 7u)');
  } finally {
    Object.assign(modelStatus, originalStatus);
    device.destroy();
  }
});
