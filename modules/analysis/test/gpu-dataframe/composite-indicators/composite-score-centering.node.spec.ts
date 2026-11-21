// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {GPUCompositeScore} from '../../../src/dataframe-analysis/composite-indicators/gpu-composite-score';
import {GPU_COMPOSITE_SCORE_PARAMETER_LENGTH} from '../../../src/dataframe-analysis/composite-indicators/composite-score-parameters';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

function withKernels(action: (kernels: ReturnType<typeof getGPUCommandNodeKernelSources>) => void) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = (length: number) =>
    createTransientView(graph, `center-${serial++}`, 'float32', length);
  try {
    const nodes = new GPUCompositeScore({
      indicators: view(4),
      indicatorCount: 2,
      parameters: view(GPU_COMPOSITE_SCORE_PARAMETER_LENGTH),
      enablePrincipalComponent: true,
      enableRank: true,
      output: {
        score: view(2),
        scaled: view(4),
        columnStatistics: view(8),
        loadings: view(2),
        principalComponentSummary: view(3)
      }
    }).getCommandNodes(graph);
    action(getGPUCommandNodeKernelSources(nodes));
  } finally {
    device.destroy();
  }
}

it('B1 moments and score retain a separate reference and mean offset', () => {
  withKernels(kernels => {
    const score = kernels.find(kernel => kernel.source.includes('var weightedSum'))!;
    expect(score.source).toContain('let meanOffset = bitcast<f32>(rowValid[');
    expect(score.source).toContain(
      'getCompositeDeviation(value, reference, meanOffset) / deviation'
    );
    expect(score.source).toContain('let magnitude = bitcast<f32>(bits & 0x7fffffffu)');
    const moments = kernels.find(kernel => kernel.source.includes('let leftMean'))!;
    expect(moments.source).toContain('let leftReference = bitcast<f32>(rowValid[');
  });
});

it('B2 centered tile totals are integer reductions rather than repeated large float additions', () => {
  withKernels(kernels => {
    const centered = kernels.find(kernel => kernel.id.includes('tile-centered-sums'))!;
    expect(centered.source).toContain('sum = addFixedPoint(sum, quantizeFixedPoint(');
    for (const kernel of kernels) {
      expect(() => new WgslReflect(kernel.source)).not.toThrow();
      expect(auditWGSLModule(kernel.source, kernel.constants)).toEqual([]);
    }
  });
});
