// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUSpaceTimeCrossCorrelation} from '../../../src/gpu-spatial-analysis/space-time-statistics/gpu-space-time-cross-correlation';
import {
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {SCALED_QUOTIENT_WGSL} from '../../../src/analysis-core/kernels/scaled-quotient-wgsl';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('normalizes through bounded mantissas instead of a potentially flushed reciprocal', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    recordWGSLKernelNodes(graph);
    const first = createTransientView(graph, 'first', 'float32', 4);
    const second = createTransientView(graph, 'second', 'float32', 4);
    const correlations = createTransientView(graph, 'correlations', 'float32', 1);
    const pValues = createTransientView(graph, 'p-values', 'float32', 1);
    const nodes = new GPUSpaceTimeCrossCorrelation({
      first,
      second,
      cellCount: 1,
      periodCount: 4,
      lags: [0],
      layout: 'cell-major',
      output: {correlations, pValues}
    }).getCommandNodes(graph);
    const record = getWGSLKernelNodeRecord(graph, nodes[0].id)!;
    expect(record.body).not.toMatch(/\/ (?:firstScale|secondScale)\b/);
    expect(record.body).toContain('getScaledQuotient(firstMinimum, firstScale, 1.0, 0)');
    // The node interpreter lacks frexp/ldexp. The headless companion checks closed-form
    // correlations at both extremes; here pin the production helper and every call site.
    expect(record.declarations).toContain(SCALED_QUOTIENT_WGSL);
    expect(record.body.match(/getScaledQuotient\(/g)).toHaveLength(6);
  } finally {
    device.destroy();
  }
});
