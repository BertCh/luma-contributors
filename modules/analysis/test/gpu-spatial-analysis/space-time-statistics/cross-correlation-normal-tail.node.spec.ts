// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {GPUSpaceTimeCrossCorrelation} from '../../../src/gpu-spatial-analysis/space-time-statistics/gpu-space-time-cross-correlation';
import {ERFC_VALUE_WGSL} from '../../../src/analysis-core/kernels/wgsl-common';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

it('uses the shared complementary normal tail with SciPy central and tail expectations', () => {
  // /usr/bin/python3: scipy.special.erfc([0,.5,1,2,4,8]). The Fisher test uses erfc(z/sqrt(2)),
  // already the two-sided normal tail; a second factor of two would be incorrect.
  const expected = [
    1, 0.4795001221869535, 0.15729920705028516, 0.004677734981047266, 1.541725790028002e-8,
    1.1224297172982928e-29
  ];
  const values = Float32Array.of(0, 0.5, 1, 2, 4, 8),
    actual = new Float32Array(6);
  const source = `${ERFC_VALUE_WGSL}
@group(0) @binding(0) var<storage, read> values: array<f32>;
@group(0) @binding(1) var<storage, read_write> actual: array<f32>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) index: vec3u) {
  actual[index.x] = erfcValue(values[index.x]);
}`;
  new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups('main', [6, 1, 1], {
    0: {0: values, 1: actual}
  });
  expected.forEach((value, index) => {
    expect(Number.isFinite(actual[index])).toBe(true);
    expect(actual[index]).toBeGreaterThan(0);
    expect(Math.abs(actual[index] / value - 1)).toBeLessThan(2e-6);
  });
});

for (const adjusted of [false, true])
  for (const counted of [false, true]) {
    it(`keeps optional output writers when adjusted=${adjusted}, counted=${counted}`, () => {
      const device = createNullWebGPUDevice(),
        graph = new GPUCommandGraph(device);
      let serial = 0;
      const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
        createTransientView(graph, `normal-tail-${serial++}`, format, length);
      try {
        const contributor = new GPUSpaceTimeCrossCorrelation({
          first: view('float32', 8),
          second: view('float32', 8),
          layout: 'cell-major',
          cellCount: 1,
          periodCount: 8,
          lags: [0],
          output: {
            correlations: view('float32', 1),
            pValues: view('float32', 1),
            adjustedPValues: adjusted ? view('float32', 1) : undefined,
            testedCount: counted ? view('uint32', 1) : undefined
          }
        });
        const kernels = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
        const correlate = kernels.find(kernel => kernel.id.endsWith('-correlate'))!.source;
        expect(correlate).toContain(ERFC_VALUE_WGSL);
        expect(correlate).toContain('erfcValue(z * 0.70710678)');
        expect(correlate).not.toContain('erfcPositive');
        expect(kernels.some(kernel => /adjusted\[.*\] =/.test(kernel.source))).toBe(adjusted);
        expect(kernels.some(kernel => kernel.id.endsWith('-publish-tested-count'))).toBe(counted);
      } finally {
        device.destroy();
      }
    });
  }
