// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computePettittTestOnCPU} from '../../../src/gpu-spatial-analysis/change-points/cpu-pettitt-test-operation';
import {computeMultipleTestingAdjustmentOnCPU} from '../../../src/gpu-spatial-analysis/multiple-testing/cpu-multiple-testing-adjustment-operation';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {GPURasterFalseDiscoveryRateMask} from '../../../src/gpu-spatial-analysis/multiple-testing/gpu-raster-false-discovery-rate-mask';
import {createPettittCompositionCountOutputs} from './pettitt-composition-fixture';

it('B1 the executable Pettitt composition uses compatible count aliases', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const counts = createPettittCompositionCountOutputs(name =>
      createTransientView(graph, name, 'uint32', 1)
    );
    expect(() =>
      new GPURasterFalseDiscoveryRateMask({
        cellCount: 2,
        method: 'bonferroni',
        pValues: createTransientView(graph, 'p', 'float32', 2),
        parameters: createTransientView(graph, 'parameters', 'float32', 1),
        output: {
          ...counts,
          adjustedPValues: createTransientView(graph, 'adjusted', 'float32', 2),
          significant: createTransientView(graph, 'significant', 'uint32', 2)
        }
      }).getCommandNodes(graph)
    ).not.toThrow();
    expect(counts.finiteTestCount).not.toBe(counts.effectiveFamilySize);
  } finally {
    device.destroy();
  }
});

it('DGN-01 Pettitt composition excludes zero/one finite observations and counts a tested constant', () => {
  for (const control of [
    new Array<number>(12).fill(NaN),
    [7, ...new Array<number>(11).fill(NaN)],
    new Array<number>(12).fill(7)
  ]) {
    const constant = control.every(Number.isFinite);
    const result = computePettittTestOnCPU(
      Float32Array.from([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, ...control]),
      {cellCount: 2, sliceCount: 12, layout: 'cell-major'}
    );
    const adjusted = computeMultipleTestingAdjustmentOnCPU(result.pValues, 'bonferroni');
    // The unique middle split has 6*6 positive cross-pairs: K=36, n=12.
    // 2*exp(-6*36^2/(12^3+12^2)) = .03140780113025614 (Python math.exp).
    expect(Number.isFinite(result.pValues[0])).toBe(true);
    expect(result.pValues[0]).toBe(Math.fround(0.03140780113025614));
    expect(adjusted.adjustedPValues[0]).toBe(
      Math.fround(constant ? 0.06281560226051228 : 0.03140780113025614)
    );
    if (constant) {
      expect(result.pValues[1]).toBe(1);
      expect(adjusted.adjustedPValues[1]).toBe(1);
    } else {
      expect(result.pValues[1]).toBeNaN();
      expect(adjusted.adjustedPValues[1]).toBeNaN();
    }
    expect(
      Array.from(adjusted.adjustedPValues, value => Number(Number.isFinite(value) && value <= 0.05))
    ).toEqual([constant ? 0 : 1, 0]);
    expect(Array.from(adjusted.finiteTestCount)).toEqual([constant ? 2 : 1]);
    expect(Array.from(adjusted.effectiveFamilySize)).toEqual([constant ? 2 : 1]);
  }
});
