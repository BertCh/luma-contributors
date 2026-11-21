// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUSpatialLagImpactsOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-lag-impacts-operation';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {readFloat32} from '../../utils/gpu-contributor-test-utils';
import {createRegressionTestContext} from './math-review-regression-test-utils';

it('FRM-01 GPU impacts operation preserves omission and suppresses an actual bound failing-fit status', async () => {
  for (const fitStatus of [undefined, 0, 2]) {
    const context = await createRegressionTestContext();
    requireTestDevice(context.device);
    try {
      const impacts = context.output('float32', 3);
      const operation = new GPUSpatialLagImpactsOperation({
        id: 'impacts-status',
        coefficients: 'coefficients',
        coefficientStride: 1,
        rhoIndex: 2,
        predictorCount: 1,
        rowCount: 2,
        maximumPower: 1,
        traces: {powerTraces: 'traces', powerSums: 'sums', infinityNorm: 'norm'},
        ...(fitStatus !== undefined ? {status: 'fitStatus'} : {}),
        output: {impacts: 'impacts'}
      });
      const views = {
        coefficients: context.input('float32', [0, 2, 0]),
        powerTraces: context.input('float32', [0]),
        powerSums: context.input('float32', [2]),
        infinityNorm: context.input('float32', [1]),
        ...(fitStatus !== undefined ? {status: context.input('uint32', [fitStatus])} : {}),
        impacts: impacts.view
      };
      context.graph.add(
        operation.createContributor(name => views[name as keyof typeof views] as never)
      );
      context.run();
      const actual = await readFloat32(impacts.buffer, 3);
      expect(actual).toHaveLength(3);
      // rho = 0 gives direct = total = beta = 2, indirect = 0, irrespective of traces.
      if (fitStatus === 2) expect(actual.every(Number.isNaN)).toBe(true);
      else expect(Array.from(actual)).toEqual([2, 0, 2]);
      operation.destroy();
    } finally {
      context.destroy();
    }
  }
});
