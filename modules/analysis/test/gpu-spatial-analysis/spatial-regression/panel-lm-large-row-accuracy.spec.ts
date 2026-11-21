// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUSpatialPanelErrorMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-error-maximum-likelihood';
import {getPanelDiagnosticNodes} from '../../../src/gpu-spatial-analysis/spatial-regression/panel-diagnostics';
import {readFloat32, readUint32} from '../../utils/gpu-contributor-test-utils';
import {createRegressionTestContext} from './math-review-regression-test-utils';
import {
  createLargePanelScene,
  PANEL_PERIOD_COUNT,
  PANEL_LM_ERROR
} from './panel-lm-large-row-accuracy.test-utils';

it('pooled panel LM error at 1048576 rows matches the independent float64 statistic', {
  timeout: 120000
}, async () => {
  const context = await createRegressionTestContext();
  try {
    const scene = createLargePanelScene();
    const tests = context.output('float32', 6);
    const status = context.output('uint32', 1);
    const contributor = new GPUSpatialPanelErrorMaximumLikelihood({
      predictorCount: 1,
      periodCount: PANEL_PERIOD_COUNT,
      predictors: context.input('float32', scene.predictors),
      response: context.input('float32', scene.response),
      weights: {
        offsets: context.input('uint32', scene.offsets),
        neighbors: context.input('uint32', scene.neighbors),
        weights: context.input('float32', scene.weights)
      },
      diagnostics: {tests: tests.view, status: status.view},
      output: {
        table: context.output('float32', 12).view,
        summary: context.output('float32', 22).view,
        status: {
          code: context.output('uint32', 1).view,
          converged: context.output('uint32', 1).view
        }
      }
    });
    // Run the actual pooled diagnostic graph (including both OLS fits). Its definition uses
    // the untransformed input and does not depend on the separate ML likelihood search.
    context.graph.add({getCommandNodes: graph => getPanelDiagnosticNodes(graph, contributor)});
    context.run();
    expect(Array.from(await readUint32(status.buffer, 1))).toEqual([0]);
    const statistic = (await readFloat32(tests.buffer, 6))[0];
    expect(Number.isFinite(statistic)).toBe(true);
    expect(Math.abs(statistic / PANEL_LM_ERROR - 1)).toBeLessThan(1e-3);
  } finally {
    context.destroy();
  }
});
