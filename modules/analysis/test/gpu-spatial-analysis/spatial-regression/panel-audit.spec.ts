// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUSpatialPanelLagMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-lag-maximum-likelihood';
import {readFloat32, readUint32} from '../../utils/gpu-contributor-test-utils';
import {createRegressionTestContext} from './math-review-regression-test-utils';
import {
  createWGSLKernelNode,
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  createPanelAuditInput,
  expectPanelAuditRows,
  getPanelAuditExpected,
  PANEL_AUDIT_EXPECTED
} from './panel-audit.test-utils';

for (const route of ['random', 'unit', 'tiled'] as const)
  for (const scene of ['ordinary', 'affine', 'island'] as const)
    it(`panel audit every output versus independent likelihood: ${route}, ${scene}`, async () => {
      const effects = route === 'random' ? 'random' : 'unit';
      const expected = getPanelAuditExpected(effects, scene === 'affine');
      const data = createPanelAuditInput(scene === 'affine', scene === 'island');
      const context = await createRegressionTestContext();
      try {
        const output = {
          coefficients: context.output('float32', 4),
          covariance: context.output('float32', 9),
          summary: context.output('float32', 6),
          status: context.output('uint32', 1),
          fitted: context.output('float32', 16),
          residuals: context.output('float32', 16)
        };
        const hausman = {tests: context.output('float32', 3), status: context.output('uint32', 1)};
        context.graph.add(
          new GPUSpatialPanelLagMaximumLikelihood({
            periodCount: 4,
            predictorCount: 1,
            effects,
            eigenvalues: route === 'tiled' ? context.input('float32', [-1, 0, 0, 1]) : undefined,
            weights: {
              offsets: context.input('uint32', data.weights.offsets),
              neighbors: context.input('uint32', data.weights.neighbors),
              weights: context.input('float32', data.weights.weights)
            },
            predictors: context.input('float32', data.predictors),
            response: context.input('float32', data.response),
            output: Object.fromEntries(
              Object.entries(output).map(([name, value]) => [name, value.view])
            ) as {
              coefficients: typeof output.coefficients.view;
              covariance: typeof output.covariance.view;
              summary: typeof output.summary.view;
              status: typeof output.status.view;
              fitted: typeof output.fitted.view;
              residuals: typeof output.residuals.view;
            },
            hausman:
              effects === 'random'
                ? {tests: hausman.tests.view, status: hausman.status.view}
                : undefined
          })
        );
        context.run();
        expect(Array.from(await readUint32(output.status.buffer, 1))).toEqual([
          scene === 'island' ? 1 : 0
        ]);
        for (const port of [
          'coefficients',
          'covariance',
          'summary',
          'fitted',
          'residuals'
        ] as const) {
          const actual = await readFloat32(output[port].buffer, output[port].view.length);
          if (scene === 'island') expect(Array.from(actual).every(Number.isNaN)).toBe(true);
          else if (port === 'summary') {
            expectPanelAuditRows(actual.slice(0, 2), expected.summary, 5e-4);
            expect(actual[2]).toBeGreaterThanOrEqual(1);
            expect(actual[2]).toBeLessThanOrEqual(effects === 'unit' ? 1 : 64);
            expect(Number.isFinite(actual[3])).toBe(true);
            expect(actual[3]).toBeLessThanOrEqual(1e-5);
            // 32 golden steps on [-.999,.999]: 1.998*((sqrt(5)-1)/2)^32.
            // Float32 subtraction of the bracket ends contributes at most two ulps here.
            expectPanelAuditRows(actual.slice(4), [4.102955015841747e-7, 0], 1.2e-7);
          } else {
            // 16 rows, well-conditioned design: 5e-4 covers float32 reductions and
            // dense log-determinant comparison rounding near the profile minimum.
            expectPanelAuditRows(actual, expected[port], 5e-4);
          }
        }
        if (effects === 'random') {
          const state = await readUint32(hausman.status.buffer, 1);
          const tests = await readFloat32(hausman.tests.buffer, 3);
          expect(Array.from(state)).toEqual([scene === 'island' ? 1 : 0]);
          if (scene === 'island') expect(Array.from(tests).every(Number.isNaN)).toBe(true);
          else expectPanelAuditRows(tests, PANEL_AUDIT_EXPECTED.random.hausman, 1e-3);
        }
      } finally {
        context.destroy();
      }
    });

it('reference native LU trace and analytic profile score match the independent ring derivatives', async () => {
  const data = createPanelAuditInput();
  const context = await createRegressionTestContext();
  try {
    recordWGSLKernelNodes(context.graph);
    const contributor = new GPUSpatialPanelLagMaximumLikelihood({
      periodCount: 4,
      predictorCount: 1,
      weights: {
        offsets: context.input('uint32', data.weights.offsets),
        neighbors: context.input('uint32', data.weights.neighbors),
        weights: context.input('float32', data.weights.weights)
      },
      predictors: context.input('float32', data.predictors),
      response: context.input('float32', data.response),
      output: {
        coefficients: context.output('float32', 4).view,
        covariance: context.output('float32', 9).view,
        summary: context.output('float32', 6).view,
        status: context.output('uint32', 1).view,
        fitted: context.output('float32', 16).view,
        residuals: context.output('float32', 16).view
      }
    });
    const nodes = contributor.getCommandNodes(context.graph);
    context.graph.add(
      nodes.filter(
        node =>
          node.id.includes('-within-pack-') ||
          ['-pooled-means', '-dense-weights', '-pack'].some(suffix => node.id.endsWith(suffix))
      )
    );
    const fit = nodes.find(node => node.id.endsWith('-random-fit'))!;
    const record = getWGSLKernelNodeRecord(context.graph, fit.id)!;
    const output = context.output('float32', 12);
    context.graph.add(
      createWGSLKernelNode(context.graph, {
        id: 'native-panel-score-probe',
        operation: 'PanelScoreOracle',
        invocationCount: 3,
        bindings: [
          ...record.bindings,
          {name: 'probe', view: output.view, type: 'f32', access: 'read_write'}
        ],
        declarations: record.declarations,
        body: `let rho=-.5+f32(index)*.5;let trace=getPanelLagTrace(rho);
let direct=vec4f(17.123268364327416,56.76483801338246,-.3033754550574301,0.0);
let score=getPanelLagProfileScore(rho,direct,trace,16.0,4.0,bitcast<u32>(centers[centersOffset+2u]));
probe[probeOffset+4u*index]=trace.x;probe[probeOffset+4u*index+1u]=trace.y;probe[probeOffset+4u*index+2u]=score.x;probe[probeOffset+4u*index+3u]=score.y;`
      })
    );
    context.run();
    // Fraction spectrum [-1,0,0,1] and NumPy SVD/SciPy derivative values in panel-audit.oracle.py.
    const expected = [
      -1.3333333333333333, 1, -14.577717216187487, 1, 0, 1, 12.329535210089412, 1,
      1.3333333333333333, 1, 18.90580121063879, 1
    ];
    expectPanelAuditRows(await readFloat32(output.buffer, 12), expected, 2e-6);
  } finally {
    context.destroy();
  }
});
