// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getPanelWithinNodes} from '../../../src/gpu-spatial-analysis/spatial-regression/panel-within-nodes';
import {getPanelDiagnosticNodes} from '../../../src/gpu-spatial-analysis/spatial-regression/panel-diagnostics';
import {GPUSpatialPanelErrorMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-error-maximum-likelihood';
import {readFloat32, readUint32} from '../../utils/gpu-contributor-test-utils';
import {createRegressionTestContext} from './math-review-regression-test-utils';
import {createPanelAuditInput, expectPanelAuditRows} from './panel-audit.test-utils';
import {PANEL_DIAGNOSTIC_AUDIT_EXPECTED} from './panel-regression-audit.test-utils';
import reviewDiagnostics from './panel-review-diagnostics.json';

for (const effects of ['unit', 'time'] as const)
  for (const shifted of [false, true])
    it(`within ${effects} GPU output matches exact group means, affine=${shifted}`, async () => {
      const data = createPanelAuditInput(shifted);
      const context = await createRegressionTestContext();
      try {
        const predictors = context.output('float32', 16),
          response = context.output('float32', 16);
        context.graph.add(
          getPanelWithinNodes(context.graph, {
            id: 'audit-within',
            effects,
            entityCount: 4,
            periodCount: 4,
            predictorCount: 1,
            predictors: context.input('float32', data.predictors),
            response: context.input('float32', data.response),
            output: {predictors: predictors.view, response: response.view}
          })
        );
        context.run();
        // Hand-summed means, as in panel-helpers-audit.node.spec.ts.
        const predictorMeans =
          effects === 'unit' ? [0.25, 0.25, 0.25, 0.5] : [0.75, -0.25, 0.5, 0.25];
        const responseMeans =
          effects === 'unit' ? [3.75, 1.25, 2.5, 3.5] : [3.75, 2.25, 2.25, 2.75];
        const group = (row: number) => (effects === 'unit' ? row % 4 : Math.floor(row / 4));
        expectPanelAuditRows(
          await readFloat32(predictors.buffer, 16),
          Array.from(
            data.predictors,
            (value, row) => value - predictorMeans[group(row)] - (shifted ? 16 : 0)
          ),
          0
        );
        expectPanelAuditRows(
          await readFloat32(response.buffer, 16),
          Array.from(
            data.response,
            (value, row) =>
              value - (shifted ? 2 : 1) * responseMeans[group(row)] - (shifted ? 16 : 0)
          ),
          0
        );
      } finally {
        context.destroy();
      }
    });

for (const scene of [
  'ordinary',
  'affine',
  'incomplete',
  'nonfinite',
  'rankZero',
  'nearby',
  'nearbyShift',
  'nearbyNonuniform',
  'scaled',
  'zeroFit',
  'nearRank',
  'nearRankReversed'
] as const)
  it(`pooled panel LM every GPU output versus independent OLS/projection: ${scene}`, async () => {
    const data = createPanelAuditInput(scene === 'affine');
    if (scene === 'zeroFit')
      data.response = Float32Array.of(-2, 5, 0, -3, ...new Array(12).fill(0));
    if (scene === 'scaled')
      data.response = Float32Array.from(data.response, value => value * 2 ** 32);
    if (
      scene === 'rankZero' ||
      scene === 'nearby' ||
      scene === 'nearbyShift' ||
      scene === 'nearbyNonuniform' ||
      scene === 'nearRank' ||
      scene === 'nearRankReversed'
    ) {
      data.predictors = Float32Array.from(
        {length: 16},
        (_, row) => (Math.floor(row / 4) + 1) * (row % 2 ? -1 : 1)
      );
      if (scene !== 'rankZero')
        data.predictors[0] +=
          scene === 'nearRank' || scene === 'nearRankReversed' ? 100 * 2 ** -23 : 0.25;
      if (scene === 'nearRankReversed') {
        const reverse = (values: Float32Array) =>
          Float32Array.from(
            {length: 16},
            (_, row) => values[(3 - Math.floor(row / 4)) * 4 + (row % 4)]
          );
        data.predictors = reverse(data.predictors);
        data.response = reverse(data.response);
      }
    }
    if (scene === 'nearbyShift')
      data.response = Float32Array.from(data.response, value => value + 32768);
    if (scene === 'nearbyNonuniform') {
      data.response = Float32Array.from(data.response, value => value + 16);
      data.weights.weights[0] = 0.25;
      data.weights.weights[1] = 0.25;
    }
    if (scene === 'nonfinite') data.response[7] = NaN;
    const context = await createRegressionTestContext();
    try {
      const tests = context.output('float32', 6),
        status = context.output('uint32', 1);
      const contributor = new GPUSpatialPanelErrorMaximumLikelihood({
        periodCount: 4,
        predictorCount: 1,
        weights: {
          offsets: context.input('uint32', data.weights.offsets),
          neighbors: context.input('uint32', data.weights.neighbors),
          weights: context.input('float32', data.weights.weights),
          status: {candidateOverflow: context.input('uint32', [scene === 'incomplete' ? 1 : 0])}
        },
        predictors: context.input('float32', data.predictors),
        response: context.input('float32', data.response),
        output: {
          table: context.output('float32', 12).view,
          summary: context.output('float32', 22).view,
          status: {
            code: context.output('uint32', 1).view,
            converged: context.output('uint32', 1).view
          }
        },
        diagnostics: {tests: tests.view, status: status.view}
      });
      // Exercise the complete diagnostic graph, including both actual OLS contributors.
      context.graph.add(getPanelDiagnosticNodes(context.graph, contributor));
      context.run();
      const invalid = scene === 'incomplete' || scene === 'nonfinite';
      expect(Array.from(await readUint32(status.buffer, 1))).toEqual([
        invalid ? 1 : scene === 'rankZero' || scene === 'zeroFit' ? 2 : 0
      ]);
      const actual = await readFloat32(tests.buffer, 6);
      if (invalid || scene === 'rankZero' || scene === 'zeroFit')
        expect(Array.from(actual).every(Number.isNaN)).toBe(true);
      else
        expectPanelAuditRows(
          actual,
          scene === 'nearRank' || scene === 'nearRankReversed'
            ? [
                2.4832931471755644, 0.11506111315877793, 0.1924145021982694, 0.6609146001913988,
                0.19241497912743916, 0.6609142062220347
              ]
            : scene === 'nearbyNonuniform'
              ? reviewDiagnostics.nearbyNonuniform.tests
              : scene === 'nearby' || scene === 'nearbyShift'
                ? reviewDiagnostics.nearby.tests
                : PANEL_DIAGNOSTIC_AUDIT_EXPECTED,
          5e-4
        );
    } finally {
      context.destroy();
    }
  });
