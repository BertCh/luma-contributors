// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUSpatialPanelRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-regression';
import {readFloat32, readUint32} from '../../utils/gpu-contributor-test-utils';
import {createRegressionTestContext} from './math-review-regression-test-utils';
import {createPanelAuditInput, expectPanelAuditRows} from './panel-audit.test-utils';
import {PANEL_REGRESSION_AUDIT_EXPECTED} from './panel-regression-audit.test-utils';

for (const effects of ['pooled', 'unit'] as const)
  for (const scene of [
    'ordinary',
    'island',
    'scaled',
    'terminal',
    'descending',
    'first',
    'neighbor',
    ...(effects === 'unit' ? (['entityShift', 'largeEntityShift'] as const) : [])
  ] as const)
    it(`panel IV every output versus SVD/dummy IV: ${effects}, ${scene}`, async () => {
      const data = createPanelAuditInput(false, scene === 'island');
      const responseScale = scene === 'scaled' ? 2 : scene === 'largeEntityShift' ? 1 / 4096 : 1;
      const precisionScale = scene === 'largeEntityShift' ? responseScale : 1;
      if (scene === 'scaled') data.response = Float32Array.from(data.response, value => 2 * value);
      if (scene === 'entityShift') {
        data.response = Float32Array.from(data.response, (value, row) => value + 16 * (row % 4));
        data.predictors = Float32Array.from(data.predictors, value => value + 16);
      }
      if (scene === 'largeEntityShift')
        data.response = Float32Array.from(
          data.response,
          (value, row) => value / 4096 + (row % 2 ? -3000 : 3000)
        );
      if (scene === 'terminal') data.weights.offsets[4] = 9;
      if (scene === 'descending') data.weights.offsets[2] = 1;
      if (scene === 'neighbor') data.weights.neighbors[0] = 4;
      if (scene === 'first') {
        data.weights.offsets = Uint32Array.from(data.weights.offsets, value => value + 1);
        data.weights.neighbors = Uint32Array.from([0, ...data.weights.neighbors]);
        data.weights.weights = Float32Array.from([0, ...data.weights.weights]);
      }
      const original =
        PANEL_REGRESSION_AUDIT_EXPECTED[
          scene === 'island' ? (effects === 'unit' ? 'islandUnit' : 'islandPooled') : effects
        ];
      const expected = {
        table: original.table.map((value, index) =>
          index < 8 && index % 4 < 2 ? responseScale * value : value
        ),
        summary: original.summary.map((value, index) =>
          index === 1 || index === 2 ? responseScale * responseScale * value : value
        ),
        residuals: original.residuals.map(value => responseScale * value)
      };
      const context = await createRegressionTestContext();
      try {
        const table = context.output('float32', 12),
          summary = context.output('float32', 16);
        const residuals = context.output('float32', 16),
          status = context.output('uint32', 1);
        context.graph.add(
          new GPUSpatialPanelRegression({
            effects,
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
              table: table.view,
              summary: summary.view,
              residuals: residuals.view,
              status: {code: status.view}
            }
          })
        );
        context.run();
        const state = (await readUint32(status.buffer, 1))[0];
        const actualTable = await readFloat32(table.buffer, 12);
        const actualSummary = await readFloat32(summary.buffer, 16);
        const actualResiduals = await readFloat32(residuals.buffer, 16);
        if (['terminal', 'descending', 'first', 'neighbor'].includes(scene)) {
          // Existing solver may fail first at rank validation (1) or nonfinite moments (3).
          expect([1, 3]).toContain(state);
          expect(Array.from(actualTable).every(Number.isNaN)).toBe(true);
          expect(Array.from(actualResiduals).every(Number.isNaN)).toBe(true);
          expect(actualSummary[0]).toBe(16);
          expect(actualSummary.slice(1).every(Number.isNaN)).toBe(true);
        } else {
          expect(state).toBe(0);
          // Sixteen rows, modest condition numbers; includes float32 IV refinement and
          // normal-tail approximation. Conditioning/first-stage summary uses 2e-3.
          for (let slot = 0; slot < 12; slot++)
            expectPanelAuditRows(
              [actualTable[slot]],
              [expected.table[slot]],
              5e-4 * (slot < 8 && slot % 4 < 2 ? precisionScale : 1)
            );
          expectPanelAuditRows(actualResiduals, expected.residuals, 5e-4 * precisionScale);
          for (let slot = 0; slot < 16; slot++)
            expectPanelAuditRows(
              [actualSummary[slot]],
              [expected.summary[slot]],
              2e-3 * (slot === 1 || slot === 2 ? precisionScale * precisionScale : 1)
            );
        }
      } finally {
        context.destroy();
      }
    });
