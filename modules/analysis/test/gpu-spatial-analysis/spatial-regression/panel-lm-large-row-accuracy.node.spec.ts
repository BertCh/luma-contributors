// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  createLargePanelScene,
  PANEL_ENTITY_COUNT,
  PANEL_ROW_COUNT,
  PANEL_LM_ERROR
} from './panel-lm-large-row-accuracy.test-utils';

it('the smaller pooled-panel fixture discriminates serial float32 LM accumulation', () => {
  const scene = createLargePanelScene();
  // Hand-worked OLS: sum x=sum u=x'u=0, so beta=(1,2), RSS=n, df=n-2.
  let predictorSum = 0;
  let errorSum = 0;
  let predictorError = 0;
  let score = 0;
  let trace = 0;
  let serialScore = 0;
  let serialTrace = 0;
  for (let row = 0; row < PANEL_ROW_COUNT; row++) {
    const entity = row % PANEL_ENTITY_COUNT;
    const periodStart = row - entity;
    const residual = scene.response[row] - 1 - 2 * scene.predictors[row];
    expect(residual).toBe(scene.errors[row]);
    predictorSum += scene.predictors[row];
    errorSum += residual;
    predictorError += scene.predictors[row] * residual;
    let lag = 0;
    let serialLag = 0;
    let rowTrace = 0;
    let serialRowTrace = 0;
    for (let slot = scene.offsets[entity]; slot < scene.offsets[entity + 1]; slot++) {
      const weight = scene.weights[slot];
      const neighborResidual = scene.errors[periodStart + scene.neighbors[slot]];
      lag += weight * neighborResidual;
      serialLag = Math.fround(serialLag + Math.fround(weight * neighborResidual));
      rowTrace += weight * (weight + weight);
      serialRowTrace = Math.fround(
        serialRowTrace + Math.fround(weight * Math.fround(weight + weight))
      );
    }
    score += residual * lag;
    trace += rowTrace;
    serialScore = Math.fround(serialScore + Math.fround(residual * serialLag));
    serialTrace = Math.fround(serialTrace + serialRowTrace);
  }
  expect([predictorSum, errorSum, predictorError]).toEqual([0, 0, 0]);
  const variance = PANEL_ROW_COUNT / (PANEL_ROW_COUNT - 2);
  const statistic = score ** 2 / (variance ** 2 * trace);
  const serialStatistic = serialScore ** 2 / (variance ** 2 * serialTrace);
  expect(Number.isFinite(statistic)).toBe(true);
  expect(Math.abs(statistic / PANEL_LM_ERROR - 1)).toBeLessThan(1e-9);
  expect(Number.isFinite(serialStatistic)).toBe(true);
  // This validates the LM statistic, including OLS residuals and RSS/(n-p), not just trace drift.
  expect(Math.abs(serialStatistic / PANEL_LM_ERROR - 1)).toBeGreaterThan(1e-3);
});
