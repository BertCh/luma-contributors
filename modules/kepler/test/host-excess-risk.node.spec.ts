// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import * as arrow from 'apache-arrow';
import {NullDevice} from '@luma.gl/core/test-utils';
import {KeplerComputeSession, type KeplerSessionExplorer} from '../src/session/compute-session';
import {AnalysisRuntime} from '../../../examples/deck/kepler-compute/src/analysis-runtime';
import {getAnalysisRecipeColumns} from '../../../examples/deck/kepler-compute/src/analysis-recipe-settings';

test('excess risk recipes preserve both fields and reject invalid populations before mutation', async () => {
  const device = new NullDevice({});
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
  const dataset = session.importArrowDataset({
    id: 'rates',
    table: arrow.tableFromArrays({
      events: new Float32Array([10, 20]),
      population: new Float32Array([100, 100]),
      label: ['a', 'b']
    })
  });
  const explorer = {
    dataset,
    isCurrent: true,
    currentRevision: 0
  } as unknown as KeplerSessionExplorer;
  const analysis = new AnalysisRuntime({session, explorer, requestUpdate: () => {}});
  try {
    expect(analysis.settings.denominatorColumn).toBe('population');
    analysis.setMode('excess-risk');
    const saved = analysis.serializeRecipe();
    expect(JSON.parse(saved).operation).toEqual({
      kind: 'excess-risk',
      events: 'events',
      populations: 'population'
    });
    expect(getAnalysisRecipeColumns(JSON.parse(saved))).toEqual(['events', 'population']);
    expect(analysis.resultColumn).toBe('analysis_excess_risk');
    expect(analysis.getTableColumns()?.columns.analysis_excess_risk.format).toBe('float32');
    analysis.setDenominatorColumn('events');
    expect(getAnalysisRecipeColumns(JSON.parse(analysis.serializeRecipe()))).toEqual(['events']);
    analysis.restoreRecipe(saved);
    expect(analysis.settings.denominatorColumn).toBe('population');
    const previous = analysis.settings;
    expect(() => analysis.setDenominatorColumn('label')).toThrow('Float32 population');
    const invalid = JSON.parse(saved);
    invalid.operation.populations = 'missing';
    expect(() => analysis.restoreRecipe(JSON.stringify(invalid))).toThrow('Float32 population');
    expect(analysis.settings).toEqual(previous);
  } finally {
    await analysis.destroy(Promise.resolve());
    await session.destroy(Promise.resolve());
    device.destroy();
  }
});
