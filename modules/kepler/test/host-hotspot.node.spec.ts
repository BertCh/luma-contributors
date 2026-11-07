// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import * as arrow from 'apache-arrow';
import {NullDevice} from '@luma.gl/core/test-utils';
import {KeplerComputeSession, type KeplerSessionExplorer} from '../src/session/compute-session';
import {
  AnalysisRuntime,
  type HotspotSettings
} from '../../../examples/deck/kepler-compute/src/analysis-runtime';
import {validateAnalysisRecipeCapacity} from '../../../examples/deck/kepler-compute/src/analysis-recipe-settings';
import {normalizeHotspotSettings} from '../../../examples/deck/kepler-compute/src/analysis-hotspot-settings';

test('hotspots require explicit shape and capacity before mutation and preserve independent settings', async () => {
  const device = new NullDevice({});
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
  const dataset = session.importArrowDataset({
    id: 'lattice',
    table: arrow.tableFromArrays({value: new Float32Array(9)})
  });
  const explorer = {
    dataset,
    isCurrent: true,
    currentRevision: 0
  } as unknown as KeplerSessionExplorer;
  const analysis = new AnalysisRuntime({session, explorer, requestUpdate: () => {}});
  const settings: HotspotSettings = {
    width: 3,
    height: 3,
    criterion: 'rook',
    radius: 1,
    minimumNeighbors: 1,
    selfWeight: 1,
    multipleTesting: 'none'
  };
  try {
    const before = analysis.settings;
    expect(analysis.rowCount).toBe(9);
    expect(() => analysis.setMode('hot-spots-lattice')).toThrow('Explicitly configure');
    expect(() => analysis.configureHotspots({...settings, width: 2})).toThrow('width × height');
    expect(() => analysis.configureHotspots({...settings, radius: 33})).toThrow('radius');
    expect(() => analysis.configureHotspots({...settings, neighborCapacity: 23})).toThrow(
      '24 slots'
    );
    expect(analysis.settings).toEqual(before);
    const beforeConfigurationBytes = session.registry.memoryUsage.totalBytes;
    analysis.configureHotspots(settings);
    expect(analysis.settings.mode).toBe('hot-spots-lattice');
    expect(analysis.settings.hotspot?.neighborCapacity).toBe(24);
    expect(analysis.getTableColumns()).toBeUndefined();
    expect(session.registry.memoryUsage.totalBytes).toBe(beforeConfigurationBytes);
    settings.width = 1;
    analysis.settings.hotspot!.height = 1;
    expect(analysis.settings.hotspot?.width).toBe(3);
    expect(analysis.settings.hotspot?.height).toBe(3);
    const saved = JSON.parse(analysis.serializeRecipe());
    saved.operation.neighborCapacity = 25;
    analysis.restoreRecipe(JSON.stringify(saved));
    expect(analysis.settings.hotspot?.neighborCapacity).toBe(25);
    saved.operation.neighborCapacity = 23;
    expect(() => validateAnalysisRecipeCapacity(saved, 9)).toThrow('24 slots');
    expect(() => analysis.restoreRecipe(JSON.stringify(saved))).toThrow('24 slots');
    expect(analysis.settings.hotspot?.neighborCapacity).toBe(25);
    expect(
      normalizeHotspotSettings({...settings, width: 3, criterion: 'queen'}, 9).neighborCapacity
    ).toBe(40);
    expect(
      normalizeHotspotSettings({...settings, width: 1, height: 3, radius: 3}, 3).neighborCapacity
    ).toBe(6);
  } finally {
    await analysis.destroy(Promise.resolve());
    await session.destroy(Promise.resolve());
    device.destroy();
  }
  expect(session.registry.memoryUsage.totalBytes).toBe(0);
});
