// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import * as arrow from 'apache-arrow';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {KeplerComputeSession} from '../src/session/compute-session';
import {GPUKeplerTableInspector} from '../src/exploration/gpu-table-inspector';
import {
  AnalysisRuntime,
  type HotspotSettings
} from '../../../examples/deck/kepler-compute/src/analysis-runtime';
import {requireTestDevice} from '../../graph/test/utils/require-test-device';

test('explicit hotspot lattice matches an independent Gi-star oracle across chunks and selected populations', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
  const values = [0, 0, 0, 0, 1, 10, 0, 10, 10];
  const first = arrow.tableFromArrays({
    value: new Float32Array(values.slice(0, 4)),
    row: new Float32Array([0, 1, 2, 3])
  });
  const second = arrow.tableFromArrays({
    value: new Float32Array(values.slice(4)),
    row: new Float32Array([4, 5, 6, 7, 8])
  });
  session.importArrowDataset({id: 'lattice', table: first.concat(second)});
  const explorer = session.openExplorer('lattice', {
    workingMemoryBudgetBytes: 8 * 1024 * 1024,
    dimensions: [{id: 'row', column: 'row', kind: 'range'}]
  });
  const analysis = new AnalysisRuntime({session, explorer, requestUpdate: () => {}});
  const inspector = new GPUKeplerTableInspector({
    session,
    explorer,
    sortMemoryBudgetBytes: 8 * 1024 * 1024,
    getDerivedColumns: () => analysis.getTableColumns()
  });
  const settings: HotspotSettings = {
    width: 3,
    height: 3,
    criterion: 'rook',
    radius: 1,
    minimumNeighbors: 1,
    selfWeight: 1,
    multipleTesting: 'none'
  };
  const render = async (minimumIncludedRow: number, minimumNeighbors: number) => {
    const encoder = device.createCommandEncoder();
    explorer.encode(encoder);
    analysis.encode(encoder);
    device.submit(encoder.finish());
    await analysis.readSummary();
    const expected = calculateLatticeOracle(values, minimumIncludedRow, minimumNeighbors);
    expect(analysis.summary?.hotspot).toEqual({
      weightsOverflow: 0,
      weightsRequiredCount: expected.links,
      invalidCount: expected.invalidCount
    });
    expect(analysis.summary?.count).toBe(expected.scores.filter(Number.isFinite).length);
    const snapshot = analysis.getTableColumns();
    expect(snapshot).toBeDefined();
    for (const column of Object.values(snapshot!.columns))
      expect(column.data.map(chunk => chunk.length)).toEqual([4, 5]);
    const page = await inspector.readPage({
      offset: 0,
      limit: 9,
      columns: Object.keys(snapshot!.columns),
      population: 'all'
    });
    expect(page).not.toBeNull();
    for (let row = 0; row < 9; row++) {
      const observed = page!.columns['analysis_hot_spot_z_score'][row];
      const probability = page!.columns['analysis_hot_spot_p_value'][row];
      if (Number.isFinite(expected.scores[row])) {
        expect(observed).toBeCloseTo(expected.scores[row], 4);
        expect(probability).toBeCloseTo(twoSidedNormalProbability(expected.scores[row]), 5);
      } else {
        expect(observed).toBeNaN();
        expect(probability).toBeNaN();
      }
      expect(page!.columns['analysis_hot_spot_bin'][row]).toBe(expected.bins[row]);
      expect(page!.columns['analysis_hot_spot_neighbor_count'][row]).toBe(
        expected.neighborCounts[row]
      );
    }
  };
  try {
    analysis.configureHotspots(settings);
    await render(0, 1);
    expect(analysis.resultColumn).toBe('analysis_hot_spot_z_score');
    expect((await analysis.readRow(8))?.kind).toBe('hot-spots-lattice');
    const saved = analysis.serializeRecipe();
    analysis.setMode('off');
    analysis.restoreRecipe(saved);
    expect(analysis.settings.hotspot).toEqual({...settings, neighborCapacity: 24});
    analysis.setPopulation('selected');
    explorer.setRange('row', 4, 8);
    await render(4, 1);
    analysis.configureHotspots({...settings, minimumNeighbors: 3});
    await render(4, 3);
    expect(analysis.summary?.hotspot?.invalidCount).toBe(4);
  } finally {
    await inspector.destroy();
    const fence = device.createFence();
    try {
      await analysis.destroy(fence.signaled);
      await session.destroy(fence.signaled);
    } finally {
      fence.destroy();
    }
  }
  expect(session.registry.memoryUsage.totalBytes).toBe(0);
});

/** Direct double-precision Gi* formula on a hand-defined 3×3 rook graph; no contributor helpers. */
function calculateLatticeOracle(
  values: readonly number[],
  minimumIncludedRow: number,
  minimumNeighbors: number
) {
  const included = values.slice(minimumIncludedRow);
  const count = included.length;
  const mean = included.reduce((total, value) => total + value, 0) / count;
  const variance = included.reduce((total, value) => total + (value - mean) ** 2, 0) / count;
  let links = 0;
  let invalidCount = 0;
  const neighborCounts: number[] = [];
  const scores = values.map((value, row) => {
    if (row < minimumIncludedRow) {
      neighborCounts.push(0);
      return NaN;
    }
    const neighbors = values
      .map((_, index) => index)
      .filter(
        index =>
          index >= minimumIncludedRow &&
          Math.abs((index % 3) - (row % 3)) +
            Math.abs(Math.floor(index / 3) - Math.floor(row / 3)) ===
            1
      );
    links += neighbors.length;
    neighborCounts.push(neighbors.length);
    const weightSum = neighbors.length + 1;
    const spread = (count * weightSum - weightSum * weightSum) / (count - 1);
    if (neighbors.length < minimumNeighbors || spread <= 0 || variance <= 0) {
      invalidCount++;
      return NaN;
    }
    const localSum = value + neighbors.reduce((total, neighbor) => total + values[neighbor], 0);
    return (localSum - mean * weightSum) / Math.sqrt(variance * spread);
  });
  const bins = scores.map(score => {
    if (!Number.isFinite(score)) return 0;
    const probability = twoSidedNormalProbability(score);
    const level = probability <= 0.01 ? 3 : probability <= 0.05 ? 2 : probability <= 0.1 ? 1 : 0;
    return level ? Math.sign(score) * level : 0;
  });
  return {scores, bins, neighborCounts, links, invalidCount};
}

function twoSidedNormalProbability(value: number): number {
  const absolute = Math.abs(value);
  const reciprocal = 1 / (1 + 0.2316419 * absolute);
  return (
    ((2 * Math.exp((-absolute * absolute) / 2)) / Math.sqrt(2 * Math.PI)) *
    reciprocal *
    (0.31938153 +
      reciprocal *
        (-0.356563782 +
          reciprocal * (1.781477937 + reciprocal * (-1.821255978 + reciprocal * 1.330274429))))
  );
}
