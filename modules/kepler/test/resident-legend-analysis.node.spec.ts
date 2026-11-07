// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {getAnalysisLegendModel} from '../../../examples/deck/kepler-compute/src/analysis-legend';
import {getResidentChannelLegendModel} from '../../../examples/deck/kepler-compute/src/resident-channel-legend-model';
import {makeAnalysisTooltipProvider} from '../../../examples/deck/kepler-compute/src/analysis-tooltip-provider';
import {getTooltipComputedProvider} from '../../../examples/deck/kepler-compute/src/resident-tooltip-fields';

describe('resident channel legend model', () => {
  it('maps linear and sqrt ticks into the visual range', () => {
    const linear = getResidentChannelLegendModel({
      scale: 'linear',
      domain: [0, 10],
      range: [1, 11],
      tickCount: 3
    });
    expect(linear).toEqual({
      status: 'ready',
      ticks: [
        {value: 0, size: 1},
        {value: 5, size: 6},
        {value: 10, size: 11}
      ]
    });
    const sqrt = getResidentChannelLegendModel({
      scale: 'sqrt',
      domain: [0, 4],
      range: [0, 2],
      tickCount: 3
    });
    expect(sqrt.status === 'ready' && sqrt.ticks[1].size).toBeCloseTo((Math.sqrt(2) / 2) * 2, 5);
  });

  it('reports unsupported scales, invalid log domains and empty domains', () => {
    expect(
      getResidentChannelLegendModel({scale: 'quantile', domain: [0, 1], range: [0, 1]}).status
    ).toBe('unsupported');
    expect(
      getResidentChannelLegendModel({scale: 'log', domain: [0, 1], range: [0, 1]}).status
    ).toBe('unsupported');
    expect(
      getResidentChannelLegendModel({scale: 'linear', domain: null, range: [0, 1]}).status
    ).toBe('empty');
  });
});

describe('analysis legend model', () => {
  it('has fixed categorical entries and missing counts for Moran', () => {
    const model = getAnalysisLegendModel('local-moran-points', {localMoran: {invalidCount: 3}});
    expect(model?.entries.map(entry => entry.label)).toContain('High-high');
    expect(model?.missingCount).toBe(3);
  });
  it('has no legend for off, OLS and grid modes', () => {
    expect(getAnalysisLegendModel('off', null)).toBeNull();
    expect(getAnalysisLegendModel('ols', null)).toBeNull();
    expect(getAnalysisLegendModel('idw-points', {minimum: 0, maximum: 1})).toBeNull();
  });
  it('labels ramp classes from the summary breaks', () => {
    const model = getAnalysisLegendModel('quantile', {
      minimum: 0,
      maximum: 9,
      breaks: [0, 3, 6, 9],
      nullCount: 0
    });
    expect(model?.entries).toHaveLength(3);
    expect(model?.entries[0].label).toBe('0 to 3');
    expect(model?.missingCount).toBe(0);
  });
});

describe('analysis tooltip provider', () => {
  const analysis = {
    datasetId: 'a',
    settings: {column: 'value'},
    readRow: async (row: number) =>
      row === 1 ? null : {sourceRowIndex: row, value: row === 2 ? NaN : 4, kind: 'quantile'}
  } as any;
  it('returns a labelled computed value, null when unavailable, and ignores other datasets', async () => {
    const provider = makeAnalysisTooltipProvider(analysis);
    expect(await provider('a', 'l', 0)).toEqual({'Analysis (quantile) of value': 4});
    expect(await provider('a', 'l', 1)).toBeNull();
    expect(await provider('a', 'l', 2)).toEqual({'Analysis (quantile) of value': null});
    expect(await provider('b', 'l', 0)).toBeNull();
    expect(getTooltipComputedProvider('a')).toBeUndefined();
  });
});
