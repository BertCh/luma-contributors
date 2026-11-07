// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it, vi} from 'vitest';
import {
  BRUSH_CIRCLE_SEGMENT_COUNT,
  makeBrushCircleSelection
} from '../../../examples/deck/kepler-compute/src/resident-brush-geometry';
import {createResidentChannelDomainReader} from '../../../examples/deck/kepler-compute/src/resident-channel-domain';
import {
  getInterpolationCellIndex,
  makeInterpolationTooltipValues
} from '../../../examples/deck/kepler-compute/src/interpolation-tooltip';
import {getNeighborTooltipLabel} from '../../../examples/deck/kepler-compute/src/analysis-neighbor-tooltip';
import {makeAnalysisTooltipProvider} from '../../../examples/deck/kepler-compute/src/analysis-tooltip-provider';

describe('brush circle geometry', () => {
  it('keeps one topology for ready and inactive brushes', () => {
    const ready = makeBrushCircleSelection([-74, 40.7], 5000);
    const inactive = makeBrushCircleSelection(null, 0);
    expect(ready.status).toBe('ready');
    expect(inactive.status).toBe('inactive');
    for (const result of [ready, inactive]) {
      expect(result.selection.kind).toBe('polygon');
      if (result.selection.kind === 'polygon')
        expect(result.selection.polygons[0][0]).toHaveLength(BRUSH_CIRCLE_SEGMENT_COUNT);
    }
  });
  it('scales longitude radius by latitude and refuses polar or antimeridian brushes', () => {
    const result = makeBrushCircleSelection([0, 60], 111320);
    if (result.selection.kind !== 'polygon') throw new Error('expected polygon');
    const ring = result.selection.polygons[0][0];
    const longitudes = ring.map(vertex => vertex[0]);
    const latitudes = ring.map(vertex => vertex[1]);
    expect(Math.max(...latitudes)).toBeCloseTo(61, 5);
    expect(Math.max(...longitudes)).toBeCloseTo(2, 5);
    expect(makeBrushCircleSelection([0, 89], 1000).status).toBe('unsupported');
    expect(makeBrushCircleSelection([179.99, 0], 100000).status).toBe('unsupported');
  });
});

describe('channel domain reader', () => {
  it('reads the global domain once per revision and notifies', async () => {
    const readColumnStatistics = vi.fn(async () => ({
      stats: {
        quantiles: [
          {label: 'Min', value: 2},
          {label: 'Max', value: 9}
        ]
      }
    }));
    const notify = vi.fn();
    const reader = createResidentChannelDomainReader({
      resolveProvider: () => ({sourceRevision: 1, readColumnStatistics}) as never,
      notify
    });
    expect(reader.getDomain('a', 'value')).toBeNull();
    await vi.waitFor(() => expect(notify).toHaveBeenCalled());
    expect(reader.getDomain('a', 'value')).toEqual([2, 9]);
    expect(readColumnStatistics).toHaveBeenCalledTimes(1);
    expect(readColumnStatistics).toHaveBeenCalledWith({column: 'value', population: 'all'});
  });
  it('reports a refusal instead of a domain', async () => {
    const notify = vi.fn();
    const reader = createResidentChannelDomainReader({
      resolveProvider: () =>
        ({
          sourceRevision: 1,
          readColumnStatistics: async () => {
            throw new Error('too many rows');
          }
        }) as never,
      notify
    });
    reader.getDomain('a', 'value');
    await vi.waitFor(() => expect(notify).toHaveBeenCalled());
    expect(reader.getDomain('a', 'value')).toBeNull();
    expect(reader.getFailure('a', 'value')).toBe('too many rows');
  });
});

describe('interpolation and neighbor tooltips', () => {
  const grid = {bounds: [0, 0, 10, 10] as const, width: 10, height: 10};
  it('finds north-up grid cells', () => {
    expect(getInterpolationCellIndex(grid, 0.5, 9.5)).toBe(0);
    expect(getInterpolationCellIndex(grid, 9.5, 0.5)).toBe(99);
    expect(getInterpolationCellIndex(grid, 10, 0)).toBe(99);
    expect(getInterpolationCellIndex(grid, 11, 5)).toBeNull();
  });
  it('labels kriging cells with interval and variance', () => {
    expect(
      makeInterpolationTooltipValues(
        {value: 3, variance: 0.5, lower: 1, upper: 5, confidenceLevel: 0.95},
        'value'
      )
    ).toEqual({
      'Interpolated value': 3,
      'Kriging variance': 0.5,
      'Kriging interval lower (0.95)': 1,
      'Kriging interval upper (0.95)': 5
    });
    expect(makeInterpolationTooltipValues({value: 3, neighbors: 4}, 'value')).toEqual({
      'Interpolated value': 3,
      'Interpolation neighbors': 4
    });
  });
  it('labels neighbor columns', () => {
    expect(getNeighborTooltipLabel('analysis_local_moran_neighbor_count')).toBe(
      'Local moran neighbor count'
    );
  });
  it('reads the interpolated value at the picked point for IDW', async () => {
    const readInterpolationPage = vi.fn(async () => [{cellIndex: 99, value: 7, neighbors: 3}]);
    const provider = makeAnalysisTooltipProvider(
      {
        datasetId: 'a',
        settings: {mode: 'idw-points', column: 'value', idw: {grid}} as never,
        readRow: async () => null,
        summary: {} as never,
        readInterpolationPage: readInterpolationPage as never
      },
      {readPosition: async () => [9.5, 0.5]}
    );
    expect(await provider('a', 'layer', 4)).toEqual({
      'Interpolated value': 7,
      'Interpolation neighbors': 3
    });
    expect(readInterpolationPage).toHaveBeenCalledWith(99, 1);
    expect(await provider('b', 'layer', 4)).toBeNull();
  });
});
