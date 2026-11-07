// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {
  NATIVE_AGGREGATION_NAMES,
  getAggregationStylePlan,
  getCustomColorThresholds,
  getReferenceBisectRight,
  getReferenceCellAppearance,
  getReferenceCellStatistics,
  getReferenceChannelScale,
  getReferenceCustomIndex,
  getReferenceElevation,
  getReferencePaletteIndex,
  getReferencePercentileThresholds,
  getReferenceQuantile,
  getReferenceStatistic,
  isAdvancedAggregationLayer,
  type AggregationPlanLayer
} from '../../../examples/deck/kepler-compute/src/aggregation-statistics-plan';
import {getCellKeySource} from '../../../examples/deck/kepler-compute/src/aggregation-statistics-gpu';
import {getNativeAggregationSettings} from '../../../examples/deck/kepler-compute/src/aggregation-runtime';

function makeLayer(
  config: Partial<AggregationPlanLayer['config']> & {
    visConfig?: Partial<AggregationPlanLayer['config']['visConfig']>;
  } = {},
  type: 'grid' | 'hexagon' = 'grid'
): AggregationPlanLayer {
  return {
    type,
    config: {
      colorField: null,
      colorScale: 'quantize',
      sizeField: null,
      ...config,
      visConfig: {
        percentile: [0, 100],
        elevationPercentile: [0, 100],
        sizeRange: [0, 500],
        colorRange: {colors: ['#000000', '#ffffff']},
        ...config.visConfig
      }
    }
  };
}

describe('native aggregation reference statistics (d3 semantics)', () => {
  const values = [4, 1, null, 2, 2, Number.NaN, 9];
  test('count includes null rows; others skip missing values', () => {
    expect(getReferenceStatistic(values, 'count')).toBe(7);
    expect(getReferenceStatistic(values, 'sum')).toBe(18);
    expect(getReferenceStatistic(values, 'average')).toBe(3.6);
    expect(getReferenceStatistic(values, 'minimum')).toBe(1);
    expect(getReferenceStatistic(values, 'maximum')).toBe(9);
    expect(getReferenceStatistic(values, 'median')).toBe(2);
    expect(getReferenceStatistic(values, 'mode')).toBe(2);
    expect(getReferenceStatistic(values, 'countUnique')).toBe(4);
  });
  test('variance and deviation are sample statistics, undefined below two values', () => {
    const variance = getReferenceStatistic([2, 4, 4, 4, 5, 5, 7, 9], 'variance');
    expect(variance).toBeCloseTo(32 / 7, 12);
    expect(getReferenceStatistic([2, 4, 4, 4, 5, 5, 7, 9], 'stdev')).toBeCloseTo(
      Math.sqrt(32 / 7),
      12
    );
    expect(getReferenceStatistic([5], 'variance')).toBeNaN();
    expect(getReferenceStatistic([5, null], 'stdev')).toBeNaN();
  });
  test('median of an even count averages the middle pair', () => {
    expect(getReferenceStatistic([1, 2, 3, 10], 'median')).toBe(2.5);
    expect(getReferenceQuantile([1, 2, 3, 4, 5], 0.25)).toBe(2);
  });
  test('mode ties resolve to the smallest value; all-null cells are undefined', () => {
    expect(getReferenceStatistic([3, 3, 1, 1, 2], 'mode')).toBe(1);
    expect(getReferenceStatistic([null, null], 'mode')).toBeNaN();
    expect(getReferenceStatistic([null, null], 'average')).toBeNaN();
    expect(getReferenceStatistic([null, null], 'sum')).toBe(0);
    expect(getReferenceStatistic([null, null], 'countUnique')).toBe(0);
  });
  test('cell grouping skips out-of-range keys and leaves empty cells undefined', () => {
    const result = getReferenceCellStatistics(
      [0, 2, 2, 0xffffffff, 5],
      [1, 3, 5, 100, 7],
      'average',
      4
    );
    expect(Array.from(result)).toEqual([1, Number.NaN, 4, Number.NaN]);
  });
});

describe('deck.gl channel scale reference', () => {
  const cells = Float64Array.from({length: 200}, (_, index) => (index % 50) + 1);
  test('percentile thresholds use the 99-quantile table', () => {
    const thresholds = getReferencePercentileThresholds(cells);
    expect(thresholds).toHaveLength(99);
    expect(thresholds[49]).toBeCloseTo(
      getReferenceQuantile(
        [...cells].sort((a, b) => a - b),
        0.5
      ),
      12
    );
    expect(getReferenceBisectRight([1, 2, 2, 5], 2)).toBe(3);
  });
  test('quantize cutoff comes from thresholds, infinite when out of range', () => {
    const thresholds = getReferencePercentileThresholds(cells);
    const scale = getReferenceChannelScale({
      values: cells,
      scale: 'quantize',
      lowerPercentile: 10,
      upperPercentile: 90
    });
    expect(scale.domain).toEqual([1, 50]);
    expect(scale.cutoff).toEqual([thresholds[9], thresholds[89]]);
    const unbounded = getReferenceChannelScale({
      values: cells,
      scale: 'quantize',
      lowerPercentile: 0,
      upperPercentile: 100
    });
    expect(unbounded.cutoff).toEqual([-Infinity, Infinity]);
    // Quirk preserved from deck.gl: an upper percentile of zero indexes thresholds[-1].
    expect(
      getReferenceChannelScale({
        values: cells,
        scale: 'linear',
        lowerPercentile: 0,
        upperPercentile: 0
      }).cutoff[1]
    ).toBe(Infinity);
  });
  test('quantile scale classifies into percentile indices over [0, 99]', () => {
    const scale = getReferenceChannelScale({
      values: cells,
      scale: 'quantile',
      lowerPercentile: 20,
      upperPercentile: 80
    });
    expect(scale.domain).toEqual([0, 99]);
    expect(scale.cutoff).toEqual([20, 79]);
    expect(Math.max(...scale.scaled)).toBeLessThanOrEqual(99);
    expect(scale.scaled[0]).toBe(
      getReferenceBisectRight(getReferencePercentileThresholds(cells), 1)
    );
  });
  test('custom breaks classify like classifyBinsByCustomBreaks', () => {
    expect(getReferenceCustomIndex(5, [10, 20], 3)).toBe(0);
    expect(getReferenceCustomIndex(10, [10, 20], 3)).toBe(1);
    expect(getReferenceCustomIndex(25, [10, 20], 3)).toBe(2);
    expect(getReferenceCustomIndex(Number.NaN, [10, 20], 3)).toBeNaN();
    expect(
      getCustomColorThresholds([
        [20, '#fff'],
        [10, '#000'],
        [null, '#111']
      ])
    ).toEqual([10, 20]);
  });
  test('cell appearance follows the grid-cell vertex shader', () => {
    const domain: [number, number] = [0, 100];
    expect(getReferenceCellAppearance(50, domain, [-Infinity, Infinity])).toEqual({
      visible: true,
      fraction: 0.5
    });
    expect(getReferenceCellAppearance(5, domain, [10, 90]).visible).toBe(false);
    expect(getReferenceCellAppearance(95, domain, [10, 90]).visible).toBe(false);
    expect(getReferenceCellAppearance(Number.NaN, domain, [-Infinity, Infinity]).visible).toBe(
      false
    );
    // The clipped domain starts at the cutoff, not at the data minimum.
    expect(getReferenceCellAppearance(50, domain, [10, 90]).fraction).toBeCloseTo(0.5);
    expect(getReferenceCellAppearance(7, [7, 7], [-Infinity, Infinity]).fraction).toBe(0);
    expect(getReferencePaletteIndex(1, 6)).toBe(5);
    expect(getReferencePaletteIndex(0.5, 6)).toBe(3);
    expect(getReferenceElevation(0.5, [0, 500])).toBe(250);
  });
});

describe('aggregation style planner', () => {
  test('accepts every native aggregation for a weighted color channel', () => {
    for (const aggregation of NATIVE_AGGREGATION_NAMES) {
      const layer = makeLayer({
        colorField: {name: 'value'},
        visConfig: {colorAggregation: aggregation}
      });
      const plan = getAggregationStylePlan(layer);
      expect(plan.color.aggregation).toBe(aggregation);
      expect(plan.color.weightColumn).toBe(aggregation === 'count' ? undefined : 'value');
    }
  });
  test('count and sum/average with default controls stay on the fast path', () => {
    expect(isAdvancedAggregationLayer(makeLayer())).toBe(false);
    expect(
      isAdvancedAggregationLayer(
        makeLayer({colorField: {name: 'value'}, visConfig: {colorAggregation: 'average'}})
      )
    ).toBe(false);
    for (const aggregation of ['minimum', 'maximum', 'median', 'stdev', 'variance', 'mode'])
      expect(
        isAdvancedAggregationLayer(
          makeLayer({colorField: {name: 'value'}, visConfig: {colorAggregation: aggregation}})
        )
      ).toBe(true);
    expect(isAdvancedAggregationLayer(makeLayer({visConfig: {percentile: [10, 100]}}))).toBe(true);
    expect(isAdvancedAggregationLayer(makeLayer({visConfig: {enable3d: true}}))).toBe(true);
  });
  test('elevation plan appears for extrusion or elevation percentiles only', () => {
    expect(getAggregationStylePlan(makeLayer()).elevation).toBeUndefined();
    const extruded = getAggregationStylePlan(
      makeLayer({
        sizeField: {name: 'height'},
        sizeScale: 'sqrt',
        visConfig: {enable3d: true, sizeAggregation: 'maximum', sizeRange: [10, 900]}
      })
    );
    expect(extruded.extruded).toBe(true);
    expect(extruded.elevation).toMatchObject({
      aggregation: 'maximum',
      weightColumn: 'height',
      scale: 'linear'
    });
    expect(extruded.elevationRange).toEqual([10, 900]);
    const filtered = getAggregationStylePlan(
      makeLayer({visConfig: {elevationPercentile: [0, 80]}})
    );
    expect(filtered.extruded).toBe(false);
    expect(filtered.elevation?.aggregation).toBe('count');
  });
  test('unsupported controls return specific reasons', () => {
    expect(getAggregationStylePlan(makeLayer({colorScale: 'jenks'})).color.scale).toBe('jenks');
    expect(getAggregationStylePlan(makeLayer({colorScale: 'ordinal'})).color.scale).toBe('ordinal');
    for (const colorScale of ['jenks', 'ordinal'])
      expect(() =>
        getAggregationStylePlan(makeLayer({colorScale, visConfig: {percentile: [10, 90]}}))
      ).toThrow(/percentile/);
    expect(() => getAggregationStylePlan(makeLayer({colorScale: 'weird'}))).toThrow(/weird/);
    expect(() =>
      getAggregationStylePlan(
        makeLayer({
          colorScale: 'custom',
          visConfig: {
            percentile: [5, 95],
            colorRange: {colors: ['#000', '#fff'], colorMap: [[1, '#000']]}
          }
        })
      )
    ).toThrow(/percentile/);
    expect(() =>
      getAggregationStylePlan(
        makeLayer({colorField: {name: 'x'}, visConfig: {colorAggregation: 'geometricMean'}})
      )
    ).toThrow(/geometricMean/);
  });
  test('interpolation follows the native texture filter', () => {
    expect(getAggregationStylePlan(makeLayer({colorScale: 'linear'})).interpolation).toBe('linear');
    expect(getAggregationStylePlan(makeLayer({colorScale: 'quantile'})).interpolation).toBe(
      'nearest'
    );
  });
});

describe('resident aggregation settings', () => {
  const base = {
    id: 'layer',
    type: 'grid' as const,
    config: {
      columnMode: 'points',
      columns: {lat: {value: 'latitude'}, lng: {value: 'longitude'}},
      colorField: {name: 'value'},
      colorScale: 'quantile',
      sizeField: null,
      textLabel: [],
      visConfig: {
        worldUnitSize: 1,
        coverage: 1,
        enable3d: false,
        colorAggregation: 'median',
        percentile: [0, 100],
        elevationPercentile: [0, 100],
        sizeRange: [0, 500],
        colorRange: {colors: ['#112233', '#ffeedd']}
      }
    }
  };
  test('advanced statistics, elevation and percentiles no longer throw', () => {
    const settings = getNativeAggregationSettings(base);
    expect(settings.advanced?.color.aggregation).toBe('median');
    expect(settings.statistic).toBe('count');
    const extruded = getNativeAggregationSettings({
      ...base,
      config: {
        ...base.config,
        sizeField: {name: 'height'},
        visConfig: {...base.config.visConfig, enable3d: true, percentile: [10, 90]}
      }
    });
    expect(extruded.advanced?.extruded).toBe(true);
    expect(extruded.advanced?.color.lowerPercentile).toBe(10);
  });
  test('plain count keeps the established fast path', () => {
    const settings = getNativeAggregationSettings({
      ...base,
      config: {...base.config, colorField: null, colorScale: 'quantize'}
    });
    expect(settings.advanced).toBeUndefined();
    expect(settings.statistic).toBe('count');
  });
  test('jenks and ordinal use the advanced lookup pipeline', () => {
    for (const colorScale of ['jenks', 'ordinal'])
      expect(
        getNativeAggregationSettings({...base, config: {...base.config, colorScale}}).advanced
          ?.color.scale
      ).toBe(colorScale);
  });
});

describe('cell key kernel source', () => {
  test('grid keys use the GPUGridBinning coordinate arithmetic', () => {
    const source = getCellKeySource({
      kind: 'grid',
      bounds: [0, 0, 40, 20],
      gridSize: [4, 2],
      cellCount: 8,
      cellSize: [10, 10]
    });
    expect(source.declarations).toContain('const COLUMNS: u32 = 4u;');
    expect(source.body).toContain('getCoordinate(y, minimumY, maximumY, ROWS) * COLUMNS');
  });
  test('hexagon keys select the d3 assignment when the layout requests it', () => {
    const layout = {
      kind: 'hexagon' as const,
      bounds: [0, 0, 100, 100] as const,
      gridSize: [10, 10] as const,
      cellCount: 100,
      radius: 10
    };
    expect(getCellKeySource({...layout, assignment: 'd3'}).body).toContain(
      'getPointDensityD3HexagonCell'
    );
    expect(getCellKeySource(layout).body).toContain('getPointDensityHexagonCell(');
  });
});
