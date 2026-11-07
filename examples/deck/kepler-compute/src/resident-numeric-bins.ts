// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {CHART_COLORS, type ChartBin} from '@kepler.gl/charts';
import type {NativeChartConfiguration} from './resident-chart-plan';
import {
  makeResidentCategoryChartBins,
  makeResidentChartSeriesBins,
  truncateResidentChartBins,
  type ResidentCategorySummary
} from './resident-chart-bins';

/** Explicit intervals retain domain order; native integer auto keeps category ranking on fallback. */
export function makeResidentNumericChartBins(
  chart: NativeChartConfiguration,
  summaries: readonly ResidentCategorySummary[],
  boundaries: readonly number[],
  datasetColor?: readonly number[],
  categoryValues?: readonly number[]
): ChartBin[] {
  if (chart.type !== 'barChart' && chart.type !== 'horizontalBar' && chart.type !== 'lineChart')
    return [];
  const line = chart.type === 'lineChart';
  if (categoryValues)
    return makeResidentCategoryChartBins(
      chart,
      categoryValues.map((value, index) => ({
        ...summaries[index],
        category: String(value),
        filterValue: Number.isFinite(value) ? [value, value] : [String(value)]
      })),
      datasetColor
    );
  const colors =
    !line && chart.chartDisplay?.['colorRange']?.colors?.length
      ? chart.chartDisplay['colorRange'].colors
      : CHART_COLORS;
  const solidColor = chart.chartDisplay?.['color'] ?? datasetColor;
  const bins = boundaries.slice(0, -1).map((minimum, index) => {
    const maximum = boundaries[index + 1];
    const summary = summaries[index];
    return {
      key: `${minimum}\u001f${maximum}`,
      value: summary?.value ?? 0,
      count: summary?.count ?? 0,
      color:
        !line && chart.colorBy === 'none'
          ? solidColor && solidColor.length >= 3
            ? `rgb(${solidColor[0]}, ${solidColor[1]}, ${solidColor[2]})`
            : CHART_COLORS[0]
          : colors[index % colors.length],
      filterValue: [minimum, maximum],
      series: makeResidentChartSeriesBins(chart, summary?.series)
    };
  });
  const axis = chart.type === 'horizontalBar' ? chart.yAxis : chart.xAxis;
  if (!line && axis?.field?.type === 'integer' && axis.aggregation !== 'numericBin') {
    bins.sort((first, second) => second.value - first.value);
    return truncateResidentChartBins(chart, bins);
  }
  return bins;
}
