// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  CHART_COLORS,
  DEFAULT_NUM_GROUPS,
  MAX_CHART_POINTS,
  OTHERS_KEY,
  type ChartBin
} from '@kepler.gl/charts';
import type {NativeChartConfiguration} from './resident-chart-plan';

export type ResidentCategorySummary = {
  category: string;
  value: number;
  count: number;
  /** Exact first selected source row, computed on the GPU, for native insertion-order ties/colors. */
  firstRow: number;
  filterValue?: ChartBin['filterValue'];
  series?: readonly Omit<ResidentCategorySummary, 'filterValue' | 'series'>[];
};

/** Format only the bounded group table; source rows never enter the native CPU chart aggregator. */
export function makeResidentCategoryChartBins(
  chart: NativeChartConfiguration,
  summaries: readonly ResidentCategorySummary[],
  datasetColor?: readonly number[]
): ChartBin[] {
  if (chart.type !== 'barChart' && chart.type !== 'horizontalBar' && chart.type !== 'lineChart')
    return [];
  const line = chart.type === 'lineChart';
  const colorRange = line ? undefined : chart.chartDisplay?.['colorRange'];
  const colors = colorRange?.colors?.length ? colorRange.colors : CHART_COLORS;
  const solidColor = chart.chartDisplay?.['color'] ?? datasetColor;
  const bins: ChartBin[] = summaries
    .filter(summary => summary.count > 0 && summary.category !== '')
    .sort((first, second) => first.firstRow - second.firstRow)
    .map((summary, index) => ({
      key: summary.category,
      count: summary.count,
      value: summary.value,
      color:
        !line && chart.colorBy === 'none'
          ? solidColor && solidColor.length >= 3
            ? `rgb(${solidColor[0]}, ${solidColor[1]}, ${solidColor[2]})`
            : CHART_COLORS[0]
          : colors[index % colors.length],
      filterValue: summary.filterValue ?? [summary.category],
      series: makeResidentChartSeriesBins(chart, summary.series)
    }));
  if (line)
    return bins
      .sort((first, second) => first.key.localeCompare(second.key))
      .slice(0, MAX_CHART_POINTS);
  bins.sort((first, second) => second.value - first.value);
  return truncateResidentChartBins(chart, bins);
}

/** Shared native category ranking limit, including integer charts that fall back to intervals. */
export function truncateResidentChartBins(
  chart: NativeChartConfiguration,
  bins: ChartBin[]
): ChartBin[] {
  if (chart.type !== 'barChart' && chart.type !== 'horizontalBar') return [];
  const maximumGroups = chart.numGroups ?? DEFAULT_NUM_GROUPS;
  if (maximumGroups <= 0) return bins.slice(0, MAX_CHART_POINTS);
  if (bins.length <= maximumGroups) return bins;
  const head = bins.slice(0, maximumGroups);
  if (chart.groupOthers) {
    const remainder = bins.slice(maximumGroups);
    head.push({
      key: OTHERS_KEY,
      value: remainder.reduce((total, bin) => total + bin.value, 0),
      count: remainder.reduce((total, bin) => total + bin.count, 0),
      color: CHART_COLORS[maximumGroups % CHART_COLORS.length]
    });
  }
  return head;
}

/** Native series palettes are local to each category; solid outer bars still use the default palette. */
export function makeResidentChartSeriesBins(
  chart: NativeChartConfiguration,
  summaries: ResidentCategorySummary['series']
): ChartBin[] | undefined {
  if (!summaries) return undefined;
  const palette =
    (chart.type === 'barChart' || chart.type === 'horizontalBar') && chart.colorBy !== 'none'
      ? chart.chartDisplay?.['colorRange']?.colors
      : undefined;
  const colors = palette?.length ? palette : CHART_COLORS;
  return summaries.map((summary, index) => ({
    key: summary.category,
    value: summary.value,
    count: summary.count,
    color: colors[index % colors.length]
  }));
}
