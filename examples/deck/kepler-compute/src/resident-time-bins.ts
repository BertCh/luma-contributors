// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {CHART_COLORS, type ChartBin} from '@kepler.gl/charts';
import {makeResidentChartSeriesBins, type ResidentCategorySummary} from './resident-chart-bins';
import type {NativeChartConfiguration} from './resident-chart-plan';
import type {ResidentTimePeriod} from './resident-time-periods';

/** Calendar lines retain chronological order even when formatted year labels sort differently. */
export function makeResidentTimeChartBins(
  chart: NativeChartConfiguration,
  summaries: readonly ResidentCategorySummary[],
  periods: readonly ResidentTimePeriod[],
  datasetColor?: readonly number[]
): ChartBin[] {
  const bars = chart.type === 'barChart' || chart.type === 'horizontalBar';
  const colorRange = bars ? chart.chartDisplay?.['colorRange'] : undefined;
  const colors = colorRange?.colors?.length ? colorRange.colors : CHART_COLORS;
  const solidColor =
    (bars ? (chart.chartDisplay as {color?: readonly number[]} | undefined)?.color : undefined) ??
    datasetColor;
  return periods.map((period, index) => ({
    key: period.label,
    value: summaries[index].value,
    count: summaries[index].count,
    color:
      bars && chart.colorBy === 'none'
        ? solidColor && solidColor.length >= 3
          ? `rgb(${solidColor[0]}, ${solidColor[1]}, ${solidColor[2]})`
          : CHART_COLORS[0]
        : colors[index % colors.length],
    filterValue: [period.start, period.end],
    series: makeResidentChartSeriesBins(chart, summaries[index].series)
  }));
}

export {makeResidentTimeBarPeriods} from './resident-time-bar-periods';
