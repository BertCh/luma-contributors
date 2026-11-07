// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ResidentCategorySummary} from './resident-chart-bins';

/** Fold only bounded GPU cells. Native totals sum per-series statistics, including means/minima. */
export function foldResidentChartSeries(
  cells: readonly ResidentCategorySummary[],
  categories: readonly string[],
  series: readonly string[]
): ResidentCategorySummary[] {
  const stride = series.length + 1;
  return categories.map((category, categoryIndex) => {
    const groups = cells.slice(categoryIndex * stride, (categoryIndex + 1) * stride);
    const present = groups
      .slice(0, series.length)
      .map((summary, index) => ({...summary, category: series[index]}))
      .filter(summary => summary.count > 0 && summary.category !== '')
      .sort((first, second) => first.firstRow - second.firstRow);
    return {
      category,
      value: present.reduce((total, summary) => total + summary.value, 0),
      count: groups.reduce((total, summary) => total + summary.count, 0),
      firstRow: groups.reduce((first, summary) => Math.min(first, summary.firstRow), 0xffffffff),
      series: present
    };
  });
}
