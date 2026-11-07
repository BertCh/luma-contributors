// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {CHART_COLORS, type ChartBin} from '@kepler.gl/charts';

export type ResidentChartSeriesPoint = {groupIndex: number; value: number};
export type ResidentChartSeriesLayout = {
  groups: readonly ChartBin[];
  series: {
    key: string;
    color: string;
    dash: string;
    values: readonly (ChartBin | undefined)[];
    segments: readonly ResidentChartSeriesPoint[][];
  }[];
  aggregateGroups: {bin: ChartBin; groupIndex: number}[];
  domain: readonly [number, number];
};

function hashSeriesKey(key: string): number {
  let hash = 2166136261;
  for (let index = 0; index < key.length; index++)
    hash = Math.imul(hash ^ key.charCodeAt(index), 16777619);
  return hash >>> 0;
}

/** Stable across category ordering and filtered-out series, unlike native per-group palettes. */
export function getResidentChartSeriesColor(
  key: string,
  colors: readonly string[] = CHART_COLORS
): string {
  const palette = colors.length ? colors : CHART_COLORS;
  return palette[hashSeriesKey(key) % palette.length];
}

/** Missing observations break a line; a supplied zero remains an actual observation. */
export function makeResidentChartLineSegments(
  values: readonly (ChartBin | undefined)[]
): ResidentChartSeriesPoint[][] {
  const segments: ResidentChartSeriesPoint[][] = [];
  let segment: ResidentChartSeriesPoint[] = [];
  for (const [groupIndex, bin] of values.entries()) {
    if (bin && Number.isFinite(bin.value)) segment.push({groupIndex, value: bin.value});
    else if (segment.length) {
      segments.push(segment);
      segment = [];
    }
  }
  if (segment.length) segments.push(segment);
  return segments;
}

/** Layout reads bounded bins only; neither series marks nor aggregate-only bins invent filters. */
export function makeResidentChartSeriesLayout(
  bins: readonly ChartBin[],
  colors?: readonly string[]
): ResidentChartSeriesLayout {
  const keys = [
    ...new Set(bins.flatMap(bin => bin.series?.map(series => series.key) ?? []))
  ].sort();
  const groups = bins.map(bin => new Map(bin.series?.map(series => [series.key, series])));
  const series = keys.map(key => {
    const values = groups.map(group => group.get(key));
    return {
      key,
      color: getResidentChartSeriesColor(key, colors),
      dash: ['', '6 3', '2 3', '8 3 2 3'][(hashSeriesKey(key) >>> 8) % 4],
      values,
      segments: makeResidentChartLineSegments(values)
    };
  });
  const aggregateGroups = bins.flatMap((bin, groupIndex) =>
    bin.series === undefined && Number.isFinite(bin.value) ? [{bin, groupIndex}] : []
  );
  let minimum = 0;
  let maximum = 0;
  for (const item of series)
    for (const bin of item.values)
      if (bin && Number.isFinite(bin.value)) {
        minimum = Math.min(minimum, bin.value);
        maximum = Math.max(maximum, bin.value);
      }
  for (const {bin} of aggregateGroups) {
    minimum = Math.min(minimum, bin.value);
    maximum = Math.max(maximum, bin.value);
  }
  return {
    groups: bins,
    series,
    aggregateGroups,
    domain: minimum === maximum ? [-1, 1] : [minimum, maximum]
  };
}
