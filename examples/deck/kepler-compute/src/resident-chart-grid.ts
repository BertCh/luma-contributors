// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ChartBin} from '@kepler.gl/charts';
import type {NativeChartConfiguration} from './resident-chart-plan';

/** Native heatmap and pivot tables cap both axes at 10 entries. */
const DEFAULT_NUM_GROUPS = 10; // Mirrors @kepler.gl/charts DEFAULT_NUM_GROUPS (value import loads deck.gl).
export const MAXIMUM_RESIDENT_GRID_AXIS_ENTRIES = DEFAULT_NUM_GROUPS;

export type ResidentGridCell = {
  x: string;
  y: string;
  value: number;
  filterValueX?: ChartBin['filterValue'];
  filterValueY?: ChartBin['filterValue'];
};
export type ResidentPivotTable = {
  rowKeys: string[];
  columnKeys: string[];
  values: Record<string, Record<string, number>>;
};

type GridAxis = {field?: {name: string; type?: string | null} | null; aggregation?: string | null};
export type GridChartConfiguration = NativeChartConfiguration & {
  type: 'heatmapChart' | 'pivotTable';
  xAxis?: GridAxis | null;
  yAxis?: GridAxis | null;
  value?: GridAxis | null;
  numGroups?: number | null;
};

export function isResidentGridChart(
  chart: NativeChartConfiguration
): chart is GridChartConfiguration {
  return chart.type === 'heatmapChart' || chart.type === 'pivotTable';
}

/** The per-axis entry limit native uses: numGroups clamped into 1..10, defaulting to 10. */
export function getResidentGridAxisLimit(numGroups: number | null | undefined): number {
  return Math.min(
    Math.max(1, numGroups && numGroups > 0 ? numGroups : DEFAULT_NUM_GROUPS),
    MAXIMUM_RESIDENT_GRID_AXIS_ENTRIES
  );
}

/**
 * A heatmap or pivot is a category summary with a series: the X axis groups, the Y axis is the
 * series dictionary and the value axis is the measure. The existing bounded GPU category/numeric
 * runtimes then supply every (x, y) cell; nothing reads source rows. The Y axis must therefore be a
 * prepared string dictionary (the planner reports it otherwise).
 */
export function makeResidentGridSyntheticChart(
  chart: NativeChartConfiguration
): NativeChartConfiguration | undefined {
  if (!isResidentGridChart(chart)) return undefined;
  const limit = getResidentGridAxisLimit(chart.numGroups);
  const categoricalX = chart.xAxis?.field?.type === 'string';
  return {
    ...chart,
    type: 'barChart',
    xAxis: chart.xAxis,
    yAxis: chart.value ?? {field: null, aggregation: 'count'},
    groupBy: chart.yAxis ? {...chart.yAxis, aggregation: 'uniqueBin'} : undefined,
    // Categorical X ranking is done by the grid reducer (count order), not by bar value order.
    numGroups: categoricalX ? 0 : limit,
    groupOthers: false,
    colorBy: 'none'
  } as unknown as NativeChartConfiguration;
}

const compareKeys = (first: string, second: string) => first.localeCompare(second);

/** Native pickAxisEntries: ordered axes keep their domain order, others rank by count then name. */
export function pickResidentGridEntries<Entry extends {key: string; count: number}>(
  entries: readonly Entry[],
  limit: number,
  ordered: boolean
): Entry[] {
  if (ordered) return entries.slice(0, limit);
  return [...entries]
    .sort((first, second) => second.count - first.count || compareKeys(first.key, second.key))
    .slice(0, limit)
    .sort((first, second) => compareKeys(first.key, second.key));
}

function collectRows(bins: readonly ChartBin[], limit: number) {
  const totals = new Map<
    string,
    {key: string; count: number; filterValue?: ChartBin['filterValue']}
  >();
  for (const bin of bins)
    for (const series of bin.series ?? []) {
      const entry = totals.get(series.key) ?? {key: series.key, count: 0};
      entry.count += series.count;
      totals.set(series.key, entry);
    }
  return pickResidentGridEntries([...totals.values()], limit, false);
}

/** Heatmap cells from per-X bins that carry the Y series; a full grid with zeros for empty cells. */
export function makeResidentHeatmapCells(
  bins: readonly ChartBin[],
  props: {limit: number; orderedX: boolean}
): ResidentGridCell[] {
  const columns = pickResidentGridEntries(
    bins.map(bin => ({key: bin.key, count: bin.count, bin})),
    props.limit,
    props.orderedX
  );
  const rows = collectRows(bins, props.limit);
  return columns.flatMap(({bin}) =>
    rows.map(row => ({
      x: bin.key,
      y: row.key,
      value: bin.series?.find(series => series.key === row.key)?.value ?? 0,
      filterValueX: bin.filterValue ?? [bin.key],
      filterValueY: [row.key]
    }))
  );
}

/** Pivot table: Y series keys are rows, X groups are columns. */
export function makeResidentPivotTable(
  bins: readonly ChartBin[],
  props: {limit: number; orderedX: boolean}
): ResidentPivotTable {
  const cells = makeResidentHeatmapCells(bins, props);
  const columnKeys: string[] = [];
  const rowKeys: string[] = [];
  const values: Record<string, Record<string, number>> = {};
  for (const cell of cells) {
    if (!columnKeys.includes(cell.x)) columnKeys.push(cell.x);
    if (!rowKeys.includes(cell.y)) rowKeys.push(cell.y);
    values[cell.y] = values[cell.y] ?? {};
    values[cell.y][cell.x] = cell.value;
  }
  return {rowKeys, columnKeys, values};
}
