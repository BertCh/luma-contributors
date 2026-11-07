// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {computeLayerChart, type ChartableDataset, type ChartViewData} from '@kepler.gl/charts';
import type {LayerChartConfig} from '@kepler.gl/types';

/** Native layer charts summarise the hovered feature, or every feature sharing its id value. */
export const MAXIMUM_RESIDENT_LAYER_CHART_ROWS = 256;

export type ResidentLayerChartValue = string | number | boolean | null;
export type ResidentLayerChartRow = Record<string, ResidentLayerChartValue>;

export type ResidentLayerChartDependencies = {
  /** Bounded source-row read, normally the resident tooltip reader. */
  readRows(
    rowIndexes: readonly number[],
    fieldNames: readonly string[]
  ): Promise<Map<number, ResidentLayerChartRow> | null>;
  /**
   * Optional resident lookup of other rows with the same id value (at most `limit`).
   * Without it a chart with an id field summarises the hovered row only and says so.
   */
  findSiblingRows?(
    idField: string,
    value: ResidentLayerChartValue,
    limit: number
  ): Promise<number[] | null>;
};

export type ResidentLayerChartResult = {
  view: ChartViewData;
  rowCount: number;
  notes: string[];
};

export function isLayerChartForLayer(chart: {type?: string; layerId?: string}, layerId: string) {
  return chart.type === 'layerChart' && chart.layerId === layerId;
}

/** Every field the native chart reads for one hovered row set. */
export function getResidentLayerChartFieldNames(chart: LayerChartConfig): string[] {
  const names = [
    chart.axis?.field?.name,
    chart.xAxis?.field?.name,
    chart.yAxis?.field?.name,
    chart.chartDisplay?.idField
  ].filter((name): name is string => Boolean(name));
  return [...new Set(names)];
}

/** Adapts bounded read rows to the dataset interface the native chart computation expects. */
export function makeResidentLayerChartDataset(
  datasetId: string,
  chart: LayerChartConfig,
  rows: ReadonlyMap<number, ResidentLayerChartRow>
): ChartableDataset {
  const fieldTypes = new Map<string, string>();
  for (const axis of [chart.axis, chart.xAxis, chart.yAxis])
    if (axis?.field) fieldTypes.set(axis.field.name, axis.field.type);
  const idField = chart.chartDisplay?.idField;
  if (idField && !fieldTypes.has(idField)) fieldTypes.set(idField, 'string');
  const indexes = [...rows.keys()];
  return {
    id: datasetId,
    allIndexes: indexes,
    filteredIndex: indexes,
    fields: [...fieldTypes].map(([name, type], fieldIdx) => ({name, type, fieldIdx})),
    getValue: (fieldName, rowIndex) => rows.get(rowIndex)?.[fieldName] ?? null
  };
}

/**
 * Computes one hover chart with the native computeLayerChart over a bounded row set read from the
 * resident dataset. Returns null when the read was stale (caller retries or keeps the old view).
 */
export async function readResidentLayerChart(
  datasetId: string,
  chart: LayerChartConfig,
  hoveredRow: number,
  dependencies: ResidentLayerChartDependencies
): Promise<ResidentLayerChartResult | null> {
  const fieldNames = getResidentLayerChartFieldNames(chart);
  const notes: string[] = [];
  const idField = chart.chartDisplay?.idField;
  let rowIndexes = [hoveredRow];
  const hovered = await dependencies.readRows([hoveredRow], fieldNames);
  const hoveredValues = hovered?.get(hoveredRow);
  if (!hovered || !hoveredValues) return null;
  if (idField) {
    const idValue = hoveredValues[idField];
    if (idValue === null || idValue === undefined || idValue === '') {
      notes.push(`The hovered feature has no ${idField} value; showing the hovered row only.`);
    } else if (!dependencies.findSiblingRows) {
      notes.push(
        `Grouping by ${idField} needs a resident row lookup that is not available; showing the hovered row only.`
      );
    } else {
      const siblings = await dependencies.findSiblingRows(
        idField,
        idValue,
        MAXIMUM_RESIDENT_LAYER_CHART_ROWS + 1
      );
      if (!siblings) return null;
      if (siblings.length > MAXIMUM_RESIDENT_LAYER_CHART_ROWS)
        notes.push(
          `Showing the first ${MAXIMUM_RESIDENT_LAYER_CHART_ROWS} rows with this ${idField}.`
        );
      rowIndexes = [
        hoveredRow,
        ...siblings
          .filter(row => row !== hoveredRow)
          .slice(0, MAXIMUM_RESIDENT_LAYER_CHART_ROWS - 1)
      ];
    }
  }
  const rows = new Map<number, ResidentLayerChartRow>([[hoveredRow, hoveredValues]]);
  const missing = rowIndexes.filter(row => row !== hoveredRow);
  if (missing.length) {
    const extra = await dependencies.readRows(missing, fieldNames);
    if (!extra) return null;
    for (const [row, values] of extra) rows.set(row, values);
  }
  const dataset = makeResidentLayerChartDataset(datasetId, chart, rows);
  return {
    view: computeLayerChart(chart, dataset, [...rows.keys()]),
    rowCount: rows.size,
    notes
  };
}
