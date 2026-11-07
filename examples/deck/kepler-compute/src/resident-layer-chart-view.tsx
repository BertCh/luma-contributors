// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useEffect, useState} from 'react';
import {ChartRenderer} from '@kepler.gl/charts';
import type {LayerChartConfig} from '@kepler.gl/types';
import {
  readResidentLayerChart,
  type ResidentLayerChartDependencies,
  type ResidentLayerChartResult
} from './resident-layer-chart';
import type {ResidentTooltipReader} from './resident-tooltip';

/** Builds row-reading dependencies from the existing resident tooltip reader (bounded, revalidated). */
export function makeResidentLayerChartDependencies(
  datasetId: string,
  layerId: string,
  readTooltip: ResidentTooltipReader,
  findSiblingRows?: ResidentLayerChartDependencies['findSiblingRows']
): ResidentLayerChartDependencies {
  return {
    async readRows(rowIndexes, fieldNames) {
      const rows = new Map<number, Record<string, string | number | boolean | null>>();
      for (const row of rowIndexes) {
        const values = await readTooltip(datasetId, layerId, row, fieldNames);
        if (!values) return null;
        rows.set(row, values);
      }
      return rows;
    },
    findSiblingRows
  };
}

/** Native hover charts for a resident layer; the computation is the native one over bounded rows. */
export function ResidentLayerChart({
  chart,
  datasetId,
  hoveredRow,
  dependencies,
  revisionKey
}: {
  chart: LayerChartConfig;
  datasetId: string;
  hoveredRow: number;
  dependencies: ResidentLayerChartDependencies;
  revisionKey: string;
}) {
  const [result, setResult] = useState<ResidentLayerChartResult | null | undefined>(undefined);
  const chartKey = JSON.stringify([chart.axis, chart.xAxis, chart.yAxis, chart.chartDisplay]);
  useEffect(() => {
    let current = true;
    setResult(undefined);
    void readResidentLayerChart(datasetId, chart, hoveredRow, dependencies).then(
      value => current && setResult(value),
      () => current && setResult(null)
    );
    return () => {
      current = false;
    };
  }, [datasetId, hoveredRow, chartKey, revisionKey, dependencies]);
  return (
    <section aria-label={`Layer chart: ${chart.title}`} data-resident-layer-chart>
      <h4>{chart.title}</h4>
      {result === undefined ? (
        <p role="status">Reading hovered rows…</p>
      ) : result === null ? (
        <p role="status">The layer chart is stale; hover again.</p>
      ) : (
        <>
          <ChartRenderer data={result.view} />
          {result.notes.map(note => (
            <p key={note} role="status">
              {note}
            </p>
          ))}
        </>
      )}
    </section>
  );
}

/** All layer charts configured for the hovered resident layer, sharing one dependency object. */
export function ResidentLayerCharts({
  charts,
  datasetId,
  layerId,
  hoveredRow,
  readTooltip,
  revisionKey,
  findSiblingRows
}: {
  charts: readonly {type?: string; layerId?: string}[];
  datasetId: string;
  layerId: string;
  hoveredRow: number;
  readTooltip: ResidentTooltipReader;
  revisionKey: string;
  findSiblingRows?: ResidentLayerChartDependencies['findSiblingRows'];
}) {
  const layerCharts = charts.filter(
    chart => chart.type === 'layerChart' && chart.layerId === layerId
  ) as LayerChartConfig[];
  const [dependencies] = useState(() =>
    makeResidentLayerChartDependencies(datasetId, layerId, readTooltip, findSiblingRows)
  );
  if (!layerCharts.length) return null;
  return (
    <>
      {layerCharts.map(chart => (
        <ResidentLayerChart
          key={chart.id}
          chart={chart}
          datasetId={datasetId}
          hoveredRow={hoveredRow}
          dependencies={dependencies}
          revisionKey={revisionKey}
        />
      ))}
    </>
  );
}
