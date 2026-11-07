// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {clamp} from '@math.gl/core';

/** Color stops mirror the fixed palettes allocated by the analysis output classes (RGB, 0-255). */
export type AnalysisLegendEntry = {color: readonly [number, number, number]; label: string};
export type AnalysisLegendModel = {
  title: string;
  kind: 'categorical' | 'ramp';
  entries: readonly AnalysisLegendEntry[];
  /** Rows with no result (null source, invalid or excluded); count only when the summary has it. */
  missingCount?: number | null;
};
export type AnalysisLegendSummary = {
  minimum?: number;
  maximum?: number;
  breaks?: readonly number[];
  nullCount?: number;
  dbscan?: {clusterCount: number; noiseCount: number; excludedCount: number};
  localMoran?: {invalidCount: number};
  quantileLisa?: {invalidQuantile: number; islandCount: number};
};

const RAMP_COLORS: readonly (readonly [number, number, number])[] = [
  [49, 54, 149],
  [69, 117, 180],
  [116, 173, 209],
  [171, 217, 233],
  [254, 224, 144],
  [253, 174, 97],
  [244, 109, 67],
  [165, 0, 38]
];
const MORAN_ENTRIES: readonly AnalysisLegendEntry[] = [
  {color: [215, 25, 28], label: 'High-high'},
  {color: [171, 217, 233], label: 'Low-high'},
  {color: [44, 123, 182], label: 'Low-low'},
  {color: [253, 174, 97], label: 'High-low'},
  {color: [150, 150, 150], label: 'Not reported'}
];
const QUANTILE_LISA_ENTRIES: readonly AnalysisLegendEntry[] = [
  {color: [215, 25, 28], label: 'Significant'},
  {color: [150, 150, 150], label: 'Not significant'},
  {color: [253, 174, 97], label: 'Isolated'},
  {color: [65, 65, 65], label: 'Undefined'}
];
const DBSCAN_COLORS: readonly (readonly [number, number, number])[] = [
  [31, 119, 180],
  [255, 127, 14],
  [44, 160, 44],
  [214, 39, 40],
  [148, 103, 189]
];

function formatLegendNumber(value: number): string {
  return Number(value.toPrecision(5)).toLocaleString('en-US', {maximumSignificantDigits: 5});
}

/** Returns null for modes that produce no color result (off, interpolation grids, OLS, cartogram). */
export function getAnalysisLegendModel(
  mode: string,
  summary: AnalysisLegendSummary | null
): AnalysisLegendModel | null {
  if (mode === 'off' || mode === 'ols' || mode === 'dorling-cartogram') return null;
  if (mode.startsWith('local-moran'))
    return {
      title: 'Local Moran quadrant',
      kind: 'categorical',
      entries: MORAN_ENTRIES,
      missingCount: summary?.localMoran?.invalidCount ?? null
    };
  if (mode.startsWith('quantile-lisa'))
    return {
      title: 'Quantile LISA',
      kind: 'categorical',
      entries: QUANTILE_LISA_ENTRIES,
      missingCount: summary?.quantileLisa?.invalidQuantile ?? null
    };
  if (mode === 'dbscan-points') {
    const clusterCount = summary?.dbscan?.clusterCount ?? DBSCAN_COLORS.length;
    const shown = Math.min(clusterCount, DBSCAN_COLORS.length);
    return {
      title: 'DBSCAN cluster',
      kind: 'categorical',
      entries: [
        ...DBSCAN_COLORS.slice(0, shown).map((color, index) => ({
          color,
          label: `Cluster ${index + 1}`
        })),
        ...(clusterCount > shown
          ? [{color: DBSCAN_COLORS[0], label: `+${clusterCount - shown} more (colors repeat)`}]
          : []),
        {
          color: [150, 150, 150],
          label: `Noise${summary?.dbscan ? ` (${summary.dbscan.noiseCount})` : ''}`
        }
      ],
      missingCount: summary?.dbscan?.excludedCount ?? null
    };
  }
  if (mode === 'idw-points' || mode === 'kriging-points') return null;
  if (!summary || summary.minimum === undefined || summary.maximum === undefined) return null;
  const breaks = summary.breaks?.length ? summary.breaks : [summary.minimum, summary.maximum];
  const lastEdge = breaks[breaks.length - 1];
  const entries = RAMP_COLORS.slice(0, clamp(breaks.length - 1, 1, 8)).map((color, index) => ({
    color,
    label: `${formatLegendNumber(breaks[index])} to ${formatLegendNumber(breaks[index + 1] ?? lastEdge)}`
  }));
  return {
    title: `Analysis (${mode})`,
    kind: 'ramp',
    entries,
    missingCount: summary.nullCount ?? null
  };
}
