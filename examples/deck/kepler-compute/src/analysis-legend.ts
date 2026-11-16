// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getGPUAnalysisResultClasses} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {getKeplerResultRampColors, getKeplerResultToneColor} from '@luma-contributors/kepler';

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
  classCount?: number;
  column?: string;
  hotspot?: {invalidCount: number};
  nullCount?: number;
  dbscan?: {clusterCount: number; noiseCount: number; excludedCount: number};
  localMoran?: {invalidCount: number};
  quantileLisa?: {invalidQuantile: number; islandCount: number};
};

const MORAN_ENTRIES: readonly AnalysisLegendEntry[] = getGPUAnalysisResultClasses(
  'local-moran-clusters',
  'catalog-text'
).map(entry => ({label: entry.label, color: getKeplerResultToneColor(entry.tone)}));
const HOTSPOT_ENTRIES: readonly AnalysisLegendEntry[] = getGPUAnalysisResultClasses(
  'hot-spot-confidence-bins'
).map(entry => ({label: entry.label, color: getKeplerResultToneColor(entry.tone)}));
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
      title: `Local Moran cluster${summary?.column ? ` (${summary.column})` : ''}`,
      kind: 'categorical',
      entries: MORAN_ENTRIES,
      missingCount: summary?.localMoran?.invalidCount ?? null
    };
  if (mode.startsWith('hot-spots'))
    return {
      title: `Hot spot confidence${summary?.column ? ` (${summary.column})` : ''}`,
      kind: 'categorical',
      entries: [
        ...HOTSPOT_ENTRIES,
        ...(summary?.hotspot?.invalidCount
          ? [
              {
                label: 'Excluded: no neighbours',
                color: getKeplerResultToneColor('excluded')
              }
            ]
          : [])
      ],
      missingCount: summary?.hotspot?.invalidCount ?? null
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
  const finiteBreaks = summary.breaks?.filter(Number.isFinite);
  const classCount = summary.classCount ?? Math.max(1, (finiteBreaks?.length ?? 2) - 1);
  const breaks = finiteBreaks?.length
    ? finiteBreaks.slice(0, classCount + 1)
    : [summary.minimum, summary.maximum];
  const entries =
    classCount > 0
      ? getKeplerResultRampColors('sequential', classCount).map((color, index) => ({
          color,
          label: `${index ? 'above ' : ''}${formatLegendNumber(breaks[index])} to ${formatLegendNumber(breaks[index + 1])}`
        }))
      : [];

  return {
    title: `Analysis (${mode})${summary.column ? ` of ${summary.column}` : ''}`,
    kind: 'ramp',
    entries,
    missingCount: summary.nullCount ?? null
  };
}
