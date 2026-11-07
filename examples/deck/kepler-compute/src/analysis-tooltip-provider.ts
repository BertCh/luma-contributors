// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {AnalysisRuntime} from './analysis-runtime';
import {readNeighborTooltipValues} from './analysis-neighbor-tooltip';
import {getInterpolationCellIndex, makeInterpolationTooltipValues} from './interpolation-tooltip';
import {
  registerTooltipComputedProvider,
  type ResidentTooltipComputedProvider,
  type ResidentTooltipValue
} from './resident-tooltip-fields';

/** Human label of the computed column shown in tooltips for the active analysis mode. */
export function getAnalysisTooltipLabel(mode: string, column: string): string {
  return `Analysis (${mode}) of ${column}`;
}

type AnalysisTooltipSource = Pick<AnalysisRuntime, 'datasetId' | 'settings' | 'readRow'> &
  Partial<
    Pick<AnalysisRuntime, 'getTableColumns' | 'readInterpolationPage' | 'resultColumn' | 'summary'>
  >;

export type AnalysisTooltipOptions = {
  /** Bounded eight-byte read of one source row's [longitude, latitude]; enables interpolation values. */
  readPosition?(sourceRowIndex: number): Promise<readonly [number, number] | null>;
};

type InterpolationGridSettings = {
  bounds: readonly [number, number, number, number];
  width: number;
  height: number;
};

async function readInterpolationTooltip(
  analysis: AnalysisTooltipSource,
  options: AnalysisTooltipOptions,
  sourceRowIndex: number
): Promise<Record<string, ResidentTooltipValue> | null> {
  const settings = analysis.settings;
  const grid = (settings.mode === 'idw-points' ? settings.idw : settings.kriging)?.grid as
    | InterpolationGridSettings
    | undefined;
  if (!grid || !analysis.summary || !analysis.readInterpolationPage || !options.readPosition)
    return null;
  const position = await options.readPosition(sourceRowIndex);
  if (!position) return null;
  const cellIndex = getInterpolationCellIndex(grid, position[0], position[1]);
  if (cellIndex === null) return null;
  const [cell] = await analysis.readInterpolationPage(cellIndex, 1);
  return cell ? makeInterpolationTooltipValues(cell, settings.column) : null;
}

/**
 * Bounded readbacks of the already-picked source row: the four-byte analysis result, the companion
 * neighbor columns of neighbor-based analyses, and for IDW/kriging the interpolated surface value
 * at the picked point's location. Inactive, stale or unavailable analysis yields null so the
 * tooltip shows the original values only. Non-finite results are null.
 */
export function makeAnalysisTooltipProvider(
  analysis: AnalysisTooltipSource,
  options: AnalysisTooltipOptions = {}
): ResidentTooltipComputedProvider {
  return async (datasetId, _layerId, sourceRowIndex) => {
    if (datasetId !== analysis.datasetId) return null;
    const mode = analysis.settings.mode;
    if (mode === 'idw-points' || mode === 'kriging-points')
      return readInterpolationTooltip(analysis, options, sourceRowIndex);
    const row = await analysis.readRow(sourceRowIndex);
    if (!row) return null;
    const settings = analysis.settings;
    const values: Record<string, ResidentTooltipValue> = {
      [getAnalysisTooltipLabel(row.kind, settings.column)]: Number.isFinite(row.value)
        ? row.value
        : null
    };
    const snapshot = analysis.getTableColumns?.();
    if (snapshot && analysis.resultColumn)
      Object.assign(
        values,
        await readNeighborTooltipValues(snapshot, analysis.resultColumn, sourceRowIndex)
      );
    return values;
  };
}

/** Registers one provider per analysis dataset; returns the unregister function. */
export function registerAnalysisTooltipProvider(
  analysis: AnalysisTooltipSource,
  options: AnalysisTooltipOptions = {}
): () => void {
  return registerTooltipComputedProvider(
    analysis.datasetId,
    makeAnalysisTooltipProvider(analysis, options)
  );
}
