// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ResidentColumnEdits} from './resident-column-edits';
import {getResidentColumnDisplayName, getVisibleResidentColumns} from './resident-column-edits';

/**
 * Propagation of resident column edits to the Kepler references that name a column.
 *
 * Resident GPU buffers, layers and filters are keyed by the SOURCE column name, and the Kepler
 * table behind them is not rebuilt, so edits propagate as follows:
 * - delete: dependent filters are removed (removeFilter), the tooltip field entry is dropped
 *   (interactionConfigChange), and a column that a layer still uses is refused with the layer
 *   named (native would delete the layer; a resident layer owns GPU buffers and is never removed
 *   implicitly). Deleted columns are also left out of table and export headers.
 * - format: the dataset field display format is set through Kepler's setColumnDisplayFormat.
 * - rename: an alias that table headers, export headers and column pickers show; layer, filter and
 *   tooltip references keep the source name because the Kepler field and GPU column keep it.
 */
export type ResidentVisStateSnapshot = {
  layers: readonly ResidentLayerReference[];
  filters: readonly ResidentFilterReference[];
  interactionConfig?: {
    tooltip?: {
      config?: {fieldsToShow?: Record<string, readonly {name: string; format?: string | null}[]>};
    };
  };
};

export type ResidentLayerReference = {
  id: string;
  config: {
    dataId?: string | null;
    label?: string;
    columns?: Record<string, {value?: string | null} | undefined>;
    textLabel?: readonly {field?: {name?: string} | null}[];
    [key: string]: unknown;
  };
  visualChannels?: Record<string, {field: string}>;
};
export type ResidentFilterReference = {
  id: string;
  dataId: readonly string[];
  name: readonly string[];
  endName?: string | readonly string[];
};

export type ResidentColumnReferences = {
  /** Layers using the column, with the roles it plays (position, color channel, label...). */
  layers: {layerId: string; label: string; roles: string[]}[];
  /** Filters naming the column for this dataset. */
  filters: {filterId: string; index: number}[];
  tooltip: boolean;
};

export function findResidentColumnReferences(
  state: ResidentVisStateSnapshot,
  datasetId: string,
  columnName: string
): ResidentColumnReferences {
  const layers: ResidentColumnReferences['layers'] = [];
  for (const layer of state.layers) {
    if (layer.config.dataId !== datasetId) continue;
    const roles: string[] = [];
    for (const [key, value] of Object.entries(layer.config.columns ?? {}))
      if (value?.value === columnName) roles.push(`column ${key}`);
    for (const [channel, descriptor] of Object.entries(layer.visualChannels ?? {})) {
      const field = layer.config[descriptor.field] as {name?: string} | null | undefined;
      if (field?.name === columnName) roles.push(`channel ${channel}`);
    }
    for (const label of layer.config.textLabel ?? [])
      if (label.field?.name === columnName) roles.push('text label');
    if (roles.length)
      layers.push({layerId: layer.id, label: layer.config.label ?? layer.id, roles});
  }
  const filters: ResidentColumnReferences['filters'] = [];
  state.filters.forEach((filter, index) => {
    const valueIndex = filter.dataId.indexOf(datasetId);
    if (valueIndex < 0) return;
    const endNames = filter.endName === undefined ? [] : [filter.endName].flat();
    if (filter.name[valueIndex] === columnName || endNames[valueIndex] === columnName)
      filters.push({filterId: filter.id, index});
  });
  const tooltip = Boolean(
    state.interactionConfig?.tooltip?.config?.fieldsToShow?.[datasetId]?.some(
      field => field.name === columnName
    )
  );
  return {layers, filters, tooltip};
}

export type ResidentColumnDeletePlan =
  | {
      ok: true;
      /** Filter indices, highest first, so each removeFilter keeps the remaining indices valid. */
      removeFilterIndices: number[];
      /** Replacement fieldsToShow for the dataset, or undefined when the tooltip does not use it. */
      tooltipFieldsToShow?: {name: string; format?: string | null}[];
    }
  | {ok: false; reason: string};

export function planResidentColumnDelete(
  state: ResidentVisStateSnapshot,
  datasetId: string,
  columnName: string
): ResidentColumnDeletePlan {
  const references = findResidentColumnReferences(state, datasetId, columnName);
  if (references.layers.length) {
    const first = references.layers[0];
    return {
      ok: false,
      reason: `Layer "${first.label}" uses this column (${first.roles.join(', ')})${references.layers.length > 1 ? ` and ${references.layers.length - 1} more layer(s) do too` : ''}. Change the layer first; resident layers are never deleted implicitly.`
    };
  }
  const fieldsToShow = state.interactionConfig?.tooltip?.config?.fieldsToShow?.[datasetId];
  return {
    ok: true,
    removeFilterIndices: references.filters.map(filter => filter.index).sort((a, b) => b - a),
    ...(references.tooltip && fieldsToShow
      ? {tooltipFieldsToShow: fieldsToShow.filter(field => field.name !== columnName)}
      : {})
  };
}

/** Column headers for export: deleted columns removed, renamed columns under their alias. */
export function getResidentExportColumns(
  columns: readonly string[],
  edits: ResidentColumnEdits
): {source: string; header: string}[] {
  return getVisibleResidentColumns(columns, edits).map(source => ({
    source,
    header: getResidentColumnDisplayName(edits, source)
  }));
}

/** Dispatchers the host provides, already bound to the Kepler instance. */
export type ResidentColumnEditHost = {
  getVisState(): ResidentVisStateSnapshot | undefined;
  removeFilter(index: number): void;
  setTooltipFieldsToShow(
    datasetId: string,
    fieldsToShow: {name: string; format?: string | null}[]
  ): void;
};

/** Apply a delete: returns what was changed, or the refusal reason. */
export function applyResidentColumnDelete(
  host: ResidentColumnEditHost,
  datasetId: string,
  columnName: string
): {ok: true; removedFilters: number; tooltipUpdated: boolean} | {ok: false; reason: string} {
  const state = host.getVisState();
  if (!state) return {ok: true, removedFilters: 0, tooltipUpdated: false};
  const plan = planResidentColumnDelete(state, datasetId, columnName);
  if (!plan.ok) return plan;
  for (const index of plan.removeFilterIndices) host.removeFilter(index);
  if (plan.tooltipFieldsToShow) host.setTooltipFieldsToShow(datasetId, plan.tooltipFieldsToShow);
  return {
    ok: true,
    removedFilters: plan.removeFilterIndices.length,
    tooltipUpdated: Boolean(plan.tooltipFieldsToShow)
  };
}
