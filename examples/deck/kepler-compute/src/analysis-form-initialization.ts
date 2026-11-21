// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useState, useSyncExternalStore, type SetStateAction} from 'react';
import {resolveGPUAnalysisDefault} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';
import {
  analysisDatasetContext,
  getAnalysisDatasetRoles,
  guessAnalysisPointColumns,
  type AnalysisDatasetContextEntry
} from './analysis-dataset-context';
import {makePointCoordinateDraft} from './analysis-point-coordinate-settings';
import type {AnalysisRuntime} from './analysis-runtime';

/** Only explicit row-major metadata establishes adjacency; a square row count does not. */
export function getAnalysisLatticeShape(metadata: unknown, rowCount: number) {
  if (!metadata || typeof metadata !== 'object') return undefined;
  const grid = (metadata as {grid?: {width?: number; height?: number; order?: string}}).grid;
  return grid?.order === 'row-major' &&
    Number.isSafeInteger(grid.width) &&
    Number.isSafeInteger(grid.height) &&
    grid.width! > 0 &&
    grid.height! > 0 &&
    grid.width! * grid.height! === rowCount
    ? {width: String(grid.width), height: String(grid.height)}
    : undefined;
}

/** Translate shared AX metadata into host drafts without materializing any source values. */
export function getAnalysisFormInitialization(
  entry: AnalysisDatasetContextEntry | undefined,
  columns: readonly string[],
  metadata?: unknown,
  rowCount = 0,
  publicationPointColumns?: {longitude: string; latitude: string}
) {
  const roles = entry ? getAnalysisDatasetRoles(entry) : undefined;
  const pointColumns =
    roles?.pointColumns ??
    publicationPointColumns ??
    guessAnalysisPointColumns(columns.map(name => ({name, type: 'real'})));
  const preparedPair =
    pointColumns &&
    columns.includes(pointColumns.longitude) &&
    columns.includes(pointColumns.latitude)
      ? pointColumns
      : undefined;
  const excluded = new Set([
    pointColumns?.longitude,
    pointColumns?.latitude,
    roles?.timeColumn,
    roles?.geometryColumn
  ]);
  const values = columns.filter(column => !excluded.has(column));
  const column =
    roles?.valueColumn && values.includes(roles.valueColumn)
      ? roles.valueColumn
      : values.includes('value')
        ? 'value'
        : (values[0] ?? '');
  const denominatorColumn = values.find(name => name !== column) ?? column;
  const coordinates = makePointCoordinateDraft();
  const planar = entry?.profile.coordinates?.model === 'planar';
  const extent = entry?.profile.extent;
  if (preparedPair) {
    if (planar) {
      coordinates.coordinateSpace = 'planar';
      coordinates.xColumn = preparedPair.longitude;
      coordinates.yColumn = preparedPair.latitude;
      if (extent?.every(Number.isFinite)) {
        [coordinates.minX, coordinates.minY, coordinates.maxX, coordinates.maxY] =
          extent.map(String);
      }
      // A planar profile declares neither a CRS nor metre units. Leave CRS confirmation explicit.
    } else {
      coordinates.longitudeColumn = preparedPair.longitude;
      coordinates.latitudeColumn = preparedPair.latitude;
    }
  }
  const bounds =
    entry &&
    !planar &&
    extent &&
    extent.every(Number.isFinite) &&
    extent[0] < extent[2] &&
    extent[1] < extent[3] &&
    extent[0] >= -180 &&
    extent[2] <= 180 &&
    extent[1] >= -90 &&
    extent[3] <= 90
      ? (['extent-west', 'extent-south', 'extent-east', 'extent-north'] as const).map(
          rule => resolveGPUAnalysisDefault(rule, entry.profile).value
        )
      : undefined;
  const interpolation = {
    longitudeColumn: coordinates.longitudeColumn,
    latitudeColumn: coordinates.latitudeColumn,
    west: bounds ? String(bounds[0]) : '',
    south: bounds ? String(bounds[1]) : '',
    east: bounds ? String(bounds[2]) : '',
    north: bounds ? String(bounds[3]) : ''
  };
  return {
    column,
    denominatorColumn,
    valueColumns: values,
    coordinates,
    interpolation,
    geometryColumn: roles?.geometryColumn ?? '',
    timeColumn: roles?.timeColumn ?? '',
    lattice: getAnalysisLatticeShape(metadata, entry?.profile.rowCount ?? rowCount),
    boundsSource: bounds
      ? 'Dataset extent (source coordinate units)'
      : 'Enter geographic grid bounds; no usable geographic extent is available',
    coordinateSource:
      entry?.dataset.coordinates || publicationPointColumns
        ? 'Publication coordinate roles'
        : 'Guessed coordinate roles; check the fields and coordinate system'
  };
}

export function useAnalysisFormInitialization(analysis: AnalysisRuntime) {
  useSyncExternalStore(
    analysisDatasetContext.subscribe,
    analysisDatasetContext.getVersion,
    analysisDatasetContext.getVersion
  );
  return analysis.formInitialization;
}

export type AnalysisFormDraftState<Draft extends object> = {
  defaults: Draft;
  draft: Draft;
  edited: ReadonlySet<keyof Draft>;
};

/** Richer metadata may update only untouched fields, including explicitly cleared controls. */
export function reconcileAnalysisFormDraft<Draft extends object>(
  state: AnalysisFormDraftState<Draft>,
  defaults: Draft
): AnalysisFormDraftState<Draft> {
  const draft = {...state.draft};
  for (const field of Object.keys(defaults) as (keyof Draft)[]) {
    if (!state.edited.has(field)) draft[field] = defaults[field];
  }
  return {...state, defaults, draft};
}

export function editAnalysisFormDraft<Draft extends object>(
  state: AnalysisFormDraftState<Draft>,
  draft: Draft
): AnalysisFormDraftState<Draft> {
  const edited = new Set(state.edited);
  for (const field of Object.keys(draft) as (keyof Draft)[]) {
    if (!Object.is(draft[field], state.draft[field])) edited.add(field);
  }
  return {...state, draft, edited};
}

/** Recipe changes reset the draft; profile refinements preserve edits made before Apply. */
export function useAnalysisFormDraft<Draft extends object>(
  defaults: Draft,
  savedKey: string | undefined,
  scopeKey?: unknown
): [Draft, (action: SetStateAction<Draft>) => void, (field: keyof Draft) => void] {
  const signature = JSON.stringify(defaults);
  const [state, setState] = useState(() => ({
    defaults,
    draft: defaults,
    edited: new Set<keyof Draft>() as ReadonlySet<keyof Draft>,
    signature,
    savedKey,
    scopeKey
  }));
  let current = state;
  if (state.savedKey !== savedKey || state.scopeKey !== scopeKey) {
    current = {
      ...state,
      defaults,
      draft: defaults,
      edited: new Set(),
      signature,
      savedKey,
      scopeKey
    };
    setState(current);
  } else if (state.signature !== signature) {
    current = {...state, ...reconcileAnalysisFormDraft(state, defaults), signature};
    setState(current);
  }
  return [
    current.draft,
    action =>
      setState(previous => ({
        ...previous,
        ...editAnalysisFormDraft(
          previous,
          typeof action === 'function'
            ? (action as (draft: Draft) => Draft)(previous.draft)
            : action
        )
      })),
    field => setState(previous => ({...previous, edited: new Set([...previous.edited, field])}))
  ];
}
