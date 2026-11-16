// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useSyncExternalStore} from 'react';
import {
  describeKeplerDatasetProfile,
  type KeplerDatasetProfile,
  type KeplerDatasetProfileDeclarations
} from '@luma-contributors/kepler';
import {
  GPU_ANALYSIS_MEASUREMENT_LEVELS,
  GPU_ANALYSIS_ROW_SEMANTICS
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {
  analysisDatasetContext,
  ANALYSIS_PROFILE_LIMITS,
  getAnalysisDatasetRoles
} from './analysis-dataset-context';
import type {CatalogResidentDataset} from './analysis-catalog-resident-input';

/** A question is answerable without disabling the operation or its field controls. */
export function AnalysisRowMeaning({
  profile,
  onDeclare
}: {
  profile: KeplerDatasetProfile;
  onDeclare(value: NonNullable<KeplerDatasetProfileDeclarations['rowSemantics']>): void;
}) {
  return (
    <section aria-label={`Row meaning for ${profile.label}`} data-testid="ax-row-meaning">
      <p>
        {profile.rowSemantics?.source === 'declared'
          ? `Row meaning: ${profile.rowSemantics.value} (declared).`
          : `Row meaning is ${profile.rowSemantics ? `${profile.rowSemantics.value} (guessed)` : 'unknown'}. What does each row represent? You can run while this is uncertain; check the interpretation.`}
      </p>
      {GPU_ANALYSIS_ROW_SEMANTICS.map(value => (
        <button
          key={value}
          type="button"
          aria-pressed={
            profile.rowSemantics?.source === 'declared' && profile.rowSemantics.value === value
          }
          onClick={() => onDeclare(value)}
        >
          {value === 'events' ? 'Events' : value === 'objects' ? 'Objects' : 'Surface samples'}
        </button>
      ))}
    </section>
  );
}

/** Host declarations and bounded refinement, shared with every analysis surface. */
export function AnalysisProfileDeclarations({
  datasets
}: {
  datasets: readonly CatalogResidentDataset[];
}) {
  useSyncExternalStore(analysisDatasetContext.subscribe, analysisDatasetContext.getVersion);
  return (
    <section aria-label="Dataset analysis profiles" data-testid="analysis-profiles">
      <details>
        <summary>Dataset meaning and analysis limits</summary>
        <p>
          Metadata appears immediately. Statistics use original source values on request: at most{' '}
          {ANALYSIS_PROFILE_LIMITS.maximumRows.toLocaleString()} rows,{' '}
          {ANALYSIS_PROFILE_LIMITS.maximumColumns} columns,{' '}
          {ANALYSIS_PROFILE_LIMITS.maximumVertices.toLocaleString()} vertices and{' '}
          {ANALYSIS_PROFILE_LIMITS.maximumReadbackBytes / 1024 / 1024} MiB of readback. Column and
          spacing summaries use at most {ANALYSIS_PROFILE_LIMITS.maximumSampleSize} rows; sampled
          checks cannot certify the full population. Each operation may have smaller limits.
        </p>
        {datasets.map(dataset => {
          const entry = analysisDatasetContext.get(dataset.id);
          if (!entry) return null;
          const declarations = analysisDatasetContext.getDeclarations(dataset.id) ?? {};
          const roles = getAnalysisDatasetRoles(entry);
          const description = describeKeplerDatasetProfile(entry.profile);
          const update = (next: KeplerDatasetProfileDeclarations) =>
            analysisDatasetContext.setDeclarations(dataset.id, next);
          return (
            <fieldset key={dataset.id} data-dataset-id={dataset.id}>
              <legend>{dataset.label}</legend>
              <p>{description.headline}</p>
              <p>
                Coordinates:{' '}
                {entry.profile.pointColumns
                  ? `${entry.profile.pointColumns.longitude}/${entry.profile.pointColumns.latitude} (${dataset.coordinates ? 'publication roles' : 'guessed'})`
                  : (dataset.geometryColumn ?? 'not mapped')}
                . Time:{' '}
                {roles.timeColumn ? `${roles.timeColumn} (${roles.timeSource})` : 'not mapped'}.
                Value:{' '}
                {roles.valueColumn ? `${roles.valueColumn} (${roles.valueSource})` : 'not mapped'}.
              </p>
              <p role="status">
                {entry.stage === 'metadata'
                  ? 'Metadata ready; statistics not read'
                  : entry.stage === 'reading'
                    ? 'Reading bounded source statistics…'
                    : entry.stage === 'statistics'
                      ? 'Bounded statistics ready'
                      : entry.limitation}
              </p>
              <button
                type="button"
                disabled={
                  entry.stage === 'reading' ||
                  !dataset.provider.readColumnValues ||
                  dataset.rowCount > ANALYSIS_PROFILE_LIMITS.maximumRows
                }
                onClick={() => {
                  void analysisDatasetContext.readStatistics(dataset.id).catch(() => {});
                }}
              >
                Compute bounded statistics
              </button>
              {!dataset.provider.readColumnValues ? (
                <p>Original readback unavailable; metadata only.</p>
              ) : null}
              {dataset.rowCount > ANALYSIS_PROFILE_LIMITS.maximumRows ? (
                <p>Source exceeds the bounded CPU profile row limit. GPU profiling is required.</p>
              ) : null}
              <p>Correct a guess when an analysis depends on what these rows measure.</p>
              <label>
                Row meaning
                <select
                  aria-label={`Row meaning for ${dataset.label}`}
                  value={declarations.rowSemantics ?? ''}
                  onChange={event => {
                    const {rowSemantics: previous, ...remaining} = declarations;
                    update(
                      event.target.value
                        ? {
                            ...remaining,
                            rowSemantics: event.target.value as NonNullable<typeof previous>
                          }
                        : remaining
                    );
                  }}
                >
                  <option value="">
                    Use guess
                    {entry.profile.rowSemantics
                      ? ` (${entry.profile.rowSemantics.value})`
                      : ' / unknown'}
                  </option>
                  {GPU_ANALYSIS_ROW_SEMANTICS.map(value => (
                    <option key={value} value={value}>
                      {value}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Coordinate system
                <select
                  aria-label={`Coordinate system for ${dataset.label}`}
                  value={declarations.coordinateModel ?? ''}
                  onChange={event => {
                    const {coordinateModel: previous, ...remaining} = declarations;
                    update(
                      event.target.value
                        ? {
                            ...remaining,
                            coordinateModel: event.target.value as NonNullable<typeof previous>
                          }
                        : remaining
                    );
                  }}
                >
                  <option value="">
                    Use guess
                    {entry.profile.coordinates
                      ? ` (${entry.profile.coordinates.model})`
                      : ' / unknown'}
                  </option>
                  <option value="geographic">Geographic (degrees; distances in metres)</option>
                  <option value="planar">Planar (source coordinate units)</option>
                </select>
              </label>
              {entry.profile.columns.map(column => (
                <label key={column.name} style={{display: 'block'}}>
                  {column.name}:{' '}
                  {column.measurement
                    ? `${column.measurement} (${column.measurementSource ?? 'guess'})`
                    : 'measurement unknown'}
                  <select
                    aria-label={`Measurement of ${column.name} for ${dataset.label}`}
                    value={
                      declarations.measurement &&
                      Object.hasOwn(declarations.measurement, column.name)
                        ? declarations.measurement[column.name]
                        : ''
                    }
                    onChange={event => {
                      // Computed keys create own properties even for __proto__ and constructor.
                      const measurement = {
                        ...declarations.measurement,
                        [column.name]: event.target.value as NonNullable<typeof column.measurement>
                      };
                      if (!event.target.value) delete measurement[column.name];
                      update({...declarations, measurement});
                    }}
                  >
                    <option value="">Use guess / unknown</option>
                    {GPU_ANALYSIS_MEASUREMENT_LEVELS.map(value => (
                      <option key={value} value={value}>
                        {value}
                      </option>
                    ))}
                  </select>
                </label>
              ))}
              <button
                type="button"
                onClick={() => analysisDatasetContext.setDeclarations(dataset.id, undefined)}
              >
                Reset declarations
              </button>
            </fieldset>
          );
        })}
      </details>
    </section>
  );
}
