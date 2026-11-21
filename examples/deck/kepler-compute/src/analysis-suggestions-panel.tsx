// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useMemo, useState, useSyncExternalStore} from 'react';
import {GPU_ANALYSIS_MEASUREMENT_LEVELS} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';
import {analysisDatasetContext} from './analysis-dataset-context';
import {makeCatalogPointColumns} from './analysis-catalog-input';
import type {CatalogResidentDataset} from './analysis-catalog-resident-input';
import {
  answerAnalysisSuggestionQuestion,
  browseAnalysisSuggestions,
  createAnalysisSuggestionLaunch,
  createAnalysisSuggestionModel,
  type AnalysisSuggestionLaunch
} from './analysis-suggestion-model';

/** Suggestions belong to a loaded dataset, while paired analyses see every current profile. */
export function AnalysisSuggestionsPanel({
  datasetId,
  datasets,
  onLaunch
}: {
  datasetId: string;
  datasets: readonly CatalogResidentDataset[];
  onLaunch(launch: AnalysisSuggestionLaunch): void;
}) {
  const version = useSyncExternalStore(
    analysisDatasetContext.subscribe,
    analysisDatasetContext.getVersion,
    analysisDatasetContext.getVersion
  );
  const [browsing, setBrowsing] = useState(false);
  const [error, setError] = useState('');
  const profiles = datasets.flatMap(dataset => {
    const entry = analysisDatasetContext.get(dataset.id);
    return entry ? [entry.profile] : [];
  });
  const profileKey = JSON.stringify(profiles);
  const model = useMemo(
    () => createAnalysisSuggestionModel(datasetId, profiles),
    [datasetId, profileKey, version]
  );
  const groups = useMemo(
    () => (browsing ? browseAnalysisSuggestions(datasetId, profiles) : []),
    [browsing, datasetId, profileKey, version]
  );
  const dataset = datasets.find(candidate => candidate.id === datasetId);
  if (!dataset || !analysisDatasetContext.get(datasetId)) return null;
  const question = model.questions[0];
  const launch = (target: Parameters<typeof createAnalysisSuggestionLaunch>[0]) => {
    try {
      const currentProfiles = datasets.flatMap(source => {
        const entry = analysisDatasetContext.get(source.id);
        return entry ? [entry.profile] : [];
      });
      onLaunch(createAnalysisSuggestionLaunch(target, currentProfiles));
      setError('');
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure));
    }
  };
  return (
    <section
      data-testid="ax-suggestions"
      data-dataset-id={datasetId}
      aria-label={`Suggested for ${dataset.label}`}
    >
      <h3>Suggested for this data</h3>
      {model.suggestions.length === 0 && (
        <p>No suggestions yet. Review the dataset meaning or browse compatible operations.</p>
      )}
      {!profiles.find(profile => profile.datasetId === datasetId)?.pointColumns &&
        !profiles.find(profile => profile.datasetId === datasetId)?.geometryColumns.length && (
          <p>
            No geometry or coordinate pair is available to the catalog. Prepare supported coordinate
            columns or a GeoJSON geometry column to enable spatial suggestions.
          </p>
        )}
      {question && (
        <label>
          {question.prompt}
          <select
            aria-label={question.prompt}
            value=""
            onChange={event => {
              if (!event.target.value) return;
              analysisDatasetContext.setDeclarations(
                question.datasetId,
                answerAnalysisSuggestionQuestion(
                  question,
                  event.target.value,
                  analysisDatasetContext.getDeclarations(question.datasetId)
                )
              );
            }}
          >
            <option value="">
              Choose an answer for{' '}
              {profiles.find(profile => profile.datasetId === question.datasetId)?.label}
            </option>
            {question.options.map(option => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </label>
      )}
      <div aria-label="Column role chips" style={{display: 'flex', flexWrap: 'wrap', gap: 6}}>
        {profiles
          .find(profile => profile.datasetId === datasetId)
          ?.columns.filter(column => column.type === 'number' || column.type === 'integer')
          .map(column => (
            <label
              key={column.name}
              style={{border: '1px solid currentColor', borderRadius: 12, padding: '2px 6px'}}
            >
              {column.name} ({column.measurementSource ?? 'guess'})
              <select
                aria-label={`Role of ${column.name} for ${dataset.label}`}
                value={column.measurement ?? ''}
                onChange={event => {
                  if (!event.target.value) return;
                  analysisDatasetContext.setDeclarations(
                    datasetId,
                    answerAnalysisSuggestionQuestion(
                      {
                        id: 'measurement',
                        datasetId,
                        column: column.name,
                        prompt: '',
                        options: GPU_ANALYSIS_MEASUREMENT_LEVELS,
                        affects: []
                      },
                      event.target.value,
                      analysisDatasetContext.getDeclarations(datasetId)
                    )
                  );
                }}
              >
                <option value="">Measurement unknown</option>
                {GPU_ANALYSIS_MEASUREMENT_LEVELS.map(value => (
                  <option key={value} value={value}>
                    {value}
                  </option>
                ))}
              </select>
            </label>
          ))}
      </div>
      <ul style={{paddingLeft: 18}}>
        {model.suggestions.map(suggestion => (
          <li key={suggestion.id} style={{marginBottom: 12}}>
            <button
              type="button"
              data-suggestion-id={suggestion.id}
              data-binding-id={suggestion.bindingId}
              data-input-assignments={JSON.stringify(suggestion.applicability.inputAssignments)}
              data-field-bindings={JSON.stringify(suggestion.applicability.fieldBindings)}
              data-input-mappings={JSON.stringify(
                Object.fromEntries(
                  Object.entries(suggestion.applicability.inputAssignments).map(
                    ([inputName, assignedId]) => {
                      const profile = profiles.find(
                        candidate => candidate.datasetId === assignedId
                      );
                      return [
                        inputName,
                        {
                          mode: 'native',
                          datasetId: assignedId ?? '',
                          geometry: profile?.pointColumns
                            ? makeCatalogPointColumns('', '')
                            : (profile?.geometryColumns[0] ?? ''),
                          longitude: profile?.pointColumns?.longitude ?? null,
                          latitude: profile?.pointColumns?.latitude ?? null
                        }
                      ];
                    }
                  )
                )
              )}
              onClick={() => launch(suggestion)}
            >
              {suggestion.title}
            </button>
            <p>{suggestion.reasons.join('; ')}</p>
            {Object.entries(suggestion.applicability.fieldBindings).length > 0 && (
              <p>
                Fields:{' '}
                {Object.entries(suggestion.applicability.fieldBindings)
                  .map(
                    ([role, columns]) =>
                      `${role}: ${typeof columns === 'string' ? columns : columns.join(', ')}`
                  )
                  .join('; ')}
              </p>
            )}
            <small>
              {suggestion.status === 'ready'
                ? 'Ready to configure'
                : 'Review mapping or confirmations in the form'}
            </small>
            {suggestion.applicability.missing.map((missing, index) => (
              <p key={index}>
                {missing.what} {missing.remedy}
              </p>
            ))}
            {suggestion.applicability.warnings.map((warning, index) => (
              <p key={index}>{warning}</p>
            ))}
            {suggestion.notes.map((note, index) => (
              <p key={index}>{note}</p>
            ))}
          </li>
        ))}
        {model.blocked.map(suggestion => (
          <li key={suggestion.id} style={{marginBottom: 12, opacity: 0.65}}>
            <button type="button" disabled>
              {suggestion.title}
            </button>
            <p>{suggestion.reasons.join('; ')}</p>
            {suggestion.applicability.missing.map((missing, index) => (
              <p key={index}>
                {missing.what} {missing.remedy}
              </p>
            ))}
          </li>
        ))}
      </ul>
      <p>
        Correct role, row-meaning and coordinate guesses in Dataset meaning and analysis limits.
      </p>
      <details onToggle={event => setBrowsing(event.currentTarget.open)}>
        <summary>Browse all that fit</summary>
        {browsing && groups.length === 0 && <p>No catalog operations fit the current profile.</p>}
        {groups.map(group => (
          <div key={group.question}>
            <h4>{group.label}</h4>
            <ul style={{paddingLeft: 18}}>
              {group.bindings.map(candidate => (
                <li key={candidate.bindingId}>
                  <button
                    type="button"
                    data-binding-id={candidate.bindingId}
                    data-input-assignments={JSON.stringify(
                      candidate.applicability.inputAssignments
                    )}
                    onClick={() => launch(candidate)}
                  >
                    {candidate.title}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </details>
      {model.dropped.map(message => (
        <p key={message}>{message}</p>
      ))}
      {error && <p role="alert">{error}</p>}
    </section>
  );
}
