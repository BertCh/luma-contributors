// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRequirement} from '@luma-contributors/analysis/analysis-core';
import {
  KEPLER_CATALOG_BINDINGS,
  KEPLER_CURATED_ANALYSES,
  getCatalogBinding,
  listKeplerApplicableBindings,
  suggestKeplerAnalyses,
  type CatalogParameterValue,
  type KeplerDatasetProfile,
  type KeplerDatasetProfileDeclarations,
  type KeplerOperationApplicability,
  type KeplerSuggestionQuestion
} from '@luma-contributors/kepler';
import {makeCatalogPointColumns} from './analysis-catalog-input';
import {analysisDatasetContext, type AnalysisDatasetContextEntry} from './analysis-dataset-context';
import type {AnalysisCatalogInputDraft} from './analysis-catalog-form-model';

export type AnalysisSuggestionSnapshot = {
  entry: AnalysisDatasetContextEntry;
  owner: object;
  provider: AnalysisDatasetContextEntry['dataset']['provider'];
  readRevision: string;
  key: string;
};

export type AnalysisSuggestionLaunch = {
  bindingId: string;
  suggestionId: string;
  drafts: Record<string, AnalysisCatalogInputDraft>;
  overrides: Record<string, CatalogParameterValue>;
  revisions: Record<string, KeplerDatasetProfile['revision']>;
  snapshots: Record<string, AnalysisSuggestionSnapshot>;
};

/** All loaded profiles participate, including partners outside this dataset card. */
export function createAnalysisSuggestionModel(
  datasetId: string,
  profiles: readonly KeplerDatasetProfile[],
  executionEngine: 'cpu' | 'gpu' = 'gpu'
) {
  const result = suggestKeplerAnalyses(KEPLER_CATALOG_BINDINGS, profiles, {
    requiredDatasetId: datasetId,
    maximumCount: 8,
    executionEngine
  });
  const suggestions = result.suggestions.filter(suggestion =>
    suggestion.datasetIds.includes(datasetId)
  );
  const blocked = result.blocked.filter(suggestion => suggestion.datasetIds.includes(datasetId));
  const relevantIds = new Set([...suggestions, ...blocked].map(suggestion => suggestion.id));
  return {
    suggestions,
    blocked,
    questions: result.questions.filter(question =>
      question.affects.some(identifier => relevantIds.has(identifier))
    ),
    dropped: result.dropped
  };
}

/** Browse is lazy and uses the same applicability assignments as the ranked suggestions. */
export function browseAnalysisSuggestions(
  datasetId: string,
  profiles: readonly KeplerDatasetProfile[]
) {
  return listKeplerApplicableBindings(KEPLER_CATALOG_BINDINGS, profiles, {
    requiredDatasetId: datasetId
  })
    .map(group => ({
      ...group,
      bindings: group.bindings.filter(candidate =>
        Object.values(candidate.applicability.inputAssignments).includes(datasetId)
      )
    }))
    .filter(group => group.bindings.length > 0);
}

/** Unfilled inputs stay empty native mappings: opening must never substitute fixture data. */
export function createAnalysisSuggestionLaunch(
  target: {
    bindingId: string;
    id?: string;
    curatedAnalysisId?: string;
    applicability: KeplerOperationApplicability;
  },
  profiles: readonly KeplerDatasetProfile[],
  getEntry: (
    datasetId: string
  ) => AnalysisDatasetContextEntry | undefined = analysisDatasetContext.get
): AnalysisSuggestionLaunch {
  const binding = getCatalogBinding(target.bindingId);
  validateRequirement('ax-suggestions', Boolean(binding), 'bindingId must name a catalog binding');
  validateRequirement(
    'ax-suggestions',
    target.applicability.status !== 'blocked',
    'applicability must be unblocked before opening an analysis'
  );
  const drafts: AnalysisSuggestionLaunch['drafts'] = {};
  const revisions: AnalysisSuggestionLaunch['revisions'] = {};
  const snapshots: AnalysisSuggestionLaunch['snapshots'] = {};
  for (const input of binding!.inputs) {
    const datasetId = target.applicability.inputAssignments[input.name];
    const profile = profiles.find(candidate => candidate.datasetId === datasetId);
    validateRequirement(
      'ax-suggestions',
      !datasetId || Boolean(profile),
      `${input.name} dataset must have a current profile`
    );
    const field = profile?.pointColumns
      ? makeCatalogPointColumns(profile.pointColumns.longitude, profile.pointColumns.latitude)
      : (profile?.geometryColumns[0] ?? '');
    drafts[`${binding!.id}:${input.name}`] = {
      mode: 'native',
      datasetId: datasetId ?? '',
      field,
      text: ''
    };
    if (profile) {
      const entry = getEntry(profile.datasetId);
      validateRequirement(
        'ax-suggestions',
        entry?.profile === profile,
        `${input.name} profile must belong to the current dataset snapshot`
      );
      revisions[profile.datasetId] = profile.revision;
      snapshots[profile.datasetId] = {
        entry: entry!,
        owner: entry!.dataset.owner ?? entry!.dataset.provider,
        provider: entry!.dataset.provider,
        readRevision: entry!.readRevision,
        key: entry!.key
      };
    }
  }
  const curated = KEPLER_CURATED_ANALYSES.find(
    analysis => analysis.id === target.curatedAnalysisId
  );
  const overrides: AnalysisSuggestionLaunch['overrides'] = {
    ...(curated?.execution.status === 'available' ? curated.execution.parameters : {})
  };
  for (const [parameter, columns] of Object.entries(target.applicability.fieldBindings))
    overrides[parameter] = typeof columns === 'string' ? columns : columns.join(',');
  return {
    bindingId: binding!.id,
    suggestionId: target.id ?? `binding:${binding!.id}`,
    drafts,
    overrides,
    revisions,
    snapshots
  };
}

/** Numeric revisions alone cannot identify a replacement source or provider snapshot. */
export function isAnalysisSuggestionLaunchCurrent(
  launch: AnalysisSuggestionLaunch,
  getEntry: (
    datasetId: string
  ) => AnalysisDatasetContextEntry | undefined = analysisDatasetContext.get
): boolean {
  return Object.entries(launch.revisions).every(([datasetId, revision]) => {
    const entry = getEntry(datasetId);
    const snapshot = launch.snapshots[datasetId];
    return Boolean(
      entry &&
        snapshot &&
        entry === snapshot.entry &&
        entry.profile.revision === revision &&
        entry.key === snapshot.key &&
        entry.dataset.provider === snapshot.provider &&
        (entry.dataset.owner ?? entry.dataset.provider) === snapshot.owner &&
        entry.readRevision === snapshot.readRevision &&
        entry.dataset.provider.getRevision() === snapshot.readRevision &&
        entry.dataset.isCurrent()
    );
  });
}

/** Only the answered declaration changes; the host context owns persistence and refinement. */
export function answerAnalysisSuggestionQuestion(
  question: KeplerSuggestionQuestion,
  answer: string,
  declarations: KeplerDatasetProfileDeclarations = {}
): KeplerDatasetProfileDeclarations {
  validateRequirement(
    'ax-suggestions',
    question.options.includes(answer),
    'answer must be one of the question options'
  );
  if (question.id === 'row-semantics')
    return {
      ...declarations,
      rowSemantics: answer as KeplerDatasetProfileDeclarations['rowSemantics']
    };
  if (question.id === 'coordinate-model')
    return {
      ...declarations,
      coordinateModel: answer as KeplerDatasetProfileDeclarations['coordinateModel']
    };
  validateRequirement('ax-suggestions', Boolean(question.column), 'measurement must name a column');
  return {
    ...declarations,
    measurement: {
      ...declarations.measurement,
      [question.column!]: answer as NonNullable<
        KeplerDatasetProfileDeclarations['measurement']
      >[string]
    }
  };
}
