// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getGPUAnalysisQuestion} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';
import {getCatalogBinding} from '../analysis/catalog-bindings';
import {KEPLER_CURATED_ANALYSES} from '../curated/curated-analyses';
import {
  buildKeplerCuratedAnalysisRequest,
  createKeplerCuratedAnalysisMapping,
  updateKeplerCuratedAnalysisMapping,
  type KeplerCuratedAnalysisMapping
} from '../curated/curated-analysis-mapping';
import type {KeplerCuratedAnalysis} from '../curated/curated-analysis-types';
import {evaluateKeplerOperationApplicability} from '../descriptors/applicability';
import {resolveKeplerOperationDescriptor} from '../descriptors/operation-descriptors';
import {evaluateKeplerWorkbenchInputs} from '../descriptors/workbench-adapters';
import type {KeplerDatasetProfile} from '../profile/dataset-profile-types';
import {createCuratedWorkflowAnalysisTemplate} from './workflow-analysis-template';
import {searchWorkflowOperations, type WorkflowOperationMatch} from './workflow-operations';
import type {WorkflowOperationRegistry} from './workflow-types';

export type WorkflowAnalysisSelection = {
  analysis: KeplerCuratedAnalysis;
  template: ReturnType<typeof createCuratedWorkflowAnalysisTemplate>;
  mapping: KeplerCuratedAnalysisMapping;
};

/** Preserve meaningful slot assignments; anchor the active dataset only when it is absent. */
export function createWorkflowAnalysisSelection(
  analysis: KeplerCuratedAnalysis,
  profiles: readonly KeplerDatasetProfile[],
  activeDatasetId?: string
): WorkflowAnalysisSelection {
  let mapping = createKeplerCuratedAnalysisMapping(analysis, profiles);
  const activeSlot = mapping.slots.find(slot =>
    slot.candidates.some(
      candidate => candidate.datasetId === activeDatasetId && candidate.fit === 'fits'
    )
  );
  if (
    activeSlot &&
    activeDatasetId &&
    !mapping.slots.some(slot => slot.datasetId === activeDatasetId)
  ) {
    mapping = updateKeplerCuratedAnalysisMapping(
      mapping,
      {slot: activeSlot.slot, datasetId: activeDatasetId},
      analysis,
      profiles
    );
  }
  return {analysis, template: createCuratedWorkflowAnalysisTemplate(analysis), mapping};
}

type SearchResult = {
  id: string;
  title: string;
  summary: string;
  question: string;
  score: number;
  status: 'ready' | 'needs-mapping' | 'blocked';
  reason: string;
};

export type WorkflowAnalysisSearchMatch = SearchResult &
  (
    | {kind: 'template'; selection: WorkflowAnalysisSelection; available: boolean}
    | {kind: 'operation'; match: WorkflowOperationMatch}
  );

function scoreWords(query: string, primary: string, secondary: string): number | undefined {
  const words = query.toLowerCase().trim().split(/\s+/).filter(Boolean);
  const scores = words.map(word =>
    primary.toLowerCase().startsWith(word)
      ? 20
      : primary.toLowerCase().includes(word)
        ? 10
        : secondary.toLowerCase().includes(word)
          ? 1
          : 0
  );
  return scores.some(score => score === 0)
    ? undefined
    : scores.reduce<number>((total, score) => total + score, 0);
}

/** Questions resolve to concrete templates/operations, using the existing mapping and checks. */
export function searchWorkflowAnalyses(
  registry: WorkflowOperationRegistry,
  query: string,
  options: {
    profiles: readonly KeplerDatasetProfile[];
    activeDatasetId?: string;
    analyses?: readonly KeplerCuratedAnalysis[];
    executionEngine?: 'gpu' | 'cpu';
    limit?: number;
  }
): WorkflowAnalysisSearchMatch[] {
  const {profiles, activeDatasetId, executionEngine = 'gpu'} = options;
  const matches: WorkflowAnalysisSearchMatch[] = [];
  for (const analysis of options.analyses ?? KEPLER_CURATED_ANALYSES) {
    const question = getGPUAnalysisQuestion(analysis.question);
    const score = scoreWords(
      query,
      `${analysis.name} ${analysis.title} ${analysis.id}`,
      `${analysis.method} ${question.label} ${question.summary} ${question.examples.join(' ')}`
    );
    if (score === undefined) continue;
    const selection = createWorkflowAnalysisSelection(analysis, profiles, activeDatasetId);
    const {mapping} = selection;
    const execution = analysis.execution;
    const available =
      execution.status === 'available' && Boolean(registry.get(execution.bindingId));
    const usesActiveDataset =
      !activeDatasetId || mapping.slots.some(slot => slot.datasetId === activeDatasetId);
    let status: SearchResult['status'] = mapping.ready ? 'ready' : 'needs-mapping';
    let reason = mapping.problems
      .filter(problem => !problem.acknowledged)
      .map(problem => problem.message)
      .join('; ');
    if (available && execution.status === 'available') {
      const parameters = mapping.complete
        ? buildKeplerCuratedAnalysisRequest(analysis, mapping).parameters
        : mapping.parameterValues;
      const applicability = evaluateKeplerWorkbenchInputs(
        execution.bindingId,
        Object.fromEntries(
          mapping.slots.map(slot => [
            execution.inputs[slot.slot],
            profiles.find(profile => profile.datasetId === slot.datasetId)
          ])
        ),
        {executionEngine, overrides: parameters}
      );
      if (applicability.status === 'blocked') status = 'blocked';
      else if (applicability.status !== 'ready') status = 'needs-mapping';
      reason = [
        reason,
        ...applicability.missing.map(missing =>
          [missing.what, missing.remedy].filter(Boolean).join(' — ')
        )
      ]
        .filter(Boolean)
        .join('; ');
    } else {
      status = 'blocked';
      reason =
        execution.status === 'needs-binding'
          ? execution.reason
          : 'This template binding is not registered.';
    }
    if (!usesActiveDataset) {
      status = 'blocked';
      reason = 'The active dataset cannot fill a slot. Choose a compatible dataset.';
    }
    matches.push({
      kind: 'template',
      id: `template:${analysis.id}`,
      title: analysis.name,
      summary: analysis.title,
      question: question.label,
      score,
      status,
      reason: reason || 'Dataset slots and fields are mapped.',
      selection,
      available
    });
  }
  const lexicalMatches = new Map(
    searchWorkflowOperations(registry, query, {limit: Infinity}).map(match => [
      match.operation.id,
      match
    ])
  );
  for (const operation of registry.list()) {
    const binding = getCatalogBinding(operation.id);
    const descriptor = binding && resolveKeplerOperationDescriptor(binding).descriptor;
    const questions = descriptor?.questions.map(getGPUAnalysisQuestion) ?? [];
    const question = questions.map(question => question.label).join(' · ');
    const lexical = lexicalMatches.get(operation.id);
    const questionScore = scoreWords(
      query,
      '',
      questions
        .flatMap(question => [question.label, question.summary, ...question.examples])
        .join(' ')
    );
    if (!lexical && questionScore === undefined) continue;
    const applicability =
      binding && descriptor
        ? evaluateKeplerOperationApplicability(binding, descriptor, profiles, {
            requiredDatasetId: activeDatasetId,
            executionEngine
          })
        : undefined;
    matches.push({
      kind: 'operation',
      id: `operation:${operation.id}`,
      title: operation.title,
      summary: operation.summary,
      question,
      score: lexical?.score ?? questionScore ?? 0,
      status:
        applicability?.status === 'blocked'
          ? 'blocked'
          : applicability?.status === 'ready'
            ? 'ready'
            : 'needs-mapping',
      reason:
        applicability?.missing
          .map(missing => [missing.what, missing.remedy].filter(Boolean).join(' — '))
          .join('; ') || 'Configure this operation in the workflow inspector.',
      match: lexical ?? {operation, score: questionScore ?? 0}
    });
  }
  const order = {ready: 0, 'needs-mapping': 1, blocked: 2};
  return matches
    .sort(
      (left, right) =>
        Number(left.status === 'blocked') - Number(right.status === 'blocked') ||
        Number(right.kind === 'template') - Number(left.kind === 'template') ||
        right.score - left.score ||
        order[left.status] - order[right.status] ||
        left.id.localeCompare(right.id)
    )
    .slice(0, Math.max(0, options.limit ?? 60));
}
