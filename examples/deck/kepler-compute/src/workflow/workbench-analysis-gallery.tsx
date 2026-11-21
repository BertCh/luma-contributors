// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useId, useMemo, useRef, useState, useSyncExternalStore} from 'react';
import styled from 'styled-components';
import {GPU_ANALYSIS_QUESTION_RECORDS} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';
import {
  createKeplerCuratedAnalysisMapping,
  getKeplerCuratedAnalysis,
  listCuratedWorkflowAnalysisTemplates,
  replayKeplerCuratedAnalysisMapping,
  type KeplerCuratedAnalysisMapping,
  type WorkflowAnalysisTemplate
} from '@luma-contributors/kepler';
import {WorkbenchAnalysisMapping} from './workbench-analysis-mapping';
import {
  getWorkbenchAnalysisMappingIssues,
  getWorkbenchAnalysisProfiles,
  getWorkbenchMappingParameters,
  launchWorkbenchAnalysis,
  updateWorkbenchAnalysisMapping
} from './workbench-analysis-launch';
import {WorkbenchButton, WorkbenchField, WorkbenchSection} from './workbench-styles';
import {WORKBENCH_COLORS} from './workbench-contract';
import type {WorkbenchStore} from './workbench-store';

const Cards = styled.div`
  display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 10px;
  & article { padding: 10px; border: 1px solid ${WORKBENCH_COLORS.border}; }
  & h4 { margin: 0; font-size: 13px; }
`;
const noSubscription = () => () => {};
const zeroRevision = () => 0;

/** The shared curated templates, filed under the shared question vocabulary. */
export function WorkbenchAnalysisGallery({
  store: suppliedStore,
  loadStore
}: {
  store?: WorkbenchStore;
  loadStore?: () => Promise<WorkbenchStore>;
}) {
  const identifier = useId();
  const [loadedStore, setLoadedStore] = useState<WorkbenchStore>();
  const store = suppliedStore ?? loadedStore;
  useSyncExternalStore(
    store?.subscribe ?? noSubscription,
    store ? () => store.getState().revision : zeroRevision,
    zeroRevision
  );
  const templates = useMemo(listCuratedWorkflowAnalysisTemplates, []);
  const [question, setQuestion] = useState('');
  const [selection, setSelection] = useState<{
    template: WorkflowAnalysisTemplate;
    mapping: KeplerCuratedAnalysisMapping;
  }>();
  const [busy, setBusy] = useState(false);
  const running = useRef(false);
  const [error, setError] = useState('');
  const profiles = store ? getWorkbenchAnalysisProfiles(store) : [];
  const analysis = selection?.template.curatedAnalysisId
    ? getKeplerCuratedAnalysis(selection.template.curatedAnalysisId)
    : undefined;
  const mapping =
    analysis && selection
      ? replayKeplerCuratedAnalysisMapping(analysis, selection.mapping.choices, profiles, {
          parameters: getWorkbenchMappingParameters(selection.mapping)
        })
      : undefined;

  async function getStore(): Promise<WorkbenchStore> {
    if (store) return store;
    if (!loadStore) throw new Error('WorkbenchAnalysisGallery: a workflow host is required.');
    const loaded = await loadStore();
    setLoadedStore(loaded);
    return loaded;
  }

  async function perform(action: () => Promise<void>): Promise<void> {
    if (running.current) return;
    running.current = true;
    setBusy(true);
    setError('');
    try {
      await action();
    } catch (failure) {
      const message = failure instanceof Error ? failure.message : String(failure);
      setError(message);
      store?.setNotice({kind: 'error', message});
    } finally {
      running.current = false;
      setBusy(false);
    }
  }

  async function refineProfiles(
    workbench: WorkbenchStore,
    datasetIds: readonly string[]
  ): Promise<void> {
    const {analysisDatasetContext} = await import('../analysis-dataset-context');
    const results = await Promise.allSettled(
      [...new Set(datasetIds)]
        .filter(datasetId => analysisDatasetContext.get(datasetId))
        .map(datasetId => analysisDatasetContext.readStatistics(datasetId))
    );
    const failures = results.flatMap(result =>
      result.status === 'rejected' ? [String(result.reason)] : []
    );
    // A test/custom host may already supply full profiles outside the native context. The plan
    // still checks their evidence; a native refinement error stays visible to the user.
    if (failures.length) setError(failures.join('\n'));
    workbench.notifyExternalChange();
  }

  return (
    <WorkbenchSection data-testid="ax-workflow-gallery" aria-label="Workflows by question">
      <h3>What would you like to find out?</h3>
      {error && <p role="alert">{error}</p>}
      {selection && analysis && mapping && store ? (
        <WorkbenchAnalysisMapping
          analysis={analysis}
          mapping={mapping}
          profiles={profiles}
          busy={busy}
          executionIssues={getWorkbenchAnalysisMappingIssues(selection.template, store, mapping)}
          onChange={change => {
            const changed = updateWorkbenchAnalysisMapping(mapping, change, analysis, profiles);
            setSelection({...selection, mapping: changed});
            if ('datasetId' in change)
              void perform(() => refineProfiles(store, [change.datasetId]));
          }}
          onParametersChange={parameters =>
            setSelection({
              ...selection,
              mapping: replayKeplerCuratedAnalysisMapping(analysis, mapping.choices, profiles, {
                parameters
              })
            })
          }
          onRowSemanticsChange={(datasetId, semantics) =>
            void perform(async () => {
              const {analysisDatasetContext} = await import('../analysis-dataset-context');
              analysisDatasetContext.setDeclarations(datasetId, {
                ...analysisDatasetContext.getDeclarations(datasetId),
                rowSemantics: semantics
              });
              store.notifyExternalChange();
            })
          }
          onRun={() =>
            void perform(async () => {
              await launchWorkbenchAnalysis(selection.template, store, mapping);
            })
          }
          onCancel={() => setSelection(undefined)}
        />
      ) : (
        <>
          <WorkbenchField>
            <label htmlFor={`${identifier}-question`}>Browse by question</label>
            <select
              id={`${identifier}-question`}
              data-testid="ax-workflow-question"
              value={question}
              disabled={busy}
              onChange={event => setQuestion(event.target.value)}
            >
              <option value="">All questions</option>
              {GPU_ANALYSIS_QUESTION_RECORDS.map(record => (
                <option value={record.id} key={record.id}>
                  {record.label}
                </option>
              ))}
            </select>
          </WorkbenchField>
          {GPU_ANALYSIS_QUESTION_RECORDS.filter(record => !question || record.id === question).map(
            record => {
              const matching = templates.filter(template => template.question === record.id);
              return (
                matching.length > 0 && (
                  <section key={record.id} aria-label={record.label}>
                    <h3>{record.label}</h3>
                    <Cards>
                      {matching.map(template => {
                        const curated = getKeplerCuratedAnalysis(template.curatedAnalysisId!);
                        const available = curated?.execution.status === 'available';
                        const sampleAvailable = available && Boolean(template.sample);
                        const reason =
                          curated?.execution.status === 'needs-binding'
                            ? curated.execution.reason
                            : !sampleAvailable
                              ? 'No licensed sample is registered.'
                              : undefined;
                        return (
                          <article data-testid={`ax-workflow-${template.id}`} key={template.id}>
                            <h4>{template.title}</h4>
                            <p>{template.method}</p>
                            <p>Needs: {template.slots.map(slot => slot.label).join(', ')}</p>
                            {template.sample && (
                              <p>
                                Sample: {template.sample.id} · {template.sample.license} ·{' '}
                                {template.sample.attribution}
                              </p>
                            )}
                            {reason && <p>{reason}</p>}
                            <WorkbenchButton
                              data-testid={`ax-sample-run-${template.id}`}
                              disabled={busy || !sampleAvailable}
                              title={reason}
                              onClick={() =>
                                void perform(async () => {
                                  await launchWorkbenchAnalysis(template, await getStore());
                                })
                              }
                            >
                              Run sample
                            </WorkbenchButton>{' '}
                            <WorkbenchButton
                              data-testid={`ax-use-data-${template.id}`}
                              disabled={busy || !available}
                              title={!available ? reason : undefined}
                              onClick={() =>
                                void perform(async () => {
                                  const workbench = await getStore();
                                  const initial = createKeplerCuratedAnalysisMapping(
                                    curated!,
                                    getWorkbenchAnalysisProfiles(workbench)
                                  );
                                  await refineProfiles(
                                    workbench,
                                    initial.slots.flatMap(slot =>
                                      slot.datasetId ? [slot.datasetId] : []
                                    )
                                  );
                                  setSelection({
                                    template,
                                    mapping: replayKeplerCuratedAnalysisMapping(
                                      curated!,
                                      initial.choices,
                                      getWorkbenchAnalysisProfiles(workbench)
                                    )
                                  });
                                })
                              }
                            >
                              Use my data
                            </WorkbenchButton>
                            <details aria-label={`About ${template.title}`}>
                              <summary>What the result means</summary>
                              <p>{template.reads.summary}</p>
                              {template.reads.cautions.map((caution, index) => (
                                <p key={index}>{caution}</p>
                              ))}
                            </details>
                          </article>
                        );
                      })}
                    </Cards>
                  </section>
                )
              );
            }
          )}
        </>
      )}
    </WorkbenchSection>
  );
}
