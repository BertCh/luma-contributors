// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useId, useMemo, useState} from 'react';
import styled from 'styled-components';
import {
  addWorkflowStep,
  applyWorkflowRecommendation,
  getWorkflowRecommendationSubject,
  instantiateWorkflowTemplate,
  listWorkflowTemplates,
  makeUniqueWorkflowId,
  recommendWorkflowOperations,
  updateWorkflowInput,
  WORKFLOW_LIMITS,
  WorkflowError,
  type WorkflowDatasetProfile,
  type WorkflowDocument,
  type WorkflowInput,
  type WorkflowRecommendation,
  type WorkflowRunnerState,
  type WorkflowRunRecord,
  type WorkflowStep,
  type WorkflowStepState
} from '@luma-contributors/kepler';
import {useWorkbenchState} from './use-workbench';
import {WORKBENCH_COLORS, WORKBENCH_STATUS, WORKBENCH_TEST_IDS} from './workbench-contract';
import type {WorkbenchStore} from './workbench-store';
import {
  WorkbenchButton,
  WorkbenchField,
  WorkbenchMuted,
  WorkbenchSection,
  WorkbenchStatusBadge,
  WorkbenchSurface
} from './workbench-styles';

const Panel = styled(WorkbenchSurface)`
  width: 100%;
  min-width: 0;
  background: transparent;
  & h2 { margin: 0 0 4px; font-size: 14px; }
  & p { margin: 0 0 8px; }
  & section { min-width: 0; }
  & input, & select { min-height: 24px; }
  & input[type='checkbox'] { min-width: 24px; }
  & select { text-overflow: ellipsis; }
  & .error { overflow-wrap: anywhere; }
  & .unit {
    max-width: 50%;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
`;

const Row = styled.div`
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 6px;
  min-width: 0;
  & > select { flex: 1 1 140px; width: 0; }
`;

const Truncated = styled.span`
  display: block;
  min-width: 0;
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
`;

const StepList = styled.ol`
  display: grid;
  gap: 8px;
  margin: 0;
  padding: 0;
  list-style: none;
  & > li { min-width: 0; }
`;

const StepButton = styled(WorkbenchButton)`
  display: grid;
  gap: 4px;
  width: 100%;
  height: auto;
  min-height: 44px;
  padding: 6px;
  text-align: left;
  &[aria-current='true'] { border-color: ${WORKBENCH_COLORS.accent}; }
`;

const StepActions = styled(Row)`
  justify-content: flex-end;
  margin-top: 3px;
`;

const Problem = styled.div`
  color: ${WORKBENCH_COLORS.danger};
  overflow-wrap: anywhere;
  margin-top: 3px;
`;

const PublicationProblem = styled(Problem)`
  color: ${WORKBENCH_COLORS.warning};
`;

const TemplateList = styled.ul`
  display: grid;
  gap: 6px;
  list-style: none;
  margin: 8px 0;
  padding: 0;
  & > li { min-width: 0; }
  & button { width: 100%; justify-content: flex-start; }
  & p { margin: 3px 0; }
`;

const QuickStartDetails = styled.details`
  min-width: 0;
  & > summary {
    min-height: 24px;
    cursor: pointer;
  }
  & > summary:focus-visible {
    outline: 2px solid ${WORKBENCH_COLORS.focus};
    outline-offset: 1px;
  }
`;

type QuickStartSource = Parameters<typeof addWorkflowStep>[2];

export function getSidePanelDatasetSource(dataset: WorkflowDatasetProfile): QuickStartSource {
  const geometryField = dataset.geometryFields[0];
  return {
    operationId: 'source-dataset',
    label: (dataset.label.trim() || 'Dataset').slice(0, WORKFLOW_LIMITS.maximumTitleCharacters),
    parameters: {
      dataset: dataset.id,
      datasetLabel: dataset.label,
      ...(geometryField
        ? {geometryMode: 'geojson-field', geometryField}
        : dataset.pointColumns
          ? {
              geometryMode: 'point-columns',
              longitudeField: dataset.pointColumns.longitude,
              latitudeField: dataset.pointColumns.latitude
            }
          : {geometryMode: 'none'})
    }
  };
}

function startSidePanelWorkflow(
  store: WorkbenchStore,
  source: QuickStartSource,
  title: string,
  recommendation?: WorkflowRecommendation
): void {
  // Replay needs the exact host label; reject unsupported source values before creating anything.
  try {
    store.registry.get(source.operationId)?.normalizeParameters(source.parameters ?? {});
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    const guidance =
      error instanceof WorkflowError && error.details.parameter === 'datasetLabel'
        ? ' Rename the dataset to use a shorter label, then try again.'
        : '';
    store.setNotice({kind: 'error', message: message + guidance});
    return;
  }
  if (!store.getState().document && !store.createWorkflow({title})) return;
  const document = store.getState().document;
  if (!document) return;
  const stepId = makeUniqueWorkflowId(
    source.operationId,
    document.steps.map(step => step.id)
  );
  let selectedStepId = stepId;
  if (
    !store.edit(`quick-start:${stepId}`, current => {
      const added = addWorkflowStep(current, store.registry, {...source, id: stepId});
      if (!recommendation) return added.document;
      const analysis = applyWorkflowRecommendation(added.document, store.registry, recommendation, {
        stepId,
        output: 'data'
      });
      selectedStepId = analysis.stepId;
      return analysis.document;
    })
  )
    return;
  store.select([selectedStepId]);
  store.setDock({open: true, inspectorTab: 'parameters'});
  if (!recommendation) store.openPalette({connectFrom: {stepId, output: 'data'}});
}

function DatasetRecommendations({
  dataset,
  store
}: {
  dataset: WorkflowDatasetProfile;
  store: WorkbenchStore;
}): React.JSX.Element {
  const [open, setOpen] = useState(false);
  const getRecommendations = useMemo(() => {
    const cache = new WeakMap<WorkflowDatasetProfile, WorkflowRecommendation[]>();
    return (profile: WorkflowDatasetProfile) => {
      let recommendations = cache.get(profile);
      if (!recommendations) {
        recommendations = recommendWorkflowOperations(
          store.registry,
          getWorkflowRecommendationSubject(profile),
          {limit: 5}
        );
        cache.set(profile, recommendations);
      }
      return recommendations;
    };
  }, [store.registry]);
  const recommendations = open ? getRecommendations(dataset) : [];
  return (
    <QuickStartDetails
      aria-label={`Suggested analyses for ${dataset.label}`}
      open={open}
      onToggle={event => setOpen(event.currentTarget.open)}
    >
      <summary>Suggested analyses</summary>
      {open && (
        <TemplateList aria-label={`Suggested analyses for ${dataset.label}`}>
          {recommendations.map(recommendation => (
            <li key={recommendation.id}>
              <WorkbenchButton
                type="button"
                data-testid={WORKBENCH_TEST_IDS.recommendation(
                  `${dataset.id}-${recommendation.id}`
                )}
                title={recommendation.label}
                onClick={() => {
                  const source = getSidePanelDatasetSource(dataset);
                  startSidePanelWorkflow(store, source, source.label ?? 'Dataset', recommendation);
                }}
              >
                <Truncated>{recommendation.label}</Truncated>
              </WorkbenchButton>
              <Truncated as="p" title={recommendation.reason}>
                <WorkbenchMuted>{recommendation.reason}</WorkbenchMuted>
              </Truncated>
            </li>
          ))}
          {!recommendations.length && (
            <li>
              <WorkbenchMuted>No suggested analyses for this dataset.</WorkbenchMuted>
            </li>
          )}
        </TemplateList>
      )}
    </QuickStartDetails>
  );
}

/** Keep runner order, retaining placeholders if a runner has not yet seen the document. */
export function getSidePanelSteps(
  document: WorkflowDocument | null,
  runner: WorkflowRunnerState | null
): WorkflowStep[] {
  const steps = new Map(document?.steps.map(step => [step.id, step]));
  const ordered: WorkflowStep[] = [];
  for (const stepId of runner?.order ?? []) {
    const step = steps.get(stepId);
    if (step) ordered.push(step);
    steps.delete(stepId);
  }
  return [...ordered, ...steps.values()];
}

export function getSidePanelRunLabel(runner: WorkflowRunnerState | null): string {
  if (!runner) return 'Run';
  if (runner.running) return 'Running';
  const count = runner.order.filter(stepId => {
    const status = runner.steps[stepId]?.status;
    return status === 'ready' || status === 'stale' || status === 'error';
  }).length;
  return count ? `Run ${count.toLocaleString()} ${count === 1 ? 'step' : 'steps'}` : 'Up to date';
}

export function formatSidePanelDuration(milliseconds: number): string {
  return milliseconds < 1000
    ? `${Math.round(milliseconds).toLocaleString()} ms`
    : `${(milliseconds / 1000).toLocaleString(undefined, {
        minimumFractionDigits: 1,
        maximumFractionDigits: 1
      })} s`;
}

export function getSidePanelRunSummary(record: WorkflowRunRecord | undefined): string {
  if (!record) return '';
  if (record.status === 'error')
    return `Failed: ${record.steps.find(step => step.error)?.error?.message ?? 'Workflow could not run.'}`;
  return `${record.status === 'done' ? 'Done' : 'Cancelled'} in ${formatSidePanelDuration(record.durationMilliseconds)}`;
}

/** Empty numeric tokens stay invalid rather than silently becoming zero. */
export function parseSidePanelInputText(
  input: WorkflowInput,
  text: string
): WorkflowInput['value'] {
  const parseNumber = (token: string) => (token.trim() ? Number(token) : Number.NaN);
  if (input.kind === 'number' || input.kind === 'integer') return parseNumber(text);
  if (input.kind === 'number-list') return text.split(',').map(parseNumber);
  return text;
}

export function getSidePanelInputProblem(
  input: WorkflowInput,
  value: WorkflowInput['value'] = input.value
): string | null {
  const describeProblem = (problem: string) => `${input.label}: ${problem}.`;
  if (input.kind === 'number' || input.kind === 'integer' || input.kind === 'number-list') {
    const numbers = Array.isArray(value) ? value : [value];
    if (
      (input.kind === 'number-list' && (!Array.isArray(value) || !value.length)) ||
      (input.kind !== 'number-list' && Array.isArray(value)) ||
      numbers.some(number => typeof number !== 'number' || !Number.isFinite(number))
    )
      return describeProblem(
        input.kind === 'number-list'
          ? 'Enter a non-empty list of finite numbers'
          : 'Enter a finite number'
      );
    for (const number of numbers as number[]) {
      if (input.kind === 'integer' && !Number.isInteger(number))
        return describeProblem(`Enter a whole number; got ${number.toLocaleString()}`);
      const unit = input.unit ? ` ${input.unit}` : '';
      if (
        (input.minimum !== undefined && number < input.minimum) ||
        (input.maximum !== undefined && number > input.maximum)
      ) {
        const range =
          input.minimum !== undefined && input.maximum !== undefined
            ? `between ${input.minimum.toLocaleString()} and ${input.maximum.toLocaleString()}`
            : input.minimum !== undefined
              ? `at least ${input.minimum.toLocaleString()}`
              : `at most ${input.maximum!.toLocaleString()}`;
        return describeProblem(`Value must be ${range}${unit}; got ${number.toLocaleString()}`);
      }
    }
  } else if (input.kind === 'boolean' && typeof value !== 'boolean') {
    return describeProblem('Choose on or off');
  } else if (
    input.kind === 'enum' &&
    (typeof value !== 'string' || !input.options?.includes(value))
  ) {
    return describeProblem(`Choose one of ${input.options?.join(', ') || '(no options)'}`);
  } else if (input.kind === 'string' && typeof value !== 'string') {
    return describeProblem('Enter text');
  }
  return null;
}

function equalInputValues(left: WorkflowInput['value'], right: WorkflowInput['value']): boolean {
  return Array.isArray(left) && Array.isArray(right)
    ? left.length === right.length && left.every((value, index) => Object.is(value, right[index]))
    : Object.is(left, right);
}

export function getSidePanelDisplayLabel(
  step: WorkflowStep,
  state: WorkflowStepState | undefined
): string | null {
  const requested = step.display ?? [];
  if (!requested.length) return null;
  const published = requested.flatMap(output => state?.published?.[output] ?? []);
  if (!published.length) return 'Display requested';
  if (published.some(output => output.key !== state?.key))
    return 'Previous result displayed on map';
  return published.length === requested.length
    ? 'Displayed on map'
    : 'Some outputs displayed on map';
}

function WorkflowInputField({
  input,
  store
}: {
  input: WorkflowInput;
  store: WorkbenchStore;
}): React.JSX.Element {
  const fieldId = useId();
  const [draft, setDraft] = useState<{text: string; storedValue: WorkflowInput['value']} | null>(
    null
  );
  // Retain partial input only while the stored value still matches this field's latest edit.
  const activeDraft = draft && equalInputValues(draft.storedValue, input.value) ? draft : null;
  if (draft && !activeDraft) setDraft(null);
  const typedText = activeDraft?.text ?? null;
  const text =
    typedText ?? (Array.isArray(input.value) ? input.value.join(', ') : String(input.value));
  const value = typedText === null ? input.value : parseSidePanelInputText(input, typedText);
  const problem = getSidePanelInputProblem(input, value);
  const problemId = `${fieldId}-problem`;
  const changeValue = (next: WorkflowInput['value']) =>
    store.edit(`input:${input.id}`, document =>
      updateWorkflowInput(document, input.id, {value: next})
    );
  const common = {
    id: fieldId,
    'data-testid': WORKBENCH_TEST_IDS.workflowInput(input.id),
    'aria-invalid': Boolean(problem),
    'aria-describedby': problem ? problemId : undefined,
    title: input.label
  };
  return (
    <WorkbenchField>
      <label htmlFor={fieldId}>
        <Truncated title={input.label}>{input.label}</Truncated>
        {input.unit && (
          <span className="unit" title={input.unit}>
            {input.unit}
          </span>
        )}
      </label>
      {input.kind === 'boolean' ? (
        <input
          {...common}
          type="checkbox"
          checked={input.value === true}
          onChange={event => changeValue(event.currentTarget.checked)}
        />
      ) : input.kind === 'enum' ? (
        <select
          {...common}
          value={String(input.value)}
          onChange={event => changeValue(event.currentTarget.value)}
        >
          {!input.options?.includes(String(input.value)) && (
            <option value={String(input.value)}>{String(input.value)} (unavailable)</option>
          )}
          {input.options?.map(option => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
      ) : (
        <input
          {...common}
          type={input.kind === 'number' || input.kind === 'integer' ? 'number' : 'text'}
          value={text}
          min={input.minimum}
          max={input.maximum}
          step={input.kind === 'integer' ? 1 : 'any'}
          onChange={event => {
            const nextText = event.currentTarget.value;
            const nextValue = parseSidePanelInputText(input, nextText);
            const accepted = changeValue(nextValue);
            setDraft({text: nextText, storedValue: accepted ? nextValue : input.value});
          }}
          onBlur={() => setDraft(null)}
        />
      )}
      {problem && (
        <span id={problemId} className="error" aria-live="polite">
          {problem}
        </span>
      )}
    </WorkbenchField>
  );
}

/** Kepler side-panel tab: workflows, step list, workflow inputs and run controls. */
export function WorkbenchSidePanel({store}: {store: WorkbenchStore}): React.JSX.Element | null {
  const panelId = useId();
  const state = useWorkbenchState(store);
  const {document, runner} = state;
  const datasets = store.host.listDatasets();
  const drawingCount = store.host.listDrawings?.().length ?? 0;
  const steps = useMemo(() => getSidePanelSteps(document, runner), [document, runner]);
  const templates = useMemo(
    () => (state.workflows.length ? [] : listWorkflowTemplates().slice(0, 3)),
    [state.workflows]
  );
  const runLabel = getSidePanelRunLabel(runner);
  const summary = getSidePanelRunSummary(runner?.lastRun);
  const runningStep = steps.find(step => step.id === runner?.runningStepId);
  const runningLabel =
    runningStep &&
    (runningStep.label ??
      store.registry.get(runningStep.operationId)?.title ??
      runningStep.operationId);
  const firstError = runner?.issues.find(issue => issue.severity === 'error')?.message;
  return (
    <Panel data-testid={WORKBENCH_TEST_IDS.sidePanel} aria-label="Workflows">
      <WorkbenchSection>
        <h2>Workflows</h2>
        <Truncated
          as="p"
          title="Chain analysis operations and re-run them when data or settings change."
        >
          <WorkbenchMuted>
            Chain analysis operations and re-run them when data or settings change.
          </WorkbenchMuted>
        </Truncated>
        <WorkbenchButton
          type="button"
          data-testid={WORKBENCH_TEST_IDS.sidePanelOpenDock}
          onClick={() => store.setDock({open: !state.dock.open})}
          aria-expanded={state.dock.open}
        >
          {state.dock.open ? 'Hide workbench' : 'Open workbench'}
        </WorkbenchButton>
      </WorkbenchSection>
      <WorkbenchSection aria-label="Choose a workflow">
        {state.workflows.length ? (
          <Row>
            <select
              aria-label="Workflow"
              data-testid={WORKBENCH_TEST_IDS.workflowSelect}
              value={state.activeWorkflowId ?? ''}
              title={document?.title ?? 'Choose a workflow'}
              onChange={event => store.setActiveWorkflow(event.currentTarget.value || null)}
            >
              {!document && <option value="">Choose a workflow</option>}
              {state.workflows.map(workflow => (
                <option key={workflow.id} value={workflow.id}>
                  {workflow.title}
                </option>
              ))}
            </select>
            <WorkbenchButton
              type="button"
              data-testid={WORKBENCH_TEST_IDS.workflowNew}
              aria-label="New workflow"
              onClick={() => store.createWorkflow()}
            >
              New
            </WorkbenchButton>
          </Row>
        ) : (
          <>
            <p>No workflows yet. Start a workflow or try a template.</p>
            <WorkbenchButton
              type="button"
              data-testid={WORKBENCH_TEST_IDS.workflowNew}
              onClick={() => store.createWorkflow()}
            >
              New workflow
            </WorkbenchButton>
            <TemplateList aria-label="Workflow templates">
              {templates.map(template => (
                <li key={template.id}>
                  <WorkbenchButton
                    type="button"
                    data-testid={WORKBENCH_TEST_IDS.template(template.id)}
                    title={template.title}
                    onClick={() =>
                      store.addWorkflow(
                        instantiateWorkflowTemplate(
                          template.id,
                          state.workflows.map(workflow => workflow.id)
                        )
                      )
                    }
                  >
                    <Truncated>{template.title}</Truncated>
                  </WorkbenchButton>
                  <Truncated as="p" title={template.summary}>
                    <WorkbenchMuted>{template.summary}</WorkbenchMuted>
                  </Truncated>
                </li>
              ))}
            </TemplateList>
            <WorkbenchButton
              type="button"
              data-testid={WORKBENCH_TEST_IDS.templates}
              onClick={() => store.setDialog('templates')}
            >
              More templates
            </WorkbenchButton>
            {state.templates.length > 0 && (
              <>
                <h3>Your templates</h3>
                <TemplateList aria-label="Your templates">
                  {state.templates.map(template => (
                    <li key={template.id}>
                      <WorkbenchButton
                        type="button"
                        data-testid={WORKBENCH_TEST_IDS.userTemplate(template.id)}
                        title={template.title}
                        onClick={() => store.useTemplate(template.id)}
                      >
                        <Truncated>{template.title}</Truncated>
                      </WorkbenchButton>
                    </li>
                  ))}
                </TemplateList>
              </>
            )}
          </>
        )}
      </WorkbenchSection>
      <WorkbenchSection aria-label="Start from data">
        <QuickStartDetails open={!document?.steps.length}>
          <summary>Start from data</summary>
          <TemplateList aria-label="Datasets">
            {datasets.map(dataset => (
              <li key={dataset.id}>
                <Truncated title={dataset.label}>{dataset.label}</Truncated>
                <WorkbenchMuted>
                  {dataset.rowCount.toLocaleString()} rows ·{' '}
                  {dataset.resident ? 'resident' : 'table'}
                </WorkbenchMuted>
                <WorkbenchButton
                  type="button"
                  data-testid={WORKBENCH_TEST_IDS.datasetQuickStart(dataset.id)}
                  aria-label={`Use ${dataset.label} in a workflow`}
                  title={`Use ${dataset.label} in a workflow`}
                  onClick={() =>
                    startSidePanelWorkflow(
                      store,
                      getSidePanelDatasetSource(dataset),
                      (dataset.label.trim() || 'Dataset').slice(
                        0,
                        WORKFLOW_LIMITS.maximumTitleCharacters
                      )
                    )
                  }
                >
                  Use in a workflow
                </WorkbenchButton>
                <DatasetRecommendations dataset={dataset} store={store} />
              </li>
            ))}
          </TemplateList>
          {drawingCount > 0 ? (
            <>
              <WorkbenchMuted>{drawingCount.toLocaleString()} drawn shapes</WorkbenchMuted>
              <WorkbenchButton
                type="button"
                data-testid={WORKBENCH_TEST_IDS.drawingQuickStart}
                aria-label="Use drawn shapes in a workflow"
                title="Use drawn shapes in a workflow"
                onClick={() =>
                  startSidePanelWorkflow(
                    store,
                    {operationId: 'source-drawing', parameters: {shapes: 'all'}},
                    'Drawn shapes'
                  )
                }
              >
                Use in a workflow
              </WorkbenchButton>
            </>
          ) : (
            <WorkbenchMuted>Draw a polygon on the map to use it as an input.</WorkbenchMuted>
          )}
        </QuickStartDetails>
      </WorkbenchSection>
      {document && (
        <>
          <WorkbenchSection aria-label="Run workflow">
            <Row>
              <WorkbenchButton
                type="button"
                $variant="primary"
                data-testid={WORKBENCH_TEST_IDS.runAll}
                disabled={!runner || runner.running || runLabel === 'Up to date' || !steps.length}
                onClick={() => {
                  void store.run();
                }}
              >
                {runLabel}
              </WorkbenchButton>
              {runner?.running && (
                <WorkbenchButton
                  type="button"
                  data-testid={WORKBENCH_TEST_IDS.cancel}
                  onClick={() => store.cancel()}
                >
                  Cancel
                </WorkbenchButton>
              )}
            </Row>
            <div aria-live="polite" role="status">
              {runner?.running && (
                <Truncated title={runningLabel ? `Running ${runningLabel}` : 'Running workflow'}>
                  {runningLabel ? `Running ${runningLabel}` : 'Running workflow'}
                </Truncated>
              )}
              {summary && <Truncated title={summary}>{summary}</Truncated>}
              {firstError && <Problem>{firstError}</Problem>}
              {runner?.issues
                .filter(issue => issue.code === 'publish-failed')
                .map((issue, index) => (
                  <PublicationProblem key={index}>{issue.message}</PublicationProblem>
                ))}
            </div>
          </WorkbenchSection>
          <WorkbenchSection aria-label="Workflow inputs">
            <h3>Inputs</h3>
            {document.inputs.length ? (
              document.inputs.map(input => (
                <WorkflowInputField
                  key={`${document.id}:${input.id}`}
                  input={input}
                  store={store}
                />
              ))
            ) : (
              <WorkbenchMuted>No workflow inputs.</WorkbenchMuted>
            )}
          </WorkbenchSection>
          <WorkbenchSection aria-label="Workflow steps">
            <h3>Steps</h3>
            <StepList role="list" aria-label="Workflow steps">
              {steps.map(step => {
                const operation = store.registry.get(step.operationId);
                const stepState = runner?.steps[step.id];
                const status =
                  stepState?.status ??
                  (step.disabled ? 'disabled' : operation ? 'ready' : 'not-configured');
                const label = step.label ?? operation?.title ?? step.operationId;
                const error =
                  stepState?.issues.find(issue => issue.severity === 'error') ?? stepState?.error;
                const publicationProblems =
                  stepState?.issues.filter(issue => issue.code === 'publish-failed') ?? [];
                const displayLabel = getSidePanelDisplayLabel(step, stepState);
                const firstOutput = operation?.outputs[0]?.name;
                const output = firstOutput
                  ? stepState?.result?.outputs[firstOutput]
                  : Object.values(stepState?.result?.outputs ?? {})[0];
                return (
                  <li key={step.id}>
                    <StepButton
                      type="button"
                      data-testid={WORKBENCH_TEST_IDS.stepRow(step.id)}
                      data-status={status}
                      aria-label={`Select ${label}`}
                      title={label}
                      aria-current={state.selectedStepIds.includes(step.id) ? 'true' : undefined}
                      aria-describedby={[
                        `${panelId}-${step.id}-status`,
                        ...(error ? [`${panelId}-${step.id}-error`] : []),
                        ...publicationProblems.map(
                          (_, index) => `${panelId}-${step.id}-publication-${index}`
                        )
                      ].join(' ')}
                      onClick={() => {
                        store.select([step.id]);
                        store.setDock({open: true, inspectorTab: 'parameters'});
                      }}
                    >
                      <Truncated title={label}>{label}</Truncated>
                      {operation?.title !== label && (
                        <Truncated title={operation?.title ?? 'Unknown operation'}>
                          <WorkbenchMuted>{operation?.title ?? 'Unknown operation'}</WorkbenchMuted>
                        </Truncated>
                      )}
                      <Row as="span" id={`${panelId}-${step.id}-status`}>
                        <WorkbenchStatusBadge $status={status} data-status={status}>
                          {WORKBENCH_STATUS[status].label}
                        </WorkbenchStatusBadge>
                        {status === 'done' && output && (
                          <WorkbenchMuted>
                            {output.table.rowCount.toLocaleString()} rows
                          </WorkbenchMuted>
                        )}
                        {displayLabel && displayLabel !== 'Display requested' && (
                          <span role="img" aria-label={displayLabel} title={displayLabel}>
                            <svg
                              width="16"
                              height="16"
                              viewBox="0 0 20 20"
                              fill="none"
                              stroke="currentColor"
                              aria-hidden="true"
                            >
                              <path d="M1 10s3-6 9-6 9 6 9 6-3 6-9 6-9-6-9-6Z" />
                              <circle cx="10" cy="10" r="2.5" />
                            </svg>
                          </span>
                        )}
                        {displayLabel === 'Display requested' && (
                          <WorkbenchMuted>{displayLabel}</WorkbenchMuted>
                        )}
                      </Row>
                    </StepButton>
                    {error && (
                      <Problem id={`${panelId}-${step.id}-error`} aria-live="polite">
                        {error.message}
                      </Problem>
                    )}
                    {publicationProblems.map((issue, index) => (
                      <PublicationProblem
                        key={index}
                        id={`${panelId}-${step.id}-publication-${index}`}
                        aria-live="polite"
                      >
                        {issue.message}
                      </PublicationProblem>
                    ))}
                    <StepActions>
                      <WorkbenchButton
                        type="button"
                        data-testid={WORKBENCH_TEST_IDS.runToHere}
                        aria-label={`Run to here: ${label}`}
                        title={`Run to here: ${label}`}
                        disabled={
                          !runner ||
                          runner.running ||
                          ['disabled', 'blocked', 'not-configured'].includes(status)
                        }
                        onClick={() => {
                          void store.run({target: step.id});
                        }}
                      >
                        Run to here
                      </WorkbenchButton>
                    </StepActions>
                  </li>
                );
              })}
            </StepList>
            {!steps.length && (
              <WorkbenchMuted>No steps yet. Open the workbench to add a step.</WorkbenchMuted>
            )}
          </WorkbenchSection>
        </>
      )}
      {state.notice && (
        <WorkbenchSection aria-live="polite">
          {state.notice.kind === 'error' ? (
            <Problem>{state.notice.message}</Problem>
          ) : (
            <WorkbenchMuted>{state.notice.message}</WorkbenchMuted>
          )}
        </WorkbenchSection>
      )}
    </Panel>
  );
}
