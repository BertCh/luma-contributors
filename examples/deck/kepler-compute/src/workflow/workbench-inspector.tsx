// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useEffect, useId, useMemo, useState} from 'react';
import styled from 'styled-components';
import {
  connectWorkflowSteps,
  disconnectWorkflowInput,
  duplicateWorkflowSteps,
  removeWorkflowInput,
  removeWorkflowSteps,
  renameWorkflow,
  replaceWorkflowStepOperation,
  setWorkflowDisplayStyle,
  updateWorkflowInput,
  updateWorkflowStep,
  WORKFLOW_LIMITS,
  type WorkflowDocument,
  type WorkflowDisplayStyle,
  type WorkflowOperation,
  type WorkflowParameterSpec,
  type WorkflowStep,
  type WorkflowStepPatch
} from '@luma-contributors/kepler';
import {
  WORKBENCH_COLORS,
  WORKBENCH_PORT_LABELS,
  WORKBENCH_STATUS,
  WORKBENCH_TEST_IDS
} from './workbench-contract';
import {WorkbenchParameterForm, WorkbenchParameterValue} from './workbench-parameter-form';
import {
  formatWorkbenchCapacity,
  getWorkbenchConnectionChoices,
  getWorkbenchInspectorTabIndex,
  getWorkbenchInspectorIssues,
  hasWorkbenchRunOptionOverrides,
  getWorkbenchIssueSuggestions,
  getWorkbenchDisplayColumns,
  getWorkbenchSourceParameters
} from './workbench-parameter-model';
import {WorkbenchResultView} from './workbench-result-view';
import type {WorkbenchInspectorTab, WorkbenchStore} from './workbench-store';
import {
  WorkbenchButton,
  WorkbenchField,
  WorkbenchSection,
  WorkbenchStatusBadge,
  WorkbenchSurface,
  WorkbenchTabs
} from './workbench-styles';
import {useWorkbenchState} from './use-workbench';

const INSPECTOR_TABS: readonly WorkbenchInspectorTab[] = ['parameters', 'result', 'contract'];
const TAB_LABELS = {parameters: 'Parameters', result: 'Result', contract: 'Contract'};
const ENGINE_LABELS = {
  host: 'Host application',
  'cpu-reference': 'CPU reference',
  gpu: 'GPU required',
  'gpu-or-cpu': 'GPU when available, with a CPU reference'
};

const InspectorSurface = styled(WorkbenchSurface)`
  & label {
    min-height: 24px;
    align-items: center;
  }
  & input[type='checkbox'],
  & input[type='radio'] {
    min-width: 24px;
    min-height: 24px;
    vertical-align: middle;
  }
  & [role='tabpanel']:focus-visible {
    outline: 2px solid ${WORKBENCH_COLORS.focus};
    outline-offset: -2px;
  }
`;

const DisclosureSummary = styled.summary`
  min-height: 24px;
  cursor: pointer;
  details[open] > & {
    margin-bottom: 8px;
  }
  &:focus-visible {
    outline: 2px solid ${WORKBENCH_COLORS.focus};
    outline-offset: 1px;
  }
`;

const SuggestionButtons = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-top: 4px;
`;

function StepLabel({
  step,
  operation,
  store
}: {
  step: WorkflowStep;
  operation?: WorkflowOperation;
  store: WorkbenchStore;
}): React.JSX.Element {
  const identifier = useId();
  const [draft, setDraft] = useState<string | null>(null);
  useEffect(() => {
    setDraft(current =>
      current !== null && current.trim() === (step.label ?? '') ? current : null
    );
  }, [step.label]);
  return (
    <WorkbenchField>
      <label htmlFor={identifier}>Step label</label>
      <input
        id={identifier}
        data-testid={WORKBENCH_TEST_IDS.stepLabel}
        aria-label="Step label"
        placeholder={operation?.title ?? step.operationId}
        value={draft ?? step.label ?? ''}
        maxLength={WORKFLOW_LIMITS.maximumTitleCharacters}
        onChange={event => {
          const label = event.target.value;
          setDraft(label);
          store.edit(`label:${step.id}`, document =>
            updateWorkflowStep(document, step.id, {label})
          );
        }}
        onBlur={() => setDraft(null)}
      />
    </WorkbenchField>
  );
}

function StepNote({step, store}: {step: WorkflowStep; store: WorkbenchStore}): React.JSX.Element {
  const identifier = useId();
  return (
    <WorkbenchField>
      <label htmlFor={identifier}>Note</label>
      <textarea
        id={identifier}
        aria-label="Note"
        rows={3}
        value={step.note ?? ''}
        maxLength={WORKFLOW_LIMITS.maximumNoteCharacters}
        onChange={event => {
          const note = event.target.value;
          store.edit(`note:${step.id}`, document => updateWorkflowStep(document, step.id, {note}));
        }}
      />
    </WorkbenchField>
  );
}

function WorkflowSettings({
  store,
  document
}: {
  store: WorkbenchStore;
  document: WorkflowDocument;
}): React.JSX.Element {
  const state = useWorkbenchState(store);
  const identifier = useId();
  const inputs = useMemo(
    () =>
      document.inputs.map(input => ({
        input,
        spec: {
          name: input.id,
          label: input.label,
          kind: input.kind,
          defaultValue: input.value,
          unit: input.unit,
          minimum: input.minimum,
          maximum: input.maximum,
          options: input.options
        } satisfies WorkflowParameterSpec,
        users: document.steps
          .filter(step =>
            Object.values(step.parameters).some(
              binding => binding.from === 'input' && binding.inputId === input.id
            )
          )
          .map(
            step => step.label ?? store.registry.get(step.operationId)?.title ?? step.operationId
          )
      })),
    [document, store.registry]
  );
  return (
    <>
      <WorkbenchSection>
        <h3>Workflow settings</h3>
        <WorkbenchField>
          <label htmlFor={`${identifier}-description`}>Description</label>
          <textarea
            id={`${identifier}-description`}
            aria-label="Description"
            rows={3}
            maxLength={WORKFLOW_LIMITS.maximumNoteCharacters}
            value={document.description ?? ''}
            onChange={event => {
              const description = event.target.value;
              store.edit(`description:${document.id}`, currentDocument =>
                renameWorkflow(currentDocument, currentDocument.title, description)
              );
            }}
          />
        </WorkbenchField>
      </WorkbenchSection>
      <WorkbenchSection>
        <h3>Workflow inputs</h3>
        {!inputs.length && <p>Promote a step parameter to edit it here.</p>}
        {inputs.map(({input, spec, users}) => (
          <WorkbenchField key={input.id}>
            <label htmlFor={`${identifier}-${input.id}-label`}>Input label</label>
            <input
              id={`${identifier}-${input.id}-label`}
              aria-label={`Label for ${input.label}`}
              maxLength={WORKFLOW_LIMITS.maximumTitleCharacters}
              value={input.label}
              onChange={event => {
                const label = event.target.value;
                store.edit(`input-label:${input.id}`, currentDocument =>
                  updateWorkflowInput(currentDocument, input.id, {label})
                );
              }}
            />
            <WorkbenchParameterValue
              spec={spec}
              value={input.value}
              testId={WORKBENCH_TEST_IDS.workflowInput(input.id)}
              onCommit={value =>
                store.edit(`input-value:${input.id}`, currentDocument =>
                  updateWorkflowInput(currentDocument, input.id, {value})
                )
              }
            />
            <span className="hint">
              {users.length ? `Used by ${users.join(', ')}` : 'No steps use this input.'}
            </span>
            <WorkbenchButton
              type="button"
              aria-label={`Remove workflow input ${input.label}`}
              onClick={() =>
                store.edit(`remove-input:${input.id}`, currentDocument =>
                  removeWorkflowInput(currentDocument, input.id)
                )
              }
            >
              Remove
            </WorkbenchButton>
          </WorkbenchField>
        ))}
      </WorkbenchSection>
      <WorkbenchSection aria-live="polite">
        <h3>Workflow issues</h3>
        {state.runner?.issues.length ? (
          <ul>
            {state.runner.issues.map((issue, index) => (
              <li key={`${issue.code}-${index}`}>
                {issue.severity}: {issue.message}
              </li>
            ))}
          </ul>
        ) : (
          <p>No workflow issues.</p>
        )}
      </WorkbenchSection>
    </>
  );
}

function StepParameters({
  store,
  document,
  step,
  operation
}: {
  store: WorkbenchStore;
  document: WorkflowDocument;
  step: WorkflowStep;
  operation: WorkflowOperation;
}): React.JSX.Element {
  const state = useWorkbenchState(store);
  const identifier = useId();
  const sourceOperations = useMemo(
    () =>
      operation.category === 'source'
        ? store.registry.list().filter(candidate => candidate.category === 'source')
        : [],
    [operation.category, store.registry]
  );
  const connections = useMemo(
    () =>
      operation.inputs.map(input => ({
        input,
        choices: getWorkbenchConnectionChoices(
          document,
          store.registry,
          state.runner,
          step.id,
          input
        )
      })),
    [document, operation, store.registry, state.runner, step.id]
  );
  const stepState = state.runner?.steps[step.id];
  const {problems, notes} = useMemo(
    () => getWorkbenchInspectorIssues(stepState?.issues ?? [], step.label ?? operation.title),
    [stepState?.issues, step.label, operation.title]
  );
  const suggestions = useMemo(
    () =>
      getWorkbenchIssueSuggestions(document, store.registry, state.runner, stepState?.issues ?? []),
    [document, store.registry, state.runner, stepState?.issues]
  );
  const displayOutputs = useMemo(
    () =>
      operation.outputs.map(output => ({
        output,
        columns: getWorkbenchDisplayColumns(step.id, output, state.runner)
      })),
    [operation, step.id, state.runner]
  );
  const status = stepState?.status ?? (step.disabled ? 'disabled' : 'not-configured');
  function patchStep(kind: string, patch: WorkflowStepPatch): void {
    store.edit(`${kind}:${step.id}`, currentDocument =>
      updateWorkflowStep(currentDocument, step.id, patch)
    );
  }
  function patchDisplayStyle(output: string, patch: WorkflowDisplayStyle | undefined): void {
    store.edit(`display-style:${step.id}:${output}`, currentDocument => {
      const currentStyle = currentDocument.steps.find(candidate => candidate.id === step.id)
        ?.displayStyle?.[output];
      return setWorkflowDisplayStyle(
        currentDocument,
        step.id,
        output,
        patch === undefined ? undefined : {...currentStyle, ...patch}
      );
    });
  }
  return (
    <>
      <WorkbenchSection>
        {operation.category === 'source' && (
          <WorkbenchField>
            <label htmlFor={`${identifier}-source`}>Source</label>
            <select
              id={`${identifier}-source`}
              aria-label="Source"
              data-testid={WORKBENCH_TEST_IDS.sourceType}
              value={operation.id}
              onChange={event => {
                const operationId = event.target.value;
                store.edit(`source-type:${step.id}`, currentDocument => {
                  const currentStep = currentDocument.steps.find(
                    candidate => candidate.id === step.id
                  );
                  if (currentStep?.operationId === operationId) return currentDocument;
                  const target = store.registry.get(operationId);
                  const parameters =
                    currentStep && target
                      ? getWorkbenchSourceParameters(
                          currentStep,
                          target,
                          target.id === 'source-dataset' ? store.host.listDatasets() : []
                        )
                      : undefined;
                  return replaceWorkflowStepOperation(
                    currentDocument,
                    store.registry,
                    step.id,
                    operationId,
                    parameters
                  );
                });
              }}
            >
              {sourceOperations.map(source => (
                <option key={source.id} value={source.id}>
                  {source.title}
                </option>
              ))}
            </select>
          </WorkbenchField>
        )}
        <StepLabel key={step.id} step={step} operation={operation} store={store} />
        <div aria-live="polite">
          <WorkbenchStatusBadge $status={status} data-status={status}>
            {WORKBENCH_STATUS[status].label}
          </WorkbenchStatusBadge>
          {problems.length > 0 && (
            <ul data-testid={WORKBENCH_TEST_IDS.stepIssues}>
              {problems.map((issue, index) => (
                <li key={`${issue.code}-${index}`}>
                  {issue.severity}: {issue.message}
                  {suggestions[index]?.length > 0 && (
                    <SuggestionButtons>
                      {suggestions[index].map(suggestion => (
                        <WorkbenchButton
                          type="button"
                          key={suggestion.id}
                          data-testid={WORKBENCH_TEST_IDS.suggestion(suggestion.id)}
                          title={suggestion.description}
                          onClick={() =>
                            store.edit(`suggestion:${suggestion.id}`, currentDocument =>
                              suggestion.apply(currentDocument, store.registry)
                            )
                          }
                        >
                          {suggestion.label}
                        </WorkbenchButton>
                      ))}
                    </SuggestionButtons>
                  )}
                </li>
              ))}
            </ul>
          )}
        </div>
      </WorkbenchSection>
      <WorkbenchSection>
        <h3>Parameters</h3>
        <WorkbenchParameterForm store={store} stepId={step.id} />
      </WorkbenchSection>
      <WorkbenchSection>
        <h3>Inputs</h3>
        {!connections.length && <p>This step reads no inputs.</p>}
        {connections.map(({input, choices}) => {
          const connection = step.inputs[input.name];
          const selected = connection ? JSON.stringify([connection.stepId, connection.output]) : '';
          return (
            <WorkbenchField key={input.name}>
              <label htmlFor={`${identifier}-${input.name}`}>{input.label}</label>
              <select
                id={`${identifier}-${input.name}`}
                aria-label={input.label}
                data-testid={WORKBENCH_TEST_IDS.inputBinding(input.name)}
                value={selected}
                aria-describedby={`${identifier}-${input.name}-hint`}
                onChange={event => {
                  const choice = choices.find(candidate => candidate.value === event.target.value);
                  store.edit(`connection:${step.id}:${input.name}`, currentDocument =>
                    choice
                      ? connectWorkflowSteps(currentDocument, store.registry, {
                          from: choice.connection,
                          to: {stepId: step.id, input: input.name}
                        })
                      : disconnectWorkflowInput(currentDocument, step.id, input.name)
                  );
                }}
              >
                <option value="">Not connected</option>
                {selected && !choices.some(choice => choice.value === selected) && (
                  <option value={selected}>
                    Unavailable connection: {connection.stepId} · {connection.output}
                  </option>
                )}
                {choices.map(choice => (
                  <option key={choice.value} value={choice.value}>
                    {choice.label}
                  </option>
                ))}
              </select>
              <span className="hint" id={`${identifier}-${input.name}-hint`}>
                {input.accepts.map(kind => WORKBENCH_PORT_LABELS[kind]).join(', ')} ·{' '}
                {formatWorkbenchCapacity(input)}
                {input.optional ? ' · Optional' : ''}
              </span>
            </WorkbenchField>
          );
        })}
      </WorkbenchSection>
      <WorkbenchSection as="details" open={hasWorkbenchRunOptionOverrides(step)}>
        <DisclosureSummary>Run options</DisclosureSummary>
        {operation.engine === 'gpu-or-cpu' && (
          <WorkbenchField>
            <label htmlFor={`${identifier}-engine`}>Engine</label>
            <select
              id={`${identifier}-engine`}
              aria-label="Engine"
              value={step.engine ?? ''}
              onChange={event =>
                patchStep('engine', {
                  engine:
                    event.target.value === ''
                      ? undefined
                      : event.target.value === 'cpu'
                        ? 'cpu'
                        : 'auto'
                })
              }
            >
              <option value="">Workflow default</option>
              <option value="auto">GPU when available</option>
              <option value="cpu">CPU reference</option>
            </select>
          </WorkbenchField>
        )}
        <WorkbenchField>
          <label htmlFor={`${identifier}-fallback`}>If the GPU engine cannot run</label>
          <select
            id={`${identifier}-fallback`}
            aria-label="If the GPU engine cannot run"
            value={step.onFallback ?? 'warn'}
            onChange={event =>
              patchStep('fallback', {onFallback: event.target.value === 'stop' ? 'stop' : 'warn'})
            }
          >
            <option value="warn">Continue on CPU and warn</option>
            <option value="stop">Stop</option>
          </select>
        </WorkbenchField>
        <WorkbenchField>
          <span className="label">Show on the map</span>
          {displayOutputs.map(({output, columns}) => {
            const style = step.displayStyle?.[output.name];
            const displayed = step.display?.includes(output.name) ?? false;
            return (
              <div key={output.name}>
                <label>
                  <span>
                    <input
                      type="checkbox"
                      checked={displayed}
                      aria-label={`Show ${output.label} on the map`}
                      onChange={event => {
                        const checked = event.target.checked;
                        store.edit(`display:${step.id}:${output.name}`, currentDocument => {
                          const display =
                            currentDocument.steps.find(candidate => candidate.id === step.id)
                              ?.display ?? [];
                          return updateWorkflowStep(currentDocument, step.id, {
                            display: checked
                              ? [...display, output.name]
                              : display.filter(name => name !== output.name)
                          });
                        });
                      }}
                    />{' '}
                    {output.label}
                  </span>
                </label>
                {displayed && (
                  <WorkbenchField>
                    <label htmlFor={`${identifier}-color-${output.name}`}>Colour by</label>
                    <select
                      id={`${identifier}-color-${output.name}`}
                      aria-label={`Colour by for ${output.label}`}
                      data-testid={WORKBENCH_TEST_IDS.displayColorColumn(output.name)}
                      value={style?.colorColumn ?? ''}
                      onChange={event =>
                        patchDisplayStyle(
                          output.name,
                          event.target.value ? {colorColumn: event.target.value} : undefined
                        )
                      }
                    >
                      <option value="">Default colour</option>
                      {style?.colorColumn && !columns.includes(style.colorColumn) && (
                        <option value={style.colorColumn}>
                          Missing column "{style.colorColumn}"
                        </option>
                      )}
                      {columns.map(column => (
                        <option key={column} value={column}>
                          {column}
                        </option>
                      ))}
                    </select>
                    {style?.colorColumn && (
                      <>
                        <label htmlFor={`${identifier}-scale-${output.name}`}>Scale</label>
                        <select
                          id={`${identifier}-scale-${output.name}`}
                          aria-label={`Scale for ${output.label}`}
                          data-testid={WORKBENCH_TEST_IDS.displayColorScale(output.name)}
                          value={style.colorScale ?? ''}
                          onChange={event => {
                            const value = event.target.value;
                            patchDisplayStyle(output.name, {
                              colorScale:
                                value === 'quantile' || value === 'quantize' || value === 'ordinal'
                                  ? value
                                  : undefined
                            });
                          }}
                        >
                          <option value="">Automatic</option>
                          <option value="quantile">Quantile</option>
                          <option value="quantize">Equal interval</option>
                          <option value="ordinal">Categories</option>
                        </select>
                        <label htmlFor={`${identifier}-palette-${output.name}`}>Palette</label>
                        <select
                          id={`${identifier}-palette-${output.name}`}
                          aria-label={`Palette for ${output.label}`}
                          data-testid={WORKBENCH_TEST_IDS.displayPalette(output.name)}
                          value={style.palette ?? ''}
                          onChange={event => {
                            const value = event.target.value;
                            patchDisplayStyle(output.name, {
                              palette:
                                value === 'sequential' ||
                                value === 'diverging' ||
                                value === 'categorical'
                                  ? value
                                  : undefined
                            });
                          }}
                        >
                          <option value="">Automatic</option>
                          <option value="sequential">Sequential</option>
                          <option value="diverging">Diverging</option>
                          <option value="categorical">Categorical</option>
                        </select>
                      </>
                    )}
                  </WorkbenchField>
                )}
              </div>
            );
          })}
        </WorkbenchField>
        <WorkbenchField>
          <label>
            <span>
              <input
                type="checkbox"
                aria-label="Disabled"
                checked={Boolean(step.disabled)}
                onChange={event => patchStep('disabled', {disabled: event.target.checked})}
              />{' '}
              Disabled
            </span>
          </label>
        </WorkbenchField>
        <StepNote step={step} store={store} />
      </WorkbenchSection>
      {notes.length > 0 && (
        <WorkbenchSection as="details">
          <DisclosureSummary>Notes ({notes.length.toLocaleString()})</DisclosureSummary>
          <ul>
            {notes.map((issue, index) => (
              <li key={`${issue.code}-${index}`}>
                {issue.severity}: {issue.message}
              </li>
            ))}
          </ul>
        </WorkbenchSection>
      )}
    </>
  );
}

function OperationContract({operation}: {operation: WorkflowOperation}): React.JSX.Element {
  return (
    <WorkbenchSection>
      <h3>{operation.title}</h3>
      <p>{operation.summary}</p>
      <h3>Declared behaviour</h3>
      <ul>
        {operation.declaredBehavior.map(behavior => (
          <li key={behavior}>{behavior}</li>
        ))}
      </ul>
      <h3>Limitation</h3>
      <p>{operation.limitation}</p>
      <h3>Ports</h3>
      <ul>
        {operation.inputs.map(input => (
          <li key={`input-${input.name}`}>
            Input: {input.label} ·{' '}
            {input.accepts.map(kind => WORKBENCH_PORT_LABELS[kind]).join(', ')} ·{' '}
            {formatWorkbenchCapacity(input)}
          </li>
        ))}
        {operation.outputs.map(output => (
          <li key={`output-${output.name}`}>
            Output: {output.label} ·{' '}
            {output.kindFrom
              ? `Same representation as input ${output.kindFrom}`
              : WORKBENCH_PORT_LABELS[output.kind]}{' '}
            · {formatWorkbenchCapacity(output)}
          </li>
        ))}
      </ul>
      <h3>Engine support</h3>
      <p>{ENGINE_LABELS[operation.engine]}</p>
      <h3>Mirrored contributors</h3>
      <p>{operation.contributorNames.join(', ') || 'None'}</p>
      <h3>Origin</h3>
      <p>{operation.origin === 'catalog' ? 'Catalog binding' : 'Workflow operation'}</p>
      <h3>Operation id</h3>
      <code style={{userSelect: 'text'}}>{operation.id}</code>
    </WorkbenchSection>
  );
}

/** Parameters, result and contract of the selected step. */
export function WorkbenchInspector({store}: {store: WorkbenchStore}): React.JSX.Element | null {
  const state = useWorkbenchState(store);
  const identifier = useId();
  const document = state.document;
  const selected = useMemo(
    () => document?.steps.filter(step => state.selectedStepIds.includes(step.id)) ?? [],
    [document, state.selectedStepIds]
  );
  const step = selected[0];
  const selectionTarget = useMemo(
    () =>
      selected
        .map(selectedStep => selectedStep.id)
        .sort()
        .join(','),
    [selected]
  );
  const operation = step && store.registry.get(step.operationId);
  let content: React.JSX.Element;
  if (!document)
    content = (
      <WorkbenchSection>
        <p>Create or select a workflow to inspect it.</p>
      </WorkbenchSection>
    );
  else if (!selected.length)
    content = <WorkflowSettings key={document.id} store={store} document={document} />;
  else if (selected.length > 1)
    content = (
      <WorkbenchSection>
        <p>{selected.length.toLocaleString()} steps selected</p>
        <WorkbenchButton
          type="button"
          onClick={() =>
            store.edit(`disable-selection:${selectionTarget}`, currentDocument =>
              selected.reduce(
                (changed, selectedStep) =>
                  updateWorkflowStep(changed, selectedStep.id, {disabled: true}),
                currentDocument
              )
            )
          }
        >
          Disable
        </WorkbenchButton>
        <WorkbenchButton
          type="button"
          onClick={() =>
            store.edit(`enable-selection:${selectionTarget}`, currentDocument =>
              selected.reduce(
                (changed, selectedStep) =>
                  updateWorkflowStep(changed, selectedStep.id, {disabled: false}),
                currentDocument
              )
            )
          }
        >
          Enable
        </WorkbenchButton>
        <WorkbenchButton
          type="button"
          onClick={() => {
            let duplicated: string[] = [];
            const accepted = store.edit(
              `duplicate-selection:${selectionTarget}`,
              currentDocument => {
                const result = duplicateWorkflowSteps(
                  currentDocument,
                  selected.map(selectedStep => selectedStep.id)
                );
                duplicated = result.stepIds;
                return result.document;
              }
            );
            if (accepted) store.select(duplicated);
          }}
        >
          Duplicate
        </WorkbenchButton>
        <WorkbenchButton
          type="button"
          $variant="danger"
          onClick={() =>
            store.edit(`delete-selection:${selectionTarget}`, currentDocument =>
              removeWorkflowSteps(
                currentDocument,
                selected.map(selectedStep => selectedStep.id)
              )
            )
          }
        >
          Delete
        </WorkbenchButton>
      </WorkbenchSection>
    );
  else if (!operation)
    content = (
      <WorkbenchSection>
        <p>
          Operation <code>{step.operationId}</code> is not in this build.
        </p>
        <StepLabel key={`${document.id}:${step.id}`} step={step} store={store} />
        <StepNote step={step} store={store} />
        <WorkbenchButton
          type="button"
          $variant="danger"
          onClick={() =>
            store.edit(`delete:${step.id}`, currentDocument =>
              removeWorkflowSteps(currentDocument, [step.id])
            )
          }
        >
          Delete
        </WorkbenchButton>
      </WorkbenchSection>
    );
  else
    content = (
      <>
        <WorkbenchTabs aria-label="Step inspector tabs">
          {INSPECTOR_TABS.map((tab, index) => (
            <button
              type="button"
              key={tab}
              role="tab"
              id={`${identifier}-tab-${tab}`}
              data-testid={WORKBENCH_TEST_IDS.inspectorTab(tab)}
              aria-selected={state.dock.inspectorTab === tab}
              aria-controls={`${identifier}-panel-${tab}`}
              tabIndex={state.dock.inspectorTab === tab ? 0 : -1}
              onClick={() => store.setDock({inspectorTab: tab})}
              onKeyDown={event => {
                const nextIndex = getWorkbenchInspectorTabIndex(event.key, index);
                if (nextIndex === null) return;
                event.preventDefault();
                store.setDock({inspectorTab: INSPECTOR_TABS[nextIndex]});
                const nextTab = event.currentTarget.parentElement?.children[nextIndex];
                if (nextTab instanceof HTMLElement) nextTab.focus();
              }}
            >
              {TAB_LABELS[tab]}
            </button>
          ))}
        </WorkbenchTabs>
        {INSPECTOR_TABS.map(tab => (
          <div
            key={tab}
            role="tabpanel"
            id={`${identifier}-panel-${tab}`}
            aria-labelledby={`${identifier}-tab-${tab}`}
            hidden={state.dock.inspectorTab !== tab}
            tabIndex={0}
          >
            {state.dock.inspectorTab === tab &&
              (tab === 'result' ? (
                <WorkbenchResultView store={store} />
              ) : tab === 'contract' ? (
                <OperationContract operation={operation} />
              ) : (
                <StepParameters
                  key={`${document!.id}:${step.id}`}
                  store={store}
                  document={document!}
                  step={step}
                  operation={operation}
                />
              ))}
          </div>
        ))}
      </>
    );
  return (
    <InspectorSurface
      data-testid={WORKBENCH_TEST_IDS.inspector}
      aria-label="Step inspector"
      role="region"
    >
      {content}
    </InspectorSurface>
  );
}
