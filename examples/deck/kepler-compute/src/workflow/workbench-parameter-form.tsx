// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useEffect, useId, useMemo, useRef, useState} from 'react';
import styled from 'styled-components';
import {
  demoteWorkflowParameter,
  formatWorkflowParameterValue,
  getWorkflowParameterProblem,
  promoteWorkflowParameter,
  resolveWorkflowParameters,
  setWorkflowParameter,
  updateWorkflowInput,
  WORKFLOW_EXPRESSION_FUNCTIONS,
  WORKFLOW_LIMITS,
  type CatalogParameterValue,
  type WorkflowDatasetProfile,
  type WorkflowParameterSpec
} from '@luma-contributors/kepler';
import {WORKBENCH_TEST_IDS} from './workbench-contract';
import {
  coerceWorkbenchParameterText,
  formatWorkbenchParameterHint,
  formatWorkbenchDrawings,
  getWorkbenchColumnSuggestions,
  getWorkbenchDatasetParameters,
  getWorkbenchParameterRows,
  getWorkbenchParameterControlSpec,
  insertWorkbenchExpressionColumn,
  type WorkbenchColumnSuggestions
} from './workbench-parameter-model';
import type {WorkbenchStore} from './workbench-store';
import {WorkbenchButton, WorkbenchField, WorkbenchMuted} from './workbench-styles';
import {useWorkbenchState} from './use-workbench';

const PROMOTABLE_KINDS = new Set(['number', 'integer', 'boolean', 'enum', 'string', 'number-list']);

const ParameterField = styled(WorkbenchField)`
  & > label,
  & [role='radiogroup'] label {
    min-height: 24px;
    align-items: center;
  }
  & [role='radiogroup'] label {
    display: inline-flex;
    gap: 4px;
    margin-right: 8px;
  }
  & input[type='checkbox'],
  & input[type='radio'] {
    min-width: 24px;
    min-height: 24px;
  }
`;

const ParameterHintRow = styled.div`
  display: flex;
  align-items: center;
  gap: 6px;
  & > .hint {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  & > button {
    flex: none;
    margin-left: auto;
  }
`;

/** Shared control for step parameters and workflow inputs; only unfinished input is local state. */
export function WorkbenchParameterValue({
  spec,
  value,
  onCommit,
  testId,
  errorTestId,
  suggestions = {columns: [], hints: []},
  datasets = [],
  datasetLabel,
  bindingIdentity = 'value',
  contextLabel,
  differentDefault,
  onReset,
  operationId
}: {
  spec: WorkflowParameterSpec;
  value: CatalogParameterValue;
  onCommit: (value: CatalogParameterValue) => void;
  testId: string;
  errorTestId?: string;
  suggestions?: WorkbenchColumnSuggestions;
  datasets?: readonly WorkflowDatasetProfile[];
  datasetLabel?: string;
  bindingIdentity?: string;
  contextLabel?: string;
  differentDefault?: CatalogParameterValue;
  onReset?: () => void;
  operationId?: string;
}): React.JSX.Element {
  const identifier = useId();
  const errorId = `${identifier}-error`;
  const hintId = `${identifier}-hint`;
  const descriptionId = `${identifier}-description`;
  const sqlHintId = `${identifier}-sql-tables`;
  const isSqlQuery = operationId === 'table-sql' && spec.name === 'query' && spec.kind === 'text';
  const canLoadFile =
    operationId === 'source-geojson' && spec.name === 'text' && spec.kind === 'text';
  const textValue = formatWorkflowParameterValue(spec, value);
  const [draft, setDraft] = useState<string | null>(null);
  const [fileProblem, setFileProblem] = useState<string | null>(null);
  const focused = useRef(false);
  const currentDraft = useRef(draft);
  currentDraft.current = draft;
  const textArea = useRef<HTMLTextAreaElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const fileRevision = useRef(0);
  const previousValue = useRef({textValue, bindingIdentity});
  const submittedValue = useRef<{textValue: string; bindingIdentity: string} | null>(null);
  const postponedExternalValue = useRef(false);
  const deferred = spec.kind === 'text' || spec.kind === 'expression';
  const text = draft ?? textValue;
  const problem = fileProblem ?? getWorkflowParameterProblem(spec, draft === null ? value : draft);
  const hint = formatWorkbenchParameterHint(spec, differentDefault);
  // An undo replaces the transient text, except while the user is finishing invalid input.
  useEffect(() => {
    if (
      previousValue.current.textValue === textValue &&
      previousValue.current.bindingIdentity === bindingIdentity
    )
      return;
    previousValue.current = {textValue, bindingIdentity};
    const submitted = submittedValue.current;
    if (
      submitted?.textValue === textValue &&
      (submitted.bindingIdentity === bindingIdentity ||
        (submitted.bindingIdentity === 'default' && bindingIdentity === 'static'))
    ) {
      submittedValue.current = null;
      return;
    }
    submittedValue.current = null;
    fileRevision.current++;
    const pending = currentDraft.current;
    if (focused.current && pending !== null && getWorkflowParameterProblem(spec, pending)) {
      postponedExternalValue.current = true;
      return;
    }
    setDraft(null);
    setFileProblem(null);
  }, [textValue, bindingIdentity, spec]);
  useEffect(
    () => () => {
      fileRevision.current++;
    },
    []
  );

  function changeText(next: string, commit = !deferred): void {
    fileRevision.current++;
    postponedExternalValue.current = false;
    submittedValue.current = null;
    setDraft(next);
    setFileProblem(null);
    if (commit && getWorkflowParameterProblem(spec, next) === null) {
      const coerced = coerceWorkbenchParameterText(spec, next);
      submittedValue.current = {
        textValue: formatWorkflowParameterValue(spec, coerced),
        bindingIdentity
      };
      onCommit(coerced);
    }
  }

  function commitText(): void {
    if (getWorkflowParameterProblem(spec, text) === null) {
      const coerced = coerceWorkbenchParameterText(spec, text);
      submittedValue.current = {
        textValue: formatWorkflowParameterValue(spec, coerced),
        bindingIdentity
      };
      onCommit(coerced);
    }
  }

  const control = {
    id: identifier,
    'data-testid': testId,
    'aria-label': spec.label,
    'aria-invalid': Boolean(problem),
    'aria-describedby':
      [
        spec.description ? descriptionId : null,
        isSqlQuery ? sqlHintId : null,
        hint || onReset ? hintId : null,
        problem ? errorId : null
      ]
        .filter(Boolean)
        .join(' ') || undefined,
    onFocus: () => {
      focused.current = true;
    },
    onBlur: () => {
      focused.current = false;
      if (postponedExternalValue.current) {
        postponedExternalValue.current = false;
        setDraft(null);
        setFileProblem(null);
        return;
      }
      if (deferred) commitText();
    }
  };
  let input: React.JSX.Element;
  if (spec.kind === 'boolean') {
    input = (
      <input
        {...control}
        type="checkbox"
        checked={value === true}
        onChange={event => onCommit(event.target.checked)}
      />
    );
  } else if (spec.kind === 'enum' && (spec.options?.length ?? 0) <= 3) {
    input = (
      <div {...control} role="radiogroup">
        {spec.options?.map(option => (
          <label key={option}>
            <input
              type="radio"
              name={identifier}
              value={option}
              checked={value === option}
              aria-label={`${spec.label}: ${option}`}
              onChange={() => onCommit(option)}
            />{' '}
            {option}
          </label>
        ))}
      </div>
    );
  } else if (spec.kind === 'enum' || spec.kind === 'dataset') {
    const missingDataset =
      spec.kind === 'dataset' && !datasets.some(dataset => dataset.id === value);
    input = (
      <select {...control} value={textValue} onChange={event => onCommit(event.target.value)}>
        {spec.kind === 'dataset' ? (
          <>
            {missingDataset && (
              <option value={textValue}>
                {value ? `Missing dataset "${datasetLabel || textValue}"` : 'Choose a dataset'}
              </option>
            )}
            {datasets.map(dataset => (
              <option key={dataset.id} value={dataset.id}>
                {dataset.label} · {dataset.rowCount.toLocaleString()} rows ·{' '}
                {dataset.resident ? 'resident' : 'table'}
              </option>
            ))}
          </>
        ) : (
          <>
            {!spec.options?.includes(textValue) && <option value={textValue}>{textValue}</option>}
            {spec.options?.map(option => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </>
        )}
      </select>
    );
  } else if (deferred) {
    input = (
      <textarea
        {...control}
        ref={textArea}
        rows={isSqlQuery ? 8 : spec.kind === 'text' ? 6 : 3}
        style={{fontFamily: 'monospace'}}
        value={text}
        onChange={event => changeText(event.target.value)}
        onKeyDown={event => {
          if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
            event.preventDefault();
            commitText();
          }
        }}
      />
    );
  } else {
    input = (
      <input
        {...control}
        type="text"
        value={text}
        inputMode={spec.kind === 'number' || spec.kind === 'integer' ? 'decimal' : undefined}
        list={spec.kind === 'column' ? `${identifier}-columns` : undefined}
        onChange={event => changeText(event.target.value)}
      />
    );
  }

  return (
    <>
      <label htmlFor={identifier}>
        {spec.label}
        {spec.unit && <span className="unit">{spec.unit === 'crs' ? 'CRS units' : spec.unit}</span>}
      </label>
      {input}
      {isSqlQuery && (
        <span className="hint" id={sqlHintId}>
          Tables: input1, input2, input3
        </span>
      )}
      {spec.description && (
        <span className="hint" id={descriptionId}>
          {spec.description}
        </span>
      )}
      {(hint || onReset) && (
        <ParameterHintRow id={hintId}>
          {hint && (
            <span className="hint" title={hint}>
              {hint}
            </span>
          )}
          {onReset && (
            <WorkbenchButton type="button" aria-label={`Reset ${spec.label}`} onClick={onReset}>
              Reset
            </WorkbenchButton>
          )}
        </ParameterHintRow>
      )}
      {problem && (
        <span id={errorId} data-testid={errorTestId} className="error" role="alert">
          {contextLabel ? `${contextLabel}: ` : ''}
          {spec.label} {problem}.
        </span>
      )}
      {spec.kind === 'text' && (
        <>
          <span className="hint">
            {text.length.toLocaleString()} /{' '}
            {(spec.maximumLength ?? WORKFLOW_LIMITS.maximumTextCharacters).toLocaleString()}{' '}
            characters
          </span>
          {canLoadFile && (
            <>
              <WorkbenchButton
                type="button"
                onClick={() => fileInput.current?.click()}
                aria-label={`Load file for ${spec.label}`}
              >
                Load file
              </WorkbenchButton>
              <input
                ref={fileInput}
                type="file"
                accept=".json,.geojson"
                hidden
                aria-label={`File for ${spec.label}`}
                onChange={async event => {
                  const file = event.target.files?.[0];
                  event.target.value = '';
                  if (!file) return;
                  const revision = ++fileRevision.current;
                  const limit = spec.maximumLength ?? WORKFLOW_LIMITS.maximumTextCharacters;
                  // UTF-8 files may use several bytes per character; validate decoded text below.
                  if (file.size > Math.max(limit * 4, 4)) {
                    setFileProblem(
                      `file is too large; the limit is ${limit.toLocaleString()} characters`
                    );
                    return;
                  }
                  try {
                    const loadedText = await file.text();
                    if (revision === fileRevision.current) {
                      const loadedProblem = getWorkflowParameterProblem(spec, loadedText);
                      if (loadedProblem) setFileProblem(loadedProblem);
                      else changeText(loadedText, true);
                    }
                  } catch (error) {
                    if (revision === fileRevision.current)
                      setFileProblem(
                        `file could not be read: ${error instanceof Error ? error.message : String(error)}`
                      );
                  }
                }}
              />
            </>
          )}
        </>
      )}
      {spec.kind === 'column' && (
        <datalist id={`${identifier}-columns`}>
          {suggestions.columns.map(column => (
            <option key={column} value={column} />
          ))}
        </datalist>
      )}
      {(spec.kind === 'column-list' || spec.kind === 'expression') && (
        <div>
          {suggestions.columns.map(column => (
            <WorkbenchButton
              type="button"
              key={column}
              aria-label={`Insert column ${column} into ${spec.label}`}
              onMouseDown={event => event.preventDefault()}
              onClick={() => {
                if (spec.kind === 'column-list') {
                  const columns = text
                    .split(',')
                    .map(name => name.trim())
                    .filter(Boolean);
                  if (!columns.includes(column)) changeText([...columns, column].join(', '));
                } else {
                  const element = textArea.current;
                  const inserted = insertWorkbenchExpressionColumn(
                    text,
                    column,
                    element?.selectionStart ?? text.length,
                    element?.selectionEnd ?? text.length
                  );
                  changeText(inserted.text);
                  queueMicrotask(() => {
                    element?.focus();
                    element?.setSelectionRange(inserted.cursor, inserted.cursor);
                  });
                }
              }}
            >
              {column}
            </WorkbenchButton>
          ))}
        </div>
      )}
      {suggestions.hints.map(hint => (
        <span key={hint} className="hint">
          {hint}
        </span>
      ))}
      {spec.kind === 'expression' && (
        <details>
          <WorkbenchButton as="summary">Functions</WorkbenchButton>
          <span>{WORKFLOW_EXPRESSION_FUNCTIONS.join(', ')}</span>
        </details>
      )}
    </>
  );
}

/** Generated parameter form of one step: one control per parameter spec, inline range errors, promote toggles. */
export function WorkbenchParameterForm({
  store,
  stepId
}: {
  store: WorkbenchStore;
  stepId: string;
}): React.JSX.Element | null {
  const state = useWorkbenchState(store);
  const document = state.document;
  const step = document?.steps.find(candidate => candidate.id === stepId);
  const operation = step && store.registry.get(step.operationId);
  const datasets = useMemo(() => store.host.listDatasets(), [store, state.document, state.runner]);
  const rows = useMemo(
    () =>
      document && step && operation
        ? getWorkbenchParameterRows(document, step, operation).map(row => ({
            ...row,
            controlSpec: getWorkbenchParameterControlSpec(
              row.spec,
              document.inputs.find(input => input.id === row.inputId)
            ),
            suggestions: ['column', 'column-list', 'expression'].includes(row.spec.kind)
              ? getWorkbenchColumnSuggestions(
                  document,
                  step,
                  operation,
                  row.spec,
                  store.registry,
                  state.runner,
                  datasets
                )
              : undefined
          }))
        : [],
    [document, step, operation, store, state.runner, datasets]
  );
  const resolvedValues = useMemo(
    () =>
      document && step && operation ? resolveWorkflowParameters(document, step, operation) : {},
    [document, step, operation]
  );
  const drawings = useMemo(
    () =>
      operation?.id === 'source-drawing' && store.host.listDrawings
        ? store.host.listDrawings()
        : null,
    [store, operation, state.document, state.runner]
  );
  if (!document || !step || !operation) return null;
  return (
    <>
      {rows.map(row => {
        const spec = row.spec;
        const workflowInput = document.inputs.find(input => input.id === row.inputId);
        const datasetLabel = resolvedValues['datasetLabel'];
        return (
          <ParameterField key={`${stepId}:${spec.name}`}>
            <WorkbenchParameterValue
              spec={row.controlSpec}
              operationId={operation.id}
              value={row.value}
              testId={WORKBENCH_TEST_IDS.parameter(spec.name)}
              errorTestId={WORKBENCH_TEST_IDS.parameterError(spec.name)}
              suggestions={row.suggestions}
              datasets={datasets}
              datasetLabel={typeof datasetLabel === 'string' ? datasetLabel : undefined}
              bindingIdentity={row.inputId ? `input:${row.inputId}` : row.bindingKind}
              contextLabel={step.label ?? operation.title}
              differentDefault={row.isDefault ? undefined : spec.defaultValue}
              onReset={
                row.bindingKind === 'default'
                  ? undefined
                  : () =>
                      store.edit(`parameter:${stepId}:${spec.name}`, currentDocument =>
                        setWorkflowParameter(currentDocument, stepId, spec.name, undefined)
                      )
              }
              onCommit={value =>
                store.edit(`parameter:${stepId}:${spec.name}`, currentDocument => {
                  const currentStep = currentDocument.steps.find(
                    candidate => candidate.id === stepId
                  );
                  const binding = currentStep?.parameters[spec.name];
                  if (binding?.from === 'input')
                    return updateWorkflowInput(currentDocument, binding.inputId, {value});
                  const dataset =
                    spec.kind === 'dataset'
                      ? datasets.find(candidate => candidate.id === value)
                      : undefined;
                  const values =
                    dataset && currentStep
                      ? getWorkbenchDatasetParameters(currentStep, operation, spec.name, dataset)
                      : {[spec.name]: value};
                  return Object.entries(values).reduce(
                    (changed, [name, parameterValue]) =>
                      setWorkflowParameter(changed, stepId, name, {
                        from: 'static',
                        value: parameterValue
                      }),
                    currentDocument
                  );
                })
              }
            />
            {row.inputId && (
              <span className="hint">Workflow input: {workflowInput?.label ?? row.inputId}</span>
            )}
            <div>
              {PROMOTABLE_KINDS.has(spec.kind) && (
                <WorkbenchButton
                  type="button"
                  data-testid={WORKBENCH_TEST_IDS.parameterPromote(spec.name)}
                  aria-label={`${row.inputId ? 'Remove workflow input for' : 'Make workflow input for'} ${spec.label}`}
                  aria-pressed={Boolean(row.inputId)}
                  title="Make this a workflow input"
                  onClick={() =>
                    store.edit(`promote:${stepId}:${spec.name}`, currentDocument =>
                      row.inputId
                        ? demoteWorkflowParameter(currentDocument, stepId, spec.name)
                        : promoteWorkflowParameter(
                            currentDocument,
                            store.registry,
                            stepId,
                            spec.name
                          ).document
                    )
                  }
                >
                  {row.inputId ? 'Demote' : 'Promote'}
                </WorkbenchButton>
              )}
            </div>
          </ParameterField>
        );
      })}
      {drawings !== null && (
        <WorkbenchMuted as="p" aria-live="polite">
          {formatWorkbenchDrawings(drawings)}
        </WorkbenchMuted>
      )}
    </>
  );
}
