// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  AnalysisSensitivityStrip,
  type AnalysisSensitivityControl
} from './analysis-sensitivity-strip';
import {getAnalysisSensitivityRuntimeKey} from './analysis-sensitivity-model';
import {createContext, useContext, useEffect, useRef, useState, type ReactNode} from 'react';
import {
  getWorkflowParameterProblem,
  type CatalogParameterSpec,
  type CatalogParameterValue,
  type WorkflowParameterSpec
} from '@luma-contributors/kepler';
import {WorkbenchParameterValue} from './workflow/workbench-parameter-form';
import {ANALYSIS_SHEET_PREVIEW_POLICY, type AnalysisSheetRow} from './analysis-sheet-model';

/** Discovery, saved invocations and lens access stay with the catalog host. */
const CoordinateModeContext = createContext({
  coordinateSystem: 'EPSG:4326',
  setCoordinateSystem: (_value: string): void => {}
});

export function AnalysisSheet({children}: {children: ReactNode}) {
  const [coordinateSystem, setCoordinateSystem] = useState('EPSG:4326');
  return (
    <CoordinateModeContext.Provider value={{coordinateSystem, setCoordinateSystem}}>
      <section aria-label="Analysis sheet" data-testid="ax-analysis-sheet">
        {children}
      </section>
    </CoordinateModeContext.Provider>
  );
}

/** Resident executors keep their settings contract while using the shared generated controls. */
export type ResidentAnalysisSheetField = {
  specification: WorkflowParameterSpec;
  controlId: string;
  value: CatalogParameterValue;
  source: string;
  columns?: readonly string[];
};

/** Manual Apply only: editing a resident form never schedules an unguarded live rebuild. */
export function ResidentAnalysisSheetForm({
  fields,
  checks,
  method,
  submitLabel,
  actions = [],
  children,
  onDraftChange,
  onChange,
  onApply
}: {
  fields: readonly ResidentAnalysisSheetField[];
  checks: readonly string[];
  method: string;
  submitLabel: string;
  /** Optional resident actions share the same transient validation and current field values. */
  actions?: readonly {
    label: string;
    disabled?: boolean;
    onInvoke(values: Readonly<Record<string, CatalogParameterValue>>): void | Promise<void>;
  }[];
  /** Related draft actions must read through the same transient validation as Apply. */
  children?(form: {
    disabled: boolean;
    readValues(): Readonly<Record<string, CatalogParameterValue>>;
  }): ReactNode;
  /** Mark explicit edits or invalidate pending work before a focused text draft commits. */
  onDraftChange?(name: string): void;
  onChange(name: string, value: CatalogParameterValue): void;
  onApply(values: Readonly<Record<string, CatalogParameterValue>>): void;
}) {
  const invalidDrafts = useRef(new Map<string, string>());
  const pendingDrafts = useRef(new Map<string, string>());
  const current = useRef({fields, checks});
  current.current = {fields, checks};
  const [, refresh] = useState(0);
  const [error, setError] = useState('');
  function getCurrentFields() {
    return current.current.fields.map(field => ({
      ...field,
      value: pendingDrafts.current.get(field.specification.name) ?? field.value
    }));
  }
  function getProblems() {
    return [
      ...current.current.checks,
      ...getCurrentFields().flatMap(field => {
        const problem =
          invalidDrafts.current.get(field.specification.name) ??
          getWorkflowParameterProblem(field.specification, field.value);
        return problem ? [`${field.specification.label}: ${problem}`] : [];
      })
    ];
  }
  function readValues() {
    const failures = getProblems();
    if (failures.length) throw new Error(failures.join('; '));
    return Object.fromEntries(
      getCurrentFields().map(field => [field.specification.name, field.value])
    );
  }
  const problems = getProblems();
  function renderFields(group: readonly ResidentAnalysisSheetField[]) {
    return group.map(field => (
      <div
        key={field.specification.name}
        data-ax-parameter={field.specification.name}
        data-apply="rebuild"
      >
        <WorkbenchParameterValue
          spec={{...field.specification, apply: 'rebuild'}}
          controlId={field.controlId}
          enumControl="select"
          testId={field.controlId}
          value={field.value}
          coordinateSystem={
            getCurrentFields().find(
              candidate => candidate.specification.name === 'coordinateSystem'
            )?.value
          }
          suggestions={{columns: [...(field.columns ?? [])], hints: []}}
          commitOnRelease
          onDraftChange={text => {
            const name = field.specification.name;
            if (text !== undefined) pendingDrafts.current.set(name, text);
            onDraftChange?.(name);
          }}
          onDraftReset={() => {
            const name = field.specification.name;
            const changed = pendingDrafts.current.delete(name);
            const invalid = invalidDrafts.current.delete(name);
            if (changed || invalid) refresh(version => version + 1);
          }}
          onValidityChange={problem => {
            const name = field.specification.name;
            const previous = invalidDrafts.current.get(name);
            if (problem) invalidDrafts.current.set(name, problem);
            else invalidDrafts.current.delete(name);
            if (previous !== (problem ?? undefined)) refresh(version => version + 1);
          }}
          onCommit={value => {
            const name = field.specification.name;
            pendingDrafts.current.delete(name);
            // Guard a submit captured before React renders the committed parameter.
            current.current = {
              ...current.current,
              fields: current.current.fields.map(candidate =>
                candidate.specification.name === name ? {...candidate, value} : candidate
              )
            };
            onChange(name, value);
          }}
        />
        <small data-testid="ax-resident-parameter-source">{field.source}</small>
      </div>
    ));
  }
  return (
    <AnalysisSheet>
      <p className="hint" data-testid="ax-resident-method">
        {method}
      </p>
      {children?.({disabled: problems.length > 0, readValues})}
      <form
        onSubmit={event => {
          event.preventDefault();
          const failures = getProblems();
          if (failures.length) {
            setError(failures.join('; '));
            return;
          }
          try {
            onApply(readValues());
            setError('');
          } catch (failure) {
            setError(failure instanceof Error ? failure.message : String(failure));
          }
        }}
      >
        <fieldset>
          <legend>Data</legend>
          {renderFields(fields.filter(field => field.specification.fieldRole))}
        </fieldset>
        <fieldset>
          <legend>Key settings</legend>
          {renderFields(
            fields.filter(
              field => !field.specification.fieldRole && field.specification.level !== 'advanced'
            )
          )}
        </fieldset>
        {fields.some(field => field.specification.level === 'advanced') && (
          <details>
            <summary>All settings</summary>
            {renderFields(
              fields.filter(
                field => !field.specification.fieldRole && field.specification.level === 'advanced'
              )
            )}
          </details>
        )}
        <section aria-label="Checks" role="status">
          <h3>Checks</h3>
          {problems.length ? (
            <ul>
              {problems.map(problem => (
                <li key={problem}>{problem}</li>
              ))}
            </ul>
          ) : (
            <p>Ready to apply to the full analysis population.</p>
          )}
        </section>
        <AnalysisSheetPreviewScope
          available={false}
          reason="Manual Apply. The current result stays visible while you edit; apply the settings to rebuild this analysis."
        />
        <button type="submit" disabled={problems.length > 0}>
          {submitLabel}
        </button>
        {actions.map(action => (
          <button
            key={action.label}
            type="button"
            disabled={action.disabled || problems.length > 0}
            onClick={async () => {
              const failures = getProblems();
              if (failures.length) {
                setError(failures.join('; '));
                return;
              }
              try {
                await action.onInvoke(readValues());
                setError('');
              } catch (failure) {
                setError(failure instanceof Error ? failure.message : String(failure));
              }
            }}
          >
            {action.label}
          </button>
        ))}
        {error && (
          <p className="error" role="alert">
            {error}
          </p>
        )}
      </form>
    </AnalysisSheet>
  );
}

export function AnalysisSheetParameter({
  spec,
  row,
  value,
  source,
  onChange,
  onValidityChange,
  sensitivity
}: {
  // Retain the catalog spec at the host boundary; controls share the workflow adapter.
  spec: CatalogParameterSpec;
  row: AnalysisSheetRow;
  value: CatalogParameterValue;
  source: string;
  onChange(value: unknown): void;
  sensitivity?: AnalysisSensitivityControl;
  onValidityChange?(problem: string | null): void;
}) {
  const {coordinateSystem, setCoordinateSystem} = useContext(CoordinateModeContext);
  useEffect(() => {
    if (spec.name === 'coordinateSystem') setCoordinateSystem(String(value));
  }, [spec.name, value, setCoordinateSystem]);
  if (spec.aliasFor) return null;
  return (
    <div data-ax-parameter={row.specification.name} data-apply={row.apply}>
      <small data-testid="ax-parameter-apply">
        {row.apply === 'live'
          ? 'Live · when retained preview is available'
          : 'Rebuild · applies on release'}
      </small>
      {row.input && <small>{row.required ? 'Required field' : 'Optional field'}</small>}
      <WorkbenchParameterValue
        spec={{
          ...row.specification,
          ...spec,
          kind: row.specification.kind
        }}
        coordinateSystem={coordinateSystem}
        value={value}
        onCommit={onChange}
        onValidityChange={onValidityChange}
        commitOnRelease={row.apply === 'rebuild'}
        testId={`catalog-parameter-${row.specification.name}`}
        errorTestId={`catalog-parameter-error-${row.specification.name}`}
        suggestions={{columns: row.columns, hints: []}}
      />
      {row.apply === 'live' && !sensitivity && (
        <small>
          Sensitivity unavailable: this parameter needs a qualified retained GPU executor and a
          ready input assignment.
        </small>
      )}
      {sensitivity && (
        <AnalysisSensitivityStrip
          key={getAnalysisSensitivityRuntimeKey(sensitivity)}
          control={sensitivity}
        />
      )}
      <small data-testid="catalog-parameter-source" style={{display: 'block'}}>
        {source}
      </small>
    </div>
  );
}

export function AnalysisSheetPreviewScope({
  available,
  reason
}: {
  available: boolean;
  reason: string;
}) {
  return (
    <section
      aria-label="Preview scope"
      data-testid="ax-preview-scope"
      data-scope={ANALYSIS_SHEET_PREVIEW_POLICY.scope}
    >
      <h3>Preview scope</h3>
      <p>{ANALYSIS_SHEET_PREVIEW_POLICY.description}</p>
      <p>{ANALYSIS_SHEET_PREVIEW_POLICY.inference}</p>
      <p data-testid="ax-preview-availability" data-available={available}>
        {reason}
      </p>
    </section>
  );
}
