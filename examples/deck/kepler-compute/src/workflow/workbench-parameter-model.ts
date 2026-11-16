// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  checkWorkflowConnection,
  getKeplerWorkbenchParameterDefaults,
  profileKeplerWorkbenchData,
  getWorkflowAnalysisProfiles,
  setWorkflowParameter,
  updateWorkflowInput,
  resolveWorkflowParameterOverrides,
  type KeplerWorkbenchProfiles,
  type WorkflowHost,
  getWorkflowColumnNames,
  getWorkflowSuggestions,
  isWorkflowParameterVisible,
  resolveWorkflowParameters,
  type CatalogParameterValue,
  type WorkflowConnection,
  type WorkflowDatasetProfile,
  type WorkflowDrawing,
  type WorkflowDocument,
  type WorkflowInput,
  type WorkflowIssue,
  type WorkflowInputPortSpec,
  type WorkflowOperation,
  type WorkflowOperationRegistry,
  type WorkflowOutputPortSpec,
  type WorkflowParameterSpec,
  type WorkflowRunnerState,
  type WorkflowStep,
  type WorkflowSuggestion
} from '@luma-contributors/kepler';
import {WORKBENCH_PORT_LABELS} from './workbench-contract';

export type WorkbenchParameterRow = {
  spec: WorkflowParameterSpec;
  value: CatalogParameterValue;
  bindingKind: 'default' | 'static' | 'input';
  isDefault: boolean;
  inputId?: string;
};

export function formatWorkbenchParameterHint(
  spec: WorkflowParameterSpec,
  differentDefault?: CatalogParameterValue
): string {
  const range =
    spec.minimum !== undefined && spec.maximum !== undefined
      ? `${spec.minimum.toLocaleString()} to ${spec.maximum.toLocaleString()}`
      : spec.minimum !== undefined
        ? `At least ${spec.minimum.toLocaleString()}`
        : spec.maximum !== undefined
          ? `At most ${spec.maximum.toLocaleString()}`
          : '';
  const defaultText =
    differentDefault === undefined
      ? ''
      : `default ${
          typeof differentDefault === 'number'
            ? differentDefault.toLocaleString()
            : Array.isArray(differentDefault)
              ? differentDefault.map(value => value.toLocaleString()).join(', ')
              : String(differentDefault)
        }`;
  return [range, defaultText].filter(Boolean).join(' · ');
}

export function hasWorkbenchRunOptionOverrides(step: WorkflowStep): boolean {
  return (
    step.engine !== undefined ||
    step.onFallback === 'stop' ||
    Boolean(step.display?.length || step.disabled || step.note) ||
    Object.keys(step.displayStyle ?? {}).length > 0
  );
}

/** A failed repair provider must not prevent inspection of the issue it was meant to repair. */
export function getWorkbenchIssueSuggestions(
  document: WorkflowDocument,
  registry: WorkflowOperationRegistry,
  runner: WorkflowRunnerState | null,
  issues: readonly WorkflowIssue[]
): WorkflowSuggestion[][] {
  return issues
    .filter(issue => issue.severity !== 'info')
    .map(issue => {
      try {
        return getWorkflowSuggestions(document, registry, runner, issue);
      } catch {
        return [];
      }
    });
}

export function getWorkbenchDisplayColumns(
  stepId: string,
  output: WorkflowOutputPortSpec,
  runner: WorkflowRunnerState | null
): string[] {
  const data = runner?.steps[stepId]?.result?.outputs[output.name];
  return data ? getWorkflowColumnNames(data) : [...output.identityColumns];
}

export function formatWorkbenchDrawings(drawings: readonly WorkflowDrawing[]): string {
  return drawings.length
    ? `${drawings.length.toLocaleString()} ${drawings.length === 1 ? 'shape' : 'shapes'}: ${drawings.map(drawing => drawing.label).join(', ')}`
    : 'No shapes drawn yet. Use the draw tools on the map.';
}

export function getWorkbenchInspectorIssues(
  issues: readonly WorkflowIssue[],
  stepLabel: string
): {problems: WorkflowIssue[]; notes: WorkflowIssue[]} {
  const problems: WorkflowIssue[] = [];
  const notes: WorkflowIssue[] = [];
  for (const issue of issues) {
    let message = issue.message;
    if (message.startsWith(`${stepLabel}:`) || message.startsWith(`${stepLabel} `)) {
      message = message.slice(stepLabel.length).replace(/^:\s*/, '').trimStart();
    }
    // Only declared counts in prose are localized; identifiers and already formatted counts stay intact.
    message = message.replace(
      /(^|\s)(\d+)(?=\s+(?:rows|vertices)\b)/g,
      (match: string, space: string, count: string) => {
        const value = Number(count);
        return Number.isSafeInteger(value) ? `${space}${value.toLocaleString()}` : match;
      }
    );
    const formatted = {...issue, message};
    if (issue.severity === 'info') notes.push(formatted);
    else problems.push(formatted);
  }
  return {problems, notes};
}

/** A shared input may narrow a parameter, but cannot relax the operation's contract. */
export function getWorkbenchParameterControlSpec(
  spec: WorkflowParameterSpec,
  input?: WorkflowInput
): WorkflowParameterSpec {
  if (!input) return spec;
  return {
    ...spec,
    kind: spec.kind === 'number' && input.kind === 'integer' ? 'integer' : spec.kind,
    ...(input.minimum === undefined
      ? {}
      : {minimum: Math.max(spec.minimum ?? -Infinity, input.minimum)}),
    ...(input.maximum === undefined
      ? {}
      : {maximum: Math.min(spec.maximum ?? Infinity, input.maximum)}),
    ...(spec.kind === 'enum' && input.kind === 'enum'
      ? {options: spec.options?.filter(option => input.options?.includes(option)) ?? []}
      : {})
  };
}

function equalValues(
  left: CatalogParameterValue | undefined,
  right: CatalogParameterValue | undefined
): boolean {
  return Array.isArray(left) && Array.isArray(right)
    ? left.length === right.length && left.every((value, index) => value === right[index])
    : left === right;
}

/** Resolve before filtering so hidden parameters can still control visibility. */
export function getWorkbenchParameterRows(
  document: WorkflowDocument,
  step: WorkflowStep,
  operation: WorkflowOperation,
  analysisOptions?: {profilesByInput: KeplerWorkbenchProfiles; executionEngine: 'gpu' | 'cpu'}
): WorkbenchParameterRow[] {
  const overrides = resolveWorkflowParameterOverrides(document, step, operation);
  const defaults =
    operation.analysisDescriptor && analysisOptions
      ? getKeplerWorkbenchParameterDefaults(operation.analysisBinding ?? operation.id, undefined, {
          ...analysisOptions,
          overrides
        })
      : undefined;
  const rows = operation.parameters.map(originalSpec => {
    const resolution = defaults?.resolutions.find(entry => entry.parameter === originalSpec.name);
    const baseline =
      defaults && analysisOptions && Object.hasOwn(overrides, originalSpec.name)
        ? getKeplerWorkbenchParameterDefaults(
            operation.analysisBinding ?? operation.id,
            undefined,
            {
              ...analysisOptions,
              overrides: Object.fromEntries(
                Object.entries(overrides).filter(([name]) => name !== originalSpec.name)
              )
            }
          )
        : defaults;
    const defaultValue = baseline?.values[originalSpec.name] ?? originalSpec.defaultValue;
    const spec = defaults
      ? {
          ...originalSpec,
          defaultValue,
          description: [
            originalSpec.description,
            resolution
              ? `Default source: ${resolution.source}. ${resolution.resolution?.explanation ?? resolution.reason ?? ''}`
              : ''
          ]
            .filter(Boolean)
            .join(' ')
        }
      : originalSpec;
    const binding = Object.hasOwn(step.parameters, spec.name)
      ? step.parameters[spec.name]
      : undefined;
    const value =
      binding?.from === 'static'
        ? binding.value
        : binding?.from === 'input'
          ? (document.inputs.find(input => input.id === binding.inputId)?.value ??
            spec.defaultValue)
          : spec.defaultValue;
    return {
      spec,
      value: value ?? '',
      bindingKind: binding?.from ?? 'default',
      isDefault: equalValues(value, spec.defaultValue),
      ...(binding?.from === 'input' ? {inputId: binding.inputId} : {})
    } satisfies WorkbenchParameterRow;
  });
  const values = Object.fromEntries(rows.map(row => [row.spec.name, row.value]));
  return rows.filter(
    row =>
      !defaults?.inactive.includes(row.spec.name) && isWorkflowParameterVisible(row.spec, values)
  );
}

/** Profiles are reused from AX sources and successful immutable intermediate results. */
export function getWorkbenchAnalysisOptions(
  document: WorkflowDocument,
  step: WorkflowStep,
  operation: WorkflowOperation,
  registry: WorkflowOperationRegistry,
  host: WorkflowHost,
  runner?: WorkflowRunnerState | null
): {profilesByInput: KeplerWorkbenchProfiles; executionEngine: 'gpu' | 'cpu'} | undefined {
  if (!operation.analysisDescriptor) return undefined;
  const profilesByInput = getWorkflowAnalysisProfiles(document, step, registry, host);
  for (const [input, connection] of Object.entries(step.inputs)) {
    const state = runner?.steps[connection.stepId];
    const data = state?.status === 'done' ? state.result?.outputs[connection.output] : undefined;
    if (data) profilesByInput[input] = data.analysisProfile ?? profileKeplerWorkbenchData(data);
  }
  return {
    profilesByInput,
    executionEngine:
      (step.engine ?? document.settings.engine) === 'cpu' || !host.getDevice() ? 'cpu' : 'gpu'
  };
}

/** Apply the method's settings while retaining user field mappings and promoted controls. */
export function applyWorkbenchCuratedPreset(
  document: WorkflowDocument,
  step: WorkflowStep,
  operation: WorkflowOperation,
  curatedAnalysisId: string
): WorkflowDocument {
  const preset = operation.analysisPresets?.find(
    candidate => candidate.curatedAnalysisId === curatedAnalysisId
  );
  if (!preset) return document;
  let changed = {
    ...document,
    steps: document.steps.map(candidate =>
      candidate.id === step.id ? {...candidate, curatedAnalysisId} : candidate
    )
  };
  for (const [name, value] of Object.entries(preset.executionParameters)) {
    if (operation.parameters.find(parameter => parameter.name === name)?.fieldRole) continue;
    const binding = step.parameters[name];
    changed =
      binding?.from === 'input'
        ? updateWorkflowInput(changed, binding.inputId, {value})
        : setWorkflowParameter(changed, step.id, name, {from: 'static', value});
  }
  return changed;
}

/** Call only after getWorkflowParameterProblem accepts the text. */
export function coerceWorkbenchParameterText(
  spec: WorkflowParameterSpec,
  text: string
): CatalogParameterValue {
  if (spec.kind === 'number' || spec.kind === 'integer') return Number(text);
  if (spec.kind === 'number-list') return text.split(',').map(value => Number(value.trim()));
  return text;
}

export type WorkbenchColumnSuggestions = {columns: string[]; hints: string[]};

/** Latest results, including stale results, are useful suggestions without becoming bindings. */
export function getWorkbenchColumnSuggestions(
  document: WorkflowDocument,
  step: WorkflowStep,
  operation: WorkflowOperation,
  spec: WorkflowParameterSpec,
  registry: WorkflowOperationRegistry,
  runner: WorkflowRunnerState | null,
  datasets: readonly WorkflowDatasetProfile[] = [],
  profilesByInput?: KeplerWorkbenchProfiles
): WorkbenchColumnSuggestions {
  const role = spec.fieldRole;
  const profile = spec.columnsFrom ? profilesByInput?.[spec.columnsFrom] : undefined;
  if (role && profile)
    return {
      columns: profile.columns
        .filter(
          column =>
            role.accepts.includes(column.type) ||
            (column.type === 'integer' && role.accepts.includes('number'))
        )
        .map(column => column.name),
      hints: []
    };
  const datasetParameter = operation.parameters.find(
    parameter => parameter.name === spec.columnsFrom && parameter.kind === 'dataset'
  );
  if (datasetParameter) {
    const datasetId = resolveWorkflowParameters(document, step, operation)[datasetParameter.name];
    return {
      columns:
        datasets.find(dataset => dataset.id === datasetId)?.fields.map(field => field.name) ?? [],
      hints: []
    };
  }
  const columns = new Set<string>();
  const hints = new Set<string>();
  for (const input of operation.inputs) {
    if (spec.columnsFrom && spec.columnsFrom !== input.name) continue;
    const connection = step.inputs[input.name];
    if (!connection) continue;
    const producer = document.steps.find(candidate => candidate.id === connection.stepId);
    if (!producer) continue;
    const producerOperation = registry.get(producer.operationId);
    const data = runner?.steps[producer.id]?.result?.outputs[connection.output];
    if (data) {
      for (const column of getWorkflowColumnNames(data)) {
        const type = data.table.columnTypes[column];
        const accepted =
          !spec.fieldRole ||
          spec.fieldRole.accepts.some(accepts =>
            accepts === 'number'
              ? type === 'float64' || type === 'int32'
              : accepts === 'integer'
                ? type === 'int32'
                : accepts === 'category'
                  ? type === 'string'
                  : accepts === 'boolean'
                    ? type === 'boolean'
                    : false
          );
        if (accepted) columns.add(column);
      }
    } else {
      for (const column of producerOperation?.outputs.find(
        output => output.name === connection.output
      )?.identityColumns ?? [])
        columns.add(column);
      hints.add(
        `Run ${producer.label ?? producerOperation?.title ?? producer.operationId} to list its columns`
      );
    }
  }
  return {columns: [...columns], hints: [...hints]};
}

export type WorkbenchConnectionChoice = {
  value: string;
  label: string;
  connection: WorkflowConnection;
};

export function getWorkbenchConnectionChoices(
  document: WorkflowDocument,
  registry: WorkflowOperationRegistry,
  _runner: WorkflowRunnerState | null,
  stepId: string,
  input: WorkflowInputPortSpec
): WorkbenchConnectionChoice[] {
  const choices: WorkbenchConnectionChoice[] = [];
  for (const producer of document.steps) {
    if (producer.id === stepId) continue;
    const operation = registry.get(producer.operationId);
    for (const output of operation?.outputs ?? []) {
      const connection = {stepId: producer.id, output: output.name};
      if (
        !checkWorkflowConnection(document, registry, {
          from: connection,
          to: {stepId, input: input.name}
        }).valid
      )
        continue;
      choices.push({
        value: JSON.stringify([producer.id, output.name]),
        label: `${producer.label ?? operation!.title} · ${output.label} (${WORKBENCH_PORT_LABELS[output.kind]})`,
        connection
      });
    }
  }
  return choices;
}

export function formatWorkbenchCapacity(
  port: WorkflowInputPortSpec | WorkflowOutputPortSpec
): string {
  return `Up to ${port.maximumRows.toLocaleString()} rows${
    'maximumVertices' in port && port.maximumVertices !== undefined
      ? ` · ${port.maximumVertices.toLocaleString()} vertices`
      : ''
  }`;
}

/** Custom values and workflow input bindings take precedence over dataset inference. */
export function getWorkbenchDatasetParameters(
  step: WorkflowStep,
  operation: WorkflowOperation,
  parameterName: string,
  dataset: WorkflowDatasetProfile
): Record<string, CatalogParameterValue> {
  const values: Record<string, CatalogParameterValue> = {[parameterName]: dataset.id};
  if (operation.parameters.some(spec => spec.name === 'datasetLabel'))
    values['datasetLabel'] = dataset.label;
  const inferred: Record<string, string> = dataset.geometryFields[0]
    ? {geometryMode: 'geojson-field', geometryField: dataset.geometryFields[0]}
    : dataset.pointColumns
      ? {
          geometryMode: 'point-columns',
          longitudeField: dataset.pointColumns.longitude,
          latitudeField: dataset.pointColumns.latitude
        }
      : {geometryMode: 'none'};
  for (const [name, value] of Object.entries(inferred)) {
    const spec = operation.parameters.find(parameter => parameter.name === name);
    const binding = step.parameters[name];
    if (
      spec &&
      (!binding || (binding.from === 'static' && equalValues(binding.value, spec.defaultValue)))
    )
      values[name] = value;
  }
  return values;
}

/** A source switch starts from the new source's defaults before inferring a sole host dataset. */
export function getWorkbenchSourceParameters(
  step: WorkflowStep,
  operation: WorkflowOperation,
  datasets: readonly WorkflowDatasetProfile[]
): Record<string, CatalogParameterValue> {
  if (operation.id !== 'source-dataset' || datasets.length !== 1) return {};
  const datasetParameter = operation.parameters.find(parameter => parameter.kind === 'dataset');
  return datasetParameter
    ? getWorkbenchDatasetParameters(
        {...step, parameters: {}},
        operation,
        datasetParameter.name,
        datasets[0]
      )
    : {};
}

export function insertWorkbenchExpressionColumn(
  text: string,
  column: string,
  start: number,
  end = start
): {text: string; cursor: number} {
  const name =
    /^[a-zA-Z_][a-zA-Z0-9_]*$/.test(column) &&
    !['true', 'false', 'null', 'and', 'or', 'not'].includes(column)
      ? column
      : `\`${column.replace(/\\/g, '\\\\').replace(/`/g, '\\`')}\``;
  return {text: text.slice(0, start) + name + text.slice(end), cursor: start + name.length};
}

export function getWorkbenchInspectorTabIndex(key: string, current: number): number | null {
  if (key === 'ArrowRight' || key === 'ArrowDown') return (current + 1) % 3;
  if (key === 'ArrowLeft' || key === 'ArrowUp') return (current + 2) % 3;
  if (key === 'Home') return 0;
  if (key === 'End') return 2;
  return null;
}
