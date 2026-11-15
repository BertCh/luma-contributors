// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CatalogParameterValue} from '../analysis/catalog-binding-types';
import {
  isWorkflowId,
  makeUniqueWorkflowId,
  parseWorkflowDocument,
  serializeWorkflowDocument
} from './workflow-document';
import {getWorkflowPortCompatibility} from './workflow-operations';
import {
  WORKFLOW_LIMITS,
  WorkflowError,
  type WorkflowConnection,
  type WorkflowDocument,
  type WorkflowDisplayStyle,
  type WorkflowInput,
  type WorkflowOperationRegistry,
  type WorkflowParameterSpec,
  type WorkflowParameterBinding,
  type WorkflowSettings,
  type WorkflowStep
} from './workflow-types';

function copyParameterValue(value: CatalogParameterValue): CatalogParameterValue {
  return Array.isArray(value) ? [...value] : value;
}

function copyParameterBinding(binding: WorkflowParameterBinding): WorkflowParameterBinding {
  return binding.from === 'static'
    ? {from: 'static', value: copyParameterValue(binding.value)}
    : {from: 'input', inputId: binding.inputId};
}

function checkContentLimit(
  label: string,
  field: string,
  count: number,
  limit: keyof typeof WORKFLOW_LIMITS
): void {
  const maximum = WORKFLOW_LIMITS[limit];
  if (count > maximum)
    throw new WorkflowError(
      'limit-exceeded',
      `${label}: ${field} must be at most ${limit} (${maximum}); got ${count}.`
    );
}

// Check persistence budgets without checking operation parameter ranges.
function finishEdit(document: WorkflowDocument, label = document.title): WorkflowDocument {
  checkContentLimit(
    document.title,
    'Description characters',
    document.description?.length ?? 0,
    'maximumNoteCharacters'
  );
  for (const input of document.inputs) {
    if (typeof input.value === 'string')
      checkContentLimit(
        input.label,
        'Value characters',
        input.value.length,
        'maximumTextCharacters'
      );
  }
  for (const step of document.steps) {
    const stepLabel = step.label ?? step.operationId;
    checkContentLimit(
      stepLabel,
      'Note characters',
      step.note?.length ?? 0,
      'maximumNoteCharacters'
    );
    for (const [name, binding] of Object.entries(step.parameters)) {
      if (binding.from === 'static' && typeof binding.value === 'string')
        checkContentLimit(
          stepLabel,
          `Parameter ${name} characters`,
          binding.value.length,
          'maximumTextCharacters'
        );
    }
  }
  // Match the serializer's omission of empty presentation fields and disabled=false.
  const persisted = {
    ...document,
    title: document.title.trim(),
    description: document.description || undefined,
    steps: document.steps.map(step => ({
      ...step,
      label: step.label || undefined,
      disabled: step.disabled || undefined,
      display: step.display?.length ? step.display : undefined,
      note: step.note || undefined
    }))
  };
  checkContentLimit(
    label,
    'Document characters',
    JSON.stringify(persisted).length,
    'maximumDocumentCharacters'
  );
  return document;
}

// Edits preserve unchanged objects so snapshot history and React can share their structure.
function equalValues(left: unknown, right: unknown): boolean {
  if (Object.is(left, right)) return true;
  if (!left || !right || typeof left !== 'object' || typeof right !== 'object') return false;
  if (Array.isArray(left) !== Array.isArray(right)) return false;
  const leftFields = left as Record<string, unknown>;
  const rightFields = right as Record<string, unknown>;
  const keys = Object.keys(leftFields);
  return (
    keys.length === Object.keys(rightFields).length &&
    keys.every(
      key => Object.hasOwn(rightFields, key) && equalValues(leftFields[key], rightFields[key])
    )
  );
}

function requireStep(document: WorkflowDocument, stepId: string): WorkflowStep {
  const step = document.steps.find(candidate => candidate.id === stepId);
  if (!step)
    throw new WorkflowError('unknown-step', `Step ${stepId}: The step does not exist.`, {stepId});
  return step;
}

function requireOperation(
  step: Pick<WorkflowStep, 'id' | 'label' | 'operationId'>,
  registry: WorkflowOperationRegistry
) {
  const operation = registry.get(step.operationId);
  if (!operation)
    throw new WorkflowError(
      'unknown-operation',
      `${step.label ?? step.operationId}: Operation ${step.operationId} is unknown.`,
      {stepId: step.id}
    );
  return operation;
}

function requireParameter(step: WorkflowStep, registry: WorkflowOperationRegistry, name: string) {
  const operation = requireOperation(step, registry);
  const parameter = operation.parameters.find(candidate => candidate.name === name);
  if (!parameter)
    throw new WorkflowError(
      'unknown-parameter',
      `${step.label ?? operation.title}: Parameter ${name} does not exist.`,
      {stepId: step.id, parameter: name}
    );
  return parameter;
}

function requireInput(document: WorkflowDocument, inputId: string): WorkflowInput {
  const input = document.inputs.find(candidate => candidate.id === inputId);
  if (!input)
    throw new WorkflowError(
      'unknown-workflow-input',
      `Workflow input ${inputId}: The input does not exist.`
    );
  return input;
}

function checkStepCapacity(document: WorkflowDocument, additional: number): void {
  if (document.steps.length + additional > WORKFLOW_LIMITS.maximumSteps)
    throw new WorkflowError(
      'limit-exceeded',
      `${document.title}: Steps must not exceed maximumSteps (${WORKFLOW_LIMITS.maximumSteps}).`
    );
}

function requireFreeId(identifier: string, taken: readonly string[], label: string): void {
  if (!isWorkflowId(identifier))
    throw new WorkflowError(
      'parameter-out-of-range',
      `${label}: Id must match [a-z0-9][a-z0-9-]* and be at most ${WORKFLOW_LIMITS.maximumIdCharacters} characters.`
    );
  if (taken.includes(identifier))
    throw new WorkflowError('duplicate-id', `${label}: Id ${identifier} is already in use.`);
}

function mapSteps(
  document: WorkflowDocument,
  update: (step: WorkflowStep) => WorkflowStep
): WorkflowDocument {
  const steps = document.steps.map(update);
  return steps.every((step, index) => step === document.steps[index])
    ? document
    : {...document, steps};
}

function replaceStep(document: WorkflowDocument, replacement: WorkflowStep): WorkflowDocument {
  const updated = mapSteps(document, step =>
    step.id === replacement.id && !equalValues(step, replacement) ? replacement : step
  );
  return updated === document
    ? document
    : finishEdit(updated, replacement.label ?? replacement.operationId);
}

function normalizePosition(
  position: {x: number; y: number},
  label = 'Canvas position'
): {x: number; y: number} {
  function normalizeCoordinate(coordinate: number): number {
    if (!Number.isFinite(coordinate))
      throw new WorkflowError(
        'parameter-out-of-range',
        `${label}: Canvas coordinates must be finite numbers.`
      );
    return Math.max(
      -WORKFLOW_LIMITS.maximumCanvasCoordinate,
      Math.min(WORKFLOW_LIMITS.maximumCanvasCoordinate, Math.round(coordinate))
    );
  }
  return {x: normalizeCoordinate(position.x), y: normalizeCoordinate(position.y)};
}

export type WorkflowStepPatch = Partial<
  Pick<WorkflowStep, 'label' | 'disabled' | 'display' | 'engine' | 'onFallback' | 'note'>
>;

/**
 * Adds a step. The id derives from the operation id. With `connectFrom`, the first input port
 * that can read that output (or `connectTo`) is connected, and without a position the step is
 * placed to the right of the producing step.
 */
export function addWorkflowStep(
  document: WorkflowDocument,
  registry: WorkflowOperationRegistry,
  options: {
    operationId: string;
    id?: string;
    label?: string;
    position?: {x: number; y: number};
    parameters?: Record<string, CatalogParameterValue>;
    connectFrom?: WorkflowConnection;
    connectTo?: string;
  }
): {document: WorkflowDocument; stepId: string} {
  const operation = requireOperation(
    {
      operationId: options.operationId,
      label: options.label,
      id: options.id ?? options.operationId
    },
    registry
  );
  checkStepCapacity(document, 1);
  const taken = document.steps.map(step => step.id);
  const stepId = options.id ?? makeUniqueWorkflowId(operation.id, taken);
  requireFreeId(stepId, taken, options.label ?? operation.title);
  const parameters: Record<string, WorkflowParameterBinding> = {};
  for (const [name, value] of Object.entries(options.parameters ?? {})) {
    if (!operation.parameters.some(parameter => parameter.name === name))
      throw new WorkflowError(
        'unknown-parameter',
        `${options.label ?? operation.title}: Parameter ${name} does not exist.`,
        {stepId, parameter: name}
      );
    Object.defineProperty(parameters, name, {
      value: {from: 'static', value: copyParameterValue(value)},
      enumerable: true,
      configurable: true,
      writable: true
    });
  }
  const upstream = options.connectFrom
    ? requireStep(document, options.connectFrom.stepId)
    : undefined;
  let position = options.position;
  if (!position) {
    if (upstream) {
      const anchor = normalizePosition(
        {x: upstream.position.x + 280, y: upstream.position.y},
        options.label ?? operation.title
      );
      const rows = [anchor.y];
      // Search downward first, then upward if the canvas boundary cuts off those rows.
      for (
        let distance = 150;
        anchor.y + distance <= WORKFLOW_LIMITS.maximumCanvasCoordinate;
        distance += 150
      )
        rows.push(anchor.y + distance);
      for (
        let distance = 150;
        anchor.y - distance >= -WORKFLOW_LIMITS.maximumCanvasCoordinate;
        distance += 150
      )
        rows.push(anchor.y - distance);
      position = rows
        .map(y => ({x: anchor.x, y}))
        .find(
          candidate =>
            !document.steps.some(
              existing =>
                Math.abs(existing.position.x - candidate.x) < 40 &&
                Math.abs(existing.position.y - candidate.y) < 40
            )
        );
      if (!position)
        throw new WorkflowError(
          'limit-exceeded',
          `${options.label ?? operation.title}: No free row within maximumCanvasCoordinate (${WORKFLOW_LIMITS.maximumCanvasCoordinate}).`,
          {stepId}
        );
    } else {
      const rightmost = document.steps.reduce<WorkflowStep | undefined>(
        (current, candidate) =>
          !current || candidate.position.x > current.position.x ? candidate : current,
        undefined
      );
      position = rightmost
        ? {x: rightmost.position.x + 280, y: rightmost.position.y}
        : {x: 0, y: 0};
    }
  }
  const step: WorkflowStep = {
    id: stepId,
    operationId: operation.id,
    inputs: {},
    parameters,
    position: normalizePosition(position, options.label ?? operation.title),
    ...(options.label?.trim() ? {label: options.label.trim()} : {})
  };
  let result = {...document, steps: [...document.steps, step]};
  if (options.connectFrom && upstream) {
    const upstreamOperation = requireOperation(upstream, registry);
    const output = upstreamOperation.outputs.find(
      port => port.name === options.connectFrom!.output
    );
    if (!output)
      throw new WorkflowError(
        'unknown-output-port',
        `${upstream.label ?? upstreamOperation.title}: Output ${options.connectFrom.output} does not exist.`,
        {stepId: upstream.id, port: options.connectFrom.output}
      );
    const input =
      options.connectTo ??
      operation.inputs.find(
        port =>
          getWorkflowPortCompatibility(output.kind, port) !== 'no' &&
          !Object.hasOwn(step.inputs, port.name)
      )?.name;
    if (input !== undefined)
      result = connectWorkflowSteps(result, registry, {
        from: options.connectFrom,
        to: {stepId, input}
      });
  }
  return {document: finishEdit(result, step.label ?? operation.title), stepId};
}

/** Removes steps and every connection that reads them. */
export function removeWorkflowSteps(
  document: WorkflowDocument,
  stepIds: readonly string[]
): WorkflowDocument {
  const removed = new Set(
    stepIds.filter(stepId => document.steps.some(step => step.id === stepId))
  );
  if (!removed.size) return document;
  const steps = document.steps
    .filter(step => !removed.has(step.id))
    .map(step => {
      const inputs = Object.fromEntries(
        Object.entries(step.inputs).filter(([, connection]) => !removed.has(connection.stepId))
      );
      return Object.keys(inputs).length === Object.keys(step.inputs).length
        ? step
        : {...step, inputs};
    });
  return finishEdit({...document, steps});
}

type ConnectionEdit = {from: WorkflowConnection; to: {stepId: string; input: string}};

function validateConnection(
  document: WorkflowDocument,
  registry: WorkflowOperationRegistry,
  connection: ConnectionEdit
): void {
  const source = requireStep(document, connection.from.stepId);
  const target = requireStep(document, connection.to.stepId);
  const sourceOperation = requireOperation(source, registry);
  const targetOperation = requireOperation(target, registry);
  const label = target.label ?? targetOperation.title;
  const output = sourceOperation.outputs.find(port => port.name === connection.from.output);
  if (!output)
    throw new WorkflowError(
      'unknown-output-port',
      `${source.label ?? sourceOperation.title}: Output ${connection.from.output} does not exist.`,
      {stepId: source.id, port: connection.from.output}
    );
  const input = targetOperation.inputs.find(port => port.name === connection.to.input);
  if (!input)
    throw new WorkflowError(
      'unknown-input-port',
      `${label}: Input ${connection.to.input} does not exist.`,
      {stepId: target.id, port: connection.to.input}
    );
  const visited = new Set<string>();
  const pending = [source.id];
  while (pending.length) {
    const stepId = pending.pop()!;
    if (stepId === target.id)
      throw new WorkflowError('cycle', `${label}: Input ${input.label} cannot create a cycle.`, {
        stepId: target.id,
        port: input.name
      });
    if (visited.has(stepId)) continue;
    visited.add(stepId);
    const step = document.steps.find(candidate => candidate.id === stepId);
    if (step) pending.push(...Object.values(step.inputs).map(upstream => upstream.stepId));
  }
  if (getWorkflowPortCompatibility(output.kind, input) === 'no')
    throw new WorkflowError(
      'incompatible-port',
      `${label}: Input ${input.label} accepts ${input.accepts.join(', ')}; output ${output.label} has kind ${output.kind}.`,
      {stepId: target.id, port: input.name}
    );
}

/** Connects an output to an input port, replacing what the port read. Rejects cycles, unknown ports and incompatible kinds. */
export function connectWorkflowSteps(
  document: WorkflowDocument,
  registry: WorkflowOperationRegistry,
  connection: ConnectionEdit
): WorkflowDocument {
  validateConnection(document, registry, connection);
  const step = requireStep(document, connection.to.stepId);
  return replaceStep(document, {
    ...step,
    inputs: {...step.inputs, [connection.to.input]: {...connection.from}}
  });
}

/** Whether `connectWorkflowSteps` would accept the connection, and why not. */
export function checkWorkflowConnection(
  document: WorkflowDocument,
  registry: WorkflowOperationRegistry,
  connection: ConnectionEdit
): {valid: boolean; reason?: string} {
  try {
    validateConnection(document, registry, connection);
    return {valid: true};
  } catch (error) {
    if (!(error instanceof WorkflowError)) throw error;
    return {valid: false, reason: error.message};
  }
}

export function disconnectWorkflowInput(
  document: WorkflowDocument,
  stepId: string,
  input: string
): WorkflowDocument {
  const step = requireStep(document, stepId);
  if (!Object.hasOwn(step.inputs, input)) return document;
  const inputs = {...step.inputs};
  delete inputs[input];
  return replaceStep(document, {...step, inputs});
}

/** Sets one parameter binding; `undefined` restores the operation default. */
export function setWorkflowParameter(
  document: WorkflowDocument,
  stepId: string,
  parameterName: string,
  binding: WorkflowParameterBinding | undefined
): WorkflowDocument {
  const step = requireStep(document, stepId);
  const parameters =
    binding === undefined
      ? {...step.parameters}
      : {...step.parameters, [parameterName]: copyParameterBinding(binding)};
  if (binding === undefined) delete parameters[parameterName];
  return replaceStep(document, {...step, parameters});
}

export function updateWorkflowStep(
  document: WorkflowDocument,
  stepId: string,
  patch: WorkflowStepPatch
): WorkflowDocument {
  const step = requireStep(document, stepId);
  if (patch.note !== undefined && patch.note.length > WORKFLOW_LIMITS.maximumNoteCharacters)
    throw new WorkflowError(
      'limit-exceeded',
      `${step.label ?? step.operationId}: Note must be at most maximumNoteCharacters (${WORKFLOW_LIMITS.maximumNoteCharacters}) characters.`,
      {stepId}
    );
  const replacement = {...step, ...patch};
  if (Object.hasOwn(patch, 'label')) replacement.label = patch.label?.trim() || undefined;
  if (Object.hasOwn(patch, 'display'))
    replacement.display = patch.display?.length ? [...new Set(patch.display)] : undefined;
  for (const key of Object.keys(replacement) as (keyof WorkflowStep)[]) {
    if (replacement[key] === undefined) delete replacement[key];
  }
  return replaceStep(document, replacement);
}

export function moveWorkflowSteps(
  document: WorkflowDocument,
  positions: Readonly<Record<string, {x: number; y: number}>>
): WorkflowDocument {
  for (const stepId of Object.keys(positions)) requireStep(document, stepId);
  const updated = mapSteps(document, step => {
    if (!Object.hasOwn(positions, step.id)) return step;
    const position = normalizePosition(positions[step.id], step.label ?? step.operationId);
    return equalValues(position, step.position) ? step : {...step, position};
  });
  return updated === document ? document : finishEdit(updated);
}

/** Copies steps with new ids; connections among the copies are kept, connections to other steps are kept as inputs. */
export function duplicateWorkflowSteps(
  document: WorkflowDocument,
  stepIds: readonly string[],
  offset: {x: number; y: number} = {x: 40, y: 40}
): {document: WorkflowDocument; stepIds: string[]} {
  const originals = [...new Set(stepIds)].map(stepId => requireStep(document, stepId));
  checkStepCapacity(document, originals.length);
  const taken = new Set(document.steps.map(step => step.id));
  const identifiers = new Map<string, string>();
  for (const step of originals) {
    const identifier = makeUniqueWorkflowId(step.id, taken);
    identifiers.set(step.id, identifier);
    taken.add(identifier);
  }
  const copies = originals.map(step => ({
    ...step,
    id: identifiers.get(step.id)!,
    ...(step.label === undefined ? {} : {label: `${step.label} copy`}),
    inputs: Object.fromEntries(
      Object.entries(step.inputs).map(([name, connection]) => [
        name,
        {...connection, stepId: identifiers.get(connection.stepId) ?? connection.stepId}
      ])
    ),
    position: normalizePosition(
      {x: step.position.x + offset.x, y: step.position.y + offset.y},
      step.label ?? step.operationId
    )
  }));
  return {
    document: copies.length
      ? finishEdit({...document, steps: [...document.steps, ...copies]})
      : document,
    stepIds: copies.map(step => step.id)
  };
}

/** Turns a step parameter into a workflow input holding its current value, and binds the step to it. */
export function promoteWorkflowParameter(
  document: WorkflowDocument,
  registry: WorkflowOperationRegistry,
  stepId: string,
  parameterName: string,
  options: {label?: string; inputId?: string} = {}
): {document: WorkflowDocument; inputId: string} {
  const step = requireStep(document, stepId);
  const parameter = requireParameter(step, registry, parameterName);
  const binding = Object.hasOwn(step.parameters, parameterName)
    ? step.parameters[parameterName]
    : undefined;
  if (binding?.from === 'input') {
    requireInput(document, binding.inputId);
    return {document, inputId: binding.inputId};
  }
  const kind = parameter.kind;
  if (
    kind !== 'number' &&
    kind !== 'integer' &&
    kind !== 'boolean' &&
    kind !== 'enum' &&
    kind !== 'string' &&
    kind !== 'number-list'
  )
    throw new WorkflowError(
      'parameter-out-of-range',
      `${step.label ?? registry.get(step.operationId)!.title}: ${parameter.label} cannot be promoted; text, dataset, column, column-list and expression parameters cannot be promoted.`,
      {stepId, parameter: parameterName}
    );
  if (document.inputs.length >= WORKFLOW_LIMITS.maximumInputs)
    throw new WorkflowError(
      'limit-exceeded',
      `${step.label ?? step.operationId}: Workflow inputs must not exceed maximumInputs (${WORKFLOW_LIMITS.maximumInputs}).`,
      {stepId, parameter: parameterName}
    );
  const taken = document.inputs.map(input => input.id);
  const inputId = options.inputId ?? makeUniqueWorkflowId(parameterName, taken);
  requireFreeId(inputId, taken, step.label ?? step.operationId);
  const input: WorkflowInput = {
    id: inputId,
    label: options.label ?? parameter.label,
    kind,
    value: copyParameterValue(binding?.from === 'static' ? binding.value : parameter.defaultValue),
    ...(parameter.unit === undefined ? {} : {unit: parameter.unit}),
    ...(parameter.minimum === undefined ? {} : {minimum: parameter.minimum}),
    ...(parameter.maximum === undefined ? {} : {maximum: parameter.maximum}),
    ...(parameter.options === undefined ? {} : {options: [...parameter.options]})
  };
  return {
    document: setWorkflowParameter(
      {...document, inputs: [...document.inputs, input]},
      stepId,
      parameterName,
      {from: 'input', inputId}
    ),
    inputId
  };
}

function acceptWorkflowInput(parameter: WorkflowParameterSpec, input: WorkflowInput): boolean {
  return (
    (input.kind === parameter.kind || (input.kind === 'integer' && parameter.kind === 'number')) &&
    (input.kind !== 'enum' ||
      Boolean(
        input.options?.length && input.options.every(option => parameter.options?.includes(option))
      ))
  );
}

/** Binds a step parameter to an existing workflow input of a compatible kind. */
export function bindWorkflowParameter(
  document: WorkflowDocument,
  registry: WorkflowOperationRegistry,
  stepId: string,
  parameterName: string,
  inputId: string
): WorkflowDocument {
  const step = requireStep(document, stepId);
  const parameter = requireParameter(step, registry, parameterName);
  const input = requireInput(document, inputId);
  if (!acceptWorkflowInput(parameter, input))
    throw new WorkflowError(
      'parameter-out-of-range',
      `${step.label ?? registry.get(step.operationId)!.title}: ${parameter.label} requires kind ${parameter.kind}${parameter.kind === 'enum' ? ` with options ${parameter.options?.join(', ') ?? '(none)'}` : ''}; workflow input ${input.label} has incompatible kind or options (${input.kind}).`,
      {stepId, parameter: parameterName}
    );
  return setWorkflowParameter(document, stepId, parameterName, {from: 'input', inputId});
}

/** Replaces an input binding with the input's current value. */
export function demoteWorkflowParameter(
  document: WorkflowDocument,
  stepId: string,
  parameterName: string
): WorkflowDocument {
  const step = requireStep(document, stepId);
  const binding = Object.hasOwn(step.parameters, parameterName)
    ? step.parameters[parameterName]
    : undefined;
  return binding?.from === 'input'
    ? setWorkflowParameter(document, stepId, parameterName, {
        from: 'static',
        value: requireInput(document, binding.inputId).value
      })
    : document;
}

function validateInput(input: WorkflowInput): void {
  function reject(problem: string): never {
    throw new WorkflowError('parameter-out-of-range', `${input.label}: Value ${problem}.`);
  }
  if (input.minimum !== undefined && !Number.isFinite(input.minimum))
    reject('minimum must be finite');
  if (input.maximum !== undefined && !Number.isFinite(input.maximum))
    reject('maximum must be finite');
  if (input.minimum !== undefined && input.maximum !== undefined && input.minimum > input.maximum)
    reject('minimum must be at most maximum');
  function validateNumber(value: unknown): void {
    if (typeof value !== 'number' || !Number.isFinite(value)) reject('must be a finite number');
    if (input.kind === 'integer' && !Number.isInteger(value))
      reject(`must be a whole number; got ${value}`);
    if (input.minimum !== undefined && value < input.minimum)
      reject(`must be at least ${input.minimum}; got ${value}`);
    if (input.maximum !== undefined && value > input.maximum)
      reject(`must be at most ${input.maximum}; got ${value}`);
  }
  switch (input.kind) {
    case 'number':
    case 'integer':
      validateNumber(input.value);
      break;
    case 'number-list':
      if (!Array.isArray(input.value) || !input.value.length)
        reject('must be a non-empty list of numbers');
      for (const value of input.value) validateNumber(value);
      break;
    case 'boolean':
      if (typeof input.value !== 'boolean') reject('must be a boolean');
      break;
    case 'enum':
      if (typeof input.value !== 'string' || !input.options?.includes(input.value))
        reject(`must be one of ${input.options?.join(', ') ?? '(none)'}`);
      break;
    case 'string':
      if (typeof input.value !== 'string') reject('must be a string');
      break;
  }
}

export function updateWorkflowInput(
  document: WorkflowDocument,
  inputId: string,
  patch: Partial<Omit<WorkflowInput, 'id' | 'kind'>>
): WorkflowDocument {
  const input = requireInput(document, inputId);
  const replacement = {
    ...input,
    ...patch,
    ...(patch.value === undefined ? {} : {value: copyParameterValue(patch.value)}),
    ...(patch.options === undefined ? {} : {options: [...patch.options]})
  };
  for (const key of ['unit', 'minimum', 'maximum', 'options'] as const) {
    if (replacement[key] === undefined) delete replacement[key];
  }
  validateInput(replacement);
  return equalValues(input, replacement)
    ? document
    : finishEdit(
        {
          ...document,
          inputs: document.inputs.map(candidate => (candidate === input ? replacement : candidate))
        },
        replacement.label
      );
}

/** Removes a workflow input; every step bound to it keeps the value as a static binding. */
export function removeWorkflowInput(document: WorkflowDocument, inputId: string): WorkflowDocument {
  const input = requireInput(document, inputId);
  const updated = mapSteps(document, step => {
    const bindings = Object.entries(step.parameters);
    if (!bindings.some(([, binding]) => binding.from === 'input' && binding.inputId === inputId))
      return step;
    return {
      ...step,
      parameters: Object.fromEntries(
        bindings.map(([name, binding]) => [
          name,
          binding.from === 'input' && binding.inputId === inputId
            ? {from: 'static', value: input.value}
            : binding
        ])
      )
    };
  });
  return finishEdit({
    ...updated,
    inputs: document.inputs.filter(candidate => candidate.id !== inputId)
  });
}

export function updateWorkflowSettings(
  document: WorkflowDocument,
  patch: Partial<WorkflowSettings>
): WorkflowDocument {
  const settings = {...document.settings, ...patch};
  return equalValues(settings, document.settings) ? document : finishEdit({...document, settings});
}

export function renameWorkflow(
  document: WorkflowDocument,
  title: string,
  description?: string
): WorkflowDocument {
  const trimmedTitle = title.trim();
  if (!trimmedTitle || trimmedTitle.length > WORKFLOW_LIMITS.maximumTitleCharacters)
    throw new WorkflowError(
      'limit-exceeded',
      `${document.title}: Title must have 1 to maximumTitleCharacters (${WORKFLOW_LIMITS.maximumTitleCharacters}) characters.`
    );
  const replacement = {...document, title: trimmedTitle};
  if (description === undefined) delete replacement.description;
  else replacement.description = description;
  return equalValues(document, replacement) ? document : finishEdit(replacement);
}

/**
 * Layered left-to-right layout: column = longest path from a source, rows ordered to keep
 * connected steps near each other. Deterministic.
 */
export function layoutWorkflow(
  document: WorkflowDocument,
  options: {columnSpacing?: number; rowSpacing?: number; origin?: {x: number; y: number}} = {}
): WorkflowDocument {
  const columns = new Map<string, number>();
  const sorted = [...document.steps].sort((left, right) =>
    left.id < right.id ? -1 : left.id > right.id ? 1 : 0
  );
  let progress = true;
  while (progress) {
    progress = false;
    for (const step of sorted) {
      if (columns.has(step.id)) continue;
      const upstream = Object.values(step.inputs).map(connection => columns.get(connection.stepId));
      if (upstream.some(column => column === undefined)) continue;
      columns.set(step.id, upstream.length ? Math.max(...(upstream as number[])) + 1 : 0);
      progress = true;
    }
  }
  const finalColumn = columns.size ? Math.max(...columns.values()) + 1 : 0;
  for (const step of sorted) if (!columns.has(step.id)) columns.set(step.id, finalColumn);
  const rows = new Map<string, number>();
  const positions: Record<string, {x: number; y: number}> = {};
  for (let column = 0; column <= finalColumn; column++) {
    function getBarycentre(step: WorkflowStep): number {
      const connected = [
        ...new Set(Object.values(step.inputs).map(connection => connection.stepId))
      ].filter(stepId => columns.get(stepId) === column - 1 && rows.has(stepId));
      return connected.length
        ? connected.reduce((sum, stepId) => sum + rows.get(stepId)!, 0) / connected.length
        : Infinity;
    }
    const members = sorted
      .filter(step => columns.get(step.id) === column)
      .sort((left, right) => {
        const leftMean = getBarycentre(left);
        const rightMean = getBarycentre(right);
        return leftMean === rightMean
          ? left.id < right.id
            ? -1
            : left.id > right.id
              ? 1
              : 0
          : leftMean < rightMean
            ? -1
            : 1;
      });
    members.forEach((step, row) => {
      rows.set(step.id, row);
      positions[step.id] = {
        x: (options.origin?.x ?? 0) + column * (options.columnSpacing ?? 280),
        y: (options.origin?.y ?? 0) + row * (options.rowSpacing ?? 150)
      };
    });
  }
  return moveWorkflowSteps(document, positions);
}

// ---------------------------------------------------------------------------------------------
// Composition: display style, clipboard and inlining another workflow
// ---------------------------------------------------------------------------------------------

/** Sets or clears (`undefined`) how one displayed output is drawn. Never changes a result key. */
export function setWorkflowDisplayStyle(
  document: WorkflowDocument,
  stepId: string,
  output: string,
  style: WorkflowDisplayStyle | undefined
): WorkflowDocument {
  const step = requireStep(document, stepId);
  // Validate just the requested presentation change, including undefined optional fields,
  // without rejecting unrelated parameters that the planner is responsible for reporting.
  const validated = parseWorkflowDocument(
    serializeWorkflowDocument({
      schemaVersion: 1,
      id: 'display-style',
      title: 'Display style',
      inputs: [],
      settings: {engine: 'auto', autoRun: false},
      steps: [
        {
          id: step.id,
          operationId: 'display-style',
          label: step.label ?? step.operationId,
          inputs: {},
          parameters: {},
          position: {x: 0, y: 0},
          displayStyle: {[output]: style === undefined ? {} : style}
        }
      ]
    })
  );
  const replacement = {...step};
  let displayStyle = {...step.displayStyle};
  const normalized = validated.steps[0].displayStyle?.[output];
  if (normalized) displayStyle = {...displayStyle, [output]: normalized};
  else delete displayStyle[output];
  if (Object.keys(displayStyle).length) replacement.displayStyle = displayStyle;
  else delete replacement.displayStyle;
  return replaceStep(document, replacement);
}

/** Steps copied out of a document, with the workflow inputs they are bound to. Serialisable. */
export type WorkflowClipboard = {
  schemaVersion: 1;
  steps: WorkflowStep[];
  inputs: WorkflowInput[];
  /** Origin for local paste placement. Legacy clipboards without it are treated as foreign. */
  sourceWorkflowId?: string;
};

function copyWorkflowInput(input: WorkflowInput): WorkflowInput {
  return {
    ...input,
    value: copyParameterValue(input.value),
    ...(input.options === undefined ? {} : {options: [...input.options]})
  };
}

function copyWorkflowStep(step: WorkflowStep): WorkflowStep {
  return {
    ...step,
    inputs: Object.fromEntries(
      Object.entries(step.inputs).map(([name, connection]) => [name, {...connection}])
    ),
    parameters: Object.fromEntries(
      Object.entries(step.parameters).map(([name, binding]) => [
        name,
        copyParameterBinding(binding)
      ])
    ),
    position: {...step.position},
    ...(step.display === undefined ? {} : {display: [...step.display]}),
    ...(step.displayStyle === undefined
      ? {}
      : {
          displayStyle: Object.fromEntries(
            Object.entries(step.displayStyle).map(([output, style]) => [output, {...style}])
          )
        })
  };
}

/**
 * Copies steps for a later paste. Connections among the copied steps are kept; connections that
 * read a step outside the copy are dropped. Display flags are cleared so a paste never publishes
 * by itself.
 */
export function copyWorkflowSteps(
  document: WorkflowDocument,
  stepIds: readonly string[]
): WorkflowClipboard {
  const originals = [...new Set(stepIds)].map(stepId => requireStep(document, stepId));
  const selected = new Set(originals.map(step => step.id));
  const boundInputs = new Set<string>();
  const steps = originals.map(original => {
    const step = copyWorkflowStep(original);
    step.inputs = Object.fromEntries(
      Object.entries(step.inputs).filter(([, connection]) => selected.has(connection.stepId))
    );
    delete step.display;
    for (const binding of Object.values(step.parameters)) {
      if (binding.from === 'input') {
        requireInput(document, binding.inputId);
        boundInputs.add(binding.inputId);
      }
    }
    return step;
  });
  return {
    schemaVersion: 1,
    sourceWorkflowId: document.id,
    steps,
    inputs: document.inputs.filter(input => boundInputs.has(input.id)).map(copyWorkflowInput)
  };
}

/** Strict reader for clipboard text from outside the application. */
export function parseWorkflowClipboard(value: unknown): WorkflowClipboard {
  if (typeof value === 'string') {
    checkContentLimit('Clipboard', 'Text characters', value.length, 'maximumDocumentCharacters');
    try {
      value = JSON.parse(value) as unknown;
    } catch (error) {
      throw new WorkflowError('malformed', 'Clipboard: Text must be valid JSON.', {cause: error});
    }
  }
  if (
    !value ||
    typeof value !== 'object' ||
    Array.isArray(value) ||
    (Object.getPrototypeOf(value) !== Object.prototype && Object.getPrototypeOf(value) !== null)
  ) {
    throw new WorkflowError(
      'malformed',
      'Clipboard: Expected an object with schemaVersion, steps and inputs.'
    );
  }
  const fields: Record<string, unknown> = {};
  for (const key of Reflect.ownKeys(value)) {
    const descriptor = Object.getOwnPropertyDescriptor(value, key)!;
    if (
      typeof key !== 'string' ||
      !['schemaVersion', 'steps', 'inputs', 'sourceWorkflowId'].includes(key) ||
      !descriptor.enumerable ||
      !('value' in descriptor)
    ) {
      throw new WorkflowError(
        'malformed',
        `Clipboard: Field ${String(key)} must be an enumerable own data field named schemaVersion, steps, inputs or sourceWorkflowId.`
      );
    }
    fields[key] = descriptor.value;
  }
  const document = parseWorkflowDocument({
    schemaVersion: fields['schemaVersion'],
    id: Object.hasOwn(fields, 'sourceWorkflowId') ? fields['sourceWorkflowId'] : 'clipboard',
    title: 'Clipboard',
    steps: fields['steps'],
    inputs: fields['inputs'],
    settings: {engine: 'auto', autoRun: false}
  });
  return {
    schemaVersion: 1,
    steps: document.steps,
    inputs: document.inputs,
    ...(Object.hasOwn(fields, 'sourceWorkflowId') ? {sourceWorkflowId: document.id} : {})
  };
}

function matchWorkflowInput(left: WorkflowInput, right: WorkflowInput): boolean {
  return (
    left.id === right.id &&
    left.kind === right.kind &&
    left.unit === right.unit &&
    left.minimum === right.minimum &&
    left.maximum === right.maximum &&
    equalValues(left.options, right.options)
  );
}

function checkCompositionLimits(
  document: WorkflowDocument,
  addedSteps: number,
  addedInputs: number
): void {
  const violations = [
    {name: 'maximumSteps' as const, existing: document.steps.length, added: addedSteps},
    {name: 'maximumInputs' as const, existing: document.inputs.length, added: addedInputs}
  ].filter(limit => limit.existing + limit.added > WORKFLOW_LIMITS[limit.name]);
  if (violations.length)
    throw new WorkflowError(
      'limit-exceeded',
      `${document.title}: ${violations.map(limit => `${limit.name} (${WORKFLOW_LIMITS[limit.name]}) is exceeded by total ${limit.existing + limit.added} (${limit.existing} existing + ${limit.added} added)`).join('; ')}.`
    );
}

function composeWorkflowSteps(
  document: WorkflowDocument,
  sourceSteps: readonly WorkflowStep[],
  sourceInputs: readonly WorkflowInput[],
  position: {x: number; y: number}
): {document: WorkflowDocument; stepIds: string[]} {
  const inputIds = new Map<string, string>();
  const takenInputIds = new Set(document.inputs.map(input => input.id));
  const addedInputs: WorkflowInput[] = [];
  for (const input of sourceInputs) {
    const existing = document.inputs.find(candidate => matchWorkflowInput(candidate, input));
    if (existing) inputIds.set(input.id, existing.id);
    else {
      const inputId = makeUniqueWorkflowId(input.id, takenInputIds);
      takenInputIds.add(inputId);
      inputIds.set(input.id, inputId);
      addedInputs.push({...copyWorkflowInput(input), id: inputId});
    }
  }
  checkCompositionLimits(document, sourceSteps.length, addedInputs.length);
  if (!sourceSteps.length && !addedInputs.length) return {document, stepIds: []};
  const takenStepIds = new Set(document.steps.map(step => step.id));
  const stepIds = new Map<string, string>();
  for (const step of sourceSteps) {
    const stepId = makeUniqueWorkflowId(step.id, takenStepIds);
    takenStepIds.add(stepId);
    stepIds.set(step.id, stepId);
  }
  const anchor = normalizePosition(position);
  const minimumX = sourceSteps.length ? Math.min(...sourceSteps.map(step => step.position.x)) : 0;
  const minimumY = sourceSteps.length ? Math.min(...sourceSteps.map(step => step.position.y)) : 0;
  const maximumX = sourceSteps.length ? Math.max(...sourceSteps.map(step => step.position.x)) : 0;
  const maximumY = sourceSteps.length ? Math.max(...sourceSteps.map(step => step.position.y)) : 0;
  const limit = WORKFLOW_LIMITS.maximumCanvasCoordinate;
  const horizontalOffset = Math.max(
    -limit - minimumX,
    Math.min(limit - maximumX, anchor.x - minimumX)
  );
  const verticalOffset = Math.max(
    -limit - minimumY,
    Math.min(limit - maximumY, anchor.y - minimumY)
  );
  const addedSteps = sourceSteps.map(original => {
    const step = copyWorkflowStep(original);
    step.id = stepIds.get(original.id)!;
    step.inputs = Object.fromEntries(
      Object.entries(original.inputs)
        .filter(([, connection]) => stepIds.has(connection.stepId))
        .map(([name, connection]) => [
          name,
          {stepId: stepIds.get(connection.stepId)!, output: connection.output}
        ])
    );
    step.parameters = Object.fromEntries(
      Object.entries(step.parameters).map(([name, binding]) => {
        if (binding.from === 'static') return [name, binding];
        const inputId = inputIds.get(binding.inputId);
        if (inputId === undefined)
          throw new WorkflowError(
            'unknown-workflow-input',
            `${step.label ?? step.operationId}: Parameter ${name} refers to missing workflow input ${binding.inputId}.`,
            {stepId: step.id, parameter: name}
          );
        return [name, {from: 'input', inputId}];
      })
    );
    step.position = normalizePosition(
      {x: original.position.x + horizontalOffset, y: original.position.y + verticalOffset},
      step.label ?? step.operationId
    );
    return step;
  });
  return {
    document: finishEdit({
      ...document,
      inputs: addedInputs.length ? [...document.inputs, ...addedInputs] : document.inputs,
      steps: addedSteps.length ? [...document.steps, ...addedSteps] : document.steps
    }),
    stepIds: addedSteps.map(step => step.id)
  };
}

/**
 * Pastes copied steps under new ids with the top-left step at `position` (default: 40 px below
 * and right of the originals when the clipboard explicitly names this document as its origin,
 * else right of the right-most step).
 * A bound workflow input is reused when the document has an input with the same id, kind and
 * constraints, otherwise it is added under a free id.
 */
export function pasteWorkflowSteps(
  document: WorkflowDocument,
  _registry: WorkflowOperationRegistry,
  clipboard: WorkflowClipboard,
  position?: {x: number; y: number}
): {document: WorkflowDocument; stepIds: string[]} {
  const originals = clipboard.steps.map(original =>
    document.steps.find(
      step => step.id === original.id && step.operationId === original.operationId
    )
  );
  if (!position) {
    if (
      clipboard.sourceWorkflowId === document.id &&
      originals.length &&
      originals.every((step): step is WorkflowStep => step !== undefined)
    ) {
      position = {
        x: Math.min(...originals.map(step => step.position.x)) + 40,
        y: Math.min(...originals.map(step => step.position.y)) + 40
      };
    } else {
      const rightmost = document.steps.reduce<WorkflowStep | undefined>(
        (current, candidate) =>
          !current || candidate.position.x > current.position.x ? candidate : current,
        undefined
      );
      position = rightmost
        ? {x: rightmost.position.x + 280, y: rightmost.position.y}
        : {x: 0, y: 0};
    }
  }
  return composeWorkflowSteps(document, clipboard.steps, clipboard.inputs, position);
}

/**
 * Inlines every step of another workflow or template into this one: composition by copy. Ids are
 * made unique, the source's workflow inputs come along (reused or added as in a paste), display
 * flags are kept, and the block is placed at `position` (default: below the lowest step).
 */
export function insertWorkflowSteps(
  document: WorkflowDocument,
  _registry: WorkflowOperationRegistry,
  source: WorkflowDocument,
  position?: {x: number; y: number}
): {document: WorkflowDocument; stepIds: string[]} {
  if (!position) {
    const lowest = document.steps.reduce<WorkflowStep | undefined>(
      (current, candidate) =>
        !current || candidate.position.y > current.position.y ? candidate : current,
      undefined
    );
    position = lowest ? {x: lowest.position.x, y: lowest.position.y + 150} : {x: 0, y: 0};
  }
  return composeWorkflowSteps(document, source.steps, source.inputs, position);
}

/**
 * Changes the operation of a step in place: the id, label, position and note stay; input
 * connections are kept for input ports of the same name that can still read their producer;
 * static parameters are kept for parameters of the same name and kind; connections that read an
 * output of this step are kept when the new operation has an output of that name, and removed
 * otherwise; display flags and styles are kept for outputs that still exist. Meant for swapping
 * one source for another (embedded GeoJSON for a dataset) without rewiring a workflow.
 */
export function replaceWorkflowStepOperation(
  document: WorkflowDocument,
  registry: WorkflowOperationRegistry,
  stepId: string,
  operationId: string,
  parameters: Record<string, CatalogParameterValue> = {}
): WorkflowDocument {
  const step = requireStep(document, stepId);
  const operation = requireOperation({...step, operationId}, registry);
  if (step.operationId === operationId && !Object.keys(parameters).length) return document;
  const previousOperation = registry.get(step.operationId);
  const parameterSpecs = new Map(
    operation.parameters.map(parameter => [parameter.name, parameter])
  );
  const bindings: Record<string, WorkflowParameterBinding> = Object.fromEntries(
    Object.entries(step.parameters).filter(([name, binding]) => {
      const parameter = parameterSpecs.get(name);
      if (!parameter) return false;
      if (binding.from === 'static')
        return (
          previousOperation?.parameters.some(
            previous => previous.name === name && previous.kind === parameter.kind
          ) ?? false
        );
      const input = document.inputs.find(candidate => candidate.id === binding.inputId);
      return input !== undefined && acceptWorkflowInput(parameter, input);
    })
  );
  for (const [name, value] of Object.entries(parameters)) {
    if (!parameterSpecs.has(name))
      throw new WorkflowError(
        'unknown-parameter',
        `${step.label ?? operation.title}: Parameter ${name} does not exist.`,
        {stepId, parameter: name}
      );
    Object.defineProperty(bindings, name, {
      value: {from: 'static', value: copyParameterValue(value)},
      enumerable: true,
      configurable: true,
      writable: true
    });
  }
  const inputs = Object.fromEntries(
    Object.entries(step.inputs).filter(([name, connection]) => {
      const input = operation.inputs.find(port => port.name === name);
      const producer = document.steps.find(candidate => candidate.id === connection.stepId);
      const output =
        producer &&
        registry.get(producer.operationId)?.outputs.find(port => port.name === connection.output);
      return (
        input !== undefined &&
        output !== undefined &&
        getWorkflowPortCompatibility(output.kind, input) !== 'no'
      );
    })
  );
  const outputs = new Set(operation.outputs.map(output => output.name));
  const replacement: WorkflowStep = {...step, operationId, inputs, parameters: bindings};
  if (step.display !== undefined) {
    const display = step.display.filter(output => outputs.has(output));
    if (display.length) replacement.display = display;
    else delete replacement.display;
  }
  if (step.displayStyle !== undefined) {
    const displayStyle = Object.fromEntries(
      Object.entries(step.displayStyle).filter(([output]) => outputs.has(output))
    );
    if (Object.keys(displayStyle).length) replacement.displayStyle = displayStyle;
    else delete replacement.displayStyle;
  }
  const updated = mapSteps(document, candidate => {
    if (candidate.id === stepId)
      return equalValues(candidate, replacement) ? candidate : replacement;
    const keptInputs = Object.fromEntries(
      Object.entries(candidate.inputs).filter(
        ([, connection]) => connection.stepId !== stepId || outputs.has(connection.output)
      )
    );
    return Object.keys(keptInputs).length === Object.keys(candidate.inputs).length
      ? candidate
      : {...candidate, inputs: keptInputs};
  });
  return updated === document ? document : finishEdit(updated, step.label ?? operation.title);
}
