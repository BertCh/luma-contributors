// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KeplerJSONValue} from '../recipes/sandbox-state';
import {
  WORKFLOW_LIMITS,
  WORKFLOW_SCHEMA_VERSION,
  WorkflowError,
  type WorkflowCollection,
  type WorkflowDocument,
  type WorkflowDisplayStyle,
  type WorkflowInput,
  type WorkflowParameterBinding,
  type WorkflowStep
} from './workflow-types';

type ObjectFields = Record<string, unknown>;
const SPECIAL_KEYS = new Set(['__proto__', 'constructor', 'prototype']);

function reject(path: string, requirement: string): never {
  throw new WorkflowError('malformed', `${path}: ${requirement}.`);
}

function checkLimit(path: string, count: number, name: string, maximum: number): void {
  if (count > maximum) {
    throw new WorkflowError('limit-exceeded', `${path}: ${name} is ${maximum}; got ${count}.`);
  }
}

/** Inspect descriptors before reading values, so accessors and inherited data never participate. */
function readObject(
  value: unknown,
  path: string,
  fields?: readonly string[],
  undefinedOptionals: readonly string[] = []
): ObjectFields {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    reject(path, 'must be an object');
  }
  const prototype = Object.getPrototypeOf(value) as object | null;
  if (prototype !== null && prototype !== Object.prototype) {
    const inheritedKey = Reflect.ownKeys(prototype)[0];
    reject(
      inheritedKey === undefined ? path : `${path}.${String(inheritedKey)}`,
      'inherited fields are not allowed'
    );
  }
  for (const key in value) {
    if (!Object.hasOwn(value, key)) reject(`${path}.${key}`, 'inherited fields are not allowed');
  }
  const result: ObjectFields = Object.create(null);
  for (const key of Reflect.ownKeys(value)) {
    const fieldPath = `${path}.${String(key)}`;
    if (typeof key !== 'string' || SPECIAL_KEYS.has(key) || (fields && !fields.includes(key))) {
      reject(fieldPath, 'unknown or special field');
    }
    const descriptor = Object.getOwnPropertyDescriptor(value, key)!;
    if (!('value' in descriptor) || !descriptor.enumerable) {
      reject(fieldPath, 'must be an enumerable own data field');
    }
    // Serializers accept TypeScript's representation of absent optional fields.
    if (descriptor.value === undefined && undefinedOptionals.includes(key)) continue;
    result[key] = descriptor.value;
  }
  return result;
}

function readArray(
  value: unknown,
  path: string,
  limit?: {name: string; maximum: number}
): unknown[] {
  if (!Array.isArray(value)) reject(path, 'must be an array');
  const length = value.length;
  // Reject counts before enumerating keys or allocating a copy of an untrusted array.
  if (limit) checkLimit(path, length, limit.name, limit.maximum);
  if (Object.getPrototypeOf(value) !== Array.prototype)
    reject(path, 'inherited fields are not allowed');
  for (const key of Reflect.ownKeys(value)) {
    if (key === 'length') continue;
    if (typeof key !== 'string') reject(`${path}.${String(key)}`, 'unknown or special array field');
    const index = Number(key);
    if (!Number.isInteger(index) || index < 0 || index >= length || String(index) !== key) {
      reject(`${path}.${key}`, 'unknown or special array field');
    }
  }
  const result: unknown[] = [];
  for (let index = 0; index < length; index++) {
    const descriptor = Object.getOwnPropertyDescriptor(value, String(index));
    if (!descriptor || !('value' in descriptor) || !descriptor.enumerable) {
      reject(`${path}[${index}]`, 'must be an own data item');
    }
    result.push(descriptor.value);
  }
  return result;
}

function readString(value: unknown, path: string): string {
  if (typeof value !== 'string') reject(path, 'must be a string');
  return value;
}

function readNumber(value: unknown, path: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) reject(path, 'must be a finite number');
  return Object.is(value, -0) ? 0 : value;
}

function readBoolean(value: unknown, path: string): boolean {
  if (typeof value !== 'boolean') reject(path, 'must be a boolean');
  return value;
}

function readChoice<const Choice extends string>(
  value: unknown,
  path: string,
  choices: readonly Choice[]
): Choice {
  if (typeof value !== 'string' || !choices.includes(value as Choice)) {
    reject(path, `must be one of ${choices.join(', ')}`);
  }
  return value as Choice;
}

function readId(value: unknown, path: string): string {
  const identifier = readString(value, path);
  checkLimit(path, identifier.length, 'maximumIdCharacters', WORKFLOW_LIMITS.maximumIdCharacters);
  if (!isWorkflowId(identifier)) reject(path, 'must match [a-z0-9][a-z0-9-]*');
  return identifier;
}

function readOptionalText(
  fields: ObjectFields,
  name: string,
  path: string,
  maximum: number,
  limit: string
): string | undefined {
  if (!Object.hasOwn(fields, name)) return undefined;
  const text = readString(fields[name], `${path}.${name}`);
  checkLimit(`${path}.${name}`, text.length, limit, maximum);
  return text || undefined;
}

function readParameterValue(value: unknown, path: string): WorkflowInput['value'] {
  if (typeof value === 'string') {
    checkLimit(path, value.length, 'maximumTextCharacters', WORKFLOW_LIMITS.maximumTextCharacters);
    return value;
  }
  if (typeof value === 'boolean') return value;
  if (typeof value === 'number') return readNumber(value, path);
  const numbers = readArray(value, path, {
    name: 'maximum parameter number-list length',
    maximum: 64
  });
  return numbers.map((number, index) => readNumber(number, `${path}[${index}]`));
}

function readBinding(value: unknown, path: string): WorkflowParameterBinding {
  const fields = readObject(value, path, ['from', 'value', 'inputId']);
  const from = readChoice(fields['from'], `${path}.from`, ['static', 'input']);
  if (from === 'static') {
    if (Object.hasOwn(fields, 'inputId'))
      reject(`${path}.inputId`, 'unknown field for a static binding');
    return {from, value: readParameterValue(fields['value'], `${path}.value`)};
  }
  if (Object.hasOwn(fields, 'value')) reject(`${path}.value`, 'unknown field for an input binding');
  return {from, inputId: readString(fields['inputId'], `${path}.inputId`)};
}

function readInput(value: unknown, path: string, omitUndefinedOptionals: boolean): WorkflowInput {
  const fields = readObject(
    value,
    path,
    ['id', 'label', 'kind', 'value', 'unit', 'minimum', 'maximum', 'options'],
    omitUndefinedOptionals ? ['unit', 'minimum', 'maximum', 'options'] : []
  );
  const id = readId(fields['id'], `${path}.id`);
  const label = readString(fields['label'], `${path}.label`);
  const kind = readChoice(fields['kind'], `${path}.kind`, [
    'number',
    'integer',
    'boolean',
    'enum',
    'string',
    'number-list'
  ]);
  let inputValue: WorkflowInput['value'];
  switch (kind) {
    case 'number':
    case 'integer':
      inputValue = readNumber(fields['value'], `${path}.value`);
      if (kind === 'integer' && !Number.isInteger(inputValue))
        reject(`${path}.value`, 'must be an integer');
      break;
    case 'boolean':
      inputValue = readBoolean(fields['value'], `${path}.value`);
      break;
    case 'number-list': {
      const numbers = readArray(fields['value'], `${path}.value`);
      if (numbers.length === 0) reject(`${path}.value`, 'must be a non-empty number-list');
      inputValue = numbers.map((number, index) => readNumber(number, `${path}.value[${index}]`));
      break;
    }
    default:
      inputValue = readString(fields['value'], `${path}.value`);
      checkLimit(
        `${path}.value`,
        inputValue.length,
        'maximumTextCharacters',
        WORKFLOW_LIMITS.maximumTextCharacters
      );
  }
  const input: WorkflowInput = {id, label, kind, value: inputValue};
  if (Object.hasOwn(fields, 'unit')) input.unit = readString(fields['unit'], `${path}.unit`);
  if (Object.hasOwn(fields, 'minimum'))
    input.minimum = readNumber(fields['minimum'], `${path}.minimum`);
  if (Object.hasOwn(fields, 'maximum'))
    input.maximum = readNumber(fields['maximum'], `${path}.maximum`);
  if (input.minimum !== undefined && input.maximum !== undefined && input.minimum > input.maximum) {
    reject(`${path}.minimum`, 'minimum must be less than or equal to maximum');
  }
  if (kind === 'enum') {
    const options = readArray(fields['options'], `${path}.options`).map((option, index) =>
      readString(option, `${path}.options[${index}]`)
    );
    if (options.length === 0) reject(`${path}.options`, 'must be non-empty');
    if (!options.includes(inputValue as string))
      reject(`${path}.value`, 'must be a member of options');
    input.options = options;
  } else if (Object.hasOwn(fields, 'options')) {
    reject(`${path}.options`, 'is only allowed for enum inputs');
  }
  return input;
}

function readDisplayStyles(
  value: unknown,
  path: string,
  omitUndefinedOptionals: boolean
): Record<string, WorkflowDisplayStyle> | undefined {
  const outputs = readObject(value, path);
  const styles: Record<string, WorkflowDisplayStyle> = {};
  for (const output of Object.keys(outputs).sort()) {
    const outputPath = `${path}.${output}`;
    if (output.trim().length === 0) reject(outputPath, 'output name must be non-empty');
    const fields = readObject(
      outputs[output],
      outputPath,
      ['colorColumn', 'colorScale', 'palette'],
      omitUndefinedOptionals ? ['colorColumn', 'colorScale', 'palette'] : []
    );
    const style: WorkflowDisplayStyle = {};
    if (Object.hasOwn(fields, 'colorColumn')) {
      const column = readString(fields['colorColumn'], `${outputPath}.colorColumn`);
      if (column.length === 0)
        reject(`${outputPath}.colorColumn`, 'must contain at least 1 character');
      checkLimit(`${outputPath}.colorColumn`, column.length, 'maximum colorColumn characters', 64);
      style.colorColumn = column;
    }
    if (Object.hasOwn(fields, 'colorScale')) {
      style.colorScale = readChoice(fields['colorScale'], `${outputPath}.colorScale`, [
        'quantile',
        'quantize',
        'ordinal'
      ]);
    }
    if (Object.hasOwn(fields, 'palette')) {
      style.palette = readChoice(fields['palette'], `${outputPath}.palette`, [
        'sequential',
        'diverging',
        'categorical'
      ]);
    }
    if (Object.keys(style).length) styles[output] = style;
  }
  return Object.keys(styles).length ? styles : undefined;
}

function readStepFields(
  value: unknown,
  path: string,
  omitUndefinedOptionals: boolean
): WorkflowStep {
  const fields = readObject(
    value,
    path,
    [
      'id',
      'operationId',
      'label',
      'inputs',
      'parameters',
      'disabled',
      'display',
      'displayStyle',
      'engine',
      'onFallback',
      'position',
      'note'
    ],
    omitUndefinedOptionals
      ? ['label', 'disabled', 'display', 'displayStyle', 'engine', 'onFallback', 'note']
      : []
  );
  const id = readId(fields['id'], `${path}.id`);
  const operationId = readString(fields['operationId'], `${path}.operationId`);
  if (operationId.trim().length === 0) reject(`${path}.operationId`, 'must be non-empty');
  checkLimit(`${path}.operationId`, operationId.length, 'maximum operationId characters', 80);
  const label = Object.hasOwn(fields, 'label')
    ? readString(fields['label'], `${path}.label`) || undefined
    : undefined;
  const step: WorkflowStep = {
    id,
    operationId,
    ...(label ? {label} : {}),
    inputs: {},
    parameters: {},
    position: {x: 0, y: 0}
  };
  const connections = readObject(fields['inputs'], `${path}.inputs`);
  for (const port of Object.keys(connections).sort()) {
    const connectionPath = `${path}.inputs.${port}`;
    const connection = readObject(connections[port], connectionPath, ['stepId', 'output']);
    const stepId = readString(connection['stepId'], `${connectionPath}.stepId`);
    if (stepId === id) reject(`${connectionPath}.stepId`, 'a step must not read itself');
    step.inputs[port] = {
      stepId,
      output: readString(connection['output'], `${connectionPath}.output`)
    };
  }
  const parameters = readObject(fields['parameters'], `${path}.parameters`);
  for (const parameter of Object.keys(parameters).sort()) {
    step.parameters[parameter] = readBinding(
      parameters[parameter],
      `${path}.parameters.${parameter}`
    );
  }
  if (Object.hasOwn(fields, 'disabled')) {
    if (readBoolean(fields['disabled'], `${path}.disabled`)) step.disabled = true;
  }
  if (Object.hasOwn(fields, 'display')) {
    const display = readArray(fields['display'], `${path}.display`).map((output, index) => {
      const name = readString(output, `${path}.display[${index}]`);
      if (name.trim().length === 0) reject(`${path}.display[${index}]`, 'must be non-empty');
      return name;
    });
    if (new Set(display).size !== display.length)
      reject(`${path}.display`, 'output names must be distinct');
    if (display.length) step.display = display;
  }
  if (Object.hasOwn(fields, 'displayStyle')) {
    const displayStyle = readDisplayStyles(
      fields['displayStyle'],
      `${path}.displayStyle`,
      omitUndefinedOptionals
    );
    if (displayStyle) step.displayStyle = displayStyle;
  }
  if (Object.hasOwn(fields, 'engine'))
    step.engine = readChoice(fields['engine'], `${path}.engine`, ['auto', 'cpu']);
  if (Object.hasOwn(fields, 'onFallback'))
    step.onFallback = readChoice(fields['onFallback'], `${path}.onFallback`, ['warn', 'stop']);
  const position = readObject(fields['position'], `${path}.position`, ['x', 'y']);
  for (const coordinate of ['x', 'y'] as const) {
    const number = readNumber(position[coordinate], `${path}.position.${coordinate}`);
    checkLimit(
      `${path}.position.${coordinate}`,
      Math.abs(number),
      'maximumCanvasCoordinate',
      WORKFLOW_LIMITS.maximumCanvasCoordinate
    );
    step.position[coordinate] = number;
  }
  const note = readOptionalText(
    fields,
    'note',
    path,
    WORKFLOW_LIMITS.maximumNoteCharacters,
    'maximumNoteCharacters'
  );
  if (note) step.note = note;
  return {
    id: step.id,
    operationId: step.operationId,
    ...(step.label ? {label: step.label} : {}),
    inputs: step.inputs,
    parameters: step.parameters,
    ...(step.disabled ? {disabled: true} : {}),
    ...(step.display ? {display: step.display} : {}),
    ...(step.displayStyle ? {displayStyle: step.displayStyle} : {}),
    ...(step.engine ? {engine: step.engine} : {}),
    ...(step.onFallback ? {onFallback: step.onFallback} : {}),
    position: step.position,
    ...(step.note ? {note: step.note} : {})
  };
}

function readStep(value: unknown, path: string, omitUndefinedOptionals: boolean): WorkflowStep {
  try {
    return readStepFields(value, path, omitUndefinedOptionals);
  } catch (error) {
    if (!(error instanceof WorkflowError)) throw error;
    const labelDescriptor =
      value && typeof value === 'object'
        ? Object.getOwnPropertyDescriptor(value, 'label')
        : undefined;
    const operationDescriptor =
      value && typeof value === 'object'
        ? Object.getOwnPropertyDescriptor(value, 'operationId')
        : undefined;
    const labelValue: unknown = labelDescriptor?.value;
    const label =
      typeof labelValue === 'string' && labelValue.length > 0
        ? labelValue
        : operationDescriptor?.value;
    throw new WorkflowError(
      error.code,
      `${typeof label === 'string' ? `${label}: ` : ''}${error.message}`,
      {cause: error}
    );
  }
}

function readDistinctItems<Item extends {id: string}>(
  value: unknown,
  path: string,
  maximum: number,
  limit: string,
  readItem: (item: unknown, path: string) => Item,
  itemLabel?: (item: Item) => string
): Item[] {
  const items = readArray(value, path, {name: limit, maximum});
  const identifiers = new Set<string>();
  return items.map((item, index) => {
    const itemPath = `${path}[${index}]`;
    const parsed = readItem(item, itemPath);
    if (identifiers.has(parsed.id))
      throw new WorkflowError(
        'duplicate-id',
        `${itemLabel ? `${itemLabel(parsed)}: ` : ''}${itemPath}.id: duplicate id ${parsed.id}.`
      );
    identifiers.add(parsed.id);
    return parsed;
  });
}

function readDocument(
  value: unknown,
  path: string,
  omitUndefinedOptionals = false
): WorkflowDocument {
  const fields = readObject(
    value,
    path,
    ['schemaVersion', 'id', 'title', 'description', 'inputs', 'steps', 'settings'],
    omitUndefinedOptionals ? ['description'] : []
  );
  if (fields['schemaVersion'] !== WORKFLOW_SCHEMA_VERSION)
    reject(`${path}.schemaVersion`, 'must be 1');
  const id = readId(fields['id'], `${path}.id`);
  const title = readString(fields['title'], `${path}.title`).trim();
  if (!title) reject(`${path}.title`, 'must contain at least 1 character after trimming');
  checkLimit(
    `${path}.title`,
    title.length,
    'maximumTitleCharacters',
    WORKFLOW_LIMITS.maximumTitleCharacters
  );
  const description = readOptionalText(
    fields,
    'description',
    path,
    WORKFLOW_LIMITS.maximumNoteCharacters,
    'maximumNoteCharacters'
  );
  const inputs = readDistinctItems(
    fields['inputs'],
    `${path}.inputs`,
    WORKFLOW_LIMITS.maximumInputs,
    'maximumInputs',
    (input, inputPath) => readInput(input, inputPath, omitUndefinedOptionals)
  );
  const steps = readDistinctItems(
    fields['steps'],
    `${path}.steps`,
    WORKFLOW_LIMITS.maximumSteps,
    'maximumSteps',
    (step, stepPath) => readStep(step, stepPath, omitUndefinedOptionals),
    step => step.label || step.operationId
  );
  const settings = readObject(fields['settings'], `${path}.settings`, ['engine', 'autoRun']);
  const document: WorkflowDocument = {
    schemaVersion: WORKFLOW_SCHEMA_VERSION,
    id,
    title,
    ...(description ? {description} : {}),
    inputs,
    steps,
    settings: {
      engine: readChoice(settings['engine'], `${path}.settings.engine`, ['auto', 'cpu']),
      autoRun: readBoolean(settings['autoRun'], `${path}.settings.autoRun`)
    }
  };
  checkLimit(
    path,
    JSON.stringify(document).length,
    'maximumDocumentCharacters',
    WORKFLOW_LIMITS.maximumDocumentCharacters
  );
  return document;
}

/** True for `[a-z0-9][a-z0-9-]*` within `WORKFLOW_LIMITS.maximumIdCharacters`. */
export function isWorkflowId(value: string): boolean {
  return (
    typeof value === 'string' &&
    value.length <= WORKFLOW_LIMITS.maximumIdCharacters &&
    /^[a-z0-9]/.test(value) &&
    !/[^a-z0-9-]/.test(value)
  );
}

/** Lower-cases and hyphenates `base`, then appends `-2`, `-3`, ... until the id is free. */
export function makeUniqueWorkflowId(base: string, taken: Iterable<string>): string {
  const stem =
    base
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-+|-+$/g, '') || 'workflow';
  const identifiers = new Set(taken);
  const first = stem.slice(0, WORKFLOW_LIMITS.maximumIdCharacters).replace(/-+$/g, '');
  if (!identifiers.has(first)) return first;
  for (let number = 2; ; number++) {
    const suffix = `-${number}`;
    const prefix = stem
      .slice(0, WORKFLOW_LIMITS.maximumIdCharacters - suffix.length)
      .replace(/-+$/g, '');
    const candidate = `${prefix}${suffix}`;
    if (!identifiers.has(candidate)) return candidate;
  }
}

/** An empty document with default settings. */
export function createWorkflowDocument(
  options: {id?: string; title?: string; description?: string} = {}
): WorkflowDocument {
  const title = options.title ?? 'Untitled workflow';
  return parseWorkflowDocument({
    schemaVersion: WORKFLOW_SCHEMA_VERSION,
    id: options.id ?? makeUniqueWorkflowId(title, []),
    title,
    ...(options.description === undefined ? {} : {description: options.description}),
    inputs: [],
    steps: [],
    settings: {engine: 'auto', autoRun: false}
  });
}

/**
 * Strict reader: unknown fields, wrong types, duplicate ids, ids that are not workflow ids and
 * values over `WORKFLOW_LIMITS` are rejected with a `WorkflowError` (`malformed`,
 * `duplicate-id`, `limit-exceeded`). It checks structure only: unknown operation ids, dangling
 * connections and cycles are left to `planWorkflow`, so a saved map survives catalog changes.
 */
export function parseWorkflowDocument(value: unknown): WorkflowDocument {
  return readDocument(value, 'document');
}

/** Canonical JSON: fixed field order, absent optional fields omitted. Round-trips through the parser. */
export function serializeWorkflowDocument(document: WorkflowDocument): KeplerJSONValue {
  return readDocument(document, 'document', true) as unknown as KeplerJSONValue;
}

function readCollection(value: unknown, omitUndefinedOptionals = false): WorkflowCollection {
  const fields = readObject(
    value,
    'collection',
    ['schemaVersion', 'workflows', 'activeWorkflowId', 'templates'],
    omitUndefinedOptionals ? ['activeWorkflowId', 'templates'] : []
  );
  if (fields['schemaVersion'] !== WORKFLOW_SCHEMA_VERSION)
    reject('collection.schemaVersion', 'must be 1');
  const workflows = readDistinctItems(
    fields['workflows'],
    'collection.workflows',
    WORKFLOW_LIMITS.maximumWorkflows,
    'maximumWorkflows',
    (document, documentPath) => readDocument(document, documentPath, omitUndefinedOptionals)
  );
  const collection: WorkflowCollection = {schemaVersion: WORKFLOW_SCHEMA_VERSION, workflows};
  if (Object.hasOwn(fields, 'activeWorkflowId')) {
    const activeWorkflowId = readId(fields['activeWorkflowId'], 'collection.activeWorkflowId');
    if (!workflows.some(workflow => workflow.id === activeWorkflowId))
      reject('collection.activeWorkflowId', 'must name a workflow in workflows');
    collection.activeWorkflowId = activeWorkflowId;
  }
  if (Object.hasOwn(fields, 'templates')) {
    const templates = readDistinctItems(
      fields['templates'],
      'collection.templates',
      WORKFLOW_LIMITS.maximumUserTemplates,
      'maximumUserTemplates',
      (document, documentPath) => readDocument(document, documentPath, omitUndefinedOptionals)
    );
    if (templates.length) collection.templates = templates;
  }
  return collection;
}

/** Strict reader of the saved extension member; any malformed workflow rejects the whole member. */
export function parseWorkflowCollection(value: unknown): WorkflowCollection {
  return readCollection(value);
}

/** Undefined when there is nothing to save. */
export function serializeWorkflowCollection(
  collection: WorkflowCollection
): KeplerJSONValue | undefined {
  const canonical = readCollection(collection, true);
  return canonical.workflows.length || canonical.templates?.length
    ? (canonical as unknown as KeplerJSONValue)
    : undefined;
}
