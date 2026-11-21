// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getGPUAnalysisCatalogGeometryForDataKind,
  getGPUAnalysisParameterComparisons,
  GPU_ANALYSIS_CAUTIONS,
  GPU_ANALYSIS_SUMMARY_TEMPLATES,
  GPU_ANALYSIS_COLUMN_TYPES,
  GPU_ANALYSIS_DEFAULT_RULES,
  GPU_ANALYSIS_MEASUREMENT_LEVELS,
  GPU_ANALYSIS_QUESTIONS,
  GPU_ANALYSIS_ROW_SEMANTICS,
  GPU_SPATIAL_ANALYSIS_DATA_KINDS
} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';
import type {CatalogParameterValue} from '../analysis/catalog-binding-types';
import {getKeplerOperationDescriptor} from '../descriptors/operation-descriptors';
import {parseWorkflowDocument} from './workflow-document';
import {getWorkflowOperationRegistry} from './workflow-operations';
import {normalizeWorkflowParameters} from './workflow-parameters';
import {planWorkflow} from './workflow-plan';
import type {
  WorkflowAnalysisRoleSource,
  WorkflowAnalysisTemplate
} from './workflow-analysis-template';
import {
  WorkflowError,
  WORKFLOW_LIMITS,
  type WorkflowOperationRegistry,
  type WorkflowPortKind
} from './workflow-types';

/** Independent envelope budget; the nested document retains its own parser limits. */
export const WORKFLOW_ANALYSIS_TEMPLATE_LIMITS = {
  maximumCharacters: 2097152,
  maximumDepth: 64,
  maximumMembers: 100000,
  maximumArrayLength: WORKFLOW_LIMITS.maximumParameterNumbers,
  maximumSlots: 64,
  maximumFields: 64,
  maximumExposed: 64,
  maximumTextCharacters: 16384
} as const;

type TemplateRecord = Record<string, unknown>;

function reject(path: string, requirement: string): never {
  throw new WorkflowError('malformed', `workflow-analysis-template ${path}: ${requirement}.`);
}

function checkLimit(
  path: string,
  count: number,
  name: keyof typeof WORKFLOW_ANALYSIS_TEMPLATE_LIMITS
): void {
  const maximum = WORKFLOW_ANALYSIS_TEMPLATE_LIMITS[name];
  if (count > maximum)
    throw new WorkflowError(
      'limit-exceeded',
      `workflow-analysis-template ${path}: ${name} is ${maximum}; got ${count}.`
    );
}

/** Copy only own data descriptors; never invoke accessors, toJSON or inherited properties. */
function copyTemplateData(value: unknown, omitUndefinedOptionals = false): unknown {
  let members = 0;
  let characters = 0;
  const ancestors = new Set<object>();
  const copy = (item: unknown, path: string, depth: number): unknown => {
    checkLimit(path, ++members, 'maximumMembers');
    checkLimit(path, depth, 'maximumDepth');
    if (typeof item === 'string') {
      characters += item.length;
      checkLimit(path, characters, 'maximumCharacters');
      return item;
    }
    if (item === null || typeof item === 'boolean') return item;
    if (typeof item === 'number' && Number.isFinite(item)) return Object.is(item, -0) ? 0 : item;
    if (!item || typeof item !== 'object') reject(path, 'must contain finite JSON data');
    if (ancestors.has(item)) reject(path, 'cyclic object data is not allowed');
    ancestors.add(item);
    const array = Array.isArray(item);
    const prototype = Object.getPrototypeOf(item);
    if (
      prototype !== (array ? Array.prototype : Object.prototype) &&
      !(prototype === null && !array)
    )
      reject(path, 'inherited fields are not allowed');
    const result: unknown[] | TemplateRecord = array ? [] : {};
    if (array) checkLimit(path, item.length, 'maximumArrayLength');
    for (const key of Reflect.ownKeys(item)) {
      if (array && key === 'length') continue;
      if (typeof key !== 'string' || ['__proto__', 'prototype', 'constructor'].includes(key))
        reject(path, 'special fields are not allowed');
      if (array && (!/^(0|[1-9][0-9]*)$/.test(key) || Number(key) >= item.length))
        reject(`${path}.${key}`, 'unknown array field');
      const descriptor = Object.getOwnPropertyDescriptor(item, key)!;
      if (!('value' in descriptor) || !descriptor.enumerable)
        reject(`${path}.${key}`, 'must be an enumerable own data field');
      if (!array && omitUndefinedOptionals && descriptor.value === undefined) continue;
      characters += key.length;
      checkLimit(path, characters, 'maximumCharacters');
      (result as TemplateRecord)[key] = copy(descriptor.value, `${path}.${key}`, depth + 1);
    }
    if (array && Object.keys(result).length !== item.length)
      reject(path, 'sparse arrays are not allowed');
    ancestors.delete(item);
    return result;
  };
  return copy(value, 'template', 0);
}

function record(value: unknown, path: string, keys?: readonly string[]): TemplateRecord {
  if (!value || typeof value !== 'object' || Array.isArray(value))
    reject(path, 'must be an object');
  const result = value as TemplateRecord;
  for (const key of Object.keys(result)) {
    if (keys && !keys.includes(key)) reject(`${path}.${key}`, 'unknown field');
  }
  return result;
}

function text(value: unknown, path: string): string {
  if (typeof value !== 'string' || !value.trim()) reject(path, 'must be non-empty text');
  checkLimit(path, value.length, 'maximumTextCharacters');
  return value;
}

function choice<const Value extends string>(
  value: unknown,
  path: string,
  choices: readonly Value[]
): Value {
  const result = text(value, path);
  if (!choices.includes(result as Value)) reject(path, `must be one of ${choices.join(', ')}`);
  return result as Value;
}

function list(
  value: unknown,
  path: string,
  maximum: number = WORKFLOW_ANALYSIS_TEMPLATE_LIMITS.maximumArrayLength
): unknown[] {
  if (!Array.isArray(value) || value.length > maximum)
    reject(path, `must be an array of at most ${maximum} items`);
  return value;
}

function finite(value: unknown, path: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) reject(path, 'must be a finite number');
  return value;
}

function parameterValue(value: unknown, path: string): CatalogParameterValue {
  if (typeof value === 'string' || typeof value === 'boolean') return value;
  if (typeof value === 'number') return finite(value, path);
  return list(value, path, WORKFLOW_LIMITS.maximumParameterNumbers).map((item, index) =>
    finite(item, `${path}[${index}]`)
  );
}

function distinct(values: readonly string[], path: string): void {
  if (new Set(values).size !== values.length) reject(path, 'duplicate entries are not allowed');
}

function readReference(value: unknown, path: string): WorkflowAnalysisRoleSource {
  const fields = record(value, path, ['stepId', 'parameter']);
  return {
    stepId: text(fields['stepId'], `${path}.stepId`),
    parameter: text(fields['parameter'], `${path}.parameter`)
  };
}

function readCondition(value: unknown, path: string): void {
  const condition = record(value, path);
  if ('anyOf' in condition) {
    record(condition, path, ['anyOf']);
    const branches = list(condition['anyOf'], `${path}.anyOf`);
    if (!branches.length) reject(`${path}.anyOf`, 'must contain a condition');
    branches.forEach((branch, index) => readCondition(branch, `${path}.anyOf[${index}]`));
    return;
  }
  const parameterKey = 'parameter' in condition ? 'parameter' : 'whenParameter';
  record(condition, path, [parameterKey, 'equals']);
  text(condition[parameterKey], `${path}.${parameterKey}`);
  const equals = list(condition['equals'], `${path}.equals`);
  if (!equals.length) reject(`${path}.equals`, 'must contain a value');
  for (const item of equals) {
    if (!['string', 'number', 'boolean'].includes(typeof item))
      reject(`${path}.equals`, 'must contain scalar values');
  }
}

function readField(value: unknown, path: string): void {
  const field = record(value, path, [
    'role',
    'label',
    'accepts',
    'measurement',
    'required',
    'multiple',
    'sources'
  ]);
  text(field['role'], `${path}.role`);
  text(field['label'], `${path}.label`);
  const accepts = list(field['accepts'], `${path}.accepts`).map(item =>
    choice(item, `${path}.accepts`, GPU_ANALYSIS_COLUMN_TYPES)
  );
  if (!accepts.length) reject(`${path}.accepts`, 'must contain a column type');
  distinct(accepts, `${path}.accepts`);
  if (field['measurement'] !== undefined) {
    const measurements = list(field['measurement'], `${path}.measurement`).map(item =>
      choice(item, `${path}.measurement`, GPU_ANALYSIS_MEASUREMENT_LEVELS)
    );
    if (!measurements.length) reject(`${path}.measurement`, 'must contain a measurement level');
    distinct(measurements, `${path}.measurement`);
  }
  if (typeof field['required'] !== 'boolean') {
    readCondition(field['required'], `${path}.required`);
  }
  if (field['multiple'] !== undefined) {
    const bounds = record(field['multiple'], `${path}.multiple`, ['minimum', 'maximum']);
    const minimum = finite(bounds['minimum'], `${path}.multiple.minimum`);
    const maximum = finite(bounds['maximum'], `${path}.multiple.maximum`);
    if (
      !Number.isInteger(minimum) ||
      !Number.isInteger(maximum) ||
      minimum < 0 ||
      maximum < minimum ||
      maximum > 64
    )
      reject(`${path}.multiple`, 'must have integer bounds 0 <= minimum <= maximum <= 64');
  }
  const sources = list(field['sources'], `${path}.sources`, 64).map((source, index) =>
    readReference(source, `${path}.sources[${index}]`)
  );
  distinct(
    sources.map(source => `${source.stepId}.${source.parameter}`),
    `${path}.sources`
  );
}

function readPolicy(value: unknown, path: string): void {
  const policy = record(value, path);
  switch (policy['from']) {
    case 'constant':
      record(policy, path, ['from']);
      return;
    case 'time-encoding':
      record(policy, path, ['from', 'input', 'role']);
      text(policy['input'], `${path}.input`);
      text(policy['role'], `${path}.role`);
      return;
    case 'preset':
      record(policy, path, ['from', 'value', 'reason', 'input', 'whenGeometry']);
      parameterValue(policy['value'], `${path}.value`);
      text(policy['reason'], `${path}.reason`);
      if (policy['input'] !== undefined) text(policy['input'], `${path}.input`);
      if (policy['whenGeometry'] !== undefined) {
        choice(policy['whenGeometry'], `${path}.whenGeometry`, ['point', 'line', 'polygon']);
        text(policy['input'], `${path}.input`);
      }
      return;
    case 'data': {
      record(policy, path, [
        'from',
        'input',
        'rule',
        'factor',
        'role',
        'factorByParameter',
        'fallback'
      ]);
      text(policy['input'], `${path}.input`);
      choice(policy['rule'], `${path}.rule`, GPU_ANALYSIS_DEFAULT_RULES);
      if (policy['role'] !== undefined) text(policy['role'], `${path}.role`);
      if (String(policy['rule']).startsWith('column-')) text(policy['role'], `${path}.role`);
      if (policy['factor'] !== undefined) finite(policy['factor'], `${path}.factor`);
      if (policy['factorByParameter'] !== undefined) {
        const factor = record(policy['factorByParameter'], `${path}.factorByParameter`, [
          'parameter',
          'factors'
        ]);
        text(factor['parameter'], `${path}.factorByParameter.parameter`);
        for (const number of Object.values(
          record(factor['factors'], `${path}.factorByParameter.factors`)
        ))
          finite(number, `${path}.factorByParameter.factors`);
      }
      if (policy['fallback'] !== undefined) {
        const fallback = record(policy['fallback'], `${path}.fallback`, ['rule', 'factor']);
        choice(fallback['rule'], `${path}.fallback.rule`, GPU_ANALYSIS_DEFAULT_RULES);
        if (String(fallback['rule']).startsWith('column-')) text(policy['role'], `${path}.role`);
        if (fallback['factor'] !== undefined) finite(fallback['factor'], `${path}.fallback.factor`);
      }
      return;
    }
    case 'field-measurement': {
      record(policy, path, ['from', 'input', 'role', 'values']);
      text(policy['input'], `${path}.input`);
      text(policy['role'], `${path}.role`);
      const values = record(policy['values'], `${path}.values`, GPU_ANALYSIS_MEASUREMENT_LEVELS);
      if (!Object.keys(values).length) reject(`${path}.values`, 'must contain a measurement value');
      for (const [name, item] of Object.entries(values))
        parameterValue(item, `${path}.values.${name}`);
      return;
    }
    default:
      reject(`${path}.from`, 'unknown default policy');
  }
}

/** Structural parsing deliberately does not require installed operations or an executable graph. */
export function parseWorkflowAnalysisTemplate(value: unknown): WorkflowAnalysisTemplate {
  let decoded = value;
  if (typeof value === 'string') {
    checkLimit('template', value.length, 'maximumCharacters');
    try {
      decoded = JSON.parse(value);
    } catch {
      reject('template', 'must be valid JSON');
    }
  }
  const template = record(copyTemplateData(decoded), 'template', [
    'schemaVersion',
    'id',
    'title',
    'question',
    'method',
    'curatedAnalysisId',
    'document',
    'slots',
    'fixedSources',
    'exposed',
    'sample',
    'reads'
  ]);
  if (template['schemaVersion'] !== 1) reject('template.schemaVersion', 'must be 1');
  for (const key of ['id', 'title', 'method']) text(template[key], `template.${key}`);
  if (template['curatedAnalysisId'] !== undefined)
    text(template['curatedAnalysisId'], 'template.curatedAnalysisId');
  choice(template['question'], 'template.question', GPU_ANALYSIS_QUESTIONS);
  template['document'] = parseWorkflowDocument(template['document']);
  const slots = list(
    template['slots'],
    'template.slots',
    WORKFLOW_ANALYSIS_TEMPLATE_LIMITS.maximumSlots
  ).map((item, index) => {
    const path = `template.slots[${index}]`;
    const slot = record(item, path, ['name', 'label', 'kind', 'rowSemantics', 'source', 'fields']);
    text(slot['name'], `${path}.name`);
    text(slot['label'], `${path}.label`);
    choice(slot['kind'], `${path}.kind`, GPU_SPATIAL_ANALYSIS_DATA_KINDS);
    if (slot['rowSemantics'] !== undefined) {
      const semantics = list(slot['rowSemantics'], `${path}.rowSemantics`).map(value =>
        choice(value, `${path}.rowSemantics`, GPU_ANALYSIS_ROW_SEMANTICS)
      );
      if (!semantics.length) reject(`${path}.rowSemantics`, 'must contain row semantics');
      distinct(semantics, `${path}.rowSemantics`);
    }
    const source = record(slot['source'], `${path}.source`, ['stepId', 'output']);
    text(source['stepId'], `${path}.source.stepId`);
    text(source['output'], `${path}.source.output`);
    const fields = list(
      slot['fields'],
      `${path}.fields`,
      WORKFLOW_ANALYSIS_TEMPLATE_LIMITS.maximumFields
    );
    fields.forEach((field, index) => readField(field, `${path}.fields[${index}]`));
    distinct(
      fields.map(field => String((field as TemplateRecord)['role'])),
      `${path}.fields`
    );
    return slot;
  });
  distinct(
    slots.map(slot => String(slot['name'])),
    'template.slots'
  );
  if (template['fixedSources'] !== undefined) {
    const fixed = list(template['fixedSources'], 'template.fixedSources', 64).map(value =>
      text(value, 'template.fixedSources')
    );
    distinct(fixed, 'template.fixedSources');
  }
  const exposed = list(
    template['exposed'],
    'template.exposed',
    WORKFLOW_ANALYSIS_TEMPLATE_LIMITS.maximumExposed
  ).map((item, index) => {
    const path = `template.exposed[${index}]`;
    const parameter = record(item, path, ['inputId', 'target', 'defaultPolicy']);
    text(parameter['inputId'], `${path}.inputId`);
    const target = readReference(parameter['target'], `${path}.target`);
    readPolicy(parameter['defaultPolicy'], `${path}.defaultPolicy`);
    return `${target.stepId}.${target.parameter}`;
  });
  distinct(exposed, 'template.exposed');
  if (template['sample'] !== undefined) {
    const sample = record(template['sample'], 'template.sample', [
      'id',
      'license',
      'attribution',
      'reference'
    ]);
    for (const key of ['id', 'license', 'attribution', 'reference'])
      text(sample[key], `template.sample.${key}`);
  }
  const reads = record(template['reads'], 'template.reads', ['summary', 'cautions', 'outputs']);
  text(reads['summary'], 'template.reads.summary');
  const cautions = list(reads['cautions'], 'template.reads.cautions').map(value =>
    text(value, 'template.reads.cautions')
  );
  distinct(cautions, 'template.reads.cautions');
  if (reads['outputs'] !== undefined) {
    const outputs = list(reads['outputs'], 'template.reads.outputs').map((item, index) => {
      const path = `template.reads.outputs[${index}]`;
      const output = record(item, path, ['source', 'interpretation']);
      const source = record(output['source'], `${path}.source`, ['stepId', 'output']);
      text(source['stepId'], `${path}.source.stepId`);
      text(source['output'], `${path}.source.output`);
      const interpretation = record(output['interpretation'], `${path}.interpretation`, [
        'summary',
        'cautions'
      ]);
      choice(
        interpretation['summary'],
        `${path}.interpretation.summary`,
        GPU_ANALYSIS_SUMMARY_TEMPLATES
      );
      const cautions = list(interpretation['cautions'], `${path}.interpretation.cautions`).map(
        caution => choice(caution, `${path}.interpretation.cautions`, GPU_ANALYSIS_CAUTIONS)
      );
      distinct(cautions, `${path}.interpretation.cautions`);
      return `${source['stepId']}.${source['output']}`;
    });
    distinct(outputs, 'template.reads.outputs');
  }
  // The object and JSON entry points share the same wire budget. Count escapes, punctuation
  // and primitive values after safe copying and document canonicalisation, before accepting it.
  checkLimit('template', JSON.stringify(template).length, 'maximumCharacters');
  return template as WorkflowAnalysisTemplate;
}

/** Defensive JSON data suitable for the existing workspace persistence machinery. */
export function serializeWorkflowAnalysisTemplate(
  template: WorkflowAnalysisTemplate
): WorkflowAnalysisTemplate {
  return parseWorkflowAnalysisTemplate(copyTemplateData(template, true));
}

export type WorkflowAnalysisTemplateIssue = {code: string; message: string; path?: string};
export type WorkflowAnalysisTemplateValidation = {
  executable: boolean;
  issues: readonly WorkflowAnalysisTemplateIssue[];
};

/**
 * Validate reuse references plus the existing planner's cycles, ports and parameter checks.
 * `executable` means the template can be mapped with this registry. Dataset availability and
 * statistical applicability still require the host's mapping/profile checks before a run.
 */
export function validateWorkflowAnalysisTemplate(
  value: unknown,
  options: {registry?: WorkflowOperationRegistry} = {}
): WorkflowAnalysisTemplateValidation {
  let template: WorkflowAnalysisTemplate;
  try {
    template = parseWorkflowAnalysisTemplate(value);
  } catch (error) {
    return {executable: false, issues: [{code: 'malformed', message: String(error)}]};
  }
  const registry = options.registry ?? getWorkflowOperationRegistry();
  const observedKinds: Record<string, Record<string, WorkflowPortKind>> = {};
  for (const slot of template.slots) {
    const geometry = getGPUAnalysisCatalogGeometryForDataKind(slot.kind);
    const kind = slot.kind === 'surface' ? 'raster' : geometry === 'none' ? 'table' : geometry;
    if (kind)
      observedKinds[slot.source.stepId] = {
        ...observedKinds[slot.source.stepId],
        [slot.source.output]: kind
      };
  }
  const plan = planWorkflow(template.document, {registry, observedKinds});
  const issues: WorkflowAnalysisTemplateIssue[] = [
    ...plan.issues,
    ...Object.values(plan.steps).flatMap(step => step.issues)
  ]
    .filter(issue => issue.severity === 'error')
    .map(({code, message}) => ({code, message}));
  const report = (code: string, path: string, message: string) =>
    issues.push({code, path, message: `workflow-analysis-template ${path}: ${message}.`});
  const steps = new Map(template.document.steps.map(step => [step.id, step]));
  const inputs = new Map(template.document.inputs.map(input => [input.id, input]));
  const sourceOperations = ['source-dataset', 'source-geojson', 'source-drawing'];
  const coveredSources = new Set<string>();
  const roleOwners = new Map<string, {path: string; role: string}>();
  const hasAncestor = (target: string, source: string, visited = new Set<string>()): boolean => {
    if (target === source) return true;
    if (visited.has(target)) return false;
    visited.add(target);
    return Object.values(steps.get(target)?.inputs ?? {}).some(connection =>
      hasAncestor(connection.stepId, source, visited)
    );
  };
  const checkReference = (reference: WorkflowAnalysisRoleSource, path: string) => {
    const step = steps.get(reference.stepId);
    if (!step) report('unknown-step', path, `step ${reference.stepId} does not exist`);
    const operation = step && registry.get(step.operationId);
    const parameter = operation?.parameters.find(
      parameter => parameter.name === reference.parameter
    );
    if (operation && !parameter)
      report(
        'unknown-parameter',
        path,
        `parameter ${reference.parameter} does not exist on ${step?.operationId}`
      );
    return {step, operation, parameter};
  };
  for (const slot of template.slots) {
    const path = `slots.${slot.name}`;
    const source = steps.get(slot.source.stepId);
    if (!source || !sourceOperations.includes(source.operationId))
      report('invalid-slot-source', path, 'source must refer to an existing workflow source step');
    if (coveredSources.has(slot.source.stepId))
      report('duplicate-slot-source', path, 'source is already assigned to another slot');
    coveredSources.add(slot.source.stepId);
    const operation = source && registry.get(source.operationId);
    if (operation && !operation.outputs.some(output => output.name === slot.source.output))
      report('unknown-output-port', path, `source output ${slot.source.output} does not exist`);
    if (getGPUAnalysisCatalogGeometryForDataKind(slot.kind) === 'none') {
      // A generic workflow table might contain geometry, so the planner permits it with a
      // runtime check. An AX slot explicitly declared geometry-free cannot satisfy that check.
      for (const consumer of template.document.steps) {
        for (const [name, connection] of Object.entries(consumer.inputs)) {
          if (connection.stepId !== slot.source.stepId || connection.output !== slot.source.output)
            continue;
          const input = registry
            .get(consumer.operationId)
            ?.inputs.find(input => input.name === name);
          if (input && !input.accepts.includes(slot.kind === 'surface' ? 'raster' : 'table'))
            report(
              'incompatible-port',
              path,
              `${slot.kind} has no geometry for ${consumer.id}.${name}`
            );
        }
      }
    }
    for (const field of slot.fields) {
      const fieldPath = `${path}.${field.role}`;
      if (field.required !== false && !field.sources.length)
        report('missing-role-source', fieldPath, 'required role needs a parameter reference');
      for (const reference of field.sources) {
        const {step, operation, parameter} = checkReference(reference, fieldPath);
        const ownerKey = `${reference.stepId}.${reference.parameter}`;
        const previousOwner = roleOwners.get(ownerKey);
        if (previousOwner && previousOwner.role !== field.role)
          report(
            'conflicting-role-source',
            fieldPath,
            `parameter is also mapped by ${previousOwner.path}`
          );
        roleOwners.set(ownerKey, {path: fieldPath, role: field.role});
        const describedInputs = step
          ? (getKeplerOperationDescriptor(step.operationId)
              ?.inputs.filter(input =>
                input.fields.some(field => field.parameter === reference.parameter)
              )
              .map(input => input.name) ?? [])
          : [];
        const columnInputs = describedInputs.length
          ? describedInputs
          : parameter?.columnsFrom
            ? [parameter.columnsFrom]
            : [];
        const isColumnParameter = parameter?.kind === 'column' || parameter?.kind === 'column-list';
        const consumesSlot =
          reference.stepId === slot.source.stepId
            ? isColumnParameter &&
              operation?.parameters.some(
                candidate =>
                  candidate.name === parameter.columnsFrom && candidate.kind === 'dataset'
              )
            : columnInputs.length
              ? columnInputs.some(input => {
                  const connection = step?.inputs[input];
                  return connection && hasAncestor(connection.stepId, slot.source.stepId);
                })
              : Object.values(step?.inputs ?? {}).some(connection =>
                  hasAncestor(connection.stepId, slot.source.stepId)
                );
        if (!consumesSlot)
          report('unrelated-role-source', fieldPath, 'column parameter does not consume this slot');
        if (
          parameter &&
          !isColumnParameter &&
          !(parameter.kind === 'string' && describedInputs.length)
        )
          report(
            'invalid-role-parameter',
            fieldPath,
            'role must write a descriptor field or a column parameter with dataset provenance'
          );
        if (typeof field.required !== 'boolean' && operation) {
          // Conditional roles use the target operation's parameter vocabulary; every branch of an
          // any-of condition names a parameter of that operation.
          for (const comparison of getGPUAnalysisParameterComparisons(field.required)) {
            const conditionParameter =
              'parameter' in comparison ? comparison.parameter : comparison.whenParameter;
            const controllingParameter = operation.parameters.find(
              parameter => parameter.name === conditionParameter
            );
            if (!controllingParameter)
              report(
                'unknown-parameter',
                fieldPath,
                `condition parameter ${conditionParameter} does not exist`
              );
            else
              for (const value of comparison.equals) {
                try {
                  normalizeWorkflowParameters(
                    [controllingParameter],
                    {[controllingParameter.name]: value},
                    {operationTitle: operation.title}
                  );
                } catch (error) {
                  report('invalid-role-condition', fieldPath, String(error));
                }
              }
          }
        }
        if (step?.parameters[reference.parameter]?.from === 'input')
          report('conflicting-role-source', fieldPath, 'column role is also a promoted input');
      }
    }
  }
  for (const sourceId of template.fixedSources ?? []) {
    const source = steps.get(sourceId);
    if (!source || !sourceOperations.includes(source.operationId))
      report('invalid-fixed-source', 'fixedSources', `source ${sourceId} must name a source step`);
    if (coveredSources.has(sourceId))
      report('duplicate-slot-source', 'fixedSources', `source ${sourceId} is also a slot`);
    coveredSources.add(sourceId);
  }
  for (const step of template.document.steps) {
    if (sourceOperations.includes(step.operationId) && !coveredSources.has(step.id))
      report('missing-slot', 'slots', `source ${step.id} has no dataset slot`);
  }
  for (const output of template.reads.outputs ?? []) {
    const path = `reads.outputs.${output.source.stepId}.${output.source.output}`;
    const step = steps.get(output.source.stepId);
    if (!step) report('unknown-step', path, 'reading refers to a missing step');
    const operation = step && registry.get(step.operationId);
    if (operation && !operation.outputs.some(port => port.name === output.source.output))
      report('unknown-output-port', path, 'reading refers to a missing output');
  }
  const policyDependencies = new Map<string, string>();
  for (const exposed of template.exposed) {
    const path = `exposed.${exposed.inputId}`;
    const {step, operation, parameter} = checkReference(exposed.target, path);
    const binding = step?.parameters[exposed.target.parameter];
    if (
      !inputs.has(exposed.inputId) ||
      binding?.from !== 'input' ||
      binding.inputId !== exposed.inputId
    )
      report('invalid-exposed-input', path, 'target must reference this existing promoted input');
    const policy = exposed.defaultPolicy;
    if (
      'input' in policy &&
      policy.input !== undefined &&
      operation &&
      !operation.inputs.some(input => input.name === policy.input)
    )
      report('unknown-input-port', path, `default input ${policy.input} does not exist`);
    if ('role' in policy && policy.role !== undefined) {
      const role = policy.role;
      const connection = step?.inputs[policy.input];
      const slot = template.slots.find(
        slot =>
          connection &&
          hasAncestor(connection.stepId, slot.source.stepId) &&
          slot.fields.some(field => field.role === role)
      );
      if (!slot?.fields.some(field => field.role === role))
        report('unknown-role', path, `default role ${role} is absent from input ${policy.input}`);
    }
    if (
      policy.from === 'time-encoding' &&
      parameter &&
      (parameter.kind !== 'enum' ||
        !['iso8601', 'unix-second', 'unix-millisecond'].every(encoding =>
          parameter.options?.includes(encoding)
        ))
    )
      report(
        'invalid-default-policy',
        path,
        'time-encoding requires an enum with ISO and Unix timestamp options'
      );
    if (policy.from === 'data') {
      if (parameter && !['number', 'integer'].includes(parameter.kind))
        report('invalid-default-policy', path, 'data rule requires a numeric parameter');
      if (policy.factorByParameter) {
        const dependency = policy.factorByParameter.parameter;
        if (operation && !operation.parameters.some(parameter => parameter.name === dependency))
          report(
            'unknown-parameter',
            path,
            `default factor parameter ${dependency} does not exist`
          );
        policyDependencies.set(
          `${exposed.target.stepId}.${exposed.target.parameter}`,
          `${exposed.target.stepId}.${dependency}`
        );
      }
    }
    const values =
      policy.from === 'preset'
        ? [policy.value]
        : policy.from === 'field-measurement'
          ? Object.values(policy.values)
          : [];
    if (parameter)
      for (const value of values) {
        try {
          normalizeWorkflowParameters(
            [parameter],
            {[parameter.name]: value},
            {operationTitle: operation!.title}
          );
          const input = inputs.get(exposed.inputId);
          if (input)
            normalizeWorkflowParameters(
              [{...input, name: input.id, defaultValue: input.value}],
              {[input.id]: value},
              {operationTitle: template.title}
            );
        } catch (error) {
          report('invalid-default-value', path, String(error));
        }
      }
  }
  for (const name of policyDependencies.keys()) {
    const visited = new Set<string>();
    let current: string | undefined = name;
    while (current !== undefined) {
      if (visited.has(current)) {
        report('default-cycle', 'exposed', `default policies cycle through ${current}`);
        break;
      }
      visited.add(current);
      current = policyDependencies.get(current);
    }
  }
  return {executable: issues.length === 0, issues};
}
