// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRequirement} from '@luma-contributors/analysis/analysis-core';
import {
  evaluateGPUAnalysisParameterCondition,
  matchGPUAnalysisFieldRole,
  GPU_ANALYSIS_COLUMN_TYPES,
  GPU_ANALYSIS_MEASUREMENT_LEVELS,
  type GPUAnalysisQuestion
} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';
import type {CatalogParameterValue} from '../analysis/catalog-binding-types';
import {getKeplerOperationDescriptor} from '../descriptors/operation-descriptors';
import type {KeplerDatasetProfile} from '../profile/dataset-profile-types';
import {createWorkflowAnalysisTemplateFromDocument} from './workflow-analysis-template';
import type {
  WorkflowAnalysisRoleSource,
  WorkflowAnalysisTemplate,
  WorkflowAnalysisTemplateSlot
} from './workflow-analysis-template';
import {
  serializeWorkflowAnalysisTemplate,
  validateWorkflowAnalysisTemplate
} from './workflow-analysis-template-parser';
import {parseWorkflowDocument, serializeWorkflowDocument} from './workflow-document';
import {
  demoteWorkflowParameter,
  promoteWorkflowParameter,
  updateWorkflowInput
} from './workflow-edit';
import {parseWorkflowExpression} from './workflow-expression';
import {getWorkflowOperationRegistry} from './workflow-operations';
import {normalizeWorkflowParameters} from './workflow-parameters';
import {resolveWorkflowParameters} from './workflow-plan';
import type {WorkflowDocument, WorkflowOperationRegistry, WorkflowStep} from './workflow-types';

export type WorkflowSaveTemplateOptions = {
  id: string;
  title: string;
  question: GPUAnalysisQuestion;
  method: string;
  /** Unselected sources remain explicitly fixed. */
  slots: readonly WorkflowAnalysisTemplateSlot[];
  exposed: readonly WorkflowAnalysisRoleSource[];
};
export type WorkflowTemplateMapping = {
  datasets: Readonly<Record<string, KeplerDatasetProfile>>;
  /** `<slot>.<role>` to chosen columns. */
  columns: Readonly<Record<string, readonly string[]>>;
  /** Promoted input ids; omitted values keep their saved setting. */
  parameters?: Readonly<Record<string, CatalogParameterValue>>;
};
function requireSave(condition: boolean, requirement: string): void {
  validateRequirement('WorkflowSaveTemplate', condition, requirement);
}
function readValues(
  document: WorkflowDocument,
  step: WorkflowStep,
  registry: WorkflowOperationRegistry
) {
  return resolveWorkflowParameters(document, step, registry.get(step.operationId)!);
}

/** Trace source dependencies, optionally requiring known unchanged column names. */
function readsTemplateSource(
  document: WorkflowDocument,
  stepId: string,
  sourceId: string,
  requirePreservedColumns = true,
  visited = new Set<string>()
): boolean {
  if (stepId === sourceId) return true;
  if (visited.has(stepId)) return false;
  visited.add(stepId);
  const step = document.steps.find(step => step.id === stepId);
  return Boolean(
    step &&
      (!requirePreservedColumns ||
        ['table-sort-rows', 'table-filter-rows', 'table-drop-geometry'].includes(
          step.operationId
        )) &&
      Object.values(step.inputs).some(connection =>
        readsTemplateSource(document, connection.stepId, sourceId, requirePreservedColumns, visited)
      )
  );
}

/** Replace identity/geometry while retaining population, limits and their promoted bindings. */
function retainTemplateSourceSettings(
  step: WorkflowStep,
  registry: WorkflowOperationRegistry
): WorkflowStep['parameters'] {
  const identityParameters = new Set([
    'dataset',
    'datasetLabel',
    'geometryMode',
    'geometryField',
    'longitudeField',
    'latitudeField'
  ]);
  const datasetParameters = registry.get('source-dataset')!.parameters;
  return Object.fromEntries(
    Object.entries(step.parameters).filter(
      ([name]) =>
        !identityParameters.has(name) &&
        datasetParameters.some(parameter => parameter.name === name)
    )
  );
}

/** Fixed expressions must be checked before the host presents a mapped template as runnable. */
function checkTemplateExpressionColumns(
  document: WorkflowDocument,
  template: WorkflowAnalysisTemplate,
  mapping: WorkflowTemplateMapping,
  registry: WorkflowOperationRegistry
): void {
  for (const step of document.steps) {
    for (const parameter of registry.get(step.operationId)!.parameters) {
      if (parameter.kind !== 'expression') continue;
      const connection = step.inputs[parameter.columnsFrom ?? ''];
      if (!connection) continue;
      const slots = template.slots.filter(slot =>
        readsTemplateSource(document, connection.stepId, slot.source.stepId, false)
      );
      if (!slots.length) continue;
      const expression = readValues(document, step, registry)[parameter.name];
      requireSave(
        typeof expression === 'string',
        `${step.id}.${parameter.name} must be an expression`
      );
      const references = parseWorkflowExpression(expression as string).columns;
      if (!references.length) continue;
      for (const slot of slots) {
        requireSave(
          readsTemplateSource(document, connection.stepId, slot.source.stepId),
          `${step.id}.${parameter.name} has column-changing steps between its input and slot ${slot.name}; expression-column remapping is unsupported. Keep this source fixed or edit the expression and workflow before saving the template`
        );
        const profile = mapping.datasets[slot.name];
        const available = new Set([
          ...profile.columns.map(column => column.name),
          ...profile.geometryColumns
        ]);
        const missing = references.filter(column => !available.has(column));
        requireSave(
          !missing.length,
          `${step.id}.${parameter.name} keeps fixed expression columns ${missing.join(', ')} that dataset ${profile.datasetId} does not contain; expression-column remapping is unsupported. Rename these columns or edit the expression before saving the template`
        );
      }
    }
  }
}

/** Infer reusable roles from actual consumers, without merging constraints across saved values. */
export function getWorkflowSaveTemplateSlots(
  document: WorkflowDocument,
  registry: WorkflowOperationRegistry = getWorkflowOperationRegistry()
): WorkflowAnalysisTemplateSlot[] {
  const inferred = createWorkflowAnalysisTemplateFromDocument({
    id: document.id,
    title: document.title,
    summary: document.description ?? document.title,
    document
  });
  return inferred.slots.map(slot => {
    const fields: WorkflowAnalysisTemplateSlot['fields'][number][] = [];
    for (const step of document.steps) {
      for (const input of getKeplerOperationDescriptor(step.operationId)?.inputs ?? []) {
        const connection = step.inputs[input.name];
        if (!connection || !readsTemplateSource(document, connection.stepId, slot.source.stepId))
          continue;
        for (const field of input.fields) {
          if (
            fields.some(existing =>
              existing.sources.some(
                source => source.stepId === step.id && source.parameter === field.parameter
              )
            )
          )
            continue;
          const {
            parameter,
            measurementByGeometry: ignoredGeometry,
            autoAssign: ignoredAutoAssign,
            ...role
          } = field;
          const baseRole = fields.some(field => field.role === role.role)
            ? `${role.role}-${step.id}`
            : role.role;
          let roleName = baseRole;
          for (let suffix = 2; fields.some(field => field.role === roleName); suffix++)
            roleName = `${baseRole}-${suffix}`;
          fields.push({...role, role: roleName, sources: [{stepId: step.id, parameter}]});
        }
      }
      for (const parameter of registry.get(step.operationId)?.parameters ?? []) {
        if (!['column', 'column-list'].includes(parameter.kind)) continue;
        const connection = step.inputs[parameter.columnsFrom ?? ''];
        if (!connection || !readsTemplateSource(document, connection.stepId, slot.source.stepId))
          continue;
        if (
          fields.some(field =>
            field.sources.some(
              source => source.stepId === step.id && source.parameter === parameter.name
            )
          )
        )
          continue;
        fields.push({
          role: `${parameter.name}-${step.id}`,
          label: parameter.label,
          accepts: GPU_ANALYSIS_COLUMN_TYPES,
          measurement: GPU_ANALYSIS_MEASUREMENT_LEVELS,
          required: Boolean(readValues(document, step, registry)[parameter.name]),
          ...(parameter.kind === 'column-list' ? {multiple: {minimum: 1, maximum: 64}} : {}),
          sources: [{stepId: step.id, parameter: parameter.name}]
        });
      }
    }
    return {...slot, fields};
  });
}

/** Save over the existing document and promoted-input contracts, preserving explicit settings. */
export function saveWorkflowAnalysisTemplate(
  source: WorkflowDocument,
  options: WorkflowSaveTemplateOptions,
  registry: WorkflowOperationRegistry = getWorkflowOperationRegistry()
): WorkflowAnalysisTemplate {
  let document = parseWorkflowDocument(serializeWorkflowDocument(source));
  for (const step of document.steps)
    for (const [parameter, binding] of Object.entries(step.parameters)) {
      if (
        binding.from === 'input' &&
        !options.exposed.some(target => target.stepId === step.id && target.parameter === parameter)
      )
        document = demoteWorkflowParameter(document, step.id, parameter);
    }
  const exposed: WorkflowAnalysisTemplate['exposed'][number][] = [];
  for (const target of options.exposed) {
    requireSave(
      !options.slots.some(slot =>
        slot.fields.some(field =>
          field.sources.some(
            source => source.stepId === target.stepId && source.parameter === target.parameter
          )
        )
      ),
      `exposed ${target.stepId}.${target.parameter} must not also be a field role`
    );
    const promoted = promoteWorkflowParameter(document, registry, target.stepId, target.parameter);
    document = promoted.document;
    exposed.push({inputId: promoted.inputId, target, defaultPolicy: {from: 'constant'}});
  }
  const selected = new Set(options.slots.map(slot => slot.source.stepId));
  const sources = document.steps.filter(step =>
    ['source-dataset', 'source-geojson', 'source-drawing'].includes(step.operationId)
  );
  const fixedSources = sources.filter(step => !selected.has(step.id)).map(step => step.id);
  for (const slot of options.slots) {
    const step = document.steps.find(step => step.id === slot.source.stepId);
    requireSave(
      Boolean(step && sources.includes(step)),
      `slot ${slot.name} must name a source step`
    );
    const settings = retainTemplateSourceSettings(step!, registry);
    step!.operationId = 'source-dataset';
    step!.parameters = {
      ...settings,
      dataset: {from: 'static', value: slot.name},
      geometryMode: {from: 'static', value: 'none'}
    };
    for (const field of slot.fields)
      for (const target of field.sources) {
        const consumer = document.steps.find(step => step.id === target.stepId);
        requireSave(Boolean(consumer), `role ${field.role} must name an existing step`);
        consumer!.parameters[target.parameter] = {from: 'static', value: field.role};
      }
  }
  const usedInputs = new Set(
    document.steps.flatMap(step =>
      Object.values(step.parameters).flatMap(binding =>
        binding.from === 'input' ? [binding.inputId] : []
      )
    )
  );
  document = {
    ...document,
    id: options.id,
    title: options.title,
    inputs: document.inputs.filter(input => usedInputs.has(input.id)),
    settings: {...document.settings, autoRun: false}
  };
  const inferred = createWorkflowAnalysisTemplateFromDocument({
    id: options.id,
    title: options.title,
    summary: options.method,
    document
  });
  const template = serializeWorkflowAnalysisTemplate({
    schemaVersion: 1,
    id: options.id,
    title: options.title,
    question: options.question,
    method: options.method,
    document,
    slots: options.slots,
    exposed,
    ...(fixedSources.length ? {fixedSources} : {}),
    reads: inferred.reads
  });
  const validation = validateWorkflowAnalysisTemplate(template, {registry});
  requireSave(validation.executable, validation.issues.map(issue => issue.message).join('\n'));
  return template;
}

/** Explicit choices use AX type/measurement matching and the ordinary workflow runner. */
export function mapWorkflowAnalysisTemplate(
  supplied: WorkflowAnalysisTemplate,
  mapping: WorkflowTemplateMapping,
  registry: WorkflowOperationRegistry = getWorkflowOperationRegistry()
): WorkflowDocument {
  const template = serializeWorkflowAnalysisTemplate(supplied);
  const validation = validateWorkflowAnalysisTemplate(template, {registry});
  requireSave(validation.executable, validation.issues.map(issue => issue.message).join('\n'));
  let document = template.document;
  for (const [inputId, value] of Object.entries(mapping.parameters ?? {})) {
    requireSave(
      template.exposed.some(exposed => exposed.inputId === inputId),
      `parameters ${inputId} must name an exposed input`
    );
    document = updateWorkflowInput(document, inputId, {value});
  }
  const assigned = new Map<string, string>();
  for (const slot of template.slots) {
    const profile = mapping.datasets[slot.name];
    requireSave(Boolean(profile), `slot ${slot.name} needs a dataset`);
    requireSave(
      profile.kind === slot.kind || (slot.kind === 'network' && profile.kind === 'lines'),
      `slot ${slot.name} needs ${slot.kind}`
    );
    const source = document.steps.find(step => step.id === slot.source.stepId)!;
    const geometry = profile.geometryColumns[0];
    source.operationId = 'source-dataset';
    source.parameters = {
      ...retainTemplateSourceSettings(source, registry),
      ...Object.fromEntries(
        Object.entries({
          dataset: profile.datasetId,
          datasetLabel: profile.label,
          geometryMode: geometry
            ? 'geojson-field'
            : profile.pointColumns
              ? 'point-columns'
              : 'none',
          ...(geometry
            ? {geometryField: geometry}
            : profile.pointColumns
              ? {
                  longitudeField: profile.pointColumns.longitude,
                  latitudeField: profile.pointColumns.latitude
                }
              : {})
        }).map(([name, value]) => [name, {from: 'static', value}])
      )
    };
    for (const field of slot.fields) {
      const columns = mapping.columns[`${slot.name}.${field.role}`] ?? [];
      const required = field.sources.some(target =>
        typeof field.required === 'boolean'
          ? field.required
          : evaluateGPUAnalysisParameterCondition(
              field.required,
              readValues(
                document,
                document.steps.find(step => step.id === target.stepId)!,
                registry
              )
            )
      );
      const minimum = required || columns.length > 0 ? (field.multiple?.minimum ?? 1) : 0;
      requireSave(
        columns.length >= minimum &&
          columns.length <= (field.multiple?.maximum ?? 1) &&
          new Set(columns).size === columns.length,
        `role ${slot.name}.${field.role} needs ${minimum} to ${field.multiple?.maximum ?? 1} distinct columns`
      );
      const candidates = matchGPUAnalysisFieldRole(field, profile.columns);
      requireSave(
        columns.every(column => candidates.some(candidate => candidate.name === column)),
        `role ${slot.name}.${field.role} needs compatible columns`
      );
      for (const target of field.sources) {
        const step = document.steps.find(step => step.id === target.stepId)!;
        const parameter = registry
          .get(step.operationId)!
          .parameters.find(parameter => parameter.name === target.parameter)!;
        const multiple = parameter.kind === 'column-list' || Boolean(field.multiple);
        requireSave(
          !multiple || columns.every(column => !column.includes(',') && column.trim() === column),
          `role ${field.role} column-list names must have no commas or surrounding whitespace`
        );
        const value = multiple ? columns.join(',') : (columns[0] ?? '');
        const key = `${target.stepId}.${target.parameter}`;
        requireSave(
          !assigned.has(key) || assigned.get(key) === value,
          `role ${field.role} must agree with the other slot mapping ${key}`
        );
        normalizeWorkflowParameters(
          [parameter],
          {[parameter.name]: value},
          {operationTitle: step.label ?? step.operationId}
        );
        assigned.set(key, value);
        step.parameters[target.parameter] = {from: 'static', value};
      }
    }
  }
  checkTemplateExpressionColumns(document, template, mapping, registry);
  return parseWorkflowDocument(serializeWorkflowDocument(document));
}
