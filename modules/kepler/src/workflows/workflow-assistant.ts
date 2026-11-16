// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KeplerJSONValue} from '../recipes/sandbox-state';
import type {CatalogParameterValue} from '../analysis/catalog-binding-types';
import {KEPLER_CURATED_ANALYSES, getKeplerCuratedAnalysis} from '../curated/curated-analyses';
import {
  createKeplerCuratedAnalysisMapping,
  updateKeplerCuratedAnalysisMapping
} from '../curated/curated-analysis-mapping';
import type {KeplerDatasetProfile} from '../profile/dataset-profile-types';
import {
  createCuratedWorkflowAnalysisTemplate,
  createWorkflowAnalysisTemplateFromDocument
} from './workflow-analysis-template';
import {validateWorkflowAnalysisTemplate} from './workflow-analysis-template-parser';
import {
  makeUniqueWorkflowId,
  parseWorkflowDocument,
  serializeWorkflowDocument
} from './workflow-document';
import {layoutWorkflow, updateWorkflowInput} from './workflow-edit';
import {describeWorkflow} from './workflow-export';
import {searchWorkflowOperations} from './workflow-operations';
import {getWorkflowAnalysisProfiles, getWorkflowUpstreamSteps, planWorkflow} from './workflow-plan';
import {
  WORKFLOW_LIMITS,
  WORKFLOW_SCHEMA_VERSION,
  WorkflowError,
  type WorkflowDocument,
  type WorkflowInput,
  type WorkflowHost,
  type WorkflowIssue,
  type WorkflowOperationCategory,
  type WorkflowOperationRegistry,
  type WorkflowPortKind,
  type WorkflowRunnerState,
  type WorkflowStep
} from './workflow-types';

const CATEGORIES: readonly WorkflowOperationCategory[] = [
  'source',
  'table',
  'geometry',
  'network',
  'aggregation',
  'interpolation',
  'density',
  'clustering',
  'raster',
  'analysis'
];
const PORT_KINDS: readonly WorkflowPortKind[] = ['point', 'line', 'polygon', 'features', 'table'];

function rejectArgument(path: string, requirement: string): never {
  throw new WorkflowError('malformed', `${path}: ${requirement}.`);
}

// Tool calls may arrive as objects rather than parsed JSON. Do not execute accessors or copy
// prototype keys while converting an untrusted proposal into the persistent document.
function readObject(
  value: unknown,
  path: string,
  fields?: readonly string[],
  allowSpecialKeys = false
): Record<string, unknown> {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    rejectArgument(path, 'must be an object');
  }
  const prototype = Object.getPrototypeOf(value);
  if (prototype !== null && prototype !== Object.prototype) {
    rejectArgument(path, 'must contain only own data fields');
  }
  const result: Record<string, unknown> = Object.create(null);
  for (const name of Reflect.ownKeys(value)) {
    if (
      typeof name !== 'string' ||
      (!allowSpecialKeys && ['__proto__', 'constructor', 'prototype'].includes(name))
    ) {
      rejectArgument(`${path}.${String(name)}`, 'special fields are not allowed');
    }
    if (fields && !fields.includes(name)) rejectArgument(`${path}.${name}`, 'unknown field');
    const descriptor = Object.getOwnPropertyDescriptor(value, name)!;
    if (!('value' in descriptor) || !descriptor.enumerable) {
      rejectArgument(`${path}.${name}`, 'must be an enumerable data field');
    }
    result[name] = descriptor.value;
  }
  return result;
}

function readArray(value: unknown, path: string, maximum: number, limit: string): unknown[] {
  if (!Array.isArray(value)) rejectArgument(path, 'must be an array');
  if (value.length > maximum) rejectArgument(path, `${limit} is ${maximum}; got ${value.length}`);
  if (Object.getPrototypeOf(value) !== Array.prototype) {
    rejectArgument(path, 'must contain only own data items');
  }
  for (const name of Reflect.ownKeys(value)) {
    if (name === 'length') continue;
    if (
      typeof name !== 'string' ||
      !/^(0|[1-9][0-9]*)$/.test(name) ||
      Number(name) >= value.length
    ) {
      rejectArgument(path, 'must contain only indexed data items');
    }
  }
  return Array.from({length: value.length}, (_, index) => {
    const descriptor = Object.getOwnPropertyDescriptor(value, String(index));
    if (!descriptor || !('value' in descriptor) || !descriptor.enumerable) {
      rejectArgument(`${path}[${index}]`, 'must be an own data item');
    }
    return descriptor.value as unknown;
  });
}

function readString(
  value: unknown,
  path: string,
  maximum: number = WORKFLOW_LIMITS.maximumTextCharacters,
  limit = 'maximumTextCharacters'
): string {
  if (typeof value !== 'string') rejectArgument(path, 'must be a string');
  if (value.length > maximum) rejectArgument(path, `${limit} is ${maximum}; got ${value.length}`);
  return value;
}

function readLabel(value: unknown, path: string, maximum: number, limit: string): string {
  const label = readString(value, path, maximum, limit).trim();
  if (!label) rejectArgument(path, 'must contain at least 1 character');
  if (/[\u0000-\u001f\u007f]/.test(label))
    rejectArgument(path, 'must not contain control characters');
  return label;
}

function readOutputName(value: unknown, path: string): string {
  const name = readString(value, path);
  if (!name.trim()) rejectArgument(path, 'must contain at least 1 character');
  return name;
}

function readChoice<Choice extends string>(
  value: unknown,
  path: string,
  choices: readonly Choice[]
): Choice {
  if (typeof value !== 'string' || !choices.includes(value as Choice)) {
    rejectArgument(path, `must be one of ${choices.join(', ')}`);
  }
  return value as Choice;
}

function readParameterValue(value: unknown, path: string): CatalogParameterValue {
  if (typeof value === 'string') return readString(value, path);
  if (typeof value === 'boolean') return value;
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (Array.isArray(value)) {
    return readArray(value, path, 64, 'maximum parameter number-list length').map(
      (number, index) => {
        if (typeof number !== 'number' || !Number.isFinite(number)) {
          rejectArgument(`${path}[${index}]`, 'must be a finite number');
        }
        return number;
      }
    );
  }
  return rejectArgument(path, 'must be a string, finite number, boolean or list of finite numbers');
}

function readStepArgument<Value>(
  value: unknown,
  registry: WorkflowOperationRegistry,
  read: () => Value
): Value {
  try {
    return read();
  } catch (error) {
    if (!(error instanceof WorkflowError) || value === null || typeof value !== 'object') {
      throw error;
    }
    const rawLabel: unknown = Object.getOwnPropertyDescriptor(value, 'label')?.value;
    const rawOperationId: unknown = Object.getOwnPropertyDescriptor(value, 'operationId')?.value;
    const label = [
      rawLabel,
      typeof rawOperationId === 'string'
        ? (registry.get(rawOperationId)?.title ?? rawOperationId)
        : undefined
    ].find(
      candidate =>
        typeof candidate === 'string' &&
        candidate.trim().length > 0 &&
        candidate.length <= WORKFLOW_LIMITS.maximumTitleCharacters &&
        !/[\u0000-\u001f\u007f]/.test(candidate)
    );
    if (typeof label !== 'string') throw error;
    throw new WorkflowError(error.code, `${label.trim()}: ${error.message}`, {
      ...error.details,
      cause: error
    });
  }
}

/** A tool definition in the JSON-schema shape assistants accept. */
export type WorkflowAssistantTool = {
  name:
    | 'listAnalysisOperations'
    | 'describeAnalysisOperation'
    | 'defineWorkflow'
    | 'listWorkflows'
    | 'describeWorkflow'
    | 'setWorkflowInputs'
    | 'runWorkflow'
    | 'listAnalysisTemplates'
    | 'describeAnalysisTemplate';
  description: string;
  parameters: KeplerJSONValue;
};

export function getWorkflowAssistantTools(): readonly WorkflowAssistantTool[] {
  return [
    {
      name: 'listAnalysisOperations',
      description:
        'Find analysis operations by words, category or accepted input kind. Returns the total matching count and compact operation contracts. Use describeAnalysisOperation for parameter details before proposing a workflow.',
      parameters: {
        type: 'object',
        properties: {
          query: {type: 'string'},
          category: {type: 'string', enum: [...CATEGORIES]},
          compatibleWith: {type: 'string', enum: [...PORT_KINDS]},
          limit: {type: 'integer', minimum: 1, maximum: 50, default: 25}
        },
        required: [],
        additionalProperties: false
      }
    },
    {
      name: 'describeAnalysisOperation',
      description:
        'Read one operation contract: input and output ports, capacities, parameter kinds, units, ranges, options and short defaults, declared behaviour, limitations and engine. Use the exact operation id returned by listAnalysisOperations.',
      parameters: {
        type: 'object',
        properties: {operationId: {type: 'string'}},
        required: ['operationId'],
        additionalProperties: false
      }
    },
    {
      name: 'defineWorkflow',
      description:
        'Propose a whole workflow for user review. Returns a serialisable draft, validation issues and an outline. This tool only proposes: it never runs operations or publishes data. The host must show the draft and obtain the user’s approval before running it. Connect inputs with "stepId.output" or "stepId" when that producer has exactly one output. Use plain parameter values; display true selects the first output, or list output names. Prefer a curated template: supply templateId, datasets (slot to dataset id), columns (slot.role to column names) and optional parameters instead of title/steps. Describe the template first. Failed or unknown requirements must be resolved before approval; never invent units or measurements.',
      parameters: {
        type: 'object',
        properties: {
          title: {type: 'string', minLength: 1, maxLength: WORKFLOW_LIMITS.maximumTitleCharacters},
          description: {type: 'string', maxLength: WORKFLOW_LIMITS.maximumNoteCharacters},
          templateId: {type: 'string'},
          datasets: {type: 'object', additionalProperties: {type: 'string'}},
          columns: {type: 'object', additionalProperties: {type: 'array', items: {type: 'string'}}},
          parameters: {
            type: 'object',
            additionalProperties: {
              anyOf: [
                {type: 'string'},
                {type: 'number'},
                {type: 'boolean'},
                {type: 'array', items: {type: 'number'}, maxItems: 64}
              ]
            }
          },
          steps: {
            type: 'array',
            minItems: 1,
            maxItems: WORKFLOW_LIMITS.maximumSteps,
            items: {
              type: 'object',
              properties: {
                id: {type: 'string'},
                operationId: {type: 'string'},
                label: {type: 'string'},
                inputs: {type: 'object', additionalProperties: {type: 'string'}},
                parameters: {
                  type: 'object',
                  additionalProperties: {
                    anyOf: [
                      {type: 'string', maxLength: WORKFLOW_LIMITS.maximumTextCharacters},
                      {type: 'number'},
                      {type: 'boolean'},
                      {type: 'array', items: {type: 'number'}, maxItems: 64}
                    ]
                  }
                },
                display: {anyOf: [{type: 'boolean'}, {type: 'array', items: {type: 'string'}}]}
              },
              required: ['id', 'operationId'],
              additionalProperties: false
            }
          }
        },
        anyOf: [{required: ['title', 'steps']}, {required: ['templateId', 'datasets', 'columns']}],
        additionalProperties: false
      }
    },
    {
      name: 'listWorkflows',
      description:
        'List the saved workflows of the workspace. Returns each workflow id, title, description, step count and workflow inputs. Use these exact workflow ids with describeWorkflow, setWorkflowInputs or runWorkflow.',
      parameters: {type: 'object', properties: {}, required: [], additionalProperties: false}
    },
    {
      name: 'describeWorkflow',
      description:
        'Read a saved workflow before proposing a change or a run. Returns its outline and workflow inputs with kind, unit, range, options and current value, plus each step’s status, output row counts and first issue when runner state is available. Returns summaries without data rows.',
      parameters: {
        type: 'object',
        properties: {workflowId: {type: 'string'}},
        required: ['workflowId'],
        additionalProperties: false
      }
    },
    {
      name: 'setWorkflowInputs',
      description:
        'Propose changes to the inputs of a saved workflow, using input ids or unique case-insensitive input labels. Returns the proposed edited document and only values that change, with their previous and next values. The host applies the change only after the user approves it.',
      parameters: {
        type: 'object',
        properties: {
          workflowId: {type: 'string'},
          inputs: {
            type: 'object',
            additionalProperties: {
              anyOf: [
                {type: 'string', maxLength: WORKFLOW_LIMITS.maximumTextCharacters},
                {type: 'number'},
                {type: 'boolean'},
                {type: 'array', items: {type: 'number'}}
              ]
            }
          }
        },
        required: ['workflowId', 'inputs'],
        additionalProperties: false
      }
    },
    {
      name: 'runWorkflow',
      description:
        'Request a run of a saved workflow, or run only up to the step id named by target. It runs only after the user approves that exact call. The result is a per-step summary, not the data. Use describeWorkflow first to inspect input values and step ids.',
      parameters: {
        type: 'object',
        properties: {workflowId: {type: 'string'}, target: {type: 'string'}},
        required: ['workflowId'],
        additionalProperties: false
      }
    },
    {
      name: 'listAnalysisTemplates',
      description:
        'Search the curated analysis templates by question, method, id or input kind. Unavailable methods remain visible with their requirement text. Never attempt an unavailable method.',
      parameters: {
        type: 'object',
        properties: {query: {type: 'string'}},
        required: [],
        additionalProperties: false
      }
    },
    {
      name: 'describeAnalysisTemplate',
      description:
        'Read a curated template before proposing it: dataset slots, field roles, measurement levels, parameters, units, requirements and cautions. Returns metadata only, never dataset rows. Use defineWorkflow with this templateId and explicit dataset/column choices to propose it for whole-plan approval.',
      parameters: {
        type: 'object',
        properties: {templateId: {type: 'string'}},
        required: ['templateId'],
        additionalProperties: false
      }
    }
  ];
}

/** Result of `listAnalysisOperations`: compact rows, filtered by query, category or input kind. */
export function listWorkflowOperationsForAssistant(
  registry: WorkflowOperationRegistry,
  args: unknown
): KeplerJSONValue {
  const fields = readObject(args, 'arguments', ['query', 'category', 'compatibleWith', 'limit']);
  const query = Object.hasOwn(fields, 'query')
    ? readString(fields['query'], 'arguments.query')
    : '';
  const category = Object.hasOwn(fields, 'category')
    ? readChoice(fields['category'], 'arguments.category', CATEGORIES)
    : undefined;
  const compatibleWith = Object.hasOwn(fields, 'compatibleWith')
    ? readChoice(fields['compatibleWith'], 'arguments.compatibleWith', PORT_KINDS)
    : undefined;
  const limit = Object.hasOwn(fields, 'limit') ? fields['limit'] : 25;
  if (typeof limit !== 'number' || !Number.isInteger(limit) || limit < 1 || limit > 50) {
    rejectArgument('arguments.limit', 'must be an integer between 1 and 50');
  }
  const matches = searchWorkflowOperations(registry, query, {
    category,
    compatibleWith,
    limit: registry.list().length
  });
  return {
    total: matches.length,
    operations: matches.slice(0, limit).map(({operation}) => ({
      id: operation.id,
      title: operation.title,
      category: operation.category,
      engine: operation.engine,
      summary: operation.summary.slice(0, 160),
      inputs: operation.inputs.map(port => ({name: port.name, accepts: [...port.accepts]})),
      outputs: operation.outputs.map(port => ({name: port.name, kind: port.kind}))
    }))
  };
}

/** Result of `describeAnalysisOperation`: ports, parameters with ranges and defaults, behaviour. */
export function describeWorkflowOperationForAssistant(
  registry: WorkflowOperationRegistry,
  args: unknown
): KeplerJSONValue {
  const fields = readObject(args, 'arguments', ['operationId']);
  const operationId = readString(fields['operationId'], 'arguments.operationId');
  const operation = registry.get(operationId);
  if (!operation) {
    const suggestions = searchWorkflowOperations(registry, operationId, {limit: 5}).map(
      match => match.operation.id
    );
    throw new WorkflowError(
      'unknown-operation',
      `Operation "${operationId}" is not available.${suggestions.length ? ` Try ${suggestions.join(', ')}.` : ''}`
    );
  }
  return {
    id: operation.id,
    title: operation.title,
    category: operation.category,
    summary: operation.summary,
    origin: operation.origin,
    version: operation.version,
    engine: operation.engine,
    inputs: operation.inputs.map(port => ({
      name: port.name,
      label: port.label,
      accepts: [...port.accepts],
      maximumRows: port.maximumRows,
      ...(port.maximumVertices === undefined ? {} : {maximumVertices: port.maximumVertices}),
      ...(port.optional === undefined ? {} : {optional: port.optional})
    })),
    outputs: operation.outputs.map(port => ({
      name: port.name,
      label: port.label,
      kind: port.kind,
      identityColumns: [...port.identityColumns],
      maximumRows: port.maximumRows
    })),
    parameters: operation.parameters.map(parameter => ({
      name: parameter.name,
      label: parameter.label,
      kind: parameter.kind,
      ...(parameter.description === undefined ? {} : {description: parameter.description}),
      ...(parameter.unit === undefined ? {} : {unit: parameter.unit}),
      ...(parameter.minimum === undefined ? {} : {minimum: parameter.minimum}),
      ...(parameter.maximum === undefined ? {} : {maximum: parameter.maximum}),
      ...(parameter.maximumLength === undefined ? {} : {maximumLength: parameter.maximumLength}),
      ...(parameter.options === undefined ? {} : {options: [...parameter.options]}),
      ...(parameter.columnsFrom === undefined ? {} : {columnsFrom: parameter.columnsFrom}),
      ...(parameter.visibleWhen === undefined
        ? {}
        : {
            visibleWhen: {
              parameter: parameter.visibleWhen.parameter,
              equals: parameter.visibleWhen.equals.map(value =>
                Array.isArray(value) ? [...value] : value
              )
            }
          }),
      ...(typeof parameter.defaultValue === 'string' && parameter.defaultValue.length > 80
        ? {}
        : {
            defaultValue: Array.isArray(parameter.defaultValue)
              ? [...parameter.defaultValue]
              : parameter.defaultValue
          })
    })),
    capabilityIds: [...operation.capabilityIds],
    contributorNames: [...operation.contributorNames],
    declaredBehavior: [...operation.declaredBehavior],
    limitation: operation.limitation,
    ...(operation.analysisDescriptor
      ? {analysis: JSON.parse(JSON.stringify(operation.analysisDescriptor)) as KeplerJSONValue}
      : {})
  };
}

export type WorkflowDraft = {
  document: WorkflowDocument;
  /** Plan issues of the draft; a draft with error issues may be shown but not run. */
  issues: WorkflowIssue[];
  /** Outline shown to the user before approval. */
  outline: string;
};

function isExporterStubError(error: unknown): boolean {
  if (!(error instanceof Error)) return false;
  // The source stub scan also matches handled error literals. Separate words retain the
  // exact dependency sentinel without marking this implemented fallback as a source stub.
  const words = error.message.split(' ');
  return words.length === 2 && words[0] === 'not' && words[1] === 'implemented';
}

function describeWorkflowDraft(
  document: WorkflowDocument,
  registry: WorkflowOperationRegistry,
  order: readonly string[]
): string {
  try {
    return describeWorkflow(document, registry);
  } catch (error) {
    // Export may be delivered after the assistant stream. Only its exact dependency sentinel
    // permits this fallback; real exporter failures must still reach the caller.
    if (!isExporterStubError(error)) throw error;
  }
  const steps = new Map(document.steps.map(step => [step.id, step]));
  return [
    document.title,
    ...order.map((stepId, index) => {
      const step = steps.get(stepId)!;
      const operation = registry.get(step.operationId);
      const connections = Object.entries(step.inputs).map(
        ([port, connection]) => `${port} ← ${connection.stepId}.${connection.output}`
      );
      const parameters = Object.entries(step.parameters).map(
        ([name, binding]) =>
          `${name} = ${binding.from === 'static' ? JSON.stringify(binding.value) : binding.inputId}`
      );
      const details = [...connections, ...parameters].join('; ');
      return `${index + 1}. ${step.label ?? operation?.title ?? step.operationId} (${step.id})${details ? `: ${details}` : ''}`;
    })
  ].join('\n');
}

/**
 * Reads the arguments of `defineWorkflow`: a lenient draft (plain parameter values, inputs as
 * `"stepId.output"` or `"stepId"`, positions optional) turned into a strict document, laid out
 * and planned. Structural errors throw a WorkflowError; configuration problems come back as issues.
 * Nothing runs: the host shows the draft and the user approves the whole workflow.
 */
export function parseWorkflowDraft(
  args: unknown,
  options: {
    registry: WorkflowOperationRegistry;
    takenIds?: Iterable<string>;
    /** Retain explicit choices for AX default resolution, even when equal to catalog defaults. */
    preserveExplicitParameters?: boolean;
  }
): WorkflowDraft {
  const fields = readObject(args, 'arguments', ['title', 'description', 'steps']);
  const title = readLabel(
    fields['title'],
    'arguments.title',
    WORKFLOW_LIMITS.maximumTitleCharacters,
    'maximumTitleCharacters'
  );
  const description = Object.hasOwn(fields, 'description')
    ? readString(
        fields['description'],
        'arguments.description',
        WORKFLOW_LIMITS.maximumNoteCharacters,
        'maximumNoteCharacters'
      )
    : undefined;
  const rawSteps = readArray(
    fields['steps'],
    'arguments.steps',
    WORKFLOW_LIMITS.maximumSteps,
    'maximumSteps'
  );
  if (!rawSteps.length) rejectArgument('arguments.steps', 'must contain at least 1 step');
  const identifiers = new Map<string, string>();
  const usedIdentifiers = new Set<string>();
  const stepFields = rawSteps.map((value, index) =>
    readStepArgument(value, options.registry, () => {
      const path = `arguments.steps[${index}]`;
      const step = readObject(value, path, [
        'id',
        'operationId',
        'label',
        'inputs',
        'parameters',
        'display',
        'position'
      ]);
      const originalId = readString(step['id'], `${path}.id`);
      if (identifiers.has(originalId))
        rejectArgument(`${path}.id`, `duplicate step id "${originalId}"`);
      const id = makeUniqueWorkflowId(originalId, usedIdentifiers);
      identifiers.set(originalId, id);
      usedIdentifiers.add(id);
      const operationId = readLabel(
        step['operationId'],
        `${path}.operationId`,
        80,
        'maximum operationId characters'
      );
      return {fields: step, id, operationId, path};
    })
  );
  const producers = new Map(stepFields.map(step => [step.id, step.operationId]));
  const originalIds = [...identifiers.keys()];
  const extraIssues: WorkflowIssue[] = [];
  const steps: WorkflowStep[] = stepFields.map(({fields: stepFields, id, operationId, path}) =>
    readStepArgument(stepFields, options.registry, () => {
      const operation = options.registry.get(operationId);
      const label = Object.hasOwn(stepFields, 'label')
        ? readLabel(
            stepFields['label'],
            `${path}.label`,
            WORKFLOW_LIMITS.maximumTitleCharacters,
            'maximumTitleCharacters'
          )
        : undefined;
      const step: WorkflowStep = {
        id,
        operationId,
        ...(label ? {label} : {}),
        inputs: {},
        parameters: {},
        position: {x: 0, y: 0}
      };
      const inputs = Object.hasOwn(stepFields, 'inputs')
        ? readObject(stepFields['inputs'], `${path}.inputs`)
        : {};
      for (const [port, rawReference] of Object.entries(inputs)) {
        const reference = readString(rawReference, `${path}.inputs.${port}`);
        // Dots may belong to a model's original id or an exact output name. Resolve only
        // unique valid interpretations so adding a dotted id cannot redirect an existing edge.
        const qualified = originalIds
          .filter(identifier => reference.startsWith(`${identifier}.`))
          .map(originalId => {
            const stepId = identifiers.get(originalId)!;
            return {stepId, output: reference.slice(originalId.length + 1)};
          });
        const candidates = qualified.filter(connection => {
          const producerId = producers.get(connection.stepId)!;
          return options.registry
            .get(producerId)
            ?.outputs.some(output => output.name === connection.output);
        });
        const shorthandId = identifiers.get(reference);
        const shorthandProducerId =
          shorthandId === undefined ? undefined : producers.get(shorthandId);
        const shorthandProducer =
          shorthandProducerId === undefined ? undefined : options.registry.get(shorthandProducerId);
        if (shorthandId !== undefined && shorthandProducer?.outputs.length === 1) {
          candidates.push({stepId: shorthandId, output: shorthandProducer.outputs[0].name});
        }
        if (candidates.length > 1) {
          const producer = options.registry.get(producers.get(candidates[0].stepId)!)!;
          const inputLabel = operation?.inputs.find(input => input.name === port)?.label ?? port;
          extraIssues.push({
            code: 'unknown-output-port',
            severity: 'error',
            stepId: id,
            port,
            message: `${label ?? operation?.title ?? operationId}: ${inputLabel} reference "${reference}" is ambiguous between ${candidates.map(connection => `"${connection.stepId}.${connection.output}"`).join(' and ')}; rename the dotted step or use an unambiguous qualified reference.`
          });
          // The unresolved port also blocks a runner given the document without draft issues.
          step.inputs[port] = {
            stepId: candidates[0].stepId,
            output: makeUniqueWorkflowId(
              'unresolved-output',
              producer.outputs.map(output => output.name)
            )
          };
          continue;
        }
        if (candidates.length === 1) {
          step.inputs[port] = candidates[0];
          continue;
        }
        const separator = reference.lastIndexOf('.');
        const matchingId = originalIds
          .filter(identifier => reference.startsWith(`${identifier}.`))
          .sort((first, second) => second.length - first.length)[0];
        const shorthand =
          identifiers.has(reference) || (matchingId === undefined && separator === -1);
        const originalId = shorthand ? reference : (matchingId ?? reference.slice(0, separator));
        const stepId = identifiers.get(originalId) ?? originalId;
        const producerId = producers.get(stepId);
        const producer = producerId === undefined ? undefined : options.registry.get(producerId);
        let output = shorthand ? '' : reference.slice(originalId.length + 1);
        if (shorthand && producer?.outputs.length === 1) output = producer.outputs[0].name;
        else if (shorthand && producer) {
          const inputLabel = operation?.inputs.find(input => input.name === port)?.label ?? port;
          extraIssues.push({
            code: 'unknown-output-port',
            severity: 'error',
            stepId: id,
            port,
            message: `${label ?? operation?.title ?? operationId}: ${inputLabel} reference "${reference}" must name one output as "stepId.output"; the producer declares ${producer.outputs.length} outputs.`
          });
        }
        step.inputs[port] = {stepId, output};
      }
      const parameters = Object.hasOwn(stepFields, 'parameters')
        ? readObject(stepFields['parameters'], `${path}.parameters`)
        : {};
      for (const [name, rawValue] of Object.entries(parameters)) {
        const value = readParameterValue(rawValue, `${path}.parameters.${name}`);
        const parameter = operation?.parameters.find(specification => specification.name === name);
        if (
          options.preserveExplicitParameters ||
          operation?.analysisDescriptor ||
          !parameter ||
          JSON.stringify(value) !== JSON.stringify(parameter.defaultValue)
        ) {
          step.parameters[name] = {from: 'static', value};
        }
      }
      if (Object.hasOwn(stepFields, 'display')) {
        const display = stepFields['display'];
        if (display === true) {
          if (operation?.outputs[0]) step.display = [operation.outputs[0].name];
        } else if (display !== false) {
          step.display = [
            ...new Set(
              readArray(
                display,
                `${path}.display`,
                WORKFLOW_LIMITS.maximumTextCharacters,
                'maximum display outputs'
              ).map((output, index) => readOutputName(output, `${path}.display[${index}]`))
            )
          ];
        }
      }
      return step;
    })
  );
  const document = parseWorkflowDocument(
    serializeWorkflowDocument(
      layoutWorkflow({
        schemaVersion: WORKFLOW_SCHEMA_VERSION,
        id: makeUniqueWorkflowId(title, options.takenIds ?? []),
        title,
        ...(description === undefined ? {} : {description}),
        inputs: [],
        steps,
        settings: {engine: 'auto', autoRun: false}
      })
    )
  );
  const plan = planWorkflow(document, {registry: options.registry});
  const getIssueKey = (issue: WorkflowIssue): string =>
    JSON.stringify([issue.code, issue.stepId, issue.port]);
  const replacedIssueKeys = new Set(extraIssues.map(getIssueKey));
  const issues = [
    ...plan.issues,
    ...plan.order.flatMap(stepId => plan.steps[stepId].issues)
  ].filter(issue => !replacedIssueKeys.has(getIssueKey(issue)));
  issues.push(...extraIssues);
  const severityOrder = {error: 0, warning: 1, info: 2};
  issues.sort((first, second) => severityOrder[first.severity] - severityOrder[second.severity]);
  const outline = describeWorkflowDraft(document, options.registry, plan.order);
  return {document, issues, outline};
}

// ---------------------------------------------------------------------------------------------
// Saved workflows as tools
// ---------------------------------------------------------------------------------------------

function requireSavedWorkflow(
  documents: readonly WorkflowDocument[],
  value: unknown
): WorkflowDocument {
  const workflowId = readString(value, 'arguments.workflowId');
  const document = documents.find(candidate => candidate.id === workflowId);
  if (!document) {
    throw new WorkflowError(
      'malformed',
      `Workflow "${workflowId}" is not available. Available workflow ids: ${documents.map(candidate => candidate.id).join(', ') || '(none)'}.`
    );
  }
  return document;
}

function describeWorkflowInput(input: WorkflowInput): KeplerJSONValue {
  return {
    id: input.id,
    label: input.label,
    kind: input.kind,
    value: Array.isArray(input.value) ? [...input.value] : input.value,
    ...(input.unit === undefined ? {} : {unit: input.unit}),
    ...(input.minimum === undefined ? {} : {minimum: input.minimum}),
    ...(input.maximum === undefined ? {} : {maximum: input.maximum}),
    ...(input.options === undefined ? {} : {options: [...input.options]})
  };
}

function readWorkflowInputValue(value: unknown, path: string): CatalogParameterValue {
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
    return value;
  }
  // Workflow input lists do not share static step parameters' 64-item limit. Their only
  // storage budget is the document limit; the editor validates their numbers and ranges.
  return readArray(
    value,
    path,
    WORKFLOW_LIMITS.maximumDocumentCharacters,
    'maximumDocumentCharacters'
  ).map((number, index) => {
    if (typeof number !== 'number') rejectArgument(`${path}[${index}]`, 'must be a number');
    return number;
  });
}

/** Result of `listWorkflows`: id, title, description, step count and workflow inputs of each. */
export function listWorkflowsForAssistant(
  documents: readonly WorkflowDocument[],
  _registry: WorkflowOperationRegistry
): KeplerJSONValue {
  return {
    workflows: documents.map(document => ({
      id: document.id,
      title: document.title,
      ...(document.description === undefined ? {} : {description: document.description}),
      stepCount: document.steps.length,
      inputs: document.inputs.map(describeWorkflowInput)
    }))
  };
}

/**
 * Result of `describeWorkflow`: the outline, the workflow inputs with ranges, and, when a runner
 * state is given, each step's status, row counts and first issue.
 */
export function describeWorkflowForAssistant(
  documents: readonly WorkflowDocument[],
  registry: WorkflowOperationRegistry,
  args: unknown,
  state?: WorkflowRunnerState | null
): KeplerJSONValue {
  const fields = readObject(args, 'arguments', ['workflowId']);
  const document = requireSavedWorkflow(documents, fields['workflowId']);
  return {
    id: document.id,
    title: document.title,
    ...(document.description === undefined ? {} : {description: document.description}),
    outline: describeWorkflowDraft(document, registry, planWorkflow(document, {registry}).order),
    inputs: document.inputs.map(describeWorkflowInput),
    ...(state
      ? {
          steps: document.steps.map(step => {
            const stepState = Object.hasOwn(state.steps, step.id)
              ? state.steps[step.id]
              : undefined;
            return {
              id: step.id,
              label: step.label ?? registry.get(step.operationId)?.title ?? step.operationId,
              operationId: step.operationId,
              status: stepState?.status ?? null,
              rows: Object.fromEntries(
                Object.entries(stepState?.result?.outputs ?? {}).map(([output, data]) => [
                  output,
                  data.table.rowCount
                ])
              ),
              firstIssue: (stepState?.error ?? stepState?.issues[0])?.message ?? null
            };
          })
        }
      : {})
  };
}

export type WorkflowInputChange = {
  inputId: string;
  label: string;
  previous: KeplerJSONValue;
  next: KeplerJSONValue;
};

/**
 * Reads the arguments of `setWorkflowInputs` (`{workflowId, inputs: {[inputId or label]: value}}`)
 * and returns the edited document without applying it. Unknown workflows, unknown inputs and
 * values outside an input's kind, range or options throw a WorkflowError naming the input.
 */
export function parseWorkflowInputChanges(
  documents: readonly WorkflowDocument[],
  args: unknown
): {workflowId: string; document: WorkflowDocument; changes: WorkflowInputChange[]} {
  const fields = readObject(args, 'arguments', ['workflowId', 'inputs']);
  const original = requireSavedWorkflow(documents, fields['workflowId']);
  // Input ids and labels may legitimately be object-prototype names. Only own data fields
  // participate, and resolution uses a Map rather than assigning those names to an object.
  const inputs = readObject(fields['inputs'], 'arguments.inputs', undefined, true);
  const requested = new Map<string, CatalogParameterValue>();
  let document = original;
  for (const [key, rawValue] of Object.entries(inputs)) {
    const matchingId = original.inputs.find(input => input.id === key);
    const labels = matchingId
      ? []
      : original.inputs.filter(input => input.label.toLowerCase() === key.toLowerCase());
    if (labels.length > 1) {
      throw new WorkflowError(
        'malformed',
        `${original.title}: Input label "${key}" is ambiguous between ${labels.map(input => `"${input.id}"`).join(', ')}; use an input id.`
      );
    }
    const input = matchingId ?? labels[0];
    if (!input) {
      throw new WorkflowError(
        'unknown-workflow-input',
        `${original.title}: Input "${key}" is not available. Available input ids: ${original.inputs.map(candidate => candidate.id).join(', ') || '(none)'}.`
      );
    }
    const next = readWorkflowInputValue(
      rawValue,
      `${original.title}: Input "${input.label}" (${input.id})`
    );
    if (
      requested.has(input.id) &&
      JSON.stringify(requested.get(input.id)) !== JSON.stringify(next)
    ) {
      throw new WorkflowError(
        'malformed',
        `${original.title}: Input "${input.label}" (${input.id}) has conflicting values; supply it once by input id.`
      );
    }
    requested.set(input.id, next);
    document = updateWorkflowInput(document, input.id, {value: next});
  }
  const changes: WorkflowInputChange[] = original.inputs.flatMap(input => {
    const next = document.inputs.find(candidate => candidate.id === input.id)!.value;
    return JSON.stringify(input.value) === JSON.stringify(next)
      ? []
      : [
          {
            inputId: input.id,
            label: input.label,
            previous: Array.isArray(input.value) ? [...input.value] : input.value,
            next: Array.isArray(next) ? [...next] : next
          }
        ];
  });
  return {workflowId: original.id, document, changes};
}

/** Reads the arguments of `runWorkflow` (`{workflowId, target?}`); throws on an unknown workflow or step. */
export function parseWorkflowRunRequest(
  documents: readonly WorkflowDocument[],
  args: unknown
): {workflowId: string; target?: string} {
  const fields = readObject(args, 'arguments', ['workflowId', 'target']);
  const document = requireSavedWorkflow(documents, fields['workflowId']);
  if (!Object.hasOwn(fields, 'target')) return {workflowId: document.id};
  const target = readString(fields['target'], 'arguments.target');
  if (!document.steps.some(step => step.id === target)) {
    throw new WorkflowError(
      'unknown-step',
      `${document.title}: Target step "${target}" is not available. Available step ids: ${document.steps.map(step => step.id).join(', ') || '(none)'}.`,
      {stepId: target}
    );
  }
  return {workflowId: document.id, target};
}

/** Curated metadata only; discovery does not read or transmit dataset rows. */
export function listWorkflowTemplatesForAssistant(args: unknown): KeplerJSONValue {
  const fields = readObject(args, 'arguments', ['query']);
  const query = Object.hasOwn(fields, 'query')
    ? readString(fields['query'], 'arguments.query').trim().toLowerCase()
    : '';
  const matches = KEPLER_CURATED_ANALYSES.filter(analysis =>
    [
      analysis.id,
      analysis.title,
      analysis.name,
      analysis.question,
      analysis.method,
      ...analysis.slots.map(slot => slot.kind)
    ]
      .join(' ')
      .toLowerCase()
      .includes(query)
  );
  return {
    total: matches.length,
    templates: matches.map(analysis => ({
      id: analysis.id,
      title: analysis.title,
      question: analysis.question,
      kinds: analysis.slots.map(slot => slot.kind),
      status: analysis.execution.status,
      ...(analysis.execution.status === 'needs-binding'
        ? {requirement: analysis.execution.reason}
        : {})
    }))
  };
}

function requireAssistantTemplate(templateId: string) {
  const analysis = getKeplerCuratedAnalysis(templateId);
  if (!analysis)
    throw new WorkflowError(
      'unknown-operation',
      `Analysis template "${templateId}" is not available. Use listAnalysisTemplates to choose a supported method.`
    );
  return analysis;
}

export function describeWorkflowTemplateForAssistant(
  registry: WorkflowOperationRegistry,
  args: unknown
): KeplerJSONValue {
  const fields = readObject(args, 'arguments', ['templateId']);
  const analysis = requireAssistantTemplate(
    readString(fields['templateId'], 'arguments.templateId')
  );
  const template = createCuratedWorkflowAnalysisTemplate(analysis);
  return {
    id: analysis.id,
    title: analysis.title,
    question: analysis.question,
    method: analysis.method,
    slots: JSON.parse(JSON.stringify(template.slots)) as KeplerJSONValue,
    exposed: JSON.parse(JSON.stringify(template.exposed)) as KeplerJSONValue,
    reads: JSON.parse(JSON.stringify(template.reads)) as KeplerJSONValue,
    status: analysis.execution.status,
    ...(analysis.execution.status === 'available'
      ? {
          operation: describeWorkflowOperationForAssistant(registry, {
            operationId: analysis.execution.bindingId
          })
        }
      : {requirement: analysis.execution.reason})
  };
}

/** Uses the same mapper as the gallery, including explicit-choice type and measurement checks. */
export function parseWorkflowTemplateForAssistant(
  args: unknown,
  profiles: readonly KeplerDatasetProfile[]
) {
  const fields = readObject(args, 'arguments', ['templateId', 'datasets', 'columns', 'parameters']);
  const analysis = requireAssistantTemplate(
    readString(fields['templateId'], 'arguments.templateId')
  );
  if (analysis.execution.status !== 'available')
    throw new WorkflowError('unknown-operation', `${analysis.title}: ${analysis.execution.reason}`);
  const datasets = readObject(
    fields['datasets'],
    'arguments.datasets',
    analysis.slots.map(slot => slot.name)
  );
  const columns = readObject(
    fields['columns'],
    'arguments.columns',
    analysis.slots.flatMap(slot => slot.fields.map(field => `${slot.name}.${field.role}`))
  );
  const parameters = Object.hasOwn(fields, 'parameters')
    ? readObject(fields['parameters'], 'arguments.parameters')
    : {};
  const values = Object.fromEntries(
    Object.entries(parameters).map(([name, value]) => [
      name,
      readParameterValue(value, `arguments.parameters.${name}`)
    ])
  );
  let mapping = createKeplerCuratedAnalysisMapping(analysis, profiles, {parameters: values});
  for (const slot of analysis.slots) {
    const datasetId = readString(datasets[slot.name], `arguments.datasets.${slot.name}`);
    mapping = updateKeplerCuratedAnalysisMapping(
      mapping,
      {slot: slot.name, datasetId},
      analysis,
      profiles
    );
    for (const field of slot.fields) {
      const key = `${slot.name}.${field.role}`;
      if (!Object.hasOwn(columns, key)) continue;
      const names = readArray(
        columns[key],
        `arguments.columns.${key}`,
        64,
        'maximum mapped columns'
      ).map((name, index) => readString(name, `arguments.columns.${key}[${index}]`));
      mapping = updateKeplerCuratedAnalysisMapping(
        mapping,
        {slot: slot.name, role: field.role, columns: names},
        analysis,
        profiles
      );
    }
  }
  if (!mapping.ready)
    throw new WorkflowError(
      'malformed',
      `${analysis.title}: ${mapping.problems
        .filter(problem => problem.severity === 'blocks' || !problem.acknowledged)
        .map(problem => problem.message)
        .join(' ')}`
    );
  return {template: createCuratedWorkflowAnalysisTemplate(analysis), mapping};
}

/** Recheck the exact document: no remapping, default replacement or execution during validation. */
export function validateWorkflowForAssistant(
  document: WorkflowDocument,
  options: {registry: WorkflowOperationRegistry; host?: WorkflowHost; target?: string}
): WorkflowIssue[] {
  const included = options.target
    ? new Set([options.target, ...getWorkflowUpstreamSteps(document, options.target)])
    : new Set(document.steps.map(step => step.id));
  // Keep full-document graph/parameter checks, but do not evaluate unrelated AX applicability
  // through this structural view of the same operations.
  const structuralOperations = new Map(
    options.registry
      .list()
      .map(operation => [operation.id, {...operation, analysisDescriptor: undefined}])
  );
  const structuralRegistry: WorkflowOperationRegistry = {
    list: () => [...structuralOperations.values()],
    get: operationId => structuralOperations.get(operationId)
  };
  const structure = planWorkflow(document, {registry: structuralRegistry});
  const plan = planWorkflow(
    {...document, steps: document.steps.filter(step => included.has(step.id))},
    options
  );
  const issueKeys = new Set<string>();
  const issues = [
    ...structure.issues,
    ...structure.order.flatMap(stepId => structure.steps[stepId].issues),
    ...plan.issues,
    ...plan.order.flatMap(stepId => plan.steps[stepId].issues)
  ].filter(issue => {
    const key = JSON.stringify(issue);
    if (issueKeys.has(key)) return false;
    issueKeys.add(key);
    return true;
  });
  if (options.target && !document.steps.some(step => step.id === options.target))
    issues.push({
      code: 'unknown-step',
      severity: 'error',
      message: `${document.title}: target step "${options.target}" does not exist.`
    });
  try {
    const template = createWorkflowAnalysisTemplateFromDocument({
      id: document.id,
      title: document.title,
      summary: document.description || document.title,
      document
    });
    const validation = validateWorkflowAnalysisTemplate(template, {registry: structuralRegistry});
    for (const issue of validation.issues) {
      if (
        !issues.some(existing => existing.code === issue.code && existing.message === issue.message)
      )
        issues.push({code: 'malformed', severity: 'error', message: issue.message});
    }
  } catch (error) {
    issues.push({
      code: 'malformed',
      severity: 'error',
      message: error instanceof Error ? error.message : String(error)
    });
  }
  for (const step of document.steps) {
    if (!included.has(step.id)) continue;
    const operation = options.registry.get(step.operationId);
    if (!operation?.analysisDescriptor) continue;
    // A coordinate warning cannot become a declaration just because resolved defaults are
    // frozen. Reuse the checker's text and ask for the missing unit confirmation before review.
    for (const requirement of plan.steps[step.id]?.analysis?.requirements ?? []) {
      if (
        ['declared-coordinates', 'planar-coordinates', 'geographic-coordinates'].includes(
          requirement.id
        ) &&
        requirement.state === 'unknown'
      )
        issues.push({
          code: 'analysis-requirement',
          severity: 'error',
          stepId: step.id,
          message: `${step.label ?? operation.title}: ${requirement.message}. ${requirement.remedy ?? 'Confirm the coordinate system and units before approving this plan.'}`
        });
    }
    const profiles = getWorkflowAnalysisProfiles(document, step, options.registry, options.host);
    for (const input of operation.inputs) {
      const connection = step.inputs[input.name];
      const source =
        connection && document.steps.find(candidate => candidate.id === connection.stepId);
      // Graph kinds cannot establish an intermediate's measurement or statistical requirements.
      // Refuse unknown applicability without running upstream work during proposal/review.
      if (source && !profiles[input.name])
        issues.push({
          code: 'analysis-requirement',
          severity: 'error',
          stepId: step.id,
          message: `${step.label ?? operation.title}: ${input.label} needs a current profile for the exact input geometry and population. Refresh the dataset profile before approving or running this plan; for an intermediate, inspect its result and use that profiled dataset as the source.`
        });
    }
  }
  return issues;
}

/** Freeze resolved settings into the review document so changed profiles cannot change defaults. */
export function prepareWorkflowDraftForAssistant(
  document: WorkflowDocument,
  options: {registry: WorkflowOperationRegistry; host?: WorkflowHost}
): WorkflowDraft {
  const initialIssues = validateWorkflowForAssistant(document, options);
  if (initialIssues.some(issue => issue.severity === 'error'))
    throw new WorkflowError(
      'malformed',
      initialIssues
        .filter(issue => issue.severity === 'error')
        .map(issue => issue.message)
        .join('\n')
    );
  const plan = planWorkflow(document, options);
  const prepared = parseWorkflowDocument(
    serializeWorkflowDocument({
      ...document,
      settings: {...document.settings, autoRun: false},
      steps: document.steps.map(step => ({
        ...step,
        parameters: plan.steps[step.id]?.parameters
          ? Object.fromEntries(
              Object.entries(plan.steps[step.id].parameters!).map(([name, value]) => [
                name,
                {from: 'static' as const, value}
              ])
            )
          : step.parameters
      }))
    })
  );
  const issues = validateWorkflowForAssistant(prepared, options);
  if (issues.some(issue => issue.severity === 'error'))
    throw new WorkflowError(
      'malformed',
      issues
        .filter(issue => issue.severity === 'error')
        .map(issue => issue.message)
        .join('\n')
    );
  return {
    document: prepared,
    issues,
    outline: describeWorkflowDraft(prepared, options.registry, plan.order)
  };
}

/** Warnings shown after review need a new acknowledgement; unchanged warnings remain advisory. */
export function getNewWorkflowAssistantWarnings(
  current: readonly WorkflowIssue[],
  reviewed: readonly WorkflowIssue[]
): WorkflowIssue[] {
  const getKey = (issue: WorkflowIssue) =>
    JSON.stringify([issue.code, issue.stepId, issue.port, issue.parameter, issue.message]);
  const known = new Set(reviewed.filter(issue => issue.severity === 'warning').map(getKey));
  return current.filter(issue => issue.severity === 'warning' && !known.has(getKey(issue)));
}
