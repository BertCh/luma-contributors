// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  parseWorkflowAnalysisTemplate,
  serializeWorkflowAnalysisTemplate
} from './workflow-analysis-template-parser';
import type {WorkflowAnalysisTemplate} from './workflow-analysis-template';
import type {CatalogParameterValue} from '../analysis/catalog-binding-types';
import {parseCatalogInvocation} from '../analysis/catalog-bindings';
import {
  createWorkflowDocument,
  makeUniqueWorkflowId,
  parseWorkflowDocument,
  serializeWorkflowDocument
} from './workflow-document';
import {layoutWorkflow} from './workflow-edit';
import {getWorkflowTopologicalOrder} from './workflow-plan';
import {
  WORKFLOW_LIMITS,
  WorkflowError,
  type WorkflowDocument,
  type WorkflowInput,
  type WorkflowOperationRegistry,
  type WorkflowStep
} from './workflow-types';

/** Import bounds distinguish document content from pretty-printing whitespace. */
export const WORKFLOW_JSON_LIMITS = {
  // Canonical values nest at most six levels. Two-space indentation and newlines add less
  // than fifteen characters per compact character; the final newline needs one more.
  maximumFormattedCharacters: WORKFLOW_LIMITS.maximumDocumentCharacters * 16 + 1
} as const;

function checkJSONTextLimit(count: number, name: string, maximum: number): void {
  if (count > maximum) {
    throw new WorkflowError(
      'limit-exceeded',
      `Workflow JSON: ${name} is ${maximum}; got ${count}.`
    );
  }
}

function countCompactCharacters(text: string): number {
  let count = 0;
  let quoted = false;
  let escaped = false;
  for (let index = 0; index < text.length; index++) {
    const character = text[index];
    if (quoted) {
      count++;
      if (escaped) escaped = false;
      else if (character === '\\') escaped = true;
      else if (character === '"') quoted = false;
    } else if (character === '"') {
      count++;
      quoted = true;
    } else if (
      character !== ' ' &&
      character !== '\t' &&
      character !== '\r' &&
      character !== '\n'
    ) {
      count++;
    }
  }
  return count;
}

/** Pretty-printed canonical JSON of one document. */
export function exportWorkflowJSON(document: WorkflowDocument): string {
  return `${JSON.stringify(serializeWorkflowDocument(document), null, 2)}\n`;
}

/** Strict import of `exportWorkflowJSON` text; throws a WorkflowError on any deviation. */
export function importWorkflowJSON(text: string): WorkflowDocument {
  checkJSONTextLimit(
    text.length,
    'maximumFormattedCharacters',
    WORKFLOW_JSON_LIMITS.maximumFormattedCharacters
  );
  if (text.length > WORKFLOW_LIMITS.maximumDocumentCharacters) {
    checkJSONTextLimit(
      countCompactCharacters(text),
      'maximumDocumentCharacters',
      WORKFLOW_LIMITS.maximumDocumentCharacters
    );
  }
  let value: unknown;
  try {
    // Count whitespace without removing it: joining tokens could turn malformed JSON into data.
    value = JSON.parse(text);
  } catch (cause) {
    const reason = cause instanceof Error ? cause.message : String(cause);
    throw new WorkflowError('malformed', `Workflow JSON is not valid JSON: ${reason}.`, {cause});
  }
  return parseWorkflowDocument(value);
}

function orderSteps(document: WorkflowDocument): WorkflowStep[] {
  const steps = new Map(document.steps.map(step => [step.id, step]));
  return getWorkflowTopologicalOrder(document).order.map(stepId => steps.get(stepId)!);
}

function sanitizeComment(text: string): string {
  return text.replace(/\*\//g, '* /').replace(/[\r\n\u2028\u2029]/g, ' ');
}

function makePascalName(text: string): string {
  return text
    .split(/[^a-zA-Z0-9]+/)
    .filter(Boolean)
    .map(word => word[0].toUpperCase() + word.slice(1))
    .join('');
}

function makeInputNames(inputs: readonly WorkflowInput[]): Map<string, string> {
  const taken = new Set(Object.getOwnPropertyNames(Object.prototype));
  return new Map(
    inputs.map(input => {
      const pascalName = makePascalName(input.id) || 'Input';
      const stem = /^[0-9]/.test(pascalName)
        ? `input${pascalName}`
        : pascalName[0].toLowerCase() + pascalName.slice(1);
      let name = stem;
      for (let suffix = 2; taken.has(name); suffix++) name = `${stem}${suffix}`;
      taken.add(name);
      return [input.id, name];
    })
  );
}

function getInputType(input: WorkflowInput): string {
  switch (input.kind) {
    case 'number':
    case 'integer':
      return 'number';
    case 'boolean':
      return 'boolean';
    case 'number-list':
      return 'number[]';
    case 'enum':
      return input.options?.map(option => JSON.stringify(option)).join(' | ') || 'string';
    default:
      return 'string';
  }
}

function describeInput(input: WorkflowInput): string {
  return [
    input.label,
    ...(input.unit ? [`Unit: ${input.unit}.`] : []),
    ...(input.minimum === undefined ? [] : [`Minimum: ${input.minimum}.`]),
    ...(input.maximum === undefined ? [] : [`Maximum: ${input.maximum}.`]),
    `Default: ${JSON.stringify(input.value)}.`
  ].join(' ');
}

function renderInput(input: WorkflowInput, name: string): string {
  const property = JSON.stringify(name);
  const fallback = JSON.stringify(input.value);
  // Overrides must be own properties of the caller's input object.
  const value = `Object.hasOwn(inputs, ${property}) ? (inputs[${property}] ?? ${fallback}) : ${fallback}`;
  return `{${Object.entries(input)
    .map(
      ([key, field]) => `${JSON.stringify(key)}: ${key === 'value' ? value : JSON.stringify(field)}`
    )
    .join(', ')}}`;
}

/**
 * TypeScript module text that rebuilds the workflow with `@luma-contributors/kepler`: workflow
 * inputs become function arguments with their current values as defaults, steps appear in
 * topological order, and the function returns a runner ready to run.
 */
export function exportWorkflowTypeScript(
  document: WorkflowDocument,
  registry: WorkflowOperationRegistry
): string {
  const canonical = parseWorkflowDocument(serializeWorkflowDocument(document));
  const inputNames = makeInputNames(canonical.inputs);
  const lines = [
    "import {WorkflowRunner, getWorkflowOperationRegistry, parseWorkflowDocument, type WorkflowHost} from '@luma-contributors/kepler';",
    '',
    `/** ${sanitizeComment(canonical.title)}: ${canonical.steps.length} steps. Generated from the analysis workbench. */`,
    `export function create${makePascalName(canonical.title) || 'Untitled'}Workflow(host: WorkflowHost, inputs: {`
  ];
  for (const input of canonical.inputs) {
    lines.push(
      `  /** ${sanitizeComment(describeInput(input))} */`,
      `  ${JSON.stringify(inputNames.get(input.id))}?: ${getInputType(input)};`
    );
  }
  lines.push('} = {}): WorkflowRunner {', '  const document = parseWorkflowDocument({');
  for (const [key, value] of Object.entries(canonical)) {
    if (key === 'inputs') {
      lines.push('    "inputs": [');
      for (const input of canonical.inputs) {
        lines.push(`      ${renderInput(input, inputNames.get(input.id)!)},`);
      }
      lines.push('    ],');
    } else if (key === 'steps') {
      lines.push('    "steps": [');
      for (const step of orderSteps(canonical)) {
        const operation = registry.get(step.operationId);
        const title = operation?.title ?? `unknown operation ${JSON.stringify(step.operationId)}`;
        lines.push(
          `      // ${sanitizeComment(`${step.label ?? operation?.title ?? step.operationId} — ${title}`)}`,
          ...JSON.stringify(step, null, 2)
            .split('\n')
            .map(
              (line, index, allLines) => `      ${line}${index === allLines.length - 1 ? ',' : ''}`
            )
        );
      }
      lines.push('    ],');
    } else {
      lines.push(`    ${JSON.stringify(key)}: ${JSON.stringify(value)},`);
    }
  }
  lines.push(
    '  });',
    '  const runner = new WorkflowRunner({host, registry: getWorkflowOperationRegistry()});',
    '  runner.setDocument(document);',
    '  return runner;',
    '}',
    ''
  );
  return lines.join('\n');
}

function equalValues(
  left: CatalogParameterValue,
  right: CatalogParameterValue | undefined
): boolean {
  return JSON.stringify(left) === JSON.stringify(right);
}

function formatValue(value: CatalogParameterValue): string {
  return typeof value === 'string' && value.length > 120
    ? `${value.length} characters`
    : JSON.stringify(value);
}

function orderNames(names: string[], declared: readonly {name: string}[]): string[] {
  return [
    ...declared.map(spec => spec.name).filter(name => names.includes(name)),
    ...names.filter(name => !declared.some(spec => spec.name === name)).sort()
  ];
}

/** Plain-text outline of a workflow: steps in order, their inputs and non-default parameters. */
export function describeWorkflow(
  document: WorkflowDocument,
  registry: WorkflowOperationRegistry
): string {
  const lines = [`${document.title}: ${document.steps.length} steps.`];
  for (const input of document.inputs) {
    lines.push(
      `  ${input.label} = ${formatValue(input.value)}${input.unit ? ` ${input.unit}` : ''}`
    );
  }
  const steps = new Map(document.steps.map(step => [step.id, step]));
  function getStepLabel(step: WorkflowStep): string {
    return step.label ?? registry.get(step.operationId)?.title ?? step.operationId;
  }
  orderSteps(document).forEach((step, index) => {
    const operation = registry.get(step.operationId);
    const title = operation?.title ?? `unknown operation ${JSON.stringify(step.operationId)}`;
    lines.push(
      `${index + 1}. ${getStepLabel(step)} — ${title}${step.disabled ? ' [disabled]' : ''}`
    );
    for (const name of orderNames(Object.keys(step.inputs), operation?.inputs ?? [])) {
      const connection = step.inputs[name];
      const producer = steps.get(connection.stepId);
      const output = producer
        ? registry.get(producer.operationId)?.outputs.find(port => port.name === connection.output)
        : undefined;
      const port = operation?.inputs.find(input => input.name === name);
      lines.push(
        `  reads ${port?.label ?? name} from ${producer ? getStepLabel(producer) : connection.stepId} · ${output?.label ?? connection.output}`
      );
    }
    for (const name of orderNames(Object.keys(step.parameters), operation?.parameters ?? [])) {
      const binding = step.parameters[name];
      const parameter = operation?.parameters.find(spec => spec.name === name);
      if (
        binding.from === 'static' &&
        parameter &&
        equalValues(binding.value, parameter.defaultValue)
      )
        continue;
      const input =
        binding.from === 'input'
          ? document.inputs.find(candidate => candidate.id === binding.inputId)
          : undefined;
      const value =
        binding.from === 'input'
          ? `workflow input ${JSON.stringify(input?.label ?? binding.inputId)}`
          : formatValue(binding.value);
      lines.push(`  ${parameter?.label ?? name}: ${value}`);
    }
    if (step.display?.length) lines.push('  shown on the map');
  });
  return `${lines.join('\n')}\n`;
}

/** Where one input of a saved catalog run came from (the host's catalog history record). */
export type WorkflowHistoryInputReference =
  | {mode: 'sample'}
  | {mode: 'text'; text: string; profileDeclarations?: string}
  | {mode: 'native'; datasetId: string; field: string};

export type WorkflowHistoryRecord = {
  id: string;
  operationId: string;
  /** Output of `serializeCatalogInvocation`. */
  invocation: string;
  inputs: Record<string, WorkflowHistoryInputReference>;
  engine: 'auto' | 'cpu';
};

function isWorkflowTableSnapshot(text: string): boolean {
  try {
    const input: unknown = JSON.parse(text);
    return typeof input === 'object' && input !== null && 'type' in input && input.type === 'table';
  } catch {
    return false;
  }
}

/**
 * Promotes saved catalog runs to a workflow: one step per run with its recorded parameters, one
 * source step per distinct input (identical inputs are shared), laid out left to right. Runs are
 * not connected to each other, because a catalog run never read another run's output.
 */
export function makeWorkflowFromCatalogHistory(
  records: readonly WorkflowHistoryRecord[],
  options: {
    registry: WorkflowOperationRegistry;
    /** GeoJSON text of a built-in sample input, when the host has one. */
    getSampleText?(operationId: string, inputName: string): string | undefined;
    title?: string;
    takenIds?: Iterable<string>;
  }
): WorkflowDocument {
  const title = options.title ?? 'Promoted from history';
  let document = createWorkflowDocument({
    title,
    id: makeUniqueWorkflowId(title, options.takenIds ?? [])
  });
  const sources = new Map<string, string>();
  let textCount = 0;
  let recordCount = 0;
  for (const record of records) {
    let parsed: ReturnType<typeof parseCatalogInvocation>;
    try {
      parsed = parseCatalogInvocation(record.invocation);
    } catch {
      // An unsupported or malformed history entry must not prevent restoring the other runs.
      continue;
    }
    const operation = options.registry.get(parsed.binding.id);
    if (!operation) continue;
    const taken = new Set(document.steps.map(step => step.id));
    const pendingSources = new Map(sources);
    const additions: WorkflowStep[] = [];
    let pendingTextCount = textCount;
    const step: WorkflowStep = {
      id: makeUniqueWorkflowId(record.id, taken),
      operationId: operation.id,
      inputs: {},
      parameters: {},
      ...(record.engine === 'cpu' ? {engine: 'cpu' as const} : {}),
      position: {x: 0, y: 0}
    };
    taken.add(step.id);
    for (const parameter of operation.parameters) {
      const value = parsed.invocation.parameters[parameter.name];
      if (value !== undefined) {
        step.parameters[parameter.name] = {from: 'static', value};
      }
    }
    for (const input of operation.inputs) {
      const reference = record.inputs[input.name];
      if (!reference) continue;
      const text =
        reference.mode === 'sample'
          ? options.getSampleText?.(operation.id, input.name)
          : reference.mode === 'text'
            ? reference.text
            : undefined;
      if (reference.mode !== 'native' && text === undefined) continue;
      const key =
        reference.mode === 'native'
          ? JSON.stringify(['native', reference.datasetId, reference.field])
          : JSON.stringify([
              'text',
              text,
              reference.mode === 'text' ? reference.profileDeclarations : undefined
            ]);
      let sourceId = pendingSources.get(key);
      if (!sourceId) {
        const label =
          reference.mode === 'native'
            ? `Dataset ${reference.datasetId}`
            : reference.mode === 'sample'
              ? `Sample: ${input.label}`
              : `${isWorkflowTableSnapshot(text!) ? 'Table snapshot' : 'GeoJSON'} ${++pendingTextCount}`;
        sourceId = makeUniqueWorkflowId(label, taken);
        taken.add(sourceId);
        const parameters: WorkflowStep['parameters'] =
          reference.mode === 'native'
            ? {
                dataset: {from: 'static', value: reference.datasetId},
                geometryMode: {from: 'static', value: reference.field ? 'geojson-field' : 'none'},
                geometryField: {from: 'static', value: reference.field}
              }
            : {
                text: {from: 'static', value: text!},
                ...(reference.mode === 'text' && reference.profileDeclarations
                  ? {
                      profileDeclarations: {
                        from: 'static' as const,
                        value: reference.profileDeclarations
                      }
                    }
                  : {})
              };
        additions.push({
          id: sourceId,
          operationId:
            reference.mode === 'native'
              ? 'source-dataset'
              : isWorkflowTableSnapshot(text!)
                ? 'source-table'
                : 'source-geojson',
          label,
          inputs: {},
          parameters,
          position: {x: 0, y: 0}
        });
        pendingSources.set(key, sourceId);
      }
      step.inputs[input.name] = {stepId: sourceId, output: 'data'};
    }
    if (document.steps.length + additions.length + 1 > WORKFLOW_LIMITS.maximumSteps) break;
    document = {...document, steps: [...document.steps, ...additions, step]};
    for (const [key, sourceId] of pendingSources) sources.set(key, sourceId);
    textCount = pendingTextCount;
    recordCount++;
  }
  if (!recordCount) {
    throw new WorkflowError('malformed', 'Catalog history contains no supported runs to promote.');
  }
  return layoutWorkflow(document);
}

/** Compact export stays within the AX-30 wire budget. */
export function exportWorkflowAnalysisTemplateJSON(template: WorkflowAnalysisTemplate): string {
  return JSON.stringify(serializeWorkflowAnalysisTemplate(template));
}

export function importWorkflowAnalysisTemplateJSON(text: string): WorkflowAnalysisTemplate {
  return parseWorkflowAnalysisTemplate(text);
}
