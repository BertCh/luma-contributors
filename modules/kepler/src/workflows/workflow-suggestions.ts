// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CatalogParameterValue} from '../analysis/catalog-binding-types';
import {
  addWorkflowStep,
  connectWorkflowSteps,
  disconnectWorkflowInput,
  moveWorkflowSteps,
  removeWorkflowSteps,
  setWorkflowParameter,
  updateWorkflowStep
} from './workflow-edit';
import {getWorkflowPortCompatibility} from './workflow-operations';
import {getWorkflowDownstreamSteps, planWorkflow, resolveWorkflowParameters} from './workflow-plan';
import {
  WorkflowError,
  type WorkflowConnection,
  type WorkflowData,
  type WorkflowDocument,
  type WorkflowIssue,
  type WorkflowOperationRegistry,
  type WorkflowRunnerState,
  type WorkflowStep
} from './workflow-types';

/** A one-click repair for an issue. `apply` is an ordinary edit: it returns a new document. */
export type WorkflowSuggestion = {
  /** Stable within one call; `<kind>:<stepId>[:<detail>]`. */
  id: string;
  /** Button text: a verb phrase, at most 48 characters. */
  label: string;
  /** One sentence saying what will change. */
  description: string;
  apply(document: WorkflowDocument, registry: WorkflowOperationRegistry): WorkflowDocument;
};

type Edit = WorkflowSuggestion['apply'];

function truncateLabel(value: string, maximum: number): string {
  return value.length <= maximum ? value : `${value.slice(0, maximum - 1)}…`;
}

function formatLabel(prefix: string, value: string, suffix = ''): string {
  return `${prefix}${truncateLabel(value, 48 - prefix.length - suffix.length)}${suffix}`;
}

function getStepLabel(step: WorkflowStep, registry: WorkflowOperationRegistry): string {
  return step.label ?? registry.get(step.operationId)?.title ?? step.operationId;
}

function getFeederPosition(document: WorkflowDocument, consumer: WorkflowStep) {
  const feederIds = new Set(Object.values(consumer.inputs).map(connection => connection.stepId));
  const feeders = document.steps.filter(step => feederIds.has(step.id));
  const lowest = feeders.length
    ? Math.max(...feeders.map(step => step.position.y))
    : consumer.position.y;
  return {x: consumer.position.x - 280, y: lowest + 150};
}

function getCurrentOutput(state: WorkflowRunnerState | null, connection: WorkflowConnection) {
  const step = state?.steps[connection.stepId];
  return step?.status === 'done' && step.result?.key === step.key
    ? step.result?.outputs[connection.output]
    : undefined;
}

function getJoinParameters(
  target: WorkflowData,
  attributes: WorkflowData,
  targetIdentities: readonly string[],
  attributeIdentities: readonly string[]
): Record<string, CatalogParameterValue> | undefined {
  const targetKey = [target.identityColumn, ...targetIdentities].find(
    name =>
      name !== undefined &&
      name !== target.table.geometryColumn &&
      Object.hasOwn(target.table.columns, name)
  );
  if (!targetKey) return undefined;
  // Shared identity survives table operations such as Summarise; source_id is the catalog's
  // declared reference to an input feature, rather than the output row's own identity.
  const attributesKey = [
    ...(attributes.identityColumn === 'source_id' || attributeIdentities.includes('source_id')
      ? ['source_id']
      : []),
    ...(attributes.identityColumn === targetKey || attributeIdentities.includes(targetKey)
      ? [targetKey]
      : [])
  ].find(name => Object.hasOwn(attributes.table.columns, name));
  if (!attributesKey) return undefined;
  const columns = Object.keys(attributes.table.columns).filter(
    name => name !== attributesKey && name !== attributes.table.geometryColumn
  );
  let prefix = '';
  let suffix = 1;
  while (columns.some(name => Object.hasOwn(target.table.columns, `${prefix}${name}`))) {
    prefix = suffix === 1 ? 'attributes_' : `attributes${suffix}_`;
    suffix++;
  }
  // An empty columns parameter deliberately selects all non-key, non-geometry attributes,
  // including names which cannot be represented by the comma-separated column picker.
  return {targetKey, attributesKey, columns: '', prefix, unmatched: 'keep'};
}

function getErrorKey(issue: WorkflowIssue): string {
  return JSON.stringify([issue.stepId, issue.code, issue.port, issue.parameter, issue.message]);
}

function getPlanErrors(document: WorkflowDocument, registry: WorkflowOperationRegistry) {
  const plan = planWorkflow(document, {registry});
  return [...plan.issues, ...Object.values(plan.steps).flatMap(step => step.issues)].filter(
    issue => issue.severity === 'error'
  );
}

/** Revalidate on application too: the canvas or registry may have changed since rendering. */
function applySafeEdit(
  document: WorkflowDocument,
  registry: WorkflowOperationRegistry,
  edit: Edit
): WorkflowDocument {
  const previousErrors = new Set(getPlanErrors(document, registry).map(getErrorKey));
  const result = edit(document, registry);
  const introduced = getPlanErrors(result, registry).find(
    issue => !previousErrors.has(getErrorKey(issue))
  );
  if (introduced) {
    throw new WorkflowError(introduced.code, introduced.message, introduced);
  }
  return result;
}

function getEditDistance(first: string, second: string): number {
  if (Math.abs(first.length - second.length) > 3) return 4;
  let previous = Array.from({length: second.length + 1}, (_, index) => index);
  for (let firstIndex = 0; firstIndex < first.length; firstIndex++) {
    const current = [firstIndex + 1];
    for (let secondIndex = 0; secondIndex < second.length; secondIndex++) {
      current.push(
        Math.min(
          previous[secondIndex + 1] + 1,
          current[secondIndex] + 1,
          previous[secondIndex] + (first[firstIndex] === second[secondIndex] ? 0 : 1)
        )
      );
    }
    if (current.every(distance => distance > 3)) return 4;
    previous = current;
  }
  return previous[second.length];
}

function compareNames(first: string, second: string): number {
  return first < second ? -1 : first > second ? 1 : 0;
}

/**
 * Repairs for one issue of one step, best first, at most three. Empty when nothing safe can be
 * offered. Pure: reads the document, the registry and the runner state, never runs anything.
 */
export function getWorkflowSuggestions(
  document: WorkflowDocument,
  registry: WorkflowOperationRegistry,
  state: WorkflowRunnerState | null,
  issue: WorkflowIssue
): WorkflowSuggestion[] {
  const step = document.steps.find(candidate => candidate.id === issue.stepId);
  if (!step) return [];
  const operation = registry.get(step.operationId);
  const suggestions: WorkflowSuggestion[] = [];
  function offer(kind: string, label: string, description: string, edit: Edit, detail?: string) {
    const id = `${kind}:${step!.id}${detail === undefined ? '' : `:${detail}`}`;
    if (suggestions.some(suggestion => suggestion.id === id)) return;
    const apply: Edit = (current, currentRegistry) => applySafeEdit(current, currentRegistry, edit);
    try {
      if (apply(document, registry) === document) return;
    } catch (error) {
      // Expected edit refusals (cycles, limits, invalid defaults) make a repair unavailable.
      if (error instanceof WorkflowError) return;
      throw error;
    }
    suggestions.push({
      id,
      label: truncateLabel(label, 48),
      description,
      apply
    });
  }
  const input = operation?.inputs.find(port => port.name === issue.port);
  const connection = input ? step.inputs[input.name] : undefined;
  const producer = document.steps.find(candidate => candidate.id === connection?.stepId);
  const producerOperation = producer && registry.get(producer.operationId);
  const producerOutput = producerOperation?.outputs.find(port => port.name === connection?.output);
  const producerData =
    producer && connection
      ? state?.steps[producer.id]?.result?.outputs[connection.output]
      : undefined;
  const parameter = operation?.parameters.find(candidate => candidate.name === issue.parameter);

  function addInputStep(
    operationId: string,
    parameters: Record<string, CatalogParameterValue>,
    inputs: Record<string, WorkflowConnection>,
    placement: 'between' | 'feeder' = 'between'
  ): Edit {
    return (current, currentRegistry) => {
      const currentTarget = current.steps.find(candidate => candidate.id === step!.id);
      const currentProducer = current.steps.find(candidate => candidate.id === producer!.id);
      if (!currentTarget || !currentProducer || !input)
        throw new WorkflowError(
          'unknown-step',
          `${getStepLabel(step!, currentRegistry)}: The connected step no longer exists.`,
          {stepId: step!.id}
        );
      let positioned = current;
      if (placement === 'between') {
        const movedIds = new Set([
          currentTarget.id,
          ...getWorkflowDownstreamSteps(current, currentTarget.id)
        ]);
        positioned = moveWorkflowSteps(
          current,
          Object.fromEntries(
            current.steps
              .filter(candidate => movedIds.has(candidate.id))
              .map(candidate => [
                candidate.id,
                {x: candidate.position.x + 280, y: candidate.position.y}
              ])
          )
        );
      }
      const added = addWorkflowStep(positioned, currentRegistry, {
        operationId,
        parameters,
        position:
          placement === 'between'
            ? currentTarget.position
            : getFeederPosition(current, currentTarget)
      });
      let result = added.document;
      for (const [name, from] of Object.entries(inputs)) {
        result = connectWorkflowSteps(result, currentRegistry, {
          from,
          to: {stepId: added.stepId, input: name}
        });
      }
      const output = currentRegistry.get(operationId)?.outputs[0];
      if (!output)
        throw new WorkflowError(
          'unknown-output-port',
          `${getStepLabel(step!, currentRegistry)}: Repair operation requires an output.`,
          {stepId: step!.id}
        );
      return connectWorkflowSteps(result, currentRegistry, {
        from: {stepId: added.stepId, output: output.name},
        to: {stepId: step!.id, input: input.name}
      });
    };
  }

  switch (issue.code) {
    case 'capacity-exceeded':
    case 'capacity-may-exceed': {
      if (!input || !connection || !producer || !producerOutput) break;
      const identity = producerOutput.identityColumns[0];
      const column =
        identity && (!producerData || Object.hasOwn(producerData.table.columns, identity))
          ? identity
          : Object.keys(producerData?.table.columns ?? {})[0];
      if (column && input.maximumRows > 0) {
        offer(
          'insert-sort',
          'Insert Sort rows with a limit',
          `Sort the producer's rows and keep at most ${input.maximumRows} rows.`,
          addInputStep('table-sort-rows', {column, limit: input.maximumRows}, {rows: connection}),
          input.name
        );
      }
      offer(
        'insert-filter',
        'Insert Filter rows',
        'Insert a filter with expression true; edit the expression to reduce rows.',
        addInputStep('table-filter-rows', {expression: 'true'}, {rows: connection}),
        input.name
      );
      break;
    }
    case 'geometry-required': {
      if (!input || !connection || !producer || !producerOperation || !producerOutput) break;
      const attributes = getCurrentOutput(state, connection);
      if (!attributes) break;
      const originalPlan = planWorkflow(document, {registry});
      if (!originalPlan.steps[producer.id]?.key) break;
      for (const geometryInput of producerOperation.inputs) {
        if (!geometryInput.accepts.some(kind => kind !== 'table')) continue;
        const upstreamConnection = producer.inputs[geometryInput.name];
        const upstream = document.steps.find(
          candidate => candidate.id === upstreamConnection?.stepId
        );
        const output =
          upstream &&
          registry
            .get(upstream.operationId)
            ?.outputs.find(candidate => candidate.name === upstreamConnection.output);
        if (!upstream || !output) continue;
        const target = getCurrentOutput(state, upstreamConnection);
        if (
          !target ||
          target.kind === 'table' ||
          !target.table.geometryColumn ||
          target.table.rowCount > input.maximumRows ||
          getWorkflowPortCompatibility(target.kind, input) !== 'yes'
        )
          continue;
        const parameters = getJoinParameters(
          target,
          attributes,
          output.identityColumns,
          producerOutput.identityColumns
        );
        if (!parameters) continue;
        const edit = addInputStep(
          'table-join-attributes',
          parameters,
          {target: upstreamConnection, attributes: connection},
          'feeder'
        );
        offer(
          'join-back',
          formatLabel('Join back to ', getStepLabel(upstream, registry)),
          'Restore geometry by joining the producer attributes to its geometry input.',
          (current, currentRegistry) => {
            const currentPlan = planWorkflow(current, {registry: currentRegistry});
            if (
              [producer.id, upstream.id].some(
                identifier =>
                  currentPlan.steps[identifier]?.key !== originalPlan.steps[identifier]?.key
              )
            )
              throw new WorkflowError(
                'upstream-not-configured',
                `${getStepLabel(step, currentRegistry)}: ${input.label} join keys require current upstream results; rerun upstream steps before repairing geometry.`,
                {stepId: step.id, port: input.name}
              );
            return edit(current, currentRegistry);
          },
          upstream.id
        );
        if (suggestions.length === 3) break;
      }
      break;
    }
    case 'input-required': {
      if (!input) break;
      const candidates = document.steps
        .filter(candidate => candidate.id !== step.id && !candidate.disabled)
        .sort(
          (first, second) =>
            Math.hypot(first.position.x - step.position.x, first.position.y - step.position.y) -
              Math.hypot(
                second.position.x - step.position.x,
                second.position.y - step.position.y
              ) || compareNames(first.id, second.id)
        );
      for (const candidate of candidates) {
        for (const output of registry.get(candidate.operationId)?.outputs ?? []) {
          const kind =
            state?.steps[candidate.id]?.result?.outputs[output.name]?.kind ?? output.kind;
          if (getWorkflowPortCompatibility(kind, input) !== 'yes') continue;
          const outputLabel = truncateLabel(output.label, 20);
          offer(
            'connect',
            formatLabel('Connect ', getStepLabel(candidate, registry), ` · ${outputLabel}`),
            `Connect ${getStepLabel(candidate, registry)} to ${input.label}.`,
            (current, currentRegistry) =>
              connectWorkflowSteps(current, currentRegistry, {
                from: {stepId: candidate.id, output: output.name},
                to: {stepId: step.id, input: input.name}
              }),
            `${candidate.id}:${output.name}`
          );
          if (suggestions.length === 2) break;
        }
        if (suggestions.length === 2) break;
      }
      if (!suggestions.length) {
        offer(
          'add-source',
          'Add a GeoJSON source',
          'Add a GeoJSON source connected to this input.',
          (current, currentRegistry) => {
            const consumer = current.steps.find(candidate => candidate.id === step.id);
            if (!consumer)
              throw new WorkflowError(
                'unknown-step',
                `${getStepLabel(step, currentRegistry)}: The step no longer exists.`,
                {stepId: step.id}
              );
            const added = addWorkflowStep(current, currentRegistry, {
              operationId: 'source-geojson',
              position: getFeederPosition(current, consumer)
            });
            const output = currentRegistry.get('source-geojson')?.outputs[0];
            if (!output)
              throw new WorkflowError(
                'unknown-output-port',
                `${getStepLabel(step, currentRegistry)}: GeoJSON source requires an output.`,
                {stepId: step.id}
              );
            return connectWorkflowSteps(added.document, currentRegistry, {
              from: {stepId: added.stepId, output: output.name},
              to: {stepId: step.id, input: input.name}
            });
          },
          input.name
        );
      }
      break;
    }
    case 'incompatible-port':
      if (input && connection)
        offer(
          'disconnect',
          formatLabel('Disconnect "', input.label, '"'),
          'Disconnect the incompatible output from this input.',
          current => disconnectWorkflowInput(current, step.id, input.name),
          input.name
        );
      break;
    case 'unknown-workflow-input':
      if (parameter)
        offer(
          'reset-default',
          formatLabel('Reset "', parameter.label, '" to its default'),
          'Restore the operation default instead of the missing workflow input.',
          current => setWorkflowParameter(current, step.id, parameter.name, undefined),
          parameter.name
        );
      break;
    case 'parameter-out-of-range': {
      if (!parameter || !operation) break;
      const binding = step.parameters[parameter.name];
      const currentValue = resolveWorkflowParameters(document, step, operation)[parameter.name];
      if (
        (parameter.kind === 'number' || parameter.kind === 'integer') &&
        binding?.from === 'static' &&
        typeof binding.value === 'number'
      ) {
        const value = binding.value;
        const bound =
          parameter.minimum !== undefined && value < parameter.minimum
            ? parameter.minimum
            : parameter.maximum !== undefined && value > parameter.maximum
              ? parameter.maximum
              : undefined;
        if (bound !== undefined && Number.isFinite(bound))
          offer(
            'set-bound',
            formatLabel('Set "', parameter.label, `" to ${bound}`),
            'Set the parameter to the nearest allowed bound.',
            current =>
              setWorkflowParameter(current, step.id, parameter.name, {
                from: 'static',
                value: bound
              }),
            parameter.name
          );
      } else if (
        parameter.kind === 'column' &&
        typeof currentValue === 'string' &&
        /(?:does not exist|does not have|not found|unknown column|missing column|no (?:such )?column)/i.test(
          issue.message
        )
      ) {
        const names = [
          ...new Set(
            Object.entries(step.inputs)
              .filter(([name]) => !parameter.columnsFrom || parameter.columnsFrom === name)
              .flatMap(([, from]) =>
                Object.keys(
                  state?.steps[from.stepId]?.result?.outputs[from.output]?.table.columns ?? {}
                )
              )
          )
        ];
        const currentName = currentValue.toLowerCase();
        const matches = names
          .map(name => ({name, distance: getEditDistance(currentName, name.toLowerCase())}))
          .filter(candidate => candidate.name !== currentValue && candidate.distance <= 3)
          .sort(
            (first, second) =>
              first.distance - second.distance || compareNames(first.name, second.name)
          );
        for (const match of matches.slice(0, 2))
          offer(
            'use-column',
            formatLabel('Use column "', match.name, '"'),
            'Use an existing upstream column with a similar name.',
            current =>
              setWorkflowParameter(current, step.id, parameter.name, {
                from: 'static',
                value: match.name
              }),
            `${parameter.name}:${match.name}`
          );
      }
      break;
    }
    case 'upstream-disabled': {
      const visited = new Set<string>([step.id]);
      let pending = Object.values(step.inputs).map(from => from.stepId);
      while (pending.length && !suggestions.length) {
        const next: string[] = [];
        for (const identifier of [...new Set(pending)].sort(compareNames)) {
          if (visited.has(identifier)) continue;
          visited.add(identifier);
          const upstream = document.steps.find(candidate => candidate.id === identifier);
          if (!upstream) continue;
          if (upstream.disabled) {
            offer(
              'enable',
              formatLabel('Enable ', getStepLabel(upstream, registry)),
              'Enable the nearest disabled upstream step.',
              current => updateWorkflowStep(current, upstream.id, {disabled: false}),
              upstream.id
            );
            if (suggestions.length) break;
          }
          next.push(...Object.values(upstream.inputs).map(from => from.stepId));
        }
        pending = next;
      }
      break;
    }
    case 'unknown-operation':
      offer(
        'remove',
        'Remove this step',
        'Remove the unavailable step and connections that read it.',
        current => removeWorkflowSteps(current, [step.id])
      );
      break;
    case 'gpu-fallback':
      if (step.onFallback === 'stop')
        offer(
          'continue-cpu',
          'Continue on the CPU engine when the GPU cannot run',
          'Allow CPU fallback with a warning when the GPU cannot run.',
          current => updateWorkflowStep(current, step.id, {onFallback: 'warn'})
        );
      break;
    default:
      break;
  }
  return suggestions.slice(0, 3);
}
