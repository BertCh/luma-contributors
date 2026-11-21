// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getCatalogParameterRenameMessage,
  resolveCatalogParameterAliases
} from '../analysis/catalog-parameter-constraints';
import {getRemovedPointCloudIcpParameterMessage} from '../analysis/catalog-pointcloud-icp-parameters';
import {parseCatalogFeatureCollection} from '../analysis/catalog-geojson';
import {parseKeplerDatasetProfileDeclarations} from '../profile/dataset-profile-cache';
import type {KeplerDatasetProfile} from '../profile/dataset-profile-types';
import {evaluateKeplerWorkbenchInputs} from '../descriptors/workbench-adapters';
import {hashWorkflowValue, makeWorkflowDataFromFeatures} from './workflow-data';
import {getWorkflowPortCompatibility} from './workflow-operations';
import {makeWorkflowSourceGeoJSONData} from './workflow-source-operations';
import {WORKFLOW_LIMITS, WorkflowError} from './workflow-types';
import type {
  WorkflowDocument,
  WorkflowHost,
  WorkflowInput,
  WorkflowIssue,
  WorkflowOperation,
  WorkflowOperationRegistry,
  WorkflowParameterSpec,
  WorkflowPlan,
  WorkflowPlannedStep,
  WorkflowPortKind,
  WorkflowStep
} from './workflow-types';

type WorkflowGraph = {
  steps: Map<string, WorkflowStep>;
  upstream: Map<string, Set<string>>;
  downstream: Map<string, Set<string>>;
};

function makeWorkflowGraph(document: WorkflowDocument): WorkflowGraph {
  const steps = new Map(document.steps.map(step => [step.id, step]));
  const upstream = new Map(document.steps.map(step => [step.id, new Set<string>()]));
  const downstream = new Map(document.steps.map(step => [step.id, new Set<string>()]));
  for (const step of document.steps) {
    for (const connection of Object.values(step.inputs)) {
      if (!steps.has(connection.stepId)) continue;
      upstream.get(step.id)!.add(connection.stepId);
      downstream.get(connection.stepId)!.add(step.id);
    }
  }
  return {steps, upstream, downstream};
}

function compareIds(first: string, second: string): number {
  return first < second ? -1 : first > second ? 1 : 0;
}

function collectReachable(neighbors: Map<string, Set<string>>, stepId: string): Set<string> {
  const reached = new Set<string>();
  const pending = [...(neighbors.get(stepId) ?? [])];
  while (pending.length) {
    const current = pending.pop()!;
    if (reached.has(current)) continue;
    reached.add(current);
    pending.push(...(neighbors.get(current) ?? []));
  }
  return reached;
}

function orderWorkflowGraph(graph: WorkflowGraph): {order: string[]; cyclic: string[]} {
  const compareSteps = (first: string, second: string): number => {
    const firstPosition = graph.steps.get(first)!.position;
    const secondPosition = graph.steps.get(second)!.position;
    return (
      firstPosition.x - secondPosition.x ||
      firstPosition.y - secondPosition.y ||
      compareIds(first, second)
    );
  };
  const order: string[] = [];
  const remaining = new Set(graph.steps.keys());
  const indegrees = new Map(
    [...graph.upstream].map(([stepId, upstream]) => [stepId, upstream.size])
  );
  const ready = [...remaining].filter(stepId => indegrees.get(stepId) === 0).sort(compareSteps);
  function drainReady(): void {
    while (ready.length) {
      const stepId = ready.shift()!;
      order.push(stepId);
      remaining.delete(stepId);
      for (const downstreamId of graph.downstream.get(stepId)!) {
        if (!remaining.has(downstreamId)) continue;
        const indegree = indegrees.get(downstreamId)! - 1;
        indegrees.set(downstreamId, indegree);
        if (indegree === 0) ready.push(downstreamId);
      }
      ready.sort(compareSteps);
    }
  }
  drainReady();
  // Kahn's remainder also includes dependents of cycles. Only a path back to the same node
  // identifies a cycle member; dependents must remain eligible for a blocked status.
  const cyclic = [...remaining]
    .filter(stepId => collectReachable(graph.upstream, stepId).has(stepId))
    .sort(compareIds);
  for (const stepId of cyclic) {
    remaining.delete(stepId);
    for (const downstreamId of graph.downstream.get(stepId)!) {
      indegrees.set(downstreamId, indegrees.get(downstreamId)! - 1);
    }
  }
  ready.push(...[...remaining].filter(stepId => indegrees.get(stepId) === 0));
  ready.sort(compareSteps);
  drainReady();
  return {order: [...order, ...cyclic], cyclic};
}

/**
 * Deterministic topological order (ties by position.x, position.y, id). Steps on a cycle, and
 * steps that read a missing step, are listed in `cyclic` / left in `order` respectively.
 */
export function getWorkflowTopologicalOrder(document: WorkflowDocument): {
  order: string[];
  cyclic: string[];
} {
  return orderWorkflowGraph(makeWorkflowGraph(document));
}

/** Every step `stepId` depends on, transitively, in topological order (without `stepId`). */
export function getWorkflowUpstreamSteps(document: WorkflowDocument, stepId: string): string[] {
  const graph = makeWorkflowGraph(document);
  const reached = collectReachable(graph.upstream, stepId);
  return orderWorkflowGraph(graph).order.filter(
    current => current !== stepId && reached.has(current)
  );
}

/** Every step that depends on `stepId`, transitively, in topological order. */
export function getWorkflowDownstreamSteps(document: WorkflowDocument, stepId: string): string[] {
  const graph = makeWorkflowGraph(document);
  const reached = collectReachable(graph.downstream, stepId);
  return orderWorkflowGraph(graph).order.filter(
    current => current !== stepId && reached.has(current)
  );
}

/** Raw parameter values of a step: operation defaults, static bindings, workflow inputs resolved. */
export function resolveWorkflowParameters(
  document: WorkflowDocument,
  step: WorkflowStep,
  operation: WorkflowOperation
): Record<string, unknown> {
  const inputs = new Map(document.inputs.map(input => [input.id, input.value]));
  const explicit: Record<string, unknown> = {};
  for (const parameter of operation.parameters) {
    const binding = Object.hasOwn(step.parameters, parameter.name)
      ? step.parameters[parameter.name]
      : undefined;
    const value =
      binding?.from === 'static'
        ? binding.value
        : binding?.from === 'input'
          ? inputs.get(binding.inputId)
          : undefined;
    if (value !== undefined) explicit[parameter.name] = Array.isArray(value) ? [...value] : value;
  }
  let resolved = explicit;
  try {
    resolved = resolveCatalogParameterAliases(operation.parameters, explicit, (name, problem) => {
      throw new WorkflowError('parameter-out-of-range', `${operation.id}: ${name} ${problem}`);
    });
  } catch (cause) {
    if (!(cause instanceof WorkflowError)) throw cause;
    // This is the raw inspector resolver: retain invalid edits so forms can render their
    // named error. Planning, preflight and execution reject them in normalizeParameters.
    // In particular, never replace a genuinely conflicting explicit pair with one spelling.
  }
  const defaults = resolveCatalogParameterAliases(
    operation.parameters,
    Object.fromEntries(
      operation.parameters.flatMap(parameter =>
        parameter.defaultValue === undefined
          ? []
          : [
              [
                parameter.name,
                Array.isArray(parameter.defaultValue)
                  ? [...parameter.defaultValue]
                  : parameter.defaultValue
              ]
            ]
      )
    ),
    (name, problem) => {
      throw new WorkflowError('parameter-out-of-range', `${operation.id}: ${name} ${problem}`);
    }
  );
  return {...defaults, ...resolved};
}

/** Explicit values only: filling operation constants here would erase AX default provenance. */
export function resolveWorkflowParameterOverrides(
  document: WorkflowDocument,
  step: WorkflowStep,
  operation: WorkflowOperation
): Record<string, import('../analysis/catalog-binding-types').CatalogParameterValue> {
  const values = resolveWorkflowParameters(document, step, operation);
  return Object.fromEntries(
    Object.keys(step.parameters)
      .filter(name => name in values)
      .map(name => [
        name,
        values[name] as import('../analysis/catalog-binding-types').CatalogParameterValue
      ])
  );
}

const SOURCE_PROFILE_CACHE = new Map<string, KeplerDatasetProfile>();

/** Profiles for direct sources; intermediate requirements are checked when their actual data arrives. */
export function getWorkflowAnalysisProfiles(
  document: WorkflowDocument,
  step: WorkflowStep,
  registry: WorkflowOperationRegistry,
  host?: WorkflowHost
): Record<string, KeplerDatasetProfile | undefined> {
  return Object.fromEntries(
    Object.entries(step.inputs).map(([name, connection]) => {
      const source = document.steps.find(candidate => candidate.id === connection.stepId);
      const operation = source && registry.get(source.operationId);
      if (!source || !operation || connection.output !== 'data') return [name, undefined];
      const values = resolveWorkflowParameters(document, source, operation);
      if (source.operationId === 'source-dataset' && values['population'] !== 'selected') {
        const dataset =
          host?.listDatasets().find(candidate => candidate.id === values['dataset']) ??
          host
            ?.listDatasets()
            .filter(candidate => candidate.label === values['datasetLabel'])
            .find((_, index, matches) => matches.length === 1 && index === 0);
        const profile = dataset ? host?.getAnalysisProfile?.(dataset.id) : undefined;
        if (!profile) return [name, undefined];
        if (values['geometryMode'] === 'none' && profile.kind === 'surface') return [name, profile];
        if (values['geometryMode'] === 'none')
          return [
            name,
            {
              ...profile,
              kind: 'attributes',
              catalogGeometry: undefined,
              geometryColumns: [],
              pointColumns: undefined,
              coordinates: undefined,
              spacing: undefined
            }
          ];
        const mappingMatches =
          values['geometryMode'] === 'point-columns'
            ? profile.pointColumns?.longitude === values['longitudeField'] &&
              profile.pointColumns?.latitude === values['latitudeField']
            : profile.geometryColumns.includes(String(values['geometryField']));
        return [name, mappingMatches ? profile : undefined];
      }
      if (source.operationId === 'source-geojson' && typeof values['text'] === 'string') {
        const text = values['text'];
        try {
          const profileKey = JSON.stringify([
            source.id,
            source.label,
            values['name'],
            text,
            values['coordinateSystem'],
            values['profileDeclarations']
          ]);
          let profile = SOURCE_PROFILE_CACHE.get(profileKey);
          if (!profile) {
            profile = makeWorkflowSourceGeoJSONData(parseCatalogFeatureCollection(text), {
              id: source.id,
              label: String(values['name'] || 'GeoJSON'),
              coordinateSystem: values['coordinateSystem'] as string,
              declarations: values['profileDeclarations']
                ? parseKeplerDatasetProfileDeclarations(values['profileDeclarations'] as string)
                : {}
            }).analysisProfile!;
            if (SOURCE_PROFILE_CACHE.size >= WORKFLOW_LIMITS.maximumSteps)
              SOURCE_PROFILE_CACHE.clear();
            SOURCE_PROFILE_CACHE.set(profileKey, profile);
          }
          return [name, profile];
        } catch {
          return [name, undefined];
        }
      }
      return [name, undefined];
    })
  );
}

function validateWorkflowInputBinding(
  input: WorkflowInput,
  parameter: WorkflowParameterSpec
): void {
  const numericInput = input.kind === 'number' || input.kind === 'integer';
  const numericParameter = parameter.kind === 'number' || parameter.kind === 'integer';
  const stringParameter = [
    'string',
    'enum',
    'text',
    'dataset',
    'column',
    'column-list',
    'expression'
  ].includes(parameter.kind);
  const compatible =
    (numericInput && numericParameter) ||
    ((input.kind === 'string' || input.kind === 'enum') && stringParameter) ||
    (input.kind === 'boolean' && parameter.kind === 'boolean') ||
    (input.kind === 'number-list' && parameter.kind === 'number-list');
  if (!compatible) {
    throw new WorkflowError(
      'parameter-out-of-range',
      `"${parameter.label}" cannot bind workflow input "${input.label}" of kind ${input.kind} to a ${parameter.kind} parameter.`,
      {parameter: parameter.name}
    );
  }
  const value = input.value;
  let requirement: string | undefined;
  if (numericInput && (typeof value !== 'number' || !Number.isFinite(value))) {
    requirement = 'a finite number';
  } else if (input.kind === 'integer' && !Number.isInteger(value)) {
    requirement = 'an integer';
  } else if (input.kind === 'boolean' && typeof value !== 'boolean') {
    requirement = 'a boolean';
  } else if ((input.kind === 'string' || input.kind === 'enum') && typeof value !== 'string') {
    requirement = 'a string';
  } else if (input.kind === 'enum' && !input.options?.includes(value as string)) {
    requirement = `one of ${(input.options ?? []).map(option => JSON.stringify(option)).join(', ')}`;
  } else if (
    input.kind === 'number-list' &&
    (!Array.isArray(value) ||
      value.length === 0 ||
      value.some(number => typeof number !== 'number' || !Number.isFinite(number)))
  ) {
    requirement = 'a non-empty list of finite numbers';
  }
  if (!requirement && (numericInput || input.kind === 'number-list')) {
    const numbers = typeof value === 'number' ? [value] : (value as readonly number[]);
    if (input.minimum !== undefined && numbers.some(number => number < input.minimum!)) {
      requirement = `at least ${input.minimum}`;
    } else if (input.maximum !== undefined && numbers.some(number => number > input.maximum!)) {
      requirement = `at most ${input.maximum}`;
    }
  }
  if (requirement) {
    throw new WorkflowError(
      'parameter-out-of-range',
      `"${parameter.label}" bound to workflow input "${input.label}" must be ${requirement}; got ${JSON.stringify(value)}.`,
      {parameter: parameter.name}
    );
  }
}

function readParameterError(error: unknown): Pick<WorkflowIssue, 'code' | 'message' | 'parameter'> {
  const fallback = {
    code: 'parameter-out-of-range' as const,
    message: 'parameter normalization failed with an unreadable error.'
  };
  try {
    const message = error instanceof Error ? error.message : String(error);
    if (typeof message !== 'string') return fallback;
    return error instanceof WorkflowError
      ? {code: error.code, message, parameter: error.details.parameter}
      : {code: 'parameter-out-of-range', message};
  } catch {
    // Thrown objects can lack a primitive representation or expose a throwing message getter.
    return fallback;
  }
}

function inferParameterName(message: string, operation: WorkflowOperation): string | undefined {
  return operation.parameters
    .filter(parameter => {
      if (!parameter.label) return false;
      let position = message.indexOf(parameter.label);
      while (position !== -1) {
        const before = message[position - 1];
        const after = message[position + parameter.label.length];
        if (
          (!before || !/[\p{L}\p{N}_]/u.test(before)) &&
          (!after || !/[\p{L}\p{N}_]/u.test(after))
        )
          return true;
        position = message.indexOf(parameter.label, position + 1);
      }
      return false;
    })
    .sort(
      (first, second) =>
        second.label.length - first.label.length || compareIds(first.name, second.name)
    )[0]?.name;
}

export type WorkflowPlanOptions = {
  registry: WorkflowOperationRegistry;
  /** Supplies external revision keys of source steps; without it sources key on parameters only. */
  host?: WorkflowHost;
  /**
   * Kinds observed in the last results, by step id and output name. They settle `maybe`
   * compatibility into `yes` or `no` before the next run.
   */
  observedKinds?: Readonly<Record<string, Readonly<Record<string, WorkflowPortKind>>>>;
};

/** Resolve inherited representations for planning, editing and host port summaries. Unknown sources defer. */
export function getWorkflowOutputKind(
  document: WorkflowDocument,
  registry: WorkflowOperationRegistry,
  stepId: string,
  outputName: string,
  options: Pick<WorkflowPlanOptions, 'host' | 'observedKinds'> = {},
  visited = new Set<string>()
): WorkflowPortKind {
  const producerObservations =
    options.observedKinds && Object.hasOwn(options.observedKinds, stepId)
      ? options.observedKinds[stepId]
      : undefined;
  const observed =
    producerObservations && Object.hasOwn(producerObservations, outputName)
      ? producerObservations[outputName]
      : undefined;
  if (observed) return observed;
  const producer = document.steps.find(step => step.id === stepId);
  const operation = producer && registry.get(producer.operationId);
  const output = operation?.outputs.find(port => port.name === outputName);
  if (!producer || !operation || !output || visited.has(stepId)) return 'features';
  visited.add(stepId);
  if (output.kindFrom) {
    const connection = producer.inputs[output.kindFrom];
    return connection
      ? getWorkflowOutputKind(
          document,
          registry,
          connection.stepId,
          connection.output,
          options,
          visited
        )
      : 'features';
  }
  if (producer.operationId === 'source-dataset') {
    const values = resolveWorkflowParameters(document, producer, operation);
    const dataset = options.host
      ?.listDatasets()
      .find(candidate => candidate.id === values['dataset']);
    const profile = dataset ? options.host?.getAnalysisProfile?.(dataset.id) : undefined;
    if (profile?.kind === 'surface') return 'raster';
    if (values['geometryMode'] === 'none') return profile ? 'table' : 'features';
    if (values['geometryMode'] === 'point-columns') return 'point';
    if (
      profile?.catalogGeometry &&
      profile.geometryColumns.includes(String(values['geometryField']))
    )
      return profile.catalogGeometry;
  }
  if (producer.operationId === 'source-geojson') {
    try {
      const values = resolveWorkflowParameters(document, producer, operation);
      return makeWorkflowDataFromFeatures(parseCatalogFeatureCollection(String(values['text'])), {
        id: producer.id,
        label: producer.label ?? operation.title
      }).kind;
    } catch {
      return 'features';
    }
  }
  return output.kind;
}

/**
 * Static plan: order, normalised parameters, issues, status and content key per step.
 *
 * key(step) = hash(operation id, operation version, normalised parameters, engine choice,
 * external revision key, and for each input port in name order: port name, key(upstream step),
 * output name). Disabled steps, steps with an error issue and everything downstream of them have
 * a null key. Pure and synchronous; never runs an operation.
 */
export function planWorkflow(
  document: WorkflowDocument,
  options: WorkflowPlanOptions
): WorkflowPlan {
  const graph = makeWorkflowGraph(document);
  const {order, cyclic} = orderWorkflowGraph(graph);
  const cycleMembers = new Set(cyclic);
  const positions = new Map(order.map((stepId, index) => [stepId, index]));
  const comparePosition = (first: string, second: string): number =>
    positions.get(first)! - positions.get(second)!;
  const steps: Record<string, WorkflowPlannedStep> = Object.fromEntries(
    order.map(stepId => {
      const step = graph.steps.get(stepId)!;
      const operation = options.registry.get(step.operationId);
      return [
        stepId,
        {
          stepId,
          ...(operation ? {operation} : {}),
          engine:
            operation?.engine === 'cpu-reference' || operation?.engine === 'host'
              ? 'cpu'
              : (step.engine ?? document.settings.engine),
          key: null,
          status: 'ready',
          issues: [],
          upstream: [...graph.upstream.get(stepId)!].sort(comparePosition),
          downstream: [...graph.downstream.get(stepId)!].sort(comparePosition)
        }
      ];
    })
  );
  const getLabel = (stepId: string): string => {
    const step = graph.steps.get(stepId)!;
    return step.label ?? steps[stepId].operation?.title ?? stepId;
  };
  const issues: WorkflowIssue[] = [];
  if (document.steps.length > WORKFLOW_LIMITS.maximumSteps) {
    issues.push({
      severity: 'error',
      code: 'limit-exceeded',
      message: `${document.title}: steps must not exceed ${WORKFLOW_LIMITS.maximumSteps}; got ${document.steps.length}.`
    });
  }
  if (cyclic.length) {
    issues.push({
      severity: 'error',
      code: 'cycle',
      message: `${document.title}: a cycle connects ${cyclic.map(getLabel).join(', ')}.`
    });
  }
  const workflowInputs = new Map(document.inputs.map(input => [input.id, input]));
  for (const stepId of order) {
    const step = graph.steps.get(stepId)!;
    const planned = steps[stepId];
    const operation = planned.operation;
    const label = getLabel(stepId);
    const addIssue = (
      code: WorkflowIssue['code'],
      severity: WorkflowIssue['severity'],
      message: string,
      details: Pick<WorkflowIssue, 'port' | 'parameter'> = {}
    ): void => {
      planned.issues.push({code, severity, message: `${label}: ${message}`, stepId, ...details});
    };
    if (cycleMembers.has(stepId)) {
      addIssue('cycle', 'error', 'cannot read itself, directly or through a cycle.');
    }
    if (!operation) {
      addIssue('unknown-operation', 'error', `operation "${step.operationId}" is not available.`);
      continue;
    }
    for (const name of Object.keys(step.inputs).sort(compareIds)) {
      const connection = step.inputs[name];
      const input = operation.inputs.find(port => port.name === name);
      if (!input) {
        addIssue('unknown-input-port', 'error', `input "${name}" is not declared.`, {port: name});
      }
      const producer = graph.steps.get(connection.stepId);
      if (!producer) {
        addIssue(
          'unknown-step',
          'error',
          `"${input?.label ?? name}" refers to missing step "${connection.stepId}".`,
          {port: name}
        );
        continue;
      }
      const producerOperation = steps[producer.id].operation;
      if (!producerOperation) continue;
      const output = producerOperation.outputs.find(port => port.name === connection.output);
      if (!output) {
        addIssue(
          'unknown-output-port',
          'error',
          `"${input?.label ?? name}" refers to output "${connection.output}" that ${getLabel(producer.id)} does not declare.`,
          {port: name}
        );
        continue;
      }
      if (!input) continue;
      const kind = getWorkflowOutputKind(
        document,
        options.registry,
        producer.id,
        output.name,
        options
      );
      const compatibility = getWorkflowPortCompatibility(kind, input);
      const connectionMessage = `"${input.label}" accepts ${input.accepts.map(formatKind).join(' or ')}; "${output.label}" of ${getLabel(producer.id)} is ${formatKind(kind)}.`;
      if (compatibility === 'no') {
        addIssue('incompatible-port', 'error', connectionMessage, {port: name});
      } else if (compatibility === 'maybe') {
        addIssue('unverified-port', 'info', connectionMessage, {port: name});
      }
      if (kind !== 'raster' && output.maximumRows > input.maximumRows) {
        addIssue(
          'capacity-may-exceed',
          'info',
          `"${input.label}" accepts at most ${input.maximumRows} rows; "${output.label}" of ${getLabel(producer.id)} allows up to ${output.maximumRows} rows (an upper bound).`,
          {port: name}
        );
      }
    }
    for (const input of operation.inputs) {
      if (!input.optional && !Object.hasOwn(step.inputs, input.name)) {
        addIssue('input-required', 'error', `needs an input for "${input.label}".`, {
          port: input.name
        });
      }
    }
    for (const name of Object.keys(step.parameters).sort(compareIds)) {
      const parameter = operation.parameters.find(specification => specification.name === name);
      if (!parameter) {
        const problem =
          getRemovedPointCloudIcpParameterMessage(step.operationId, name) ??
          getCatalogParameterRenameMessage(operation.parameters, name) ??
          `parameter "${name}" is not declared.`;
        addIssue('unknown-parameter', 'error', problem, {parameter: name});
      }
      const binding = step.parameters[name];
      if (binding.from === 'input' && !workflowInputs.has(binding.inputId)) {
        addIssue(
          'unknown-workflow-input',
          'error',
          `"${parameter?.label ?? name}" refers to missing workflow input "${binding.inputId}".`,
          {parameter: name}
        );
      }
    }
    try {
      for (const parameter of operation.parameters) {
        const binding = Object.hasOwn(step.parameters, parameter.name)
          ? step.parameters[parameter.name]
          : undefined;
        if (binding?.from !== 'input') continue;
        const input = workflowInputs.get(binding.inputId);
        if (input) validateWorkflowInputBinding(input, parameter);
      }
      const profiles = operation.analysisDescriptor
        ? getWorkflowAnalysisProfiles(document, step, options.registry, options.host)
        : {};
      const hasProfiles =
        operation.analysisDescriptor && operation.inputs.every(input => profiles[input.name]);
      if (hasProfiles) {
        planned.analysis = evaluateKeplerWorkbenchInputs(
          operation.analysisBinding ?? operation.id,
          profiles,
          {
            overrides: resolveWorkflowParameterOverrides(document, step, operation),
            executionEngine:
              planned.engine === 'cpu' || (options.host && !options.host.getDevice())
                ? 'cpu'
                : 'gpu'
          }
        );
        for (const missing of planned.analysis.missing) {
          addIssue('analysis-requirement', 'error', `${missing.what}. ${missing.remedy}`);
        }
        for (const warning of planned.analysis.warnings)
          addIssue('analysis-requirement', 'warning', warning);
      }
      planned.parameters = operation.normalizeParameters(
        planned.analysis?.parameters ?? resolveWorkflowParameters(document, step, operation)
      );
    } catch (error) {
      const {message, code, parameter: explicitParameter} = readParameterError(error);
      const operationPrefix = `${operation.title}: `;
      const stepPrefix = `${label}: `;
      const detail = message.startsWith(stepPrefix)
        ? message.slice(stepPrefix.length)
        : message.startsWith(operationPrefix)
          ? message.slice(operationPrefix.length)
          : message;
      const parameter = explicitParameter ?? inferParameterName(detail, operation);
      addIssue(code, 'error', detail, parameter ? {parameter} : {});
    }
    for (const output of step.display ?? []) {
      if (!operation.outputs.some(port => port.name === output)) {
        addIssue(
          'display-unknown-output',
          'warning',
          `display output "${output}" is not declared.`,
          {
            port: output
          }
        );
      }
    }
  }
  // Determine configuration for every step before walking upstream: cycle members are appended
  // to the order, so a dependent can precede its invalid producer.
  for (const stepId of order) {
    steps[stepId].status = graph.steps.get(stepId)!.disabled
      ? 'disabled'
      : steps[stepId].issues.some(issue => issue.severity === 'error')
        ? 'not-configured'
        : 'ready';
  }
  for (const stepId of order) {
    const planned = steps[stepId];
    if (planned.status !== 'ready') continue;
    const pending = [...planned.upstream];
    const visited = new Set<string>([stepId]);
    for (let index = 0; index < pending.length; index++) {
      const upstreamId = pending[index];
      if (visited.has(upstreamId)) continue;
      visited.add(upstreamId);
      const upstream = steps[upstreamId];
      if (upstream.status === 'disabled' || upstream.status === 'not-configured') {
        const disabled = upstream.status === 'disabled';
        planned.status = 'blocked';
        planned.issues.push({
          code: disabled ? 'upstream-disabled' : 'upstream-not-configured',
          severity: 'info',
          stepId,
          message: `${getLabel(stepId)}: upstream step "${getLabel(upstreamId)}" is ${disabled ? 'disabled' : 'not configured'}.`
        });
        break;
      }
      pending.push(...upstream.upstream);
    }
  }
  for (const stepId of order) {
    const planned = steps[stepId];
    if (planned.status !== 'ready') continue;
    const step = graph.steps.get(stepId)!;
    const operation = planned.operation!;
    let externalRevision: string | undefined;
    if (options.host && operation.getExternalRevisionKey) {
      try {
        externalRevision = operation.getExternalRevisionKey(options.host, planned.parameters!);
      } catch {
        // A source will report the host failure when run; keep the static plan available.
        externalRevision = 'error';
      }
    }
    planned.key = hashWorkflowValue([
      operation.id,
      operation.version,
      planned.parameters,
      planned.engine,
      ...(planned.analysis ? [planned.analysis] : []),
      // Runtime AX can derive defaults only after intermediate inputs arrive. Explicit choices
      // must remain distinct even when static normalization fills the same catalog constant.
      ...(operation.analysisDescriptor
        ? [resolveWorkflowParameterOverrides(document, step, operation)]
        : []),
      ...(step.curatedAnalysisId ? [{curatedAnalysisId: step.curatedAnalysisId}] : []),
      externalRevision,
      Object.keys(step.inputs)
        .sort(compareIds)
        .map(name => {
          const connection = step.inputs[name];
          return [name, steps[connection.stepId].key, connection.output];
        })
    ]);
  }
  return {order, issues, steps};
}

function formatKind(kind: WorkflowPortKind): string {
  return kind === 'point' || kind === 'line' || kind === 'polygon' ? `${kind}s` : kind;
}
