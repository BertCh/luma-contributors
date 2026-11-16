// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {formatWorkflowParameterValue} from './workflow-parameters';
import {getWorkflowTopologicalOrder} from './workflow-plan';
import type {WorkflowPreflight} from './workflow-preflight';
import type {
  WorkflowDocument,
  WorkflowOperation,
  WorkflowOperationRegistry,
  WorkflowRunRecord,
  WorkflowRunnerState,
  WorkflowStep
} from './workflow-types';

function escapeMarkdown(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\\/g, '\\\\')
    .replace(/[|`*_\[\]#]/g, character => `\\${character}`)
    .replace(/\r\n|[\r\n\u2028\u2029]/g, '\\n');
}

function formatDuration(milliseconds: number): string {
  return milliseconds < 1000
    ? `${Math.round(milliseconds)} ms`
    : `${Number((milliseconds / 1000).toFixed(1))} s`;
}

function getStepLabel(
  step: WorkflowStep | undefined,
  operation: WorkflowOperation | undefined,
  fallback: string
): string {
  return step?.label ?? operation?.title ?? fallback;
}

function orderNames(names: string[], specifications: readonly {name: string}[]): string[] {
  return [
    ...specifications.map(specification => specification.name).filter(name => names.includes(name)),
    ...names
      .filter(name => !specifications.some(specification => specification.name === name))
      .sort()
  ];
}

/**
 * A run report in Markdown: what the workflow is, what ran, on which engine, how long it took,
 * with which parameters, and what each step reported. Meant to be pasted next to a result so
 * someone else can reproduce it. Deterministic for the same inputs except for the timestamps it
 * prints from the record.
 */
export function makeWorkflowRunReport(
  document: WorkflowDocument,
  registry: WorkflowOperationRegistry,
  state: WorkflowRunnerState,
  record: WorkflowRunRecord,
  preflight?: WorkflowPreflight
): string {
  const steps = new Map(document.steps.map(step => [step.id, step]));
  const lines = [
    `# ${escapeMarkdown(document.title)}`,
    '',
    `Run ${escapeMarkdown(record.runId)} · ${record.status} · ${record.steps.length} steps · ${formatDuration(record.durationMilliseconds)} · started ${new Date(record.startedAt).toISOString()}`
  ];
  if (document.inputs.length) {
    lines.push('', '## Inputs', '');
    for (const input of document.inputs) {
      const value = Array.isArray(input.value) ? input.value.join(', ') : String(input.value);
      lines.push(
        `- ${escapeMarkdown(input.label)}: ${escapeMarkdown(value)}${input.unit ? ` ${escapeMarkdown(input.unit)}` : ''}`
      );
    }
  }
  if (preflight) {
    lines.push(
      '',
      '## Preflight (current document, cold run)',
      '',
      `GPU scratch peak: ${preflight.peakScratchBytes ?? 'unknown'} bytes; command nodes: ${preflight.commandNodeCount ?? 'unknown'}.`,
      'Logical GPU transients only; excludes inputs, outputs, staging, JavaScript heap and driver overhead.',
      '',
      '| Step | Scratch bytes | Command nodes | Basis |',
      '| --- | --- | --- | --- |'
    );
    for (const estimate of preflight.steps) {
      lines.push(
        `| ${escapeMarkdown(estimate.stepId)} | ${estimate.scratchBytes ?? 'unknown'} | ${estimate.commandNodeCount ?? 'unknown'} | ${escapeMarkdown(estimate.reason)} |`
      );
    }
    lines.push('', '### Materialisation boundaries', '');
    for (const boundary of preflight.boundaries) {
      lines.push(
        `- ${escapeMarkdown(boundary.producer)}.${escapeMarkdown(boundary.output)} → ${escapeMarkdown(boundary.consumer)}.${escapeMarkdown(boundary.input)}: ${boundary.materialization === 'cpu-raster' ? 'CPU raster band' : 'CPU table'}; ${escapeMarkdown(boundary.reason)}`
      );
    }
  }
  lines.push(
    '',
    '## Steps',
    '',
    '| # | Step | Operation | Status | Engine | Time | Output rows |',
    '| --- | --- | --- | --- | --- | --- | --- |'
  );
  record.steps.forEach((entry, index) => {
    const operation = registry.get(entry.operationId);
    const label = getStepLabel(steps.get(entry.stepId), operation, entry.stepId);
    const rows = orderNames(Object.keys(entry.outputRows ?? {}), operation?.outputs ?? []).map(
      name => `${name}: ${entry.outputRows![name]}`
    );
    const engine =
      entry.engine === 'gpu'
        ? 'GPU'
        : entry.engine === 'cpu-reference'
          ? 'CPU'
          : entry.engine === 'host'
            ? 'Host'
            : '—';
    lines.push(
      `| ${index + 1} | ${escapeMarkdown(label)} | ${escapeMarkdown(operation?.title ?? entry.operationId)} | ${entry.status} | ${engine} | ${entry.cacheHit ? 'cached' : formatDuration(entry.durationMilliseconds)} | ${escapeMarkdown(rows.join(', '))} |`
    );
  });
  lines.push('', '## Details');
  record.steps.forEach((entry, index) => {
    const operation = registry.get(entry.operationId);
    const label = getStepLabel(steps.get(entry.stepId), operation, entry.stepId);
    const result = state.steps[entry.stepId]?.result;
    // A failed rerun may still hold the previous success; it is not evidence for this run.
    const runResult = entry.status === 'done' && result?.key === entry.key ? result : undefined;
    lines.push(
      '',
      `### ${index + 1}. ${escapeMarkdown(label)}`,
      '',
      `- Operation: ${escapeMarkdown(entry.operationId)}`
    );
    for (const input of entry.inputs) {
      const producer = steps.get(input.stepId);
      const producerOperation = producer && registry.get(producer.operationId);
      const portLabel =
        operation?.inputs.find(port => port.name === input.port)?.label ?? input.port;
      const outputLabel =
        producerOperation?.outputs.find(port => port.name === input.output)?.label ?? input.output;
      lines.push(
        `- ${escapeMarkdown(portLabel)} ← ${escapeMarkdown(getStepLabel(producer, producerOperation, input.stepId))} · ${escapeMarkdown(outputLabel)} (${input.rowCount} rows)`
      );
    }
    for (const name of orderNames(
      Object.keys(runResult?.parameters ?? {}),
      operation?.parameters ?? []
    )) {
      const parameter = operation?.parameters.find(specification => specification.name === name);
      const value = runResult!.parameters[name];
      const formatted =
        typeof value === 'string' && value.length > 120
          ? `${value.length} characters`
          : parameter
            ? formatWorkflowParameterValue(parameter, value)
            : Array.isArray(value)
              ? value.join(', ')
              : String(value);
      lines.push(
        `- ${escapeMarkdown(parameter?.label ?? name)}: ${escapeMarkdown(formatted)}${parameter?.unit ? ` ${escapeMarkdown(parameter.unit)}` : ''}`
      );
    }
    const sensitivity = entry.sensitivity ?? runResult?.sensitivity;
    if (sensitivity) {
      lines.push(
        `- Exploratory parameter search: ${escapeMarkdown(sensitivity.parameter)} = ${sensitivity.values.join(', ')}; chosen ${sensitivity.chosenValue ?? 'none'}.`
      );
      lines.push(
        `- Search population: ${escapeMarkdown(sensitivity.population.scope)}; ${sensitivity.population.rowCount}/${sensitivity.population.sourceRowCount} rows; selection ${escapeMarkdown(sensitivity.population.selectionKey)}; revision ${escapeMarkdown(sensitivity.revisionKey)}.`
      );
      lines.push(
        '- Correction within one map does not control error across this parameter search.'
      );
    }
    const contributors = entry.dispatchedContributors ?? runResult?.dispatchedContributors ?? [];
    if (contributors.length)
      lines.push(`- GPU contributors: ${contributors.map(escapeMarkdown).join(', ')}`);
    const fallbackReason = entry.fallbackReason ?? runResult?.fallbackReason;
    if (fallbackReason) lines.push(`- Fallback: ${escapeMarkdown(fallbackReason)}`);
    for (const diagnostic of runResult?.diagnostics ?? []) {
      lines.push(
        `- Diagnostic: ${diagnostic.severity} · ${escapeMarkdown(diagnostic.code)} · ${diagnostic.count} · ${escapeMarkdown(diagnostic.message)}`
      );
    }
    const error =
      entry.error ?? (entry.status === 'error' ? state.steps[entry.stepId]?.error : undefined);
    if (error)
      lines.push(`- Error: ${escapeMarkdown(error.code)} · ${escapeMarkdown(error.message)}`);
  });
  const notes: string[] = [];
  const addWarning = (code: string, message: string): void => {
    const note = `- Warning: ${escapeMarkdown(code)} · ${escapeMarkdown(message)}`;
    if (!notes.includes(note)) notes.push(note);
  };
  for (const issue of state.issues) {
    if (issue.severity === 'warning') addWarning(issue.code, issue.message);
  }
  for (const step of [...document.steps].sort((first, second) =>
    first.id < second.id ? -1 : first.id > second.id ? 1 : 0
  )) {
    for (const issue of state.steps[step.id]?.issues ?? []) {
      if (issue.severity === 'warning') addWarning(issue.code, issue.message);
    }
  }
  for (const entry of record.steps) {
    const result = state.steps[entry.stepId]?.result;
    if (entry.status === 'done' && result?.key === entry.key) {
      for (const diagnostic of result.diagnostics) {
        if (diagnostic.severity === 'warning') addWarning(diagnostic.code, diagnostic.message);
      }
    }
    if (entry.status === 'skipped' || entry.status === 'cancelled') {
      notes.push(
        `- ${escapeMarkdown(getStepLabel(steps.get(entry.stepId), registry.get(entry.operationId), entry.stepId))}: ${entry.status}`
      );
    }
  }
  lines.push('', '## Notes', '', ...(notes.length ? notes : ['None.']));
  return `${lines.join('\n')}\n`;
}

function quoteDot(value: string): string {
  return `"${value
    .replace(/\\/g, '\\\\')
    .replace(/"/g, '\\"')
    .replace(/\r\n|[\r\n\u2028\u2029]/g, '\\n')
    .replace(/\t/g, '\\t')}"`;
}

/** The workflow graph in Graphviz DOT, one node per step and one edge per connection. */
export function makeWorkflowDot(
  document: WorkflowDocument,
  registry: WorkflowOperationRegistry,
  state?: WorkflowRunnerState | null
): string {
  const steps = new Map(document.steps.map(step => [step.id, step]));
  // Graph export uses id ties so moving nodes on the canvas does not change the text.
  const {order} = getWorkflowTopologicalOrder({
    ...document,
    steps: document.steps.map(step => ({...step, position: {x: 0, y: 0}}))
  });
  const lines = [`digraph ${quoteDot(document.id)} {`, '  rankdir=LR;'];
  for (const stepId of order) {
    const step = steps.get(stepId)!;
    const operation = registry.get(step.operationId);
    const labels = [
      getStepLabel(step, operation, step.operationId),
      operation?.title ?? step.operationId
    ];
    if (state) {
      const current = state.steps[stepId];
      const outputs = current?.result?.outputs;
      const firstOutput = orderNames(Object.keys(outputs ?? {}), operation?.outputs ?? [])[0];
      labels.push(
        `${current?.status ?? 'not-configured'}${firstOutput ? ` · ${outputs![firstOutput].table.rowCount} rows` : ''}`
      );
    }
    lines.push(
      `  ${quoteDot(stepId)} [label=${quoteDot(labels.join('\n'))}${step.disabled ? ', style=dashed' : ''}];`
    );
  }
  for (const stepId of order) {
    const step = steps.get(stepId)!;
    const operation = registry.get(step.operationId);
    for (const name of Object.keys(step.inputs).sort()) {
      const connection = step.inputs[name];
      const producer = steps.get(connection.stepId);
      const producerOperation = producer && registry.get(producer.operationId);
      const outputLabel =
        producerOperation?.outputs.find(port => port.name === connection.output)?.label ??
        connection.output;
      const inputLabel = operation?.inputs.find(port => port.name === name)?.label ?? name;
      lines.push(
        `  ${quoteDot(connection.stepId)} -> ${quoteDot(stepId)} [label=${quoteDot(`${outputLabel} → ${inputLabel}`)}];`
      );
    }
  }
  lines.push('}');
  return `${lines.join('\n')}\n`;
}
