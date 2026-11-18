// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPU_COMPUTATION_RECORD_SCHEMA} from './computation-record-types';
import {hashGPUComputationText} from './hash-computation-text';
import {getGPUComputationIdentityIssues} from './computation-identity-validation';
import type {
  GPUComputationGraph,
  GPUComputationSourceRef,
  GPUComputationAdapter,
  GPUComputationAxes,
  GPUComputationFormulaRef,
  GPUDecisionRecord,
  GPUAnalysisChoice,
  GPUWorkflowRecord,
  GPUComputationRecord,
  GPUComputationAttributes
} from './computation-record-types';

// Bump when analysis correctness changes invalidate comparisons with earlier captures.
const DEFAULT_IMPLEMENTATION_REVISION = 'analysis-2026-11-18-review-1';

/** A graph, optionally accompanied by the sources returned by its description. */
export type GPUComputationGraphInput =
  | GPUComputationGraph
  | {
      graph: GPUComputationGraph;
      sources: readonly GPUComputationSourceRef[];
    };

/** Structure and identity used to assemble one computation record. */
export type CreateGPUComputationRecordProps = {
  id: string;
  adapter: GPUComputationAdapter;
  domains: readonly string[];
  graphs: readonly GPUComputationGraphInput[];
  axes?: Partial<GPUComputationAxes>;
  formulas?: readonly GPUComputationFormulaRef[];
  sources?: readonly GPUComputationSourceRef[];
  decisions?: readonly GPUDecisionRecord[];
  choices?: readonly GPUAnalysisChoice[];
  workflow?: GPUWorkflowRecord;
  runId?: string;
  /** Unique execution identity; defaults to a random UUID. Supply explicitly for fixtures. */
  executionId?: string;
  /** Per-frame parameter values included in deterministic run identity. */
  parameters?: GPUComputationAttributes;
  /** Implementation revision; defaults to `analysis-2026-11-18-review-1`. Change after correctness fixes. */
  implementationRevision?: string;
  createdAt?: string;
  datasetVersion?: string;
  seed?: string | number;
};

/** Assembles a record, deduplicating sources by id; conflicting retained text throws. */
export function createGPUComputationRecord(
  props: CreateGPUComputationRecordProps
): GPUComputationRecord {
  validateIdentity(props);
  const adapter = {...props.adapter};
  const choices =
    props.choices?.map(choice => ({
      ...choice,
      ...(choice.alternatives && {alternatives: [...choice.alternatives]})
    })) ?? [];
  const graphs = props.graphs.map(input => {
    const graph = 'graph' in input ? input.graph : input;
    return {...graph, nodes: graph.nodes.map(node => ({...node}))};
  });
  const sources = new Map<string, GPUComputationSourceRef>();
  const addSource = (source: GPUComputationSourceRef): void => {
    const previous = sources.get(source.id);
    if (
      previous?.text !== undefined &&
      source.text !== undefined &&
      previous.text !== source.text
    ) {
      throw new Error(`Conflicting source text: ${source.id}`);
    }
    if (!previous || (previous.text === undefined && source.text !== undefined)) {
      sources.set(source.id, source);
    }
  };
  props.sources?.forEach(addSource);
  for (const input of props.graphs) {
    if ('graph' in input) input.sources.forEach(addSource);
  }
  return {
    schema: GPU_COMPUTATION_RECORD_SCHEMA,
    id: props.id,
    runId: props.runId ?? getGPUComputationRunId({...props, graphs, adapter, choices}),
    executionId: props.executionId ?? globalThis.crypto.randomUUID(),
    implementationRevision: props.implementationRevision ?? DEFAULT_IMPLEMENTATION_REVISION,
    ...(props.parameters !== undefined && {parameters: {...props.parameters}}),
    createdAt: props.createdAt ?? new Date().toISOString(),
    adapter,
    domains: props.domains,
    graphs,
    axes: {
      axes: props.axes?.axes ?? [],
      keySpaces: props.axes?.keySpaces ?? [],
      relations: props.axes?.relations ?? []
    },
    refs: {formulas: props.formulas ?? [], sources: [...sources.values()]},
    decisions: props.decisions ?? [],
    choices,
    ...(props.workflow !== undefined && {workflow: props.workflow})
  };
}

/** Hashes ordered graph/node ids, adapter, parameters, choices, revision, dataset and seed. */
export function getGPUComputationRunId(props: {
  graphs: readonly GPUComputationGraph[];
  adapter: GPUComputationAdapter;
  datasetVersion?: string;
  seed?: string | number;
  parameters?: GPUComputationAttributes;
  implementationRevision?: string;
  choices?: readonly GPUAnalysisChoice[];
}): string {
  validateIdentity(props);
  return `run-${hashGPUComputationText(
    JSON.stringify([
      props.graphs.map(graph => [graph.id, graph.nodes.map(node => node.id)]),
      [
        props.adapter.backend,
        props.adapter.variant ?? null,
        props.adapter.vendor ?? null,
        props.adapter.architecture ?? null,
        props.adapter.description ?? null
      ],
      props.datasetVersion ?? null,
      props.seed ?? null,
      props.implementationRevision ?? DEFAULT_IMPLEMENTATION_REVISION,
      Object.entries(props.parameters ?? {}).sort(([first], [second]) =>
        first < second ? -1 : first > second ? 1 : 0
      ),
      (props.choices ?? []).map(choice => [choice.stage, choice.name, choice.value])
    ])
  )}`;
}

function validateIdentity(props: Parameters<typeof getGPUComputationIdentityIssues>[0]): void {
  const issue = getGPUComputationIdentityIssues(props)[0];
  if (issue) throw new Error(`computation-record ${issue.path}: ${issue.message}`);
}

/** Reads the DeviceInfo vendor, architecture (falling back to GPU family), and renderer. */
export function getGPUComputationAdapter(
  device: {
    type: string;
    info: {vendor?: string; renderer?: string; gpu?: string; gpuArchitecture?: string};
  },
  variant?: string
): GPUComputationAdapter {
  const architecture = device.info.gpuArchitecture ?? device.info.gpu;
  return {
    backend: device.type,
    ...(device.info.vendor !== undefined && {vendor: device.info.vendor}),
    ...(architecture !== undefined && {architecture}),
    ...(device.info.renderer !== undefined && {description: device.info.renderer}),
    ...(variant !== undefined && {variant})
  };
}
