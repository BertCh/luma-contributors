// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPU_COMPUTATION_RECORD_SCHEMA} from './computation-record-types';
import {hashGPUComputationText} from './hash-computation-text';
import {mergeGPUComputationSourceRefs} from './merge-computation-source';
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
// Concrete integration target in dev-docs/release-package.md, not the accepted peer range.
// Update alongside computational dependency upgrades; callers may supply a custom build revision.
const DEFAULT_DEPENDENCY_REVISION = '@luma.gl/graph@10.0.0-alpha.16';

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
  /** Concrete computational dependency revision; defaults to `@luma.gl/graph@10.0.0-alpha.16`. */
  dependencyRevision?: string;
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
    const merged = previous ? mergeGPUComputationSourceRefs(previous, source) : {...source};
    // A shared default cannot represent a node whose entry point was unknown.
    // Keep the shared default unknown and restate known node entry points below.
    if (previous && (previous.entryPoint === undefined || source.entryPoint === undefined))
      delete merged.entryPoint;
    sources.set(source.id, merged);
  };
  props.sources?.forEach(addSource);
  for (const input of props.graphs) {
    if ('graph' in input) input.sources.forEach(addSource);
  }
  // Node entry points are stored only where they differ from the shared source ref; restate them
  // against the merged ref, which may come from another graph's first node.
  props.graphs.forEach((input, graphIndex) => {
    const described = new Map(
      ('graph' in input ? input.sources : (props.sources ?? [])).map(source => [source.id, source])
    );
    for (const node of graphs[graphIndex].nodes as {source?: string; entryPoint?: string}[]) {
      const own = node.source === undefined ? undefined : described.get(node.source);
      const merged = node.source === undefined ? undefined : sources.get(node.source);
      if (!own || !merged) continue;
      const entryPoint = node.entryPoint ?? own.entryPoint;
      if (entryPoint === undefined || entryPoint === merged.entryPoint) delete node.entryPoint;
      else node.entryPoint = entryPoint;
    }
  });
  return structuredClone<GPUComputationRecord>({
    schema: GPU_COMPUTATION_RECORD_SCHEMA,
    id: props.id,
    ...(props.datasetVersion !== undefined && {datasetVersion: props.datasetVersion}),
    ...(props.seed !== undefined && {seed: props.seed}),
    runId: props.runId ?? getGPUComputationRunId({...props, graphs, adapter, choices}),
    executionId: props.executionId ?? globalThis.crypto.randomUUID(),
    implementationRevision: props.implementationRevision ?? DEFAULT_IMPLEMENTATION_REVISION,
    dependencyRevision: props.dependencyRevision ?? DEFAULT_DEPENDENCY_REVISION,
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
  });
}

/** Hashes ordered graph/node ids, adapter, parameters, choices, revision, dataset and seed. */
export function getGPUComputationRunId(props: {
  graphs: readonly GPUComputationGraph[];
  adapter: GPUComputationAdapter;
  datasetVersion?: string;
  seed?: string | number;
  parameters?: GPUComputationAttributes;
  implementationRevision?: string;
  /** Concrete computational dependency revision; defaults to `@luma.gl/graph@10.0.0-alpha.16`. */
  dependencyRevision?: string;
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
      props.dependencyRevision ?? DEFAULT_DEPENDENCY_REVISION,
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
