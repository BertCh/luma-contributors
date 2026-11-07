// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  CompiledGPUCommandGraph,
  GPUCommandGraphEncodingStats,
  GPUCommandGraphNodePreflight
} from '@luma.gl/graph/gpu-core';
import type {
  GPUComputationEdge,
  GPUComputationEdgeKind,
  GPUComputationGraph,
  GPUComputationNode,
  GPUComputationResource,
  GPUComputationSourceRef
} from './computation-record-types';
import type {GPUComputationRecorder} from './computation-recorder';

/** Compiled graph structure and its deduplicated source references. */
export type GPUCommandGraphDescription = {
  /** Scheduled graph, logical resources and dependencies. */
  graph: GPUComputationGraph;
  /** Sources in first scheduled occurrence order. */
  sources: readonly GPUComputationSourceRef[];
};

/** Options naming the described graph. */
export type DescribeGPUCommandGraphOptions = {
  /** Record graph id; defaults to `<compiled.id>-compiled`. Must not contain ':'. */
  graphId?: string;
  /** Display label; defaults to the compiled graph id. */
  label?: string;
  /** Domain pack; defaults to `gpu-core`. */
  domain?: string;
  /** Annotations recorded while the graph was built; without it nodes carry published facts only. */
  recorder?: GPUComputationRecorder;
};

const READ_USAGES = new Set([
  'storage-read',
  'storage-read-write',
  'uniform',
  'copy-source',
  'indirect',
  'vertex',
  'index',
  'sampled'
]);
const WRITE_USAGES = new Set([
  'storage-write',
  'storage-read-write',
  'copy-destination',
  'render-attachment'
]);

/**
 * Describes a compiled graph from its public `id`, `stats` and `preflight` report, merged with
 * the annotations a {@link GPUComputationRecorder} collected. Touches no GPU state.
 *
 * Dependency edges are inferred from declared resource usage at buffer or texture granularity, in
 * scheduled order. Texture mip and layer ranges are not distinguished, so a few `war`, `waw` and
 * `raw` edges can be conservative. Explicit `dependsOn` edges are present only for recorded nodes.
 */
export function describeGPUCommandGraph(
  compiled: Pick<CompiledGPUCommandGraph, 'id' | 'stats' | 'preflight'>,
  options: DescribeGPUCommandGraphOptions = {}
): GPUCommandGraphDescription {
  const graphId = options.graphId ?? `${compiled.id}-compiled`;
  validateGraphId(graphId);
  const recorder = options.recorder;
  const preflightById = new Map(compiled.preflight.nodes.map(node => [node.id, node]));
  const sources = new Map<string, GPUComputationSourceRef>();
  const lifetimes = new Map<string, {firstNode: string; lastNode: string}>();
  const summaries = new Map<string, {kind: GPUComputationResource['kind']; bytes: number}>();
  const edges = new Map<string, GPUComputationEdge>();
  const addEdge = (from: string, to: string, kind: GPUComputationEdgeKind, resource?: string) => {
    if (from === to) return;
    edges.set(`${from}\0${to}\0${kind}\0${resource ?? ''}`, {
      from,
      to,
      kind,
      ...(resource !== undefined && {resource})
    });
  };
  const lastWriter = new Map<string, string>();
  const readersSinceWrite = new Map<string, string[]>();
  const nodes: GPUComputationNode[] = [];

  for (const nodeId of compiled.stats.nodeOrder) {
    const node = preflightById.get(nodeId) as GPUCommandGraphNodePreflight | undefined;
    if (!node) continue;
    const record = recorder?.getNodeRecord(nodeId);
    if (record?.source) {
      const previous = sources.get(record.source.id);
      if (previous && previous.text !== record.source.text) {
        throw new Error(`Source id "${record.source.id}" has conflicting text`);
      }
      if (!previous) sources.set(record.source.id, record.source);
    }
    const reads = new Set<string>();
    const writes = new Set<string>();
    for (const resource of node.resources ?? []) {
      const id = resource.id;
      const usage: string = resource.usage;
      const isRead = READ_USAGES.has(usage);
      const isWrite = WRITE_USAGES.has(usage);
      if (isRead) reads.add(id);
      if (isWrite) writes.add(id);
      const summary = summaries.get(id);
      summaries.set(id, {
        kind: resource.kind,
        bytes: Math.max(summary?.bytes ?? 0, resource.byteLength ?? 0)
      });
      const lifetime = lifetimes.get(id);
      if (lifetime) lifetime.lastNode = nodeId;
      else lifetimes.set(id, {firstNode: nodeId, lastNode: nodeId});
      const writer = lastWriter.get(id);
      if (isRead && writer) addEdge(writer, nodeId, 'raw', id);
      if (isWrite) {
        for (const reader of readersSinceWrite.get(id) ?? []) addEdge(reader, nodeId, 'war', id);
        if (writer && !isRead) addEdge(writer, nodeId, 'waw', id);
      }
    }
    // Update hazard state after the node's own resource uses, so a node never depends on itself.
    for (const id of reads) {
      if (!writes.has(id))
        readersSinceWrite.set(id, [...(readersSinceWrite.get(id) ?? []), nodeId]);
    }
    for (const id of writes) {
      lastWriter.set(id, nodeId);
      readersSinceWrite.set(id, []);
    }
    for (const dependency of record?.dependsOn ?? []) addEdge(dependency, nodeId, 'explicit');

    const workload = getWorkload(node);
    nodes.push(
      omitUndefined({
        id: nodeId,
        label: record?.label ?? record?.dedupName ?? node.description ?? nodeId,
        namespace: record?.namespace ?? '',
        kind: node.type,
        op: record?.op ?? node.operation,
        datum: record?.datum,
        loop: record?.loop,
        dedupName: record?.dedupName,
        derivedFrom: record?.derivedFrom,
        formula: record?.formula,
        source: record?.source?.id,
        workgroupSize: record?.workgroupSize,
        bindings: record?.bindings,
        dispatch: record?.dispatch,
        reads: [...reads],
        writes: [...writes],
        workload,
        condition: node.condition
          ? {
              id: node.condition.id,
              source: node.condition.source,
              mode: node.condition.mode,
              ...(node.condition.bufferId !== undefined && {resource: node.condition.bufferId})
            }
          : undefined,
        publication: node.publication ? {...node.publication} : undefined,
        attrs: {...record?.attrs}
      }) as GPUComputationNode
    );
  }

  const resources: GPUComputationResource[] = [];
  for (const [id, summary] of summaries) {
    const handle = recorder?.getResource(id);
    const base = {id, kind: summary.kind, ...lifetimes.get(id)};
    if (handle && 'transient' in handle) {
      const isBuffer = 'byteLength' in handle;
      resources.push({
        ...base,
        bytes: isBuffer ? handle.byteLength : summary.bytes,
        ownership: handle.transient ? 'transient' : 'imported',
        ...(!isBuffer && {
          format: handle.format,
          attrs: {
            width: handle.width,
            height: handle.height,
            depth: handle.depth,
            mipLevels: handle.mipLevels,
            dimension: handle.dimension
          }
        })
      });
    } else {
      // Without the handle, ownership is unknown; imported is the conservative statement.
      resources.push({...base, bytes: summary.bytes, ownership: 'imported'});
    }
  }

  return {
    graph: {
      id: graphId,
      form: 'compiled',
      domain: options.domain ?? 'gpu-core',
      label: options.label ?? compiled.id,
      nodes,
      edges: [...edges.values()],
      resources
    },
    sources: [...sources.values()]
  };
}

/**
 * Groups encoded nodes into physical compute passes using the published encoding rule: consecutive
 * executed compute nodes share a pass when the encoding coalesced them, and every other node type
 * ends the pass. Returns `undefined` when the grouping disagrees with `stats.computePassCount`.
 */
export function getGPUCommandGraphPassIndices(
  stats: GPUCommandGraphEncodingStats
): Record<string, number> | undefined {
  const coalesced = stats.coalescedComputeNodeCount > 0;
  const indices: Record<string, number> = {};
  let passCount = 0;
  let open = false;
  for (const node of stats.nodes) {
    if (node.condition?.outcome === 'skipped') continue;
    if (node.type !== 'compute') {
      open = false;
      continue;
    }
    if (!open || !coalesced) passCount++;
    open = coalesced;
    indices[node.id] = passCount - 1;
  }
  return passCount === stats.computePassCount ? indices : undefined;
}

/**
 * Describes the realized passes of one encoding. Per-node pass membership is not published, so it
 * is inferred by {@link getGPUCommandGraphPassIndices}; when that disagrees with the encoding's
 * pass count, each compute node is listed in its own pass and the graph gets `passGrouping: 'unknown'`.
 */
export function describeGPUCommandGraphEncoding(
  stats: GPUCommandGraphEncodingStats,
  props: {compiledGraphId: string; graphId?: string}
): GPUComputationGraph {
  const graphId = props.graphId ?? `${props.compiledGraphId}-encoded`;
  validateGraphId(graphId);
  validateGraphId(props.compiledGraphId);
  const inferred = getGPUCommandGraphPassIndices(stats);
  const nodes: GPUComputationNode[] = [];
  const passes = new Map<number, GPUComputationNode>();
  const skipped: string[] = [];
  let fallbackPassIndex = 0;
  for (const node of stats.nodes) {
    if (node.condition?.outcome === 'skipped') {
      skipped.push(node.id);
      continue;
    }
    const derivedFrom = `${props.compiledGraphId}:${node.id}`;
    if (node.type === 'copy') {
      nodes.push({
        id: `copy-${node.id}`,
        label: node.id,
        namespace: '',
        kind: 'copy',
        derivedFrom: [derivedFrom],
        reads: [],
        writes: [],
        attrs: {}
      });
      continue;
    }
    const passIndex = node.type === 'compute' && inferred ? inferred[node.id] : fallbackPassIndex++;
    const previous = passes.get(passIndex);
    if (previous) {
      previous.derivedFrom = [...previous.derivedFrom!, derivedFrom];
      previous.attrs = {...previous.attrs, nodeCount: previous.derivedFrom.length};
    } else {
      const pass: GPUComputationNode = {
        id: `pass-${passIndex}`,
        kind: 'pass',
        label: `${node.type} pass ${passIndex}`,
        namespace: '',
        derivedFrom: [derivedFrom],
        reads: [],
        writes: [],
        attrs: {passType: node.type, nodeCount: 1}
      };
      passes.set(passIndex, pass);
      nodes.push(pass);
    }
  }
  return {
    id: graphId,
    form: 'encoded',
    domain: 'gpu-core',
    nodes,
    edges: [],
    attrs: {
      ...(inferred ? {} : {passGrouping: 'unknown'}),
      ...(skipped.length ? {skipped: skipped.join(',')} : {})
    }
  };
}

function getWorkload(node: GPUCommandGraphNodePreflight) {
  // Preflight fills unannotated nodes with zeros; only nodes naming an operation carry a workload.
  if (node.operation === undefined) return undefined;
  return {
    operation: node.operation,
    variant: node.variant,
    commandCount: node.commandCount,
    maximumWorkgroupCount: node.maximumWorkgroupCount,
    maximumInvocationCount: node.maximumInvocationCount,
    readByteLength: node.readByteLength,
    writeByteLength: node.writeByteLength
  };
}

function validateGraphId(id: string): void {
  // Graph ids are joined to node ids with ':' in derivedFrom references.
  if (id.includes(':')) throw new Error(`graph id "${id}" must not contain ':'`);
}

function omitUndefined<Value extends object>(value: Value): Value {
  return Object.fromEntries(
    Object.entries(value)
      .filter(([, entry]) => entry !== undefined)
      .map(([key, entry]) => [
        key,
        entry !== null && typeof entry === 'object' && !Array.isArray(entry)
          ? omitUndefined(entry)
          : entry
      ])
  ) as Value;
}
