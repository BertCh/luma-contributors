// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPU_COMPUTATION_METRICS_SCHEMA,
  GPU_COMPUTATION_CAPTURES_SCHEMA
} from './computation-record-types';
import type {
  GPUComputationMetrics,
  GPUComputationCaptures,
  GPUComputationEncodingFlags,
  GPUComputationAdapter,
  GPUComputationSeries,
  GPUComputationNodeMetrics
} from './computation-record-types';

/**
 * Structural subset of the published `GPUCommandGraphTimingReport`, so it can be passed directly.
 * `gpuStartTimeMilliseconds`/`gpuEndTimeMilliseconds` produce a `ms` timebase; the nanosecond
 * fields remain for timings measured outside luma.gl. Do not mix the two in one report.
 */
export type GPUComputationTimingInput = {
  nodes: readonly {
    id: string;
    cpuEncodeTimeMilliseconds: number;
    /** Measured duration in milliseconds, independent of endpoint availability. */
    gpuTimeMilliseconds?: number;
    gpuStartTimeMilliseconds?: number;
    gpuEndTimeMilliseconds?: number;
    gpuBeginNanoseconds?: number;
    gpuEndNanoseconds?: number;
    passIndex?: number;
    condition?: {outcome?: string};
  }[];
};

/** Builds relative endpoint metrics and preserves independently measured millisecond durations. */
export function createGPUComputationMetrics(props: {
  runId: string;
  executionId?: string;
  graph: string;
  timing: GPUComputationTimingInput;
  encoding: GPUComputationEncodingFlags;
  adapter?: GPUComputationAdapter;
  series?: readonly GPUComputationSeries[];
  counters?: Readonly<Record<string, Readonly<Record<string, number>>>>;
  /** Node id to physical pass index, from `getGPUCommandGraphPassIndices`. */
  passIndices?: Readonly<Record<string, number>>;
}): GPUComputationMetrics {
  const milliseconds = props.timing.nodes.some(node => node.gpuStartTimeMilliseconds !== undefined);
  const getBegin = (node: GPUComputationTimingInput['nodes'][number]) =>
    milliseconds ? node.gpuStartTimeMilliseconds : node.gpuBeginNanoseconds;
  const getEnd = (node: GPUComputationTimingInput['nodes'][number]) =>
    milliseconds ? node.gpuEndTimeMilliseconds : node.gpuEndNanoseconds;
  let origin = Infinity;
  for (const node of props.timing.nodes) {
    const begin = getBegin(node);
    const end = getEnd(node);
    if (
      begin !== undefined &&
      end !== undefined &&
      Number.isFinite(begin) &&
      Number.isFinite(end) &&
      end >= begin
    )
      origin = Math.min(origin, begin);
  }
  const nodes: Record<string, GPUComputationNodeMetrics> = Object.fromEntries(
    props.timing.nodes.map(node => {
      const outcome = node.condition?.outcome;
      const begin = getBegin(node);
      const end = getEnd(node);
      const validPair =
        begin !== undefined &&
        end !== undefined &&
        Number.isFinite(begin) &&
        Number.isFinite(end) &&
        end >= begin;
      const passIndex = node.passIndex ?? props.passIndices?.[node.id];
      return [
        node.id,
        {
          cpuEncodeMs: node.cpuEncodeTimeMilliseconds,
          ...(node.gpuTimeMilliseconds !== undefined &&
            Number.isFinite(node.gpuTimeMilliseconds) &&
            node.gpuTimeMilliseconds >= 0 && {gpuTimeMs: node.gpuTimeMilliseconds}),
          ...(validPair && {gpuBegin: begin! - origin}),
          ...(validPair && {gpuEnd: end! - origin}),
          ...(passIndex !== undefined && {passIndex}),
          ...(node.condition !== undefined && {
            outcome:
              outcome === 'skipped'
                ? 'skipped'
                : outcome === 'gpu-resolved' || outcome === 'indirect'
                  ? 'indirect'
                  : 'executed'
          }),
          ...(props.counters?.[node.id] !== undefined && {counters: props.counters[node.id]})
        }
      ];
    })
  );
  return {
    schema: GPU_COMPUTATION_METRICS_SCHEMA,
    runId: props.runId,
    ...(props.executionId !== undefined && {executionId: props.executionId}),
    graph: props.graph,
    timebase: milliseconds ? 'ms' : 'ns',
    encoding: props.encoding,
    nodes,
    ...(props.adapter !== undefined && {adapter: props.adapter}),
    ...(props.series !== undefined && {series: props.series})
  };
}

/** Builds a captures sidecar with empty arrays filled in. */
export function createGPUComputationCaptures(props: {
  runId: string;
  executionId?: string;
  snapshots?: GPUComputationCaptures['snapshots'];
  stamps?: GPUComputationCaptures['stamps'];
  replays?: GPUComputationCaptures['replays'];
}): GPUComputationCaptures {
  return {
    schema: GPU_COMPUTATION_CAPTURES_SCHEMA,
    runId: props.runId,
    ...(props.executionId !== undefined && {executionId: props.executionId}),
    snapshots: props.snapshots ?? [],
    stamps: props.stamps ?? [],
    replays: props.replays ?? []
  };
}
