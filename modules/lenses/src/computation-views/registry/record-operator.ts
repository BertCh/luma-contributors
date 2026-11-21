// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  resolveGPUComputationCaptureGraph,
  type GPUComputationCapture
} from '@luma-contributors/analysis/computation-record';
import {getLensScenario} from '../../computation-lenses/registry-implementation';
import type {LensRecord, ViewContext} from './view-registration';

/** A capture's explicit observation node or unique writer within its own graph. */
export function getCaptureProducer(record: LensRecord, capture: GPUComputationCapture) {
  const graph = resolveGPUComputationCaptureGraph(record, capture);
  const observed = graph?.nodes.find(node => node.id === capture.afterNode);
  if (!capture.resource) return observed;
  const writers = graph?.nodes.filter(node => node.writes.includes(capture.resource!)) ?? [];
  if (writers.length === 1) return writers[0];
  return observed && writers.includes(observed) ? observed : undefined;
}

/** Resolve an explicit host element, the registered scenario, or a capture's recorded producer. */
export function getRecordOperator(context: ViewContext, name?: string): string | undefined {
  const explicit = context.options?.['element'];
  if (typeof explicit === 'string') return explicit;
  const scenario = getLensScenario(context.record.id);
  if (scenario) return scenario.element;
  const capture = [
    ...(context.record.captures?.snapshots ?? []),
    ...(context.record.captures?.stamps ?? []),
    ...(context.record.captures?.replays ?? [])
  ].find(candidate => candidate.name === name || candidate.id === name);
  const producer = capture && getCaptureProducer(context.record, capture);
  if (producer?.op) return producer.op;
  const operators = new Set(
    context.record.graphs.flatMap(candidate =>
      candidate.nodes.flatMap(node => (node.op ? [node.op] : []))
    )
  );
  return operators.size === 1 ? [...operators][0] : undefined;
}
