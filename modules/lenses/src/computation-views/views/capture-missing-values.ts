// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  resolveGPUComputationCaptureGraph,
  type GPUComputationCapture
} from '@luma-contributors/analysis/computation-record';
import {getCaptureProducer} from '../registry/record-operator';
import type {LensRecord} from '../registry/view-registration';
import {getAllCaptures} from './record-queries';

/** Binding evidence survives renamed captures and caller-named backing resources. */
function getRasterJoinEvidence(record: LensRecord, capture: GPUComputationCapture) {
  if (capture.elementType !== 'u32' || !capture.resource) return undefined;
  const graph = resolveGPUComputationCaptureGraph(record, capture);
  const writers =
    graph?.nodes.filter(
      node =>
        node.op === 'GPURasterJoin' &&
        node.writes.includes(capture.resource!) &&
        node.bindings?.some(
          binding =>
            binding.resource === capture.resource &&
            binding.access !== 'read' &&
            [
              'pointZones',
              'counts',
              'outsideOut',
              'boundaryCounts',
              'pointBoundaryMask',
              'unassignedOut'
            ].includes(binding.name)
        )
    ) ?? [];
  const prefixes = new Set(
    writers.map(node => node.id.replace(/-(lookup|aggregate|tallies-publish)$/, ''))
  );
  return graph && writers.length
    ? {graph, prefix: prefixes.size === 1 ? [...prefixes][0] : undefined}
    : undefined;
}

/** Join result ports require their own producer's graph/frame completeness certificate. */
export function isUncertifiedRasterJoinCapture(
  record: LensRecord,
  capture: GPUComputationCapture,
  observation: GPUComputationCapture = capture
): boolean {
  const evidence = getRasterJoinEvidence(record, capture);
  if (!evidence) return false;
  const {graph, prefix} = evidence;
  if (
    prefix === undefined ||
    resolveGPUComputationCaptureGraph(record, observation) !== graph ||
    observation.iteration !== capture.iteration
  )
    return true;
  const statusResources = new Set(
    graph.nodes
      .filter(
        node =>
          node.op === 'RasterProducerCompleteness' &&
          (node.id === `${prefix}-producer-incomplete` ||
            (node.id.startsWith(`${prefix}-producer-incomplete-`) &&
              /^\d+$/.test(node.id.slice(`${prefix}-producer-incomplete-`.length))))
      )
      .flatMap(node =>
        (node.bindings ?? [])
          .filter(binding => binding.name === 'incomplete' && binding.access !== 'read')
          .map(binding => binding.resource)
      )
  );
  const certificates = getAllCaptures(record).filter(
    candidate =>
      candidate.resource !== undefined &&
      statusResources.has(candidate.resource) &&
      resolveGPUComputationCaptureGraph(record, candidate) === graph &&
      candidate.iteration === capture.iteration
  );
  return (
    certificates.length !== 1 ||
    certificates.some(
      certificate =>
        certificate.elementType !== 'u32' ||
        (certificate.components ?? 1) !== 1 ||
        certificate.values.length !== 1 ||
        certificate.values[0] !== 0
    )
  );
}

/** Resolve producer and certificate once, then interpret lanes without walking graph metadata. */
export function createCaptureValueInterpreter(record: LensRecord, capture: GPUComputationCapture) {
  const uncertified = isUncertifiedRasterJoinCapture(record, capture);
  let catchment = false;
  if (
    capture.elementType === 'u32' &&
    (capture.name === 'reachableFacilities' || capture.name === 'result.reachableFacilities')
  ) {
    const producer = getCaptureProducer(record, capture);
    catchment =
      producer?.op !== undefined
        ? producer.op === 'GPUCatchmentAccessibility'
        : capture.graph === undefined &&
          record.id === 'gpu-spatial-analysis.catchment-accessibility';
  }
  const isUnknown = (value: number) => uncertified || (catchment && value === 0xffffffff);
  return {isUnknown, interpret: (value: number) => (isUnknown(value) ? NaN : value)};
}

/** Producer-scoped unavailable values carry no numeric claim. Captured storage stays intact. */
export function isUnknownCaptureValue(
  record: LensRecord,
  capture: GPUComputationCapture,
  value: number
): boolean {
  return createCaptureValueInterpreter(record, capture).isUnknown(value);
}

/** Numeric readers use NaN for scoped missing lanes, without changing captured storage. */
export function getInterpretedCaptureValue(
  record: LensRecord,
  capture: GPUComputationCapture,
  value: number
): number {
  return createCaptureValueInterpreter(record, capture).interpret(value);
}
