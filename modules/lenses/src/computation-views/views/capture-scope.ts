// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  resolveGPUComputationCaptureGraph,
  type GPUComputationCapture
} from '@luma-contributors/analysis/computation-record';
import type {LensRecord} from '../registry/view-registration';
import {
  getAllCaptures,
  getCaptureFirstKey,
  getCaptureRowCount,
  getCaptureValue
} from './record-queries';

/** Resolve identity before name, within a uniquely attributed observation graph and frame. */
export function resolveCaptureCompanion(
  record: LensRecord,
  observation: GPUComputationCapture,
  reference?: string,
  allowStatic = true
): GPUComputationCapture | undefined {
  if (reference === undefined) return undefined;
  const graph = resolveGPUComputationCaptureGraph(record, observation);
  if (!graph) return undefined;
  const captures = getAllCaptures(record);
  const identities = captures.filter(candidate => candidate.id === reference);
  const candidates = (
    identities.length ? identities : captures.filter(candidate => candidate.name === reference)
  ).filter(
    candidate =>
      resolveGPUComputationCaptureGraph(record, candidate) === graph &&
      (candidate.iteration === observation.iteration ||
        (allowStatic && candidate.iteration === undefined))
  );
  const exactFrame = candidates.filter(candidate => candidate.iteration === observation.iteration);
  const applicable = exactFrame.length ? exactFrame : candidates;
  return applicable.length === 1 ? applicable[0] : undefined;
}

/** Shared count certificate and captured absolute-key window for point drawing and scales. */
export function resolvePointCaptureWindow(
  record: LensRecord,
  points: GPUComputationCapture
): {count: number; first: number; end: number} | undefined {
  const geometry = points.geometry;
  if (geometry?.kind !== 'points') return undefined;
  const first = getCaptureFirstKey(points);
  if (geometry.countCapture === undefined)
    return {count: getCaptureRowCount(points), first, end: first + getCaptureRowCount(points)};
  const graph = resolveGPUComputationCaptureGraph(record, points);
  if (!graph) return undefined;
  const certificate = resolveCaptureCompanion(record, points, geometry.countCapture, false);
  const count = certificate && getCaptureValue(certificate, getCaptureFirstKey(certificate));
  const resource = graph.resources?.find(
    candidate =>
      candidate.id === points.resource &&
      candidate.kind === 'buffer' &&
      candidate.bytesKind === 'logical'
  );
  const capacity = Math.min(
    record.axes.keySpaces.find(space => space.id === points.keySpace)?.size ?? Infinity,
    resource ? Math.floor(resource.bytes / ((points.components ?? 2) * 4)) : Infinity
  );
  if (
    !certificate ||
    getCaptureRowCount(certificate) !== 1 ||
    (certificate.components ?? 1) !== 1 ||
    count === undefined ||
    !Number.isSafeInteger(count) ||
    count < 0 ||
    count === 0xffffffff ||
    count > capacity
  )
    return undefined;
  return {count, first, end: Math.min(count, first + getCaptureRowCount(points))};
}
