// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUComputationNodeMetrics} from '@luma-contributors/analysis/computation-record';

/** A measured duration is evidence even when it is zero; it supplies no timeline endpoints. */
export function getMeasuredGPUTimeMs(measurement?: {gpuTimeMs?: number}): number | undefined {
  const duration = measurement?.gpuTimeMs;
  return duration !== undefined && Number.isFinite(duration) && duration >= 0
    ? duration
    : undefined;
}

/** Undefined means unmeasured; explicit skipped outcomes override timing evidence. */
export function getGPUExecutionEvidence(
  measurement?: Pick<GPUComputationNodeMetrics, 'outcome' | 'gpuTimeMs' | 'gpuBegin' | 'gpuEnd'>
): boolean | undefined {
  if (!measurement) return undefined;
  if (measurement.outcome === 'executed') return true;
  if (measurement.outcome === 'skipped') return false;
  if (getMeasuredGPUTimeMs(measurement) !== undefined) return true;
  const {gpuBegin, gpuEnd} = measurement;
  return gpuBegin !== undefined &&
    gpuEnd !== undefined &&
    Number.isFinite(gpuBegin) &&
    Number.isFinite(gpuEnd) &&
    gpuEnd >= gpuBegin
    ? true
    : undefined;
}
