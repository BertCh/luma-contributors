// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateCapacity} from '../kernels/gpu-contributor-utils';

/** WebGPU's default binding size; requesting less would reduce portable analysis capacity. */
export const ANALYSIS_DEFAULT_STORAGE_BUFFER_BINDING_SIZE = 128 * 1024 * 1024;
/** WebGPU's default buffer size; larger allocations require an explicit device request. */
export const ANALYSIS_DEFAULT_BUFFER_SIZE = 256 * 1024 * 1024;
/** WebGPU's default storage binding count, the floor for analysis device requests. */
export const ANALYSIS_DEFAULT_STORAGE_BUFFERS_PER_SHADER_STAGE = 8;
/** Sixteen bindings cover the analysis profile without demanding every adapter binding. */
export const ANALYSIS_TARGET_STORAGE_BUFFERS_PER_SHADER_STAGE = 16;
/** WebGPU's default workgroup memory; optional requests must never lower it. */
export const ANALYSIS_DEFAULT_COMPUTE_WORKGROUP_STORAGE_SIZE = 16384;
/** WebGPU's default workgroup invocation count; optional requests must never lower it. */
export const ANALYSIS_DEFAULT_COMPUTE_INVOCATIONS_PER_WORKGROUP = 256;
/** Byte sizes and requested counts must remain exact JavaScript integers. */
const ANALYSIS_MAXIMUM_SAFE_INTEGER = Number.MAX_SAFE_INTEGER;

/** Structural subset accepted from either native adapter limits or luma.gl device limits. */
export type AnalysisDeviceLimits = {
  maxStorageBuffersPerShaderStage: number;
  maxStorageBufferBindingSize: number;
  maxBufferSize: number;
  maxComputeWorkgroupStorageSize?: number;
  maxComputeInvocationsPerWorkgroup?: number;
};

/** Optional targets, clamped to adapter support and floored at WebGPU defaults. */
export type AnalysisRequiredLimitsOptions = {
  /** Defaults to {@link ANALYSIS_TARGET_STORAGE_BUFFERS_PER_SHADER_STAGE}. */
  requestedStorageBuffersPerShaderStage?: number;
  /** Request only when a contributor needs additional workgroup memory. */
  maxComputeWorkgroupStorageSize?: number;
  /** Request only when a contributor needs additional workgroup invocations. */
  maxComputeInvocationsPerWorkgroup?: number;
};

/**
 * Returns `requiredLimits` for `luma.createDevice`, raising buffer sizes to the adapter maxima.
 * Pass the adapter's limits before creating the device: an existing device cannot be upgraded.
 * A missing adapter returns an empty record so hosts can retain their normal fallback path.
 * Optional compute targets are capped at support; contributors must still check device limits.
 */
export function getAnalysisRequiredLimits(
  adapterLimits?: AnalysisDeviceLimits | null,
  options: AnalysisRequiredLimitsOptions = {}
): Record<string, number> {
  for (const [name, value] of Object.entries(options)) {
    if (value !== undefined) {
      validateCapacity('analysis-device-limits', name, value, 1, ANALYSIS_MAXIMUM_SAFE_INTEGER);
    }
  }
  if (!adapterLimits) {
    return {};
  }
  const requiredLimits: Record<string, number> = {
    maxStorageBuffersPerShaderStage: getAnalysisLimitTarget(
      adapterLimits.maxStorageBuffersPerShaderStage,
      ANALYSIS_DEFAULT_STORAGE_BUFFERS_PER_SHADER_STAGE,
      options.requestedStorageBuffersPerShaderStage ??
        ANALYSIS_TARGET_STORAGE_BUFFERS_PER_SHADER_STAGE
    ),
    maxStorageBufferBindingSize: getAnalysisLimitTarget(
      adapterLimits.maxStorageBufferBindingSize,
      ANALYSIS_DEFAULT_STORAGE_BUFFER_BINDING_SIZE
    ),
    maxBufferSize: getAnalysisLimitTarget(adapterLimits.maxBufferSize, ANALYSIS_DEFAULT_BUFFER_SIZE)
  };
  if (options.maxComputeWorkgroupStorageSize !== undefined) {
    requiredLimits['maxComputeWorkgroupStorageSize'] = getAnalysisLimitTarget(
      adapterLimits.maxComputeWorkgroupStorageSize,
      ANALYSIS_DEFAULT_COMPUTE_WORKGROUP_STORAGE_SIZE,
      options.maxComputeWorkgroupStorageSize
    );
  }
  if (options.maxComputeInvocationsPerWorkgroup !== undefined) {
    requiredLimits['maxComputeInvocationsPerWorkgroup'] = getAnalysisLimitTarget(
      adapterLimits.maxComputeInvocationsPerWorkgroup,
      ANALYSIS_DEFAULT_COMPUTE_INVOCATIONS_PER_WORKGROUP,
      options.maxComputeInvocationsPerWorkgroup
    );
  }
  return requiredLimits;
}

/**
 * Largest whole row count that fits one packed storage binding and its backing buffer.
 * Pass `device.limits`, not adapter limits, when warning before a run. Include padding and
 * per-row scratch in `bytesPerRow` for the largest binding. This is a byte ceiling only:
 * contributor-specific algorithm limits, buffer offsets, and total device memory still apply.
 */
export function getAnalysisMaximumRowCount(
  deviceLimits: Pick<AnalysisDeviceLimits, 'maxStorageBufferBindingSize' | 'maxBufferSize'>,
  bytesPerRow: number
): number {
  validateCapacity(
    'analysis-device-limits',
    'bytesPerRow',
    bytesPerRow,
    1,
    ANALYSIS_MAXIMUM_SAFE_INTEGER
  );
  const byteLength = Math.min(deviceLimits.maxStorageBufferBindingSize, deviceLimits.maxBufferSize);
  validateCapacity(
    'analysis-device-limits',
    'device binding bytes',
    byteLength,
    0,
    ANALYSIS_MAXIMUM_SAFE_INTEGER
  );
  return Math.floor(byteLength / bytesPerRow);
}

/** Floors malformed or undersized adapter limits at the WebGPU default. */
function getAnalysisLimitTarget(
  adapterLimit: number | undefined,
  defaultLimit: number,
  requestedLimit?: number
): number {
  const supportedLimit =
    adapterLimit !== undefined && Number.isSafeInteger(adapterLimit)
      ? Math.max(defaultLimit, adapterLimit)
      : defaultLimit;
  return requestedLimit === undefined
    ? supportedLimit
    : Math.max(defaultLimit, Math.min(supportedLimit, requestedLimit));
}
