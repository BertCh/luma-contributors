// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateGPUComputationRecord,
  validateGPUComputationSidecars,
  type GPUComputationRecord,
  type GPUComputationRecordWithSidecars
} from '@luma-contributors/analysis/computation-record';

/** Checks joins at loading and view boundaries, including legacy pairs without execution ids. */
export function assertLensSidecarIdentity(record: GPUComputationRecordWithSidecars): void {
  for (const sidecar of [
    ...(record.metrics ?? []),
    ...(record.captures ? [record.captures] : [])
  ]) {
    if (sidecar.runId !== record.runId) throw new Error('Lens sidecar runId does not match record');
    if (sidecar.executionId !== record.executionId)
      throw new Error('Lens sidecar executionId does not match record');
  }
}

/** Fails a scenario run whose record is malformed, so a plausible but wrong card never renders. */
export function assertValidLensRecord(record: GPUComputationRecord): void {
  const validation = validateGPUComputationRecord(record);
  if (!validation.valid) throw new Error(JSON.stringify(validation.issues));
}

/** Fails a scenario run whose capture or metric sidecars do not match their record. */
export function assertValidLensSidecars(
  record: GPUComputationRecord,
  sidecars: Parameters<typeof validateGPUComputationSidecars>[1]
): void {
  const validation = validateGPUComputationSidecars(record, sidecars);
  if (!validation.valid) throw new Error(JSON.stringify(validation.issues));
}
