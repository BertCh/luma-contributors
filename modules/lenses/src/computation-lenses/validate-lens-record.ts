// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateGPUComputationRecord,
  validateGPUComputationSidecars,
  type GPUComputationRecord
} from '@luma-contributors/analysis/computation-record';

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
