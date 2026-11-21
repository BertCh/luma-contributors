// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getPackedUint32Values,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

/** A numeric uint32 view or exact byte readback of status words. */
export type RegressionStatusReadback = Uint32Array | Uint8Array | ArrayBuffer;

/** Capture a real status word before interpreting model coefficients. @internal */
export function validateRegressionStatusReadback(
  reader: string,
  status: RegressionStatusReadback,
  path = 'status'
): number {
  try {
    return getPackedUint32Values(status, 1)[0];
  } catch (error) {
    validateRequirement(
      reader,
      false,
      `${path} must hold at least one uint32 code in whole 4-byte elements; ${error instanceof Error ? error.message : String(error)}`
    );
    throw error;
  }
}
