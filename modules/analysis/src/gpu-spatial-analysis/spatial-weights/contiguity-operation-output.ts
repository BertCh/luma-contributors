// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';

/** @internal Named contiguity forms write three diagnostics; no metric or separate count port. */
export function validateContiguityOperationOutput(
  id: string,
  output: {status: object; distances?: unknown}
): void {
  validateRequirement(
    id,
    output.distances === undefined,
    'output.distances is unsupported by topology-only contiguity; attach a separately measured distance view after producing the CSR'
  );
  for (const name of Object.keys(output)) {
    validateRequirement(
      id,
      ['offsets', 'neighbors', 'weights', 'status'].includes(name),
      `output.${name} is unsupported by named contiguity; accepted destinations are offsets, neighbors, weights and status`
    );
  }
  for (const [name, value] of Object.entries(output.status)) {
    if (value !== undefined)
      validateRequirement(
        id,
        ['overflow', 'requiredCount', 'candidateOverflow'].includes(name),
        `output.status.${name} is unsupported by named contiguity; use terminal offsets for the retained count`
      );
  }
}
