// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CPUAZPRegionsInput} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-azp-regions-operation';
import type {AZPRegionsOperationProps} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-azp-regions-operation';

// Oracle literals: regionalization-audit-azp.oracle.py. Directed path plus an island;
// unused label slot four; constant second column tests the zero population deviation rule.
export function createAuditAZPInput(scale = 1, shift = 0): CPUAZPRegionsInput {
  return {
    offsets: Uint32Array.of(0, 1, 2, 3, 4, 5, 6, 7, 7, 7),
    neighbors: Uint32Array.of(1, 2, 3, 4, 5, 6, 7),
    weights: Float32Array.of(2, 0, -1, 7, 3, 9, 2),
    values: Float32Array.from(
      [0, 2, 8, 10, 12, 14, 30, 32, 100].flatMap(value => [value * scale + shift, 7])
    ),
    initialLabels: Uint32Array.of(0, 0, 0, 1, 1, 1, 2, 2, 3),
    extensiveValues: new Float32Array(9).fill(1),
    extensiveThreshold: Float32Array.of(2),
    parameters: Uint32Array.of(0)
  };
}

export function createAuditAZPProps(): AZPRegionsOperationProps {
  return {
    rowCount: 9,
    weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    values: 'values',
    columnCount: 2,
    standardize: false,
    extensiveValues: 'extensiveValues',
    extensiveThreshold: 'extensiveThreshold',
    initialLabels: 'initialLabels',
    regionCapacity: 5,
    maximumIterations: 8,
    parameters: 'parameters',
    output: {labels: 'labels', objective: 'objective', status: 'status'}
  };
}
