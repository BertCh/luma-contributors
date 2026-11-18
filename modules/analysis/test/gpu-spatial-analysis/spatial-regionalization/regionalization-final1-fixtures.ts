// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** A path whose region boundary needs two separate AZP rounds to reach the singleton ten. */
export function createRegionalizationBudgetPath() {
  return {
    offsets: Uint32Array.of(0, 1, 3, 5, 7, 9, 10),
    neighbors: Uint32Array.of(1, 0, 2, 1, 3, 2, 4, 3, 5, 4),
    weights: new Float32Array(10).fill(1),
    values: Float32Array.of(0, 0, 0, 0, 0, 10),
    initialLabels: Uint32Array.of(0, 0, 0, 1, 1, 1),
    parameters: Uint32Array.of(0)
  };
}

/** Dropping edge 0-1 changes the best raw Ward merge from cost 0.5 to 40.5. */
export function createWardLossPath(missingEdge: boolean) {
  return {
    offsets: missingEdge ? Uint32Array.of(0, 0, 1, 2) : Uint32Array.of(0, 1, 3, 4),
    neighbors: missingEdge ? Uint32Array.of(2, 1) : Uint32Array.of(1, 0, 2, 1),
    weights: new Float32Array(missingEdge ? 2 : 4).fill(1),
    values: Float32Array.of(0, 1, 10),
    parameters: Uint32Array.of(0)
  };
}

/** Seed 2 plants the only complete seed at row 0; one enclave step can reach row 1, not row 2. */
export function createMaxPEnclaveBudgetPath() {
  return {
    offsets: Uint32Array.of(0, 1, 3, 5, 6),
    neighbors: Uint32Array.of(1, 0, 2, 1, 3, 2),
    weights: new Float32Array(6).fill(1),
    values: Float32Array.of(0, 1, 2, 3),
    extensiveValues: Float32Array.of(1, 0, 0, 0),
    threshold: Float32Array.of(1),
    parameters: Uint32Array.of(2)
  };
}
