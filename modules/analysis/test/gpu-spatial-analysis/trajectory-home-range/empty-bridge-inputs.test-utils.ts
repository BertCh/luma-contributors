// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {RecipeTestFixture} from '../recipes/recipe-harness';

/** Zero logical fixes, independently of the storage buffer's physical capacity. */
export function createEmptyBridgeInputs(fixture: RecipeTestFixture, trackOffsets: Uint32Array) {
  return {
    // Empty logical views still need nonzero storage backing for aligned binding preflight.
    // Poison the unused backing row to catch accidental reads without inventing a live fix.
    positions: fixture.input('positions', Float32Array.of(NaN, NaN), 'float32x2', 0),
    timestamps: fixture.input('timestamps', Float32Array.of(NaN), 'float32', 0),
    trackOffsets: fixture.input('offsets', trackOffsets, 'uint32', trackOffsets.length),
    motionVariances: fixture.input('motion', Float32Array.of(NaN), 'float32', 0),
    locationErrors: fixture.input('errors', Float32Array.of(NaN), 'float32', 0),
    parameters: fixture.input('parameters', Float32Array.of(0, 0, 1, 1), 'float32', 4)
  };
}
