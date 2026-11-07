// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import {vec3} from '@math.gl/core';

/** Asserts that a 3-vector has unit length within 1e-6. Non-finite lengths fail. */
export function assertUnitVector(vector: ArrayLike<number>): void {
  // Expected a finite unit-length direction vector
  assert(Math.abs(vec3.length(vector as number[]) - 1) < 1e-6);
}
