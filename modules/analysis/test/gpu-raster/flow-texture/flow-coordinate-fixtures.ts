// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';
import type {createParityGraph} from '../../oracles/parity-wp2-geodesic-harness';

type CoordinateRig = ReturnType<typeof createParityGraph>;

/** Mutable particle state that is read back after submission. */
export function createCoordinateParticleState(rig: CoordinateRig) {
  const positions = rig.output('positions', 1, 'float32x2');
  const ages = rig.output('ages', 1, 'uint32');
  const generations = rig.output('generations', 1, 'uint32');
  positions.buffer.write(Float32Array.of(50, 50));
  ages.buffer.write(Uint32Array.of(0));
  generations.buffer.write(Uint32Array.of(0));
  return {positions, ages, generations};
}

/** Reused CSR storage with nonzero stale sentinels. */
export function createCoordinateOffsets(rig: CoordinateRig) {
  const offsets = rig.output('offsets', 5, 'uint32');
  offsets.buffer.write(Uint32Array.of(99, 99, 99, 99, 99));
  return offsets;
}

/** Independent straight-line fixture assertions shared by CPU and GPU checks. */
export function expectCoordinateStreamline(values: ArrayLike<number>) {
  expect(values.length).toBe(34);
  // Independent Python integer Philox-4x32-10, counter [0,0,0,0], key [8,2]:
  // words [1295583915,1141723941,473967449,1081980456], top 24 bits / 2^24 * 100.
  const seedX = 30.165159702301025;
  const seedY = 26.582831144332886;
  for (let point = 0; point < 17; point++) {
    expect(Number.isFinite(values[2 * point])).toBe(true);
    expect(Number.isFinite(values[2 * point + 1])).toBe(true);
    expect(values[2 * point]).toBeCloseTo(Math.fround(seedX + (point - 8) * 0.25), 5);
    expect(values[2 * point + 1]).toBeCloseTo(Math.fround(seedY), 5);
  }
}
