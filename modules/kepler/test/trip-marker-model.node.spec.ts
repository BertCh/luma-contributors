// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {
  findTripMarkerModel,
  makeTripMarkerBaseRotation
} from '../../../examples/deck/kepler-compute/src/trip-marker-model';

describe('trip marker models', () => {
  test('identity adjustments leave the heading rotation untouched', () => {
    expect(makeTripMarkerBaseRotation({}, [0, 0, 0])).toEqual([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  });

  test('adjust yaw is negated like native toQuaternion(roll, -pitch, -yaw)', () => {
    // Rz(-90): x axis maps to -y, y axis maps to +x (column-major columns).
    const rotation = makeTripMarkerBaseRotation({adjustYaw: 90}, [0, 0, 0]);
    expect(rotation[0]).toBeCloseTo(0);
    expect(rotation[1]).toBeCloseTo(-1);
    expect(rotation[3]).toBeCloseTo(1);
    expect(rotation[4]).toBeCloseTo(0);
    expect(rotation[8]).toBeCloseTo(1);
  });

  test('model angles compose after the adjustments', () => {
    const rotation = makeTripMarkerBaseRotation({adjustRoll: 90}, [0, 0, 90]);
    // Rx(90) * Rz(90): x -> y -> z.
    expect(rotation[0]).toBeCloseTo(0);
    expect(rotation[1]).toBeCloseTo(0);
    expect(rotation[2]).toBeCloseTo(1);
  });

  test('finds gallery and custom native models', () => {
    expect(findTripMarkerModel({scenegraph: 'car'})?.url).toMatch(/Car\.glb$/);
    expect(
      findTripMarkerModel({scenegraph: 'custom', scenegraphCustomModelUrl: 'https://x/m.glb'})?.url
    ).toBe('https://x/m.glb');
    expect(findTripMarkerModel({scenegraph: 'custom'})).toBeNull();
    expect(findTripMarkerModel({scenegraph: 'missing'})).toBeNull();
  });
});
