// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../utils/require-test-device';
import {createMedialAxisFixture} from './medial-axis-fixture';
import {computeMedialAxisReference, computeFilledRectangleReference} from './medial-axis-oracle';

it('medial axis matches complete masks and radii for every binary 3x3 image in one resident atlas', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // A background frame isolates each tile, including its exterior-background semantics.
  const width = 32 * 5;
  const height = 16 * 5;
  const values = new Uint32Array(width * height);
  const expectedAxis = Array(width * height).fill(0);
  const expectedRadius = Array(width * height).fill(0);
  for (let bits = 0; bits < 512; bits++) {
    const tile = Array.from({length: 9}, (_, cell) => ((bits >>> cell) & 1) * 7);
    const reference = computeMedialAxisReference(3, 3, tile);
    for (let cell = 0; cell < 9; cell++) {
      const column = (bits % 32) * 5 + 1 + (cell % 3);
      const row = Math.floor(bits / 32) * 5 + 1 + Math.floor(cell / 3);
      const target = row * width + column;
      values[target] = tile[cell];
      expectedAxis[target] = reference.medialAxis[cell];
      expectedRadius[target] = reference.radius[cell];
    }
  }
  const fixture = createMedialAxisFixture(device, width, height, values);
  try {
    fixture.submit();
    const actual = await fixture.read();
    expect(actual.medialAxis).toEqual(expectedAxis);
    expect(actual.radius).toEqual(expectedRadius);
    expect(actual.axisTail).toEqual([77, 77, 77]);
    expect(actual.radiusTail).toEqual([77, 77, 77]);
    fixture.update(new Uint32Array(width * height));
    fixture.submit();
    const cleared = await fixture.read();
    expect(cleared.medialAxis.every(value => value === 0)).toBe(true);
    expect(cleared.radius.every(value => value === 0)).toBe(true);
  } finally {
    fixture.destroy();
  }
});

it('medial axis processes a complete filled rectangle larger than 16384 cells', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const width = 257;
  const height = 193;
  const fixture = createMedialAxisFixture(
    device,
    width,
    height,
    new Uint32Array(width * height).fill(1)
  );
  try {
    fixture.submit();
    const actual = await fixture.read();
    const reference = computeFilledRectangleReference(width, height);
    expect(actual.medialAxis).toEqual(reference.medialAxis);
    expect(actual.radius).toEqual(reference.radius);
  } finally {
    fixture.destroy();
  }
});

it('medial axis handles primitive diagonal rays and exact tangent containment', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const width = 129;
  const height = 129;
  const values = new Uint32Array(width * height).fill(1);
  values[40 * width + 40] = 0;
  const fixture = createMedialAxisFixture(device, width, height, values);
  try {
    fixture.submit();
    const actual = await fixture.read();
    // p=(63,63), b=(40,40), q=(64,64): r(p)=23sqrt(2), r(q)=24sqrt(2).
    // Rounded f32 addition can reject this exact containment; integer squares cannot.
    const focus = 63 * width + 63;
    expect(actual.radius[focus]).toBe(Math.fround(Math.sqrt(1058)));
    expect(actual.medialAxis[focus]).toBe(0);
    expect(actual.radius[40 * width + 40]).toBe(0);
    expect(actual.medialAxis[40 * width + 40]).toBe(0);
  } finally {
    fixture.destroy();
  }
});
