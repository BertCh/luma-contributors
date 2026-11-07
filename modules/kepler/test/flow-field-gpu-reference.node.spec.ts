// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {
  buildFlowFieldGrid,
  type FlowFieldColumnMode,
  type FlowFieldSamples
} from '../../../examples/deck/kepler-compute/src/flow-field-grid';
import {
  FLOW_FIELD_FIXED_POINT_LIMIT,
  buildFlowFieldGridReference,
  getFlowFieldExtent,
  getFlowFieldFilteredBounds,
  getFlowFieldFixedPoint
} from '../../../examples/deck/kepler-compute/src/flow-field-gpu-reference';
import {
  isFlowFieldVisible,
  planFlowField
} from '../../../examples/deck/kepler-compute/src/flow-field-plan';

function makeSamples(mode: FlowFieldColumnMode): FlowFieldSamples {
  const longitudes: number[] = [];
  const latitudes: number[] = [];
  const first: number[] = [];
  const second: number[] = [];
  // Deterministic jittered scatter with several samples per grid cell and a swirl/hill field.
  let seed = 12345;
  const random = () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 4294967296;
  };
  for (let index = 0; index < 4000; index++) {
    const longitude = -74 + random() * 0.4;
    const latitude = 40.5 + random() * 0.3;
    longitudes.push(longitude);
    latitudes.push(latitude);
    const dx = longitude + 73.8;
    const dy = latitude - 40.65;
    if (mode === 'UV') {
      first.push(-dy * 30 + random());
      second.push(dx * 30 + random());
    } else if (mode === 'SPEED_DIR') {
      first.push(2 + random() * 8);
      second.push(random() * 360);
    } else {
      first.push(300 - 900 * Math.hypot(dx, dy) * 30);
    }
  }
  return {longitudes, latitudes, first, second: mode === 'ELEVATION' ? undefined : second};
}

function expectGridsClose(
  reference: ReturnType<typeof buildFlowFieldGridReference>,
  cpu: ReturnType<typeof buildFlowFieldGrid>,
  tolerance: number
) {
  expect(reference).not.toBeNull();
  expect(cpu).not.toBeNull();
  if (!reference || !cpu) return;
  expect(reference.columns).toBe(cpu.columns);
  expect(reference.rows).toBe(cpu.rows);
  expect(reference.bounds).toEqual(cpu.bounds);
  expect(reference.texels.length).toBe(cpu.texels.length);
  let compared = 0;
  for (let index = 0; index < cpu.texels.length; index += 4) {
    // Validity must match exactly; velocities only differ by fixed-point quantization.
    expect(reference.texels[index + 2]).toBe(cpu.texels[index + 2]);
    if (!cpu.texels[index + 2]) continue;
    compared++;
    expect(Math.abs(reference.texels[index] - cpu.texels[index])).toBeLessThan(tolerance);
    expect(Math.abs(reference.texels[index + 1] - cpu.texels[index + 1])).toBeLessThan(tolerance);
  }
  expect(compared).toBeGreaterThan(20);
}

describe('GPU flow field grid design (CPU reference)', () => {
  for (const mode of ['UV', 'SPEED_DIR', 'ELEVATION'] as const) {
    test(`${mode} matches the native CPU grid within quantization`, () => {
      const samples = makeSamples(mode);
      for (const smoothing of [0, 1, 3]) {
        const options = {mode, gridResolution: 24, smoothing};
        const cpu = buildFlowFieldGrid(samples, options);
        const reference = buildFlowFieldGridReference(samples, options);
        // Elevation gradients divide by ~1e5 m cell sizes, so quantization error stays tiny.
        expectGridsClose(reference, cpu, mode === 'ELEVATION' ? 1e-4 : 5e-3);
      }
    });
  }

  test('a filter that keeps the extent corners matches the CPU grid built from the same selection', () => {
    const samples = makeSamples('UV');
    const extent = getFlowFieldExtent(samples)!;
    const selected = new Uint32Array(samples.longitudes.length);
    for (let index = 0; index < selected.length; index++)
      selected[index] = index % 3 === 0 || index % 7 === 0 ? 1 : 0;
    // Keep the rows that define the extent so CPU and GPU extents agree.
    for (let index = 0; index < selected.length; index++)
      if (
        samples.longitudes[index] === extent.bounds[0] ||
        samples.longitudes[index] === extent.bounds[2] ||
        samples.latitudes[index] === extent.bounds[1] ||
        samples.latitudes[index] === extent.bounds[3]
      )
        selected[index] = 1;
    const filtered = {...samples, selected};
    const options = {mode: 'UV' as const, gridResolution: 20, smoothing: 1};
    expectGridsClose(
      buildFlowFieldGridReference(filtered, options),
      buildFlowFieldGrid(filtered, options),
      5e-3
    );
    const unfiltered = buildFlowFieldGridReference(samples, options)!;
    const reference = buildFlowFieldGridReference(filtered, options)!;
    expect(reference.sampleCount).toBeLessThan(unfiltered.sampleCount);
    expect(reference.texels).not.toEqual(unfiltered.texels);
  });

  test('filtered bounds follow the selected rows while the cell counts stay those of the full extent', () => {
    const samples = makeSamples('UV');
    const selected = new Uint32Array(samples.longitudes.length);
    // Only a tight cluster is selected.
    for (let index = 0; index < selected.length; index++)
      selected[index] =
        samples.longitudes[index] < -73.9 && samples.latitudes[index] < 40.6 ? 1 : 0;
    const options = {mode: 'UV' as const, gridResolution: 16, smoothing: 0};
    const reference = buildFlowFieldGridReference({...samples, selected}, options)!;
    const full = buildFlowFieldGridReference(samples, options)!;
    // Same texture size, so the GPU never reallocates for a filter change.
    expect(reference.columns).toBe(full.columns);
    expect(reference.rows).toBe(full.rows);
    // Bounds shrink to the selected rows, like native (and the CPU grid's bounds).
    expect(reference.bounds).toEqual(getFlowFieldFilteredBounds({...samples, selected}));
    expect(reference.bounds).toEqual(buildFlowFieldGrid({...samples, selected}, options)!.bounds);
    expect(reference.bounds[2]).toBeLessThan(full.bounds[2]);
    expect(reference.bounds[3]).toBeLessThan(full.bounds[3]);
    // The tighter bounds fill (almost) the whole texture instead of a corner of it.
    const filledCount = (grid: {texels: Float32Array}) =>
      grid.texels.filter((_, index) => index % 4 === 2 && grid.texels[index] > 0.5).length;
    expect(filledCount(reference)).toBeGreaterThan(filledCount(full) / 2);
  });

  test('speed is normalized into the texel w channel and altitude is a smoothed cell mean', () => {
    const samples = {
      ...makeSamples('UV'),
      altitude: makeSamples('UV').first.map((value, index) => 100 + (index % 10))
    };
    const reference = buildFlowFieldGridReference(samples, {
      mode: 'UV',
      gridResolution: 12,
      smoothing: 1
    })!;
    let minimum = Infinity;
    let maximum = -Infinity;
    for (let cell = 0; cell < reference.columns * reference.rows; cell++) {
      if (reference.texels[cell * 4 + 2] < 0.5) continue;
      const normalized = reference.texels[cell * 4 + 3];
      expect(normalized).toBeGreaterThanOrEqual(0);
      expect(normalized).toBeLessThanOrEqual(1);
      minimum = Math.min(minimum, normalized);
      maximum = Math.max(maximum, normalized);
    }
    expect(minimum).toBe(0);
    expect(maximum).toBe(1);
    expect(reference.scaledTexels[3]).toBe(reference.texels[3]);
    expect(reference.altitude).toBeDefined();
    for (const value of reference.altitude!) {
      if (Number.isNaN(value)) continue;
      expect(value).toBeGreaterThanOrEqual(100 - 1e-3);
      expect(value).toBeLessThanOrEqual(109 + 1e-3);
    }
    // ELEVATION mode uses the first column as its altitude.
    const elevation = buildFlowFieldGridReference(makeSamples('ELEVATION'), {
      mode: 'ELEVATION',
      gridResolution: 12,
      smoothing: 0
    })!;
    expect(elevation.altitude).toBeDefined();
    expect(
      buildFlowFieldGridReference(makeSamples('UV'), {
        mode: 'UV',
        gridResolution: 12,
        smoothing: 0
      })!.altitude
    ).toBeUndefined();
  });

  test('fully filtered or non-finite input yields no field', () => {
    const samples = makeSamples('UV');
    const none = new Uint32Array(samples.longitudes.length);
    const reference = buildFlowFieldGridReference(
      {...samples, selected: none},
      {mode: 'UV', gridResolution: 16, smoothing: 0}
    );
    // No selected rows cannot form a grid: the GPU path shows an invalid (empty) field.
    expect(reference).toBeNull();
    expect(
      getFlowFieldExtent({
        longitudes: [0, 1, 2, NaN],
        latitudes: [0, 1, 2, 3],
        first: [1, 1, 1, 1],
        second: [1, 1, 1, 1]
      })
    ).toBeNull();
  });

  test('fixed-point accumulation cannot overflow int32', () => {
    for (const mode of ['UV', 'SPEED_DIR', 'ELEVATION'] as const) {
      const extent = getFlowFieldExtent(makeSamples(mode))!;
      for (const rowCount of [4000, 5_000_000, 100_000_000]) {
        const fixedPoint = getFlowFieldFixedPoint(mode, {...extent, validCount: rowCount});
        const halfFirst =
          mode === 'SPEED_DIR'
            ? extent.maximumMagnitude
            : (extent.firstRange[1] - extent.firstRange[0]) / 2;
        // Worst case: every row lands in one cell at the largest deviation from the center.
        expect(rowCount * halfFirst * fixedPoint.scaleFirst).toBeLessThanOrEqual(
          FLOW_FIELD_FIXED_POINT_LIMIT * 1.0001
        );
        expect(fixedPoint.scaleFirst).toBeGreaterThan(0);
      }
    }
  });

  test('GPU-resolved visual scale equals the CPU scale', () => {
    const samples = makeSamples('UV');
    const options = {mode: 'UV' as const, gridResolution: 24, smoothing: 2};
    const plan = planFlowField({}, {bounds: getFlowFieldExtent(samples)!.bounds});
    expect(plan.velocityScale).toBe(1);
    expect(plan.velocityScaleNumerator).toBeGreaterThan(0);
    const reference = buildFlowFieldGridReference(samples, options, plan.velocityScaleNumerator)!;
    const cpuScale = planFlowField(
      {},
      {bounds: reference.bounds, maximumSpeed: reference.maximumSpeed}
    ).velocityScale;
    expect(reference.velocityScale).toBeCloseTo(cpuScale, 8);
    let fastest = 0;
    for (let index = 0; index < reference.scaledTexels.length; index += 4)
      fastest = Math.max(
        fastest,
        Math.hypot(reference.scaledTexels[index], reference.scaledTexels[index + 1])
      );
    expect(fastest * 1).toBeCloseTo(plan.velocityScaleNumerator, 3);
    const physical = planFlowField({}, {bounds: reference.bounds}, 'physical');
    expect(physical.velocityScaleNumerator).toBe(0);
    expect(buildFlowFieldGridReference(samples, options, 0)!.velocityScale).toBe(1);
  });

  test('viewport culling intersects the field bounds, including world copies', () => {
    const field: [number, number, number, number] = [-74, 40, -73, 41];
    expect(isFlowFieldVisible([-74.5, 40.2, -73.5, 40.8], field)).toBe(true);
    expect(isFlowFieldVisible([-60, 40.2, -50, 40.8], field)).toBe(false);
    expect(isFlowFieldVisible([-74.5, 50, -73.5, 52], field)).toBe(false);
    expect(isFlowFieldVisible([285.5, 40.2, 286.5, 40.8], field)).toBe(true);
    expect(isFlowFieldVisible([-180, -80, 180, 80], field)).toBe(true);
  });
});
