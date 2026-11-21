// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUGreatCircleArcs,
  GPULineSegmentize,
  getGPUGreatCircleArcsParameterValues,
  getGPULineSegmentizeParameterValues
} from '../../../src/gpu-spatial-analysis/line-segmentize/index';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createLineLineFixture} from './line-test-utils';

for (const route of ['arcs', 'segmentize'] as const) {
  it(`${route} keeps both polar half-turn branches continuous at the endpoint (GEO-121)`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createLineLineFixture(device, {
      capacity: 10,
      lineCapacity: 2,
      inputs: {
        sources: {values: Float32Array.of(0, 80, 0, 80), format: 'float32x2'},
        targets: {values: Float32Array.of(180, 80, -180, 80), format: 'float32x2'},
        positions: {values: Float32Array.of(0, 80, 180, 80, 0, 80, -180, 80), format: 'float32x2'},
        offsets: {values: Uint32Array.of(0, 2, 4), format: 'uint32'}
      },
      createContributor: (inputs, parameters, output) =>
        route === 'arcs'
          ? new GPUGreatCircleArcs({
              sources: inputs['sources'] as GraphDataView<'float32x2'>,
              targets: inputs['targets'] as GraphDataView<'float32x2'>,
              maximumSegmentCount: 4,
              parameters,
              output
            })
          : new GPULineSegmentize({
              positions: inputs['positions'] as GraphDataView<'float32x2'>,
              lineOffsets: inputs['offsets'] as GraphDataView<'uint32'>,
              spatialContext: {
                coordinateSpace: 'longitude-latitude',
                metric: 'great-circle',
                sphereRadius: 1
              },
              maximumPieceCount: 4,
              parameters,
              output
            })
    });
    try {
      const result = await fixture.run(
        route === 'arcs'
          ? getGPUGreatCircleArcsParameterValues({maximumSegmentLength: 0, minimumSegments: 4})
          : // The path is exactly 20 degrees: dividing by 3.9 robustly selects four pieces.
            getGPULineSegmentizeParameterValues({maximumSegmentLength: (20 * Math.PI) / 180 / 3.9})
      );
      expect(result.lineOffsets).toEqual([0, 5, 10]);
      expect(result.positions).toHaveLength(10);
      for (let line = 0; line < 2; line++) {
        const positions = result.positions.slice(5 * line, 5 * line + 5);
        // Hand geometry: two meridians meet at the north pole, 5 degrees per piece.
        // Pole longitude has no unique value and is deliberately not asserted.
        for (const [index, latitude] of [80, 85, 90, 85, 80].entries()) {
          expect(Number.isFinite(positions[index][1])).toBe(true);
          expect(positions[index][1]).toBeCloseTo(latitude, 3);
        }
        for (const index of [0, 1, 3, 4]) expect(Number.isFinite(positions[index][0])).toBe(true);
        expect(positions[0][0]).toBe(0);
        expect(positions[1][0]).toBeCloseTo(0, 3);
        expect(Math.abs(positions[3][0])).toBeCloseTo(180, 3);
        expect(Math.abs(positions[4][0])).toBeCloseTo(180, 3);
        // Either sign at the half-turn describes the meridian, but a 360-degree final jump
        // violates the documented unwrapped continuity. This assertion discriminates the lead.
        expect(Math.abs(positions[4][0] - positions[3][0])).toBeLessThan(0.01);
      }
    } finally {
      fixture.destroy();
    }
  });
}

it('great-circle arcs emit NaN interiors at the antipodal cutoff and finite control coordinates (GEO-120)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createLineLineFixture(device, {
    capacity: 10,
    lineCapacity: 2,
    inputs: {
      sources: {values: Float32Array.of(0, 0, 0, 0), format: 'float32x2'},
      targets: {values: Float32Array.of(180, 0, 179, 0), format: 'float32x2'}
    },
    createContributor: (inputs, parameters, output) =>
      new GPUGreatCircleArcs({
        sources: inputs['sources'] as GraphDataView<'float32x2'>,
        targets: inputs['targets'] as GraphDataView<'float32x2'>,
        maximumSegmentCount: 4,
        parameters,
        output
      })
  });
  try {
    const result = await fixture.run(
      getGPUGreatCircleArcsParameterValues({
        maximumSegmentLength: 0,
        minimumSegments: 4
      })
    );
    expect(result.lineOffsets).toEqual([0, 5, 10]);
    expect(result.positions).toHaveLength(10);
    expect(result.positions[0]).toEqual([0, 0]);
    expect(Number.isFinite(result.positions[4][0])).toBe(true);
    expect(Math.abs(result.positions[4][0])).toBe(180);
    expect(result.positions[4][1]).toBe(0);
    for (const vertex of result.positions.slice(1, 4)) {
      expect(vertex[0]).toBeNaN();
      expect(vertex[1]).toBeNaN();
    }
    // Equatorial control: pi - angle = pi/180 radians, safely above 1e-6.
    for (const [index, longitude] of [0, 44.75, 89.5, 134.25, 179].entries()) {
      const vertex = result.positions[5 + index];
      expect(vertex.every(Number.isFinite)).toBe(true);
      expect(vertex[0]).toBeCloseTo(longitude, 3);
      expect(vertex[1]).toBeCloseTo(0, 3);
    }
  } finally {
    fixture.destroy();
  }
});

it('great-circle cutoff distinguishes representable targets on both sides of 1e-6 radians (S2)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createLineLineFixture(device, {
    capacity: 10,
    lineCapacity: 2,
    inputs: {
      sources: {values: Float32Array.of(0, 0, 0, 0), format: 'float32x2'},
      targets: {values: Float32Array.of(179.99998, 0, 179.9998, 0), format: 'float32x2'}
    },
    createContributor: (inputs, parameters, output) =>
      new GPUGreatCircleArcs({
        sources: inputs['sources'] as GraphDataView<'float32x2'>,
        targets: inputs['targets'] as GraphDataView<'float32x2'>,
        maximumSegmentCount: 4,
        parameters,
        output
      })
  });
  try {
    const result = await fixture.run(
      getGPUGreatCircleArcsParameterValues({maximumSegmentLength: 0, minimumSegments: 4})
    );
    expect(result.positions).toHaveLength(10);
    // Equatorial supplements of the f32 targets are about 2.66e-7 and 3.46e-6 radians.
    // These reject both an exact-antipodes-only cutoff and the formerly claimed 1e-3 cutoff.
    for (const vertex of result.positions.slice(1, 4)) {
      expect(vertex[0]).toBeNaN();
      expect(vertex[1]).toBeNaN();
    }
    for (const vertex of result.positions.slice(5, 10))
      expect(vertex.every(Number.isFinite)).toBe(true);
    expect(result.positions[4]).toEqual([179.99998474121094, 0]);
    expect(result.positions[9]).toEqual([179.9998016357422, 0]);
  } finally {
    fixture.destroy();
  }
});

for (const route of ['arcs', 'segmentize'] as const) {
  it(`${route} retains a small nonzero terminal longitude from a nonzero near-antipodal origin (B3)`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createLineLineFixture(device, {
      capacity: 10,
      lineCapacity: 2,
      inputs: {
        sources: {values: Float32Array.of(179.99998, 0, 170, 0), format: 'float32x2'},
        targets: {values: Float32Array.of(0.000001, 0, -170, 0), format: 'float32x2'},
        positions: {
          values: Float32Array.of(179.99998, 0, 0.000001, 0, 170, 0, -170, 0),
          format: 'float32x2'
        },
        offsets: {values: Uint32Array.of(0, 2, 4), format: 'uint32'}
      },
      createContributor: (inputs, parameters, output) =>
        route === 'arcs'
          ? new GPUGreatCircleArcs({
              sources: inputs['sources'] as GraphDataView<'float32x2'>,
              targets: inputs['targets'] as GraphDataView<'float32x2'>,
              maximumSegmentCount: 4,
              parameters,
              output
            })
          : new GPULineSegmentize({
              positions: inputs['positions'] as GraphDataView<'float32x2'>,
              lineOffsets: inputs['offsets'] as GraphDataView<'uint32'>,
              spatialContext: {
                coordinateSpace: 'longitude-latitude',
                metric: 'great-circle',
                sphereRadius: 1
              },
              maximumPieceCount: 4,
              parameters,
              output
            })
    });
    try {
      const result = await fixture.run(
        route === 'arcs'
          ? getGPUGreatCircleArcsParameterValues({maximumSegmentLength: 0, minimumSegments: 4})
          : getGPULineSegmentizeParameterValues({maximumSegmentLength: 0.08})
      );
      expect(result.positions).toHaveLength(10);
      expect(result.positions[0]).toEqual([179.99998474121094, 0]);
      // Literal f32(1e-6), retained bit for bit; source + f32(target-source) would be zero.
      expect(result.positions[4]).toEqual([9.999999974752427e-7, 0]);
      for (const vertex of result.positions.slice(1, 4)) {
        expect(vertex[0]).toBeNaN();
        expect(vertex[1]).toBeNaN();
      }
      expect(result.positions[9]).toEqual([190, 0]);
    } finally {
      fixture.destroy();
    }
  });
}
