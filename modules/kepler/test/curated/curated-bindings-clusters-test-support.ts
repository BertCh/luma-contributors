// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';
import {
  measurePathLength,
  measureRingSignedArea,
  type CatalogCoordinateSystem,
  type CatalogPosition
} from '../../src/analysis/catalog-geojson';

/** Preserve every hull edge, allowing either winding and a different first vertex. */
export function expectCuratedClusterRingAdjacency(
  ring: readonly CatalogPosition[],
  expectedRing: readonly CatalogPosition[],
  report: string
): void {
  expect(ring.length, `${report}; ring length`).toBe(expectedRing.length);
  expect(
    ring.length,
    `${report}; polygon ring needs at least three vertices and closure`
  ).toBeGreaterThanOrEqual(4);
  expect(ring.at(-1), `${report}; ring closure`).toEqual(ring[0]);
  expect(expectedRing.at(-1), `${report}; reference ring closure`).toEqual(expectedRing[0]);
  const vertices = ring.slice(0, -1);
  const expectedVertices = expectedRing.slice(0, -1);
  const firstVertex = expectedVertices.findIndex(
    position => position[0] === vertices[0][0] && position[1] === vertices[0][1]
  );
  const preservesAdjacency =
    firstVertex >= 0 &&
    [1, -1].some(direction =>
      vertices.every((position, index) => {
        const expected =
          expectedVertices[(firstVertex + direction * index + vertices.length) % vertices.length];
        return position[0] === expected[0] && position[1] === expected[1];
      })
    );
  expect(
    preservesAdjacency,
    `${report}; hull ring adjacency; actual=${JSON.stringify(ring)}; expected=${JSON.stringify(expectedRing)}`
  ).toBe(true);
}

/** Check readback measurements against the geometry that the binding actually publishes. */
export function expectCuratedClusterFootprintMeasures(
  ring: readonly CatalogPosition[],
  measurements: {area: number; perimeter: number},
  coordinateSystem: CatalogCoordinateSystem,
  report: string
): void {
  const expected = {
    area: Math.abs(measureRingSignedArea(ring, coordinateSystem)),
    perimeter: measurePathLength(ring, coordinateSystem)
  };
  for (const column of ['area', 'perimeter'] as const) {
    const actual = measurements[column];
    const wanted = expected[column];
    const evidence = `${report}; published geometry ${column}=${wanted}; readback ${column}=${actual}`;
    expect(Number.isFinite(actual), evidence).toBe(true);
    // Same f32 budget as CPU/GPU fixture parity: geographic upload has sub-metre lon/lat
    // quanta, while planar input is translated to its local origin before upload.
    const geographic = coordinateSystem !== 'planar';
    const tolerance = Math.max(geographic ? 0.1 : 1e-3, wanted * (geographic ? 5e-3 : 1e-4));
    expect(Math.abs(actual - wanted), evidence).toBeLessThanOrEqual(tolerance);
  }
}
