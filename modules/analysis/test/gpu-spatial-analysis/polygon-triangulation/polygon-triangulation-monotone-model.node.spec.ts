// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {WgslReflect} from 'wgsl_reflect';
import {runMonotoneKernels} from './polygon-triangulation-monotone-harness';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {GPUPolygonTriangulation} from '../../../src/analysis-core/polygon-triangulation/gpu-polygon-triangulation';
import {
  checkTriangulation,
  createMonotoneCases,
  createRandomStarPolygon,
  createLargePolygonRings,
  getRingsChecksum,
  type LargePolygonDescriptor,
  createTwoOptPolygon,
  getPolygonArea,
  trianglesOverlap
} from './polygon-triangulation-monotone-cases';
import {triangulateMonotoneModel, orientExactSign} from './polygon-triangulation-monotone-model';
import largeFixture from '../../oracles/fixtures/geometry-triangulation-large.json';
import {triangulateMonotoneOnCPU} from '../../../src/analysis-core/polygon-triangulation/cpu-monotone-triangulation';
import {
  TOUCHING_HOLE_RINGS,
  expectTouchingHoleTiling
} from './polygon-triangulation-touching-hole-contract';

describe('monotone triangulation CPU model', () => {
  for (const entry of createMonotoneCases()) {
    it(`triangulates ${entry.name}`, () => {
      const result = triangulateMonotoneModel(entry.rings);
      expect(result.failure).toBeUndefined();
      expect(result.valid).toBe(true);
      const check = checkTriangulation(entry.rings, result.triangles);
      expect(check.minimumSignedArea).toBeGreaterThanOrEqual(0);
      expect(check.area).toBe(getPolygonArea(entry.rings));
      expect(check.boundaryMatches).toBe(true);
    });
  }
});

describe('monotone triangulation CPU model, randomized', () => {
  it('triangulates 2-opt polygons with sparse and dense coordinates', () => {
    let tested = 0;
    for (let seed = 1; seed <= 120; seed++) {
      const dense = seed % 2 === 0;
      const polygon = createTwoOptPolygon(20 + (seed % 60), seed, dense ? 30 : 5000);
      if (!polygon) continue;
      tested++;
      const result = triangulateMonotoneModel([polygon]);
      expect(result.failure, `seed ${seed}`).toBeUndefined();
      const check = checkTriangulation([polygon], result.triangles);
      expect(check.minimumSignedArea, `seed ${seed}`).toBeGreaterThanOrEqual(0);
      expect(check.area, `seed ${seed}`).toBe(getPolygonArea([polygon]));
      expect(check.boundaryMatches, `seed ${seed}`).toBe(true);
      const brute = triangulateMonotoneModel([polygon], {rayShooting: 'brute'});
      expect(brute.triangles, `seed ${seed} brute`).toEqual(result.triangles);
    }
    expect(tested).toBeGreaterThan(100);
  });

  it('triangulates a 20000-vertex random star polygon', () => {
    const polygon = createRandomStarPolygon(20_000, 20261014);
    const result = triangulateMonotoneModel([polygon]);
    expect(result.failure).toBeUndefined();
    expect(result.triangles.length).toBe(19_998);
    const check = checkTriangulation([polygon], result.triangles);
    expect(check.minimumSignedArea).toBeGreaterThanOrEqual(0);
    const expected = getPolygonArea([polygon]);
    expect(Math.abs(check.area - expected) / expected).toBeLessThan(1e-9);
    expect(check.boundaryMatches).toBe(true);
  }, 120_000);

  it('does not overlap triangles pairwise on the small cases', () => {
    for (const entry of createMonotoneCases().filter(c => c.rings.flat().length < 120)) {
      const result = triangulateMonotoneModel(entry.rings);
      const positions = entry.rings.flat();
      const triangles = result.triangles.map(
        ([a, b, c]) => [positions[a], positions[b], positions[c]] as const
      );
      for (let i = 0; i < triangles.length; i++) {
        for (let j = i + 1; j < triangles.length; j++) {
          expect(trianglesOverlap(triangles[i], triangles[j]), `${entry.name} ${i}/${j}`).toBe(
            false
          );
        }
      }
    }
  });

  it('flags invalid input instead of producing triangles', () => {
    const bowtie = triangulateMonotoneModel([
      [
        [0, 0],
        [10, 10],
        [10, 0],
        [0, 10]
      ]
    ]);
    expect(bowtie.valid).toBe(false);
    const twoVertexShell = triangulateMonotoneModel([
      [
        [0, 0],
        [5, 5]
      ]
    ]);
    expect(twoVertexShell.valid).toBe(false);
  });
});

it('checks the geometric boundary when a hole and shell share distinct rows at one vertex', () => {
  const rings: [number, number][][] = [
    [
      [0, 0],
      [100, 0],
      [100, 100],
      [0, 100]
    ],
    [
      [0, 0],
      [30, 20],
      [20, 50]
    ]
  ];
  // Earcut's triangulation uses both input rows 0 and 4 for the same point (0, 0).
  const triangles: [number, number, number][] = [
    [3, 4, 6],
    [5, 0, 1],
    [1, 2, 3],
    [6, 5, 1],
    [1, 3, 6]
  ];
  const check = checkTriangulation(rings, triangles);
  expect(check.boundaryMatches).toBe(true);
  expect(check.area).toBe(9450);
  expect(check.minimumSignedArea).toBeGreaterThan(0);
  expect(checkTriangulation(rings, triangles.slice(1)).boundaryMatches).toBe(false);
});

// GEO-124: preserve row counts and indices; degenerates are part of the valid output contract.
it('retains two zero-area rows while tiling two holes touching at (4,4)', () => {
  const result = triangulateMonotoneOnCPU(TOUCHING_HOLE_RINGS);
  expect(result.valid).toBe(true);
  expect(result.failure).toBeUndefined();
  expectTouchingHoleTiling(result.triangles);
});

it('GEO-131 uses literal exact orientation signs and the nonfinite sentinel', () => {
  // Cross products: +1, -1, 0; the subnormal product is positive even though f64 underflows.
  expect(orientExactSign([0, 0], [1, 0], [0, 1])).toBe(1);
  expect(orientExactSign([0, 0], [0, 1], [1, 0])).toBe(-1);
  expect(orientExactSign([0, 0], [1, 1], [2, 2])).toBe(0);
  expect(orientExactSign([0, 0], [Number.MIN_VALUE, 0], [0, Number.MIN_VALUE])).toBe(1);
  // Review S2: both determinant products must overflow to exercise the exact fallback.
  const magnitude = 2 ** 700;
  const lowerCoordinate = magnitude * (1 - Number.EPSILON);
  const firstProduct = magnitude * lowerCoordinate;
  const secondProduct = magnitude * magnitude;
  expect(firstProduct).toBe(Infinity);
  expect(secondProduct).toBe(Infinity);
  // The exact difference is -magnitude^2 * Number.EPSILON, so its sign is -1.
  expect(orientExactSign([0, 0], [magnitude, magnitude], [magnitude, lowerCoordinate])).toBe(-1);
  for (const value of [NaN, Infinity, -Infinity]) {
    expect(orientExactSign([0, 0], [value, 0], [0, 1])).toBe(2);
    const result = triangulateMonotoneOnCPU([
      [
        [0, 0],
        [value, 0],
        [0, 1]
      ]
    ]);
    expect(result.valid).toBe(false);
    expect(result.triangles).toEqual([]);
  }
});

it('GEO-131 exercises production triangulation against pinned Shapely areas and boundary coverage', () => {
  const cases = largeFixture.cases as (LargePolygonDescriptor & {
    vertexCount: number;
    triangleCount: number;
    checksum: number;
    area: number;
    cdtArea: number;
  })[];
  for (const entry of cases.filter(candidate => candidate.vertexCount <= 20_000)) {
    const rings = createLargePolygonRings(entry);
    expect(getRingsChecksum(rings), entry.name).toBe(entry.checksum);
    const result = triangulateMonotoneModel(rings);
    expect(result.valid, entry.name).toBe(true);
    expect(result.triangles.length, entry.name).toBe(entry.triangleCount);
    const check = checkTriangulation(rings, result.triangles);
    expect(Number.isFinite(check.area), entry.name).toBe(true);
    expect(Math.abs(check.area - entry.area) / entry.area, entry.name).toBeLessThan(1e-9);
    expect(Math.abs(check.area - entry.cdtArea) / entry.cdtArea, entry.name).toBeLessThan(1e-9);
    expect(check.boundaryMatches, entry.name).toBe(true);
  }
});

it('GEO-131 preserves Earcut routing and parses extracted kernel declarations', () => {
  for (const largePolygonMethod of ['none', undefined] as const) {
    const graph = new GPUCommandGraph(createNullWebGPUDevice(), {
      id: largePolygonMethod ?? 'default'
    });
    const contributor = new GPUPolygonTriangulation({
      id: 'route-contract',
      positions: createTransientView(graph, 'positions', 'float32x2', 12),
      ringOffsets: createTransientView(graph, 'rings', 'uint32', 4),
      polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 2),
      featureOffsets: createTransientView(graph, 'features', 'uint32', 2),
      ...(largePolygonMethod === 'none' ? {largePolygonMethod} : {}),
      output: {
        indices: createTransientView(graph, 'indices', 'uint32', 48),
        valid: createTransientView(graph, 'valid', 'uint32', 1),
        triangleCount: createTransientView(graph, 'counts', 'uint32', 1)
      }
    });
    const nodes = contributor.getCommandNodes(graph);
    if (largePolygonMethod === undefined) {
      const firstEarcut = nodes.findIndex(node => node.id === 'route-contract-earcut');
      const monotone = nodes.findIndex(node => node.id.startsWith('route-contract-monotone-'));
      const fallbackEarcut = nodes.findIndex(node => node.id === 'route-contract-earcut-fallback');
      expect(firstEarcut).toBeGreaterThanOrEqual(0);
      expect(monotone).toBeGreaterThan(firstEarcut);
      expect(monotone).toBeLessThan(fallbackEarcut);
    }
    const kernels = getGPUCommandNodeKernelSources(nodes);
    const earcut = kernels.filter(kernel => kernel.id.includes('-earcut'));
    expect(earcut.map(kernel => kernel.id)).toEqual(
      largePolygonMethod === 'none'
        ? ['route-contract-earcut']
        : ['route-contract-earcut', 'route-contract-earcut-fallback']
    );
    for (const kernel of earcut) {
      expect(() => new WgslReflect(kernel.source), kernel.id).not.toThrow();
    }
    expect(kernels.at(-1)?.id).toBe(
      largePolygonMethod === 'none' ? 'route-contract-earcut' : 'route-contract-earcut-fallback'
    );
  }
});

it('review B2 ignores nonfinite vertices in discarded holes and rejects live rings', () => {
  const shell: [number, number][] = [
    [0, 0],
    [10, 0],
    [10, 10],
    [0, 10]
  ];
  // A one-row ring contributes no area: the hand-worked result is the 100-unit square.
  const discarded = triangulateMonotoneOnCPU([shell, [[NaN, 23]]]);
  expect(discarded.valid).toBe(true);
  expect(discarded.triangles).toHaveLength(2);
  const check = checkTriangulation([shell], discarded.triangles);
  expect(Number.isFinite(check.area)).toBe(true);
  expect(check.area).toBe(100);
  expect(check.boundaryMatches).toBe(true);
  for (const value of [NaN, Infinity, -Infinity]) {
    const live = triangulateMonotoneOnCPU([
      shell,
      [
        [value, 2],
        [4, 2],
        [4, 4],
        [2, 4]
      ]
    ]);
    expect(live.valid).toBe(false);
    expect(live.triangles).toEqual([]);
  }
});

it('review B3 rejects padding and coincident-row substitutions in counted degenerates', () => {
  const result = triangulateMonotoneOnCPU(TOUCHING_HOLE_RINGS);
  const padding = result.triangles.map<[number, number, number]>(triangle =>
    ['9,6,8', '11,8,6'].includes(triangle.join(',')) ? [0, 0, 0] : [...triangle]
  );
  expect(() => expectTouchingHoleTiling(padding)).toThrow();
  const coincident = result.triangles.map<[number, number, number]>(
    triangle => triangle.map(row => (row === 8 ? 6 : row)) as [number, number, number]
  );
  expect(() => expectTouchingHoleTiling(coincident)).toThrow();
});

it('review B3 pins counted degenerate source rows from the interpreted monotone kernels', () => {
  // Integer coordinates make the interpreter's orientation products exact for this fixture.
  const result = runMonotoneKernels([TOUCHING_HOLE_RINGS]);
  expect(result.valid).toEqual([1]);
  expect(result.triangleCount).toEqual([14]);
  const triangles = Array.from({length: 14}, (_, triangle): [number, number, number] => [
    result.indices[triangle * 3],
    result.indices[triangle * 3 + 1],
    result.indices[triangle * 3 + 2]
  ]);
  expectTouchingHoleTiling(triangles);
});
