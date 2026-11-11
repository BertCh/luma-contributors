// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// FX-17: the winding number of `GPUS2PolygonCovering` sums `atan2(numerator, denominator)` per
// ring edge. On the headless test GPU, `atan2(y, -0.0)` returns minus half pi for positive y (and
// plus half pi for negative y), as it did in the cap path. These configurations make
// `denominator` exactly zero, of both signs, with a nonzero numerator of either sign
// (perpendicular axis vertices seen from points on coordinate planes, which S2 cell vertices on
// face center lines are), and make both tiny (polygon vertices on face centers and face corners,
// which are S2 cell vertices). Each cover is compared with a binary64 CPU port of the traversal.
// Without `s2AnalysisAtan2` the `axis-wedge-negative-zero` cases fail.

import {Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUS2PolygonCovering,
  type GPUS2PolygonCoveringRefinement
} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-polygon-covering';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

type Vector = [number, number, number];
type CoveringCase = {
  name: string;
  rings: Vector[][];
  mode: 'exterior' | 'interior';
  complement: boolean;
  maximumLevel: number;
};
type CoveringResult = {cells: bigint[]; required: number; status: number; visited: number};

const TOLERANCE = 1e-6;
const CAPACITY = 4096;
const MAXIMUM_VISITED_CELLS = 65536;

function crossVectors(left: Vector, right: Vector): Vector {
  return [
    left[1] * right[2] - left[2] * right[1],
    left[2] * right[0] - left[0] * right[2],
    left[0] * right[1] - left[1] * right[0]
  ];
}
function dotVectors(left: Vector, right: Vector): number {
  return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}
function normalizeVector(vector: Vector): Vector {
  const length = Math.hypot(...vector);
  return [vector[0] / length, vector[1] / length, vector[2] / length];
}
function negateVector(vector: Vector): Vector {
  return [-vector[0], -vector[1], -vector[2]];
}

// Binary64 port of the dggs S2 cell geometry the kernel uses.
function getCellLevel(cell: bigint): number {
  let trailingZeros = 0;
  while (trailingZeros < 64 && ((cell >> BigInt(trailingZeros)) & 1n) === 0n) trailingZeros++;
  return trailingZeros > 60 ? 0 : 30 - Math.floor(trailingZeros / 2);
}
function getCellIJ(cell: bigint): [number, number] {
  const level = getCellLevel(cell);
  let point: [number, number] = [0, 0];
  for (let hilbertLevel = 1; hilbertLevel <= level; hilbertLevel++) {
    const digitLevel = level - hilbertLevel + 1;
    const digit = Number((cell >> BigInt(2 * (30 - digitLevel) + 1)) & 3n);
    const digitXY: [number, number] =
      digit === 1 ? [0, 1] : digit === 2 ? [1, 1] : digit === 3 ? [1, 0] : [0, 0];
    const size = 2 ** (hilbertLevel - 1);
    if (digitXY[1] === 0) {
      if (digitXY[0] === 1) point = [size - 1 - point[0], size - 1 - point[1]];
      point = [point[1], point[0]];
    }
    point = [point[0] + size * digitXY[0], point[1] + size * digitXY[1]];
  }
  return Number(cell >> 61n) & 1 ? [point[1], point[0]] : point;
}
function convertSTToUV(st: number): number {
  return st >= 0.5 ? (4 * st * st - 1) / 3 : (1 - 4 * (1 - st) * (1 - st)) / 3;
}
function convertFaceUVToXYZ(face: number, u: number, v: number): Vector {
  switch (face) {
    case 0:
      return [1, u, v];
    case 1:
      return [-u, 1, v];
    case 2:
      return [-u, -v, 1];
    case 3:
      return [-1, -v, -u];
    case 4:
      return [v, -1, -u];
    default:
      return [v, u, -1];
  }
}
function getCellVertex(cell: bigint, slot: number): Vector {
  const level = getCellLevel(cell);
  const [i, j] = getCellIJ(cell);
  const scale = 2 ** level;
  const s = (i + (slot === 1 || slot === 2 ? 1 : 0)) / scale;
  const t = (j + (slot >= 2 ? 1 : 0)) / scale;
  return normalizeVector(
    convertFaceUVToXYZ(Number(cell >> 61n), convertSTToUV(s), convertSTToUV(t))
  );
}

/** Binary64 port of the serial traversal of `GPUS2PolygonCovering`, with an exact `atan2`. */
function computeReferenceCovering(reference: CoveringCase): CoveringResult {
  const rings = reference.rings;
  const onArc = (first: Vector, second: Vector, point: Vector): boolean => {
    const normal = normalizeVector(crossVectors(first, second));
    return (
      Math.abs(dotVectors(normal, point)) <= TOLERANCE &&
      dotVectors(crossVectors(first, point), normal) >= -TOLERANCE &&
      dotVectors(crossVectors(point, second), normal) >= -TOLERANCE
    );
  };
  const contains = (point: Vector): [boolean, boolean] => {
    let inside = false;
    let boundary = false;
    rings.forEach((ring, ringIndex) => {
      let winding = 0;
      ring.forEach((first, vertex) => {
        const second = ring[(vertex + 1) % ring.length]!;
        boundary = boundary || onArc(first, second, point);
        const numerator = dotVectors(point, crossVectors(first, second));
        const denominator =
          dotVectors(first, second) - dotVectors(point, first) * dotVectors(point, second);
        if (Math.abs(numerator) + Math.abs(denominator) > 1e-12) {
          winding += Math.atan2(numerator, denominator);
        }
      });
      inside = ringIndex === 0 ? winding > 3.14159265 : inside && winding <= 3.14159265;
    });
    return [reference.complement ? !inside : inside, boundary];
  };
  const classify = (cell: bigint): [boolean, boolean] => {
    const vertices = [0, 1, 2, 3].map(slot => getCellVertex(cell, slot));
    const crosses = vertices.map((vertex, edge) => crossVectors(vertex, vertices[(edge + 1) % 4]!));
    const normals = crosses.map(normalizeVector);
    const cellContains = (point: Vector) =>
      normals.every(normal => dotVectors(normal, point) >= -TOLERANCE);
    const cellOnArc = (first: Vector, second: Vector, normal: Vector, point: Vector) =>
      Math.abs(dotVectors(normal, point)) <= TOLERANCE &&
      dotVectors(crossVectors(first, point), normal) >= -TOLERANCE &&
      dotVectors(crossVectors(point, second), normal) >= -TOLERANCE;
    let intersects = false;
    let contained = true;
    let touched = false;
    for (const ring of rings)
      for (const vertex of ring)
        if (cellContains(vertex)) {
          touched = true;
          intersects = true;
        }
    for (let edge = 0; edge < 4; edge++) {
      const first = vertices[edge]!;
      const second = vertices[(edge + 1) % 4]!;
      const normal = normals[edge]!;
      const [inside, boundary] = contains(first);
      intersects = intersects || inside || boundary;
      contained = contained && inside && !boundary;
      for (const ring of rings)
        ring.forEach((third, vertex) => {
          const fourth = ring[(vertex + 1) % ring.length]!;
          const crossing = crossVectors(crosses[edge]!, crossVectors(third, fourth));
          if (Math.hypot(...crossing) > TOLERANCE * TOLERANCE) {
            const point = normalizeVector(crossing);
            if (
              (cellOnArc(first, second, normal, point) && onArc(third, fourth, point)) ||
              (cellOnArc(first, second, normal, negateVector(point)) &&
                onArc(third, fourth, negateVector(point)))
            ) {
              touched = true;
              intersects = true;
            }
          } else if (
            cellOnArc(first, second, normal, third) ||
            cellOnArc(first, second, normal, fourth)
          ) {
            touched = true;
            intersects = true;
          }
        });
    }
    return [intersects, contained && !touched];
  };
  const stack: bigint[] = [];
  for (let face = 5; face >= 0; face--) stack.push(BigInt(2 * face + 1) << 60n);
  const cells: bigint[] = [];
  let visited = 0;
  while (stack.length > 0) {
    const cell = stack.pop()!;
    visited++;
    const [intersects, contained] = classify(cell);
    if (!intersects) continue;
    const level = getCellLevel(cell);
    const emit = contained || (reference.mode !== 'interior' && level === reference.maximumLevel);
    if (emit) {
      cells.push(cell);
    } else if (level < reference.maximumLevel) {
      const marker = 1n << BigInt(2 * (30 - level));
      const childMarker = marker >> 2n;
      const first = cell - marker + childMarker;
      for (let child = 3; child >= 0; child--)
        stack.push(first + BigInt(child) * (childMarker << 1n));
    }
  }
  return {cells, required: cells.length, status: 0, visited};
}

async function executeCovering(
  device: Device,
  reference: CoveringCase,
  refinement: GPUS2PolygonCoveringRefinement
): Promise<CoveringResult> {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const allocate = (bytes: number) => {
    const buffer = device.createBuffer({
      byteLength: bytes,
      usage: Buffer.STORAGE | Buffer.COPY_DST | Buffer.COPY_SRC
    });
    buffers.push(buffer);
    return buffer;
  };
  const positionValues = reference.rings.flat(2);
  const offsetValues = [0];
  for (const ring of reference.rings) offsetValues.push(offsetValues.at(-1)! + ring.length);
  const positions = allocate(positionValues.length * 4);
  positions.write(new Float32Array(positionValues));
  const offsets = allocate(offsetValues.length * 4);
  offsets.write(new Uint32Array(offsetValues));
  const cells = allocate(CAPACITY * 8);
  const counts = allocate(8);
  const status = allocate(4);
  const visited = allocate(4);
  graph.add(
    new GPUS2PolygonCovering({
      positions: importGraphBuffer(
        graph,
        'positions',
        positions,
        'float32x3',
        positionValues.length / 3
      ),
      ringOffsets: importGraphBuffer(graph, 'offsets', offsets, 'uint32', offsetValues.length),
      maximumLevel: reference.maximumLevel,
      mode: reference.mode,
      complement: reference.complement,
      capacity: CAPACITY,
      maximumVisitedCells: MAXIMUM_VISITED_CELLS,
      refinement,
      output: {
        cells: importGraphBuffer(graph, 'cells', cells, 'uint32x2', CAPACITY),
        counts: importGraphBuffer(graph, 'counts', counts, 'uint32x2', 1),
        status: importGraphBuffer(graph, 'status', status, 'uint32', 1),
        visitedCount: importGraphBuffer(graph, 'visited', visited, 'uint32', 1)
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const words = await readUint32(cells, CAPACITY * 2);
    const countWords = await readUint32(counts, 2);
    return {
      cells: Array.from(
        {length: countWords[0]!},
        (_, index) => BigInt(words[index * 2]!) + (BigInt(words[index * 2 + 1]!) << 32n)
      ),
      required: countWords[1]!,
      status: (await readUint32(status, 1))[0]!,
      visited: (await readUint32(visited, 1))[0]!
    };
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}

const X: Vector = [1, 0, 0];
const Y: Vector = [0, 1, 0];
const Z: Vector = [0, 0, 1];
const INVERSE_SQUARE_ROOT_TWO = Math.SQRT1_2;
const INVERSE_SQUARE_ROOT_THREE = 1 / Math.sqrt(3);

/**
 * Rotations by signed axis permutations (determinant one, so rings stay counterclockwise): exact
 * in float32, so zero denominators stay exactly zero.
 */
const SYMMETRIES: ((vector: Vector) => Vector)[] = [
  vector => vector,
  ([x, y, z]) => [-y, x, z],
  ([x, y, z]) => [z, x, y],
  ([x, y, z]) => [-x, -y, z],
  ([x, y, z]) => [y, -z, -x],
  ([x, y, z]) => [x, -z, y]
];

/**
 * Rings (counterclockwise seen from outside) whose winding meets `denominator == 0`:
 * - `axis-wedge`: edge X to Y seen from cell vertices with x == 0 and z > 0 (numerator > 0).
 * - `octant`: every edge joins perpendicular axes; cell vertices on the coordinate planes.
 * - `notched`: a nonconvex ring whose reflex vertex makes interior points lie right of an
 *   axis-to-axis edge (numerator < 0 with denominator 0).
 * - `face-corners`: the ring is the boundary of face 0 (cube corners), so the cell vertices at
 *   the corners make the numerator and denominator both tiny.
 * - `face-center-vertex`: a ring vertex on the face center Z, a cell vertex at level 1 and deeper.
 */
const BASE_RINGS: {name: string; rings: Vector[][]}[] = [
  {name: 'axis-wedge', rings: [[X, Y, [-INVERSE_SQUARE_ROOT_TWO, 0, INVERSE_SQUARE_ROOT_TWO]]]},
  {name: 'octant', rings: [[X, Y, Z]]},
  // Signed zeros chosen so that dot(X, Y) is -0 and, at a cell vertex (-0, y, z), the product
  // dot(point, X) * dot(point, Y) is +0: the denominator is -0 with a positive numerator.
  {
    name: 'axis-wedge-negative-zero',
    rings: [
      [
        [1, -0, 0],
        [-0, 1, -0],
        [-INVERSE_SQUARE_ROOT_TWO, -0, INVERSE_SQUARE_ROOT_TWO]
      ]
    ]
  },
  {
    name: 'octant-negative-zero',
    rings: [
      [
        [1, -0, 0],
        [-0, 1, -0],
        [0, -0, 1]
      ]
    ]
  },
  {
    name: 'notched',
    rings: [
      [
        X,
        normalizeVector([0.2, 1, 0.1]),
        normalizeVector([-0.6, 0.4, 1]),
        normalizeVector([0.15, 0.15, 1]),
        normalizeVector([0.6, -0.5, 1])
      ]
    ]
  },
  {
    name: 'face-corners',
    rings: [
      [
        [INVERSE_SQUARE_ROOT_THREE, -INVERSE_SQUARE_ROOT_THREE, -INVERSE_SQUARE_ROOT_THREE],
        [INVERSE_SQUARE_ROOT_THREE, INVERSE_SQUARE_ROOT_THREE, -INVERSE_SQUARE_ROOT_THREE],
        [INVERSE_SQUARE_ROOT_THREE, INVERSE_SQUARE_ROOT_THREE, INVERSE_SQUARE_ROOT_THREE],
        [INVERSE_SQUARE_ROOT_THREE, -INVERSE_SQUARE_ROOT_THREE, INVERSE_SQUARE_ROOT_THREE]
      ]
    ]
  },
  {
    name: 'face-center-vertex',
    rings: [[Z, normalizeVector([0.5, -0.4, 1]), normalizeVector([0.6, 0.5, 1])]]
  },
  {
    name: 'wedge-with-hole',
    rings: [
      [X, Y, [-INVERSE_SQUARE_ROOT_TWO, 0, INVERSE_SQUARE_ROOT_TWO]],
      [
        normalizeVector([0.05, 0.3, 0.6]),
        normalizeVector([0.3, 0.3, 0.6]),
        normalizeVector([0.2, 0.6, 0.6])
      ]
    ]
  }
];

function createCases(): CoveringCase[] {
  const cases: CoveringCase[] = [];
  for (const base of BASE_RINGS)
    SYMMETRIES.forEach((symmetry, symmetryIndex) => {
      for (const mode of ['exterior', 'interior'] as const)
        cases.push({
          name: `${base.name}-${symmetryIndex}-${mode}`,
          rings: base.rings.map(ring => ring.map(symmetry)),
          mode,
          complement: base.name === 'octant' && symmetryIndex === 0,
          maximumLevel: 4
        });
    });
  return cases;
}

it('S2 polygon covering winding agrees with a binary64 reference where atan2 has a zero second argument', {
  timeout: 300000
}, async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const cases = createCases();
  // The configurations really do produce exactly zero denominators with numerators of both signs.
  let positiveZeroDenominators = 0;
  let negativeZeroDenominators = 0;
  let negativeZeroSignedDenominators = 0;
  for (const reference of cases) {
    const expected = computeReferenceCovering(reference);
    expect(expected.cells.length, reference.name).toBeGreaterThan(0);
    for (const refinement of ['serial', 'level-synchronous'] as const) {
      const actual = await executeCovering(device, reference, refinement);
      const note = `${reference.name} ${refinement}`;
      expect(actual.status, note).toBe(0);
      expect(actual.required, note).toBe(expected.required);
      expect(actual.visited, note).toBe(expected.visited);
      expect(actual.cells, note).toEqual(expected.cells);
    }
    for (const ring of reference.rings)
      for (const cell of expected.cells)
        for (let slot = 0; slot < 4; slot++) {
          const point = getCellVertex(cell, slot);
          ring.forEach((first, vertex) => {
            const second = ring[(vertex + 1) % ring.length]!;
            const denominator =
              dotVectors(first, second) - dotVectors(point, first) * dotVectors(point, second);
            const numerator = dotVectors(point, crossVectors(first, second));
            if (denominator === 0 && numerator > 1e-6) positiveZeroDenominators++;
            if (denominator === 0 && numerator < -1e-6) negativeZeroDenominators++;
            if (Object.is(denominator, -0) && numerator > 1e-6) negativeZeroSignedDenominators++;
          });
        }
  }
  expect(positiveZeroDenominators).toBeGreaterThan(0);
  expect(negativeZeroDenominators).toBeGreaterThan(0);
  expect(negativeZeroSignedDenominators).toBeGreaterThan(0);
});
