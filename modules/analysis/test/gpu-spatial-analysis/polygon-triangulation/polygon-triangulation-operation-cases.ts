// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CPUPolygonTriangulationInput} from '../../../src/analysis-core/polygon-triangulation/cpu-polygon-triangulation-operation';
import {getPolygonTriangulationIndexCount} from '../../../src/analysis-core/polygon-triangulation/gpu-polygon-triangulation';
import type {PolygonTriangulationOperationProps} from '../../../src/analysis-core/polygon-triangulation/gpu-polygon-triangulation-operation';
import type {SearchPoint} from './polygon-triangulation-search-cases';

export const OPERATION_POLYGONS: SearchPoint[][][] = [
  [
    [
      [0, 0],
      [10, 0],
      [10, 8],
      [6, 8],
      [6, 5],
      [0, 5]
    ],
    [
      [2, 1],
      [2, 3],
      [4, 3],
      [4, 1]
    ]
  ],
  [
    [
      [20, 0],
      [24, 0],
      [25, 3],
      [22, 2],
      [20, 4]
    ]
  ],
  [
    [
      [30, 0],
      [40, 0],
      [40, 10],
      [30, 10]
    ],
    // The leftmost hole vertex lies inside the shell edge: a zero-length bridge.
    [
      [30, 4],
      [33, 3],
      [33, 5]
    ]
  ]
];

// Hand-derived layout: one live ring beginning at vertex 1 reserves ringEnd - 3 triangles.
// The two cases straddle signed int32 overflow of the index product: 2147483646 and
// 2147483649 (which becomes -2147483647). Both fail, but only the first has padding slots.
export const CLAMPED_RING_END_CASES = [
  {ringEnd: 715827885, triangleEnd: 715827882, indices: [1, 1, 1, 1, 1, 1]},
  {ringEnd: 715827886, triangleEnd: 715827883, indices: [0, 0, 0, 0, 0, 0]}
] as const;

export function createClampedRingEndInput(ringEnd: number): CPUPolygonTriangulationInput {
  return {
    positions: Float32Array.of(99, 99, 0, 0, 4, 0, 0, 4),
    featureOffsets: Uint32Array.of(0, 1),
    polygonOffsets: Uint32Array.of(0, 1),
    ringOffsets: Uint32Array.of(1, ringEnd)
  };
}

export function packOperationPolygons(polygons: SearchPoint[][][]): CPUPolygonTriangulationInput {
  const positions: number[] = [];
  const ringOffsets = [0];
  const polygonOffsets = [0];
  for (const polygon of polygons) {
    for (const ring of polygon) {
      positions.push(...ring.flat());
      ringOffsets.push(positions.length / 2);
    }
    polygonOffsets.push(ringOffsets.length - 1);
  }
  return {
    positions: Float32Array.from(positions),
    // Two features, the first owning two polygons; feature grouping must not alter slices.
    featureOffsets: Uint32Array.from(
      polygons.length >= 2 ? [0, 2, polygons.length] : [0, polygons.length]
    ),
    polygonOffsets: Uint32Array.from(polygonOffsets),
    ringOffsets: Uint32Array.from(ringOffsets)
  };
}

export function getTriangulationOperationProps(
  input: CPUPolygonTriangulationInput,
  indexCapacity = Math.max(
    1,
    getPolygonTriangulationIndexCount(
      input.positions.length / 2,
      input.ringOffsets.length - 1,
      input.polygonOffsets.length - 1
    )
  )
): PolygonTriangulationOperationProps {
  return {
    id: 'polygon-twin',
    positions: 'positions',
    featureOffsets: 'featureOffsets',
    polygonOffsets: 'polygonOffsets',
    ringOffsets: 'ringOffsets',
    polygonCount: input.polygonOffsets.length - 1,
    indexCapacity,
    output: {
      indices: 'indices',
      valid: 'valid',
      triangleCount: 'triangleCount',
      triangleOffsets: 'triangleOffsets'
    }
  };
}

/** Empty polygon, a frame (area 12), and a convex quadrilateral (area 4). */
export function createPaddingTriangulationInput(
  leading = 0,
  trailing = 0,
  value = 0
): CPUPolygonTriangulationInput {
  const input = packOperationPolygons([
    [],
    [
      [
        [0, 0],
        [4, 0],
        [4, 4],
        [0, 4]
      ],
      [
        [1, 1],
        [1, 3],
        [3, 3],
        [3, 1]
      ]
    ],
    [
      [
        [10, 0],
        [13, 0],
        [12, 2],
        [10, 1]
      ]
    ]
  ]);
  const positions = new Float32Array(input.positions.length + 2 * (leading + trailing)).fill(value);
  positions.set(input.positions, 2 * leading);
  return {
    ...input,
    positions,
    ringOffsets: Uint32Array.from(input.ringOffsets, offset => offset + leading)
  };
}
