// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandNodeProducer, GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPUFuzzyContiguity} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-fuzzy-contiguity';
import {GPUIslandAttachment} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-island-attachment';
import {GPULatticeWeights} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-lattice-weights';
import {GPUSpatialLag} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-lag';
import {AUDIT_ALGEBRA_LEFT, AUDIT_ALGEBRA_MASK} from './audit-weights-a-algebra-fixture';
import {flattenFeatures, type OraclePolygons} from './spatial-weights-oracle';

type Format = 'uint32' | 'float32' | 'float32x2';
export type AuditBuilderIO = {
  input: <ValueFormat extends Format>(
    values: Uint32Array | Float32Array,
    format: ValueFormat
  ) => GraphDataView<ValueFormat>;
  output: <ValueFormat extends 'uint32' | 'float32'>(
    name: string,
    format: ValueFormat,
    expected: number[]
  ) => GraphDataView<ValueFormat>;
};
export const AUDIT_BUILDER_NAMES = [
  'fuzzy',
  'islandsDirected',
  'islandsReciprocal',
  'latticeQueen',
  'latticeRook',
  'lag',
  'normalizedLag'
] as const;
export type AuditBuilderName = (typeof AUDIT_BUILDER_NAMES)[number];

/** Literal Shapely/dense-matrix/distance expectations from audit-weights-a-helpers.oracle.py. */
export function createAuditBuilder(
  io: AuditBuilderIO,
  name: AuditBuilderName,
  scale = 1,
  shift = 0,
  nearestSearch: 'scan' | 'grid' = 'scan'
): GPUCommandNodeProducer {
  const output = io.output;
  const input = io.input;
  if (name === 'lag' || name === 'normalizedLag') {
    const expected =
      name === 'lag'
        ? [37, 50, 0, 0, 105, 118, 8, 16, 22, 24, 27, 30, 0, 0, 0, 0]
        : [
            2.8461538461538463, 3.8461538461538463, 0, 0, 8.076923076923077, 9.076923076923077, 1,
            2, 11, 12, 9, 10, 0, 0, 0, 0
          ];
    return new GPUSpatialLag({
      values: input(
        Uint32Array.of(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16),
        'uint32'
      ),
      columnCount: 2,
      normalize: name === 'normalizedLag',
      mask: input(AUDIT_ALGEBRA_MASK, 'uint32'),
      weights: {
        offsets: input(AUDIT_ALGEBRA_LEFT.offsets, 'uint32'),
        neighbors: input(AUDIT_ALGEBRA_LEFT.neighbors, 'uint32'),
        weights: input(AUDIT_ALGEBRA_LEFT.weights, 'float32')
      },
      output: output('lag', 'float32', expected)
    });
  }
  if (name === 'latticeQueen' || name === 'latticeRook') {
    const queen = name === 'latticeQueen';
    const neighbors = queen
      ? [1, 4, 5, 0, 4, 5, 7, 0, 1, 5, 0, 1, 4, 3]
      : [1, 4, 0, 5, 7, 0, 5, 1, 4, 3];
    return new GPULatticeWeights({
      width: 4,
      height: 2,
      criterion: queen ? 'queen' : 'rook',
      cellSize: [3 * scale, 4 * scale],
      mask: input(Uint32Array.of(1, 1, 0, 1, 1, 1, 0, 1), 'uint32'),
      output: {
        offsets: output(
          'offsets',
          'uint32',
          queen ? [0, 3, 6, 6, 7, 10, 13, 13, 14] : [0, 2, 4, 4, 5, 7, 9, 9, 10]
        ),
        neighbors: output('neighbors', 'uint32', neighbors),
        weights: output('weights', 'float32', Array(neighbors.length).fill(1)),
        distances: output(
          'distances',
          'float32',
          (queen ? [3, 4, 5, 3, 5, 4, 4, 4, 5, 3, 5, 4, 3, 4] : [3, 4, 3, 4, 4, 4, 3, 4, 3, 4]).map(
            value => value * scale
          )
        ),
        status: {
          overflow: output('overflow', 'uint32', [0]),
          requiredCount: output('requiredCount', 'uint32', [neighbors.length]),
          candidateOverflow: output('candidateOverflow', 'uint32', [0])
        }
      }
    });
  }
  if (name === 'fuzzy') {
    const features: OraclePolygons[] = [
      [0, 0],
      [1, 0],
      [2.25, 0],
      [10, 0],
      [4, 4],
      [4.5, 4]
    ].map(([x, y]) => [
      [
        [
          [shift + scale * x, -shift + scale * y],
          [shift + scale * (x + 1), -shift + scale * y],
          [shift + scale * (x + 1), -shift + scale * (y + 1)],
          [shift + scale * x, -shift + scale * (y + 1)]
        ]
      ]
    ]);
    features.push(
      [],
      [
        [
          [
            [NaN, 0],
            [0, 0],
            [0, 1]
          ]
        ]
      ]
    );
    const polygons = flattenFeatures(features);
    return new GPUFuzzyContiguity({
      tolerance: 0.25 * scale,
      polygons: {
        kind: 'polygons',
        positions: input(polygons.positions, 'float32x2'),
        ringOffsets: input(polygons.ringOffsets, 'uint32'),
        polygonOffsets: input(polygons.polygonOffsets, 'uint32'),
        featureOffsets: input(polygons.featureOffsets, 'uint32')
      },
      output: {
        offsets: output('offsets', 'uint32', [0, 1, 3, 4, 4, 5, 6, 6, 6]),
        neighbors: output('neighbors', 'uint32', [1, 0, 2, 1, 5, 4]),
        weights: output('weights', 'float32', [1, 1, 1, 1, 1, 1]),
        distances: output('distances', 'float32', [0, 0, 0.25 * scale, 0.25 * scale, 0, 0]),
        added: output('added', 'uint32', [0, 0, 1, 1, 0, 0]),
        featureStatus: output('featureStatus', 'uint32', [0, 0, 0, 0, 0, 0, 1, 2]),
        status: {
          overflow: output('overflow', 'uint32', [0]),
          requiredCount: output('requiredCount', 'uint32', [6]),
          candidateOverflow: output('candidateOverflow', 'uint32', [0])
        }
      }
    });
  }
  const reciprocal = name === 'islandsReciprocal';
  const neighbors = reciprocal ? [1, 2, 0, 0, 4, 5, 3, 3, 7, 5] : [1, 0, 0, 4, 3, 3, 5];
  return new GPUIslandAttachment({
    nearestSearch,
    reciprocal,
    positions: input(
      Float32Array.from([0, 2, 1, 5, 5, 10, NaN, 30].flatMap(x => [shift + scale * x, -shift])),
      'float32x2'
    ),
    weights: {
      offsets: input(Uint32Array.of(0, 1, 2, 2, 2, 2, 2, 2, 2), 'uint32'),
      neighbors: input(Uint32Array.of(1, 0), 'uint32'),
      weights: input(Float32Array.of(2, 3), 'float32')
    },
    output: {
      offsets: output(
        'offsets',
        'uint32',
        reciprocal ? [0, 2, 3, 4, 6, 7, 9, 9, 10] : [0, 1, 2, 3, 4, 5, 6, 6, 7]
      ),
      neighbors: output('neighbors', 'uint32', neighbors),
      weights: output(
        'weights',
        'float32',
        reciprocal ? [2, 1, 3, 1, 1, 1, 1, 1, 1, 1] : [2, 3, 1, 1, 1, 1, 1]
      ),
      added: output(
        'added',
        'uint32',
        reciprocal ? [0, 1, 0, 1, 1, 1, 1, 1, 1, 1] : [0, 0, 1, 1, 1, 1, 1]
      ),
      attachment: output(
        'attachment',
        'uint32',
        [0xffffffff, 0xffffffff, 0, 4, 3, 3, 0xffffffff, 5]
      ),
      status: {
        overflow: output('overflow', 'uint32', [0]),
        requiredCount: output('requiredCount', 'uint32', [neighbors.length]),
        candidateOverflow: output('candidateOverflow', 'uint32', [0]),
        uncertainCount: output('uncertainCount', 'uint32', [0])
      }
    }
  });
}
