// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPUCanonicalGeometryEquality} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-canonical-geometry-equality';
import {GPUCanonicalPolygonEquality} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-canonical-polygon-equality';

export type EqualityView = <Format extends 'uint32' | 'uint32x2' | 'uint32x4' | 'float32x2'>(
  format: Format,
  values: number[]
) => GraphDataView<Format>;

export function createEqualityMetricFixture(
  kind: 'tree' | 'polygon',
  toleranceMetric: 'per-coordinate' | 'euclidean' | undefined,
  view: EqualityView,
  delta: readonly [number, number] = [0.75, 0.75],
  tolerance = 1
) {
  const output = view('uint32', [99]);
  const common = {id: 'metric-test', tolerance, toleranceMetric, output};
  const contributor =
    kind === 'tree'
      ? new GPUCanonicalGeometryEquality({
          ...common,
          left: {
            positions: view('float32x2', [0, 0]),
            nodes: view('uint32x4', [0, 0, 1, 0]),
            children: view('uint32', [0])
          },
          right: {
            positions: view('float32x2', [delta[0], delta[1]]),
            nodes: view('uint32x4', [0, 0, 1, 0]),
            children: view('uint32', [0])
          },
          pairs: view('uint32x2', [0, 0])
        })
      : new GPUCanonicalPolygonEquality({
          ...common,
          // Canonical clockwise triangles; every vertex moves (.75,.75).
          left: {
            positions: view('float32x2', [0, 0, 0, 2, 2, 0]),
            ringOffsets: view('uint32', [0, 3]),
            polygonOffsets: view('uint32', [0, 1]),
            featureOffsets: view('uint32', [0, 1])
          },
          right: {
            positions: view('float32x2', [
              delta[0],
              delta[1],
              delta[0],
              delta[1] + 2,
              delta[0] + 2,
              delta[1]
            ]),
            ringOffsets: view('uint32', [0, 3]),
            polygonOffsets: view('uint32', [0, 1]),
            featureOffsets: view('uint32', [0, 1])
          }
        });
  return {contributor, output};
}
