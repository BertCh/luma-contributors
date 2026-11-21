// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCanonicalGeometryEquality} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-canonical-geometry-equality';
import {GPUCanonicalPolygonEquality} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-canonical-polygon-equality';
import type {EqualityView} from './canonical-equality-fixture';

export function createSubnormalEqualityFixture(kind: 'tree' | 'polygon', view: EqualityView) {
  const minimumNormal = 2 ** -126;
  const output = view('uint32', [99]);
  const common = {tolerance: minimumNormal, toleranceMetric: 'euclidean' as const, output};
  // Stored first coordinates differ by .75N, within tolerance N; all others coincide.
  const contributor =
    kind === 'tree'
      ? new GPUCanonicalGeometryEquality({
          ...common,
          left: {
            positions: view('float32x2', [1.5 * minimumNormal, 0]),
            nodes: view('uint32x4', [0, 0, 1, 0]),
            children: view('uint32', [0])
          },
          right: {
            positions: view('float32x2', [0.75 * minimumNormal, 0]),
            nodes: view('uint32x4', [0, 0, 1, 0]),
            children: view('uint32', [0])
          },
          pairs: view('uint32x2', [0, 0])
        })
      : new GPUCanonicalPolygonEquality({
          ...common,
          left: {
            positions: view('float32x2', [1.5 * minimumNormal, 0, 1.5 * minimumNormal, 2, 2, 0]),
            ringOffsets: view('uint32', [0, 3]),
            polygonOffsets: view('uint32', [0, 1]),
            featureOffsets: view('uint32', [0, 1])
          },
          right: {
            positions: view('float32x2', [0.75 * minimumNormal, 0, 1.5 * minimumNormal, 2, 2, 0]),
            ringOffsets: view('uint32', [0, 3]),
            polygonOffsets: view('uint32', [0, 1]),
            featureOffsets: view('uint32', [0, 1])
          }
        });
  return {contributor, output};
}
