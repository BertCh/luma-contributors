// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPULineClipByPolygon,
  GPUSharedPaths
} from '../../../src/gpu-spatial-analysis/line-clip/index';
import type {OracleView} from '../line-merge/line-merge-output-oracle-fixture';
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';

export function createSharedTruncationFixture(
  graph: GPUCommandGraph,
  view: OracleView,
  aliasStatus = false
) {
  const packed = view('uint32', 2);
  const overflow = graph.createDataView(packed.buffer, {format: 'uint32', length: 1});
  const uncertainCount = graph.createDataView(packed.buffer, {
    format: 'uint32',
    length: 1,
    byteOffset: aliasStatus ? 0 : 4
  });
  const count = view('uint32', 1);
  const requiredCount = view('uint32', 1);
  const lines = () => ({
    kind: 'lines' as const,
    positions: view('float32x2', 4, [0, 0, 1, 0, 0, 2, 1, 2]),
    lineOffsets: view('uint32', 3, [0, 2, 4])
  });
  const contributor = new GPUSharedPaths({
    left: lines(),
    right: lines(),
    intersectionCapacity: 8,
    output: {
      runs: {
        positions: view('float32x2', 2),
        offsets: view('uint32', 2),
        leftSourceIds: view('uint32', 1),
        rightSourceIds: view('uint32', 1),
        forward: view('uint32', 1)
      },
      status: {count, requiredCount, overflow, uncertainCount}
    }
  });
  return {contributor, packed, count, requiredCount};
}

export function createClipPaddingFixture(view: OracleView, intersectionCapacity: number) {
  const count = view('uint32', 1);
  const overflow = view('uint32', 1);
  const uncertainCount = view('uint32', 1);
  // Line bbox [-4,-3]^2 is disjoint from the triangle bbox [1e-30,3.2e38] x [0,3.2e38].
  // Only unused midpoint rows can land at the old finite padding value (3e38,3e38).
  const contributor = new GPULineClipByPolygon({
    spatialSort: false,
    lines: {
      kind: 'lines',
      positions: view('float32x2', 2, [-4, -4, -3, -3]),
      lineOffsets: view('uint32', 2, [0, 2])
    },
    polygons: {
      kind: 'polygons',
      positions: view('float32x2', 4, [1e-30, 1e-30, 3.2e38, 3.2e38, 3.2e38, 0, 1e-30, 1e-30]),
      featureOffsets: view('uint32', 2, [0, 1]),
      polygonOffsets: view('uint32', 2, [0, 1]),
      ringOffsets: view('uint32', 2, [0, 4])
    },
    intersectionCapacity,
    candidateCapacity: 32,
    output: {
      pieces: {
        geometry: {kind: 'lines', positions: view('float32x2', 4), lineOffsets: view('uint32', 3)},
        sourceIds: view('uint32', 2),
        status: {count, overflow}
      },
      status: {uncertainCount}
    }
  });
  return {contributor, count, overflow, uncertainCount};
}

export function createClipUncertaintyFixture(view: OracleView, includeUncertain: boolean) {
  const count = view('uint32', 1);
  const overflow = view('uint32', 1);
  const uncertainCount = includeUncertain ? view('uint32', 1) : undefined;
  // The short diagonal overlaps the triangle's diagonal edge, but its orientation
  // terms against the two incident edges span about 2^226 and cannot be certified.
  const contributor = new GPULineClipByPolygon({
    spatialSort: false,
    lines: {
      kind: 'lines',
      positions: view('float32x2', 2, [1e-30, 1e-30, 2e-30, 2e-30]),
      lineOffsets: view('uint32', 2, [0, 2])
    },
    polygons: {
      kind: 'polygons',
      positions: view('float32x2', 4, [1.5e-30, 1.5e-30, 1e38, 1e38, 1e38, 0, 1.5e-30, 1.5e-30]),
      featureOffsets: view('uint32', 2, [0, 1]),
      polygonOffsets: view('uint32', 2, [0, 1]),
      ringOffsets: view('uint32', 2, [0, 4])
    },
    intersectionCapacity: 8,
    candidateCapacity: 32,
    output: {
      pieces: {
        geometry: {kind: 'lines', positions: view('float32x2', 8), lineOffsets: view('uint32', 5)},
        sourceIds: view('uint32', 4),
        status: {count, overflow}
      },
      status: {uncertainCount}
    }
  });
  return {contributor, count, overflow, uncertainCount};
}

export function createSharedUncertaintyFixture(view: OracleView, includeUncertain: boolean) {
  const count = view('uint32', 1);
  const overflow = view('uint32', 1);
  const uncertainCount = includeUncertain ? view('uint32', 1) : undefined;
  // One collinear candidate with the same unrepresentable orientation exponent span.
  const contributor = new GPUSharedPaths({
    left: {
      kind: 'lines',
      positions: view('float32x2', 2, [1e-30, 1e-30, 2e-30, 2e-30]),
      lineOffsets: view('uint32', 2, [0, 2])
    },
    right: {
      kind: 'lines',
      positions: view('float32x2', 2, [1.5e-30, 1.5e-30, 1e38, 1e38]),
      lineOffsets: view('uint32', 2, [0, 2])
    },
    intersectionCapacity: 4,
    output: {
      runs: {
        positions: view('float32x2', 2),
        offsets: view('uint32', 2),
        leftSourceIds: view('uint32', 1),
        rightSourceIds: view('uint32', 1),
        forward: view('uint32', 1)
      },
      status: {count, overflow, uncertainCount}
    }
  });
  return {contributor, count, overflow, uncertainCount};
}

export function createClipCapacityFixture(
  view: OracleView,
  capacity: number,
  optional: 'edges' | 'required' | 'vertices' | 'counts' | 'parts' | 'absent'
) {
  const positions = view('float32x2', capacity);
  const lineOffsets = view('uint32', 3);
  const sourceIds = view('uint32', 2);
  const sourcePartRows = optional === 'parts' ? view('uint32', 2) : undefined;
  const count = view('uint32', 1);
  const overflow = view('uint32', 1);
  const vertexSourceEdges = optional === 'edges' ? view('uint32', capacity) : undefined;
  const requiredVertexCount =
    optional === 'required' || optional === 'counts' ? view('uint32', 1) : undefined;
  const vertexCount =
    optional === 'vertices' || optional === 'counts' ? view('uint32', 1) : undefined;
  // Square [0,4]^2 clips line 0 to (0,1)-(2,1)-(4,1), edges 0 then 1.
  // Line 1 stays (1,2)-(2,2)-(3,2), edges 3 then 4: six required vertices.
  const contributor = new GPULineClipByPolygon({
    lines: {
      kind: 'lines',
      positions: view('float32x2', 6, [-1, 1, 2, 1, 5, 1, 1, 2, 2, 2, 3, 2]),
      lineOffsets: view('uint32', 3, [0, 3, 6]),
      sourceIds: view('uint32', 2, [71, 159])
    },
    polygons: {
      kind: 'polygons',
      positions: view('float32x2', 5, [0, 0, 4, 0, 4, 4, 0, 4, 0, 0]),
      ringOffsets: view('uint32', 2, [0, 5]),
      polygonOffsets: view('uint32', 2, [0, 1]),
      featureOffsets: view('uint32', 2, [0, 1])
    },
    intersectionCapacity: 8,
    candidateCapacity: 32,
    output: {
      pieces: {
        geometry: {kind: 'lines', positions, lineOffsets},
        sourceIds,
        sourcePartRows,
        vertexSourceEdges,
        status: {count, overflow, vertexCount, requiredVertexCount}
      }
    }
  });
  return {
    contributor,
    positions,
    lineOffsets,
    sourceIds,
    sourcePartRows,
    count,
    overflow,
    vertexSourceEdges,
    vertexCount,
    requiredVertexCount
  };
}
