// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma-contributors/graph/gpu-core';
import type {GPUPolygonizeOutput} from '../../../src/gpu-spatial-analysis/geometry-topology/topology-types';

type Format = 'uint32' | 'float32' | 'float32x2' | 'float32x4';
export function createTopologyOutput(
  view: <Value extends Format>(format: Value, length: number) => GraphDataView<Value>
): GPUPolygonizeOutput {
  const segments = 32;
  const rings = 8;
  const status = () => ({
    count: view('uint32', 1),
    requiredCount: view('uint32', 1),
    overflow: view('uint32', 1),
    candidateOverflow: view('uint32', 1)
  });
  return {
    nodedSegments: {
      endpoints: view('float32x4', segments),
      sourceFeatureIds: view('uint32', segments),
      sourceRingIds: view('uint32', segments),
      sourceEdgeIds: view('uint32', segments),
      sourceStartParameters: view('float32', segments),
      sourceEndParameters: view('float32', segments),
      status: status()
    },
    directedEdges: {
      endpoints: view('float32x4', 2 * segments),
      sourceSegmentIds: view('uint32', 2 * segments),
      sourceFeatureIds: view('uint32', 2 * segments),
      twinIds: view('uint32', 2 * segments),
      ringIds: view('uint32', 2 * segments),
      flags: view('uint32', 2 * segments),
      status: status()
    },
    polygons: {
      kind: 'polygons',
      positions: view('float32x2', 2 * segments),
      featureOffsets: view('uint32', rings + 1),
      polygonOffsets: view('uint32', rings + 1),
      ringOffsets: view('uint32', rings + 1),
      sourceIds: view('uint32', rings)
    },
    diagnostics: {
      edgeClasses: view('uint32', segments),
      closedRingCount: view('uint32', 1),
      openChainCount: view('uint32', 1),
      cutEdgeCount: view('uint32', 1),
      dangleCount: view('uint32', 1),
      unusedEdgeCount: view('uint32', 1)
    },
    status: status()
  };
}
