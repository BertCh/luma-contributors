// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Caller-owned output buffers of the topology operations (GPUPolygonize, GPUMakeValid,
// GPUPolygonPrecisionReduction): noded segments, directed edges, polygons, diagnostics, status.

import type {GPUPolygonizeOutput} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {RunBuffer, RunBuilder} from '../../../define-run';

export type TopologyCapacity = {segments: number; rings: number; vertices: number};

export type TopologyBuffers = {
  output: GPUPolygonizeOutput;
  /** Output polygon vertices and ring offsets, for drawing. */
  positions: RunBuffer<'float32x2'>;
  ringOffsets: RunBuffer<'uint32'>;
  ringCapacity: number;
  vertexCapacity: number;
};

/** Allocates every output buffer of a topology operation at the given capacities. */
export function createTopologyBuffers(
  builder: RunBuilder,
  capacity: TopologyCapacity
): TopologyBuffers {
  const {segments, rings, vertices} = capacity;
  let counter = 0;
  const buffer = <Format extends 'uint32' | 'float32' | 'float32x2' | 'float32x4'>(
    format: Format,
    length: number
  ) => builder.allocate(`topology-${counter++}`, format, length);
  const status = () => ({
    count: buffer('uint32', 1).view,
    requiredCount: buffer('uint32', 1).view,
    overflow: buffer('uint32', 1).view,
    candidateOverflow: buffer('uint32', 1).view
  });
  const positions = buffer('float32x2', vertices);
  const ringOffsets = buffer('uint32', rings + 1);
  return {
    positions,
    ringOffsets,
    ringCapacity: rings,
    vertexCapacity: vertices,
    output: {
      nodedSegments: {
        endpoints: buffer('float32x4', segments).view,
        sourceFeatureIds: buffer('uint32', segments).view,
        sourceRingIds: buffer('uint32', segments).view,
        sourceEdgeIds: buffer('uint32', segments).view,
        sourceStartParameters: buffer('float32', segments).view,
        sourceEndParameters: buffer('float32', segments).view,
        status: status()
      },
      directedEdges: {
        endpoints: buffer('float32x4', 2 * segments).view,
        sourceSegmentIds: buffer('uint32', 2 * segments).view,
        sourceFeatureIds: buffer('uint32', 2 * segments).view,
        twinIds: buffer('uint32', 2 * segments).view,
        ringIds: buffer('uint32', 2 * segments).view,
        flags: buffer('uint32', 2 * segments).view,
        status: status()
      },
      polygons: {
        kind: 'polygons',
        positions: positions.view,
        featureOffsets: buffer('uint32', rings + 1).view,
        polygonOffsets: buffer('uint32', rings + 1).view,
        ringOffsets: ringOffsets.view,
        sourceIds: buffer('uint32', rings).view
      },
      diagnostics: {
        edgeClasses: buffer('uint32', segments).view,
        closedRingCount: buffer('uint32', 1).view,
        openChainCount: buffer('uint32', 1).view,
        cutEdgeCount: buffer('uint32', 1).view,
        dangleCount: buffer('uint32', 1).view,
        unusedEdgeCount: buffer('uint32', 1).view
      },
      status: status()
    }
  };
}
