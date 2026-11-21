// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUPolygonOverlayOutput} from '../../../src/gpu-spatial-analysis/polygon-overlay/gpu-polygon-overlay';
import {GPUBufferSurface} from '../../../src/gpu-spatial-analysis/polygon-buffer/gpu-buffer-surface';
import {createOutputOracleFixture} from '../outline-geometry/output-oracle-fixture';

export const BUFFER_OPTIONAL_STATUS = [
  'uncertainCount',
  'ringCount',
  'requiredRingCount',
  'nodedSegmentRequiredCount'
] as const;
export const UNWRITTEN_STATUS = [
  'invalidCount',
  'iterationCount',
  'converged',
  'iterationLimitReached',
  'approximation'
] as const;

export function createBufferOutputOracle(
  device: Device,
  optional?: (typeof BUFFER_OPTIONAL_STATUS)[number],
  settings: {
    positions?: number[];
    lineOffsets?: number[];
    capStyle?: 'flat' | 'square' | 'round';
    distance?: number;
    vertexTolerance?: number;
    boundaryCapacity?: number;
    vertexCapacity?: number;
    ringCounts?: boolean;
    intersections?: number;
    nodedSegments?: number;
  } = {}
) {
  const fixture = createOutputOracleFixture(device);
  const {graph, column, output} = fixture;
  const status = {
    count: output('count', 'uint32', 1),
    overflow: output('overflow', 'uint32', 1),
    candidateOverflow: output('candidateOverflow', 'uint32', 1),
    ...(optional ? {[optional]: output(optional, 'uint32', 1)} : {}),
    ...(settings.ringCounts
      ? {
          ringCount: output('ringCount', 'uint32', 1),
          requiredRingCount: output('requiredRingCount', 'uint32', 1)
        }
      : {})
  };
  const boundary = {
    endpoints: output('endpoints', 'float32x4', settings.boundaryCapacity ?? 4),
    operandIds: output('operandIds', 'uint32', settings.boundaryCapacity ?? 4),
    sourceFeatureIds: output('sourceFeatureIds', 'uint32', settings.boundaryCapacity ?? 4),
    status: {
      count: output('boundaryCount', 'uint32', 1),
      requiredCount: output('boundaryRequiredCount', 'uint32', 1),
      overflow: output('boundaryOverflow', 'uint32', 1),
      candidateOverflow: output('boundaryCandidateOverflow', 'uint32', 1)
    }
  };
  const result: GPUPolygonOverlayOutput = {
    geometry: {
      kind: 'polygons',
      // The nested assembly reserves a closing vertex, even though polygon output omits it.
      positions: output('positions', 'float32x2', settings.vertexCapacity ?? 5),
      featureOffsets: output('featureOffsets', 'uint32', 2),
      polygonOffsets: output('polygonOffsets', 'uint32', 2),
      ringOffsets: output('ringOffsets', 'uint32', 2)
    },
    status,
    boundary
  };
  const props = {
    geometry: {
      kind: 'lines' as const,
      positions: column('input', 'float32x2', new Float32Array(settings.positions ?? [0, 0, 4, 0])),
      lineOffsets: column('lines', 'uint32', new Uint32Array(settings.lineOffsets ?? [0, 2])),
      featureOffsets: column(
        'inputFeatures',
        'uint32',
        new Uint32Array([0, 0, (settings.lineOffsets?.length ?? 2) - 1])
      )
    },
    parameters: column(
      'parameters',
      'float32',
      new Float32Array([settings.distance ?? 1, 5, 0, 0])
    ),
    joinStyle: 'mitre' as const,
    capStyle: settings.capStyle ?? 'flat',
    vertexTolerance: settings.vertexTolerance ?? 0.25,
    capacity: {
      intersections: settings.intersections ?? 16,
      nodedSegments: settings.nodedSegments ?? 32
    },
    output: result
  };
  const contributor = new GPUBufferSurface(props);
  const nodes = contributor.getCommandNodes(graph);
  for (const node of nodes) graph.add(node);
  return {...fixture, props, nodes};
}
