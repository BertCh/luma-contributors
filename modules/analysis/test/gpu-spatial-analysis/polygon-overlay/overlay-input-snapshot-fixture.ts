// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {recordWGSLKernelNodes} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import type {Device} from '@luma.gl/core';
import {
  GPUPolygonOverlay,
  type GPUPolygonOverlayProps
} from '../../../src/gpu-spatial-analysis/polygon-overlay/gpu-polygon-overlay';
import {createOutputOracleFixture} from '../outline-geometry/output-oracle-fixture';

export function createOverlayInputSnapshotFixture(
  device: Device,
  options: {mutate: boolean; packed: boolean; ids: boolean}
) {
  const fixture = createOutputOracleFixture(device);
  const {graph, column, output} = fixture;
  const packedValues = new Uint32Array(66);
  packedValues.set([999, 101]);
  packedValues.set([888, 202], 64);
  const packed = column('packed-identities', 'uint32', packedValues);
  let identityReads = 0;
  const operand = (
    name: string,
    points: number[],
    identities: number[],
    byteOffset: number
  ): GPUPolygonOverlayProps['left'] => {
    const sourceIds = options.packed
      ? graph.createDataView(packed.buffer, {format: 'uint32', length: 2, byteOffset})
      : column(`${name}-ids`, 'uint32', Uint32Array.from(identities));
    const result: GPUPolygonOverlayProps['left'] = {
      kind: 'polygons',
      positions: column(`${name}-positions`, 'float32x2', Float32Array.from(points)),
      ringOffsets: column(`${name}-rings`, 'uint32', Uint32Array.of(0, 3)),
      polygonOffsets: column(`${name}-polygons`, 'uint32', Uint32Array.of(0, 1)),
      featureOffsets: column(`${name}-features`, 'uint32', Uint32Array.of(0, 0, 1))
    };
    if (options.ids)
      Object.defineProperty(result, 'sourceIds', {
        enumerable: true,
        configurable: true,
        get() {
          identityReads++;
          return sourceIds;
        }
      });
    return result;
  };
  const left = operand('left', [0, 0, 2, 0, 0, 2], [999, 101], 0);
  const right = operand('right', [1, 0, 3, 0, 1, 2], [888, 202], 256);
  const boundary = {
    endpoints: output('boundary', 'float32x4', 8),
    operandIds: output('operands', 'uint32', 8),
    sourceFeatureIds: output('owners', 'uint32', 8),
    status: {
      count: output('boundary-count', 'uint32', 1),
      requiredCount: output('boundary-required', 'uint32', 1),
      overflow: output('boundary-overflow', 'uint32', 1),
      candidateOverflow: output('boundary-candidate', 'uint32', 1)
    }
  };
  const featureSources = {
    sourceRows: output('source-rows', 'uint32', 2),
    operandIds: output('source-operands', 'uint32', 2),
    mixed: output('mixed', 'uint32', 2)
  };
  recordWGSLKernelNodes(graph);
  const originalLeftPositions = left.positions;
  const originalRightFeatures = right.featureOffsets;
  const producer = new GPUPolygonOverlay({
    id: 'snapshot-overlay',
    left,
    right,
    operation: 'intersection',
    vertexTolerance: 2 ** -16,
    capacity: {intersections: 32, nodedSegments: 32},
    output: {
      geometry: {
        kind: 'polygons',
        positions: output('result', 'float32x2', 16),
        ringOffsets: output('rings', 'uint32', 3),
        polygonOffsets: output('polygons', 'uint32', 3),
        featureOffsets: output('features', 'uint32', 3)
      },
      status: {
        count: output('count', 'uint32', 1),
        overflow: output('overflow', 'uint32', 1),
        candidateOverflow: output('candidate', 'uint32', 1)
      },
      boundary,
      featureSources
    }
  });
  const readsAtConstruction = identityReads;
  if (options.mutate) {
    delete left.sourceIds;
    delete right.sourceIds;
    // Alter the maps too; the original graph-view identities must remain captured.
    left.positions = column('changed-left', 'float32x2', new Float32Array(6));
    right.featureOffsets = column('changed-right', 'uint32', Uint32Array.of(0, 0, 0));
  }
  const nodes = producer.getCommandNodes(graph);
  return {
    ...fixture,
    nodes,
    boundary,
    featureSources,
    readsAtConstruction,
    identityReads,
    producer,
    originalLeftPositions,
    originalRightFeatures
  };
}
