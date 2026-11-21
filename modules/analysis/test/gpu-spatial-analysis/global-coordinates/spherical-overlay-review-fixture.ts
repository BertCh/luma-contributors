// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  GPUSphericalPolygonOverlay,
  type GPUSphericalPolygonOverlayProps
} from '../../../src/gpu-spatial-analysis/global-coordinates/gpu-spherical-polygon-overlay';
import {recordWGSLKernelNodes} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createOracleFixture} from './output-oracle-fixture';
import {
  GPUPolygonOverlay,
  type GPUPolygonOverlayProps
} from '../../../src/gpu-spatial-analysis/polygon-overlay/gpu-polygon-overlay';

export function createPlanarIdentityFixture(device: Device) {
  const fixture = createOracleFixture(device);
  const operand = (points: number[], identities: number[]) => ({
    kind: 'polygons' as const,
    positions: fixture.input('float32x2', points),
    ringOffsets: fixture.input('uint32', [0, 3]),
    polygonOffsets: fixture.input('uint32', [0, 1]),
    featureOffsets: fixture.input('uint32', [0, 0, 1]),
    sourceIds: fixture.input('uint32', identities)
  });
  const outputs = {
    positions: fixture.output('float32x2', 16),
    rings: fixture.output('uint32', 3),
    polygons: fixture.output('uint32', 3),
    features: fixture.output('uint32', 3),
    count: fixture.output('uint32', 1),
    overflow: fixture.output('uint32', 1),
    candidate: fixture.output('uint32', 1),
    endpoints: fixture.output('float32x4', 8),
    operands: fixture.output('uint32', 8),
    owners: fixture.output('uint32', 8),
    boundaryCount: fixture.output('uint32', 1),
    boundaryRequired: fixture.output('uint32', 1),
    boundaryOverflow: fixture.output('uint32', 1),
    boundaryCandidate: fixture.output('uint32', 1),
    sourceRows: fixture.output('uint32', 2),
    sourceOperands: fixture.output('uint32', 2),
    mixed: fixture.output('uint32', 2)
  };
  const props: GPUPolygonOverlayProps = {
    id: 'planar-identity',
    left: operand([0, 0, 2, 0, 0, 2], [999, 101]),
    right: operand([1, 0, 3, 0, 1, 2], [888, 202]),
    operation: 'intersection',
    vertexTolerance: 2 ** -16,
    capacity: {intersections: 32, nodedSegments: 32},
    output: {
      geometry: {
        kind: 'polygons',
        positions: outputs.positions.view,
        ringOffsets: outputs.rings.view,
        polygonOffsets: outputs.polygons.view,
        featureOffsets: outputs.features.view
      },
      status: {
        count: outputs.count.view,
        overflow: outputs.overflow.view,
        candidateOverflow: outputs.candidate.view
      },
      boundary: {
        endpoints: outputs.endpoints.view,
        operandIds: outputs.operands.view,
        sourceFeatureIds: outputs.owners.view,
        status: {
          count: outputs.boundaryCount.view,
          requiredCount: outputs.boundaryRequired.view,
          overflow: outputs.boundaryOverflow.view,
          candidateOverflow: outputs.boundaryCandidate.view
        }
      },
      featureSources: {
        sourceRows: outputs.sourceRows.view,
        operandIds: outputs.sourceOperands.view,
        mixed: outputs.mixed.view
      }
    }
  };
  recordWGSLKernelNodes(fixture.graph);
  const nodes = new GPUPolygonOverlay(props).getCommandNodes(fixture.graph);
  fixture.graph.add(nodes);
  return {...fixture, props, outputs, nodes};
}

export function createSphericalReviewFixture(
  device: Device,
  options: {
    right?: number[][];
    leftRow?: number;
    rightRow?: number;
    leftComplement?: boolean;
    operation?: GPUSphericalPolygonOverlayProps['operation'];
    boundaryCapacity?: number;
    ringCapacity?: number;
    leftIds?: number[];
    rightIds?: number[];
    packBoundary?: boolean;
    packFeatureComplement?: boolean;
  } = {}
) {
  const fixture = createOracleFixture(device);
  const operand = (coordinates: number[][], row: number, sourceIds?: number[]) => ({
    kind: 'polygons' as const,
    positions: fixture.input(
      'float32x3',
      coordinates.flatMap(([horizontal, vertical]) => {
        const length = Math.hypot(horizontal, vertical, 1);
        return [horizontal / length, vertical / length, 1 / length];
      })
    ),
    ringOffsets: fixture.input('uint32', [0, 3]),
    polygonOffsets: fixture.input('uint32', [0, 1]),
    featureOffsets: fixture.input('uint32', [...Array(row + 1).fill(0), 1]),
    ...(sourceIds ? {sourceIds: fixture.input('uint32', sourceIds)} : {})
  });
  const ringCapacity = options.ringCapacity ?? 2;
  const boundaryCapacity = options.boundaryCapacity ?? 8;
  const outputs = {
    positions: fixture.output('float32x3', 16),
    rings: fixture.output('uint32', ringCapacity + 1),
    polygons: fixture.output('uint32', ringCapacity + 1),
    features: fixture.output('uint32', ringCapacity + 1),
    count: fixture.output('uint32', 1),
    overflow: fixture.output('uint32', 1),
    candidate: fixture.output('uint32', 1),
    ringCount: fixture.output('uint32', 1),
    requiredRingCount: fixture.output('uint32', 1),
    complement: fixture.output('uint32', 1),
    endpoints: fixture.output('float32x4', boundaryCapacity),
    operands: fixture.output('uint32', boundaryCapacity),
    owners: fixture.output('uint32', boundaryCapacity),
    boundaryCount: fixture.output('uint32', 1),
    boundaryRequired: fixture.output('uint32', 1),
    boundaryOverflow: fixture.output('uint32', 1),
    boundaryCandidate: fixture.output('uint32', 1),
    sourceRows: fixture.output('uint32', ringCapacity),
    sourceOperands: fixture.output('uint32', ringCapacity),
    mixed: fixture.output('uint32', ringCapacity)
  };
  if (options.packBoundary) {
    const packed = fixture.output('uint32', 128);
    outputs.operands = {
      ...outputs.operands,
      view: fixture.graph.createDataView(packed.view.buffer, {
        format: 'uint32',
        byteOffset: 0,
        length: boundaryCapacity
      }),
      read: async () => (await packed.read()).slice(0, boundaryCapacity)
    };
    outputs.owners = {
      ...outputs.owners,
      view: fixture.graph.createDataView(packed.view.buffer, {
        format: 'uint32',
        byteOffset: 256,
        length: boundaryCapacity
      }),
      read: async () => (await packed.read()).slice(64, 64 + boundaryCapacity)
    };
  }
  if (options.packFeatureComplement) {
    const packed = fixture.output('uint32', 128);
    outputs.sourceRows = {
      ...outputs.sourceRows,
      view: fixture.graph.createDataView(packed.view.buffer, {
        format: 'uint32',
        byteOffset: 0,
        length: ringCapacity
      }),
      read: async () => (await packed.read()).slice(0, ringCapacity),
      poison: () => packed.poison()
    };
    outputs.complement = {
      ...outputs.complement,
      view: fixture.graph.createDataView(packed.view.buffer, {
        format: 'uint32',
        byteOffset: 256,
        length: 1
      }),
      read: async () => (await packed.read()).slice(64, 65),
      poison: () => packed.poison()
    };
  }
  const props: GPUSphericalPolygonOverlayProps = {
    id: 'spherical-review',
    left: operand(
      [
        [0, 0],
        [2, 0],
        [0, 2]
      ],
      options.leftRow ?? 0,
      options.leftIds
    ),
    right: operand(
      options.right ?? [
        [1, 0],
        [3, 0],
        [1, 2]
      ],
      options.rightRow ?? 0,
      options.rightIds
    ),
    operation: options.operation ?? 'intersection',
    leftComplement: options.leftComplement,
    chartCenter: [0, 0, 1],
    vertexTolerance: 2 ** -16,
    capacity: {intersections: 32, nodedSegments: 32},
    output: {
      geometry: {
        kind: 'polygons',
        positions: outputs.positions.view,
        ringOffsets: outputs.rings.view,
        polygonOffsets: outputs.polygons.view,
        featureOffsets: outputs.features.view
      },
      status: {
        count: outputs.count.view,
        overflow: outputs.overflow.view,
        candidateOverflow: outputs.candidate.view,
        ringCount: outputs.ringCount.view,
        requiredRingCount: outputs.requiredRingCount.view
      },
      complement: outputs.complement.view,
      boundary: {
        endpoints: outputs.endpoints.view,
        operandIds: outputs.operands.view,
        sourceFeatureIds: outputs.owners.view,
        status: {
          count: outputs.boundaryCount.view,
          requiredCount: outputs.boundaryRequired.view,
          overflow: outputs.boundaryOverflow.view,
          candidateOverflow: outputs.boundaryCandidate.view
        }
      },
      featureSources: {
        sourceRows: outputs.sourceRows.view,
        operandIds: outputs.sourceOperands.view,
        mixed: outputs.mixed.view
      }
    }
  };
  recordWGSLKernelNodes(fixture.graph);
  const nodes = new GPUSphericalPolygonOverlay(props).getCommandNodes(fixture.graph);
  fixture.graph.add(nodes);
  return {...fixture, outputs, props, nodes};
}
