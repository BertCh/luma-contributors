// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {GPUSphericalArcOverlay} from '../../../src/gpu-spatial-analysis/global-coordinates/gpu-spherical-arc-overlay';
import {GPUS2PolygonCovering} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-polygon-covering';
import {GPUSphericalPolygonContains} from '../../../src/gpu-spatial-analysis/global-coordinates/gpu-spherical-polygon-contains';
import {GPUS2CellMeasures} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-cell-measures';
import {GPUS2CapCovering} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-cap-covering';
import {GPUSphericalConvexContains} from '../../../src/gpu-spatial-analysis/global-coordinates/gpu-spherical-convex-contains';
import {WgslReflect} from 'wgsl_reflect';
import {GPUCommandGraph, createTransientView} from '@luma-contributors/graph/gpu-core';
import {GPUDatumGridTransform} from '../../../src/gpu-project/gpu-datum-grid-transform';
import {GPUS2CellHierarchy} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-cell-hierarchy';
import {GPUSphericalConvexIntersection} from '../../../src/gpu-spatial-analysis/global-coordinates/gpu-spherical-convex-intersection';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';

it('global-coordinate contributors parse within baseline WebGPU bindings', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let serial = 0;
  const view = <Format extends 'float32x2' | 'float32x3' | 'float32x4' | 'uint32' | 'uint32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `global-${serial++}`, format, length);
  for (const direction of ['forward', 'inverse'] as const) {
    for (const operation of ['horizontal', 'vertical', 'combined'] as const) {
      const contributor = new GPUDatumGridTransform({
        positions: view('float32x3', 4),
        samples: view('float32x3', 4),
        validity: view('uint32', 4),
        grids: [{width: 2, height: 2, sampleOffset: 0, origin: [0, 0], spacing: [1, 1]}],
        operation,
        direction,
        sourceVerticalReference: 'ellipsoid',
        targetVerticalReference: 'orthometric',
        output: {positions: view('float32x3', 4), status: view('uint32', 4)}
      });
      for (const kernel of getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph))) {
        expect(() => new WgslReflect(kernel.source)).not.toThrow();
        expect(auditWGSLModule(kernel.source, kernel.constants)).toEqual([]);
        expect(getStorageBindingCount(kernel.source)).toBeLessThanOrEqual(8);
      }
    }
  }
  for (const operation of ['parent', 'children', 'range', 'neighbors'] as const) {
    const contributor = new GPUS2CellHierarchy({
      cells: view('uint32x2', 4),
      operation,
      level: 0,
      output: {
        cells: view(
          'uint32x2',
          operation === 'children' || operation === 'neighbors' ? 16 : operation === 'range' ? 8 : 4
        ),
        status: view('uint32', 4)
      }
    });
    for (const kernel of getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph))) {
      expect(() => new WgslReflect(kernel.source)).not.toThrow();
      expect(auditWGSLModule(kernel.source, kernel.constants)).toEqual([]);
    }
  }
  const spherical = new GPUSphericalConvexIntersection({
    firstPositions: view('float32x3', 6),
    secondPositions: view('float32x3', 6),
    firstOffsets: view('uint32', 3),
    secondOffsets: view('uint32', 3),
    capacity: 6,
    output: {
      positions: view('float32x3', 12),
      counts: view('uint32x2', 2),
      status: view('uint32', 2)
    }
  });
  for (const kernel of getGPUCommandNodeKernelSources(spherical.getCommandNodes(graph))) {
    expect(() => new WgslReflect(kernel.source)).not.toThrow();
    expect(auditWGSLModule(kernel.source, kernel.constants)).toEqual([]);
    expect(getStorageBindingCount(kernel.source)).toBe(7);
  }
  const additional = [
    new GPUSphericalArcOverlay({
      left: {
        positions: view('float32x3', 4),
        ringOffsets: view('uint32', 2),
        referenceInside: true
      },
      right: {
        positions: view('float32x3', 4),
        ringOffsets: view('uint32', 2),
        referenceInside: true
      },
      referencePoint: [0, 0, 1],
      operation: 'intersection',
      output: {
        positions: view('float32x3', 64),
        ringOffsets: view('uint32', 17),
        counts: view('uint32', 4),
        status: view('uint32', 1)
      }
    }),
    new GPUS2PolygonCovering({
      positions: view('float32x3', 8),
      ringOffsets: view('uint32', 3),
      maximumLevel: 15,
      capacity: 128,
      output: {
        cells: view('uint32x2', 128),
        counts: view('uint32x2', 1),
        status: view('uint32', 1),
        visitedCount: view('uint32', 1)
      }
    }),
    new GPUSphericalPolygonContains({
      positions: view('float32x3', 8),
      ringOffsets: view('uint32', 3),
      polygonOffsets: view('uint32', 2),
      queries: view('float32x3', 4),
      polygonIndices: view('uint32', 4),
      output: {contains: view('uint32', 4), status: view('uint32', 4)}
    }),
    new GPUSphericalConvexIntersection({
      ...spherical.props,
      output: {...spherical.props.output, measures: view('float32x4', 2)}
    }),
    new GPUS2CellMeasures({
      cells: view('uint32x2', 2),
      output: {measures: view('float32x2', 2), status: view('uint32', 2)}
    }),
    new GPUS2CapCovering({
      caps: view('float32x4', 2),
      level: 2,
      capacity: 10,
      output: {cells: view('uint32x2', 20), counts: view('uint32x2', 2), status: view('uint32', 2)}
    }),
    new GPUSphericalConvexContains({
      positions: view('float32x3', 3),
      offsets: view('uint32', 2),
      queries: view('float32x3', 2),
      polygonIndices: view('uint32', 2),
      output: {contains: view('uint32', 2), status: view('uint32', 2)}
    })
  ];
  for (const contributor of additional)
    for (const kernel of getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph))) {
      expect(auditWGSLModule(kernel.source, kernel.constants)).toEqual([]);
      expect(getStorageBindingCount(kernel.source)).toBeLessThanOrEqual(8);
    }
});

it('spherical chart composition rejects original-input aliases and complement/status aliases', async () => {
  const {GPUSphericalPolygonOverlay} = await import(
    '../../../src/gpu-spatial-analysis/global-coordinates/gpu-spherical-polygon-overlay'
  );
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32x3' | 'float32x4' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `alias-${serial++}`, format, length);
  const operand = () => ({
    kind: 'polygons' as const,
    positions: view('float32x3', 3),
    ringOffsets: view('uint32', 2),
    polygonOffsets: view('uint32', 2),
    featureOffsets: view('uint32', 2)
  });
  const status = () => ({
    count: view('uint32', 1),
    requiredCount: view('uint32', 1),
    overflow: view('uint32', 1),
    candidateOverflow: view('uint32', 1)
  });
  const left = operand();
  const right = operand();
  const resultStatus = status();
  const props = {
    left,
    right,
    chartCenter: [0, 0, 1] as const,
    operation: 'intersection' as const,
    vertexTolerance: 2 ** -18,
    capacity: {intersections: 16, nodedSegments: 32},
    output: {
      geometry: {
        kind: 'polygons' as const,
        positions: view('float32x3', 32),
        ringOffsets: view('uint32', 9),
        polygonOffsets: view('uint32', 9),
        featureOffsets: view('uint32', 9),
        sourceIds: view('uint32', 8)
      },
      status: resultStatus,
      complement: view('uint32', 1),
      boundary: {
        endpoints: view('float32x4', 32),
        operandIds: view('uint32', 32),
        sourceFeatureIds: view('uint32', 32),
        status: status()
      }
    }
  };
  expect(
    () =>
      new GPUSphericalPolygonOverlay({
        ...props,
        output: {...props.output, complement: resultStatus.overflow}
      })
  ).toThrow();
  const overlapping = graph.createDataView(left.positions.buffer, {format: 'uint32', length: 2});
  expect(
    () =>
      new GPUSphericalPolygonOverlay({
        ...props,
        output: {...props.output, geometry: {...props.output.geometry, ringOffsets: overlapping}}
      })
  ).toThrow();
  expect(
    () =>
      new GPUSphericalPolygonOverlay({
        ...props,
        output: {
          ...props.output,
          complement: graph.createDataView(left.ringOffsets.buffer, {format: 'uint32', length: 1})
        }
      })
  ).toThrow();
  device.destroy();
});
