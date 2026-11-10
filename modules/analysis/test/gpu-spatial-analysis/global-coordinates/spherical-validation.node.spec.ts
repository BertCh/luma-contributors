// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {GPUSphericalArcOverlay} from '../../../src/gpu-spatial-analysis/global-coordinates/gpu-spherical-arc-overlay';
import {GPUSphericalConvexContains} from '../../../src/gpu-spatial-analysis/global-coordinates/gpu-spherical-convex-contains';
import {GPUSphericalConvexIntersection} from '../../../src/gpu-spatial-analysis/global-coordinates/gpu-spherical-convex-intersection';
import {GPUSphericalPolygonContains} from '../../../src/gpu-spatial-analysis/global-coordinates/gpu-spherical-polygon-contains';
import {GPUSphericalPolygonOverlay} from '../../../src/gpu-spatial-analysis/global-coordinates/gpu-spherical-polygon-overlay';
import {GPUS2CapCovering} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-cap-covering';
import {GPUS2CellHierarchy} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-cell-hierarchy';
import {GPUS2PolygonCovering} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-polygon-covering';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

type Format = 'float32x3' | 'float32x4' | 'uint32' | 'uint32x2';

function createViews() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <F extends Format>(format: F, length: number) =>
    createTransientView(graph, `validation-${serial++}`, format, length);
  return {device, graph, view};
}

const PLANAR = {coordinateSpace: 'planar', metric: 'native'} as const;
const GREAT_CIRCLE_EDGES = {
  coordinateSpace: 'longitude-latitude',
  metric: 'none',
  edgeModel: 'great-circle'
} as const;
const LINEAR_EDGES = {
  coordinateSpace: 'longitude-latitude',
  metric: 'none',
  edgeModel: 'linear'
} as const;

it('spherical and S2 covering classes take a longitude-latitude spatial context and reject planar', () => {
  const {device, graph, view} = createViews();
  try {
    const make = (
      spatialContext?: typeof PLANAR | typeof GREAT_CIRCLE_EDGES | typeof LINEAR_EDGES
    ) => [
      () =>
        new GPUSphericalConvexContains({
          spatialContext,
          positions: view('float32x3', 3),
          offsets: view('uint32', 2),
          queries: view('float32x3', 2),
          polygonIndices: view('uint32', 2),
          output: {contains: view('uint32', 2), status: view('uint32', 2)}
        }),
      () =>
        new GPUSphericalConvexIntersection({
          spatialContext,
          firstPositions: view('float32x3', 3),
          secondPositions: view('float32x3', 3),
          firstOffsets: view('uint32', 2),
          secondOffsets: view('uint32', 2),
          capacity: 6,
          output: {
            positions: view('float32x3', 6),
            counts: view('uint32x2', 1),
            status: view('uint32', 1)
          }
        }),
      () =>
        new GPUSphericalPolygonContains({
          spatialContext,
          positions: view('float32x3', 3),
          ringOffsets: view('uint32', 2),
          polygonOffsets: view('uint32', 2),
          queries: view('float32x3', 2),
          polygonIndices: view('uint32', 2),
          output: {contains: view('uint32', 2), status: view('uint32', 2)}
        }),
      () =>
        new GPUSphericalArcOverlay({
          spatialContext,
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
            positions: view('float32x3', 16),
            ringOffsets: view('uint32', 3),
            counts: view('uint32', 4),
            status: view('uint32', 1)
          }
        }),
      () =>
        new GPUS2PolygonCovering({
          spatialContext,
          positions: view('float32x3', 3),
          ringOffsets: view('uint32', 2),
          maximumLevel: 3,
          capacity: 8,
          output: {
            cells: view('uint32x2', 8),
            counts: view('uint32x2', 1),
            status: view('uint32', 1)
          }
        }),
      () =>
        new GPUS2CapCovering({
          spatialContext,
          caps: view('float32x4', 1),
          level: 2,
          capacity: 4,
          output: {
            cells: view('uint32x2', 4),
            counts: view('uint32x2', 1),
            status: view('uint32', 1)
          }
        })
    ];
    for (const construct of make(undefined)) {
      expect(construct().spatialContext).toEqual({
        coordinateSpace: 'longitude-latitude',
        metric: 'none',
        edgeModel: 'great-circle'
      });
    }
    for (const construct of make(PLANAR)) {
      expect(construct).toThrow(/does not support planar coordinates/);
    }
    for (const construct of make(LINEAR_EDGES)) {
      expect(construct).toThrow(/needs great-circle edges/);
    }
    // The polygon overlay needs the full planar overlay output, so it is checked on its own below.
    for (const construct of make(GREAT_CIRCLE_EDGES)) {
      expect(construct().spatialContext.coordinateSpace).toBe('longitude-latitude');
    }
    expect(graph).toBeDefined();
  } finally {
    device.destroy();
  }
});

it('GPUSphericalPolygonOverlay validates its spatial context before its chart', () => {
  const {device, view} = createViews();
  try {
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
    const props = {
      left: operand(),
      right: operand(),
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
        status: status(),
        complement: view('uint32', 1),
        boundary: {
          endpoints: view('float32x4', 32),
          operandIds: view('uint32', 32),
          sourceFeatureIds: view('uint32', 32),
          status: status()
        }
      }
    };
    expect(() => new GPUSphericalPolygonOverlay({...props, spatialContext: PLANAR})).toThrow(
      /spherical-polygon-overlay does not support planar coordinates/
    );
    expect(() => new GPUSphericalPolygonOverlay({...props, chartCenter: [0, 0, 2]})).toThrow(
      /spherical-polygon-overlay chartCenter must be a finite unit vector/
    );
    expect(
      new GPUSphericalPolygonOverlay({...props, spatialContext: GREAT_CIRCLE_EDGES}).spatialContext
        .coordinateSpace
    ).toBe('longitude-latitude');
  } finally {
    device.destroy();
  }
});

it('spherical and S2 classes name the limit that a rejected prop exceeds', () => {
  const {device, view} = createViews();
  try {
    const arcOperand = (length: number) => ({
      positions: view('float32x3', length),
      ringOffsets: view('uint32', 2),
      referenceInside: true
    });
    const arc =
      (leftLength: number, outputLength = 16) =>
      () =>
        new GPUSphericalArcOverlay({
          left: arcOperand(leftLength),
          right: arcOperand(4),
          referencePoint: [0, 0, 1],
          operation: 'union',
          output: {
            positions: view('float32x3', outputLength),
            ringOffsets: view('uint32', 3),
            counts: view('uint32', 4),
            status: view('uint32', 1)
          }
        });
    expect(arc(124)).not.toThrow();
    expect(arc(125)).toThrow(/spherical-arc-overlay left plus right positions must be at most 128/);
    expect(arc(8, 1025)).toThrow(/output.positions must hold 3 to 1024 entries/);
    expect(
      () =>
        new GPUSphericalArcOverlay({
          ...arc(8)().props,
          referencePoint: [0, 0, 2],
          output: arc(8)().props.output
        })
    ).toThrow(/spherical-arc-overlay referencePoint must be a finite unit vector/);

    const intersection = (capacity: number) => () =>
      new GPUSphericalConvexIntersection({
        firstPositions: view('float32x3', 3),
        secondPositions: view('float32x3', 3),
        firstOffsets: view('uint32', 2),
        secondOffsets: view('uint32', 2),
        capacity,
        output: {
          positions: view('float32x3', capacity),
          counts: view('uint32x2', 1),
          status: view('uint32', 1)
        }
      });
    expect(intersection(64)).not.toThrow();
    expect(intersection(65)).toThrow(
      /spherical-convex-intersection capacity must be an integer in \[1, 64\]/
    );
    const contains = (tolerance: number) => () =>
      new GPUSphericalConvexContains({
        tolerance,
        positions: view('float32x3', 3),
        offsets: view('uint32', 2),
        queries: view('float32x3', 1),
        polygonIndices: view('uint32', 1),
        output: {contains: view('uint32', 1), status: view('uint32', 1)}
      });
    expect(contains(1e-2)).not.toThrow();
    expect(contains(1e-7)).not.toThrow();
    expect(contains(2e-2)).toThrow(/spherical-convex-contains tolerance must be a finite number/);
    const covering =
      (maximumLevel: number, maximumVisitedCells = 64) =>
      () =>
        new GPUS2PolygonCovering({
          positions: view('float32x3', 3),
          ringOffsets: view('uint32', 2),
          maximumLevel,
          maximumVisitedCells,
          capacity: 4,
          output: {
            cells: view('uint32x2', 4),
            counts: view('uint32x2', 1),
            status: view('uint32', 1)
          }
        });
    expect(covering(15)).not.toThrow();
    expect(covering(16)).toThrow(
      /s2-polygon-covering maximumLevel must be an integer in \[0, 15\]/
    );
    expect(covering(3, 5)).toThrow(/maximumVisitedCells must be an integer in \[6, 1048576\]/);
    expect(covering(3, 1048577)).toThrow(
      /maximumVisitedCells must be an integer in \[6, 1048576\]/
    );
    const cap = (level: number, refinement?: 'uniform' | 'level-synchronous') => () =>
      new GPUS2CapCovering({
        refinement,
        caps: view('float32x4', 1),
        level,
        capacity: 4,
        output: {
          cells: view('uint32x2', 4),
          counts: view('uint32x2', 1),
          status: view('uint32', 1)
        }
      });
    expect(cap(8)).not.toThrow();
    expect(cap(9)).toThrow(/s2-cap-covering level must be an integer in \[0, 8\]/);
    expect(cap(9, 'uniform')).toThrow(/level must be an integer in \[0, 8\]/);
    expect(cap(15, 'level-synchronous')).not.toThrow();
    expect(cap(16, 'level-synchronous')).toThrow(/level must be an integer in \[0, 15\]/);
    const hierarchy = (level: number) => () =>
      new GPUS2CellHierarchy({
        cells: view('uint32x2', 1),
        operation: 'parent',
        level,
        output: {cells: view('uint32x2', 1), status: view('uint32', 1)}
      });
    expect(hierarchy(30)).not.toThrow();
    expect(hierarchy(31)).toThrow(/s2-cell-hierarchy level must be an integer in \[0, 30\]/);
  } finally {
    device.destroy();
  }
});

it('S2 cap covering rejects a frontierCapacity below its capacity or on the uniform tier', () => {
  const {device, view} = createViews();
  try {
    const make = (refinement: 'uniform' | 'level-synchronous', frontierCapacity: number) => () =>
      new GPUS2CapCovering({
        refinement,
        frontierCapacity,
        caps: view('float32x4', 2),
        level: 4,
        capacity: 10,
        output: {
          cells: view('uint32x2', 20),
          counts: view('uint32x2', 2),
          status: view('uint32', 2)
        }
      });
    expect(make('level-synchronous', 10)).not.toThrow();
    expect(make('level-synchronous', 9)).toThrow(/frontierCapacity must be an integer in \[10,/);
    expect(make('uniform', 64)).toThrow(/frontierCapacity needs refinement 'level-synchronous'/);
  } finally {
    device.destroy();
  }
});

it('level-synchronous S2 coverings fit core shader budgets', () => {
  const {device, graph, view} = createViews();
  try {
    const contributor = new GPUS2CapCovering({
      refinement: 'level-synchronous',
      caps: view('float32x4', 3),
      level: 15,
      mode: 'interior',
      capacity: 256,
      output: {
        cells: view('uint32x2', 768),
        counts: view('uint32x2', 3),
        status: view('uint32', 3)
      }
    });
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    for (const kernel of getGPUCommandNodeKernelSources(nodes)) {
      expect(auditWGSLModule(kernel.source, kernel.constants)).toEqual([]);
    }
  } finally {
    device.destroy();
  }
});
