// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import type {GPUVectorFormat} from '@luma.gl/graph/gpu-data';
import type {S2CaptureSpec, S2Uploads} from './s2-lens-run';
import {runExample as runHierarchy} from './s2-cell-hierarchy.example';
import {runExample as runMeasures} from './s2-cell-measures.example';
import {runExample as runCovering} from './s2-cap-covering.example';
import {runExample as runContains} from './spherical-convex-contains.example';
import {runExample as runIntersection} from './spherical-convex-intersection.example';

import {runExample as runS2PolygonCovering} from './s2-polygon-covering.example';
import {runExample as runSphericalPolygonContains} from './spherical-polygon-contains.example';
import {runExample as runSphericalPolygonOverlay} from './spherical-polygon-overlay.example';
import {runExample as runSphericalArcOverlay} from './spherical-arc-overlay.example';

/** Small analytic fixtures. Every displayed value is captured from an actual GPU input/output. */
export async function buildGlobalCoordinateScenario(
  element: string,
  graph: GPUCommandGraph<undefined>,
  uploads: S2Uploads
): Promise<Record<string, S2CaptureSpec>> {
  const captures: Record<string, S2CaptureSpec> = {};
  const input = <Format extends GPUVectorFormat>(
    name: string,
    format: Format,
    values: number[]
  ) => {
    const view = uploads.view(
      name,
      format.startsWith('uint') ? Uint32Array.from(values) : Float32Array.from(values),
      format
    );
    captures[name] = {view, role: 'input'};
    return view;
  };
  const output = <Format extends GPUVectorFormat>(name: string, format: Format, length: number) => {
    const components = Number(format.match(/x([234])$/)?.[1] ?? 1);
    const view = uploads.view(
      name,
      format.startsWith('uint')
        ? new Uint32Array(length * components)
        : new Float32Array(length * components),
      format,
      length
    );
    captures[name] = {view, role: 'output'};
    return view;
  };
  if (element === 'GPUS2CellHierarchy' || element === 'GPUS2CellMeasures') {
    // Face root IDs are exactly (2 * face + 1) << 60, stored low/high.
    const cells = input(
      'inputCells',
      'uint32x2',
      Array.from({length: 6}, (_, face) => [0, (2 * face + 1) * 0x10000000]).flat()
    );
    if (element === 'GPUS2CellHierarchy')
      await runHierarchy(graph, {
        cells,
        operation: 'children',
        output: {cells: output('children', 'uint32x2', 24), status: output('status', 'uint32', 6)}
      });
    else
      await runMeasures(graph, {
        cells,
        output: {
          measures: output('areaPerimeter', 'float32x2', 6),
          status: output('status', 'uint32', 6)
        }
      });
  } else if (element === 'GPUS2CapCovering') {
    await runCovering(graph, {
      caps: input('caps', 'float32x4', [0, 0, 1, 0.1]),
      level: 0,
      capacity: 6,
      output: {
        cells: output('coverCells', 'uint32x2', 6),
        counts: output('counts', 'uint32x2', 1),
        status: output('status', 'uint32', 1)
      }
    });
  } else if (element === 'GPUSphericalConvexContains') {
    await runContains(graph, {
      positions: input('octantVertices', 'float32x3', [1, 0, 0, 0, 1, 0, 0, 0, 1]),
      offsets: input('polygonOffsets', 'uint32', [0, 3]),
      queries: input('queries', 'float32x3', [
        1 / Math.sqrt(3),
        1 / Math.sqrt(3),
        1 / Math.sqrt(3),
        -1,
        0,
        0,
        1,
        0,
        0
      ]),
      polygonIndices: input('polygonIndices', 'uint32', [0, 0, 0]),
      boundary: 'closed',
      output: {contains: output('contains', 'uint32', 3), status: output('status', 'uint32', 3)}
    });
  } else if (element === 'GPUSphericalConvexIntersection') {
    await runIntersection(graph, {
      firstPositions: input('firstVertices', 'float32x3', [1, 0, 0, 0, 1, 0, 0, 0, 1]),
      secondPositions: input('secondVertices', 'float32x3', [1, 0, 0, 0, 1, 0, 0, 0, 1]),
      firstOffsets: input('firstOffsets', 'uint32', [0, 3]),
      secondOffsets: input('secondOffsets', 'uint32', [0, 3]),
      capacity: 6,
      output: {
        positions: output('intersectionVertices', 'float32x3', 6),
        counts: output('counts', 'uint32x2', 1),
        status: output('status', 'uint32', 1),
        measures: output('areaFractionsUnion', 'float32x4', 1)
      }
    });
  } else if (element === 'GPUSphericalPolygonContains') {
    await runSphericalPolygonContains(graph, {
      positions: input('vertices', 'float32x3', [1, 0, 0, 0, 1, 0, 0, 0, 1]),
      ringOffsets: input('ringOffsets', 'uint32', [0, 3]),
      polygonOffsets: input('polygonOffsets', 'uint32', [0, 1]),
      queries: input('queries', 'float32x3', [
        1 / Math.sqrt(3),
        1 / Math.sqrt(3),
        1 / Math.sqrt(3),
        -1,
        0,
        0,
        1,
        0,
        0
      ]),
      polygonIndices: input('polygonIndices', 'uint32', [0, 0, 0]),
      output: {contains: output('contains', 'uint32', 3), status: output('status', 'uint32', 3)}
    });
  } else if (element === 'GPUS2PolygonCovering') {
    await runS2PolygonCovering(graph, {
      positions: input('vertices', 'float32x3', [1, 0, 0, 0, 1, 0, 0, 0, 1]),
      ringOffsets: input('ringOffsets', 'uint32', [0, 3]),
      maximumLevel: 2,
      capacity: 96,
      output: {
        cells: output('coverCells', 'uint32x2', 96),
        counts: output('counts', 'uint32x2', 1),
        status: output('status', 'uint32', 1),
        visitedCount: output('visited', 'uint32', 1)
      }
    });
  } else if (element === 'GPUSphericalArcOverlay') {
    const vertices = [1, 0, 0, 0, 1, 0, 0, 0, 1];
    await runSphericalArcOverlay(graph, {
      left: {
        positions: input('left', 'float32x3', vertices),
        ringOffsets: input('leftOffsets', 'uint32', [0, 3]),
        referenceInside: false
      },
      right: {
        positions: input('right', 'float32x3', vertices),
        ringOffsets: input('rightOffsets', 'uint32', [0, 3]),
        referenceInside: false
      },
      referencePoint: [-1 / Math.sqrt(3), -1 / Math.sqrt(3), -1 / Math.sqrt(3)],
      operation: 'intersection',
      output: {
        positions: output('intersectionVertices', 'float32x3', 16),
        ringOffsets: output('rings', 'uint32', 5),
        counts: output('counts', 'uint32', 4),
        status: output('status', 'uint32', 1)
      }
    });
  } else if (element === 'GPUSphericalPolygonOverlay') {
    const vertices = [1, 0, 0, 0, 1, 0, 0, 0, 1];
    const operand = (name: string) => ({
      kind: 'polygons' as const,
      positions: input(name + 'Vertices', 'float32x3', vertices),
      ringOffsets: input(name + 'Rings', 'uint32', [0, 3]),
      polygonOffsets: input(name + 'Polygons', 'uint32', [0, 1]),
      featureOffsets: input(name + 'Features', 'uint32', [0, 1])
    });
    // The overlay result has no feature requirement (`requiredCount`); its boundary has one.
    const resultStatus = (name: string) => ({
      count: output(name + 'Count', 'uint32', 1),
      overflow: output(name + 'Overflow', 'uint32', 1),
      candidateOverflow: output(name + 'CandidateOverflow', 'uint32', 1)
    });
    const status = (name: string) => ({
      ...resultStatus(name),
      requiredCount: output(name + 'Required', 'uint32', 1)
    });
    await runSphericalPolygonOverlay(graph, {
      left: operand('left'),
      right: operand('right'),
      chartCenter: [1 / Math.sqrt(3), 1 / Math.sqrt(3), 1 / Math.sqrt(3)],
      operation: 'intersection',
      vertexTolerance: 2 ** -18,
      capacity: {intersections: 32, nodedSegments: 64},
      output: {
        geometry: {
          kind: 'polygons',
          positions: output('intersectionVertices', 'float32x3', 64),
          ringOffsets: output('rings', 'uint32', 9),
          polygonOffsets: output('polygons', 'uint32', 9),
          featureOffsets: output('features', 'uint32', 9)
        },
        status: resultStatus('result'),
        complement: output('complement', 'uint32', 1),
        boundary: {
          endpoints: output('boundaryEndpoints', 'float32x4', 64),
          operandIds: output('boundaryOperands', 'uint32', 64),
          sourceFeatureIds: output('boundarySources', 'uint32', 64),
          status: status('boundary')
        }
      }
    });
  } else throw new Error(`Unknown global-coordinate example: ${element}`);
  return captures;
}
