// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUS2PolygonCovering,
  GPUSphericalPolygonContains,
  GPUSphericalPolygonOverlay,
  GPUSphericalArcOverlay,
  GPUS2CellHierarchy,
  GPUS2CellMeasures,
  GPUS2CapCovering,
  GPUSphericalConvexContains,
  GPUSphericalConvexIntersection
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {KernelAuditCase} from './kernel-audit-context';

const cases: KernelAuditCase[] = [];
for (const operation of ['parent', 'children', 'range', 'neighbors'] as const) {
  cases.push({
    exportName: 'GPUS2CellHierarchy',
    name: `${operation}, level30, binding-limited output`,
    build(context) {
      const stride = operation === 'parent' ? 1 : operation === 'range' ? 2 : 4;
      const count = context.maximumRows(stride * 8);
      context.graph.add(
        new GPUS2CellHierarchy({
          cells: context.view('uint32x2', count),
          operation,
          level: 30,
          output: {
            cells: context.view('uint32x2', count * stride),
            status: context.view('uint32', count)
          }
        })
      );
    }
  });
}
cases.push({
  exportName: 'GPUS2CellMeasures',
  name: 'binding-limited rows, geometry precision status',
  build(context) {
    const count = context.maximumRows(8);
    context.graph.add(
      new GPUS2CellMeasures({
        cells: context.view('uint32x2', count),
        output: {measures: context.view('float32x2', count), status: context.view('uint32', count)}
      })
    );
  }
});
for (const mode of ['exterior', 'interior'] as const)
  cases.push({
    exportName: 'GPUS2CapCovering',
    name: `${mode},level8,full candidate capacity,maximum tolerance`,
    build(context) {
      const capacity = 6 * 4 ** 8;
      const count = context.maximumRows(capacity * 8);
      context.graph.add(
        new GPUS2CapCovering({
          caps: context.view('float32x4', count),
          level: 8,
          mode,
          tolerance: 1e-2,
          capacity,
          output: {
            cells: context.view('uint32x2', count * capacity),
            counts: context.view('uint32x2', count),
            status: context.view('uint32', count)
          }
        })
      );
    }
  });
for (const mode of ['exterior', 'interior'] as const)
  cases.push({
    exportName: 'GPUS2CapCovering',
    name: `${mode},level-synchronous,level15,frontier-limited caps`,
    build(context) {
      const capacity = 1024;
      const frontierCapacity = 8 * capacity + 64;
      const count = context.maximumRows(frontierCapacity * 16);
      context.graph.add(
        new GPUS2CapCovering({
          refinement: 'level-synchronous',
          caps: context.view('float32x4', count),
          level: 15,
          mode,
          tolerance: 1e-2,
          capacity,
          output: {
            cells: context.view('uint32x2', count * capacity),
            counts: context.view('uint32x2', count),
            status: context.view('uint32', count)
          }
        })
      );
    }
  });
for (const mode of ['interior', 'exterior'] as const)
  cases.push({
    exportName: 'GPUS2PolygonCovering',
    name: `${mode},level-synchronous,level15,maximum traversal budget`,
    build(context) {
      context.graph.add(
        new GPUS2PolygonCovering({
          refinement: 'level-synchronous',
          positions: context.view('float32x3', 128),
          ringOffsets: context.view('uint32', 3),
          maximumLevel: 15,
          maximumVisitedCells: 1048576,
          capacity: 65536,
          mode,
          complement: true,
          output: {
            cells: context.view('uint32x2', 65536),
            counts: context.view('uint32x2', 1),
            status: context.view('uint32', 1),
            visitedCount: context.view('uint32', 1)
          }
        })
      );
    }
  });
for (const boundary of ['open', 'closed'] as const)
  for (const complement of [false, true])
    cases.push({
      exportName: 'GPUSphericalConvexContains',
      name: `${boundary},complement=${complement},32 vertices,binding-limited queries`,
      build(context) {
        const count = context.maximumRows(12);
        context.graph.add(
          new GPUSphericalConvexContains({
            positions: context.view('float32x3', 32),
            offsets: context.view('uint32', 2),
            queries: context.view('float32x3', count),
            polygonIndices: context.view('uint32', count),
            boundary,
            complement,
            tolerance: 1e-2,
            output: {contains: context.view('uint32', count), status: context.view('uint32', count)}
          })
        );
      }
    });
cases.push({
  exportName: 'GPUSphericalConvexIntersection',
  name: '32 vertices per input,capacity64,every output,binding-limited rows',
  build(context) {
    const count = context.maximumRows(64 * 12);
    context.graph.add(
      new GPUSphericalConvexIntersection({
        firstPositions: context.view('float32x3', count * 32),
        secondPositions: context.view('float32x3', count * 32),
        firstOffsets: context.view('uint32', count + 1),
        secondOffsets: context.view('uint32', count + 1),
        capacity: 64,
        tolerance: 1e-2,
        output: {
          positions: context.view('float32x3', count * 64),
          counts: context.view('uint32x2', count),
          status: context.view('uint32', count),
          measures: context.view('float32x4', count)
        }
      })
    );
  }
});
for (const operation of ['intersection', 'union', 'difference', 'symmetric-difference'] as const) {
  cases.push({
    exportName: 'GPUSphericalArcOverlay',
    name: `${operation},128 input vertices,1024 output vertices`,
    build(context) {
      const operand = () => ({
        positions: context.view('float32x3', 64),
        ringOffsets: context.view('uint32', 2),
        referenceInside: true
      });
      context.graph.add(
        new GPUSphericalArcOverlay({
          left: operand(),
          right: operand(),
          referencePoint: [0, 0, 1],
          operation,
          output: {
            positions: context.view('float32x3', 1024),
            ringOffsets: context.view('uint32', 343),
            counts: context.view('uint32', 4),
            status: context.view('uint32', 1)
          }
        })
      );
    }
  });
  cases.push({
    exportName: 'GPUSphericalPolygonOverlay',
    name: `${operation},chart snap topology`,
    build(context) {
      const operand = () => ({
        kind: 'polygons' as const,
        positions: context.view('float32x3', 64),
        ringOffsets: context.view('uint32', 2),
        polygonOffsets: context.view('uint32', 2),
        featureOffsets: context.view('uint32', 2)
      });
      const status = () => ({
        count: context.view('uint32', 1),
        requiredCount: context.view('uint32', 1),
        overflow: context.view('uint32', 1),
        candidateOverflow: context.view('uint32', 1)
      });
      context.graph.add(
        new GPUSphericalPolygonOverlay({
          left: operand(),
          right: operand(),
          chartCenter: [0, 0, 1],
          minimumCosine: 0.001,
          vertexTolerance: 2 ** -18,
          operation,
          leftComplement: true,
          rightComplement: true,
          capacity: {intersections: 1024, nodedSegments: 2048},
          output: {
            geometry: {
              kind: 'polygons',
              positions: context.view('float32x3', 2048),
              ringOffsets: context.view('uint32', 257),
              polygonOffsets: context.view('uint32', 257),
              featureOffsets: context.view('uint32', 257),
              sourceIds: context.view('uint32', 256)
            },
            complement: context.view('uint32', 1),
            status: status(),
            boundary: {
              endpoints: context.view('float32x4', 2048),
              operandIds: context.view('uint32', 2048),
              sourceFeatureIds: context.view('uint32', 2048),
              status: status()
            }
          }
        })
      );
    }
  });
}
for (const mode of ['interior', 'exterior'] as const)
  cases.push({
    exportName: 'GPUS2PolygonCovering',
    name: `${mode},level15,maximum traversal budget`,
    build(context) {
      context.graph.add(
        new GPUS2PolygonCovering({
          positions: context.view('float32x3', 128),
          ringOffsets: context.view('uint32', 3),
          maximumLevel: 15,
          maximumVisitedCells: 1048576,
          capacity: 65536,
          mode,
          complement: true,
          output: {
            cells: context.view('uint32x2', 65536),
            counts: context.view('uint32x2', 1),
            status: context.view('uint32', 1),
            visitedCount: context.view('uint32', 1)
          }
        })
      );
    }
  });
for (const boundary of ['open', 'closed'] as const)
  cases.push({
    exportName: 'GPUSphericalPolygonContains',
    name: `${boundary},holed,binding-limited queries`,
    build(context) {
      const count = context.maximumRows(12);
      context.graph.add(
        new GPUSphericalPolygonContains({
          positions: context.view('float32x3', 128),
          ringOffsets: context.view('uint32', 3),
          polygonOffsets: context.view('uint32', 2),
          queries: context.view('float32x3', count),
          polygonIndices: context.view('uint32', count),
          boundary,
          complement: true,
          output: {contains: context.view('uint32', count), status: context.view('uint32', count)}
        })
      );
    }
  });
export const COVERAGE_GLOBAL_CASES: readonly KernelAuditCase[] = cases;
