// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUNormalizeGeometry,
  GPUCanonicalGeometryEquality,
  GPUVectorStraightSkeleton,
  GPUOffsetCurveComponents,
  GPUNormalizePolygons,
  GPUCanonicalPolygonEquality,
  GPUSubdividePolygons,
  GPUPolygonPrecisionReduction,
  GPULinePrecisionReduction,
  GPUSegmentSnap,
  GPULineLoopRemoval,
  GPURatioConcaveHull,
  GPUMaximumInscribedCircle,
  GPUConvexStraightSkeleton
} from '../../../src/gpu-spatial-analysis/geometry-refinement/index';
import type {KernelAuditCase, KernelAuditContext} from './kernel-audit-context';
import {createTopologyOutput} from '../../gpu-spatial-analysis/geometry-refinement/topology-output';

const cases: KernelAuditCase[] = [];
function add(exportName: string, name: string, build: KernelAuditCase['build']) {
  cases.push({exportName, name, build});
}
function polygons(context: KernelAuditContext, vertices = 128) {
  return {
    positions: context.view('float32x2', vertices),
    ringOffsets: context.view('uint32', 5),
    polygonOffsets: context.view('uint32', 3),
    featureOffsets: context.view('uint32', 3)
  };
}
add(
  'GPUNormalizePolygons',
  'multi-feature polygons with holes and canonical part ordering',
  context => {
    const input = polygons(context);
    context.graph.add(
      new GPUNormalizePolygons({
        ...input,
        output: {
          positions: context.view('float32x2', 128),
          ringOffsets: context.view('uint32', 5),
          polygonOffsets: context.view('uint32', 3)
        }
      })
    );
  }
);
add(
  'GPUCanonicalPolygonEquality',
  'two independent geometries, nonzero coordinate tolerance',
  context => {
    context.graph.add(
      new GPUCanonicalPolygonEquality({
        left: polygons(context),
        right: polygons(context),
        tolerance: 1e-5,
        output: context.view('uint32', 2)
      })
    );
  }
);
add(
  'GPUSubdividePolygons',
  'maximum 128-vertex convex pieces, multi-feature holes, bounded work',
  context => {
    context.graph.add(
      new GPUSubdividePolygons({
        ...polygons(context),
        maximumVertices: 128,
        maximumWork: 4096,
        output: {
          positions: context.view('float32x2', 128 * 128),
          pieces: context.view('uint32x4', 128),
          status: context.view('uint32x4', 1)
        }
      })
    );
  }
);
add('GPUSubdividePolygons', 'rectilinear work queue and clipped bounded pieces', context => {
  context.graph.add(
    new GPUSubdividePolygons({
      ...polygons(context),
      method: 'rectilinear',
      maximumVertices: 4,
      maximumSplits: 65536,
      output: {
        positions: context.view('float32x2', 512),
        pieces: context.view('uint32x4', 128),
        status: context.view('uint32x4', 1)
      }
    })
  );
});
for (const mode of ['valid-output', 'keep-collapsed'] as const) {
  add(
    'GPUPolygonPrecisionReduction',
    `${mode}, hot-pixel topology with source IDs and uncertainty outputs`,
    context => {
      const output = createTopologyOutput(context.view);
      for (const status of [
        output.status,
        output.nodedSegments.status,
        output.directedEdges.status
      ]) {
        status.uncertainCount = context.view('uint32', 1);
      }
      context.graph.add(
        new GPUPolygonPrecisionReduction({
          polygons: {
            kind: 'polygons',
            ...polygons(context, 16),
            sourceIds: context.view('uint32', 2)
          },
          intersectionCapacity: 64,
          gridSize: 0.5,
          mode,
          lattice: context.view('float32', 4),
          output
        })
      );
    }
  );
  add('GPULinePrecisionReduction', `${mode}, multiple lines with non-binary grid`, context => {
    context.graph.add(
      new GPULinePrecisionReduction({
        positions: context.view('float32x2', 128),
        lineOffsets: context.view('uint32', 17),
        gridSize: 0.1,
        mode,
        output: {
          positions: context.view('float32x2', 128),
          lineOffsets: context.view('uint32', 17),
          status: context.view('uint32', 16)
        }
      })
    );
  });
}
add(
  'GPUSegmentSnap',
  'all outputs including validity, reference insertions and endpoint snapping',
  context => {
    context.graph.add(
      new GPUSegmentSnap({
        positions: context.view('float32x2', 128),
        lineOffsets: context.view('uint32', 17),
        referencePositions: context.view('float32x2', 128),
        tolerance: 0.5,
        maximumVerticesPerLine: 128,
        output: {
          positions: context.view('float32x2', 2048),
          sourceEdges: context.view('uint32', 2048),
          counts: context.view('uint32x4', 16),
          validity: context.view('uint32', 16)
        }
      })
    );
  }
);
add(
  'GPULineLoopRemoval',
  'exact noding and chronological loop erasure across multiple lines',
  context => {
    context.graph.add(
      new GPULineLoopRemoval({
        lines: {
          kind: 'lines',
          positions: context.view('float32x2', 128),
          lineOffsets: context.view('uint32', 17)
        },
        intersectionCapacity: 256,
        output: {
          positions: context.view('float32x2', 1024),
          lineOffsets: context.view('uint32', 17),
          sourceEdges: context.view('uint32', 1024),
          status: context.view('uint32x4', 1)
        }
      })
    );
  }
);
for (const allowHoles of [false, true]) {
  add('GPURatioConcaveHull', `border erosion with allowHoles=${allowHoles}`, context => {
    const capacity = 2 * 128 - 5;
    context.graph.add(
      new GPURatioConcaveHull({
        positions: context.view('float32x2', 128),
        ratio: 0.2,
        allowHoles,
        maximumIterations: capacity,
        output: {
          triangles: context.view('uint32x3', capacity),
          triangleValidity: context.view('uint32', capacity),
          boundaryEdges: context.view('uint32x2', capacity * 3),
          edgeValidity: context.view('uint32', capacity * 3),
          status: {converged: context.view('uint32', 1)}
        }
      })
    );
  });
}
add(
  'GPUMaximumInscribedCircle',
  'maximum 4096 live cells, holes and multiple features, every output',
  context => {
    context.graph.add(
      new GPUMaximumInscribedCircle({
        ...polygons(context),
        tolerance: 1e-5,
        cellCapacity: 4096,
        maximumIterations: 65536,
        output: {
          circles: context.view('float32x4', 2),
          witnesses: context.view('float32x2', 2),
          status: context.view('uint32', 2)
        }
      })
    );
  }
);
add(
  'GPUConvexStraightSkeleton',
  'maximum 64 vertices and 128 events, inward offset and full provenance',
  context => {
    context.graph.add(
      new GPUConvexStraightSkeleton({
        positions: context.view('float32x2', 128),
        ringOffsets: context.view('uint32', 3),
        maximumVertices: 64,
        maximumIterations: 128,
        segmentCapacity: 128,
        offsetDistance: 0.5,
        eventTolerance: 1e-6,
        output: {
          segments: context.view('float32x4', 256),
          sourceEdges: context.view('uint32x2', 256),
          offsetPositions: context.view('float32x2', 128),
          status: context.view('uint32x4', 2)
        }
      })
    );
  }
);

add('GPUNormalizeGeometry', 'nested mixed tree with explicit node statuses', context => {
  context.graph.add(
    new GPUNormalizeGeometry({
      positions: context.view('float32x2', 256),
      nodes: context.view('uint32x4', 128),
      children: context.view('uint32', 128),
      output: {
        positions: context.view('float32x2', 256),
        children: context.view('uint32', 128),
        status: context.view('uint32', 128)
      }
    })
  );
});
add(
  'GPUCanonicalGeometryEquality',
  'paired mixed roots, tolerance and nonfinite status',
  context => {
    const tree = () => ({
      positions: context.view('float32x2', 256),
      nodes: context.view('uint32x4', 128),
      children: context.view('uint32', 128)
    });
    context.graph.add(
      new GPUCanonicalGeometryEquality({
        left: tree(),
        right: tree(),
        pairs: context.view('uint32x2', 128),
        tolerance: 1e-6,
        output: context.view('uint32', 128)
      })
    );
  }
);
add(
  'GPUVectorStraightSkeleton',
  'maximum1024vertices4096events,holes and full offset links',
  context => {
    context.graph.add(
      new GPUVectorStraightSkeleton({
        ...polygons(context, 1024),
        maximumVertices: 1024,
        maximumEvents: 4096,
        segmentCapacity: 8192,
        offsetDistance: 0.25,
        output: {
          segments: context.view('float32x4', 16384),
          sourceEdges: context.view('uint32x2', 16384),
          offsetPositions: context.view('float32x2', 18432),
          offsetLinks: context.view('uint32x2', 18432),
          status: context.view('uint32x4', 2)
        }
      })
    );
  }
);
add('GPUOffsetCurveComponents', 'noding,side/distance selection,multipart provenance', context => {
  const original = {
    positions: context.view('float32x2', 32),
    lineOffsets: context.view('uint32', 3)
  };
  const raw = {positions: context.view('float32x2', 544), lineOffsets: context.view('uint32', 3)};
  context.graph.add(
    new GPUOffsetCurveComponents({
      original,
      raw,
      rowsPerVertex: 17,
      distance: 1,
      tolerance: 0.01,
      segmentCapacity: 1024,
      intersectionCapacity: 512,
      output: {
        positions: context.view('float32x2', 2048),
        sourceEdges: context.view('uint32', 2048),
        components: context.view('uint32x4', 1024),
        status: context.view('uint32x4', 2)
      }
    })
  );
});

export const COVERAGE_GEOMETRY_CASES: readonly KernelAuditCase[] = cases;
