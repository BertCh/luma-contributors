// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import type {GPUVectorFormat} from '@luma.gl/graph/gpu-data';
import type {S2CaptureSpec, S2Uploads} from './s2-lens-run';
import {runExample as runGPUNormalizeGeometry} from './normalize-geometry.example';
import {runExample as runGPUCanonicalGeometryEquality} from './canonical-geometry-equality.example';
import {runExample as runGPUVectorStraightSkeleton} from './vector-straight-skeleton.example';
import {runExample as runGPUOffsetCurveComponents} from './offset-curve-components.example';
import {runExample as normalize} from './normalize-polygons.example';
import {runExample as equal} from './canonical-polygon-equality.example';
import {runExample as subdivide} from './subdivide-polygons.example';
import {runExample as reducePolygon} from './polygon-precision-reduction.example';
import {runExample as reduceLine} from './line-precision-reduction.example';
import {runExample as snap} from './segment-snap.example';
import {runExample as removeLoops} from './line-loop-removal.example';
import {runExample as hull} from './ratio-concave-hull.example';
import {runExample as circle} from './maximum-inscribed-circle.example';
import {runExample as skeleton} from './convex-straight-skeleton.example';

/** Small explicit examples: all arithmetic is performed by the recorded GPU graph. */
export async function buildGeometryRefinementScenario(
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
  const polygon = (prefix: string, coordinates = [4, 2, 0, 2, 0, 0, 4, 0]) => ({
    positions: input(`${prefix}-positions`, 'float32x2', coordinates),
    ringOffsets: input(`${prefix}-ring-offsets`, 'uint32', [0, 4]),
    polygonOffsets: input(`${prefix}-polygon-offsets`, 'uint32', [0, 1]),
    featureOffsets: input(`${prefix}-feature-offsets`, 'uint32', [0, 1])
  });
  const canonicalOutput = (prefix: string) => ({
    positions: output(`${prefix}-positions`, 'float32x2', 4),
    ringOffsets: output(`${prefix}-ring-offsets`, 'uint32', 2),
    polygonOffsets: output(`${prefix}-polygon-offsets`, 'uint32', 2)
  });
  switch (element) {
    case 'GPUNormalizeGeometry': {
      await runGPUNormalizeGeometry(graph, {
        positions: input('input-positions', 'float32x2', [4, 0, 0, 0, 2, 2]),
        nodes: input('input-nodes', 'uint32x4', [2, 0, 2, 10, 0, 2, 1, 11, 7, 0, 2, 12]),
        children: input('input-children', 'uint32', [1, 0]),
        output: {
          positions: output('normalized-positions', 'float32x2', 3),
          children: output('normalized-children', 'uint32', 2),
          status: output('status', 'uint32', 3)
        }
      });
      break;
    }
    case 'GPUCanonicalGeometryEquality': {
      const left = {
        positions: input('left-positions', 'float32x2', [0, 0, 4, 0, 2, 2]),
        nodes: input('left-nodes', 'uint32x4', [2, 0, 2, 10, 0, 2, 1, 11, 7, 0, 2, 12]),
        children: input('left-children', 'uint32', [0, 1])
      };
      const right = {
        positions: input('right-positions', 'float32x2', [2, 2, 0, 0, 4, 0]),
        nodes: input('right-nodes', 'uint32x4', [0, 0, 1, 99, 2, 1, 2, 98, 7, 0, 2, 97]),
        children: input('right-children', 'uint32', [1, 0])
      };
      await runGPUCanonicalGeometryEquality(graph, {
        left,
        right,
        pairs: input('root-pairs', 'uint32x2', [2, 2]),
        output: output('equality', 'uint32', 1)
      });
      break;
    }
    case 'GPUVectorStraightSkeleton': {
      await runGPUVectorStraightSkeleton(graph, {
        positions: input(
          'positions',
          'float32x2',
          [0, 0, 10, 0, 10, 10, 0, 10, 3, 3, 3, 7, 7, 7, 7, 3]
        ),
        ringOffsets: input('ring-offsets', 'uint32', [0, 4, 8]),
        polygonOffsets: input('polygon-offsets', 'uint32', [0, 2]),
        featureOffsets: input('feature-offsets', 'uint32', [0, 1]),
        maximumVertices: 8,
        maximumEvents: 16,
        segmentCapacity: 32,
        offsetDistance: 0.25,
        output: {
          segments: output('segments', 'float32x4', 32),
          sourceEdges: output('source-edges', 'uint32x2', 32),
          offsetPositions: output('offset-positions', 'float32x2', 40),
          offsetLinks: output('offset-links', 'uint32x2', 40),
          status: output('status', 'uint32x4', 1)
        }
      });
      break;
    }
    case 'GPUOffsetCurveComponents': {
      await runGPUOffsetCurveComponents(graph, {
        original: {
          positions: input('original-positions', 'float32x2', [0, 0, 4, 0, 4, 4]),
          lineOffsets: input('original-offsets', 'uint32', [0, 3])
        },
        raw: {
          positions: input('raw-positions', 'float32x2', [0, 0.5, 3.5, 0.5, 3.5, 4]),
          lineOffsets: input('raw-offsets', 'uint32', [0, 3])
        },
        distance: 0.5,
        tolerance: 0.001,
        segmentCapacity: 16,
        intersectionCapacity: 16,
        output: {
          positions: output('positions', 'float32x2', 32),
          sourceEdges: output('source-edges', 'uint32', 32),
          components: output('components', 'uint32x4', 8),
          status: output('status', 'uint32x4', 2)
        }
      });
      break;
    }
    case 'GPUNormalizePolygons':
      await normalize(graph, {...polygon('input'), output: canonicalOutput('normalized')});
      break;
    case 'GPUCanonicalPolygonEquality': {
      const left = polygon('left');
      const right = polygon('right', [0, 0, 0, 2, 4, 2, 4, 0]);
      const leftOutput = canonicalOutput('left-normalized');
      const rightOutput = canonicalOutput('right-normalized');
      await normalize(graph, {...left, id: 'normalize-left', output: leftOutput});
      await normalize(graph, {...right, id: 'normalize-right', output: rightOutput});
      await equal(graph, {
        left: {...left, ...leftOutput},
        right: {...right, ...rightOutput},
        tolerance: 0,
        output: output('equality', 'uint32', 1)
      });
      break;
    }
    case 'GPUSubdividePolygons':
      await subdivide(graph, {
        ...polygon('input'),
        maximumVertices: 4,
        maximumWork: 128,
        output: {
          positions: output('piece-positions', 'float32x2', 16),
          pieces: output('pieces', 'uint32x4', 4),
          status: output('status', 'uint32x4', 1)
        }
      });
      break;
    case 'GPUPolygonPrecisionReduction': {
      const status = (prefix: string) => ({
        count: output(`${prefix}-count`, 'uint32', 1),
        requiredCount: output(`${prefix}-required-count`, 'uint32', 1),
        overflow: output(`${prefix}-overflow`, 'uint32', 1),
        candidateOverflow: output(`${prefix}-candidate-overflow`, 'uint32', 1),
        uncertainCount: output(`${prefix}-uncertain-count`, 'uint32', 1)
      });
      await reducePolygon(graph, {
        polygons: {kind: 'polygons', ...polygon('input', [0.2, 0.2, 4.2, 0.2, 4.2, 2.2, 0.2, 2.2])},
        intersectionCapacity: 32,
        gridSize: 1,
        mode: 'valid-output',
        lattice: output('lattice', 'float32', 4),
        output: {
          nodedSegments: {
            endpoints: output('noded-endpoints', 'float32x4', 32),
            sourceFeatureIds: output('noded-feature-ids', 'uint32', 32),
            sourceRingIds: output('noded-ring-ids', 'uint32', 32),
            sourceEdgeIds: output('noded-edge-ids', 'uint32', 32),
            sourceStartParameters: output('noded-start-parameters', 'float32', 32),
            sourceEndParameters: output('noded-end-parameters', 'float32', 32),
            status: status('noded')
          },
          directedEdges: {
            endpoints: output('directed-endpoints', 'float32x4', 64),
            sourceSegmentIds: output('directed-segment-ids', 'uint32', 64),
            sourceFeatureIds: output('directed-feature-ids', 'uint32', 64),
            twinIds: output('directed-twin-ids', 'uint32', 64),
            ringIds: output('directed-ring-ids', 'uint32', 64),
            flags: output('directed-flags', 'uint32', 64),
            status: status('directed')
          },
          polygons: {
            kind: 'polygons',
            positions: output('polygon-positions', 'float32x2', 64),
            featureOffsets: output('polygon-feature-offsets', 'uint32', 9),
            polygonOffsets: output('polygon-offsets', 'uint32', 9),
            ringOffsets: output('polygon-ring-offsets', 'uint32', 9),
            sourceIds: output('polygon-source-ids', 'uint32', 8)
          },
          diagnostics: {
            edgeClasses: output('edge-classes', 'uint32', 32),
            closedRingCount: output('closed-ring-count', 'uint32', 1),
            openChainCount: output('open-chain-count', 'uint32', 1),
            cutEdgeCount: output('cut-edge-count', 'uint32', 1),
            dangleCount: output('dangle-count', 'uint32', 1),
            unusedEdgeCount: output('unused-edge-count', 'uint32', 1)
          },
          status: {
            count: output('result-count', 'uint32', 1),
            requiredRingCount: output('result-required-rings', 'uint32', 1),
            overflow: output('result-overflow', 'uint32', 1),
            candidateOverflow: output('result-candidate-overflow', 'uint32', 1),
            uncertainCount: output('result-uncertain-count', 'uint32', 1)
          }
        }
      });
      break;
    }
    case 'GPULinePrecisionReduction':
      await reduceLine(graph, {
        positions: input('positions', 'float32x2', [0.1, 0.1, 0.2, 0.2]),
        lineOffsets: input('line-offsets', 'uint32', [0, 2]),
        gridSize: 1,
        mode: 'keep-collapsed',
        output: {
          positions: output('rounded-positions', 'float32x2', 2),
          lineOffsets: output('rounded-offsets', 'uint32', 2),
          status: output('status', 'uint32', 1)
        }
      });
      break;
    case 'GPUSegmentSnap':
      await snap(graph, {
        positions: input('positions', 'float32x2', [0, 0, 6, 0]),
        lineOffsets: input('line-offsets', 'uint32', [0, 2]),
        referencePositions: input('references', 'float32x2', [4, 0.1, 2, 0]),
        tolerance: 0.2,
        maximumVerticesPerLine: 6,
        output: {
          positions: output('snapped-positions', 'float32x2', 6),
          sourceEdges: output('source-edges', 'uint32', 6),
          counts: output('counts', 'uint32x4', 1),
          issueCode: output('issue-code', 'uint32', 1)
        }
      });
      break;
    case 'GPULineLoopRemoval':
      await removeLoops(graph, {
        lines: {
          kind: 'lines',
          positions: input('positions', 'float32x2', [0, 0, 4, 4, 0, 4, 4, 0]),
          lineOffsets: input('line-offsets', 'uint32', [0, 4])
        },
        intersectionCapacity: 16,
        output: {
          positions: output('erased-positions', 'float32x2', 16),
          lineOffsets: output('erased-offsets', 'uint32', 2),
          sourceEdges: output('source-edges', 'uint32', 16),
          status: output('status', 'uint32x4', 1)
        }
      });
      break;
    case 'GPURatioConcaveHull':
      await hull(graph, {
        positions: input('positions', 'float32x2', [0, 0, 0, 3, 1, 1, 4, 0, 5, 3]),
        ratio: 0.1,
        allowHoles: false,
        maximumIterations: 8,
        output: {
          triangles: output('triangles', 'uint32x3', 5),
          triangleValidity: output('triangle-validity', 'uint32', 5),
          boundaryEdges: output('boundary-edges', 'uint32x2', 15),
          edgeValidity: output('edge-validity', 'uint32', 15),
          status: {converged: output('converged', 'uint32', 1)}
        }
      });
      break;
    case 'GPUMaximumInscribedCircle':
      await circle(graph, {
        ...polygon('input'),
        tolerance: 0.01,
        cellCapacity: 128,
        maximumIterations: 128,
        output: {
          circles: output('circles', 'float32x4', 1),
          witnesses: output('witnesses', 'float32x2', 1),
          status: output('status', 'uint32', 1)
        }
      });
      break;
    case 'GPUConvexStraightSkeleton':
      await skeleton(graph, {
        positions: input('positions', 'float32x2', [0, 0, 4, 0, 4, 2, 0, 2]),
        ringOffsets: input('ring-offsets', 'uint32', [0, 4]),
        maximumVertices: 4,
        maximumIterations: 8,
        segmentCapacity: 8,
        offsetDistance: 0.5,
        output: {
          segments: output('segments', 'float32x4', 8),
          sourceEdges: output('source-edges', 'uint32x2', 8),
          offsetPositions: output('offset-positions', 'float32x2', 4),
          status: output('status', 'uint32x4', 1)
        }
      });
      break;
    default:
      throw new Error(`Unknown geometry refinement lens ${element}`);
  }
  return captures;
}
