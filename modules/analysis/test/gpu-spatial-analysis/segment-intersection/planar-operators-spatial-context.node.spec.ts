// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPU_PLANAR_SPATIAL_CONTEXT,
  type GPUSpatialContext
} from '../../../src/analysis-core/contracts/spatial-context';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {GPUAffineTransform} from '../../../src/gpu-spatial-analysis/geometry-edit/gpu-affine-transform';
import {GPUGeometryCleanup} from '../../../src/gpu-spatial-analysis/geometry-edit/gpu-geometry-cleanup';
import {GPUGeometryOrientation} from '../../../src/gpu-spatial-analysis/geometry-edit/gpu-geometry-orientation';
import {GPUGeometryPredicates} from '../../../src/gpu-spatial-analysis/geometry-predicates/gpu-geometry-predicates';
import {GPULineNoding} from '../../../src/gpu-spatial-analysis/geometry-topology/gpu-line-noding';
import {GPUMakeValid} from '../../../src/gpu-spatial-analysis/geometry-topology/gpu-make-valid';
import {GPUPolygonize} from '../../../src/gpu-spatial-analysis/geometry-topology/gpu-polygonize';
import {GPUGeometryValidity} from '../../../src/gpu-spatial-analysis/geometry-validity/gpu-geometry-validity';
import {GPULineClipByPolygon} from '../../../src/gpu-spatial-analysis/line-clip/gpu-line-clip-by-polygon';
import {GPUSharedPaths} from '../../../src/gpu-spatial-analysis/line-clip/gpu-shared-paths';
import {GPULineMerge} from '../../../src/gpu-spatial-analysis/line-merge/gpu-line-merge';
import {GPULineSplit} from '../../../src/gpu-spatial-analysis/line-split/gpu-line-split';
import {GPUBufferSurface} from '../../../src/gpu-spatial-analysis/polygon-buffer/gpu-buffer-surface';
import {GPUCoverageSimplification} from '../../../src/gpu-spatial-analysis/polygon-coverage-simplification/gpu-coverage-simplification';
import {GPUPolygonOverlay} from '../../../src/gpu-spatial-analysis/polygon-overlay/gpu-polygon-overlay';
import {GPUPolygonTriangulation} from '../../../src/gpu-spatial-analysis/polygon-triangulation/gpu-polygon-triangulation';
import {GPURectangleClip} from '../../../src/gpu-spatial-analysis/rectangle-clip/gpu-rectangle-clip';
import {GPUSegmentIntersection} from '../../../src/gpu-spatial-analysis/segment-intersection/gpu-segment-intersection';
import {GPUVertexSnap} from '../../../src/gpu-spatial-analysis/vertex-snap/gpu-vertex-snap';

// The matrix checks coordinate-space support. Constructors that capture nested ports before
// validation need complete fixtures; the separate getter tests pin validation order explicitly.
type PlanarConstructor = new (props: never) => {readonly spatialContext: unknown};
// GPULineSimplification has a great-circle path; its context matrix is in line-simplification/.
const PLANAR_OPERATORS: Record<string, unknown> = {
  GPUAffineTransform,
  GPUBufferSurface,
  GPUCoverageSimplification,
  GPUGeometryCleanup,
  GPUGeometryOrientation,
  GPUGeometryPredicates,
  GPUGeometryValidity,
  GPULineClipByPolygon,
  GPULineMerge,
  GPULineNoding,
  GPULineSplit,
  GPUMakeValid,
  GPUPolygonOverlay,
  GPUPolygonTriangulation,
  GPUPolygonize,
  GPURectangleClip,
  GPUSegmentIntersection,
  GPUSharedPaths,
  GPUVertexSnap
};

// GPULineMerge and GPURectangleClip also accept longitude-latitude contexts (linear edges).
const LONGITUDE_LATITUDE_OPERATORS = new Set(['GPULineMerge', 'GPURectangleClip']);

const COMPLETE_FIXTURE_OPERATORS = [
  'GPUCoverageSimplification',
  'GPUGeometryPredicates',
  'GPULineClipByPolygon',
  'GPULineSplit',
  'GPUPolygonOverlay',
  'GPUVertexSnap'
] as const;

function createCompleteOperatorFactories(): Record<
  string,
  (spatialContext: GPUSpatialContext | undefined) => {readonly spatialContext: unknown}
> {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'float32x4' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `context-fixture-${serial++}`, format, length);
  const polygons = () => ({
    kind: 'polygons' as const,
    positions: view('float32x2', 4),
    ringOffsets: view('uint32', 2),
    polygonOffsets: view('uint32', 2),
    featureOffsets: view('uint32', 2)
  });
  const lines = () => ({
    kind: 'lines' as const,
    positions: view('float32x2', 2),
    lineOffsets: view('uint32', 2)
  });
  const status = () => ({
    count: view('uint32', 1),
    overflow: view('uint32', 1),
    candidateOverflow: view('uint32', 1)
  });
  const pieces = () => ({geometry: lines(), sourceIds: view('uint32', 1), status: status()});
  return {
    GPUCoverageSimplification: spatialContext =>
      new GPUCoverageSimplification({
        ...polygons(),
        spatialContext,
        parameters: view('float32', 4),
        output: {
          positions: view('float32x2', 4),
          ringOffsets: view('uint32', 2),
          status: {overflow: view('uint32', 1)}
        }
      }),
    GPUGeometryPredicates: spatialContext =>
      new GPUGeometryPredicates({
        spatialContext,
        geometry: {kind: 'points', positions: view('float32x2', 1)},
        output: {numCoordinates: view('uint32', 1)}
      }),
    GPULineClipByPolygon: spatialContext =>
      new GPULineClipByPolygon({
        spatialContext,
        lines: lines(),
        polygons: polygons(),
        intersectionCapacity: 4,
        candidateCapacity: 8,
        output: {pieces: pieces()}
      }),
    GPULineSplit: spatialContext =>
      new GPULineSplit({
        spatialContext,
        lines: lines(),
        intersectionCapacity: 4,
        output: {pieces: pieces()}
      }),
    GPUPolygonOverlay: spatialContext =>
      new GPUPolygonOverlay({
        spatialContext,
        left: polygons(),
        right: polygons(),
        operation: 'intersection',
        vertexTolerance: 1e-5,
        capacity: {intersections: 8, nodedSegments: 16},
        output: {
          geometry: polygons(),
          status: status(),
          boundary: {
            endpoints: view('float32x4', 8),
            operandIds: view('uint32', 8),
            sourceFeatureIds: view('uint32', 8),
            status: {...status(), requiredCount: view('uint32', 1)}
          }
        }
      }),
    GPUVertexSnap: spatialContext =>
      new GPUVertexSnap({
        spatialContext,
        positions: view('float32x2', 1),
        referencePositions: view('float32x2', 1),
        parameters: view('float32', 4),
        output: {positions: view('float32x2', 1)}
      })
  };
}

it.each(COMPLETE_FIXTURE_OPERATORS)('%s context fixture is otherwise valid', name => {
  const create = createCompleteOperatorFactories()[name];
  // Constructor validation is independent of buffer contents. A valid planar control ensures
  // longitude-latitude rejection is not being tested with a malformed property bag.
  expect(() => create(GPU_PLANAR_SPATIAL_CONTEXT)).not.toThrow();
});

it('GPUPolygonTriangulation rejects unsupported contexts before evaluating output getters', () => {
  for (const metric of ['none', 'great-circle'] as const) {
    let outputReadCount = 0;
    expect(
      () =>
        new GPUPolygonTriangulation({
          spatialContext: {coordinateSpace: 'longitude-latitude', metric},
          get output() {
            outputReadCount++;
            throw new Error('output getter must not run for an unsupported context');
          }
        } as never)
    ).toThrow(/does not support longitude-latitude coordinates; project positions/);
    expect(outputReadCount).toBe(0);
  }
});

it('GPUSegmentIntersection rejects unsupported contexts before evaluating geometry or output getters', () => {
  for (const metric of ['none', 'great-circle'] as const) {
    let contextReadCount = 0;
    let geometryReadCount = 0;
    let outputReadCount = 0;
    expect(
      () =>
        new GPUSegmentIntersection({
          get spatialContext() {
            contextReadCount++;
            return {coordinateSpace: 'longitude-latitude', metric};
          },
          get left() {
            geometryReadCount++;
            throw new Error('geometry getter must not run for an unsupported context');
          },
          get output() {
            outputReadCount++;
            throw new Error('output getter must not run for an unsupported context');
          }
        } as never)
    ).toThrow(/does not support longitude-latitude coordinates; project positions/);
    expect(contextReadCount).toBe(1);
    expect(geometryReadCount).toBe(0);
    expect(outputReadCount).toBe(0);
  }
});

it('GPUGeometryPredicates rejects unsupported contexts before evaluating geometry getters', () => {
  for (const metric of ['none', 'great-circle'] as const) {
    let geometryReadCount = 0;
    let otherReadCount = 0;
    expect(
      () =>
        new GPUGeometryPredicates({
          spatialContext: {coordinateSpace: 'longitude-latitude', metric},
          get geometry() {
            geometryReadCount++;
            throw new Error('geometry getter must not run for an unsupported context');
          },
          get other() {
            otherReadCount++;
            throw new Error('other getter must not run for an unsupported context');
          }
        } as never)
    ).toThrow(/does not support longitude-latitude coordinates; project positions/);
    expect(geometryReadCount).toBe(0);
    expect(otherReadCount).toBe(0);
  }
});

for (const [name, operator] of Object.entries(PLANAR_OPERATORS)) {
  const acceptsLongitudeLatitude = LONGITUDE_LATITUDE_OPERATORS.has(name);
  it(`${name} ${acceptsLongitudeLatitude ? 'accepts planar and longitude-latitude' : 'rejects longitude-latitude'} spatial contexts${name === 'GPURectangleClip' ? ' and rejects the great-circle edge model' : ''}`, () => {
    const Operator = operator as PlanarConstructor;
    const createComplete = createCompleteOperatorFactories()[name];
    const createForContext = (spatialContext: GPUSpatialContext | undefined) =>
      createComplete ? createComplete(spatialContext) : new Operator({spatialContext} as never);
    for (const metric of ['none', 'great-circle'] as const) {
      const create = () => createForContext({coordinateSpace: 'longitude-latitude', metric});
      if (acceptsLongitudeLatitude) {
        try {
          expect(create().spatialContext).toBeTruthy();
        } catch (error) {
          expect(String((error as Error).message)).not.toMatch(/does not support|spatial context/);
        }
      } else {
        expect(create).toThrow(
          /does not support longitude-latitude coordinates; project positions/
        );
      }
    }
    if (name === 'GPURectangleClip') {
      expect(
        () =>
          new Operator({
            spatialContext: {
              coordinateSpace: 'longitude-latitude',
              metric: 'great-circle',
              edgeModel: 'great-circle'
            }
          } as never)
      ).toThrow(/edge/);
    }
    // The planar context passes the context check; other property validation may still throw.
    for (const spatialContext of [undefined, GPU_PLANAR_SPATIAL_CONTEXT]) {
      try {
        const instance = createForContext(spatialContext);
        expect(instance.spatialContext).toBeTruthy();
      } catch (error) {
        expect(String((error as Error).message)).not.toMatch(/does not support|spatial context/);
      }
    }
  });
}
