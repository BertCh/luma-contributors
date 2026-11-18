// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPU_PLANAR_SPATIAL_CONTEXT} from '../../../src/analysis-core/contracts/spatial-context';
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

// Every planar-only planar-sweep operator resolves its spatial context before it reads any other
// property, so a context-only property bag is enough to observe the context check.
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
    for (const metric of ['none', 'great-circle'] as const) {
      const create = () =>
        new Operator({spatialContext: {coordinateSpace: 'longitude-latitude', metric}} as never);
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
        const instance = new Operator({spatialContext} as never);
        expect(instance.spatialContext).toBeTruthy();
      } catch (error) {
        expect(String((error as Error).message)).not.toMatch(/does not support|spatial context/);
      }
    }
  });
}
