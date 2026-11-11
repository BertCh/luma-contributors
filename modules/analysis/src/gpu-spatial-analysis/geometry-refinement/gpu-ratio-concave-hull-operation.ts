// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {getDelaunayMaximumTriangleCount} from '../delaunay-tessellation/gpu-delaunay-tessellation';
import {GPURatioConcaveHull, type GPURatioConcaveHullProps} from './gpu-ratio-concave-hull';

/** Graph values for a ratio hull, with the point count made explicit for output allocation. */
export type RatioConcaveHullOperationProps = Pick<
  GPURatioConcaveHullProps,
  'id' | 'ratio' | 'allowHoles' | 'maximumIterations' | 'spatialContext'
> & {
  positions: string;
  pointCount: number;
  output: {
    triangles: string;
    triangleValidity: string;
    boundaryEdges: string;
    edgeValidity: string;
    status: {converged: string};
  };
};

/**
 * GPU ratio hull as a graph operation; numeric props are compile-time.
 * The shared operation base rejects zero-length outputs, so pointCount must be at least three.
 * Collinear sites are supported; the pure CPU function also accepts fewer than three sites.
 */
export class GPURatioConcaveHullOperation extends GPUContributorOperation {
  constructor(props: RatioConcaveHullOperationProps) {
    const capacity = getDelaunayMaximumTriangleCount(props.pointCount);
    super({
      id: props.id ?? 'ratio-concave-hull',
      inputs: {positions: props.positions},
      outputs: {
        triangles: {value: props.output.triangles, format: 'uint32x3', length: capacity},
        triangleValidity: {
          value: props.output.triangleValidity,
          format: 'uint32',
          length: capacity
        },
        boundaryEdges: {
          value: props.output.boundaryEdges,
          format: 'uint32x2',
          length: capacity * 3
        },
        edgeValidity: {value: props.output.edgeValidity, format: 'uint32', length: capacity * 3},
        converged: {value: props.output.status.converged, format: 'uint32', length: 1}
      },
      createContributor: getView =>
        new GPURatioConcaveHull({
          id: props.id,
          ratio: props.ratio,
          allowHoles: props.allowHoles,
          maximumIterations: props.maximumIterations,
          spatialContext: props.spatialContext,
          positions: getView<'float32x2'>('positions'),
          output: {
            triangles: getView<'uint32x3'>('triangles'),
            triangleValidity: getView<'uint32'>('triangleValidity'),
            boundaryEdges: getView<'uint32x2'>('boundaryEdges'),
            edgeValidity: getView<'uint32'>('edgeValidity'),
            status: {converged: getView<'uint32'>('converged')}
          }
        })
    });
  }
}
