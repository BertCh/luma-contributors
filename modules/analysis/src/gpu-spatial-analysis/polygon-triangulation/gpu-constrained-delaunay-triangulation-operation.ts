// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  GPUPolygonTriangulationOperation,
  snapshotPolygonTriangulationOperationProps,
  type PolygonTriangulationOperationProps
} from '../../analysis-core/polygon-triangulation/gpu-polygon-triangulation-operation';
import {GPUConstrainedDelaunayTriangulation} from './gpu-constrained-delaunay-triangulation';

/** Polygon graph values with per-polygon constrained-Delaunay completion. */
export type ConstrainedDelaunayTriangulationOperationProps = Omit<
  PolygonTriangulationOperationProps,
  'output'
> & {
  maximumFlips?: number;
  flipMethod?: 'auto' | 'serial' | 'parallel';
  maximumVerticesPerPolygon?: number;
  output: PolygonTriangulationOperationProps['output'] & {status: {converged: string}};
};

/**
 * GPU constrained triangulation as a graph operation; all numeric props are compile-time.
 * The shared operation base requires at least one polygon slot and one index row.
 * Empty polygon slots are supported; the pure CPU function also accepts zero-length outputs.
 */
export class GPUConstrainedDelaunayTriangulationOperation extends GPUContributorOperation {
  constructor(props: ConstrainedDelaunayTriangulationOperationProps) {
    props = snapshotConstructionFields(props, [
      'id',
      'spatialContext',
      'positions',
      'featureOffsets',
      'polygonOffsets',
      'ringOffsets',
      'polygonCount',
      'indexCapacity',
      'maximumWork',
      'useZOrderHash',
      'largePolygonMethod',
      'monotoneMinimumVertices',
      'monotoneMaximumVertices',
      'maximumFlips',
      'flipMethod',
      'maximumVerticesPerPolygon',
      'output'
    ]);
    const snapshot = snapshotPolygonTriangulationOperationProps(props);
    props = Object.freeze({
      ...props,
      ...snapshot,
      output: Object.freeze({
        ...snapshot.output,
        status: snapshotConstructionFields(props.output.status, ['converged'])
      })
    });
    const id = props.id ?? 'constrained-delaunay';
    const polygon = new GPUPolygonTriangulationOperation({...props, id});
    super({
      id,
      inputs: polygon.inputs,
      outputs: {
        ...polygon.outputs,
        converged: {
          value: props.output.status.converged,
          format: 'uint32',
          length: props.polygonCount
        }
      },
      createContributor: getView =>
        new GPUConstrainedDelaunayTriangulation({
          id,
          spatialContext: props.spatialContext,
          maximumWork: props.maximumWork,
          useZOrderHash: props.useZOrderHash,
          largePolygonMethod: props.largePolygonMethod,
          monotoneMinimumVertices: props.monotoneMinimumVertices,
          monotoneMaximumVertices: props.monotoneMaximumVertices,
          maximumFlips: props.maximumFlips,
          flipMethod: props.flipMethod,
          maximumVerticesPerPolygon: props.maximumVerticesPerPolygon,
          positions: getView<'float32x2'>('positions'),
          featureOffsets: getView<'uint32'>('featureOffsets'),
          polygonOffsets: getView<'uint32'>('polygonOffsets'),
          ringOffsets: getView<'uint32'>('ringOffsets'),
          output: {
            indices: getView<'uint32'>('indices'),
            valid: getView<'uint32'>('valid'),
            triangleCount:
              props.output.triangleCount !== undefined
                ? getView<'uint32'>('triangleCount')
                : undefined,
            triangleOffsets:
              props.output.triangleOffsets !== undefined
                ? getView<'uint32'>('triangleOffsets')
                : undefined,
            status: {converged: getView<'uint32'>('converged')}
          }
        })
    });
  }
}
