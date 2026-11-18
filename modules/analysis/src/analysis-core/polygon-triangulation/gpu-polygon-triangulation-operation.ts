// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../operations/gpu-contributor-operation';
import {validateCapacity} from '../kernels/gpu-contributor-utils';
import {
  snapshotConstructionFields,
  snapshotConstructionSpatialContext
} from '../kernels/construction-snapshot';
import {
  GPUPolygonTriangulation,
  type GPUPolygonTriangulationProps
} from './gpu-polygon-triangulation';

/** Graph value names in place of the contributor's packed views. */
export type PolygonTriangulationOperationProps = Pick<
  GPUPolygonTriangulationProps,
  | 'id'
  | 'spatialContext'
  | 'maximumWork'
  | 'useZOrderHash'
  | 'largePolygonMethod'
  | 'monotoneMinimumVertices'
  | 'monotoneMaximumVertices'
> & {
  positions: string;
  featureOffsets: string;
  polygonOffsets: string;
  ringOffsets: string;
  /** Number of polygons (polygonOffsets rows minus one). */
  polygonCount: number;
  /** Caller-sized index buffer, including padding and spare capacity. */
  indexCapacity: number;
  output: {
    indices: string;
    valid: string;
    triangleCount?: string;
    triangleOffsets?: string;
  };
};

/** @internal Capture one shape/port contract for both operation implementations. */
export function snapshotPolygonTriangulationOperationProps(
  props: PolygonTriangulationOperationProps
): PolygonTriangulationOperationProps {
  const snapshot = snapshotConstructionFields(props, [
    'id',
    'spatialContext',
    'maximumWork',
    'useZOrderHash',
    'largePolygonMethod',
    'monotoneMinimumVertices',
    'monotoneMaximumVertices',
    'positions',
    'featureOffsets',
    'polygonOffsets',
    'ringOffsets',
    'polygonCount',
    'indexCapacity',
    'output'
  ]);
  return Object.freeze({
    ...snapshot,
    ...(snapshot.spatialContext
      ? {spatialContext: snapshotConstructionSpatialContext(snapshot.spatialContext)}
      : {}),
    output: snapshotConstructionFields(snapshot.output, [
      'indices',
      'valid',
      'triangleCount',
      'triangleOffsets'
    ])
  });
}

/**
 * Polygon triangulation as a graph operation. Compile with GPU_CONTRIBUTOR_OPERATION_LOWERING.
 * Every numeric property is compile-time; all input contents are per-frame graph values.
 * The output retains the contributor's polygon slices, including degenerate padding.
 * The shared operation base requires polygonCount and indexCapacity to be at least one;
 * empty polygon slots are supported. The pure CPU function also accepts zero-length outputs.
 */
export class GPUPolygonTriangulationOperation extends GPUContributorOperation {
  constructor(props: PolygonTriangulationOperationProps) {
    props = snapshotPolygonTriangulationOperationProps(props);
    const id = props.id ?? 'polygon-triangulation';
    if (props.maximumWork !== undefined) validateCapacity(id, 'maximumWork', props.maximumWork);
    const {output, polygonCount, indexCapacity} = props;
    super({
      id,
      inputs: {
        positions: props.positions,
        featureOffsets: props.featureOffsets,
        polygonOffsets: props.polygonOffsets,
        ringOffsets: props.ringOffsets
      },
      outputs: {
        indices: {value: output.indices, format: 'uint32', length: indexCapacity},
        valid: {value: output.valid, format: 'uint32', length: polygonCount},
        ...(output.triangleCount !== undefined
          ? {triangleCount: {value: output.triangleCount, format: 'uint32', length: polygonCount}}
          : {}),
        ...(output.triangleOffsets !== undefined
          ? {
              triangleOffsets: {
                value: output.triangleOffsets,
                format: 'uint32',
                length: polygonCount + 1
              }
            }
          : {})
      },
      createContributor: getView =>
        new GPUPolygonTriangulation({
          id,
          spatialContext: props.spatialContext,
          maximumWork: props.maximumWork,
          useZOrderHash: props.useZOrderHash,
          largePolygonMethod: props.largePolygonMethod,
          monotoneMinimumVertices: props.monotoneMinimumVertices,
          monotoneMaximumVertices: props.monotoneMaximumVertices,
          positions: getView<'float32x2'>('positions'),
          featureOffsets: getView<'uint32'>('featureOffsets'),
          polygonOffsets: getView<'uint32'>('polygonOffsets'),
          ringOffsets: getView<'uint32'>('ringOffsets'),
          output: {
            indices: getView<'uint32'>('indices'),
            valid: getView<'uint32'>('valid'),
            triangleCount:
              output.triangleCount !== undefined ? getView<'uint32'>('triangleCount') : undefined,
            triangleOffsets:
              output.triangleOffsets !== undefined
                ? getView<'uint32'>('triangleOffsets')
                : undefined
          }
        })
    });
  }
}
