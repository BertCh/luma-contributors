// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  GPUSphericalArcOverlay,
  type GPUSphericalArcOverlayProps
} from './gpu-spherical-arc-overlay';

/** Packed graph operands and caller-sized output capacities. Input lengths come from vectors. */
export type SphericalArcOverlayOperationProps = Pick<
  GPUSphericalArcOverlayProps,
  'id' | 'spatialContext' | 'referencePoint' | 'operation' | 'tolerance'
> & {
  left: {
    positions: string;
    ringOffsets: string;
    referenceInside: boolean;
  };
  right: {
    positions: string;
    ringOffsets: string;
    referenceInside: boolean;
  };
  capacity: number;
  ringCapacity: number;
  output: {positions: string; ringOffsets: string; counts: string; status: string};
};

/** GPU spherical arrangement as a graph operation; use GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUSphericalArcOverlayOperation extends GPUContributorOperation {
  constructor(props: SphericalArcOverlayOperationProps) {
    const id = props.id ?? 'spherical-arc-overlay';
    super({
      id,
      inputs: {
        leftPositions: props.left.positions,
        leftRingOffsets: props.left.ringOffsets,
        rightPositions: props.right.positions,
        rightRingOffsets: props.right.ringOffsets
      },
      outputs: {
        positions: {value: props.output.positions, format: 'float32x3', length: props.capacity},
        ringOffsets: {
          value: props.output.ringOffsets,
          format: 'uint32',
          length: props.ringCapacity + 1
        },
        counts: {value: props.output.counts, format: 'uint32', length: 4},
        status: {value: props.output.status, format: 'uint32', length: 1}
      },
      createContributor: getView =>
        new GPUSphericalArcOverlay({
          ...props,
          id,
          left: {
            positions: getView<'float32x3'>('leftPositions'),
            ringOffsets: getView<'uint32'>('leftRingOffsets'),
            referenceInside: props.left.referenceInside
          },
          right: {
            positions: getView<'float32x3'>('rightPositions'),
            ringOffsets: getView<'uint32'>('rightRingOffsets'),
            referenceInside: props.right.referenceInside
          },
          output: {
            positions: getView<'float32x3'>('positions'),
            ringOffsets: getView<'uint32'>('ringOffsets'),
            counts: getView<'uint32'>('counts'),
            status: getView<'uint32'>('status')
          }
        })
    });
  }
}
