// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  GPUBrownianBridgeUtilization,
  resolveBrownianBridgeTailPolicy,
  type GPUBrownianBridgeUtilizationProps
} from './gpu-brownian-bridge-utilization';

/** Graph-value names replacing contributor views; lengths and grid size are compile-time. */
export type BrownianBridgeUtilizationOperationProps = Pick<
  GPUBrownianBridgeUtilizationProps,
  'id' | 'spatialContext' | 'temporalContext' | 'gridSize' | 'subStepCount' | 'tailPolicy'
> & {
  positions: string;
  timestamps: string;
  timestampFormat?: 'float32' | 'uint32x2';
  trackOffsets: string;
  motionVariances: string;
  locationErrors: string;
  parameters: string;
  rowCount: number;
  output: {density: string};
};

/** GPU bridge density as a graph operation; use GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUBrownianBridgeUtilizationOperation extends GPUContributorOperation {
  constructor(props: BrownianBridgeUtilizationOperationProps) {
    const id = props.id ?? 'brownian-bridge-utilization';
    validateRequirement(
      id,
      props.timestampFormat === undefined ||
        props.timestampFormat === 'float32' ||
        props.timestampFormat === 'uint32x2',
      `timestampFormat must be float32 or uint32x2; received ${props.timestampFormat}`
    );
    props = {...props, gridSize: [...props.gridSize], output: {...props.output}};
    resolveBrownianBridgeTailPolicy(id, props.tailPolicy, 'truncate-at-seven-sigma');
    super({
      id,
      inputs: {
        positions: props.positions,
        timestamps: props.timestamps,
        trackOffsets: props.trackOffsets,
        motionVariances: props.motionVariances,
        locationErrors: props.locationErrors,
        parameters: props.parameters
      },
      outputs: {
        density: {
          value: props.output.density,
          format: 'float32',
          length: props.gridSize[0] * props.gridSize[1]
        }
      },
      createContributor: getView => {
        const positions = getView<'float32x2'>('positions');
        validateRequirement(
          id,
          positions.length === props.rowCount,
          `positions length must equal declared rowCount ${props.rowCount}; received ${positions.length}`
        );
        return new GPUBrownianBridgeUtilization({
          id,
          spatialContext: props.spatialContext,
          temporalContext: props.temporalContext,
          gridSize: props.gridSize,
          subStepCount: props.subStepCount,
          tailPolicy: props.tailPolicy,
          positions,
          timestamps:
            props.timestampFormat === 'uint32x2'
              ? getView<'uint32x2'>('timestamps')
              : getView<'float32'>('timestamps'),
          trackOffsets: getView<'uint32'>('trackOffsets'),
          motionVariances: getView<'float32'>('motionVariances'),
          locationErrors: getView<'float32'>('locationErrors'),
          parameters: getView<'float32'>('parameters'),
          output: {density: getView<'float32'>('density')}
        });
      }
    });
  }
}
