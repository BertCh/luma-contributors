// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import type {GPUSpaceTimeWeightsLayout} from '../spatial-weights/spatial-weights-space-time-stages';
import {
  GPULocationChangePoints,
  type GPULocationChangePointCost
} from './gpu-location-change-points';

/** Shared graph-value props; parameters uses GPULocationChangePoints.getParameterValues layout. */
export type LocationChangePointsOperationProps = {
  id?: string;
  values: string;
  parameters: string;
  layout: GPUSpaceTimeWeightsLayout;
  cellCount: number;
  sliceCount: number;
  cost: GPULocationChangePointCost;
  maximumChangePoints: number;
  output: {
    changePointCount: string;
    changePoints: string;
    requiredChangePointCounts?: string;
    optimalCost?: string;
    status?: {overflow: string};
  };
};

/** Vector operation form; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPULocationChangePointsOperation extends GPUContributorOperation {
  constructor(props: LocationChangePointsOperationProps) {
    super({
      id: props.id ?? 'location-change-points',
      inputs: {values: props.values, parameters: props.parameters},
      outputs: {
        changePointCount: {
          value: props.output.changePointCount,
          format: 'uint32',
          length: props.cellCount
        },
        changePoints: {
          value: props.output.changePoints,
          format: 'uint32',
          length: props.cellCount * props.maximumChangePoints
        },
        ...(props.output.requiredChangePointCounts !== undefined
          ? {
              requiredChangePointCounts: {
                value: props.output.requiredChangePointCounts,
                format: 'uint32' as const,
                length: props.cellCount
              }
            }
          : {}),
        ...(props.output.optimalCost !== undefined
          ? {
              optimalCost: {
                value: props.output.optimalCost,
                format: 'float32' as const,
                length: props.cellCount
              }
            }
          : {}),
        ...(props.output.status !== undefined
          ? {overflow: {value: props.output.status.overflow, format: 'uint32' as const, length: 1}}
          : {})
      },
      createContributor: getView =>
        new GPULocationChangePoints({
          ...props,
          values: getView<'float32'>('values'),
          parameters: getView<'float32'>('parameters'),
          output: {
            changePointCount: getView<'uint32'>('changePointCount'),
            changePoints: getView<'uint32'>('changePoints'),
            requiredChangePointCounts:
              props.output.requiredChangePointCounts !== undefined
                ? getView<'uint32'>('requiredChangePointCounts')
                : undefined,
            optimalCost:
              props.output.optimalCost !== undefined
                ? getView<'float32'>('optimalCost')
                : undefined,
            status:
              props.output.status !== undefined
                ? {overflow: getView<'uint32'>('overflow')}
                : undefined
          }
        })
    });
  }
}
