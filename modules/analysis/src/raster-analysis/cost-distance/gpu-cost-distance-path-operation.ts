// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUCostDistancePath} from './gpu-cost-distance-path';

/** Graph value names in place of the views of `GPUCostDistancePath`. */
export type CostDistancePathOperationProps = {
  id?: string;
  grid: GPURasterMetadata;
  /** `uint32`, one predecessor code per cell. */
  backLinks: string;
  /** `uint32`, one target cell. */
  target: string;
  /** Explicit capacity of the compact path. */
  capacity: number;
  output: {
    path: {
      ids: string;
      status: {count: string; overflow: string; requiredCount: string};
    };
    status?: {iterationLimitReached?: string};
  };
};

/** Vector operation form; compile with `GPU_CONTRIBUTOR_OPERATION_LOWERING`. */
export class GPUCostDistancePathOperation extends GPUContributorOperation {
  constructor(props: CostDistancePathOperationProps) {
    const id = props.id ?? 'cost-distance-path';
    const {path} = props.output;
    const iterationLimitReached = props.output.status?.iterationLimitReached;
    super({
      id,
      inputs: {backLinks: props.backLinks, target: props.target},
      outputs: {
        ids: {value: path.ids, format: 'uint32', length: props.capacity},
        count: {value: path.status.count, format: 'uint32', length: 1},
        overflow: {value: path.status.overflow, format: 'uint32', length: 1},
        requiredCount: {value: path.status.requiredCount, format: 'uint32', length: 1},
        ...(iterationLimitReached !== undefined
          ? {
              iterationLimitReached: {
                value: iterationLimitReached,
                format: 'uint32' as const,
                length: 1
              }
            }
          : {})
      },
      createContributor: getView =>
        new GPUCostDistancePath({
          id,
          grid: props.grid,
          backLinks: getView<'uint32'>('backLinks'),
          target: getView<'uint32'>('target'),
          output: {
            path: {
              ids: getView<'uint32'>('ids'),
              status: {
                count: getView<'uint32'>('count'),
                overflow: getView<'uint32'>('overflow'),
                requiredCount: getView<'uint32'>('requiredCount')
              }
            },
            status:
              iterationLimitReached !== undefined
                ? {iterationLimitReached: getView<'uint32'>('iterationLimitReached')}
                : undefined
          }
        })
    });
  }
}
