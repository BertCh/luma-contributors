// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {snapshotRasterGrid} from '../raster-grid/raster-construction-snapshot';
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
      status: {count: string; overflow: string; requiredCount?: string; candidateOverflow?: string};
    };
    status?: {iterationLimitReached?: string};
  };
};

/** Vector operation form; compile with `GPU_CONTRIBUTOR_OPERATION_LOWERING`. */
export class GPUCostDistancePathOperation extends GPUContributorOperation {
  constructor(props: CostDistancePathOperationProps) {
    props = snapshotConstructionFields(props, [
      'id',
      'grid',
      'backLinks',
      'target',
      'capacity',
      'output'
    ]);
    const id = props.id ?? 'cost-distance-path';
    const capturedPath = snapshotConstructionFields(props.output.path, ['ids', 'status']);
    const path = Object.freeze({
      ...capturedPath,
      status: snapshotConstructionFields(capturedPath.status, [
        'count',
        'overflow',
        'requiredCount',
        'candidateOverflow'
      ])
    });
    const grid = snapshotRasterGrid(props.grid);
    const requiredCount = path.status.requiredCount;
    const candidateOverflow = path.status.candidateOverflow;
    const iterationLimitReached = props.output.status?.iterationLimitReached;
    super({
      id,
      inputs: {backLinks: props.backLinks, target: props.target},
      outputs: {
        ids: {value: path.ids, format: 'uint32', length: props.capacity},
        count: {value: path.status.count, format: 'uint32', length: 1},
        overflow: {value: path.status.overflow, format: 'uint32', length: 1},
        ...(requiredCount !== undefined
          ? {requiredCount: {value: requiredCount, format: 'uint32' as const, length: 1}}
          : {}),
        ...(candidateOverflow !== undefined
          ? {candidateOverflow: {value: candidateOverflow, format: 'uint32' as const, length: 1}}
          : {}),
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
          grid,
          backLinks: getView<'uint32'>('backLinks'),
          target: getView<'uint32'>('target'),
          output: {
            path: {
              ids: getView<'uint32'>('ids'),
              status: {
                count: getView<'uint32'>('count'),
                overflow: getView<'uint32'>('overflow'),
                ...(requiredCount !== undefined
                  ? {requiredCount: getView<'uint32'>('requiredCount')}
                  : {}),
                ...(candidateOverflow !== undefined
                  ? {candidateOverflow: getView<'uint32'>('candidateOverflow')}
                  : {})
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
