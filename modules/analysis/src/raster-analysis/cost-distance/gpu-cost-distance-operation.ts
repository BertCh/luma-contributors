// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUCostDistance, type GPUCostDistanceProps} from './gpu-cost-distance';
import {
  resolveRasterOperationBand,
  type RasterOperationBand
} from '../raster-grid/raster-operation-band';

/** Buffer-backed cost distance. Stateful continuation is not supported by the mixed executor. */
export type CostDistanceOperationProps = Pick<
  GPUCostDistanceProps,
  | 'id'
  | 'grid'
  | 'cellSizeMode'
  | 'maximumIterations'
  | 'maximumTieIterations'
  | 'sweepAfterIteration'
> & {
  friction: RasterOperationBand;
  parameters: string;
  frictionParameters?: string;
  sources?: string;
  sourceCosts?: string;
  sourceCount?: string;
  sourceMask?: string;
  costBreaks?: string;
  costBreakCount?: number;
  output: {
    costs: string;
    settledAt?: string;
    backLinks?: string;
    bands?: string;
    bandCounts?: string;
    status?: {converged?: string; iterationLimitReached?: string; iterationCount?: string};
  };
};

/** Cost distance as a vector operation; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUCostDistanceOperation extends GPUContributorOperation {
  constructor(props: CostDistanceOperationProps) {
    const length = props.grid.width * props.grid.height;
    const optionalInputs = [
      'frictionParameters',
      'sources',
      'sourceCosts',
      'sourceCount',
      'sourceMask',
      'costBreaks'
    ] as const;
    super({
      id: props.id ?? 'cost-distance',
      inputs: {
        values: props.friction.storage.values,
        parameters: props.parameters,
        ...(props.friction.validity !== undefined ? {validity: props.friction.validity} : {}),
        ...Object.fromEntries(
          optionalInputs.filter(port => props[port] !== undefined).map(port => [port, props[port]!])
        )
      },
      outputs: {
        costs: {value: props.output.costs, format: 'float32', length},
        ...Object.fromEntries(
          (['settledAt', 'backLinks', 'bands', 'bandCounts'] as const)
            .filter(port => props.output[port] !== undefined)
            .map(port => [
              port,
              {
                value: props.output[port]!,
                format: 'uint32' as const,
                length: port === 'bandCounts' ? props.costBreakCount! : length
              }
            ])
        ),
        ...Object.fromEntries(
          Object.entries(props.output.status ?? {}).map(([port, value]) => [
            port,
            {value, format: 'uint32' as const, length: 1}
          ])
        )
      },
      createContributor: getView =>
        new GPUCostDistance({
          ...props,
          friction: resolveRasterOperationBand(props.friction, getView),
          parameters: getView<'float32'>('parameters'),
          frictionParameters:
            props.frictionParameters !== undefined
              ? getView<'float32'>('frictionParameters')
              : undefined,
          sources: props.sources !== undefined ? getView<'uint32'>('sources') : undefined,
          sourceCosts:
            props.sourceCosts !== undefined ? getView<'float32'>('sourceCosts') : undefined,
          sourceCount:
            props.sourceCount !== undefined ? getView<'uint32'>('sourceCount') : undefined,
          sourceMask: props.sourceMask !== undefined ? getView<'uint32'>('sourceMask') : undefined,
          costBreaks: props.costBreaks !== undefined ? getView<'float32'>('costBreaks') : undefined,
          output: {
            costs: getView<'float32'>('costs'),
            settledAt:
              props.output.settledAt !== undefined ? getView<'uint32'>('settledAt') : undefined,
            backLinks:
              props.output.backLinks !== undefined ? getView<'uint32'>('backLinks') : undefined,
            bands: props.output.bands !== undefined ? getView<'uint32'>('bands') : undefined,
            bandCounts:
              props.output.bandCounts !== undefined ? getView<'uint32'>('bandCounts') : undefined,
            status: {
              iterationCount:
                props.output.status?.iterationCount !== undefined
                  ? getView<'uint32'>('iterationCount')
                  : undefined,
              converged:
                props.output.status?.converged !== undefined
                  ? getView<'uint32'>('converged')
                  : undefined,
              iterationLimitReached:
                props.output.status?.iterationLimitReached !== undefined
                  ? getView<'uint32'>('iterationLimitReached')
                  : undefined
            }
          }
        })
    });
  }
}
