// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPURasterWalkingCost, type GPURasterWalkingCostProps} from '../gpu-raster-walking-cost';
import {
  resolveRasterOperationBand,
  type RasterOperationBand
} from '../raster-grid/raster-operation-band';

/** Buffer-band graph values shared by both directed walking-cost operations. */
export type RasterWalkingCostOperationProps = Pick<
  GPURasterWalkingCostProps,
  'id' | 'grid' | 'cellSizeMode' | 'model' | 'maximumIterations'
> & {
  elevation: RasterOperationBand;
  friction: RasterOperationBand;
  sources: string;
  parameters: string;
  output: {
    costs: string;
    validity: string;
    status: {
      converged: string;
      iterationLimitReached?: string;
      iterationCount?: string;
      invalidCount?: string;
    };
  };
};

/** GPU directed walking cost as a vector operation; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPURasterWalkingCostOperation extends GPUContributorOperation {
  constructor(props: RasterWalkingCostOperationProps) {
    const length = props.grid.width * props.grid.height;
    super({
      id: props.id ?? 'raster-walking-cost',
      inputs: {
        elevationValues: props.elevation.storage.values,
        frictionValues: props.friction.storage.values,
        ...(props.elevation.validity !== undefined
          ? {elevationValidity: props.elevation.validity}
          : {}),
        ...(props.friction.validity !== undefined
          ? {frictionValidity: props.friction.validity}
          : {}),
        sources: props.sources,
        parameters: props.parameters
      },
      outputs: {
        costs: {value: props.output.costs, format: 'float32', length},
        validity: {value: props.output.validity, format: 'uint32', length},
        ...Object.fromEntries(
          Object.entries(props.output.status)
            .filter(([, value]) => value !== undefined)
            .map(([port, value]) => [port, {value: value!, format: 'uint32' as const, length: 1}])
        )
      },
      createContributor: getView =>
        new GPURasterWalkingCost({
          ...props,
          elevation: resolveRasterOperationBand(props.elevation, port =>
            getView(port === 'values' ? 'elevationValues' : 'elevationValidity')
          ),
          friction: resolveRasterOperationBand(props.friction, port =>
            getView(port === 'values' ? 'frictionValues' : 'frictionValidity')
          ),
          sources: getView<'uint32'>('sources'),
          parameters: getView<'float32'>('parameters'),
          output: {
            costs: getView<'float32'>('costs'),
            validity: getView<'uint32'>('validity'),
            status: {
              converged: getView<'uint32'>('converged'),
              iterationLimitReached:
                props.output.status.iterationLimitReached !== undefined
                  ? getView<'uint32'>('iterationLimitReached')
                  : undefined,
              iterationCount:
                props.output.status.iterationCount !== undefined
                  ? getView<'uint32'>('iterationCount')
                  : undefined,
              invalidCount:
                props.output.status.invalidCount !== undefined
                  ? getView<'uint32'>('invalidCount')
                  : undefined
            }
          }
        })
    });
  }
}
