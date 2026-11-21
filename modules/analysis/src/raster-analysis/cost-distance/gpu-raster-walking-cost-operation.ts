// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {
  snapshotRasterGrid,
  snapshotRasterWalkingModel
} from '../raster-grid/raster-construction-snapshot';
import {GPURasterWalkingCost, type GPURasterWalkingCostProps} from '../gpu-raster-walking-cost';
import {
  resolveRasterOperationBand,
  snapshotRasterOperationBand,
  validateRasterOperationOutputNames,
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
    props = snapshotConstructionFields(props, [
      'id',
      'grid',
      'cellSizeMode',
      'model',
      'maximumIterations',
      'elevation',
      'friction',
      'sources',
      'parameters',
      'output'
    ]);
    const output = snapshotConstructionFields(props.output, ['costs', 'validity', 'status']);
    props = Object.freeze({
      ...props,
      grid: snapshotRasterGrid(props.grid),
      model: snapshotRasterWalkingModel(props.model, props.id ?? 'raster-walking-cost'),
      elevation: snapshotRasterOperationBand(props.elevation),
      friction: snapshotRasterOperationBand(props.friction),
      output: Object.freeze({
        ...output,
        status: snapshotConstructionFields(output.status, [
          'converged',
          'iterationLimitReached',
          'iterationCount',
          'invalidCount'
        ])
      })
    });
    const id = props.id ?? 'raster-walking-cost';
    validateRasterOperationOutputNames(
      id,
      {
        'output.costs': props.output.costs,
        'output.validity': props.output.validity,
        'output.status.converged': props.output.status.converged
      },
      {
        'output.status.iterationLimitReached': props.output.status.iterationLimitReached,
        'output.status.iterationCount': props.output.status.iterationCount,
        'output.status.invalidCount': props.output.status.invalidCount
      },
      {
        'elevation.storage.values': props.elevation.storage.values,
        'friction.storage.values': props.friction.storage.values,
        ...(props.elevation.validity !== undefined
          ? {'elevation.validity': props.elevation.validity}
          : {}),
        ...(props.friction.validity !== undefined
          ? {'friction.validity': props.friction.validity}
          : {}),
        sources: props.sources,
        parameters: props.parameters
      }
    );
    const length = props.grid.width * props.grid.height;
    super({
      id,
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
        converged: {value: props.output.status.converged, format: 'uint32', length: 1},
        ...Object.fromEntries(
          Object.entries(props.output.status)
            .filter(([port, value]) => port !== 'converged' && value !== undefined)
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
