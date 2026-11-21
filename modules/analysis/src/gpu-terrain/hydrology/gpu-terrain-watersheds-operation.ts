// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUTerrainWatersheds, type GPUTerrainWatershedsProps} from './gpu-terrain-watersheds';
import type {TerrainDrainageOperationStatus} from './cpu-terrain-drainage-utils';
import {validateCapacity} from '../../analysis-core/kernels/gpu-contributor-utils';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {snapshotRasterGrid} from '../../raster-analysis/raster-grid/raster-construction-snapshot';
import {validateTerrainFlowDirectionEncoding} from './terrain-flow-tracing';

/** Watershed graph names in place of views. Grid dimensions explicitly size the output. */
export type TerrainWatershedsOperationProps = Omit<
  GPUTerrainWatershedsProps,
  'flowDirections' | 'pourPoints' | 'output'
> & {
  flowDirections: string;
  pourPoints?: string;
  output: {labels: string; status?: TerrainDrainageOperationStatus};
};

/** @internal Shared named-field capture for GPU declaration and CPU execution. */
export function snapshotTerrainWatershedsOperationProps(
  props: TerrainWatershedsOperationProps
): TerrainWatershedsOperationProps {
  const captured = snapshotConstructionFields(props, [
    'id',
    'grid',
    'directionEncoding',
    'maximumIterations',
    'flowDirections',
    'pourPoints',
    'output'
  ]);
  const capturedGrid = snapshotRasterGrid(captured.grid);
  const output = snapshotConstructionFields(captured.output, ['labels', 'status']);
  return Object.freeze({
    ...captured,
    grid: capturedGrid,
    output: {
      ...output,
      status:
        output.status &&
        snapshotConstructionFields(output.status, [
          'converged',
          'iterationLimitReached',
          'iterationCount'
        ])
    }
  });
}

/** Vector operation form. Grid and options are compile-time; direction and point rows are per-frame. */
export class GPUTerrainWatershedsOperation extends GPUContributorOperation {
  constructor(props: TerrainWatershedsOperationProps) {
    props = snapshotTerrainWatershedsOperationProps(props);
    const id = props.id ?? 'terrain-watersheds';
    validateTerrainFlowDirectionEncoding(id, props.directionEncoding);
    if (props.maximumIterations !== undefined)
      validateCapacity(id, 'maximumIterations', props.maximumIterations, 1, 1024);
    super({
      id,
      inputs: {
        flowDirections: props.flowDirections,
        ...(props.pourPoints !== undefined ? {pourPoints: props.pourPoints} : {})
      },
      outputs: {
        labels: {
          value: props.output.labels,
          format: 'uint32',
          length: props.grid.width * props.grid.height
        },
        ...Object.fromEntries(
          Object.entries(props.output.status ?? {})
            .filter(([, value]) => value !== undefined)
            .map(([name, value]) => [name, {value: value!, format: 'uint32' as const, length: 1}])
        )
      },
      createContributor: getView =>
        new GPUTerrainWatersheds({
          ...props,
          id,
          flowDirections: getView<'uint32'>('flowDirections'),
          pourPoints: props.pourPoints !== undefined ? getView<'uint32'>('pourPoints') : undefined,
          output: {
            labels: getView<'uint32'>('labels'),
            status: {
              converged:
                props.output.status?.converged !== undefined
                  ? getView<'uint32'>('converged')
                  : undefined,
              iterationLimitReached:
                props.output.status?.iterationLimitReached !== undefined
                  ? getView<'uint32'>('iterationLimitReached')
                  : undefined,
              iterationCount:
                props.output.status?.iterationCount !== undefined
                  ? getView<'uint32'>('iterationCount')
                  : undefined
            }
          }
        })
    });
  }
}
