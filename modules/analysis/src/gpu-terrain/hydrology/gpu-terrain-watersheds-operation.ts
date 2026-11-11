// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUTerrainWatersheds, type GPUTerrainWatershedsProps} from './gpu-terrain-watersheds';
import type {TerrainDrainageOperationStatus} from './cpu-terrain-drainage-utils';

/** Watershed graph names in place of views. Grid dimensions explicitly size the output. */
export type TerrainWatershedsOperationProps = Omit<
  GPUTerrainWatershedsProps,
  'flowDirections' | 'pourPoints' | 'output'
> & {
  flowDirections: string;
  pourPoints?: string;
  output: {labels: string; status?: TerrainDrainageOperationStatus};
};

/** Vector operation form. Grid and options are compile-time; direction and point rows are per-frame. */
export class GPUTerrainWatershedsOperation extends GPUContributorOperation {
  constructor(props: TerrainWatershedsOperationProps) {
    const id = props.id ?? 'terrain-watersheds';
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
