// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUTerrainStreamOrder, type GPUTerrainStreamOrderProps} from './gpu-terrain-stream-order';
import type {TerrainDrainageOperationStatus} from './cpu-terrain-drainage-utils';

/** Graph names shared by both stream-order operations; grid dimensions size the output. */
export type TerrainStreamOrderOperationProps = Omit<
  GPUTerrainStreamOrderProps,
  'flowDirections' | 'streams' | 'output'
> & {
  flowDirections: string;
  streams: string;
  output: {streamOrder: string; status?: TerrainDrainageOperationStatus};
};

/** Vector operation form. Input rows are per-frame; all other props are compile-time. */
export class GPUTerrainStreamOrderOperation extends GPUContributorOperation {
  constructor(props: TerrainStreamOrderOperationProps) {
    const id = props.id ?? 'terrain-stream-order';
    super({
      id,
      inputs: {flowDirections: props.flowDirections, streams: props.streams},
      outputs: {
        streamOrder: {
          value: props.output.streamOrder,
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
        new GPUTerrainStreamOrder({
          ...props,
          id,
          flowDirections: getView<'uint32'>('flowDirections'),
          streams: getView<'uint32'>('streams'),
          output: {
            streamOrder: getView<'uint32'>('streamOrder'),
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
