// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  GPUTerrainCatchmentSummary,
  type GPUTerrainCatchmentSummaryProps
} from './gpu-terrain-catchment-summary';

/** Packed graph names; the grid, basinCount and histogramBins specify all output lengths. */
export type TerrainCatchmentSummaryOperationProps = Omit<
  GPUTerrainCatchmentSummaryProps,
  'elevation' | 'validity' | 'basinIds' | 'streams' | 'flowDirections' | 'output'
> & {
  elevation: string;
  validity?: string;
  basinIds: string;
  streams: string;
  flowDirections: string;
  output: {statistics: string; hypsometry: string; profiles: string; profileStatus: string};
};

/** GPU summary as a vector operation; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUTerrainCatchmentSummaryOperation extends GPUContributorOperation {
  constructor(props: TerrainCatchmentSummaryOperationProps) {
    const count = props.grid.width * props.grid.height;
    super({
      id: props.id ?? 'terrain-catchment-summary',
      inputs: {
        elevation: props.elevation,
        basinIds: props.basinIds,
        streams: props.streams,
        flowDirections: props.flowDirections,
        ...(props.validity !== undefined ? {validity: props.validity} : {})
      },
      outputs: {
        statistics: {
          value: props.output.statistics,
          format: 'float32',
          length: props.basinCount * 6
        },
        hypsometry: {
          value: props.output.hypsometry,
          format: 'float32',
          length: props.basinCount * props.histogramBins
        },
        profiles: {value: props.output.profiles, format: 'float32', length: count * 3},
        profileStatus: {value: props.output.profileStatus, format: 'uint32', length: count}
      },
      createContributor: getView =>
        new GPUTerrainCatchmentSummary({
          ...props,
          elevation: getView<'float32'>('elevation'),
          basinIds: getView<'uint32'>('basinIds'),
          streams: getView<'uint32'>('streams'),
          flowDirections: getView<'uint32'>('flowDirections'),
          validity: props.validity !== undefined ? getView<'uint32'>('validity') : undefined,
          output: {
            statistics: getView<'float32'>('statistics'),
            hypsometry: getView<'float32'>('hypsometry'),
            profiles: getView<'float32'>('profiles'),
            profileStatus: getView<'uint32'>('profileStatus')
          }
        })
    });
  }
}
