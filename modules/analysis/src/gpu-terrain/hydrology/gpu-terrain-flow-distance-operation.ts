// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  GPUTerrainFlowDistance,
  type GPUTerrainFlowDistanceProps
} from './gpu-terrain-flow-distance';

/** Graph-value props shared by the distance operation and its CPU twin. */
export type TerrainFlowDistanceOperationProps = Omit<
  GPUTerrainFlowDistanceProps,
  'flowDirections' | 'streams' | 'transitionCosts' | 'output'
> & {
  flowDirections: string;
  streams?: string;
  transitionCosts?: string;
  output: {distance: string; validity: string; terminalCells: string};
};

/** Vector operation form. Input contents are per-frame; grid and other props are compile-time. */
export class GPUTerrainFlowDistanceOperation extends GPUContributorOperation {
  constructor(props: TerrainFlowDistanceOperationProps) {
    const id = props.id ?? 'terrain-flow-distance';
    const length = props.grid.width * props.grid.height;
    super({
      id,
      inputs: {
        flowDirections: props.flowDirections,
        ...(props.streams !== undefined ? {streams: props.streams} : {}),
        ...(props.transitionCosts !== undefined ? {transitionCosts: props.transitionCosts} : {})
      },
      outputs: {
        distance: {value: props.output.distance, format: 'float32', length},
        validity: {value: props.output.validity, format: 'uint32', length},
        terminalCells: {value: props.output.terminalCells, format: 'uint32', length}
      },
      createContributor: getView =>
        new GPUTerrainFlowDistance({
          ...props,
          id,
          flowDirections: getView<'uint32'>('flowDirections'),
          streams: props.streams !== undefined ? getView<'uint32'>('streams') : undefined,
          transitionCosts:
            props.transitionCosts !== undefined ? getView<'float32'>('transitionCosts') : undefined,
          output: {
            distance: getView<'float32'>('distance'),
            validity: getView<'uint32'>('validity'),
            terminalCells: getView<'uint32'>('terminalCells')
          }
        })
    });
  }
}
