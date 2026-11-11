// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUTerrainBreach, type GPUTerrainBreachProps} from './gpu-terrain-breach';

/** Graph-value names with the pit count made explicit; path rows remain pit-major scalar uint32. */
export type TerrainBreachOperationProps = Omit<
  GPUTerrainBreachProps,
  'elevation' | 'validity' | 'pitCells' | 'burnDepth' | 'output'
> & {
  elevation: string;
  validity?: string;
  pitCells: string;
  pitCount: number;
  burnDepth?: string;
  output: {
    elevation: string;
    cutDepth: string;
    modificationMask: string;
    pathCells: string;
    pathLengths: string;
    pitStatus: string;
    unresolvedCount: string;
  };
};

/** Vector breach operation; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUTerrainBreachOperation extends GPUContributorOperation {
  constructor(props: TerrainBreachOperationProps) {
    const count = props.grid.width * props.grid.height;
    super({
      id: props.id ?? 'terrain-breach',
      inputs: {
        elevation: props.elevation,
        pitCells: props.pitCells,
        ...(props.validity !== undefined ? {validity: props.validity} : {}),
        ...(props.burnDepth !== undefined ? {burnDepth: props.burnDepth} : {})
      },
      outputs: {
        elevationOutput: {value: props.output.elevation, format: 'float32', length: count},
        cutDepth: {value: props.output.cutDepth, format: 'float32', length: count},
        modificationMask: {value: props.output.modificationMask, format: 'uint32', length: count},
        pathCells: {
          value: props.output.pathCells,
          format: 'uint32',
          length: props.pitCount * (props.maximumSteps + 1)
        },
        pathLengths: {value: props.output.pathLengths, format: 'uint32', length: props.pitCount},
        pitStatus: {value: props.output.pitStatus, format: 'uint32', length: props.pitCount},
        unresolvedCount: {value: props.output.unresolvedCount, format: 'uint32', length: 1}
      },
      createContributor: getView =>
        new GPUTerrainBreach({
          ...props,
          elevation: getView<'float32'>('elevation'),
          pitCells: getView<'uint32'>('pitCells'),
          validity: props.validity !== undefined ? getView<'uint32'>('validity') : undefined,
          burnDepth: props.burnDepth !== undefined ? getView<'float32'>('burnDepth') : undefined,
          output: {
            elevation: getView<'float32'>('elevationOutput'),
            cutDepth: getView<'float32'>('cutDepth'),
            modificationMask: getView<'uint32'>('modificationMask'),
            pathCells: getView<'uint32'>('pathCells'),
            pathLengths: getView<'uint32'>('pathLengths'),
            pitStatus: getView<'uint32'>('pitStatus'),
            unresolvedCount: getView<'uint32'>('unresolvedCount')
          }
        })
    });
  }
}
