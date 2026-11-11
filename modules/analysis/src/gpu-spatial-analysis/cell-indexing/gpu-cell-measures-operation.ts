// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUCellMeasures, type GPUCellMeasuresProps} from './gpu-cell-measures';

/** Graph values replacing the contributor's packed views. */
export type CellMeasuresOperationProps = Pick<
  GPUCellMeasuresProps,
  'id' | 'family' | 'wordOrder' | 'maximumEdgeCount' | 'units'
> & {
  cells: string;
  rowCount: number;
  output: {
    areas?: string;
    perimeters?: string;
    edgeLengths?: string;
    edgeCounts?: string;
    pentagons?: string;
  };
};

/** GPU cell measures as a graph operation; use GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUCellMeasuresOperation extends GPUContributorOperation {
  constructor(props: CellMeasuresOperationProps) {
    const id = props.id ?? 'cell-measures';
    super({
      id,
      inputs: {cells: props.cells},
      outputs: Object.fromEntries(
        (['areas', 'perimeters', 'edgeLengths', 'edgeCounts', 'pentagons'] as const).flatMap(
          port =>
            props.output[port] === undefined
              ? []
              : [
                  [
                    port,
                    {
                      value: props.output[port]!,
                      format: port === 'edgeCounts' || port === 'pentagons' ? 'uint32' : 'float32',
                      length:
                        props.rowCount *
                        (port === 'edgeLengths' ? (props.maximumEdgeCount ?? 6) : 1)
                    }
                  ]
                ]
        )
      ),
      createContributor: getView =>
        new GPUCellMeasures({
          ...props,
          id,
          cells: getView<'uint32x2'>('cells'),
          output: {
            areas: props.output.areas === undefined ? undefined : getView<'float32'>('areas'),
            perimeters:
              props.output.perimeters === undefined ? undefined : getView<'float32'>('perimeters'),
            edgeLengths:
              props.output.edgeLengths === undefined
                ? undefined
                : getView<'float32'>('edgeLengths'),
            edgeCounts:
              props.output.edgeCounts === undefined ? undefined : getView<'uint32'>('edgeCounts'),
            pentagons:
              props.output.pentagons === undefined ? undefined : getView<'uint32'>('pentagons')
          }
        })
    });
  }
}
