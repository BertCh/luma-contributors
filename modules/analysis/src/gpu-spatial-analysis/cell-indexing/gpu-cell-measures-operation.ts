// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import {resolveCellPrecision} from './cell-precision';
import {snapshotCellConstruction} from './cell-construction-snapshot';
import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  GPUCellMeasures,
  type GPUCellMeasuresProps,
  type GPUCellMeasuresOutput
} from './gpu-cell-measures';

/** Graph values replacing the contributor's packed views. */
export type CellMeasuresOperationProps = Pick<
  GPUCellMeasuresProps,
  'id' | 'family' | 'wordOrder' | 'maximumEdgeCount' | 'units' | 'precision'
> & {
  cells: string;
  rowCount: number;
  output: {[Port in keyof GPUCellMeasuresOutput]?: string} & {
    [Port in keyof GPUCellMeasuresOutput]-?: Record<Port, string>;
  }[keyof GPUCellMeasuresOutput];
};

/** GPU cell measures as a graph operation; use GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUCellMeasuresOperation extends GPUContributorOperation {
  constructor(props: CellMeasuresOperationProps) {
    props = snapshotCellConstruction(props, [
      'id',
      'family',
      'cells',
      'wordOrder',
      'maximumEdgeCount',
      'units',
      'precision',
      'rowCount',
      ['output', ['areas', 'perimeters', 'edgeLengths', 'edgeCounts', 'pentagons']]
    ]);
    const id = props.id ?? 'cell-measures';
    resolveCellPrecision(id, props.precision);
    validateRequirement(
      id,
      (['areas', 'perimeters', 'edgeLengths', 'edgeCounts', 'pentagons'] as const).some(
        port => props.output[port] !== undefined
      ),
      'output must contain at least one terminal measure: areas, perimeters, edgeLengths, edgeCounts or pentagons'
    );
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
                        (port === 'edgeLengths'
                          ? (props.maximumEdgeCount ?? (props.family === 'h3' ? 6 : 4))
                          : 1)
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
          } as GPUCellMeasuresOutput
        })
    });
  }
}
