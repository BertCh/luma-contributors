// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {resolveCellPrecision} from './cell-precision';
import {resolveCellLongitudePolicy} from './cell-longitude-policy';
import {validateCellIndexResolution} from './cell-index-families';
import {validateCapacity} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  GPU_LONGITUDE_LATITUDE_SPATIAL_CONTEXT,
  GPU_LONGITUDE_LATITUDE_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {validatePointOperationRows} from './cell-operation-rows';
import {snapshotCellConstruction} from './cell-construction-snapshot';
import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUPointToCell, type GPUPointToCellProps} from './gpu-point-to-cell';

/** Graph-value properties shared by the GPU operation and its CPU twin. */
export type PointToCellOperationProps = Omit<
  GPUPointToCellProps,
  'positions' | 'mask' | 'output'
> & {
  /** Compile-time number of positions and output rows. */
  rowCount: number;
  /** Longitude/latitude degrees, float32x2. */
  positions: string;
  /** Optional uint32 row mask. */
  mask?: string;
  output: {cells: string; validity?: string};
};

/** Vector operation form; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUPointToCellOperation extends GPUContributorOperation {
  constructor(props: PointToCellOperationProps) {
    props = snapshotCellConstruction(props, [
      'id',
      'family',
      'resolution',
      'precision',
      'longitudePolicy',
      'spatialContext',
      'positions',
      'mask',
      'rowCount',
      ['output', ['cells', 'validity']]
    ]);
    const id = props.id ?? 'point-to-cell';
    resolveCellPrecision(id, props.precision, props.family === 'h3');
    resolveCellLongitudePolicy(id, props.longitudePolicy);
    validateCellIndexResolution(id, props.family, props.resolution);
    validateCapacity(id, 'rowCount', props.rowCount, 0);
    resolveGPUSpatialContext(
      id,
      props.spatialContext ?? GPU_LONGITUDE_LATITUDE_SPATIAL_CONTEXT,
      GPU_LONGITUDE_LATITUDE_SPATIAL_CONTEXT_SUPPORT
    );
    super({
      id,
      emptyOutputPorts: ['cells', 'validity'],
      inputs: {positions: props.positions, ...(props.mask === undefined ? {} : {mask: props.mask})},
      outputs: {
        cells: {value: props.output.cells, format: 'uint32x2', length: props.rowCount},
        ...(props.output.validity === undefined
          ? {}
          : {
              validity: {
                value: props.output.validity,
                format: 'uint32' as const,
                length: props.rowCount
              }
            })
      },
      createContributor: getView => {
        const positions = getView<'float32x2'>('positions', 'float32x2');
        const mask = props.mask === undefined ? undefined : getView<'uint32'>('mask', 'uint32');
        validatePointOperationRows(id, props.rowCount, positions, mask);
        if (props.rowCount === 0) return {getCommandNodes: () => []};
        return new GPUPointToCell({
          ...props,
          positions,
          mask,
          output: {
            cells: getView<'uint32x2'>('cells'),
            validity:
              props.output.validity === undefined ? undefined : getView<'uint32'>('validity')
          }
        });
      }
    });
  }
}
