// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {resolveHealpixIndexScheme} from './healpix-scheme';
import {HEALPIX_MAXIMUM_ORDER} from './healpix-index-wgsl';
import {validateCapacity} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  GPU_LONGITUDE_LATITUDE_SPATIAL_CONTEXT,
  GPU_LONGITUDE_LATITUDE_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {validatePointOperationRows} from '../cell-indexing/cell-operation-rows';
import {snapshotCellConstruction} from '../cell-indexing/cell-construction-snapshot';
import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUHealpixPointToCell, type GPUHealpixPointToCellProps} from './gpu-healpix-point-to-cell';

/** Same properties as the contributor, with graph names and an explicit position row count. */
export type HealpixPointToCellOperationProps = Pick<
  GPUHealpixPointToCellProps,
  'id' | 'order' | 'scheme' | 'spatialContext'
> & {
  positions: string;
  mask?: string;
  rowCount: number;
  output: {cells: string; validity?: string};
};

/** HEALPix projection as a graph operation; use GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUHealpixPointToCellOperation extends GPUContributorOperation {
  constructor(props: HealpixPointToCellOperationProps) {
    props = snapshotCellConstruction(props, [
      'id',
      'order',
      'scheme',
      'spatialContext',
      'positions',
      'mask',
      'rowCount',
      ['output', ['cells', 'validity']]
    ]);
    const id = props.id ?? 'healpix-point-to-cell';
    resolveHealpixIndexScheme(id, props.scheme);
    validateCapacity(id, 'order', props.order, 0, HEALPIX_MAXIMUM_ORDER);
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
          : {validity: {value: props.output.validity, format: 'uint32', length: props.rowCount}})
      },
      createContributor: getView => {
        const positions = getView<'float32x2'>('positions', 'float32x2');
        const mask = props.mask === undefined ? undefined : getView<'uint32'>('mask', 'uint32');
        validatePointOperationRows(id, props.rowCount, positions, mask);
        if (props.rowCount === 0) return {getCommandNodes: () => []};
        return new GPUHealpixPointToCell({
          id,
          order: props.order,
          scheme: props.scheme,
          spatialContext: props.spatialContext,
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
