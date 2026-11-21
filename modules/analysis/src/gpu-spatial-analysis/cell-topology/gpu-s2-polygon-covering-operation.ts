// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateS2CoveringOptions} from './s2-covering-option-validation';
import {snapshotS2PolygonOperationProps} from './s2-covering-construction-snapshot';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUS2PolygonCovering, type GPUS2PolygonCoveringProps} from './gpu-s2-polygon-covering';

/** The same properties and packed outputs for either covering operation. */
export type S2PolygonCoveringOperationProps = Omit<
  GPUS2PolygonCoveringProps,
  'positions' | 'ringOffsets' | 'output'
> & {
  positions: string;
  ringOffsets: string;
  /** Explicit compile-time input view lengths. */
  positionCount: number;
  ringOffsetCount: number;
  output: {cells: string; counts: string; status: string; visitedCount?: string};
};

/** Operation form; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUS2PolygonCoveringOperation extends GPUContributorOperation {
  constructor(props: S2PolygonCoveringOperationProps) {
    props = snapshotS2PolygonOperationProps(props);
    validateS2CoveringOptions(props.id ?? 's2-polygon-covering', props, [
      'serial',
      'level-synchronous'
    ]);
    const id = props.id ?? 's2-polygon-covering';
    validateCapacity(id, 'positionCount', props.positionCount, 0);
    validateCapacity(id, 'ringOffsetCount', props.ringOffsetCount, 2);
    super({
      id: props.id ?? 's2-polygon-covering',
      inputs: {positions: props.positions, ringOffsets: props.ringOffsets},
      outputs: {
        cells: {value: props.output.cells, format: 'uint32x2', length: props.capacity},
        counts: {value: props.output.counts, format: 'uint32x2', length: 1},
        status: {value: props.output.status, format: 'uint32', length: 1},
        ...(props.output.visitedCount === undefined
          ? {}
          : {
              visitedCount: {value: props.output.visitedCount, format: 'uint32' as const, length: 1}
            })
      },
      createContributor: getView => {
        const positions = getView<'float32x3'>('positions');
        const ringOffsets = getView<'uint32'>('ringOffsets');
        validateRequirement(
          id,
          positions.length === props.positionCount,
          `positionCount must equal declared ${props.positionCount}; received positions.length ${positions.length}`
        );
        validateRequirement(
          id,
          ringOffsets.length === props.ringOffsetCount,
          `ringOffsetCount must equal declared ${props.ringOffsetCount}; received ringOffsets.length ${ringOffsets.length}`
        );
        return new GPUS2PolygonCovering({
          ...props,
          positions,
          ringOffsets,
          output: {
            cells: getView<'uint32x2'>('cells'),
            counts: getView<'uint32x2'>('counts'),
            status: getView<'uint32'>('status'),
            visitedCount:
              props.output.visitedCount === undefined
                ? undefined
                : getView<'uint32'>('visitedCount')
          }
        });
      }
    });
  }
}
