// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

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
      createContributor: getView =>
        new GPUS2PolygonCovering({
          ...props,
          positions: getView<'float32x3'>('positions'),
          ringOffsets: getView<'uint32'>('ringOffsets'),
          output: {
            cells: getView<'uint32x2'>('cells'),
            counts: getView<'uint32x2'>('counts'),
            status: getView<'uint32'>('status'),
            visitedCount:
              props.output.visitedCount === undefined
                ? undefined
                : getView<'uint32'>('visitedCount')
          }
        })
    });
  }
}
