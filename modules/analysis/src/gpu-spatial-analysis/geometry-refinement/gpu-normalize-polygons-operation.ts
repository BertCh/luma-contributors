// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUNormalizePolygons} from './gpu-normalize-polygons';

/** Graph values replace polygon views; reuse input featureOffsets with the output hierarchy. */
export type NormalizePolygonsOperationProps = {
  id?: string;
  positions: string;
  ringOffsets: string;
  polygonOffsets: string;
  featureOffsets: string;
  rowCount: number;
  ringCount: number;
  polygonCount: number;
  featureCount: number;
  output: {positions: string; ringOffsets: string; polygonOffsets: string};
};

/** GPU polygon normalization operation; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUNormalizePolygonsOperation extends GPUContributorOperation {
  constructor(props: NormalizePolygonsOperationProps) {
    super({
      id: props.id ?? 'normalize-polygons',
      inputs: {
        positions: props.positions,
        ringOffsets: props.ringOffsets,
        polygonOffsets: props.polygonOffsets,
        featureOffsets: props.featureOffsets
      },
      outputs: {
        positionsOutput: {
          value: props.output.positions,
          format: 'float32x2',
          length: props.rowCount
        },
        ringOffsetsOutput: {
          value: props.output.ringOffsets,
          format: 'uint32',
          length: props.ringCount + 1
        },
        polygonOffsetsOutput: {
          value: props.output.polygonOffsets,
          format: 'uint32',
          length: props.polygonCount + 1
        }
      },
      createContributor: getView =>
        new GPUNormalizePolygons({
          id: props.id,
          positions: getView<'float32x2'>('positions'),
          ringOffsets: getView<'uint32'>('ringOffsets'),
          polygonOffsets: getView<'uint32'>('polygonOffsets'),
          featureOffsets: getView<'uint32'>('featureOffsets'),
          output: {
            positions: getView<'float32x2'>('positionsOutput'),
            ringOffsets: getView<'uint32'>('ringOffsetsOutput'),
            polygonOffsets: getView<'uint32'>('polygonOffsetsOutput')
          }
        })
    });
  }
}
