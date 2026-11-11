// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUNormalizeGeometry} from './gpu-normalize-geometry';

/** Packed postorder-tree graph values. Nodes stay in their original order. */
export type NormalizeGeometryOperationProps = {
  id?: string;
  positions: string;
  nodes: string;
  children: string;
  rowCount: number;
  nodeCount: number;
  childCount: number;
  maximumNestingDepth?: number;
  output: {positions: string; children: string; status: string};
};

/** GPU tree normalization operation; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUNormalizeGeometryOperation extends GPUContributorOperation {
  constructor(props: NormalizeGeometryOperationProps) {
    super({
      id: props.id ?? 'normalize-geometry',
      inputs: {positions: props.positions, nodes: props.nodes, children: props.children},
      outputs: {
        positionsOutput: {
          value: props.output.positions,
          format: 'float32x2',
          length: props.rowCount
        },
        childrenOutput: {
          value: props.output.children,
          format: 'uint32',
          length: props.childCount
        },
        status: {value: props.output.status, format: 'uint32', length: props.nodeCount}
      },
      createContributor: getView =>
        new GPUNormalizeGeometry({
          id: props.id,
          maximumNestingDepth: props.maximumNestingDepth,
          positions: getView<'float32x2'>('positions'),
          nodes: getView<'uint32x4'>('nodes'),
          children: getView<'uint32'>('children'),
          output: {
            positions: getView<'float32x2'>('positionsOutput'),
            children: getView<'uint32'>('childrenOutput'),
            status: getView<'uint32'>('status')
          }
        }),
      emptyOutputPorts: ['positionsOutput', 'childrenOutput', 'status']
    });
  }
}
