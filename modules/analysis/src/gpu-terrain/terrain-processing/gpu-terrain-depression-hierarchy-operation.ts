// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  GPUTerrainDepressionHierarchy,
  type GPUTerrainDepressionHierarchyProps
} from './gpu-terrain-depression-hierarchy';

/** Graph-value names, with an explicit shared-stage count formerly inferred from stages. */
export type TerrainDepressionHierarchyOperationProps = Omit<
  GPUTerrainDepressionHierarchyProps,
  'elevation' | 'validity' | 'outlets' | 'stages' | 'output'
> & {
  elevation: string;
  validity?: string;
  outlets: string;
  stages: string;
  stageCount: number;
  output: {
    nodes: string;
    levels: string;
    cellNodes: string;
    curves: string;
    status: string;
    ports?: {status: {requiredCount?: string; overflow?: string}};
  };
};

/** Vector hierarchy operation; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUTerrainDepressionHierarchyOperation extends GPUContributorOperation {
  constructor(props: TerrainDepressionHierarchyOperationProps) {
    super({
      id: props.id ?? 'terrain-depression-hierarchy',
      inputs: {
        elevation: props.elevation,
        outlets: props.outlets,
        stages: props.stages,
        ...(props.validity !== undefined ? {validity: props.validity} : {})
      },
      outputs: {
        nodes: {value: props.output.nodes, format: 'uint32', length: props.nodeCapacity * 4},
        levels: {value: props.output.levels, format: 'float32', length: props.nodeCapacity * 2},
        cellNodes: {
          value: props.output.cellNodes,
          format: 'uint32',
          length: props.grid.width * props.grid.height
        },
        curves: {
          value: props.output.curves,
          format: 'float32',
          length: props.nodeCapacity * props.stageCount * 2
        },
        status: {value: props.output.status, format: 'uint32', length: 3},
        ...(props.output.ports?.status.requiredCount !== undefined
          ? {
              requiredCount: {
                value: props.output.ports.status.requiredCount,
                format: 'uint32' as const,
                length: 1
              }
            }
          : {}),
        ...(props.output.ports?.status.overflow !== undefined
          ? {
              overflow: {
                value: props.output.ports.status.overflow,
                format: 'uint32' as const,
                length: 1
              }
            }
          : {})
      },
      createContributor: getView =>
        new GPUTerrainDepressionHierarchy({
          ...props,
          elevation: getView<'float32'>('elevation'),
          outlets: getView<'uint32'>('outlets'),
          stages: getView<'float32'>('stages'),
          validity: props.validity !== undefined ? getView<'uint32'>('validity') : undefined,
          output: {
            nodes: getView<'uint32'>('nodes'),
            levels: getView<'float32'>('levels'),
            cellNodes: getView<'uint32'>('cellNodes'),
            curves: getView<'float32'>('curves'),
            status: getView<'uint32'>('status'),
            ports: {
              status: {
                requiredCount:
                  props.output.ports?.status.requiredCount !== undefined
                    ? getView<'uint32'>('requiredCount')
                    : undefined,
                overflow:
                  props.output.ports?.status.overflow !== undefined
                    ? getView<'uint32'>('overflow')
                    : undefined
              }
            }
          }
        })
    });
  }
}
