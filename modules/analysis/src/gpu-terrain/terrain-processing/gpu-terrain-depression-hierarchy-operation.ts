// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
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
  /** Positive integer number of shared query stages. */
  stageCount: number;
  output: {
    nodes: string;
    levels: string;
    cellNodes: string;
    curves: string;
    /** Required graph-value names for the one-row uint32 status outputs. */
    status: {count: string; requiredCount: string; overflow: string};
  };
};

/** Vector hierarchy operation; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUTerrainDepressionHierarchyOperation extends GPUContributorOperation {
  constructor(props: TerrainDepressionHierarchyOperationProps) {
    const {output} = props;
    const id = props.id ?? 'terrain-depression-hierarchy';
    validateRequirement(
      id,
      !Object.hasOwn(output ?? {}, 'ports'),
      'output.ports was removed; supply graph value names in output.status.count, output.status.requiredCount and output.status.overflow'
    );
    validateRequirement(
      id,
      typeof output?.status !== 'string',
      'output.status must be a map; supply graph value names in output.status.count, output.status.requiredCount and output.status.overflow'
    );
    for (const port of ['count', 'requiredCount', 'overflow'] as const) {
      const name = output?.status?.[port];
      validateRequirement(
        id,
        typeof name === 'string' && name.length > 0,
        `output.status.${port} must be a non-empty graph value name`
      );
    }
    validateCapacity(id, 'stageCount', props.stageCount);
    const contributorProps = {
      id,
      grid: {...props.grid, affine: [...props.grid.affine] as typeof props.grid.affine},
      nodeCapacity: props.nodeCapacity,
      cellSizeMode: props.cellSizeMode
    };
    const hasValidity = props.validity !== undefined;
    super({
      id,
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
        count: {value: output.status.count, format: 'uint32', length: 1},
        requiredCount: {value: output.status.requiredCount, format: 'uint32', length: 1},
        overflow: {value: output.status.overflow, format: 'uint32', length: 1}
      },
      createContributor: getView =>
        new GPUTerrainDepressionHierarchy({
          ...contributorProps,
          elevation: getView<'float32'>('elevation'),
          outlets: getView<'uint32'>('outlets'),
          stages: getView<'float32'>('stages'),
          validity: hasValidity ? getView<'uint32'>('validity') : undefined,
          output: {
            nodes: getView<'uint32'>('nodes'),
            levels: getView<'float32'>('levels'),
            cellNodes: getView<'uint32'>('cellNodes'),
            curves: getView<'float32'>('curves'),
            status: {
              count: getView<'uint32'>('count'),
              requiredCount: getView<'uint32'>('requiredCount'),
              overflow: getView<'uint32'>('overflow')
            }
          }
        })
    });
  }
}
