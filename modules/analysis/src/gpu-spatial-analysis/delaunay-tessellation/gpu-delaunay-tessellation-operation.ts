// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import {snapshotDelaunayTessellationOperationProps} from './delaunay-construction-snapshot';
import {
  GPUContributorOperation,
  type GPUContributorOperationOutput
} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  GPUDelaunayTessellation,
  type GPUDelaunayTessellationProps
} from './gpu-delaunay-tessellation';

/**
 * Packed graph values for Delaunay construction; scalar output lengths are explicit.
 * Status diagnostics are exposed only when requested by the caller.
 */
export type DelaunayTessellationOperationProps = Pick<
  GPUDelaunayTessellationProps,
  'id' | 'spatialContext'
> & {
  positions: string;
  capacity: number;
  groups?: {ids: string; count: number};
  output: {
    triangles: string;
    status: {
      count: string;
      requiredCount: string;
      overflow: string;
      invalidCount?: string;
      duplicateCount?: string;
      converged?: string;
      iterationCount?: string;
      iterationLimitReached?: string;
      candidateOverflow?: string;
      uncertainCount?: string;
      approximation?: string;
    };
    groups?: {triangleCounts: string; triangleOffsets?: string; status: {converged: string}};
  };
};

/**
 * GPU Delaunay as a graph operation; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING.
 * Capacity zero reports demand through mandatory scalar status without triangle payload.
 */
export class GPUDelaunayTessellationOperation extends GPUContributorOperation {
  constructor(props: DelaunayTessellationOperationProps) {
    props = snapshotDelaunayTessellationOperationProps(props);
    for (const port of ['count', 'requiredCount', 'overflow'] as const) {
      const value = props.output.status[port];
      validateRequirement(
        props.id ?? 'delaunay-tessellation',
        typeof value === 'string' && value.length > 0,
        `output.status.${port} must name a graph value; received ${String(value)}`
      );
    }
    const outputs: Record<string, GPUContributorOperationOutput> = {
      triangles: {value: props.output.triangles, format: 'uint32x3', length: props.capacity}
    };
    for (const [port, value] of Object.entries(props.output.status)) {
      if (value !== undefined) outputs[port] = {value, format: 'uint32', length: 1};
    }
    if (props.output.groups && props.groups) {
      outputs['groupTriangleCounts'] = {
        value: props.output.groups.triangleCounts,
        format: 'uint32',
        length: props.groups.count
      };
      outputs['groupConverged'] = {
        value: props.output.groups.status.converged,
        format: 'uint32',
        length: props.groups.count
      };
      if (props.output.groups.triangleOffsets !== undefined)
        outputs['groupTriangleOffsets'] = {
          value: props.output.groups.triangleOffsets,
          format: 'uint32',
          length: props.groups.count
        };
    }
    super({
      id: props.id ?? 'delaunay-tessellation',
      inputs: {positions: props.positions, ...(props.groups ? {groupIds: props.groups.ids} : {})},
      outputs,
      emptyOutputPorts: ['triangles'],
      createContributor: getView =>
        new GPUDelaunayTessellation({
          id: props.id,
          spatialContext: props.spatialContext,
          positions: getView<'float32x2'>('positions'),
          groups: props.groups
            ? {ids: getView<'uint32'>('groupIds'), count: props.groups.count}
            : undefined,
          output: {
            triangles: getView<'uint32x3'>('triangles'),
            status: {
              count: getView<'uint32'>('count'),
              requiredCount: getView<'uint32'>('requiredCount'),
              overflow: getView<'uint32'>('overflow'),
              ...Object.fromEntries(
                Object.entries(props.output.status)
                  .filter(
                    ([port, value]) =>
                      !['count', 'requiredCount', 'overflow'].includes(port) && value !== undefined
                  )
                  .map(([port]) => [port, getView<'uint32'>(port)])
              )
            },
            groups: props.output.groups
              ? {
                  triangleCounts: getView<'uint32'>('groupTriangleCounts'),
                  triangleOffsets:
                    props.output.groups.triangleOffsets !== undefined
                      ? getView<'uint32'>('groupTriangleOffsets')
                      : undefined,
                  status: {converged: getView<'uint32'>('groupConverged')}
                }
              : undefined
          }
        })
    });
  }
}
