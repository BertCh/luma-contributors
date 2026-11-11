// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUIsolines, type GPUIsolinesProps} from './gpu-isolines';

/** Logical values and explicit capacities shared by the isolines operations. */
export type IsolinesOperationProps = Pick<GPUIsolinesProps, 'id' | 'grid' | 'noDataValue'> & {
  values: string;
  mask?: string;
  levels: string;
  parameters: string;
  segmentCapacity: number;
  output: {
    segments: string;
    segmentLevels: string;
    segmentEdges?: string;
    status: {count: string; requiredCount?: string; overflow: string; polylineOverflow?: string};
    polylines?: {
      vertexCapacity: number;
      vertices: string;
      polylineOffsets: string;
      polylineLevels: string;
      polylineClosed: string;
      polylineCount: string;
      vertexCount: string;
    };
  };
};

/** Isolines with optional stitching; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUIsolinesOperation extends GPUContributorOperation {
  constructor(props: IsolinesOperationProps) {
    const {segmentCapacity, output} = props;
    const polylines = output.polylines;
    super({
      id: props.id ?? 'isolines',
      inputs: {
        values: props.values,
        levels: props.levels,
        parameters: props.parameters,
        ...(props.mask !== undefined ? {mask: props.mask} : {})
      },
      outputs: {
        segments: {value: output.segments, format: 'float32x4', length: segmentCapacity},
        segmentLevels: {value: output.segmentLevels, format: 'uint32', length: segmentCapacity},
        ...(output.segmentEdges !== undefined
          ? {
              segmentEdges: {
                value: output.segmentEdges,
                format: 'uint32x2' as const,
                length: segmentCapacity
              }
            }
          : {}),
        ...Object.fromEntries(
          Object.entries(output.status).map(([port, value]) => [
            port,
            {value, format: 'uint32' as const, length: 1}
          ])
        ),
        ...(polylines
          ? {
              vertices: {
                value: polylines.vertices,
                format: 'float32x2' as const,
                length: polylines.vertexCapacity
              },
              polylineOffsets: {
                value: polylines.polylineOffsets,
                format: 'uint32' as const,
                length: segmentCapacity + 1
              },
              polylineLevels: {
                value: polylines.polylineLevels,
                format: 'uint32' as const,
                length: segmentCapacity
              },
              polylineClosed: {
                value: polylines.polylineClosed,
                format: 'uint32' as const,
                length: segmentCapacity
              },
              polylineCount: {value: polylines.polylineCount, format: 'uint32' as const, length: 1},
              vertexCount: {value: polylines.vertexCount, format: 'uint32' as const, length: 1}
            }
          : {})
      },
      createContributor: getView =>
        new GPUIsolines({
          ...props,
          values: getView<'float32'>('values'),
          levels: getView<'float32'>('levels'),
          parameters: getView<'float32'>('parameters'),
          mask: props.mask !== undefined ? getView<'uint32'>('mask') : undefined,
          output: {
            segments: getView<'float32x4'>('segments'),
            segmentLevels: getView<'uint32'>('segmentLevels'),
            segmentEdges:
              output.segmentEdges !== undefined ? getView<'uint32x2'>('segmentEdges') : undefined,
            status: {
              count: getView<'uint32'>('count'),
              overflow: getView<'uint32'>('overflow'),
              requiredCount:
                output.status.requiredCount !== undefined
                  ? getView<'uint32'>('requiredCount')
                  : undefined,
              polylineOverflow:
                output.status.polylineOverflow !== undefined
                  ? getView<'uint32'>('polylineOverflow')
                  : undefined
            },
            polylines: polylines
              ? {
                  vertices: getView<'float32x2'>('vertices'),
                  polylineOffsets: getView<'uint32'>('polylineOffsets'),
                  polylineLevels: getView<'uint32'>('polylineLevels'),
                  polylineClosed: getView<'uint32'>('polylineClosed'),
                  polylineCount: getView<'uint32'>('polylineCount'),
                  vertexCount: getView<'uint32'>('vertexCount')
                }
              : undefined
          }
        })
    });
  }
}
