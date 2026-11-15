// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUSpatialContext} from '../../analysis-core/contracts/spatial-context';
import {
  GPUContributorOperation,
  type GPUContributorOperationOutput
} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUMinimumBounds} from './gpu-minimum-bounds';

/** Graph names in place of views; lengths and requested columns are compile-time. */
export type MinimumBoundsOperationProps = {
  id?: string;
  spatialContext?: GPUSpatialContext;
  positions: string;
  rowCount: number;
  maximumHullVertices?: number;
  output: {
    rectangleCorners?: string;
    rectangleSizes?: string;
    circles?: string;
    longestLines?: string;
    diameters?: string;
    bounds?: string;
    hullSizes?: string;
    status: {overflow: string; groupOverflow?: string};
  };
} & (
  | {
      labels: string;
      groupCount: number;
      ringOffsets?: never;
      polygonOffsets?: never;
      featureOffsets?: never;
      featureCount?: never;
    }
  | {
      labels?: undefined;
      groupCount?: never;
      ringOffsets: string;
      polygonOffsets: string;
      featureOffsets: string;
      featureCount: number;
    }
);

/** GPU minimum bounds operation; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUMinimumBoundsOperation extends GPUContributorOperation {
  constructor(props: MinimumBoundsOperationProps) {
    const length = props.labels !== undefined ? props.groupCount : props.featureCount;
    const outputs: Record<string, GPUContributorOperationOutput> = {
      overflow: {value: props.output.status.overflow, format: 'uint32', length: 1}
    };
    if (props.output.status.groupOverflow !== undefined) {
      outputs['groupOverflow'] = {
        value: props.output.status.groupOverflow,
        format: 'uint32',
        length: 1
      };
    }
    for (const port of [
      'rectangleCorners',
      'rectangleSizes',
      'circles',
      'longestLines',
      'diameters',
      'bounds',
      'hullSizes'
    ] as const) {
      const value = props.output[port];
      if (value !== undefined)
        outputs[port] = {
          value,
          format:
            port === 'hullSizes'
              ? 'uint32'
              : port === 'rectangleCorners'
                ? 'float32x2'
                : port === 'circles'
                  ? 'float32x3'
                  : port === 'diameters'
                    ? 'float32'
                    : 'float32x4',
          length: port === 'rectangleCorners' ? length * 4 : length
        };
    }
    super({
      id: props.id ?? 'minimum-bounds',
      inputs: {
        positions: props.positions,
        ...(props.labels !== undefined
          ? {labels: props.labels}
          : {
              ringOffsets: props.ringOffsets,
              polygonOffsets: props.polygonOffsets,
              featureOffsets: props.featureOffsets
            })
      },
      outputs,
      createContributor: getView =>
        new GPUMinimumBounds({
          id: props.id,
          spatialContext: props.spatialContext,
          maximumHullVertices: props.maximumHullVertices,
          ...(props.labels !== undefined
            ? {
                positions: getView<'float32x2'>('positions'),
                labels: getView<'uint32'>('labels'),
                groupCount: props.groupCount
              }
            : {
                positions: getView<'float32x2'>('positions'),
                ringOffsets: getView<'uint32'>('ringOffsets'),
                polygonOffsets: getView<'uint32'>('polygonOffsets'),
                featureOffsets: getView<'uint32'>('featureOffsets')
              }),
          output: {
            rectangleCorners:
              props.output.rectangleCorners !== undefined
                ? getView<'float32x2'>('rectangleCorners')
                : undefined,
            rectangleSizes:
              props.output.rectangleSizes !== undefined
                ? getView<'float32x4'>('rectangleSizes')
                : undefined,
            circles:
              props.output.circles !== undefined ? getView<'float32x3'>('circles') : undefined,
            longestLines:
              props.output.longestLines !== undefined
                ? getView<'float32x4'>('longestLines')
                : undefined,
            diameters:
              props.output.diameters !== undefined ? getView<'float32'>('diameters') : undefined,
            bounds: props.output.bounds !== undefined ? getView<'float32x4'>('bounds') : undefined,
            hullSizes:
              props.output.hullSizes !== undefined ? getView<'uint32'>('hullSizes') : undefined,
            status: {
              overflow: getView<'uint32'>('overflow'),
              groupOverflow:
                props.output.status.groupOverflow !== undefined
                  ? getView<'uint32'>('groupOverflow')
                  : undefined
            }
          }
        })
    });
  }
}
