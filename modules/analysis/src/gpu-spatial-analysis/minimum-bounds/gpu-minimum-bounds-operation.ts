// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  snapshotConstructionFields,
  snapshotConstructionSpatialContext
} from '../../analysis-core/kernels/construction-snapshot';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
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
  labelsComplete?: string;
  /** CPU host interpretation; GPU views remain resident storage. */
  storage?: 'compact' | 'resident-capacity';
  positions: string;
  rowCount: number;
  maximumHullVertices?: number;
  output: {
    rectangleCorners?: string;
    rectangleSizes?: string;
    circles?: string;
    longestLines?: string;
    diameters?: string;
    sourceBounds?: string;
    labelsComplete?: string;
    precisionLimited?: string;
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

/**
 * GPU minimum bounds operation; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING.
 * At lowering, positions must contain exactly `rowCount` rows. On the polygon route,
 * featureOffsets must contain exactly `featureCount + 1` rows, including the terminal entry.
 */
export class GPUMinimumBoundsOperation extends GPUContributorOperation {
  /** Label provenance alongside the incumbent geometry; omission denotes categorical labels. */
  readonly partitionProvenance: Readonly<{labelsComplete?: string}>;
  constructor(props: MinimumBoundsOperationProps) {
    props = snapshotConstructionFields(props, [
      'id',
      'spatialContext',
      'storage',
      'positions',
      'rowCount',
      'maximumHullVertices',
      'output',
      'labels',
      'labelsComplete',
      'groupCount',
      'ringOffsets',
      'polygonOffsets',
      'featureOffsets',
      'featureCount'
    ]);
    validateRequirement(
      props.id ?? 'minimum-bounds',
      !('bounds' in props.output),
      'output.bounds was removed; use output.sourceBounds'
    );
    const output = snapshotConstructionFields(props.output, [
      'rectangleCorners',
      'rectangleSizes',
      'circles',
      'longestLines',
      'diameters',

      'sourceBounds',
      'labelsComplete',
      'precisionLimited',
      'hullSizes',
      'status'
    ]);
    props = Object.freeze({
      ...props,
      ...(props.spatialContext
        ? {spatialContext: snapshotConstructionSpatialContext(props.spatialContext)}
        : {}),
      output: Object.freeze({
        ...output,
        status: snapshotConstructionFields(output.status, ['overflow', 'groupOverflow'])
      })
    });
    const id = props.id ?? 'minimum-bounds';
    validateRequirement(
      id,
      props.storage === undefined || ['compact', 'resident-capacity'].includes(props.storage),
      `storage must be compact or resident-capacity; received ${props.storage}`
    );
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

      'sourceBounds',
      'labelsComplete',
      'precisionLimited',
      'hullSizes'
    ] as const) {
      const value = props.output[port];
      if (value !== undefined)
        outputs[port] = {
          value,
          format:
            port === 'hullSizes' || port === 'precisionLimited' || port === 'labelsComplete'
              ? 'uint32'
              : port === 'rectangleCorners'
                ? 'float32x2'
                : port === 'circles'
                  ? 'float32x3'
                  : port === 'diameters'
                    ? 'float32'
                    : 'float32x4',
          length: port === 'labelsComplete' ? 1 : port === 'rectangleCorners' ? length * 4 : length
        };
    }
    super({
      emptyOutputPorts: Object.keys(outputs).filter(
        port => !['overflow', 'groupOverflow'].includes(port)
      ),
      id,
      inputs: {
        ...(props.labelsComplete === undefined ? {} : {inputLabelsComplete: props.labelsComplete}),
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
      createContributor: getView => {
        const positions = getView<'float32x2'>('positions');
        validateRequirement(
          id,
          positions.length === props.rowCount,
          `positions length must equal rowCount (${props.rowCount})`
        );
        const input =
          props.labels !== undefined
            ? {positions, labels: getView<'uint32'>('labels'), groupCount: props.groupCount}
            : (() => {
                const featureOffsets = getView<'uint32'>('featureOffsets');
                validateRequirement(
                  id,
                  featureOffsets.length === props.featureCount + 1,
                  `featureOffsets length must equal featureCount + 1 (${props.featureCount + 1})`
                );
                return {
                  positions,
                  ringOffsets: getView<'uint32'>('ringOffsets'),
                  polygonOffsets: getView<'uint32'>('polygonOffsets'),
                  featureOffsets
                };
              })();
        return new GPUMinimumBounds({
          id: props.id,
          spatialContext: props.spatialContext,
          maximumHullVertices: props.maximumHullVertices,
          labelsComplete:
            props.labelsComplete === undefined
              ? undefined
              : getView<'uint32'>('inputLabelsComplete'),
          ...input,
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
            labelsComplete:
              props.output.labelsComplete === undefined
                ? undefined
                : getView<'uint32'>('labelsComplete'),
            sourceBounds:
              props.output.sourceBounds !== undefined
                ? getView<'float32x4'>('sourceBounds')
                : undefined,
            precisionLimited:
              props.output.precisionLimited !== undefined
                ? getView<'uint32'>('precisionLimited')
                : undefined,
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
        });
      }
    });
    this.partitionProvenance = Object.freeze({
      labelsComplete: props.output.labelsComplete ?? props.labelsComplete
    });
  }
}
