// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUContributorOperation,
  type GPUContributorOperationOutput
} from '../../analysis-core/operations/gpu-contributor-operation';
import type {GPUSpatialContext, GPUTemporalContext} from '../../analysis-core/contracts/index';
import {
  GPUSpaceTimeDiggleTest,
  GPU_SPACE_TIME_DIGGLE_SUMMARY_LENGTH
} from './gpu-space-time-diggle-test';
import type {GPUSpaceTimeKEdgeCorrection} from './space-time-k-function-nodes';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  snapshotConstructionFields,
  snapshotConstructionSpatialContext
} from '../../analysis-core/kernels/construction-snapshot';

/** Shared graph-value props. Parameters keep the space-time K and Diggle seed packer layouts. */
export type SpaceTimeDiggleTestOperationProps = {
  id?: string;
  spatialContext?: GPUSpatialContext;
  temporalContext?: GPUTemporalContext;
  positions: string;
  times: string;
  rowCount: number;
  mask?: string;
  parameters: string;
  randomParameters: string;
  gridSize: readonly [number, number];
  distanceClassCount: number;
  timeClassCount: number;
  edgeCorrection?: GPUSpaceTimeKEdgeCorrection;
  permutationCount: number;
  output: {
    d: string;
    statistics: string;
    summary: string;
    permutationMean?: string;
    permutationStandardDeviation?: string;
    k?: string;
    spatialK?: string;
    temporalK?: string;
    status?: string;
  };
};

/** Captures declared descriptors, including nested frame and calendar-axis metadata. */
export function snapshotSpaceTimeDiggleOperationProps(
  input: SpaceTimeDiggleTestOperationProps
): SpaceTimeDiggleTestOperationProps {
  const props = snapshotConstructionFields(input, [
    'id',
    'spatialContext',
    'temporalContext',
    'positions',
    'times',
    'rowCount',
    'mask',
    'parameters',
    'randomParameters',
    'gridSize',
    'distanceClassCount',
    'timeClassCount',
    'edgeCorrection',
    'permutationCount',
    'output'
  ]);
  const temporal =
    props.temporalContext &&
    snapshotConstructionFields(props.temporalContext, [
      'unit',
      'epoch',
      'encoding',
      'timeZone',
      'axis'
    ]);
  let axis = temporal?.axis;
  if (axis?.kind === 'regular') {
    const captured = snapshotConstructionFields(axis, ['kind', 'step']);
    axis = {
      ...captured,
      step:
        typeof captured.step === 'object' && captured.step !== null
          ? snapshotConstructionFields(captured.step, ['calendarUnit', 'interval'])
          : captured.step
    };
  } else if (axis) {
    axis = snapshotConstructionFields(axis, ['kind']);
  }
  return {
    ...props,
    gridSize: [...props.gridSize] as [number, number],
    ...(props.spatialContext
      ? {spatialContext: snapshotConstructionSpatialContext(props.spatialContext)}
      : {}),
    ...(temporal ? {temporalContext: {...temporal, axis: axis!}} : {}),
    output: snapshotConstructionFields(props.output, [
      'd',
      'statistics',
      'summary',
      'permutationMean',
      'permutationStandardDeviation',
      'k',
      'spatialK',
      'temporalK',
      'status'
    ])
  };
}

/** Vector operation form; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUSpaceTimeDiggleTestOperation extends GPUContributorOperation {
  constructor(inputProps: SpaceTimeDiggleTestOperationProps) {
    const props = snapshotSpaceTimeDiggleOperationProps(inputProps);
    const id = props.id ?? 'space-time-diggle-test';
    validateCapacity(id, 'rowCount', props.rowCount);
    const outputs: Record<string, GPUContributorOperationOutput> = {};
    for (const [port, value] of Object.entries(props.output))
      if (value !== undefined)
        outputs[port] = {
          value,
          format: port === 'status' ? 'uint32' : 'float32',
          length:
            port === 'status'
              ? 1
              : port === 'statistics'
                ? props.permutationCount + 1
                : port === 'summary'
                  ? GPU_SPACE_TIME_DIGGLE_SUMMARY_LENGTH
                  : port === 'spatialK'
                    ? props.distanceClassCount
                    : port === 'temporalK'
                      ? props.timeClassCount
                      : props.distanceClassCount * props.timeClassCount
        };
    super({
      id: props.id ?? 'space-time-diggle-test',
      inputs: {
        positions: props.positions,
        times: props.times,
        parameters: props.parameters,
        randomParameters: props.randomParameters,
        ...(props.mask === undefined ? {} : {mask: props.mask})
      },
      outputs,
      createContributor: getView => {
        const positions = getView<'float32x2'>('positions');
        validateRequirement(
          id,
          positions.length === props.rowCount,
          'positions length must match rowCount'
        );
        return new GPUSpaceTimeDiggleTest({
          ...props,
          positions,
          times: getView<'float32'>('times'),
          parameters: getView<'float32'>('parameters'),
          randomParameters: getView<'uint32'>('randomParameters'),
          mask: props.mask === undefined ? undefined : getView<'uint32'>('mask'),
          output: {
            d: getView<'float32'>('d'),
            statistics: getView<'float32'>('statistics'),
            summary: getView<'float32'>('summary'),
            permutationMean:
              props.output.permutationMean === undefined
                ? undefined
                : getView<'float32'>('permutationMean'),
            permutationStandardDeviation:
              props.output.permutationStandardDeviation === undefined
                ? undefined
                : getView<'float32'>('permutationStandardDeviation'),
            k: props.output.k === undefined ? undefined : getView<'float32'>('k'),
            spatialK:
              props.output.spatialK === undefined ? undefined : getView<'float32'>('spatialK'),
            temporalK:
              props.output.temporalK === undefined ? undefined : getView<'float32'>('temporalK'),
            status: props.output.status === undefined ? undefined : getView<'uint32'>('status')
          }
        });
      }
    });
  }
}
