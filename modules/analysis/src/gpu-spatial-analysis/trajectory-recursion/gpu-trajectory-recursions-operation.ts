// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  resolveGPUTemporalContext,
  type GPUTemporalAxis
} from '../../analysis-core/contracts/temporal-context';
import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  snapshotConstructionFields,
  snapshotConstructionSpatialContext
} from '../../analysis-core/kernels/construction-snapshot';
import {
  GPUTrajectoryRecursions,
  type GPUTrajectoryRecursionsProps
} from './gpu-trajectory-recursions';

/** Graph-value names replacing contributor views, with explicit lengths. */
export type TrajectoryRecursionsOperationProps = Pick<
  GPUTrajectoryRecursionsProps,
  'id' | 'spatialContext' | 'temporalContext' | 'maximumTrackSegments'
> & {
  positions: string;
  timestamps: string;
  timestampFormat?: 'float32' | 'uint32x2';
  trackOffsets: string;
  radii: string;
  rowCount: number;
  radiusCount: number;
  output: {
    revisits?: string;
    residenceTimes?: string;
    firstPassageTimes?: string;
    firstPassageCensored?: string;
    status?: {candidateOverflow?: string};
  };
};

/** @internal Capture the recursion operation contract before validation or adapter construction. */
export function snapshotTrajectoryRecursionsOperationProps(
  input: TrajectoryRecursionsOperationProps
): TrajectoryRecursionsOperationProps {
  const props = snapshotConstructionFields(input, [
    'id',
    'spatialContext',
    'temporalContext',
    'maximumTrackSegments',
    'positions',
    'timestamps',
    'timestampFormat',
    'trackOffsets',
    'radii',
    'rowCount',
    'radiusCount',
    'output'
  ]);
  const output = snapshotConstructionFields(props.output, [
    'revisits',
    'residenceTimes',
    'firstPassageTimes',
    'firstPassageCensored',
    'status'
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
  return Object.freeze({
    ...props,
    ...(props.spatialContext
      ? {spatialContext: snapshotConstructionSpatialContext(props.spatialContext)}
      : {}),
    ...(temporal
      ? {
          temporalContext: Object.freeze({
            ...temporal,
            axis: snapshotRecursionAxis(temporal.axis)
          })
        }
      : {}),
    output: Object.freeze({
      ...output,
      ...(output.status
        ? {status: snapshotConstructionFields(output.status, ['candidateOverflow'])}
        : {})
    })
  });
}

function snapshotRecursionAxis(axis: GPUTemporalAxis): GPUTemporalAxis {
  // Missing axes must reach the named temporal validator, not the field snapshot helper.
  if (!axis) return axis;
  const kind = axis.kind;
  if (kind !== 'regular') return Object.freeze({kind});
  const step = axis.step;
  return Object.freeze({
    kind,
    step:
      typeof step === 'object' && step !== null
        ? snapshotConstructionFields(step, ['calendarUnit', 'interval'])
        : step
  });
}

/** GPU recursion analysis as a graph operation; use GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUTrajectoryRecursionsOperation extends GPUContributorOperation {
  constructor(props: TrajectoryRecursionsOperationProps) {
    props = snapshotTrajectoryRecursionsOperationProps(props);
    const id = props.id ?? 'trajectory-recursions';
    const {spatialContext, temporalContext, maximumTrackSegments, timestampFormat, output} = props;
    validateRequirement(
      id,
      timestampFormat === undefined ||
        timestampFormat === 'float32' ||
        timestampFormat === 'uint32x2',
      'timestampFormat must be float32 or uint32x2'
    );
    if (temporalContext)
      resolveGPUTemporalContext(id, temporalContext, {
        encodings: [timestampFormat === 'uint32x2' ? 'int64-words' : 'float32-relative'],
        axisKinds: ['irregular']
      });
    validateCapacity(id, 'rowCount', props.rowCount, 0);
    validateCapacity(id, 'radiusCount', props.radiusCount);
    const length = props.rowCount * props.radiusCount;
    super({
      id,
      emptyOutputPorts: ['revisits', 'residenceTimes', 'firstPassageTimes', 'firstPassageCensored'],
      inputs: {
        positions: props.positions,
        timestamps: props.timestamps,
        trackOffsets: props.trackOffsets,
        radii: props.radii
      },
      outputs: {
        ...Object.fromEntries(
          (['revisits', 'firstPassageCensored'] as const).flatMap(port =>
            output[port] ? [[port, {value: output[port]!, format: 'uint32' as const, length}]] : []
          )
        ),
        ...Object.fromEntries(
          (['residenceTimes', 'firstPassageTimes'] as const).flatMap(port =>
            output[port] ? [[port, {value: output[port]!, format: 'float32' as const, length}]] : []
          )
        ),
        ...(output.status?.candidateOverflow
          ? {
              candidateOverflow: {
                value: output.status.candidateOverflow,
                format: 'uint32',
                length: 1
              }
            }
          : {})
      },
      createContributor: getView =>
        new GPUTrajectoryRecursions({
          id,
          spatialContext,
          temporalContext,
          maximumTrackSegments,
          positions: getView<'float32x2'>('positions'),
          timestamps:
            timestampFormat === 'uint32x2'
              ? getView<'uint32x2'>('timestamps')
              : getView<'float32'>('timestamps'),
          trackOffsets: getView<'uint32'>('trackOffsets'),
          radii: getView<'float32'>('radii'),
          output: {
            revisits: output.revisits ? getView<'uint32'>('revisits') : undefined,
            residenceTimes: output.residenceTimes
              ? getView<'float32'>('residenceTimes')
              : undefined,
            firstPassageTimes: output.firstPassageTimes
              ? getView<'float32'>('firstPassageTimes')
              : undefined,
            firstPassageCensored: output.firstPassageCensored
              ? getView<'uint32'>('firstPassageCensored')
              : undefined,
            status: output.status?.candidateOverflow
              ? {candidateOverflow: getView<'uint32'>('candidateOverflow')}
              : undefined
          }
        })
    });
  }
}
