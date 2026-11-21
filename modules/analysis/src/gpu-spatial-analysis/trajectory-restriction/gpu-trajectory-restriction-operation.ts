// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUContributorOperationOutput} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  GPUTrajectoryRestriction,
  type GPUTrajectoryRestrictionProps
} from './gpu-trajectory-restriction';

/** Graph value names instead of views; all lengths formerly inferred from views are explicit. */
export type TrajectoryRestrictionOperationProps = Pick<
  GPUTrajectoryRestrictionProps,
  | 'id'
  | 'spatialContext'
  | 'temporalContext'
  | 'pairwiseTracks'
  | 'intervalsPerPair'
  | 'polygonRingVertexCounts'
  | 'timeBoundary'
  | 'geometryBoundary'
  | 'intervalsPerSegment'
  | 'mergeIntervals'
> & {
  positions: string;
  timestamps: string;
  trackOffsets: string;
  timestampFormat?: 'float32' | 'uint32x2';
  timeOrigin?: string;
  /** Seven scalars followed by polygon coordinates, as documented by GPUTrajectoryRestriction. */
  parameters: string;
  rowCount: number;
  trackCount: number;
  pairs?: {first: string; second: string; count?: string};
  /** Required with pairs; static pair counts are inferred from pairwiseTracks. */
  pairCapacity?: number;
  /** Required with compactIntervals. */
  compactCapacity?: number;
  output: {
    intervals: string;
    positions: string;
    status: string;
    relationships: string;
    compactIntervals?: string;
    compactPositions?: string;
    compactSources?: string;
    compactStatus?: string;
    compactSummary?: string;
  };
};

/** @internal Validate the array and operation forms' trajectory input names. */
export function validateRestrictionInputNames(
  id: string,
  input: {timestamps: unknown; trackOffsets: unknown}
): void {
  validateRequirement(id, !('times' in input), 'times was removed; use timestamps');
  validateRequirement(id, !('offsets' in input), 'offsets was removed; use trackOffsets');
  validateRequirement(
    id,
    input.timestamps !== undefined && input.trackOffsets !== undefined,
    'timestamps and trackOffsets are required'
  );
}

/**
 * Vector operation form of GPUTrajectoryRestriction, including word time and optional compaction.
 * Compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. Numeric parameters remain a per-frame value.
 * The inherited empty-output adapter preserves zero-row tracks/pairs without padding graph values.
 */
export class GPUTrajectoryRestrictionOperation extends GPUContributorOperation {
  constructor(props: TrajectoryRestrictionOperationProps) {
    validateRestrictionInputNames(props.id ?? 'trajectory-restriction', props);
    const pairCount = props.pairs ? props.pairCapacity! : props.pairwiseTracks?.length;
    const intervalCount =
      pairCount === undefined
        ? props.rowCount * (props.intervalsPerSegment ?? 1)
        : pairCount * props.intervalsPerPair!;
    const outputs: Record<string, GPUContributorOperationOutput> = {
      intervals: {value: props.output.intervals, format: 'float32x2', length: intervalCount},
      positions: {value: props.output.positions, format: 'float32x4', length: intervalCount},
      status: {value: props.output.status, format: 'uint32', length: intervalCount},
      relationships: {
        value: props.output.relationships,
        format: 'uint32x4',
        length: pairCount ?? props.trackCount
      }
    };
    for (const [port, format] of [
      ['compactIntervals', 'float32x2'],
      ['compactPositions', 'float32x4'],
      ['compactSources', 'uint32x2'],
      ['compactStatus', 'uint32'],
      ['compactSummary', 'uint32']
    ] as const) {
      if (props.output[port] !== undefined)
        outputs[port] = {
          value: props.output[port]!,
          format,
          length: port === 'compactSummary' ? 3 : props.compactCapacity!
        };
    }
    super({
      id: props.id ?? 'trajectory-restriction',
      inputs: {
        sourcePositions: props.positions,
        times: props.timestamps,
        offsets: props.trackOffsets,
        parameters: props.parameters,
        ...(props.timeOrigin !== undefined ? {timeOrigin: props.timeOrigin} : {}),
        ...(props.pairs
          ? {
              pairFirst: props.pairs.first,
              pairSecond: props.pairs.second,
              ...(props.pairs.count !== undefined ? {pairCount: props.pairs.count} : {})
            }
          : {})
      },
      outputs,
      createContributor: getView =>
        new GPUTrajectoryRestriction({
          id: props.id,
          spatialContext: props.spatialContext,
          temporalContext: props.temporalContext,
          positions: getView<'float32x2'>('sourcePositions'),
          times:
            props.timestampFormat === 'uint32x2'
              ? getView<'uint32x2'>('times')
              : getView<'float32'>('times'),
          offsets: getView<'uint32'>('offsets'),
          parameters: getView<'float32'>('parameters'),
          timeOrigin: props.timeOrigin !== undefined ? getView<'uint32'>('timeOrigin') : undefined,
          pairwiseTracks: props.pairwiseTracks,
          pairs: props.pairs
            ? {
                first: getView<'uint32'>('pairFirst'),
                second: getView<'uint32'>('pairSecond'),
                count: props.pairs.count !== undefined ? getView<'uint32'>('pairCount') : undefined
              }
            : undefined,
          intervalsPerPair: props.intervalsPerPair,
          polygonRingVertexCounts: props.polygonRingVertexCounts,
          timeBoundary: props.timeBoundary,
          geometryBoundary: props.geometryBoundary,
          intervalsPerSegment: props.intervalsPerSegment,
          mergeIntervals: props.mergeIntervals,
          output: {
            intervals: getView<'float32x2'>('intervals'),
            positions: getView<'float32x4'>('positions'),
            status: getView<'uint32'>('status'),
            relationships: getView<'uint32x4'>('relationships'),
            compactIntervals:
              props.output.compactIntervals !== undefined
                ? getView<'float32x2'>('compactIntervals')
                : undefined,
            compactPositions:
              props.output.compactPositions !== undefined
                ? getView<'float32x4'>('compactPositions')
                : undefined,
            compactSources:
              props.output.compactSources !== undefined
                ? getView<'uint32x2'>('compactSources')
                : undefined,
            compactStatus:
              props.output.compactStatus !== undefined
                ? getView<'uint32'>('compactStatus')
                : undefined,
            compactSummary:
              props.output.compactSummary !== undefined
                ? getView<'uint32'>('compactSummary')
                : undefined
          }
        }),
      emptyOutputPorts: Object.keys(outputs).filter(port => port !== 'compactSummary')
    });
  }
}
