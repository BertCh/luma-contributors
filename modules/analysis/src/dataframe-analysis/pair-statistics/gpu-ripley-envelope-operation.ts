// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUContributorOperation,
  type GPUContributorOperationOutput
} from '../../analysis-core/operations/gpu-contributor-operation';
import type {GPUSpatialContext} from '../../analysis-core/contracts/spatial-context';
import {
  snapshotConstructionFields,
  snapshotConstructionSpatialContext
} from '../../analysis-core/kernels/construction-snapshot';
import {GPURipleyEnvelope, type GPURipleyEnvelopeStatistic} from './gpu-ripley-envelope';
import {GPU_RIPLEY_ENVELOPE_STATISTICS_LENGTH} from './ripley-parameters';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Shared vector props. Prepared polygon study windows are omitted; this form uses rectangles. */
export type RipleyEnvelopeOperationProps = {
  id?: string;
  spatialContext?: GPUSpatialContext;
  positions: string;
  rowCount: number;
  parameters: string;
  gridSize: readonly [number, number];
  radiusCount: number;
  simulationCount: number;
  maximumAttempts?: number;
  rank?: number;
  statistic?: GPURipleyEnvelopeStatistic;
  mask?: string;
  observedK?: string;
  output: {
    lower?: string;
    upper?: string;
    globalLower?: string;
    globalUpper?: string;
    theoretical?: string;
    statistics?: string;
    unplacedCount?: string;
  };
};

/** @internal Capture declared fields and selected output names for both operation backends. */
export function snapshotRipleyEnvelopeOperation(
  props: RipleyEnvelopeOperationProps
): RipleyEnvelopeOperationProps {
  const snapshot = snapshotConstructionFields(props, [
    'id',
    'spatialContext',
    'positions',
    'rowCount',
    'parameters',
    'gridSize',
    'radiusCount',
    'simulationCount',
    'maximumAttempts',
    'rank',
    'statistic',
    'mask',
    'observedK',
    'output'
  ]);
  return Object.freeze({
    ...snapshot,
    spatialContext:
      snapshot.spatialContext === undefined
        ? undefined
        : snapshotConstructionSpatialContext(snapshot.spatialContext),
    gridSize: Object.freeze([...snapshot.gridSize]) as readonly [number, number],
    output: snapshotConstructionFields<RipleyEnvelopeOperationProps['output']>(snapshot.output, [
      'lower',
      'upper',
      'globalLower',
      'globalUpper',
      'theoretical',
      'statistics',
      'unplacedCount'
    ])
  });
}

/**
 * Rectangle-only GPURipleyEnvelope operation. Polygon preparation owns GPU views outside graph
 * values, so that optional window port is omitted. Compile with GPU_CONTRIBUTOR_OPERATION_LOWERING.
 */
export class GPURipleyEnvelopeOperation extends GPUContributorOperation {
  constructor(callerProps: RipleyEnvelopeOperationProps) {
    const props = snapshotRipleyEnvelopeOperation(callerProps);
    const id = props.id ?? 'ripley-envelope';
    const rowCount = props.rowCount;
    validateCapacity(id, 'rowCount', rowCount);
    const outputs: Record<string, GPUContributorOperationOutput> = {};
    for (const [port, value] of Object.entries(props.output))
      if (value !== undefined)
        outputs[port] = {
          value,
          format: port === 'unplacedCount' ? 'uint32' : 'float32',
          length:
            port === 'unplacedCount'
              ? 1
              : port === 'statistics'
                ? GPU_RIPLEY_ENVELOPE_STATISTICS_LENGTH
                : props.radiusCount
        };
    super({
      id,
      inputs: {
        positions: props.positions,
        parameters: props.parameters,
        ...(props.mask === undefined ? {} : {mask: props.mask}),
        ...(props.observedK === undefined ? {} : {observedK: props.observedK})
      },
      outputs,
      createContributor: getView => {
        const positions = getView<'float32x2'>('positions');
        validateRequirement(
          id,
          positions.length === rowCount,
          'positions.length must match rowCount'
        );
        return new GPURipleyEnvelope({
          ...props,
          positions,
          parameters: getView<'float32'>('parameters'),
          mask: props.mask === undefined ? undefined : getView<'uint32'>('mask'),
          observedK: props.observedK === undefined ? undefined : getView<'float32'>('observedK'),
          output: {
            lower: props.output.lower === undefined ? undefined : getView<'float32'>('lower'),
            upper: props.output.upper === undefined ? undefined : getView<'float32'>('upper'),
            globalLower:
              props.output.globalLower === undefined
                ? undefined
                : getView<'float32'>('globalLower'),
            globalUpper:
              props.output.globalUpper === undefined
                ? undefined
                : getView<'float32'>('globalUpper'),
            theoretical:
              props.output.theoretical === undefined
                ? undefined
                : getView<'float32'>('theoretical'),
            statistics:
              props.output.statistics === undefined ? undefined : getView<'float32'>('statistics'),
            unplacedCount:
              props.output.unplacedCount === undefined
                ? undefined
                : getView<'uint32'>('unplacedCount')
          }
        });
      }
    });
  }
}
