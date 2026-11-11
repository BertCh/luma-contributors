// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUContributorOperation,
  type GPUContributorOperationOutput
} from '../../analysis-core/operations/gpu-contributor-operation';
import type {GPUSpatialContext} from '../../analysis-core/contracts/spatial-context';
import {GPURipleyEnvelope, type GPURipleyEnvelopeStatistic} from './gpu-ripley-envelope';
import {GPU_RIPLEY_ENVELOPE_STATISTICS_LENGTH} from './ripley-parameters';

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

/**
 * Rectangle-only GPURipleyEnvelope operation. Polygon preparation owns GPU views outside graph
 * values, so that optional window port is omitted. Compile with GPU_CONTRIBUTOR_OPERATION_LOWERING.
 */
export class GPURipleyEnvelopeOperation extends GPUContributorOperation {
  constructor(props: RipleyEnvelopeOperationProps) {
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
      id: props.id ?? 'ripley-envelope',
      inputs: {
        positions: props.positions,
        parameters: props.parameters,
        ...(props.mask === undefined ? {} : {mask: props.mask}),
        ...(props.observedK === undefined ? {} : {observedK: props.observedK})
      },
      outputs,
      createContributor: getView =>
        new GPURipleyEnvelope({
          ...props,
          positions: getView<'float32x2'>('positions'),
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
        })
    });
  }
}
