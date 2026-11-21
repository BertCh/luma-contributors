// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUTemporalContext} from '../../analysis-core/contracts/temporal-context';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  snapshotConstructionFields,
  snapshotConstructionSpatialContext
} from '../../analysis-core/kernels/construction-snapshot';
import type {GPUSpatialWeights} from '../spatial-weights/spatial-weights';

type SpaceTimeConstructionProps = {
  pairs?: GPUSpatialWeights;
  neighbors?: GPUSpatialWeights;
  output: {
    statistics?: GraphDataView;
    counts?: GraphDataView;
    summary: GraphDataView<'float32'>;
    permutationCount?: GraphDataView<'uint32'>;
    usablePermutationCount?: GraphDataView<'uint32'>;
  };
  temporalContext?: GPUTemporalContext;
  distanceTransform?: {constant: number; power: number};
  timeTransform?: {constant: number; power: number};
};

/** Capture declared destinations and neighborhood status before validation or deferred lowering. @internal */
export function snapshotSpaceTimeTestProps<Props extends SpaceTimeConstructionProps>(
  props: Props,
  fields: readonly (keyof Props)[],
  neighborhood: 'pairs' | 'neighbors'
): Props {
  const captured = snapshotConstructionFields(props, fields);
  const id = (captured as {id?: string}).id ?? 'space-time-test';
  validateRequirement(
    id,
    Boolean(captured[neighborhood] && typeof captured[neighborhood] === 'object'),
    `${neighborhood} must be a weights port map; received ${String(captured[neighborhood])}`
  );
  validateRequirement(
    id,
    Boolean(captured.output && typeof captured.output === 'object'),
    `output must be a port map; received ${String(captured.output)}`
  );
  const weights = snapshotConstructionFields(captured[neighborhood]!, [
    'offsets',
    'neighbors',
    'weights',
    'distances',
    'status',
    'information',
    'spatialContext'
  ]);
  const status = weights.status;
  const information =
    weights.information &&
    snapshotConstructionFields(weights.information, [
      'rowCount',
      'transform',
      'symmetric',
      'hasSelfWeights',
      'provenance'
    ]);
  const provenance = information?.provenance;
  return Object.freeze({
    ...captured,
    [neighborhood]: Object.freeze({
      ...weights,
      status:
        status &&
        snapshotConstructionFields(status, [
          'count',
          'requiredCount',
          'overflow',
          'candidateOverflow',
          'uncertainCount',
          'invalidCount',
          'iterationCount',
          'converged',
          'iterationLimitReached',
          'approximation'
        ]),
      information:
        information &&
        Object.freeze({
          ...information,
          provenance:
            provenance &&
            snapshotConstructionFields(provenance, ['source', 'criterion', 'k', 'radius', 'order'])
        }),
      spatialContext:
        weights.spatialContext && snapshotConstructionSpatialContext(weights.spatialContext)
    }),
    output: snapshotConstructionFields(captured.output, [
      'counts',
      'statistics',
      'summary',
      'permutationCount',
      'usablePermutationCount'
    ]),
    temporalContext:
      captured.temporalContext && snapshotSpaceTimeTemporalContext(captured.temporalContext),
    distanceTransform:
      captured.distanceTransform &&
      snapshotConstructionFields(captured.distanceTransform, ['constant', 'power']),
    timeTransform:
      captured.timeTransform &&
      snapshotConstructionFields(captured.timeTransform, ['constant', 'power'])
  });
}

/** Capture temporal metadata, including calendar steps, without copying any live data. @internal */
export function snapshotSpaceTimeTemporalContext(context: GPUTemporalContext): GPUTemporalContext {
  const captured = snapshotConstructionFields(context, [
    'unit',
    'epoch',
    'encoding',
    'timeZone',
    'axis'
  ]);
  const axis = captured.axis;
  // Preserve invalid declarations for the named temporal-context validator to reject.
  if (!axis) return captured;
  const capturedKind = snapshotConstructionFields(axis, ['kind']);
  const kind = capturedKind.kind;
  if (kind !== 'regular' && kind !== 'irregular') return captured;
  if (kind === 'irregular') return Object.freeze({...captured, axis: Object.freeze({kind})});
  const capturedStep = snapshotConstructionFields(
    axis as Extract<GPUTemporalContext['axis'], {kind: 'regular'}>,
    ['step']
  );
  const step = capturedStep.step;
  return Object.freeze({
    ...captured,
    axis: Object.freeze({
      kind,
      step:
        step && typeof step === 'object'
          ? snapshotConstructionFields(step, ['calendarUnit', 'interval'])
          : step
    })
  });
}
