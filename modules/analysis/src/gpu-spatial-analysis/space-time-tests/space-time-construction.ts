// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

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
    statistics: GraphDataView;
    summary: GraphDataView<'float32'>;
    permutationCount?: GraphDataView<'uint32'>;
  };
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
  const weights = snapshotConstructionFields(captured[neighborhood]!, [
    'offsets',
    'neighbors',
    'weights',
    'distances',
    'status',
    'information',
    'spatialContext'
  ]);
  return Object.freeze({
    ...captured,
    [neighborhood]: Object.freeze({
      ...weights,
      status:
        weights.status &&
        snapshotConstructionFields(weights.status, [
          'overflow',
          'candidateOverflow',
          'requiredCount'
        ]),
      information:
        weights.information &&
        snapshotConstructionFields(weights.information, [
          'rowCount',
          'transform',
          'symmetric',
          'hasSelfWeights',
          'provenance'
        ]),
      spatialContext:
        weights.spatialContext && snapshotConstructionSpatialContext(weights.spatialContext)
    }),
    output: snapshotConstructionFields(captured.output, [
      'statistics',
      'summary',
      'permutationCount'
    ]),
    distanceTransform:
      captured.distanceTransform &&
      snapshotConstructionFields(captured.distanceTransform, ['constant', 'power']),
    timeTransform:
      captured.timeTransform &&
      snapshotConstructionFields(captured.timeTransform, ['constant', 'power'])
  });
}
