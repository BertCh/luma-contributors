// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';

/** Capture impacts topology and port maps once, preserving live GPU view contents. @internal */
export function snapshotSpatialLagImpactsProps<
  Props extends {
    traces: {powerTraces: unknown; powerSums: unknown; infinityNorm: unknown};
    output: {impacts: unknown};
  }
>(input: Props): Props {
  const props = snapshotConstructionFields(input, [
    'id',
    'coefficients',
    'coefficientStride',
    'rhoIndex',
    'firstPredictorIndex',
    'predictorCount',
    'rowCount',
    'maximumPower',
    'traces',
    'status',
    'rhoBound',
    'output'
  ] as readonly (keyof Props)[]);
  return Object.freeze({
    ...props,
    traces: snapshotConstructionFields(props.traces, ['powerTraces', 'powerSums', 'infinityNorm']),
    output: snapshotConstructionFields(props.output, ['impacts'])
  });
}
