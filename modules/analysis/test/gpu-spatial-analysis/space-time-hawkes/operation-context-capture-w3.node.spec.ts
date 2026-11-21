// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  GPUSpaceTimeHawkesOperation,
  type SpaceTimeHawkesOperationProps
} from '../../../src/gpu-spatial-analysis/space-time-hawkes/gpu-space-time-hawkes-operation';
import type {GPUSpaceTimeHawkes} from '../../../src/gpu-spatial-analysis/space-time-hawkes/gpu-space-time-hawkes';
import {constructNetworkOperationContributor} from '../../utils/network-operation-test-utils';

it('W2-14 captures nested Hawkes metadata before deferred lowering', () => {
  const props: SpaceTimeHawkesOperationProps = {
    id: 'captured-hawkes',
    rowCount: 2,
    positions: 'positions',
    times: 'times',
    parameters: 'parameters',
    temporalContext: {
      unit: 'second',
      epoch: 0n,
      encoding: 'float32-relative',
      axis: {kind: 'regular', step: {calendarUnit: 'day', interval: 1}}
    },
    output: {estimates: 'estimates'}
  };
  const operation = new GPUSpaceTimeHawkesOperation(props);
  props.temporalContext!.epoch = 99n;
  if (
    props.temporalContext!.axis.kind === 'regular' &&
    typeof props.temporalContext!.axis.step === 'object'
  )
    props.temporalContext!.axis.step.interval = 5;
  props.rowCount = 100;
  props.output.estimates = 'changed';
  const contributor = constructNetworkOperationContributor(operation, {
    positions: {format: 'float32x2', length: 2},
    times: {format: 'float32', length: 2},
    parameters: {format: 'float32', length: 9}
  }) as GPUSpaceTimeHawkes;
  expect(contributor.temporalContext).toMatchObject({epoch: 0n, axis: {step: {interval: 1}}});
  expect(operation.outputs['estimates'].value).toBe('estimates');
});
