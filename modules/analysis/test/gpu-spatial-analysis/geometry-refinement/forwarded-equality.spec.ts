// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {createOracleFixture} from '../global-coordinates/output-oracle-fixture';
import type {EqualityView} from './canonical-equality-fixture';
import {createSubnormalEqualityFixture} from './forwarded-equality-fixture';

it.each([
  'tree',
  'polygon'
] as const)('forward metric N1 canonical %s compares subnormal stored coordinates before flushing', async kind => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createOracleFixture(device);
  const output = fixture.output('uint32', 1);
  const view: EqualityView = (format, values) =>
    format === 'uint32' && values.length === 1 && values[0] === 99
      ? (output.view as GraphDataView<typeof format>)
      : fixture.input(format, values);
  const equality = createSubnormalEqualityFixture(kind, view);
  fixture.graph.add(equality.contributor);
  const compiled = fixture.graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await output.read()).toEqual([1]);
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});
