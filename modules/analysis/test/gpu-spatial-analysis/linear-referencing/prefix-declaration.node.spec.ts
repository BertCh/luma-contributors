// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {afterAll, expect, it} from 'vitest';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createOutputOracleFixture} from '../cell-topology/output-oracle-test-utils';
import {addPrefixLocateConsumers} from './prefix-locate-fixture';

const device = createNullWebGPUDevice();
afterAll(() => device.destroy());
it('B4 declares both distance and fraction consumers on one graph', () => {
  const fixture = createOutputOracleFixture(device);
  try {
    const positions = fixture.input('positions', 'float32x2', Float32Array.of(0, 0, 1, 0));
    const offsets = fixture.input('offsets', 'uint32', Uint32Array.of(0, 2));
    expect(addPrefixLocateConsumers(fixture, positions, offsets, 2, 1)).toHaveLength(2);
  } finally {
    fixture.destroy();
  }
});
