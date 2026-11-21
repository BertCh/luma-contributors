// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it} from 'vitest';
import {GPUMapMatchingOperation} from '../../../src/gpu-network/map-matching/gpu-map-matching-operation';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {expectOracleRows, runPoisonedOperation} from '../alternative-routes/output-oracle-harness';
import {getMatchingGraphInputs} from './map-matching-operation-fixture';
import {createMatchingOutputOracle, getMatchingOutputExpected} from './map-matching-output-oracles';

for (const routeNodeBudget of [8, 16])
  it(`pins isolated matching ports and omitted outputs at budget ${routeNodeBudget}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    for (const port of [
      'matchedOffsets',
      'matchedCount',
      'breakCount',
      'routeOverflow',
      'absent'
    ] as const) {
      const {input, props} = createMatchingOutputOracle(routeNodeBudget);
      if (port === 'routeOverflow') props.output.status = {routeOverflow: 'routeOverflow'};
      else if (port !== 'absent') props.output[port] = port;
      const result = await runPoisonedOperation(
        device,
        new GPUMapMatchingOperation(props),
        getMatchingGraphInputs(input)
      );
      const expected = getMatchingOutputExpected(routeNodeBudget);
      expectOracleRows(result['matchedEdges'], expected.matchedEdges);
      // Binary fractions and offsets on horizontal length-two edges are exact in float32.
      if (port !== 'absent') expectOracleRows(result[port], expected[port]);
    }
  });
