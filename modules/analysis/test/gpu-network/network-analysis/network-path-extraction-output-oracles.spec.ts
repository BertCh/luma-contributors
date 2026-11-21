// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUNetworkPathExtractionOperation} from '../../../src/gpu-network/network-analysis/gpu-network-path-extraction-operation';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {poisonOperationOutputs} from '../adjacency-matrix/network-output-oracle-test-utils';
import {PATH_INPUTS, createPathOracleProps} from './network-path-extraction-output-oracles.fixture';

it('path operation publishes hand-enumerated offsets costs and clamped counts with optional ports alone', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const capacity of [2, 4])
    for (const selected of [undefined, 'lines', 'pathCosts', 'edges', 'edgeLines'] as const) {
      const operation = poisonOperationOutputs(
        new GPUNetworkPathExtractionOperation(createPathOracleProps(capacity, selected))
      );
      const values = await runOperationGraph(device, {
        inputs: PATH_INPUTS,
        operations: [operation],
        outputs: Object.values(operation.outputs).map(output => output.value)
      });
      expect(Array.from(values['ids'])).toEqual(capacity === 2 ? [41, 17] : [41, 17, 90, 41]);
      expect(Array.from(values['count'])).toEqual([capacity]);
      expect(Array.from(values['required'])).toEqual([4]);
      expect(Array.from(values['overflow'])).toEqual([capacity === 2 ? 1 : 0]);
      // Offsets describe full paths even when their stored prefix is truncated.
      if (selected === 'lines') expect(Array.from(values['lines'])).toEqual([0, 3, 4]);
      if (selected === 'pathCosts') expect(Array.from(values['pathCosts'])).toEqual([5, 0]);
      if (selected === 'edges' || selected === 'edgeLines') {
        expect(Array.from(values['edgeIdsOut'])).toEqual([501]);
        expect(Array.from(values['edgeCount'])).toEqual([1]);
        expect(Array.from(values['edgeRequired'])).toEqual([2]);
        expect(Array.from(values['edgeOverflow'])).toEqual([1]);
        if (selected === 'edgeLines') expect(Array.from(values['edgeLines'])).toEqual([0, 2, 2]);
      }
    }
});
