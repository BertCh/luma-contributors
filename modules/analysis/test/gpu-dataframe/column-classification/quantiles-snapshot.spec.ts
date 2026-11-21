// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {CPUColumnQuantilesOperation} from '../../../src/dataframe-analysis/column-classification/cpu-column-quantiles-operation';
import {GPUColumnQuantilesOperation} from '../../../src/dataframe-analysis/column-classification/gpu-column-quantiles-operation';
import {getGPUColumnQuantilesParameterValues} from '../../../src/dataframe-analysis/column-classification/column-quantiles-parameters';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {createQuantilesSnapshotFixture} from './quantiles-snapshot.test-utils';

it.each([
  false,
  true
])('B1 executes captured bounds under the original name, inherited=%s', async inherited => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const Operation of [CPUColumnQuantilesOperation, GPUColumnQuantilesOperation]) {
    const fixture = createQuantilesSnapshotFixture(inherited);
    const operation = new Operation(fixture.props);
    fixture.mutate();
    const result = await runOperationGraph(device, {
      operations: [operation],
      inputs: {
        values: {format: 'float32', values: Float32Array.of(1, 3)},
        parameters: {
          format: 'float32',
          values: getGPUColumnQuantilesParameterValues({quantiles: [0.5], filterRange: [0, 1]})
        }
      },
      outputs: ['initial-bounds', 'quantiles']
    });
    // The full filter range includes both observations; their median is exactly 2.
    expect(Array.from(result['initial-bounds'])).toEqual([1, 3]);
    expect(result).not.toHaveProperty('filterBounds');
    expect(Array.from(result['quantiles'])).toEqual([2]);
    expect(fixture.reads).toEqual({
      values: 1,
      parameters: 1,
      rowCount: 1,
      quantiles: 1,
      validCount: 1,
      filterBounds: 1
    });
  }
});
