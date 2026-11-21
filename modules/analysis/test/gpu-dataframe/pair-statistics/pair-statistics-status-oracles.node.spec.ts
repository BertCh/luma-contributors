// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeRipleyEnvelopeOnCPU,
  CPURipleyEnvelopeOperation
} from '../../../src/dataframe-analysis/pair-statistics/cpu-ripley-envelope-operation';
import {getGPURipleyParameterValues} from '../../../src/dataframe-analysis/pair-statistics/ripley-parameters';

it('CPU rectangle envelope has zero unplaced draws by direct uniform mapping', async () => {
  // No rejection: each finite uniform coordinate lies within the unit square.
  const result = computeRipleyEnvelopeOnCPU(
    {
      positions: new Float32Array([0, 0, 1, 1]),
      parameters: getGPURipleyParameterValues({
        bounds: [0, 0, 1, 1],
        maximumDistance: 2,
        edgeCorrection: 'none'
      })
    },
    {radiusCount: 1, simulationCount: 2}
  );
  expect(Array.from(result.unplacedCount)).toEqual([0]);
  const positions = new Float32Array([0, 0, 1, 1]);
  const parameters = getGPURipleyParameterValues({
    bounds: [0, 0, 1, 1],
    maximumDistance: 2,
    edgeCorrection: 'none'
  });
  const operation = new CPURipleyEnvelopeOperation({
    positions: 'positions',
    parameters: 'parameters',
    rowCount: 2,
    gridSize: [2, 2],
    radiusCount: 1,
    simulationCount: 2,
    output: {lower: 'lower', unplacedCount: 'unplaced'}
  });
  const rows = (await operation.execute({
    inputs: {
      positions: {
        format: 'float32x2',
        length: 2,
        chunks: [{values: positions, length: 2}],
        pack: () => positions
      },
      parameters: {
        format: 'float32',
        length: parameters.length,
        chunks: [{values: parameters, length: parameters.length}],
        pack: () => parameters
      }
    }
  })) as Record<string, Uint32Array>;
  expect(Array.from(rows['unplaced'])).toEqual([0]);
});
