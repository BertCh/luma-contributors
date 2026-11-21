// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeTerrainBranchingFlowOnCPU,
  CPUTerrainBranchingFlowOperation
} from '../../../src/gpu-terrain/terrain-processing/cpu-terrain-branching-flow-operation';
import {
  BRANCHING_STATUS_GRID,
  BRANCHING_STATUS_SCENES,
  createBranchingStatusFractions,
  createBranchingStatusProps
} from './branching-flow-status-oracles-fixtures';

for (const scene of BRANCHING_STATUS_SCENES) {
  it(`CPU branching status counts hand-derived unresolved cells: ${scene.name}`, async () => {
    const inputs = {
      fractions: createBranchingStatusFractions(scene.cycle),
      values: new Float32Array([1, 1, 1]),
      validity: new Uint32Array(scene.validity)
    };
    const result = computeTerrainBranchingFlowOnCPU(inputs, {
      grid: BRANCHING_STATUS_GRID,
      mode: 'transport'
    });
    expect(Array.from(result.status)).toEqual(scene.expectedStatus);
    expect(Array.from(result.unresolvedCount)).toEqual(scene.expectedCount);
    const operation = new CPUTerrainBranchingFlowOperation(createBranchingStatusProps());
    const output = await operation.execute({
      inputs: Object.fromEntries(
        Object.entries(inputs).map(([port, values]) => [
          port,
          {
            format: port === 'validity' ? ('uint32' as const) : ('float32' as const),
            length: values.length,
            chunks: [{values, length: values.length}],
            pack: () => values
          }
        ])
      )
    });
    expect(Array.from(output['status'] as Uint32Array)).toEqual(scene.expectedStatus);
    expect(Array.from(output['unresolvedCount'] as Uint32Array)).toEqual(scene.expectedCount);
  });
}
