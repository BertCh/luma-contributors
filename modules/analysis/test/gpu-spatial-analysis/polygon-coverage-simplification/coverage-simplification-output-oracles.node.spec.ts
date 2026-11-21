// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {createUnrepairableCrossingOracle} from './coverage-simplification-output-oracles';

useCPUExecutorTestHooks();
it('measures the crossing still present after the final repair round', () => {
  const device = createNullWebGPUDevice();
  const fixture = createUnrepairableCrossingOracle(device);
  try {
    const memory = createCPUGraphMemory();
    for (const entry of fixture.columns) memory.write(entry.view, entry.values);
    executeCommandNodesOnCPU(fixture.nodes, memory);
    expect(
      Array.from(
        memory.readUint32(fixture.columns.find(entry => entry.name === 'statistics')!.view)
      )
    ).toEqual([1, 1, 0, 0]);
  } finally {
    fixture.destroy();
    device.destroy();
  }
});
