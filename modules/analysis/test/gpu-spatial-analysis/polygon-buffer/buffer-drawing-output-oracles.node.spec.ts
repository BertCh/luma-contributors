// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {createBufferDrawingOracle} from './buffer-drawing-output-oracles';

it('withholds retained buffer geometry while either completeness flag is set', async () => {
  const device = createNullWebGPUDevice();
  const fixture = await createBufferDrawingOracle(device, true);
  try {
    const memory = createCPUGraphMemory();
    for (const entry of fixture.columns) memory.write(entry.view, entry.values);
    for (const [overflow, candidateOverflow, expected] of [
      [0, 1, 0],
      [0, 0, 1],
      [1, 0, 0]
    ]) {
      memory.write(fixture.overflow.view, new Uint32Array([overflow]));
      memory.write(fixture.candidateOverflow.view, new Uint32Array([candidateOverflow]));
      executeCommandNodesOnCPU(fixture.nodes, memory);
      const ringCount = fixture.columns.find(entry => entry.name === 'drawing-ring-count')!;
      expect(Array.from(memory.readUint32(ringCount.view))).toEqual([expected]);
    }
  } finally {
    fixture.destroy();
    device.destroy();
  }
});
