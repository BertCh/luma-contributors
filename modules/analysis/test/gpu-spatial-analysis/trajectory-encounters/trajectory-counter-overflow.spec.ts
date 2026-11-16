// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createOutputBuffer, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {createTrajectoryCounterOverflowFixture} from './trajectory-counter-overflow-fixture';
import {expectTrajectoryCounterOverflowReadback} from './trajectory-counter-overflow-assertions';

it('saturates concurrent production hit reservations and preserves the output prefix', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const state = createOutputBuffer(device, 1);
  const pairs = createOutputBuffer(device, 4);
  const overflow = createOutputBuffer(device, 1);
  const fixture = createTrajectoryCounterOverflowFixture(graph, {
    state: importGraphBuffer(graph, 'output-state', state, 'uint32', 1),
    pairs: importGraphBuffer(graph, 'output-pairs', pairs, 'uint32x2', 2),
    overflow: importGraphBuffer(graph, 'output-overflow', overflow, 'uint32', 1)
  });
  graph.add({id: 'counter-overflow', getCommandNodes: () => fixture.nodes});
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expectTrajectoryCounterOverflowReadback({
      state: await readUint32(state, 1),
      pairs: await readUint32(pairs, 4),
      overflow: await readUint32(overflow, 1)
    });
  } finally {
    compiled.destroy();
    state.destroy();
    pairs.destroy();
    overflow.destroy();
  }
});
