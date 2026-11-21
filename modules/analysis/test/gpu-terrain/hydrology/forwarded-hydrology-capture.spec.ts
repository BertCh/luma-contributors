// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPUTerrainWatersheds} from '../../../src/gpu-terrain/hydrology/gpu-terrain-watersheds';
import {
  GPUTerrainFlow,
  getGPUTerrainFlowParameterValues
} from '../../../src/gpu-terrain/hydrology/gpu-terrain-flow';
import {createParityGraph} from '../../oracles/parity-wp2-geodesic-harness';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';

it('direct watershed honors inherited one-round budget and iteration-count destination', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = createParityGraph(device);
  const grid = {
    width: 4,
    height: 1,
    affine: [1, 0, 0, 0, -1, 1] as const,
    pixelInterpretation: 'area' as const
  };
  const labels = rig.output('labels', 4, 'uint32');
  const iterationCount = rig.output('iterations', 1, 'uint32');
  const status = Object.create({iterationCount: iterationCount.view});
  const props = Object.create({
    grid,
    flowDirections: rig.input('directions', [1, 1, 1, 0], 'uint32').view,
    maximumIterations: 1,
    output: Object.create({labels: labels.view, status})
  });
  rig.graph.add(new GPUTerrainWatersheds(props));
  const compiled = rig.graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    // Synchronous doubling maps [1,2,3,3] -> [2,3,3,3]. Cell 0 has not reached the outlet.
    expect(await readUint32(labels.buffer, 4)).toEqual([0xfffffffe, 3, 3, 3]);
    expect(await readUint32(iterationCount.buffer, 1)).toEqual([1]);
  } finally {
    compiled.destroy();
    rig.destroy();
  }
});

it('direct flow retains inherited fill settings and selected writer after caller mutation', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = createParityGraph(device);
  const grid = {
    width: 2,
    height: 2,
    affine: [1, 0, 0, 0, -1, 2] as const,
    pixelInterpretation: 'area' as const
  };
  const filled = rig.output('filled', 4, 'float32');
  const inherited = {
    grid,
    elevation: {
      id: 'z',
      format: 'float32' as const,
      storage: {kind: 'buffer' as const, values: rig.input('z', [4, 3, 2, 1], 'float32').view}
    },
    parameters: rig.input('parameters', [...getGPUTerrainFlowParameterValues({}, grid)], 'float32')
      .view,
    fillDepressions: true,
    maximumFillIterations: 1,
    output: {filledElevation: filled.view}
  };
  const contributor = new GPUTerrainFlow(Object.create(inherited));
  inherited.fillDepressions = false;
  rig.graph.add(contributor);
  const compiled = rig.graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    // All four cells are boundary cells, so fill preserves their finite elevations exactly.
    expect(await readFloat32(filled.buffer, 4)).toEqual([4, 3, 2, 1]);
  } finally {
    compiled.destroy();
    rig.destroy();
  }
});
