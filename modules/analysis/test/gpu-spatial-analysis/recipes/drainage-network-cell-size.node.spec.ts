// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {addDrainageNetworkRecipe} from '../../../src/gpu-spatial-analysis/recipes/drainage-network-recipe';
import {GPUTerrainFlow} from '../../../src/gpu-terrain/hydrology/gpu-terrain-flow';
import {GPUTerrainFlowDistance} from '../../../src/gpu-terrain/hydrology/gpu-terrain-flow-distance';
import {GPUTerrainStreamNetwork} from '../../../src/gpu-terrain/hydrology/gpu-terrain-stream-network';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it.each([
  undefined,
  'uniform',
  'geographic',
  'web-mercator'
] as const)('uses cellSizeMode %s for flow, downstream distance and stream reaches', cellSizeMode => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const {contributors} = addDrainageNetworkRecipe(graph, {
      flow: {
        grid: {width: 2, height: 1, affine: [1, 0, 0, 0, -1, 0.5], pixelInterpretation: 'area'},
        elevation: {
          id: 'elevation',
          format: 'float32',
          storage: {kind: 'buffer', values: createTransientView(graph, 'elevation', 'float32', 2)}
        },
        parameters: createTransientView(graph, 'parameters', 'float32', 8),
        cellSizeMode
      }
    });
    const flow = contributors.find(contributor => contributor instanceof GPUTerrainFlow);
    const distance = contributors.find(
      contributor => contributor instanceof GPUTerrainFlowDistance
    );
    const network = contributors.find(
      contributor => contributor instanceof GPUTerrainStreamNetwork
    );
    expect(flow).toBeInstanceOf(GPUTerrainFlow);
    expect(distance).toBeInstanceOf(GPUTerrainFlowDistance);
    expect(network).toBeInstanceOf(GPUTerrainStreamNetwork);
    expect([
      flow?.props.cellSizeMode,
      distance?.props.cellSizeMode,
      network?.props.cellSizeMode
    ]).toEqual([cellSizeMode, cellSizeMode, cellSizeMode]);
  } finally {
    device.destroy();
  }
});
