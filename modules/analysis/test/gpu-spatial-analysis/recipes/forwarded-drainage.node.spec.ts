// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {addDrainageNetworkRecipe} from '../../../src/gpu-spatial-analysis/recipes/drainage-network-recipe';
import {GPUTerrainFlow} from '../../../src/gpu-terrain/hydrology/gpu-terrain-flow';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('captures inherited drainage flow settings and reads getters once', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const values = createTransientView(graph, 'elevation', 'float32', 4);
    let reads = 0;
    const flow = Object.create({
      grid: {width: 2, height: 2, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
      elevation: {id: 'elevation', format: 'float32', storage: {kind: 'buffer', values}},
      parameters: createTransientView(graph, 'parameters', 'float32', 8),
      get fillDepressions() {
        reads++;
        return true;
      },
      maximumFillIterations: 1
    });
    const result = addDrainageNetworkRecipe(graph, {flow});
    expect(reads).toBe(1);
    expect(result.intermediates.flowStatus.fillConverged?.length).toBe(1);
    const contributor = result.contributors.find(
      (item): item is GPUTerrainFlow => item instanceof GPUTerrainFlow
    );
    if (!contributor) throw new Error('expected terrain flow stage');
    const prototype = Object.getPrototypeOf(flow);
    const ownResult = addDrainageNetworkRecipe(graph, {
      id: 'own-flow',
      flow: {
        grid: prototype.grid,
        elevation: prototype.elevation,
        parameters: prototype.parameters,
        fillDepressions: true,
        maximumFillIterations: 1
      }
    });
    expect(ownResult.status.stages.map(stage => stage.stage)).toEqual(
      result.status.stages.map(stage => stage.stage)
    );
    expect(contributor.props.fillDepressions).toBe(true);
    expect(contributor.props.maximumFillIterations).toBe(1);
    prototype.maximumFillIterations = 1024;
    prototype.grid.affine[0] = 99;
    expect(contributor.props.maximumFillIterations).toBe(1);
    expect(contributor.props.grid.affine[0]).toBe(1);
    expect(reads).toBe(1);
  } finally {
    device.destroy();
  }
});
