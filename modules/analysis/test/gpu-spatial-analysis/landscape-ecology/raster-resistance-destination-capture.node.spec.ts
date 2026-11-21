// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import type {VertexFormat} from '@luma.gl/core';
import {expect, it} from 'vitest';
import {
  addRasterResistanceConnectivityRecipe,
  type GPURasterResistanceConnectivityProps
} from '../../../src/gpu-spatial-analysis/landscape-ecology/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('captures raster resistance output and nested destinations before validation', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const view = <Format extends VertexFormat>(name: string, format: Format, length: number) =>
      createTransientView(graph, name, format, length);
    const first = view('converged', 'uint32', 1);
    // This view is too short. The recipe must bind the same view it preflighted.
    const second = view('empty', 'uint32', 0);
    let outputReads = 0;
    let convergedReads = 0;
    const output: GPURasterResistanceConnectivityProps['output'] = {
      potentials: view('potentials', 'float32', 3),
      vertexCurrents: view('vertex-currents', 'float32', 3),
      diagnostics: view('diagnostics', 'float32x4', 1),
      solverStatus: view('solver-status', 'uint32x4', 1),
      status: {
        get converged() {
          return ++convergedReads === 1 ? first : second;
        }
      }
    };
    const result = addRasterResistanceConnectivityRecipe(graph, {
      grid: {width: 3, height: 1, affine: [1, 0, 0, 0, -1, 0], pixelInterpretation: 'area'},
      resistance: {
        id: 'resistance',
        format: 'float32',
        storage: {kind: 'buffer', values: view('resistance', 'float32', 3)}
      },
      parameters: view('parameters', 'float32', 8),
      source: 0,
      sink: 2,
      get output() {
        outputReads++;
        return output;
      }
    });
    expect(result.status.stages.find(stage => stage.stage === 'solve')?.status.converged).toBe(
      first
    );
    expect({outputReads, convergedReads}).toEqual({outputReads: 1, convergedReads: 1});
  } finally {
    device.destroy();
  }
});
