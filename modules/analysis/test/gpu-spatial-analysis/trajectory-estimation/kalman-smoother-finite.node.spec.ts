// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUTrajectoryKalman} from '../../../src/gpu-spatial-analysis/trajectory-estimation/gpu-trajectory-kalman';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

it('TRJ-107 checks both backward outputs before publishing either', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const view = <Format extends 'float32' | 'float32x2' | 'float32x4' | 'uint32'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, name, format, length);
    const contributor = new GPUTrajectoryKalman({
      positions: view('positions', 'float32x2', 2),
      times: view('times', 'float32', 2),
      offsets: view('offsets', 'uint32', 2),
      parameters: view('parameters', 'float32', 4),
      output: {
        filteredState: view('state', 'float32x4', 2),
        filteredCovariance: view('covariance', 'float32x4', 2),
        status: view('status', 'uint32', 2),
        smoothedState: view('smoothed', 'float32x4', 2),
        smoothedCovariance: view('smoothed-covariance', 'float32x4', 2)
      }
    });
    const source = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph)).find(kernel =>
      kernel.id.endsWith('-smooth')
    )!.source;
    expect(source).toContain('finite(correctedState.x)');
    expect(source).toContain('finite(correctedCovariance.w)');
    expect(source.indexOf('finite(correctedCovariance.w)')).toBeLessThan(
      source.indexOf('storeSmoothedState(observation, correctedState)')
    );
  } finally {
    device.destroy();
  }
});
