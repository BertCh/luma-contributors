// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUTrajectoryKalman} from '../../../src/gpu-spatial-analysis/trajectory-estimation/gpu-trajectory-kalman';
import {computeTrajectoryKalmanOnCPU} from '../../../src/gpu-spatial-analysis/trajectory-estimation/cpu-trajectory-kalman-operation';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {createKalmanLocalMetadata} from '../../../src/gpu-spatial-analysis/trajectory-estimation/kalman-local-publication';
import {getViewBinding} from '@luma.gl/graph/gpu-core';
import {GPUTrajectoryKalmanOperation} from '../../../src/gpu-spatial-analysis/trajectory-estimation/gpu-trajectory-kalman-operation';

it('S3 GPU Kalman named form validates covarianceMode at construction', () => {
  expect(
    () =>
      new GPUTrajectoryKalmanOperation({
        positions: 'positions',
        timestamps: 'timestamps',
        trackOffsets: 'trackOffsets',
        parameters: 'parameters',
        rowCount: 2,
        covarianceMode: 'nonsense' as 'full',
        output: {filteredState: 'state', filteredCovariance: 'covariance', status: 'status'}
      })
  ).toThrow(/covarianceMode.*isotropic.*full/);
});

it.each([
  'isotropic',
  'full'
] as const)('B1 %s zero-row Kalman has only bindable views', covarianceMode => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const contributor = new GPUTrajectoryKalman({
      covarianceMode,
      positions: createTransientView(graph, 'positions', 'float32x2', 0),
      times: createTransientView(graph, 'times', 'float32', 0),
      offsets: createTransientView(graph, 'offsets', 'uint32', 1),
      parameters: createTransientView(
        graph,
        'parameters',
        'float32',
        covarianceMode === 'full' ? 10 : 4
      ),
      output: {
        filteredState: createTransientView(graph, 'state', 'float32x4', 0),
        filteredCovariance: createTransientView(graph, 'covariance', 'float32x4', 0),
        status: createTransientView(graph, 'status', 'uint32', 0)
      }
    });
    // A zero-row graph has no output elements and needs no dispatch or metadata binding.
    const nodes = contributor.getCommandNodes(graph);
    const metadata = createKalmanLocalMetadata(graph, 'zero-binding-probe', contributor.props);
    expect(() => getViewBinding(metadata.origins, () => null as never)).not.toThrow();
    for (const node of nodes)
      for (const resource of node.resources ?? []) {
        if ('buffer' in resource)
          expect(() => getViewBinding(resource.buffer as never, () => null as never)).not.toThrow();
      }
    expect(nodes.length).toBe(0);
  } finally {
    device.destroy();
  }
});

it('B6 Kalman preserves resident clamping for unowned observations and an excessive singleton', () => {
  for (const trackOffsets of [Uint32Array.of(3, 4), Uint32Array.of(999)]) {
    const result = computeTrajectoryKalmanOnCPU({
      positions: new Float32Array(4),
      timestamps: Float32Array.of(0, 1),
      trackOffsets,
      parameters: Float32Array.of(0, 1, 1, 0)
    });
    expect(Array.from(result.filteredState)).toEqual(Array(8).fill(0));
    expect(Array.from(result.status)).toEqual([0, 0]);
  }
});

it.each([
  'isotropic',
  'full'
] as const)('B3 %s reference retains the final small repeated measurement', covarianceMode => {
  const result = computeTrajectoryKalmanOnCPU(
    {
      positions: Float32Array.from(
        Array.from({length: 10}, (_, row) => [row === 0 ? 500000 : 0.01, 0]).flat()
      ),
      timestamps: Float32Array.from({length: 10}, (_, row) => 3600 * row),
      trackOffsets: Uint32Array.of(0, 10),
      parameters:
        covarianceMode === 'full'
          ? Float32Array.of(0.01, 0, 0.01, 0.0001, 0, 0.0001, 1, 0, 1, 0)
          : Float32Array.of(0.01, 0.0001, 1, 0)
    },
    {covarianceMode, smooth: true}
  );
  for (const state of [result.filteredState, result.smoothedState!]) {
    expect(Number.isFinite(state[36])).toBe(true);
    expect(state[36]).toBe(0.009999999776482582);
  }
});

it.each([
  'isotropic',
  'full'
] as const)('FLT-09 %s retains local state until after smoothing', covarianceMode => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const covarianceRows = covarianceMode === 'full' ? 36 : 9;
    const contributor = new GPUTrajectoryKalman({
      covarianceMode,
      positions: createTransientView(graph, 'positions', 'float32x2', 9),
      times: createTransientView(graph, 'times', 'float32', 9),
      offsets: createTransientView(graph, 'offsets', 'uint32', 2),
      parameters: createTransientView(
        graph,
        'parameters',
        'float32',
        covarianceMode === 'full' ? 10 : 4
      ),
      output: {
        filteredState: createTransientView(graph, 'filtered', 'float32x4', 9),
        filteredCovariance: createTransientView(graph, 'covariance', 'float32x4', covarianceRows),
        status: createTransientView(graph, 'status', 'uint32', 9),
        smoothedState: createTransientView(graph, 'smoothed', 'float32x4', 9),
        smoothedCovariance: createTransientView(
          graph,
          'smoothed-covariance',
          'float32x4',
          covarianceRows
        )
      }
    });
    const kernels = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
    const metadata = createKalmanLocalMetadata(graph, 'binding-probe', contributor.props);
    expect(() => getViewBinding(metadata.origins, () => null as never)).not.toThrow();
    if (covarianceMode === 'isotropic')
      expect(
        () =>
          new GPUTrajectoryKalman({...contributor.props, covarianceMode: 'nonsense' as 'isotropic'})
      ).toThrow(/covarianceMode.*isotropic.*full/);
    expect(kernels.at(-1)!.id).toContain('publish-absolute');
    for (const kernel of kernels) expect(kernel.source).not.toContain('@group(0) @binding(8)');
    expect(kernels.find(kernel => kernel.id.endsWith('-smooth'))!.source).toContain('rowOrigin(');
    expect(kernels.find(kernel => kernel.id.endsWith('-filter'))!.source).toContain(
      'kalmanPositionSubtract'
    );
    // q=0,r=1,v=1: second-row gain=(2/3,1/3), then seven unit predictions:
    // final local position 2/3+7/3=3, velocity 1/3. No later measurement revises it.
    const result = computeTrajectoryKalmanOnCPU(
      {
        positions: Float32Array.from([10_000_000, 0, 10_000_001, 0, ...Array(14).fill(NaN)]),
        timestamps: Float32Array.from({length: 9}, (_, row) => row),
        trackOffsets: Uint32Array.of(0, 9),
        parameters:
          covarianceMode === 'full'
            ? Float32Array.of(0, 0, 0, 1, 0, 1, 1, 0, 1, 0)
            : Float32Array.of(0, 1, 1, 0)
      },
      {covarianceMode, smooth: true}
    );
    expect(result.filteredState[32] - 10_000_000).toBe(3);
    expect(Number.isFinite(result.filteredState[34])).toBe(true);
    expect(result.filteredState[34]).toBeCloseTo(1 / 3, 6);
    expect(result.status[8]).toBe(2);
  } finally {
    device.destroy();
  }
});
