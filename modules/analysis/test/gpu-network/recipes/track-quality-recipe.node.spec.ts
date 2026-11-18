// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {
  addTrackQualityRecipe,
  type GPUTrackQualityRecipeProps
} from '../../../src/gpu-network/recipes/track-quality-recipe';
import {GPUTrajectoryMetrics} from '../../../src/gpu-spatial-analysis/trajectory-analysis/gpu-trajectory-metrics';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

function createProps(graph: GPUCommandGraph): GPUTrackQualityRecipeProps {
  return {
    trajectory: {
      positions: createTransientView(graph, 'positions', 'float32x2', 2),
      timestamps: createTransientView(graph, 'timestamps', 'float32', 2),
      trackOffsets: createTransientView(graph, 'track-offsets', 'uint32', 2),
      temporalContext: {
        encoding: 'float32-relative',
        unit: 'millisecond',
        epoch: 0n,
        axis: {kind: 'irregular'}
      }
    },
    thresholds: createTransientView(graph, 'thresholds', 'float32', 2)
  };
}

it('network:H3 forwards the declared millisecond context to trajectory metrics', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph);
    const result = addTrackQualityRecipe(graph, props);
    const metrics = result.contributors.find(
      contributor => contributor instanceof GPUTrajectoryMetrics
    ) as GPUTrajectoryMetrics;
    expect(metrics).toBeInstanceOf(GPUTrajectoryMetrics);
    // Recipe configuration is captured; GPU views retain the caller's identity.
    expect(metrics.props.temporalContext).toStrictEqual(props.trajectory.temporalContext);
    expect(metrics.props.timestamps).toBe(props.trajectory.timestamps);
    Object.assign(props.trajectory.temporalContext!, {unit: 'second'});
    expect(metrics.props.temporalContext?.unit).toBe('millisecond');
  } finally {
    device.destroy();
  }
});

it('network:H3 rejects mismatched timestamp encoding before scheduling and allows a corrected retry', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph);
    const context = props.trajectory.temporalContext;
    props.trajectory.temporalContext = {
      encoding: 'int64-words',
      unit: 'millisecond',
      epoch: 'unix',
      axis: {kind: 'irregular'}
    };
    expect(
      () =>
        new GPUTrajectoryMetrics({
          ...props.trajectory,
          output: {stepSpeeds: createTransientView(graph, 'standalone-speeds', 'float32', 2)}
        })
    ).toThrow(/does not support int64-words times/);
    const add = vi.spyOn(graph, 'add');
    expect(() => addTrackQualityRecipe(graph, props)).toThrow(/does not support int64-words times/);
    expect(add).not.toHaveBeenCalled();
    props.trajectory.temporalContext = context;
    expect(() => addTrackQualityRecipe(graph, props)).not.toThrow();
  } finally {
    device.destroy();
  }
});
