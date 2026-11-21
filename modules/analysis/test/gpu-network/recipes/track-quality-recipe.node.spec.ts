// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {
  addTrackQualityRecipe,
  type GPUTrackQualityRecipeProps
} from '../../../src/gpu-network/recipes/track-quality-recipe';
import {GPUTrajectoryMetrics} from '../../../src/gpu-spatial-analysis/trajectory-analysis/gpu-trajectory-metrics';
import {
  computeTrackSimilarityOnCPU,
  GPUTrackSimilarity,
  decodeGPUTrackSimilarityPairStatus
} from '../../../src/gpu-spatial-analysis/track-similarity/index';
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

it('publishes per-pair similarity evidence without map matching (STA-10)', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const props = createProps(graph);
    props.trajectory.positions = createTransientView(graph, 'two-tracks', 'float32x2', 6);
    props.trajectory.timestamps = createTransientView(graph, 'two-track-times', 'float32', 6);
    props.trajectory.trackOffsets = createTransientView(graph, 'two-track-offsets', 'uint32', 3);
    props.similarity = {
      pairA: createTransientView(graph, 'pair-a', 'uint32', 1),
      pairB: createTransientView(graph, 'pair-b', 'uint32', 1),
      maximumFrechetVertexCount: 2
    };
    const recipe = addTrackQualityRecipe(graph, props);
    const stage = recipe.status.stages.find(entry => entry.stage === 'similarity');
    expect(stage).toBeDefined();
    // Read through the additive per-pair layout; it must reference the existing output.
    const layout = (stage as {layout?: {kind: string; view: unknown}} | undefined)?.layout;
    expect(layout?.kind).toBe('per-pair-bits');
    expect(layout?.view === recipe.outputs.similarity!.pairStatus).toBe(true);
    const contributor = recipe.contributors.find(
      entry => entry instanceof GPUTrackSimilarity
    ) as GPUTrackSimilarity;
    expect(contributor.props.output.pairStatus === layout?.view).toBe(true);
    expect(recipe.outputs.matching).toBeUndefined();
    // Parallel three-vertex tracks separated by 1: Hausdorff=1 by projection;
    // shorter vertex count 3 > cap 2 leaves sequence measures unavailable.
    const reference = computeTrackSimilarityOnCPU(
      {
        linesA: {
          positions: Float32Array.of(0, 0, 1, 0, 2, 0, 0, 1, 1, 1, 2, 1),
          lineOffsets: Uint32Array.of(0, 3, 6)
        },
        pairA: Uint32Array.of(0),
        pairB: Uint32Array.of(1)
      },
      {maximumFrechetVertexCount: 2, measures: ['hausdorff', 'frechet']}
    );
    expect(Number.isFinite(reference.hausdorff[0])).toBe(true);
    expect(Array.from(reference.hausdorff)).toEqual([1]);
    expect(Array.from(reference.frechet)).toEqual([NaN]);
    expect(Array.from(reference.pairStatus)).toEqual([2]);
    expect(decodeGPUTrackSimilarityPairStatus(reference.pairStatus)[0]).toMatchObject({
      frechetCapExceeded: true,
      uncappedMeasuresValid: true,
      dynamicProgramMeasuresValid: false
    });
  } finally {
    device.destroy();
  }
});
