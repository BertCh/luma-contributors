// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {RecipeTestFixture} from '../gpu-spatial-analysis/recipes/recipe-harness';
import {requireTestDevice} from '../utils/require-test-device';
import {createWGSLKernelNode} from '../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  addExposureByZoneRecipe,
  addRegionalRasterChangeRecipe
} from '../../src/gpu-spatial-analysis/recipes/raster-summary-recipes';
import {
  addTrackQualityRecipe,
  GPU_TRACK_QUALITY_FLAGS
} from '../../src/gpu-network/recipes/track-quality-recipe';
import {getGPUChangeDetectionParameterValues} from '../../src/raster-analysis/change-detection/index';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';

const grid: GPURasterMetadata = {
  width: 2,
  height: 1,
  affine: [1, 0, 0, 0, 1, 0],
  pixelInterpretation: 'area'
};
function publish<Format extends 'float32' | 'uint32'>(
  fixture: RecipeTestFixture,
  name: string,
  source: GraphDataView<Format>
) {
  const output = fixture.output(name, source.format, source.length);
  const type = source.format === 'float32' ? 'f32' : 'u32';
  fixture.graph.add(
    createWGSLKernelNode(fixture.graph, {
      id: name,
      operation: 'review-copy',
      invocationCount: source.length,
      bindings: [
        {name: 'source', view: source, type, access: 'read'},
        {name: 'destination', view: output.view, type, access: 'read_write'}
      ],
      body: 'destination[destinationOffset+index]=source[sourceOffset+index];'
    })
  );
  return output;
}
async function fixture(name: string) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  return new RecipeTestFixture(device, name);
}

it('unknown exposure cutoff preserves observed support and publishes unknown classification', async () => {
  const setup = await fixture('review-exposure');
  try {
    const result = addExposureByZoneRecipe(setup.graph, {
      grid,
      zones: setup.input('zones', new Uint32Array([0, 0]), 'uint32', 2),
      zoneCount: 1,
      hazard: {
        id: 'hazard',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: setup.input('hazard', new Float32Array([4, NaN]), 'float32', 2)
        }
      },
      population: {
        id: 'population',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: setup.input('population', new Float32Array([10, 20]), 'float32', 2)
        }
      },
      weightMeaning: 'count',
      threshold: setup.input('threshold', new Float32Array([NaN]), 'float32', 1)
    });
    const observed = publish(setup, 'observed-out', result.outputs.observedPopulation),
      coverage = publish(setup, 'coverage-out', result.outputs.coverage),
      exposed = publish(setup, 'exposed-out', result.outputs.exposedPopulation),
      share = publish(setup, 'share-out', result.outputs.exposedShare),
      invalid = publish(setup, 'invalid-out', result.outputs.thresholdInvalidCount);
    setup.run();
    expect(await setup.readFloat32(observed, 1)).toEqual([10]);
    expect((await setup.readFloat32(coverage, 1))[0]).toBeCloseTo(1 / 3, 6);
    expect(await setup.readFloat32(exposed, 1)).toEqual([NaN]);
    expect(await setup.readFloat32(share, 1)).toEqual([NaN]);
    expect(await setup.readUint32(invalid, 1)).toEqual([1]);
  } finally {
    setup.destroy();
  }
});

it('invalid regional change cutoff is unknown rather than zero changed area', async () => {
  const setup = await fixture('review-change');
  try {
    const result = addRegionalRasterChangeRecipe(setup.graph, {
      grid,
      change: {
        slices: setup.input('slices', new Float32Array([1, 3, 4, 8]), 'float32', 4),
        sliceCount: 2,
        parameters: setup.parameters(
          'parameters',
          'float32',
          getGPUChangeDetectionParameterValues({beforeSlice: 0, afterSlice: 1})
        )
      },
      zones: setup.input('zones', new Uint32Array([0, 0]), 'uint32', 2),
      zoneCount: 1,
      cellAreas: setup.input('areas', new Float32Array([1, 1]), 'float32', 2),
      threshold: setup.input('threshold', new Float32Array([-1]), 'float32', 1)
    });
    const changed = publish(setup, 'changed-out', result.outputs.changedArea),
      coverage = publish(setup, 'coverage-out', result.outputs.coverage),
      invalid = publish(setup, 'invalid-out', result.outputs.thresholdInvalidCount);
    setup.run();
    expect(await setup.readFloat32(changed, 1)).toEqual([NaN]);
    expect(await setup.readFloat32(coverage, 1)).toEqual([1]);
    expect(await setup.readUint32(invalid, 1)).toEqual([1]);
  } finally {
    setup.destroy();
  }
});

it('track quality point-centred speed test flags the outlier fix and not the fix after it', async () => {
  const run = async (positions: number[], speedOutlierMode?: 'step' | 'point-centred') => {
    const setup = await fixture('review-point-centred');
    try {
      const count = positions.length / 2;
      const result = addTrackQualityRecipe(setup.graph, {
        speedOutlierMode,
        trajectory: {
          positions: setup.input('positions', Float32Array.from(positions), 'float32x2', count),
          timestamps: setup.input(
            'times',
            Float32Array.from({length: count}, (_, row) => row),
            'float32',
            count
          ),
          trackOffsets: setup.input('offsets', new Uint32Array([0, count]), 'uint32', 2)
        },
        thresholds: setup.input('thresholds', new Float32Array([5, 100]), 'float32', 2)
      });
      const flags = publish(setup, 'flags-out', result.outputs.flags);
      setup.run();
      return await setup.readUint32(flags, count);
    } finally {
      setup.destroy();
    }
  };
  const speed = GPU_TRACK_QUALITY_FLAGS.excessiveSpeed;
  const interiorOutlier = [0, 0, 1, 0, 502, 0, 3, 0, 4, 0];
  // The documented default stays step-based: the row after the outlier is flagged too.
  expect(await run(interiorOutlier)).toEqual([0, 0, speed, speed, 0]);
  expect(await run(interiorOutlier, 'step')).toEqual([0, 0, speed, speed, 0]);
  // Point-centred: only the row whose both adjacent steps are excessive.
  expect(await run(interiorOutlier, 'point-centred')).toEqual([0, 0, speed, 0, 0]);
  // An outlier first row is visible without blaming the fix after it.
  expect(await run([500, 0, 1, 0, 2, 0, 3, 0], 'point-centred')).toEqual([speed, 0, 0, 0]);
  // The track-start fix is not flagged for the outlier next to it.
  expect(await run([0, 0, 500, 0, 2, 0, 3, 0], 'point-centred')).toEqual([0, speed, 0, 0]);
  // An outlier last row, and a valid last fix after an interior outlier.
  expect(await run([0, 0, 1, 0, 2, 0, 900, 0], 'point-centred')).toEqual([0, 0, 0, speed]);
  expect(await run([0, 0, 1, 0, 900, 0, 3, 0], 'point-centred')).toEqual([0, 0, speed, 0]);
  // A single fast step cannot say which end is wrong: both ends are flagged.
  expect(await run([0, 0, 500, 0], 'point-centred')).toEqual([speed, speed]);
});

it('track quality reports invalid initial timestamps and invalid thresholds', async () => {
  const setup = await fixture('review-track');
  try {
    const result = addTrackQualityRecipe(setup.graph, {
      trajectory: {
        positions: setup.input('positions', new Float32Array([0, 0, 1, 0]), 'float32x2', 2),
        timestamps: setup.input('times', new Float32Array([NaN, 1]), 'float32', 2),
        trackOffsets: setup.input('offsets', new Uint32Array([0, 2]), 'uint32', 2)
      },
      thresholds: setup.input('thresholds', new Float32Array([NaN, 1]), 'float32', 2)
    });
    const flags = publish(setup, 'flags-out', result.outputs.flags);
    setup.run();
    expect(await setup.readUint32(flags, 2)).toEqual([
      GPU_TRACK_QUALITY_FLAGS.invalidTime | GPU_TRACK_QUALITY_FLAGS.invalidThreshold,
      GPU_TRACK_QUALITY_FLAGS.invalidTime | GPU_TRACK_QUALITY_FLAGS.invalidThreshold
    ]);
  } finally {
    setup.destroy();
  }
});
