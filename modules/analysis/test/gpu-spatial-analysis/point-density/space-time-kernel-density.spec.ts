// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  getGPUSpaceTimeKernelDensityParameterValues,
  GPUSpaceTimeKernelDensity,
  GPUSpaceTimeRelativeRisk,
  GPU_SPACE_TIME_KERNEL_DENSITY_PARAMETER_LENGTH,
  type GPUKernelDensityKernel,
  type GPUSpaceTimeTemporalKernel
} from '../../../src/gpu-spatial-analysis/point-density/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {makeExtentGrid} from '../spatial-interpolation/extent-grid';
import {
  createSpaceTimeDensityScene,
  getSpaceTimeDensityReference,
  type SpaceTimeDensityScene
} from './space-time-kernel-density-reference';

/**
 * GPU versus the binary64 reference: `|gpu - reference| <= 2e-4 * max|reference|`, the tolerance of
 * `GPUKernelDensity` (f32 accumulation, the f32 erfc fit and the quadrant mass cancellation).
 */
const RELATIVE_TOLERANCE = 2e-4;

const EXTENT = [0, 0, 12, 8] as const;
const COLUMNS = 8;
const ROWS = 6;
const SLICES = 7;
const BANDWIDTH = 1.4;
const TEMPORAL_BANDWIDTH = 1.7;
const FIRST_SLICE_TIME = 1;
const SLICE_INTERVAL = 2;

type Options = {
  spatialKernel: GPUKernelDensityKernel;
  temporalKernel: GPUSpaceTimeTemporalKernel;
  spatialEdge: boolean;
  temporalEdge: boolean;
  normalization: 'intensity' | 'probability';
  weighted: boolean;
  timeWindow?: readonly [number, number];
  /** Scene seed, default 7. */
  seed?: number;
};

function getSettings(options: Options) {
  return {
    bandwidth: BANDWIDTH,
    temporalBandwidth: TEMPORAL_BANDWIDTH,
    firstSliceTime: FIRST_SLICE_TIME,
    sliceInterval: SLICE_INTERVAL,
    timeWindow: options.timeWindow
  };
}

async function runDensity(device: Device, options: Options, scene: SpaceTimeDensityScene) {
  const grid = makeExtentGrid(COLUMNS, ROWS, EXTENT);
  const voxelCount = COLUMNS * ROWS * SLICES;
  const pointCount = scene.times.length;
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'space-time-density-parameters',
    format: 'float32',
    length: GPU_SPACE_TIME_KERNEL_DENSITY_PARAMETER_LENGTH
  });
  const valuesOut = track(createOutputBuffer(device, voxelCount));
  const weightSumOut = track(createOutputBuffer(device, 1));
  const invalidCountOut = track(createOutputBuffer(device, 1));
  const graph = new GPUCommandGraph(device, {id: 'space-time-density-graph'});
  graph.add(
    new GPUSpaceTimeKernelDensity({
      id: 'stkde',
      positions: importGraphBuffer(
        graph,
        'positions',
        track(createInputBuffer(device, Float32Array.from(scene.positions))),
        'float32x2',
        pointCount
      ),
      times: importGraphBuffer(
        graph,
        'times',
        track(createInputBuffer(device, Float32Array.from(scene.times))),
        'float32',
        pointCount
      ),
      weights:
        options.weighted && scene.weights
          ? importGraphBuffer(
              graph,
              'weights',
              track(createInputBuffer(device, Float32Array.from(scene.weights))),
              'float32',
              pointCount
            )
          : undefined,
      parameters: parameterBuffer.importToGraph(graph),
      grid,
      sliceCount: SLICES,
      indexGridSize: [4, 3],
      indexBounds: EXTENT,
      spatialKernel: options.spatialKernel,
      temporalKernel: options.temporalKernel,
      spatialEdgeCorrection: options.spatialEdge ? 'uniform' : 'none',
      temporalEdgeCorrection: options.temporalEdge ? 'uniform' : 'none',
      normalization: options.normalization,
      output: {
        values: importGraphBuffer(graph, 'values-out', valuesOut, 'float32', voxelCount),
        weightSum: importGraphBuffer(graph, 'weight-sum-out', weightSumOut, 'float32', 1),
        status: {
          invalidCount: importGraphBuffer(graph, 'invalid-count-out', invalidCountOut, 'uint32', 1)
        }
      }
    })
  );
  const compiled = graph.compile();
  try {
    parameterBuffer.write(
      getGPUSpaceTimeKernelDensityParameterValues(getSettings(options), grid, SLICES)
    );
    submitGraph(device, compiled, undefined);
    return {
      values: Float32Array.from(await readFloat32(valuesOut, voxelCount)),
      weightSum: (await readFloat32(weightSumOut, 1))[0],
      invalidCount: (await readUint32(invalidCountOut, 1))[0]
    };
  } finally {
    compiled.destroy();
    parameterBuffer.destroy();
    for (const buffer of buffers) {
      buffer.destroy();
    }
  }
}

function getReference(options: Options, scene: SpaceTimeDensityScene) {
  return getSpaceTimeDensityReference({
    scene: {...scene, weights: options.weighted ? scene.weights : undefined},
    columns: COLUMNS,
    rows: ROWS,
    extent: EXTENT,
    bandwidth: BANDWIDTH,
    temporalBandwidth: TEMPORAL_BANDWIDTH,
    firstSliceTime: FIRST_SLICE_TIME,
    sliceInterval: SLICE_INTERVAL,
    sliceCount: SLICES,
    spatialKernel: options.spatialKernel,
    temporalKernel: options.temporalKernel,
    spatialEdge: options.spatialEdge,
    temporalEdge: options.temporalEdge,
    timeWindow: options.timeWindow,
    normalization: options.normalization
  });
}

function expectField(actual: ArrayLike<number>, expected: readonly number[], label: string): void {
  const scale = Math.max(...expected.filter(Number.isFinite).map(Math.abs));
  expect(scale, `${label} reference is not all zero`).toBeGreaterThan(0);
  let worst = 0;
  for (const [voxel, value] of expected.entries()) {
    if (Number.isNaN(value)) {
      expect(actual[voxel], `${label} voxel ${voxel} should be NaN`).toBeNaN();
    } else {
      expect(Number.isFinite(actual[voxel]), `${label} voxel ${voxel} finite`).toBe(true);
      worst = Math.max(worst, Math.abs(actual[voxel] - value));
    }
  }
  expect(worst, label).toBeLessThanOrEqual(RELATIVE_TOLERANCE * scale);
}

const CASES: Options[] = [
  {
    spatialKernel: 'gaussian',
    temporalKernel: 'gaussian',
    spatialEdge: false,
    temporalEdge: false,
    normalization: 'intensity',
    weighted: true
  },
  {
    spatialKernel: 'quartic',
    temporalKernel: 'epanechnikov',
    spatialEdge: true,
    temporalEdge: true,
    normalization: 'probability',
    weighted: true,
    seed: 11
  },
  {
    spatialKernel: 'epanechnikov',
    temporalKernel: 'quartic',
    spatialEdge: true,
    temporalEdge: false,
    normalization: 'intensity',
    weighted: false,
    seed: 3
  },
  {
    spatialKernel: 'gaussian',
    temporalKernel: 'gaussian',
    spatialEdge: true,
    temporalEdge: true,
    normalization: 'probability',
    weighted: false,
    seed: 5
  },
  {
    // A time window narrower than the data and the slices: points and slices outside are dropped.
    spatialKernel: 'quartic',
    temporalKernel: 'quartic',
    spatialEdge: false,
    temporalEdge: true,
    normalization: 'intensity',
    weighted: true,
    timeWindow: [2, 11],
    seed: 9
  }
];

it.each(
  CASES.map(options => ({
    name: `${options.spatialKernel}/${options.temporalKernel} space ${options.spatialEdge ? 'uniform' : 'none'} time ${options.temporalEdge ? 'uniform' : 'none'} ${options.normalization} weighted=${options.weighted}`,
    options
  }))
)('GPUSpaceTimeKernelDensity matches the binary64 reference: $name', async ({options}) => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = createSpaceTimeDensityScene(options.seed ?? 7, options.weighted);
  const reference = getReference(options, scene);
  const run = await runDensity(device, options, scene);
  expectField(run.values, reference.values, 'density');
  expect(run.invalidCount).toBe(reference.invalidCount);
  expect(Math.abs(run.weightSum - reference.weightSum)).toBeLessThanOrEqual(
    1e-5 * Math.max(1, reference.weightSum)
  );
});

it('GPUSpaceTimeKernelDensity is bitwise reproducible and keeps time order inside index cells', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Every point sits in two index cells and many share a time, so only the stable sort and the
  // restored in-cell order fix the summation order.
  const scene: SpaceTimeDensityScene = {positions: [], times: [], weights: []};
  for (let index = 0; index < 400; index++) {
    scene.positions.push(
      Math.fround(5 + 0.4 * Math.sin(index)),
      Math.fround(3 + 0.4 * Math.cos(index * 1.3))
    );
    scene.times.push(Math.fround((index * 7) % 13));
    scene.weights!.push(Math.fround(1 + (index % 5) / 3));
  }
  const options = CASES[0];
  const first = await runDensity(device, options, scene);
  const second = await runDensity(device, options, scene);
  expect(Array.from(first.values)).toEqual(Array.from(second.values));
  expectField(first.values, getReference(options, scene).values, 'dense cell');
});

it('GPUSpaceTimeKernelDensity ignores a row order permutation', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = createSpaceTimeDensityScene(21, true);
  const count = scene.times.length;
  const order = Array.from({length: count}, (_, index) => (index * 37 + 11) % count);
  expect(new Set(order).size).toBe(count);
  const shuffled: SpaceTimeDensityScene = {
    positions: order.flatMap(row => [scene.positions[2 * row], scene.positions[2 * row + 1]]),
    times: order.map(row => scene.times[row]),
    weights: order.map(row => scene.weights![row])
  };
  const original = await runDensity(device, CASES[0], scene);
  const permuted = await runDensity(device, CASES[0], shuffled);
  expectField(permuted.values, Array.from(original.values), 'permuted rows');
});

it('GPUSpaceTimeRelativeRisk matches the ratio of the two binary64 densities', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const grid = makeExtentGrid(COLUMNS, ROWS, EXTENT);
  const voxelCount = COLUMNS * ROWS * SLICES;
  const caseScene = createSpaceTimeDensityScene(7, true);
  const controlScene = createSpaceTimeDensityScene(19, true);
  // Compact temporal kernels leave voxels with no control mass and no case mass.
  const options: Options = {
    spatialKernel: 'quartic',
    temporalKernel: 'epanechnikov',
    spatialEdge: true,
    temporalEdge: true,
    normalization: 'probability',
    weighted: true
  };
  const caseReference = getReference(options, caseScene).values;
  const controlReference = getReference(options, controlScene).values;
  for (const scale of ['log', 'ratio'] as const) {
    const buffers: Buffer[] = [];
    const track = (buffer: Buffer) => {
      buffers.push(buffer);
      return buffer;
    };
    const parameterBuffer = new GPUParameterBuffer(device, {
      id: 'space-time-risk-parameters',
      format: 'float32',
      length: GPU_SPACE_TIME_KERNEL_DENSITY_PARAMETER_LENGTH
    });
    const riskOut = track(createOutputBuffer(device, voxelCount));
    const caseInvalidOut = track(createOutputBuffer(device, 1));
    const graph = new GPUCommandGraph(device, {id: 'space-time-risk-graph'});
    const view = <Format extends 'float32x2' | 'float32'>(
      name: string,
      data: number[],
      format: Format,
      length: number
    ) =>
      importGraphBuffer(
        graph,
        name,
        track(createInputBuffer(device, Float32Array.from(data))),
        format,
        length
      );
    graph.add(
      new GPUSpaceTimeRelativeRisk({
        id: 'risk',
        casePositions: view(
          'case-positions',
          caseScene.positions,
          'float32x2',
          caseScene.times.length
        ),
        caseTimes: view('case-times', caseScene.times, 'float32', caseScene.times.length),
        caseWeights: view('case-weights', caseScene.weights!, 'float32', caseScene.times.length),
        controlPositions: view(
          'control-positions',
          controlScene.positions,
          'float32x2',
          controlScene.times.length
        ),
        controlTimes: view(
          'control-times',
          controlScene.times,
          'float32',
          controlScene.times.length
        ),
        controlWeights: view(
          'control-weights',
          controlScene.weights!,
          'float32',
          controlScene.times.length
        ),
        parameters: parameterBuffer.importToGraph(graph),
        grid,
        sliceCount: SLICES,
        indexGridSize: [4, 3],
        indexBounds: EXTENT,
        spatialKernel: options.spatialKernel,
        temporalKernel: options.temporalKernel,
        scale,
        output: {
          values: importGraphBuffer(graph, 'risk-out', riskOut, 'float32', voxelCount),
          status: {
            caseInvalidCount: importGraphBuffer(
              graph,
              'case-invalid-out',
              caseInvalidOut,
              'uint32',
              1
            )
          }
        }
      })
    );
    const compiled = graph.compile();
    try {
      parameterBuffer.write(
        getGPUSpaceTimeKernelDensityParameterValues(getSettings(options), grid, SLICES)
      );
      submitGraph(device, compiled, undefined);
      const risk = Array.from(await readFloat32(riskOut, voxelCount));
      expect((await readUint32(caseInvalidOut, 1))[0]).toBe(
        getReference(options, caseScene).invalidCount
      );
      let finiteCount = 0;
      let infiniteCount = 0;
      let nanCount = 0;
      for (let voxel = 0; voxel < voxelCount; voxel++) {
        const numerator = caseReference[voxel];
        const denominator = controlReference[voxel];
        if (!(denominator > 0) || Number.isNaN(numerator)) {
          expect(risk[voxel], `${scale} voxel ${voxel}`).toBeNaN();
          nanCount++;
        } else if (numerator === 0) {
          expect(risk[voxel], `${scale} voxel ${voxel}`).toBe(scale === 'log' ? -Infinity : 0);
          infiniteCount++;
        } else {
          const expected =
            scale === 'log' ? Math.log(numerator / denominator) : numerator / denominator;
          expect(Math.abs(risk[voxel] - expected), `${scale} voxel ${voxel}`).toBeLessThanOrEqual(
            1e-3 * Math.max(1, Math.abs(expected))
          );
          finiteCount++;
        }
      }
      // The scene exercises every branch.
      expect(finiteCount).toBeGreaterThan(20);
      expect(infiniteCount + nanCount).toBeGreaterThan(0);
    } finally {
      compiled.destroy();
      parameterBuffer.destroy();
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
  }
});
