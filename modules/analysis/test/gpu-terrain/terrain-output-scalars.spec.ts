// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {makeGPURasterMetadata} from '../../src/raster-analysis/index';
import {requireTestDevice} from '../../src/testing/require-test-device';
import {
  GPUTerrainSummits,
  getGPUTerrainSummitsParameterValues
} from '../../src/gpu-terrain/terrain-features/gpu-terrain-summits';
import {
  GPUTerrainFlow,
  getGPUTerrainFlowParameterValues
} from '../../src/gpu-terrain/hydrology/gpu-terrain-flow';
import {GPUTerrainStreamOrder} from '../../src/gpu-terrain/hydrology/gpu-terrain-stream-order';
import {GPUTerrainWatersheds} from '../../src/gpu-terrain/hydrology/gpu-terrain-watersheds';
import {
  GPUPointHorizonProfile,
  getGPUPointHorizonProfileParameterValues
} from '../../src/gpu-terrain/point-horizon/index';
import {GPUTerrainFlowFractions} from '../../src/gpu-terrain/terrain-processing/gpu-terrain-flow-fractions';
import {createTerrainReviewFixture} from './terrain-processing/terrain-review-fixture';

it('summit candidate loss is zero independently of final capacity overflow', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const includeCandidateOverflow of [true, false]) {
    const rig = createTerrainReviewFixture(device);
    const grid = makeGPURasterMetadata({width: 3, height: 3});
    const candidateOverflow = rig.output('uint32', 1);
    candidateOverflow.buffer.write(Uint32Array.of(123));
    const count = rig.output('uint32', 1);
    const requiredCount = rig.output('uint32', 1);
    const overflow = rig.output('uint32', 1);
    // A single central summit, no earlier candidate cap; capacity zero loses the final row only.
    rig.graph.add(
      new GPUTerrainSummits({
        grid,
        maximumRadiusPixels: 1,
        elevation: {
          id: 'elevation',
          format: 'float32',
          storage: {kind: 'buffer', values: rig.input('float32', [0, 0, 0, 0, 10, 0, 0, 0, 0]).view}
        },
        parameters: rig.input(
          'float32',
          getGPUTerrainSummitsParameterValues({radius: 1, minimumDrop: 1}, grid)
        ).view,
        output: {
          summits: {
            ids: rig.output('uint32', 0).view,
            status: {
              count: count.view,
              requiredCount: requiredCount.view,
              overflow: overflow.view,
              candidateOverflow: includeCandidateOverflow ? candidateOverflow.view : undefined
            }
          }
        }
      })
    );
    const compiled = rig.run();
    try {
      expect(await count.read()).toEqual([0]);
      expect(await requiredCount.read()).toEqual([1]);
      expect(await overflow.read()).toEqual([1]);
      if (includeCandidateOverflow) expect(await candidateOverflow.read()).toEqual([0]);
    } finally {
      compiled.destroy();
      rig.destroy();
    }
  }
});

it('singleton flow counts one fill, three flat confirmations and one tour confirmation', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = createTerrainReviewFixture(device);
  const grid = makeGPURasterMetadata({width: 1, height: 1});
  const counts = [rig.output('uint32', 1), rig.output('uint32', 1), rig.output('uint32', 1)];
  counts.forEach(output => output.buffer.write(Uint32Array.of(123)));
  // Epsilon selects the relaxation fill. On a singleton every loop is unchanged on round one;
  // flat resolution runs its three independent confirming loops. No scheduling races are possible.
  rig.graph.add(
    new GPUTerrainFlow({
      grid,
      fillDepressions: true,
      resolveFlats: true,
      maximumFillIterations: 1,
      maximumFlatIterations: 1,
      maximumAccumulationIterations: 1,
      elevation: {
        id: 'elevation',
        format: 'float32',
        storage: {kind: 'buffer', values: rig.input('float32', [7]).view}
      },
      parameters: rig.input('float32', getGPUTerrainFlowParameterValues({fillEpsilon: 0.25}, grid))
        .view,
      output: {
        accumulation: rig.output('float32', 1).view,
        status: {
          fillIterationCount: counts[0].view,
          flatsIterationCount: counts[1].view,
          accumulationIterationCount: counts[2].view
        }
      }
    })
  );
  const compiled = rig.run();
  try {
    expect(await Promise.all(counts.map(output => output.read()))).toEqual([[1], [3], [1]]);
  } finally {
    compiled.destroy();
    rig.destroy();
  }
});

it('filled-only flow publishes flat diagnostics into poisoned caller buffers', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = createTerrainReviewFixture(device);
  const grid = makeGPURasterMetadata({width: 1, height: 1});
  const filled = rig.output('float32', 1);
  const converged = rig.output('uint32', 1);
  const rounds = rig.output('uint32', 1);
  converged.buffer.write(Uint32Array.of(123));
  rounds.buffer.write(Uint32Array.of(123));
  rig.graph.add(
    new GPUTerrainFlow({
      grid,
      fillDepressions: true,
      resolveFlats: true,
      maximumFillIterations: 1,
      maximumFlatIterations: 1,
      elevation: {
        id: 'elevation',
        format: 'float32',
        storage: {kind: 'buffer', values: rig.input('float32', [7]).view}
      },
      parameters: rig.input('float32', getGPUTerrainFlowParameterValues({fillEpsilon: 0.25}, grid))
        .view,
      output: {
        filledElevation: filled.view,
        status: {flatsConverged: converged.view, flatsIterationCount: rounds.view}
      }
    })
  );
  const compiled = rig.run();
  try {
    expect(await filled.read()).toEqual([7]);
    expect(await converged.read()).toEqual([1]);
    expect(await rounds.read()).toEqual([3]);
  } finally {
    compiled.destroy();
    rig.destroy();
  }
});

it('pooled flat diagnostics use disjoint aligned binding windows and overwrite poison', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = createTerrainReviewFixture(device);
  const grid = makeGPURasterMetadata({width: 1, height: 1});
  const words = rig.output('uint32', 65);
  words.buffer.write(new Uint32Array(65).fill(123));
  const statusView = (byteOffset: number) =>
    rig.graph.createDataView(words.view.buffer, {format: 'uint32', length: 1, byteOffset});
  rig.graph.add(
    new GPUTerrainFlow({
      grid,
      fillDepressions: true,
      resolveFlats: true,
      maximumFillIterations: 1,
      maximumFlatIterations: 1,
      elevation: {
        id: 'elevation',
        format: 'float32',
        storage: {kind: 'buffer', values: rig.input('float32', [7]).view}
      },
      parameters: rig.input('float32', getGPUTerrainFlowParameterValues({fillEpsilon: 0.25}, grid))
        .view,
      output: {
        filledElevation: rig.output('float32', 1).view,
        status: {flatsConverged: statusView(0), flatsIterationCount: statusView(256)}
      }
    })
  );
  const compiled = rig.run();
  try {
    const values = await words.read();
    expect([values[0], values[64]]).toEqual([1, 3]);
    expect(values.slice(1, 64)).toEqual(new Array(63).fill(123));
  } finally {
    compiled.destroy();
    rig.destroy();
  }
});

for (const kind of ['order', 'watershed'] as const) {
  it(`singleton ${kind} counts its first unchanged confirmation`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = createTerrainReviewFixture(device);
    const grid = makeGPURasterMetadata({width: 1, height: 1});
    const iterationCount = rig.output('uint32', 1);
    iterationCount.buffer.write(Uint32Array.of(123));
    const common = {grid, flowDirections: rig.input('uint32', [0]).view};
    const result = rig.output('uint32', 1);
    if (kind === 'order')
      rig.graph.add(
        new GPUTerrainStreamOrder({
          ...common,
          streams: rig.input('uint32', [1]).view,
          output: {streamOrder: result.view, status: {iterationCount: iterationCount.view}}
        })
      );
    else
      rig.graph.add(
        new GPUTerrainWatersheds({
          ...common,
          output: {labels: result.view, status: {iterationCount: iterationCount.view}}
        })
      );
    const compiled = rig.run();
    try {
      expect(await iterationCount.read()).toEqual([1]);
      expect(await result.read()).toEqual([kind === 'order' ? 1 : 0]);
    } finally {
      compiled.destroy();
      rig.destroy();
    }
  });
}

for (const traversal of ['march', 'pyramid'] as const) {
  it(`point-horizon ${traversal} counts exactly one declared sample per ray`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = createTerrainReviewFixture(device);
    const grid = makeGPURasterMetadata({width: 5, height: 5});
    const tangent = rig.output('float32', 4);
    const samples = rig.output('uint32', 4);
    // min=max=1 declares the singleton lattice {1}; all four cardinal samples are inside.
    // Ground 10, eye 12, no curvature: (10-12)/1 = -2, regardless of traversal.
    rig.graph.add(
      new GPUPointHorizonProfile({
        grid,
        azimuthCount: 4,
        minimumDistance: 1,
        maximumDistance: 1,
        cellSize: 1,
        traversal,
        elevation: {
          id: 'elevation',
          format: 'float32',
          storage: {
            kind: 'buffer',
            values: rig.input('float32', new Float32Array(25).fill(10)).view
          }
        },
        observers: rig.graph.createDataView(rig.input('float32', [2, 2, 2, 0]).view.buffer, {
          format: 'float32x4',
          length: 1
        }),
        parameters: rig.input('float32', getGPUPointHorizonProfileParameterValues({}, grid)).view,
        output: {tangent: tangent.view, samples: samples.view}
      })
    );
    const compiled = rig.run();
    try {
      expect(await tangent.read()).toEqual([-2, -2, -2, -2]);
      expect(await samples.read()).toEqual([1, 1, 1, 1]);
    } finally {
      compiled.destroy();
      rig.destroy();
    }
  });
}

for (const routing of ['d-infinity', 'mfd-freeman', 'mfd-quinn'] as const) {
  it(`flow-fraction ${routing} edge lengths use the independent rectangular metric`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = createTerrainReviewFixture(device);
    const grid = makeGPURasterMetadata({width: 3, height: 3, cellSize: [3, 4]});
    const lengths = rig.output('float32', 72);
    rig.graph.add(
      new GPUTerrainFlowFractions({
        grid,
        routing,
        elevation: rig.input('float32', [9, 8, 7, 6, 5, 4, 3, 2, 1]).view,
        flowDirections: rig.input('uint32', [4, 4, 4, 4, 4, 4, 1, 1, 0]).view,
        parameters: rig.input('float32', getGPUTerrainFlowParameterValues({}, grid)).view,
        output: {fractions: rig.output('float32', 72).view, edgeLengths: lengths.view}
      })
    );
    const compiled = rig.run();
    try {
      // E, SE, S, SW, W, NW, N, NE: Pythagoras on a 3-by-4 rectangle gives diagonal 5.
      expect((await lengths.read()).slice(32, 40)).toEqual([3, 5, 4, 5, 3, 5, 4, 5]);
    } finally {
      compiled.destroy();
      rig.destroy();
    }
  });
}
