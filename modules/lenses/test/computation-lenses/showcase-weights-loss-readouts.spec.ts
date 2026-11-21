// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPU_PLANAR_SPATIAL_CONTEXT} from '@luma-contributors/analysis/analysis-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  getGPUSpatialAutocorrelationParameterValues,
  getGPUPermutationParameterValues,
  GPUSpatialWeightsAlgebra
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {requireTestDevice} from '../../../analysis/src/testing/require-test-device';
import {
  readFloat32,
  readUint32,
  submitGraph
} from '../../../analysis/src/testing/gpu-contributor-test-utils';
import {SpatialAnalysisResources} from '../../../../examples/deck/spatial-analysis-showcase/src/engine/resources';
import {
  compileStatGraph,
  createStatBuffers
} from '../../../../examples/deck/spatial-analysis-showcase/src/scenes/weights/b4-hotspot-stats';
import {createAnalysisSpace} from '../../../../examples/deck/spatial-analysis-showcase/src/scenes/weights/b4-space';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {importGraphBuffer} from '../../../../examples/deck/spatial-analysis-showcase/src/engine/graph-buffers';

it('scene algebra status follows a truncated higher-order writer and an independent fitting block writer', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const resources = new SpatialAnalysisResources(device, 'scene-algebra-status');
  try {
    const overflow = resources.createBuffer('analysis-overflow', 4);
    const candidateOverflow = resources.createBuffer('analysis-candidate', 4);
    const total = resources.createBuffer('analysis-total', 4);
    const searchOverflow = resources.createBuffer('search-overflow', Uint32Array.of(0));
    const searchCandidate = resources.createBuffer('search-candidate', Uint32Array.of(0));
    const space = await createAnalysisSpace({
      device,
      resources,
      id: 'algebra-space',
      spatialContext: GPU_PLANAR_SPATIAL_CONTEXT,
      rows: 3,
      slots: 4,
      unionSlots: 8,
      positions: resources.createBuffer('positions', Float32Array.of(0, 0, 1, 0, 2, 0)),
      values: resources.createBuffer('values', Float32Array.of(1, 2, 3)),
      mask: resources.createBuffer('mask', Uint32Array.of(1, 1, 1)),
      categories: Uint32Array.of(0, 0, 1),
      drawAllLinks: false,
      producer: {overflow, candidateOverflow, total}
    });
    const compile = async (block: boolean) => {
      const graph = new GPUCommandGraph<void>(device);
      const output = {
        spatialContext: GPU_PLANAR_SPATIAL_CONTEXT,
        offsets: importGraphBuffer(graph, 'offsets', space.csr.offsets, 'uint32', 4),
        neighbors: importGraphBuffer(graph, 'neighbors', space.csr.neighbors, 'uint32', 4),
        weights: importGraphBuffer(graph, 'weights', space.csr.weights, 'float32', 4),
        status: {
          overflow: importGraphBuffer(graph, 'overflow', overflow, 'uint32', 1),
          candidateOverflow: importGraphBuffer(graph, 'candidate', candidateOverflow, 'uint32', 1),
          requiredCount: importGraphBuffer(graph, 'total', total, 'uint32', 1)
        }
      };
      graph.add(
        block
          ? new GPUSpatialWeightsAlgebra({
              operation: 'block',
              groupCount: 2,
              groupIds: importGraphBuffer(
                graph,
                'groups',
                resources.createBuffer('groups', Uint32Array.of(0, 0, 1)),
                'uint32',
                3
              ),
              output
            })
          : new GPUSpatialWeightsAlgebra({
              operation: 'higherOrder',
              order: 3,
              cumulative: true,
              workCapacity: 9,
              weights: {
                spatialContext: GPU_PLANAR_SPATIAL_CONTEXT,
                offsets: importGraphBuffer(
                  graph,
                  'input-offsets',
                  resources.createBuffer('input-offsets', Uint32Array.of(0, 2, 4, 6)),
                  'uint32',
                  4
                ),
                neighbors: importGraphBuffer(
                  graph,
                  'input-neighbors',
                  resources.createBuffer('input-neighbors', Uint32Array.of(1, 2, 0, 2, 0, 1)),
                  'uint32',
                  6
                ),
                weights: importGraphBuffer(
                  graph,
                  'input-weights',
                  resources.createBuffer('input-weights', new Float32Array(6).fill(1)),
                  'float32',
                  6
                ),
                status: {
                  overflow: importGraphBuffer(graph, 'input-overflow', searchOverflow, 'uint32', 1),
                  candidateOverflow: importGraphBuffer(
                    graph,
                    'input-candidate',
                    searchCandidate,
                    'uint32',
                    1
                  )
                }
              },
              output
            })
      );
      return resources.track(await graph.compileAsync());
    };
    const higher = await compile(false),
      block = await compile(true);
    space.setDisplayChoice('summary', 'mean');
    const run = (compiled: typeof higher) => {
      const encoder = device.createCommandEncoder();
      compiled.encode(encoder, {parameters: undefined});
      space.encode(encoder, {
        normalizeLag: false,
        matrix: 'weights',
        display: 'summary',
        summary: 'mean',
        includeFocal: false,
        focalWeight: 1,
        rowStandardized: false
      });
      device.submit(encoder.finish());
    };
    run(higher); // Three mutually connected rows require 6 slots; final capacity is only 4.
    expect(await readFloat32(space.shown, 3)).toEqual([NaN, NaN, NaN]);
    searchOverflow.write(Uint32Array.of(1));
    searchCandidate.write(Uint32Array.of(1));
    run(block); // Independent groups [0,0,1] require just the two reciprocal links.
    expect(await readFloat32(space.shown, 3)).toEqual([2, 1, NaN]);
    expect(await readFloat32(space.shownExtent, 2)).toEqual([1, 2]);
    searchOverflow.write(Uint32Array.of(0));
    run(higher); // Candidate-only input loss is inherited even with clear search final overflow.
    expect(await readUint32(candidateOverflow)).toEqual([1]);
    expect(await readFloat32(space.shown, 3)).toEqual([NaN, NaN, NaN]);
  } finally {
    resources.destroy();
  }
});

it('showcase neighborhood graph suppresses candidate-only and final loss then restores a literal range', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const resources = new SpatialAnalysisResources(device, 'neighborhood-loss-test');
  try {
    const overflow = resources.createBuffer('overflow', Uint32Array.of(0));
    const candidateOverflow = resources.createBuffer('candidate-overflow', Uint32Array.of(1));
    const space = await createAnalysisSpace({
      spatialContext: GPU_PLANAR_SPATIAL_CONTEXT,
      device,
      resources,
      id: 'neighborhood-loss',
      rows: 2,
      slots: 2,
      unionSlots: 4,
      positions: resources.createBuffer('positions', Float32Array.of(0, 0, 1, 0)),
      values: resources.createBuffer('values', Float32Array.of(1, 2)),
      mask: resources.createBuffer('mask', Uint32Array.of(1, 1)),
      categories: Uint32Array.of(0, 1),
      drawAllLinks: false,
      producer: {
        overflow,
        candidateOverflow,
        total: resources.createBuffer('total', Uint32Array.of(2))
      }
    });
    space.csr.offsets.write(Uint32Array.of(0, 1, 2));
    space.csr.neighbors.write(Uint32Array.of(1, 0));
    space.csr.weights.write(Float32Array.of(1, 1));
    space.setDisplayChoice('summary', 'mean');
    const run = () => {
      const encoder = device.createCommandEncoder();
      space.encode(encoder, {
        normalizeLag: false,
        matrix: 'weights',
        display: 'summary',
        summary: 'mean',
        includeFocal: false,
        focalWeight: 1,
        rowStandardized: false
      });
      device.submit(encoder.finish());
    };
    run();
    expect(await readFloat32(space.shown, 2)).toEqual([NaN, NaN]);
    candidateOverflow.write(Uint32Array.of(0));
    overflow.write(Uint32Array.of(1));
    run();
    expect(await readFloat32(space.shown, 2)).toEqual([NaN, NaN]);
    overflow.write(Uint32Array.of(0));
    run();
    // Reciprocal unit edges: each mean is the other row's value; recovered extent is [1,2].
    expect(await readFloat32(space.shown, 2)).toEqual([2, 1]);
    expect(await readFloat32(space.shownExtent, 2)).toEqual([1, 2]);
  } finally {
    resources.destroy();
  }
});

it('polygon hot-spots statistics graph suppresses candidate-only loss and recovers', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const resources = new SpatialAnalysisResources(device, 'hotspot-loss-test');
  try {
    // Four places on a cycle; each has its two adjacent places. Finite, nonconstant values
    // give defined Gi* statistics; dropping any raw contiguity candidates invalidates all rows.
    const weights = {
      offsets: resources.createBuffer('offsets', Uint32Array.of(0, 2, 4, 6, 8)),
      neighbors: resources.createBuffer('neighbors', Uint32Array.of(1, 3, 0, 2, 1, 3, 0, 2)),
      weights: resources.createBuffer('weights', new Float32Array(8).fill(1))
    };
    const overflow = resources.createBuffer('overflow', Uint32Array.of(0));
    const candidateOverflow = resources.createBuffer('candidate-overflow', Uint32Array.of(1));
    const parameterValues = getGPUSpatialAutocorrelationParameterValues({significanceLevel: 0.05});
    const parameters = resources.createParameterBuffer(
      'parameters',
      'float32',
      parameterValues.length,
      parameterValues
    );
    const permutationValues = getGPUPermutationParameterValues({
      permutationCount: 19,
      seed: 1,
      significanceLevel: 0.05
    });
    const permutationParameters = resources.createParameterBuffer(
      'permutation',
      'uint32',
      permutationValues.length,
      permutationValues
    );
    const buffers = createStatBuffers(resources, 4, 'result');
    const compiled = await compileStatGraph({
      device,
      resources,
      id: 'hotspot-loss',
      spatialContext: GPU_PLANAR_SPATIAL_CONTEXT,
      rows: 4,
      slots: 8,
      weights,
      weightsStatus: {overflow, candidateOverflow},
      values: resources.createBuffer('values', Float32Array.of(1, 2, 4, 8)),
      mask: resources.createBuffer('mask', new Uint32Array(4).fill(1)),
      buffers,
      parameters,
      permutationParameters,
      variant: {
        statistic: 'gi-star',
        inference: 'analytic',
        selfWeight: 1,
        falseDiscoveryRate: false,
        alternative: 'two-sided',
        rowStandardized: false,
        maximumNeighbors: 4
      }
    });
    submitGraph(device, compiled, undefined);
    expect((await readFloat32(buffers.zScores, 4)).every(Number.isNaN)).toBe(true);
    candidateOverflow.write(Uint32Array.of(0));
    submitGraph(device, compiled, undefined);
    // Gi*: mean=15/4, S=sqrt(115)/4, three unit weights including self. The
    // finite-population factor is sqrt((4*3-3^2)/(4-1))=1; z=(4*localSum-45)/sqrt(115).
    const scores = await readFloat32(buffers.zScores, 4);
    expect(scores).toHaveLength(4);
    const expected = [
      -0.09325048082403138, -1.5852581740085334, 1.0257552890643453, 0.6527533657682196
    ];
    scores.forEach((value, index) => {
      expect(Number.isFinite(value)).toBe(true);
      expect(value).toBeCloseTo(expected[index], 5);
    });
  } finally {
    resources.destroy();
  }
});
