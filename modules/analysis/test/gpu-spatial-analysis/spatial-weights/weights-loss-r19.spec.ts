// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {getCoreDefaultBuffer} from '../../../src/analysis-core/contracts/network-topology';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GPUContiguityWeights} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-contiguity-weights';
import {GPUAnalysisContiguityStatistics} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-analysis-contiguity-statistics';
import {GPUIslandAttachment} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-island-attachment';
import {GPUSpatialLag} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-lag';
import {GPUArealInterpolation} from '../../../src/gpu-spatial-analysis/areal-interpolation/gpu-areal-interpolation';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import {GPUMultivariateLocalGearyPermutationTest} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/gpu-multivariate-local-geary-permutation-test';
import {GPUSpatialEmpiricalBayesRates} from '../../../src/gpu-spatial-analysis/rate-smoothing/gpu-spatial-empirical-bayes-rates';
import {GPUChunkedSpatialWeightsTranspose} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights-transpose';
import {GPUGlobalSpatialStatistics} from '../../../src/gpu-spatial-analysis/global-spatial-statistics/gpu-global-spatial-statistics';
import {GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT} from '../../../src/gpu-spatial-analysis/global-spatial-statistics/global-spatial-statistics-layout';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from './spatial-weights-harness';
import {createIslandMoran, importPublishedWeights} from './weights-loss-r19-fixtures';

function expectFiniteValues(actual: number[], expected: readonly number[]) {
  expect(actual).toHaveLength(expected.length);
  actual.forEach((value, index) => {
    expect(Number.isFinite(value), `entry ${index}`).toBe(true);
    expect(value, `entry ${index}`).toBeCloseTo(expected[index], 5);
  });
}

for (const unweightedFastPath of [true, false]) {
  it(`WRT-02 areal interpolation clears stale candidate loss before lag (fast=${unweightedFastPath})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new WeightsRig(device);
    let compiled: ReturnType<typeof rig.graph.compile> | undefined;
    try {
      const output = rig.weightsOutput(1, 1);
      const overflow = rig.output('uint32', 1);
      const candidateOverflow = rig.output('uint32', 1);
      const producer = new GPUArealInterpolation({
        grid: makeGPURasterMetadata({width: 1, height: 1}),
        sourceZones: rig.input(Uint32Array.of(0), 'uint32', 1),
        targetZones: rig.input(Uint32Array.of(0), 'uint32', 1),
        sourceZoneCount: 1,
        targetZoneCount: 1,
        unweightedFastPath,
        output: {
          weights: {
            ...output.spatialWeights,
            status: {candidateOverflow: candidateOverflow.view}
          },
          status: {overflow: overflow.view}
        }
      });
      const lag = rig.output('float32', 1);
      rig.graph.add(producer);
      rig.graph.add(
        new GPUSpatialLag({
          weights: producer.props.output.weights,
          values: rig.input(Float32Array.of(7), 'float32', 1),
          output: lag.view
        })
      );
      compiled = rig.graph.compile();
      for (const stale of [1, 7]) {
        candidateOverflow.buffer.write(Uint32Array.of(stale));
        submitGraph(device, compiled, undefined);
        expect(await readUint32(candidateOverflow.buffer, 1)).toEqual([0]);
        expect(await readUint32(overflow.buffer, 1)).toEqual([0]);
        // Sole cell belongs entirely to both zones: share=1, so lag=1*7.
        expectFiniteValues(await readFloat32(lag.buffer, 1), [7]);
      }
    } finally {
      compiled?.destroy();
      rig.destroy();
    }
  });

  it(`B3 direct areal weights suppress final truncation and recover without a nested overflow alias (fast=${unweightedFastPath})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new WeightsRig(device);
    let compiled: ReturnType<typeof rig.graph.compile> | undefined;
    try {
      const output = rig.weightsOutput(1, 1);
      const overflow = rig.output('uint32', 1);
      const candidateOverflow = rig.output('uint32', 1);
      const sourceZones = rig.input(Uint32Array.of(0, 1), 'uint32', 2);
      const producer = new GPUArealInterpolation({
        grid: makeGPURasterMetadata({width: 2, height: 1}),
        sourceZones,
        targetZones: rig.input(Uint32Array.of(0, 0), 'uint32', 2),
        sourceZoneCount: 2,
        targetZoneCount: 1,
        unweightedFastPath,
        output: {
          weights: {...output.spatialWeights, status: {candidateOverflow: candidateOverflow.view}},
          status: {overflow: overflow.view}
        }
      });
      expect(producer.props.output.weights.status?.overflow).toBe(overflow.view);
      const lag = rig.output('float32', 1);
      rig.graph.add(producer);
      rig.graph.add(
        new GPUSpatialLag({
          weights: producer.props.output.weights,
          values: rig.input(Float32Array.of(7, 11), 'float32', 2),
          output: lag.view
        })
      );
      compiled = rig.graph.compile();
      for (const truncated of [true, false, true, false]) {
        // Two source-target pairs need two slots; coalescing both cells into source 0 needs one.
        rig.bufferOf(sourceZones).write(Uint32Array.of(0, truncated ? 1 : 0));
        candidateOverflow.buffer.write(Uint32Array.of(1));
        submitGraph(device, compiled, undefined);
        expect(await readUint32(candidateOverflow.buffer, 1)).toEqual([0]);
        expect(await readUint32(overflow.buffer, 1)).toEqual([Number(truncated)]);
        if (truncated) expect(await readFloat32(lag.buffer, 1)).toEqual([NaN]);
        // Source 0 covers both cells on recovery: its extensive share is 2/2, so lag is 7.
        else expectFiniteValues(await readFloat32(lag.buffer, 1), [7]);
      }
    } finally {
      compiled?.destroy();
      rig.destroy();
    }
  });
}

it('review B9 clean transpose overwrites prefilled chunk candidate status and computes downstream Moran', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  let compiled: ReturnType<typeof rig.graph.compile> | undefined;
  try {
    // Input 0->1->2->0; transpose is 0->2, 1->0, 2->1, with no islands.
    const input = rig.uploadWeights({
      offsets: [0, 1, 2, 3],
      neighbors: [1, 2, 0],
      weights: [1, 1, 1],
      distances: []
    });
    const output = rig.weightsOutput(3, 3);
    const candidate = rig.output('uint32', 1);
    const inheritedOverflow = rig.input(Uint32Array.of(0), 'uint32', 1);
    const inheritedCandidateOverflow = rig.input(Uint32Array.of(0), 'uint32', 1);
    const transpose = new GPUChunkedSpatialWeightsTranspose({
      weights: {
        rowCount: 3,
        status: {overflow: inheritedOverflow, candidateOverflow: inheritedCandidateOverflow},
        chunks: [{rowStart: 0, rowCount: 3, weights: input}]
      },
      output: {
        chunks: [
          {
            rowStart: 0,
            rowCount: 3,
            weights: {
              ...output.spatialWeights,
              status: {
                count: rig.output('uint32', 1).view,
                requiredCount: rig.output('uint32', 1).view,
                overflow: rig.output('uint32', 1).view,
                candidateOverflow: candidate.view
              }
            }
          }
        ]
      }
    });
    const scores = rig.output('float32', 3),
      quadrants = rig.output('uint32', 3),
      summary = rig.output('float32', GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length),
      capturedFlags = rig.output('uint32', 2);
    rig.graph.add(transpose);
    rig.graph.add(createIslandMoran(rig, transpose.outputWeights, scores.view, quadrants.view));
    rig.graph.add(
      new GPUGlobalSpatialStatistics({
        mode: 'reference',
        weights: transpose.outputWeights,
        values: rig.input(Float32Array.of(1, 2, 4), 'float32', 3),
        statistics: ['moran'],
        output: {summary: summary.view}
      })
    );
    const flags = transpose.outputWeights.status!;
    rig.graph.add(
      createWGSLKernelNode(rig.graph, {
        id: 'review-b9-capture-flags',
        operation: 'ReviewTransposeReadback',
        invocationCount: 1,
        bindings: [
          {name: 'overflow', view: flags.overflow!, type: 'u32', access: 'read'},
          {name: 'candidate', view: flags.candidateOverflow!, type: 'u32', access: 'read'},
          {name: 'output', view: capturedFlags.view, type: 'u32', access: 'read_write'}
        ],
        body: 'output[outputOffset] = overflow[overflowOffset]; output[outputOffset + 1u] = candidate[candidateOffset];'
      })
    );
    compiled = rig.graph.compile();
    for (const [garbage, inheritedFinalLoss, inheritedCandidateLoss] of [
      [42, 0, 0],
      [7, 0, 0],
      [42, 1, 0],
      [7, 0, 1],
      [7, 0, 0]
    ]) {
      candidate.buffer.write(Uint32Array.of(garbage));
      rig.bufferOf(inheritedOverflow).write(Uint32Array.of(inheritedFinalLoss));
      rig.bufferOf(inheritedCandidateOverflow).write(Uint32Array.of(inheritedCandidateLoss));
      submitGraph(device, compiled, undefined);
      expect(await readUint32(candidate.buffer, 1)).toEqual([0]);
      const incomplete = Boolean(inheritedFinalLoss || inheritedCandidateLoss);
      expect(await readUint32(capturedFlags.buffer, 2)).toEqual([0, Number(incomplete)]);
      expect(await output.read()).toMatchObject({offsets: [0, 1, 2, 3], neighbors: [2, 0, 1]});
      if (incomplete) {
        expect(await readFloat32(scores.buffer, 3)).toEqual([NaN, NaN, NaN]);
        const moran = (
          await readFloat32(summary.buffer, GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length)
        )[GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.moran];
        expect(Number.isNaN(moran)).toBe(true);
        continue;
      }
      // c=(-4,-1,5)/3; sum_ij c_i*c_j=-7/3, sum_i c_i²=14/3 and n/S0=1.
      expectFiniteValues(
        [
          (await readFloat32(summary.buffer, GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length))[
            GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.moran
          ]
        ],
        [-0.5]
      );
      expectFiniteValues(await readFloat32(scores.buffer, 3), [-1, 1, 1]);
    }
  } finally {
    compiled?.destroy();
    rig.destroy();
  }
});

it('review B8 cross-weight lag selects source row 2 from three values', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const weights = rig.uploadWeights({
      offsets: [0, 1],
      neighbors: [2],
      weights: [1],
      distances: []
    });
    const output = rig.output('float32', 1);
    rig.run(
      new GPUSpatialLag({
        weights,
        values: rig.input(Float32Array.of(10, 20, 30), 'float32', 3),
        output: output.view
      })
    );
    expectFiniteValues(await readFloat32(output.buffer, 1), [30]);
  } finally {
    rig.destroy();
  }
});

it('SSR-125 queen 2x2 producer reports candidate-only loss, then exact and tiled Geary recover', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const producer = new WeightsRig(device);
  let producerCompiled: ReturnType<typeof producer.graph.compile> | undefined;
  try {
    const generated = producer.weightsOutput(4, 12);
    const overflow = producer.output('uint32', 1),
      candidateOverflow = producer.output('uint32', 1);
    const contiguity = new GPUContiguityWeights({
      polygons: producer.polygons({
        positions: Float32Array.of(
          0,
          0,
          1,
          0,
          1,
          1,
          0,
          1,
          1,
          0,
          2,
          0,
          2,
          1,
          1,
          1,
          0,
          1,
          1,
          1,
          1,
          2,
          0,
          2,
          1,
          1,
          2,
          1,
          2,
          2,
          1,
          2
        ),
        ringOffsets: Uint32Array.of(0, 4, 8, 12, 16),
        polygonOffsets: Uint32Array.of(0, 1, 2, 3, 4),
        featureOffsets: Uint32Array.of(0, 1, 2, 3, 4)
      }),
      criterion: 'queen',
      pairCapacity: 1,
      output: {
        ...generated.spatialWeights,
        status: {overflow: overflow.view, candidateOverflow: candidateOverflow.view}
      }
    });
    producer.graph.add(contiguity);
    producerCompiled = producer.graph.compile();
    expect(contiguity.outputWeights.status?.candidateOverflow).toBe(candidateOverflow.view);
    // All four squares meet at (1,1); each row has the other three rows as neighbors.
    const complete = {
      offsets: [0, 3, 6, 9, 12],
      neighbors: [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2],
      weights: Array(12).fill(1),
      distances: []
    };
    for (const strategy of ['exact', 'tiled'] as const) {
      const rig = new WeightsRig(device);
      let compiled: ReturnType<WeightsRig['graph']['compile']> | undefined;
      try {
        submitGraph(device, producerCompiled, undefined);
        expect([
          ...(await readUint32(overflow.buffer, 1)),
          ...(await readUint32(candidateOverflow.buffer, 1))
        ]).toEqual([0, 1]);
        const weights = importPublishedWeights(rig.graph, contiguity.outputWeights);
        const statistic = rig.output('float32', 4),
          pValues = rig.output('float32', 4),
          classes = rig.output('uint32', 4);
        rig.graph.add(
          new GPUMultivariateLocalGearyPermutationTest({
            weights,
            values: rig.input(Float32Array.of(0, 1, 2, 3), 'float32', 4),
            variableCount: 1,
            permutationCount: 7,
            strategy,
            rowTileSize: 2,
            permutationTileSize: 3,
            output: {statistic: statistic.view, pseudoPValues: pValues.view, classes: classes.view}
          })
        );
        compiled = rig.graph.compile();
        submitGraph(device, compiled, undefined);
        expect((await readFloat32(statistic.buffer, 4)).every(Number.isNaN)).toBe(true);
        expect((await readFloat32(pValues.buffer, 4)).every(Number.isNaN)).toBe(true);
        expect(await readUint32(classes.buffer, 4)).toEqual(Array(4).fill(0xffffffff));
        getCoreDefaultBuffer(weights.offsets.buffer.defaultBuffer!).write(
          Uint32Array.from(complete.offsets)
        );
        getCoreDefaultBuffer(weights.neighbors.buffer.defaultBuffer!).write(
          Uint32Array.from(complete.neighbors)
        );
        getCoreDefaultBuffer(weights.weights.buffer.defaultBuffer!).write(
          Float32Array.from(complete.weights)
        );
        candidateOverflow.buffer.write(Uint32Array.of(0));
        submitGraph(device, compiled, undefined);
        // Population variance of [0,1,2,3] is 5/4; mean squared differences are [14,6,6,14]/3.
        expectFiniteValues(await readFloat32(statistic.buffer, 4), [
          56 / 15,
          8 / 5,
          8 / 5,
          56 / 15
        ]);
        expect(
          (await readFloat32(pValues.buffer, 4)).every(
            value => Number.isFinite(value) && value > 0 && value <= 1
          )
        ).toBe(true);
        expect(await readUint32(classes.buffer, 4)).toEqual([0, 0, 0, 0]);
      } finally {
        compiled?.destroy();
        rig.destroy();
      }
    }
  } finally {
    producerCompiled?.destroy();
    producer.destroy();
  }
});

it('SSR-125 island attachment preserves provenance and recovers literal three-row adjacency', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  let compiled: ReturnType<typeof rig.graph.compile> | undefined;
  try {
    // Representatives lie at x=0,1,4. Rows 0<->1 are linked; genuine island 2 chooses row 1.
    const weights = rig.uploadWeights({
      offsets: [0, 1, 2, 2],
      neighbors: [1, 0],
      weights: [1, 1],
      distances: []
    });
    const loss = rig.input(Uint32Array.of(0), 'uint32', 1);
    weights.status = {candidateOverflow: loss};
    const result = rig.weightsOutput(3, 6);
    const overflow = rig.output('uint32', 1),
      candidateOverflow = rig.output('uint32', 1);
    const attachment = rig.output('uint32', 3);
    const attach = new GPUIslandAttachment({
      weights,
      positions: rig.input(Float32Array.of(0, 0, 1, 0, 4, 0), 'float32x2', 3),
      reciprocal: false,
      nearestSearch: 'scan',
      output: {
        ...result.spatialWeights,
        attachment: attachment.view,
        added: rig.output('uint32', 6).view,
        status: {overflow: overflow.view, candidateOverflow: candidateOverflow.view}
      }
    });
    const z = rig.output('float32', 3),
      quadrants = rig.output('uint32', 3);
    rig.graph.add(attach);
    rig.graph.add(createIslandMoran(rig, attach.outputWeights, z.view, quadrants.view));
    compiled = rig.graph.compile();
    for (const incomplete of [false, true, false]) {
      // Truncation empties row 0, so attachment can invent a plausible replacement 0->1.
      rig
        .bufferOf(weights.offsets)
        .write(Uint32Array.from(incomplete ? [0, 0, 1, 1] : [0, 1, 2, 2]));
      rig.bufferOf(weights.neighbors).write(Uint32Array.from(incomplete ? [0, 0] : [1, 0]));
      rig.bufferOf(loss).write(Uint32Array.of(Number(incomplete)));
      submitGraph(device, compiled, undefined);
      expect(await readUint32(candidateOverflow.buffer, 1)).toEqual([Number(incomplete)]);
      expect(await readUint32(overflow.buffer, 1)).toEqual([0]);
      if (incomplete) {
        expect((await readFloat32(z.buffer, 3)).every(Number.isNaN)).toBe(true);
        expect(await readUint32(quadrants.buffer, 3)).toEqual([0, 0, 0]);
      } else {
        expect(await readUint32(attachment.buffer, 3)).toEqual([0xffffffff, 0xffffffff, 1]);
        expect(await result.read()).toMatchObject({
          offsets: [0, 1, 2, 3],
          neighbors: [1, 0, 1],
          weights: [1, 1, 1]
        });
        // Conditional randomization has two donors per row; each observed lag chooses the
        // donor giving the larger local I. A two-point distribution lies one sigma above its mean.
        expectFiniteValues(await readFloat32(z.buffer, 3), [1, 1, 1]);
      }
    }
  } finally {
    compiled?.destroy();
    rig.destroy();
  }
});

for (const chunked of [false, true]) {
  it(`SSR-113 lag reports both causes and restores hand-worked lag, chunked=${chunked}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new WeightsRig(device);
    let compiled: ReturnType<typeof rig.graph.compile> | undefined;
    try {
      const overflow = rig.input(Uint32Array.of(0), 'uint32', 1),
        candidateOverflow = rig.input(Uint32Array.of(0), 'uint32', 1);
      const weights = chunked
        ? {
            rowCount: 3,
            status: {overflow},
            chunks: [
              {
                rowStart: 0,
                rowCount: 1,
                weights: rig.uploadWeights({
                  offsets: [0, 1],
                  neighbors: [1],
                  weights: [1],
                  distances: []
                })
              },
              {
                rowStart: 1,
                rowCount: 2,
                weights: {
                  ...rig.uploadWeights({
                    offsets: [0, 2, 3],
                    neighbors: [0, 2, 1],
                    weights: [1, 1, 1],
                    distances: []
                  }),
                  status: {candidateOverflow}
                }
              }
            ]
          }
        : {
            ...rig.uploadWeights({
              offsets: [0, 1, 3, 4],
              neighbors: [1, 0, 2, 1],
              weights: [1, 1, 1, 1],
              distances: []
            }),
            status: {overflow, candidateOverflow}
          };
      const output = rig.output('float32', 3),
        packedStatus = rig.output('uint32', 3);
      packedStatus.buffer.write(Uint32Array.of(0, 0, 42));
      const status = {
        overflow: rig.graph.createDataView(packedStatus.view.buffer, {
          format: 'uint32',
          length: 1,
          byteOffset: 0
        }),
        candidateOverflow: rig.graph.createDataView(packedStatus.view.buffer, {
          format: 'uint32',
          length: 1,
          byteOffset: 4
        }),
        count: rig.graph.createDataView(packedStatus.view.buffer, {
          format: 'uint32',
          length: 1,
          byteOffset: 8
        })
      };
      rig.graph.add(
        new GPUSpatialLag({
          weights,
          values: rig.input(Float32Array.of(10, 20, 30), 'float32', 3),
          output: output.view,
          status
        })
      );
      compiled = rig.graph.compile();
      for (const flags of [
        [1, 0],
        [0, 1],
        [0, 0]
      ]) {
        rig.bufferOf(overflow).write(Uint32Array.of(flags[0]));
        rig.bufferOf(candidateOverflow).write(Uint32Array.of(flags[1]));
        submitGraph(device, compiled, undefined);
        expect(await readUint32(packedStatus.buffer, 3)).toEqual([...flags, 42]);
        const values = await readFloat32(output.buffer, 3);
        if (flags[0] || flags[1]) expect(values.every(Number.isNaN)).toBe(true);
        else expectFiniteValues(values, [20, 40, 20]); // 20; 10+30; 20.
      }
    } finally {
      compiled?.destroy();
      rig.destroy();
    }
  });

  it(`SSR-125 spatial EB rates recover the literal pooled rate on one graph, chunked=${chunked}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new WeightsRig(device);
    let compiled: ReturnType<typeof rig.graph.compile> | undefined;
    try {
      const loss = rig.input(Uint32Array.of(0), 'uint32', 1);
      const weights = chunked
        ? {
            rowCount: 3,
            chunks: [
              {
                rowStart: 0,
                rowCount: 1,
                weights: rig.uploadWeights({
                  offsets: [0, 1],
                  neighbors: [1],
                  weights: [1],
                  distances: []
                })
              },
              {
                rowStart: 1,
                rowCount: 2,
                weights: {
                  ...rig.uploadWeights({
                    offsets: [0, 1, 1],
                    neighbors: [0],
                    weights: [1],
                    distances: []
                  }),
                  status: {candidateOverflow: loss}
                }
              }
            ]
          }
        : {
            ...rig.uploadWeights({
              offsets: [0, 1, 2, 2],
              neighbors: [1, 0],
              weights: [1, 1],
              distances: []
            }),
            status: {candidateOverflow: loss}
          };
      const spatialRates = rig.output('float32', 3),
        smoothedRates = rig.output('float32', 3),
        priorMeans = rig.output('float32', 3),
        priorVariances = rig.output('float32', 3);
      rig.graph.add(
        new GPUSpatialEmpiricalBayesRates({
          weights,
          events: rig.input(Float32Array.of(1, 3, 4), 'float32', 3),
          populations: rig.input(Float32Array.of(10, 10, 10), 'float32', 3),
          output: {
            spatialRates: spatialRates.view,
            smoothedRates: smoothedRates.view,
            priorMeans: priorMeans.view,
            priorVariances: priorVariances.view
          }
        })
      );
      compiled = rig.graph.compile();
      for (const flag of [0, 1, 0]) {
        rig.bufferOf(loss).write(Uint32Array.of(flag));
        submitGraph(device, compiled, undefined);
        for (const result of [spatialRates, smoothedRates, priorMeans, priorVariances]) {
          const values = await readFloat32(result.buffer, 3);
          if (flag) expect(values.every(Number.isNaN)).toBe(true);
          // Neighborhoods {0,1},{0,1},{2}: pooled E/B=4/20,4/20,4/10.
          // For the pair s²=.01 and m/(B/n)=.02, so a clamps to zero; island variance is zero.
          else expectFiniteValues(values, result === priorVariances ? [0, 0, 0] : [0.2, 0.2, 0.4]);
        }
      }
    } finally {
      compiled?.destroy();
      rig.destroy();
    }
  });
}

it('Tests §5 contiguity profile pins the chain-plus-island six-word layout and independent losses', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const capacities of [
    {capacity: 4, pairCapacity: 64},
    {capacity: 1, pairCapacity: 64},
    {capacity: 4, pairCapacity: 1}
  ]) {
    const rig = new WeightsRig(device);
    try {
      const output = rig.weightsOutput(4, capacities.capacity),
        overflow = rig.output('uint32', 1),
        candidateOverflow = rig.output('uint32', 1),
        requiredCount = rig.output('uint32', 1);
      const cardinality = rig.output('uint32', 4),
        counts = rig.output('uint32', 6),
        median = rig.output('float32', 1),
        validity = rig.output('uint32', 1);
      // Unit squares at x=0,1,2,10: undirected chain 0--1--2 and disconnected row 3.
      rig.run(
        new GPUAnalysisContiguityStatistics({
          criterion: 'rook',
          pairCapacity: capacities.pairCapacity,
          polygons: rig.polygons({
            positions: Float32Array.from(
              [0, 1, 2, 10].flatMap(left => [left, 0, left + 1, 0, left + 1, 1, left, 1])
            ),
            ringOffsets: Uint32Array.of(0, 4, 8, 12, 16),
            polygonOffsets: Uint32Array.of(0, 1, 2, 3, 4),
            featureOffsets: Uint32Array.of(0, 1, 2, 3, 4)
          }),
          output: {
            weights: {
              ...output.spatialWeights,
              status: {
                overflow: overflow.view,
                candidateOverflow: candidateOverflow.view,
                requiredCount: requiredCount.view
              }
            },
            cardinality: cardinality.view,
            counts: counts.view,
            median: median.view,
            validity: validity.view
          }
        })
      );
      const exact = capacities.capacity === 4 && capacities.pairCapacity === 64;
      expect([
        ...(await readUint32(overflow.buffer, 1)),
        ...(await readUint32(candidateOverflow.buffer, 1))
      ]).toEqual(
        capacities.pairCapacity === 1 ? [0, 1] : capacities.capacity === 1 ? [1, 0] : [0, 0]
      );
      expect(await readUint32(validity.buffer, 1)).toEqual([Number(exact)]);
      expect(await readUint32(requiredCount.buffer, 1)).toEqual([
        capacities.pairCapacity === 1 ? 1 : 4
      ]);
      // 4 directed slots, 0 asymmetric, 1 isolate, min degree 0, max degree 2, 1 empty row.
      expect(await readUint32(counts.buffer, 6)).toEqual(
        exact ? [4, 0, 1, 0, 2, 1] : Array(6).fill(0xffffffff)
      );
      expect(await readUint32(cardinality.buffer, 4)).toEqual(
        exact ? [1, 2, 1, 0] : Array(4).fill(0xffffffff)
      );
      const middle = (await readFloat32(median.buffer, 1))[0];
      if (exact)
        expectFiniteValues([middle], [1]); // sorted [0,1,1,2], middle average=1.
      else expect(middle).toBeNaN();
    } finally {
      rig.destroy();
    }
  }
});
