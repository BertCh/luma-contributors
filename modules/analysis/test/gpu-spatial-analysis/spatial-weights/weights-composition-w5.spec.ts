// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {WeightsRig} from './spatial-weights-harness';
import {readUint32, readFloat32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GPUSpatialWeightsTransform} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-transform';
import {GPUChunkedSpatialWeightsTransform} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights-transform';
import {GPUSpatialWeightsAlgebra} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-algebra';
import {GPUChunkedSpatialWeightsAlgebra} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights-algebra';
import {GPUSpatialWeightsTranspose} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-transpose';
import {GPUChunkedSpatialWeightsTranspose} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights-transpose';
import {GPUSpatialWeightsValidation} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-validation';
import type {GPUSpatialWeights} from '../../../src/gpu-spatial-analysis/spatial-weights/spatial-weights';

for (const route of [
  'whole-transform',
  'chunk-transform',
  'whole-transpose',
  'packed-transpose',
  'chunk-transpose'
] as const)
  for (const flags of [false, true])
    for (const child of route.startsWith('whole') ? [false] : [false, true])
      it(`CPL-03: ${route} to consumer recovers root/child=${child}, flags=${flags}`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const rig = new WeightsRig(device);
        try {
          const complete = rig.input(Uint32Array.of(1), 'uint32', 1);
          const status = {
            complete,
            ...(flags
              ? {
                  overflow: rig.input(Uint32Array.of(0), 'uint32', 1),
                  candidateOverflow: rig.input(Uint32Array.of(0), 'uint32', 1)
                }
              : {})
          };
          const base = rig.uploadWeights({
            offsets: [0, 1, 2],
            neighbors: [1, 0],
            weights: [2, 4],
            distances: []
          });
          const whole = {...base, status};
          const chunks = {
            rowCount: 2,
            ...(child ? {} : {status}),
            chunks: [{rowStart: 0, rowCount: 2, weights: {...base, ...(child ? {status} : {})}}]
          };
          const destination = (capacity = 2) => ({
            ...rig.weightsOutput(2, capacity).spatialWeights,
            status: {
              overflow: rig.output('uint32', 1).view,
              candidateOverflow: rig.output('uint32', 1).view,
              requiredCount: rig.output('uint32', 1).view
            }
          });
          const transposeDestination = destination();
          const producer =
            route === 'whole-transform'
              ? new GPUSpatialWeightsTransform({
                  operation: 'binary',
                  weights: whole,
                  output: rig.output('float32', 2).view
                })
              : route === 'chunk-transform'
                ? new GPUChunkedSpatialWeightsTransform({
                    operation: 'binary',
                    weights: chunks,
                    output: [rig.output('float32', 2).view]
                  })
                : route === 'whole-transpose'
                  ? new GPUSpatialWeightsTranspose({
                      weights: whole,
                      output: rig.weightsOutput(2, 2).spatialWeights
                    })
                  : route === 'packed-transpose'
                    ? new GPUChunkedSpatialWeightsTranspose({
                        weights: chunks,
                        output: rig.weightsOutput(2, 2).spatialWeights
                      })
                    : new GPUChunkedSpatialWeightsTranspose({
                        weights: chunks,
                        output: {
                          chunks: [
                            {
                              rowStart: 0,
                              rowCount: 2,
                              weights: {
                                ...transposeDestination,
                                status: {
                                  ...transposeDestination.status,
                                  count: rig.output('uint32', 1).view
                                }
                              }
                            }
                          ]
                        }
                      });
          const weights = producer.outputWeights;
          const consumer =
            'chunks' in weights
              ? new GPUChunkedSpatialWeightsAlgebra({
                  operation: 'selfWeight',
                  selfWeight: 1,
                  weights,
                  output: {chunks: [{rowStart: 0, rowCount: 2, weights: destination(4)}]}
                })
              : new GPUSpatialWeightsAlgebra({
                  operation: 'selfWeight',
                  selfWeight: 1,
                  weights,
                  output: destination(4)
                });
          const observed = rig.output('uint32', 1);
          rig.graph.add(producer);
          rig.graph.add(consumer);
          rig.graph.add({
            getCommandNodes: graph => [
              createWGSLKernelNode(graph, {
                id: 'w5-read-loss',
                operation: 'Review',
                invocationCount: 1,
                bindings: [
                  {
                    name: 'loss',
                    view: consumer.outputWeights.status!.candidateOverflow!,
                    type: 'u32',
                    access: 'read'
                  },
                  {name: 'result', view: observed.view, type: 'u32', access: 'read_write'}
                ],
                body: 'result[resultOffset] = loss[lossOffset];'
              })
            ]
          });
          const compiled = rig.graph.compile();
          try {
            for (const value of [1, 0, 1]) {
              rig.bufferOf(complete).write(Uint32Array.of(value));
              submitGraph(device, compiled, undefined);
              // Positive certificate zero is loss even with both negative flags zero.
              expect(await readUint32(observed.buffer, 1)).toEqual([1 - value]);
            }
          } finally {
            compiled.destroy();
          }
        } finally {
          rig.destroy();
        }
      });

it.each([
  'parent',
  'child',
  'undefined',
  'override'
] as const)('W5-03: binary to adaptive preserves defined provenance (%s)', async kind => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const information = {rowCount: 2, transform: 'raw' as const, symmetric: 'unknown' as const};
    const base = rig.uploadWeights({
      offsets: [0, 1, 2],
      neighbors: [1, 0],
      weights: [1, 1],
      distances: [7, 11]
    });
    const weights: GPUSpatialWeights = {
      ...base,
      ...(kind === 'parent'
        ? {}
        : {
            information: {
              ...information,
              provenance: {
                source: 'composition-fixture',
                criterion: kind === 'undefined' ? undefined : 'knn'
              }
            }
          })
    };
    const binary = new GPUChunkedSpatialWeightsTransform({
      operation: 'binary',
      weights: {
        rowCount: 2,
        ...(kind === 'child'
          ? {}
          : {
              information: {
                ...information,
                provenance: {
                  source: 'composition-fixture',
                  criterion: kind === 'override' ? 'distance' : 'knn'
                }
              }
            }),
        chunks: [{rowStart: 0, rowCount: 2, weights}]
      }
    });
    const output = rig.output('float32', 2);
    const adaptive = new GPUChunkedSpatialWeightsTransform({
      operation: 'kernel',
      kernel: 'uniform',
      bandwidth: 'adaptive',
      weights: binary.outputWeights,
      output: [output.view]
    });
    rig.run(binary, adaptive);
    // Sole neighbor defines h, so K_uniform(d/h)=1/2 at each row.
    expect(await readFloat32(output.buffer, 2)).toEqual([0.5, 0.5]);
  } finally {
    rig.destroy();
  }
});

it.each([
  false,
  true
])('W5-05: validation keeps disjoint published counts intact (chunked=%s)', async chunked => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const backing = rig.input(new Uint32Array(66).fill(1), 'uint32', 66);
    const neighborCounts = rig.graph.createDataView(backing.buffer, {format: 'uint32', length: 2});
    const flags = rig.graph.createDataView(backing.buffer, {
      format: 'uint32',
      byteOffset: 256,
      length: 2
    });
    const base = {
      ...rig.uploadWeights({offsets: [0, 1, 2], neighbors: [1, 0], weights: [1, 1], distances: []}),
      neighborCounts
    };
    const weights = chunked
      ? {rowCount: 2, chunks: [{rowStart: 0, rowCount: 2, weights: base}]}
      : base;
    const invalidCount = rig.output('uint32', 1);
    expect(
      () =>
        new GPUSpatialWeightsValidation({
          weights,
          output: {flags: neighborCounts, status: {invalidCount: invalidCount.view}}
        })
    ).toThrow(/overlap|share|buffer/);
    rig.run(
      new GPUSpatialWeightsValidation({
        weights,
        output: {flags, status: {invalidCount: invalidCount.view}}
      })
    );
    const actual = await readUint32(rig.bufferOf(backing), 66);
    expect(actual.slice(0, 2)).toEqual([1, 1]);
    expect(actual.slice(64)).toEqual([0, 0]);
    expect(await readUint32(invalidCount.buffer, 1)).toEqual([0]);
  } finally {
    rig.destroy();
  }
});

it('W5-04: chunked symmetrization rejects neighbors and destination on one physical buffer', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const backing = rig.input(new Uint32Array(66), 'uint32', 66);
    const neighbors = rig.graph.createDataView(backing.buffer, {format: 'uint32', length: 2});
    const output = rig.graph.createDataView(backing.buffer, {
      format: 'float32',
      byteOffset: 256,
      length: 2
    });
    const weights = {
      ...rig.uploadWeights({offsets: [0, 1, 2], neighbors: [1, 0], weights: [2, 4], distances: []}),
      neighbors
    };
    expect(
      () =>
        new GPUChunkedSpatialWeightsTransform({
          operation: 'average-reciprocal',
          weights: {rowCount: 2, chunks: [{rowStart: 0, rowCount: 2, weights}]},
          output: [output]
        })
    ).toThrow(/buffer|share/);
  } finally {
    rig.destroy();
  }
});
