// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUChunkedSpatialWeightsTranspose} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights-transpose';
import {WeightsRig} from './spatial-weights-harness';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPUChunkedSpatialWeightsAlgebra} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights-algebra';
import {readUint32} from '../../utils/gpu-contributor-test-utils';
import {CPUSpatialWeightsAlgebraOperation} from '../../../src/gpu-spatial-analysis/spatial-weights/cpu-spatial-weights-algebra-operation';
import {GPUSpatialWeightsAlgebraOperation} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-algebra-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {GPUIslandAttachment} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-island-attachment';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GPUChunkedSpatialWeightsTransform} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights-transform';
import {readFloat32} from '../../utils/gpu-contributor-test-utils';
import {GPUSpatialLag} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-lag';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {GPUSpatialWeightsAlgebra} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-algebra';
import {GPUSpatialWeightsTransform} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-transform';
import {CPUSpatialWeightsValidationOperation} from '../../../src/gpu-spatial-analysis/spatial-weights/cpu-spatial-weights-validation-operation';
import {GPUSpatialWeightsValidationOperation} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-validation-operation';
import {createReviewCarryNodes} from './audit-weights-a-review.test-utils';
import {addContiguityOrderCorrelogramRecipe} from '../../../src/gpu-spatial-analysis/recipes/contiguity-order-correlogram-recipe';
import {GPUSpatialWeightsTranspose} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-transpose';

it('review 2 R14: global and symmetrization nodes write the captured published views', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const operation of ['double', 'variance', 'average-reciprocal'] as const) {
    const rig = new WeightsRig(device);
    try {
      const chunks = [0, 1].map(rowStart => ({
        rowStart,
        rowCount: 1,
        weights: rig.uploadWeights({
          offsets: [0, 1],
          neighbors: [1 - rowStart],
          weights: [rowStart ? 3 : 2],
          distances: []
        })
      }));
      const original = [rig.output('float32', 1), rig.output('float32', 1)];
      const output = original.map(port => port.view);
      for (const port of original) port.buffer.write(Float32Array.of(-999));
      const replacement = rig.input(Float32Array.of(99), 'float32', 1);
      const transform = new GPUChunkedSpatialWeightsTransform({
        operation,
        weights: {rowCount: 2, chunks},
        output
      });
      output[0] = replacement;
      for (const [index, chunk] of transform.outputWeights.chunks.entries())
        expect(chunk.weights.weights).toBe(original[index].view);
      const published = rig.output('float32', 2);
      rig.run(transform, {
        getCommandNodes: graph => [
          createWGSLKernelNode(graph, {
            id: 'review2-transform-published',
            operation: 'Review',
            invocationCount: 1,
            bindings: [
              {
                name: 'first',
                view: transform.outputWeights.chunks[0].weights.weights,
                type: 'f32',
                access: 'read'
              },
              {
                name: 'second',
                view: transform.outputWeights.chunks[1].weights.weights,
                type: 'f32',
                access: 'read'
              },
              {name: 'result', view: published.view, type: 'f32', access: 'read_write'}
            ],
            body: 'result[resultOffset] = first[firstOffset]; result[resultOffset + 1u] = second[secondOffset];'
          })
        ]
      });
      const actual = await readFloat32(published.buffer, 2);
      // Double: [2,3]/5; variance: s=[1,1], n/Q=1; symmetric mean: (2+3)/2.
      const expected =
        operation === 'double' ? [0.4, 0.6] : operation === 'variance' ? [1, 1] : [2.5, 2.5];
      for (const [index, value] of actual.entries()) {
        expect(Number.isFinite(value)).toBe(true);
        expect(Math.abs(value - expected[index])).toBeLessThanOrEqual(1e-6);
      }
      expect(await readFloat32(rig.bufferOf(replacement), 1)).toEqual([99]);
    } finally {
      rig.destroy();
    }
  }
});

it('review 2 R13: inherited validator names and CSR ports produce independent literals', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  class Names {
    get offsets() {
      return 'o';
    }
    get neighbors() {
      return 'n';
    }
    get weights() {
      return 'w';
    }
  }
  for (const Operation of [
    GPUSpatialWeightsValidationOperation,
    CPUSpatialWeightsValidationOperation
  ]) {
    const result = await runOperationGraph(device, {
      inputs: {
        o: {format: 'uint32', values: Uint32Array.of(0, 1, 2)},
        n: {format: 'uint32', values: Uint32Array.of(0, 0)},
        w: {format: 'float32', values: Float32Array.of(1, 1)}
      },
      operations: [
        new Operation({
          rowCount: 2,
          weights: new Names(),
          output: {flags: 'flags', status: {invalidCount: 'invalid'}}
        })
      ],
      outputs: ['flags', 'invalid']
    });
    expect(Array.from(result['flags'])).toEqual([8, 0]);
    expect(Array.from(result['invalid'])).toEqual([1]);
  }
  const rig = new WeightsRig(device);
  try {
    const fields = rig.uploadWeights({
      offsets: [0, 1, 2],
      neighbors: [1, 0],
      weights: [2, 3],
      distances: []
    });
    class CSR {
      get offsets() {
        return fields.offsets;
      }
      get neighbors() {
        return fields.neighbors;
      }
      get weights() {
        return fields.weights;
      }
    }
    const output = rig.output('float32', 2);
    rig.run(
      new GPUChunkedSpatialWeightsTransform({
        operation: 'row',
        weights: {rowCount: 2, chunks: [{rowStart: 0, rowCount: 2, weights: new CSR()}]},
        output: [output.view]
      })
    );
    expect(await readFloat32(output.buffer, 2)).toEqual([1, 1]);
  } finally {
    rig.destroy();
  }
});

it('review 2 R13: inherited overflow survives packed transpose and suppresses a lag consumer', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const chunked of [false, true]) {
    const rig = new WeightsRig(device);
    try {
      const overflow = rig.input(Uint32Array.of(1), 'uint32', 1);
      class Status {
        get overflow() {
          return overflow;
        }
      }
      const input = rig.uploadWeights({
        offsets: [0, 1, 2],
        neighbors: [1, 0],
        weights: [2, 4],
        distances: []
      });
      const output = rig.weightsOutput(2, 2);
      const operation = chunked
        ? new GPUChunkedSpatialWeightsTranspose({
            weights: {
              rowCount: 2,
              status: new Status(),
              chunks: [{rowStart: 0, rowCount: 2, weights: input}]
            },
            output: output.spatialWeights
          })
        : new GPUSpatialWeightsTranspose({
            weights: {...input, status: new Status()},
            output: output.spatialWeights
          });
      expect(operation.outputWeights.status?.overflow).toBe(overflow);
      const lag = rig.output('float32', 2);
      rig.run(
        operation,
        new GPUSpatialLag({
          weights: operation.outputWeights,
          values: rig.input(Float32Array.of(10, 20), 'float32', 2),
          output: lag.view
        })
      );
      expect(await output.read()).toEqual({
        offsets: [0, 1, 2],
        neighbors: [1, 0],
        weights: [4, 2],
        distances: []
      });
      expect(await readUint32(rig.bufferOf(overflow), 1)).toEqual([1]);
      for (const value of await readFloat32(lag.buffer, 2)) expect(Number.isNaN(value)).toBe(true);
    } finally {
      rig.destroy();
    }
  }
});

it('review 2 R10: island attachment ignores replacement port identities while reading original contents', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  const other = new WeightsRig(device);
  try {
    const inputLoss = rig.input(Uint32Array.of(1), 'uint32', 1);
    const weights = {
      ...rig.uploadWeights({offsets: [0, 0, 1], neighbors: [0], weights: [2], distances: []}),
      status: {overflow: inputLoss}
    };
    const output = rig.weightsOutput(2, 2);
    const count = rig.output('uint32', 1);
    const replacement = rig.output('uint32', 1);
    const required = rig.output('uint32', 1);
    const overflow = rig.output('uint32', 1);
    const attachment = rig.output('uint32', 2);
    const added = rig.output('uint32', 2);
    count.buffer.write(Uint32Array.of(99));
    replacement.buffer.write(Uint32Array.of(99));
    const destination = {
      ...output.spatialWeights,
      attachment: attachment.view,
      added: added.view,
      status: {count: count.view, requiredCount: required.view, overflow: overflow.view}
    };
    const operation = new GPUIslandAttachment({
      nearestSearch: 'scan',
      reciprocal: false,
      weights,
      positions: rig.input(Float32Array.of(0, 0, 1, 0), 'float32x2', 2),
      output: destination
    });
    weights.status = {overflow: other.input(Uint32Array.of(0), 'uint32', 1)};
    destination.status.count = replacement.view;
    expect(operation.outputWeights.status!.count).toBe(count.view);
    const captured = rig.output('uint32', 2);
    rig.graph.add(operation);
    rig.graph.add({
      getCommandNodes: graph => [
        createWGSLKernelNode(graph, {
          id: 'review2-island-published',
          operation: 'Review',
          invocationCount: 1,
          bindings: [
            {
              name: 'count',
              view: operation.outputWeights.status!.count!,
              type: 'u32',
              access: 'read'
            },
            {
              name: 'candidate',
              view: operation.outputWeights.status!.candidateOverflow!,
              type: 'u32',
              access: 'read'
            },
            {name: 'result', view: captured.view, type: 'u32', access: 'read_write'}
          ],
          body: 'result[resultOffset] = count[countOffset]; result[resultOffset + 1u] = candidate[candidateOffset];'
        })
      ]
    });
    const compiled = rig.graph.compile();
    try {
      for (const flag of [1, 0]) {
        rig.bufferOf(inputLoss).write(Uint32Array.of(flag));
        submitGraph(device, compiled, undefined);
        expect(await readUint32(captured.buffer, 2)).toEqual([2, flag]);
        expect(await readUint32(replacement.buffer, 1)).toEqual([99]);
        expect(await output.read()).toEqual({
          offsets: [0, 1, 2],
          neighbors: [1, 0],
          weights: [1, 2],
          distances: []
        });
        expect(await readUint32(attachment.buffer, 2)).toEqual([1, 0xffffffff]);
        expect(await readUint32(added.buffer, 2)).toEqual([1, 0]);
        expect(await readUint32(required.buffer, 1)).toEqual([2]);
        expect(await readUint32(overflow.buffer, 1)).toEqual([0]);
      }
    } finally {
      compiled.destroy();
    }
  } finally {
    rig.destroy();
    other.destroy();
  }
});

it('review 2 R9: kernels may share the unused source buffer and row chunks may share disjoint floats', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const layout of ['whole kernel', 'chunked kernel', 'chunked row'] as const) {
    const rig = new WeightsRig(device);
    try {
      const values = layout === 'chunked row' ? [2, 3] : [2, 3, 99, 99];
      const backing = rig.input(Float32Array.from(values), 'float32', values.length);
      const create = (byteOffset: number, length: number) =>
        rig.graph.createDataView(backing.buffer, {format: 'float32', byteOffset, length});
      if (layout === 'chunked row') {
        const chunks = [0, 1].map(rowStart => ({
          rowStart,
          rowCount: 1,
          weights: {
            ...rig.uploadWeights({
              offsets: [0, 1],
              neighbors: [1 - rowStart],
              weights: [values[rowStart]],
              distances: []
            }),
            weights: create(rowStart * 4, 1)
          }
        }));
        rig.run(
          new GPUChunkedSpatialWeightsTransform({operation: 'row', weights: {rowCount: 2, chunks}})
        );
        expect(await readFloat32(rig.bufferOf(backing), 2)).toEqual([1, 1]);
      } else {
        const weights = {
          ...rig.uploadWeights({
            offsets: [0, 1, 2],
            neighbors: [1, 0],
            weights: [2, 3],
            distances: [7, 7]
          }),
          weights: create(0, 2)
        };
        const output = create(8, 2);
        rig.run(
          layout === 'whole kernel'
            ? new GPUSpatialWeightsTransform({
                operation: 'kernel',
                kernel: 'uniform',
                bandwidth: 8,
                weights,
                output
              })
            : new GPUChunkedSpatialWeightsTransform({
                operation: 'kernel',
                kernel: 'uniform',
                bandwidth: 8,
                weights: {rowCount: 2, chunks: [{rowStart: 0, rowCount: 2, weights}]},
                output: [output]
              })
        );
        expect(await readFloat32(rig.bufferOf(backing), 4)).toEqual([2, 3, 0.5, 0.5]);
      }
    } finally {
      rig.destroy();
    }
  }
});

it('review 2 R7: intersection derives matrix guarantees from chunk-only exclusions in both operand orders', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const reverse of [false, true])
    for (const allExcluded of [false, true]) {
      const rig = new WeightsRig(device);
      try {
        const diagonal = {
          rowCount: 2,
          information: {
            rowCount: 2,
            transform: 'raw' as const,
            symmetric: 'unknown' as const,
            hasSelfWeights: true,
            provenance: {source: 'review'}
          },
          chunks: [0, 1].map(rowStart => ({
            rowStart,
            rowCount: 1,
            weights: rig.uploadWeights({
              offsets: [0, 2],
              neighbors: [0, 1],
              weights: rowStart ? [4, 9] : [9, 2],
              distances: []
            })
          }))
        };
        const excluded = {
          rowCount: 2,
          chunks: [0, 1].map(rowStart => ({
            rowStart,
            rowCount: 1,
            weights: {
              ...rig.uploadWeights({
                offsets: [0, 1],
                neighbors: [1 - rowStart],
                weights: [rowStart ? 5 : 3],
                distances: []
              }),
              information: {
                rowCount: 1,
                transform: 'raw' as const,
                symmetric: 'unknown' as const,
                hasSelfWeights: rowStart === 0 || allExcluded ? false : undefined,
                provenance: {source: 'review'}
              }
            }
          }))
        };
        const outputs = [rig.weightsOutput(1, 1), rig.weightsOutput(1, 1)];
        const overflow = [rig.output('uint32', 1), rig.output('uint32', 1)];
        const required = [rig.output('uint32', 1), rig.output('uint32', 1)];
        const operation = new GPUChunkedSpatialWeightsAlgebra({
          operation: 'intersection',
          left: reverse ? excluded : diagonal,
          right: reverse ? diagonal : excluded,
          output: {
            chunks: outputs.map((output, rowStart) => ({
              rowStart,
              rowCount: 1,
              weights: {
                ...output.spatialWeights,
                status: {overflow: overflow[rowStart].view, requiredCount: required[rowStart].view}
              }
            }))
          }
        });
        expect(
          operation.outputWeights.chunks.map(chunk => chunk.weights.information?.hasSelfWeights)
        ).toEqual([false, allExcluded ? false : true]);
        expect(operation.outputWeights.information?.hasSelfWeights).toBe(!allExcluded);
        rig.run(operation);
        for (const row of [0, 1]) {
          expect(await outputs[row].read()).toEqual({
            offsets: [0, 1],
            neighbors: [1 - row],
            weights: [reverse ? (row ? 5 : 3) : row ? 4 : 2],
            distances: []
          });
          expect(await readUint32(overflow[row].buffer, 1)).toEqual([0]);
          expect(await readUint32(required[row].buffer, 1)).toEqual([1]);
        }
      } finally {
        rig.destroy();
      }
    }
});

it('review 2 R5: binary then adaptive uniform retains chunk-only kNN metadata', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const weights = {
      rowCount: 2,
      chunks: [0, 1].map(rowStart => ({
        rowStart,
        rowCount: 1,
        weights: {
          ...rig.uploadWeights({
            offsets: [0, 1],
            neighbors: [1 - rowStart],
            weights: [rowStart ? 4 : 2],
            distances: [7]
          }),
          information: {
            rowCount: 1,
            transform: 'raw' as const,
            symmetric: true as const,
            provenance: {source: 'review', criterion: 'knn', k: 1}
          }
        }
      }))
    };
    const binary = new GPUChunkedSpatialWeightsTransform({
      id: 'review2-binary',
      operation: 'binary',
      weights
    });
    const output = [rig.output('float32', 1), rig.output('float32', 1)];
    const kernel = new GPUChunkedSpatialWeightsTransform({
      id: 'review2-kernel',
      operation: 'kernel',
      kernel: 'uniform',
      bandwidth: 'adaptive',
      weights: binary.outputWeights,
      output: output.map(port => port.view)
    });
    rig.run(binary, kernel);
    for (const port of output) expect(await readFloat32(port.buffer, 1)).toEqual([0.5]);
    for (const chunk of binary.outputWeights.chunks) {
      expect(await readFloat32(rig.bufferOf(chunk.weights.weights), 1)).toEqual([1]);
      expect(chunk.weights.information?.provenance?.criterion).toBe('knn');
      expect(chunk.weights.information?.rowCount).toBe(1);
      expect(chunk.weights.information?.transform).toBe('binary');
    }
  } finally {
    rig.destroy();
  }
});

it('review 2 R4: truncated transpose publishes local count one without an inherited matrix count two', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const status = {
      count: rig.input(Uint32Array.of(2), 'uint32', 1),
      requiredCount: rig.input(Uint32Array.of(2), 'uint32', 1),
      overflow: rig.input(Uint32Array.of(0), 'uint32', 1),
      candidateOverflow: rig.input(Uint32Array.of(1), 'uint32', 1)
    };
    const output = rig.weightsOutput(2, 1);
    const count = rig.output('uint32', 1);
    const requiredCount = rig.output('uint32', 1);
    const overflow = rig.output('uint32', 1);
    const candidateOverflow = rig.output('uint32', 1);
    const operation = new GPUChunkedSpatialWeightsTranspose({
      weights: {
        rowCount: 2,
        status,
        chunks: [
          {
            rowStart: 0,
            rowCount: 2,
            weights: rig.uploadWeights({
              offsets: [0, 1, 2],
              neighbors: [1, 0],
              weights: [2, 4],
              distances: []
            })
          }
        ]
      },
      output: {
        chunks: [
          {
            rowStart: 0,
            rowCount: 2,
            weights: {
              ...output.spatialWeights,
              status: {
                count: count.view,
                requiredCount: requiredCount.view,
                overflow: overflow.view,
                candidateOverflow: candidateOverflow.view
              }
            }
          }
        ]
      }
    });
    expect(operation.outputWeights.status?.count === undefined).toBe(true);
    expect(operation.outputWeights.status?.requiredCount === undefined).toBe(true);
    const publishedLoss = rig.output('uint32', 2);
    const aggregate = operation.outputWeights.status!;
    rig.graph.add(operation);
    rig.graph.add({
      getCommandNodes: graph => [
        createWGSLKernelNode(graph, {
          id: 'w5-transpose-loss',
          operation: 'Review',
          invocationCount: 1,
          bindings: [
            {name: 'overflow', view: aggregate.overflow!, type: 'u32', access: 'read'},
            {name: 'candidate', view: aggregate.candidateOverflow!, type: 'u32', access: 'read'},
            {name: 'result', view: publishedLoss.view, type: 'u32', access: 'read_write'}
          ],
          body: 'result[resultOffset] = overflow[overflowOffset]; result[resultOffset + 1u] = candidate[candidateOffset];'
        })
      ]
    });
    const retry = rig.graph.compile();
    try {
      submitGraph(device, retry, undefined);
      expect(await readUint32(publishedLoss.buffer, 2)).toEqual([1, 1]);
      rig.bufferOf(status.candidateOverflow!).write(Uint32Array.of(0));
      submitGraph(device, retry, undefined);
      expect(await readUint32(publishedLoss.buffer, 2)).toEqual([1, 0]);
    } finally {
      retry.destroy();
    }
    expect(await output.read()).toEqual({
      offsets: [0, 1, 1],
      neighbors: [1],
      weights: [4],
      distances: []
    });
    expect(await readUint32(count.buffer, 1)).toEqual([1]);
    expect(await readUint32(requiredCount.buffer, 1)).toEqual([2]);
    expect(await readUint32(overflow.buffer, 1)).toEqual([1]);
    expect(await readUint32(candidateOverflow.buffer, 1)).toEqual([0]);
  } finally {
    rig.destroy();
  }
});

it('review 2 R3: recipe keeps higher-order loss but omits the untrusted requirement', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const table = rig.output('float32', 20);
    const incomplete = rig.output('uint32', 2);
    const result = addContiguityOrderCorrelogramRecipe(rig.graph, {
      id: 'review2-recipe',
      maximumOrder: 2,
      slotCapacity: 4,
      workCapacity: 1,
      weights: rig.uploadWeights({
        offsets: [0, 2, 3, 4],
        neighbors: [1, 2, 2, 1],
        weights: [1, 1, 1, 1],
        distances: []
      }),
      values: rig.input(Float32Array.of(1, 2, 4), 'float32', 3),
      output: {correlogram: table.view, incompleteOrders: incomplete.view}
    });
    const raw = result.contributors.find(
      contributor =>
        contributor instanceof GPUSpatialWeightsAlgebra &&
        contributor.id === 'review2-recipe-order-2-ring'
    ) as GPUSpatialWeightsAlgebra;
    const described = result.intermediates.orderWeights[1];
    expect(described.status!.requiredCount === undefined).toBe(true);
    const captured = rig.output('uint32', 2);
    rig.graph.add({
      getCommandNodes: graph => [
        createWGSLKernelNode(graph, {
          id: 'review2-recipe-capture',
          operation: 'Review',
          invocationCount: 1,
          bindings: [
            {
              name: 'required',
              view: raw.props.output.status.requiredCount!,
              type: 'u32',
              access: 'read'
            },
            {
              name: 'candidate',
              view: described.status!.candidateOverflow!,
              type: 'u32',
              access: 'read'
            },
            {name: 'result', view: captured.view, type: 'u32', access: 'read_write'}
          ],
          body: 'result[resultOffset] = required[requiredOffset]; result[resultOffset + 1u] = candidate[candidateOffset];'
        })
      ]
    });
    const compiled = rig.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      expect(await readUint32(captured.buffer, 2)).toEqual([1, 1]);
      expect(await readUint32(incomplete.buffer, 2)).toEqual([0, 1]);
      const actual = await readFloat32(table.buffer, 20);
      // n=3, S0=3, z'z=14/3, z'Wz=-2: Moran I=-3/7. Order two is incomplete.
      expect(Number.isFinite(actual[0])).toBe(true);
      expect(actual[0]).toBeCloseTo(-3 / 7, 5);
      for (const value of actual.slice(10, 18)) expect(Number.isNaN(value)).toBe(true);
    } finally {
      compiled.destroy();
    }
  } finally {
    rig.destroy();
  }
});

it('review 2 R2: cumulative geographic unit resolution rejects conflicts and retains compatible meters', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const outer = {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const;
    const operand = (right: boolean, units: 'meters' | 'kilometers') => ({
      rowCount: 2,
      spatialContext: outer,
      chunks: [
        {
          rowStart: 0,
          rowCount: 2,
          weights: {
            ...rig.uploadWeights(
              right
                ? {offsets: [0, 0, 1], neighbors: [0], weights: [4], distances: [11]}
                : {offsets: [0, 1, 1], neighbors: [1], weights: [2], distances: [7]}
            ),
            spatialContext: {...outer, units}
          }
        }
      ]
    });
    const output = rig.weightsOutput(2, 2, true);
    const overflow = rig.output('uint32', 1);
    const required = rig.output('uint32', 1);
    const construct = (units: 'meters' | 'kilometers') =>
      new GPUChunkedSpatialWeightsAlgebra({
        operation: 'union',
        left: operand(false, 'meters'),
        right: operand(true, units),
        output: {
          chunks: [
            {
              rowStart: 0,
              rowCount: 2,
              weights: {
                ...output.spatialWeights,
                status: {overflow: overflow.view, requiredCount: required.view}
              }
            }
          ]
        }
      });
    expect(() => construct('kilometers')).toThrow(/units differ/);
    const operation = construct('meters');
    expect(operation.outputWeights.spatialContext?.units).toBe('meters');
    rig.run(operation);
    expect(await output.read()).toEqual({
      offsets: [0, 1, 2],
      neighbors: [1, 0],
      weights: [2, 4],
      distances: [7, 11]
    });
    expect(await readUint32(overflow.buffer, 1)).toEqual([0]);
    expect(await readUint32(required.buffer, 1)).toEqual([2]);
  } finally {
    rig.destroy();
  }
});

it('review 2 R1: corrected transpose views retain required distances and independent status words', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const packed of [false, true]) {
    const rig = new WeightsRig(device);
    try {
      const chunks = [0, 1].map(rowStart => ({
        rowStart,
        rowCount: 1,
        weights: rig.uploadWeights({
          offsets: [0, 1],
          neighbors: [1 - rowStart],
          weights: [rowStart ? 4 : 2],
          distances: [rowStart ? 11 : 7]
        })
      }));
      const output = rig.weightsOutput(2, 2, true);
      const count = rig.output('uint32', 1);
      const required = rig.output('uint32', 1);
      const overflow = rig.output('uint32', 1);
      const candidate = rig.output('uint32', 1);
      const destination = {
        ...output.spatialWeights,
        status: {
          count: count.view,
          requiredCount: required.view,
          overflow: overflow.view,
          candidateOverflow: candidate.view
        }
      };
      const operation = packed
        ? new GPUChunkedSpatialWeightsTranspose({
            weights: {rowCount: 2, chunks},
            output: output.spatialWeights
          })
        : new GPUChunkedSpatialWeightsTranspose({
            weights: {rowCount: 2, chunks},
            output: {chunks: [{rowStart: 0, rowCount: 2, weights: destination}]}
          });
      const distances = chunks[1].weights.distances;
      chunks[1].weights.distances = undefined;
      expect(() => operation.getCommandNodes(rig.graph)).toThrow(/distances/);
      chunks[1].weights.distances = distances;
      if (!packed) {
        destination.status.candidateOverflow = count.view;
        expect(() => operation.getCommandNodes(rig.graph)).toThrow(/separate buffers|overlap/);
        Reflect.deleteProperty(destination.status, 'candidateOverflow');
        expect(() => operation.getCommandNodes(rig.graph)).toThrow(
          /ports and shapes must match construction/
        );
        destination.status.candidateOverflow = candidate.view;
      }
      rig.run(operation);
      // Hand transpose swaps the source rows, retaining each edge's measurement.
      expect(await output.read()).toEqual({
        offsets: [0, 1, 2],
        neighbors: [1, 0],
        weights: [4, 2],
        distances: [11, 7]
      });
      if (!packed) {
        expect(await readUint32(count.buffer, 1)).toEqual([2]);
        expect(await readUint32(required.buffer, 1)).toEqual([2]);
        expect(await readUint32(overflow.buffer, 1)).toEqual([0]);
        expect(await readUint32(candidate.buffer, 1)).toEqual([0]);
      }
    } finally {
      rig.destroy();
    }
  }
});

it('R12: production lattice/algebra prefix pipelines clamp both sides of a uint32 carry', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const kind of ['lattice', 'algebra'] as const) {
    const rig = new WeightsRig(device);
    try {
      const output = rig.weightsOutput(3, 4);
      const overflow = rig.output('uint32', 1);
      const requiredCount = rig.output('uint32', 1);
      rig.run({
        getCommandNodes: graph =>
          createReviewCarryNodes(graph, kind, {
            ...output.spatialWeights,
            status: {overflow: overflow.view, requiredCount: requiredCount.view}
          })
      });
      // Integer oracle: [0,2^32-2,2^32+1,2^32+8], all prefixes saturated before capacity clamp.
      expect((await output.read()).offsets).toEqual([0, 4, 4, 4]);
      expect(await readUint32(requiredCount.buffer, 1)).toEqual([0xffffffff]);
      expect(await readUint32(overflow.buffer, 1)).toEqual([1]);
    } finally {
      rig.destroy();
    }
  }
});

it('R11: both named validators retain both distinct outputs for a two-row self violation', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const Operation of [
    CPUSpatialWeightsValidationOperation,
    GPUSpatialWeightsValidationOperation
  ]) {
    const props = {
      id: 'r11-validation',
      rowCount: 2,
      weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
      output: {flags: 'flags', status: {invalidCount: 'invalid'}}
    };
    expect(
      () => new Operation({...props, output: {flags: 'dup', status: {invalidCount: 'dup'}}})
    ).toThrow(/no other port uses/);
    const result = await runOperationGraph(device, {
      inputs: {
        offsets: {format: 'uint32', values: Uint32Array.of(0, 1, 2)},
        neighbors: {format: 'uint32', values: Uint32Array.of(0, 0)},
        weights: {format: 'float32', values: Float32Array.of(1, 1)}
      },
      operations: [new Operation(props)],
      outputs: ['flags', 'invalid']
    });
    // Only row zero has a self violation (bit 8); exactly one row is invalid.
    expect(Array.from(result['flags'])).toEqual([8, 0]);
    expect(Array.from(result['invalid'])).toEqual([1]);
  }
});

it('R10: replacing caller status preserves the original loss view across submissions', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const original = rig.input(Uint32Array.of(1), 'uint32', 1);
    const weights = {
      ...rig.uploadWeights({offsets: [0, 0], neighbors: [], weights: [], distances: []}, 1),
      status: {overflow: original}
    };
    const output = rig.weightsOutput(1, 1);
    const overflow = rig.output('uint32', 1);
    const attachment = rig.output('uint32', 1);
    const added = rig.output('uint32', 1);
    added.buffer.write(Uint32Array.of(99));
    const operation = new GPUIslandAttachment({
      nearestSearch: 'scan',
      reciprocal: false,
      weights,
      positions: rig.input(Float32Array.of(0, 0), 'float32x2', 1),
      output: {
        ...output.spatialWeights,
        attachment: attachment.view,
        added: added.view,
        status: {overflow: overflow.view}
      }
    });
    weights.status = {overflow: rig.input(Uint32Array.of(0), 'uint32', 1)};
    const captured = rig.output('uint32', 1);
    rig.graph.add(operation);
    rig.graph.add({
      getCommandNodes: graph => [
        createWGSLKernelNode(graph, {
          id: 'r10-copy-status',
          operation: 'ReviewStatus',
          invocationCount: 1,
          bindings: [
            {
              name: 'source',
              view: operation.outputWeights.status!.candidateOverflow!,
              type: 'u32',
              access: 'read'
            },
            {name: 'destination', view: captured.view, type: 'u32', access: 'read_write'}
          ],
          body: 'destination[destinationOffset] = source[sourceOffset];'
        })
      ]
    });
    const compiled = rig.graph.compile();
    try {
      const flags: number[] = [];
      for (const value of [1, 0]) {
        rig.bufferOf(original).write(Uint32Array.of(value));
        submitGraph(device, compiled, undefined);
        flags.push((await readUint32(captured.buffer, 1))[0]);
        expect((await output.read()).offsets).toEqual([0, 0]);
        expect(await readUint32(overflow.buffer, 1)).toEqual([0]);
        expect(await readUint32(attachment.buffer, 1)).toEqual([0xffffffff]);
        // No retained edge writes added[0]; keep the unused-tail canary.
        expect(await readUint32(added.buffer, 1)).toEqual([99]);
      }
      expect(flags).toEqual([1, 0]);
    } finally {
      compiled.destroy();
    }
  } finally {
    rig.destroy();
  }
});

it('R9: whole/chunked exact-range in-place transforms preserve the rest of the buffer', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const chunked of [false, true]) {
    const rig = new WeightsRig(device);
    try {
      const backing = rig.input(Float32Array.of(2, 3, 99, 99), 'float32', 4);
      const source = rig.graph.createDataView(backing.buffer, {
        format: 'float32',
        byteOffset: 0,
        length: 2
      });
      const exact = rig.graph.createDataView(backing.buffer, {
        format: 'float32',
        byteOffset: 0,
        length: 2
      });
      const destination = rig.graph.createDataView(backing.buffer, {
        format: 'float32',
        byteOffset: 8,
        length: 2
      });
      const weights = {
        ...rig.uploadWeights({
          offsets: [0, 1, 2],
          neighbors: [1, 0],
          weights: [2, 3],
          distances: []
        }),
        weights: source
      };
      const construct = (output: typeof source) =>
        chunked
          ? new GPUChunkedSpatialWeightsTransform({
              operation: 'row',
              weights: {rowCount: 2, chunks: [{rowStart: 0, rowCount: 2, weights}]},
              output: [output]
            })
          : new GPUSpatialWeightsTransform({operation: 'row', weights, output});
      expect(() => construct(destination)).toThrow(/output.*same byte range/);
      rig.run(construct(exact));
      // One neighbor per row: w / sum(w) = 1. The unused backing tail is a canary.
      expect(await readFloat32(rig.bufferOf(backing), 4)).toEqual([1, 1, 99, 99]);
    } finally {
      rig.destroy();
    }
  }
});

it('R1: packed transpose preserves both distance measurements from separate chunks', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const weights = {
      rowCount: 2,
      chunks: [
        {
          rowStart: 0,
          rowCount: 1,
          weights: rig.uploadWeights({
            offsets: [0, 1],
            neighbors: [1],
            weights: [2],
            distances: [7]
          })
        },
        {
          rowStart: 1,
          rowCount: 1,
          weights: rig.uploadWeights({
            offsets: [0, 1],
            neighbors: [0],
            weights: [4],
            distances: [11]
          })
        }
      ]
    };
    const output = rig.weightsOutput(2, 2, true);
    rig.run(new GPUChunkedSpatialWeightsTranspose({weights, output: output.spatialWeights}));
    // Hand transpose: the second input row becomes output row 0, with its own distance 11.
    expect(await output.read()).toEqual({
      offsets: [0, 1, 2],
      neighbors: [1, 0],
      weights: [4, 2],
      distances: [11, 7]
    });
  } finally {
    rig.destroy();
  }
});

it('R8: whole/chunked extreme normalization follows scaled norms and nonfinite total suppression', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const operation of ['variance', 'double'] as const)
    for (const chunked of [false, true]) {
      const rig = new WeightsRig(device);
      try {
        const magnitude = operation === 'variance' ? 1e22 : 3e38;
        const expected =
          operation === 'variance'
            ? [0.8786796564403575, 0.8786796564403575, 1.2426406871192852]
            : [0, 0, 0];
        let actual: number[];
        if (chunked) {
          const chunks = [0, 1, 2].map(rowStart => ({
            rowStart,
            rowCount: 1,
            weights: rig.uploadWeights(
              rowStart === 0
                ? {
                    offsets: [0, 2],
                    neighbors: [1, 2],
                    weights: [magnitude, magnitude],
                    distances: []
                  }
                : rowStart === 1
                  ? {offsets: [0, 1], neighbors: [0], weights: [1], distances: []}
                  : {offsets: [0, 0], neighbors: [], weights: [], distances: []},
              1
            )
          }));
          const outputs = chunks.map(chunk =>
            rig.output('float32', chunk.weights.neighbors.length)
          );
          rig.run(
            new GPUChunkedSpatialWeightsTransform({
              operation,
              weights: {rowCount: 3, chunks},
              output: outputs.map(output => output.view)
            })
          );
          actual = [
            ...(await readFloat32(outputs[0].buffer, 2)),
            ...(await readFloat32(outputs[1].buffer, 1))
          ];
        } else {
          const output = rig.output('float32', 3);
          rig.run(
            new GPUSpatialWeightsTransform({
              operation,
              weights: rig.uploadWeights({
                offsets: [0, 2, 3, 3],
                neighbors: [1, 2, 0],
                weights: [magnitude, magnitude, 1],
                distances: []
              }),
              output: output.view
            })
          );
          actual = await readFloat32(output.buffer, 3);
        }
        for (const [index, value] of actual.entries()) {
          expect(Number.isFinite(value)).toBe(true);
          // Scaled f32 norms plus one global reduction; absolute error <=2e-6.
          expect(Math.abs(value - expected[index])).toBeLessThanOrEqual(2e-6);
        }
      } finally {
        rig.destroy();
      }
    }
});

it('R7: whole/chunked intersections have diagonal-free literals and metadata in both operand orders', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const chunked of [false, true])
    for (const flags of [
      [true, false],
      [false, true],
      [undefined, false],
      [false, undefined]
    ] as const) {
      const rig = new WeightsRig(device);
      try {
        const operands = flags.map(hasSelfWeights => ({
          ...rig.uploadWeights(
            hasSelfWeights === false
              ? {offsets: [0, 1, 2], neighbors: [1, 0], weights: [3, 5], distances: []}
              : {offsets: [0, 2, 3], neighbors: [0, 1, 0], weights: [9, 2, 4], distances: []}
          ),
          information: {
            rowCount: 2,
            transform: 'raw' as const,
            symmetric: 'unknown' as const,
            hasSelfWeights
          }
        }));
        const destination = rig.weightsOutput(2, 2);
        const overflow = rig.output('uint32', 1);
        const required = rig.output('uint32', 1);
        const candidate = rig.output('uint32', 1);
        const output = {
          ...destination.spatialWeights,
          status: {
            overflow: overflow.view,
            requiredCount: required.view,
            candidateOverflow: candidate.view
          }
        };
        const operation = chunked
          ? new GPUChunkedSpatialWeightsAlgebra({
              operation: 'intersection',
              left: {
                rowCount: 2,
                information: operands[0].information,
                chunks: [{rowStart: 0, rowCount: 2, weights: operands[0]}]
              },
              right: {
                rowCount: 2,
                information: operands[1].information,
                chunks: [{rowStart: 0, rowCount: 2, weights: operands[1]}]
              },
              output: {chunks: [{rowStart: 0, rowCount: 2, weights: output}]}
            })
          : new GPUSpatialWeightsAlgebra({
              operation: 'intersection',
              left: operands[0],
              right: operands[1],
              output
            });
        expect(operation.outputWeights.information?.hasSelfWeights).toBe(false);
        rig.run(operation);
        expect(await destination.read()).toEqual({
          offsets: [0, 1, 2],
          neighbors: [1, 0],
          weights: flags[0] === false ? [3, 5] : [2, 4],
          distances: []
        });
        expect(await readUint32(required.buffer, 1)).toEqual([2]);
        expect(await readUint32(overflow.buffer, 1)).toEqual([0]);
        expect(await readUint32(candidate.buffer, 1)).toEqual([0]);
      } finally {
        rig.destroy();
      }
    }
});

it('R6: symmetrized local-status chunks chain to lag, then suppress it after local loss', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  const flags = [
    rig.input(Uint32Array.of(0), 'uint32', 1),
    rig.input(Uint32Array.of(0), 'uint32', 1)
  ];
  const outputs = [rig.output('float32', 1), rig.output('float32', 1)];
  const lag = rig.output('float32', 2);
  const operation = new GPUChunkedSpatialWeightsTransform({
    operation: 'average-reciprocal',
    weights: {
      rowCount: 2,
      chunks: [0, 1].map(rowStart => ({
        rowStart,
        rowCount: 1,
        weights: {
          ...rig.uploadWeights({
            offsets: [0, 1],
            neighbors: [1 - rowStart],
            weights: [rowStart ? 4 : 2],
            distances: []
          }),
          status: {overflow: flags[rowStart]}
        }
      }))
    },
    output: outputs.map(output => output.view)
  });
  rig.graph.add(operation);
  rig.graph.add(
    new GPUSpatialLag({
      weights: operation.outputWeights,
      values: rig.input(Float32Array.of(10, 20), 'float32', 2),
      output: lag.view
    })
  );
  const compiled = rig.graph.compile();
  try {
    for (const loss of [0, 1, 0]) {
      rig.bufferOf(flags[0]).write(Uint32Array.of(loss));
      submitGraph(device, compiled, undefined);
      for (const output of outputs) expect(await readFloat32(output.buffer, 1)).toEqual([3]);
      const actual = await readFloat32(lag.buffer, 2);
      if (loss) for (const value of actual) expect(value).toBeNaN();
      else expect(actual).toEqual([60, 30]);
    }
  } finally {
    compiled.destroy();
    rig.destroy();
  }
});

it('R5: outer-only kNN provenance enables adaptive uniform kernels on plain chunks', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const outputs = [rig.output('float32', 1), rig.output('float32', 1)];
    rig.run(
      new GPUChunkedSpatialWeightsTransform({
        operation: 'kernel',
        kernel: 'uniform',
        bandwidth: 'adaptive',
        weights: {
          rowCount: 2,
          information: {
            rowCount: 2,
            transform: 'raw',
            symmetric: true,
            provenance: {source: 'review', criterion: 'knn', k: 1}
          },
          chunks: [0, 1].map(rowStart => ({
            rowStart,
            rowCount: 1,
            weights: rig.uploadWeights({
              offsets: [0, 1],
              neighbors: [1 - rowStart],
              weights: [1],
              distances: [7]
            })
          }))
        },
        output: outputs.map(output => output.view)
      })
    );
    // K_uniform(7 / (7 * epsilon)) = 1/2, independently of either old weight.
    for (const output of outputs) expect(await readFloat32(output.buffer, 1)).toEqual([0.5]);
  } finally {
    rig.destroy();
  }
});

it('R4: returned output descriptions carry freshly written count and candidate words', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const output = rig.weightsOutput(2, 2);
    const count = rig.output('uint32', 1);
    const required = rig.output('uint32', 1);
    const overflow = rig.output('uint32', 1);
    const candidate = rig.output('uint32', 1);
    const islandCount = rig.output('uint32', 1);
    for (const port of [count, required, overflow, candidate, islandCount])
      port.buffer.write(Uint32Array.of(99));
    const transpose = new GPUChunkedSpatialWeightsTranspose({
      weights: {
        rowCount: 2,
        chunks: [
          {
            rowStart: 0,
            rowCount: 2,
            weights: rig.uploadWeights({
              offsets: [0, 1, 2],
              neighbors: [1, 0],
              weights: [2, 4],
              distances: []
            })
          }
        ]
      },
      output: {
        chunks: [
          {
            rowStart: 0,
            rowCount: 2,
            weights: {
              ...output.spatialWeights,
              status: {
                count: count.view,
                requiredCount: required.view,
                overflow: overflow.view,
                candidateOverflow: candidate.view
              }
            }
          }
        ]
      }
    });
    const empty = rig.weightsOutput(1, 1);
    const islandOverflow = rig.output('uint32', 1);
    const island = new GPUIslandAttachment({
      id: 'r4-island',
      reciprocal: false,
      nearestSearch: 'scan',
      positions: rig.input(Float32Array.of(0, 0), 'float32x2', 1),
      weights: rig.uploadWeights({offsets: [0, 0], neighbors: [], weights: [], distances: []}, 1),
      output: {
        ...empty.spatialWeights,
        attachment: rig.output('uint32', 1).view,
        added: rig.output('uint32', 1).view,
        status: {overflow: islandOverflow.view, count: islandCount.view}
      }
    });
    const published = rig.output('uint32', 4);
    const status = transpose.outputWeights.chunks[0].weights.status!;
    // The chunk description deliberately omits loss flags; the matrix aggregate carries them.
    expect(status.candidateOverflow).toBeUndefined();
    const aggregate = transpose.outputWeights.status!;
    rig.run(transpose, island, {
      getCommandNodes: graph => [
        createWGSLKernelNode(graph, {
          id: 'r4-published',
          operation: 'Review',
          invocationCount: 1,
          bindings: [
            {name: 'count', view: status.count!, type: 'u32', access: 'read'},
            {name: 'candidate', view: candidate.view, type: 'u32', access: 'read'},
            {
              name: 'islandCount',
              view: island.outputWeights.status!.count!,
              type: 'u32',
              access: 'read'
            },
            {
              name: 'inherited',
              view: aggregate.candidateOverflow!,
              type: 'u32',
              access: 'read'
            },
            {name: 'result', view: published.view, type: 'u32', access: 'read_write'}
          ],
          // The raw caller chunk candidate word (filled with 99) must be reset to zero, and the
          // input carries no loss, so the inherited aggregate is zero too.
          body: 'result[resultOffset] = count[countOffset]; result[resultOffset + 1u] = candidate[candidateOffset]; result[resultOffset + 2u] = islandCount[islandCountOffset]; result[resultOffset + 3u] = inherited[inheritedOffset];'
        })
      ]
    });
    expect(await readUint32(published.buffer, 4)).toEqual([2, 0, 0, 0]);
    expect(await readUint32(required.buffer, 1)).toEqual([2]);
    expect(await readUint32(overflow.buffer, 1)).toEqual([0]);
    expect(await output.read()).toEqual({
      offsets: [0, 1, 2],
      neighbors: [1, 0],
      weights: [4, 2],
      distances: []
    });
    expect((await empty.read()).offsets).toEqual([0, 0]);
    expect(await readUint32(islandOverflow.buffer, 1)).toEqual([0]);
  } finally {
    rig.destroy();
  }
});

it('R3: both named backends expose retained-work counts with both intermediate budgets', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const Operation of [CPUSpatialWeightsAlgebraOperation, GPUSpatialWeightsAlgebraOperation]) {
    for (const workCapacity of [1, 8]) {
      const result = await runOperationGraph(device, {
        inputs: {
          offsets: {format: 'uint32', values: Uint32Array.of(0, 2, 3, 4)},
          neighbors: {format: 'uint32', values: Uint32Array.of(1, 2, 2, 1)},
          weights: {format: 'float32', values: Float32Array.of(1, 1, 1, 1)}
        },
        operations: [
          new Operation({
            operation: 'higherOrder',
            rowCount: 3,
            capacity: 4,
            order: 2,
            workCapacity,
            weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
            output: {
              offsets: 'outOffsets',
              neighbors: 'outNeighbors',
              weights: 'outWeights',
              status: {
                overflow: 'overflow',
                requiredCount: 'required',
                candidateOverflow: 'candidate'
              }
            }
          })
        ],
        outputs: ['outOffsets', 'outNeighbors', 'outWeights', 'overflow', 'required', 'candidate']
      });
      const partial = workCapacity === 1;
      expect(Array.from(result['outOffsets'])).toEqual(partial ? [0, 1, 1, 1] : [0, 0, 0, 0]);
      expect(Array.from(result['outNeighbors'].slice(0, partial ? 1 : 0))).toEqual(
        partial ? [2] : []
      );
      expect(Array.from(result['outWeights'].slice(0, partial ? 1 : 0))).toEqual(
        partial ? [1] : []
      );
      expect(Array.from(result['required'])).toEqual([Number(partial)]);
      expect(Array.from(result['candidate'])).toEqual([Number(partial)]);
      expect(Array.from(result['overflow'])).toEqual([0]);
    }
  }
});

it('R2: chunked union carries a right-only meter context and preserves both measurements', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  const context = {coordinateSpace: 'planar', metric: 'native', units: 'meters'} as const;
  const output = rig.weightsOutput(2, 2, true);
  const overflow = rig.output('uint32', 1);
  const required = rig.output('uint32', 1);
  try {
    const operation = new GPUChunkedSpatialWeightsAlgebra({
      operation: 'union',
      left: {
        rowCount: 2,
        chunks: [
          {
            rowStart: 0,
            rowCount: 2,
            weights: rig.uploadWeights({
              offsets: [0, 1, 1],
              neighbors: [1],
              weights: [2],
              distances: [7]
            })
          }
        ]
      },
      right: {
        rowCount: 2,
        spatialContext: context,
        chunks: [
          {
            rowStart: 0,
            rowCount: 2,
            weights: rig.uploadWeights({
              offsets: [0, 0, 1],
              neighbors: [0],
              weights: [4],
              distances: [11]
            })
          }
        ]
      },
      output: {
        chunks: [
          {
            rowStart: 0,
            rowCount: 2,
            weights: {
              ...output.spatialWeights,
              status: {overflow: overflow.view, requiredCount: required.view}
            }
          }
        ]
      }
    });
    expect(operation.outputWeights.spatialContext).toEqual(context);
    expect(operation.outputWeights.chunks[0].weights.spatialContext).toEqual(context);
    rig.run(operation);
    expect(await output.read()).toEqual({
      offsets: [0, 1, 2],
      neighbors: [1, 0],
      weights: [2, 4],
      distances: [7, 11]
    });
    expect(await readUint32(overflow.buffer, 1)).toEqual([0]);
    expect(await readUint32(required.buffer, 1)).toEqual([2]);
  } finally {
    rig.destroy();
  }
});
