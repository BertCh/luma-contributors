// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it, vi} from 'vitest';
import {GPUChunkedSpatialWeightsAlgebra} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights-algebra';
import {GPUSpatialWeightsAlgebra} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-algebra';
import {GPUChunkedSpatialWeightsTranspose} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights-transpose';
import {createReviewSetup, createReviewCarryNodes} from './audit-weights-a-review.test-utils';
import type {GPUSpatialContext} from '../../../src/analysis-core/contracts/spatial-context';
import {computeSpatialWeightsAlgebraOnCPU} from '../../../src/gpu-spatial-analysis/spatial-weights/cpu-spatial-weights-algebra-operation';
import {GPUIslandAttachment} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-island-attachment';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {executeCommandNodesOnCPU} from '../segment-intersection/wgsl-graph-executor';
import {GPUChunkedSpatialWeightsTransform} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights-transform';
import {GPUSpatialWeightsTransform} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-transform';
import {getGPUCommandNodeKernelSources} from '../../utils/gpu-contributor-test-utils';
import {executeWeightsAuditKernel} from './weights-audit-executor';
import {CPUSpatialWeightsValidationOperation} from '../../../src/gpu-spatial-analysis/spatial-weights/cpu-spatial-weights-validation-operation';
import {GPUSpatialWeightsValidationOperation} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-validation-operation';
import {addContiguityOrderCorrelogramRecipe} from '../../../src/gpu-spatial-analysis/recipes/contiguity-order-correlogram-recipe';
import {GPUSpatialWeightsTranspose} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-transpose';
import {GPUReduction} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';

useCPUExecutorTestHooks();

it.each([
  ['double', false],
  ['double', true],
  ['average-reciprocal', false],
  ['average-reciprocal', true]
] as const)('v3-1: %s shaders avoid the Metal-rejected reserved identifier (chunked=%s)', (operation, chunked) => {
  const setup = createReviewSetup();
  try {
    const chunks = [0, 1].map(rowStart => ({rowStart, rowCount: 1, weights: setup.weights(1, 1)}));
    const transform = chunked
      ? new GPUChunkedSpatialWeightsTransform({
          operation,
          weights: {rowCount: 2, chunks},
          output: [setup.view('float32', 1), setup.view('float32', 1)]
        })
      : new GPUSpatialWeightsTransform({
          operation,
          weights: setup.weights(),
          output: setup.view('float32', 2)
        });
    const kernels = getGPUCommandNodeKernelSources(transform.getCommandNodes(setup.graph)).filter(
      kernel => kernel.source.includes('fn getSpatialWeightsDoubleScale(')
    );
    expect(kernels).toHaveLength(chunked ? 2 : 1);
    // Chromium/Metal's WGSL validator rejects `target` even in an uncalled helper parameter.
    // The CPU interpreter accepts it, so numerical interpreter comparisons cannot catch this.
    for (const kernel of kernels) {
      const shader = kernel.source.replace(/\/\*[\s\S]*?\*\/|\/\/[^\n]*/g, '');
      expect.soft(/\btarget\b/.test(shader), kernel.id).toBe(false);
    }
  } finally {
    setup.device.destroy();
  }
});

it.each([
  'double',
  'variance',
  'average-reciprocal'
] as const)('review 2 R14: %s writes the captured published destination after caller array replacement', operation => {
  const setup = createReviewSetup();
  // Only the upstream collective is replaced; production partial/apply and their bindings run.
  const reduction = vi
    .spyOn(GPUReduction.prototype, 'getCommandNodes')
    .mockImplementation(function (this: GPUReduction, graph) {
      const source = this.input as import('@luma.gl/graph/gpu-core').GraphDataView<'float32'>;
      const {output, id} = this;
      return [
        createWGSLKernelNode(graph, {
          id: `${id}-review-serial-sum`,
          operation: 'ReviewReduction',
          invocationCount: 1,
          bindings: [
            {name: 'source', view: source, type: 'f32', access: 'read'},
            {name: 'total', view: output, type: 'f32', access: 'read_write'}
          ],
          body: `var sum = 0.0; for (var row = 0u; row < ${source.length}u; row++) { sum += source[sourceOffset + row]; } total[totalOffset] = sum;`
        })
      ];
    });
  try {
    const chunks = [0, 1].map(rowStart => ({
      rowStart,
      rowCount: 1,
      weights: {
        offsets: setup.input(Uint32Array.of(0, 1), 'uint32'),
        neighbors: setup.input(Uint32Array.of(1 - rowStart), 'uint32'),
        weights: setup.input(Float32Array.of(rowStart ? 3 : 2), 'float32')
      }
    }));
    const output = [setup.view('float32', 1), setup.view('float32', 1)];
    const original = [...output];
    const replacement = setup.input(Float32Array.of(99), 'float32');
    const transform = new GPUChunkedSpatialWeightsTransform({
      operation,
      weights: {rowCount: 2, chunks},
      output
    });
    output[0] = replacement;
    executeCommandNodesOnCPU(transform.getCommandNodes(setup.graph), setup.memory);
    // Double: [2,3]/5; variance: s=[1,1], n/Q=1; symmetric mean: (2+3)/2.
    const expected =
      operation === 'double' ? [0.4, 0.6] : operation === 'variance' ? [1, 1] : [2.5, 2.5];
    for (const [index, chunk] of transform.outputWeights.chunks.entries()) {
      expect(chunk.weights.weights).toBe(original[index]);
      const actual = setup.memory.readFloat32(chunk.weights.weights)[0];
      expect.soft(Number.isFinite(actual)).toBe(true);
      expect.soft(Math.abs(actual - expected[index])).toBeLessThanOrEqual(1e-6);
    }
    expect.soft(setup.memory.readFloat32(replacement)[0]).toBe(99);
  } finally {
    reduction.mockRestore();
    setup.device.destroy();
  }
});

it('review 2 R13: both validators capture inherited input-name getters', () => {
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
  const props = {
    rowCount: 2,
    weights: new Names(),
    output: {flags: 'flags', status: {invalidCount: 'invalid'}}
  };
  expect(new GPUSpatialWeightsValidationOperation(props).inputs).toEqual({
    offsets: 'o',
    neighbors: 'n',
    weights: 'w'
  });
  const operation = new CPUSpatialWeightsValidationOperation(props);
  expect(operation.inputs).toEqual({offsets: 'o', neighbors: 'n', weights: 'w'});
  const vector = (format: 'uint32' | 'float32', values: Uint32Array | Float32Array) => ({
    format,
    length: values.length,
    chunks: [{values, length: values.length}],
    pack: () => values
  });
  expect(
    operation.execute({
      inputs: {
        offsets: vector('uint32', Uint32Array.of(0, 1, 2)),
        neighbors: vector('uint32', Uint32Array.of(0, 0)),
        weights: vector('float32', Float32Array.of(1, 1))
      }
    })
  ).toEqual({flags: Uint32Array.of(8, 0), invalid: Uint32Array.of(1)});
});

it('review 2 R13: chunk transforms capture inherited CSR getters', () => {
  const setup = createReviewSetup();
  try {
    const fields = {
      offsets: setup.input(Uint32Array.of(0, 1, 2), 'uint32'),
      neighbors: setup.input(Uint32Array.of(1, 0), 'uint32'),
      weights: setup.input(Float32Array.of(2, 3), 'float32')
    };
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
    const operation = new GPUChunkedSpatialWeightsTransform({
      operation: 'row',
      weights: {rowCount: 2, chunks: [{rowStart: 0, rowCount: 2, weights: new CSR()}]}
    });
    executeCommandNodesOnCPU(operation.getCommandNodes(setup.graph), setup.memory);
    expect(
      Array.from(setup.memory.readFloat32(operation.outputWeights.chunks[0].weights.weights))
    ).toEqual([1, 1]);
  } finally {
    setup.device.destroy();
  }
});

it.each([
  false,
  true
])('review 2 R13: packed transpose retains an inherited loss getter (chunked=%s)', chunked => {
  const setup = createReviewSetup();
  try {
    const overflow = setup.input(Uint32Array.of(1), 'uint32');
    class Status {
      get overflow() {
        return overflow;
      }
    }
    const input = {
      offsets: setup.input(Uint32Array.of(0, 1, 2), 'uint32'),
      neighbors: setup.input(Uint32Array.of(1, 0), 'uint32'),
      weights: setup.input(Float32Array.of(2, 4), 'float32')
    };
    const output = setup.weights();
    const operation = chunked
      ? new GPUChunkedSpatialWeightsTranspose({
          weights: {
            rowCount: 2,
            status: new Status(),
            chunks: [{rowStart: 0, rowCount: 2, weights: input}]
          },
          output
        })
      : new GPUSpatialWeightsTranspose({weights: {...input, status: new Status()}, output});
    expect.soft(operation.outputWeights.status?.overflow === overflow).toBe(true);
    executeCommandNodesOnCPU(operation.getCommandNodes(setup.graph), setup.memory);
    expect(Array.from(setup.memory.readUint32(output.offsets))).toEqual([0, 1, 2]);
    expect(Array.from(setup.memory.readUint32(overflow))).toEqual([1]);
  } finally {
    setup.device.destroy();
  }
});

it('review 2 R10: island preflight and publication retain captured input and output views', () => {
  for (const foreign of [false, true]) {
    const setup = createReviewSetup();
    const other = createReviewSetup();
    try {
      const inputLoss = setup.input(Uint32Array.of(1), 'uint32');
      const weights = {
        offsets: setup.input(Uint32Array.of(0, 0, 1), 'uint32'),
        neighbors: setup.input(Uint32Array.of(0), 'uint32'),
        weights: setup.input(Float32Array.of(2), 'float32'),
        status: {overflow: inputLoss}
      };
      const count = setup.view('uint32', 1);
      const replacement = setup.view('uint32', 1);
      const output = {
        ...setup.output(2, 2),
        attachment: setup.view('uint32', 2),
        added: setup.view('uint32', 2),
        status: {...setup.output(2, 2).status, count}
      };
      const operation = new GPUIslandAttachment({
        nearestSearch: 'scan',
        reciprocal: false,
        weights,
        positions: setup.input(Float32Array.of(0, 0, 1, 0), 'float32x2'),
        output
      });
      weights.status = {
        overflow: foreign ? other.view('uint32', 1) : setup.input(Uint32Array.of(0), 'uint32')
      };
      output.status.count = replacement;
      const nodes = operation.getCommandNodes(setup.graph);
      const actual: number[] = [];
      for (const value of [1, 0]) {
        setup.memory.write(inputLoss, Uint32Array.of(value));
        executeCommandNodesOnCPU(nodes, setup.memory);
        actual.push(setup.memory.readUint32(operation.outputWeights.status!.candidateOverflow!)[0]);
        expect.soft(setup.memory.readUint32(operation.outputWeights.status!.count!)[0]).toBe(2);
        expect.soft(setup.memory.readUint32(replacement)[0]).toBe(99);
      }
      expect(actual).toEqual([1, 0]);
      expect(Array.from(setup.memory.readUint32(output.offsets))).toEqual([0, 1, 2]);
      expect(Array.from(setup.memory.readUint32(output.neighbors))).toEqual([1, 0]);
      expect(Array.from(setup.memory.readFloat32(output.weights))).toEqual([1, 2]);
      expect(Array.from(setup.memory.readUint32(output.attachment))).toEqual([1, 0xffffffff]);
      expect(Array.from(setup.memory.readUint32(output.added))).toEqual([1, 0]);
    } finally {
      setup.device.destroy();
      other.device.destroy();
    }
  }
});

it.each([
  'whole kernel',
  'chunked kernel',
  'chunked row'
] as const)('review 2 R9: accepts safe separate-dispatch and unused-source layouts (%s)', layout => {
  const setup = createReviewSetup();
  try {
    const backing = setup.input(
      Float32Array.from(layout === 'chunked row' ? [2, 3] : [2, 3, 99, 99]),
      'float32'
    );
    const create = (byteOffset: number, length: number) =>
      setup.graph.createDataView(backing.buffer, {format: 'float32', byteOffset, length});
    if (layout === 'chunked row') {
      const chunks = [0, 1].map(rowStart => ({
        rowStart,
        rowCount: 1,
        weights: {
          offsets: setup.input(Uint32Array.of(0, 1), 'uint32'),
          neighbors: setup.input(Uint32Array.of(1 - rowStart), 'uint32'),
          weights: create(rowStart * 4, 1)
        }
      }));
      const operation = new GPUChunkedSpatialWeightsTransform({
        operation: 'row',
        weights: {rowCount: 2, chunks}
      });
      executeCommandNodesOnCPU(operation.getCommandNodes(setup.graph), setup.memory);
      expect(Array.from(setup.memory.readFloat32(backing))).toEqual([1, 1]);
    } else {
      const weights = {
        offsets: setup.input(Uint32Array.of(0, 1, 2), 'uint32'),
        neighbors: setup.input(Uint32Array.of(1, 0), 'uint32'),
        weights: create(0, 2),
        distances: setup.input(Float32Array.of(7, 7), 'float32')
      };
      const output = create(8, 2);
      const operation =
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
            });
      executeCommandNodesOnCPU(operation.getCommandNodes(setup.graph), setup.memory);
      // K_uniform(7/8)=1/2; the source weights are never bound or read by a kernel transform.
      expect(Array.from(setup.memory.readFloat32(backing))).toEqual([2, 3, 0.5, 0.5]);
    }
  } finally {
    setup.device.destroy();
  }
});

it('review 2 R7: intersection resolves chunk-only diagonal exclusions then aggregates all output chunks', () => {
  for (const reverse of [false, true])
    for (const allExcluded of [false, true]) {
      const setup = createReviewSetup();
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
            weights: {
              offsets: setup.input(Uint32Array.of(0, 2), 'uint32'),
              neighbors: setup.input(Uint32Array.of(0, 1), 'uint32'),
              weights: setup.input(Float32Array.of(rowStart ? 4 : 9, rowStart ? 9 : 2), 'float32')
            }
          }))
        };
        const excluded = {
          rowCount: 2,
          chunks: [0, 1].map(rowStart => ({
            rowStart,
            rowCount: 1,
            weights: {
              offsets: setup.input(Uint32Array.of(0, 1), 'uint32'),
              neighbors: setup.input(Uint32Array.of(1 - rowStart), 'uint32'),
              weights: setup.input(Float32Array.of(rowStart ? 5 : 3), 'float32'),
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
        const output = [setup.output(1, 1), setup.output(1, 1)];
        const operation = new GPUChunkedSpatialWeightsAlgebra({
          operation: 'intersection',
          left: reverse ? excluded : diagonal,
          right: reverse ? diagonal : excluded,
          output: {chunks: output.map((weights, rowStart) => ({rowStart, rowCount: 1, weights}))}
        });
        expect
          .soft(
            operation.outputWeights.chunks.map(chunk => chunk.weights.information?.hasSelfWeights)
          )
          .toEqual([false, allExcluded ? false : true]);
        expect.soft(operation.outputWeights.information?.hasSelfWeights).toBe(!allExcluded);
        executeCommandNodesOnCPU(operation.getCommandNodes(setup.graph), setup.memory);
        expect(output.map(chunk => Array.from(setup.memory.readUint32(chunk.offsets)))).toEqual([
          [0, 1],
          [0, 1]
        ]);
        expect(output.map(chunk => setup.memory.readUint32(chunk.neighbors)[0])).toEqual([1, 0]);
        expect(output.map(chunk => setup.memory.readFloat32(chunk.weights)[0])).toEqual(
          reverse ? [3, 5] : [2, 4]
        );
      } finally {
        setup.device.destroy();
      }
    }
});

it('review 2 R5: binary output retains chunk-only kNN provenance for an adaptive kernel chain', () => {
  const setup = createReviewSetup();
  try {
    const weights = {
      rowCount: 2,
      chunks: [0, 1].map(rowStart => ({
        rowStart,
        rowCount: 1,
        weights: {
          offsets: setup.input(Uint32Array.of(0, 1), 'uint32'),
          neighbors: setup.input(Uint32Array.of(1 - rowStart), 'uint32'),
          weights: setup.input(Float32Array.of(rowStart ? 4 : 2), 'float32'),
          distances: setup.input(Float32Array.of(7), 'float32'),
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
    const output = [setup.view('float32', 1), setup.view('float32', 1)];
    const kernel = new GPUChunkedSpatialWeightsTransform({
      id: 'review2-kernel',
      operation: 'kernel',
      kernel: 'uniform',
      bandwidth: 'adaptive',
      weights: binary.outputWeights,
      output
    });
    executeCommandNodesOnCPU(
      [...binary.getCommandNodes(setup.graph), ...kernel.getCommandNodes(setup.graph)],
      setup.memory
    );
    expect(output.map(view => setup.memory.readFloat32(view)[0])).toEqual([0.5, 0.5]);
    expect(
      binary.outputWeights.chunks.map(chunk => setup.memory.readFloat32(chunk.weights.weights)[0])
    ).toEqual([1, 1]);
    expect(
      binary.outputWeights.chunks.map(chunk => chunk.weights.information?.provenance?.criterion)
    ).toEqual(['knn', 'knn']);
    expect(binary.outputWeights.chunks.map(chunk => chunk.weights.information?.transform)).toEqual([
      'binary',
      'binary'
    ]);
  } finally {
    setup.device.destroy();
  }
});

it('review 2 R4: truncated transpose omits inherited matrix cardinalities and retains input loss', () => {
  const setup = createReviewSetup();
  try {
    const status = {
      count: setup.input(Uint32Array.of(2), 'uint32'),
      requiredCount: setup.input(Uint32Array.of(2), 'uint32'),
      overflow: setup.input(Uint32Array.of(0), 'uint32'),
      candidateOverflow: setup.input(Uint32Array.of(1), 'uint32')
    };
    const output = {
      ...setup.output(2, 1),
      status: {...setup.output(2, 1).status, count: setup.view('uint32', 1)}
    };
    const operation = new GPUChunkedSpatialWeightsTranspose({
      weights: {
        rowCount: 2,
        status,
        chunks: [
          {
            rowStart: 0,
            rowCount: 2,
            weights: {
              offsets: setup.input(Uint32Array.of(0, 1, 2), 'uint32'),
              neighbors: setup.input(Uint32Array.of(1, 0), 'uint32'),
              weights: setup.input(Float32Array.of(2, 4), 'float32')
            }
          }
        ]
      },
      output: {chunks: [{rowStart: 0, rowCount: 2, weights: output}]}
    });
    const nodes = operation.getCommandNodes(setup.graph);
    executeCommandNodesOnCPU(nodes, setup.memory);
    expect(Array.from(setup.memory.readUint32(output.offsets))).toEqual([0, 1, 1]);
    expect(Array.from(setup.memory.readUint32(output.status.count))).toEqual([1]);
    expect(Array.from(setup.memory.readUint32(output.status.requiredCount))).toEqual([2]);
    expect(Array.from(setup.memory.readUint32(output.status.overflow))).toEqual([1]);
    expect.soft(operation.outputWeights.status?.count === undefined).toBe(true);
    expect.soft(operation.outputWeights.status?.requiredCount === undefined).toBe(true);
    const aggregate = operation.outputWeights.status!;
    expect(Array.from(setup.memory.readUint32(aggregate.overflow!))).toEqual([1]);
    expect(Array.from(setup.memory.readUint32(aggregate.candidateOverflow!))).toEqual([1]);
    setup.memory.write(status.candidateOverflow, Uint32Array.of(0));
    executeCommandNodesOnCPU(nodes, setup.memory);
    expect(Array.from(setup.memory.readUint32(aggregate.overflow!))).toEqual([1]);
    expect(Array.from(setup.memory.readUint32(aggregate.candidateOverflow!))).toEqual([0]);
  } finally {
    setup.device.destroy();
  }
});

it('review 2 R3: recipe publishes described higher-order weights and retains raw work diagnostics separately', () => {
  const setup = createReviewSetup();
  try {
    const result = addContiguityOrderCorrelogramRecipe(setup.graph, {
      id: 'review2-recipe',
      maximumOrder: 2,
      slotCapacity: 4,
      workCapacity: 1,
      weights: {
        offsets: setup.input(Uint32Array.of(0, 2, 3, 4), 'uint32'),
        neighbors: setup.input(Uint32Array.of(1, 2, 2, 1), 'uint32'),
        weights: setup.input(Float32Array.of(1, 1, 1, 1), 'float32')
      },
      values: setup.input(Float32Array.of(1, 2, 4), 'float32')
    });
    const nodes = (
      setup.graph as unknown as {nodes: import('@luma.gl/graph/gpu-core').GPUCommandNode[]}
    ).nodes;
    executeCommandNodesOnCPU(
      nodes.filter(
        node =>
          node.id.startsWith('review2-recipe-order-2-ring') ||
          node.id.startsWith('review2-recipe-order-2-standardize')
      ),
      setup.memory
    );
    const raw = result.contributors.find(
      contributor =>
        contributor instanceof GPUSpatialWeightsAlgebra &&
        contributor.id === 'review2-recipe-order-2-ring'
    ) as GPUSpatialWeightsAlgebra;
    const described = result.intermediates.orderWeights[1];
    // SciPy oracle from R3: exact order two has zero slots; the retained work rediscovered 0->2.
    expect(Array.from(setup.memory.readUint32(raw.props.output.status.requiredCount!))).toEqual([
      1
    ]);
    expect(Array.from(setup.memory.readUint32(described.status!.candidateOverflow!))).toEqual([1]);
    expect.soft(described.status!.requiredCount).toBeUndefined();
    expect(Array.from(setup.memory.readUint32(described.offsets))).toEqual([0, 1, 1, 1]);
    expect(setup.memory.readUint32(described.neighbors)[0]).toBe(2);
    expect(setup.memory.readFloat32(described.weights)[0]).toBe(1);
  } finally {
    setup.device.destroy();
  }
});

it('review 2 R2: unspecified outer units cannot conceal incompatible valid chunk declarations', () => {
  const setup = createReviewSetup();
  try {
    const outer: GPUSpatialContext = {
      coordinateSpace: 'longitude-latitude',
      metric: 'great-circle'
    };
    const operand = (units: 'meters' | 'kilometers') => ({
      rowCount: 2,
      spatialContext: outer,
      chunks: [
        {
          rowStart: 0,
          rowCount: 2,
          weights: {...setup.weights(2, 2, true), spatialContext: {...outer, units}}
        }
      ]
    });
    const construct = (units: 'meters' | 'kilometers') =>
      new GPUChunkedSpatialWeightsAlgebra({
        operation: 'union',
        left: operand('meters'),
        right: operand(units),
        output: {chunks: [{rowStart: 0, rowCount: 2, weights: setup.output(2, 4, true)}]}
      });
    expect.soft(() => construct('kilometers')).toThrow(/units differ/);
    const compatible = construct('meters').outputWeights;
    expect(compatible.spatialContext?.units).toBe('meters');
    expect(compatible.chunks[0].weights.spatialContext?.units).toBe('meters');
  } finally {
    setup.device.destroy();
  }
});

it('review 2 R1: transpose revalidates required distances and corrected status aliases before scratch', () => {
  for (const packed of [false, true]) {
    const setup = createReviewSetup();
    try {
      const chunks = [0, 1].map(rowStart => ({
        rowStart,
        rowCount: 1,
        weights: {
          offsets: setup.input(Uint32Array.of(0, 1), 'uint32'),
          neighbors: setup.input(Uint32Array.of(1 - rowStart), 'uint32'),
          weights: setup.input(Float32Array.of(rowStart ? 4 : 2), 'float32'),
          distances: setup.input(Float32Array.of(rowStart ? 11 : 7), 'float32') as
            | import('@luma.gl/graph/gpu-core').GraphDataView<'float32'>
            | undefined
        }
      }));
      const output = setup.output(2, 2, true);
      const count = setup.view('uint32', 1);
      const destination = {...output, status: {...output.status, count}};
      const operation = packed
        ? new GPUChunkedSpatialWeightsTranspose({
            weights: {rowCount: 2, chunks},
            output: {...output, status: undefined}
          })
        : new GPUChunkedSpatialWeightsTranspose({
            weights: {rowCount: 2, chunks},
            output: {chunks: [{rowStart: 0, rowCount: 2, weights: destination}]}
          });
      const distance = chunks[1].weights.distances;
      chunks[1].weights.distances = undefined;
      const allocation = vi.spyOn(setup.graph, 'createTransientBuffer');
      expect.soft(() => operation.getCommandNodes(setup.graph)).toThrow(/distances/);
      expect.soft(allocation).not.toHaveBeenCalled();
      chunks[1].weights.distances = distance;
      allocation.mockClear();
      if (!packed) {
        destination.status.candidateOverflow = count;
        expect
          .soft(() => operation.getCommandNodes(setup.graph))
          .toThrow(/separate buffers|overlap/);
        expect.soft(allocation).not.toHaveBeenCalled();
        Reflect.deleteProperty(destination.status, 'candidateOverflow');
        expect
          .soft(() => operation.getCommandNodes(setup.graph))
          .toThrow(/ports and shapes must match construction/);
        expect.soft(allocation).not.toHaveBeenCalled();
        destination.status.candidateOverflow = setup.view('uint32', 1);
      }
      executeCommandNodesOnCPU(operation.getCommandNodes(setup.graph), setup.memory);
      expect(Array.from(setup.memory.readFloat32(output.distances!))).toEqual([11, 7]);
      if (!packed) {
        expect(Array.from(setup.memory.readUint32(destination.status.count))).toEqual([2]);
        expect(Array.from(setup.memory.readUint32(destination.status.candidateOverflow))).toEqual([
          0
        ]);
      }
      allocation.mockRestore();
    } finally {
      setup.device.destroy();
    }
  }
});

it('R12: lattice and algebra clamp every prefix monotonically across a uint32 carry', () => {
  for (const kind of ['lattice', 'algebra'] as const) {
    const setup = createReviewSetup();
    try {
      const output = setup.output(3, 4);
      executeCommandNodesOnCPU(createReviewCarryNodes(setup.graph, kind, output), setup.memory);
      // Exact prefixes [0,4294967294,4294967297,4294967304], clamped to four slots.
      expect.soft(Array.from(setup.memory.readUint32(output.offsets))).toEqual([0, 4, 4, 4]);
      expect(Array.from(setup.memory.readUint32(output.status.requiredCount))).toEqual([
        0xffffffff
      ]);
      expect(Array.from(setup.memory.readUint32(output.status.overflow))).toEqual([1]);
    } finally {
      setup.device.destroy();
    }
  }
});

it('R11: both named validators refuse colliding flags and invalid-count outputs', () => {
  // Two rows: 0->0 violates self, 1->0 is valid; distinct outputs must be [8,0] and [1].
  const props = {
    id: 'r11-validation',
    rowCount: 2,
    weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    output: {flags: 'dup', status: {invalidCount: 'dup'}}
  };
  for (const Constructor of [
    GPUSpatialWeightsValidationOperation,
    CPUSpatialWeightsValidationOperation
  ]) {
    expect(() => new Constructor(props)).toThrow(/r11-validation.*no other port uses/);
  }
});

it('R10: island attachment captures loss views but reads their changing contents', () => {
  const setup = createReviewSetup();
  try {
    const original = setup.input(Uint32Array.of(1), 'uint32');
    const weights = {
      offsets: setup.input(Uint32Array.of(0, 0), 'uint32'),
      neighbors: setup.view('uint32', 1),
      weights: setup.view('float32', 1),
      status: {overflow: original}
    };
    const output = {
      ...setup.weights(1, 1),
      attachment: setup.view('uint32', 1),
      added: setup.view('uint32', 1),
      status: {overflow: setup.view('uint32', 1)}
    };
    const operation = new GPUIslandAttachment({
      nearestSearch: 'scan',
      reciprocal: false,
      weights,
      positions: setup.input(Float32Array.of(0, 0), 'float32x2'),
      output
    });
    weights.status = {overflow: setup.input(Uint32Array.of(0), 'uint32')};
    const nodes = operation.getCommandNodes(setup.graph);
    const flags: number[] = [];
    for (const value of [1, 0]) {
      setup.memory.write(original, Uint32Array.of(value));
      executeCommandNodesOnCPU(nodes, setup.memory);
      flags.push(setup.memory.readUint32(operation.outputWeights.status!.candidateOverflow!)[0]);
      expect(Array.from(setup.memory.readUint32(output.attachment))).toEqual([0xffffffff]);
      expect(Array.from(setup.memory.readUint32(output.added))).toEqual([99]);
    }
    expect(flags).toEqual([1, 0]);
  } finally {
    setup.device.destroy();
  }
});

it('R9: transform refuses distinct views in one buffer and retains exact-view in-place support', () => {
  for (const chunked of [false, true]) {
    const setup = createReviewSetup();
    try {
      const backing = setup.input(Float32Array.of(2, 3, 99, 99), 'float32');
      const source = setup.graph.createDataView(backing.buffer, {
        format: 'float32',
        byteOffset: 0,
        length: 2
      });
      const destination = setup.graph.createDataView(backing.buffer, {
        format: 'float32',
        byteOffset: 8,
        length: 2
      });
      const weights = {
        offsets: setup.input(Uint32Array.of(0, 1, 2), 'uint32'),
        neighbors: setup.input(Uint32Array.of(1, 0), 'uint32'),
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
      expect.soft(() => construct(destination)).toThrow(/output.*same.*range|output.*buffer/);
      const exact = setup.graph.createDataView(backing.buffer, {
        format: 'float32',
        byteOffset: 0,
        length: 2
      });
      executeCommandNodesOnCPU(construct(exact).getCommandNodes(setup.graph), setup.memory);
      expect(Array.from(setup.memory.readFloat32(backing))).toEqual([1, 1, 99, 99]);
    } finally {
      setup.device.destroy();
    }
  }
});

it('R1: chunked algebra applies the whole distance-availability rules', () => {
  const setup = createReviewSetup();
  try {
    const left = setup.weights(2, 2, true);
    const right = setup.weights(2, 2, false);
    for (const operation of ['union', 'symmetric-difference'] as const) {
      expect(
        () =>
          new GPUSpatialWeightsAlgebra({operation, left, right, output: setup.output(2, 4, true)})
      ).toThrow(/distances/);
      expect(
        () =>
          new GPUChunkedSpatialWeightsAlgebra({
            operation,
            left: {rowCount: 2, chunks: [{rowStart: 0, rowCount: 2, weights: left}]},
            right: {rowCount: 2, chunks: [{rowStart: 0, rowCount: 2, weights: right}]},
            output: {chunks: [{rowStart: 0, rowCount: 2, weights: setup.output(2, 4, true)}]}
          })
      ).toThrow(/distances/);
    }
  } finally {
    setup.device.destroy();
  }
});

it('R8: chunked normalization agrees with scaled norms and the nonfinite-total rule', () => {
  for (const operation of ['variance', 'double'] as const)
    for (const chunked of [false, true]) {
      const setup = createReviewSetup();
      try {
        const magnitude = operation === 'variance' ? 1e22 : 3e38;
        const chunks = [0, 1, 2].map(rowStart => ({
          rowStart,
          rowCount: 1,
          weights: setup.weights(1, rowStart === 0 ? 2 : 1)
        }));
        const outputViews = chunks.map(chunk =>
          setup.view('float32', chunk.weights.neighbors.length)
        );
        const nodes = chunked
          ? new GPUChunkedSpatialWeightsTransform({
              operation,
              weights: {rowCount: 3, chunks},
              output: outputViews
            }).getCommandNodes(setup.graph)
          : new GPUSpatialWeightsTransform({
              operation,
              weights: setup.weights(3, 3),
              output: setup.view('float32', 3)
            }).getCommandNodes(setup.graph);
        const records = getGPUCommandNodeKernelSources(nodes);
        const partials = new Float32Array(3);
        const total = new Float32Array(1);
        const buffers = chunked
          ? [
              {
                offsets: Uint32Array.of(0, 2),
                source: Float32Array.of(magnitude, magnitude),
                values: Float32Array.of(-999, -999),
                partials,
                total
              },
              {
                offsets: Uint32Array.of(0, 1),
                source: Float32Array.of(1),
                values: Float32Array.of(-999),
                partials,
                total
              },
              {
                offsets: Uint32Array.of(0, 0),
                source: Float32Array.of(777),
                values: Float32Array.of(-999),
                partials,
                total
              }
            ]
          : [
              {
                offsets: Uint32Array.of(0, 2, 3, 3),
                source: Float32Array.of(magnitude, magnitude, 1),
                values: Float32Array.of(-999, -999, -999),
                partials,
                total
              }
            ];
        for (const [index, buffer] of buffers.entries())
          executeWeightsAuditKernel(
            records.find(record =>
              record.id.endsWith(chunked ? `-partials-chunk${index}` : '-partials')
            )!,
            buffer
          );
        // Only the collective sum is replaced. Preserve float32 overflow in the reference sum.
        total[0] = partials.reduce((sum, value) => Math.fround(sum + value), 0);
        for (const [index, buffer] of buffers.entries())
          executeWeightsAuditKernel(
            records.find(record =>
              record.id.endsWith(chunked ? `-apply-chunk${index}` : '-apply')
            )!,
            buffer
          );
        const actual = chunked
          ? [...buffers[0].values, ...buffers[1].values]
          : Array.from(buffers[0].values);
        const expected =
          operation === 'variance'
            ? [0.8786796564403575, 0.8786796564403575, 1.2426406871192852]
            : [0, 0, 0];
        for (const [index, value] of actual.entries()) {
          expect.soft(Number.isFinite(value)).toBe(true);
          expect.soft(Math.abs(value - expected[index])).toBeLessThanOrEqual(2e-6);
        }
        if (chunked) expect(buffers[2].values[0]).toBe(-999);
        process.stdout.write(`R8 ${operation} chunked=${chunked}: ${JSON.stringify(actual)}\n`);
      } finally {
        setup.device.destroy();
      }
    }
});

it('R7: intersection excludes self metadata when either operand excludes its diagonal', () => {
  for (const flags of [
    [true, false],
    [false, true],
    [undefined, false],
    [false, undefined]
  ] as const) {
    const setup = createReviewSetup();
    const {input, memory, graph} = setup;
    try {
      const operands = flags.map(hasSelfWeights => ({
        offsets: input(
          Uint32Array.from(hasSelfWeights === false ? [0, 1, 2] : [0, 2, 3]),
          'uint32'
        ),
        neighbors: input(Uint32Array.from(hasSelfWeights === false ? [1, 0] : [0, 1, 0]), 'uint32'),
        weights: input(Float32Array.from(hasSelfWeights === false ? [3, 5] : [9, 2, 4]), 'float32'),
        information: {
          rowCount: 2,
          transform: 'raw' as const,
          symmetric: 'unknown' as const,
          hasSelfWeights
        }
      }));
      const wholeOutput = setup.output();
      const whole = new GPUSpatialWeightsAlgebra({
        operation: 'intersection',
        left: operands[0],
        right: operands[1],
        output: wholeOutput
      });
      const chunkOutput = setup.output();
      const chunked = new GPUChunkedSpatialWeightsAlgebra({
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
        output: {chunks: [{rowStart: 0, rowCount: 2, weights: chunkOutput}]}
      });
      expect.soft(whole.outputWeights.information?.hasSelfWeights).toBe(false);
      expect.soft(chunked.outputWeights.information?.hasSelfWeights).toBe(false);
      for (const [operation, output] of [
        [whole, wholeOutput],
        [chunked, chunkOutput]
      ] as const) {
        executeCommandNodesOnCPU(operation.getCommandNodes(graph), memory);
        expect(Array.from(memory.readUint32(output.offsets))).toEqual([0, 1, 2]);
        expect(Array.from(memory.readUint32(output.neighbors))).toEqual([1, 0]);
      }
    } finally {
      setup.device.destroy();
    }
  }
});

it('R6: average-reciprocal accepts complete local loss ports and preserves them on its public result', () => {
  const setup = createReviewSetup();
  const {input, view, memory, graph} = setup;
  try {
    const outputs = [view('float32', 1), view('float32', 1)];
    const chunks = [0, 1].map(rowStart => ({
      rowStart,
      rowCount: 1,
      weights: {
        offsets: input(Uint32Array.of(0, 1), 'uint32'),
        neighbors: input(Uint32Array.of(1 - rowStart), 'uint32'),
        weights: input(Float32Array.of(rowStart ? 4 : 2), 'float32'),
        status: {overflow: input(Uint32Array.of(0), 'uint32')}
      }
    }));
    const operation = new GPUChunkedSpatialWeightsTransform({
      operation: 'average-reciprocal',
      weights: {rowCount: 2, chunks},
      output: outputs
    });
    executeCommandNodesOnCPU(operation.getCommandNodes(graph), memory);
    expect(outputs.map(output => memory.readFloat32(output)[0])).toEqual([3, 3]);
    expect(operation.outputWeights.chunks[0].weights.status?.overflow).toBe(
      chunks[0].weights.status.overflow
    );
  } finally {
    setup.device.destroy();
  }
});

it('R5: adaptive chunks inherit outer-only kNN provenance and retain local row counts', () => {
  const setup = createReviewSetup();
  try {
    const operation = new GPUChunkedSpatialWeightsTransform({
      operation: 'kernel',
      kernel: 'uniform',
      bandwidth: 'adaptive',
      weights: {
        rowCount: 2,
        spatialContext: {coordinateSpace: 'planar', metric: 'native', units: 'meters'},
        information: {
          rowCount: 2,
          transform: 'raw',
          symmetric: true,
          provenance: {source: 'review', criterion: 'knn', k: 1}
        },
        chunks: [0, 1].map(rowStart => ({
          rowStart,
          rowCount: 1,
          weights: setup.weights(1, 1, true)
        }))
      },
      output: [setup.view('float32', 1), setup.view('float32', 1)]
    });
    expect(
      operation.outputWeights.chunks.map(chunk => chunk.weights.information?.rowCount)
    ).toEqual([1, 1]);
    expect(operation.outputWeights.chunks[0].weights.spatialContext?.units).toBe('meters');
    expect(operation.getCommandNodes(setup.graph).length).toBeGreaterThan(0);
  } finally {
    setup.device.destroy();
  }
});

it('R4: returned transpose and island descriptions expose overwritten diagnostics', () => {
  const setup = createReviewSetup();
  const {input, view, memory, graph} = setup;
  try {
    const source = {
      rowCount: 2,
      chunks: [
        {
          rowStart: 0,
          rowCount: 2,
          weights: {
            offsets: input(Uint32Array.of(0, 1, 2), 'uint32'),
            neighbors: input(Uint32Array.of(1, 0), 'uint32'),
            weights: input(Float32Array.of(2, 4), 'float32')
          }
        }
      ]
    };
    const destination = {
      ...setup.output(),
      status: {...setup.output().status, count: view('uint32', 1)}
    };
    const transpose = new GPUChunkedSpatialWeightsTranspose({
      weights: source,
      output: {chunks: [{rowStart: 0, rowCount: 2, weights: destination}]}
    });
    executeCommandNodesOnCPU(transpose.getCommandNodes(graph), memory);
    const published = transpose.outputWeights.chunks[0].weights.status!;
    expect(Array.from(memory.readUint32(published.count!))).toEqual([2]);
    expect.soft(Array.from(memory.readUint32(destination.status.candidateOverflow!))).toEqual([0]);
    const count = view('uint32', 1);
    const island = new GPUIslandAttachment({
      id: 'r4-island',
      reciprocal: false,
      nearestSearch: 'scan',
      positions: input(Float32Array.of(0, 0), 'float32x2'),
      weights: {
        offsets: input(Uint32Array.of(0, 0), 'uint32'),
        neighbors: view('uint32', 1),
        weights: view('float32', 1)
      },
      output: {
        ...setup.output(1, 1),
        attachment: view('uint32', 1),
        added: view('uint32', 1),
        status: {overflow: view('uint32', 1), count}
      }
    });
    executeCommandNodesOnCPU(island.getCommandNodes(graph), memory);
    expect(Array.from(memory.readUint32(island.outputWeights.status!.count!))).toEqual([0]);
  } finally {
    setup.device.destroy();
  }
});

it('R4: constructors refuse destination diagnostics they do not write', () => {
  const setup = createReviewSetup();
  try {
    const inputWeights = {
      rowCount: 2,
      chunks: [{rowStart: 0, rowCount: 2, weights: setup.weights()}]
    };
    expect
      .soft(
        () =>
          new GPUChunkedSpatialWeightsTranspose({
            weights: inputWeights,
            output: {...setup.weights(), status: {count: setup.view('uint32', 1)}}
          })
      )
      .toThrow(/output.status/);
    const destination = {
      ...setup.output(),
      status: {
        ...setup.output().status,
        count: setup.view('uint32', 1),
        invalidCount: setup.view('uint32', 1)
      }
    };
    expect
      .soft(
        () =>
          new GPUChunkedSpatialWeightsTranspose({
            weights: inputWeights,
            output: {chunks: [{rowStart: 0, rowCount: 2, weights: destination}]}
          })
      )
      .toThrow(/invalidCount/);
    expect(
      () =>
        new GPUIslandAttachment({
          nearestSearch: 'scan',
          reciprocal: false,
          weights: setup.weights(1, 1),
          positions: setup.input(Float32Array.of(0, 0), 'float32x2'),
          output: {
            ...setup.output(1, 1),
            attachment: setup.view('uint32', 1),
            added: setup.view('uint32', 1),
            status: {overflow: setup.view('uint32', 1), invalidCount: setup.view('uint32', 1)}
          }
        })
    ).toThrow(/invalidCount/);
  } finally {
    setup.device.destroy();
  }
});

it('R3: candidate loss counts retained work and can exceed the exact shortest-path requirement', () => {
  const weights = {
    offsets: Uint32Array.of(0, 2, 3, 4),
    neighbors: Uint32Array.of(1, 2, 2, 1),
    weights: Float32Array.of(1, 1, 1, 1)
  };
  const partial = computeSpatialWeightsAlgebraOnCPU(
    {weights},
    {operation: 'higherOrder', order: 2, capacity: 4, workCapacity: 1}
  );
  const complete = computeSpatialWeightsAlgebraOnCPU(
    {weights},
    {operation: 'higherOrder', order: 2, capacity: 4, workCapacity: 8}
  );
  // SciPy shortest_path oracle: all exact-order-two rows are empty. Retained work forgot 0->2.
  expect(Array.from(partial.offsets)).toEqual([0, 1, 1, 1]);
  expect(partial.neighbors[0]).toBe(2);
  expect(partial.weights[0]).toBe(1);
  expect(Array.from(partial.status.requiredCount)).toEqual([1]);
  expect(Array.from(partial.status.candidateOverflow)).toEqual([1]);
  expect(Array.from(partial.status.overflow)).toEqual([0]);
  expect(Array.from(complete.offsets)).toEqual([0, 0, 0, 0]);
  expect(Array.from(complete.status.requiredCount)).toEqual([0]);
  expect(Array.from(complete.status.candidateOverflow)).toEqual([0]);
  expect(Array.from(complete.status.overflow)).toEqual([0]);
  expect(partial.status.requiredCount[0]).toBeGreaterThan(complete.status.requiredCount[0]);
});

it('R2: chunked algebra rejects incompatible units and resolves right-only context', () => {
  const setup = createReviewSetup();
  const meters: GPUSpatialContext = {coordinateSpace: 'planar', metric: 'native', units: 'meters'};
  try {
    const operand = (spatialContext?: GPUSpatialContext) => ({
      rowCount: 2,
      spatialContext,
      chunks: [{rowStart: 0, rowCount: 2, weights: setup.weights(2, 2, true)}]
    });
    const construct = (leftContext?: GPUSpatialContext, rightContext?: GPUSpatialContext) =>
      new GPUChunkedSpatialWeightsAlgebra({
        operation: 'union',
        left: operand(leftContext),
        right: operand(rightContext),
        output: {chunks: [{rowStart: 0, rowCount: 2, weights: setup.output(2, 4, true)}]}
      });
    expect(() =>
      construct(meters, {...meters, units: 'kilometers' as GPUSpatialContext['units']})
    ).toThrow(/spatial context|units/);
    for (const leftContext of [meters, undefined]) {
      const result = construct(leftContext, meters).outputWeights;
      expect(result.spatialContext).toEqual(meters);
      expect(result.chunks[0].weights.spatialContext).toEqual(meters);
    }
  } finally {
    setup.device.destroy();
  }
});

it('R1: packed transpose requires requested distances on every input chunk', () => {
  const setup = createReviewSetup();
  try {
    expect(
      () =>
        new GPUChunkedSpatialWeightsTranspose({
          weights: {
            rowCount: 2,
            chunks: [
              {rowStart: 0, rowCount: 1, weights: setup.weights(1, 1, true)},
              {rowStart: 1, rowCount: 1, weights: setup.weights(1, 1, false)}
            ]
          },
          output: setup.weights(2, 2, true)
        })
    ).toThrow(/distances/);
  } finally {
    setup.device.destroy();
  }
});
