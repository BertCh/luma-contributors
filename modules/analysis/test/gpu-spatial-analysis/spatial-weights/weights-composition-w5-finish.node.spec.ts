// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// R19 wave 5, stream spatial-stats-w5-weights-composition (finishing pass; the cut-off
// implementer's cases are in weights-composition-w5.node.spec.ts). Host construction plus the CPU WGSL
// interpreter (not GPU execution). Every expectation below is a hand-worked literal.

import {expect, it} from 'vitest';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {createReviewSetup} from './audit-weights-a-review.test-utils';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {executeCommandNodesOnCPU} from '../segment-intersection/wgsl-graph-executor';
import type {GPUSpatialContext} from '../../../src/analysis-core/contracts/spatial-context';
import {GPUSpatialWeightsTransform} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-transform';
import {GPUChunkedSpatialWeightsTransform} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights-transform';
import {GPUSpatialWeightsAlgebra} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-algebra';
import {GPUChunkedSpatialWeightsAlgebra} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights-algebra';
import {GPUSpatialWeightsTranspose} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-transpose';
import {GPUChunkedSpatialWeightsTranspose} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights-transpose';
import {GPUSpatialWeightsValidation} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-validation';
import {
  getGPUSpatialWeightsLossAndCompletenessViews,
  getGPUSpatialWeightsLossViews
} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights';
import type {GPUSpatialWeights} from '../../../src/gpu-spatial-analysis/spatial-weights/spatial-weights';
import type {GPUChunkedSpatialWeights} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights';
import {snapshotWeightsPort} from '../../../src/gpu-spatial-analysis/spatial-weights/weights-construction-snapshot';

useCPUExecutorTestHooks();

type Setup = ReturnType<typeof createReviewSetup>;

/** Two reciprocal edges 0 -> 1 (weight 2) and 1 -> 0 (weight 4). */
function createPair(setup: Setup) {
  return {
    offsets: setup.input(Uint32Array.of(0, 1, 2), 'uint32'),
    neighbors: setup.input(Uint32Array.of(1, 0), 'uint32'),
    weights: setup.input(Float32Array.of(2, 4), 'float32')
  };
}

/** A certificate alone, or with both negative loss flags present and clear. */
function createStatus(setup: Setup, kind: 'complete-only' | 'both-flags-clear') {
  const complete = setup.input(Uint32Array.of(1), 'uint32');
  return {
    complete,
    status:
      kind === 'complete-only'
        ? {complete}
        : {
            overflow: setup.input(Uint32Array.of(0), 'uint32'),
            candidateOverflow: setup.input(Uint32Array.of(0), 'uint32'),
            complete
          }
  };
}

/** Writes the certificate 1 -> 0 -> 1 and reads the consumer's loss word after each run. */
function readRecovery(
  setup: Setup,
  complete: GraphDataView<'uint32'>,
  nodes: Parameters<typeof executeCommandNodesOnCPU>[0],
  loss: GraphDataView<'uint32'>
): number[] {
  return [1, 0, 1].map(value => {
    setup.memory.write(complete, Uint32Array.of(value));
    executeCommandNodesOnCPU(nodes, setup.memory);
    return setup.memory.readUint32(loss)[0];
  });
}

// CPL-03: the shared snapshot keeps status.complete, identity-preserving transforms pass it on,
// and a completeness-aware consumer inverts it on every submission.
it.each([
  'complete-only',
  'both-flags-clear'
] as const)('CPL-03 whole transform keeps a %s certificate for its consumer', kind => {
  const setup = createReviewSetup();
  try {
    const {complete, status} = createStatus(setup, kind);
    const transform = new GPUSpatialWeightsTransform({
      operation: 'row',
      weights: {...createPair(setup), status},
      output: setup.view('float32', 2)
    });
    expect(transform.outputWeights.status?.complete === complete).toBe(true);
    const output = setup.output(2, 4);
    const union = new GPUSpatialWeightsAlgebra({
      operation: 'union',
      left: transform.outputWeights,
      right: transform.outputWeights,
      output
    });
    const nodes = [
      ...transform.getCommandNodes(setup.graph),
      ...union.getCommandNodes(setup.graph)
    ];
    // Row standardization of one slot per row gives [1, 1]; the union of W with itself is W.
    expect(
      readRecovery(setup, complete, nodes, union.outputWeights.status!.candidateOverflow!)
    ).toEqual([0, 1, 0]);
    expect(Array.from(setup.memory.readFloat32(output.weights, 2))).toEqual([1, 1]);
  } finally {
    setup.device.destroy();
  }
});

it.each([
  ['complete-only', 'root'],
  ['complete-only', 'chunk'],
  ['both-flags-clear', 'root'],
  ['both-flags-clear', 'chunk']
] as const)('CPL-03 chunked transform keeps a %s %s certificate for its consumer', (kind, level) => {
  const setup = createReviewSetup();
  try {
    const {complete, status} = createStatus(setup, kind);
    const transform = new GPUChunkedSpatialWeightsTransform({
      operation: 'row',
      weights: {
        rowCount: 2,
        ...(level === 'root' ? {status} : {}),
        chunks: [
          {
            rowStart: 0,
            rowCount: 2,
            weights: {...createPair(setup), ...(level === 'chunk' ? {status} : {})}
          }
        ]
      },
      output: [setup.view('float32', 2)]
    });
    const published =
      level === 'root'
        ? transform.outputWeights.status
        : transform.outputWeights.chunks[0].weights.status;
    expect(published?.complete === complete).toBe(true);
    const union = new GPUChunkedSpatialWeightsAlgebra({
      operation: 'union',
      left: transform.outputWeights,
      right: transform.outputWeights,
      output: {chunks: [{rowStart: 0, rowCount: 2, weights: setup.output(2, 4)}]}
    });
    const nodes = [
      ...transform.getCommandNodes(setup.graph),
      ...union.getCommandNodes(setup.graph)
    ];
    expect(
      readRecovery(setup, complete, nodes, union.outputWeights.status!.candidateOverflow!)
    ).toEqual([0, 1, 0]);
  } finally {
    setup.device.destroy();
  }
});

it.each([
  'complete-only',
  'both-flags-clear'
] as const)('CPL-03 whole transpose forwards a %s certificate to its consumer', kind => {
  const setup = createReviewSetup();
  try {
    const {complete, status} = createStatus(setup, kind);
    const transpose = new GPUSpatialWeightsTranspose({
      weights: {...createPair(setup), status},
      output: setup.weights(2, 2)
    });
    expect(transpose.outputWeights.status?.complete === complete).toBe(true);
    const union = new GPUSpatialWeightsAlgebra({
      operation: 'union',
      left: transpose.outputWeights,
      right: transpose.outputWeights,
      output: setup.output(2, 4)
    });
    const nodes = [
      ...transpose.getCommandNodes(setup.graph),
      ...union.getCommandNodes(setup.graph)
    ];
    expect(
      readRecovery(setup, complete, nodes, union.outputWeights.status!.candidateOverflow!)
    ).toEqual([0, 1, 0]);
  } finally {
    setup.device.destroy();
  }
});

it.each([
  ['packed', 'root'],
  ['packed', 'chunk'],
  ['chunked', 'root'],
  ['chunked', 'chunk']
] as const)('CPL-03 chunked transpose to %s output folds a %s certificate', (layout, level) => {
  const setup = createReviewSetup();
  try {
    const {complete, status} = createStatus(setup, 'complete-only');
    const transpose = new GPUChunkedSpatialWeightsTranspose({
      weights: {
        rowCount: 2,
        ...(level === 'root' ? {status} : {}),
        chunks: [
          {
            rowStart: 0,
            rowCount: 2,
            weights: {...createPair(setup), ...(level === 'chunk' ? {status} : {})}
          }
        ]
      },
      output:
        layout === 'packed'
          ? setup.weights(2, 2)
          : {
              chunks: [
                {
                  rowStart: 0,
                  rowCount: 2,
                  weights: {
                    ...setup.output(2, 2),
                    status: {...setup.output(2, 2).status, count: setup.view('uint32', 1)}
                  }
                }
              ]
            }
    });
    const nodes = transpose.getCommandNodes(setup.graph);
    const published = transpose.outputWeights.status!;
    if (layout === 'packed' && level === 'root') {
      // Packed output forwards the matrix certificate itself; a consumer inverts it.
      expect(published.complete === complete).toBe(true);
      const union = new GPUSpatialWeightsAlgebra({
        operation: 'union',
        left: transpose.outputWeights as never,
        right: transpose.outputWeights as never,
        output: setup.output(2, 4)
      });
      expect(
        readRecovery(
          setup,
          complete,
          [...nodes, ...union.getCommandNodes(setup.graph)],
          union.outputWeights.status!.candidateOverflow!
        )
      ).toEqual([0, 1, 0]);
    } else {
      // Otherwise the transpose folds complete == 0 into its own candidateOverflow aggregate.
      expect(readRecovery(setup, complete, nodes, published.candidateOverflow!)).toEqual([0, 1, 0]);
    }
  } finally {
    setup.device.destroy();
  }
});

it('CPL-03 the loss-only list still refuses certificates; the aware list returns both polarities', () => {
  const setup = createReviewSetup();
  try {
    const rootComplete = setup.input(Uint32Array.of(1), 'uint32');
    const chunkComplete = setup.input(Uint32Array.of(1), 'uint32');
    const overflow = setup.input(Uint32Array.of(0), 'uint32');
    const weights = {
      rowCount: 2,
      status: {overflow, complete: rootComplete},
      chunks: [
        {
          rowStart: 0,
          rowCount: 2,
          weights: {...createPair(setup), status: {complete: chunkComplete}}
        }
      ]
    };
    // The construction snapshot preserves both certificates by identity.
    const captured = snapshotWeightsPort(weights);
    expect(captured.status.complete === rootComplete).toBe(true);
    expect(captured.chunks[0].weights.status.complete === chunkComplete).toBe(true);
    expect(() => getGPUSpatialWeightsLossViews(captured)).toThrow(/completeness-aware/);
    const views = getGPUSpatialWeightsLossAndCompletenessViews(captured);
    expect(views.loss.length === 1 && views.loss[0] === overflow).toBe(true);
    expect(views.complete.length).toBe(2);
    expect(views.complete[0] === rootComplete && views.complete[1] === chunkComplete).toBe(true);
  } finally {
    setup.device.destroy();
  }
});

// W5-01: matrix aggregates report destination truncation and inherited loss independently.
it.each([
  {capacity: 1, inherited: 0, expected: [1, 0]},
  {capacity: 2, inherited: 1, expected: [0, 1]}
])('W5-01 chunked transpose aggregates: capacity $capacity, input candidate $inherited', fixture => {
  const setup = createReviewSetup();
  try {
    const candidateOverflow = setup.input(Uint32Array.of(fixture.inherited), 'uint32');
    const destination = {
      ...setup.output(2, fixture.capacity),
      status: {...setup.output(2, fixture.capacity).status, count: setup.view('uint32', 1)}
    };
    const transpose = new GPUChunkedSpatialWeightsTranspose({
      weights: {
        rowCount: 2,
        status: {candidateOverflow},
        chunks: [{rowStart: 0, rowCount: 2, weights: createPair(setup)}]
      },
      output: {chunks: [{rowStart: 0, rowCount: 2, weights: destination}]}
    });
    executeCommandNodesOnCPU(transpose.getCommandNodes(setup.graph), setup.memory);
    const aggregate = transpose.outputWeights.status!;
    expect([
      setup.memory.readUint32(aggregate.overflow!)[0],
      setup.memory.readUint32(aggregate.candidateOverflow!)[0]
    ]).toEqual(fixture.expected);
    // The raw caller-owned chunk candidate word was reset from its 99 fill; it is not published.
    expect(Array.from(setup.memory.readUint32(destination.status.candidateOverflow))).toEqual([0]);
    expect(transpose.outputWeights.chunks[0].weights.status?.candidateOverflow).toBeUndefined();
  } finally {
    setup.device.destroy();
  }
});

// W5-02: distances combined or compared to a bandwidth need metric-result compatibility.
const IMPLICIT_METERS: GPUSpatialContext = {
  coordinateSpace: 'longitude-latitude',
  metric: 'great-circle'
};
const KILOMETERS: GPUSpatialContext = {...IMPLICIT_METERS, units: 'kilometers'};

it.each([
  'implicit-first',
  'kilometers-first'
] as const)('W5-02 algebra refuses [7 m, 0.007 km] distances (%s)', order => {
  const setup = createReviewSetup();
  try {
    // Both operands store distance 7 in meters and 0.007 in kilometers: the same 7 m edge.
    // A merged result labeled kilometers would report the first one 1000 times too long.
    const operand = (spatialContext: GPUSpatialContext, distance: number) => ({
      ...createPair(setup),
      distances: setup.input(Float32Array.of(distance, distance), 'float32'),
      spatialContext
    });
    const [left, right] =
      order === 'implicit-first'
        ? [operand(IMPLICIT_METERS, 7), operand(KILOMETERS, 0.007)]
        : [operand(KILOMETERS, 0.007), operand(IMPLICIT_METERS, 7)];
    expect(
      () =>
        new GPUSpatialWeightsAlgebra({
          operation: 'union',
          left,
          right,
          output: setup.output(2, 4, true)
        })
    ).toThrow(/units differ/);
    const chunked = (weights: typeof left) => ({
      rowCount: 2,
      chunks: [{rowStart: 0, rowCount: 2, weights}]
    });
    expect(
      () =>
        new GPUChunkedSpatialWeightsAlgebra({
          operation: 'union',
          left: chunked(left),
          right: chunked(right),
          output: {chunks: [{rowStart: 0, rowCount: 2, weights: setup.output(2, 4, true)}]}
        })
    ).toThrow(/units differ/);
    // Control: two implicit-meter operands still merge.
    const control = new GPUChunkedSpatialWeightsAlgebra({
      operation: 'union',
      left: chunked(operand(IMPLICIT_METERS, 7)),
      right: chunked(operand(IMPLICIT_METERS, 7)),
      output: {chunks: [{rowStart: 0, rowCount: 2, weights: setup.output(2, 4, true)}]}
    });
    expect(control.outputWeights.spatialContext?.metric).toBe('great-circle');
  } finally {
    setup.device.destroy();
  }
});

it.each([
  'linear-first',
  'ellipsoidal-first'
] as const)('W5-02 chunked algebra refuses an invalid assembled edge model (%s)', order => {
  const setup = createReviewSetup();
  try {
    // Each context is valid alone; filling the ellipsoidal one with edgeModel 'linear' is not.
    const linear: GPUSpatialContext = {
      coordinateSpace: 'longitude-latitude',
      metric: 'none',
      edgeModel: 'linear'
    };
    const ellipsoidal: GPUSpatialContext = {
      coordinateSpace: 'longitude-latitude',
      metric: 'ellipsoidal'
    };
    const operand = (spatialContext: GPUSpatialContext) => ({
      rowCount: 2,
      spatialContext,
      chunks: [{rowStart: 0, rowCount: 2, weights: setup.weights(2, 2, true)}]
    });
    const [left, right] =
      order === 'linear-first'
        ? [operand(linear), operand(ellipsoidal)]
        : [operand(ellipsoidal), operand(linear)];
    expect(
      () =>
        new GPUChunkedSpatialWeightsAlgebra({
          operation: 'union',
          left,
          right,
          output: {chunks: [{rowStart: 0, rowCount: 2, weights: setup.output(2, 4, true)}]}
        })
    ).toThrow(/incompatible spatial contexts|edgeModel/);
  } finally {
    setup.device.destroy();
  }
});

it.each([
  ['kernel', true],
  ['row', false]
] as const)('W5-02 %s transform bandwidth context check (refuses: %s)', (operation, refuses) => {
  const setup = createReviewSetup();
  try {
    const pair = () => ({
      ...createPair(setup),
      distances: setup.input(Float32Array.of(7, 7), 'float32')
    });
    const kernelProps = operation === 'kernel' ? {bandwidth: 0.01, kernel: 'uniform' as const} : {};
    const construct = [
      // Whole: implicit-meter distances against a kilometer bandwidth context.
      () =>
        new GPUSpatialWeightsTransform({
          operation,
          ...kernelProps,
          weights: {...pair(), spatialContext: IMPLICIT_METERS},
          spatialContext: KILOMETERS,
          output: setup.view('float32', 2)
        }),
      // Chunked: implicit-meter matrix against a kilometer chunk.
      () =>
        new GPUChunkedSpatialWeightsTransform({
          operation,
          ...kernelProps,
          weights: {
            rowCount: 2,
            spatialContext: IMPLICIT_METERS,
            chunks: [{rowStart: 0, rowCount: 2, weights: {...pair(), spatialContext: KILOMETERS}}]
          },
          output: [setup.view('float32', 2)]
        })
    ];
    for (const create of construct) {
      // A row transform never reads distances: coordinate compatibility remains its contract.
      if (refuses) expect(create).toThrow(/units differ/);
      else expect(create).not.toThrow();
    }
  } finally {
    setup.device.destroy();
  }
});

// W5-03: a chunk's undefined provenance field must not erase the matrix declaration.
it.each([
  {name: 'parent-only', parent: 'knn', child: 'absent', accepted: true},
  {name: 'child-only', parent: 'absent', child: 'knn', accepted: true},
  {name: 'explicitly undefined child', parent: 'knn', child: undefined, accepted: true},
  {name: 'child overrides knn', parent: 'knn', child: 'distance-band', accepted: false},
  {name: 'child overrides band', parent: 'distance-band', child: 'knn', accepted: true}
] as const)('W5-03 binary then adaptive kernel: $name', fixture => {
  const setup = createReviewSetup();
  try {
    const provenance = (criterion: string | undefined | 'absent') =>
      criterion === 'absent' ? {source: 'review'} : {source: 'review', criterion, k: 1};
    const information = (criterion: string | undefined | 'absent') => ({
      rowCount: 1,
      transform: 'raw' as const,
      symmetric: true as const,
      provenance: provenance(criterion)
    });
    const binary = new GPUChunkedSpatialWeightsTransform({
      operation: 'binary',
      weights: {
        rowCount: 2,
        information: {...information(fixture.parent), rowCount: 2},
        chunks: [0, 1].map(rowStart => ({
          rowStart,
          rowCount: 1,
          weights: {
            offsets: setup.input(Uint32Array.of(0, 1), 'uint32'),
            neighbors: setup.input(Uint32Array.of(1 - rowStart), 'uint32'),
            weights: setup.input(Float32Array.of(rowStart ? 4 : 2), 'float32'),
            distances: setup.input(Float32Array.of(7), 'float32'),
            information: information(fixture.child)
          }
        }))
      }
    });
    const output = [setup.view('float32', 1), setup.view('float32', 1)];
    const kernel = () =>
      new GPUChunkedSpatialWeightsTransform({
        operation: 'kernel',
        kernel: 'uniform',
        bandwidth: 'adaptive',
        weights: binary.outputWeights,
        output
      });
    if (!fixture.accepted) {
      expect(kernel).toThrow(/adaptive' needs kNN/);
      return;
    }
    const adaptive = kernel();
    executeCommandNodesOnCPU(
      [...binary.getCommandNodes(setup.graph), ...adaptive.getCommandNodes(setup.graph)],
      setup.memory
    );
    // Each row's only neighbor is at 7: h = 7 * 1.0000001, z < 1, uniform K = 1/2.
    expect(output.map(view => setup.memory.readFloat32(view)[0])).toEqual([0.5, 0.5]);
  } finally {
    setup.device.destroy();
  }
});

it('W5-03 an undefined chunk information field keeps the matrix value', () => {
  const setup = createReviewSetup();
  try {
    const transform = new GPUChunkedSpatialWeightsTransform({
      operation: 'binary',
      weights: {
        rowCount: 2,
        information: {rowCount: 2, transform: 'raw', symmetric: true, hasSelfWeights: false},
        chunks: [
          {
            rowStart: 0,
            rowCount: 2,
            weights: {
              ...createPair(setup),
              information: {
                rowCount: 2,
                transform: 'raw',
                symmetric: true,
                hasSelfWeights: undefined
              }
            }
          }
        ]
      }
    });
    expect(transform.outputWeights.chunks[0].weights.information?.hasSelfWeights).toBe(false);
  } finally {
    setup.device.destroy();
  }
});

// W5-04: symmetrization reads neighbors while writing weights; no buffer may hold both.
it.each([
  'whole',
  'chunked'
])('W5-04 %s average-reciprocal refuses a destination sharing the neighbors buffer', layout => {
  const setup = createReviewSetup();
  try {
    const backing = setup.view('uint32', 66);
    const neighbors = setup.graph.createDataView(backing.buffer, {format: 'uint32', length: 2});
    const destination = setup.graph.createDataView(backing.buffer, {
      format: 'float32',
      byteOffset: 256,
      length: 2
    });
    const weights = {...createPair(setup), neighbors};
    const construct = (operation: 'average-reciprocal' | 'row' | 'binary') =>
      layout === 'whole'
        ? new GPUSpatialWeightsTransform({operation, weights, output: destination})
        : new GPUChunkedSpatialWeightsTransform({
            operation,
            weights: {rowCount: 2, chunks: [{rowStart: 0, rowCount: 2, weights}]},
            output: [destination]
          });
    expect(() => construct('average-reciprocal')).toThrow(/share buffers/);
    // Row and binary never bind neighbors, so the disjoint layout stays supported.
    for (const operation of ['row', 'binary'] as const)
      expect(() => construct(operation).getCommandNodes(setup.graph)).not.toThrow();
  } finally {
    setup.device.destroy();
  }
});

it.each([
  'whole',
  'chunked'
])('W5-04 %s average-reciprocal keeps an unused disjoint distance buffer', layout => {
  const setup = createReviewSetup();
  try {
    const backing = setup.view('float32', 66);
    const distances = setup.graph.createDataView(backing.buffer, {format: 'float32', length: 2});
    const destination = setup.graph.createDataView(backing.buffer, {
      format: 'float32',
      byteOffset: 256,
      length: 2
    });
    const weights = {...createPair(setup), distances};
    const transform =
      layout === 'whole'
        ? new GPUSpatialWeightsTransform({
            operation: 'average-reciprocal',
            weights,
            output: destination
          })
        : new GPUChunkedSpatialWeightsTransform({
            operation: 'average-reciprocal',
            weights: {rowCount: 2, chunks: [{rowStart: 0, rowCount: 2, weights}]},
            output: [destination]
          });
    // The symmetrization dispatch binds offsets, neighbors, source and destination only.
    expect(() => transform.getCommandNodes(setup.graph)).not.toThrow();
  } finally {
    setup.device.destroy();
  }
});

// W5-05: validation must not write published neighborCounts.
it.each([
  'whole',
  'chunk'
] as const)('W5-05 %s validation refuses flags over published neighborCounts', layout => {
  const setup = createReviewSetup();
  try {
    const backing = setup.view('uint32', 66);
    const neighborCounts = setup.graph.createDataView(backing.buffer, {
      format: 'uint32',
      length: 2
    });
    const counted = {...createPair(setup), neighborCounts};
    const weights =
      layout === 'whole'
        ? counted
        : {rowCount: 2, chunks: [{rowStart: 0, rowCount: 2, weights: counted}]};
    const construct = (flags: GraphDataView<'uint32'>) =>
      new GPUSpatialWeightsValidation<GPUSpatialWeights | GPUChunkedSpatialWeights>({
        weights,
        output: {flags, status: {invalidCount: setup.view('uint32', 1)}}
      });
    expect(() => construct(neighborCounts)).toThrow(/overlap/);
    const partial = setup.graph.createDataView(backing.buffer, {
      format: 'uint32',
      byteOffset: 4,
      length: 2
    });
    expect(() => construct(partial)).toThrow(/overlap/);
    // A disjoint view of the same buffer is legal (the row kernel binds flags, not the counts).
    const flags = setup.graph.createDataView(backing.buffer, {
      format: 'uint32',
      byteOffset: 256,
      length: 2
    });
    const validation = construct(flags);
    const described = validation.outputWeights;
    const published = (
      'chunks' in described ? described.chunks[0].weights : described
    ) as typeof counted;
    expect(published.neighborCounts === neighborCounts).toBe(true);
    expect(() => validation.getCommandNodes(setup.graph)).not.toThrow();
  } finally {
    setup.device.destroy();
  }
});

// W5-06: double normalization is kept only when every slot provably survives.
it.each([
  // Neighbors [2, 0] with two columns: slot 0 -> 2 is dropped, 0.5 of the unit mass survives.
  {
    name: 'out-of-range square',
    neighbors: [2, 0],
    symmetric: 'unknown',
    columns: 2,
    retained: [0.5],
    offsets: [0, 1, 1],
    transform: 'unknown'
  },
  // One column keeps only 1 -> 0: again 0.5.
  {
    name: 'rectangular',
    neighbors: [1, 0],
    symmetric: true,
    columns: 1,
    retained: [0.5],
    offsets: [0, 1],
    transform: 'unknown'
  },
  // Undeclared symmetry: the full-domain result is not provable from the handle.
  {
    name: 'undeclared full domain',
    neighbors: [1, 0],
    symmetric: 'unknown',
    columns: 2,
    retained: [0.5, 0.5],
    offsets: [0, 1, 2],
    transform: 'unknown'
  },
  // Declared symmetric and square: W' = W, all unit mass survives and double is proven.
  {
    name: 'declared symmetric square',
    neighbors: [1, 0],
    symmetric: true,
    columns: 2,
    retained: [0.5, 0.5],
    offsets: [0, 1, 2],
    transform: 'double'
  }
] as const)('W5-06 transpose double metadata: $name', fixture => {
  for (const chunked of [false, true]) {
    const setup = createReviewSetup();
    try {
      const information = {
        rowCount: 2,
        transform: 'double' as const,
        symmetric: fixture.symmetric
      };
      const weights = {
        offsets: setup.input(Uint32Array.of(0, 1, 2), 'uint32'),
        neighbors: setup.input(Uint32Array.from(fixture.neighbors), 'uint32'),
        weights: setup.input(Float32Array.of(0.5, 0.5), 'float32')
      };
      const output = setup.weights(fixture.columns, 2);
      const transpose = chunked
        ? new GPUChunkedSpatialWeightsTranspose({
            weights: {rowCount: 2, chunks: [{rowStart: 0, rowCount: 2, weights}], information},
            columnCount: fixture.columns,
            output
          })
        : new GPUSpatialWeightsTranspose({
            weights: {...weights, information},
            columnCount: fixture.columns,
            output
          });
      executeCommandNodesOnCPU(transpose.getCommandNodes(setup.graph), setup.memory);
      expect(Array.from(setup.memory.readUint32(output.offsets))).toEqual(fixture.offsets);
      const count = fixture.offsets[fixture.columns];
      expect(Array.from(setup.memory.readFloat32(output.weights, count))).toEqual(fixture.retained);
      expect(transpose.outputWeights.information?.transform).toBe(fixture.transform);
    } finally {
      setup.device.destroy();
    }
  }
});
