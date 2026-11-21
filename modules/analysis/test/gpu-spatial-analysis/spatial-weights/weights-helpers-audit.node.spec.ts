// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  createIslandAttachmentFallbackNodes,
  ISLAND_ATTACHMENT_NEAREST_DECLARATIONS
} from '../../../src/gpu-spatial-analysis/spatial-weights/island-attachment-fallback';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {executeWeightsAuditKernel} from './weights-audit-executor';
import {
  createNearestComparisonAuditNode,
  FALLBACK_ALL_NEAREST,
  FALLBACK_POSITIONS
} from './weights-helpers-audit-fixtures';
import {
  getKeyPairSortNodes,
  getKeyGroupNodes
} from '../../../src/gpu-spatial-analysis/spatial-weights/key-pair-grouping';
import {
  PAIR_HIGH,
  PAIR_LOW,
  PAIR_ORDER,
  PAIR_FLAGS,
  PAIR_GROUP_INDEX,
  PAIR_BOUNDARIES
} from './weights-helpers-audit-fixtures';
import {getSlotGroupingNodes} from '../../../src/gpu-spatial-analysis/spatial-weights/slot-grouping';
import {SORTED_SLOT_KEYS, COLUMN_OFFSETS} from './weights-helpers-audit-fixtures';
import {Buffer} from '@luma.gl/core';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {validateSpatialStatisticsBindings} from '../../../src/gpu-spatial-analysis/spatial-weights/spatial-statistics-bindings';
import {createCollectiveAuditNode} from './weights-helpers-audit-fixtures';
import {findStatisticsBarrierRisks} from './statistics-wgsl-audit';
import {createCountArithmeticAuditNode} from './weights-helpers-audit-fixtures';
import {
  ORDINARY_COUNTS,
  ORDINARY_COUNT_VALUES,
  ORDINARY_COUNT_BITS
} from './weights-helpers-audit-fixtures';
import {resolveStatisticsPopulationCount} from '../../../src/gpu-spatial-analysis/spatial-weights/spatial-statistics-population-output';
import {createRowAuditNode, ROW_NEIGHBORS, ROW_RESULT} from './weights-helpers-audit-fixtures';
import {validateSpatialStatisticsWritableDispatches} from '../../../src/gpu-spatial-analysis/spatial-weights/spatial-statistics-writable-dispatches';
import {
  createCsrStageAuditNodes,
  createContainsAuditNode,
  STAGE_LEFT,
  STAGE_RIGHT,
  SET_EXPECTED
} from './weights-helpers-audit-fixtures';
import type {GPUCommandNode} from '@luma.gl/graph/gpu-core';
import {createSortRowsAuditNodes} from './weights-helpers-audit-fixtures';
import {
  getWeightsLossSources,
  validateWeightsLossSources,
  createWeightsLossNodes,
  createWeightsIncompleteNodes
} from '../../../src/gpu-spatial-analysis/spatial-weights/spatial-weights-loss';
import {
  createRowLocalAuditNodes,
  getRowLocalAuditExpected,
  SELF_INPUT,
  SELF_WEIGHTS,
  SUBGRAPH_MASK
} from './weights-helpers-audit-fixtures';
import {computeSpatialWeightsAlgebraOnCPU} from '../../../src/gpu-spatial-analysis/spatial-weights/cpu-spatial-weights-algebra-operation';
import type {GPUSpatialWeightsCombineRule} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-algebra';
import {
  createSpaceTimeAuditNodes,
  SPACE_TIME_INPUT,
  SPACE_TIME_EXPECTED,
  SPACE_TIME_LAGS
} from './weights-helpers-audit-fixtures';
import {createSpatialWeightsSuppressionNodes} from '../../../src/gpu-spatial-analysis/spatial-weights/spatial-weights-suppression';
import {createStatisticsKernelNode} from '../../../src/gpu-spatial-analysis/spatial-weights/statistics-kernel-node';
import {createStatisticsGuardAuditNode} from './weights-helpers-audit-fixtures';
import {
  snapshotWeightsFields,
  snapshotWeightsPort,
  snapshotWeightsAlgebraProps,
  snapshotContiguityOperationProps,
  snapshotWeightsSpatialContext
} from '../../../src/gpu-spatial-analysis/spatial-weights/weights-construction-snapshot';
import {GPUSpatialWeightsSummary} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-summary';
import {GPUSpatialWeightsTransform} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-transform';
import {GPUSpatialWeightsTranspose} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-transpose';
import {GPUSpatialWeightsValidation} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-validation';
import type {GPUSpatialWeights} from '../../../src/gpu-spatial-analysis/spatial-weights/spatial-weights';
import {addWeightsAuditReadback} from './weights-helpers-audit-fixtures';

it('combine-rule oracle distinguishes min from left and max from right', () => {
  expect(getRowLocalAuditExpected('union', 'min').weights).not.toEqual(
    getRowLocalAuditExpected('union', 'left').weights
  );
  expect(getRowLocalAuditExpected('union', 'max').weights).not.toEqual(
    getRowLocalAuditExpected('union', 'right').weights
  );
});

it.each([
  'summary',
  'transpose',
  'transform'
] as const)('%s refuses writable aliases of other published ports', kind => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
      createTransientView(graph, `alias-${serial++}`, format, length);
    const weights = {
      offsets: view('uint32', 9),
      neighbors: view('uint32', 8),
      weights: view('float32', 8),
      distances: view('float32', 8)
    };
    const shared = view('uint32', 9);
    const create = () =>
      kind === 'summary'
        ? new GPUSpatialWeightsSummary({
            weights,
            output: {
              statistics: view('float32', 3),
              counts: graph.createDataView(shared.buffer, {format: 'uint32', length: 6}),
              cardinality: graph.createDataView(shared.buffer, {format: 'uint32', length: 8})
            }
          })
        : kind === 'transpose'
          ? new GPUSpatialWeightsTranspose({
              weights,
              output: {
                offsets: shared,
                neighbors: graph.createDataView(shared.buffer, {format: 'uint32', length: 8}),
                weights: view('float32', 8),
                distances: view('float32', 8)
              }
            })
          : new GPUSpatialWeightsTransform({
              operation: 'binary',
              weights,
              output: weights.distances
            });
    expect(create).toThrow(/overlap|disjoint|buffer|alias/);
  } finally {
    device.destroy();
  }
});

it('audit readback accepts storage-only transients and preserves exact words, view prefixes and tails', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const sourceBuffer = graph.createTransientBuffer({
      id: 'readback-source',
      byteLength: 64,
      usage: Buffer.STORAGE
    });
    const destinationBuffer = graph.createTransientBuffer({
      id: 'readback-destination',
      byteLength: 64,
      usage: Buffer.STORAGE
    });
    const source = graph.createDataView(sourceBuffer, {format: 'uint32', byteOffset: 4, length: 8});
    const destination = graph.createDataView(destinationBuffer, {
      format: 'uint32',
      byteOffset: 8,
      length: 10
    });
    let node!: GPUCommandNode;
    expect(() => {
      node = addWeightsAuditReadback(graph, 'readback', source, destination);
    }).not.toThrow();
    expect(sourceBuffer.usage).toBe(Buffer.STORAGE);
    expect(destinationBuffer.usage).toBe(Buffer.STORAGE);
    const [kernel] = getGPUCommandNodeKernelSources([node]);
    // Identity is a closed form, including uint32 values that lose bits through f32.
    const words = [0, 1, 0x80000000, 0xffffffff, 7, 23, 100, 0x7fffffff];
    const sourceValues = Uint32Array.from([99, ...words, 99]);
    const destinationValues = new Uint32Array(12).fill(99);
    executeWeightsAuditKernel(kernel, {sourceValues, destinationValues});
    expect(Array.from(destinationValues)).toEqual([99, 99, ...words, 99, 99]);
  } finally {
    device.destroy();
  }
});

/** Replace only the upstream scan and the out-of-scope scan-total reduction. */
function executeCsrStages(
  nodes: readonly GPUCommandNode[],
  inputs: Record<string, Uint32Array | Float32Array>,
  rows: number,
  capacity: number,
  initialOverflow = 99
) {
  const kernels = getGPUCommandNodeKernelSources(nodes);
  const buffers = {
    ...inputs,
    counts: new Uint32Array(rows),
    starts: new Uint32Array(rows),
    scanTotal: new Uint32Array(1),
    outOffsets: new Uint32Array(rows + 1).fill(99),
    overflow: Uint32Array.of(initialOverflow),
    totalNeighbors: Uint32Array.of(99)
  };
  executeWeightsAuditKernel(kernels.find(kernel => kernel.id.endsWith('-counts'))!, buffers);
  let total = 0;
  buffers.counts.forEach((count, index) => {
    buffers.starts[index] = total;
    total += count;
  });
  buffers.scanTotal[0] = total;
  executeWeightsAuditKernel(kernels.find(kernel => kernel.id.endsWith('-offsets'))!, buffers);
  const neighbors = new Uint32Array(capacity).fill(99);
  const weights = new Float32Array(capacity).fill(99);
  const distances = new Float32Array(capacity).fill(99);
  for (const [name, output] of [
    ['neighbors', neighbors],
    ['weights', weights],
    ['distances', distances]
  ] as const) {
    const kernel = kernels.find(kernel => kernel.id.endsWith(`-emit-${name}`));
    if (kernel) executeWeightsAuditKernel(kernel, {...buffers, outValues: output});
  }
  const sorting = kernels.find(kernel => kernel.id.endsWith('-sort-rows'));
  if (sorting) executeWeightsAuditKernel(sorting, {...buffers, outValues: neighbors});
  return {
    offsets: Array.from(buffers.outOffsets),
    neighbors: Array.from(neighbors),
    weights: Array.from(weights),
    distances: Array.from(distances),
    overflow: buffers.overflow[0],
    requiredCount: buffers.totalNeighbors[0]
  };
}

it('fallback comparator and batch cursors match independent nearest and Cartesian-batch oracles', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const positions = createTransientView(graph, 'positions', 'float32x2', 8);
    const result = createTransientView(graph, 'nearest', 'uint32', 8);
    const [comparison] = getGPUCommandNodeKernelSources([
      createNearestComparisonAuditNode(graph, positions, result)
    ]);
    for (const [shift, scale] of [
      [0, 1],
      [100, 2]
    ]) {
      const values = Float32Array.from(FALLBACK_POSITIONS, value => shift + scale * value);
      const nearest = new Uint32Array(8).fill(99);
      executeWeightsAuditKernel(comparison, {positions: values, result: nearest});
      expect(Array.from(nearest)).toEqual(FALLBACK_ALL_NEAREST);
    }
    const kernels = getGPUCommandNodeKernelSources(
      createIslandAttachmentFallbackNodes(graph, {
        id: 'fallback',
        positions,
        unresolved: createTransientView(graph, 'mask', 'uint32', 8),
        attachment: result,
        nearestDeclarations: ISLAND_ATTACHMENT_NEAREST_DECLARATIONS,
        queryBatchSize: 2,
        representativeBatchSize: 3
      })
    );
    const initialize = kernels.find(kernel => kernel.id === 'fallback-fallback-initialize')!;
    const advances = kernels.filter(kernel => kernel.id.includes('-fallback-advance-'));
    // ceil(8/2)*ceil(8/3)=12 conservative Cartesian batches. Five
    // unresolved rows occupy only three query batches; the remainder are empty.
    expect(advances).toHaveLength(12);
    const buffers = {
      cursor: new Uint32Array(5).fill(99),
      count: Uint32Array.of(5),
      positions: Float32Array.from(FALLBACK_POSITIONS)
    };
    executeWeightsAuditKernel(initialize, buffers);
    expect(Array.from(buffers.cursor)).toEqual([0, 0, 2, 3, 2]);
    for (const kernel of advances) executeWeightsAuditKernel(kernel, buffers);
    expect(Array.from(buffers.cursor)).toEqual([0, 5, 0, 3, 2]);
  } finally {
    device.destroy();
  }
});

it('weights spatial context captures frame getters once and copies coordinate tuples', () => {
  let reads = 0;
  const origin: [number, number] = [1, 2];
  const captured = snapshotWeightsSpatialContext({
    coordinateSpace: 'planar',
    metric: 'native',
    units: 'meters',
    frame: {
      id: 'frame',
      kind: 'crs',
      get origin() {
        reads++;
        return origin;
      }
    }
  });
  origin[0] = 99;
  expect(reads).toBe(1);
  expect(captured.frame?.origin).toEqual([1, 2]);
  expect(Object.isFrozen(captured.frame?.origin)).toBe(true);
});

it('weights construction snapshots copy every metadata layer while preserving opaque graph identities', () => {
  let reads = 0;
  const inherited = Object.create({
    get capacity() {
      reads++;
      return 8;
    }
  }) as {capacity: number; extra?: number};
  inherited.extra = 99;
  const fields = snapshotWeightsFields(inherited, ['capacity']);
  expect(fields.capacity).toBe(8);
  expect(reads).toBe(1);
  expect('extra' in fields).toBe(false);
  expect(Object.isFrozen(fields)).toBe(true);
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    for (let index = 0; index < 8; index++) {
      const view = createTransientView(graph, `snapshot-${index}`, 'uint32', 1);
      const provenance = {source: 'oracle', criterion: 'knn', k: 2, radius: index + 1, order: 1};
      const port = {
        offsets: view,
        neighbors: view,
        weights: view,
        status: {overflow: view},
        information: {rowCount: 8, transform: 'raw', symmetric: false, provenance}
      };
      const captured = snapshotWeightsPort(port);
      provenance.radius = 99;
      port.status.overflow = createTransientView(graph, `replacement-${index}`, 'uint32', 1);
      expect(captured.information.provenance.radius).toBe(index + 1);
      expect(captured.status.overflow).toBe(view);
      expect(captured.neighbors).toBe(view);
      expect(Object.isFrozen(captured.information.provenance)).toBe(true);
      const lags = [{lag: 1, weight: 2, includeSelf: true}];
      const props = {
        id: 'oracle',
        operation: 'spaceTime',
        weights: port,
        output: port,
        lags,
        periodCount: 3,
        rowCount: 24,
        capacity: 8
      };
      const algebra = snapshotWeightsAlgebraProps(props);
      lags[0].weight = 99;
      props.capacity = 99;
      expect(algebra.capacity).toBe(8);
      expect(algebra.lags[0].weight).toBe(2);
      expect(algebra.weights.neighbors).toBe(view);
      expect(Object.isFrozen(algebra.lags)).toBe(true);
      const polygons = {
        positions: 'positions',
        ringOffsets: 'rings',
        polygonOffsets: 'parts',
        featureOffsets: 'features'
      };
      const contiguity = snapshotContiguityOperationProps({
        criterion: 'rook',
        polygons,
        output: port,
        featureCount: 8,
        capacity: 8
      });
      polygons.positions = 'changed';
      expect(contiguity.polygons.positions).toBe('positions');
      expect(contiguity.output.neighbors).toBe(view);
      const chunks = [{rowStart: 0, rowCount: 1, weights: port}];
      const chunked = snapshotWeightsPort({rowCount: 1, chunks});
      chunks[0].rowStart = 99;
      expect(chunked.chunks[0].rowStart).toBe(0);
      expect(chunked.chunks[0].weights.neighbors).toBe(view);
    }
  } finally {
    device.destroy();
  }
});

it('summary transform transpose and validation retain construction-time ports after caller map mutation', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
      createTransientView(graph, `consumer-${serial++}`, format, length);
    const weights: GPUSpatialWeights = {
      offsets: view('uint32', 9),
      neighbors: view('uint32', 8),
      weights: view('float32', 8)
    };
    const neighbors = weights.neighbors;
    const summaryOutput = {
      statistics: view('float32', 3),
      counts: view('uint32', 6),
      cardinality: view('uint32', 8)
    };
    const statistics = summaryOutput.statistics;
    const transformProps = {operation: 'binary' as const, weights, output: view('float32', 8)};
    const transformed = transformProps.output;
    const transposeOutput = {
      offsets: view('uint32', 9),
      neighbors: view('uint32', 8),
      weights: view('float32', 8)
    };
    const transposeNeighbors = transposeOutput.neighbors;
    const validationOutput = {flags: view('uint32', 8), status: {invalidCount: view('uint32', 1)}};
    const invalid = validationOutput.status.invalidCount;
    const summary = new GPUSpatialWeightsSummary({weights, output: summaryOutput});
    const transform = new GPUSpatialWeightsTransform(transformProps);
    const transpose = new GPUSpatialWeightsTranspose({weights, output: transposeOutput});
    const validation = new GPUSpatialWeightsValidation({weights, output: validationOutput});
    weights.neighbors = view('uint32', 8);
    summaryOutput.statistics = view('float32', 3);
    transformProps.output = view('float32', 8);
    transposeOutput.neighbors = view('uint32', 8);
    validationOutput.status.invalidCount = view('uint32', 1);
    for (const contributor of [summary, transform, transpose, validation]) {
      expect((contributor.props.weights as GPUSpatialWeights).neighbors).toBe(neighbors);
      contributor.getCommandNodes(graph);
    }
    expect(summary.props.output.statistics).toBe(statistics);
    expect(transform.props.output).toBe(transformed);
    expect(transpose.props.output.neighbors).toBe(transposeNeighbors);
    expect(validation.props.output.status.invalidCount).toBe(invalid);
  } finally {
    device.destroy();
  }
});

it('cooperative kernel guard excludes padded 2D workgroups without varying barrier participation', () => {
  const device = createNullWebGPUDevice();
  try {
    Object.defineProperty(device.limits, 'maxComputeWorkgroupsPerDimension', {
      value: 3,
      configurable: true
    });
    const graph = new GPUCommandGraph(device);
    const node = createStatisticsGuardAuditNode(
      graph,
      createTransientView(graph, 'guard-result', 'uint32', 9),
      32
    );
    expect(node.workload?.maximumWorkgroupCount).toBe(9); // ceil(32/4)=8, packed into 3x3.
    const [kernel] = getGPUCommandNodeKernelSources([node]);
    expect(findStatisticsBarrierRisks(kernel.source)).toEqual([]);
    const result = new Uint32Array(9).fill(99);
    executeWeightsAuditKernel(kernel, {result}, [3, 3, 1]);
    expect(Array.from(result)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 99]);
  } finally {
    device.destroy();
  }
});

it('suppression independently covers all six fields, both loss sources, logical limits and untouched tails', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
      createTransientView(graph, `suppression-${serial++}`, format, length);
    const overflow = view('uint32', 1);
    const candidate = view('uint32', 1);
    const weights = {
      rowCount: 8,
      status: {overflow},
      chunks: [
        {
          rowStart: 0,
          rowCount: 8,
          weights: {
            offsets: view('uint32', 9),
            neighbors: view('uint32', 8),
            weights: view('float32', 8),
            status: {candidateOverflow: candidate}
          }
        }
      ]
    };
    const output = {
      output: view('float32', 10),
      statistics: view('float32', 5),
      summary: view('float32', 6),
      counts: view('uint32', 8),
      cardinality: view('uint32', 10),
      permutationCount: view('uint32', 3)
    };
    const lengths = {
      output: 8,
      statistics: 3,
      summary: 4,
      counts: 6,
      cardinality: 8,
      permutationCount: 1
    };
    const kernels = getGPUCommandNodeKernelSources(
      createSpatialWeightsSuppressionNodes(graph, 'suppress', 'audit', weights, output, lengths)
    );
    expect(kernels).toHaveLength(2);
    for (const active of [false, true])
      for (const kernel of kernels) {
        const buffers: Record<string, Float32Array | Uint32Array> = {
          loss: Uint32Array.of(active ? 7 : 0)
        };
        for (const [name, port] of Object.entries(output))
          buffers[name] =
            port.format === 'float32'
              ? Float32Array.from({length: port.length}, (_, index) => index + 1)
              : Uint32Array.from({length: port.length}, (_, index) => index + 1);
        executeWeightsAuditKernel(kernel, buffers);
        for (const [name, port] of Object.entries(output)) {
          const logical = lengths[name as keyof typeof lengths];
          const values = Array.from(buffers[name]);
          if (active) {
            if (port.format === 'float32')
              expect(values.slice(0, logical).every(Number.isNaN)).toBe(true);
            else expect(values.slice(0, logical)).toEqual(Array(logical).fill(0));
          } else
            expect(values.slice(0, logical)).toEqual(
              Array.from({length: logical}, (_, index) => index + 1)
            );
          expect(values.slice(logical)).toEqual(
            Array.from({length: port.length - logical}, (_, index) => logical + index + 1)
          );
        }
      }
  } finally {
    device.destroy();
  }
});

it('cooperative kernel guard skips zero-invocation dispatches', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const result = createTransientView(graph, 'zero-result', 'uint32', 1);
    const [kernel] = getGPUCommandNodeKernelSources([
      createStatisticsKernelNode(graph, {
        id: 'zero-guard',
        operation: 'audit',
        invocationCount: 0,
        workgroupSize: 4,
        guardIndex: false,
        bindings: [{name: 'result', view: result, type: 'u32', access: 'read_write'}],
        body: 'workgroupBarrier(); if (localInvocationIndex == 0u) { result[resultOffset] = 7u; }'
      })
    ]);
    const values = Uint32Array.of(99);
    executeWeightsAuditKernel(kernel, {result: values});
    expect(Array.from(values)).toEqual([99]);
  } finally {
    device.destroy();
  }
});

it('space-time stages and CPU twin equal independent Kronecker products for both layouts, lag signs and diagonal replacement', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const layout of ['cell-major', 'period-major'] as const)
      for (const scale of [1, 2])
        for (const capacity of [10, 48]) {
          const graph = new GPUCommandGraph(device);
          let serial = 0;
          const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
            createTransientView(graph, `st-${serial++}`, format, length);
          const output = {
            offsets: view('uint32', 13),
            neighbors: view('uint32', capacity),
            weights: view('float32', capacity),
            status: {overflow: view('uint32', 1), requiredCount: view('uint32', 1)}
          };
          const actual = executeCsrStages(
            createSpaceTimeAuditNodes(
              graph,
              {
                offsets: view('uint32', 5),
                neighbors: view('uint32', 5),
                weights: view('float32', 5)
              },
              output,
              layout,
              scale
            ),
            {
              baseOffsets: Uint32Array.from(SPACE_TIME_INPUT.offsets),
              baseNeighbors: Uint32Array.from(SPACE_TIME_INPUT.neighbors),
              baseWeights: Float32Array.from(SPACE_TIME_INPUT.weights)
            },
            12,
            capacity
          );
          const expected = SPACE_TIME_EXPECTED[layout];
          const used = Math.min(capacity, 43);
          expect(actual.offsets).toEqual(expected.offsets.map(value => Math.min(capacity, value)));
          expect(actual.neighbors.slice(0, used)).toEqual(expected.neighbors.slice(0, used));
          expect(actual.weights.slice(0, used)).toEqual(
            expected.weights.slice(0, used).map(value => value * scale)
          );
          expect(actual.requiredCount).toBe(43);
          expect(actual.overflow).toBe(Number(capacity < 43));
          const source = {
            offsets: Uint32Array.from(SPACE_TIME_INPUT.offsets),
            neighbors: Uint32Array.from(SPACE_TIME_INPUT.neighbors),
            weights: Float32Array.from(SPACE_TIME_INPUT.weights)
          };
          const cpu = computeSpatialWeightsAlgebraOnCPU(
            {weights: source},
            {
              operation: 'spaceTime',
              capacity,
              periodCount: 3,
              layout,
              lags: SPACE_TIME_LAGS.map(lag => ({...lag, weight: lag.weight * scale}))
            }
          );
          expect(Array.from(cpu.offsets)).toEqual(actual.offsets);
          expect(Array.from(cpu.neighbors).slice(0, used)).toEqual(
            expected.neighbors.slice(0, used)
          );
          expect(Array.from(cpu.weights).slice(0, used)).toEqual(
            expected.weights.slice(0, used).map(value => value * scale)
          );
          expect(Array.from(cpu.status.overflow)).toEqual([Number(capacity < 43)]);
          expect(Array.from(cpu.status.requiredCount)).toEqual([43]);
          expect(Array.from(cpu.status.candidateOverflow)).toEqual([0]);
        }
  } finally {
    device.destroy();
  }
});

it('row-local helpers and CPU twin match every rule, optional distance, self replacement, mask and chunk oracle', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const operation of [
      'union',
      'intersection',
      'difference',
      'symmetric-difference',
      'selfWeight',
      'subgraph'
    ] as const) {
      for (const rule of (operation === 'selfWeight' || operation === 'subgraph'
        ? ['sum']
        : [
            'left',
            'right',
            'sum',
            'min',
            'max',
            'product',
            'binary'
          ]) as GPUSpatialWeightsCombineRule[]) {
        for (const scale of [1, 2])
          for (const chunked of [false, true])
            for (const perRow of operation === 'selfWeight' ? [false, true] : [true]) {
              const graph = new GPUCommandGraph(device);
              let serial = 0;
              const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
                createTransientView(graph, `local-${serial++}`, format, length);
              const rowStart = chunked ? 4 : 0;
              const rows = chunked ? 4 : 8;
              const capacity = 7;
              const original = operation === 'selfWeight' ? SELF_INPUT : STAGE_LEFT;
              const slice = (source: typeof STAGE_LEFT) => {
                const begin = source.offsets[rowStart];
                const end = source.offsets[rowStart + rows];
                return {
                  offsets: source.offsets
                    .slice(rowStart, rowStart + rows + 1)
                    .map(value => value - begin),
                  neighbors: [...source.neighbors.slice(begin, end), 0],
                  weights: [...source.weights.slice(begin, end).map(value => value * scale), 0],
                  distances: [...source.distances.slice(begin, end), 0]
                };
              };
              const left = slice(original);
              const right = slice(STAGE_RIGHT);
              const port = (source: typeof left) => ({
                offsets: view('uint32', rows + 1),
                neighbors: view('uint32', source.neighbors.length),
                weights: view('float32', source.weights.length),
                distances: view('float32', source.distances.length)
              });
              const output = {
                offsets: view('uint32', rows + 1),
                neighbors: view('uint32', capacity),
                weights: view('float32', capacity),
                distances: view('float32', capacity),
                status: {overflow: view('uint32', 1), requiredCount: view('uint32', 1)}
              };
              const actual = executeCsrStages(
                createRowLocalAuditNodes(
                  graph,
                  port(left),
                  port(right),
                  view('uint32', 8),
                  perRow ? view('float32', 8) : 0.5 * scale,
                  output,
                  operation,
                  rule,
                  chunked ? {rowStart: 4, totalRows: 8} : undefined
                ),
                {
                  leftOffsets: Uint32Array.from(left.offsets),
                  leftNeighbors: Uint32Array.from(left.neighbors),
                  leftWeights: Float32Array.from(left.weights),
                  leftDistances: Float32Array.from(left.distances),
                  rightOffsets: Uint32Array.from(right.offsets),
                  rightNeighbors: Uint32Array.from(right.neighbors),
                  rightWeights: Float32Array.from(right.weights),
                  rightDistances: Float32Array.from(right.distances),
                  sourceOffsets: Uint32Array.from(left.offsets),
                  sourceNeighbors: Uint32Array.from(left.neighbors),
                  sourceWeights: Float32Array.from(left.weights),
                  sourceDistances: Float32Array.from(left.distances),
                  mask: Uint32Array.from(SUBGRAPH_MASK),
                  selfWeights: Float32Array.from(SELF_WEIGHTS, value => value * scale)
                },
                rows,
                capacity
              );
              const expected = getRowLocalAuditExpected(operation, rule, scale, perRow);
              const begin = expected.offsets[rowStart];
              const end = expected.offsets[rowStart + rows];
              const used = Math.min(capacity, end - begin);
              expect(actual.offsets).toEqual(
                expected.offsets
                  .slice(rowStart, rowStart + rows + 1)
                  .map(value => Math.min(value - begin, capacity))
              );
              expect(actual.neighbors.slice(0, used)).toEqual(
                expected.neighbors.slice(begin, begin + used)
              );
              expect(actual.weights.slice(0, used)).toEqual(
                expected.weights.slice(begin, begin + used)
              );
              expect(actual.distances.slice(0, used)).toEqual(
                expected.distances.slice(begin, begin + used)
              );
              expect(actual.requiredCount).toBe(end - begin);
              expect(actual.overflow).toBe(Number(end - begin > capacity));
              if (!chunked) {
                const cpuRows = (source: typeof STAGE_LEFT) => ({
                  offsets: Uint32Array.from(source.offsets),
                  neighbors: Uint32Array.from(source.neighbors),
                  weights: Float32Array.from(source.weights, value => value * scale),
                  distances: Float32Array.from(source.distances)
                });
                const cpu = computeSpatialWeightsAlgebraOnCPU(
                  {
                    left: cpuRows(original),
                    right: cpuRows(STAGE_RIGHT),
                    weights: cpuRows(original),
                    mask: Uint32Array.from(SUBGRAPH_MASK),
                    selfWeights: Float32Array.from(SELF_WEIGHTS, value => value * scale)
                  },
                  operation === 'selfWeight'
                    ? {
                        operation,
                        capacity,
                        distances: true,
                        ...(!perRow ? {selfWeight: 0.5 * scale} : {})
                      }
                    : operation === 'subgraph'
                      ? {operation, capacity, distances: true}
                      : {operation, capacity, distances: true, weightRule: rule}
                );
                expect(Array.from(cpu.offsets)).toEqual(actual.offsets);
                expect(Array.from(cpu.neighbors).slice(0, used)).toEqual(
                  expected.neighbors.slice(0, used)
                );
                expect(Array.from(cpu.weights).slice(0, used)).toEqual(
                  expected.weights.slice(0, used)
                );
                expect(Array.from(cpu.distances!).slice(0, used)).toEqual(
                  expected.distances.slice(0, used)
                );
                expect(Array.from(cpu.status.overflow)).toEqual([actual.overflow]);
                expect(Array.from(cpu.status.requiredCount)).toEqual([end]);
                expect(Array.from(cpu.status.candidateOverflow)).toEqual([0]);
              }
            }
      }
    }
  } finally {
    device.destroy();
  }
}, 120000);

it('loss aggregation deduplicates sources, ORs across the seven-input batch boundary and preserves merge state', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const inputs = Array.from({length: 8}, (_, index) =>
      createTransientView(graph, `loss-${index}`, 'uint32', 1)
    );
    const output = createTransientView(graph, 'loss-output', 'uint32', 1);
    const sources = [
      {
        status: {overflow: inputs[0], candidateOverflow: inputs[1]},
        chunks: inputs.slice(2).map(view => ({weights: {status: {overflow: view}}}))
      },
      {status: {overflow: inputs[0]}}
    ];
    const collected = getWeightsLossSources(sources);
    expect(collected.overflow).toEqual([inputs[0], ...inputs.slice(2)]);
    expect(collected.candidateOverflow).toEqual([inputs[1]]);
    expect(validateWeightsLossSources('audit', sources)).toEqual([
      ...collected.overflow,
      ...collected.candidateOverflow
    ]);
    expect(createWeightsIncompleteNodes(graph, 'none', [])).toEqual({view: undefined, nodes: []});
    const incomplete = createWeightsIncompleteNodes(graph, 'combined', sources);
    expect(incomplete.view?.length).toBe(1);
    expect(incomplete.nodes).toHaveLength(2);
    for (const merge of [false, true]) {
      const kernels = getGPUCommandNodeKernelSources(
        createWeightsLossNodes(graph, `or-${merge}`, inputs, output, merge)
      );
      expect(kernels).toHaveLength(2);
      for (const active of [false, true]) {
        const result = Uint32Array.of(1);
        kernels.forEach((kernel, batch) => {
          const buffers: Record<string, Uint32Array> = {loss: result};
          inputs.slice(batch * 7, batch * 7 + 7).forEach((_, index) => {
            buffers[`lossInput${index}`] = Uint32Array.of(active && batch === 1 ? 7 : 0);
          });
          executeWeightsAuditKernel(kernel, buffers);
        });
        expect(result[0]).toBe(merge || active ? 1 : 0);
      }
    }
    expect(createWeightsLossNodes(graph, 'keep-empty', [], output, true)).toEqual([]);
    expect(() => createWeightsLossNodes(graph, 'alias', [output], output)).toThrow(/merge/);
  } finally {
    device.destroy();
  }
});

it('binary frontier CSR sorting orders each row and preserves its constant weight attribute', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const output = {
      offsets: createTransientView(graph, 'o', 'uint32', 9),
      neighbors: createTransientView(graph, 'n', 'uint32', 24),
      weights: createTransientView(graph, 'w', 'float32', 24),
      status: {
        overflow: createTransientView(graph, 'overflow', 'uint32', 1),
        requiredCount: createTransientView(graph, 'count', 'uint32', 1)
      }
    };
    const actual = executeCsrStages(createSortRowsAuditNodes(graph, output), {}, 8, 24);
    expect(actual.offsets).toEqual([0, 3, 6, 9, 12, 15, 18, 21, 24]);
    expect(actual.neighbors).toEqual(Array.from({length: 8}, () => [0, 1, 2]).flat());
    expect(actual.weights).toEqual(Array(24).fill(1));
    expect(actual.overflow).toBe(0);
    expect(actual.requiredCount).toBe(24);
  } finally {
    device.destroy();
  }
});

it('CSR stages and all merge visitors match independent set literals, clamping and inherited overflow', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const operation of [
      undefined,
      'union',
      'intersection',
      'difference',
      'symmetric-difference'
    ] as const) {
      for (const capacity of [7, 20]) {
        for (const initializes of [false, true]) {
          const graph = new GPUCommandGraph(device);
          let serial = 0;
          const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
            createTransientView(graph, `csr-${serial++}`, format, length);
          const source = (count: number) => ({
            offsets: view('uint32', 9),
            neighbors: view('uint32', count),
            weights: view('float32', count)
          });
          const output = {
            offsets: view('uint32', 9),
            neighbors: view('uint32', capacity),
            weights: view('float32', capacity),
            status: {overflow: view('uint32', 1), requiredCount: view('uint32', 1)}
          };
          const actual = executeCsrStages(
            createCsrStageAuditNodes(graph, source(10), source(9), output, operation, initializes),
            {
              // The interpreter eagerly reads select/&& operands and throws on
              // OOB, unlike WGSL's robust buffer reads. The extra unused word
              // models that ignored zero read; CSR offsets still bound all rows.
              leftOffsets: Uint32Array.from(STAGE_LEFT.offsets),
              leftNeighbors: Uint32Array.from([...STAGE_LEFT.neighbors, 0]),
              rightOffsets: Uint32Array.from(STAGE_RIGHT.offsets),
              rightNeighbors: Uint32Array.from([...STAGE_RIGHT.neighbors, 0])
            },
            8,
            capacity,
            1
          );
          const expected = operation ? SET_EXPECTED[operation] : STAGE_LEFT;
          expect(actual.offsets).toEqual(expected.offsets.map(value => Math.min(value, capacity)));
          const used = Math.min(capacity, expected.neighbors.length);
          expect(actual.neighbors.slice(0, used)).toEqual(expected.neighbors.slice(0, used));
          const expectedWeights = expected.offsets
            .slice(0, -1)
            .flatMap((begin, row) =>
              expected.neighbors
                .slice(begin, expected.offsets[row + 1])
                .map(id => (row + 1) * 10 + id)
            );
          expect(actual.weights).toEqual([
            ...expectedWeights.slice(0, used),
            ...Array(capacity - used).fill(99)
          ]);
          expect(actual.overflow).toBe(
            !initializes || expected.neighbors.length > capacity ? 1 : 0
          );
          expect(actual.requiredCount).toBe(expected.neighbors.length);
        }
      }
    }
    const graph = new GPUCommandGraph(device);
    const [contains] = getGPUCommandNodeKernelSources([
      createContainsAuditNode(
        graph,
        createTransientView(graph, 'keys', 'uint32', 8),
        createTransientView(graph, 'queries', 'uint32', 8),
        createTransientView(graph, 'answers', 'uint32', 8)
      )
    ]);
    const result = new Uint32Array(8).fill(99);
    executeWeightsAuditKernel(contains, {
      keys: Uint32Array.of(0, 2, 2, 4, 8, 10, 12, 14, 0),
      queries: Uint32Array.of(0, 1, 2, 7, 8, 9, 14, 15),
      result
    });
    expect(Array.from(result)).toEqual([1, 0, 1, 0, 1, 0, 1, 0]);
  } finally {
    device.destroy();
  }
});

it('writable dispatch validation distinguishes logical overlap from co-bound aligned overlap', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const packed = createTransientView(graph, 'packed-outputs', 'uint32', 128);
    const fields = Object.fromEntries(
      Array.from({length: 8}, (_, index) => [
        `field${index}`,
        graph.createDataView(packed.buffer, {format: 'uint32', byteOffset: 4 * index, length: 1})
      ])
    );
    expect(() =>
      validateSpatialStatisticsWritableDispatches(
        'audit',
        fields,
        Object.keys(fields).map(name => [name])
      )
    ).not.toThrow();
    expect(() =>
      validateSpatialStatisticsWritableDispatches('audit', fields, [['field0', 'field1']])
    ).toThrow(/aligned binding byte ranges/);
    expect(() =>
      validateSpatialStatisticsWritableDispatches(
        'audit',
        {...fields, duplicate: fields['field0']},
        []
      )
    ).toThrow(/logical byte ranges/);
    const aligned = graph.createDataView(packed.buffer, {
      format: 'uint32',
      byteOffset: 256,
      length: 1
    });
    expect(() =>
      validateSpatialStatisticsWritableDispatches('audit', {first: fields['field0'], aligned}, [
        ['first', 'aligned']
      ])
    ).not.toThrow();
    const physical = device.createBuffer({byteLength: 512, usage: Buffer.STORAGE});
    try {
      const first = importGraphBuffer(graph, 'first-output', physical, 'uint32', 128);
      const second = importGraphBuffer(graph, 'second-output', physical, 'uint32', 128);
      expect(() =>
        validateSpatialStatisticsWritableDispatches('audit', {first, second}, [])
      ).toThrow(/import it once/);
    } finally {
      physical.destroy();
    }
  } finally {
    device.destroy();
  }
});

it('row dispatch preserves global rows, chunk origin, aligned prefixes and logical capacity', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const packedNeighbors = createTransientView(graph, 'neighbors', 'uint32', 12);
    const neighbors = graph.createDataView(packedNeighbors.buffer, {
      format: 'uint32',
      byteOffset: 8,
      length: 8
    });
    const packedResult = createTransientView(graph, 'result', 'uint32', 15);
    const result = graph.createDataView(packedResult.buffer, {
      format: 'uint32',
      byteOffset: 4,
      length: 14
    });
    const [kernel] = getGPUCommandNodeKernelSources([createRowAuditNode(graph, neighbors, result)]);
    const parameters = new Uint32Array(64);
    parameters.set([3, 8, 8, 0, 2, 1]);
    parameters[36] = 3;
    const output = new Uint32Array(15).fill(99);
    executeWeightsAuditKernel(kernel, {
      neighbors: Uint32Array.from([88, 88, ...ROW_NEIGHBORS]),
      result: output,
      graphChunkGroupParameters: parameters
    });
    // Eight global rows 3..10; two neighbor-prefix words do not count as capacity.
    expect(Array.from(output)).toEqual(ROW_RESULT);
  } finally {
    device.destroy();
  }
});

it('population count resolves either spelling to exactly the supplied view and refuses ambiguous aliases', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    for (let index = 0; index < 8; index++) {
      const view = createTransientView(graph, `population-${index}`, 'uint32', 1);
      expect(resolveStatisticsPopulationCount('audit', {includedCount: view})).toBe(view);
      expect(resolveStatisticsPopulationCount('audit', {count: view})).toBe(view);
      expect(() =>
        resolveStatisticsPopulationCount('audit', {count: view, includedCount: view})
      ).toThrow(/provide only one/);
    }
    expect(resolveStatisticsPopulationCount('audit', {})).toBeUndefined();
  } finally {
    device.destroy();
  }
});

it('ordinary count arithmetic matches an independent rational IEEE rounding oracle bit for bit', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const node = createCountArithmeticAuditNode(
      graph,
      createTransientView(graph, 'counts', 'uint32', 8),
      createTransientView(graph, 'values', 'float32', 8),
      createTransientView(graph, 'result', 'uint32', 32)
    );
    const [kernel] = getGPUCommandNodeKernelSources([node]);
    const result = new Uint32Array(32).fill(99);
    executeWeightsAuditKernel(kernel, {
      counts: Uint32Array.from(ORDINARY_COUNTS),
      values: Float32Array.from(ORDINARY_COUNT_VALUES),
      result
    });
    expect(Array.from(result)).toEqual(ORDINARY_COUNT_BITS);
  } finally {
    device.destroy();
  }
});

it('collective control validates its full uniform shape and aligned origin before lowering', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const values = createTransientView(graph, 'values', 'float32', 32);
    const result = createTransientView(graph, 'result', 'float32', 10);
    const packed = createTransientView(graph, 'control', 'uint32', 128);
    expect(() =>
      createCollectiveAuditNode(
        graph,
        values,
        result,
        graph.createDataView(packed.buffer, {format: 'uint32', byteOffset: 4, length: 64})
      )
    ).toThrow(/control.byteOffset/);
    expect(() =>
      createCollectiveAuditNode(
        graph,
        values,
        result,
        createTransientView(graph, 'short', 'uint32', 1)
      )
    ).toThrow(/control.length/);
    const foreign = createTransientView(new GPUCommandGraph(device), 'foreign', 'uint32', 64);
    expect(() => createCollectiveAuditNode(graph, values, result, foreign)).toThrow(/graph/);
    Object.defineProperty(device.limits, 'maxUniformBufferBindingSize', {
      value: 256,
      configurable: true
    });
    expect(() => createCollectiveAuditNode(graph, values, result, packed)).toThrow(
      /control binding byteLength/
    );
    const valid = graph.createDataView(packed.buffer, {
      format: 'uint32',
      byteOffset: 256,
      length: 64
    });
    const node = createCollectiveAuditNode(graph, values, result, valid);
    expect(node.workload?.maximumInvocationCount).toBe(32);
    expect(node.workload?.maximumWorkgroupCount).toBe(8);
    expect(node.workload?.readByteLength).toBe(128);
    expect(node.workload?.writeByteLength).toBe(40);
    const [kernel] = getGPUCommandNodeKernelSources([node]);
    expect(findStatisticsBarrierRisks(kernel.source)).toEqual([]);
  } finally {
    device.destroy();
  }
});

it('statistics binding preflight checks nested views, aligned prefixes and physical import identity', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {
      value: 256,
      configurable: true
    });
    const views = Array.from({length: 8}, (_, index) =>
      createTransientView(graph, `nested-${index}`, 'uint32', 2)
    );
    const cyclic: Record<string, unknown> = {
      chunks: views.map(view => ({weights: {view}})),
      contents: new Uint32Array(1000)
    };
    cyclic['cycle'] = cyclic;
    expect(() =>
      validateSpatialStatisticsBindings(graph, 'audit', cyclic, {scratch: 256})
    ).not.toThrow();
    expect(() => validateSpatialStatisticsBindings(graph, 'audit', cyclic, {scratch: 257})).toThrow(
      /audit scratch needs 257 bytes/
    );
    const packed = createTransientView(graph, 'packed', 'uint32', 128);
    const prefixed = graph.createDataView(packed.buffer, {
      format: 'uint32',
      byteOffset: 252,
      length: 2
    });
    // Two logical words plus the 252-byte aligned prefix require 260 bytes.
    expect(() => validateSpatialStatisticsBindings(graph, 'audit', {nested: {prefixed}})).toThrow(
      /260 bytes/
    );
    const physical = device.createBuffer({byteLength: 32, usage: Buffer.STORAGE});
    try {
      const first = importGraphBuffer(graph, 'first-import', physical, 'uint32', 8);
      const second = importGraphBuffer(graph, 'second-import', physical, 'uint32', 8);
      expect(() =>
        validateSpatialStatisticsBindings(graph, 'audit', {first, repeated: first})
      ).not.toThrow();
      expect(() => validateSpatialStatisticsBindings(graph, 'audit', {first, second})).toThrow(
        /import it once/
      );
    } finally {
      physical.destroy();
    }
  } finally {
    device.destroy();
  }
});

it('slot-group offsets include empty columns and exclude every unused tail key', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const grouping = getSlotGroupingNodes(graph, {
      id: 'slots',
      operation: 'audit',
      columns: 5,
      slotKeys: createTransientView(graph, 'keys', 'uint32', 8),
      slotIndices: createTransientView(graph, 'ids', 'uint32', 8),
      columnOffsets: createTransientView(graph, 'offsets', 'uint32', 6)
    });
    const kernel = getGPUCommandNodeKernelSources(grouping.nodes).find(
      kernel => kernel.id === 'slots-column-offsets'
    )!;
    const offsets = new Uint32Array(6).fill(99);
    executeWeightsAuditKernel(kernel, {
      sortedKeys: Uint32Array.from(SORTED_SLOT_KEYS),
      segmentOffsets: offsets
    });
    expect(Array.from(offsets)).toEqual(COLUMN_OFFSETS);
  } finally {
    device.destroy();
  }
});

it('pair grouping matches independent lexicographic runs and gives each invalid item its own group', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const high = createTransientView(graph, 'high', 'uint32', 8);
    const low = createTransientView(graph, 'low', 'uint32', 8);
    const sorted = getKeyPairSortNodes(graph, 'pair', 'audit', 8, high, low);
    const groups = getKeyGroupNodes(graph, 'groups', 'audit', 8, high, low, sorted.sortedItems);
    const kernels = getGPUCommandNodeKernelSources([...sorted.nodes, ...groups.nodes]);
    for (const shift of [0, 8]) {
      const buffers = {
        keyHigh: Uint32Array.from(PAIR_HIGH, value =>
          value === 0xffffffff ? value : value + shift
        ),
        keyLow: Uint32Array.from(PAIR_LOW, value => (value === 0xffffffff ? value : value + shift)),
        itemIds: new Uint32Array(8).fill(99),
        sortedItems: Uint32Array.from(PAIR_ORDER),
        flags: new Uint32Array(8).fill(99),
        groupIndex: Uint32Array.from(PAIR_GROUP_INDEX),
        groupStarts: new Uint32Array(9).fill(99)
      };
      executeWeightsAuditKernel(kernels.find(kernel => kernel.id === 'pair-iota')!, buffers);
      expect(Array.from(buffers.itemIds)).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
      executeWeightsAuditKernel(kernels.find(kernel => kernel.id === 'groups-run-flags')!, buffers);
      expect(Array.from(buffers.flags)).toEqual(PAIR_FLAGS);
      // Upstream inclusive scan is supplied by its definition; the GPU test
      // below also exercises the actual sort and scan implementations.
      executeWeightsAuditKernel(
        kernels.find(kernel => kernel.id === 'groups-group-starts')!,
        buffers
      );
      expect(Array.from(buffers.groupStarts)).toEqual([...PAIR_BOUNDARIES, 99, 99]);
    }
  } finally {
    device.destroy();
  }
});
