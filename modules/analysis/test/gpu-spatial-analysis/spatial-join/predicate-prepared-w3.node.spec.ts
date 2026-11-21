// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {
  GPUSpatialJoinPrepared,
  getGPUSpatialJoinPreparedMemoryPlan
} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-join-prepared';
import {Buffer} from '@luma.gl/core';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUSpatialPredicateJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-predicate-join';
import {decodeGPUSpatialJoinGeometry} from '../../../src/gpu-spatial-analysis/spatial-join/spatial-join-geometry';
import {packGPUSpatialRelatePattern} from '../../../src/gpu-spatial-analysis/spatial-join/spatial-relate-types';
import {getGPUSpatialPredicateJoinExecutionPlan} from '../../../src/gpu-spatial-analysis/spatial-join/spatial-predicate-route-plan';
import {getParsedKernelSources} from '../../utils/gpu-contributor-test-utils';
import {getGPUPartitionMemoryPlan} from '../../../src/analysis-core/contracts/partitioning';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

function createFixture() {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const scalar = (name: string) => createTransientView(graph, name, 'uint32', 1);
  const pairs = {
    leftRows: createTransientView(graph, 'left-rows', 'uint32', 9),
    rightRows: createTransientView(graph, 'right-rows', 'uint32', 9),
    status: {count: scalar('count'), overflow: scalar('overflow')}
  };
  const props = {
    left: {kind: 'points' as const, positions: createTransientView(graph, 'left', 'float32x2', 1)},
    right: {
      kind: 'points' as const,
      positions: createTransientView(graph, 'right', 'float32x2', 9)
    },
    predicate: 'intersects' as const,
    output: {pairs}
  };
  return {graph, scalar, pairs, props};
}

it('GEO-103 exposes typed graph-owned candidate loss and preserves supplied destinations', () => {
  const {props, scalar, pairs} = createFixture();
  const join = new GPUSpatialPredicateJoin(props);
  expect(join.candidateCapacity).toBe(8);
  expect(join.outputPairs?.status.candidateOverflow).toBeDefined();
  expect(join.outputPairs?.status.count).toBe(pairs.status.count);
  const loss = scalar('supplied-loss');
  const supplied = new GPUSpatialPredicateJoin({
    ...props,
    id: 'supplied',
    output: {pairs: {...pairs, status: {...pairs.status, candidateOverflow: loss}}}
  });
  expect(supplied.outputPairs?.status.candidateOverflow).toBe(loss);
});

it('B1 constructs replacement pairs-only joins without reserving the same loss resource ID', () => {
  const {props} = createFixture();
  const first = new GPUSpatialPredicateJoin({...props, id: 'replacement'});
  const replacement = new GPUSpatialPredicateJoin({...first.props, candidateCapacity: 9});
  expect(replacement.outputPairs?.status.candidateOverflow).toBeDefined();
  expect(replacement.outputPairs?.status.candidateOverflow).not.toBe(
    first.outputPairs?.status.candidateOverflow
  );
});

it('B3 retains captured caller storage before declaration and after destroy', () => {
  const {graph, scalar} = createFixture();
  const storage = {
    nodeMinima: createTransientView(graph, 'external-minima', 'float32x2', 1),
    nodeMaxima: createTransientView(graph, 'external-maxima', 'float32x2', 1),
    nodeChildren: createTransientView(graph, 'external-children', 'uint32x2', 1),
    leafIds: scalar('external-leaves'),
    count: scalar('external-count'),
    overflow: scalar('external-overflow')
  };
  const prepared = new GPUSpatialJoinPrepared({
    geometry: {
      kind: 'points',
      positions: createTransientView(graph, 'external-point', 'float32x2', 1)
    },
    storage
  });
  expect(prepared.storage.nodeMinima).toBe(storage.nodeMinima);
  prepared.destroy();
  for (const name of [
    'nodeMinima',
    'nodeMaxima',
    'nodeChildren',
    'leafIds',
    'count',
    'overflow'
  ] as const)
    expect(prepared.storage[name]).toBe(storage[name]);
});

it('SCL-14 sizes the padded whole-domain BVH at the 128 MiB boundary', () => {
  const limits = {maxStorageBufferBindingSize: 128 * 1024 * 1024, maxBufferSize: 256 * 1024 * 1024};
  const fits = getGPUSpatialJoinPreparedMemoryPlan(
    {kind: 'points', featureCount: 8388608, vertexCount: 8388608},
    limits
  );
  expect(fits.leafCapacity).toBe(8388608);
  expect(fits.largestBindingBytes).toBe(134217720); // 8 * (2 * 8388608 - 1).
  expect(fits.feasible).toBe(true);
  const exceeds = getGPUSpatialJoinPreparedMemoryPlan(
    {kind: 'points', featureCount: 8388609, vertexCount: 8388609},
    limits
  );
  expect(exceeds.leafCapacity).toBe(16777216);
  expect(exceeds.largestBindingBytes).toBe(268435448);
  expect(exceeds.feasible).toBe(false);
  const {graph} = createFixture();
  const partitioning = {
    length: 8388609,
    partitions: [
      {id: 0, coreStart: 0, coreEnd: 4194304, haloStart: 0, haloEnd: 4194304},
      {id: 1, coreStart: 4194304, coreEnd: 8388609, haloStart: 4194304, haloEnd: 8388609}
    ],
    seamOwnership: {kind: 'lowest-partition-id' as const},
    haloDistance: 0
  };
  const prepared = new GPUSpatialJoinPrepared({
    geometry: {
      kind: 'points',
      positions: createTransientView(graph, 'partitioned-points', 'float32x2', 8388609)
    },
    partitioning
  });
  expect(prepared.leafCapacity).toBe(exceeds.leafCapacity);
  expect(getGPUPartitionMemoryPlan(partitioning, 8).peakPartitionBytes).toBe(33554440);
});

it('UGE-02 trims padded geometry through every terminal live offset and retains live empties', () => {
  const decoded = decodeGPUSpatialJoinGeometry(
    {
      kind: 'polygons',
      positions: Float32Array.of(0, 0, 1, 0, 1, 1, 0, 1, 99, 99),
      featureOffsets: Uint32Array.of(0, 1, 1, 1),
      polygonOffsets: Uint32Array.of(0, 1, 1, 1),
      ringOffsets: Uint32Array.of(0, 4, 4, 4)
    },
    Uint32Array.of(2)
  );
  // One square followed by one genuinely empty live feature, then a padding slot.
  expect(Array.from(decoded.featureOffsets!)).toEqual([0, 1, 1]);
  expect(Array.from(decoded.polygonOffsets!)).toEqual([0, 1]);
  expect(Array.from(decoded.ringOffsets!)).toEqual([0, 4]);
  expect(Array.from(decoded.positions)).toEqual([0, 0, 1, 0, 1, 1, 0, 1]);
  expect(() =>
    decodeGPUSpatialJoinGeometry({kind: 'points', positions: new Uint8Array(3)}, 0)
  ).toThrow();
  const positionValues = Float32Array.of(1, 2, 99, 100);
  const bytes = new Uint8Array(positionValues.byteLength + 1);
  bytes.set(new Uint8Array(positionValues.buffer), 1);
  expect(
    Array.from(
      decodeGPUSpatialJoinGeometry(
        {kind: 'points', positions: bytes.subarray(1)},
        Uint32Array.of(1).buffer
      ).positions
    )
  ).toEqual([1, 2]);
  expect(() =>
    decodeGPUSpatialJoinGeometry({kind: 'points', positions: positionValues}, new Uint32Array(0))
  ).toThrow();
  expect(() =>
    decodeGPUSpatialJoinGeometry(
      {kind: 'lines', positions: positionValues, lineOffsets: Uint32Array.of(0, 3)},
      1
    )
  ).toThrow(/positions needs 3/);
});

it('UGE-12 accepts inner pairs and unmatched from one search', () => {
  const {graph, props, scalar} = createFixture();
  const unmatched = {
    ids: createTransientView(graph, 'unmatched', 'uint32', 1),
    status: {count: scalar('unmatched-count'), overflow: scalar('unmatched-overflow')}
  };
  const join = new GPUSpatialPredicateJoin({...props, output: {...props.output, unmatched}});
  expect(() => getParsedKernelSources(join.getCommandNodes(graph))).not.toThrow();
});

it('SCL-16 predicate planning reproduces constructor route selection', () => {
  const shape = {
    predicate: 'within' as const,
    leftKind: 'lines' as const,
    rightKind: 'lines' as const,
    leftCount: 1,
    rightCount: 1,
    leftVertexCount: 32,
    rightVertexCount: 32
  };
  expect(getGPUSpatialPredicateJoinExecutionPlan(shape).route).toBe('relate');
  expect(getGPUSpatialPredicateJoinExecutionPlan({...shape, leftVertexCount: 31}).route).toBe(
    'fast'
  );
  expect(getGPUSpatialPredicateJoinExecutionPlan({...shape, multipart: true}).route).toBe('fast');
  expect(getGPUSpatialPredicateJoinExecutionPlan({...shape, leftKind: 'points'}).route).toBe(
    'fast'
  );
  expect(getGPUSpatialPredicateJoinExecutionPlan({...shape, predicate: 'dwithin'}).route).toBe(
    'workgroup-distance'
  );
  expect(
    getGPUSpatialPredicateJoinExecutionPlan({...shape, predicate: 'dwithin', greatCircle: true})
      .route
  ).toBe('fast');
  expect(getGPUSpatialPredicateJoinExecutionPlan({...shape, predicate: 'covers'}).route).toBe(
    'relate'
  );
  expect(() =>
    getGPUSpatialPredicateJoinExecutionPlan({...shape, engine: 'invalid' as 'auto'})
  ).toThrow(/auto, fast or relate/);
  const {graph, scalar, props} = createFixture();
  const join = new GPUSpatialPredicateJoin({
    ...props,
    leftFeatureCount: scalar('live-count'),
    leftStatus: {overflow: scalar('upstream-overflow'), complete: scalar('upstream-complete')},
    output: {
      ...props.output,
      unmatched: {
        ids: scalar('unmatched'),
        status: {count: scalar('unmatched-count'), overflow: scalar('unmatched-overflow')}
      },
      status: {candidateOverflow: scalar('aggregate-candidate-loss')}
    }
  });
  const sources = getParsedKernelSources(join.getCommandNodes(graph));
  expect(sources.has(`${join.id}-live-left-bounds`)).toBe(true);
  expect(join.usesRelateEngine).toBe(false);
});

it('B1 plans great-circle dwithin and rejects great-circle intersects', () => {
  const shape = {
    greatCircle: true,
    leftKind: 'points',
    rightKind: 'points',
    leftCount: 1,
    rightCount: 1,
    leftVertexCount: 1,
    rightVertexCount: 1
  } as const;
  expect(getGPUSpatialPredicateJoinExecutionPlan({...shape, predicate: 'dwithin'})).toMatchObject({
    route: 'fast',
    feasible: true
  });
  expect(() =>
    getGPUSpatialPredicateJoinExecutionPlan({...shape, predicate: 'intersects'})
  ).toThrow(/greatCircle requires predicate 'dwithin'.*intersects/);
});

it('ING-12 rejects noninteger, nonfinite or unrepresentable two-word slot capacities', () => {
  for (const slotCount of [NaN, Infinity, 1.5, 0, -1, 0x80000000, Number.MAX_SAFE_INTEGER]) {
    expect(() => packGPUSpatialRelatePattern('T********', slotCount)).toThrow(/slotCount/);
  }
  // T = bits 1..3; each * = all four bits. A spare slot must match nothing.
  expect(Array.from(packGPUSpatialRelatePattern('T********', 2))).toEqual([0xfffffffe, 15, 0, 0]);
});

it('LIF-11 releases prepared join acquisitions and publishes readiness only on success', () => {
  const {graph} = createFixture();
  const input = graph.device.createBuffer({
    byteLength: 9 * 8,
    usage: Buffer.STORAGE | Buffer.COPY_DST
  });
  const geometry = {
    kind: 'points' as const,
    positions: importGraphBuffer(graph, 'persistent-points', input, 'float32x2', 9)
  };
  const prepared = new GPUSpatialJoinPrepared({id: 'failure-join', geometry});
  const created: {destroyed: boolean}[] = [];
  const allocate = graph.device.createBuffer.bind(graph.device);
  const spy = vi.spyOn(graph.device, 'createBuffer').mockImplementation(settings => {
    if (settings && 'id' in settings && settings.id === 'failure-join-node-minima')
      throw new Error('injected node-minima');
    const buffer = allocate(settings);
    created.push(buffer);
    return buffer;
  });
  try {
    expect(() => graph.add(prepared)).toThrow('injected node-minima');
    expect(prepared.isDeclaredIn(graph)).toBe(false);
    expect(created.filter(buffer => !buffer.destroyed)).toHaveLength(0);
    expect(() => prepared.storage).toThrow(/needs graph.add/);
    spy.mockRestore();
    const fresh = new GPUCommandGraph(graph.device);
    expect(() => fresh.add(prepared)).not.toThrow();
    expect(prepared.isDeclaredIn(fresh)).toBe(true);
    expect(prepared.isDeclaredIn(graph)).toBe(false);
    const freshRight = {
      kind: 'points' as const,
      positions: importGraphBuffer(fresh, 'fresh-right', input, 'float32x2', 9)
    };
    const integers = (name: string, length = 1) =>
      createTransientView(fresh, name, 'uint32', length);
    const join = new GPUSpatialPredicateJoin({
      left: {kind: 'points', positions: createTransientView(fresh, 'fresh-left', 'float32x2', 1)},
      right: freshRight,
      predicate: 'intersects',
      prepared,
      output: {
        pairs: {
          leftRows: integers('fresh-left-rows', 9),
          rightRows: integers('fresh-right-rows', 9),
          status: {count: integers('fresh-count'), overflow: integers('fresh-overflow')}
        }
      }
    });
    expect(() => fresh.add(join)).not.toThrow();
  } finally {
    spy.mockRestore();
    prepared.destroy();
    input.destroy();
    graph.device.destroy();
  }
});
