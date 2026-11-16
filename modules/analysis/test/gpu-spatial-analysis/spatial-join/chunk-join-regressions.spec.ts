// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph, type GPUCommandNode, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {createSaturatedCountScanNodes} from '../../../src/analysis-core/kernels/scan-total';
import {createTransientView} from '@luma.gl/graph/gpu-core';
import {GPUNearestFeatureJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-nearest-feature-join';
import {GPUPointInPolygonJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-point-in-polygon-join';
import {GPUBoundsFilter} from '../../../src/gpu-spatial-analysis/bounds-filter/gpu-bounds-filter';
import {
  createSpatialJoinChunkNode,
  getCandidateTruncationNodes
} from '../../../src/gpu-spatial-analysis/spatial-join/spatial-join-passes';
import {
  createInputBuffer,
  createOutputBuffer,
  createVectorView,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

const NO_FEATURE = 0xffffffff;

/** Independent buffers and deliberately unaligned logical offsets exercise runtime binding offsets. */
function createFixture(
  device: Device,
  kind: 'join' | 'bounds' | 'nearest',
  lengths: number[],
  accepted: Set<number>,
  candidateCapacity: number,
  compactCapacity = 20,
  packed = false
) {
  const graph = new GPUCommandGraph(device, {id: `chunk-regression-${kind}`});
  const buffers: Buffer[] = [];
  const totalRows = lengths.reduce((sum, length) => sum + length, 0);
  const importView = <Format extends 'uint32' | 'float32' | 'float32x2' | 'float32x4'>(
    name: string,
    buffer: Buffer,
    format: Format,
    length: number,
    byteOffset = 0
  ) => {
    buffers.push(buffer);
    const handle = graph.importBuffer(
      {id: name, byteLength: buffer.byteLength, usage: buffer.usage},
      buffer
    );
    return graph.createDataView(handle, {format, length, byteOffset});
  };
  const scalarBuffers: Record<string, Buffer> = {};
  const scalar = (name: string) => {
    const buffer = createOutputBuffer(device, 1);
    scalarBuffers[name] = buffer;
    return importView(name, buffer, 'uint32', 1);
  };
  const idsBuffer = createOutputBuffer(device, compactCapacity);
  const ids = importView('compact-ids', idsBuffer, 'uint32', compactCapacity);
  const count = scalar('count');
  const overflow = scalar('overflow');
  const requiredCount = scalar('requiredCount');
  const featureCounts = scalar('featureCounts');
  const candidateCount = scalar('candidateCount');
  const candidateOverflow = scalar('candidateOverflow');
  const uncertainCount = scalar('uncertainCount');
  const joinOverflow = scalar('joinOverflow');
  const flagReadbackBuffer = createOutputBuffer(device, totalRows);
  const flagReadback = importView('flag-readback', flagReadbackBuffer, 'uint32', totalRows);
  const assignmentReadbackBuffer = createOutputBuffer(device, totalRows);
  const assignmentReadback = importView(
    'assignment-readback',
    assignmentReadbackBuffer,
    'uint32',
    totalRows
  );
  const inputRecords: {buffer: Buffer; length: number; firstRow: number; prefixWords: number}[] =
    [];
  const flagChunks: GraphDataView<'uint32'>[] = [];
  const assignmentChunks: GraphDataView<'uint32'>[] = [];
  const sourceChunks: GraphDataView<'uint32'>[] = [];
  const rowWidth = kind === 'bounds' ? 4 : 2;
  const makeValues = (
    length: number,
    firstRow: number,
    prefixWords: number,
    visible: Set<number>
  ) => {
    const values = new Float32Array(prefixWords + Math.max(length, 1) * rowWidth);
    for (let row = 0; row < length; row++)
      values.fill(
        visible.has(firstRow + row) ? 0.5 : 2,
        prefixWords + row * rowWidth,
        prefixWords + (row + 1) * rowWidth
      );
    return values;
  };
  let firstRow = 0;
  const positionChunks = lengths.map((length, chunkIndex) => {
    const prefixWords = 1 + chunkIndex * 3;
    const inputBuffer = createInputBuffer(
      device,
      makeValues(length, firstRow, prefixWords, accepted)
    );
    inputRecords.push({buffer: inputBuffer, length, firstRow, prefixWords});
    const positions = importView(
      `rows-${chunkIndex}`,
      inputBuffer,
      kind === 'bounds' ? 'float32x4' : 'float32x2',
      length,
      prefixWords * 4
    );
    // These caller-owned flags have STORAGE usage only: no COPY_DST initialization or clearing.
    const flagBuffer = device.createBuffer({
      byteLength: (prefixWords + Math.max(length, 1)) * 4,
      usage: Buffer.STORAGE
    });
    flagChunks.push(
      importView(`flags-${chunkIndex}`, flagBuffer, 'uint32', length, prefixWords * 4)
    );
    const assignedBuffer = device.createBuffer({
      byteLength: (prefixWords + Math.max(length, 1)) * 4,
      usage: Buffer.STORAGE
    });
    assignmentChunks.push(
      importView(`assigned-${chunkIndex}`, assignedBuffer, 'uint32', length, prefixWords * 4)
    );
    const sourceValues = new Uint32Array(prefixWords + Math.max(length, 1));
    for (let row = 0; row < length; row++) sourceValues[prefixWords + row] = 1000 + firstRow + row;
    sourceChunks.push(
      importView(
        `source-${chunkIndex}`,
        createInputBuffer(device, sourceValues),
        'uint32',
        length,
        prefixWords * 4
      )
    );
    firstRow += length;
    return positions;
  });
  const flags = packed ? flagChunks[0] : createVectorView('flag-chunks', 'uint32', flagChunks);
  const nearestFlags =
    kind === 'nearest'
      ? createTransientView(graph, 'nearest-flags', 'uint32', totalRows)
      : undefined;
  if (kind === 'join') {
    const positions = positionChunks as GraphDataView<'float32x2'>[];
    graph.add(
      new GPUPointInPolygonJoin({
        points: {
          kind: 'points',
          positions: packed
            ? positions[0]
            : createVectorView('point-chunks', 'float32x2', positions),
          sourceIds: packed
            ? sourceChunks[0]
            : createVectorView('source-chunks', 'uint32', sourceChunks)
        },
        polygons: {
          kind: 'polygons',
          positions: importView(
            'polygon-vertices',
            createInputBuffer(device, new Float32Array([0, 0, 1, 0, 1, 1, 0, 1])),
            'float32x2',
            4
          ),
          featureOffsets: importView(
            'features',
            createInputBuffer(device, new Uint32Array([0, 1])),
            'uint32',
            2
          ),
          polygonOffsets: importView(
            'polygons',
            createInputBuffer(device, new Uint32Array([0, 1])),
            'uint32',
            2
          ),
          ringOffsets: importView(
            'rings',
            createInputBuffer(device, new Uint32Array([0, 4])),
            'uint32',
            2
          )
        },
        candidateCapacity,
        querySort: true,
        output: {
          pointFeatureIds: packed
            ? assignmentChunks[0]
            : createVectorView('assignment-chunks', 'uint32', assignmentChunks),
          candidateTruncated: flags,
          featureCounts,
          matches: {ids, status: {count, overflow, requiredCount}},
          status: {overflow: joinOverflow, candidateCount, candidateOverflow, uncertainCount}
        }
      })
    );
  } else if (kind === 'nearest') {
    const positions = positionChunks as GraphDataView<'float32x2'>[];
    graph.add(
      new GPUNearestFeatureJoin({
        queries: {
          kind: 'points',
          positions: createVectorView('nearest-point-chunks', 'float32x2', positions),
          sourceIds: createVectorView('nearest-source-chunks', 'uint32', sourceChunks)
        },
        features: {
          kind: 'points',
          positions: importView(
            'feature-point',
            createInputBuffer(device, new Float32Array([0.5, 0.5])),
            'float32x2',
            1
          )
        },
        radius: importView(
          'radius',
          createInputBuffer(device, new Float32Array([0])),
          'float32',
          1
        ),
        candidateCapacity,
        querySort: true,
        output: {
          nearestFeatureIds: createVectorView('nearest-assignments', 'uint32', assignmentChunks),
          candidateTruncated: nearestFlags,
          featureCounts,
          matches: {ids, status: {count, overflow, requiredCount}},
          status: {overflow: joinOverflow, candidateCount, candidateOverflow}
        }
      })
    );
  } else {
    const bounds = positionChunks as GraphDataView<'float32x4'>[];
    graph.add(
      new GPUBoundsFilter({
        bounds: packed ? bounds[0] : createVectorView('bounds-chunks', 'float32x4', bounds),
        parameters: importView(
          'box',
          createInputBuffer(device, new Float32Array([0, 0, 1, 1])),
          'float32',
          4
        ),
        output: {ids, outputMask: flags, status: {count, overflow, requiredCount}}
      })
    );
  }
  // Storage-only results are mirrored by compute, so no output needs COPY_SRC either.
  firstRow = 0;
  const mirrors: GPUCommandNode[] = [];
  for (const [chunkIndex, length] of lengths.entries()) {
    if (length > 0) {
      for (const [name, source, destination] of [
        [
          'flags',
          nearestFlags
            ? graph.createDataView(nearestFlags.buffer, {
                format: 'uint32',
                length,
                byteOffset: firstRow * 4
              })
            : flagChunks[chunkIndex],
          flagReadback
        ],
        ...(kind !== 'bounds'
          ? [['assigned', assignmentChunks[chunkIndex], assignmentReadback] as const]
          : [])
      ] as const) {
        mirrors.push(
          createSpatialJoinChunkNode(graph, {
            id: `mirror-${name}-${chunkIndex}`,
            operation: 'ChunkRegression',
            invocationCount: length,
            firstRow,
            bindings: [
              {name: 'sourceRows', view: source, type: 'u32', access: 'read'},
              {name: 'destinationRows', view: destination, type: 'u32', access: 'read_write'}
            ],
            body: 'destinationRows[destinationRowsOffset + CHUNK_FIRST_ROW + index] = sourceRows[sourceRowsOffset + index];'
          })
        );
      }
    }
    firstRow += length;
  }
  graph.add({getCommandNodes: () => mirrors});
  const compiled = graph.compile();
  return {
    seedStatuses(value: number) {
      for (const buffer of Object.values(scalarBuffers)) buffer.write(new Uint32Array([value]));
    },
    async run(visible = accepted) {
      for (const record of inputRecords)
        record.buffer.write(
          makeValues(record.length, record.firstRow, record.prefixWords, visible)
        );
      submitGraph(device, compiled, undefined);
      const statuses: Record<string, number> = {};
      for (const [name, buffer] of Object.entries(scalarBuffers))
        statuses[name] = (await readUint32(buffer, 1))[0];
      return {
        statuses,
        ids: await readUint32(idsBuffer, statuses['count']),
        flags: await readUint32(flagReadbackBuffer, totalRows),
        assignments: kind !== 'bounds' ? await readUint32(assignmentReadbackBuffer, totalRows) : []
      };
    },
    destroy() {
      compiled.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  };
}

for (const capacity of [4, 5]) {
  it(`GPUPointInPolygonJoin preserves the exact candidate prefix and resets all flags (capacity ${capacity})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createFixture(
      device,
      'join',
      [2, 3, 5],
      new Set(Array.from({length: 10}, (_, index) => index)),
      capacity
    );
    try {
      const overflowing = await fixture.run();
      expect(overflowing.flags).toEqual(
        Array.from({length: 10}, (_, row) => (row < capacity ? 0 : 1))
      );
      expect(overflowing.assignments).toEqual(
        Array.from({length: 10}, (_, row) => (row < capacity ? 0 : NO_FEATURE))
      );
      expect(overflowing.ids).toEqual(Array.from({length: capacity}, (_, row) => 1000 + row));
      expect(overflowing.statuses).toEqual({
        count: capacity,
        // GPUCompactOutput reports any contributor capacity overflow, including candidates.
        overflow: 1,
        requiredCount: capacity,
        featureCounts: capacity,
        candidateCount: 10,
        candidateOverflow: 1,
        uncertainCount: 0,
        joinOverflow: 1
      });
      const reset = await fixture.run(new Set([1, 8]));
      expect(reset.flags).toEqual(new Array(10).fill(0));
      expect(reset.assignments).toEqual(
        Array.from({length: 10}, (_, row) => (row === 1 || row === 8 ? 0 : NO_FEATURE))
      );
      expect(reset.ids).toEqual([1001, 1008]);
      expect(reset.statuses).toEqual({
        count: 2,
        overflow: 0,
        requiredCount: 2,
        featureCounts: 2,
        candidateCount: 2,
        candidateOverflow: 0,
        uncertainCount: 0,
        joinOverflow: 0
      });
    } finally {
      fixture.destroy();
    }
  });
}

for (const kind of ['join', 'bounds'] as const) {
  it(`${kind} packed and independent offset chunks agree across empty, workgroup and scan boundaries`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const lengths = [0, 255, 0, 256, 65_537, 0];
    const totalRows = lengths.reduce((sum, length) => sum + length, 0);
    const accepted = new Set([
      0,
      253,
      254,
      255,
      256,
      509,
      510,
      511,
      512,
      65_535,
      65_536,
      totalRows - 1
    ]);
    const packed = createFixture(device, kind, [totalRows], accepted, 32, 20, true);
    const chunked = createFixture(device, kind, lengths, accepted, 32);
    try {
      const expectedIds = [...accepted].map(row => (kind === 'join' ? 1000 + row : row));
      const packedResult = await packed.run();
      const chunkedResult = await chunked.run();
      expect(chunkedResult).toEqual(packedResult);
      expect(chunkedResult.ids).toEqual(expectedIds);
      expect(chunkedResult.statuses['count']).toBe(accepted.size);
      expect(chunkedResult.flags).toEqual(
        Array.from({length: totalRows}, (_, row) =>
          kind === 'join' ? 0 : Number(accepted.has(row))
        )
      );
      if (kind === 'join')
        expect(chunkedResult.assignments).toEqual(
          Array.from({length: totalRows}, (_, row) => (accepted.has(row) ? 0 : NO_FEATURE))
        );
    } finally {
      packed.destroy();
      chunked.destroy();
    }
  });
}

it('GPUBoundsFilter resets storage-only masks and all compact statuses after overflow', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createFixture(
    device,
    'bounds',
    [2, 3, 5],
    new Set(Array.from({length: 10}, (_, row) => row)),
    4,
    4
  );
  try {
    const overflowing = await fixture.run();
    expect(overflowing.ids).toEqual([0, 1, 2, 3]);
    expect(overflowing.flags).toEqual(new Array(10).fill(1));
    expect(overflowing.statuses['count']).toBe(4);
    expect(overflowing.statuses['requiredCount']).toBe(10);
    expect(overflowing.statuses['overflow']).toBe(1);
    const reset = await fixture.run(new Set([1, 8]));
    expect(reset.ids).toEqual([1, 8]);
    expect(reset.flags).toEqual(
      Array.from({length: 10}, (_, row) => Number(row === 1 || row === 8))
    );
    expect(reset.statuses['count']).toBe(2);
    expect(reset.statuses['requiredCount']).toBe(2);
    expect(reset.statuses['overflow']).toBe(0);
  } finally {
    fixture.destroy();
  }
});

it('saturated candidate offsets prevent a third chunk from overwriting the retained prefix after 2^32', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device, {id: 'candidate-wrap'});
  const rowCount = 65_537;
  const counts = createTransientView(graph, 'counts', 'uint32', rowCount);
  const offsetsBuffer = createOutputBuffer(device, rowCount);
  const totalBuffer = createOutputBuffer(device, 1);
  const retainedBuffer = createOutputBuffer(device, 4);
  const handle = (name: string, buffer: Buffer, length: number) =>
    graph.createDataView(
      graph.importBuffer({id: name, byteLength: buffer.byteLength, usage: buffer.usage}, buffer),
      {format: 'uint32', length}
    );
  const offsets = handle('offsets', offsetsBuffer, rowCount);
  const total = handle('total', totalBuffer, 1);
  const retained = handle('retained', retainedBuffer, 4);
  const nodes: GPUCommandNode[] = [];
  let firstRow = 0;
  // Isolate the 65,537 x 65,536 candidate-count layout without four billion BVH visits.
  for (const [chunkIndex, length] of [1, 65_535, 1].entries()) {
    nodes.push(
      createSpatialJoinChunkNode(graph, {
        id: `counts-${chunkIndex}`,
        operation: 'CandidateWrap',
        invocationCount: length,
        firstRow,
        bindings: [{name: 'counts', view: counts, type: 'u32', access: 'read_write'}],
        body: 'counts[countsOffset + CHUNK_FIRST_ROW + index] = 65536u;'
      })
    );
    firstRow += length;
  }
  nodes.push(
    ...createSaturatedCountScanNodes(graph, {
      id: 'scan',
      operation: 'CandidateWrap',
      counts,
      offsets,
      rowCount,
      total
    })
  );
  firstRow = 0;
  for (const [chunkIndex, length] of [1, 65_535, 1].entries()) {
    nodes.push(
      createSpatialJoinChunkNode(graph, {
        id: `write-${chunkIndex}`,
        operation: 'CandidateWrap',
        invocationCount: length,
        firstRow,
        bindings: [
          {name: 'offsets', view: offsets, type: 'u32', access: 'read'},
          {name: 'retained', view: retained, type: 'u32', access: 'read_write'}
        ],
        body: `let row = CHUNK_FIRST_ROW + index;
  let slotStart = offsets[offsetsOffset + row];
  if (slotStart >= 4u) { return; }
  for (var localSlot = 0u; localSlot < 4u; localSlot++) {
    retained[retainedOffset + localSlot] = 123u + row;
  }`
      })
    );
    firstRow += length;
  }
  const flagsBuffer = createOutputBuffer(device, rowCount);
  const candidateOverflowBuffer = createOutputBuffer(device, 1);
  nodes.push(
    ...getCandidateTruncationNodes(graph, {
      id: 'wrap-status',
      operation: 'CandidateWrap',
      counts,
      offsets,
      rowCount,
      candidateCapacity: 4,
      total,
      candidateTruncated: handle('flags', flagsBuffer, rowCount),
      candidateOverflow: handle('candidate-overflow', candidateOverflowBuffer, 1)
    })
  );
  graph.add({getCommandNodes: () => nodes});
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const values = await readUint32(offsetsBuffer, rowCount);
    expect(values[0]).toBe(0);
    expect(values[65_535]).toBe(0xffff0000);
    expect(values[65_536]).toBe(0xffffffff);
    expect(values.every((value, row) => row === 0 || value >= values[row - 1])).toBe(true);
    expect(await readUint32(totalBuffer, 1)).toEqual([0xffffffff]);
    expect(await readUint32(retainedBuffer, 4)).toEqual([123, 123, 123, 123]);
    expect(await readUint32(flagsBuffer, rowCount)).toEqual(new Array(rowCount).fill(1));
    expect(await readUint32(candidateOverflowBuffer, 1)).toEqual([1]);
  } finally {
    compiled.destroy();
    for (const buffer of [
      offsetsBuffer,
      totalBuffer,
      retainedBuffer,
      flagsBuffer,
      candidateOverflowBuffer
    ])
      buffer.destroy();
  }
});

it('GPUBoundsFilter rejects overlapping storage-only mask chunks at construction', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device, {id: 'overlapping-masks'});
  const sameMaskRow = createTransientView(graph, 'same-row', 'uint32', 1);
  const bounds = createVectorView(
    'bounds',
    'float32x4',
    [0, 1, 2].map(index => createTransientView(graph, `bounds-${index}`, 'float32x4', 1))
  );
  expect(
    () =>
      new GPUBoundsFilter({
        bounds,
        parameters: createTransientView(graph, 'parameters', 'float32', 4),
        output: {
          ids: createTransientView(graph, 'ids', 'uint32', 3),
          outputMask: createVectorView('mask', 'uint32', [sameMaskRow, sameMaskRow, sameMaskRow]),
          status: {
            count: createTransientView(graph, 'count', 'uint32', 1),
            overflow: createTransientView(graph, 'overflow', 'uint32', 1)
          }
        }
      })
  ).toThrow(/.*must not overlap/);
});

for (const capacity of [4, 5]) {
  it(`shared candidate scan preserves nearest-feature join overflow semantics (capacity ${capacity})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createFixture(
      device,
      'nearest',
      [2, 3, 5],
      new Set(Array.from({length: 10}, (_, row) => row)),
      capacity
    );
    try {
      const result = await fixture.run();
      expect(result.flags).toEqual(Array.from({length: 10}, (_, row) => (row < capacity ? 0 : 1)));
      expect(result.assignments).toEqual(
        Array.from({length: 10}, (_, row) => (row < capacity ? 0 : NO_FEATURE))
      );
      expect(result.ids).toEqual(Array.from({length: capacity}, (_, row) => 1000 + row));
      expect(result.statuses).toEqual({
        count: capacity,
        // GPUCompactOutput reports any contributor capacity overflow, including candidates.
        overflow: 1,
        requiredCount: capacity,
        featureCounts: capacity,
        candidateCount: 10,
        candidateOverflow: 1,
        uncertainCount: 0,
        joinOverflow: 1
      });
      const reset = await fixture.run(new Set([1, 8]));
      expect(reset.flags).toEqual(new Array(10).fill(0));
      expect(reset.ids).toEqual([1001, 1008]);
      expect(reset.statuses).toEqual({
        count: 2,
        overflow: 0,
        requiredCount: 2,
        featureCounts: 2,
        candidateCount: 2,
        candidateOverflow: 0,
        uncertainCount: 0,
        joinOverflow: 0
      });
    } finally {
      fixture.destroy();
    }
  });
}

it('GPUPointInPolygonJoin preserves explicit source rows with query-sorted chunks [65536, 1]', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const totalRows = 65_537;
  const selected = new Set([0, 255, 256, 65_534, 65_535, 65_536]);
  const fixture = createFixture(device, 'join', [65_536, 1], selected, 32);
  try {
    const result = await fixture.run();
    expect(result.flags).toEqual(new Array(totalRows).fill(0));
    expect(result.assignments).toEqual(
      Array.from({length: totalRows}, (_, row) => (selected.has(row) ? 0 : NO_FEATURE))
    );
    expect(result.ids).toEqual([...selected].map(row => 1000 + row));
    expect(result.statuses).toEqual({
      count: 6,
      overflow: 0,
      requiredCount: 6,
      featureCounts: 6,
      candidateCount: 6,
      candidateOverflow: 0,
      uncertainCount: 0,
      joinOverflow: 0
    });
  } finally {
    fixture.destroy();
  }
});

it('join and bounds constructors reject logically separated but co-bound aligned outputs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const shared = createTransientView(graph, 'shared-writes', 'uint32', 2);
  const first = graph.createDataView(shared.buffer, {format: 'uint32', length: 1, byteOffset: 0});
  const second = graph.createDataView(shared.buffer, {format: 'uint32', length: 1, byteOffset: 4});
  const scalar = (id: string) => createTransientView(graph, id, 'uint32', 1);
  expect(
    () =>
      new GPUBoundsFilter({
        bounds: createTransientView(graph, 'bounds', 'float32x4', 1),
        parameters: createTransientView(graph, 'box', 'float32', 4),
        output: {ids: scalar('bounds-ids'), status: {count: first, overflow: second}}
      })
  ).toThrow(/aligned binding byte ranges.*must not overlap/);
  expect(
    () =>
      new GPUPointInPolygonJoin({
        candidateCapacity: 1,
        points: {kind: 'points', positions: createTransientView(graph, 'point', 'float32x2', 1)},
        polygons: {
          kind: 'polygons',
          positions: createTransientView(graph, 'vertices', 'float32x2', 4),
          featureOffsets: createTransientView(graph, 'features', 'uint32', 2),
          polygonOffsets: createTransientView(graph, 'parts', 'uint32', 2),
          ringOffsets: createTransientView(graph, 'rings', 'uint32', 2)
        },
        output: {
          pointFeatureIds: second,
          featureCounts: first,
          matches: {
            ids: scalar('match-id'),
            status: {count: scalar('count'), overflow: scalar('compact-overflow')}
          },
          status: {overflow: scalar('join-overflow')}
        }
      })
  ).toThrow(/aligned binding byte ranges.*must not overlap/);
});

for (const kind of ['join', 'nearest'] as const) {
  it(`${kind} reports compact-only overflow independently of candidate overflow and resets the next frame`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createFixture(
      device,
      kind,
      [2, 3, 5],
      new Set(Array.from({length: 10}, (_, row) => row)),
      32,
      4
    );
    try {
      const result = await fixture.run();
      expect(result.flags).toEqual(new Array(10).fill(0));
      expect(result.assignments).toEqual(new Array(10).fill(0));
      expect(result.ids).toEqual([1000, 1001, 1002, 1003]);
      expect(result.statuses).toEqual({
        count: 4,
        overflow: 1,
        requiredCount: 10,
        featureCounts: 10,
        candidateCount: 10,
        candidateOverflow: 0,
        uncertainCount: 0,
        joinOverflow: 1
      });
      const reset = await fixture.run(new Set([1, 8]));
      expect(reset.flags).toEqual(new Array(10).fill(0));
      expect(reset.assignments).toEqual(
        Array.from({length: 10}, (_, row) => (row === 1 || row === 8 ? 0 : NO_FEATURE))
      );
      expect(reset.ids).toEqual([1001, 1008]);
      expect(reset.statuses).toEqual({
        count: 2,
        overflow: 0,
        requiredCount: 2,
        featureCounts: 2,
        candidateCount: 2,
        candidateOverflow: 0,
        uncertainCount: 0,
        joinOverflow: 0
      });
    } finally {
      fixture.destroy();
    }
  });
}

it('GPUPointInPolygonJoin resets prefilled scalar statuses with only empty point chunks', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createFixture(device, 'join', [0, 0, 0], new Set(), 4);
  try {
    for (const seed of [0xffffffff, 1]) {
      fixture.seedStatuses(seed);
      const result = await fixture.run();
      expect(result.ids).toEqual([]);
      expect(result.flags).toEqual([]);
      expect(result.assignments).toEqual([]);
      expect(result.statuses).toEqual({
        count: 0,
        overflow: 0,
        requiredCount: 0,
        featureCounts: 0,
        candidateCount: 0,
        candidateOverflow: 0,
        uncertainCount: 0,
        joinOverflow: 0
      });
    }
  } finally {
    fixture.destroy();
  }
});
