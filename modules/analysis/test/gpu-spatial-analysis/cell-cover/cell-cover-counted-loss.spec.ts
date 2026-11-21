// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUCellCover,
  GPUCellCompaction,
  GPUCellSetOutline
} from '../../../src/gpu-spatial-analysis';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createOutputOracleFixture} from '../cell-topology/output-oracle-test-utils';

for (const family of ['quadbin', 'h3'] as const) {
  for (const diagnostic of [false, true]) {
    for (const publishComplete of [false, true]) {
      for (const reportLoss of [false, true]) {
        it(`W5-05 counted cover replay (${family}, diagnostic ${diagnostic}, complete ${publishComplete}, consumer loss ${reportLoss})`, async () => {
          const device = await getWebGPUTestDevice();
          requireTestDevice(device);
          const fixture = createOutputOracleFixture(device);
          const {graph, input, output} = fixture;
          // Zoom 2: column zero, rows zero and one. Their centers (-135,79.1713)
          // and (-135,40.9799) lie inside both frames. Only north=90 exceeds Mercator.
          // These adjacent cells are not four siblings, so compaction keeps both.
          // Independent Python/Shapely rectangle containment over all 122 resolution-zero
          // H3 centers (h3-js center coordinates) for [-10,10] x [10,80/90]
          // gives exactly 8039fffffffffff and 8059fffffffffff. Their centers are
          // (-0.5345171,33.7110115) and (6.3587832,14.1302725); neither touches an edge.
          const west = family === 'quadbin' ? -170 : -10;
          const east = family === 'quadbin' ? -100 : 10;
          const resolution = family === 'quadbin' ? 2 : 0;
          const positions = input(
            'positions',
            'float32x2',
            Float32Array.of(west, 10, east, 10, east, 90, west, 90)
          );
          const source = {
            cells: output('cover-cells', 'uint32x2', 4),
            status: {
              count: output('cover-count', 'uint32', 1),
              overflow: output('cover-overflow', 'uint32', 1),
              candidateOverflow: output('cover-candidate', 'uint32', 1),
              complete: publishComplete ? output('cover-complete', 'uint32', 1) : undefined,
              uncertainCount: diagnostic ? output('uncertain', 'uint32', 1) : undefined
            }
          };
          graph.add(
            new GPUCellCover({
              family,
              resolution,
              positions,
              featureOffsets: input('features', 'uint32', Uint32Array.of(0, 1)),
              polygonOffsets: input('polygons', 'uint32', Uint32Array.of(0, 1)),
              ringOffsets: input('rings', 'uint32', Uint32Array.of(0, 4)),
              candidateCapacity: family === 'quadbin' ? 4 : 1024,
              candidateEnumeration: 'bounds',
              edgeSlabs: false,
              output: {...source, featureIds: output('feature-ids', 'uint32', 4)}
            })
          );
          const cells = output('compacted', 'uint32x2', 4);
          const status = {
            count: output('count', 'uint32', 1),
            requiredCount: output('required', 'uint32', 1),
            overflow: output('overflow', 'uint32', 1),
            candidateOverflow: reportLoss ? output('candidate', 'uint32', 1) : undefined
          };
          graph.add(
            new GPUCellCompaction({
              family,
              source,
              operation: {type: 'compact', resolution},
              output: {cells, status}
            })
          );
          const compiled = graph.compile();
          try {
            const cleanNorth = family === 'quadbin' ? 85 : 80;
            for (const [north, loss] of [
              [90, 1],
              [cleanNorth, 0],
              [90, 1],
              [cleanNorth, 0]
            ]) {
              fixture.writeInput(
                positions,
                Float32Array.of(west, 10, east, 10, east, north, west, north)
              );
              const result = await fixture.submitTwice(compiled, async () => ({
                source: await fixture.readRows([
                  source.status.count,
                  source.status.overflow,
                  source.status.candidateOverflow
                ]),
                consumer: await fixture.readRows([
                  status.count,
                  status.requiredCount,
                  status.overflow
                ]),
                candidate: status.candidateOverflow
                  ? await fixture.read(status.candidateOverflow)
                  : undefined,
                uncertain: source.status.uncertainCount
                  ? await fixture.read(source.status.uncertainCount)
                  : undefined,
                complete: source.status.complete
                  ? await fixture.read(source.status.complete)
                  : undefined,
                cells: await fixture.read(cells)
              }));
              const retainedCount = !publishComplete && loss ? 0 : 2;
              const inheritedLoss = publishComplete ? loss : 0;
              const consumerCount = !reportLoss && inheritedLoss ? 0 : retainedCount;
              expect(result.source).toEqual([retainedCount, 0, 0]);
              expect(result.consumer).toEqual([consumerCount, retainedCount, 0]);
              if (reportLoss) expect(result.candidate).toEqual([inheritedLoss]);
              if (diagnostic) expect(result.uncertain).toEqual([loss]);
              if (publishComplete) expect(result.complete).toEqual([1 - loss]);
              // S2: a suppressed count makes the cell allocation unspecified.
              if (consumerCount > 0)
                expect(result.cells).toEqual(
                  family === 'quadbin'
                    ? [0xffffffff, 0x4820ffff, 0xffffffff, 0x4822ffff, 0, 0, 0, 0]
                    : [0xffffffff, 0x08039fff, 0xffffffff, 0x08059fff, 0, 0, 0, 0]
                );
            }
          } finally {
            compiled.destroy();
            fixture.destroy();
          }
        });
      }
    }
  }
}

it('B1/B2 separates clipping, capacity, both and clean replay with shared public status storage', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createOutputOracleFixture(device);
  const {graph, input, output} = fixture;
  const positions = input(
    'positions',
    'float32x2',
    Float32Array.of(-170, 70, -100, 70, -100, 90, -170, 90)
  );
  const shared = output('shared-status', 'uint32', 256);
  const word = (offset: number) =>
    graph.createDataView(shared.buffer, {
      format: 'uint32',
      byteOffset: shared.byteOffset + offset,
      length: 1
    });
  const source = {
    cells: output('cover-cells', 'uint32x2', 4),
    status: {
      count: output('cover-count', 'uint32', 1),
      overflow: output('cover-overflow', 'uint32', 1),
      candidateOverflow: word(0),
      uncertainCount: word(256),
      candidateCount: word(512),
      complete: word(768)
    }
  };
  graph.add(
    new GPUCellCover({
      family: 'quadbin',
      resolution: 2,
      positions,
      featureOffsets: input('features', 'uint32', Uint32Array.of(0, 1)),
      polygonOffsets: input('polygons', 'uint32', Uint32Array.of(0, 1)),
      ringOffsets: input('rings', 'uint32', Uint32Array.of(0, 4)),
      candidateCapacity: 1,
      candidateEnumeration: 'bounds',
      edgeSlabs: false,
      output: {...source, featureIds: output('feature-ids', 'uint32', 4)}
    })
  );
  const cells = output('compacted', 'uint32x2', 4);
  const status = {
    count: output('count', 'uint32', 1),
    overflow: output('overflow', 'uint32', 1),
    candidateOverflow: output('candidate', 'uint32', 1)
  };
  graph.add(
    new GPUCellCompaction({
      family: 'quadbin',
      source,
      operation: {type: 'compact', resolution: 2, minimumResolution: 2},
      output: {cells, status}
    })
  );
  const compiled = graph.compile();
  try {
    // Column 0, row 0 is always accepted. South=10 adds row 1 as a second
    // candidate (capacity 1); north=90 clips, while north=80/85 does not.
    for (const [south, north, capacityLoss, clipping] of [
      [70, 90, 0, 1],
      [10, 85, 1, 0],
      [10, 90, 1, 1],
      [70, 80, 0, 0],
      [70, 90, 0, 1],
      [70, 80, 0, 0]
    ]) {
      fixture.writeInput(
        positions,
        Float32Array.of(-170, south, -100, south, -100, north, -170, north)
      );
      const actual = await fixture.submitTwice(compiled, async () => ({
        publicStatus: await fixture.read(shared),
        source: await fixture.readRows([source.status.count, source.status.overflow]),
        consumer: await fixture.readRows([status.count, status.overflow, status.candidateOverflow]),
        cells: await fixture.read(cells)
      }));
      expect([
        actual.publicStatus[0],
        actual.publicStatus[64],
        actual.publicStatus[128],
        actual.publicStatus[192]
      ]).toEqual([capacityLoss, clipping, capacityLoss ? 2 : 1, capacityLoss || clipping ? 0 : 1]);
      expect(actual.source).toEqual([1, 0]);
      expect(actual.consumer).toEqual([1, 0, capacityLoss || clipping ? 1 : 0]);
      expect(actual.cells).toEqual([0xffffffff, 0x4820ffff, 0, 0, 0, 0, 0, 0]);
    }
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});

it('B4 deduplicates overlapping feature covers before counted outline and preserves loss', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createOutputOracleFixture(device);
  const {graph, input, output} = fixture;
  const rectangle = (north: number) => [-170, 10, -100, 10, -100, north, -170, north];
  const positions = input(
    'positions',
    'float32x2',
    Float32Array.from([...rectangle(85), ...rectangle(85)])
  );
  const source = {
    cells: output('cover', 'uint32x2', 8),
    status: {
      count: output('cover-count', 'uint32', 1),
      overflow: output('cover-overflow', 'uint32', 1),
      candidateOverflow: output('cover-candidate', 'uint32', 1),
      complete: output('complete', 'uint32', 1)
    }
  };
  graph.add(
    new GPUCellCover({
      family: 'quadbin',
      resolution: 2,
      positions,
      featureOffsets: input('features', 'uint32', Uint32Array.of(0, 1, 2)),
      polygonOffsets: input('polygons', 'uint32', Uint32Array.of(0, 1, 2)),
      ringOffsets: input('rings', 'uint32', Uint32Array.of(0, 4, 8)),
      candidateCapacity: 8,
      candidateEnumeration: 'bounds',
      edgeSlabs: false,
      output: {...source, featureIds: output('feature-ids', 'uint32', 8)}
    })
  );
  const sorted = {
    cells: output('sorted', 'uint32x2', 4),
    status: {
      count: output('sorted-count', 'uint32', 1),
      overflow: output('sorted-overflow', 'uint32', 1),
      candidateOverflow: output('sorted-candidate', 'uint32', 1)
    }
  };
  graph.add(
    new GPUCellCompaction({
      family: 'quadbin',
      source,
      operation: {type: 'compact', resolution: 2, minimumResolution: 2},
      output: sorted
    })
  );
  const outline = {
    cells: output('outline-cells', 'uint32x2', 16),
    rows: output('rows', 'uint32', 16),
    edgeIndices: output('edges', 'uint32', 16),
    endpoints: output('endpoints', 'float32x4', 16),
    status: {
      count: output('outline-count', 'uint32', 1),
      overflow: output('outline-overflow', 'uint32', 1),
      candidateOverflow: output('outline-candidate', 'uint32', 1)
    }
  };
  graph.add(new GPUCellSetOutline({family: 'quadbin', source: sorted, output: outline}));
  const compiled = graph.compile();
  try {
    for (const [north, complete] of [
      [85, 1],
      [90, 0],
      [85, 1]
    ]) {
      fixture.writeInput(positions, Float32Array.from([...rectangle(north), ...rectangle(north)]));
      const actual = await fixture.submitTwice(compiled, async () => ({
        source: await fixture.readRows([
          source.status.count,
          source.status.overflow,
          source.status.candidateOverflow,
          source.status.complete
        ]),
        sorted: await fixture.readRows([
          sorted.status.count,
          sorted.status.overflow,
          sorted.status.candidateOverflow
        ]),
        outline: await fixture.readRows([
          outline.status.count,
          outline.status.overflow,
          outline.status.candidateOverflow
        ]),
        cells: await fixture.read(sorted.cells)
      }));
      // Two features each cover the same two adjacent tiles: 4 rows -> 2 distinct cells -> 6 edges.
      expect(actual.source).toEqual([4, 0, 0, complete]);
      expect(actual.sorted).toEqual([2, 0, 1 - complete]);
      expect(actual.outline).toEqual([6, 0, 1 - complete]);
      if (complete)
        expect(actual.cells).toEqual([0xffffffff, 0x4820ffff, 0xffffffff, 0x4822ffff, 0, 0, 0, 0]);
    }
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});

// F4: requires the exact unowned outline-ring integration patch in review-forwarded.patch.
it('F4 direct counted outline suppresses a closed ring on incomplete source and recovers', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createOutputOracleFixture(device);
  const {graph, input, output} = fixture;
  const complete = input('complete', 'uint32', Uint32Array.of(1));
  const source = {
    cells: input('source', 'uint32x2', Uint32Array.of(0xffffffff, 0x4820ffff)),
    status: {
      count: input('source-count', 'uint32', Uint32Array.of(1)),
      overflow: input('source-overflow', 'uint32', Uint32Array.of(0)),
      candidateOverflow: input('source-candidate', 'uint32', Uint32Array.of(0)),
      complete
    }
  };
  const outline = {
    cells: output('cells', 'uint32x2', 8),
    rows: output('rows', 'uint32', 8),
    edgeIndices: output('edges', 'uint32', 8),
    endpoints: output('endpoints', 'float32x4', 8),
    status: {
      count: output('count', 'uint32', 1),
      overflow: output('overflow', 'uint32', 1),
      candidateOverflow: output('candidate', 'uint32', 1)
    }
  };
  const rings = {
    ringOffsets: output('ring-offsets', 'uint32', 3),
    positions: output('ring-positions', 'float32x2', 10),
    status: {
      count: output('ring-count', 'uint32', 1),
      overflow: output('ring-overflow', 'uint32', 1)
    }
  };
  graph.add(
    new GPUCellSetOutline({family: 'quadbin', source, output: outline, rings: {output: rings}})
  );
  const compiled = graph.compile();
  try {
    for (const value of [1, 0, 1]) {
      fixture.writeInput(complete, Uint32Array.of(value));
      const actual = await fixture.submitTwice(compiled, () =>
        fixture.readRows([
          outline.status.count,
          outline.status.candidateOverflow,
          rings.status.count
        ])
      );
      // One retained tile has four edges and one closed ring, but cannot certify the whole
      // missing-source boundary: ring publication must be suppressed when complete is zero.
      expect(actual).toEqual([4, 1 - value, value]);
    }
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});
