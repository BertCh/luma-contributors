// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import type {GPUUint32Rows} from '../../../src/analysis-core/kernels/gpu-contributor-types';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPU_SPATIAL_JOIN_NO_FEATURE as N,
  GPUPointInPolygonJoin,
  type GPUPointInPolygonJoinProps
} from '../../../src/gpu-spatial-analysis/spatial-join/index';
import {
  createInputBuffer,
  createOutputBuffer,
  createVectorView,
  readCompactIds,
  readUint32,
  sortNumbers,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  buildPolygonFeatureArrays,
  classifyPointInFeature,
  createRandom,
  joinPointsInPolygons,
  POLYGON_FEATURES,
  POLYGON_POINTS
} from './spatial-join-oracle';
import {requireTestDevice} from '../../utils/require-test-device';
import {registerLonLatPolygonStatusTests} from './lon-lat-polygon-status-fixtures';

type JoinBufferName =
  | 'points'
  | 'sourceIds'
  | 'polygonPositions'
  | 'featureOffsets'
  | 'polygonOffsets'
  | 'ringOffsets'
  | 'featureIds'
  | 'pointFeatureIds'
  | 'featureCounts'
  | 'overflow'
  | 'candidateCount'
  | 'candidateTruncated'
  | 'candidateOverflow'
  | 'uncertainCount'
  | 'matchIds'
  | 'matchCount'
  | 'matchOverflow'
  | 'matchTotal';

type JoinFixture = {
  graph: GPUCommandGraph;
  buffers: Record<JoinBufferName, Buffer>;
  props: GPUPointInPolygonJoinProps<GPUUint32Rows>;
};

function createJoinFixture(
  device: Device,
  points: [number, number][],
  options: Partial<GPUPointInPolygonJoinProps> & {
    matchCapacity?: number;
    chunkLengths?: number[];
  } = {}
): JoinFixture {
  const {matchCapacity, chunkLengths, ...overrides} = options;
  const graph = new GPUCommandGraph(device, {id: 'pip-join'});
  const arrays = buildPolygonFeatureArrays(POLYGON_FEATURES);
  const pointCount = points.length;
  const featureCount = POLYGON_FEATURES.length;
  const buffers: Record<JoinBufferName, Buffer> = {
    points: createInputBuffer(device, Float32Array.from(points.flat())),
    sourceIds: createInputBuffer(device, Uint32Array.from(points.map((_, index) => 1000 + index))),
    polygonPositions: createInputBuffer(device, arrays.polygonPositions),
    featureOffsets: createInputBuffer(device, arrays.featureOffsets),
    polygonOffsets: createInputBuffer(device, arrays.polygonOffsets),
    ringOffsets: createInputBuffer(device, arrays.ringOffsets),
    featureIds: createInputBuffer(device, Uint32Array.from([100, 101, 102, 103])),
    pointFeatureIds: createOutputBuffer(device, pointCount),
    featureCounts: createOutputBuffer(device, featureCount),
    overflow: createOutputBuffer(device, 1),
    candidateCount: createOutputBuffer(device, 1),
    candidateTruncated: createOutputBuffer(device, pointCount),
    candidateOverflow: createOutputBuffer(device, 1),
    uncertainCount: createOutputBuffer(device, 1),
    matchIds: createOutputBuffer(device, matchCapacity ?? pointCount),
    matchCount: createOutputBuffer(device, 1),
    matchOverflow: createOutputBuffer(device, 1),
    matchTotal: createOutputBuffer(device, 1)
  };
  const props: GPUPointInPolygonJoinProps<GPUUint32Rows> = {
    points: {
      kind: 'points',
      positions: importGraphBuffer(graph, 'points', buffers.points, 'float32x2', pointCount),
      sourceIds: importGraphBuffer(graph, 'source-ids', buffers.sourceIds, 'uint32', pointCount)
    },
    polygons: {
      kind: 'polygons',
      positions: importGraphBuffer(
        graph,
        'polygon-positions',
        buffers.polygonPositions,
        'float32x2',
        arrays.polygonPositions.length / 2
      ),
      featureOffsets: importGraphBuffer(
        graph,
        'feature-offsets',
        buffers.featureOffsets,
        'uint32',
        arrays.featureOffsets.length
      ),
      polygonOffsets: importGraphBuffer(
        graph,
        'polygon-offsets',
        buffers.polygonOffsets,
        'uint32',
        arrays.polygonOffsets.length
      ),
      ringOffsets: importGraphBuffer(
        graph,
        'ring-offsets',
        buffers.ringOffsets,
        'uint32',
        arrays.ringOffsets.length
      ),
      sourceIds: importGraphBuffer(graph, 'feature-ids', buffers.featureIds, 'uint32', featureCount)
    },
    candidateCapacity: 16,
    // These specs pin the closed rule (smallest row wins); the half-open default has its own spec.
    boundaryRule: 'closed',
    output: {
      pointFeatureIds: importGraphBuffer(
        graph,
        'point-feature-ids',
        buffers.pointFeatureIds,
        'uint32',
        pointCount
      ),
      featureCounts: importGraphBuffer(
        graph,
        'feature-counts',
        buffers.featureCounts,
        'uint32',
        featureCount
      ),
      matches: {
        ids: importGraphBuffer(
          graph,
          'match-ids',
          buffers.matchIds,
          'uint32',
          matchCapacity ?? pointCount
        ),
        status: {
          count: importGraphBuffer(graph, 'match-count', buffers.matchCount, 'uint32', 1),
          overflow: importGraphBuffer(graph, 'match-overflow', buffers.matchOverflow, 'uint32', 1),
          requiredCount: importGraphBuffer(graph, 'match-total', buffers.matchTotal, 'uint32', 1)
        }
      },
      status: {
        overflow: importGraphBuffer(graph, 'overflow', buffers.overflow, 'uint32', 1),
        candidateCount: importGraphBuffer(
          graph,
          'candidate-count',
          buffers.candidateCount,
          'uint32',
          1
        ),
        uncertainCount: importGraphBuffer(
          graph,
          'uncertain-count',
          buffers.uncertainCount,
          'uint32',
          1
        )
      }
    },
    ...overrides
  };
  props.output.candidateTruncated = importGraphBuffer(
    graph,
    'candidate-truncated',
    buffers.candidateTruncated,
    'uint32',
    pointCount
  );
  props.output.status.candidateOverflow = importGraphBuffer(
    graph,
    'candidate-overflow',
    buffers.candidateOverflow,
    'uint32',
    1
  );
  if (chunkLengths) {
    function chunkView<Format extends 'float32x2' | 'uint32'>(view: GraphDataView<Format>) {
      let firstRow = 0;
      return createVectorView(
        `chunks-${view.buffer.id}`,
        view.format,
        chunkLengths!.map(length => {
          const chunk = graph.createDataView(view.buffer, {
            format: view.format,
            length,
            byteOffset: firstRow * view.rowByteLength
          });
          firstRow += length;
          return chunk;
        })
      );
    }
    props.points = {
      ...props.points,
      positions: chunkView(props.points.positions as GraphDataView<'float32x2'>),
      sourceIds: chunkView(props.points.sourceIds as GraphDataView<'uint32'>)
    };
    props.output.pointFeatureIds = chunkView(
      props.output.pointFeatureIds as GraphDataView<'uint32'>
    );
    props.output.candidateTruncated = chunkView(
      props.output.candidateTruncated as GraphDataView<'uint32'>
    );
  }
  return {graph, buffers, props};
}

function destroyFixture(fixture: JoinFixture): void {
  for (const buffer of Object.values(fixture.buffers)) {
    buffer.destroy();
  }
}

it('GPUPointInPolygonJoin joins points to features and updates per frame', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createJoinFixture(device, POLYGON_POINTS);
  fixture.graph.add(new GPUPointInPolygonJoin(fixture.props));
  const compiled = fixture.graph.compile();
  const {buffers} = fixture;

  submitGraph(device, compiled, undefined);
  expect(await readUint32(buffers.pointFeatureIds, 10)).toEqual([
    100,
    N,
    100,
    100,
    101,
    102,
    102,
    N,
    N,
    N
  ]);
  expect(await readUint32(buffers.featureCounts, 4)).toEqual([3, 1, 2, 0]);
  expect(await readUint32(buffers.candidateCount, 1)).toEqual([9]);
  expect(await readUint32(buffers.uncertainCount, 1)).toEqual([0]);
  expect(await readUint32(buffers.overflow, 1)).toEqual([0]);
  expect((await readCompactIds(buffers.matchIds, buffers.matchCount)).sort(sortNumbers)).toEqual([
    1000, 1002, 1003, 1004, 1005, 1006
  ]);

  const moved = POLYGON_POINTS.map(point => [...point]);
  moved[8] = [6, 3];
  buffers.points.write(Float32Array.from(moved.flat()));
  submitGraph(device, compiled, undefined);
  expect((await readUint32(buffers.pointFeatureIds, 10))[8]).toBe(101);
  expect(await readUint32(buffers.featureCounts, 4)).toEqual([3, 2, 2, 0]);
  expect(await readUint32(buffers.candidateCount, 1)).toEqual([10]);

  compiled.destroy();
  destroyFixture(fixture);
});

it('GPUPointInPolygonJoin excludes boundary points when requested', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createJoinFixture(device, POLYGON_POINTS, {boundaryRule: 'open'});
  fixture.graph.add(new GPUPointInPolygonJoin(fixture.props));
  const compiled = fixture.graph.compile();
  submitGraph(device, compiled, undefined);
  expect((await readUint32(fixture.buffers.pointFeatureIds, 10))[3]).toBe(N);
  expect(await readUint32(fixture.buffers.featureCounts, 4)).toEqual([2, 1, 2, 0]);
  compiled.destroy();
  destroyFixture(fixture);
});

it('GPUPointInPolygonJoin reports candidate, leaf, and match overflow', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const containing = POLYGON_POINTS.map(point =>
    POLYGON_FEATURES.map((feature, row) =>
      classifyPointInFeature(point, feature) !== 'outside' ? row : -1
    ).filter(row => row >= 0)
  );

  const candidateFixture = createJoinFixture(device, POLYGON_POINTS, {candidateCapacity: 4});
  candidateFixture.graph.add(new GPUPointInPolygonJoin(candidateFixture.props));
  let compiled = candidateFixture.graph.compile();
  submitGraph(device, compiled, undefined);
  expect(await readUint32(candidateFixture.buffers.overflow, 1)).toEqual([1]);
  // Aggregate compact completeness includes intermediate loss even with ample match capacity.
  expect(await readUint32(candidateFixture.buffers.matchOverflow, 1)).toEqual([1]);
  expect(await readUint32(candidateFixture.buffers.candidateCount, 1)).toEqual([9]);
  const featureIds = await readUint32(candidateFixture.buffers.pointFeatureIds, 10);
  for (const [row, featureId] of featureIds.entries()) {
    expect(featureId === N || containing[row].includes(featureId - 100)).toBe(true);
  }
  compiled.destroy();
  destroyFixture(candidateFixture);

  const leafFixture = createJoinFixture(device, POLYGON_POINTS, {leafCapacity: 2});
  leafFixture.graph.add(new GPUPointInPolygonJoin(leafFixture.props));
  compiled = leafFixture.graph.compile();
  submitGraph(device, compiled, undefined);
  expect(await readUint32(leafFixture.buffers.overflow, 1)).toEqual([1]);
  expect(await readUint32(leafFixture.buffers.matchOverflow, 1)).toEqual([1]);
  expect((await readUint32(leafFixture.buffers.pointFeatureIds, 10)).slice(0, 7)).toEqual([
    100,
    N,
    100,
    100,
    101,
    N,
    N
  ]);
  compiled.destroy();
  destroyFixture(leafFixture);

  const matchFixture = createJoinFixture(device, POLYGON_POINTS, {matchCapacity: 3});
  matchFixture.graph.add(new GPUPointInPolygonJoin(matchFixture.props));
  compiled = matchFixture.graph.compile();
  submitGraph(device, compiled, undefined);
  const {buffers} = matchFixture;
  const matchIds = await readCompactIds(buffers.matchIds, buffers.matchCount);
  expect(matchIds.length).toBe(3);
  expect(matchIds.every(id => [1000, 1002, 1003, 1004, 1005, 1006].includes(id))).toBe(true);
  expect(await readUint32(buffers.matchTotal, 1)).toEqual([6]);
  expect(await readUint32(buffers.matchOverflow, 1)).toEqual([1]);
  expect(await readUint32(buffers.overflow, 1)).toEqual([1]);
  compiled.destroy();
  destroyFixture(matchFixture);
});

it('GPUPointInPolygonJoin preserves point chunks and matches a random oracle', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createJoinFixture(device, POLYGON_POINTS);
  const firstPoints = createInputBuffer(
    device,
    Float32Array.from(POLYGON_POINTS.slice(0, 6).flat())
  );
  const secondPoints = createInputBuffer(device, Float32Array.from(POLYGON_POINTS.slice(6).flat()));
  const firstIds = createOutputBuffer(device, 6);
  const secondIds = createOutputBuffer(device, 4);
  const {graph} = fixture;
  graph.add(
    new GPUPointInPolygonJoin({
      ...fixture.props,
      points: {
        kind: 'points',
        positions: createVectorView('chunked-points', 'float32x2', [
          importGraphBuffer(graph, 'p0', firstPoints, 'float32x2', 6),
          importGraphBuffer(graph, 'p1', secondPoints, 'float32x2', 4)
        ])
      },
      output: {
        ...fixture.props.output,
        pointFeatureIds: createVectorView('chunked-ids', 'uint32', [
          importGraphBuffer(graph, 'i0', firstIds, 'uint32', 6),
          importGraphBuffer(graph, 'i1', secondIds, 'uint32', 4)
        ]),
        matches: undefined
      }
    })
  );
  let compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  expect([...(await readUint32(firstIds, 6)), ...(await readUint32(secondIds, 4))]).toEqual([
    100,
    N,
    100,
    100,
    101,
    102,
    102,
    N,
    N,
    N
  ]);
  compiled.destroy();
  destroyFixture(fixture);
  for (const buffer of [firstPoints, secondPoints, firstIds, secondIds]) buffer.destroy();

  const random = createRandom(7);
  const lattice: [number, number][] = Array.from({length: 400}, (_, index) => [
    (index % 20) * 0.8 + 0.0625,
    Math.floor(index / 20) * 0.3 + 0.0625
  ]);
  for (let index = lattice.length - 1; index > 0; index--) {
    const swap = Math.floor(random() * (index + 1));
    [lattice[index], lattice[swap]] = [lattice[swap], lattice[index]];
  }
  const randomFixture = createJoinFixture(device, lattice, {candidateCapacity: 1024});
  randomFixture.graph.add(
    new GPUPointInPolygonJoin({
      ...randomFixture.props,
      polygons: {...randomFixture.props.polygons, sourceIds: undefined}
    })
  );
  compiled = randomFixture.graph.compile();
  submitGraph(device, compiled, undefined);
  const oracle = joinPointsInPolygons(lattice, POLYGON_FEATURES, true);
  expect(await readUint32(randomFixture.buffers.pointFeatureIds, 400)).toEqual(oracle.featureRows);
  expect(await readUint32(randomFixture.buffers.featureCounts, 4)).toEqual(oracle.counts);
  expect(await readUint32(randomFixture.buffers.overflow, 1)).toEqual([0]);
  compiled.destroy();
  destroyFixture(randomFixture);
});

registerLonLatPolygonStatusTests('GPUPointInPolygonJoin');

for (const edgeSlabs of [false, true]) {
  for (const querySort of [false, true]) {
    for (const candidateCapacity of [4, 32]) {
      for (const matchCapacity of [3, 10]) {
        it(`GPUPointInPolygonJoin packed and three uneven chunks agree (slabs ${edgeSlabs}, sorted ${querySort}, candidates ${candidateCapacity}, matches ${matchCapacity})`, async () => {
          const device = await getWebGPUTestDevice();
          requireTestDevice(device);
          const fixtures = [undefined, [2, 3, 5]].map(chunkLengths =>
            createJoinFixture(device, POLYGON_POINTS, {
              chunkLengths,
              edgeSlabs,
              querySort,
              candidateCapacity,
              matchCapacity
            })
          );
          const compiled = fixtures.map(fixture => {
            fixture.graph.add(new GPUPointInPolygonJoin(fixture.props));
            return fixture.graph.compile();
          });
          try {
            // Repeat encodings to catch per-feature counts or compact counts being cleared per chunk
            // or retained from the preceding frame.
            for (let frame = 0; frame < 2; frame++) {
              const framePoints = POLYGON_POINTS.map(point => [...point] as [number, number]);
              if (frame === 1) framePoints[8] = [6, 3];
              for (const fixture of fixtures)
                fixture.buffers.points.write(Float32Array.from(framePoints.flat()));
              for (const graph of compiled) submitGraph(device, graph, undefined);
              const results = await Promise.all(
                fixtures.map(async ({buffers}) => ({
                  assignments: await readUint32(buffers.pointFeatureIds, 10),
                  featureCounts: await readUint32(buffers.featureCounts, 4),
                  matches: await readCompactIds(buffers.matchIds, buffers.matchCount),
                  matchCount: await readUint32(buffers.matchCount, 1),
                  matchTotal: await readUint32(buffers.matchTotal, 1),
                  matchOverflow: await readUint32(buffers.matchOverflow, 1),
                  candidateTruncated: await readUint32(buffers.candidateTruncated, 10),
                  candidateOverflow: await readUint32(buffers.candidateOverflow, 1),
                  candidateCount: await readUint32(buffers.candidateCount, 1),
                  uncertainCount: await readUint32(buffers.uncertainCount, 1),
                  overflow: await readUint32(buffers.overflow, 1)
                }))
              );
              expect(results[1]).toEqual(results[0]);
              if (candidateCapacity === 32) {
                const oracle = joinPointsInPolygons(framePoints, POLYGON_FEATURES, true);
                expect(results[1].assignments).toEqual(
                  oracle.featureRows.map(row => (row === N ? N : 100 + row))
                );
                expect(results[1].featureCounts).toEqual(oracle.counts);
                // Stable compact IDs are in source-row order across chunk boundaries, also on overflow.
                expect(results[1].matches).toEqual(
                  oracle.featureRows
                    .flatMap((row, index) => (row === N ? [] : [1000 + index]))
                    .slice(0, matchCapacity)
                );
                expect(results[1].matchTotal).toEqual([frame === 0 ? 6 : 7]);
              }
            }
          } finally {
            for (const graph of compiled) graph.destroy();
            for (const fixture of fixtures) destroyFixture(fixture);
          }
        });
      }
    }
  }
}

for (const edgeSlabs of [false, true]) {
  it(`point candidates stream across pages and reset every encoding (slabs ${edgeSlabs})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createJoinFixture(device, POLYGON_POINTS, {
      candidateCapacity: 16,
      candidateTileCapacity: 2,
      edgeSlabs,
      querySort: true,
      spatialSort: true,
      chunkLengths: [1, 3, 6],
      matchCapacity: 3
    });
    fixture.graph.add(new GPUPointInPolygonJoin(fixture.props));
    const compiled = fixture.graph.compile();
    try {
      for (const populated of [true, false, true]) {
        fixture.buffers.points.write(
          populated ? Float32Array.from(POLYGON_POINTS.flat()) : new Float32Array(20).fill(1000)
        );
        submitGraph(device, compiled, undefined);
        expect(await readUint32(fixture.buffers.pointFeatureIds, 10)).toEqual(
          populated ? [100, N, 100, 100, 101, 102, 102, N, N, N] : new Array(10).fill(N)
        );
        expect(await readUint32(fixture.buffers.featureCounts, 4)).toEqual(
          populated ? [3, 1, 2, 0] : [0, 0, 0, 0]
        );
        expect(await readUint32(fixture.buffers.candidateCount, 1)).toEqual([populated ? 9 : 0]);
        expect(await readUint32(fixture.buffers.candidateOverflow, 1)).toEqual([0]);
        expect(await readUint32(fixture.buffers.candidateTruncated, 10)).toEqual(
          new Array(10).fill(0)
        );
        expect(await readUint32(fixture.buffers.uncertainCount, 1)).toEqual([0]);
        expect(await readUint32(fixture.buffers.matchTotal, 1)).toEqual([populated ? 6 : 0]);
        expect(await readCompactIds(fixture.buffers.matchIds, fixture.buffers.matchCount)).toEqual(
          populated ? [1000, 1002, 1003] : []
        );
        expect(await readUint32(fixture.buffers.matchOverflow, 1)).toEqual([Number(populated)]);
      }
    } finally {
      compiled.destroy();
      destroyFixture(fixture);
    }
  });
}

for (const edgeSlabs of [false, true]) {
  it(`point pages preserve the global candidate cutoff and compact prefix (slabs ${edgeSlabs})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const bounds = POLYGON_FEATURES.map(feature => {
      const vertices = feature.flat(2);
      return [
        Math.min(...vertices.map(point => point[0])),
        Math.min(...vertices.map(point => point[1])),
        Math.max(...vertices.map(point => point[0])),
        Math.max(...vertices.map(point => point[1]))
      ];
    });
    const candidates = POLYGON_POINTS.flatMap((point, pointRow) =>
      bounds.flatMap((box, featureRow) =>
        point[0] >= box[0] && point[0] <= box[2] && point[1] >= box[1] && point[1] <= box[3]
          ? [{pointRow, featureRow}]
          : []
      )
    );
    expect(candidates).toHaveLength(9);
    for (const candidateCapacity of [4, 7]) {
      const assigned = new Array<number>(POLYGON_POINTS.length).fill(N);
      for (const candidate of candidates.slice(0, candidateCapacity)) {
        if (
          classifyPointInFeature(
            POLYGON_POINTS[candidate.pointRow],
            POLYGON_FEATURES[candidate.featureRow]
          ) !== 'outside'
        )
          assigned[candidate.pointRow] = Math.min(
            assigned[candidate.pointRow],
            100 + candidate.featureRow
          );
      }
      const expectedTruncated = POLYGON_POINTS.map((_, pointRow) =>
        Number(
          candidates.slice(candidateCapacity).some(candidate => candidate.pointRow === pointRow)
        )
      );
      const matches = assigned.flatMap((feature, pointRow) =>
        feature === N ? [] : [1000 + pointRow]
      );
      const counts = POLYGON_FEATURES.map(
        (_, featureRow) => assigned.filter(feature => feature === 100 + featureRow).length
      );
      const fixture = createJoinFixture(device, POLYGON_POINTS, {
        candidateCapacity,
        candidateTileCapacity: 2,
        spatialSort: false,
        querySort: true,
        edgeSlabs,
        chunkLengths: [2, 1, 7],
        matchCapacity: 3
      });
      fixture.graph.add(new GPUPointInPolygonJoin(fixture.props));
      const compiled = fixture.graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        expect(await readUint32(fixture.buffers.pointFeatureIds, 10)).toEqual(assigned);
        expect(await readUint32(fixture.buffers.featureCounts, 4)).toEqual(counts);
        expect(await readUint32(fixture.buffers.candidateTruncated, 10)).toEqual(expectedTruncated);
        expect(await readUint32(fixture.buffers.candidateCount, 1)).toEqual([9]);
        expect(await readUint32(fixture.buffers.candidateOverflow, 1)).toEqual([1]);
        expect(await readUint32(fixture.buffers.matchTotal, 1)).toEqual([matches.length]);
        expect(await readCompactIds(fixture.buffers.matchIds, fixture.buffers.matchCount)).toEqual(
          matches.slice(0, 3)
        );
        expect(await readUint32(fixture.buffers.matchOverflow, 1)).toEqual([1]);
      } finally {
        compiled.destroy();
        destroyFixture(fixture);
      }
    }
  });
}
