// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect} from 'vitest';
import {
  BOW_TIE_SEGMENTS,
  canonicalizeRing,
  NONE,
  SQUARE_CANONICAL_RINGS,
  SQUARE_SEGMENTS,
  type createOracleScene,
  type Scene
} from './make-valid-output-oracles-fixtures';

export async function expectOracleOutput(
  fixture: ReturnType<typeof createOracleScene>,
  read: (view: GraphDataView) => Promise<number[]>,
  scene: Scene = 'squares'
) {
  const {output, statuses, lattice} = fixture;
  const {nodedSegments, directedEdges, polygons, diagnostics} = output;
  if (scene === 'uncertain') {
    // Each implicitly closed two-point ring contributes two traversals. The four cross-ring
    // pairs exceed the predicates' 200-order span; same-ring reversed adjacent edges do not.
    // A disconnected zero counter fails every present-destination case here.
    for (const status of Object.values(statuses)) {
      expect(await read(status.candidateOverflow)).toEqual([1]);
      if (status.uncertainCount) expect(await read(status.uncertainCount)).toEqual([4]);
    }
    expect(await read(statuses.noded.count)).toEqual([4]);
    expect(await read(statuses.directed.count)).toEqual([8]);
    return;
  }
  if (scene === 'hole') {
    // Shapely: one polygon with shell + hole, area 4*4 - 2*2 = 12. Features count
    // polygons, whereas polygon offsets count rings; copying one port to the other fails.
    expect(await read(statuses.noded.count)).toEqual([8]);
    expect(await read(statuses.directed.count)).toEqual([16]);
    expect(await read(statuses.top.count)).toEqual([1]);
    expect(await read(statuses.top.requiredRingCount)).toEqual([4]);
    for (const status of Object.values(statuses)) {
      expect(await read(status.overflow!)).toEqual([0]);
      expect(await read(status.candidateOverflow)).toEqual([0]);
    }
    expect(await read(polygons.featureOffsets)).toEqual([0, 1, 1, 1, 1, 1, 1, 1, 1]);
    expect(await read(polygons.polygonOffsets)).toEqual([0, 2, 2, 2, 2, 2, 2, 2, 2]);
    const offsets = await read(polygons.ringOffsets);
    expect(offsets).toEqual([0, 4, 8, 8, 8, 8, 8, 8, 8]);
    const positions = await read(polygons.positions);
    expect(canonicalizeRing(positions, 0, 4)).toEqual([
      [0, 0],
      [0, 4],
      [4, 0],
      [4, 4]
    ]);
    expect(canonicalizeRing(positions, 4, 8)).toEqual([
      [1, 1],
      [1, 3],
      [3, 1],
      [3, 3]
    ]);
    const areas = [0, 1].map(ring => {
      let doubledArea = 0;
      const start = offsets[ring];
      const end = offsets[ring + 1];
      for (let vertex = start; vertex < end; vertex++) {
        const next = vertex + 1 < end ? vertex + 1 : start;
        doubledArea +=
          positions[2 * vertex] * positions[2 * next + 1] -
          positions[2 * next] * positions[2 * vertex + 1];
      }
      return Math.abs(doubledArea) / 2;
    });
    expect(areas).toEqual([16, 4]);
    expect(areas[0] - areas[1]).toBe(12);
    return;
  }
  const count = scene === 'squares' ? 8 : scene === 'bow-tie' ? 6 : scene === 'bridge' ? 10 : 2;
  // Counts follow the source-edge census, including both traversals of a collapsed ring or
  // retraced bridge. Every directed side is present, including unbounded walks.
  for (const [status, expectedCount] of [
    [statuses.noded, count],
    [statuses.directed, 2 * count]
  ] as const) {
    expect(await read(status.count)).toEqual([expectedCount]);
    expect(await read(status.requiredCount)).toEqual([expectedCount]);
    expect(await read(status.overflow!)).toEqual([0]);
    expect(await read(status.candidateOverflow)).toEqual([0]);
  }
  for (const status of Object.values(statuses)) {
    if (status.uncertainCount) expect(await read(status.uncertainCount)).toEqual([0]);
  }
  expect(await read(statuses.top.count)).toEqual([scene === 'dangle' ? 0 : 2]);
  expect(await read(statuses.top.overflow!)).toEqual([0]);
  expect(await read(statuses.top.candidateOverflow)).toEqual([0]);
  if (lattice) {
    // gridSize=.3 rounds upward to .5; integer input minima (1,1) are already on that grid.
    expect(await read(lattice)).toEqual(scene === 'dangle' ? [1, 2, 0.5, 0] : [1, 1, 0.5, 0]);
  }
  const expectedDiagnostics =
    scene === 'squares'
      ? [8, 0, 0, 0, 0]
      : scene === 'bow-tie'
        ? [6, 0, 0, 0, 0]
        : scene === 'bridge'
          ? [8, 2, 2, 0, 2]
          : [0, 2, 0, 2, 2];
  const diagnosticViews = [
    diagnostics.closedRingCount,
    diagnostics.openChainCount,
    diagnostics.cutEdgeCount,
    diagnostics.dangleCount,
    diagnostics.unusedEdgeCount
  ];
  expect((await Promise.all(diagnosticViews.map(read))).flat()).toEqual(expectedDiagnostics);
  // Source classifications: closedRing=1, openChain|cut|unused=22,
  // openChain|dangle|unused=26. Retractions count source rows, not unique undirected edges.
  const classes =
    scene === 'bridge'
      ? [1, 1, 1, 1, 22, 1, 1, 1, 1, 22]
      : new Array<number>(count).fill(scene === 'dangle' ? 26 : 1);
  expect(await read(diagnostics.edgeClasses)).toEqual([
    ...classes,
    ...new Array(12 - count).fill(0)
  ]);
  if (scene === 'dangle') {
    // Closing traversal belongs to the last original row, not the coincident forward edge.
    expect(await read(nodedSegments.sourceEdgeIds)).toEqual([0, 1, ...new Array(10).fill(NONE)]);
    expect(await read(nodedSegments.sourceStartParameters)).toEqual(new Array(12).fill(0));
    expect(await read(nodedSegments.sourceEndParameters)).toEqual([1, 1, ...new Array(10).fill(0)]);
    // Two source traversals of one isolated segment, each expanded into two directed sides.
    // Every side has no continuation except its reverse, which paired-opposite traversal
    // excludes. The ring assembler's documented dangling bit is 2.
    expect(await read(directedEdges.flags)).toEqual([2, 2, 2, 2, ...new Array(20).fill(0)]);
    expect(await read(directedEdges.ringIds)).toEqual(new Array(24).fill(NONE));
  }

  if (scene === 'squares' || scene === 'bow-tie') {
    const segments = scene === 'squares' ? SQUARE_SEGMENTS : BOW_TIE_SEGMENTS;
    expect(await read(nodedSegments.endpoints)).toEqual([
      ...segments.flat(),
      ...new Array((12 - count) * 4).fill(0)
    ]);
    const sourceFeatures =
      scene === 'squares' ? [70, 70, 70, 70, 80, 80, 80, 80] : [70, 70, 70, 70, 70, 70];
    const sourceRings = scene === 'squares' ? [0, 0, 0, 0, 1, 1, 1, 1] : [0, 0, 0, 0, 0, 0];
    const sourceEdges = scene === 'squares' ? [0, 1, 2, 3, 4, 5, 6, 7] : [0, 0, 1, 2, 2, 3];
    const tail = new Array<number>(12 - count).fill(NONE);
    expect(await read(nodedSegments.sourceFeatureIds)).toEqual([...sourceFeatures, ...tail]);
    expect(await read(nodedSegments.sourceRingIds)).toEqual([...sourceRings, ...tail]);
    expect(await read(nodedSegments.sourceEdgeIds)).toEqual([...sourceEdges, ...tail]);
    expect(await read(nodedSegments.sourceStartParameters)).toEqual([
      ...(scene === 'squares' ? [0, 0, 0, 0, 0, 0, 0, 0] : [0, 0.5, 0, 0, 0.5, 0]),
      ...new Array(12 - count).fill(0)
    ]);
    expect(await read(nodedSegments.sourceEndParameters)).toEqual([
      ...(scene === 'squares' ? [1, 1, 1, 1, 1, 1, 1, 1] : [0.5, 1, 1, 0.5, 1, 1]),
      ...new Array(12 - count).fill(0)
    ]);
    // Each original segment owns its forward side 2*s and backward side 2*s+1.
    const directed = segments.flatMap(([firstX, firstY, secondX, secondY]) => [
      firstX,
      firstY,
      secondX,
      secondY,
      secondX,
      secondY,
      firstX,
      firstY
    ]);
    expect(await read(directedEdges.endpoints)).toEqual([
      ...directed,
      ...new Array((12 - count) * 8).fill(0)
    ]);
    expect(await read(directedEdges.sourceSegmentIds)).toEqual([
      ...segments.flatMap((_, index) => [index, index]),
      ...new Array(2 * (12 - count)).fill(NONE)
    ]);
    expect(await read(directedEdges.sourceFeatureIds)).toEqual([
      ...sourceFeatures.flatMap(identity => [identity, identity]),
      ...new Array(2 * (12 - count)).fill(NONE)
    ]);
    expect(await read(directedEdges.twinIds)).toEqual([
      ...segments.flatMap((_, index) => [2 * index + 1, 2 * index]),
      ...new Array(2 * (12 - count)).fill(NONE)
    ]);
    const positions = await read(polygons.positions);
    const ringOffsets = await read(polygons.ringOffsets);
    const polygonOffsets = await read(polygons.polygonOffsets);
    expect(polygonOffsets.slice(0, 3)).toEqual([0, 1, 2]);
    expect(await read(polygons.featureOffsets)).toEqual([0, 1, 2, 2, 2, 2, 2, 2, 2]);
    const canonical = [0, 1]
      .map(polygon => {
        const ring = polygonOffsets[polygon];
        return canonicalizeRing(positions, ringOffsets[ring], ringOffsets[ring + 1]);
      })
      .sort((first, second) => first[0][0] - second[0][0] || first[0][1] - second[0][1]);
    expect(canonical).toEqual(
      scene === 'squares'
        ? SQUARE_CANONICAL_RINGS
        : [
            [
              [0, 0],
              [1, 1],
              [2, 0]
            ],
            [
              [0, 2],
              [1, 1],
              [2, 2]
            ]
          ]
    );
    // Shoelace area checks ring connectivity as well as its set of vertices (Shapely areas).
    const areas = [0, 1]
      .map(polygon => {
        const ring = polygonOffsets[polygon];
        let twiceArea = 0;
        const start = ringOffsets[ring];
        const end = ringOffsets[ring + 1];
        for (let vertex = start; vertex < end; vertex++) {
          const next = vertex + 1 < end ? vertex + 1 : start;
          twiceArea +=
            positions[2 * vertex] * positions[2 * next + 1] -
            positions[2 * next] * positions[2 * vertex + 1];
        }
        return Math.abs(twiceArea) / 2;
      })
      .sort((first, second) => first - second);
    expect(areas).toEqual(scene === 'squares' ? [4, 6] : [1, 1]);
    if (scene === 'squares') {
      // Isolated CCW squares: forward sides walk bounded ring 0/2, reverse sides the
      // corresponding clockwise unbounded ring 1/3. Leaders are the lowest directed row.
      expect(await read(directedEdges.ringIds)).toEqual([
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        2,
        3,
        2,
        3,
        2,
        3,
        2,
        3,
        ...new Array(8).fill(NONE)
      ]);
      expect(await read(directedEdges.flags)).toEqual(new Array(24).fill(0));
      expect(await read(statuses.top.requiredRingCount)).toEqual([4]);
      expect((await read(polygons.sourceIds)).slice(0, 2)).toEqual([70, 80]);
    }
  }
}
