// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import shapelyFixture from './trajectory-restriction-shapely-fixture.json';
import {
  computeTrajectoryRestrictionOnCPU,
  CPUTrajectoryRestrictionOperation,
  type CPUTrajectoryRestrictionResult
} from '../../../src/gpu-spatial-analysis/trajectory-restriction/cpu-trajectory-restriction-operation';
import {GPUTrajectoryRestrictionOperation} from '../../../src/gpu-spatial-analysis/trajectory-restriction/gpu-trajectory-restriction-operation';
import {
  createDecreasingOffsetPairScene,
  createDecreasingOffsetGeometryScene,
  createPairScene,
  createPolygonScene,
  createRectangleScene,
  createReversedRangeOrderGeometryScene,
  createStaggeredPairScene,
  createWordScene,
  createZeroRowScene,
  getSceneInputs,
  getSceneProps
} from './trajectory-restriction-operation-cases';

function getPresentIntervals(
  result: CPUTrajectoryRestrictionResult,
  begin = 0,
  end = result.status.length
): number[][] {
  const intervals: number[][] = [];
  for (let row = begin; row < end; row++)
    if ((result.status[row] & 3) !== 0)
      intervals.push(Array.from(result.intervals.subarray(row * 2, row * 2 + 2)));
  return intervals;
}

it('matches independent GEOS intersections for concavity, a hole, reversal and empty/interior tracks', () => {
  const trackCount = shapelyFixture.tracks.length;
  const result = computeTrajectoryRestrictionOnCPU(
    {
      positions: Float32Array.from(shapelyFixture.tracks.flat(2)),
      timestamps: Float32Array.from(shapelyFixture.tracks.flatMap(() => [0, 16])),
      trackOffsets: Uint32Array.from({length: trackCount + 1}, (_, track) => track * 2),
      parameters: Float32Array.from([0, 16, 0, 0, 0, 0, 0, ...shapelyFixture.rings.flat(2)])
    },
    {
      polygonRingVertexCounts: shapelyFixture.rings.map(ring => ring.length),
      intervalsPerSegment: 16,
      mergeIntervals: true,
      compactCapacity: 24
    }
  );
  expect(result.compactSummary![0]).toBe(shapelyFixture.expected.length);
  expect(result.compactSummary![2]).toBe(0);
  for (const [row, expected] of shapelyFixture.expected.entries()) {
    for (let component = 0; component < 2; component++)
      expect(
        Math.abs(result.compactIntervals![row * 2 + component] - expected.interval[component])
      ).toBeLessThanOrEqual(2e-6 * Math.max(1, Math.abs(expected.interval[component])));
    for (let component = 0; component < 4; component++)
      expect(
        Math.abs(result.compactPositions![row * 4 + component] - expected.positions[component])
      ).toBeLessThanOrEqual(2e-6 * Math.max(1, Math.abs(expected.positions[component])));
    expect(Math.floor(result.compactSources![row * 2] / 2)).toBe(expected.track);
    expect(Math.floor(result.compactSources![row * 2 + 1] / 2)).toBe(expected.track);
  }
});

it('visits every overlapping segment pair in staggered tracks, in source order', () => {
  const {input, options} = createStaggeredPairScene();
  const result = computeTrajectoryRestrictionOnCPU(input, options);
  // The same [1,3] encounter is split at A's fixes (t=1,3) and B's interior fix (t=2).
  for (const begin of [0, 12])
    expect(getPresentIntervals(result, begin, begin + 12)).toEqual([
      [1, 1],
      [1, 2],
      [2, 3],
      [3, 3]
    ]);
  // Overlaps: 1 + 2 + 2 + 2, including the two last-fix endpoint matches.
  expect(Array.from(result.relationships)).toEqual([1, 0, 7, 0, 1, 0, 7, 0]);
});

it('clips two tracks against the hand-computed box, preserving all ordinary output fields', () => {
  const {input, options} = createRectangleScene();
  const result = computeTrajectoryRestrictionOnCPU(input, options);
  // Track 0 crosses x=-1 at t=1 and x=1 at t=3, then stays outside. Track 1 is stationary inside.
  expect(Array.from(result.intervals)).toEqual([1, 3, NaN, NaN, NaN, NaN, 0, 3, 3, 3]);
  expect(Array.from(result.positions)).toEqual([
    -1,
    0,
    1,
    0,
    NaN,
    NaN,
    NaN,
    NaN,
    NaN,
    NaN,
    NaN,
    NaN,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0
  ]);
  expect(Array.from(result.status)).toEqual([33, 32, 32, 49, 50]);
  expect(Array.from(result.relationships)).toEqual([1, 0, 3, 0, 1, 1, 2, 0]);
});

it('uses hand-read outer/hole intersections, collinear pieces and stable unions on separate tracks', () => {
  const {input, options} = createPolygonScene();
  for (const geometryBoundary of ['closed', 'open'] as const) {
    const result = computeTrajectoryRestrictionOnCPU(input, {...options, geometryBoundary});
    expect(getPresentIntervals(result, 0, 4)).toEqual([
      [2, 4],
      [8, 10]
    ]);
    expect(getPresentIntervals(result, 8, 12)).toEqual(
      geometryBoundary === 'closed'
        ? [
            [0, 1],
            [1, 5],
            [5, 6]
          ]
        : [
            [0, 1],
            [5, 6]
          ]
    );
    expect(Array.from(result.relationships)).toEqual([
      1,
      0,
      2,
      0,
      1,
      geometryBoundary === 'closed' ? 1 : 0,
      2,
      0
    ]);
    expect(Array.from(result.compactSummary!)).toEqual(
      geometryBoundary === 'closed' ? [3, 3, 0] : [4, 4, 0]
    );
    expect(
      Array.from(result.compactIntervals!.subarray(0, geometryBoundary === 'closed' ? 6 : 8))
    ).toEqual(geometryBoundary === 'closed' ? [2, 4, 8, 10, 0, 6] : [2, 4, 8, 10, 0, 1, 5, 6]);
    expect(
      Array.from(result.compactSources!.subarray(0, geometryBoundary === 'closed' ? 6 : 8))
    ).toEqual(geometryBoundary === 'closed' ? [0, 0, 0, 0, 2, 3] : [0, 0, 0, 0, 2, 2, 2, 3]);
    expect(Array.from(result.status.subarray(0, 2))).toEqual(
      geometryBoundary === 'closed' ? [33, 1] : [225, 193]
    );
    expect(Array.from(result.compactPositions!.subarray(0, 8))).toEqual([0, 0, 2, 0, 6, 0, 8, 0]);
  }
});

it('counts both slot truncation and compact truncation with the existing three-word summary', () => {
  const {input, options} = createPolygonScene();
  const result = computeTrajectoryRestrictionOnCPU(input, {
    ...options,
    intervalsPerSegment: 1,
    compactCapacity: 1,
    mergeIntervals: false
  });
  // First observation needs two slots, third needs three; each contributes one overflow count.
  // Three surviving slots remain (including track 1's last-fix singleton), of which one fits.
  expect(Array.from(result.status)).toEqual([289, 32, 305, 50]);
  expect(Array.from(result.compactSummary!)).toEqual([3, 1, 4]);
  expect(Array.from(result.compactIntervals!)).toEqual([2, 4]);
  expect(Array.from(result.compactSources!)).toEqual([0, 0]);
  const zero = computeTrajectoryRestrictionOnCPU(input, {...options, compactCapacity: 0});
  expect(zero.compactIntervals).toHaveLength(0);
  expect(Array.from(zero.compactSummary!)).toEqual([3, 0, 3]);
});

it('solves a closed-form distance encounter and self-overlap, invalid and inactive pair rows', () => {
  const {input, options} = createPairScene();
  const result = computeTrajectoryRestrictionOnCPU(input, options);
  // |t-2| <= 1 gives [1,3]. The stationary self-pair has a segment plus three endpoint overlaps.
  expect(getPresentIntervals(result, 0, 8)).toEqual([[1, 3]]);
  expect(getPresentIntervals(result, 8, 16)).toEqual([
    [0, 4],
    [4, 4],
    [4, 4],
    [4, 4]
  ]);
  expect(getPresentIntervals(result, 16)).toEqual([]);
  expect(Array.from(result.positions.subarray(0, 4))).toEqual([-1, 0, 1, 0]);
  expect(Array.from(result.relationships)).toEqual([
    1, 0, 4, 0, 1, 1, 4, 0, 0, 0, 4, 0, 0xffffffff, 0xffffffff, 0, 1, 0xffffffff, 0xffffffff, 0, 0
  ]);
  expect(result.status[0]).toBe(33);
  expect(Array.from(result.status.subarray(8, 12))).toEqual([49, 50, 50, 50]);
  expect(result.status[24]).toBe(4);
  expect(result.status.subarray(32).every(value => value === 0)).toBe(true);
  expect(result.intervals.subarray(32).every(Number.isNaN)).toBe(true);
  const overflow = computeTrajectoryRestrictionOnCPU(input, {...options, intervalsPerPair: 1});
  expect(overflow.status[1]).toBe(305);
  expect(Array.from(overflow.relationships)).toEqual(Array.from(result.relationships));
});

it('handles time-open and geometry-open pair boundaries and a tangent singleton', () => {
  const {input, options} = createPairScene();
  const open = computeTrajectoryRestrictionOnCPU(input, {
    ...options,
    timeBoundary: 'open',
    geometryBoundary: 'open'
  });
  expect(open.status[0]).toBe(225); // open radius intersections at t=1,3
  expect(open.status[8]).toBe(241); // stationary interior over the open time window
  expect(getPresentIntervals(open, 8, 16)).toEqual([[0, 4]]);
  const tangentInput = {
    ...input,
    positions: Float32Array.of(-2, 1, 2, 1, 0, 0, 0, 0),
    timestamps: Float32Array.of(0, 4, 0, 4),
    trackOffsets: Uint32Array.of(0, 2, 4),
    pairs: undefined
  };
  const tangentOptions = {intervalsPerPair: 4, pairwiseTracks: [[0, 1] as const]};
  expect(
    getPresentIntervals(computeTrajectoryRestrictionOnCPU(tangentInput, tangentOptions))
  ).toEqual([[2, 2]]);
  expect(
    getPresentIntervals(
      computeTrajectoryRestrictionOnCPU(tangentInput, {...tangentOptions, geometryBoundary: 'open'})
    )
  ).toEqual([]);
});

it('retains measured endpoints across strict gaps with Int64 origin carry and merge provenance', () => {
  const {input, options} = createWordScene();
  const result = computeTrajectoryRestrictionOnCPU(input, options);
  expect(getPresentIntervals(result)).toEqual([
    [0, 2],
    [2, 2],
    [7, 7]
  ]);
  expect(Array.from(result.status)).toEqual([49, 58, 50]);
  expect(Array.from(result.relationships)).toEqual([1, 0xffffffff, 3, 1]);
  expect(Array.from(result.compactIntervals!.subarray(0, 4))).toEqual([0, 2, 7, 7]);
  expect(Array.from(result.compactSources!.subarray(0, 4))).toEqual([0, 1, 2, 2]);
  expect(Array.from(result.compactSummary!)).toEqual([2, 2, 0]);
});

it('counts invalid segments once and gap-overlap unknowns separately for pairwise tracks', () => {
  const input = {
    positions: Float32Array.of(0, 0, NaN, 0, 2, 0, 0, 0, 2, 0, 4, 0),
    timestamps: Float32Array.of(0, 1, 2, 0, 2, 4),
    trackOffsets: Uint32Array.of(0, 3, 6),
    parameters: Float32Array.of(0, 10, 1, 0, 0, 0, 0)
  };
  const result = computeTrajectoryRestrictionOnCPU(input, {
    pairwiseTracks: [[0, 1]],
    intervalsPerPair: 4
  });
  expect(Array.from(result.relationships)).toEqual([1, 0xffffffff, 2, 2]);
  expect(result.status[0] & 4).toBe(4);
  const gap = computeTrajectoryRestrictionOnCPU(
    {
      ...input,
      positions: Float32Array.of(0, 0, 1, 0, 2, 0, 0, 0, 2, 0, 4, 0),
      parameters: Float32Array.of(0, 10, 1, 0, 0, 0, 1)
    },
    {pairwiseTracks: [[0, 1]], intervalsPerPair: 16}
  );
  // Both [0,1] and [1,2] overlap the other track's [0,2] gap for positive duration.
  expect(gap.relationships[3]).toBe(2);
  expect(gap.status[0] & 8).toBe(8);
});

it('handles open time endpoints, empty tracks, degenerate boxes and collapsed rings', () => {
  const input = {
    positions: Float32Array.of(0, 0, 0, 0),
    timestamps: Float32Array.of(0, 2),
    trackOffsets: Uint32Array.of(0, 0, 2),
    parameters: Float32Array.of(0, 2, 0, 0, 0, 0, 0)
  };
  const closed = computeTrajectoryRestrictionOnCPU(input);
  expect(Array.from(closed.relationships)).toEqual([0xffffffff, 0xffffffff, 0, 0, 1, 1, 2, 0]);
  expect(getPresentIntervals(closed)).toEqual([
    [0, 2],
    [2, 2]
  ]);
  const openTime = computeTrajectoryRestrictionOnCPU(input, {timeBoundary: 'open'});
  expect(Array.from(openTime.status)).toEqual([241, 0]);
  expect(Array.from(openTime.relationships.subarray(4))).toEqual([1, 1, 1, 0]);
  expect(
    getPresentIntervals(computeTrajectoryRestrictionOnCPU(input, {geometryBoundary: 'open'}))
  ).toEqual([]);
  for (const geometryBoundary of ['closed', 'open'] as const) {
    const ring = computeTrajectoryRestrictionOnCPU(
      {...input, parameters: Float32Array.of(0, 2, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0)},
      {polygonRingVertexCounts: [3], geometryBoundary}
    );
    expect(getPresentIntervals(ring)).toEqual(
      geometryBoundary === 'closed'
        ? [
            [0, 2],
            [2, 2]
          ]
        : []
    );
  }
});

it('uses exact orientation for a hand-constructed vertex touch and no tolerance band', () => {
  const parameters = Float32Array.of(0, 4, 0, 0, 0, 0, 0, 0, 0, 100000, 100000, 0, 100000);
  const options = {polygonRingVertexCounts: [3], intervalsPerSegment: 4};
  // Interior is y>x. This track is 18 float32 steps below the long diagonal and strictly outside.
  const outside = {
    positions: Float32Array.of(50000, 49999.9296875, 60000, 59999.9296875),
    timestamps: Float32Array.of(0, 4),
    trackOffsets: Uint32Array.of(0, 2),
    parameters
  };
  expect(getPresentIntervals(computeTrajectoryRestrictionOnCPU(outside, options))).toEqual([]);
  // Crosses (0,0) but remains below y=x on both sides of the vertex's neighbouring edges.
  const touch = {...outside, positions: Float32Array.of(-1, 1, 1, -1)};
  expect(getPresentIntervals(computeTrajectoryRestrictionOnCPU(touch, options))).toEqual([[2, 2]]);
  expect(
    getPresentIntervals(
      computeTrajectoryRestrictionOnCPU(touch, {...options, geometryBoundary: 'open'})
    )
  ).toEqual([]);
});

it('drops GPU-only maxima while preserving outputs and accepting truly empty input', () => {
  const {input} = createRectangleScene();
  const manyVertices = 132;
  const ringInput = {
    ...input,
    parameters: Float32Array.from([
      0,
      6,
      -1,
      -1,
      1,
      1,
      0,
      ...Array.from({length: manyVertices}, (_, vertex) => (vertex % 2 ? [1, 1] : [-1, -1])).flat()
    ])
  };
  expect(() =>
    computeTrajectoryRestrictionOnCPU(ringInput, {
      polygonRingVertexCounts: [manyVertices],
      intervalsPerSegment: 257
    })
  ).not.toThrow();
  const empty = computeTrajectoryRestrictionOnCPU(
    {
      positions: new Float32Array(0),
      timestamps: new Float32Array(0),
      trackOffsets: Uint32Array.of(0),
      parameters: input.parameters
    },
    {compactCapacity: 1}
  );
  expect(empty.relationships).toHaveLength(0);
  expect(empty.intervals).toHaveLength(0);
  expect(Array.from(empty.compactSummary!)).toEqual([0, 0, 0]);
  expect(empty.compactPositions!.every(Number.isNaN)).toBe(true);
});

it('rejects malformed arrays with their argument names and clamps excessive offsets', () => {
  const {input} = createRectangleScene();
  for (const [argument, replacement] of [
    ['positions', new Float32Array(3)],
    ['timestamps', new Float32Array(1)],
    ['trackOffsets', new Uint32Array(0)],
    ['parameters', new Float32Array(6)]
  ] as const)
    expect(() => computeTrajectoryRestrictionOnCPU({...input, [argument]: replacement})).toThrow(
      new RegExp(argument)
    );
  expect(() =>
    computeTrajectoryRestrictionOnCPU(
      {...input, pairs: {first: Uint32Array.of(0), second: new Uint32Array(0)}},
      {intervalsPerPair: 1}
    )
  ).toThrow(/pairs/);
  expect(() =>
    computeTrajectoryRestrictionOnCPU({...input, timeOrigin: Uint32Array.of(0, 0)})
  ).toThrow(/timeOrigin/);
  const clamped = computeTrajectoryRestrictionOnCPU({
    ...input,
    trackOffsets: Uint32Array.of(0, 3, 999)
  });
  expect(clamped).toEqual(computeTrajectoryRestrictionOnCPU(input));
  expect(() =>
    computeTrajectoryRestrictionOnCPU({...input, trackOffsets: Uint32Array.of(0, 3, 1, 5)})
  ).toThrow(/trackOffsets/);
});

it('preserves reversed empty pair ranges instead of rejecting decreasing clamped offsets', () => {
  const scene = createDecreasingOffsetPairScene();
  const result = computeTrajectoryRestrictionOnCPU(scene.input, scene.options);
  // Track 0 is [1,0), hence empty. Track 1 is its sole measured fix. Their product has no support.
  expect(Array.from(result.relationships)).toEqual([0xffffffff, 0xffffffff, 0, 0]);
  expect(Array.from(result.status)).toEqual([0]);
  expect(Array.from(result.intervals)).toEqual([NaN, NaN]);
  expect(Array.from(result.positions)).toEqual([NaN, NaN, NaN, NaN]);
  // A self-pair on the surviving track still uses its timestamp and has one known singleton.
  const surviving = computeTrajectoryRestrictionOnCPU(scene.input, {
    ...scene.options,
    pairwiseTracks: [[1, 1]]
  });
  expect(Array.from(surviving.relationships)).toEqual([1, 1, 1, 0]);
  expect(Array.from(surviving.intervals)).toEqual([0, 0]);
  expect(Array.from(surviving.status)).toEqual([50]);
  expect(Array.from(scene.input.trackOffsets)).toEqual([1, 0, 1]);
});

it('keeps decreasing geometry ranges empty and compacts remaining output slots in source order', () => {
  const scene = createDecreasingOffsetGeometryScene();
  const result = computeTrajectoryRestrictionOnCPU(scene.input, scene.options);
  expect(Array.from(result.relationships)).toEqual([0xffffffff, 0xffffffff, 0, 0, 1, 1, 1, 0]);
  expect(Array.from(result.status)).toEqual([50]);
  expect(Array.from(result.compactSummary!)).toEqual([1, 1, 0]);
  expect(Array.from(result.compactSources!)).toEqual([0, 0]);
  expect(Array.from(result.compactIntervals!)).toEqual([0, 0]);
  const reversedScene = createReversedRangeOrderGeometryScene();
  const reversedTracks = computeTrajectoryRestrictionOnCPU(
    reversedScene.input,
    reversedScene.options
  );
  // Track traversal visits row 1 before row 0, but compaction scans slots in row order.
  expect(Array.from(reversedTracks.compactSummary!)).toEqual([2, 2, 0]);
  expect(Array.from(reversedTracks.compactSources!)).toEqual([0, 0, 1, 1]);
  expect(Array.from(reversedTracks.compactIntervals!)).toEqual([0, 0, 1, 1]);
  const touchingScene = createReversedRangeOrderGeometryScene(true);
  const touching = computeTrajectoryRestrictionOnCPU(touchingScene.input, touchingScene.options);
  // The GPU compactor's forward cursor stays at track 0 for source rows 0 and 1: both are below
  // offsets[1]=2. Its existing rule consequently merges these equal-time singletons, although
  // they originated in different tracks. Preserve this defined malformed-offset behavior.
  expect(Array.from(touching.compactSummary!)).toEqual([1, 1, 0]);
  expect(Array.from(touching.compactSources!.subarray(0, 2))).toEqual([0, 1]);
  expect(Array.from(touching.compactIntervals!.subarray(0, 2))).toEqual([0, 0]);
});

it('returns correctly shaped zero-row arrays for zero observations, with and without tracks', () => {
  for (const trackCount of [0, 1] as const) {
    const scene = createZeroRowScene(trackCount);
    const result = computeTrajectoryRestrictionOnCPU(scene.input, scene.options);
    expect(result.intervals).toHaveLength(0);
    expect(result.positions).toHaveLength(0);
    expect(result.status).toHaveLength(0);
    expect(Array.from(result.relationships)).toEqual(
      trackCount === 0 ? [] : [0xffffffff, 0xffffffff, 0, 0]
    );
    if (trackCount === 0) {
      expect(Array.from(result.compactSummary!)).toEqual([0, 0, 0]);
      expect(result.compactIntervals).toHaveLength(0);
      expect(result.compactPositions).toHaveLength(0);
      expect(result.compactSources).toHaveLength(0);
      expect(result.compactStatus).toHaveLength(0);
    }
    const props = getSceneProps(scene);
    expect(new GPUTrajectoryRestrictionOperation(props).outputs['intervals'].length).toBe(0);
    expect(new CPUTrajectoryRestrictionOperation(props).outputs['intervals'].length).toBe(0);
  }
});

it('validates graph output shapes and disjoint value names in both operation constructors', () => {
  const props = {...getSceneProps(createRectangleScene()), id: 'restriction-check'};
  for (const Operation of [GPUTrajectoryRestrictionOperation, CPUTrajectoryRestrictionOperation]) {
    expect(() => new Operation(props)).not.toThrow();
    expect(() => new Operation({...props, rowCount: -1})).toThrow(/^restriction-check/);
    expect(
      () => new Operation({...props, output: {...props.output, positions: props.positions}})
    ).toThrow(/^restriction-check/);
    expect(
      () => new Operation({...props, output: {...props.output, status: props.output.intervals}})
    ).toThrow(/^restriction-check/);
  }
  // As in the pilot, contributor-specific GPU checks occur when its views are bound on lowering.
  for (const override of [
    {trackCount: -1},
    {intervalsPerSegment: 0},
    {intervalsPerPair: 1},
    {polygonRingVertexCounts: [2]},
    {mergeIntervals: true},
    {timeOrigin: 'origin'},
    {spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'none'} as const}
  ])
    expect(() => new CPUTrajectoryRestrictionOperation({...props, ...override})).toThrow(
      /^restriction-check/
    );
  expect(
    () => new CPUTrajectoryRestrictionOperation({...props, intervalsPerSegment: 257})
  ).not.toThrow();
});

it('executes every CPU graph port and checks explicit row, track and pair counts', async () => {
  for (const scene of [
    createRectangleScene(),
    createPolygonScene(),
    createPairScene(),
    createWordScene(),
    createDecreasingOffsetPairScene(),
    createDecreasingOffsetGeometryScene(),
    createReversedRangeOrderGeometryScene(),
    createZeroRowScene(),
    createZeroRowScene(0)
  ]) {
    const operation = new CPUTrajectoryRestrictionOperation(getSceneProps(scene));
    const graphInputs = getSceneInputs(scene);
    const inputs = Object.fromEntries(
      Object.entries(operation.inputs).map(([port, value]) => {
        const input = graphInputs[value];
        const components = input.format.endsWith('x2') ? 2 : 1;
        const length = input.values.length / components;
        return [
          port,
          {
            format: input.format,
            length,
            chunks: [{values: input.values, length}],
            pack: () => input.values
          }
        ];
      })
    );
    const outputs = await operation.execute({inputs});
    const expected = computeTrajectoryRestrictionOnCPU(scene.input, scene.options);
    for (const [port, descriptor] of Object.entries(
      new GPUTrajectoryRestrictionOperation(getSceneProps(scene)).outputs
    ))
      expect(outputs[descriptor.value]).toEqual(
        expected[port as keyof CPUTrajectoryRestrictionResult]
      );
    expect(() =>
      new CPUTrajectoryRestrictionOperation({
        ...getSceneProps(scene),
        rowCount: scene.input.positions.length / 2 + 1
      }).execute({inputs})
    ).toThrow(/positions/);
    expect(() =>
      new CPUTrajectoryRestrictionOperation({
        ...getSceneProps(scene),
        trackCount: scene.input.trackOffsets.length
      }).execute({inputs})
    ).toThrow(/trackOffsets/);
    const pairs = scene.input.pairs;
    if (pairs)
      expect(() =>
        new CPUTrajectoryRestrictionOperation({
          ...getSceneProps(scene),
          pairCapacity: pairs.first.length + 1
        }).execute({inputs})
      ).toThrow(/pairs.first/);
  }
});

it('binary64 singleton separation remains evaluable when float32 squared distance overflows', () => {
  const separation = Math.fround(1e20);
  // Independent closed form: separation^2 is finite in binary64 and greater than radius^2 = 1.
  expect(Number.isFinite(separation * separation)).toBe(true);
  expect(separation * separation).toBeGreaterThan(1);
  const result = computeTrajectoryRestrictionOnCPU(
    {
      positions: Float32Array.of(0, 0, separation, 0),
      timestamps: Float32Array.of(0, 0),
      trackOffsets: Uint32Array.of(0, 1, 2),
      parameters: Float32Array.of(0, 0, 1, 0, 0, 0, 0)
    },
    {pairwiseTracks: [[0, 1]], intervalsPerPair: 1}
  );
  expect(Array.from(result.relationships)).toEqual([0, 0, 1, 0]);
  expect(result.status[0] & 4).toBe(0);
});
