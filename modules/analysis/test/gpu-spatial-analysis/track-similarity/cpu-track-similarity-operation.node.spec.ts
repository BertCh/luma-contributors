// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {
  CPUTrackSimilarityOperation,
  computeTrackSimilarityOnCPU,
  type CPUTrackSimilarityInput
} from '../../../src/gpu-spatial-analysis/track-similarity/cpu-track-similarity-operation';
import {
  GPUTrackSimilarityOperation,
  type TrackSimilarityOperationProps
} from '../../../src/gpu-spatial-analysis/track-similarity/gpu-track-similarity-operation';
import {SHAPELY_DENSIFY_CASES} from './track-similarity-shapely-cases';

type Point = readonly [number, number];

it('honors an explicit Frechet cap without suppressing original-vertex measures', () => {
  const result = computeTrackSimilarityOnCPU(
    makeInput(
      [
        [0, 0],
        [1, 0]
      ],
      [
        [0, 0],
        [1, 0]
      ]
    ),
    {maximumFrechetVertexCount: 1, measures: ['frechet', 'hausdorff', 'maximumDistance']}
  );
  expect(result.pairStatus[0]).toBe(2);
  expect(result.frechet[0]).toBeNaN();
  expect(result.hausdorff[0]).toBe(0);
  expect(result.maximumDistance[0]).toBe(1);
});

it('bounds remote antimeridian segment distances by the endpoints in both orientations', () => {
  for (const segment of [
    [
      [100, 0],
      [-100, 0]
    ],
    [
      [-100, 0],
      [100, 0]
    ]
  ] as const) {
    const result = computeTrackSimilarityOnCPU(makeInput([[0, 0]], segment), {
      spatialContext: {
        coordinateSpace: 'longitude-latitude',
        metric: 'great-circle',
        sphereRadius: 180 / Math.PI
      },
      measures: ['hausdorff', 'symmetricSegmentPathDistance', 'maximumDistance']
    });
    expect(result.pairStatus[0]).toBe(0);
    expect(result.maximumDistance[0]).toBe(100);
    expect(result.hausdorff[0]).toBe(100);
    expect(result.symmetricSegmentPathDistance[0]).toBe(100);
  }
});

function makeInput(first: readonly Point[], second: readonly Point[]): CPUTrackSimilarityInput {
  return {
    linesA: {
      positions: Float32Array.from(first.flat()),
      lineOffsets: Uint32Array.of(0, first.length)
    },
    linesB: {
      positions: Float32Array.from(second.flat()),
      lineOffsets: Uint32Array.of(0, second.length)
    },
    pairA: Uint32Array.of(0),
    pairB: Uint32Array.of(0),
    frechetEpsilon: Float32Array.of(1.5)
  };
}

it('reports both pair lengths when the CPU helper rejects unequal pairs', () => {
  const input = makeInput([[0, 0]], [[1, 0]]);
  expect(() => computeTrackSimilarityOnCPU({...input, pairA: Uint32Array.of(0, 0)})).toThrow(
    /^computeTrackSimilarityOnCPU pairB\.length must equal pairA\.length 2; received 1$/
  );
});

it('does not densify maximum-distance-only or segment-path-only requests', () => {
  const input = makeInput(
    [
      [0, 0],
      [2, 0]
    ],
    [
      [0, 1],
      [2, 1]
    ]
  );
  const originalPush = Array.prototype.push;
  // Stop the wrong implementation after 32 points, before its billion-point allocation.
  Array.prototype.push = function (this: unknown[], ...values: unknown[]): number {
    if (this.length >= 32 && values.some(value => Array.isArray(value) && value.length === 2)) {
      throw new Error('unexpected dense point allocation');
    }
    return originalPush.apply(this, values);
  };
  let result: ReturnType<typeof computeTrackSimilarityOnCPU>;
  try {
    result = computeTrackSimilarityOnCPU(input, {
      densify: 1e-9,
      measures: ['maximumDistance', 'symmetricSegmentPathDistance']
    });
  } finally {
    Array.prototype.push = originalPush;
  }
  expect(result.maximumDistance[0]).toBe(Math.fround(Math.sqrt(5)));
  expect(result.symmetricSegmentPathDistance[0]).toBe(1);
  expect(result.hausdorff[0]).toBeNaN();
  expect(result.pairStatus[0]).toBe(0);
});

it('checks expanded vertex count before allocating points for measures that densify', () => {
  const originalPush = Array.prototype.push;
  Array.prototype.push = function (this: unknown[], ...values: unknown[]): number {
    if (this.length >= 32 && values.some(value => Array.isArray(value) && value.length === 2)) {
      throw new Error('unexpected dense point allocation');
    }
    return originalPush.apply(this, values);
  };
  try {
    expect(() =>
      computeTrackSimilarityOnCPU(
        makeInput(
          [
            [0, 0],
            [1, 0]
          ],
          [[0, 0]]
        ),
        {
          densify: 1 / 2 ** 32,
          measures: ['frechet']
        }
      )
    ).toThrow(/densifiedVertexCount/);
  } finally {
    Array.prototype.push = originalPush;
  }
});

/** Enumerates every complete monotone coupling; no DP table or recurrence is used. */
function enumerateCouplings(first: readonly Point[], second: readonly Point[], band?: number) {
  const paths: number[][] = [];
  const visitPath = (firstRow: number, secondRow: number, costs: number[]) => {
    const longerRow = first.length >= second.length ? firstRow : secondRow;
    const shorterRow = first.length >= second.length ? secondRow : firstRow;
    if (
      band !== undefined &&
      (longerRow + band < shorterRow ||
        longerRow > shorterRow + Math.abs(first.length - second.length) + band)
    )
      return;
    const path = [
      ...costs,
      Math.hypot(
        first[firstRow][0] - second[secondRow][0],
        first[firstRow][1] - second[secondRow][1]
      )
    ];
    if (firstRow === first.length - 1 && secondRow === second.length - 1) {
      paths.push(path);
      return;
    }
    if (firstRow + 1 < first.length) visitPath(firstRow + 1, secondRow, path);
    if (secondRow + 1 < second.length) visitPath(firstRow, secondRow + 1, path);
    if (firstRow + 1 < first.length && secondRow + 1 < second.length)
      visitPath(firstRow + 1, secondRow + 1, path);
  };
  visitPath(0, 0, []);
  return {
    frechet: Math.min(...paths.map(path => Math.max(...path))),
    warping: Math.min(...paths.map(path => path.reduce((sum, cost) => sum + cost, 0)))
  };
}

/** Exhaustively scores complete edit scripts rather than optimizing prefixes. */
function enumerateEdits(
  first: readonly Point[],
  second: readonly Point[],
  tolerance: number,
  gap: Point
) {
  let bestCount = Infinity;
  let bestPenalty = Infinity;
  const visitScript = (firstRow: number, secondRow: number, count: number, penalty: number) => {
    if (firstRow === first.length && secondRow === second.length) {
      bestCount = Math.min(bestCount, count);
      bestPenalty = Math.min(bestPenalty, penalty);
      return;
    }
    if (firstRow < first.length)
      visitScript(
        firstRow + 1,
        secondRow,
        count + 1,
        penalty + Math.hypot(first[firstRow][0] - gap[0], first[firstRow][1] - gap[1])
      );
    if (secondRow < second.length)
      visitScript(
        firstRow,
        secondRow + 1,
        count + 1,
        penalty + Math.hypot(second[secondRow][0] - gap[0], second[secondRow][1] - gap[1])
      );
    if (firstRow < first.length && secondRow < second.length) {
      const distance = Math.hypot(
        first[firstRow][0] - second[secondRow][0],
        first[firstRow][1] - second[secondRow][1]
      );
      visitScript(
        firstRow + 1,
        secondRow + 1,
        count + Number(distance > tolerance),
        penalty + distance
      );
    }
  };
  visitScript(0, 0, 0, 0);
  return {count: bestCount, penalty: bestPenalty};
}

/** Exhaustive ordered subsets, independent of the LCSS recurrence. */
function enumerateSubsequence(
  first: readonly Point[],
  second: readonly Point[],
  tolerance: number,
  band?: number
): number {
  const getSubsets = (length: number) =>
    Array.from({length: 2 ** length}, (_, bits) =>
      Array.from({length}, (_, index) => index).filter(index => (bits & (1 << index)) !== 0)
    );
  let longest = 0;
  for (const firstRows of getSubsets(first.length)) {
    for (const secondRows of getSubsets(second.length)) {
      if (firstRows.length !== secondRows.length) continue;
      const matches = firstRows.every((firstRow, index) => {
        const secondRow = secondRows[index];
        const longerRow = first.length >= second.length ? firstRow : secondRow;
        const shorterRow = first.length >= second.length ? secondRow : firstRow;
        return (
          (band === undefined ||
            (longerRow + band >= shorterRow &&
              longerRow <= shorterRow + Math.abs(first.length - second.length) + band)) &&
          Math.hypot(
            first[firstRow][0] - second[secondRow][0],
            first[firstRow][1] - second[secondRow][1]
          ) <= tolerance
        );
      });
      if (matches) longest = Math.max(longest, firstRows.length);
    }
  }
  return longest / Math.min(first.length, second.length);
}

it('matches exhaustive couplings, edit scripts and ordered subsets on unequal nontrivial tracks', () => {
  const first: Point[] = [
    [0, 0],
    [2, 2],
    [4, 0]
  ];
  const second: Point[] = [
    [0, 1],
    [1, 0],
    [3, 0],
    [4, 1]
  ];
  for (const band of [undefined, 0, 1]) {
    const result = computeTrackSimilarityOnCPU(makeInput(first, second), {
      dynamicTimeWarpingBandRadius: band,
      lcssBandRadius: band,
      editDistanceTolerance: 1,
      lcssTolerance: 1,
      editDistanceGapPoint: [1, -1]
    });
    const coupling = enumerateCouplings(first, second);
    const bandCoupling = enumerateCouplings(first, second, band);
    const edits = enumerateEdits(first, second, 1, [1, -1]);
    expect(result.frechet[0]).toBeCloseTo(coupling.frechet, 6);
    expect(result.dynamicTimeWarping[0]).toBeCloseTo(bandCoupling.warping, 5);
    expect(result.editDistanceOnRealSequence[0]).toBe(edits.count);
    expect(result.editDistanceWithRealPenalty[0]).toBeCloseTo(edits.penalty, 5);
    expect(result.longestCommonSubsequence[0]).toBe(
      Math.fround(enumerateSubsequence(first, second, 1, band))
    );
    expect(result.frechetWithin[0]).toBe(Number(coupling.frechet <= 1.5));
    expect(result.pairStatus[0]).toBe(0);
  }
});

it('matches committed Shapely/GEOS fixtures including densified lines and closed rings', () => {
  for (const fixture of SHAPELY_DENSIFY_CASES) {
    const result = computeTrackSimilarityOnCPU(makeInput(fixture.a, fixture.b), {
      densify: fixture.densify,
      measures: ['hausdorff', 'frechet', 'maximumDistance']
    });
    expect(result.hausdorff[0]).toBeCloseTo(fixture.hausdorff, 5);
    expect(result.frechet[0]).toBeCloseTo(fixture.frechet, 5);
    expect(result.maximumDistance[0]).toBeCloseTo(fixture.maxDistance, 5);
  }
});

it('has the hand-computed parallel-segment distances and inclusive matching', () => {
  const result = computeTrackSimilarityOnCPU(
    makeInput(
      [
        [0, 0],
        [2, 0]
      ],
      [
        [0, 1],
        [1, 1],
        [2, 1]
      ]
    ),
    {lcssTolerance: 1, editDistanceTolerance: 1}
  );
  expect(result.hausdorff[0]).toBe(1);
  expect(result.symmetricSegmentPathDistance[0]).toBe(1);
  expect(result.maximumDistance[0]).toBe(Math.fround(Math.sqrt(5)));
  expect(result.frechet[0]).toBe(Math.fround(Math.sqrt(2)));
  expect(result.dynamicTimeWarping[0]).toBe(Math.fround(2 + Math.sqrt(2)));
  expect(result.editDistanceOnRealSequence[0]).toBe(1);
  expect(result.longestCommonSubsequence[0]).toBe(1);
  expect(result.frechetWithin[0]).toBe(1);
});

it('uses original vertices for SSPD and maximum distance when densification is enabled', () => {
  const input = makeInput(
    [
      [0, 0],
      [4, 0]
    ],
    [
      [0, 0],
      [2, 3],
      [4, 0]
    ]
  );
  const original = computeTrackSimilarityOnCPU(input);
  const dense = computeTrackSimilarityOnCPU(input, {densify: 0.25});
  expect(dense.symmetricSegmentPathDistance[0]).toBe(0.5);
  expect(dense.maximumDistance[0]).toBe(4);
  expect(dense.symmetricSegmentPathDistance).toEqual(original.symmetricSegmentPathDistance);
  expect(dense.maximumDistance).toEqual(original.maximumDistance);
});

it('crosses GPU strip boundaries when the CPU cap is omitted and drops subdivision limits', () => {
  const first = Array.from({length: 2050}, () => [3, 4] as const);
  const result = computeTrackSimilarityOnCPU(makeInput(first, first), {
    measures: ['frechet']
  });
  expect(result.frechet[0] === 0).toBe(true);
  expect(result.pairStatus[0]).toBe(0);
  const dense = computeTrackSimilarityOnCPU(
    makeInput(
      [
        [0, 0],
        [1, 0]
      ],
      [[0, 0]]
    ),
    {densify: 1 / 4097, measures: ['hausdorff']}
  );
  expect(dense.hausdorff[0]).toBe(1);
});

it('keeps inactive, invalid and empty pairs distinct, including all output missing values', () => {
  const input: CPUTrackSimilarityInput = {
    linesA: {positions: Float32Array.of(0, 0, 1, 0), lineOffsets: Uint32Array.of(0, 0, 2)},
    pairA: Uint32Array.of(1, 0, 99, 1),
    pairB: Uint32Array.of(1, 1, 1, 1),
    activePairCount: Uint32Array.of(3),
    frechetEpsilon: Float32Array.of(-1)
  };
  const result = computeTrackSimilarityOnCPU(input);
  expect(Array.from(result.pairStatus)).toEqual([16, 17, 4, 4]);
  for (const [port, values] of Object.entries(result)) {
    if (port === 'pairStatus') continue;
    if (port === 'frechetWithin') expect(Array.from(values)).toEqual([0, 0, 0, 0]);
    else expect(Array.from(values.slice(1)).every(Number.isNaN)).toBe(true);
  }
  for (const epsilon of [NaN, Infinity, -1]) {
    expect(
      computeTrackSimilarityOnCPU({...input, frechetEpsilon: Float32Array.of(epsilon)})
        .frechetWithin[0]
    ).toBe(0);
  }
});

it('matches closed-form equatorial arc lengths, collapsed segments and the antimeridian', () => {
  const context = {
    coordinateSpace: 'longitude-latitude',
    metric: 'great-circle',
    sphereRadius: 1000
  } as const;
  const result = computeTrackSimilarityOnCPU(
    makeInput(
      [
        [179, 0],
        [-179, 0]
      ],
      [
        [180, 0],
        [-180, 0]
      ]
    ),
    {spatialContext: context, densify: 0.5}
  );
  const oneDegree = Math.fround((1000 * Math.PI) / 180);
  expect(result.hausdorff[0]).toBe(oneDegree);
  expect(result.symmetricSegmentPathDistance[0]).toBe(Math.fround(oneDegree / 2));
  expect(result.maximumDistance[0]).toBe(oneDegree);
  expect(result.frechet[0]).toBe(oneDegree);
  const coincident = computeTrackSimilarityOnCPU(
    makeInput(
      [[-180, 0]],
      [
        [180, 0],
        [180, 0]
      ]
    ),
    {spatialContext: context, editDistanceGapPoint: [180, 0]}
  );
  for (const port of [
    'hausdorff',
    'frechet',
    'maximumDistance',
    'symmetricSegmentPathDistance',
    'dynamicTimeWarping',
    'editDistanceWithRealPenalty'
  ] as const) {
    expect(coincident[port][0] === 0).toBe(true);
  }
});

it('retains finite float64 path costs until the single output rounding', () => {
  const result = computeTrackSimilarityOnCPU(
    makeInput(
      [
        [1e20, 0],
        [1e20, 0]
      ],
      [
        [0, 0],
        [0, 0]
      ]
    )
  );
  expect(result.frechet[0]).toBe(Math.fround(1e20));
  expect(result.dynamicTimeWarping[0]).toBe(Math.fround(2 * Math.fround(1e20)));
});

it('throws with array names for inconsistent packed arrays', () => {
  const input = makeInput([[0, 0]], [[1, 1]]);
  expect(() => computeTrackSimilarityOnCPU({...input, pairB: new Uint32Array()})).toThrow('pairB');
  expect(() => computeTrackSimilarityOnCPU({...input, activePairCount: new Uint32Array()})).toThrow(
    'activePairCount'
  );
  expect(() => computeTrackSimilarityOnCPU({...input, frechetEpsilon: new Float32Array()})).toThrow(
    'frechetEpsilon'
  );
  expect(() =>
    computeTrackSimilarityOnCPU({
      ...input,
      linesA: {...input.linesA, positions: Float32Array.of(1)}
    })
  ).toThrow('linesA.positions');
  for (const offsets of [new Uint32Array(), Uint32Array.of(0, 2), Uint32Array.of(1, 0)]) {
    expect(() =>
      computeTrackSimilarityOnCPU({...input, linesA: {...input.linesA, lineOffsets: offsets}})
    ).toThrow('linesA.lineOffsets');
  }
});

const PROPS: TrackSimilarityOperationProps = {
  id: 'similarity-check',
  linesA: {positions: 'positions', lineOffsets: 'offsets'},
  pairA: 'pairA',
  pairB: 'pairB',
  pairCount: 2,
  output: {frechet: 'frechet', pairStatus: 'status'}
};

describe('construction-time validation', () => {
  for (const Operation of [GPUTrackSimilarityOperation, CPUTrackSimilarityOperation]) {
    it(`${Operation.name} checks fixed output lengths and graph-value collisions`, () => {
      expect(() => new Operation({...PROPS, pairCount: 0})).not.toThrow();
      expect(() => new Operation({...PROPS, pairCount: 1.5})).toThrow(
        /^similarity-check.*(length|pairCount)/
      );
      expect(() => new Operation({...PROPS, output: {frechet: 'positions'}})).toThrow(
        /^similarity-check.*outputs/
      );
      expect(() => new Operation({...PROPS, output: {frechet: 'same', hausdorff: 'same'}})).toThrow(
        /^similarity-check.*outputs/
      );
    });
  }
  it('CPU operation validates numeric props and supports the documented uncapped configuration', () => {
    for (const [name, value] of [
      ['densify', -1],
      ['lcssTolerance', -1],
      ['editDistanceTolerance', Infinity],
      ['lcssBandRadius', -1],
      ['dynamicTimeWarpingBandRadius', 0.5],
      ['maximumFrechetVertexCount', 0]
    ] as const) {
      expect(() => new CPUTrackSimilarityOperation({...PROPS, [name]: value})).toThrow(
        new RegExp(`^similarity-check.*${name}`)
      );
    }
    expect(
      () => new CPUTrackSimilarityOperation({...PROPS, output: {pairStatus: 'status'}})
    ).toThrow(/^similarity-check.*output/);
    expect(
      () => new CPUTrackSimilarityOperation({...PROPS, output: {frechetWithin: 'within'}})
    ).toThrow(/^similarity-check.*frechetWithin/);
    expect(
      () =>
        new CPUTrackSimilarityOperation({
          ...PROPS,
          maximumFrechetVertexCount: 4096,
          densify: 1 / 8192
        })
    ).not.toThrow();
  });
});

it('includes a non-axial equality in all threshold decisions', () => {
  const input = {...makeInput([[0, 0]], [[3, 4]]), frechetEpsilon: Float32Array.of(5)};
  const result = computeTrackSimilarityOnCPU(input, {editDistanceTolerance: 5, lcssTolerance: 5});
  expect(Array.from(result.frechetWithin)).toEqual([1]);
});
