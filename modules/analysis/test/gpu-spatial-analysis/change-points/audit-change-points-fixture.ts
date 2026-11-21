// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Minimum length 8 leaves one segment. Exact Fraction SSE for [0, a, ..., a]
// is (7*a/8)^2 + 7*(a/8)^2 = 7*a^2/8.
export const WIDE_MEAN_COST_VALUES = Float32Array.from(
  [4, 2 ** 62, -(2 ** 62)].flatMap(value => [0, ...new Array<number>(7).fill(value)])
);
export const WIDE_MEAN_COST_EXPECTED = [14, 1.8609191940988822e37, 1.8609191940988822e37];

// Minimum length 5 leaves one segment. /usr/bin/python3 Fraction sums the squared
// deviations from the exact mean. The repeated-value cases have SSE 4*a^2/5.
const WIDE_LEVEL = 2 ** 63;
export const WIDE_MEAN_SQUARES_VALUES = Float32Array.from([
  ...[4, WIDE_LEVEL, -WIDE_LEVEL].flatMap(value => [0, ...new Array<number>(4).fill(value)]),
  // The first nonzero offset contributes to the running moments before scaling starts.
  0,
  2 ** 53,
  WIDE_LEVEL,
  WIDE_LEVEL,
  WIDE_LEVEL
]);
export const WIDE_MEAN_SQUARES_EXPECTED = [
  12.8, 6.80564733841877e37, 6.80564733841877e37, 1.019850828803084e38
];

/** Exhaustive-partition NumPy literals from audit-change-points.oracle.py. */
export const AUDIT_CHANGE_SERIES = [
  [1, 2, 1, 3, 9, 14, 10, 12],
  [1, NaN, 1, 3, 9, 14, NaN, 12],
  [3, 3, 3, 3, 3, 3, 3, 3],
  new Array<number>(8).fill(NaN),
  [7, NaN, NaN, NaN, NaN, NaN, NaN, NaN]
];
const NONE = 0xffffffff;
// Fraction gives two-slice population variances (3*2^63/2)^2 and 2^254.
// Decimal (70 digits) evaluates 2*(log(2*pi*variance)+1) for both likelihoods.
const WIDE_VARIANCE_SERIES = Float32Array.of(0, 3 * 2 ** 63, -(2 ** 127), 2 ** 127);
const WIDE_VARIANCE_EXPECTED = [181.97070406635757, 357.7945218572709];
export const WIDE_STANDARDIZATION_CASES = [
  {
    cost: 'variance',
    values: WIDE_VARIANCE_SERIES,
    minimumLength: 2,
    sliceCount: 2,
    expectedCost: WIDE_VARIANCE_EXPECTED,
    points: [NONE, NONE],
    counts: [0, 0]
  },
  {
    cost: 'mean-variance',
    values: WIDE_VARIANCE_SERIES,
    minimumLength: 2,
    sliceCount: 2,
    expectedCost: WIDE_VARIANCE_EXPECTED,
    points: [NONE, NONE],
    counts: [0, 0]
  },
  {
    cost: 'slope',
    // Three consecutive finite points have SSE (y0-2*y1+y2)^2/6.
    // Fraction gives 2^126/6 and 2^131/3, respectively. The latter overflows float32.
    // In the last cell a split at 3 fits both flat segments exactly, leaving penalty 1.
    values: Float32Array.from(
      [
        [0, WIDE_LEVEL, 3 * WIDE_LEVEL, NaN, NaN, NaN],
        [0, 2 ** 65, 0, NaN, NaN, NaN],
        [0, 0, 0, 2 ** 100, 2 ** 100, 2 ** 100]
      ].flat()
    ),
    minimumLength: 3,
    sliceCount: 6,
    expectedCost: [1.4178431955039102e37, Infinity, 1],
    points: [NONE, NONE, 3],
    counts: [0, 0, 1]
  }
] as const;

/*
 * Review 2 oracle, run with /usr/bin/python3. The additive costs use Fraction;
 * only the likelihood logs use Decimal. Sort by exact cost then earliest start.
 *
 * from fractions import Fraction as F
 * from decimal import Decimal as D, getcontext
 * from itertools import combinations
 * getcontext().prec = 65
 * pi = D('3.1415926535897932384626433832795028841971693993751058209749445923078')
 * def avg(v): return F(sum(v), len(v))
 * def decimal(x): return D(x.numerator) / D(x.denominator)
 * def fits(kind, values, penalty, minimum):
 *     center = avg(values)
 *     variance = avg([(v-center)**2 for v in values])
 *     def segment(v):
 *         n, center_local = len(v), avg(v)
 *         squares = sum((x-center_local)**2 for x in v)
 *         if kind == 'mean': return squares
 *         if kind == 'slope':
 *             t = avg(range(n))
 *             cov = sum((i-t)*(x-center_local) for i,x in enumerate(v))
 *             return squares-cov**2/sum((i-t)**2 for i in range(n))
 *         if kind == 'poisson':
 *             return 2*sum(v)*(D(n).ln()-D(sum(v)).ln())
 *         moment = avg([(x-center)**2 for x in v]) if kind == 'variance' else squares/n
 *         return n*((2*pi*decimal(max(moment, variance/10**11))).ln()+1)
 *     result = []
 *     for count in range(len(values)//minimum):
 *         for splits in combinations(range(minimum,len(values)-minimum+1), count):
 *             edges = (0,*splits,len(values))
 *             spans = list(zip(edges,edges[1:]))
 *             if all(last-first>=minimum for first,last in spans):
 *                 result.append((sum(segment(values[first:last]) for first,last in spans)
 *                                + count*penalty, splits))
 *     return sorted(result, key=lambda row: (row[0], row[1][::-1]))
 *
 * For the mixed slope, exact optimum = 2787593149816327892691964784081045188247555/3.
 * For the capacity-limited mean, exact optimum = 340282366920938463463374607431768211456.
 * Both exceed float32 range. The second mean optimum is exactly float32 max.
 */
const TIE_AMPLITUDE = 2 ** 37;
const TIE_TRANSLATION = 2 ** 60;
export const TRANSLATED_CHANGE_TIES = [
  {
    cost: 'mean',
    source: [0, 0, 0, 1, 2, 2, 2],
    amplitude: TIE_AMPLITUDE,
    translation: TIE_TRANSLATION,
    penalty: TIE_AMPLITUDE ** 2,
    point: 3,
    expected: [3.3056565380087516e22, 3.3056565380087516e22]
  },
  {
    cost: 'variance',
    source: [0, 0, 0, 1, 0, 0, 0],
    amplitude: TIE_AMPLITUDE,
    translation: TIE_TRANSLATION,
    penalty: 0,
    point: 3,
    expected: [360.7817060481817, 360.7817060481817]
  },
  {
    cost: 'mean-variance',
    source: [0, 0, 0, 1, 0, 0, 0],
    amplitude: TIE_AMPLITUDE,
    translation: TIE_TRANSLATION,
    penalty: 0,
    point: 3,
    expected: [289.9339827051792, 289.9339827051792]
  },
  {
    cost: 'poisson',
    source: [1, 1, 1, 4, 1, 1, 1],
    amplitude: 1,
    translation: 16,
    penalty: 0,
    point: 3,
    expected: [-7.834621031095917, -697.4345043545763]
  },
  {
    cost: 'slope',
    source: [-1, -1, 0, 1, 0, -1, -1, -1, 0],
    amplitude: TIE_AMPLITUDE,
    translation: TIE_TRANSLATION,
    penalty: 2.5 * TIE_AMPLITUDE ** 2,
    point: 4,
    expected: [7.555786372591432e22, 7.555786372591432e22]
  }
] as const;

export const WIDE_REACHABILITY_CASES = [
  {
    name: 'mixed-magnitude slope',
    cost: 'slope',
    values: Float32Array.of(0, 2 ** 70, 0, 2 ** 100, 2 ** 100, 2 ** 100),
    parameters: Float32Array.of(1, 3),
    point: 3,
    required: 1,
    expected: Infinity,
    overflow: 0
  },
  {
    name: 'out-of-range mean with truncated partition',
    cost: 'mean',
    values: Float32Array.of(0, 0, 2 ** 65, 2 ** 65, 0, 0),
    parameters: Float32Array.of(2 ** 127, 2),
    point: 2,
    required: 2,
    expected: Infinity,
    overflow: 1
  },
  {
    name: 'mean at float32 maximum',
    cost: 'mean',
    values: Float32Array.of(0, 2 ** 65),
    parameters: Float32Array.of(3.4028234663852886e38, 1),
    point: 1,
    required: 1,
    expected: 3.4028234663852886e38,
    overflow: 0
  }
] as const;

export const REVIEW_TWO_CHANGE_CASES = [
  ...TRANSLATED_CHANGE_TIES.map(fixture => ({
    name: `B1 translated ${fixture.cost} tie`,
    cost: fixture.cost,
    values: Float32Array.from(
      [0, fixture.translation].flatMap(translation =>
        fixture.source.map(value => value * fixture.amplitude + translation)
      )
    ),
    parameters: Float32Array.of(fixture.penalty, 3),
    sliceCount: fixture.source.length,
    points: [fixture.point, fixture.point],
    required: [1, 1],
    expected: fixture.expected,
    relativeTolerance: 2e-5,
    overflow: 0
  })),
  ...WIDE_REACHABILITY_CASES.map(fixture => ({
    ...fixture,
    name: `${fixture.cost === 'slope' ? 'S1' : 'S2'} ${fixture.name}`,
    sliceCount: fixture.values.length,
    points: [fixture.point],
    required: [fixture.required],
    expected: [fixture.expected],
    relativeTolerance: 0
  }))
];

export const AUDIT_CHANGE_CASES = [
  {
    cost: 'mean',
    minimumLength: 1,
    expected: {
      changePointCount: [1, 1, 0, 0, 0],
      changePoints: [4, 4, NONE, NONE, NONE],
      requiredChangePointCounts: [3, 2, 0, 0, 0],
      optimalCost: [13.75, 10.666666666666668, 0, NaN, 0],
      overflow: [1]
    }
  },
  {
    cost: 'variance',
    minimumLength: 2,
    expected: {
      changePointCount: [0, 0, 0, 0, 0],
      changePoints: [NONE, NONE, NONE, NONE, NONE],
      requiredChangePointCounts: [0, 0, 0, 0, 0],
      optimalCost: [48.373620443392355, 36.92448741142864, -179.92447165220128, NaN, NaN],
      overflow: [0]
    }
  },
  {
    cost: 'mean-variance',
    minimumLength: 2,
    expected: {
      changePointCount: [1, 1, 0, 0, 0],
      changePoints: [4, 3, NONE, NONE, NONE],
      requiredChangePointCounts: [1, 2, 0, 0, 0],
      optimalCost: [29.42403762017287, -16.602752155082968, -179.92447165220128, NaN, NaN],
      overflow: [1]
    }
  },
  {
    cost: 'poisson',
    minimumLength: 1,
    expected: {
      changePointCount: [1, 1, 0, 0, 0],
      changePoints: [4, 4, NONE, NONE, NONE],
      requiredChangePointCounts: [1, 1, 0, 0, 0],
      optimalCost: [
        -222.66775260963453,
        -174.07976033515118,
        -52.733389856069266,
        NaN,
        -27.242742086774385
      ],
      overflow: [0]
    }
  },
  {
    cost: 'slope',
    minimumLength: 3,
    expected: {
      changePointCount: [1, 1, 0, 0, 0],
      changePoints: [4, 4, NONE, NONE, NONE],
      requiredChangePointCounts: [1, 1, 0, 0, 0],
      optimalCost: [18, 14.42857142857143, 0, NaN, NaN],
      overflow: [0]
    }
  }
] as const;

export const AUDIT_CHANGE_AFFINE_COSTS = {
  mean: [55, 42.666666666666664, 0, NaN, 0],
  variance: [59.46397533235148, 45.242253578147995, -179.92447165220128, NaN, NaN],
  'mean-variance': [40.514392509131994, -8.284985988363625, -179.92447165220128, NaN, NaN],
  poisson: [-589.5101187757376, -459.0630695598937, -172.0089090458933, NaN, -73.89360522922723],
  slope: [72, 57.71428571428571, 0, NaN, NaN]
};

export const AUDIT_PETTITT_SERIES = [
  [1, 1, 1, 1, 8, 8, 8, 8],
  [NaN, 1, 1, NaN, 8, 8, 8, 8],
  [3, 3, 3, 3, 3, 3, 3, 3],
  new Array<number>(8).fill(NaN),
  [7, NaN, NaN, NaN, NaN, NaN, NaN, NaN]
];
export const AUDIT_PETTITT_EXPECTED = {
  statistic: [16, 8, 0, 0, 0],
  changeSlice: [4, 4, 1, NONE, NONE],
  pValues: [0.13896690244560309, 0.4357605676462449, 1, NaN, NaN]
};

export function packAuditSeries(
  series: number[][],
  layout: 'cell-major' | 'period-major',
  transform: (value: number) => number = value => value
): Float32Array {
  return Float32Array.from(
    layout === 'cell-major'
      ? series.flat()
      : series[0].flatMap((_, slice) => series.map(row => row[slice])),
    transform
  );
}
