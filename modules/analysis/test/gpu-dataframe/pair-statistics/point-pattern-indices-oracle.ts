// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getIncludedRows,
  type PairStatisticsFrame,
  type PairStatisticsScene
} from './pair-statistics-oracle';

import {
  getOracleWindowBounds,
  getOracleWindowArea,
  getOracleBoundaryDistance,
  getOracleQuadratAreas,
  isOracleInsideWindow,
  type OracleStudyWindow
} from '../../analysis-core/study-window-oracle';

const fround = Math.fround;

/** Tail of the Clark-Evans and quadrat tests. */
export type PointPatternOracleAlternative = 'two-sided' | 'greater' | 'lesser';

/** Defaults to `'two-sided'`. */
function resolveAlternative(
  alternative: PointPatternOracleAlternative | undefined
): PointPatternOracleAlternative {
  return alternative ?? 'two-sided';
}

/** Per-frame options of the Clark-Evans and quadrat p-values. */
export type PointPatternOracleOptions = {
  /** Defaults to `'donnelly'`, like `getGPUPointPatternIndicesParameterValues`. */
  clarkEvansCorrection?: 'none' | 'donnelly';
  /** Defaults to `'two-sided'`. */
  analyticAlternative?: PointPatternOracleAlternative;
};

/**
 * f64 lower and upper tails `[P(a, x), Q(a, x)]` of the regularized incomplete gamma function
 * (series for `x < a + 1`, Lentz continued fraction above; Numerical Recipes `gser`/`gcf`).
 * Verified against scipy.special.gammainc / gammaincc in the node spec.
 */
export function getGammaTailsOnCPU(a: number, x: number): [number, number] {
  if (x <= 0) {
    return [0, 1];
  }
  const logGamma = getLogGammaOnCPU(a);
  const prefix = Math.exp(a * Math.log(x) - x - logGamma);
  if (x < a + 1) {
    let term = 1 / a;
    let sum = term;
    for (let denominator = a + 1; Math.abs(term) > Math.abs(sum) * 1e-16; denominator++) {
      term *= x / denominator;
      sum += term;
    }
    return [sum * prefix, 1 - sum * prefix];
  }
  const tiny = 1e-300;
  let b = x + 1 - a;
  let c = 1 / tiny;
  let d = 1 / b;
  let h = d;
  for (let iteration = 1; iteration < 100000; iteration++) {
    const an = -iteration * (iteration - a);
    b += 2;
    d = an * d + b;
    d = Math.abs(d) < tiny ? tiny : d;
    c = b + an / c;
    c = Math.abs(c) < tiny ? tiny : c;
    d = 1 / d;
    const delta = d * c;
    h *= delta;
    if (Math.abs(delta - 1) < 1e-15) {
      break;
    }
  }
  return [1 - prefix * h, prefix * h];
}

/** Lanczos (g = 7, n = 9) log-gamma in f64. */
function getLogGammaOnCPU(value: number): number {
  const coefficients = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6,
    1.5056327351493116e-7
  ];
  const x = value - 1;
  let sum = coefficients[0];
  for (let index = 1; index < 9; index++) {
    sum += coefficients[index] / (x + index);
  }
  const t = x + 7.5;
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(sum);
}

/** Normal p-value of a standardized statistic. */
export function getNormalPValueOnCPU(
  z: number,
  alternative: PointPatternOracleAlternative
): number {
  // erfc(|z| / sqrt(2)) = Q(1/2, z^2 / 2) is the two-sided p-value; each tail is half of it.
  const twoSided = getGammaTailsOnCPU(0.5, (z * z) / 2)[1];
  if (alternative === 'two-sided') {
    return twoSided;
  }
  const smallTail = twoSided / 2;
  const isUpper = alternative === 'greater';
  return z >= 0 === isUpper ? smallTail : 1 - smallTail;
}

/** Chi-square p-value for the tail, exact in f64 for every degree of freedom. */
export function getChiSquarePValueOnCPU(
  x: number,
  degreesOfFreedom: number,
  alternative: PointPatternOracleAlternative
): number {
  const [lower, upper] = getGammaTailsOnCPU(degreesOfFreedom / 2, x / 2);
  if (alternative === 'greater') {
    return upper;
  }
  if (alternative === 'lesser') {
    return lower;
  }
  return Math.min(1, 2 * Math.min(lower, upper));
}

/** CPU result of {@link computePointPatternIndicesOnCPU}. */
export type PointPatternIndicesOracleResult = {
  /** Per row: f32 nearest-neighbor distance, NaN for excluded rows and when `n < 2`. */
  nearestNeighborDistances: number[];
  /** Per row: smallest-index nearest neighbor, `0xffffffff` when none. */
  nearestNeighborIds: number[];
  /** `[n, observed, expected, ratio, standardError, zScore, pValue]`. */
  clarkEvans: number[];
  /** Row-major quadrat counts (empty without a quadrat grid). */
  quadratCounts: number[];
  /** `[m, mean, variance, varianceToMeanRatio, chiSquare, degreesOfFreedom, pValue]`. */
  quadratStatistics: number[];
};

/**
 * Quadrat of an offset on `count` equal steps over `[0, extent]`, from the definition rather than
 * from a quotient: the largest `k` with `extent * k / count <= offset`, so a point on an interior
 * edge belongs to the upper quadrat and one on the maximum edge to the last. The products of a
 * float32 offset or extent with an integer below 2^29 are exact in binary64.
 */
function getOracleQuadratIndex(offset: number, extent: number, count: number): number {
  let index = 0;
  while (index + 1 < count && offset * count >= extent * (index + 1)) index++;
  return index;
}

/**
 * f64 Clark-Evans index and quadrat statistics with the `GPUPointPatternIndices` inclusion rule,
 * f32 squared distances (ties to the smallest row index) and quadrats from the edge definition.
 */
export function computePointPatternIndicesOnCPU(
  scene: PairStatisticsScene,
  frame: Pick<PairStatisticsFrame, 'bounds'> & PointPatternOracleOptions,
  quadratGrid: readonly [number, number] = [1, 1]
): PointPatternIndicesOracleResult {
  const [minX, minY, maxX, maxY] = frame.bounds.map(fround);
  const rows = scene.positions.length / 2;
  const included = getIncludedRows(scene, frame);
  const n = included.length;
  const width = fround(maxX - minX);
  const height = fround(maxY - minY);
  const area = fround(width * height);
  const nearestNeighborDistances = new Array<number>(rows).fill(NaN);
  const nearestNeighborIds = new Array<number>(rows).fill(0xffffffff);
  let distanceSum = 0;
  for (const focus of included) {
    const x = scene.positions[focus * 2];
    const y = scene.positions[focus * 2 + 1];
    let bestSquared = Infinity;
    let bestId = -1;
    for (const neighbor of included) {
      if (neighbor === focus) {
        continue;
      }
      const deltaX = fround(scene.positions[neighbor * 2] - x);
      const deltaY = fround(scene.positions[neighbor * 2 + 1] - y);
      const squared = fround(fround(deltaX * deltaX) + fround(deltaY * deltaY));
      if (squared < bestSquared) {
        bestSquared = squared;
        bestId = neighbor;
      }
    }
    if (bestId >= 0) {
      nearestNeighborDistances[focus] = fround(Math.sqrt(bestSquared));
      nearestNeighborIds[focus] = bestId;
      distanceSum += nearestNeighborDistances[focus];
    }
  }
  const alternative = resolveAlternative(frame.analyticAlternative);
  let clarkEvans = [n, NaN, NaN, NaN, NaN, NaN, NaN];
  if (n >= 2 && area > 0) {
    const observed = distanceSum / n;
    let expected = 0.5 / Math.sqrt(n / area);
    let standardError = 0.26136 / Math.sqrt((n * n) / area);
    if ((frame.clarkEvansCorrection ?? 'donnelly') === 'donnelly') {
      // Donnelly (1978), rectangular window of perimeter P.
      const perimeter = 2 * (width + height);
      expected += ((0.0514 + 0.041 / Math.sqrt(n)) * perimeter) / n;
      standardError = Math.sqrt(
        (0.0703 * area) / (n * n) + 0.037 * perimeter * Math.sqrt(area / n ** 5)
      );
    }
    const zScore = (observed - expected) / standardError;
    clarkEvans = [
      n,
      observed,
      expected,
      observed / expected,
      standardError,
      zScore,
      getNormalPValueOnCPU(zScore, alternative)
    ];
  }

  const [columns, quadratRows] = quadratGrid;
  const quadratCounts = new Array<number>(columns * quadratRows).fill(0);
  for (const row of included) {
    const x = scene.positions[row * 2];
    const y = scene.positions[row * 2 + 1];
    const column = width > 0 ? getOracleQuadratIndex(fround(x - minX), width, columns) : 0;
    const quadratRow =
      height > 0 ? getOracleQuadratIndex(fround(y - minY), height, quadratRows) : 0;
    quadratCounts[quadratRow * columns + column]++;
  }
  const m = quadratCounts.length;
  const mean = n / m;
  const squares = quadratCounts.reduce((sum, count) => sum + (count - mean) ** 2, 0);
  const variance = m > 1 ? squares / (m - 1) : NaN;
  const ratio = m > 1 && mean > 0 ? variance / mean : NaN;
  return {
    nearestNeighborDistances,
    nearestNeighborIds,
    clarkEvans,
    quadratCounts,
    quadratStatistics: [
      m,
      mean,
      variance,
      ratio,
      (m - 1) * ratio,
      m - 1,
      m > 1 && Number.isFinite(ratio)
        ? getChiSquarePValueOnCPU((m - 1) * ratio, m - 1, alternative)
        : NaN
    ]
  };
}

/** Guard and polygon extension of the independent point-pattern oracle. */
export function computeWindowPointPatternIndicesOnCPU(
  scene: PairStatisticsScene,
  frame: Pick<PairStatisticsFrame, 'bounds'> &
    Omit<PointPatternOracleOptions, 'clarkEvansCorrection'> & {
      clarkEvansCorrection?: 'none' | 'donnelly' | 'guard';
      guardDistance?: number;
    },
  quadratGrid: readonly [number, number],
  window?: OracleStudyWindow,
  quadratSamples = 16
): PointPatternIndicesOracleResult & {quadratAreas: number[]} {
  const bounds = window ? getOracleWindowBounds(window) : frame.bounds;
  const mask = Uint32Array.from({length: scene.positions.length / 2}, (_, row) =>
    Number(
      (!scene.mask || scene.mask[row] !== 0) &&
        (!window ||
          isOracleInsideWindow(window, [scene.positions[row * 2], scene.positions[row * 2 + 1]]))
    )
  );
  const effectiveScene = {...scene, mask};
  const effectiveFrame = {
    ...frame,
    bounds,
    clarkEvansCorrection:
      frame.clarkEvansCorrection === 'guard' ? ('none' as const) : frame.clarkEvansCorrection
  };
  const result = computePointPatternIndicesOnCPU(effectiveScene, effectiveFrame, quadratGrid);
  const included = getIncludedRows(effectiveScene, effectiveFrame);
  const count = included.length;
  const area = window
    ? getOracleWindowArea(window)
    : (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]);
  if (frame.clarkEvansCorrection === 'guard' || window) {
    const guardDistance = frame.guardDistance ?? 0;
    const focus = included.filter(row => {
      if (frame.clarkEvansCorrection !== 'guard') return true;
      const point = [scene.positions[row * 2], scene.positions[row * 2 + 1]] as const;
      const distance = window
        ? getOracleBoundaryDistance(window, point, guardDistance)
        : Math.min(
            point[0] - bounds[0],
            bounds[2] - point[0],
            point[1] - bounds[1],
            bounds[3] - point[1]
          );
      return Number.isFinite(guardDistance) && guardDistance > 0 && distance >= guardDistance;
    });
    result.clarkEvans = [count, NaN, NaN, NaN, NaN, NaN, NaN];
    if (count >= 2 && focus.length && area > 0) {
      const observed =
        focus.reduce((sum, row) => sum + result.nearestNeighborDistances[row], 0) / focus.length;
      const expected =
        window && (frame.clarkEvansCorrection ?? 'donnelly') === 'donnelly'
          ? NaN
          : 0.5 * Math.sqrt(area / count);
      const standardError = Number.isFinite(expected)
        ? 0.26136 * Math.sqrt(area / (count * focus.length))
        : NaN;
      const zScore = (observed - expected) / standardError;
      result.clarkEvans = [
        count,
        observed,
        expected,
        observed / expected,
        standardError,
        zScore,
        getNormalPValueOnCPU(zScore, resolveAlternative(frame.analyticAlternative))
      ];
    }
  }
  let quadratAreas = new Array<number>(quadratGrid[0] * quadratGrid[1]).fill(
    area / (quadratGrid[0] * quadratGrid[1])
  );
  if (window) {
    const sampledAreas = getOracleQuadratAreas(window, bounds, quadratGrid, quadratSamples);
    const sampledTotal = sampledAreas.reduce((sum, value) => sum + value, 0);
    quadratAreas = sampledAreas.map(value => (value * area) / sampledTotal);
    const effectiveCount = sampledAreas.filter(value => value > 0).length;
    const equalAreas = effectiveCount > 0 && sampledAreas.every(value => value === sampledAreas[0]);
    // spatstat quadrat.test counts every point in the window. Points in a quadrat with no area
    // sample cannot be given an expected count, so the statistics are undefined (PTS-108).
    const unsampledPoints = sampledAreas.reduce(
      (sum, value, quadrat) => (value > 0 ? sum : sum + result.quadratCounts[quadrat]),
      0
    );
    if (unsampledPoints > 0) {
      const degreesOfFreedom = effectiveCount - 1;
      result.quadratStatistics = [effectiveCount, NaN, NaN, NaN, NaN, degreesOfFreedom, NaN];
    } else if (!equalAreas) {
      // Preserve the legacy equal-area formula exactly, including operation order.
      const mean = effectiveCount > 0 ? count / effectiveCount : NaN;
      let squares = 0;
      let chiSquare = 0;
      for (let quadrat = 0; quadrat < sampledAreas.length; quadrat++) {
        if (sampledAreas[quadrat] > 0) {
          const expected = (count * sampledAreas[quadrat]) / sampledTotal;
          const residual = result.quadratCounts[quadrat] - expected;
          squares += residual ** 2;
          chiSquare += residual ** 2 / expected;
        }
      }
      const degreesOfFreedom = effectiveCount - 1;
      if (!(degreesOfFreedom > 0 && count > 0)) chiSquare = NaN;
      result.quadratStatistics = [
        effectiveCount,
        mean,
        degreesOfFreedom > 0 ? squares / degreesOfFreedom : NaN,
        chiSquare / degreesOfFreedom,
        chiSquare,
        degreesOfFreedom,
        Number.isFinite(chiSquare)
          ? getChiSquarePValueOnCPU(
              chiSquare,
              degreesOfFreedom,
              resolveAlternative(frame.analyticAlternative)
            )
          : NaN
      ];
    }
  }
  return {...result, quadratAreas};
}

function createSingleRingWindow(ring: readonly number[]): OracleStudyWindow {
  return {
    positions: new Float64Array(ring),
    polygonOffsets: new Uint32Array([0, 1]),
    ringOffsets: new Uint32Array([0, ring.length / 2]),
    featureOffsets: new Uint32Array([0, 1])
  };
}

/**
 * PTS-108 fixtures for 2 x 2 quadrats with 16 samples per axis, with expectations computed
 * independently of the oracle and the kernel: exact clipped areas of each quadrat in the window,
 * every window point counted, expected counts `n * area / A` over the quadrats of positive area
 * (spatstat `quadrat.test`), and the SciPy chi-square tails. Script:
 * `/usr/bin/python3 polygon_quadrat_oracle.py` (Sutherland-Hodgman clipping, shoelace areas,
 * `scipy.stats.chi2`), kept with the 2026-11-19 review fixes.
 *
 * Aligned: an L shape whose re-entrant corner (750, 750) lies on the sample lattice, so the
 * sampled areas are exact: [937500, 750000, 750000, 0], counts [5, 1, 3, 0], n = 9, m = 3.
 */
export const QUADRAT_ALIGNED_L_WINDOW = createSingleRingWindow([
  0, 0, 2000, 0, 2000, 750, 750, 750, 750, 2000, 0, 2000
]);
export const QUADRAT_ALIGNED_L_POSITIONS = new Float32Array([
  100, 100, 300, 200, 500, 600, 800, 300, 900, 900, 200, 800, 1500, 300, 200, 1200, 400, 1500, 600,
  1900
]);
/** `[m, mean, variance, ratio, chiSquare, df, two-sided p]` from the script. */
export const QUADRAT_ALIGNED_L_STATISTICS = [
  3, 3, 2.7751479289940826, 0.9166666666666665, 1.833333333333333, 2, 0.7996993086896947
];

/**
 * Spike: the square [0, 1000]^2 plus the strip [1000, 2000] x [400, 405], which no sample row of
 * its quadrat (y = 390.6, 421.9) reaches. Counts [45, 10, 45, 0]: the ten strip points fall in a
 * quadrat with sampled area 0. With exact areas [500000, 5000, 500000, 0] the statistic is
 * chi-square 182.405 with 2 degrees of freedom (p 4.9e-40); the old kernel dropped the strip's
 * quadrat but kept its points in `n` and reported chi-square 1.0 with 1 degree of freedom
 * (p 0.317). The lattice cannot give that quadrat an expected count, so the statistics are NaN.
 */
export const QUADRAT_SPIKE_WINDOW = createSingleRingWindow([
  0, 0, 1000, 0, 1000, 400, 2000, 400, 2000, 405, 1000, 405, 1000, 1000, 0, 1000
]);
export const QUADRAT_SPIKE_POSITIONS = Float32Array.from(
  [
    ...Array.from({length: 45}, (_, index) => [
      10 + (index % 9) * 110,
      20 + Math.floor(index / 9) * 90
    ]),
    ...Array.from({length: 10}, (_, index) => [1050 + index * 100, 402]),
    ...Array.from({length: 45}, (_, index) => [
      10 + (index % 9) * 110,
      520 + Math.floor(index / 9) * 90
    ])
  ].flat()
);
