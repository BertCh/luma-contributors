// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Binary64 CPU reference of the space-time K family (stpp 2.0-8 `STIKhat`, `PCFhat`, `LISTAhat`;
 * splancs / Diggle 1995 `D`, with `tieRule: 'splancs'` for the splancs rules at exact ties, which
 * `space-time-diggle-splancs.node.spec.ts` checks against a port of the splancs source).
 * stpp and stopp are R-only (GPL) and R is not installed here, so the
 * R fixtures are pending: this brute-force double loop, written from the definitions in the
 * `GPUSpaceTimeKFunction` documentation, is the oracle. The circle fraction of the spatial
 * isotropic weight is computed here from the crossing angles of the four side lines, not with the
 * arc formula of the source under test.
 */
export type SpaceTimeKScene = {
  /** Interleaved x, y, already rounded to float32. */
  positions: number[];
  times: number[];
  intensities?: number[];
  mask?: number[];
};

export type SpaceTimeKReferenceOptions = {
  scene: SpaceTimeKScene;
  bounds: readonly [number, number, number, number];
  timeWindow: readonly [number, number];
  maximumDistance: number;
  maximumTimeLag: number;
  distanceClassCount: number;
  timeClassCount: number;
  edgeCorrection: 'none' | 'isotropic';
  /**
   * Rule at exact ties. `'stpp'` (default; stpp 2.0-8 `stikfunction.f`, `GPUSpaceTimeKFunction`):
   * a lag that exactly reaches a time limit has temporal weight 1 and a pair at exactly the
   * largest distance or lag is counted. `'splancs'` (splancs 2.01-45 `stkhat.f`,
   * `GPUSpaceTimeDiggleTest`): that lag has weight 2, such a pair is dropped, and the class is
   * the first break `maximum (k + 1) / count` at or above the value, decided by products.
   */
  tieRule?: 'stpp' | 'splancs';
  pairCorrelationKernel?: 'box' | 'epanechnikov';
  spatialBandwidth?: number;
  temporalBandwidth?: number;
};

/** splancs class of a value: the first break at or above it; `count` when it is dropped. */
function getSplancsClass(value: number, maximum: number, count: number): number {
  if (!(value < maximum)) {
    return count;
  }
  let classIndex = 0;
  while (value * count > maximum * (classIndex + 1)) {
    classIndex++;
  }
  return classIndex;
}

export type SpaceTimeKReference = {
  included: number[];
  k: number[];
  spatialK: number[];
  temporalK: number[];
  d: number[];
  pairCorrelation: number[];
  /** Ordered pairs per class, not cumulative. */
  pairCounts: number[];
  /** `row * bins + bin`, NaN for excluded rows. */
  local: number[];
  /** Smallest distance of a scaled distance or lag to an integer class boundary, to detect ties. */
  minimumBoundaryMargin: number;
};

/** Fraction of the circle of radius `radius` about `(x, y)` inside the rectangle, exact. */
export function getCircleFractionInside(
  x: number,
  y: number,
  radius: number,
  bounds: readonly [number, number, number, number]
): number {
  if (radius === 0) {
    return 1;
  }
  const angles = [0, 2 * Math.PI];
  // Crossings of the circle with each side line.
  for (const [coordinate, center, isX] of [
    [bounds[0], x, true],
    [bounds[2], x, true],
    [bounds[1], y, false],
    [bounds[3], y, false]
  ] as const) {
    const offset = (coordinate - center) / radius;
    if (Math.abs(offset) < 1) {
      const base = Math.acos(offset);
      // isX: cos(theta) = offset; otherwise sin(theta) = offset.
      const solutions = isX ? [base, -base] : [Math.asin(offset), Math.PI - Math.asin(offset)];
      for (const angle of solutions) {
        angles.push(((angle % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI));
      }
    }
  }
  angles.sort((first, second) => first - second);
  let inside = 0;
  for (let index = 0; index + 1 < angles.length; index++) {
    const middle = 0.5 * (angles[index] + angles[index + 1]);
    const px = x + radius * Math.cos(middle);
    const py = y + radius * Math.sin(middle);
    if (px >= bounds[0] && px <= bounds[2] && py >= bounds[1] && py <= bounds[3]) {
      inside += angles[index + 1] - angles[index];
    }
  }
  return inside / (2 * Math.PI);
}

function getKernelValue(kernel: 'box' | 'epanechnikov', offset: number, bandwidth: number): number {
  const scaled = offset / bandwidth;
  if (Math.abs(scaled) > 1) {
    return 0;
  }
  return kernel === 'box' ? 1 / (2 * bandwidth) : (0.75 * (1 - scaled * scaled)) / bandwidth;
}

export function getSpaceTimeKReference(options: SpaceTimeKReferenceOptions): SpaceTimeKReference {
  const {scene, bounds, timeWindow, maximumDistance, maximumTimeLag} = options;
  const U = options.distanceClassCount;
  const V = options.timeClassCount;
  const bins = U * V;
  const rowCount = scene.times.length;
  const isotropic = options.edgeCorrection === 'isotropic';
  const included: number[] = [];
  for (let row = 0; row < rowCount; row++) {
    const x = scene.positions[2 * row];
    const y = scene.positions[2 * row + 1];
    const time = scene.times[row];
    const intensity = scene.intensities ? scene.intensities[row] : 1;
    if (
      (scene.mask ? scene.mask[row] !== 0 : true) &&
      Number.isFinite(x) &&
      Number.isFinite(y) &&
      x >= bounds[0] &&
      x <= bounds[2] &&
      y >= bounds[1] &&
      y <= bounds[3] &&
      Number.isFinite(time) &&
      time >= timeWindow[0] &&
      time <= timeWindow[1] &&
      Number.isFinite(intensity) &&
      intensity > 0
    ) {
      included.push(row);
    }
  }
  const n = included.length;
  const area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]);
  const duration = timeWindow[1] - timeWindow[0];
  const inhomogeneous = Boolean(scene.intensities);
  const lambdaBar = n / (area * duration);
  const inverseIntensity = (row: number) =>
    inhomogeneous ? lambdaBar / scene.intensities![row] : 1;
  const pairNormalizer = inhomogeneous ? n * n : n * (n - 1);
  const globalScale = (area * duration) / pairNormalizer;

  const weightedClass = new Array<number>(bins).fill(0);
  const spatialClass = new Array<number>(U).fill(0);
  const temporalCumulative = new Array<number>(V).fill(0);
  const pairCounts = new Array<number>(bins).fill(0);
  const pairCorrelationSums = new Array<number>(bins).fill(0);
  const localClass = new Array<number>(rowCount * bins).fill(0);
  let minimumBoundaryMargin = Infinity;
  const kernel = options.pairCorrelationKernel ?? 'box';
  const spatialBandwidth = options.spatialBandwidth ?? 0;
  const temporalBandwidth = options.temporalBandwidth ?? 0;
  const splancs = options.tieRule === 'splancs';
  for (const first of included) {
    const x = scene.positions[2 * first];
    const y = scene.positions[2 * first + 1];
    const time = scene.times[first];
    const radius = Math.min(time - timeWindow[0], timeWindow[1] - time);
    for (const second of included) {
      if (second === first) {
        continue;
      }
      const distance = Math.hypot(
        scene.positions[2 * second] - x,
        scene.positions[2 * second + 1] - y
      );
      const lag = Math.abs(scene.times[second] - time);
      const spatialWeight = isotropic
        ? Math.min(1 / Math.max(getCircleFractionInside(x, y, distance, bounds), 0.01), 100)
        : 1;
      const isInsideWindow = splancs ? lag < radius : lag <= radius;
      const temporalWeight = isotropic ? (isInsideWindow ? 1 : 2) : 1;
      const intensityFactor = inverseIntensity(first) * inverseIntensity(second);
      const weight = spatialWeight * temporalWeight * intensityFactor;
      const scaledDistance = (distance / maximumDistance) * U;
      const scaledLag = (lag / maximumTimeLag) * V;
      for (const [value, limit] of [
        [scaledDistance, U],
        [scaledLag, V]
      ] as const) {
        if (value > 1e-9 && value < limit + 1) {
          minimumBoundaryMargin = Math.min(
            minimumBoundaryMargin,
            Math.abs(value - Math.round(value))
          );
        }
      }
      let inDistance = scaledDistance <= U;
      let inLag = scaledLag <= V;
      let distanceClass = Math.min(Math.max(Math.ceil(scaledDistance) - 1, 0), U - 1);
      let lagClass = Math.min(Math.max(Math.ceil(scaledLag) - 1, 0), V - 1);
      if (splancs) {
        distanceClass = getSplancsClass(distance, maximumDistance, U);
        lagClass = getSplancsClass(lag, maximumTimeLag, V);
        inDistance = distanceClass < U;
        inLag = lagClass < V;
      }
      if (inDistance) {
        spatialClass[distanceClass] += spatialWeight;
      }
      if (inDistance && inLag) {
        const bin = distanceClass * V + lagClass;
        weightedClass[bin] += weight;
        pairCounts[bin]++;
        localClass[first * bins + bin] += weight;
      }
      // Cumulative temporal marginal over every pair, regardless of distance.
      for (let index = 0; index < V; index++) {
        if (splancs ? lagClass <= index : scaledLag <= index + 1) {
          temporalCumulative[index] += temporalWeight;
        }
      }
      if (spatialBandwidth > 0 && temporalBandwidth > 0) {
        for (let a = 0; a < U; a++) {
          for (let b = 0; b < V; b++) {
            const u = (maximumDistance * (a + 1)) / U;
            const v = (maximumTimeLag * (b + 1)) / V;
            pairCorrelationSums[a * V + b] +=
              weight *
              getKernelValue(kernel, u - distance, spatialBandwidth) *
              getKernelValue(kernel, v - lag, temporalBandwidth);
          }
        }
      }
    }
  }
  const valid = n >= 2;
  const k = new Array<number>(bins).fill(Number.NaN);
  const d = new Array<number>(bins).fill(Number.NaN);
  const pairCorrelation = new Array<number>(bins).fill(Number.NaN);
  const local = new Array<number>(rowCount * bins).fill(Number.NaN);
  const spatialK = new Array<number>(U).fill(Number.NaN);
  const temporalK = new Array<number>(V).fill(Number.NaN);
  if (valid) {
    for (let a = 0; a < U; a++) {
      let spatialSum = 0;
      for (let a2 = 0; a2 <= a; a2++) {
        spatialSum += spatialClass[a2];
      }
      spatialK[a] = (area / pairNormalizer) * spatialSum;
    }
    for (let b = 0; b < V; b++) {
      temporalK[b] = (duration / pairNormalizer) * temporalCumulative[b];
    }
    for (let a = 0; a < U; a++) {
      for (let b = 0; b < V; b++) {
        let sum = 0;
        for (let a2 = 0; a2 <= a; a2++) {
          for (let b2 = 0; b2 <= b; b2++) {
            sum += weightedClass[a2 * V + b2];
          }
        }
        k[a * V + b] = globalScale * sum;
        d[a * V + b] = k[a * V + b] - spatialK[a] * temporalK[b];
        if (spatialBandwidth > 0 && temporalBandwidth > 0) {
          const u = (maximumDistance * (a + 1)) / U;
          pairCorrelation[a * V + b] =
            (globalScale * pairCorrelationSums[a * V + b]) / (4 * Math.PI * u);
        }
      }
    }
    for (const row of included) {
      for (let a = 0; a < U; a++) {
        for (let b = 0; b < V; b++) {
          let sum = 0;
          for (let a2 = 0; a2 <= a; a2++) {
            for (let b2 = 0; b2 <= b; b2++) {
              sum += localClass[row * bins + a2 * V + b2];
            }
          }
          local[row * bins + a * V + b] = n * globalScale * sum;
        }
      }
    }
  }
  return {
    included,
    k,
    spatialK,
    temporalK,
    d,
    pairCorrelation,
    pairCounts,
    local,
    minimumBoundaryMargin
  };
}

/**
 * A reproducible pattern with space-time interaction (clusters that persist), plus every kind of
 * excluded row, with positions and times off every class boundary.
 */
export function createSpaceTimeKScene(
  seed: number,
  options: {intensities?: boolean; mask?: boolean} = {}
): SpaceTimeKScene {
  let state = seed;
  const random = () => {
    state = (state * 1664525 + 1013904223) % 4294967296;
    return state / 4294967296;
  };
  const scene: SpaceTimeKScene = {
    positions: [],
    times: [],
    intensities: options.intensities ? [] : undefined,
    mask: options.mask ? [] : undefined
  };
  const add = (x: number, y: number, time: number, masked = 1) => {
    scene.positions.push(Math.fround(x), Math.fround(y));
    scene.times.push(Math.fround(time));
    scene.intensities?.push(Math.fround(0.5 + 1.5 * random()));
    scene.mask?.push(masked);
  };
  for (let cluster = 0; cluster < 3; cluster++) {
    const cx = 2 + cluster * 3.1;
    const cy = 1.5 + cluster * 2.3;
    const ct = 3 + cluster * 6;
    for (let index = 0; index < 14; index++) {
      add(cx + (random() - 0.5) * 2, cy + (random() - 0.5) * 2, ct + (random() - 0.5) * 4);
    }
  }
  for (let index = 0; index < 24; index++) {
    add(random() * 10, random() * 8, random() * 20);
  }
  // Near the window edges, where the edge weights are large.
  add(0.05, 0.1, 0.2);
  add(9.9, 7.9, 19.8);
  add(0.3, 7.7, 10.1);
  // Excluded: non-finite time, time outside the window, non-finite position, outside the window,
  // masked, bad intensity.
  add(5, 4, Number.NaN);
  add(5, 4.1, -3);
  add(5, 4.2, 25);
  add(Number.NaN, 4, 5);
  add(30, 4, 5);
  add(5, -2, 5);
  if (scene.mask) {
    add(5.3, 4.3, 7.7, 0);
  }
  if (scene.intensities) {
    add(5.4, 4.4, 7.8);
    scene.intensities[scene.intensities.length - 1] = 0;
    add(5.5, 4.5, 7.9);
    scene.intensities[scene.intensities.length - 1] = Number.NaN;
  }
  return scene;
}
