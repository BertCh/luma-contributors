// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {describe, expect, it} from 'vitest';
import {
  estimateBrownianMotionVariance,
  estimateWindowedBrownianMotionVariance,
  getGPUBrownianBridgeUtilizationParameterValues,
  getGPUUtilizationVolumeIsoplethsParameterValues,
  GPUBrownianBridgeUtilization,
  GPUUtilizationVolumeIsopleths
} from '../../../src/gpu-spatial-analysis/trajectory-home-range/index';
import {GPUTrajectoryRecursions} from '../../../src/gpu-spatial-analysis/trajectory-recursion/index';
import {readFloat32, readUint32} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import numpyChecks from '../../oracles/fixtures/trajectories-lines-numpy-checks.json';
import {GraphHarness, importTrajectory} from '../trajectory-staypoints/movement-spec-utils';

function createRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return (state + 0.5) / 4294967296;
  };
}

function createNormal(random: () => number): () => number {
  return () => Math.sqrt(-2 * Math.log(random())) * Math.cos(2 * Math.PI * random());
}

describe('estimateBrownianMotionVariance (MOB-07)', () => {
  const normal = createNormal(createRandom(11));
  const trueVariance = 4;
  const count = 3001;
  const x = new Float64Array(count);
  const y = new Float64Array(count);
  const times = new Float64Array(count);
  for (let index = 1; index < count; index++) {
    const step = 0.5 + (1.5 * ((index * 7919) % 13)) / 13;
    times[index] = times[index - 1] + step;
    x[index] = x[index - 1] + Math.sqrt(trueVariance * step) * normal();
    y[index] = y[index - 1] + Math.sqrt(trueVariance * step) * normal();
  }

  it('equals the closed form mean(r^2 / (2 T alpha (1 - alpha))) without location error', () => {
    let sum = 0;
    let terms = 0;
    for (let index = 1; index + 1 < count; index += 2) {
      const span = times[index + 1] - times[index - 1];
      const alpha = (times[index] - times[index - 1]) / span;
      const residualX = x[index] - (x[index - 1] + alpha * (x[index + 1] - x[index - 1]));
      const residualY = y[index] - (y[index - 1] + alpha * (y[index + 1] - y[index - 1]));
      sum += (residualX ** 2 + residualY ** 2) / (2 * span * alpha * (1 - alpha));
      terms++;
    }
    const estimate = estimateBrownianMotionVariance({x, y, times});
    expect(estimate).toBeCloseTo(sum / terms, 6);
    // And it recovers the variance that generated the track.
    expect(Math.abs(estimate / trueVariance - 1)).toBeLessThan(0.06);
  });

  it('maximizes the likelihood when a location error is present', () => {
    const error = 1.5;
    const noise = createNormal(createRandom(5));
    const noisyX = Float64Array.from(x, value => value + error * noise());
    const noisyY = Float64Array.from(y, value => value + error * noise());
    const estimate = estimateBrownianMotionVariance(
      {x: noisyX, y: noisyY, times},
      {locationError: error}
    );
    // Independent evaluation of the log-likelihood of the Horne et al. definition.
    const logLikelihood = (variance: number) => {
      let total = 0;
      for (let index = 1; index + 1 < count; index += 2) {
        const span = times[index + 1] - times[index - 1];
        const alpha = (times[index] - times[index - 1]) / span;
        const residualX =
          noisyX[index] - (noisyX[index - 1] + alpha * (noisyX[index + 1] - noisyX[index - 1]));
        const residualY =
          noisyY[index] - (noisyY[index - 1] + alpha * (noisyY[index + 1] - noisyY[index - 1]));
        const total2 =
          span * alpha * (1 - alpha) * variance + ((1 - alpha) ** 2 + alpha ** 2 + 1) * error ** 2;
        total += -Math.log(2 * Math.PI * total2) - (residualX ** 2 + residualY ** 2) / (2 * total2);
      }
      return total;
    };
    expect(logLikelihood(estimate)).toBeGreaterThan(logLikelihood(estimate * 1.02));
    expect(logLikelihood(estimate)).toBeGreaterThan(logLikelihood(estimate * 0.98));
    expect(Math.abs(estimate / trueVariance - 1)).toBeLessThan(0.12);
  });

  it('windowed estimates follow a change in the variance of the movement', () => {
    const normalChange = createNormal(createRandom(21));
    const size = 400;
    const track = {
      x: new Float64Array(size),
      y: new Float64Array(size),
      times: new Float64Array(size)
    };
    for (let index = 1; index < size; index++) {
      track.times[index] = index;
      const variance = index < 200 ? 1 : 25;
      track.x[index] = track.x[index - 1] + Math.sqrt(variance) * normalChange();
      track.y[index] = track.y[index - 1] + Math.sqrt(variance) * normalChange();
    }
    const windowed = estimateWindowedBrownianMotionVariance(track, {windowSize: 41});
    expect(windowed.length).toBe(size - 1);
    expect(windowed[60]).toBeGreaterThan(0.4);
    expect(windowed[60]).toBeLessThan(2.5);
    expect(windowed[340]).toBeGreaterThan(12);
    expect(windowed[340]).toBeLessThan(50);
  });
});

type Track = {x: number[]; y: number[]; times: number[]};

/** Binary64 Brownian bridge density at points, from the formulas of Horne et al. (2007). */
function bridgeDensityReference(
  tracks: Track[],
  motionVariance: number[][],
  locationError: number[][],
  points: [number, number][],
  subStepCount: number
): number[] {
  let totalDuration = 0;
  for (const track of tracks) {
    for (let index = 0; index + 1 < track.times.length; index++) {
      totalDuration += track.times[index + 1] - track.times[index];
    }
  }
  return points.map(([px, py]) => {
    let total = 0;
    tracks.forEach((track, trackIndex) => {
      for (let index = 0; index + 1 < track.times.length; index++) {
        const duration = track.times[index + 1] - track.times[index];
        let segment = 0;
        for (let step = 0; step < subStepCount; step++) {
          const alpha = (step + 0.5) / subStepCount;
          const meanX = track.x[index] + alpha * (track.x[index + 1] - track.x[index]);
          const meanY = track.y[index] + alpha * (track.y[index + 1] - track.y[index]);
          const variance =
            duration * alpha * (1 - alpha) * motionVariance[trackIndex][index] +
            (1 - alpha) ** 2 * locationError[trackIndex][index] ** 2 +
            alpha ** 2 * locationError[trackIndex][index + 1] ** 2;
          segment +=
            Math.exp(-((px - meanX) ** 2 + (py - meanY) ** 2) / (2 * variance)) /
            (2 * Math.PI * variance);
        }
        total += (duration * segment) / subStepCount;
      }
    });
    return total / totalDuration;
  });
}

async function runUtilization(options: {
  tracks: Track[];
  motionVariance: number[][];
  locationError: number[][];
  gridSize: [number, number];
  extent: [number, number, number, number];
  subStepCount: number;
  words?: boolean;
}) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = new GraphHarness(device, 'mob07-utilization');
  const flat = <Value>(select: (track: Track) => Value[]) => options.tracks.flatMap(select);
  const trackOffsets = [0];
  for (const track of options.tracks) {
    trackOffsets.push(trackOffsets[trackOffsets.length - 1] + track.times.length);
  }
  const rowCount = trackOffsets[trackOffsets.length - 1];
  const trajectory = importTrajectory(harness, {
    positions: flat(track => track.x.map((value, index) => [value, track.y[index]])),
    times: options.words
      ? flat(track => track.times.map(time => BigInt(Math.round(time * 1000))))
      : Float32Array.from(flat(track => track.times)),
    trackOffsets
  });
  const motionVariances = harness.input(
    'motion-variances',
    Float32Array.from(options.motionVariance.flatMap(values => [...values, 0])),
    'float32',
    rowCount
  );
  const locationErrors = harness.input(
    'location-errors',
    Float32Array.from(options.locationError.flat()),
    'float32',
    rowCount
  );
  const parameters = harness.input(
    'parameters',
    getGPUBrownianBridgeUtilizationParameterValues({extent: options.extent}),
    'float32',
    4
  );
  const cellCount = options.gridSize[0] * options.gridSize[1];
  const density = harness.output('density', 'float32', cellCount);
  harness.graph.add(
    new GPUBrownianBridgeUtilization({
      positions: trajectory.positions,
      timestamps: trajectory.timestamps,
      trackOffsets: trajectory.trackOffsets,
      motionVariances,
      locationErrors,
      gridSize: options.gridSize,
      parameters,
      subStepCount: options.subStepCount,
      output: {density: density.view}
    })
  );
  return harness.run(() => readFloat32(density.buffer, cellCount));
}

function getCellCenters(gridSize: [number, number], extent: [number, number, number, number]) {
  const centers: [number, number][] = [];
  const width = (extent[2] - extent[0]) / gridSize[0];
  const height = (extent[3] - extent[1]) / gridSize[1];
  for (let row = 0; row < gridSize[1]; row++) {
    for (let column = 0; column < gridSize[0]; column++) {
      centers.push([extent[0] + (column + 0.5) * width, extent[1] + (row + 0.5) * height]);
    }
  }
  return {centers, cellArea: width * height};
}

describe('GPUBrownianBridgeUtilization (MOB-07)', () => {
  const gridSize: [number, number] = [48, 36];
  const extent: [number, number, number, number] = [-6, -4, 18, 14];
  const tracks: Track[] = [
    {x: [0, 3, 7, 8, 12], y: [0, 2, 1, 5, 6], times: [0, 2, 3, 6, 8]},
    {x: [10, 11, 9], y: [-1, 0, 3], times: [0, 1, 4]}
  ];
  const motionVariance = [
    [0.8, 0.8, 1.5, 0.3],
    [0.5, 2]
  ];
  const locationError = [
    [0.4, 0.3, 0.5, 0.2, 0.4],
    [0.3, 0.3, 0.6]
  ];

  it('matches a binary64 reference of the bridge formulas, with float32 and Int64-word times', async () => {
    const {centers} = getCellCenters(gridSize, extent);
    const expected = bridgeDensityReference(tracks, motionVariance, locationError, centers, 15);
    const peak = Math.max(...expected);
    for (const words of [false, true]) {
      const actual = await runUtilization({
        tracks: words
          ? tracks.map(track => ({...track, times: track.times.map(time => time)}))
          : tracks,
        motionVariance: words
          ? motionVariance.map(values => values.map(value => value / 1000))
          : motionVariance,
        locationError,
        gridSize,
        extent,
        subStepCount: 15,
        words
      });
      // Word times are in ms: the variance per ms is a thousandth, and the density is unchanged.
      expected.forEach((value, index) => {
        expect(Math.abs(actual[index] - value)).toBeLessThanOrEqual(2e-4 * peak);
      });
    }
  });

  it('places mass 1 inside a grid that covers the track, and none where nothing can reach', async () => {
    const {centers, cellArea} = getCellCenters(gridSize, extent);
    const actual = await runUtilization({
      tracks,
      motionVariance,
      locationError,
      gridSize,
      extent,
      subStepCount: 15
    });
    const mass = actual.reduce((sum, value) => sum + value, 0) * cellArea;
    expect(mass).toBeGreaterThan(0.97);
    expect(mass).toBeLessThan(1.03);
    // The corner at (-6, 14) is more than 7 standard deviations from every segment.
    expect(actual[(gridSize[1] - 1) * gridSize[0]]).toBe(0);
    expect(centers.length).toBe(actual.length);
  });

  it('with one step reduces to a single Gaussian at the segment midpoint', async () => {
    // One segment, alpha = 1/2: mean (1, 1), variance T/4 sigma^2 + (delta_a^2 + delta_b^2) / 4.
    const track: Track = {x: [-1, 3], y: [-1, 3], times: [0, 4]};
    const variance = (4 / 4) * 2 + (0.5 ** 2 + 1 ** 2) / 4;
    const small: [number, number] = [8, 8];
    const smallExtent: [number, number, number, number] = [-5, -5, 7, 7];
    const actual = await runUtilization({
      tracks: [track],
      motionVariance: [[2]],
      locationError: [[0.5, 1]],
      gridSize: small,
      extent: smallExtent,
      subStepCount: 1
    });
    const {centers} = getCellCenters(small, smallExtent);
    centers.forEach(([px, py], index) => {
      const expected =
        Math.exp(-((px - 1) ** 2 + (py - 1) ** 2) / (2 * variance)) / (2 * Math.PI * variance);
      expect(actual[index]).toBeCloseTo(expected, 5);
    });
  });
});

async function runIsopleths(values: Float32Array, level: number, shapeLength: number) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = new GraphHarness(device, 'mob07-isopleths');
  const density = harness.input('density', values, 'float32', shapeLength);
  const parameters = harness.input(
    'parameters',
    getGPUUtilizationVolumeIsoplethsParameterValues({level}),
    'float32',
    4
  );
  const volume = harness.output('volume', 'float32', shapeLength);
  const inside = harness.output('inside', 'uint32', shapeLength);
  const totalMass = harness.output('total', 'float32', 1);
  harness.graph.add(
    new GPUUtilizationVolumeIsopleths({
      density,
      parameters,
      output: {volume: volume.view, inside: inside.view, totalMass: totalMass.view}
    })
  );
  return harness.run(async () => ({
    volume: await readFloat32(volume.buffer, shapeLength),
    inside: await readUint32(inside.buffer, shapeLength),
    total: (await readFloat32(totalMass.buffer, 1))[0]
  }));
}

describe('GPUUtilizationVolumeIsopleths (MOB-07)', () => {
  it('computes the volumes of a four-cell surface by hand', async () => {
    // Densities 4, 3, 2, 1 in cell order 2, 0, 3, 1: total 10, volumes 0.4 0.7 0.9 1.0 from the densest.
    const result = await runIsopleths(Float32Array.from([3, 1, 4, 2]), 0.8, 4);
    expect(result.total).toBe(10);
    expect(result.volume[2]).toBeCloseTo(0.4, 6);
    expect(result.volume[0]).toBeCloseTo(0.7, 6);
    expect(result.volume[3]).toBeCloseTo(0.9, 6);
    expect(result.volume[1]).toBeCloseTo(1.0, 6);
    // The 80% isopleth is the smallest set reaching 0.8: cells 2, 0 and 3 (0.4, 0.7, 0.9).
    expect(result.inside).toEqual([1, 0, 1, 1]);
    const exact = await runIsopleths(Float32Array.from([3, 1, 4, 2]), 0.7, 4);
    // Mass 0.7 is reached exactly by two cells, so a third is not needed.
    expect(exact.inside.reduce((sum, value) => sum + value, 0)).toBeGreaterThanOrEqual(2);
  });

  it('gives empty cells volume 1, outside every isopleth (trajectories-lines-04)', async () => {
    // The six densities of the review: 0.5, 0.3, 0.15, 0.05, 0, 0. Cumulative shares from the
    // densest are 0.50, 0.80, 0.95, 1.00 by hand. The two empty cells hold no mass, so they are
    // the last cells any isopleth would reach: volume 1, and `volume <= level` must not select
    // them. At level 0.9 the volume idiom selects cells 0 and 1; the mask also holds cell 2, whose
    // own mass crosses the level (0.8 before it, 0.95 with it).
    const densities = Float32Array.from([0.5, 0.3, 0.15, 0.05, 0, 0]);
    const result = await runIsopleths(densities, 0.9, 6);
    [0.5, 0.8, 0.95, 1].forEach((expected, cell) =>
      expect(result.volume[cell]).toBeCloseTo(expected, 6)
    );
    expect(result.volume[4]).toBe(1);
    expect(result.volume[5]).toBe(1);
    expect(Array.from(result.volume, value => (value <= 0.9 ? 1 : 0))).toEqual([1, 1, 0, 0, 0, 0]);
    expect(Array.from(result.inside)).toEqual([1, 1, 1, 0, 0, 0]);
    // Second check: a stable descending NumPy argsort and cumsum of the same float32 densities
    // (oracles/trajectories-lines-numpy-checks.py).
    const reference = numpyChecks.volumeIsopleths;
    expect(reference.level).toBe(0.9);
    expect(reference.densities).toEqual(Array.from(densities));
    reference.volume.forEach((expected, cell) =>
      expect(result.volume[cell], `NumPy volume ${cell}`).toBeCloseTo(expected, 6)
    );
    expect(Array.from(result.inside)).toEqual(reference.inside);
    // A surface with no mass at all: every cell is empty.
    const empty = await runIsopleths(new Float32Array(4), 0.9, 4);
    expect(Array.from(empty.volume)).toEqual([1, 1, 1, 1]);
    expect(Array.from(empty.inside)).toEqual([0, 0, 0, 0]);
  });

  it('matches a binary64 sort and cumulative sum over 70,000 cells (three prefix levels)', async () => {
    const random = createRandom(9);
    const count = 70000;
    const values = Float32Array.from({length: count}, () => random() ** 3);
    values[5] = 0;
    values[77] = NaN;
    values[78] = -1;
    const level = 0.95;
    const result = await runIsopleths(values, level, count);
    const clean = Array.from(values, value => (value > 0 ? value : 0));
    const total = clean.reduce((sum, value) => sum + value, 0);
    const order = clean
      .map((value, cell) => ({value, cell}))
      .sort((left, right) => right.value - left.value || left.cell - right.cell);
    expect(Math.abs(result.total / total - 1)).toBeLessThan(1e-4);
    let cumulative = 0;
    let borderline = 0;
    for (const {value, cell} of order) {
      const before = cumulative;
      cumulative += value;
      const expectedVolume = value > 0 ? cumulative / total : 1;
      expect(Math.abs(result.volume[cell] - expectedVolume)).toBeLessThan(2e-4);
      const expectedInside = value > 0 && before / total < level ? 1 : 0;
      if (Math.abs(before / total - level) < 1e-5) {
        borderline++;
      } else {
        expect(result.inside[cell]).toBe(expectedInside);
      }
    }
    expect(borderline).toBeLessThan(40);
    // Zero, NaN and negative cells hold no mass: volume 1, never inside.
    expect(result.volume[5]).toBe(1);
    expect(result.volume[77]).toBe(1);
    expect(result.volume[78]).toBe(1);
    expect(result.inside[78]).toBe(0);
  });

  it('turns a Brownian bridge surface into a home range that grows with the level', async () => {
    const gridSize: [number, number] = [48, 36];
    const extent: [number, number, number, number] = [-6, -4, 18, 14];
    const track: Track = {x: [0, 3, 7, 8, 12], y: [0, 2, 1, 5, 6], times: [0, 2, 3, 6, 8]};
    const density = await runUtilization({
      tracks: [track],
      motionVariance: [[0.8, 0.8, 1.5, 0.3]],
      locationError: [[0.4, 0.3, 0.5, 0.2, 0.4]],
      gridSize,
      extent,
      subStepCount: 15
    });
    const sizes: number[] = [];
    for (const level of [0.5, 0.9, 0.99]) {
      const result = await runIsopleths(Float32Array.from(density), level, density.length);
      sizes.push(result.inside.reduce((sum, value) => sum + value, 0));
    }
    expect(sizes[0]).toBeGreaterThan(5);
    expect(sizes[1]).toBeGreaterThan(sizes[0]);
    expect(sizes[2]).toBeGreaterThan(sizes[1]);
  });
});

type RecursionCase = {
  positions: number[][];
  times: Float32Array | bigint[];
  trackOffsets: number[];
  radii: number[];
  maximumTrackSegments?: number;
};

async function runRecursions(options: RecursionCase) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = new GraphHarness(device, 'mob07-recursions');
  const trajectory = importTrajectory(harness, options);
  const radiusCount = options.radii.length;
  const outputRows = trajectory.rowCount * radiusCount;
  const radii = harness.input('radii', Float32Array.from(options.radii), 'float32', radiusCount);
  const revisits = harness.output('revisits', 'uint32', outputRows);
  const residence = harness.output('residence', 'float32', outputRows);
  const passage = harness.output('passage', 'float32', outputRows);
  const censored = harness.output('censored', 'uint32', outputRows);
  const overflow = harness.output('overflow', 'uint32', 1);
  harness.graph.add(
    new GPUTrajectoryRecursions({
      positions: trajectory.positions,
      timestamps: trajectory.timestamps,
      trackOffsets: trajectory.trackOffsets,
      radii,
      maximumTrackSegments: options.maximumTrackSegments,
      output: {
        revisits: revisits.view,
        residenceTimes: residence.view,
        firstPassageTimes: passage.view,
        firstPassageCensored: censored.view,
        status: {candidateOverflow: overflow.view}
      }
    })
  );
  return harness.run(async () => ({
    revisits: await readUint32(revisits.buffer, outputRows),
    residence: await readFloat32(residence.buffer, outputRows),
    passage: await readFloat32(passage.buffer, outputRows),
    censored: await readUint32(censored.buffer, outputRows),
    overflow: (await readUint32(overflow.buffer, 1))[0]
  }));
}

/** Binary64 reference of the recursion definitions, by direct segment-circle intersection. */
function recursionReference(
  positions: number[][],
  times: number[],
  trackOffsets: number[],
  radii: number[]
) {
  const rowCount = positions.length;
  const results = {
    revisits: new Array<number>(rowCount * radii.length).fill(0),
    residence: new Array<number>(rowCount * radii.length).fill(0),
    passage: new Array<number>(rowCount * radii.length).fill(0),
    censored: new Array<number>(rowCount * radii.length).fill(0)
  };
  for (let track = 0; track + 1 < trackOffsets.length; track++) {
    const start = trackOffsets[track];
    const end = trackOffsets[track + 1];
    for (let row = start; row < end; row++) {
      radii.forEach((radius, radiusIndex) => {
        const slot = row * radii.length + radiusIndex;
        const [cx, cy] = positions[row];
        let visits = 0;
        let residence = 0;
        let outsideBefore = true;
        for (let index = start; index < end; index++) {
          const [px, py] = positions[index];
          const inside = Math.hypot(px - cx, py - cy) <= radius;
          if (index === start && inside) visits++;
          if (index + 1 < end) {
            const [qx, qy] = positions[index + 1];
            const dx = qx - px;
            const dy = qy - py;
            const a = dx * dx + dy * dy;
            const b = 2 * ((px - cx) * dx + (py - cy) * dy);
            const c = (px - cx) ** 2 + (py - cy) ** 2 - radius ** 2;
            let interval: [number, number] | undefined;
            if (a === 0) {
              if (c <= 0) interval = [0, 1];
            } else if (b * b - 4 * a * c >= 0) {
              const root = Math.sqrt(b * b - 4 * a * c);
              const first = Math.max((-b - root) / (2 * a), 0);
              const last = Math.min((-b + root) / (2 * a), 1);
              if (first <= last) interval = [first, last];
            }
            if (interval) {
              if (!inside) visits++;
              residence += (interval[1] - interval[0]) * (times[index + 1] - times[index]);
            }
          }
          outsideBefore = !inside;
        }
        expect(outsideBefore === true || outsideBefore === false).toBe(true);
        results.revisits[slot] = visits - 1;
        results.residence[slot] = residence;
        // First passage from the focal fix.
        let found = false;
        for (let index = row; index + 1 < end; index++) {
          const [qx, qy] = positions[index + 1];
          if (Math.hypot(qx - cx, qy - cy) > radius) {
            const [px, py] = positions[index];
            const dx = qx - px;
            const dy = qy - py;
            const a = dx * dx + dy * dy;
            const b = 2 * ((px - cx) * dx + (py - cy) * dy);
            const c = (px - cx) ** 2 + (py - cy) ** 2 - radius ** 2;
            const exit = (-b + Math.sqrt(b * b - 4 * a * c)) / (2 * a);
            results.passage[slot] =
              times[index] - times[row] + exit * (times[index + 1] - times[index]);
            found = true;
            break;
          }
        }
        if (!found) {
          results.passage[slot] = times[end - 1] - times[row];
          results.censored[slot] = 1;
        }
      });
    }
  }
  return results;
}

describe('GPUTrajectoryRecursions (MOB-07)', () => {
  // Track 0: out and back along the x axis, one fix per time unit. Track 1: stationary.
  const positions = [
    [0, 0],
    [1, 0],
    [2, 0],
    [3, 0],
    [2, 0],
    [1, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0]
  ];
  const timesList = [0, 1, 2, 3, 4, 5, 6, 0, 1, 2];
  const trackOffsets = [0, 7, 10];

  it('matches the hand-computed out-and-back track', async () => {
    const result = await runRecursions({
      positions,
      times: Float32Array.from(timesList),
      trackOffsets,
      radii: [0.5, 1.5, 3]
    });
    const at = (row: number, radius: number) => row * 3 + radius;
    // Fix 0, r = 0.5: inside for the first half of segment 0 and the last half of segment 5.
    expect(result.residence[at(0, 0)]).toBeCloseTo(1, 5);
    expect(result.revisits[at(0, 0)]).toBe(1);
    expect(result.passage[at(0, 0)]).toBeCloseTo(0.5, 5);
    expect(result.censored[at(0, 0)]).toBe(0);
    // Fix 3 (x = 3), r = 1.5: inside for x in [1.5, 3] on the way out and back: 0.5 + 1 + 1 + 0.5.
    expect(result.residence[at(3, 1)]).toBeCloseTo(3, 5);
    expect(result.revisits[at(3, 1)]).toBe(0);
    expect(result.passage[at(3, 1)]).toBeCloseTo(1.5, 5);
    // Fix 5 (x = 1), r = 3: the track never gets 3 away from it afterwards: censored at the end.
    expect(result.passage[at(5, 2)]).toBeCloseTo(1, 5);
    expect(result.censored[at(5, 2)]).toBe(1);
    // The stationary second track never sees the first: residence is its own duration.
    expect(result.residence[at(7, 0)]).toBeCloseTo(2, 5);
    expect(result.revisits[at(7, 0)]).toBe(0);
    expect(result.censored[at(7, 0)]).toBe(1);
    expect(result.passage[at(8, 0)]).toBeCloseTo(1, 5);
    expect(result.overflow).toBe(0);
  });

  it('gives the same answers for Int64 millisecond times, scaled', async () => {
    const result = await runRecursions({
      positions,
      times: timesList.map(time => BigInt(time * 1000)),
      trackOffsets,
      radii: [0.5, 1.5, 3]
    });
    expect(result.residence[0]).toBeCloseTo(1000, 2);
    expect(result.passage[0]).toBeCloseTo(500, 2);
    expect(result.passage[3 * 3 + 1]).toBeCloseTo(1500, 2);
  });

  it('matches a binary64 reference on a random walk with several radii', async () => {
    const random = createRandom(31);
    const normal = createNormal(random);
    const count = 160;
    const walk: number[][] = [[0, 0]];
    const walkTimes = [0];
    for (let index = 1; index < count; index++) {
      walk.push([walk[index - 1][0] + 0.6 * normal(), walk[index - 1][1] + 0.6 * normal()]);
      walkTimes.push(walkTimes[index - 1] + 0.5 + random());
    }
    const radii = [0.4, 1, 2.5];
    const actual = await runRecursions({
      positions: walk,
      times: Float32Array.from(walkTimes),
      trackOffsets: [0, count],
      radii
    });
    const expected = recursionReference(walk, walkTimes, [0, count], radii);
    expect(Math.max(...expected.revisits)).toBeGreaterThan(2);
    expected.revisits.forEach((value, slot) => {
      expect(actual.revisits[slot]).toBe(value);
      expect(actual.censored[slot]).toBe(expected.censored[slot]);
      expect(Math.abs(actual.residence[slot] - expected.residence[slot])).toBeLessThan(2e-3);
      expect(Math.abs(actual.passage[slot] - expected.passage[slot])).toBeLessThan(2e-3);
    });
  });

  it('examines a window of segments around the row when the track exceeds maximumTrackSegments', async () => {
    // Out and back along x over 10 segments, so the full track revisits every circle.
    const xs = [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0];
    const options = {
      positions: xs.map(x => [x, 0]),
      times: Float32Array.from(xs.map((_, index) => index)),
      trackOffsets: [0, xs.length],
      radii: [1.5, 3.5]
    };
    const full = await runRecursions({...options, maximumTrackSegments: 16});
    const windowed = await runRecursions({...options, maximumTrackSegments: 4});
    // Row 0 sees the whole track: out (1.5) and back (1.5). A window of 4 segments sees only the way out.
    expect(full.residence[0]).toBeCloseTo(3, 5);
    expect(full.revisits[0]).toBe(1);
    expect(windowed.residence[0]).toBeCloseTo(1.5, 5);
    expect(windowed.revisits[0]).toBe(0);
    // The last row (x = 0) takes the final 4 segments (6 to 9), the window shifted to stay inside
    // the track. With r = 3.5 it is inside for x <= 3.5: 0.5 + 1 + 1 + 1.
    expect(windowed.residence[10 * 2 + 1]).toBeCloseTo(3.5, 5);
    expect(windowed.revisits[10 * 2 + 1]).toBe(0);
    expect(windowed.overflow).toBe(1);
  });

  it('reports candidateOverflow when a track is longer than the examined window', async () => {
    const long = Array.from({length: 12}, (_, index) => [index, 0]);
    const options = {
      positions: long,
      times: Float32Array.from({length: 12}, (_, index) => index),
      trackOffsets: [0, 12],
      radii: [1.5]
    };
    expect((await runRecursions({...options, maximumTrackSegments: 4})).overflow).toBe(1);
    expect((await runRecursions({...options, maximumTrackSegments: 16})).overflow).toBe(0);
  });
});
