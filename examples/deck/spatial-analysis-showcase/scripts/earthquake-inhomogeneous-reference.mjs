// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {readFile, writeFile} from 'node:fs/promises';
import {EARTH_RADIUS_METERS, measureHaversineAngleRadians, toRadians} from './lib/geodesy.mjs';

const directory = new URL('../public/data/poopdeck-earthquakes/', import.meta.url);
const [positions, types] = await Promise.all(
  ['position.bin', 'type.bin'].map(name => readFile(new URL(name, directory)))
);
const originLatitude = toRadians(36.5);
const originLongitude = toRadians(140.5);
const earthRadius = EARTH_RADIUS_METERS;
const bounds = {minimumX: -135000, maximumX: 135000, minimumY: -166000, maximumY: 166000};
const width = bounds.maximumX - bounds.minimumX;
const height = bounds.maximumY - bounds.minimumY;
const area = width * height;
const events = [];

for (let row = 0; row < types.length; row++) {
  if (types[row] !== 0) continue;
  const latitude = toRadians(positions.readFloatLE(row * 8 + 4));
  const longitudeDelta = toRadians(positions.readFloatLE(row * 8)) - originLongitude;
  const angle = measureHaversineAngleRadians(latitude, originLatitude, longitudeDelta);
  const scale = earthRadius * (angle === 0 ? 1 : angle / Math.sin(angle));
  const horizontal = scale * Math.cos(latitude) * Math.sin(longitudeDelta);
  const vertical =
    scale *
    (Math.cos(originLatitude) * Math.sin(latitude) -
      Math.sin(originLatitude) * Math.cos(latitude) * Math.cos(longitudeDelta));
  if (
    horizontal < bounds.minimumX ||
    horizontal > bounds.maximumX ||
    vertical < bounds.minimumY ||
    vertical > bounds.maximumY
  )
    continue;
  events.push({
    sourceRow: row,
    positionX: Math.fround(horizontal),
    positionY: Math.fround(vertical)
  });
}
assert.equal(events.length, 802, 'The independent archive selection must contain 802 events');

const radii = Array.from({length: 16}, (_, index) => Math.fround((index + 1) * 5000));
const intensityBandwidths = [75000, 150000, 300000];
const pairBandwidths = [10000, 25000];
const cacheCases = [
  {name: 'constant-translation-annular', model: 'constant', correction: 'translation', estimator: 'annular'},
  {name: 'constant-border-annular', model: 'constant', correction: 'border', estimator: 'annular'},
  {name: 'epanechnikov-75-exclude-border', model: 'epanechnikov', intensityBandwidth: 75000, self: 'exclude', correction: 'border', estimator: 'annular'},
  {name: 'epanechnikov-150-include-border', model: 'epanechnikov', intensityBandwidth: 150000, self: 'include', correction: 'border', estimator: 'annular'},
  {name: 'epanechnikov-300-exclude-translation-annular', model: 'epanechnikov', intensityBandwidth: 300000, self: 'exclude', correction: 'translation', estimator: 'annular'},
  {name: 'epanechnikov-75-include-translation-annular', model: 'epanechnikov', intensityBandwidth: 75000, self: 'include', correction: 'translation', estimator: 'annular'},
  {name: 'kernel-10-radius-none', model: 'epanechnikov', intensityBandwidth: 150000, self: 'exclude', correction: 'translation', estimator: 'epanechnikov', pairBandwidth: 10000, divisor: 'radius', zeroCorrection: 'none'},
  {name: 'kernel-10-distance-reflection', model: 'epanechnikov', intensityBandwidth: 150000, self: 'include', correction: 'translation', estimator: 'epanechnikov', pairBandwidth: 10000, divisor: 'distance', zeroCorrection: 'reflection'},
  {name: 'kernel-25-radius-reflection', model: 'epanechnikov', intensityBandwidth: 300000, self: 'exclude', correction: 'translation', estimator: 'epanechnikov', pairBandwidth: 25000, divisor: 'radius', zeroCorrection: 'reflection'},
  {name: 'kernel-25-distance-none', model: 'epanechnikov', intensityBandwidth: 75000, self: 'include', correction: 'translation', estimator: 'epanechnikov', pairBandwidth: 25000, divisor: 'distance', zeroCorrection: 'none'}
];
const coordinateCounts = new Map();
for (const event of events) {
  const key = `${event.positionX},${event.positionY}`;
  coordinateCounts.set(key, (coordinateCounts.get(key) ?? 0) + 1);
}
const coincidentPairCount = [...coordinateCounts.values()].reduce(
  (total, count) => total + (count * (count - 1)) / 2,
  0
);

function rectangleKernelMass(event, bandwidth) {
  // Integrate the normalized compact Epanechnikov kernel over the clipped rectangle.
  // Integrating x analytically leaves a smooth one-dimensional Simpson integral in y.
  const lowerY = Math.max(bounds.minimumY - event.positionY, -bandwidth);
  const upperY = Math.min(bounds.maximumY - event.positionY, bandwidth);
  if (!(upperY > lowerY)) return 0;
  const intervals = 128;
  const step = (upperY - lowerY) / intervals;
  let total = 0;
  for (let index = 0; index <= intervals; index++) {
    const deltaY = lowerY + index * step;
    const remainingSquared = 1 - (deltaY / bandwidth) ** 2;
    if (remainingSquared <= 0) continue;
    const extentX = bandwidth * Math.sqrt(remainingSquared);
    const lowerX = Math.max(bounds.minimumX - event.positionX, -extentX);
    const upperX = Math.min(bounds.maximumX - event.positionX, extentX);
    if (!(upperX > lowerX)) continue;
    const integral =
      (upperX - lowerX) * remainingSquared -
      (upperX ** 3 - lowerX ** 3) / (3 * bandwidth ** 2);
    total += (index === 0 || index === intervals ? 1 : index % 2 === 0 ? 2 : 4) * integral;
  }
  return (2 / (Math.PI * bandwidth ** 2)) * (step / 3) * total;
}

function estimateIntensities(scenario) {
  if (scenario.model === 'constant') {
    const suppliedIntensity = Math.fround(events.length / area);
    return events.map(() => suppliedIntensity);
  }
  const bandwidth = scenario.intensityBandwidth;
  return events.map((focus, focusIndex) => {
    let kernelTotal = 0;
    for (let sourceIndex = 0; sourceIndex < events.length; sourceIndex++) {
      if (scenario.self === 'exclude' && sourceIndex === focusIndex) continue;
      const source = events[sourceIndex];
      const horizontalDelta = (focus.positionX - source.positionX) / bandwidth;
      const verticalDelta = (focus.positionY - source.positionY) / bandwidth;
      kernelTotal += Math.max(0, 1 - horizontalDelta * horizontalDelta - verticalDelta * verticalDelta);
    }
    const mass = rectangleKernelMass(focus, bandwidth);
    assert.ok(mass > 0, 'Every event must have positive clipped kernel mass');
    return (kernelTotal * (2 / Math.PI)) / (bandwidth * bandwidth * mass);
  });
}

function getBorderDistance(event) {
  return Math.min(
    event.positionX - bounds.minimumX,
    bounds.maximumX - event.positionX,
    event.positionY - bounds.minimumY,
    bounds.maximumY - event.positionY
  );
}

function getPairWeight(first, second, firstIntensity, secondIntensity, correction) {
  const overlap = correction === 'translation'
    ? (width - Math.abs(first.positionX - second.positionX)) * (height - Math.abs(first.positionY - second.positionY))
    : 1;
  return correction === 'translation'
    ? 1 / (firstIntensity * secondIntensity * overlap)
    : 1 / (firstIntensity * secondIntensity);
}

function calculateScenario(scenario) {
  const intensities = estimateIntensities(scenario);
  let previousK = 0;
  const bins = radii.map((radius, radiusIndex) => {
    const lowerRadius = radiusIndex === 0 ? 0 : radii[radiusIndex - 1];
    let kNumerator = 0;
    let kDenominator = 0;
    let pairCorrelation = 0;
    let invalidDistanceDivisor = false;
    if (scenario.correction === 'border') {
      for (let firstIndex = 0; firstIndex < events.length; firstIndex++) {
        const first = events[firstIndex];
        if (getBorderDistance(first) <= radius) continue;
        kDenominator += 1 / intensities[firstIndex];
        for (let secondIndex = 0; secondIndex < events.length; secondIndex++) {
          if (firstIndex === secondIndex) continue;
          const second = events[secondIndex];
          if (Math.hypot(first.positionX - second.positionX, first.positionY - second.positionY) <= radius) {
            kNumerator += 1 / (intensities[firstIndex] * intensities[secondIndex]);
          }
        }
      }
    } else {
      for (let firstIndex = 0; firstIndex < events.length; firstIndex++) {
        const first = events[firstIndex];
        for (let secondIndex = firstIndex + 1; secondIndex < events.length; secondIndex++) {
          const second = events[secondIndex];
          if (Math.hypot(first.positionX - second.positionX, first.positionY - second.positionY) <= radius) {
            kNumerator +=
              2 *
              getPairWeight(
                first,
                second,
                intensities[firstIndex],
                intensities[secondIndex],
                'translation'
              );
          }
        }
      }
    }

    const kStatistic = scenario.correction === 'border' ? kNumerator / kDenominator : kNumerator;
    if (scenario.estimator === 'epanechnikov') {
      const bandwidth = scenario.pairBandwidth;
      const kernel = offset => (0.75 / bandwidth) * Math.max(0, 1 - (offset / bandwidth) ** 2);
      for (let firstIndex = 0; firstIndex < events.length; firstIndex++) {
        const first = events[firstIndex];
        for (let secondIndex = firstIndex + 1; secondIndex < events.length; secondIndex++) {
          const second = events[secondIndex];
          const distance = Math.hypot(first.positionX - second.positionX, first.positionY - second.positionY);
          const divisor = scenario.divisor === 'distance' ? distance : radius;
          if (!(divisor > 0)) {
            if (distance <= radius + bandwidth) invalidDistanceDivisor = true;
            continue;
          }
          const pairWeight = getPairWeight(first, second, intensities[firstIndex], intensities[secondIndex], 'translation');
          let kernelValue = kernel(radius - distance);
          if (scenario.zeroCorrection === 'reflection') kernelValue += kernel(radius + distance);
          pairCorrelation += (2 * pairWeight * kernelValue) / (2 * Math.PI * divisor);
        }
      }
    } else {
      pairCorrelation = (kStatistic - previousK) / (Math.PI * (radius * radius - lowerRadius * lowerRadius));
    }
    previousK = kStatistic;
    return {
      radius,
      k: kStatistic,
      lMinusR: Math.sqrt(kStatistic / Math.PI) - radius,
      pairCorrelation: invalidDistanceDivisor ? null : pairCorrelation,
      valid: !invalidDistanceDivisor
    };
  });
  return {
    name: scenario.name,
    options: scenario,
    terminal: bins.at(-1),
    bins,
    intensityRange: [Math.min(...intensities), Math.max(...intensities)]
  };
}

const scenarios = cacheCases.map(calculateScenario);
assert.ok(scenarios.every(scenario => Number.isFinite(scenario.terminal.k)));
const outputFile = process.argv[2] ?? '/tmp/earthquake-inhomogeneous-reference.json';
await writeFile(
  outputFile,
  JSON.stringify({
    source: 'poopdeck-earthquakes position/depth/type float32 archive columns; 802 type-0 rows inside fixed AEQD rectangle',
    sourceSupport: [
      'src/scenes/points/earthquake-pair-structure-data.ts: type-0 archive filter, fixed AEQD rectangle, float32 projected positions.',
      'src/scenes/points/earthquake-pair-structure.compute.ts: 16 radii through 80 km, supplied N/area intensity, option applicability and output definitions.',
      'modules/graph/src/gpu-spatial-analysis/point-process/gpu-inhomogeneous-point-pattern.ts: Epanechnikov intensity, border estimator, annular K differences, public props and invalid divisor status.',
      'modules/graph/src/gpu-spatial-analysis/point-process/translation-point-pattern.ts: unordered overlap weights, translation K reduction, compact pair-correlation kernel, divisor and reflection formulas.',
      'modules/graph/src/utils/study-window-wgsl.ts: normalized Epanechnikov kernel mass over the rectangle.'
    ],
    events: events.length,
    bounds,
    area,
    radiusCount: radii.length,
    terminalRadius: radii.at(-1),
    plannedCaseCount: cacheCases.length,
    assumptions: [
      `The distance divisor rejects coincident distinct events; this selected float32 frame has ${coincidentPairCount} coincident coordinate pairs.`,
      'Projected input coordinates are rounded to float32 before all pair distances and intensities.',
      'The compact Epanechnikov kernel intensity is independently integrated over the clipped rectangle with 128-interval Simpson quadrature in y.',
      'Translation overlap is the exact shifted rectangle intersection area.',
      'Border K erodes focal support by radius; the ordered neighbour sum is normalized by the surviving inverse-intensity focal total.',
      'Translation K is the unordered pair sum multiplied by two, weighted by inverse intensity product and inverse rectangle overlap.',
      'Kernel pair correlation mirrors the graph contract formula with half-width bandwidth, radius or distance divisor, and optional reflected kernel.',
      'This is a finite descriptive estimator on archived rows, not a stochastic model or causal/hazard inference.'
    ],
    scenarios
  }, null, 2) + '\n'
);
console.log(JSON.stringify(scenarios.map(scenario => ({name: scenario.name, terminal: scenario.terminal})), null, 2));

