// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {readFile, writeFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import {addMetersToLngLat, getDistanceScales} from '@math.gl/web-mercator';
import {EARTH_RADIUS_METERS, measureHaversineDistance} from './lib/geodesy.mjs';

const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
const repositoryRoot = path.resolve(scriptDirectory, '../../../..');
const dataDirectory = path.join(
  repositoryRoot,
  'examples/deck/spatial-analysis-showcase/public/data/bixi-flows'
);
const outputPath = path.join(
  repositoryRoot,
  'dev-docs/research/showcase-overhaul-2026-10-25/bixi-models-reference.json'
);

async function readColumn(name, Type) {
  const bytes = await readFile(path.join(dataDirectory, `${name}.bin`));
  return new Type(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
}

const manifest = JSON.parse(await readFile(path.join(dataDirectory, 'manifest.json'), 'utf8'));
const stationNames = manifest.properties.stationNames;
const [locations, departures, arrivals, pairOrigins, pairDestinations, pairCounts] =
  await Promise.all([
    readColumn('locations', Float32Array),
    readColumn('stationDepartures', Uint32Array),
    readColumn('stationArrivals', Uint32Array),
    readColumn('origin', Uint16Array),
    readColumn('destination', Uint16Array),
    readColumn('count', Uint16Array)
  ]);

const stationIds = Array.from({length: departures.length}, (_, station) => station)
  .sort((left, right) => departures[right] + arrivals[right] - departures[left] - arrivals[left] || left - right)
  .slice(0, 16);
const localIndex = new Map(stationIds.map((station, index) => [station, index]));
const stationCount = stationIds.length;
const coordinates = stationIds.map(station => [locations[station * 2], locations[station * 2 + 1]]);
const projectionOrigin = [-73.58, 45.52];
const projectionScales = getDistanceScales({
  longitude: projectionOrigin[0],
  latitude: projectionOrigin[1],
  highPrecision: true
});
const metersPerDegree = [
  projectionScales.unitsPerDegree[0] * projectionScales.metersPerUnit[0],
  projectionScales.unitsPerDegree[1] * projectionScales.metersPerUnit[1]
];
const positions = stationIds.map((station, index) => {
  const [longitude, latitude] = coordinates[index];
  let x = (longitude - projectionOrigin[0]) * metersPerDegree[0];
  let y = (latitude - projectionOrigin[1]) * metersPerDegree[1];
  const [estimatedLongitude, estimatedLatitude] = addMetersToLngLat(projectionOrigin, [x, y]);
  x += (longitude - estimatedLongitude) * metersPerDegree[0];
  y += (latitude - estimatedLatitude) * metersPerDegree[1];
  // Production projectColumn writes the refined coordinates into Float32Array before cost building.
  return [Math.fround(x), Math.fround(y)];
});
const activity = stationIds.map(station => departures[station] + arrivals[station]);
const pairCountsByIndex = new Map();
for (let row = 0; row < pairCounts.length; row++) {
  const origin = localIndex.get(pairOrigins[row]);
  const destination = localIndex.get(pairDestinations[row]);
  if (origin !== undefined && destination !== undefined)
    pairCountsByIndex.set(`${origin}:${destination}`, pairCounts[row]);
}
const pairs = [];
for (let origin = 0; origin < stationCount; origin++) {
  for (let destination = 0; destination < stationCount; destination++) {
    if (origin === destination) continue;
    const count = pairCountsByIndex.get(`${origin}:${destination}`);
    if (count === undefined) continue;
    const [originLongitude, originLatitude] = coordinates[origin];
    const [destinationLongitude, destinationLatitude] = coordinates[destination];
    const distanceKm = measureHaversineDistance(
      [originLongitude, originLatitude],
      [destinationLongitude, destinationLatitude],
      EARTH_RADIUS_METERS / 1000
    );
    pairs.push({origin, destination, count, distanceKm});
  }
}

function fit(constraint, coefficient) {
  const predictions = pairs.map(pair => Math.exp(coefficient * pair.distanceKm));
  const originMargins = new Array(stationCount).fill(0);
  const destinationMargins = new Array(stationCount).fill(0);
  const fittedOrigins = new Array(stationCount).fill(0);
  const fittedDestinations = new Array(stationCount).fill(0);
  for (const pair of pairs) {
    originMargins[pair.origin] += pair.count;
    destinationMargins[pair.destination] += pair.count;
  }
  if (constraint === 'unconstrained') {
    const scale = pairs.reduce((sum, pair) => sum + pair.count, 0) /
      predictions.reduce((sum, value) => sum + value, 0);
    for (let index = 0; index < predictions.length; index++) predictions[index] *= scale;
  } else if (constraint === 'origin' || constraint === 'destination') {
    const useOrigins = constraint === 'origin';
    const margins = useOrigins ? originMargins : destinationMargins;
    for (let station = 0; station < stationCount; station++) {
      const baseline = pairs.reduce((sum, pair, index) =>
        sum + ((useOrigins ? pair.origin : pair.destination) === station ? predictions[index] : 0), 0);
      const factor = baseline > 0 ? margins[station] / baseline : 0;
      for (let index = 0; index < pairs.length; index++) {
        if ((useOrigins ? pairs[index].origin : pairs[index].destination) === station)
          predictions[index] *= factor;
      }
    }
  } else {
    for (let iteration = 0; iteration < 2048; iteration++) {
      for (let station = 0; station < stationCount; station++) {
        const baseline = pairs.reduce((sum, pair, index) => sum + (pair.origin === station ? predictions[index] : 0), 0);
        if (baseline > 0) for (let index = 0; index < pairs.length; index++)
          if (pairs[index].origin === station) predictions[index] *= originMargins[station] / baseline;
      }
      for (let station = 0; station < stationCount; station++) {
        const baseline = pairs.reduce((sum, pair, index) => sum + (pair.destination === station ? predictions[index] : 0), 0);
        if (baseline > 0) for (let index = 0; index < pairs.length; index++)
          if (pairs[index].destination === station) predictions[index] *= destinationMargins[station] / baseline;
      }
      if (maximumMarginalError(predictions, originMargins, destinationMargins) < 1e-8) break;
    }
  }
  predictions.forEach((prediction, index) => {
    fittedOrigins[pairs[index].origin] += prediction;
    fittedDestinations[pairs[index].destination] += prediction;
  });
  const marginalError = constraint === 'unconstrained' ? 0 : maximumMarginalError(predictions, originMargins, destinationMargins, constraint);
  const deviance = predictions.reduce((sum, prediction, index) => {
    const observed = pairs[index].count;
    return sum + 2 * (observed > 0 ? observed * Math.log(observed / prediction) - observed + prediction : prediction);
  }, 0);
  return {coefficient, deviance, marginalError, predictions, fittedOrigins, fittedDestinations};
}

function maximumMarginalError(predictions, originMargins, destinationMargins, constraint = 'doubly') {
  const fittedOrigins = new Array(stationCount).fill(0);
  const fittedDestinations = new Array(stationCount).fill(0);
  predictions.forEach((prediction, index) => {
    fittedOrigins[pairs[index].origin] += prediction;
    fittedDestinations[pairs[index].destination] += prediction;
  });
  let error = 0;
  if (constraint === 'origin' || constraint === 'doubly') for (let station = 0; station < stationCount; station++)
    error = Math.max(error, Math.abs(fittedOrigins[station] - originMargins[station]) / Math.max(1, originMargins[station]));
  if (constraint === 'destination' || constraint === 'doubly') for (let station = 0; station < stationCount; station++)
    error = Math.max(error, Math.abs(fittedDestinations[station] - destinationMargins[station]) / Math.max(1, destinationMargins[station]));
  return error;
}

const constraints = ['unconstrained', 'origin', 'destination', 'doubly'];
const coefficientGrid = Array.from({length: 9}, (_, index) => -1.5 + index * 0.1875);
const gridFits = Object.fromEntries(constraints.map(constraint => {
  const candidates = coefficientGrid.map(coefficient => fit(constraint, coefficient));
  const best = candidates.reduce((winner, candidate) => candidate.deviance < winner.deviance ? candidate : winner);
  return [constraint, {coefficient: best.coefficient, deviance: best.deviance, marginalError: best.marginalError}];
}));

// Bounded calibration is independently bracketed using a dense deterministic scan for the oracle.
const boundedFits = Object.fromEntries(constraints.map(constraint => {
  const candidates = Array.from({length: 301}, (_, index) => fit(constraint, -1.5 + index * 0.005));
  const best = candidates.reduce((winner, candidate) => candidate.deviance < winner.deviance ? candidate : winner);
  return [constraint, {coefficient: best.coefficient, deviance: best.deviance, marginalError: best.marginalError}];
}));

const facilityCandidateIndexes = [0];
while (facilityCandidateIndexes.length < 8) {
  let bestStation = -1;
  let bestDistance = -1;
  for (let station = 0; station < stationCount; station++) {
    if (facilityCandidateIndexes.includes(station)) continue;
    const nearest = Math.min(...facilityCandidateIndexes.map(candidate => Math.hypot(
      positions[station][0] - positions[candidate][0], positions[station][1] - positions[candidate][1]
    )));
    if (nearest > bestDistance) { bestDistance = nearest; bestStation = station; }
  }
  facilityCandidateIndexes.push(bestStation);
}
let costs = Float32Array.from(positions.flatMap(position => facilityCandidateIndexes.map(candidate =>
  Math.hypot(position[0] - positions[candidate][0], position[1] - positions[candidate][1]) / 1000
)));
function evaluateSelection(selected, operation, demandOrder = Array.from({length: stationCount}, (_, index) => index), capacities = null) {
  const remaining = capacities ? selected.map(candidate => capacities[candidate]) : [];
  const assignments = new Array(stationCount).fill(-1);
  let objective = operation === 'p-center' ? 0 : 0;
  let unassignedCount = 0;
  for (const demandIndex of demandOrder) {
    let chosenSlot = -1;
    let chosenCost = Infinity;
    for (let slot = 0; slot < selected.length; slot++) {
      const cost = costs[demandIndex * facilityCandidateIndexes.length + selected[slot]];
      if (cost < chosenCost && (!capacities || remaining[slot] >= activity[demandIndex])) {
        chosenSlot = slot;
        chosenCost = cost;
      }
    }
    if (chosenSlot < 0) { unassignedCount++; continue; }
    assignments[demandIndex] = selected[chosenSlot];
    if (capacities) remaining[chosenSlot] -= activity[demandIndex];
    if (operation === 'p-center') objective = Math.max(objective, chosenCost);
    else if (operation === 'maximum-coverage') {
      if (chosenCost <= 1.5) objective += activity[demandIndex];
    } else if (operation === 'set-covering') {
      if (chosenCost > 1.5) unassignedCount++;
    } else objective += activity[demandIndex] * chosenCost;
  }
  return {selected, assignments, objective: unassignedCount ? null : objective, unassignedCount};
}
function combinations(length, selectCount) {
  const result = [];
  const choose = (start, prefix) => {
    if (prefix.length === selectCount) { result.push(prefix); return; }
    for (let value = start; value < length; value++) choose(value + 1, [...prefix, value]);
  };
  choose(0, []);
  return result;
}
function selectObjectiveWinner(candidates, maximize) {
  return candidates.reduce((winner, candidate) => {
    if (!winner) return candidate;
    const improvesObjective = maximize
      ? candidate.objective > winner.objective
      : candidate.objective < winner.objective;
    return improvesObjective ? candidate : winner;
  }, null);
}
const objectiveDirectionCandidates = [{id: 'lower', objective: 2}, {id: 'higher', objective: 7}];
const objectiveDirectionFixture = {
  candidateObjectives: objectiveDirectionCandidates.map(candidate => candidate.objective),
  minimumWinner: selectObjectiveWinner(objectiveDirectionCandidates, false),
  maximumWinner: selectObjectiveWinner(objectiveDirectionCandidates, true)
};
assert.equal(objectiveDirectionFixture.minimumWinner.objective, 2);
assert.equal(objectiveDirectionFixture.maximumWinner.objective, 7);
function exactBy(operation, capacities = null) {
  const candidates = combinations(facilityCandidateIndexes.length, operation === 'set-covering' ? 8 : 3).map(selection =>
    evaluateSelection(selection, operation, undefined, capacities)
  );
  const feasible = candidates.filter(candidate => candidate.unassignedCount === 0);
  return selectObjectiveWinner(feasible, operation === 'maximum-coverage');
}
const weightsTotal = activity.reduce((sum, value) => sum + value, 0);
function bestCapacitated(demandOrder) {
  return combinations(facilityCandidateIndexes.length, 3).map(selection =>
    evaluateSelection(selection, 'capacitated-p-median', demandOrder, capacities)
  ).filter(candidate => candidate.unassignedCount === 0)
    .sort((left, right) => left.objective - right.objective)[0] ?? null;
}
const capacities = new Array(facilityCandidateIndexes.length).fill(weightsTotal * 0.4);
const descendingActivityOrder = Array.from({length: stationCount}, (_, index) => index)
  .sort((left, right) => activity[right] - activity[left] || left - right);
const denseAllocation = {
  candidateOriginalStationIds: facilityCandidateIndexes.map(index => stationIds[index]),
  demandWeightTotal: weightsTotal,
  coverageRadiusKm: 1.5,
  pMedianExactSmallSet: exactBy('p-median'),
  maximumCoverageExactSmallSet: exactBy('maximum-coverage'),
  setCoveringExactSmallSet: (() => {
    for (let facilityCount = 1; facilityCount <= facilityCandidateIndexes.length; facilityCount++) {
      const complete = combinations(facilityCandidateIndexes.length, facilityCount)
        .map(selection => evaluateSelection(selection, 'set-covering'))
        .find(candidate => candidate.unassignedCount === 0);
      if (complete) return complete;
    }
    return null;
  })(),
  pCenterExact: exactBy('p-center'),
  capacitatedByStationIdOrder: bestCapacitated(
    Array.from({length: stationCount}, (_, index) => index).sort((left, right) => stationIds[left] - stationIds[right])
  ),
  capacitatedByDescendingActivity: bestCapacitated(descendingActivityOrder)
};
const denseCosts = costs;
const sparseCostCutoffKm = 3;
const sparseEntryCount = denseCosts.filter(cost => cost <= sparseCostCutoffKm).length;
costs = Float32Array.from(denseCosts, cost => cost <= sparseCostCutoffKm ? cost : Infinity);
const allocation = {
  ...denseAllocation,
  sparseCostCutoffKm,
  sparseEntryCount,
  sparseMissingDemandRows: Array.from({length: stationCount}, (_, demand) =>
    costs.slice(demand * facilityCandidateIndexes.length, (demand + 1) * facilityCandidateIndexes.length)
      .every(cost => !Number.isFinite(cost))
  ).filter(Boolean).length,
  pMedianSparseExactSmallSet: exactBy('p-median')
};

const result = {
  status: 'passed',
  provenance: 'CPU reference independently reconstructs the induced monthly pair support and LocalMetricProjection station distances from bundled binary columns; allocation positions and costs are rounded to float32 as in production before exhaustive objective evaluation.',
  sample: {
    stationIds,
    stationNames: stationIds.map(station => stationNames[station]),
    stationCount,
    observedPairCount: pairs.length,
    structuralZeroCount: stationCount,
    missingPairCount: stationCount * (stationCount - 1) - pairs.length,
    observedCountTotal: pairs.reduce((sum, pair) => sum + pair.count, 0),
    coefficientGrid
  },
  gridFits,
  boundedFits,
  allocation,
  objectiveDirectionFixture
};
assert.equal(stationCount, 16);
assert.equal(result.sample.observedPairCount + result.sample.missingPairCount, 240);
assert.equal(result.sample.structuralZeroCount, 16);
assert.deepEqual(Object.fromEntries(Object.entries(gridFits).map(([constraint, value]) => [constraint, value.coefficient])), {
  unconstrained: -0.9375, origin: -0.9375, destination: -0.9375, doubly: -1.125
});
assert(Object.values(gridFits).every(value => Number.isFinite(value.deviance) && value.marginalError < 1e-6));
assert(allocation.setCoveringExactSmallSet?.unassignedCount === 0);
assert(allocation.capacitatedByStationIdOrder && allocation.capacitatedByDescendingActivity);
assert(allocation.capacitatedByStationIdOrder.objective !== allocation.capacitatedByDescendingActivity.objective);
assert(allocation.sparseEntryCount > 0 && allocation.sparseEntryCount < stationCount * 8);
assert(allocation.pMedianSparseExactSmallSet === null || allocation.pMedianSparseExactSmallSet.unassignedCount === 0);
assert(allocation.pCenterExact && allocation.pCenterExact.unassignedCount === 0);
await writeFile(outputPath, `${JSON.stringify(result, null, 2)}\n`);
console.log(JSON.stringify({outputPath, sample: result.sample, gridFits, boundedFits, allocation}, null, 2));
