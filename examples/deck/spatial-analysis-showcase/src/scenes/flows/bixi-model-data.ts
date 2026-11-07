// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Viewport} from '@deck.gl/core';
import {haversineMeters} from '../../cartography/anchors';
import {MONTREAL_ORIGIN, type BixiFlows} from './bixi-data';

/** Small induced BIXI support shared by the interaction and hypothetical siting studies. */
export type BixiModelData = {
  stationIds: Uint32Array;
  stationNames: readonly string[];
  positions: Float32Array;
  geographicPositions: Float32Array;
  activity: Float32Array;
  candidateIds: Uint32Array;
  candidatePositions: Float32Array;
  origins: Uint32Array;
  destinations: Uint32Array;
  flows: Float32Array;
  distancesKm: Float32Array;
  pairState: Uint32Array;
  observedPairCount: number;
  missingPairCount: number;
};

const STATION_COUNT = 16;
const FACILITY_CANDIDATE_COUNT = 8;

/** Same class palette used for assignment spokes, station fills and their categorical legend. */
export const BIXI_ASSIGNMENT_PALETTE = [
  [78, 201, 255, 255],
  [255, 148, 72, 255],
  [189, 122, 255, 255],
  [87, 235, 168, 255],
  [255, 105, 168, 255],
  [245, 220, 87, 255],
  [107, 158, 255, 255],
  [255, 92, 92, 255]
] as const;

/** Deterministic high-activity station sample and induced monthly pair support. */
export function makeBixiModelData(flows: BixiFlows): BixiModelData {
  const stationIds = Array.from({length: flows.stationCount}, (_, station) => station)
    .sort(
      (left, right) =>
        flows.departures[right] +
          flows.arrivals[right] -
          (flows.departures[left] + flows.arrivals[left]) || left - right
    )
    .slice(0, STATION_COUNT);
  const localStationIds = new Map(stationIds.map((station, index) => [station, index]));
  const positions = new Float32Array(STATION_COUNT * 2);
  const geographicPositions = new Float32Array(STATION_COUNT * 2);
  const activity = new Float32Array(STATION_COUNT);
  stationIds.forEach((station, index) => {
    positions.set(flows.centers.subarray(station * 2, station * 2 + 2), index * 2);
    geographicPositions.set(flows.lngLat.subarray(station * 2, station * 2 + 2), index * 2);
    activity[index] = flows.departures[station] + flows.arrivals[station];
  });

  const candidateIndexes: number[] = [0];
  while (candidateIndexes.length < FACILITY_CANDIDATE_COUNT) {
    let nextIndex = -1;
    let greatestNearestDistance = -1;
    for (let stationIndex = 0; stationIndex < STATION_COUNT; stationIndex++) {
      if (candidateIndexes.includes(stationIndex)) continue;
      const nearestDistance = Math.min(
        ...candidateIndexes.map(candidateIndex =>
          Math.hypot(
            positions[stationIndex * 2] - positions[candidateIndex * 2],
            positions[stationIndex * 2 + 1] - positions[candidateIndex * 2 + 1]
          )
        )
      );
      if (nearestDistance > greatestNearestDistance) {
        greatestNearestDistance = nearestDistance;
        nextIndex = stationIndex;
      }
    }
    candidateIndexes.push(nextIndex);
  }
  const candidateIds = Uint32Array.from(candidateIndexes, index => stationIds[index]);
  const candidatePositions = new Float32Array(candidateIndexes.length * 2);
  candidateIndexes.forEach((stationIndex, candidateIndex) =>
    candidatePositions.set(
      positions.subarray(stationIndex * 2, stationIndex * 2 + 2),
      candidateIndex * 2
    )
  );

  const countByPair = new Map<string, number>();
  for (let row = 0; row < flows.pairs.count.length; row++) {
    const localOrigin = localStationIds.get(flows.pairs.origin[row]);
    const localDestination = localStationIds.get(flows.pairs.destination[row]);
    if (localOrigin === undefined || localDestination === undefined) continue;
    countByPair.set(`${localOrigin}:${localDestination}`, flows.pairs.count[row]);
  }
  const pairCount = STATION_COUNT * STATION_COUNT;
  const origins = new Uint32Array(pairCount);
  const destinations = new Uint32Array(pairCount);
  const pairFlows = new Float32Array(pairCount);
  const distancesKm = new Float32Array(pairCount);
  const pairState = new Uint32Array(pairCount);
  let observedPairCount = 0;
  let missingPairCount = 0;
  for (let origin = 0; origin < STATION_COUNT; origin++) {
    for (let destination = 0; destination < STATION_COUNT; destination++) {
      const pair = origin * STATION_COUNT + destination;
      origins[pair] = origin;
      destinations[pair] = destination;
      const originLongitude = flows.lngLat[stationIds[origin] * 2];
      const originLatitude = flows.lngLat[stationIds[origin] * 2 + 1];
      const destinationLongitude = flows.lngLat[stationIds[destination] * 2];
      const destinationLatitude = flows.lngLat[stationIds[destination] * 2 + 1];
      distancesKm[pair] =
        haversineMeters(
          [originLongitude, originLatitude],
          [destinationLongitude, destinationLatitude]
        ) / 1000;
      if (origin === destination) {
        pairState[pair] = 0;
      } else {
        const observedCount = countByPair.get(`${origin}:${destination}`);
        if (observedCount === undefined) {
          pairState[pair] = 2;
          missingPairCount++;
        } else {
          pairState[pair] = 1;
          pairFlows[pair] = observedCount;
          observedPairCount++;
        }
      }
    }
  }
  return {
    stationIds: Uint32Array.from(stationIds),
    stationNames: stationIds.map(station => flows.names[station]),
    positions,
    geographicPositions,
    activity,
    candidateIds,
    candidatePositions,
    origins,
    destinations,
    flows: pairFlows,
    distancesKm,
    pairState,
    observedPairCount,
    missingPairCount
  };
}

/** Coordinate frame shared with existing Montreal BIXI scenes. */
export const BIXI_MODEL_ORIGIN = MONTREAL_ORIGIN;

/** Pick a real station in screen space using its original geographic coordinate. */
export function findBixiStationAtPixel(
  data: BixiModelData,
  viewport: Viewport,
  pixel: readonly number[]
): number {
  let nearestStation = -1;
  let nearestSquaredDistance = 15 * 15;
  for (let station = 0; station < data.stationIds.length; station++) {
    const projected = viewport.project([
      data.geographicPositions[station * 2],
      data.geographicPositions[station * 2 + 1]
    ]);
    const squaredDistance = (projected[0] - pixel[0]) ** 2 + (projected[1] - pixel[1]) ** 2;
    if (squaredDistance < nearestSquaredDistance) {
      nearestSquaredDistance = squaredDistance;
      nearestStation = station;
    }
  }
  return nearestStation;
}
