// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Chicago home-to-work flows between the 226 census tracts, the zones of the flow models. Tract
// totals are summed on the CPU from the flow table, the way a flow model's inputs are prepared.

import {COMMUTE_FLOWS, getCommutePositions} from '../../../../data/commute-flows';
import {createPointIndex} from '../../../pick';
import {fitPositions} from '../../network-routing-access/shared/graphs';

export type TractFlows = ReturnType<typeof createTractFlows>;

export function createTractFlows(world: {width: number; height: number}) {
  const positions = fitPositions(getCommutePositions(world), world);
  const zoneCount = positions.length / 2;
  const origins = Uint32Array.from(COMMUTE_FLOWS, ([origin]) => origin);
  const destinations = Uint32Array.from(COMMUTE_FLOWS, ([, destination]) => destination);
  const jobs = Float32Array.from(COMMUTE_FLOWS, ([, , count]) => count);
  // Every flow as a segment, for the muted context lines.
  const segments = new Float32Array(COMMUTE_FLOWS.length * 4);
  COMMUTE_FLOWS.forEach(([origin, destination], flow) => {
    segments.set(
      [
        positions[origin * 2],
        positions[origin * 2 + 1],
        positions[destination * 2],
        positions[destination * 2 + 1]
      ],
      flow * 4
    );
  });
  const flowsByOrigin: number[][] = Array.from({length: zoneCount}, () => []);
  COMMUTE_FLOWS.forEach(([origin], flow) => flowsByOrigin[origin].push(flow));
  for (const list of flowsByOrigin) list.sort((a, b) => jobs[b] - jobs[a]);
  return {
    world,
    positions,
    zoneCount,
    origins,
    destinations,
    jobs,
    segments,
    flowsByOrigin,
    /** Trips leaving and populations (workers living plus jobs located) of each tract. */
    outflows: new Float32Array(zoneCount),
    populations: new Float32Array(zoneCount),
    index: createPointIndex(positions, zoneCount)
  };
}

/** Sums the flows of at least `minimumJobs` into tract outflows and populations. */
export function writeTractTotals(flows: TractFlows, minimumJobs: number): void {
  flows.outflows.fill(0);
  flows.populations.fill(0);
  for (let flow = 0; flow < flows.jobs.length; flow++) {
    if (flows.jobs[flow] < minimumJobs) continue;
    flows.outflows[flows.origins[flow]] += flows.jobs[flow];
    flows.populations[flows.origins[flow]] += flows.jobs[flow];
    flows.populations[flows.destinations[flow]] += flows.jobs[flow];
  }
}

/** The tract with outflow nearest a position; tracts without flows are skipped. */
export function findNearestOrigin(flows: TractFlows, x: number, y: number): number {
  let best = -1;
  let bestDistance = Infinity;
  for (let zone = 0; zone < flows.zoneCount; zone++) {
    if (!flows.flowsByOrigin[zone].length) continue;
    const distance = Math.hypot(flows.positions[zone * 2] - x, flows.positions[zone * 2 + 1] - y);
    if (distance < bestDistance) {
      bestDistance = distance;
      best = zone;
    }
  }
  return best;
}

/** The tract with the most flows leaving it, the poster's origin. */
export function findBusiestOrigin(flows: TractFlows): number {
  let best = 0;
  let total = -1;
  flows.flowsByOrigin.forEach((list, zone) => {
    const sum = list.reduce((acc, flow) => acc + flows.jobs[flow], 0);
    if (sum > total) {
      total = sum;
      best = zone;
    }
  });
  return best;
}
