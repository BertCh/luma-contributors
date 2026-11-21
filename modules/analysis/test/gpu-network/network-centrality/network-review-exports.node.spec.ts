// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import * as routes from '../../../src/gpu-network/alternative-routes';
import * as optimization from '../../../src/gpu-network/network-optimization';
import * as structure from '../../../src/gpu-network/network-structure';
it('exports the route readers and solver feasibility/diagnostic helpers from family entries', () => {
  const entries: Record<string, unknown> = {...routes, ...optimization, ...structure};
  for (const name of [
    'getGPUAlternativeRoutesOutputLayout',
    'decodeGPUAlternativeRoutes',
    'decodeNetworkLouvainStatus',
    'getNetworkFacilitySearchLimits'
  ])
    expect(typeof entries[name], name).toBe('function');
  for (const [name, value] of Object.entries({
    GPU_NETWORK_FACILITY_OPTIMIZATION_MAXIMUM_FACILITY_COUNT: 16,
    GPU_NETWORK_FACILITY_OPTIMIZATION_MAXIMUM_DEMAND_COUNT: 64,
    GPU_NETWORK_FACILITY_OPTIMIZATION_MAXIMUM_CANDIDATE_COUNT: 1048576,
    GPU_NETWORK_STRUCTURE_EXACT_MAXIMUM_NODE_COUNT: 1024,
    GPU_NETWORK_STRUCTURE_EXACT_MAXIMUM_EDGE_COUNT: 65536
  }))
    expect(entries[name], name).toBe(value);
});
