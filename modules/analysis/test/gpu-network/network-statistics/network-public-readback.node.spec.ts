// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import * as network from '../../../src/gpu-network/index';

it('exports and executes all six checked readers through gpu-network', () => {
  const words = new Uint32Array(19);
  words[0] = 65537;
  const bytes = new Uint8Array(words.buffer);
  expect(network.decodeGPUNetworkStatisticsWords).toBeTypeOf('function');
  expect(network.decodeGPUNetworkStatisticsFromBytes).toBeTypeOf('function');
  expect(network.decodeGPUNetworkCoarseningSummaryWords).toBeTypeOf('function');
  expect(network.decodeGPUNetworkCoarseningSummaryFromBytes).toBeTypeOf('function');
  expect(network.decodeGPUNetworkSubgraphFilterCountsWords).toBeTypeOf('function');
  expect(network.decodeGPUNetworkSubgraphFilterCountsFromBytes).toBeTypeOf('function');
  expect(network.MAXIMUM_GPU_NETWORK_STATISTICS_DEGREE_BIN_COUNT).toBe(4096);
  expect(network.decodeGPUNetworkStatisticsWords(words).liveNodeCount).toBe(65537);
  expect(network.decodeGPUNetworkStatisticsFromBytes({bytes}).liveNodeCount).toBe(65537);
  expect(network.decodeGPUNetworkCoarseningSummaryWords(words.subarray(0, 8)).liveNodeCount).toBe(
    65537
  );
  expect(
    network.decodeGPUNetworkCoarseningSummaryFromBytes(bytes.subarray(0, 32)).liveNodeCount
  ).toBe(65537);
  expect(
    network.decodeGPUNetworkSubgraphFilterCountsWords(words.subarray(0, 4)).liveNodeCount
  ).toBe(65537);
  expect(
    network.decodeGPUNetworkSubgraphFilterCountsFromBytes(bytes.subarray(0, 16)).liveNodeCount
  ).toBe(65537);
});
