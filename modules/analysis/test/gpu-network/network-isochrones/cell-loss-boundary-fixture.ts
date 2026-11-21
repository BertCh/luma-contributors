// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUNetworkIsochrones} from '../../../src/gpu-network/network-isochrones/index';
import {CELL_SQUARE} from './isochrones-output-oracle-fixtures';
import type {IsochroneOracleHarness} from './isochrones-output-oracle-utils';

/** Inject loss at the aggregation/outline boundary using the actual isochrone declaration. */
export function createCellLossBoundaryFixture(
  harness: Pick<IsochroneOracleHarness, 'graph' | 'input' | 'output'>,
  candidateOnly: boolean,
  sharedDestination?: 'count' | 'candidateOverflow'
) {
  const table = {
    cells: harness.input('uint32x2', CELL_SQUARE.key),
    counts: harness.input('uint32', [1]),
    status: {
      count: harness.input('uint32', [1]),
      overflow: harness.input('uint32', [candidateOnly ? 0 : 1]),
      candidateOverflow: harness.input('uint32', [candidateOnly ? 1 : 0])
    }
  };
  const outlineStatus = {
    count: harness.output('outlineCount', 'uint32', 1),
    overflow: harness.output('outlineOverflow', 'uint32', 1),
    candidateOverflow: harness.output('outlineCandidateOverflow', 'uint32', 1)
  };
  const sharedStatusBacking = sharedDestination
    ? harness.output('sharedStatus', 'uint32', 128)
    : undefined;
  if (sharedStatusBacking && sharedDestination) {
    table.status.overflow = harness.graph.createDataView(sharedStatusBacking.buffer, {
      format: 'uint32',
      length: 1,
      byteOffset: 0
    });
    outlineStatus[sharedDestination] = harness.graph.createDataView(sharedStatusBacking.buffer, {
      format: 'uint32',
      length: 1,
      byteOffset: 256
    });
  }
  const ringStatus = {
    count: harness.output('ringCount', 'uint32', 1),
    overflow: harness.output('ringOverflow', 'uint32', 1),
    candidateOverflow: harness.output('ringCandidateOverflow', 'uint32', 1),
    complete: harness.output('ringComplete', 'uint32', 1)
  };
  const contributor = new GPUNetworkIsochrones({
    topology: {
      offsets: harness.input('uint32', [0, 1]),
      neighbors: harness.input('uint32', [0]),
      weights: harness.input('float32', [0]),
      nodePositions: harness.input('float32x2', [-70, 20])
    },
    costs: harness.input('float32', [0]),
    costBreaks: harness.input('float32', [1]),
    parameters: harness.input('float32', [1, -180, -80, 180, 80, 0, 0, 0, 0, 1, 0, 0]),
    cellOutline: {
      family: 'quadbin',
      resolution: 2,
      table,
      output: {
        rows: harness.output('rows', 'uint32', 4),
        cells: harness.output('cells', 'uint32x2', 4),
        edgeIndices: harness.output('edges', 'uint32', 4),
        endpoints: harness.output('endpoints', 'float32x4', 4),
        status: outlineStatus
      },
      rings: {
        output: {
          ringOffsets: harness.output('ringOffsets', 'uint32', 2),
          positions: harness.output('ringPositions', 'float32x2', 5),
          status: ringStatus
        }
      }
    }
  });
  // Keep only the consumer boundary so injected producer evidence is not overwritten by aggregation.
  const nodes = contributor
    .getCommandNodes(harness.graph)
    .filter(node => node.id.startsWith('network-isochrones-outline'));
  return {table, outlineStatus, ringStatus, nodes, sharedStatusBacking};
}
