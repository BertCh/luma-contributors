// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUCoverageDissolve} from '../../../src/gpu-spatial-analysis/polygon-coverage-topology/gpu-coverage-dissolve';
import {createOutputOracleFixture} from '../outline-geometry/output-oracle-fixture';

/** Two labeled shells, one with a hole: three rings require fifteen closed vertices. */
export function createDissolveCapacityOracle(device: Device) {
  const fixture = createOutputOracleFixture(device);
  const {graph, column, output} = fixture;
  const contributor = new GPUCoverageDissolve({
    positions: column(
      'input',
      'float32x2',
      new Float32Array([0, 0, 4, 0, 4, 4, 0, 4, 1, 1, 1, 3, 3, 3, 3, 1, 10, 0, 12, 0, 12, 2, 10, 2])
    ),
    ringOffsets: column('rings', 'uint32', new Uint32Array([0, 4, 8, 12])),
    polygonOffsets: column('polygons', 'uint32', new Uint32Array([0, 2, 3])),
    featureOffsets: column('features', 'uint32', new Uint32Array([0, 1, 2])),
    labels: column('labels', 'uint32', new Uint32Array([37, 91])),
    output: {
      positions: output('positions', 'float32x2', 15),
      ringOffsets: output('offsets', 'uint32', 3),
      status: {
        count: output('count', 'uint32', 1),
        requiredCount: output('requiredCount', 'uint32', 1),
        overflow: output('overflow', 'uint32', 1)
      }
    }
  });
  const nodes = contributor.getCommandNodes(graph);
  for (const node of nodes) graph.add(node);
  return {...fixture, nodes};
}

/** Two equal-label squares meet at one corner: two incoming edges carry touching bit 1. */
export function createDissolveTouchOracle(
  device: Device,
  optional: 'both' | 'segmentFlags' | 'touchingSegmentCount' | undefined
) {
  const fixture = createOutputOracleFixture(device);
  const {graph, column, output} = fixture;
  const contributor = new GPUCoverageDissolve({
    positions: column(
      'input',
      'float32x2',
      new Float32Array([0, 0, 2, 0, 2, 2, 0, 2, 2, 2, 4, 2, 4, 4, 2, 4])
    ),
    ringOffsets: column('rings', 'uint32', new Uint32Array([0, 4, 8])),
    polygonOffsets: column('polygons', 'uint32', new Uint32Array([0, 1, 2])),
    featureOffsets: column('features', 'uint32', new Uint32Array([0, 1, 2])),
    labels: column('labels', 'uint32', new Uint32Array([37, 37])),
    output: {
      positions: output('positions', 'float32x2', 10),
      ringOffsets: output('offsets', 'uint32', 3),
      status: {
        count: output('count', 'uint32', 1),
        overflow: output('overflow', 'uint32', 1),
        ...(optional === 'both' || optional === 'touchingSegmentCount'
          ? {touchingSegmentCount: output('touchingSegmentCount', 'uint32', 1)}
          : {})
      },
      ...(optional === 'both' || optional === 'segmentFlags'
        ? {segmentFlags: output('segmentFlags', 'uint32', 8)}
        : {})
    }
  });
  const nodes = contributor.getCommandNodes(graph);
  for (const node of nodes) graph.add(node);
  return {...fixture, nodes};
}
