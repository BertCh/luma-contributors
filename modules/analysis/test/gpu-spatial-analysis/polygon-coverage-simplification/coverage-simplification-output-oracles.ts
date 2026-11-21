// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUCoverageSimplification} from '../../../src/gpu-spatial-analysis/polygon-coverage-simplification/gpu-coverage-simplification';
import {createOutputOracleFixture} from '../outline-geometry/output-oracle-fixture';

/** The bow-tie's two diagonal edges intersect at (2,2); no vertex is omitted at tolerance 0.
 * One repair round cannot insert new vertices, so initial/remaining/restored/overflow = [1,1,0,0]. */
export function createUnrepairableCrossingOracle(device: Device) {
  const fixture = createOutputOracleFixture(device);
  const {graph, column, output} = fixture;
  const contributor = new GPUCoverageSimplification({
    positions: column('input', 'float32x2', new Float32Array([0, 0, 4, 0, 0, 4, 4, 4])),
    ringOffsets: column('rings', 'uint32', new Uint32Array([0, 4])),
    polygonOffsets: column('polygons', 'uint32', new Uint32Array([0, 1])),
    featureOffsets: column('features', 'uint32', new Uint32Array([0, 1])),
    parameters: column('parameters', 'float32', new Float32Array(4)),
    maximumIterations: 32,
    topologyIterationCount: 1,
    output: {
      positions: output('positions', 'float32x2', 4),
      ringOffsets: output('offsets', 'uint32', 2),
      topologyStatistics: output('statistics', 'uint32', 4),
      status: {overflow: output('overflow', 'uint32', 1)}
    }
  });
  const nodes = contributor.getCommandNodes(graph);
  for (const node of nodes) graph.add(node);
  return {...fixture, nodes};
}
