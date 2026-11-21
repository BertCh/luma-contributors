// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUOffsetCurve} from '../../../src/gpu-spatial-analysis/outline-geometry/gpu-offset-curve';
import {createOutputOracleFixture} from './output-oracle-fixture';

export function createOffsetOutputOracle(
  device: Device,
  optional?: 'vertexCount' | 'requiredVertexCount'
) {
  const fixture = createOutputOracleFixture(device);
  const {column, output, graph} = fixture;
  const positions = column('input', 'float32x2', new Float32Array([0, 0, 4, 0]));
  const lineOffsets = column('lines', 'uint32', new Uint32Array([0, 2]));
  const parameters = column('parameters', 'float32', new Float32Array([1, 5, 0, 0]));
  const status = {
    count: output('count', 'uint32', 1),
    overflow: output('overflow', 'uint32', 1),
    ...(optional ? {[optional]: output(optional, 'uint32', 1)} : {})
  };
  const contributor = new GPUOffsetCurve({
    positions,
    lineOffsets,
    parameters,
    output: {
      layout: 'compact',
      positions: output('positions', 'float32x2', 1),
      lineOffsets: output('offsets', 'uint32', 2),
      status
    }
  });
  const nodes = contributor.getCommandNodes(graph);
  for (const node of nodes) graph.add(node);
  // Left normal of the horizontal segment is (0,1): two points, first retained by capacity 1.
  const expected = {
    positions: [0, 1],
    offsets: [0, 1],
    count: [1],
    overflow: [1],
    ...(optional ? {[optional]: [optional === 'vertexCount' ? 1 : 2]} : {})
  };
  return {...fixture, contributor, nodes, parameters, expected};
}
