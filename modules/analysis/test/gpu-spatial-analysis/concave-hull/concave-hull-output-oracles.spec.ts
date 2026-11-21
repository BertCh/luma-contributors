// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUConcaveHull} from '../../../src/gpu-spatial-analysis/concave-hull/gpu-concave-hull';
import {createOutputOracleFixture} from '../cell-topology/output-oracle-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it('writes the unique 3-4-5 triangle and its three directed boundary edges', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createOutputOracleFixture(device);
  const {output, input, graph} = fixture;
  const triangles = output('triangles', 'uint32x3', 1);
  const triangleValidity = output('triangle-validity', 'uint32', 1);
  const boundaryEdges = output('edges', 'uint32x2', 3);
  const edgeValidity = output('edge-validity', 'uint32', 3);
  const converged = output('converged', 'uint32', 1);
  // Only three noncollinear points: unique CCW triangle 0->1->2, radius 5/2.
  graph.add(
    new GPUConcaveHull({
      positions: input('positions', 'float32x2', new Float32Array([0, 0, 3, 0, 0, 4])),
      maximumCircumradius: 3,
      output: {triangles, triangleValidity, boundaryEdges, edgeValidity, status: {converged}}
    })
  );
  const compiled = graph.compile();
  try {
    const actual = await fixture.submitTwice(compiled, async () => ({
      triangles: await fixture.readRows([triangles]),
      edges: await fixture.readRows([boundaryEdges]),
      validity: await fixture.readRows([triangleValidity]),
      edgeValidity: await fixture.readRows([edgeValidity]),
      converged: await fixture.readRows([converged])
    }));
    // Preserve orientation while allowing any cyclic starting corner.
    const first = actual.triangles.indexOf(0);
    expect([...actual.triangles.slice(first), ...actual.triangles.slice(0, first)]).toEqual([
      0, 1, 2
    ]);
    expect(
      [0, 1, 2]
        .map(row => actual.edges.slice(row * 2, row * 2 + 2))
        .sort((left, right) => left[0] - right[0])
    ).toEqual([
      [0, 1],
      [1, 2],
      [2, 0]
    ]);
    expect(actual.validity).toEqual([1]);
    expect(actual.edgeValidity).toEqual([1, 1, 1]);
    expect(actual.converged).toEqual([1]);
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});
