// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUNormalizeGeometry} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-normalize-geometry';
import {GPUNormalizeGeometryOperation} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-normalize-geometry-operation';
import {CPUNormalizeGeometryOperation} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-normalize-geometry-operation';
import {GPUOffsetCurveComponents} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-offset-curve-components';
import {createOutputOracleFixture} from '../cell-topology/output-oracle-test-utils';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

// A finite point leaf succeeds (0), a NaN leaf is non-finite (1), and an
// unknown type 8 is invalid (2). No traversal or canonical-order oracle needed.
it('normalization status independently classifies finite, nonfinite and invalid leaves', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createOutputOracleFixture(device);
  const {graph, input, output} = fixture;
  const positions = input('positions', 'float32x2', Float32Array.of(1, 2));
  const nodes = input('nodes', 'uint32x4', Uint32Array.of(0, 0, 1, 0));
  const status = output('status', 'uint32', 1);
  graph.add(
    new GPUNormalizeGeometry({
      positions,
      nodes,
      children: input('children', 'uint32', new Uint32Array(0)),
      output: {
        positions: output('normalized', 'float32x2', 1),
        children: output('ordered', 'uint32', 0),
        status
      }
    })
  );
  const compiled = graph.compile();
  try {
    for (const frame of [
      {point: [1, 2], kind: 0, status: 0},
      {point: [NaN, 2], kind: 0, status: 1},
      {point: [1, 2], kind: 8, status: 2},
      {point: [1, 2], kind: 0, status: 0}
    ]) {
      fixture.writeInput(positions, Float32Array.from(frame.point));
      fixture.writeInput(nodes, Uint32Array.of(frame.kind, 0, 1, 0));
      expect(await fixture.submitTwice(compiled, () => fixture.read(status))).toEqual([
        frame.status
      ]);
      for (const Operation of [GPUNormalizeGeometryOperation, CPUNormalizeGeometryOperation]) {
        const result = await runOperationGraph(device, {
          inputs: {
            positions: {format: 'float32x2', values: Float32Array.from(frame.point)},
            nodes: {format: 'uint32x4', values: Uint32Array.of(frame.kind, 0, 1, 0)},
            children: {format: 'uint32', values: new Uint32Array(0)}
          },
          operations: [
            new Operation({
              positions: 'positions',
              nodes: 'nodes',
              children: 'children',
              rowCount: 1,
              nodeCount: 1,
              childCount: 0,
              output: {positions: 'normalized', children: 'ordered', status: 'status'}
            })
          ],
          outputs: ['status']
        });
        expect(Array.from(result['status'])).toEqual([frame.status]);
      }
    }
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});

// A horizontal segment offset upward by 1 is one component of two vertices.
// Vertex capacity 1 cannot publish a complete component. All eight words,
// including required counts and reserved zeros, have independent literal values.
for (const capacity of [1, 2]) {
  it(`offset component status observes all eight words (vertex capacity ${capacity})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createOutputOracleFixture(device);
    const {graph, input, output} = fixture;
    const status = output('status', 'uint32x4', 2);
    graph.add(
      new GPUOffsetCurveComponents({
        original: {
          positions: input('original', 'float32x2', Float32Array.of(0, 0, 2, 0)),
          lineOffsets: input('original-offsets', 'uint32', Uint32Array.of(0, 2))
        },
        raw: {
          positions: input('raw', 'float32x2', Float32Array.of(0, 1, 2, 1)),
          lineOffsets: input('raw-offsets', 'uint32', Uint32Array.of(0, 2))
        },
        distance: 1,
        tolerance: 0,
        segmentCapacity: 4,
        intersectionCapacity: 4,
        output: {
          positions: output('positions', 'float32x2', capacity),
          sourceEdges: output('edges', 'uint32', capacity),
          components: output('components', 'uint32x4', 1),
          status
        }
      })
    );
    const compiled = graph.compile();
    try {
      expect(await fixture.submitTwice(compiled, () => fixture.read(status))).toEqual(
        capacity === 1 ? [0, 0, 1, 2, 1, 0, 0, 0] : [1, 2, 1, 2, 0, 0, 0, 0]
      );
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  });
}
