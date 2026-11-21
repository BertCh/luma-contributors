// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {captureGraphCommandNodes} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {createGeometryFixture} from '../outline-geometry/geometry-fixture';
import {createReviewCases, resolveReviewInputViews} from './multipart-point-review-fixture';
import {GPUPointInPolygonJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-point-in-polygon-join';

// Expectations are exact binary arithmetic or Shapely closed intersections; see the adjacent
// oracle. Deliberate NaNs are checked by toEqual; finite expectations reject unexpected NaN.
for (const entry of createReviewCases()) {
  it(`WebGPU review ${entry.name} contributor and mixed operation forms`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const operation = entry.gpu();
    const outputs = Object.fromEntries(
      Object.values(operation.outputs).map(definition => [
        definition.value,
        {format: definition.format as 'float32' | 'uint32', length: definition.length}
      ])
    );
    const fixture = createGeometryFixture(device, {
      inputs: entry.gpuInputs ?? entry.inputs,
      outputs,
      create: ({inputs, outputs: views}) => {
        for (const view of Object.values(views))
          (view.buffer.defaultBuffer as Buffer).write(
            new Uint32Array(view.length).fill(0xdeadbeef),
            view.byteOffset
          );
        return {
          getCommandNodes: graph => {
            resolveReviewInputViews(entry, graph, inputs);
            return captureGraphCommandNodes(graph, () =>
              graph.add(
                operation.createContributor(
                  port =>
                    (views[operation.outputs[port]?.value] ??
                      inputs[operation.inputs[port]]) as never
                )
              )
            );
          }
        };
      }
    });
    try {
      const result = await fixture.run();
      for (const [port, expected] of Object.entries(entry.expected))
        expect(result[port], port).toEqual(expected);
    } finally {
      fixture.destroy();
      operation.destroy();
    }
    // These packed mixed operations exercise actual backend crossings as well as direct binding.
    for (const mixed of [entry.gpu(), entry.cpu()]) {
      const result = await runOperationGraph(device, {
        inputs: entry.inputs,
        operations: [mixed],
        outputs: Object.keys(entry.expected)
      });
      for (const [port, expected] of Object.entries(entry.expected))
        expect(Array.from(result[port]), port).toEqual(expected);
    }
  });
}
it('WebGPU review R10 rejects disjoint read/write views of the same six-word buffer', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32x2' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `alias-${serial++}`, format, length);
  const shared = view('uint32', 6);
  const props = {
    points: {
      kind: 'points' as const,
      positions: view('float32x2', 2),
      featureOffsets: view('uint32', 3)
    },
    polygons: {
      kind: 'polygons' as const,
      positions: view('float32x2', 8),
      featureOffsets: view('uint32', 3),
      polygonOffsets: view('uint32', 3),
      ringOffsets: view('uint32', 3)
    },
    candidateCapacity: 8,
    output: {
      pointFeatureIds: view('uint32', 2),
      featurePolygonMembership: graph.createDataView(shared.buffer, {
        format: 'uint32' as const,
        length: 4,
        byteOffset: 0
      }),
      polygonMembershipCounts: graph.createDataView(shared.buffer, {
        format: 'uint32' as const,
        length: 2,
        byteOffset: 16
      }),
      status: {overflow: view('uint32', 1)}
    }
  };
  expect(() => new GPUPointInPolygonJoin(props)).toThrow(/read|writ/);
  expect(() =>
    new GPUPointInPolygonJoin({
      ...props,
      output: {...props.output, polygonMembershipCounts: view('uint32', 2)}
    }).getCommandNodes(graph)
  ).not.toThrow();
  // The R8 cases above submit this separate-buffer combination and check every numeric result.
});
