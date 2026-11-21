// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUGroupConvexHull,
  type GPUGroupConvexHullOutput
} from '../../../src/gpu-spatial-analysis/group-geometry/gpu-group-convex-hull';
import {GPUMinimumBounds} from '../../../src/gpu-spatial-analysis/minimum-bounds/gpu-minimum-bounds';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

function createStatusFixture(withDetails: boolean) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let sequence = 0;
  const view = <Format extends 'uint32' | 'float32x2' | 'float32x4'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `hull-status-${sequence++}`, format, length);
  const positions = view('float32x2', 4);
  const labels = view('uint32', 4);
  const status: GPUGroupConvexHullOutput['status'] = {
    overflow: view('uint32', 1),
    ...(withDetails ? {groupOverflow: view('uint32', 1), vertexOverflow: view('uint32', 1)} : {})
  };
  const props = {
    id: 'status-hull',
    positions,
    labels,
    groupCount: 1,
    maximumVerticesPerGroup: 3,
    vertexCapacity: 3,
    output: {
      vertexIndices: view('uint32', 3),
      offsets: view('uint32', 2),
      counts: view('uint32', 1),
      status
    }
  };
  return {device, graph, view, props};
}

for (const withDetails of [false, true]) {
  it(`publishes boolean hull overflow and independent named flags (details=${withDetails})`, () => {
    const fixture = createStatusFixture(withDetails);
    try {
      const nodes = new GPUGroupConvexHull(fixture.props).getCommandNodes(fixture.graph);
      const publish = nodes.find(node => node.id === 'status-hull-publish-status');
      expect(publish, 'status must be normalized after all capacity stages').toBeDefined();
      // The first publication input is the internal atomic bitset. Exercise the real WGSL
      // independently of hull construction, including simultaneous limits and the next empty frame.
      const input = publish!.resources![0];
      if (!('buffer' in input)) throw new Error('Hull status publication needs a buffer input');
      const flags = input.buffer as GraphDataView<'uint32'>;
      const memory = createCPUGraphMemory();
      const {status} = fixture.props.output;
      for (const [bits, overflow, groupOverflow, vertexOverflow] of [
        [0, 0, 0, 0],
        [1, 1, 1, 0],
        [2, 1, 0, 1],
        [3, 1, 1, 1],
        [0, 0, 0, 0]
      ]) {
        memory.write(flags, new Uint32Array([bits]));
        for (const scalar of Object.values(status))
          memory.write(scalar, new Uint32Array([0xffffffff]));
        executeCommandNodesOnCPU([publish!], memory);
        expect(memory.readUint32(status.overflow)[0]).toBe(overflow);
        if (status.groupOverflow)
          expect(memory.readUint32(status.groupOverflow)[0]).toBe(groupOverflow);
        if (status.vertexOverflow)
          expect(memory.readUint32(status.vertexOverflow)[0]).toBe(vertexOverflow);
      }
      for (const source of getGPUCommandNodeKernelSources([publish!]))
        expect(auditWGSLModule(source.source, source.constants)).toEqual([]);
    } finally {
      fixture.device.destroy();
    }
  });
}

it('validates each optional diagnostic, aliasing and graph membership', () => {
  const fixture = createStatusFixture(true);
  const foreign = createStatusFixture(false);
  try {
    const {props, graph, view} = fixture;
    for (const port of ['groupOverflow', 'vertexOverflow'] as const) {
      expect(
        () =>
          new GPUGroupConvexHull({
            ...props,
            output: {
              ...props.output,
              status: {...props.output.status, [port]: view('float32x4', 1)}
            }
          })
      ).toThrow(new RegExp(`status.${port}`));
      expect(
        () =>
          new GPUGroupConvexHull({
            ...props,
            output: {...props.output, status: {...props.output.status, [port]: view('uint32', 0)}}
          })
      ).toThrow(new RegExp(`status.${port}.*1`));
      expect(
        () =>
          new GPUGroupConvexHull({
            ...props,
            output: {...props.output, status: {...props.output.status, [port]: props.labels}}
          })
      ).toThrow(/share buffers/);
      expect(() =>
        new GPUGroupConvexHull({
          ...props,
          output: {
            ...props.output,
            status: {...props.output.status, [port]: foreign.view('uint32', 1)}
          }
        }).getCommandNodes(graph)
      ).toThrow(/graph/);
    }
    expect(
      () =>
        new GPUGroupConvexHull({
          ...props,
          output: {
            ...props.output,
            status: {...props.output.status, groupOverflow: props.output.counts}
          }
        })
    ).toThrow(/status.groupOverflow must not overlap output.counts/);
    expect(
      () =>
        new GPUGroupConvexHull({
          ...props,
          output: {
            ...props.output,
            status: {...props.output.status, vertexOverflow: props.output.status.overflow}
          }
        })
    ).toThrow(/must not overlap/);
    const boundsOutput = {
      sourceBounds: view('float32x4', 1),
      status: {overflow: view('uint32', 1), groupOverflow: props.labels}
    };
    expect(
      () =>
        new GPUMinimumBounds({
          id: 'status-bounds',
          positions: props.positions,
          labels: props.labels,
          groupCount: 1,
          output: boundsOutput
        })
    ).toThrow(/share buffers/);
    expect(
      () =>
        new GPUMinimumBounds({
          positions: props.positions,
          labels: props.labels,
          groupCount: 1,
          output: {
            ...boundsOutput,
            status: {...boundsOutput.status, groupOverflow: view('uint32', 0)}
          }
        })
    ).toThrow(/status.groupOverflow.*1/);
    const shared = view('uint32', 1);
    expect(
      () =>
        new GPUMinimumBounds({
          positions: props.positions,
          labels: props.labels,
          groupCount: 1,
          output: {
            sourceBounds: view('float32x4', 1),
            hullSizes: shared,
            status: {overflow: view('uint32', 1), groupOverflow: shared}
          }
        })
    ).toThrow(/status.groupOverflow must not overlap output.hullSizes/);
  } finally {
    fixture.device.destroy();
    foreign.device.destroy();
  }
});
