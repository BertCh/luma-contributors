// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GraphVectorView} from '@luma.gl/graph/gpu-core';
import {afterAll, expect, it, vi} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {
  GPUNaturalNeighbor,
  type GPUNaturalNeighborProps
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-natural-neighbor';
import {expectValidQueryShaderLiterals} from '../linear-referencing/gpu-linear-referencing-chunks.test-utils';
import {WGSL_RESERVED_WORDS} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  createNullWebGPUDevice,
  createVectorView,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';

type ChunkedProps = GPUNaturalNeighborProps<
  GraphVectorView<'float32x2'>,
  GraphVectorView<'float32x4'>,
  GraphVectorView<'uint32'>,
  GraphVectorView<'float32' | 'float32x2'>,
  GraphVectorView<'uint32'>
>;

const device = createNullWebGPUDevice();
afterAll(() => device.destroy());
function createProps(graph: GPUCommandGraph, lengths: number[], indexed: boolean): ChunkedProps {
  const rows = <Format extends 'float32x2' | 'float32x4' | 'float32' | 'uint32'>(
    name: string,
    format: Format,
    width = 1
  ) =>
    createVectorView(
      name,
      format,
      lengths.map((length, index) => {
        const backing = createTransientView(graph, `${name}-${index}`, format, length * width + 64);
        return graph.createDataView(backing.buffer, {
          format,
          length: length * width,
          byteOffset: ((index % 3) + 1) * 4
        });
      })
    );
  return {
    positions: createTransientView(graph, 'positions', 'float32x2', 3),
    values: createTransientView(graph, 'values', 'float32', 3),
    queries: rows('queries', 'float32x2'),
    maximumNeighborCount: 3,
    triangles: indexed ? createTransientView(graph, 'triangles', 'uint32', 6) : undefined,
    queryTriangles: indexed ? rows('seeds', 'uint32') : undefined,
    output: {
      values: rows('results', 'float32x4'),
      status: rows('statuses', 'uint32'),
      weights: indexed ? rows('weights', 'float32x2', 3) : rows('weights', 'float32', 3)
    }
  };
}

it('validates complete query-major weights, seed topology and all aligned output aliases', () => {
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph, [1, 3, 2], true);
  expect(
    () =>
      new GPUNaturalNeighbor({
        ...props,
        queryTriangles: createTransientView(graph, 'wrong-seeds', 'uint32', 6)
      })
  ).toThrow(/queryTriangles.*topology/);
  expect(
    () =>
      new GPUNaturalNeighbor({
        ...props,
        output: {
          ...props.output,
          weights: createTransientView(graph, 'wrong-weights', 'float32x2', 18)
        }
      })
  ).toThrow(/weights.*topology/);
  const backing = createTransientView(graph, 'alias', 'float32x4', 128);
  const single = createProps(new GPUCommandGraph(device), [1], false);
  expect(
    () =>
      new GPUNaturalNeighbor({
        ...single,
        output: {
          values: graph.createDataView(backing.buffer, {
            format: 'float32x4',
            length: 1,
            byteOffset: 4
          }),
          status: graph.createDataView(backing.buffer, {
            format: 'uint32',
            length: 1,
            byteOffset: 20
          })
        }
      })
  ).toThrow(/aligned binding byte ranges must not overlap/);
});

it('prechecks late weight chunks before declaring resources or nodes', () => {
  const graph = new GPUCommandGraph(device);
  const contributor = new GPUNaturalNeighbor(createProps(graph, [1, 6_000_000], true));
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const createView = vi.spyOn(graph, 'createDataView');
  const addNode = vi.spyOn(graph, 'add');
  expect(() => contributor.getCommandNodes(graph)).toThrow(/weights chunk 1 needs 144000008 bytes/);
  expect(allocate).not.toHaveBeenCalled();
  expect(createView).not.toHaveBeenCalled();
  expect(addNode).not.toHaveBeenCalled();
});

for (const indexed of [false, true]) {
  it(`accepts 30 million queries with packed resident samples and triangulation (indexed=${indexed})`, () => {
    const graph = new GPUCommandGraph(device);
    const contributor = new GPUNaturalNeighbor(
      createProps(graph, Array(10).fill(3_000_000), indexed)
    );
    expect(contributor.getCommandNodes(graph)).toHaveLength(10);
    expect(contributor.estimatedCommandNodeCount).toBe(10);
  });
  it(`keeps one shader/pipeline and exact chunk workgroup counts (indexed=${indexed})`, () => {
    const pipelines = new Set<string>();
    for (const count of [1, 3, 100]) {
      const graph = new GPUCommandGraph(device);
      const lengths = Array.from({length: count}, (_, index) => [1, 7, 13][index % 3]);
      const nodes = new GPUNaturalNeighbor(createProps(graph, lengths, indexed)).getCommandNodes(
        graph
      );
      expect(() => graph.add(nodes)).not.toThrow();
      expect(nodes.map(node => node.workload?.maximumWorkgroupCount)).toEqual(
        lengths.map(length => Math.ceil(length / 2))
      );
      for (const key of getGPUCommandNodePipelineKeys(nodes)) pipelines.add(key);
      for (const {source, constants} of getGPUCommandNodeKernelSources(nodes)) {
        expect(constants).toEqual({});
        expect(new WgslReflect(source).storage.length).toBeLessThanOrEqual(8);
        expectValidQueryShaderLiterals(source);
        const identifiers =
          source.replace(/\/\/[^\n]*|\/\*[\s\S]*?\*\//g, '').match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? [];
        expect(identifiers.filter(word => WGSL_RESERVED_WORDS.has(word))).toEqual([]);
      }
    }
    expect(pipelines.size).toBe(1);
  });
}
