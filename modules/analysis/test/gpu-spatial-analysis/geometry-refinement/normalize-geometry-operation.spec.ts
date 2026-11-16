// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUNormalizeGeometry} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-normalize-geometry';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUNormalizeGeometryOperation} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-normalize-geometry-operation';
import {
  GPUNormalizeGeometryOperation,
  type NormalizeGeometryOperationProps
} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-normalize-geometry-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {getNormalizeGeometryScene} from './normalize-geometry-scene';

for (const leafOnly of [true, false])
  it(`normalizes ${leafOnly ? 'a leaf-only tree without children' : 'an entirely empty tree'} on both operations`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const props: NormalizeGeometryOperationProps = {
      positions: 'positions',
      nodes: 'nodes',
      children: 'children',
      rowCount: leafOnly ? 1 : 0,
      nodeCount: leafOnly ? 1 : 0,
      childCount: 0,
      output: {positions: 'normalized', children: 'ordered', status: 'status'}
    };
    for (const Operation of [GPUNormalizeGeometryOperation, CPUNormalizeGeometryOperation]) {
      const result = await runOperationGraph(device, {
        inputs: {
          positions: {format: 'float32x2', values: new Float32Array(leafOnly ? [1, 2] : [])},
          nodes: {format: 'uint32x4', values: new Uint32Array(leafOnly ? [0, 0, 1, 0] : [])},
          children: {format: 'uint32', values: new Uint32Array(0)}
        },
        operations: [new Operation(props)],
        outputs: ['normalized', 'ordered', 'status']
      });
      // A point is copied with status 0; an empty tree emits no coordinates or status rows.
      expect(Array.from(result['normalized'])).toEqual(leafOnly ? [1, 2] : []);
      expect(result['ordered'].length).toBe(0);
      expect(Array.from(result['status'])).toEqual(leafOnly ? [0] : []);
    }
  });

for (const invalid of [false, true])
  it(`compares every tree-normalization output (${invalid ? 'nonfinite leaf and invalid parent' : 'nested heterogeneous collections'})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const {input} = getNormalizeGeometryScene();
    if (invalid) input.positions[2] = NaN;
    const props: NormalizeGeometryOperationProps = {
      positions: 'positions',
      nodes: 'nodes',
      children: 'children',
      rowCount: input.positions.length / 2,
      nodeCount: input.nodes.length / 4,
      childCount: input.children.length,
      maximumNestingDepth: 8,
      output: {positions: 'normalized', children: 'ordered', status: 'status'}
    };
    const outputs = Object.values(props.output);
    const results = [];
    for (const Operation of [GPUNormalizeGeometryOperation, CPUNormalizeGeometryOperation])
      results.push(
        await runOperationGraph(device, {
          inputs: {
            positions: {format: 'float32x2', values: input.positions},
            nodes: {format: 'uint32x4', values: input.nodes},
            children: {format: 'uint32', values: input.children}
          },
          operations: [new Operation(props)],
          outputs
        })
      );
    // Coordinates are copied and all other fields are integers; tolerance is zero, with
    // NaNs matched explicitly and signed zeros compared numerically.
    for (const port of outputs) {
      expect(results[0][port].length).toBe(results[1][port].length);
      for (let index = 0; index < results[1][port].length; index++) {
        const expected = results[1][port][index];
        const actual = results[0][port][index];
        expect(Number.isNaN(expected) ? Number.isNaN(actual) : actual === expected).toBe(true);
      }
    }
  });

it('normalizes nested collections across several workgroups with independent ordering on repeated encodings', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = getNormalizeGeometryScene();
  const copies = 40;
  const verticesPerCopy = scene.input.positions.length / 2;
  const nodesPerCopy = scene.input.nodes.length / 4;
  const childrenPerCopy = scene.input.children.length;
  const positions = new Float32Array(scene.input.positions.length * copies);
  const expectedPositions = new Float32Array(positions.length);
  const nodes = new Uint32Array((nodesPerCopy * copies + 1) * 4);
  const children = new Uint32Array((childrenPerCopy + 1) * copies);
  const expectedChildren = new Uint32Array(children.length);
  for (let copy = 0; copy < copies; copy++) {
    for (let coordinate = 0; coordinate < scene.input.positions.length; coordinate++) {
      const translation = coordinate % 2 === 0 ? copy * 10 : 0;
      positions[copy * scene.input.positions.length + coordinate] =
        scene.input.positions[coordinate] + translation;
      expectedPositions[copy * scene.input.positions.length + coordinate] =
        scene.expected.positions[coordinate] + translation;
    }
    for (let node = 0; node < nodesPerCopy; node++) {
      const source = node * 4;
      const destination = (copy * nodesPerCopy + node) * 4;
      const kind = scene.input.nodes[source];
      nodes.set(scene.input.nodes.subarray(source, source + 4), destination);
      nodes[destination + 1] +=
        copy * ([0, 2, 3].includes(kind) ? verticesPerCopy : childrenPerCopy);
    }
    for (let child = 0; child < childrenPerCopy; child++) {
      children[copy * childrenPerCopy + child] = scene.input.children[child] + copy * nodesPerCopy;
      expectedChildren[copy * childrenPerCopy + child] =
        scene.expected.children[child] + copy * nodesPerCopy;
    }
    children[copies * childrenPerCopy + copy] = (copy + 1) * nodesPerCopy - 1;
    // Descending canonical coordinate order puts the translated last copy first.
    expectedChildren[copies * childrenPerCopy + copy] = (copies - copy) * nodesPerCopy - 1;
  }
  nodes.set([7, copies * childrenPerCopy, copies, 0], copies * nodesPerCopy * 4);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const input = (
    id: string,
    values: Float32Array | Uint32Array,
    format: 'float32x2' | 'uint32' | 'uint32x4'
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return importGraphBuffer(
      graph,
      id,
      buffer,
      format,
      values.length / (format === 'float32x2' ? 2 : format === 'uint32x4' ? 4 : 1)
    );
  };
  const normalized = createOutputBuffer(device, positions.length);
  buffers.push(normalized);
  const ordered = createOutputBuffer(device, children.length);
  buffers.push(ordered);
  const status = createOutputBuffer(device, nodes.length / 4);
  buffers.push(status);
  graph.add(
    new GPUNormalizeGeometry({
      positions: input('positions', positions, 'float32x2') as GraphDataView<'float32x2'>,
      nodes: input('nodes', nodes, 'uint32x4') as GraphDataView<'uint32x4'>,
      children: input('children', children, 'uint32') as GraphDataView<'uint32'>,
      maximumNestingDepth: 10,
      output: {
        positions: importGraphBuffer(
          graph,
          'normalized',
          normalized,
          'float32x2',
          positions.length / 2
        ),
        children: importGraphBuffer(graph, 'ordered', ordered, 'uint32', children.length),
        status: importGraphBuffer(graph, 'status', status, 'uint32', nodes.length / 4)
      }
    })
  );
  const compiled = graph.compile();
  try {
    for (let encoding = 0; encoding < 5; encoding++) {
      submitGraph(device, compiled, undefined);
      expect(await readFloat32(normalized, positions.length)).toEqual(
        Array.from(expectedPositions)
      );
      expect(await readUint32(ordered, children.length)).toEqual(Array.from(expectedChildren));
      expect(await readUint32(status, nodes.length / 4)).toEqual(Array(nodes.length / 4).fill(0));
    }
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});
