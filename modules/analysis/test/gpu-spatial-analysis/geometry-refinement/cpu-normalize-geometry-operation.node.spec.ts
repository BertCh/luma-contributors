// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeNormalizeGeometryOnCPU,
  CPUNormalizeGeometryOperation
} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-normalize-geometry-operation';
import {
  GPUNormalizeGeometryOperation,
  type NormalizeGeometryOperationProps
} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-normalize-geometry-operation';
import {getNormalizeGeometryScene} from './normalize-geometry-scene';

it('runs a leaf-only CPU operation and declares an actually empty children output on both operations', async () => {
  const positions = new Float32Array([1, 2]);
  const nodes = new Uint32Array([0, 0, 1, 0]);
  const children = new Uint32Array(0);
  const props: NormalizeGeometryOperationProps = {
    id: 'leaf-only',
    positions: 'positions',
    nodes: 'nodes',
    children: 'children',
    rowCount: 1,
    nodeCount: 1,
    childCount: 0,
    output: {positions: 'normalized', children: 'ordered', status: 'status'}
  };
  expect(computeNormalizeGeometryOnCPU({positions, nodes, children})).toEqual({
    positions,
    children,
    status: new Uint32Array([0])
  });
  const gpuOperation = new GPUNormalizeGeometryOperation(props);
  expect(gpuOperation.outputs['childrenOutput'].length).toBe(0);
  expect(gpuOperation.contract.outputs).toMatchObject({ordered: {length: 0}});
  const cpuOperation = new CPUNormalizeGeometryOperation(props);
  expect(cpuOperation.outputs['ordered'].length).toBe(0);
  const result = await cpuOperation.execute({
    inputs: {
      positions: {
        format: 'float32x2',
        length: 1,
        chunks: [{values: positions, length: 1}],
        pack: () => positions
      },
      nodes: {
        format: 'uint32x4',
        length: 1,
        chunks: [{values: nodes, length: 1}],
        pack: () => nodes
      },
      children: {
        format: 'uint32',
        length: 0,
        chunks: [{values: children, length: 0}],
        pack: () => children
      }
    }
  });
  expect(result).toEqual({
    normalized: new Float32Array([1, 2]),
    ordered: new Uint32Array(0),
    status: new Uint32Array([0])
  });
  for (const Operation of [GPUNormalizeGeometryOperation, CPUNormalizeGeometryOperation]) {
    expect(() => new Operation({...props, childCount: -1})).toThrow(/^leaf-only/);
    expect(
      () => new Operation({...props, output: {...props.output, children: 'children'}})
    ).toThrow(/^leaf-only/);
  }
});

it('matches a hand-normalized tree and preserves node identity, shell roles and nested comparisons', () => {
  const {input, expected} = getNormalizeGeometryScene();
  const originalNodes = input.nodes.slice();
  const result = computeNormalizeGeometryOnCPU(input);
  expect(result).toEqual(expected);
  expect(input.nodes).toEqual(originalNodes);
  expect(computeNormalizeGeometryOnCPU({...input, ...result})).toEqual(result);
});

it('preserves entirely empty tree vectors without padding', async () => {
  const input = {
    positions: new Float32Array(0),
    nodes: new Uint32Array(0),
    children: new Uint32Array(0)
  };
  const props: NormalizeGeometryOperationProps = {
    positions: 'positions',
    nodes: 'nodes',
    children: 'children',
    rowCount: 0,
    nodeCount: 0,
    childCount: 0,
    output: {positions: 'normalized', children: 'ordered', status: 'status'}
  };
  const gpuOperation = new GPUNormalizeGeometryOperation(props);
  for (const port of ['positionsOutput', 'childrenOutput', 'status'])
    expect(gpuOperation.outputs[port].length).toBe(0);
  const cpuOperation = new CPUNormalizeGeometryOperation(props);
  expect(computeNormalizeGeometryOnCPU(input)).toEqual({
    positions: new Float32Array(0),
    children: new Uint32Array(0),
    status: new Uint32Array(0)
  });
  expect(
    await cpuOperation.execute({
      inputs: {
        positions: {
          format: 'float32x2',
          length: 0,
          chunks: [{values: input.positions, length: 0}],
          pack: () => input.positions
        },
        nodes: {
          format: 'uint32x4',
          length: 0,
          chunks: [{values: input.nodes, length: 0}],
          pack: () => input.nodes
        },
        children: {
          format: 'uint32',
          length: 0,
          chunks: [{values: input.children, length: 0}],
          pack: () => input.children
        }
      }
    })
  ).toEqual({
    normalized: new Float32Array(0),
    ordered: new Uint32Array(0),
    status: new Uint32Array(0)
  });
});

it('keeps the first repeated minimum, zero-winding direction, empty leaves and signed zeros', () => {
  const positions = new Float32Array([0, 0, 2, 0, 0, 0, 1, 0, -0, -0]);
  const result = computeNormalizeGeometryOnCPU({
    positions,
    nodes: new Uint32Array([3, 0, 4, 0, 0, 4, 1, 1, 2, 5, 0, 2]),
    children: new Uint32Array(0)
  });
  expect(result.positions).toEqual(positions);
  expect(Object.is(result.positions[8], -0)).toBe(true);
  expect(Array.from(result.status)).toEqual([0, 0, 0]);
});

it('returns node status for bad kinds, leaf ranges, child references and nonfinite leaves', () => {
  const result = computeNormalizeGeometryOnCPU({
    positions: new Float32Array([0, 0, NaN, 1]),
    nodes: new Uint32Array([
      0, 100, 1, 0, 2, 0, 2, 1, 99, 0, 0, 2, 7, 0, 2, 3, 7, 999, 1, 4, 7, 2, 1, 5
    ]),
    children: new Uint32Array([0, 1, 5])
  });
  expect(Array.from(result.status)).toEqual([2, 1, 2, 2, 2, 2]);
  expect(Number.isNaN(result.positions[2])).toBe(true);
  expect(Array.from(result.children)).toEqual([0, 1, 5]);
});

it('normalizes trees deeper than the GPU comparison stack', () => {
  const nodes = new Uint32Array(101 * 4);
  const children = new Uint32Array(100);
  nodes.set([0, 0, 1, 0]);
  for (let node = 1; node <= 100; node++) {
    nodes.set([7, node - 1, 1, node], node * 4);
    children[node - 1] = node - 1;
  }
  const result = computeNormalizeGeometryOnCPU({
    positions: new Float32Array([1, 2]),
    nodes,
    children
  });
  expect(result.status.every(status => status === 0)).toBe(true);
  expect(result.children).toEqual(children);
});

it('rejects malformed packing and overlapping ranges and checks both operation contracts', () => {
  expect(() =>
    computeNormalizeGeometryOnCPU({
      positions: new Float32Array(1),
      nodes: new Uint32Array(4),
      children: new Uint32Array(0)
    })
  ).toThrow(/positions/);
  expect(() =>
    computeNormalizeGeometryOnCPU({
      positions: new Float32Array(2),
      nodes: new Uint32Array(3),
      children: new Uint32Array(0)
    })
  ).toThrow(/nodes/);
  expect(() =>
    computeNormalizeGeometryOnCPU({
      positions: new Float32Array(2),
      nodes: new Uint32Array([0, 0, 1, 0, 0, 0, 1, 1]),
      children: new Uint32Array(0)
    })
  ).toThrow(/nodes.*disjoint/);
  const props: NormalizeGeometryOperationProps = {
    id: 'tree-check',
    positions: 'positions',
    nodes: 'nodes',
    children: 'children',
    rowCount: 1,
    nodeCount: 1,
    childCount: 1,
    output: {positions: 'normalized', children: 'ordered', status: 'status'}
  };
  for (const Operation of [CPUNormalizeGeometryOperation, GPUNormalizeGeometryOperation]) {
    expect(() => new Operation(props)).not.toThrow();
    expect(() => new Operation({...props, nodeCount: -1})).toThrow(/^tree-check/);
    expect(
      () => new Operation({...props, output: {...props.output, positions: 'positions'}})
    ).toThrow(/^tree-check/);
  }
  expect(() => new CPUNormalizeGeometryOperation({...props, maximumNestingDepth: 0})).toThrow(
    /^tree-check.*maximumNestingDepth/
  );
  expect(
    () => new CPUNormalizeGeometryOperation({...props, maximumNestingDepth: 100})
  ).not.toThrow();
});
