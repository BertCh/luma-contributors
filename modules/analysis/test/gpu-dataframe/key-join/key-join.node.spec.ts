// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {GPUKeyLookup, type GPUKeyLookupProps} from '../../../src/dataframe-analysis/key-join/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {decodeFixedPointSum, getScaledValue, lookupKeysOnCPU, packKeys} from './key-join-oracle';

let serial = 0;

it('final1b F1 rejects candidate publication into an input join key', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const base = createProps(graph);
    const key = createTransientView(graph, 'alias-left-keys', 'uint32', 8);
    const scalar = (name: string) => createTransientView(graph, name, 'uint32', 1);
    expect(() =>
      new GPUKeyLookup({
        ...base,
        leftKeys: key,
        compactMatchedLeftRows: true,
        output: {
          rows: {
            ids: createTransientView(graph, 'retained', 'uint32', 8),
            status: {
              count: scalar('count'),
              overflow: scalar('overflow'),
              candidateOverflow: graph.createDataView(key.buffer, {format: 'uint32', length: 1})
            }
          }
        }
      }).getCommandNodes(graph)
    ).toThrow(/share buffers|overlap/);
  } finally {
    device.destroy();
  }
});

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUKeyLookupProps> = {},
  keyFormat: 'uint32' | 'uint32x2' = 'uint32'
): GPUKeyLookupProps {
  const view = <Format extends 'uint32' | 'float32' | 'uint32x2'>(format: Format, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  return {
    leftKeys: view(keyFormat, 8),
    rightKeys: view(keyFormat, 6),
    output: {matched: view('uint32', 8)},
    ...overrides
  } as GPUKeyLookupProps;
}

it('lookupKeysOnCPU oracle handles duplicates, masks, reserved keys and aggregates', () => {
  const result = lookupKeysOnCPU({
    keyBits: 32,
    leftKeys: [1n, 2n, 3n, 0xffffffffn, 1n, 2n],
    rightKeys: [2n, 1n, 2n, 0xffffffffn, 1n],
    rightMask: [1, 1, 1, 1, 0],
    leftMask: [1, 1, 1, 1, 1, 0],
    compactMatchedLeftRows: true,
    capacity: 2,
    sumScale: 1,
    gatherColumns: [{words: Uint32Array.from([10, 11, 12, 13, 14]), isFloat: false}],
    aggregates: [
      {operation: 'sum', values: Float32Array.from([1.5, 2, 3, 4, 5])},
      {operation: 'count'}
    ]
  });
  expect(result.rightRows).toEqual([1, 0, 0xffffffff, 0xffffffff, 1, 0xffffffff]);
  expect(result.matchCounts).toEqual([1, 2, 0, 0, 1, 0]);
  expect(result.matched).toEqual([1, 1, 0, 0, 1, 0]);
  expect(result.rightMatched).toEqual([1, 1, 1, 0, 0]);
  expect(result.gathered[0]).toEqual([11, 10, 0xffffffff, 0xffffffff, 11, 0xffffffff]);
  expect(result.compactRows).toEqual([0, 1]);
  expect(result.compactTotal).toBe(3);
  expect(result.compactOverflow).toBe(1);
  expect(result.aggregates[1].values).toEqual([1, 2, 0, 0, 1, 0]);
  // The `sums` words at sumScale 1 round half to even: round(1.5) = 2, plus 3 = 5.
  expect(result.aggregates[0].sums).toEqual([2n, 5n, 0n, 0n, 2n, 0n]);
  // The float sum is the sum of the values, 1.5 + 3 = 4.5, whatever the scale of the words.
  expect(result.aggregates[0].values.slice(0, 2)).toEqual([2, 4.5]);
  expect(Number.isNaN(result.aggregates[0].values[2])).toBe(true);
});

it('oracle helpers scale, decode and pack', () => {
  expect(getScaledValue(2.5, 1)).toBe(2n);
  expect(getScaledValue(-0.25, 4)).toBe(-1n);
  expect(decodeFixedPointSum(-3n * 65536n, 65536)).toBe(-3);
  expect(decodeFixedPointSum(2n ** 40n, 1)).toBe(2 ** 40);
  expect(Array.from(packKeys([0x100000002n], 64))).toEqual([2, 1]);
  expect(Array.from(packKeys([7n], 32))).toEqual([7]);
});

it('GPUKeyLookup validates its props', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {
    id: 'key-join-validation'
  });
  const view = <Format extends 'uint32' | 'float32' | 'uint32x2'>(format: Format, length: number) =>
    createTransientView(graph, `v-${serial++}`, format, length);
  expect(() => new GPUKeyLookup(createProps(graph)).getCommandNodes(graph)).not.toThrow();
  expect(() => new GPUKeyLookup(createProps(graph, {sumScale: -1}))).toThrow(/sumScale/);
  expect(
    () =>
      new GPUKeyLookup(
        createProps(graph, {
          rightKeys: view('uint32x2', 6)
        } as Partial<GPUKeyLookupProps>)
      )
  ).toThrow(/rightKeys/);
  expect(() => new GPUKeyLookup(createProps(graph, {leftMask: view('uint32', 3)}))).toThrow(
    /leftMask length/
  );
  expect(
    () => new GPUKeyLookup(createProps(graph, {output: {matched: view('uint32', 3)}}))
  ).toThrow(/output.matched length/);
  expect(
    () => new GPUKeyLookup(createProps(graph, {output: {rightMatched: view('uint32', 8)}}))
  ).toThrow(/rightMatched length/);
  // Compaction requires an explicit destination.
  expect(() => new GPUKeyLookup(createProps(graph, {compactMatchedLeftRows: true}))).toThrow(
    /rows is required/
  );
  const rows = {
    ids: view('uint32', 4),
    status: {count: view('uint32', 1), overflow: view('uint32', 1)}
  };
  expect(() => new GPUKeyLookup(createProps(graph, {output: {rows}}))).toThrow(/only supported/);
  expect(
    () => new GPUKeyLookup(createProps(graph, {compactMatchedLeftRows: true, output: {rows}}))
  ).not.toThrow();
  // Gather columns.
  expect(
    () =>
      new GPUKeyLookup(
        createProps(graph, {
          gather: [{column: view('float32', 6), output: view('uint32', 8)}]
        })
      )
  ).toThrow(/gather\[0\]\.output/);
  expect(
    () =>
      new GPUKeyLookup(
        createProps(graph, {
          gather: [{column: view('float32', 5), output: view('float32', 8)}]
        })
      )
  ).toThrow(/column length/);
  // Aggregates.
  expect(
    () =>
      new GPUKeyLookup(
        createProps(graph, {
          aggregates: [{operation: 'sum', output: view('float32', 8)}]
        })
      )
  ).toThrow(/needs a column/);
  expect(
    () =>
      new GPUKeyLookup(
        createProps(graph, {
          aggregates: [
            {
              operation: 'maximum',
              column: view('float32', 6),
              output: view('float32', 8),
              sums: view('uint32x2', 8)
            }
          ]
        })
      )
  ).toThrow(/sum or mean/);
  // Nodes are deterministic and prefixed.
  const props = createProps(graph, {
    gather: [{column: view('float32', 6), output: view('float32', 8)}],
    aggregates: [
      {
        operation: 'mean',
        column: view('float32', 6),
        output: view('float32', 8)
      }
    ]
  });
  const first = new GPUKeyLookup({...props, id: 'a'}).getCommandNodes(graph).map(node => node.id);
  expect(first.every(id => id.startsWith('a-'))).toBe(true);
  expect(first).toContain('a-probe');
});

it('count-distinct uses exact finite value identity and has independent Set-based expectations', () => {
  const result = lookupKeysOnCPU({
    keyBits: 32,
    leftKeys: [1n, 2n, 3n, 4n],
    rightKeys: [1n, 1n, 1n, 1n, 1n, 2n, 2n, 3n],
    compactMatchedLeftRows: false,
    sumScale: 1,
    aggregates: [
      {
        operation: 'count-distinct',
        values: new Float32Array([-0, 0, 2, 2, NaN, Infinity, -Infinity, -3])
      }
    ]
  });
  expect(result.aggregates[0].values).toEqual([2, 0, 1, 0]);
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const props = createProps(graph, {
      aggregates: [
        {
          operation: 'count-distinct',
          column: createTransientView(graph, 'distinct-values', 'float32', 6),
          output: createTransientView(graph, 'distinct-output', 'float32', 8)
        }
      ]
    });
    const contributor = new GPUKeyLookup(props);
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    for (const kernel of getGPUCommandNodeKernelSources(nodes))
      expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
  } finally {
    device.destroy();
  }
});

it('DPC-124 lookup oracle keeps a finite mean before narrowing an overflowing sum', () => {
  const values = new Float32Array(4).fill(Math.fround(1e38));
  const result = lookupKeysOnCPU({
    keyBits: 32,
    leftKeys: [7n, 7n, 9n],
    rightKeys: [7n, 7n, 7n, 7n],
    compactMatchedLeftRows: false,
    sumScale: 1,
    aggregates: [
      {operation: 'mean', values},
      {operation: 'sum', values},
      {operation: 'count', values}
    ]
  });
  // Closed form: mean([x,x,x,x]) = x, even though f32(4*x) overflows.
  // IEEE binary32(1e38) = 9.999999680285692e37, independently pinned as a literal.
  expect(result.aggregates[0].values).toEqual([9.999999680285692e37, 9.999999680285692e37, NaN]);
  expect(result.aggregates[0].values.slice(0, 2).every(Number.isFinite)).toBe(true);
  expect(result.aggregates[1].values).toEqual([Infinity, Infinity, NaN]);
  expect(result.aggregates[2].values).toEqual([4, 4, 0]);
  expect(Number.isNaN(result.aggregates[0].values[2])).toBe(true);
  expect(Number.isNaN(result.aggregates[1].values[2])).toBe(true);
  expect(result.rightRows).toEqual([0, 0, 0xffffffff]);
  expect(result.matchCounts).toEqual([4, 4, 0]);
});

it('B4 lookup oracle preserves cancellation below a binary64 serial sum', () => {
  const values = Float32Array.from([2 ** 54, 1, -(2 ** 54)]);
  const result = lookupKeysOnCPU({
    keyBits: 32,
    leftKeys: [7n, 9n],
    rightKeys: [7n, 7n, 7n],
    compactMatchedLeftRows: false,
    sumScale: 1,
    aggregates: [
      {operation: 'sum', values},
      {operation: 'mean', values}
    ]
  });
  // Exact cancellation leaves 1; rounding 1/3 once to binary32 gives the literal below.
  expect(result.aggregates[0].values).toEqual([1, NaN]);
  expect(result.aggregates[1].values).toEqual([0.3333333432674408, NaN]);
});

it('S2 join preflight enumerates the global named output table once', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph, {
    gather: Array.from({length: 20}, (_, index) => ({
      column: createTransientView(graph, `gather-source-${index}`, 'float32', 6),
      output: createTransientView(graph, `gather-destination-${index}`, 'float32', 8)
    }))
  });
  const entries = Object.entries;
  let scans = 0;
  const spy = vi.spyOn(Object, 'entries').mockImplementation(object => {
    if (Object.hasOwn(object, 'gather[19].output') && Object.hasOwn(object, 'output.matched'))
      scans++;
    return entries(object);
  });
  try {
    new GPUKeyLookup(props);
    expect(scans).toBe(1);
  } finally {
    spy.mockRestore();
    device.destroy();
  }
});
