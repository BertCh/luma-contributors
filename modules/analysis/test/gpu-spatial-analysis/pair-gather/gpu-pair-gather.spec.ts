// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPU_PAIR_GATHER_NO_ROW,
  GPUPairGather,
  type GPUPairGatherHow
} from '../../../src/gpu-spatial-analysis/pair-gather/index';
import {createGeometryFixture, expectClose} from '../outline-geometry/geometry-fixture';
import {RecipeTestFixture} from '../recipes/recipe-harness';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {runExample as runPairExample} from '../../../../lenses/src/computation-lenses/scenarios/gpu-spatial-analysis/pair-gather.example';
import {runExample as runPredicateExample} from '../../../../lenses/src/computation-lenses/scenarios/gpu-spatial-analysis/spatial-predicate-join.example';
import {runExample as runPreparedExample} from '../../../../lenses/src/computation-lenses/scenarios/gpu-spatial-analysis/spatial-join-prepared.example';
import {requireTestDevice} from '../../utils/require-test-device';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

type ExpectedRow = {
  left: number;
  right: number;
  lval: number;
  lcat: number;
  rval: number | null;
  rcat: number | null;
  d?: number | null;
};

it('final1 compact pairs gather rows independently of permuted caller IDs and preserve inherited candidate loss', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createGeometryFixture(device, {
    inputs: {
      left: {format: 'uint32', values: Uint32Array.of(0, 1)},
      right: {format: 'uint32', values: Uint32Array.of(0, 1)},
      callerIds: {format: 'uint32', values: Uint32Array.of(1, 0)},
      attributes: {format: 'uint32', values: Uint32Array.of(10, 20)},
      count: {format: 'uint32', values: Uint32Array.of(2)},
      overflow: {format: 'uint32', values: Uint32Array.of(0)},
      candidate: {format: 'uint32', values: Uint32Array.of(1)}
    },
    outputs: {
      left: {format: 'uint32', length: 2},
      right: {format: 'uint32', length: 2},
      attributes: {format: 'uint32', length: 2},
      callerIds: {format: 'uint32', length: 2},
      count: {format: 'uint32', length: 1},
      overflow: {format: 'uint32', length: 1},
      candidate: {format: 'uint32', length: 1}
    },
    create: ({inputs, outputs}) => {
      const input = (name: string) => inputs[name] as GraphDataView<'uint32'>;
      const output = (name: string) => outputs[name] as GraphDataView<'uint32'>;
      const status = Object.assign(
        Object.create({
          get candidateOverflow() {
            return input('candidate');
          }
        }),
        {
          count: input('count'),
          overflow: input('overflow')
        }
      );
      return new GPUPairGather({
        pairs: {leftRows: input('left'), rightRows: input('right'), status},
        rightColumns: [
          {source: input('attributes'), output: output('attributes')},
          {source: input('callerIds'), output: output('callerIds')}
        ],
        output: {
          leftRows: output('left'),
          rightRows: output('right'),
          status: {
            count: output('count'),
            overflow: output('overflow'),
            candidateOverflow: output('candidate')
          }
        }
      });
    }
  });
  try {
    const incomplete = await fixture.run();
    expect(incomplete['attributes']).toEqual([10, 20]);
    expect(incomplete['callerIds']).toEqual([1, 0]);
    expect(incomplete['candidate']).toEqual([1]);
    expect(incomplete['overflow']).toEqual([0]);
    fixture.writeInput('candidate', Uint32Array.of(0));
    const complete = await fixture.run();
    expect(complete['attributes']).toEqual([10, 20]);
    expect(complete['candidate']).toEqual([0]);
    expect(complete['count']).toEqual([2]);
  } finally {
    fixture.destroy();
  }
});

for (const kind of ['gather', 'predicate', 'prepared'] as const) {
  it(`F5 R8 ${kind} example counts the active prefix and clears bins on reuse`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = new RecipeTestFixture(device, `histogram-reuse-${kind}`);
    let compiled: ReturnType<typeof fixture.graph.compile> | undefined;
    let prepared: {destroy(): void} | undefined;
    try {
      const parkPositions = fixture.output('park-positions', 'float32x2', 4);
      parkPositions.buffer.write(new Float32Array([0, 0, 3, 0, 3, 1, 0, 1]));
      const parks = {
        kind: 'polygons' as const,
        positions: parkPositions.view,
        featureOffsets: fixture.input('park-features', new Uint32Array([0, 1]), 'uint32', 2),
        polygonOffsets: fixture.input('park-polygons', new Uint32Array([0, 1]), 'uint32', 2),
        ringOffsets: fixture.input('park-rings', new Uint32Array([0, 4]), 'uint32', 2)
      };
      const tracts = {
        kind: 'polygons' as const,
        positions: fixture.input(
          'tract-positions',
          new Float32Array([0, 0, 1, 0, 1, 1, 0, 1, 2, 0, 3, 0, 3, 1, 2, 1]),
          'float32x2',
          8
        ),
        featureOffsets: fixture.input('tract-features', new Uint32Array([0, 1, 2]), 'uint32', 3),
        polygonOffsets: fixture.input('tract-polygons', new Uint32Array([0, 1, 2]), 'uint32', 3),
        ringOffsets: fixture.input('tract-rings', new Uint32Array([0, 4, 8]), 'uint32', 3)
      };
      const pairs = (name: string) => ({
        leftRows: fixture.output(`${name}-left`, 'uint32', 2).view,
        rightRows: fixture.output(`${name}-right`, 'uint32', 2).view,
        status: {
          count: fixture.output(`${name}-count`, 'uint32', 1).view,
          overflow: fixture.output(`${name}-overflow`, 'uint32', 1).view
        }
      });
      const input = {
        parks,
        tracts,
        pairs: pairs('pairs'),
        spatialContext: {coordinateSpace: 'planar' as const, metric: 'native' as const}
      };
      const prefixControl = fixture.output('prefix-control', 'uint32', 2);
      prefixControl.buffer.write(Uint32Array.of(0, 1));
      const add = fixture.graph.add.bind(fixture.graph);
      let prefixWriters = 0;
      // Interpose only the fixture data writer, immediately before the example's actual
      // membership kernel. Keep its shader, bindings and GPUHistogram untouched.
      fixture.graph.add = node => {
        if ('id' in node && typeof node.id === 'string' && node.id.endsWith('active-pairs')) {
          prefixWriters++;
          add(
            createWGSLKernelNode(fixture.graph, {
              id: 'fixture-published-prefix',
              operation: 'review-fixture',
              invocationCount: 2,
              bindings: [
                {name: 'control', view: prefixControl.view, type: 'u32', access: 'read'},
                {name: 'rightRows', view: input.pairs.rightRows, type: 'u32', access: 'read_write'},
                {name: 'count', view: input.pairs.status.count, type: 'u32', access: 'read_write'}
              ],
              body: `if (control[controlOffset] != 0u) {
              rightRows[rightRowsOffset + index] = index;
              if (index == 0u) { count[countOffset] = control[controlOffset + 1u]; }
            }`
            })
          );
        }
        add(node);
      };
      const preparedResult =
        kind === 'prepared'
          ? await runPreparedExample(fixture.graph, {...input, candidatePairs: pairs('candidates')})
          : undefined;
      prepared = preparedResult?.prepared;
      const result =
        preparedResult ??
        (kind === 'gather'
          ? await runPairExample(fixture.graph, {
              ...input,
              parkAreas: fixture.input('areas', new Float32Array([3]), 'float32', 1)
            })
          : await runPredicateExample(fixture.graph, input));
      const bins = fixture.output('bins', 'uint32', 2);
      expect(prefixWriters).toBe(1);
      fixture.graph.add = add;
      fixture.graph.add(
        createWGSLKernelNode(fixture.graph, {
          id: 'capture-tract-bins',
          operation: 'review',
          invocationCount: 2,
          bindings: [
            {name: 'source', view: result.tractMatches, type: 'u32', access: 'read'},
            {name: 'bins', view: bins.view, type: 'u32', access: 'read_write'}
          ],
          body: 'bins[binsOffset + index] = source[sourceOffset + index];'
        })
      );
      compiled = fixture.graph.compile();
      submitGraph(device, compiled, undefined);
      expect(await fixture.readUint32(bins, 2)).toEqual([1, 1]);
      parkPositions.buffer.write(new Float32Array([2.2, 0.2, 2.8, 0.2, 2.8, 0.8, 2.2, 0.8]));
      submitGraph(device, compiled, undefined);
      expect(await fixture.readUint32(bins, 2)).toEqual([0, 1]);
      // Independent hand oracle: rightRows=[0,1], count=1 includes only bin 0.
      // Inverting select's arms would instead count spare slot 1. Reusing the same
      // compiled graph with count=0 must clear both bins, not retain old counts.
      prefixControl.buffer.write(Uint32Array.of(1, 1));
      submitGraph(device, compiled, undefined);
      expect(await fixture.readUint32(bins, 2)).toEqual([1, 0]);
      prefixControl.buffer.write(Uint32Array.of(1, 0));
      submitGraph(device, compiled, undefined);
      expect(await fixture.readUint32(bins, 2)).toEqual([0, 0]);
    } finally {
      compiled?.destroy();
      prepared?.destroy();
      fixture.destroy();
    }
  });
}

// Pinned from geopandas 1.2.0 (scratchpad build/D/gen_pairs.py). Predicate scene: 30 points
// against 5 boxes, `sjoin(predicate='intersects', how='inner' | 'left')`; rows sorted by
// (left, right). Nearest scene: 8 points against 6 points, `sjoin_nearest(distance_col='d',
// max_distance=3.0)` with all ties; the dense [query, slot] layout is what GPUNearestFeatureJoin
// writes (capacity 3, ordered by distance then feature row).
const PREDICATE = {
  lval: [
    0.25, 1.75, 3.25, 4.75, 6.25, 7.75, 9.25, 10.75, 12.25, 13.75, 15.25, 16.75, 18.25, 19.75,
    21.25, 22.75, 24.25, 25.75, 27.25, 28.75, 30.25, 31.75, 33.25, 34.75, 36.25, 37.75, 39.25,
    40.75, 42.25, 43.75
  ],
  lcat: [
    3, 10, 17, 24, 31, 38, 45, 52, 59, 66, 73, 80, 87, 94, 101, 108, 115, 122, 129, 136, 143, 150,
    157, 164, 171, 178, 185, 192, 199, 206
  ],
  rval: [0.5, 10.5, 20.5, 30.5, 40.5],
  rcat: [9, 109, 209, 309, 409],
  pairsLeft: [0, 0, 1, 1, 8, 14, 17, 18, 21, 28, 29],
  pairsRight: [0, 1, 0, 1, 2, 3, 1, 2, 0, 2, 3],
  unmatched: [2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 19, 20, 22, 23, 24, 25, 26, 27],
  inner: [
    {left: 0, right: 0, lval: 0.25, lcat: 3, rval: 0.5, rcat: 9},
    {left: 0, right: 1, lval: 0.25, lcat: 3, rval: 10.5, rcat: 109},
    {left: 1, right: 0, lval: 1.75, lcat: 10, rval: 0.5, rcat: 9},
    {left: 1, right: 1, lval: 1.75, lcat: 10, rval: 10.5, rcat: 109},
    {left: 8, right: 2, lval: 12.25, lcat: 59, rval: 20.5, rcat: 209},
    {left: 14, right: 3, lval: 21.25, lcat: 101, rval: 30.5, rcat: 309},
    {left: 17, right: 1, lval: 25.75, lcat: 122, rval: 10.5, rcat: 109},
    {left: 18, right: 2, lval: 27.25, lcat: 129, rval: 20.5, rcat: 209},
    {left: 21, right: 0, lval: 31.75, lcat: 150, rval: 0.5, rcat: 9},
    {left: 28, right: 2, lval: 42.25, lcat: 199, rval: 20.5, rcat: 209},
    {left: 29, right: 3, lval: 43.75, lcat: 206, rval: 30.5, rcat: 309}
  ],
  left: [
    {left: 0, right: 0, lval: 0.25, lcat: 3, rval: 0.5, rcat: 9},
    {left: 0, right: 1, lval: 0.25, lcat: 3, rval: 10.5, rcat: 109},
    {left: 1, right: 0, lval: 1.75, lcat: 10, rval: 0.5, rcat: 9},
    {left: 1, right: 1, lval: 1.75, lcat: 10, rval: 10.5, rcat: 109},
    {left: 2, right: -1, lval: 3.25, lcat: 17, rval: null, rcat: null},
    {left: 3, right: -1, lval: 4.75, lcat: 24, rval: null, rcat: null},
    {left: 4, right: -1, lval: 6.25, lcat: 31, rval: null, rcat: null},
    {left: 5, right: -1, lval: 7.75, lcat: 38, rval: null, rcat: null},
    {left: 6, right: -1, lval: 9.25, lcat: 45, rval: null, rcat: null},
    {left: 7, right: -1, lval: 10.75, lcat: 52, rval: null, rcat: null},
    {left: 8, right: 2, lval: 12.25, lcat: 59, rval: 20.5, rcat: 209},
    {left: 9, right: -1, lval: 13.75, lcat: 66, rval: null, rcat: null},
    {left: 10, right: -1, lval: 15.25, lcat: 73, rval: null, rcat: null},
    {left: 11, right: -1, lval: 16.75, lcat: 80, rval: null, rcat: null},
    {left: 12, right: -1, lval: 18.25, lcat: 87, rval: null, rcat: null},
    {left: 13, right: -1, lval: 19.75, lcat: 94, rval: null, rcat: null},
    {left: 14, right: 3, lval: 21.25, lcat: 101, rval: 30.5, rcat: 309},
    {left: 15, right: -1, lval: 22.75, lcat: 108, rval: null, rcat: null},
    {left: 16, right: -1, lval: 24.25, lcat: 115, rval: null, rcat: null},
    {left: 17, right: 1, lval: 25.75, lcat: 122, rval: 10.5, rcat: 109},
    {left: 18, right: 2, lval: 27.25, lcat: 129, rval: 20.5, rcat: 209},
    {left: 19, right: -1, lval: 28.75, lcat: 136, rval: null, rcat: null},
    {left: 20, right: -1, lval: 30.25, lcat: 143, rval: null, rcat: null},
    {left: 21, right: 0, lval: 31.75, lcat: 150, rval: 0.5, rcat: 9},
    {left: 22, right: -1, lval: 33.25, lcat: 157, rval: null, rcat: null},
    {left: 23, right: -1, lval: 34.75, lcat: 164, rval: null, rcat: null},
    {left: 24, right: -1, lval: 36.25, lcat: 171, rval: null, rcat: null},
    {left: 25, right: -1, lval: 37.75, lcat: 178, rval: null, rcat: null},
    {left: 26, right: -1, lval: 39.25, lcat: 185, rval: null, rcat: null},
    {left: 27, right: -1, lval: 40.75, lcat: 192, rval: null, rcat: null},
    {left: 28, right: 2, lval: 42.25, lcat: 199, rval: 20.5, rcat: 209},
    {left: 29, right: 3, lval: 43.75, lcat: 206, rval: 30.5, rcat: 309}
  ]
};
const NEAREST = {
  lval: [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0],
  lcat: [40, 41, 42, 43, 44, 45, 46, 47],
  rval: [0.5, 3.5, 6.5, 9.5, 12.5, 15.5],
  rcat: [500, 501, 502, 503, 504, 505],
  capacity: 3,
  ids: [
    0, 4294967295, 4294967295, 1, 4294967295, 4294967295, 0, 1, 4294967295, 3, 4294967295,
    4294967295, 4, 5, 4294967295, 3, 4294967295, 4294967295, 5, 4294967295, 4294967295, 4294967295,
    4294967295, 4294967295
  ],
  counts: [1, 1, 2, 1, 2, 1, 1, 0],
  dists: [
    1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.4142135623730951, -1.0, -1.0,
    1.4142135623730951, 1.4142135623730951, -1.0, 1.4142135623730951, -1.0, -1.0, 2.0, -1.0, -1.0,
    -1.0, -1.0, -1.0
  ],
  inner: [
    {left: 0, right: 0, lval: 1.0, lcat: 40, rval: 0.5, rcat: 500, d: 1.0},
    {left: 1, right: 1, lval: 3.0, lcat: 41, rval: 3.5, rcat: 501, d: 1.0},
    {left: 2, right: 0, lval: 5.0, lcat: 42, rval: 0.5, rcat: 500, d: 1.0},
    {left: 2, right: 1, lval: 5.0, lcat: 42, rval: 3.5, rcat: 501, d: 1.0},
    {left: 3, right: 3, lval: 7.0, lcat: 43, rval: 9.5, rcat: 503, d: 1.4142135623730951},
    {left: 4, right: 4, lval: 9.0, lcat: 44, rval: 12.5, rcat: 504, d: 1.4142135623730951},
    {left: 4, right: 5, lval: 9.0, lcat: 44, rval: 15.5, rcat: 505, d: 1.4142135623730951},
    {left: 5, right: 3, lval: 11.0, lcat: 45, rval: 9.5, rcat: 503, d: 1.4142135623730951},
    {left: 6, right: 5, lval: 13.0, lcat: 46, rval: 15.5, rcat: 505, d: 2.0}
  ],
  left: [
    {left: 0, right: 0, lval: 1.0, lcat: 40, rval: 0.5, rcat: 500, d: 1.0},
    {left: 1, right: 1, lval: 3.0, lcat: 41, rval: 3.5, rcat: 501, d: 1.0},
    {left: 2, right: 0, lval: 5.0, lcat: 42, rval: 0.5, rcat: 500, d: 1.0},
    {left: 2, right: 1, lval: 5.0, lcat: 42, rval: 3.5, rcat: 501, d: 1.0},
    {left: 3, right: 3, lval: 7.0, lcat: 43, rval: 9.5, rcat: 503, d: 1.4142135623730951},
    {left: 4, right: 4, lval: 9.0, lcat: 44, rval: 12.5, rcat: 504, d: 1.4142135623730951},
    {left: 4, right: 5, lval: 9.0, lcat: 44, rval: 15.5, rcat: 505, d: 1.4142135623730951},
    {left: 5, right: 3, lval: 11.0, lcat: 45, rval: 9.5, rcat: 503, d: 1.4142135623730951},
    {left: 6, right: 5, lval: 13.0, lcat: 46, rval: 15.5, rcat: 505, d: 2.0},
    {left: 7, right: -1, lval: 15.0, lcat: 47, rval: null, rcat: null, d: null}
  ]
};

type Scene = typeof PREDICATE | typeof NEAREST;
type TestDevice = NonNullable<Awaited<ReturnType<typeof getWebGPUTestDevice>>>;

it.each([
  0, 4
])('GPUPairGather gathers a uint32 column at source byte offset %i into an output byte offset', async sourceByteOffset => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const createInput = (name: string, values: number[]) => {
    const buffer = createInputBuffer(device, new Uint32Array(values));
    buffers.push(buffer);
    return importGraphBuffer(graph, name, buffer, 'uint32', values.length);
  };
  const createOutput = (name: string) => {
    const buffer = createOutputBuffer(device, 1);
    buffers.push(buffer);
    return importGraphBuffer(graph, name, buffer, 'uint32', 1);
  };
  try {
    // The row output and the column output are separate buffers: a column kernel reads the rows
    // and writes the column, and WebGPU refuses one buffer bound read-only and read-write in the
    // same pass, whatever the byte ranges.
    const leftRows = createOutput('left-rows');
    const columnBuffer = createOutputBuffer(device, 128);
    buffers.push(columnBuffer);
    const columnView = importGraphBuffer(graph, 'column-output', columnBuffer, 'uint32');
    const columnOutput = graph.createDataView(columnView.buffer, {
      format: 'uint32',
      length: 1,
      byteOffset: 256
    });
    const sourceStorage = createInput('source', sourceByteOffset === 0 ? [7] : [99, 7, 88]);
    const source = graph.createDataView(sourceStorage.buffer, {
      format: 'uint32',
      length: 1,
      byteOffset: sourceByteOffset
    });
    graph.add(
      new GPUPairGather({
        pairs: {
          leftRows: createInput('pair-left', [0]),
          rightRows: createInput('pair-right', [0]),
          status: {
            count: createInput('pair-count', [1]),
            requiredCount: createInput('pair-required', [1]),
            overflow: createInput('pair-overflow', [0])
          }
        },
        leftColumns: [{source, output: columnOutput}],
        output: {
          leftRows,
          rightRows: createOutput('right-rows'),
          status: {count: createOutput('row-count'), overflow: createOutput('overflow')}
        }
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const words = await readUint32(columnBuffer);
      expect(words[0]).toBe(0);
      expect(words[256 / Uint32Array.BYTES_PER_ELEMENT]).toBe(7);
    } finally {
      compiled.destroy();
    }
  } finally {
    for (const buffer of buffers) buffer.destroy();
  }
});

it('GPUPairGather fills unmatched and out-of-range uint32 column rows with all ones', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createGeometryFixture(device, {
    inputs: {
      pairLeft: {values: new Uint32Array([0, 1]), format: 'uint32'},
      pairRight: {values: new Uint32Array([0, 3]), format: 'uint32'},
      pairCount: {values: new Uint32Array([2]), format: 'uint32'},
      pairRequired: {values: new Uint32Array([2]), format: 'uint32'},
      pairOverflow: {values: new Uint32Array([0]), format: 'uint32'},
      unmatchedIds: {values: new Uint32Array([2]), format: 'uint32'},
      unmatchedCount: {values: new Uint32Array([1]), format: 'uint32'},
      unmatchedOverflow: {values: new Uint32Array([0]), format: 'uint32'},
      column: {values: new Uint32Array([9]), format: 'uint32'}
    },
    outputs: {
      leftRows: {format: 'uint32', length: 3},
      rightRows: {format: 'uint32', length: 3},
      rowCount: {format: 'uint32', length: 1},
      overflow: {format: 'uint32', length: 1},
      column: {format: 'uint32', length: 3}
    },
    create: ({inputs: inputViews, outputs: outputViews}) => {
      const inputs = inputViews as Record<string, GraphDataView<'uint32'>>;
      const outputs = outputViews as Record<string, GraphDataView<'uint32'>>;
      return new GPUPairGather({
        how: 'left',
        pairs: {
          leftRows: inputs['pairLeft'],
          rightRows: inputs['pairRight'],
          status: {
            count: inputs['pairCount'],
            requiredCount: inputs['pairRequired'],
            overflow: inputs['pairOverflow']
          }
        },
        unmatchedLeft: {
          ids: inputs['unmatchedIds'],
          status: {
            count: inputs['unmatchedCount'],
            overflow: inputs['unmatchedOverflow']
          }
        },
        rightColumns: [{source: inputs['column'], output: outputs['column']}],
        output: {
          leftRows: outputs['leftRows'],
          rightRows: outputs['rightRows'],
          status: {count: outputs['rowCount'], overflow: outputs['overflow']}
        }
      });
    }
  });
  try {
    const result = await fixture.run();
    expect(result['rowCount']).toEqual([3]);
    expect(result['overflow']).toEqual([0]);
    expect(result['rightRows']).toEqual([0, 3, GPU_PAIR_GATHER_NO_ROW]);
    expect(result['column']).toEqual([9, 0xffffffff, 0xffffffff]);
  } finally {
    fixture.destroy();
  }
});

function pad(values: number[], length: number): Uint32Array {
  const padded = new Uint32Array(length);
  padded.set(values);
  return padded;
}

function createColumnInputs(scene: Scene) {
  return {
    lval: {values: new Float32Array(scene.lval), format: 'float32' as const},
    lcat: {values: new Uint32Array(scene.lcat), format: 'uint32' as const},
    rval: {values: new Float32Array(scene.rval), format: 'float32' as const},
    rcat: {values: new Uint32Array(scene.rcat), format: 'uint32' as const}
  };
}

function createOutputs(capacity: number) {
  return {
    leftRows: {format: 'uint32' as const, length: capacity},
    rightRows: {format: 'uint32' as const, length: capacity},
    rowCount: {format: 'uint32' as const, length: 1},
    overflow: {format: 'uint32' as const, length: 1},
    candidateOverflow: {format: 'uint32' as const, length: 1},
    matchedCount: {format: 'uint32' as const, length: 1},
    lvalOut: {format: 'float32' as const, length: capacity},
    lcatOut: {format: 'uint32' as const, length: capacity},
    rvalOut: {format: 'float32' as const, length: capacity},
    rcatOut: {format: 'uint32' as const, length: capacity},
    slotOut: {format: 'float32' as const, length: capacity}
  };
}

function createGather(
  views: {inputs: Record<string, never>; outputs: Record<string, never>},
  how: GPUPairGatherHow,
  source: object
) {
  const {inputs, outputs} = views;
  return new GPUPairGather({
    how,
    ...source,
    leftColumns: [
      {source: inputs['lval'], output: outputs['lvalOut']},
      {source: inputs['lcat'], output: outputs['lcatOut']}
    ],
    rightColumns: [
      {source: inputs['rval'], output: outputs['rvalOut']},
      {source: inputs['rcat'], output: outputs['rcatOut']}
    ],
    output: {
      leftRows: outputs['leftRows'],
      rightRows: outputs['rightRows'],
      status: {
        count: outputs['rowCount'],
        overflow: outputs['overflow'],
        matchedCount: outputs['matchedCount'],
        candidateOverflow: outputs['candidateOverflow']
      }
    }
  });
}

function decodeRows(result: Record<string, number[]>, distanceOutput?: string): ExpectedRow[] {
  const rows: ExpectedRow[] = [];
  for (let row = 0; row < result['rowCount'][0]; row++) {
    const right = result['rightRows'][row];
    const unmatched = right === GPU_PAIR_GATHER_NO_ROW;
    const decoded: ExpectedRow = {
      left: result['leftRows'][row],
      right: unmatched ? -1 : right,
      lval: result['lvalOut'][row],
      lcat: result['lcatOut'][row],
      rval: Number.isNaN(result['rvalOut'][row]) ? null : result['rvalOut'][row],
      rcat: result['rcatOut'][row] === 0xffffffff ? null : result['rcatOut'][row]
    };
    if (distanceOutput) {
      decoded.d = Number.isNaN(result[distanceOutput][row]) ? null : result[distanceOutput][row];
    }
    rows.push(decoded);
  }
  return rows.sort((a, b) => a.left - b.left || a.right - b.right);
}

function expectRowsMatch(actual: ExpectedRow[], expected: ExpectedRow[]) {
  expect(actual.length).toBe(expected.length);
  actual.forEach((row, index) => {
    const {d: actualDistance, ...actualRest} = row;
    const {d: expectedDistance, ...expectedRest} = expected[index];
    expect(actualRest).toEqual(expectedRest);
    if (expectedDistance !== undefined) {
      if (expectedDistance === null) {
        expect(actualDistance).toBeNull();
      } else {
        expectClose(actualDistance as number, expectedDistance, 1e-6, 1e-6, `row ${index}`);
      }
    }
  });
}

function createPredicateFixture(device: TestDevice, how: GPUPairGatherHow, capacity: number) {
  return createGeometryFixture(device, {
    inputs: {
      ...createColumnInputs(PREDICATE),
      pairLeft: {values: pad(PREDICATE.pairsLeft, 16), format: 'uint32'},
      pairRight: {values: pad(PREDICATE.pairsRight, 16), format: 'uint32'},
      pairCount: {values: new Uint32Array([PREDICATE.pairsLeft.length]), format: 'uint32'},
      pairRequired: {values: new Uint32Array([PREDICATE.pairsLeft.length]), format: 'uint32'},
      pairOverflow: {values: new Uint32Array([0]), format: 'uint32'},
      unmatchedIds: {values: pad(PREDICATE.unmatched, 32), format: 'uint32'},
      unmatchedCount: {values: new Uint32Array([PREDICATE.unmatched.length]), format: 'uint32'},
      unmatchedOverflow: {values: new Uint32Array([0]), format: 'uint32'},
      slotWeights: {
        values: new Float32Array(Array.from({length: 16}, (_, slot) => slot + 0.5)),
        format: 'float32'
      }
    },
    outputs: createOutputs(capacity),
    create: ({inputs, outputs}) =>
      createGather({inputs: inputs as never, outputs: outputs as never}, how, {
        pairs: {
          leftRows: inputs['pairLeft'],
          rightRows: inputs['pairRight'],
          status: {
            count: inputs['pairCount'],
            requiredCount: inputs['pairRequired'],
            overflow: inputs['pairOverflow']
          }
        },
        ...(how === 'left'
          ? {
              unmatchedLeft: {
                ids: inputs['unmatchedIds'],
                status: {
                  count: inputs['unmatchedCount'],
                  overflow: inputs['unmatchedOverflow']
                }
              }
            }
          : {}),
        slotColumns: [{source: inputs['slotWeights'], output: outputs['slotOut']}]
      })
  });
}

for (const how of ['inner', 'left'] as const) {
  it(`GPUPairGather how=${how} matches geopandas sjoin(predicate='intersects', how='${how}')`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createPredicateFixture(device, how, 48);
    const result = await fixture.run();
    expect(result['overflow'][0]).toBe(0);
    expect(result['matchedCount'][0]).toBe(PREDICATE.pairsLeft.length);
    expectRowsMatch(decodeRows(result), PREDICATE[how] as ExpectedRow[]);
    // Matched rows keep input pair order and carry their slot column; unmatched rows get NaN.
    PREDICATE.pairsLeft.forEach((_, slot) => {
      expect(result['slotOut'][slot]).toBe(slot + 0.5);
    });
    if (how === 'left') {
      expect(result['slotOut'][PREDICATE.pairsLeft.length]).toBeNaN();
      expect(result['rightRows'][PREDICATE.pairsLeft.length]).toBe(GPU_PAIR_GATHER_NO_ROW);
    }
    expect(fixture.getCompileCount()).toBe(0);
    fixture.destroy();
  });
}

it('GPUPairGather reports overflow when the output capacity is too small', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createPredicateFixture(device, 'left', 20);
  const result = await fixture.run();
  expect(result['rowCount'][0]).toBe(20);
  expect(result['overflow'][0]).toBe(1);
  expect(result['matchedCount'][0]).toBe(PREDICATE.pairsLeft.length);
  fixture.destroy();
});

function createNearestFixture(device: TestDevice, how: GPUPairGatherHow) {
  return createGeometryFixture(device, {
    inputs: {
      ...createColumnInputs(NEAREST),
      neighborRows: {values: new Uint32Array(NEAREST.ids), format: 'uint32'},
      neighborCounts: {values: new Uint32Array(NEAREST.counts), format: 'uint32'},
      neighborDistances: {values: new Float32Array(NEAREST.dists), format: 'float32'},
      joinOverflow: {values: new Uint32Array([0]), format: 'uint32'}
    },
    outputs: createOutputs(16),
    create: ({inputs, outputs}) =>
      createGather({inputs: inputs as never, outputs: outputs as never}, how, {
        neighbors: {
          rows: inputs['neighborRows'],
          counts: inputs['neighborCounts'],
          capacity: NEAREST.capacity
        },
        candidateOverflowSources: [inputs['joinOverflow']],
        slotColumns: [{source: inputs['neighborDistances'], output: outputs['slotOut']}]
      })
  });
}

for (const how of ['inner', 'left'] as const) {
  it(`GPUPairGather flattens nearest neighbors like geopandas sjoin_nearest(how='${how}', distance_col)`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createNearestFixture(device, how);
    const result = await fixture.run();
    expect(result['overflow'][0]).toBe(0);
    expectRowsMatch(
      decodeRows({...result, rowCount: result['rowCount']}, 'slotOut'),
      NEAREST[how] as ExpectedRow[]
    );
    fixture.destroy();
  });
}

it('GPUPairGather ORs many overflow sources within the core binding limit', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Nine flags (only the last set) push the publish kernel past 8 storage buffers, so they are
  // first OR-ed into one word in groups of 7.
  const flags = Object.fromEntries(
    Array.from({length: 9}, (_, flag) => [
      `flag${flag}`,
      {values: new Uint32Array([flag === 8 ? 1 : 0]), format: 'uint32' as const}
    ])
  );
  const fixture = createGeometryFixture(device, {
    inputs: {
      ...createColumnInputs(NEAREST),
      neighborRows: {values: new Uint32Array(NEAREST.ids), format: 'uint32'},
      neighborCounts: {values: new Uint32Array(NEAREST.counts), format: 'uint32'},
      neighborDistances: {values: new Float32Array(NEAREST.dists), format: 'float32'},
      ...flags
    },
    outputs: createOutputs(16),
    create: ({inputs, outputs}) =>
      createGather({inputs: inputs as never, outputs: outputs as never}, 'left', {
        neighbors: {
          rows: inputs['neighborRows'],
          counts: inputs['neighborCounts'],
          capacity: NEAREST.capacity
        },
        overflowSources: Object.keys(flags).map(name => inputs[name]),
        slotColumns: [{source: inputs['neighborDistances'], output: outputs['slotOut']}]
      })
  });
  const result = await fixture.run();
  expect(result['overflow'][0]).toBe(1);
  expectRowsMatch(
    decodeRows({...result, rowCount: result['rowCount']}, 'slotOut'),
    NEAREST.left as ExpectedRow[]
  );
  fixture.destroy();
});
