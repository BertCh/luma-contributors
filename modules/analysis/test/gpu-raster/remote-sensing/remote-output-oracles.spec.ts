// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPURasterConservativeRemapping} from '../../../src/raster-analysis/remote-sensing/gpu-raster-conservative-remapping';
import {
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import {GPURasterRegionGraphOperation} from '../../../src/raster-analysis/remote-sensing/gpu-raster-region-graph-operation';
import {CPURasterRegionGraphOperation} from '../../../src/raster-analysis/remote-sensing/cpu-raster-region-graph-operation';
import {
  expectLiteralRows,
  poisonOperation
} from '../../analysis-core/mixed-execution/output-oracles-test-utils';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';

it('conservative prepared source rows and overlap areas match two unit rectangles with explicit padding', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const preparedPresent of [true, false]) {
    const buffers: Buffer[] = [];
    const allocate = (values: Float32Array | Uint32Array) => {
      const buffer = createOutputBuffer(device, values.length);
      buffer.write(values);
      buffers.push(buffer);
      return buffer;
    };
    const values = allocate(new Float32Array([100, 2, 4, 100]));
    const indices = allocate(new Uint32Array(6).fill(123));
    const areas = allocate(new Float32Array(6).fill(123));
    const result = allocate(new Float32Array([123]));
    const covered = allocate(new Float32Array([123]));
    const status = allocate(new Uint32Array([123]));
    const build = (weightMode: 'prepare' | 'reuse') => {
      const graph = new GPUCommandGraph(device);
      const contributor = new GPURasterConservativeRemapping({
        source: {width: 4, height: 1, origin: [0, 0], spacing: [1, 1]},
        destination: {width: 1, height: 1, origin: [1.25, 0], spacing: [1.25, 1]},
        values: importGraphBuffer(graph, 'values', values, 'float32', 4),
        normalization: 'destination-area',
        weightMode,
        preparedWeights: preparedPresent
          ? {
              indices: importGraphBuffer(graph, 'indices', indices, 'uint32', 6),
              areas: importGraphBuffer(graph, 'areas', areas, 'float32', 6)
            }
          : undefined,
        output: {
          values: importGraphBuffer(graph, 'result', result, 'float32', 1),
          coveredArea: importGraphBuffer(graph, 'covered', covered, 'float32', 1),
          status: importGraphBuffer(graph, 'status', status, 'uint32', 1)
        }
      });
      graph.add(contributor);
      return graph.compile();
    };
    const prepared = build('prepare');
    try {
      submitGraph(device, prepared);
      // Shapely oracle: destination [1.25,2.5] intersects source rows 1 and 2,
      // with areas .75 and .5. Candidate slots 0 and 1 are not source identities.
      if (preparedPresent) {
        expect(await readUint32(indices, 6)).toEqual([
          1, 2, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff
        ]);
        expect(await readFloat32(areas, 6)).toEqual([0.75, 0.5, 0, 0, 0, 0]);
      }
      expectLiteralRows(await readFloat32(result, 1), [2.8]);
      expect(await readFloat32(covered, 1)).toEqual([1.25]);
      expect(await readUint32(status, 1)).toEqual([0]);
      if (preparedPresent) {
        values.write(new Float32Array([100, 10, 14, 100]));
        result.write(new Float32Array([123]));
        const reused = build('reuse');
        try {
          submitGraph(device, reused);
          expectLiteralRows(await readFloat32(result, 1), [11.6]);
          expect(await readUint32(indices, 6)).toEqual([
            1, 2, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff
          ]);
          expect(await readFloat32(areas, 6)).toEqual([0.75, 0.5, 0, 0, 0, 0]);
        } finally {
          reused.destroy();
        }
      }
    } finally {
      prepared.destroy();
      buffers.forEach(buffer => buffer.destroy());
    }
  }
});

it('region operations write literal adjacency, mean weights, bounded partitions and pending status', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const Operation of [GPURasterRegionGraphOperation, CPURasterRegionGraphOperation]) {
    for (const maximumIterations of [1, 2]) {
      const props = {
        grid: makeGPURasterMetadata({width: 3, height: 1}),
        regionCount: 3,
        labels: 'labels',
        values: 'values',
        mergeMode: 'adaptive' as const,
        mergeThreshold: 3,
        maximumIterations,
        output: {
          adjacency: 'adjacency',
          weights: 'weights',
          mapping: 'mapping',
          status: {iterationLimitReached: 'pending'}
        }
      };
      const operation = new Operation(props);
      const rows = await runOperationGraph(device, {
        inputs: {
          labels: {format: 'uint32', values: new Uint32Array([0, 1, 2])},
          values: {format: 'float32', values: new Float32Array([0, 2, 4])}
        },
        operations: [
          operation instanceof GPURasterRegionGraphOperation
            ? poisonOperation(operation)
            : operation
        ],
        outputs: ['adjacency', 'weights', 'mapping', 'pending']
      });
      // Tie chooses 0--1, whose new mean 1 leaves the weight-3 edge to region 2 eligible.
      expectLiteralRows(rows['adjacency'], [0, 1, 0, 1, 0, 1, 0, 1, 0], 0);
      expectLiteralRows(
        rows['weights'],
        maximumIterations === 1
          ? [NaN, NaN, NaN, NaN, NaN, 3, NaN, 3, NaN]
          : new Array(9).fill(NaN),
        0
      );
      expectLiteralRows(rows['mapping'], maximumIterations === 1 ? [0, 0, 2] : [0, 0, 0], 0);
      expectLiteralRows(rows['pending'], [maximumIterations === 1 ? 1 : 0], 0);
    }
    for (const [searchStructure, maximumIterations, expectedMapping, pending] of [
      ['auto', 1, [0, 0, 1, 2], 1],
      ['dense', 2, [0, 0, 0, 1], 1],
      ['boruvka', 1, [0, 0, 2, 2], 1],
      ['boruvka', 2, [0, 0, 0, 0], 0]
    ] as const) {
      const props = {
        grid: makeGPURasterMetadata({width: 4, height: 1}),
        regionCount: 4,
        labels: 'labels',
        values: 'values',
        mergeMode: 'fixed' as const,
        searchStructure,
        mergeThreshold: 3,
        maximumIterations,
        output: {
          adjacency: 'adjacency',
          weights: 'weights',
          mapping: 'mapping',
          status: {iterationLimitReached: 'pending'}
        }
      };
      const operation = new Operation(props);
      const rows = await runOperationGraph(device, {
        inputs: {
          labels: {format: 'uint32', values: new Uint32Array([0, 1, 2, 3])},
          values: {format: 'float32', values: new Float32Array([0, 1, 4, 5])}
        },
        operations: [
          operation instanceof GPURasterRegionGraphOperation
            ? poisonOperation(operation)
            : operation
        ],
        outputs: ['adjacency', 'weights', 'mapping', 'pending']
      });
      expectLiteralRows(rows['adjacency'], [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0], 0);
      expectLiteralRows(
        rows['weights'],
        [NaN, 1, NaN, NaN, 1, NaN, 3, NaN, NaN, 3, NaN, 1, NaN, NaN, 1, NaN],
        0
      );
      expectLiteralRows(rows['mapping'], expectedMapping, 0);
      expectLiteralRows(rows['pending'], [pending], 0);
    }
  }
});
